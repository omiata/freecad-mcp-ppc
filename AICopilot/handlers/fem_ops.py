# FEM workbench operation handlers for FreeCAD MCP
#
# Linear-static structural analysis with Gmsh meshing and the CalculiX
# solver, driven through FreeCAD's own ObjectsFem factories and femtools.
#
# The solver-run and result-extraction flow (run_analysis) is ported from
# neka-nat/freecad-mcp's addon/FreeCADMCP/rpc_server/fem_executor.py and
# object_factory.py:
#   Copyright (c) 2025 Shirokuma (k tanaka)
#   Licensed under the MIT License (see LICENSES/MIT-neka-nat-freecad-mcp.txt)
# and extended here into a full dispatcher: dedicated analysis / material /
# mesh / constraint operations instead of generic object creation plus
# execute_python for the parts the generic setter couldn't reach (force
# magnitude and direction, Gmsh run), a setup check, a yield-based safety
# factor, and stable result selection.
#
# FEM modules (ObjectsFem, femmesh, femtools) are imported inside each
# method, never at module scope: importing this handler must not require
# the FEM workbench, and unit tests stub them per test.

import contextlib
import json
import math
import os
import tempfile
import time
from typing import Any

import FreeCAD

from .base import NO_ACTIVE_DOCUMENT_ERROR, BaseHandler

_ANALYSIS_TYPES = ('Fem::FemAnalysis', 'Fem::FemAnalysisPython')

# Nominal bulk properties for quick studies. E and yield in MPa, density in
# kg/m^3. Printed parts (FDM) are anisotropic and usually weaker than these
# bulk figures -- callers that care pass explicit values instead.
MATERIAL_PRESETS = {
    'steel_s235': {'name': 'Steel S235', 'youngs_modulus': 210000.0, 'poisson_ratio': 0.30,
                   'density': 7850.0, 'yield_strength': 235.0},
    'aluminium_6061_t6': {'name': 'Aluminium 6061-T6', 'youngs_modulus': 69000.0,
                          'poisson_ratio': 0.33, 'density': 2700.0, 'yield_strength': 276.0},
    'pla': {'name': 'PLA', 'youngs_modulus': 3500.0, 'poisson_ratio': 0.36,
            'density': 1240.0, 'yield_strength': 50.0},
    'petg': {'name': 'PETG', 'youngs_modulus': 2100.0, 'poisson_ratio': 0.38,
             'density': 1270.0, 'yield_strength': 50.0},
    'abs': {'name': 'ABS', 'youngs_modulus': 2200.0, 'poisson_ratio': 0.35,
            'density': 1050.0, 'yield_strength': 40.0},
}


def _normalize_faces(faces) -> "list[str] | None":
    """Accept ['Face3', 'Face5'], [3, 5] or a single 'Face3' / 3 and return
    FreeCAD sub-element names. None means the input could not be read."""
    if faces is None:
        return None
    if isinstance(faces, (str, int)):
        faces = [faces]
    out = []
    for f in faces:
        if isinstance(f, bool):
            return None
        if isinstance(f, int):
            if f < 1:
                return None
            out.append(f"Face{f}")
        elif isinstance(f, str) and f.startswith('Face') and f[4:].isdigit() and int(f[4:]) >= 1:
            out.append(f)
        elif isinstance(f, str) and f.isdigit() and int(f) >= 1:
            out.append(f"Face{f}")
        else:
            return None
    return out or None


def _unit_vector(direction) -> "tuple[float, float, float] | None":
    """[x, y, z] -> normalized tuple, or None for anything not a non-zero
    3-vector."""
    try:
        x, y, z = (float(c) for c in direction)
    except (TypeError, ValueError):
        return None
    length = math.sqrt(x * x + y * y + z * z)
    if length < 1e-12:
        return None
    return (x / length, y / length, z / length)


def _positive_number(value) -> "float | None":
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(v) or math.isinf(v) or v <= 0:
        return None
    return v


class FemOpsHandler(BaseHandler):
    """Handler for FEM workbench operations (linear static, CalculiX)."""

    _ALLOWED_OPERATIONS = frozenset({
        "create_analysis", "set_material", "create_mesh", "add_fixed",
        "add_force", "add_pressure", "add_self_weight", "check_analysis",
        "run_analysis", "get_results", "list_material_presets",
    })

    # ------------------------------------------------------------------
    # Shared resolution helpers
    # ------------------------------------------------------------------

    def _resolve_analysis(self, args: dict[str, Any]):
        """(doc, analysis, error) for args['analysis_name'] (default 'Analysis')."""
        name = args.get('analysis_name') or 'Analysis'
        return self.resolve_object(name, noun='FEM analysis', type_id=_ANALYSIS_TYPES)

    def _members(self, analysis, predicate):
        return [m for m in analysis.Group if predicate(m)]

    @staticmethod
    def _is_solver(obj) -> bool:
        tid = getattr(obj, 'TypeId', '')
        return 'Solver' in tid and hasattr(obj, 'AnalysisType')

    @staticmethod
    def _is_material(obj) -> bool:
        return hasattr(obj, 'Material') and isinstance(getattr(obj, 'Material', None), dict) \
            and 'Material' in getattr(obj, 'TypeId', '')

    @staticmethod
    def _is_mesh(obj) -> bool:
        return hasattr(obj, 'FemMesh') and hasattr(obj, 'CharacteristicLengthMax')

    @staticmethod
    def _is_result(obj) -> bool:
        return 'Result' in getattr(obj, 'TypeId', '') and hasattr(obj, 'vonMises')

    @staticmethod
    def _is_constraint(obj) -> bool:
        return 'Constraint' in getattr(obj, 'TypeId', '')

    def _resolve_face_refs(self, args: dict[str, Any], doc):
        """(obj, face_names, error) for args['object_name'] + args['faces'],
        checking every face index exists on the shape."""
        object_name = args.get('object_name', '')
        if not object_name:
            return None, None, "Error: object_name is required (the solid the faces belong to)"
        _, obj, err = self.resolve_object(object_name, doc=doc, attr='Shape')
        if err:
            return None, None, f"Error: {err}"
        faces = _normalize_faces(args.get('faces'))
        if not faces:
            return None, None, (
                "Error: faces is required -- a list like ['Face1', 'Face6'] or [1, 6]. "
                "Use measurement_operations(operation='list_faces') to see each face's "
                "normal, centroid and area.")
        n_faces = len(obj.Shape.Faces)
        bad = [f for f in faces if int(f[4:]) > n_faces]
        if bad:
            return None, None, f"Error: {object_name} has {n_faces} faces; not found: {', '.join(bad)}"
        return obj, faces, None

    def _add_to_analysis(self, analysis, obj):
        analysis.addObject(obj)

    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------

    def list_material_presets(self, args: dict[str, Any]) -> str:
        """Return the built-in nominal material presets."""
        return json.dumps({
            "presets": MATERIAL_PRESETS,
            "units": {"youngs_modulus": "MPa", "yield_strength": "MPa", "density": "kg/m^3"},
            "note": "Nominal bulk values. FDM-printed parts are anisotropic and usually weaker.",
        })

    def create_analysis(self, args: dict[str, Any]) -> str:
        """Create a FEM analysis container with a CalculiX solver inside it."""
        try:
            doc = self.get_document()
            if not doc:
                return NO_ACTIVE_DOCUMENT_ERROR
            import ObjectsFem

            name = args.get('analysis_name') or 'Analysis'
            if doc.getObject(name) is not None:
                return f"Error: an object named {name} already exists in {doc.Name}"
            analysis = ObjectsFem.makeAnalysis(doc, name)
            solver_factory = (getattr(ObjectsFem, 'makeSolverCalculiXCcxTools', None)
                              or getattr(ObjectsFem, 'makeSolverCalculixCcxTools', None))
            if solver_factory is None:
                return "Error: this FreeCAD's ObjectsFem has no CalculiX solver factory"
            solver = solver_factory(doc, args.get('solver_name') or 'CalculiX')
            solver.AnalysisType = 'static'
            self._add_to_analysis(analysis, solver)
            self.recompute(doc)
            return json.dumps({
                "analysis": analysis.Name,
                "solver": solver.Name,
                "analysis_type": "static",
                "next": "set_material, create_mesh, add_fixed, add_force/add_pressure, run_analysis",
            })
        except Exception as e:
            return f"Error creating FEM analysis: {e}"

    def set_material(self, args: dict[str, Any]) -> str:
        """Create or update the solid material of an analysis.

        Either `preset` (see list_material_presets) or explicit
        youngs_modulus (MPa) + poisson_ratio; explicit values override the
        preset's. density (kg/m^3) is needed only for add_self_weight;
        yield_strength (MPa) enables the safety factor in run_analysis.
        """
        try:
            doc, analysis, err = self._resolve_analysis(args)
            if err:
                return f"Error: {err}"
            import ObjectsFem

            values = {}
            preset = args.get('preset')
            if preset:
                key = str(preset).lower()
                if key not in MATERIAL_PRESETS:
                    return (f"Error: unknown material preset {preset!r}. "
                            f"Available: {', '.join(sorted(MATERIAL_PRESETS))}")
                values.update(MATERIAL_PRESETS[key])
            for k in ('youngs_modulus', 'poisson_ratio', 'density', 'yield_strength'):
                if args.get(k) is not None:
                    values[k] = args[k]
            if args.get('display_name'):
                values['name'] = args['display_name']

            e_mod = _positive_number(values.get('youngs_modulus'))
            if e_mod is None:
                return "Error: youngs_modulus (MPa, > 0) is required, directly or via preset"
            try:
                nu = float(values.get('poisson_ratio'))
            except (TypeError, ValueError):
                return "Error: poisson_ratio is required, directly or via preset"
            if not (0.0 <= nu < 0.5):
                return f"Error: poisson_ratio must be in [0, 0.5); got {nu}"
            density = values.get('density')
            if density is not None and _positive_number(density) is None:
                return f"Error: density must be > 0 kg/m^3; got {density}"
            yield_strength = values.get('yield_strength')
            if yield_strength is not None and _positive_number(yield_strength) is None:
                return f"Error: yield_strength must be > 0 MPa; got {yield_strength}"

            mat_name = args.get('material_name') or 'Material'
            mat = doc.getObject(mat_name)
            created = False
            if mat is None:
                mat = ObjectsFem.makeMaterialSolid(doc, mat_name)
                created = True
            elif not self._is_material(mat):
                return f"Error: {mat_name} exists and is not a FEM material"

            card = dict(mat.Material)
            card['Name'] = str(values.get('name') or mat_name)
            card['YoungsModulus'] = f"{float(e_mod)} MPa"
            card['PoissonRatio'] = str(nu)
            if density is not None:
                card['Density'] = f"{float(density)} kg/m^3"
            if yield_strength is not None:
                # Not read by CalculiX; kept on the card so run_analysis can
                # report a safety factor from the document alone.
                card['YieldStrength'] = f"{float(yield_strength)} MPa"
            mat.Material = card
            if created or mat not in analysis.Group:
                self._add_to_analysis(analysis, mat)
            self.recompute(doc)
            return json.dumps({"material": mat.Name, "created": created, "card": dict(mat.Material)})
        except Exception as e:
            return f"Error setting FEM material: {e}"

    def create_mesh(self, args: dict[str, Any]) -> str:
        """Mesh a solid with Gmsh and add the mesh to the analysis.

        max_size (mm) is the target element size; element_order '2nd'
        (default) avoids the shear locking of linear tetrahedra in bending.
        Re-running with the same mesh_name re-meshes in place.
        """
        try:
            doc, analysis, err = self._resolve_analysis(args)
            if err:
                return f"Error: {err}"
            object_name = args.get('object_name', '')
            if not object_name:
                return "Error: object_name (the solid to mesh) is required"
            _, shape_obj, err = self.resolve_object(object_name, doc=doc, attr='Shape')
            if err:
                return f"Error: {err}"
            if not shape_obj.Shape.Solids:
                return f"Error: {object_name} has no solids to mesh"

            max_size = _positive_number(args.get('max_size'))
            if max_size is None:
                bb = shape_obj.Shape.BoundBox
                # About 20 elements across the smallest dimension's scale,
                # bounded so thin parts don't explode the node count.
                max_size = max(min(bb.XLength, bb.YLength, bb.ZLength) / 4.0, bb.DiagonalLength / 40.0)
            min_size = args.get('min_size', 0.0) or 0.0
            try:
                min_size = float(min_size)
            except (TypeError, ValueError):
                return f"Error: min_size must be a number; got {args.get('min_size')!r}"
            if min_size < 0 or min_size > max_size:
                return f"Error: min_size must be between 0 and max_size ({max_size}); got {min_size}"
            order = str(args.get('element_order', '2nd'))
            if order not in ('1st', '2nd'):
                return f"Error: element_order must be '1st' or '2nd'; got {order!r}"

            import ObjectsFem
            from femmesh.gmshtools import GmshTools

            mesh_name = args.get('mesh_name') or 'Mesh'
            mesh = doc.getObject(mesh_name)
            if mesh is None:
                mesh = ObjectsFem.makeMeshGmsh(doc, mesh_name)
                self._add_to_analysis(analysis, mesh)
            elif not self._is_mesh(mesh):
                return f"Error: {mesh_name} exists and is not a Gmsh FEM mesh"
            mesh.Shape = shape_obj
            mesh.CharacteristicLengthMax = max_size
            mesh.CharacteristicLengthMin = min_size
            mesh.ElementOrder = order
            self.recompute(doc)

            t0 = time.time()
            gmsh_error = GmshTools(mesh).create_mesh()
            elapsed = time.time() - t0
            fem_mesh = mesh.FemMesh
            if gmsh_error:
                return f"Error: Gmsh failed meshing {object_name}: {gmsh_error}"
            if fem_mesh.VolumeCount == 0:
                return (f"Error: Gmsh produced no volume elements for {object_name} "
                        f"(nodes={fem_mesh.NodeCount}). Check the solid with "
                        "measurement_operations(operation='check_solid').")
            self.recompute(doc)
            return json.dumps({
                "mesh": mesh.Name,
                "shape": shape_obj.Name,
                "nodes": fem_mesh.NodeCount,
                "volume_elements": fem_mesh.VolumeCount,
                "element_order": order,
                "max_size_mm": round(max_size, 4),
                "min_size_mm": min_size,
                "seconds": round(elapsed, 2),
            })
        except Exception as e:
            return f"Error creating FEM mesh: {e}"

    def add_fixed(self, args: dict[str, Any]) -> str:
        """Fix (all translations zero) the given faces of a solid."""
        try:
            doc, analysis, err = self._resolve_analysis(args)
            if err:
                return f"Error: {err}"
            obj, faces, err = self._resolve_face_refs(args, doc)
            if err:
                return err
            import ObjectsFem

            con = ObjectsFem.makeConstraintFixed(doc, args.get('constraint_name') or 'Fixed')
            con.References = [(obj, f) for f in faces]
            self._add_to_analysis(analysis, con)
            self.recompute(doc)
            return json.dumps({"constraint": con.Name, "type": "fixed",
                               "object": obj.Name, "faces": faces})
        except Exception as e:
            return f"Error adding fixed constraint: {e}"

    def add_force(self, args: dict[str, Any]) -> str:
        """Apply a total force (N) spread over the given faces, along an
        arbitrary direction vector.

        FreeCAD's ConstraintForce takes its direction from an edge or face
        of some object, not from a vector. A hidden Part::Line helper along
        `direction` provides that edge, so any direction works without
        hunting for a parallel model edge.
        """
        try:
            doc, analysis, err = self._resolve_analysis(args)
            if err:
                return f"Error: {err}"
            obj, faces, err = self._resolve_face_refs(args, doc)
            if err:
                return err
            force = _positive_number(args.get('force'))
            if force is None:
                return "Error: force (total, in N, > 0) is required; set its sense with direction"
            direction = _unit_vector(args.get('direction'))
            if direction is None:
                return "Error: direction must be a non-zero [x, y, z] vector, e.g. [0, 0, -1]"
            import ObjectsFem

            name = args.get('constraint_name') or 'Force'
            con = ObjectsFem.makeConstraintForce(doc, name)
            helper = doc.addObject('Part::Line', f"{con.Name}_Direction")
            helper.X1, helper.Y1, helper.Z1 = 0.0, 0.0, 0.0
            helper.X2, helper.Y2, helper.Z2 = direction
            helper.Visibility = False
            self.recompute(doc)

            con.References = [(obj, f) for f in faces]
            con.Force = f"{force} N"
            con.Direction = (helper, ['Edge1'])
            con.Reversed = False
            self._add_to_analysis(analysis, con)
            self.recompute(doc)

            dv = con.DirectionVector
            applied = (dv.x, dv.y, dv.z)
            if sum(a * b for a, b in zip(applied, direction, strict=True)) < 0.999:
                return (f"Error: force direction did not take (requested {direction}, "
                        f"FreeCAD reports {applied}); constraint {con.Name} left in the "
                        "document for inspection")
            return json.dumps({"constraint": con.Name, "type": "force", "object": obj.Name,
                               "faces": faces, "force_N": force,
                               "direction": [round(c, 6) for c in direction],
                               "direction_helper": helper.Name})
        except Exception as e:
            return f"Error adding force: {e}"

    def add_pressure(self, args: dict[str, Any]) -> str:
        """Apply a pressure (MPa) normal to the given faces. Positive
        pressure pushes into the material; reversed=True pulls."""
        try:
            doc, analysis, err = self._resolve_analysis(args)
            if err:
                return f"Error: {err}"
            obj, faces, err = self._resolve_face_refs(args, doc)
            if err:
                return err
            pressure = _positive_number(args.get('pressure'))
            if pressure is None:
                return "Error: pressure (MPa, > 0) is required; use reversed=true for suction"
            import ObjectsFem

            con = ObjectsFem.makeConstraintPressure(doc, args.get('constraint_name') or 'Pressure')
            con.References = [(obj, f) for f in faces]
            con.Pressure = f"{pressure} MPa"
            con.Reversed = bool(args.get('reversed', False))
            self._add_to_analysis(analysis, con)
            self.recompute(doc)
            return json.dumps({"constraint": con.Name, "type": "pressure", "object": obj.Name,
                               "faces": faces, "pressure_MPa": pressure,
                               "reversed": con.Reversed})
        except Exception as e:
            return f"Error adding pressure: {e}"

    def add_self_weight(self, args: dict[str, Any]) -> str:
        """Add gravity load. Needs a material with density."""
        try:
            doc, analysis, err = self._resolve_analysis(args)
            if err:
                return f"Error: {err}"
            direction = _unit_vector(args.get('direction', [0, 0, -1]))
            if direction is None:
                return "Error: direction must be a non-zero [x, y, z] vector"
            g = args.get('acceleration', 9806.65)
            if _positive_number(g) is None:
                return f"Error: acceleration must be > 0 mm/s^2; got {g}"
            import ObjectsFem

            con = ObjectsFem.makeConstraintSelfWeight(doc, args.get('constraint_name') or 'SelfWeight')
            con.GravityDirection = FreeCAD.Vector(*direction)
            con.GravityAcceleration = f"{float(g)} mm/s^2"
            self._add_to_analysis(analysis, con)
            self.recompute(doc)
            return json.dumps({"constraint": con.Name, "type": "self_weight",
                               "direction": list(direction), "acceleration_mm_s2": float(g)})
        except Exception as e:
            return f"Error adding self weight: {e}"

    def _setup_report(self, analysis) -> dict:
        solvers = self._members(analysis, self._is_solver)
        materials = self._members(analysis, self._is_material)
        meshes = self._members(analysis, self._is_mesh)
        constraints = self._members(analysis, self._is_constraint)
        fixed = [c for c in constraints if 'Fixed' in c.TypeId or 'Displacement' in c.TypeId]
        loads = [c for c in constraints if c not in fixed]

        problems = []
        if not solvers:
            problems.append("no solver (create_analysis adds one)")
        if not materials:
            problems.append("no material (set_material)")
        if not meshes:
            problems.append("no mesh (create_mesh)")
        else:
            for m in meshes:
                if m.FemMesh.VolumeCount == 0:
                    problems.append(f"mesh {m.Name} is empty (re-run create_mesh)")
        if not fixed:
            problems.append("no support (add_fixed): the model would be free to move")
        if not loads:
            problems.append("no load (add_force, add_pressure or add_self_weight)")
        for c in constraints:
            if not hasattr(c, 'GravityDirection') and not c.References:
                problems.append(f"constraint {c.Name} has no faces")
        if (any(hasattr(c, 'GravityDirection') for c in constraints)
                and not any('Density' in m.Material for m in materials)):
            problems.append("self weight needs a material density")

        return {
            "analysis": analysis.Name,
            "solver": [s.Name for s in solvers],
            "materials": [m.Name for m in materials],
            "meshes": [{"name": m.Name, "nodes": m.FemMesh.NodeCount,
                        "volume_elements": m.FemMesh.VolumeCount} for m in meshes],
            "supports": [c.Name for c in fixed],
            "loads": [c.Name for c in loads],
            "results": [r.Name for r in self._members(analysis, self._is_result)],
            "ready": not problems,
            "problems": problems,
        }

    def check_analysis(self, args: dict[str, Any]) -> str:
        """Report what the analysis contains and what is missing to run it."""
        try:
            _, analysis, err = self._resolve_analysis(args)
            if err:
                return f"Error: {err}"
            return json.dumps(self._setup_report(analysis))
        except Exception as e:
            return f"Error checking FEM analysis: {e}"

    def _yield_strength(self, analysis) -> "float | None":
        for m in self._members(analysis, self._is_material):
            raw = m.Material.get('YieldStrength')
            if raw:
                # set_material writes "<value> MPa"; anything else is a
                # hand-edited card and goes through FreeCAD's unit parser.
                text = str(raw).strip()
                if text.endswith(' MPa'):
                    with contextlib.suppress(ValueError):
                        return float(text[:-4])
                try:
                    return float(FreeCAD.Units.Quantity(text).getValueAs('MPa'))
                except Exception:
                    return None
        return None

    def _result_summary(self, analysis, result_obj) -> dict:
        # vonMises / DisplacementLengths can be None on a degenerate run.
        vm = list(getattr(result_obj, 'vonMises', None) or [])
        disp = list(getattr(result_obj, 'DisplacementLengths', None) or [])
        summary = {
            "result_object": result_obj.Name,
            "node_count": len(vm),
            "max_von_mises_MPa": max(vm) if vm else None,
            "min_von_mises_MPa": min(vm) if vm else None,
            "max_displacement_mm": max(disp) if disp else None,
        }
        vectors = list(getattr(result_obj, 'DisplacementVectors', None) or [])
        if vectors:
            summary["max_displacement_components_mm"] = {
                axis: max((getattr(v, axis) for v in vectors), key=abs) for axis in ('x', 'y', 'z')
            }
            if vm:
                i_max = max(range(len(vm)), key=vm.__getitem__)
                mesh = getattr(result_obj, 'Mesh', None)
                fem_mesh = getattr(mesh, 'FemMesh', None) if mesh else None
                node_numbers = list(getattr(result_obj, 'NodeNumbers', None) or [])
                if fem_mesh is not None and i_max < len(node_numbers):
                    try:
                        p = fem_mesh.getNodeById(node_numbers[i_max])
                        summary["max_von_mises_location_mm"] = [round(p.x, 3), round(p.y, 3), round(p.z, 3)]
                    except Exception:
                        pass
        sy = self._yield_strength(analysis)
        if sy is not None and summary["max_von_mises_MPa"]:
            summary["yield_strength_MPa"] = sy
            summary["safety_factor"] = round(sy / summary["max_von_mises_MPa"], 3)
        return summary

    def run_analysis(self, args: dict[str, Any]) -> str:
        """Write the CalculiX input, run the solver, load and summarize results.

        Fails early, with the list from check_analysis, when the setup is
        incomplete. working_dir keeps the .inp/.frd/.dat files for
        inspection after a solver failure. Failures are "Error: ..."
        strings, like every other handler, so the async job reports them
        as errors rather than as a successful result.
        """
        work_dir = None
        stage = "initialization"
        try:
            stage = "analysis lookup"
            doc, analysis, err = self._resolve_analysis(args)
            if err:
                return f"Error: {err}"

            stage = "setup check"
            report = self._setup_report(analysis)
            if not report["ready"]:
                return "Error: analysis not ready to run: " + "; ".join(report["problems"])
            solver = self._members(analysis, self._is_solver)[0]

            stage = "femtools import"
            from femtools import ccxtools

            stage = "solver setup"
            fea = ccxtools.FemToolsCcx(analysis=analysis, solver=solver)
            fea.update_objects()
            work_dir = args.get('working_dir') or tempfile.mkdtemp(prefix='freecad_mcp_fem_')
            os.makedirs(work_dir, exist_ok=True)
            fea.setup_working_dir(work_dir)
            fea.setup_ccx()

            stage = "prerequisite check"
            prereq_msg = fea.check_prerequisites()
            if prereq_msg:
                return f"Error: CalculiX prerequisites failed: {prereq_msg} (working_dir: {work_dir})"

            stage = "solver execution"
            fea.purge_results()
            t0 = time.time()
            # FemToolsCcx.run() reports CalculiX failures by returning False
            # rather than raising; None is returned on success in some
            # FreeCAD versions, so only an explicit False is a failure.
            if fea.run() is False:
                return ("Error: CalculiX solver run failed; inspect the .dat/.frd/.sta "
                        f"files in {work_dir}")
            solve_seconds = time.time() - t0

            stage = "result loading"
            fea.load_results()
            results = self._members(analysis, self._is_result)
            if not results:
                return f"Error: solver ran but produced no result object (working_dir: {work_dir})"

            stage = "result extraction"
            # load_results may add more than one result object (one per
            # output step); the last one is the final state.
            summary = self._result_summary(analysis, results[-1])
            self.recompute(doc)
            summary.update({
                "success": True,
                "analysis": analysis.Name,
                "solve_seconds": round(solve_seconds, 2),
                "working_dir": work_dir,
            })
            return json.dumps(summary)
        except Exception as e:
            return (f"Error: FEM analysis failed during {stage}: {type(e).__name__}: {e}"
                    f" (working_dir: {work_dir})")

    def get_results(self, args: dict[str, Any]) -> str:
        """Summarize the latest result already in the analysis (no solve)."""
        try:
            _, analysis, err = self._resolve_analysis(args)
            if err:
                return f"Error: {err}"
            results = self._members(analysis, self._is_result)
            if not results:
                return f"Error: {analysis.Name} has no results yet; call run_analysis first"
            return json.dumps(self._result_summary(analysis, results[-1]))
        except Exception as e:
            return f"Error reading FEM results: {e}"
