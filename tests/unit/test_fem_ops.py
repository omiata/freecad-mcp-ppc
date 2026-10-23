"""Unit tests for FemOpsHandler (fem_operations).

FEM objects are plain fakes, not MagicMocks: the handler classifies
analysis members with hasattr() checks (a solver has AnalysisType, a mesh
has FemMesh, ...), and a MagicMock answers yes to every hasattr. ObjectsFem,
femmesh.gmshtools and femtools.ccxtools are stubbed into sys.modules per
test, because the handler imports them inside each method.

The live pipeline (real Gmsh + CalculiX against beam theory) is covered by
tests/integration/test_fem_ops.py.
"""

import ast
import json
import os
import sys
import types
import unittest

# _freecad_mocks puts AICopilot/ on sys.path and installs the FreeCAD mocks,
# so it must be imported before handlers.* when this file runs on its own.
from tests.unit._freecad_mocks import (  # isort: skip
    _Vec,
    assert_error_contains,
    assert_success_contains,
    make_handler,
    make_mock_doc,
    make_part_object,
    mock_FreeCAD,
    reset_mocks,
)

from handlers.fem_ops import MATERIAL_PRESETS, FemOpsHandler, _normalize_faces, _unit_vector  # isort: skip

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..'))
SERVER_PY = os.path.join(REPO_ROOT, 'freecad_mcp_server.py')


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class _Obj:
    TypeId = 'App::FeaturePython'

    def __init__(self, name):
        self.Name = name
        self.Label = name


class FakeAnalysis(_Obj):
    TypeId = 'Fem::FemAnalysis'

    def __init__(self, name):
        super().__init__(name)
        self.Group = []

    def addObject(self, obj):
        self.Group.append(obj)


class FakeSolver(_Obj):
    TypeId = 'Fem::FemSolverObjectPython'

    def __init__(self, name):
        super().__init__(name)
        self.AnalysisType = 'thermomech'


class FakeMaterial(_Obj):
    TypeId = 'App::MaterialObjectPython'

    def __init__(self, name):
        super().__init__(name)
        self.Material = {}


class FakeFemMesh:
    def __init__(self, nodes=0, volumes=0):
        self.NodeCount = nodes
        self.VolumeCount = volumes
        self.nodes = {}

    def getNodeById(self, i):
        return self.nodes[i]


class FakeMesh(_Obj):
    TypeId = 'Fem::FemMeshShapeBaseObjectPython'

    def __init__(self, name):
        super().__init__(name)
        self.FemMesh = FakeFemMesh()
        self.CharacteristicLengthMax = 0.0
        self.CharacteristicLengthMin = 0.0
        self.ElementOrder = '2nd'
        self.Shape = None


class FakeConstraint(_Obj):
    def __init__(self, name, type_id):
        super().__init__(name)
        self.TypeId = type_id
        self.References = []


class FakeForce(FakeConstraint):
    """Mimics ConstraintForce deriving DirectionVector from the Direction
    edge -- here, from the Part::Line helper's end point."""

    flip = False  # set True to simulate FreeCAD ignoring the requested sense

    def __init__(self, name):
        super().__init__(name, 'Fem::ConstraintForce')
        self.Force = None
        self.Reversed = None
        self.DirectionVector = _Vec(0, 0, 1)
        self._direction = None

    @property
    def Direction(self):
        return self._direction

    @Direction.setter
    def Direction(self, value):
        self._direction = value
        helper = value[0]
        s = -1.0 if FakeForce.flip else 1.0
        self.DirectionVector = _Vec(s * helper.X2, s * helper.Y2, s * helper.Z2)


class FakeSelfWeight(FakeConstraint):
    def __init__(self, name):
        super().__init__(name, 'Fem::ConstraintPython')
        self.GravityDirection = _Vec(0, 0, -1)
        self.GravityAcceleration = None


class FakeResult(_Obj):
    TypeId = 'Fem::FemResultObjectPython'

    def __init__(self, name, vm, disp_vectors, mesh=None):
        super().__init__(name)
        self.vonMises = vm
        self.DisplacementVectors = disp_vectors
        self.DisplacementLengths = [v.Length for v in disp_vectors]
        self.NodeNumbers = list(range(1, len(vm) + 1))
        self.Mesh = mesh


def _install_objectsfem(doc):
    """Stub ObjectsFem whose factories add fakes to doc.Objects."""
    mod = types.ModuleType('ObjectsFem')

    def factory(cls, *extra):
        def make(d, name=None):
            obj = cls(name or cls.__name__, *extra) if extra else cls(name or cls.__name__)
            doc.Objects.append(obj)
            return obj
        return make

    mod.makeAnalysis = factory(FakeAnalysis)
    mod.makeSolverCalculiXCcxTools = factory(FakeSolver)
    mod.makeMaterialSolid = factory(FakeMaterial)
    mod.makeMeshGmsh = factory(FakeMesh)
    mod.makeConstraintFixed = lambda d, name='Fixed': _append(doc, FakeConstraint(name, 'Fem::ConstraintFixed'))
    mod.makeConstraintPressure = lambda d, name='Pressure': _append(doc, FakeConstraint(name, 'Fem::ConstraintPressure'))
    mod.makeConstraintForce = lambda d, name='Force': _append(doc, FakeForce(name))
    mod.makeConstraintSelfWeight = lambda d, name='SelfWeight': _append(doc, FakeSelfWeight(name))
    sys.modules['ObjectsFem'] = mod
    return mod


def _append(doc, obj):
    doc.Objects.append(obj)
    return obj


def _install_gmsh(nodes=500, volumes=200, error=None):
    pkg = types.ModuleType('femmesh')
    mod = types.ModuleType('femmesh.gmshtools')

    class GmshTools:
        calls = []

        def __init__(self, mesh):
            self.mesh = mesh

        def create_mesh(self):
            GmshTools.calls.append(self.mesh)
            self.mesh.FemMesh = FakeFemMesh(nodes, volumes)
            return error

    mod.GmshTools = GmshTools
    pkg.gmshtools = mod
    sys.modules['femmesh'] = pkg
    sys.modules['femmesh.gmshtools'] = mod
    return GmshTools


def _install_ccx(run_return=True, prereq='', produce=True, raise_in=None):
    pkg = types.ModuleType('femtools')
    mod = types.ModuleType('femtools.ccxtools')

    class FemToolsCcx:
        def __init__(self, analysis, solver):
            self.analysis = analysis
            self.solver = solver

        def _maybe_raise(self, stage):
            if raise_in == stage:
                raise RuntimeError(f"boom in {stage}")

        def update_objects(self):
            self._maybe_raise('update_objects')

        def setup_working_dir(self, d):
            self.work_dir = d

        def setup_ccx(self):
            pass

        def check_prerequisites(self):
            return prereq

        def purge_results(self):
            self.analysis.Group = [m for m in self.analysis.Group if not isinstance(m, FakeResult)]

        def run(self):
            self._maybe_raise('run')
            return run_return

        def load_results(self):
            if not produce:
                return
            mesh = next(m for m in self.analysis.Group if isinstance(m, FakeMesh))
            mesh.FemMesh.nodes = {1: _Vec(0, 0, 0), 2: _Vec(0, 5, 10), 3: _Vec(100, 5, 5)}
            # Two output steps, like CalculiX writes: only the last is final.
            self.analysis.addObject(FakeResult('CCX_Results', [1.0, 1.0, 1.0],
                                               [_Vec(), _Vec(), _Vec()], mesh))
            self.analysis.addObject(FakeResult(
                'CCX_Results001', [12.0, 60.0, 0.5],
                [_Vec(0, 0, 0), _Vec(0, 0, -0.05), _Vec(0.001, 0, -0.19)], mesh))

    mod.FemToolsCcx = FemToolsCcx
    pkg.ccxtools = mod
    sys.modules['femtools'] = pkg
    sys.modules['femtools.ccxtools'] = mod
    return FemToolsCcx


class _FemTestBase(unittest.TestCase):
    def setUp(self):
        reset_mocks()
        self._saved = {k: sys.modules.get(k) for k in
                       ('ObjectsFem', 'femmesh', 'femmesh.gmshtools', 'femtools', 'femtools.ccxtools')}
        self.beam = make_part_object('Beam', faces=6, bbox=(100.0, 10.0, 10.0))
        self.beam.Shape.BoundBox.DiagonalLength = (100.0 ** 2 + 10.0 ** 2 + 10.0 ** 2) ** 0.5
        self.doc = make_mock_doc([self.beam])
        mock_FreeCAD.ActiveDocument = self.doc
        _install_objectsfem(self.doc)
        self.handler = make_handler(FemOpsHandler)
        FakeForce.flip = False

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
        FakeForce.flip = False

    def call(self, op, **args):
        return getattr(self.handler, op)({'operation': op, **args})

    def json_call(self, op, **args):
        result = self.call(op, **args)
        assert_success_contains(self, result)
        return json.loads(result)

    def ready_analysis(self, with_yield=True):
        """Analysis + material + mesh + support + load, ready to run."""
        _install_gmsh()
        self.json_call('create_analysis')
        extra = {} if with_yield else {'yield_strength': None}
        self.json_call('set_material', preset='steel_s235', **extra)
        if not with_yield:
            self.doc.getObject('Material').Material.pop('YieldStrength', None)
        self.json_call('create_mesh', object_name='Beam', max_size=4)
        self.json_call('add_fixed', object_name='Beam', faces=['Face1'])
        self.json_call('add_force', object_name='Beam', faces=[2], force=100, direction=[0, 0, -1])
        return self.doc.getObject('Analysis')


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

class TestHelpers(unittest.TestCase):
    def test_normalize_faces_accepts_names_ints_and_digit_strings(self):
        self.assertEqual(_normalize_faces(['Face1', 3, '5']), ['Face1', 'Face3', 'Face5'])
        self.assertEqual(_normalize_faces('Face2'), ['Face2'])
        self.assertEqual(_normalize_faces(4), ['Face4'])

    def test_normalize_faces_rejects_zero_bool_and_junk(self):
        for bad in ([0], ['Face0'], [True], ['Edge1'], ['Face'], [], None, [1.5]):
            self.assertIsNone(_normalize_faces(bad), bad)

    def test_unit_vector(self):
        self.assertEqual(_unit_vector([0, 0, -5]), (0.0, 0.0, -1.0))
        x, y, z = _unit_vector([1, 1, 0])
        self.assertAlmostEqual(x, 2 ** -0.5)
        for bad in ([0, 0, 0], [1, 2], 'z', None, [1, 'a', 0]):
            self.assertIsNone(_unit_vector(bad), bad)


# ---------------------------------------------------------------------------
# create_analysis / set_material
# ---------------------------------------------------------------------------

class TestCreateAnalysis(_FemTestBase):
    def test_creates_analysis_with_static_calculix_solver(self):
        data = self.json_call('create_analysis')
        analysis = self.doc.getObject(data['analysis'])
        self.assertEqual(len(analysis.Group), 1)
        self.assertIsInstance(analysis.Group[0], FakeSolver)
        self.assertEqual(analysis.Group[0].AnalysisType, 'static')

    def test_refuses_existing_name(self):
        self.json_call('create_analysis')
        assert_error_contains(self, self.call('create_analysis'), 'already exists')

    def test_needs_document(self):
        mock_FreeCAD.ActiveDocument = None
        assert_error_contains(self, self.call('create_analysis'), 'No active document')


class TestSetMaterial(_FemTestBase):
    def setUp(self):
        super().setUp()
        self.json_call('create_analysis')

    def test_preset_writes_card_with_units(self):
        data = self.json_call('set_material', preset='steel_s235')
        card = data['card']
        self.assertEqual(card['YoungsModulus'], '210000.0 MPa')
        self.assertEqual(card['PoissonRatio'], '0.3')
        self.assertEqual(card['Density'], '7850.0 kg/m^3')
        self.assertEqual(card['YieldStrength'], '235.0 MPa')
        self.assertTrue(data['created'])

    def test_explicit_values_override_preset(self):
        data = self.json_call('set_material', preset='pla', youngs_modulus=2800, display_name='PLA impreso')
        self.assertEqual(data['card']['YoungsModulus'], '2800.0 MPa')
        self.assertEqual(data['card']['Name'], 'PLA impreso')

    def test_preset_name_is_case_insensitive(self):
        self.json_call('set_material', preset='PETG')

    def test_reusing_name_updates_instead_of_duplicating(self):
        self.json_call('set_material', preset='pla')
        data = self.json_call('set_material', preset='abs')
        self.assertFalse(data['created'])
        analysis = self.doc.getObject('Analysis')
        self.assertEqual(sum(isinstance(m, FakeMaterial) for m in analysis.Group), 1)
        self.assertEqual(data['card']['YoungsModulus'], '2200.0 MPa')

    def test_rejects_unknown_preset(self):
        assert_error_contains(self, self.call('set_material', preset='unobtainium'),
                              'unknown material preset', 'steel_s235')

    def test_requires_modulus_and_poisson(self):
        assert_error_contains(self, self.call('set_material', poisson_ratio=0.3), 'youngs_modulus')
        assert_error_contains(self, self.call('set_material', youngs_modulus=1000), 'poisson_ratio')

    def test_poisson_boundaries(self):
        self.json_call('set_material', youngs_modulus=1000, poisson_ratio=0.0)
        assert_error_contains(self, self.call('set_material', youngs_modulus=1000, poisson_ratio=0.5), '[0, 0.5)')
        assert_error_contains(self, self.call('set_material', youngs_modulus=1000, poisson_ratio=-0.1), '[0, 0.5)')

    def test_rejects_non_positive_density_and_yield(self):
        assert_error_contains(self, self.call('set_material', preset='pla', density=0), 'density')
        assert_error_contains(self, self.call('set_material', preset='pla', yield_strength=-1), 'yield_strength')

    def test_unknown_analysis(self):
        assert_error_contains(self, self.call('set_material', analysis_name='Nope', preset='pla'),
                              'FEM analysis not found')

    def test_presets_listed(self):
        data = self.json_call('list_material_presets')
        self.assertEqual(set(data['presets']), set(MATERIAL_PRESETS))


# ---------------------------------------------------------------------------
# create_mesh
# ---------------------------------------------------------------------------

class TestCreateMesh(_FemTestBase):
    def setUp(self):
        super().setUp()
        self.json_call('create_analysis')

    def test_meshes_with_second_order_by_default(self):
        gmsh = _install_gmsh(nodes=2344, volumes=1121)
        data = self.json_call('create_mesh', object_name='Beam', max_size=4)
        self.assertEqual((data['nodes'], data['volume_elements']), (2344, 1121))
        mesh = self.doc.getObject('Mesh')
        self.assertEqual(mesh.ElementOrder, '2nd')
        self.assertEqual(mesh.CharacteristicLengthMax, 4.0)
        self.assertIs(mesh.Shape, self.beam)
        self.assertIn(mesh, self.doc.getObject('Analysis').Group)
        self.assertEqual(len(gmsh.calls), 1)

    def test_remesh_reuses_object(self):
        _install_gmsh()
        self.json_call('create_mesh', object_name='Beam', max_size=4)
        self.json_call('create_mesh', object_name='Beam', max_size=2, element_order='1st')
        analysis = self.doc.getObject('Analysis')
        self.assertEqual(sum(isinstance(m, FakeMesh) for m in analysis.Group), 1)
        self.assertEqual(self.doc.getObject('Mesh').ElementOrder, '1st')

    def test_default_size_derived_from_part(self):
        _install_gmsh()
        data = self.json_call('create_mesh', object_name='Beam')
        # bbox 100x10x10: max(10/4, diag/40) = max(2.5, 100.995/40) = 2.5249
        self.assertAlmostEqual(data['max_size_mm'], (100.0 ** 2 + 200.0) ** 0.5 / 40.0, places=3)

    def test_gmsh_error_reported(self):
        _install_gmsh(error='surface mesh failed')
        assert_error_contains(self, self.call('create_mesh', object_name='Beam', max_size=4),
                              'Gmsh failed', 'surface mesh failed')

    def test_empty_volume_mesh_is_an_error(self):
        _install_gmsh(nodes=40, volumes=0)
        assert_error_contains(self, self.call('create_mesh', object_name='Beam', max_size=4),
                              'no volume elements')

    def test_argument_validation(self):
        _install_gmsh()
        assert_error_contains(self, self.call('create_mesh'), 'object_name')
        assert_error_contains(self, self.call('create_mesh', object_name='Beam', element_order='3rd'),
                              'element_order')
        assert_error_contains(self, self.call('create_mesh', object_name='Beam', max_size=2, min_size=3),
                              'min_size')
        assert_error_contains(self, self.call('create_mesh', object_name='Ghost'), 'not found')

    def test_needs_solid(self):
        _install_gmsh()
        self.beam.Shape.Solids = []
        assert_error_contains(self, self.call('create_mesh', object_name='Beam'), 'no solids')


# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------

class TestConstraints(_FemTestBase):
    def setUp(self):
        super().setUp()
        self.json_call('create_analysis')

    def test_fixed_references_faces(self):
        data = self.json_call('add_fixed', object_name='Beam', faces=['Face1', 6])
        con = self.doc.getObject(data['constraint'])
        self.assertEqual(con.References, [(self.beam, 'Face1'), (self.beam, 'Face6')])
        self.assertIn(con, self.doc.getObject('Analysis').Group)

    def test_face_out_of_range(self):
        assert_error_contains(self, self.call('add_fixed', object_name='Beam', faces=['Face7']),
                              'has 6 faces', 'Face7')

    def test_faces_required_points_to_list_faces(self):
        assert_error_contains(self, self.call('add_fixed', object_name='Beam'), 'list_faces')
        assert_error_contains(self, self.call('add_fixed', faces=[1]), 'object_name')

    def test_force_sets_magnitude_and_arbitrary_direction(self):
        data = self.json_call('add_force', object_name='Beam', faces=[2], force=100, direction=[1, 0, -1])
        con = self.doc.getObject(data['constraint'])
        self.assertEqual(con.Force, '100.0 N')
        self.assertFalse(con.Reversed)
        helper = self.doc.getObject(data['direction_helper'])
        self.assertEqual(helper.TypeId, 'Part::Line')
        self.assertFalse(helper.Visibility)
        self.assertAlmostEqual(data['direction'][0], 2 ** -0.5, places=5)

    def test_force_direction_check_catches_flipped_sense(self):
        FakeForce.flip = True
        assert_error_contains(self, self.call('add_force', object_name='Beam', faces=[2], force=100,
                                              direction=[0, 0, -1]), 'direction did not take')

    def test_force_validation(self):
        for bad in (0, -5, 'x', None):
            assert_error_contains(self, self.call('add_force', object_name='Beam', faces=[2], force=bad,
                                                  direction=[0, 0, 1]), 'force')
        assert_error_contains(self, self.call('add_force', object_name='Beam', faces=[2], force=1,
                                              direction=[0, 0, 0]), 'direction')

    def test_pressure(self):
        data = self.json_call('add_pressure', object_name='Beam', faces=[3], pressure=0.5, reversed=True)
        con = self.doc.getObject(data['constraint'])
        self.assertEqual(con.Pressure, '0.5 MPa')
        self.assertTrue(con.Reversed)
        assert_error_contains(self, self.call('add_pressure', object_name='Beam', faces=[3], pressure=0),
                              'pressure')

    def test_self_weight(self):
        data = self.json_call('add_self_weight', direction=[0, -2, 0])
        con = self.doc.getObject(data['constraint'])
        self.assertEqual(con.GravityDirection, _Vec(0, -1, 0))
        self.assertEqual(con.GravityAcceleration, '9806.65 mm/s^2')


# ---------------------------------------------------------------------------
# check / run / results
# ---------------------------------------------------------------------------

class TestCheckAnalysis(_FemTestBase):
    def test_empty_analysis_lists_every_missing_piece(self):
        self.json_call('create_analysis')
        data = self.json_call('check_analysis')
        self.assertFalse(data['ready'])
        text = ' '.join(data['problems'])
        for piece in ('material', 'mesh', 'support', 'load'):
            self.assertIn(piece, text)

    def test_ready_analysis(self):
        self.ready_analysis()
        data = self.json_call('check_analysis')
        self.assertTrue(data['ready'], data['problems'])
        self.assertEqual(data['supports'], ['Fixed'])
        self.assertEqual(data['loads'], ['Force'])

    def test_self_weight_without_density_flagged(self):
        self.ready_analysis()
        self.doc.getObject('Material').Material.pop('Density')
        self.json_call('add_self_weight')
        data = self.json_call('check_analysis')
        self.assertIn('density', ' '.join(data['problems']))


class TestRunAnalysis(_FemTestBase):
    def test_refuses_incomplete_setup(self):
        self.json_call('create_analysis')
        _install_ccx()
        assert_error_contains(self, self.call('run_analysis'), 'not ready', 'no material')

    def test_success_summarizes_final_step_with_safety_factor(self):
        self.ready_analysis()
        _install_ccx()
        data = self.json_call('run_analysis', working_dir=self._tmp())
        self.assertTrue(data['success'])
        self.assertEqual(data['result_object'], 'CCX_Results001')
        self.assertEqual(data['max_von_mises_MPa'], 60.0)
        self.assertAlmostEqual(data['max_displacement_mm'], (0.001 ** 2 + 0.19 ** 2) ** 0.5)
        self.assertEqual(data['max_displacement_components_mm']['z'], -0.19)
        self.assertEqual(data['max_von_mises_location_mm'], [0.0, 5.0, 10.0])
        self.assertEqual(data['yield_strength_MPa'], 235.0)
        self.assertAlmostEqual(data['safety_factor'], round(235.0 / 60.0, 3))

    def test_no_safety_factor_without_yield(self):
        self.ready_analysis(with_yield=False)
        _install_ccx()
        data = self.json_call('run_analysis', working_dir=self._tmp())
        self.assertNotIn('safety_factor', data)

    def test_solver_failure_is_error_string_with_working_dir(self):
        self.ready_analysis()
        _install_ccx(run_return=False)
        wd = self._tmp()
        result = self.call('run_analysis', working_dir=wd)
        self.assertTrue(result.startswith('Error'), result)
        assert_error_contains(self, result, 'CalculiX solver run failed', wd)

    def test_run_returning_none_counts_as_success(self):
        self.ready_analysis()
        _install_ccx(run_return=None)
        self.json_call('run_analysis', working_dir=self._tmp())

    def test_prerequisite_failure(self):
        self.ready_analysis()
        _install_ccx(prereq='no ccx binary')
        assert_error_contains(self, self.call('run_analysis', working_dir=self._tmp()),
                              'prerequisites failed', 'no ccx binary')

    def test_no_result_object(self):
        self.ready_analysis()
        _install_ccx(produce=False)
        assert_error_contains(self, self.call('run_analysis', working_dir=self._tmp()), 'no result object')

    def test_exception_reports_stage(self):
        self.ready_analysis()
        _install_ccx(raise_in='run')
        result = self.call('run_analysis', working_dir=self._tmp())
        self.assertTrue(result.startswith('Error'), result)
        assert_error_contains(self, result, 'solver execution', 'boom in run')

    def test_get_results_without_and_with_run(self):
        self.ready_analysis()
        assert_error_contains(self, self.call('get_results'), 'no results yet')
        _install_ccx()
        self.json_call('run_analysis', working_dir=self._tmp())
        data = self.json_call('get_results')
        self.assertEqual(data['max_von_mises_MPa'], 60.0)

    def _tmp(self):
        import tempfile
        d = tempfile.mkdtemp(prefix='fem_unit_')
        self.addCleanup(lambda: __import__('shutil').rmtree(d, ignore_errors=True))
        return d


# ---------------------------------------------------------------------------
# Schema parity with the bridge
# ---------------------------------------------------------------------------

class TestSchemaParity(unittest.TestCase):
    def test_operation_enum_matches_allowed_operations(self):
        with open(SERVER_PY, encoding='utf-8') as f:
            src = f.read()
        start = src.index('name="fem_operations"')
        end = src.index('types.Tool(', start)
        block = src[start:end]
        marker = block.index('"description": "FEM operation to perform"')
        list_start = block.index('[', block.index('"enum":', marker))
        list_end = block.index(']', list_start)
        schema_ops = set(ast.literal_eval(block[list_start:list_end + 1]))
        self.assertEqual(schema_ops, set(FemOpsHandler._ALLOWED_OPERATIONS))

    def test_every_allowed_operation_is_a_method(self):
        for op in FemOpsHandler._ALLOWED_OPERATIONS:
            self.assertTrue(callable(getattr(FemOpsHandler, op, None)), op)


if __name__ == '__main__':
    unittest.main()
