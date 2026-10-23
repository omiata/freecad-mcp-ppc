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
from unittest.mock import MagicMock, patch

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

import handlers.fem_ops as fem_ops_module  # isort: skip
from handlers.fem_ops import (  # isort: skip
    MATERIAL_PRESETS,
    FemOpsHandler,
    _default_max_size,
    _mesh_quality,
    _normalize_faces,
    _tet_quality,
    _unit_vector,
)

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
        self.elements = {}  # volume id -> node ids

    @property
    def Nodes(self):
        return self.nodes

    @property
    def Volumes(self):
        return list(self.elements)

    def getElementNodes(self, vid):
        return self.elements[vid]

    def getNodeById(self, i):
        return self.nodes[i]


class FakeMeshRegion(_Obj):
    TypeId = 'Fem::MeshRegion'

    def __init__(self, name, mesh, size):
        super().__init__(name)
        self.CharacteristicLength = size
        self.References = []
        mesh.MeshRegionList = list(getattr(mesh, 'MeshRegionList', [])) + [self]


class FakeMesh(_Obj):
    TypeId = 'Fem::FemMeshShapeBaseObjectPython'

    def __init__(self, name):
        super().__init__(name)
        self.FemMesh = FakeFemMesh()
        self.MeshRegionList = []
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
    mod.makeMeshRegion = lambda d, mesh, size, name='MeshRegion': _append(doc, FakeMeshRegion(name, mesh, size))
    sys.modules['ObjectsFem'] = mod

    def remove(name):
        obj = doc.getObject(name)
        doc.Objects.remove(obj)
        for o in doc.Objects:
            if isinstance(getattr(o, 'MeshRegionList', None), list) and obj in o.MeshRegionList:
                o.MeshRegionList.remove(obj)
    doc.removeObject.side_effect = remove
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

    def test_tet_quality_regular_flat_and_inverted(self):
        s = 2 ** -0.5
        regular = (_Vec(1, 0, -s), _Vec(-1, 0, -s), _Vec(0, 1, s), _Vec(0, -1, s))
        q, ratio = _tet_quality(*regular)
        self.assertAlmostEqual(abs(q), 1.0, places=6)
        self.assertAlmostEqual(ratio, 1.0, places=6)
        q, _ = _tet_quality(_Vec(0, 0, 0), _Vec(1, 0, 0), _Vec(0, 1, 0), _Vec(1, 1, 0))
        self.assertAlmostEqual(q, 0.0, places=9)
        q_pos, _ = _tet_quality(_Vec(0, 0, 0), _Vec(1, 0, 0), _Vec(0, 1, 0), _Vec(0, 0, 1))
        q_neg, _ = _tet_quality(_Vec(0, 0, 0), _Vec(0, 1, 0), _Vec(1, 0, 0), _Vec(0, 0, 1))
        self.assertAlmostEqual(q_pos, -q_neg)
        _, ratio = _tet_quality(_Vec(0, 0, 0), _Vec(10, 0, 0), _Vec(0, 1, 0), _Vec(0, 0, 1))
        self.assertAlmostEqual(ratio, 101 ** 0.5)

    def test_mesh_quality_counts_poor_and_degenerate(self):
        m = FakeFemMesh()
        m.nodes = {1: _Vec(0, 0, 0), 2: _Vec(1, 0, 0), 3: _Vec(0, 1, 0), 4: _Vec(0, 0, 1),
                   5: _Vec(1, 1, 0), 6: _Vec(0.5, 0.5, 0.05)}
        m.elements = {1: [1, 2, 3, 4, 99, 99, 99, 99, 99, 99],  # TET10: corners first
                      2: [1, 2, 3, 5], 3: [1, 2, 3, 6]}
        report = _mesh_quality(m)
        self.assertEqual(report['elements_checked'], 3)
        self.assertEqual(report['degenerate_elements'], 1)  # the flat one
        self.assertEqual(report['poor_elements'], 2)         # flat + the sliver
        self.assertEqual(_mesh_quality(FakeFemMesh()), {'elements_checked': 0})

    def test_default_max_size(self):
        bb = MagicMock(XLength=100.0, YLength=60.0, ZLength=10.0, DiagonalLength=117.0)
        self.assertEqual(_default_max_size(bb, None, '2nd'), 5.0)   # smallest side stands in for the wall
        self.assertEqual(_default_max_size(bb, 2.0, '2nd'), 1.0)
        self.assertEqual(_default_max_size(bb, 2.0, '1st'), 0.5)
        self.assertEqual(_default_max_size(bb, 40.0, '2nd'), 5.85)  # chunky: diagonal / 20 caps it

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
        """No wall measured (the mock shape has no real faces): the smallest
        bounding-box side stands in for it, so 10 mm / 2."""
        _install_gmsh()
        data = self.json_call('create_mesh', object_name='Beam')
        self.assertAlmostEqual(data['max_size_mm'], 5.0, places=4)
        self.assertEqual(data['max_size_source'], 'from part size and thinnest wall')

    def test_default_size_follows_thinnest_wall(self):
        """A 2 mm wall on a 100 mm part: the old max(side/4, diagonal/40)
        rule gave 2.5 mm elements, less than one across the wall."""
        _install_gmsh()
        with patch.object(fem_ops_module, '_min_wall_thickness', return_value=(2.0, [0, 0, 1])):
            data = self.json_call('create_mesh', object_name='Beam')
            self.assertAlmostEqual(data['max_size_mm'], 1.0)          # 2 mm / 2 second-order elements
            self.assertEqual(data['elements_through_thickness_est'], 2.0)
            self.assertEqual(data['warnings'], [])
            data = self.json_call('create_mesh', object_name='Beam', element_order='1st')
            self.assertAlmostEqual(data['max_size_mm'], 0.5)          # 1st order wants 4 across

    def test_huge_mesh_needs_allow_large(self):
        """A 1000 mm3 part at 0.05 mm is ~40M elements: refused, because
        meshing and solving it would freeze FreeCAD's GUI for minutes."""
        _install_gmsh()
        assert_error_contains(self, self.call('create_mesh', object_name='Beam', max_size=0.05),
                              'elements', 'allow_large')
        self.json_call('create_mesh', object_name='Beam', max_size=0.05, allow_large=True)

    def test_coarse_given_size_warns_about_thin_wall(self):
        _install_gmsh()
        with patch.object(fem_ops_module, '_min_wall_thickness', return_value=(2.0, [5, 5, 1])):
            data = self.json_call('create_mesh', object_name='Beam', max_size=5)
        self.assertEqual(data['elements_through_thickness_est'], 0.4)
        self.assertEqual(len(data['warnings']), 1)
        self.assertIn('thinnest wall', data['warnings'][0])
        self.assertIn('max_size <= 1', data['warnings'][0])

    def test_refine_creates_regions_and_remesh_replaces_them(self):
        _install_gmsh()
        data = self.json_call('create_mesh', object_name='Beam', max_size=4,
                              refine=[{'faces': ['Face1'], 'size': 0.5}, {'faces': [2, 3], 'size': 1}])
        mesh = self.doc.getObject('Mesh')
        self.assertEqual([r.Name for r in mesh.MeshRegionList], ['Mesh_Refine1', 'Mesh_Refine2'])
        self.assertEqual(mesh.MeshRegionList[0].CharacteristicLength, 0.5)
        self.assertEqual(mesh.MeshRegionList[1].References, [(self.beam, 'Face2'), (self.beam, 'Face3')])
        self.assertEqual(len(data['refinements']), 2)
        self.json_call('create_mesh', object_name='Beam', max_size=4, refine=[{'faces': [4], 'size': 2}])
        self.assertEqual([r.Name for r in mesh.MeshRegionList], ['Mesh_Refine1'])
        self.assertEqual(mesh.MeshRegionList[0].References, [(self.beam, 'Face4')])
        self.json_call('create_mesh', object_name='Beam', max_size=4)
        self.assertEqual(mesh.MeshRegionList, [])

    def test_refine_validation(self):
        _install_gmsh()
        assert_error_contains(self, self.call('create_mesh', object_name='Beam', refine=[{'faces': [1]}]),
                              'refine[0].size')
        assert_error_contains(self, self.call('create_mesh', object_name='Beam',
                                              refine=[{'faces': ['Face9'], 'size': 1}]), 'refine[0]', 'Face9')
        assert_error_contains(self, self.call('create_mesh', object_name='Beam', refine=['Face1']),
                              'refine[0] must be an object')

    def test_mesh_quality_operation(self):
        _install_gmsh()
        self.json_call('create_mesh', object_name='Beam', max_size=4)
        mesh = self.doc.getObject('Mesh')
        mesh.FemMesh.nodes = {1: _Vec(0, 0, 0), 2: _Vec(1, 0, 0), 3: _Vec(0, 1, 0), 4: _Vec(0, 0, 1),
                              5: _Vec(1, 1, 0)}
        mesh.FemMesh.elements = {10: [1, 2, 3, 4], 11: [1, 2, 3, 5]}  # second one is flat
        with patch.object(fem_ops_module, '_min_wall_thickness', return_value=(10.0, [0, 0, 0])):
            data = self.json_call('mesh_quality')
        self.assertEqual(data['quality']['elements_checked'], 2)
        self.assertEqual(data['quality']['degenerate_elements'], 1)
        self.assertTrue(any('degenerate' in w for w in data['warnings']))
        assert_error_contains(self, self.call('mesh_quality', mesh_name='Nope'), 'no Gmsh FEM mesh')

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

    def _solved(self):
        self.ready_analysis()
        _install_ccx()
        self.json_call('run_analysis', working_dir=self._tmp())

    def test_probe_point_and_box(self):
        self._solved()
        data = self.json_call('get_results', point=[0, 5, 9], box=[50, 0, 0, 101, 10, 10])
        self.assertEqual(data['at_point']['nearest_node_mm'], [0.0, 5.0, 10.0])
        self.assertAlmostEqual(data['at_point']['distance_mm'], 1.0)
        self.assertEqual(data['at_point']['von_mises_MPa'], 60.0)
        self.assertEqual(data['in_box']['nodes'], 1)
        self.assertEqual(data['in_box']['max_von_mises_MPa'], 0.5)

    def test_probe_away_from_supports(self):
        """Fixed face is x = 0: nodes 1 and 2 sit on it, so the max away from
        it is node 3's, and the safety factor follows that value."""
        self._solved()

        class Face:
            BoundBox = MagicMock(XMin=0.0, XMax=0.0, YMin=0.0, YMax=10.0, ZMin=0.0, ZMax=10.0)

            def distToShape(self, p):
                return (abs(p.x), None, None)

        self.beam.Shape.getElement = MagicMock(return_value=Face())
        with patch.object(sys.modules['Part'], 'Vertex', side_effect=lambda p: p, create=True):
            data = self.json_call('get_results', exclude_near_supports_mm=1)
        away = data['away_from_supports']
        self.assertEqual((away['nodes_kept'], away['nodes_excluded']), (1, 2))
        self.assertEqual(away['max_von_mises_MPa'], 0.5)
        self.assertEqual(away['max_von_mises_location_mm'], [100.0, 5.0, 5.0])
        self.assertAlmostEqual(away['safety_factor'], 470.0)
        self.assertEqual(data['max_von_mises_MPa'], 60.0)  # global peak still reported

    def test_probe_validation(self):
        self._solved()
        assert_error_contains(self, self.call('get_results', point=[1, 2]), 'point must be')
        assert_error_contains(self, self.call('get_results', box=[500, 500, 500, 600, 600, 600]),
                              'no result nodes inside box')
        assert_error_contains(self, self.call('get_results', exclude_near_supports_mm=0), '> 0')

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
