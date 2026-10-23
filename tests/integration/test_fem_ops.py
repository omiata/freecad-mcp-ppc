"""Integration tests for fem_operations against a live FreeCAD instance.

Runs the real pipeline -- Gmsh meshing and the CalculiX solver bundled with
FreeCAD -- on a steel cantilever and compares the result with Euler-
Bernoulli beam theory. The unit tests' fakes can only check that the
handler drives ObjectsFem/femtools in the right order; this checks that
the numbers that come back are physically right.

Run with: python3 -m pytest tests/integration/test_fem_ops.py -v
"""

import contextlib
import json
import time

import pytest

from . import conftest as _conftest  # noqa: F401  bootstraps the session fixture
from .test_e2e_workflows import send_command

L, B, H = 100.0, 10.0, 10.0      # mm, bar along +X
F_N = 100.0                      # tip load, -Z
E_MPA = 210000.0


def _analytic():
    inertia = B * H ** 3 / 12.0
    sigma = F_N * L * (H / 2.0) / inertia
    delta = F_N * L ** 3 / (3.0 * E_MPA * inertia)
    return sigma, delta


def _fem(args: dict, timeout: float = 120.0) -> str:
    resp = send_command("fem_operations", args, timeout=timeout)
    if isinstance(resp, dict) and "result" in resp:
        return resp["result"]
    if isinstance(resp, dict) and "error" in resp:
        return resp["error"]
    return str(resp)


def _fem_json(args: dict, timeout: float = 120.0) -> dict:
    out = _fem(args, timeout=timeout)
    assert not str(out).startswith("Error"), out
    return json.loads(out)


def _exec(code: str, timeout: float = 10.0):
    return send_command("execute_python_sync", {"code": code}, timeout=timeout)


def _face_at_x(x: float) -> str:
    """Name of the Beam face whose centroid sits at the given X."""
    resp = _exec(
        "beam = FreeCAD.ActiveDocument.getObject('Beam')\n"
        f"result = next('Face%d' % i for i, f in enumerate(beam.Shape.Faces, 1) "
        f"if abs(f.CenterOfMass.x - {x}) < 1e-6)"
    )
    text = json.dumps(resp)
    for i in range(1, 7):
        if f'Face{i}' in text:
            return f'Face{i}'
    raise AssertionError(f"face at x={x} not found: {resp}")


@pytest.fixture
def cantilever_doc():
    doc_name = f"FEM_{int(time.time() * 1000) % 100000}"
    send_command("view_control", {"operation": "create_document", "document_name": doc_name})
    send_command("part_operations", {"operation": "box", "name": "Beam",
                                     "length": L, "width": B, "height": H})
    yield doc_name
    with contextlib.suppress(Exception):
        _exec(f"FreeCAD.closeDocument({doc_name!r})")


class TestCantilever:
    def test_static_cantilever_matches_beam_theory(self, cantilever_doc):
        fixed_face, tip_face = _face_at_x(0.0), _face_at_x(L)

        _fem_json({"operation": "create_analysis"})
        check = _fem_json({"operation": "check_analysis"})
        assert not check["ready"]

        _fem_json({"operation": "set_material", "preset": "steel_s235", "youngs_modulus": E_MPA})
        mesh = _fem_json({"operation": "create_mesh", "object_name": "Beam", "max_size": 4})
        assert mesh["volume_elements"] > 100
        _fem_json({"operation": "add_fixed", "object_name": "Beam", "faces": [fixed_face]})
        _fem_json({"operation": "add_force", "object_name": "Beam", "faces": [tip_face],
                   "force": F_N, "direction": [0, 0, -1]})
        assert _fem_json({"operation": "check_analysis"})["ready"]

        res = _fem_json({"operation": "run_analysis"}, timeout=300.0)
        sigma, delta = _analytic()
        assert res["success"] is True
        # 2nd-order tets: within a few percent of beam theory (measured:
        # +0.5% stress, +0.2% deflection on FreeCAD 1.1.4).
        assert res["max_von_mises_MPa"] == pytest.approx(sigma, rel=0.05)
        assert res["max_displacement_mm"] == pytest.approx(delta, rel=0.05)
        # Tip moves down (-Z), the load's direction.
        assert res["max_displacement_components_mm"]["z"] < 0
        # Peak stress at the clamped root, not at the tip.
        assert res["max_von_mises_location_mm"][0] < L / 4
        assert res["safety_factor"] == pytest.approx(235.0 / res["max_von_mises_MPa"], rel=1e-3)

        again = _fem_json({"operation": "get_results"})
        assert again["max_von_mises_MPa"] == pytest.approx(res["max_von_mises_MPa"])

    def test_incomplete_setup_is_refused_before_solving(self, cantilever_doc):
        _fem_json({"operation": "create_analysis"})
        out = _fem({"operation": "run_analysis"})
        assert "not ready" in out and "no material" in out

    def test_mid_span_stress_converges_while_root_peak_does_not_matter(self, cantilever_doc):
        """Convergence check the way a caller would do it: solve at h and
        h/2 and compare a stress read away from the clamp. At mid-span beam
        theory gives F*(L/2)*(H/2)/I = 30 MPa on the top fibre."""
        fixed_face, tip_face = _face_at_x(0.0), _face_at_x(L)
        _fem_json({"operation": "create_analysis"})
        _fem_json({"operation": "set_material", "preset": "steel_s235", "youngs_modulus": E_MPA})
        _fem_json({"operation": "add_fixed", "object_name": "Beam", "faces": [fixed_face]})
        _fem_json({"operation": "add_force", "object_name": "Beam", "faces": [tip_face],
                   "force": F_N, "direction": [0, 0, -1]})
        probe = {"point": [L / 2, B / 2, H], "exclude_near_supports_mm": 10.0}
        runs = []
        for size in (4.0, 2.0):
            _fem_json({"operation": "create_mesh", "object_name": "Beam", "max_size": size})
            runs.append(_fem_json({"operation": "run_analysis", **probe}, timeout=600.0))
        mid = [r["at_point"]["von_mises_MPa"] for r in runs]
        sigma_mid = F_N * (L / 2) * (H / 2) / (B * H ** 3 / 12.0)
        assert mid[1] == pytest.approx(sigma_mid, rel=0.05)
        assert abs(mid[1] - mid[0]) / mid[1] < 0.05          # converged at mid-span
        away = [r["away_from_supports"]["max_von_mises_MPa"] for r in runs]
        assert abs(away[1] - away[0]) / away[1] < 0.05
        for r in runs:
            assert r["away_from_supports"]["max_von_mises_location_mm"][0] >= 10.0 - 1e-6


@pytest.fixture
def plate_doc():
    doc_name = f"FEMP_{int(time.time() * 1000) % 100000}"
    send_command("view_control", {"operation": "create_document", "document_name": doc_name})
    send_command("part_operations", {"operation": "box", "name": "Plate",
                                     "length": 100, "width": 60, "height": 2})
    yield doc_name
    with contextlib.suppress(Exception):
        _exec(f"FreeCAD.closeDocument({doc_name!r})")


class TestMeshSizing:
    def test_thin_plate_default_size_follows_wall(self, plate_doc):
        """100 x 60 x 2 mm: the first default rule gave 2.5 mm elements here,
        under one across the wall."""
        _fem_json({"operation": "create_analysis"})
        mesh = _fem_json({"operation": "create_mesh", "object_name": "Plate"}, timeout=300.0)
        assert mesh["min_wall_thickness_mm"] == pytest.approx(2.0, rel=1e-3)
        assert mesh["max_size_mm"] <= 1.0 + 1e-9
        assert mesh["elements_through_thickness_est"] >= 2.0
        assert mesh["warnings"] == []
        assert mesh["quality"]["degenerate_elements"] == 0
        assert mesh["quality"]["min_shape_quality"] > 0.0

    def test_coarse_size_warns_and_refinement_adds_nodes(self, plate_doc):
        _fem_json({"operation": "create_analysis"})
        coarse = _fem_json({"operation": "create_mesh", "object_name": "Plate", "max_size": 5})
        assert any("thinnest wall" in w for w in coarse["warnings"])
        top = _exec(
            "p = FreeCAD.ActiveDocument.getObject('Plate')\n"
            "result = next('Face%d' % i for i, f in enumerate(p.Shape.Faces, 1) "
            "if abs(f.CenterOfMass.z - 2) < 1e-6)")
        top_face = next(f"Face{i}" for i in range(1, 7) if f"'Face{i}'" in json.dumps(top))
        refined = _fem_json({"operation": "create_mesh", "object_name": "Plate", "max_size": 5,
                             "refine": [{"faces": [top_face], "size": 1.0}]}, timeout=300.0)
        assert refined["nodes"] > coarse["nodes"] * 2
        assert len(refined["refinements"]) == 1
        again = _fem_json({"operation": "mesh_quality"})
        assert again["nodes"] == refined["nodes"]
        assert again["refinements"] == ["Mesh_Refine1"]
