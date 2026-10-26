"""Independent audits of the saved dynamic-cutting states, without OpenCL."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parents[1] / "docs/results/dynamic_cutting"


def components(nl):
    i, k = np.where(nl >= 0)
    graph = coo_matrix((np.ones(len(i)), (i, nl[i, k])), shape=(len(nl), len(nl)))
    return connected_components(graph, directed=False)


def test_intact_start_support_and_dynamic_separation():
    summary = json.loads((ROOT / "nominal/summary.json").read_text())
    with np.load(ROOT / "nominal/trajectory.npz") as d:
        x = d["position_m"]
        n = x.shape[1]
        assert np.isfinite(x).all()
        assert (
            summary["preset_interfaces"]
            == summary["initial_broken_bonds"]
            == summary["clamped_nodes"]
            == 0
        )
        initial, final = d["initial_neighbours"], d["final_neighbours"]
        assert components(initial)[0] == 1
        assert np.all((final == initial) | (final == -1))
        count, labels = components(final)
        sizes = np.sort(np.bincount(labels))[::-1]
        assert count == summary["final_connected_components"]
        assert (sizes > n * 0.03).sum() == 4
        assert sizes[sizes <= n * 0.03].sum() / n < 0.01
        assert x[0, :, 2].min() <= 0.45 * float(d["spacing_m"])
        assert np.all(d["particle_volume_m3"] > 0)
        np.testing.assert_allclose(
            d["particle_volume_m3"].sum() * summary["density_kg_m3"], summary["mass_kg"]
        )
        assert (d["component_labels"][0] == d["component_labels"][0, 0]).all()
    trace = np.genfromtxt(ROOT / "nominal/observables.csv", delimiter=",", names=True)
    assert np.all(np.diff(trace["connected_components"]) >= 0)
    assert np.all(np.diff(trace["broken_fraction"]) >= -1e-12)
    assert abs(trace["reaction_z_n"]).max() > 0.001


def test_changed_path_uses_same_intact_body_and_control_does_not_cut():
    with np.load(ROOT / "nominal/trajectory.npz") as a, np.load(
        ROOT / "shifted/trajectory.npz"
    ) as b:
        np.testing.assert_array_equal(a["initial_neighbours"], b["initial_neighbours"])
        np.testing.assert_array_equal(
            a["reference_position_m"], b["reference_position_m"]
        )
        assert abs(a["knife_pose_local"][0, 0] - b["knife_pose_local"][0, 0]) > 0.009
        n = len(b["initial_neighbours"])
        sizes = np.bincount(components(b["final_neighbours"])[1])
        assert (sizes > n * 0.03).sum() == 2
    with np.load(ROOT / "intact/trajectory.npz") as control:
        np.testing.assert_array_equal(
            control["initial_neighbours"], control["final_neighbours"]
        )
        assert components(control["final_neighbours"])[0] == 1
    trace = np.genfromtxt(ROOT / "intact/observables.csv", delimiter=",", names=True)
    assert abs(trace["reaction_z_n"]).max() > 0.001


def test_render_provenance_and_time_step_sensitivity():
    report = json.loads((ROOT / "nominal/render.json").read_text())
    for key, name in [
        ("trajectory_sha256", "trajectory.npz"),
        ("force_trace_sha256", "observables.csv"),
    ]:
        data = (ROOT / "nominal" / name).read_bytes()
        if name.endswith(".csv"):
            data = data.replace(b"\r\n", b"\n")
        assert report[key] == hashlib.sha256(data).hexdigest()
    assert not report["geometry_uses_knife_path"] and not report["text_overlays"]
    assert (
        report["deformation_scale"] == 1
        and report["max_replayed_tool_center_error_m"] < 1e-6
    )
    comparison = json.loads((ROOT / "temporal_comparison.json").read_text())
    assert comparison["coarse_dt_s"] > 1.9 * comparison["fine_dt_s"]
    for case in ["fine", "coarse"]:
        assert comparison[case]["initial_major_components"] == 1
        assert comparison[case]["final_major_components"] == 4
        assert comparison[case]["debris_mass_fraction"] < 0.01
