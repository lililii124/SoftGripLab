"""Audit the published solved mesh without requiring the research backend."""

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parents[1] / "docs/results/cutting"


def test_saved_cut_topology_and_irreversible_separation():
    with np.load(ROOT / "nominal/trajectory.npz") as data:
        x = data["position_m"]
        tets = data["tet_indices"]
        hard_count = int(data["hard_particle_count"])
        edges, weights = data["cut_edge_indices"], data["cut_edge_coordinates"]
        virtual = (
            x[:, edges[:, 0]] * (1 - weights[None, :, None])
            + x[:, edges[:, 1]] * weights[None, :, None]
        )
        np.testing.assert_allclose(x[:, hard_count:], virtual, atol=2e-9)
        assert np.isfinite(x).all()
        springs = data["cut_spring_indices"]
        ke = data["cut_stiffness_n_m"]
        assert np.all(np.diff(ke, axis=0) <= 1e-6)
        assert np.all(ke[0] == 500) and np.all(ke[-1] == 0)
        # Build connectivity independently from the saved element/spring indices.
        base_a, base_b = np.repeat(tets[:, 0], 3), tets[:, 1:].ravel()
        counts = []
        for row in ke:
            live = springs[row >= 1e-5]
            ia = np.r_[base_a, edges[live[:, 0], 0]]
            ib = np.r_[base_b, edges[live[:, 1], 0]]
            graph = coo_matrix(
                (np.ones(len(ia)), (ia, ib)), shape=(hard_count, hard_count)
            )
            counts.append(
                connected_components(graph, directed=False, return_labels=False)
            )
        np.testing.assert_array_equal(counts, data["connected_components"])
        assert counts[0] == 1 and counts[-1] == 4
        assert set(counts) == {1, 2, 3, 4}
        # Elements remain valid; ghost support nodes are not new material mass.
        ref = data["reference_position_m"]
        d0 = np.transpose(ref[tets[:, 1:]] - ref[tets[:, 0, None]], (0, 2, 1))
        dx = np.transpose(x[:, tets[:, 1:]] - x[:, tets[:, 0, None]], (0, 1, 3, 2))
        assert np.linalg.det(dx @ np.linalg.inv(d0)).min() > 0
        masses = data["physical_mass_kg"]
        summary = json.loads((ROOT / "nominal/summary.json").read_text())
        np.testing.assert_allclose(
            masses.sum(), summary["reference_mass_kg"], rtol=1e-10
        )
        assert np.all(masses > 0)
        fixed = data["clamped_nodes"]
        np.testing.assert_allclose(
            x[:, fixed], np.broadcast_to(x[0, fixed], x[:, fixed].shape), atol=1e-8
        )


def test_cutting_controls_and_render_provenance():
    nominal = json.loads((ROOT / "nominal/summary.json").read_text())
    control = json.loads((ROOT / "no_damage_prefix/summary.json").read_text())
    fine = json.loads((ROOT / "half_step_first_cut/summary.json").read_text())
    times = nominal["completed_cut_times_s"]
    assert 0 < times[0] < times[1] < times[2] < 4.2
    assert nominal["success"] and nominal["min_det_f"] > 0
    assert control["duration_s"] == 0.5 and control["final_connected_components"] == 1
    assert control["final_broken_fractions"] == [0, 0, 0]
    trace = np.genfromtxt(
        ROOT / "no_damage_prefix/observables.csv", delimiter=",", names=True
    )
    assert abs(trace["reaction_y_n"]).max() > 0.001
    assert fine["duration_s"] == 1.1 and fine["final_connected_components"] == 2
    assert (
        abs(fine["completed_cut_times_s"][0] - times[0])
        <= nominal["coupling_dt_s"] + 1e-9
    )
    render = json.loads((ROOT / "nominal/render.json").read_text())
    assert render["deformation_scale"] == 1
    assert render["max_replayed_tool_center_error_m"] < 1e-6
    assert (
        hashlib.sha256((ROOT / "nominal/trajectory.npz").read_bytes()).hexdigest()
        == render["trajectory_sha256"]
    )
