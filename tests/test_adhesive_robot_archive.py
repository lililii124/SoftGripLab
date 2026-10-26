"""Physical checks on public full-rate traces without the optional solver."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "docs/results/adhesive_robot"


def test_adhesive_lift_force_balance_and_release():
    report = json.loads((ROOT / "nominal/summary.json").read_text())
    trace = np.genfromtxt(ROOT / "nominal/observables.csv", delimiter=",", names=True)
    assert report["payload_anchors"] == report["payload_attachment_constraints"] == 0
    assert report["first_compressive_contact_s"] < 1.25
    assert report["pick_place_success"] and trace["min_det_f"].min() > 0.5
    carry = np.flatnonzero((trace["time_s"] > 2.8) & (trace["time_s"] < 3.8))
    assert np.all(trace["adhesive_points"][carry] > 0)
    assert np.all(trace["bottom_z_m"][carry] > 0.05)
    a, b = carry[[0, -1]]
    force = np.column_stack([trace[f"reaction_{axis}_n"] for axis in "xyz"])
    velocity = np.column_stack([trace[f"com_v{axis}_m_s"] for axis in "xyz"])
    impulse = (-force[a + 1 : b + 1] - [0, 0, report["weight_n"]]).sum(axis=0) * report[
        "coupling_dt_s"
    ]
    np.testing.assert_allclose(
        impulse, report["mass_kg"] * (velocity[b] - velocity[a]), atol=1e-6
    )
    assert trace["adhesive_points"][-1] == 0
    assert np.linalg.norm(force[-1]) < 1e-6


def test_contacted_zero_adhesion_control_cannot_lift():
    report = json.loads((ROOT / "nonadhesive/summary.json").read_text())
    assert report["first_compressive_contact_s"] < 1.25
    assert not report["pick_place_success"]
    assert report["max_com_height_m"] < 0.046
    assert report["airborne_fraction"] == 0


def test_surface_replay_provenance_and_coupling_refinement():
    render = json.loads((ROOT / "nominal/render.json").read_text())
    assert (
        render["surface_replay_sha256"]
        == hashlib.sha256((ROOT / "nominal/trajectory.npz").read_bytes()).hexdigest()
    )
    assert render["max_replayed_tool_center_error_m"] < 1e-6
    nominal = json.loads((ROOT / "nominal/summary.json").read_text())
    fine = json.loads((ROOT / "fine_coupling/summary.json").read_text())
    assert fine["pick_place_success"]
    assert (
        np.linalg.norm(np.array(nominal["final_com_m"]) - fine["final_com_m"]) < 0.0005
    )
