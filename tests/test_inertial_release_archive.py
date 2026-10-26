"""Causal and momentum checks on the acceleration-release archive."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "docs/results/inertial_release"


def read(name):
    return (
        json.loads((ROOT / name / "summary.json").read_text()),
        np.genfromtxt(ROOT / name / "observables.csv", delimiter=",", names=True),
    )


def test_only_withdrawal_history_changes_and_adhesion_stays_enabled():
    fast, a = read("fast")
    slow, b = read("slow")
    prefix = a["time_s"] <= 4.4
    for key in a.dtype.names:
        np.testing.assert_array_equal(a[key][prefix], b[key][prefix])
    for key in [
        "work_j_m2",
        "grid",
        "mass_kg",
        "young_pa",
        "interaction_range_m",
        "release_position_gain",
    ]:
        assert fast[key] == slow[key]
    for report, trace in [(fast, a), (slow, b)]:
        assert not report["adhesion_switched_off_for_release"]
        assert (
            report["payload_attachment_constraints"] == report["payload_anchors"] == 0
        )
        assert np.all(trace["adhesion_on"][trace["time_s"] >= 1.25] == 1)
        carry = (trace["time_s"] >= 2.8) & (trace["time_s"] <= 4.4)
        assert np.all(trace["bottom_z_m"][carry] > 0.05)
        assert np.all(trace["adhesive_points"][carry] > 0)
    assert fast["pick_place_success"] and fast["detached_at_end"]
    assert not slow["detached_at_end"] and b["adhesive_points"][-1] > 0


def test_upward_acceleration_release_and_free_flight_momentum():
    times = []
    for name in ["fast", "fine"]:
        report, r = read(name)
        released = (
            (r["time_s"] > 4.4)
            & (r["adhesive_points"] == 0)
            & (r["compressed_points"] == 0)
            & (r["min_gap_m"] > report["interaction_range_m"])
        )
        index = np.flatnonzero(released)[0]
        times.append(r["time_s"][index])
        assert r["head_az_m_s2"][index] > 0 and r["head_vz_m_s"][index] > 0
        assert np.max(abs(r["reaction_z_n"][index + 1 :])) < 1e-10
        assert report["pick_place_success"] and r["min_det_f"].min() > 0
        a = np.flatnonzero(r["time_s"] >= 4.4)[0]
        b = index + 1
        assert np.all(r["bottom_z_m"][a : b + 1] > 0.05)
        force = np.column_stack([r[f"reaction_{axis}_n"] for axis in "xyz"])
        velocity = np.column_stack([r[f"com_v{axis}_m_s"] for axis in "xyz"])
        impulse = (-force[a + 1 : b + 1] - [0, 0, report["weight_n"]]).sum(
            axis=0
        ) * report["coupling_dt_s"]
        np.testing.assert_allclose(
            impulse, report["mass_kg"] * (velocity[b] - velocity[a]), atol=1e-6
        )
    assert abs(times[1] - times[0]) <= read("fast")[0]["coupling_dt_s"] + 1e-12


def test_force_and_surface_render_provenance():
    report = json.loads((ROOT / "fast/render.json").read_text())
    assert (
        report["surface_replay_sha256"]
        == hashlib.sha256((ROOT / "fast/trajectory.npz").read_bytes()).hexdigest()
    )
    data = (ROOT / "fast/observables.csv").read_bytes().replace(b"\r\n", b"\n")
    assert report["force_trace_sha256"] == hashlib.sha256(data).hexdigest()
    with np.load(ROOT / "fast/trajectory.npz") as d:
        assert str(d["source_trajectory_sha256"]) == report["trajectory_sha256"]
        assert np.isfinite(d["position_m"]).all()
    assert report["max_replayed_tool_center_error_m"] < 1e-6
    assert report["deformation_scale"] == 1 and not report["text_overlays"]
