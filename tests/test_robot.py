import numpy as np
import pytest

pytest.importorskip("pybullet", reason="optional PyBullet environment is not installed")

from softgriplab.backends.robot import run_robot


@pytest.mark.robot
def test_motor_driven_pick_place_and_weak_suction_control(tmp_path):
    good = run_robot(tmp_path / "nominal", render=False)
    weak = run_robot(tmp_path / "weak", pressure_pa=1000., render=False)
    assert good["pick_place_success"]
    assert good["events"][0]["contact_confirmed"]
    assert good["events"][0]["event"] == "seal_acquired"
    assert good["events"][-1]["event"] == "commanded_release"
    assert abs(good["events"][0]["gap_m"]) <= good["suction"]["seal_tolerance_m"]
    assert good["max_payload_height_m"] > .25
    assert not weak["pick_place_success"]
    assert any(event["event"] == "capacity_exceeded" for event in weak["events"])
    assert weak["max_payload_height_m"] < .15
    np.testing.assert_allclose(weak["final_payload_position_m"][:2], [.5, -.18], atol=.005)
    trajectory = np.genfromtxt(tmp_path / "nominal/trajectory.csv", delimiter=",", names=True)
    assert np.max(trajectory["tcp_tracking_error_m"]) < .02

