import numpy as np
import pytest

pytest.importorskip("pybullet", reason="optional PyBullet environment is not installed")

from softgriplab.backends.robot import run_robot


@pytest.mark.robot
def test_friction_carries_payload_only_with_bilateral_contact(tmp_path):
    good = run_robot(tmp_path / "nominal", render=False)
    weak = run_robot(tmp_path / "frictionless", friction=0.0, render=False)
    assert good["pick_place_success"]
    assert good["payload_attachment_constraints"] == 0
    assert good["events"][0]["event"] == "bilateral_contact"
    assert good["airborne_bilateral_contact_fraction"] > 0.99
    assert good["max_contact_penetration_m"] < 0.0008
    assert good["max_payload_height_m"] > 0.30
    assert not weak["pick_place_success"]
    assert weak["max_payload_height_m"] < 0.16
    trajectory = np.genfromtxt(
        tmp_path / "nominal/trajectory.csv", delimiter=",", names=True
    )
    assert np.max(trajectory["tcp_tracking_error_m"]) < 0.02
    airborne = (trajectory["time_s"] > 3.5) & (trajectory["time_s"] < 6.0)
    assert np.min(trajectory["left_normal_n"][airborne]) > 0.1
    assert np.min(trajectory["right_normal_n"][airborne]) > 0.1
    assert np.max(np.abs(trajectory["left_gap_m"][airborne])) < 0.0008
    assert np.max(np.abs(trajectory["right_gap_m"][airborne])) < 0.0008
