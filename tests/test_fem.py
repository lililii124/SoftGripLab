import numpy as np
import pytest

pytest.importorskip("pybullet", reason="optional PyBullet environment is not installed")
from softgriplab.backends.fem import run_fem


@pytest.mark.robot
def test_compression_bulges_and_recovers_without_inverted_tetrahedra(tmp_path):
    result = run_fem(tmp_path, cells=3, frames=31, render=False)
    assert result["specimen_anchors"] == 0
    assert result["min_det_f_all_steps"] > 0.5
    assert result["minimum_height_m"] < 0.032
    assert result["maximum_width_m"] > 0.052
    assert result["final_height_m"] > 0.038
    with np.load(tmp_path / "trajectory.npz") as data:
        assert len(data["time_s"]) == 31
        # Lateral motion must be solved, even though only vertical fixture motion is prescribed.
        assert (
            np.max(abs(data["position_m"][:, :, 0] - data["reference_m"][:, 0])) > 0.001
        )


@pytest.mark.robot
def test_soft_block_requires_friction_to_lift(tmp_path):
    held = run_fem(tmp_path / "held", scene="grasp", cells=4, frames=31, render=False)
    slip = run_fem(
        tmp_path / "slip", scene="grasp", cells=4, frames=31, friction=0, render=False
    )
    assert held["maximum_center_height_m"] > 0.04
    assert slip["maximum_center_height_m"] < 0.03
    trace = np.genfromtxt(tmp_path / "held/observables.csv", delimiter=",", names=True)
    lifted = (trace["time_s"] > 1.8) & (trace["time_s"] < 2.5)
    assert np.all(trace["tool_1_contacts"][lifted] > 0)
    assert np.all(trace["tool_2_contacts"][lifted] > 0)
