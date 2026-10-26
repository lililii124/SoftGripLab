"""Optional full rollout; the public archive checks run in ordinary CI."""
import os
import pytest


@pytest.mark.mpm
@pytest.mark.robot
@pytest.mark.skipif(
    os.environ.get("SOFTGRIPLAB_FULL_COUPLED") != "1",
    reason="full coupled rollout is opt-in",
)
def test_contacted_adhesion_lifts_but_zero_work_does_not(tmp_path):
    pytest.importorskip("mpm_jax")
    pytest.importorskip("pybullet")
    from softgriplab.backends.adhesive_robot import run_adhesive_robot

    nominal = run_adhesive_robot(tmp_path / "nominal", fps=1)
    control = run_adhesive_robot(tmp_path / "control", work_j_m2=0, fps=1)
    assert nominal["pick_place_success"]
    assert not control["pick_place_success"]
    assert nominal["first_compressive_contact_s"] < 1.25
    assert control["first_compressive_contact_s"] < 1.25
    assert nominal["max_com_height_m"] > control["max_com_height_m"] + 0.04
    assert nominal["airborne_momentum_balance_residual_n_s"] < 1e-6
    assert nominal["final_contact_force_n"] < 1e-6
