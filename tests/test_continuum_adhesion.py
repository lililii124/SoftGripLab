import numpy as np
import pytest

pytest.importorskip("mpm_jax", reason="optional JAX-MPM environment is not installed")
from softgriplab.backends.continuum_adhesion import run_continuum_adhesion


@pytest.mark.mpm
def test_adhesion_changes_bulk_motion_and_tension_then_detaches(tmp_path):
    bonded = run_continuum_adhesion(tmp_path / "bonded", grid=16, frames=21)
    free = run_continuum_adhesion(tmp_path / "free", grid=16, frames=21, work_j_m2=0)
    assert bonded["tensile_samples"] > 0
    assert free["tensile_samples"] == 0
    assert bonded["max_surface_lift_m"] > free["max_surface_lift_m"] + 0.002
    assert bonded["detached_at_end"] and free["detached_at_end"]
    assert bonded["min_det_f"] > 0
    with np.load(tmp_path / "bonded/trajectory.npz") as a, np.load(
        tmp_path / "free/trajectory.npz"
    ) as b:
        np.testing.assert_allclose(a["sphere_center_m"], b["sphere_center_m"])
        assert np.max(np.abs(a["position_m"] - b["position_m"])) > 0.002
