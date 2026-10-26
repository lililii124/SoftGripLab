import numpy as np
import pytest

pytest.importorskip("mpm_jax", reason="optional JAX-MPM environment is not installed")

from softgriplab.backends.adhesive import contact_cycle


@pytest.mark.mpm
def test_jkr_cycle_retains_tension_then_detaches():
    data = contact_cycle()
    assert not data["jkr_active"][0] and not data["jkr_active"][-1]
    attached = (data["indentation_m"] < 0) & data["jkr_active"]
    assert np.any(attached)
    assert np.all(data["jkr_force_n"][attached] < 0)
    assert not np.any(data["hertz_active"][data["indentation_m"] <= 0])
    # The displacement-controlled stable branch passes the load-control minimum
    # before reaching its own displacement instability.
    assert data["jkr_force_n"].min() == pytest.approx(-1.5 * np.pi * .08 * .01, rel=.01)


@pytest.mark.mpm
def test_zero_adhesion_recovers_hertz_force_and_radius():
    data = contact_cycle(work_j_m2=0)
    np.testing.assert_allclose(data["jkr_radius_m"], data["hertz_radius_m"], atol=1e-14)
    np.testing.assert_allclose(data["jkr_force_n"], data["hertz_force_n"], atol=1e-14)
