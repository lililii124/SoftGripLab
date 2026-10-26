"""Optional continuum integration check using a free-motion momentum oracle."""

import numpy as np
import pytest
from scipy.integrate import quad

pytest.importorskip("mpm_jax", reason="optional JAX-MPM environment is not installed")

from softgriplab.backends.jax_mpm import run_grasp


@pytest.mark.mpm
def test_frictionless_vertical_momentum_balance(tmp_path):
    result = run_grasp(tmp_path / "free", grid=16, frames=8, friction=0)

    def integrand(t):
        gravity = 9.81 * np.clip((t - .03) / .015, 0, 1)
        shock = 20 * np.sin(np.pi * (t - .075) / .015) if .075 < t < .09 else 0
        return (.12 - t) * (gravity + shock)

    expected = quad(integrand, 0, .12, points=[.03, .045, .075, .09], epsabs=1e-12)[0]
    # First-order explicit stepping: error should scale with dt, not grid spacing.
    assert abs(result["final_com_drop_m"] - expected) < .5 * 30 * .12 * result["dt_s"]
    assert result["min_det_f"] > 0
