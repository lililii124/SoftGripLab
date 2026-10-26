"""Variational consistency, objectivity and force balance of 3D contact."""

import numpy as np
import pytest

pytest.importorskip("skfem")
from softgriplab.complex_contact import HyperelasticContact, PROBE_SITES


@pytest.fixture(scope="module")
def model():
    return HyperelasticContact((8, 4, 4))


def test_energy_gradient_and_consistent_tangent(model):
    rng = np.random.default_rng(30)
    u = rng.normal(0, 0.0001, model.basis.N)
    direction = rng.normal(size=len(u))
    direction /= np.linalg.norm(direction)
    center = PROBE_SITES[1].center(-0.07)
    e, gradient, K, _, _ = model.evaluate(u, center, 0.65, 0.025)
    h = 2e-6
    plus = model.evaluate(u+h*direction, center, 0.65, 0.025, tangent=False)
    minus = model.evaluate(u-h*direction, center, 0.65, 0.025, tangent=False)
    assert (plus[0]-minus[0])/(2*h) == pytest.approx(gradient@direction, rel=1e-5, abs=1e-7)
    np.testing.assert_allclose((plus[1]-minus[1])/(2*h), K@direction, rtol=3e-5, atol=2e-7)
    assert np.isfinite(e)


def test_rigid_rotation_has_zero_bulk_energy_and_stress(model):
    a = 0.53
    rotation = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    displacement = model.points@rotation.T-model.points
    u = np.zeros(model.basis.N)
    u[model.basis.nodal_dofs] = displacement.T
    energy, gradient, _, reaction, min_j = model.evaluate(u, [10, 10, 10], 0.65, 0.025)
    assert abs(energy) < 1e-10
    np.testing.assert_allclose(gradient, 0, atol=1e-11)
    np.testing.assert_allclose(reaction, 0, atol=1e-11)
    assert min_j == pytest.approx(1, abs=1e-12)


def test_equilibrium_clamps_balance_the_tool_reaction(model):
    center = PROBE_SITES[1].center(-0.04)
    u, reaction, report = model.equilibrium(center, 0.65, 0.012)
    _, gradient, _, _, _ = model.evaluate(u, center, 0.65, 0.012)
    nodal_residual = gradient[model.basis.nodal_dofs]
    fixture_force = -nodal_residual[:, model.fixed_nodes].sum(axis=1)
    np.testing.assert_allclose(fixture_force+reaction, 0, atol=1e-5)
    np.testing.assert_array_equal(u[model.fixed], 0)
    assert report["min_j"] > 0 and report["residual"] < 2e-6
