"""Independent mechanical checks for the active-identification reference."""

import numpy as np
import pytest
from scipy.integrate import trapezoid

from softgriplab.active_contact import (CALIBRATION, ContactSpec, Probe, Release,
                                       initial_state, interface_force, simulate)
from softgriplab.active_identification import acquisition_scores, log_posterior, make_bank


PARAMETERS = np.array([28000., 42000., 0.14, 1.7])


def test_adhesion_work_and_contact_sign():
    spec = ContactSpec()
    gap = np.linspace(0, spec.cutoff_m, 10001)
    force = interface_force(gap, PARAMETERS[3], spec)
    assert trapezoid(force, gap)/spec.geometry.area_m2 == pytest.approx(PARAMETERS[3], rel=1e-7)
    assert interface_force(-1e-5, PARAMETERS[3]) < 0
    assert interface_force(spec.cutoff_m*1.01, PARAMETERS[3]) == 0


def test_suspended_static_equilibrium_is_preserved():
    result = simulate(PARAMETERS, Release(0))
    np.testing.assert_allclose(result["state"], np.broadcast_to(initial_state(PARAMETERS, True)[:, None], result["state"].shape), atol=1e-14)
    assert not result["detached"][0]
    np.testing.assert_allclose(result["force_n"], 0.01*9.81, atol=1e-13)


def test_acceleration_releases_without_changing_interface():
    slow = simulate(PARAMETERS, Release(4))
    fast = simulate(PARAMETERS, Release(100))
    assert not slow["detached"][0]
    assert fast["detached"][0]
    assert np.max(fast["force_n"]) > 0.01*9.81
    assert fast["force_n"][0, -1] == 0


@pytest.mark.parametrize("protocol", [CALIBRATION, Probe("separation", 0.0028, 0.24), Release(52)])
def test_rk4_matches_independent_implicit_integration(protocol):
    result = simulate(PARAMETERS, protocol)
    reference = simulate(PARAMETERS, protocol, method="radau")
    np.testing.assert_allclose(result["force_n"], reference["force_n"], atol=3e-4, rtol=0)
    np.testing.assert_allclose(result["state"][:, :, [0, 2]], reference["state"][:, :, [0, 2]], atol=2e-6, rtol=0)
    np.testing.assert_array_equal(result["detached"], reference["detached"])


def test_probe_energy_balance_includes_bulk_dissipation():
    protocol = Probe("energy", 0.0003, 0.15, 0.3)
    result = simulate(PARAMETERS, protocol, samples=1501, dt=5e-5)
    spec = ContactSpec()
    t, state, q = result["time_s"], result["state"][0], result["tool_m"]
    b, z = state[:, 2], state[:, 4]
    k_inf, k_branch = PARAMETERS[:2]*spec.geometry.area_m2/spec.geometry.thickness_m
    gap = q-b
    # This weak probe remains on the ascending, elastic interface branch.
    assert gap.min() >= -1e-12 and gap.max() < spec.peak_gap_m
    interface_k = 2*PARAMETERS[3]*spec.geometry.area_m2/spec.cutoff_m/spec.peak_gap_m
    energy = 0.5*k_inf*b**2 + 0.5*k_branch*(b-z)**2 + 0.5*interface_k*gap**2 + 0.5*spec.top_mass_kg*state[:, 3]**2
    dissipated = trapezoid(k_branch/PARAMETERS[2]*(b-z)**2, t)
    s = np.clip(t/protocol.ramp_s, 0, 1)
    velocity = protocol.amplitude_m/protocol.ramp_s*30*s**2*(1-s)**2
    work = trapezoid(result["force_n"][0]*velocity, t)
    assert energy[-1]-energy[0]+dissipated == pytest.approx(work, rel=2e-5, abs=1e-10)


def test_batched_solver_equals_individual_runs():
    bank = make_bank(32)[:3]
    batch = simulate(bank, CALIBRATION)
    for i, parameters in enumerate(bank):
        np.testing.assert_allclose(batch["state"][i], simulate(parameters, CALIBRATION)["state"][0], atol=1e-14)


def test_active_probe_targets_task_relevant_observation():
    # Probe A reveals a nuisance variable; B reveals the release class.
    predictions = np.zeros((2, 4, 3, 2))
    predictions[0, :, :, 0] = np.array([-5, 5, -5, 5])[:, None]
    predictions[1, :, :, 0] = np.array([-5, -5, 5, 5])[:, None]
    labels = np.array([[0], [0], [1], [1]])
    scores = acquisition_scores(np.full(4, -np.log(4)), predictions, labels,
                               noise=np.ones(2), draws=256)
    assert scores[1] > 0.6 and abs(scores[0]) < 0.01
    assert np.argmax(scores) == 1
    posterior = np.exp(log_posterior(np.full(4, -np.log(4)), predictions[1],
                                    predictions[1, 0], noise=np.ones(2)))
    np.testing.assert_allclose(posterior, [0.5, 0.5, 0, 0], atol=1e-10)


def test_invalid_models_and_unresolved_steps_fail():
    with pytest.raises(ValueError):
        simulate([1, 2, 3, -1], CALIBRATION)
    with pytest.raises(ValueError):
        simulate(PARAMETERS, CALIBRATION, dt=0.01)
    with pytest.raises(ValueError):
        simulate([28000, 42000, 0.14, 0.01], Release(50))
    with pytest.raises(ValueError):
        make_bank(33)
