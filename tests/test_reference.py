"""Physics and inference checks against analytic or separately constructed oracles."""

import numpy as np
import pytest
from scipy.integrate import trapezoid

from softgriplab.demo import analytic_ramp_force
from softgriplab.grasp import (GraspSpec, assess_grasp, half_sine_acceleration,
                              plan_fixed_closure)
from softgriplab.identification import CompressionTrace, fit_material
from softgriplab.materials import Geometry, StandardLinearSolid, ramp_hold


@pytest.fixture
def material():
    return StandardLinearSolid(35000, 65000, .6)


def test_scipy_response_matches_finite_ramp_analytic_solution(material):
    t = np.linspace(0, 5, 501)
    geometry = Geometry()
    force = material.force(t, ramp_hold(t, .001, .2), geometry)
    expected = analytic_ramp_force(material, t, .001, .2, geometry)
    np.testing.assert_allclose(force, expected, rtol=1e-11, atol=1e-11)


def test_step_relaxation_and_force_control_creep(material):
    t = np.linspace(0, 15, 1501)
    geometry = Geometry()
    strain = .02
    force = material.force(t, np.full_like(t, strain * geometry.thickness_m), geometry)
    expected = geometry.area_m2 * strain * (material.equilibrium_pa +
                                           material.branch_pa * np.exp(-t / material.relaxation_s))
    np.testing.assert_allclose(force, expected, rtol=1e-10)
    normal = .5
    compression = material.compression(t, np.full_like(t, normal), geometry)
    retardation = material.relaxation_s * material.instantaneous_pa / material.equilibrium_pa
    compliance = (1 / material.equilibrium_pa -
                  (1 / material.equilibrium_pa - 1 / material.instantaneous_pa) * np.exp(-t / retardation))
    np.testing.assert_allclose(compression, normal / geometry.area_m2 * compliance * geometry.thickness_m,
                               rtol=1e-10)


def test_multirate_identification_recovers_noisy_truth(material, tmp_path):
    rng = np.random.default_rng(81)
    t = np.linspace(0, 4, 401)
    geometry = Geometry()
    traces = []
    for ramp in (.1, .8):
        u = ramp_hold(t, .001, ramp)
        force = analytic_ramp_force(material, t, .001, ramp, geometry) + rng.normal(0, .004, len(t))
        traces.append(CompressionTrace(t, u, force, geometry, .004))
    traces[0].to_csv(tmp_path / "compression.csv")
    loaded = CompressionTrace.from_csv(tmp_path / "compression.csv", geometry, .004)
    np.testing.assert_allclose(loaded.force_n, traces[0].force_n)
    result = fit_material([loaded, traces[1]])
    estimated = [result.material.equilibrium_pa, result.material.branch_pa, result.material.relaxation_s]
    np.testing.assert_allclose(estimated, [35000, 65000, .6], rtol=.015)
    assert result.jacobian_condition < 100
    validation = np.linspace(0, 2, 401)
    actual = result.material.force(validation, ramp_hold(validation, .0005, .04), geometry)
    oracle = analytic_ramp_force(material, validation, .0005, .04, geometry)
    assert np.linalg.norm(actual - oracle) / np.linalg.norm(oracle) < .01


def test_uninformative_short_record_produces_identifiability_warning(material):
    t = np.linspace(0, .002, 21)
    u = ramp_hold(t, .0002, .001)
    trace = CompressionTrace(t, u, material.force(t, u), noise_std_n=.001)
    result = fit_material([trace], initial=material)
    assert any("shorter" in warning or "conditioned" in warning for warning in result.warnings)


@pytest.mark.parametrize("time", [np.array([0., .1, .3]), np.array([1., 2., 3.]),
                                 np.array([0., 0., 1.]), np.array([0., np.nan, 1.])])
def test_invalid_time_records_are_rejected(material, time):
    with pytest.raises(ValueError):
        material.force(time, np.full(3, .001))


def test_invalid_data_and_materials_are_rejected():
    with pytest.raises(ValueError):
        StandardLinearSolid(-1, 20, .2)
    with pytest.raises(ValueError):
        Geometry(area_m2=0)
    with pytest.raises(ValueError):
        CompressionTrace(np.arange(3.), np.zeros(3), np.ones(3))
    with pytest.raises(ValueError):
        fit_material([])


def test_two_finger_friction_capacity_and_zero_friction():
    t = np.linspace(0, 1, 11)
    spec = GraspSpec(mass_kg=.1, friction=.5)
    normal = np.full_like(t, .981)  # One finger's normal force, not the sum.
    result = assess_grasp(t, normal, np.full_like(t, .001), np.zeros_like(t), spec)
    np.testing.assert_allclose(result["margin_n"], 0, atol=1e-14)
    assert result["summary"]["sticking_feasible"]
    assert not result["summary"]["design_feasible"]
    no_friction = assess_grasp(t, normal, np.full_like(t, .001), np.zeros_like(t),
                               GraspSpec(friction=0))
    assert not no_friction["summary"]["sticking_feasible"]


def test_shock_pulse_impulse_matches_integral():
    t = np.linspace(0, 1, 10001)
    pulse = half_sine_acceleration(t, .2, .1, 30)
    np.testing.assert_allclose(trapezoid(pulse, t), 2 * 30 * .1 / np.pi, rtol=1e-5)
    assert pulse[0] == pulse[-1] == 0


def test_identified_plan_matches_demand_and_flags_infeasible_case(material):
    t = np.linspace(0, 4, 801)
    acceleration = half_sine_acceleration(t, 2.5, .2, 3)
    plan = plan_fixed_closure(material, t, acceleration)
    u = ramp_hold(t, plan["closure_m"], .2)
    result = assess_grasp(t, material.force(t, u), u, acceleration, evaluation_start_s=.3)
    assert plan["feasible"]
    assert result["summary"]["design_feasible"]
    assert abs(result["summary"]["min_design_margin_n"]) < 1e-10
    impossible = plan_fixed_closure(material, t, acceleration, GraspSpec(mass_kg=10))
    assert not impossible["feasible"]
    assert impossible["required_strain"] > .08


def test_force_control_cannot_hide_excessive_creep(material):
    t = np.linspace(0, 10, 1001)
    normal = np.full_like(t, 2.)
    u = material.compression(t, normal)
    result = assess_grasp(t, normal, u, np.zeros_like(t))
    assert result["summary"]["sticking_feasible"]
    assert result["summary"]["strain_limit_exceeded"]
    assert not result["summary"]["design_feasible"]
