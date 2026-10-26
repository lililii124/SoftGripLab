"""Two-contact vertical grasp feasibility, not a post-slip dynamics solver."""

from dataclasses import dataclass

import numpy as np

from .materials import Geometry, StandardLinearSolid, positive, ramp_hold, sampled_input


@dataclass(frozen=True)
class GraspSpec:
    mass_kg: float = 0.05
    friction: float = 0.5
    strain_limit: float = 0.08
    safety_factor: float = 1.2
    gravity_m_s2: float = 9.81

    def __post_init__(self):
        positive(mass_kg=self.mass_kg, strain_limit=self.strain_limit,
                 safety_factor=self.safety_factor, gravity_m_s2=self.gravity_m_s2)
        if not np.isfinite(self.friction) or self.friction < 0:
            raise ValueError("friction must be finite and nonnegative")
        if self.strain_limit >= 1 or self.safety_factor < 1:
            raise ValueError("strain_limit must be < 1 and safety_factor >= 1")


def half_sine_acceleration(time_s, start_s, duration_s, peak_m_s2):
    positive(duration_s=duration_s)
    if not np.isfinite(start_s) or start_s < 0 or not np.isfinite(peak_m_s2):
        raise ValueError("pulse start and peak must be finite; start must be nonnegative")
    phase = (np.asarray(time_s) - start_s) / duration_s
    return np.where((phase >= 0) & (phase <= 1), peak_m_s2 * np.sin(np.pi * phase), 0)


def assess_grasp(time_s, normal_force_n, compression_m, acceleration_m_s2,
                 spec=GraspSpec(), geometry=Geometry(), evaluation_start_s=0.0):
    """N is force per finger; both contacts together support up to 2 mu N.

    Positive acceleration increases upward support demand. Negative margins
    indicate sticking is infeasible, not a prediction of slip distance or damage.
    """
    t, normal = sampled_input(time_s, normal_force_n)
    _, displacement = sampled_input(t, compression_m)
    _, acceleration = sampled_input(t, acceleration_m_s2)
    if not np.isfinite(evaluation_start_s) or not t[0] <= evaluation_start_s <= t[-1]:
        raise ValueError("evaluation start must lie inside the sampled interval")
    if np.any(normal < -1e-10) or np.any(displacement < -1e-10):
        raise ValueError("grasp requires compressive contact force and displacement")
    normal = np.maximum(normal, 0)
    demand = spec.mass_kg * np.abs(spec.gravity_m_s2 + acceleration)
    capacity = 2 * spec.friction * normal
    margin = capacity - demand
    design_margin = capacity - spec.safety_factor * demand
    strain = displacement / geometry.thickness_m
    active = t >= evaluation_start_s
    return {
        "capacity_n": capacity, "demand_n": demand, "margin_n": margin,
        "design_margin_n": design_margin, "strain": strain,
        "summary": {
            "min_margin_n": float(np.min(margin[active])),
            "min_design_margin_n": float(np.min(design_margin[active])),
            "peak_strain": float(np.max(strain[active])),
            "sticking_feasible": bool(np.all(margin[active] >= -1e-9)),
            "design_feasible": bool(np.all(design_margin[active] >= -1e-9)
                                    and np.max(strain[active]) <= spec.strain_limit + 1e-9),
            "strain_limit_exceeded": bool(np.max(strain[active]) > spec.strain_limit + 1e-9),
        },
    }


def plan_fixed_closure(material: StandardLinearSolid, time_s, acceleration_m_s2,
                       spec=GraspSpec(), geometry=Geometry(), ramp_s=0.2,
                       evaluation_start_s=0.3):
    """Minimum fixed closure for the supplied deterministic disturbance envelope.

    Linear scaling of the reference material response avoids a new optimizer.
    Infeasible plans are explicitly returned; closure is never silently clamped.
    """
    t, acceleration = sampled_input(time_s, acceleration_m_s2)
    if spec.friction == 0:
        raise ValueError("a vertical friction grasp requires positive friction")
    unit_displacement = ramp_hold(t, geometry.thickness_m, ramp_s)
    per_strain = material.force(t, unit_displacement, geometry)
    active = t >= evaluation_start_s
    if not np.any(active) or evaluation_start_s <= 0 or evaluation_start_s > t[-1]:
        raise ValueError("choose an evaluation start after initial contact, inside the record")
    if np.any(per_strain[active] <= 0):
        raise ValueError("no positive contact capacity in the evaluation interval")
    required = spec.safety_factor * spec.mass_kg * np.abs(spec.gravity_m_s2 + acceleration)
    strain = float(np.max(required[active] / (2 * spec.friction * per_strain[active])))
    return {"closure_m": strain * geometry.thickness_m,
            "required_strain": strain, "feasible": bool(strain <= spec.strain_limit),
            "assumption": "known acceleration envelope and friction; fitted material point estimate"}
