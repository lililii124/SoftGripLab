"""Axial SLS specimen with a finite-range adhesive interface (SI units).

This is a two-node, lumped-mass reference, not a 3D contact solver. The bottom
node is clamped during probing and free during release. Surface displacement,
force and separation are solved together; release never disables adhesion.
"""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from .materials import Geometry, positive


@dataclass(frozen=True)
class ContactSpec:
    geometry: Geometry = Geometry(thickness_m=0.02, area_m2=0.0002)
    top_mass_kg: float = 0.0005
    bottom_mass_kg: float = 0.0095
    peak_gap_m: float = 0.0005
    cutoff_m: float = 0.002
    compression_stiffness_n_m: float = 20000.0
    gravity_m_s2: float = 9.81

    def __post_init__(self):
        positive(top_mass_kg=self.top_mass_kg, bottom_mass_kg=self.bottom_mass_kg,
                 peak_gap_m=self.peak_gap_m, cutoff_m=self.cutoff_m,
                 compression_stiffness_n_m=self.compression_stiffness_n_m,
                 gravity_m_s2=self.gravity_m_s2)
        if self.peak_gap_m >= self.cutoff_m:
            raise ValueError("peak gap must be smaller than the interaction cutoff")


@dataclass(frozen=True)
class Probe:
    name: str
    amplitude_m: float
    ramp_s: float
    duration_s: float = 0.36

    def __post_init__(self):
        positive(amplitude_m=self.amplitude_m, ramp_s=self.ramp_s,
                 duration_s=self.duration_s)
        if self.ramp_s > self.duration_s:
            raise ValueError("probe ramp must fit within the duration")

    def position(self, t):
        s = np.clip(np.asarray(t) / self.ramp_s, 0, 1)
        return self.amplitude_m * s**3 * (10 - 15 * s + 6 * s**2)


@dataclass(frozen=True)
class Release:
    acceleration_m_s2: float
    pulse_s: float = 0.06
    duration_s: float = 0.18

    def __post_init__(self):
        positive(pulse_s=self.pulse_s, duration_s=self.duration_s)
        if not np.isfinite(self.acceleration_m_s2) or self.acceleration_m_s2 < 0:
            raise ValueError("release acceleration must be finite and nonnegative")
        if self.duration_s < self.pulse_s:
            raise ValueError("release duration must include the acceleration pulse")

    def position(self, t):
        # a(t) = a_peak (1 - cos(2 pi t/T))/2, followed by constant velocity.
        t = np.asarray(t)
        s = np.clip(t, 0, self.pulse_s)
        omega = 2 * np.pi / self.pulse_s
        q = self.acceleration_m_s2 * (s**2 / 4 + (np.cos(omega*s) - 1) / (2*omega**2))
        return q + self.acceleration_m_s2 * self.pulse_s / 2 * np.maximum(t-self.pulse_s, 0)


CALIBRATION = Probe("initial", 0.00012, 0.04, 0.06)
PROBES = tuple(Probe(f"{label}_{speed}", amplitude, ramp)
               for label, amplitude in (("small", 0.00035), ("medium", 0.0009),
                                        ("large", 0.0016), ("pull_off", 0.0028))
               for speed, ramp in (("fast", 0.025), ("slow", 0.24)))


def parameter_array(parameters):
    """Columns: E_inf [Pa], E_branch [Pa], tau [s], W [J/m2]."""
    p = np.atleast_2d(np.asarray(parameters, dtype=float))
    if p.ndim != 2 or p.shape[1] != 4 or not np.isfinite(p).all() or np.any(p <= 0):
        raise ValueError("parameters must be finite positive rows of E_inf,E_branch,tau,W")
    return p


def interface_force(gap, work, spec=ContactSpec()):
    """Triangular tensile traction; its area is exactly W per unit area.

    Compression uses a penalty spring. The tensile potential is reversible:
    no fracture damage, rate dependence or externally commanded bond switch.
    """
    peak = 2 * np.asarray(work) * spec.geometry.area_m2 / spec.cutoff_m
    shape = np.maximum(0, np.minimum(np.asarray(gap) / spec.peak_gap_m,
                                    (spec.cutoff_m-gap) / (spec.cutoff_m-spec.peak_gap_m)))
    return np.where(np.asarray(gap) < 0, spec.compression_stiffness_n_m * gap, peak*shape)


def initial_state(parameters, release=False, spec=ContactSpec()):
    p = parameter_array(parameters)
    state = np.zeros((len(p), 5))  # bottom x,v; top x,v; filtered extension z
    if release:
        weight = (spec.top_mass_kg + spec.bottom_mass_kg) * spec.gravity_m_s2
        peak = 2*p[:, 3]*spec.geometry.area_m2/spec.cutoff_m
        if np.any(peak <= weight):
            raise ValueError("release requires an initially sustainable static adhesive hold")
        gap = weight * spec.peak_gap_m / peak
        extension = (spec.bottom_mass_kg * spec.gravity_m_s2 * spec.geometry.thickness_m /
                     (p[:, 0] * spec.geometry.area_m2))
        state[:, 2] = -gap
        state[:, 0] = -gap - extension
        state[:, 4] = extension
    return state


def _rhs(state, tool, p, release, spec):
    factor = spec.geometry.area_m2 / spec.geometry.thickness_m
    extension = state[:, 2] - state[:, 0]
    bulk = factor * ((p[:, 0] + p[:, 1])*extension - p[:, 1]*state[:, 4])
    contact = interface_force(tool-state[:, 2], p[:, 3], spec)
    out = np.empty_like(state)
    if release:
        out[:, 0] = state[:, 1]
        out[:, 1] = bulk/spec.bottom_mass_kg - spec.gravity_m_s2
    else:
        out[:, :2] = 0
    out[:, 2] = state[:, 3]
    out[:, 3] = (contact-bulk)/spec.top_mass_kg - (spec.gravity_m_s2 if release else 0)
    out[:, 4] = (extension-state[:, 4])/p[:, 2]
    return out


def simulate(parameters, protocol, *, spec=ContactSpec(), dt=0.0001,
             samples=37, method="rk4"):
    """Batch RK4 or independent SciPy Radau integration of the same equations.

    Output times are identical for either method. Radau is an integration
    cross-check, NOT an independent constitutive/contact model or experiment.
    """
    p = parameter_array(parameters)
    positive(dt=dt)
    if type(samples) is not int or samples < 3:
        raise ValueError("samples must be an integer >= 3")
    if not isinstance(protocol, (Probe, Release)) or method not in ("rk4", "radau"):
        raise ValueError("provide a Probe/Release and method rk4 or radau")
    release = isinstance(protocol, Release)
    times = np.linspace(0, protocol.duration_s, samples)
    state = initial_state(p, release, spec)
    history = np.empty((len(p), samples, 5))
    history[:, 0] = state
    if method == "radau":
        for i, parameters_i in enumerate(p):
            solution = solve_ivp(
                lambda t, y: _rhs(y[None], protocol.position(t), parameters_i[None],
                                  release, spec)[0], (0, times[-1]), state[i],
                t_eval=times, method="Radau", rtol=2e-8, atol=1e-11, max_step=0.001)
            if not solution.success:
                raise RuntimeError(solution.message)
            history[i] = solution.y.T
    else:
        # Conservative bound covers compressed contact and the stiffest bulk.
        bulk_k = (p[:, 0]+p[:, 1])*spec.geometry.area_m2/spec.geometry.thickness_m
        interface_k = np.maximum(spec.compression_stiffness_n_m,
                                 2*p[:, 3]*spec.geometry.area_m2/spec.cutoff_m/spec.peak_gap_m)
        omega_bound = np.sqrt((bulk_k+interface_k)/spec.top_mass_kg + bulk_k/spec.bottom_mass_kg)
        if dt * max(float(omega_bound.max()), float((1/p[:, 2]).max())) > 0.8:
            raise ValueError("dt exceeds the conservative RK4 resolution bound; reduce dt")
        substeps = int(np.ceil((times[1]-times[0])/dt))
        h = (times[1]-times[0])/substeps
        grid = np.arange((samples-1)*substeps*2+1)*h/2
        tool = protocol.position(grid)
        for j in range(1, samples):
            for k in range((j-1)*substeps, j*substeps):
                k1 = _rhs(state, tool[2*k], p, release, spec)
                k2 = _rhs(state+h/2*k1, tool[2*k+1], p, release, spec)
                k3 = _rhs(state+h/2*k2, tool[2*k+1], p, release, spec)
                k4 = _rhs(state+h*k3, tool[2*k+2], p, release, spec)
                state = state + h/6*(k1+2*k2+2*k3+k4)
            history[:, j] = state
        if not np.isfinite(history).all():
            raise RuntimeError("contact integration produced nonfinite states")
    tool = protocol.position(times)
    gap = tool[None] - history[:, :, 2]
    force = interface_force(gap, p[:, 3, None], spec)
    # A visible mid-height marker on the axial specimen, not its internal stress.
    marker = (history[:, :, 0] + history[:, :, 2])/2
    terminal_window = times >= times[-1] - 0.02 - 1e-12
    return {"time_s": times, "tool_m": tool, "state": history, "force_n": force,
            "marker_m": marker, "gap_m": gap,
            "detached": np.all(gap[:, terminal_window] >= spec.cutoff_m, axis=1)}
