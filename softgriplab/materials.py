"""Small-strain uniaxial reference models, integrated by SciPy.

Compression is positive. Force is the force on ONE platen; displacement is
the TOTAL reduction in specimen thickness. SI units throughout.
"""

from dataclasses import dataclass

import numpy as np
from scipy.signal import lsim


def positive(**values):
    for name, value in values.items():
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")


def sampled_input(time_s, values):
    t, u = np.asarray(time_s, float), np.asarray(values, float)
    if t.ndim != 1 or u.shape != t.shape or len(t) < 3:
        raise ValueError("time and input must be equal-length 1D arrays, length >= 3")
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(u)):
        raise ValueError("time and input must be finite")
    if t[0] != 0 or np.any(np.diff(t) <= 0):
        raise ValueError("time must start at zero and strictly increase")
    if not np.allclose(np.diff(t), t[1], rtol=1e-6, atol=1e-12):
        raise ValueError("SciPy lsim requires uniform sampling; resample explicitly first")
    return t, u


@dataclass(frozen=True)
class Geometry:
    thickness_m: float = 0.02
    area_m2: float = 0.0004

    def __post_init__(self):
        positive(thickness_m=self.thickness_m, area_m2=self.area_m2)


@dataclass(frozen=True)
class StandardLinearSolid:
    equilibrium_pa: float
    branch_pa: float
    relaxation_s: float

    def __post_init__(self):
        positive(equilibrium_pa=self.equilibrium_pa, branch_pa=self.branch_pa,
                 relaxation_s=self.relaxation_s)

    @property
    def instantaneous_pa(self):
        return self.equilibrium_pa + self.branch_pa

    def force(self, time_s, compression_m, geometry=Geometry()):
        """Zero prehistory; first input may jump, then linearly interpolated samples.

        Relaxation modulus E(t) = E_inf + E_branch exp(-t/tau).
        State z is low-pass filtered strain: z'=(epsilon-z)/tau.
        Stress = E_0 epsilon - E_branch z. This is a material reference,
        not a unilateral contact solver: unloading can produce tensile stress.
        """
        t, displacement = sampled_input(time_s, compression_m)
        system = (np.array([[-1 / self.relaxation_s]]),
                  np.array([[1 / self.relaxation_s]]),
                  np.array([[-self.branch_pa]]),
                  np.array([[self.instantaneous_pa]]))
        _, stress, _ = lsim(system, displacement / geometry.thickness_m, t)
        return np.asarray(stress) * geometry.area_m2

    def compression(self, time_s, normal_force_n, geometry=Geometry()):
        """Ideal force control (creep); same zero-prehistory convention."""
        t, force = sampled_input(time_s, normal_force_n)
        # Inverse of E(s) = (E_0 tau s + E_inf)/(tau s + 1).
        transfer = ([self.relaxation_s, 1],
                    [self.instantaneous_pa * self.relaxation_s, self.equilibrium_pa])
        _, strain, _ = lsim(transfer, force / geometry.area_m2, t)
        return np.asarray(strain) * geometry.thickness_m


def ramp_hold(time_s, amplitude_m, ramp_s):
    positive(amplitude_m=amplitude_m, ramp_s=ramp_s)
    t = np.asarray(time_s, float)
    return amplitude_m * np.clip(t / ramp_s, 0, 1)
