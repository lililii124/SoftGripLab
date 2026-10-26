"""Fit material parameters to measured compression traces with SciPy."""

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from .materials import Geometry, StandardLinearSolid, positive, sampled_input


@dataclass(frozen=True)
class CompressionTrace:
    time_s: np.ndarray
    compression_m: np.ndarray
    force_n: np.ndarray
    geometry: Geometry = Geometry()
    noise_std_n: float = 1.0

    def __post_init__(self):
        t, u = sampled_input(self.time_s, self.compression_m)
        f = np.asarray(self.force_n, float)
        if f.shape != t.shape or not np.all(np.isfinite(f)):
            raise ValueError("force must have the same shape and contain finite values")
        if np.any(u < 0) or np.max(u) <= 0:
            raise ValueError("compression must be nonnegative and contain a positive load")
        positive(noise_std_n=self.noise_std_n)
        for name, value in (("time_s", t), ("compression_m", u), ("force_n", f)):
            object.__setattr__(self, name, value.copy())

    @classmethod
    def from_csv(cls, path, geometry=Geometry(), noise_std_n=1.0):
        data = np.genfromtxt(Path(path), delimiter=",", names=True, encoding="utf-8-sig")
        required = {"time_s", "compression_m", "force_n"}
        if not required.issubset(data.dtype.names or ()):
            raise ValueError(f"CSV requires columns: {', '.join(sorted(required))}")
        return cls(data["time_s"], data["compression_m"], data["force_n"],
                   geometry, noise_std_n)

    def to_csv(self, path):
        np.savetxt(path, np.column_stack((self.time_s, self.compression_m, self.force_n)),
                   delimiter=",", header="time_s,compression_m,force_n", comments="")


@dataclass(frozen=True)
class FitResult:
    material: StandardLinearSolid
    weighted_rmse: float
    jacobian_condition: float
    log_parameter_std: tuple[float, ...]
    warnings: tuple[str, ...]
    evaluations: int

    def to_dict(self):
        return asdict(self)


def fit_material(traces, initial=None):
    """Joint fit in log parameters with deterministic multi-start initialization.

    The Jacobian condition and local log-parameter standard errors are diagnostics,
    not guarantees of uniqueness. Covariance assumes independent, correctly
    weighted residuals and local linearity. No friction parameter is inferred
    from normal compression alone.
    """
    traces = tuple(traces)
    if not traces:
        raise ValueError("provide at least one compression trace")
    scales = [max(np.max(tr.force_n), tr.noise_std_n) * tr.geometry.thickness_m /
              (tr.geometry.area_m2 * np.max(tr.compression_m)) for tr in traces]
    scale = max(float(np.median(scales)), 10.0)
    duration = max(tr.time_s[-1] for tr in traces)
    starts = ([initial] if initial is not None else [
        StandardLinearSolid(scale * f, scale * (1 - f), duration * tau)
        for f, tau in ((0.4, 0.1), (0.7, 0.3), (0.2, 0.03))])

    def residual(log_parameters):
        material = StandardLinearSolid(*np.exp(log_parameters))
        return np.concatenate([(material.force(tr.time_s, tr.compression_m, tr.geometry)
                                - tr.force_n) / tr.noise_std_n for tr in traces])

    lower, upper = np.log([1e-3, 1e-3, 1e-6]), np.log([1e12, 1e12, 1e6])
    fits = [least_squares(residual,
                          np.clip(np.log(list(asdict(start).values())), lower + 1e-8, upper - 1e-8),
                          bounds=(lower, upper), x_scale="jac", max_nfev=300)
            for start in starts]
    successful = [fit for fit in fits if fit.success and np.all(np.isfinite(fit.fun))]
    if not successful:
        raise RuntimeError("material optimization failed to converge")
    best = min(successful, key=lambda fit: np.dot(fit.fun, fit.fun))
    singular = np.linalg.svd(best.jac, compute_uv=False)
    condition = float(singular[0] / max(singular[-1], np.finfo(float).tiny))
    dof = max(len(best.fun) - 3, 1)
    covariance = np.linalg.pinv(best.jac.T @ best.jac) * np.dot(best.fun, best.fun) / dof
    std = np.sqrt(np.maximum(np.diag(covariance), 0))
    material = StandardLinearSolid(*np.exp(best.x))
    warnings = []
    if condition > 1e5:
        warnings.append("Poorly conditioned fit: parameters may not be separately identifiable.")
    if np.max(std) > 0.5:
        warnings.append("Large local parameter uncertainty; collect additional loading histories.")
    if duration < 3 * material.relaxation_s:
        warnings.append("Record is shorter than three fitted relaxation times.")
    if any(tr.time_s[1] > material.relaxation_s / 5 for tr in traces):
        warnings.append("Sampling may be too coarse to resolve the fitted relaxation time.")
    if np.any(best.active_mask):
        warnings.append("One or more parameters reached an optimization bound.")
    return FitResult(material, float(np.sqrt(np.mean(best.fun ** 2))), condition,
                     tuple(map(float, std)), tuple(warnings), sum(fit.nfev for fit in fits))
