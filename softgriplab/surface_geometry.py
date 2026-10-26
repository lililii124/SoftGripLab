"""Small-scan geometry estimation under an explicit swept-ellipse prior.

Input points have no surface-node correspondences. Axial registration, end
planes, solid topology and elliptical cross-sections are assumed known.
An unobserved back surface is a prior-dependent extrapolation, not a scan.
"""

from dataclasses import dataclass
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares
from skfem import Basis, ElementTetP1, ElementVector, MeshTet

from .complex_contact import HyperelasticContact, LENGTH_M, complex_mesh
from .meshes import tetrahedral_block


@dataclass
class TubeSurface:
    stations: np.ndarray
    values: np.ndarray  # cy, cz, log(ry), log(rz), in dimensionless coordinates

    def section(self, x):
        fields = CubicSpline(self.stations, self.values, axis=0, extrapolate=False)(x)
        return fields[..., 0], fields[..., 1], np.exp(fields[..., 2]), np.exp(fields[..., 3])

    def surface(self, x, angle):
        cy, cz, ry, rz = self.section(x)
        return np.stack((x, cy+ry*np.cos(angle), cz+rz*np.sin(angle)), axis=-1)

    def map(self, reference):
        x, y, z = np.asarray(reference).T
        cy, cz, ry, rz = self.section(x)
        return np.column_stack((x, cy+ry*y*np.sqrt(1-z*z/2), cz+rz*z*np.sqrt(1-y*y/2)))

    def residual(self, points):
        """First-order normal distance to the implicit elliptical surface."""
        x, y, z = np.asarray(points).T
        cy, cz, ry, rz = self.section(x)
        a, b = (y-cy)/ry, (z-cz)/rz
        return (a*a+b*b-1)/np.maximum(2*np.sqrt((a/ry)**2+(b/rz)**2), 1e-12)


def reconstruct_surface(points_m, *, sigma_m=4e-5, stations=13):
    points = np.asarray(points_m, dtype=float)/LENGTH_M
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 6*stations:
        raise ValueError("expected at least six registered 3D scan points per axial station")
    if not np.isfinite(points).all() or sigma_m <= 0 or stations < 4:
        raise ValueError("scan, noise and station count must be finite and valid")
    if np.min(points[:, 0]) < -3 or np.max(points[:, 0]) > 3:
        raise ValueError("registered scan points must lie between the known end planes")
    if np.ptp(points[:, 0]) < 5.5:
        raise ValueError("scan must cover the axial extent; hidden axial regions are unsupported")
    knots = np.linspace(-3, 3, stations)
    design = CubicSpline(knots, np.eye(stations), axis=0, extrapolate=False)(points[:, 0])
    initial = np.tile([0., 0., np.log(.8), np.log(.72)], (stations, 1))
    lower = np.tile([-1.2, -1.2, np.log(.25), np.log(.25)], (stations, 1))
    upper = np.tile([1.2, 1.2, np.log(1.5), np.log(1.5)], (stations, 1))
    def residual(parameters, regularize=True):
        values = parameters.reshape(stations, 4)
        cy, cz, log_y, log_z = (design@values).T
        ry, rz = np.exp(log_y), np.exp(log_z)
        a, b = (points[:, 1]-cy)/ry, (points[:, 2]-cz)/rz
        distance = (a*a+b*b-1)/np.maximum(2*np.sqrt((a/ry)**2+(b/rz)**2), 1e-12)
        result = distance/(sigma_m/LENGTH_M)
        return np.r_[result, np.diff(values, n=2, axis=0).ravel()] if regularize else result
    result = least_squares(residual, initial.ravel(), bounds=(lower.ravel(), upper.ravel()),
                           loss="soft_l1", f_scale=2., max_nfev=150, ftol=1e-9, xtol=1e-9, gtol=1e-8)
    if not result.success:
        raise RuntimeError("surface reconstruction failed to converge")
    geometry = TubeSurface(knots, result.x.reshape(stations, 4))
    singular = np.linalg.svd(result.jac[:len(points)], compute_uv=False)
    report = {"scan_points": len(points), "geometry_coefficients": result.x.size,
              "scan_normal_rmse_m": float(np.sqrt(np.mean(geometry.residual(points)**2))*LENGTH_M),
              "weighted_scan_jacobian_condition": float(singular[0]/max(singular[-1], 1e-15)),
              "source": "unlabelled xyz only; no true surface or correspondence supplied",
              "prior": "registered solid swept ellipses, smooth axial profiles, known end planes",
              "uncertainty": "conditioning diagnostic only; no calibrated confidence interval"}
    return geometry, report


class ScannedContact(HyperelasticContact):
    """Reuse all bulk/contact/Newton mechanics with a reconstructed reference mesh."""

    def __init__(self, geometry, cells=(12, 4, 4)):
        super().__init__(cells)
        reference, _, _ = tetrahedral_block(size=(6, 2, 2), cells=cells)
        self.points = geometry.map(reference)
        volumes = np.linalg.det(np.transpose(self.points[self.tets[:, 1:]]-self.points[self.tets[:, 0, None]], (0,2,1)))
        if not np.isfinite(volumes).all() or np.min(volumes) <= 0:
            raise ValueError("reconstructed geometry has invalid tetrahedra")
        self.mesh = MeshTet(self.points.T, self.tets.T)
        self.basis = Basis(self.mesh, ElementVector(ElementTetP1()), intorder=1)
        self.fixed = self.basis.nodal_dofs[:, self.fixed_nodes].ravel()
        self.free = np.setdiff1d(np.arange(self.basis.N), self.fixed)
        area = np.linalg.norm(np.cross(self.points[self.faces[:,1]]-self.points[self.faces[:,0]],
                                      self.points[self.faces[:,2]]-self.points[self.faces[:,0]]), axis=1)/2
        self.area = np.zeros(len(self.points))
        for corner in range(3):
            np.add.at(self.area, self.faces[:,corner], area/3)
        self.surface = np.flatnonzero(self.area > 0)


@dataclass(frozen=True)
class PlacedProbe:
    origin: tuple
    radius: float = .65

    def center(self, offset):
        return np.asarray(self.origin)+[0,0,offset]


def plan_probes(surface):
    """Plan once from the measured surface; never recenter on hidden true geometry."""
    probes = []
    for location in (-1.6, 0., 1.5):
        radius = .65
        cy = float(surface.section(location)[0])
        x, angle = np.meshgrid(np.linspace(location-radius, location+radius, 161),
                               np.linspace(0, 2*np.pi, 241), indexing="ij")
        xyz = surface.surface(x, angle)
        d2 = (xyz[...,0]-location)**2+(xyz[...,1]-cy)**2
        heights = np.where(d2 <= radius**2, xyz[...,2]+np.sqrt(np.maximum(radius**2-d2, 0)), -np.inf)
        probes.append(PlacedProbe((location, cy, float(heights.max())), radius))
    return tuple(probes)


def synthetic_surface():
    """Validation generator only. Its coefficients never enter reconstruction."""
    x = np.linspace(-3, 3, 37)
    s = x/3
    cy = .25*np.sin(np.pi*s)+.08*np.sin(2*np.pi*s)
    cz = .10+.15*s*s+.07*np.sin(np.pi*s)
    ry = .75*(1+.20*np.cos(2*np.pi*s)+.12*s)
    rz = .90*ry*(1+.12*np.sin(np.pi*s))
    return TubeSurface(x, np.column_stack((cy, cz, np.log(ry), np.log(rz))))
