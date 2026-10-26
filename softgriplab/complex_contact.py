"""3D hyperelastic adhesive probing using scikit-fem and SciPy sparse Newton.

Lengths are nondimensionalized by LENGTH_M and energy by mu*LENGTH_M**3.
For fixed Poisson ratio and dimensionless contact penalty, shape depends on
eta = W/(mu*LENGTH_M), while reaction force scales with mu*LENGTH_M**2.
This exact similarity is used by the inverse problem, not a shape animation.
"""

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import spsolve
from skfem import Basis, BilinearForm, ElementTetP1, ElementVector, LinearForm, MeshTet
from skfem.helpers import ddot, det, inv, transpose

from .meshes import tetrahedral_block

LENGTH_M = 0.01
POISSON = 0.35
LAMBDA_OVER_MU = 2*POISSON/(1-2*POISSON)


def deformation(displacement):
    F = displacement.grad + np.eye(3)[:, :, None, None]
    J = det(F)
    return F, J, transpose(inv(F))


@LinearForm
def internal_force(v, w):
    F, J, inv_t = deformation(w["displacement"])
    stress = F + (LAMBDA_OVER_MU*np.log(J)-1)*inv_t
    return ddot(stress, v.grad)


@BilinearForm
def internal_tangent(u, v, w):
    F, J, inv_t = deformation(w["displacement"])
    geometric = np.einsum("ij...,kj...,kl...->il...", inv_t, u.grad, inv_t)
    change = (u.grad + (1-LAMBDA_OVER_MU*np.log(J))*geometric
              + LAMBDA_OVER_MU*ddot(inv_t, u.grad)*inv_t)
    return ddot(change, v.grad)


def shape_map(x, y, z):
    s = x/3
    radius = 0.78*(1+0.28*np.cos(2*np.pi*s)+0.14*s)
    cy = 0.40*np.sin(np.pi*s)
    cz = 0.22*s**2
    return np.stack((x, cy+radius*y*np.sqrt(1-z*z/2),
                     cz+0.9*radius*z*np.sqrt(1-y*y/2)), axis=-1)


def complex_mesh(cells=(16, 6, 6)):
    points, tets, faces = tetrahedral_block(size=(6, 2, 2), cells=cells)
    points = shape_map(*points.T)
    volumes = np.linalg.det(np.transpose(points[tets[:, 1:]]-points[tets[:, 0, None]], (0, 2, 1)))/6
    if np.min(volumes) <= 0:
        raise ValueError("invalid reference mesh")
    return points, tets, faces


@dataclass(frozen=True)
class SurfaceProbe:
    x: float
    radius: float = 0.65

    def center(self, offset):
        s = self.x/3
        cy = 0.40*np.sin(np.pi*s)
        # Analytic surface sampling defines the SAME tool pose for every mesh.
        x, angle = np.meshgrid(np.linspace(self.x-self.radius, self.x+self.radius, 161),
                               np.linspace(0, 2*np.pi, 241), indexing="ij")
        t = x/3
        r = 0.78*(1+0.28*np.cos(2*np.pi*t)+0.14*t)
        y = 0.40*np.sin(np.pi*t)+r*np.cos(angle)
        z = 0.22*t**2+0.9*r*np.sin(angle)
        d2 = (x-self.x)**2+(y-cy)**2
        height = np.where(d2 <= self.radius**2, z+np.sqrt(np.maximum(self.radius**2-d2, 0)), -np.inf)
        return np.array([self.x, cy, float(height.max()+offset)])


PROBE_SITES = (SurfaceProbe(-1.6), SurfaceProbe(0.0), SurfaceProbe(1.5))


class HyperelasticContact:
    def __init__(self, cells=(16, 6, 6)):
        self.points, self.tets, self.faces = complex_mesh(cells)
        self.mesh = MeshTet(self.points.T, self.tets.T)
        self.basis = Basis(self.mesh, ElementVector(ElementTetP1()), intorder=1)
        self.fixed_nodes = np.flatnonzero(np.abs(self.points[:, 0]) > 2.999)
        self.fixed = self.basis.nodal_dofs[:, self.fixed_nodes].ravel()
        self.free = np.setdiff1d(np.arange(self.basis.N), self.fixed)
        area = np.linalg.norm(np.cross(self.points[self.faces[:, 1]]-self.points[self.faces[:, 0]],
                                      self.points[self.faces[:, 2]]-self.points[self.faces[:, 0]]), axis=1)/2
        self.area = np.zeros(len(self.points))
        for corner in range(3):
            np.add.at(self.area, self.faces[:, corner], area/3)
        self.surface = np.flatnonzero(self.area > 0)
        self.peak_gap, self.cutoff, self.penalty = 0.025, 0.10, 100.0

    def contact(self, displacement, center, radius, eta, tangent=True):
        position = self.points + displacement[self.basis.nodal_dofs].T
        ids = self.surface
        delta = position[ids]-center
        distance = np.linalg.norm(delta, axis=1)
        n = delta/np.maximum(distance[:, None], 1e-12)
        gap = distance-radius
        peak = 2*eta/self.cutoff
        ascending = (gap >= 0) & (gap < self.peak_gap)
        descending = (gap >= self.peak_gap) & (gap < self.cutoff)
        compressed = gap < 0
        traction, slope, potential = np.zeros((3, len(ids)))
        traction[compressed] = self.penalty*gap[compressed]
        slope[compressed] = self.penalty
        potential[compressed] = 0.5*self.penalty*gap[compressed]**2-eta
        traction[ascending] = peak*gap[ascending]/self.peak_gap
        slope[ascending] = peak/self.peak_gap
        potential[ascending] = 0.5*peak/self.peak_gap*gap[ascending]**2-eta
        traction[descending] = peak*(self.cutoff-gap[descending])/(self.cutoff-self.peak_gap)
        slope[descending] = -peak/(self.cutoff-self.peak_gap)
        potential[descending] = -0.5*peak/(self.cutoff-self.peak_gap)*(self.cutoff-gap[descending])**2
        nodal_gradient = self.area[ids, None]*traction[:, None]*n
        gradient = np.zeros(self.basis.N)
        gradient[self.basis.nodal_dofs[:, ids]] = nodal_gradient.T
        energy = float(np.dot(self.area[ids], potential))
        # dU/dcenter = -sum(dU/dnodes); reaction on tool is -dU/dcenter.
        reaction = np.sum(nodal_gradient, axis=0)
        matrix = None
        if tangent:
            nn = n[:, :, None]*n[:, None, :]
            block = self.area[ids, None, None]*(slope[:, None, None]*nn +
                    (traction/np.maximum(distance, 1e-12))[:, None, None]*(np.eye(3)-nn))
            dofs = self.basis.nodal_dofs[:, ids].T
            rows = np.broadcast_to(dofs[:, :, None], block.shape).ravel()
            cols = np.broadcast_to(dofs[:, None, :], block.shape).ravel()
            matrix = coo_matrix((block.ravel(), (rows, cols)), shape=(self.basis.N, self.basis.N)).tocsr()
        return energy, gradient, matrix, reaction, gap

    def evaluate(self, u, center, radius, eta, tangent=True):
        field = self.basis.interpolate(u)
        F, J, _ = deformation(field)
        if np.min(J) <= 0 or not np.isfinite(J).all():
            return np.inf, None, None, None, float(np.min(J))
        log_j = np.log(J)
        density = 0.5*(np.sum(F*F, axis=(0, 1))-3)-log_j+0.5*LAMBDA_OVER_MU*log_j**2
        energy = float(np.sum(density*self.basis.dx))
        gradient = internal_force.assemble(self.basis, displacement=field)
        matrix = internal_tangent.assemble(self.basis, displacement=field) if tangent else None
        ce, cg, ck, reaction, _ = self.contact(u, center, radius, eta, tangent)
        return energy+ce, gradient+cg, matrix+ck if tangent else None, reaction, float(J.min())

    def equilibrium(self, center, radius, eta, initial=None, tolerance=2e-6):
        u = np.zeros(self.basis.N) if initial is None else np.array(initial, copy=True)
        for iteration in range(70):
            energy, gradient, matrix, reaction, min_j = self.evaluate(u, center, radius, eta)
            residual = float(np.linalg.norm(gradient[self.free], ord=np.inf))
            if residual < tolerance:
                return u, reaction, {"iterations": iteration, "residual": residual, "min_j": min_j}
            K = matrix[self.free][:, self.free]
            rhs = gradient[self.free]
            accepted = False
            for shift in (0., 0.001, 0.03, 0.3, 3., 30.):
                step = spsolve(K+diags(np.full(len(rhs), shift)), -rhs)
                direction = float(rhs@step)
                if not np.isfinite(step).all() or direction >= 0:
                    continue
                for reduction in range(15):
                    alpha = 0.5**reduction
                    trial = u.copy()
                    trial[self.free] += alpha*step
                    trial_energy = self.evaluate(trial, center, radius, eta, tangent=False)[0]
                    if np.isfinite(trial_energy) and trial_energy <= energy+1e-4*alpha*direction:
                        u = trial
                        accepted = True
                        break
                if accepted:
                    break
            if not accepted:
                raise RuntimeError(f"contact Newton line search failed, residual={residual:g}")
        raise RuntimeError(f"contact Newton failed to converge, residual={residual:g}")

    def cycle(self, eta, probe=PROBE_SITES[1], offsets=None):
        if offsets is None:
            offsets = np.r_[np.linspace(0.13, -0.16, 12), np.linspace(-0.16, 0.24, 16)[1:]]
        u = np.zeros(self.basis.N)
        positions, reactions, reports = [], [], []
        origin = probe.center(0)
        for offset in offsets:
            center = origin + [0, 0, offset]
            u, reaction, report = self.equilibrium(center, probe.radius, eta, u)
            positions.append(self.points+u[self.basis.nodal_dofs].T)
            reactions.append(reaction)
            reports.append(report)
        return {"position": np.asarray(positions), "reaction": np.asarray(reactions),
                "offsets": np.asarray(offsets), "centers": origin[None]+np.array([0, 0, 1])*np.asarray(offsets)[:, None],
                "reports": reports}

    def surface_observations(self, positions):
        """Area-weighted motion of three visible surface patches, in x,y,z.

        This uses boundary displacements only, never internal strain/stress.
        The patch definition is fixed in reference/material coordinates.
        """
        displacement = positions-self.points
        output = []
        for probe in PROBE_SITES:
            s = self.points[:, 0]/3
            cy = 0.4*np.sin(np.pi*s)
            mask = ((np.abs(self.points[:, 0]-probe.x) < 0.8) &
                    (self.points[:, 2] > 0.22*s**2+0.15) &
                    (np.abs(self.points[:, 1]-cy) < 0.75))
            weight = self.area*mask
            if weight.sum() <= 0:
                raise ValueError("mesh cannot resolve a surface observation patch")
            output.append(np.einsum("tnc,n->tc", displacement, weight)/weight.sum())
        return np.stack(output, axis=1)
