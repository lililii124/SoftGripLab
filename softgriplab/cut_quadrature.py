"""Reference-volume quadrature for planar cuts through linear tetrahedra.

Virtual-node cutting duplicates an intersected tetrahedron. Each copy integrates
only its physical portion: clipped volume scales its elastic coefficients, and
integrals of the original linear shape functions supply its lumped nodal mass.
SciPy/Qhull supplies the convex-polyhedron triangulation.
"""

from itertools import combinations

import numpy as np
from scipy.spatial import ConvexHull


def clipped_tet_quadrature(vertices, plane_x, side):
    """Return physical volume and four shape-function integrals (m³)."""
    vertices = np.asarray(vertices, dtype=float)
    if vertices.shape != (4, 3) or side not in (-1, 1):
        raise ValueError("require four 3D vertices and side -1 or +1")
    basis = (vertices[1:] - vertices[0]).T
    full_volume = abs(np.linalg.det(basis)) / 6
    if full_volume <= 0:
        raise ValueError("degenerate reference tetrahedron")
    distances = side * (vertices[:, 0] - plane_x)
    if np.all(distances >= 0):
        return full_volume, np.full(4, full_volume / 4)
    if np.all(distances <= 0):
        return 0.0, np.zeros(4)
    polygon = list(vertices[distances >= 0])
    for a, b in combinations(range(4), 2):
        if distances[a] * distances[b] < 0:
            fraction = distances[a] / (distances[a] - distances[b])
            polygon.append(vertices[a] + fraction * (vertices[b] - vertices[a]))
    polygon = np.unique(np.array(polygon), axis=0)
    hull = ConvexHull(polygon)
    center = polygon.mean(axis=0)
    integrals = np.zeros(4)
    volume = 0.0
    for triangle in hull.simplices:
        sub_tet = np.vstack((center, polygon[triangle]))
        sub_volume = abs(np.linalg.det((sub_tet[1:] - sub_tet[0]).T)) / 6
        centroid = sub_tet.mean(axis=0)
        bary = np.linalg.solve(basis, centroid - vertices[0])
        shape = np.r_[1 - bary.sum(), bary]
        integrals += sub_volume * shape
        volume += sub_volume
    if np.min(integrals) < -1e-14:
        raise ValueError("clipped region lies outside reference element")
    return volume, integrals


def apply_cut_volume_weights(builder, original_particles, cut_planes_x, density):
    """Weight DiSECt virtual tetrahedra without changing its integration kernels.

    The planned planes must be separated far enough that no element crosses two.
    Call once after topology preparation and before model finalization.
    """
    positions = np.asarray(builder.particle_q)
    masses = np.zeros(len(positions))
    fractions = []
    for element, indices in enumerate(builder.tet_indices):
        indices = np.asarray(indices)
        tet = positions[indices]
        full_volume = abs(np.linalg.det((tet[1:] - tet[0]).T)) / 6
        planes = [x for x in cut_planes_x if tet[:, 0].min() < x < tet[:, 0].max()]
        if len(planes) > 1:
            raise ValueError("cut planes must not cross the same reference tetrahedron")
        if planes:
            plane = planes[0]
            physical = tet[indices < original_particles]
            if not len(physical):
                raise ValueError("virtual tetrahedron has no original material node")
            signs = np.sign(physical[:, 0] - plane)
            if not np.all(signs == signs[0]) or signs[0] == 0:
                raise ValueError(
                    "cut topology does not identify one physical half-space"
                )
            volume, integrals = clipped_tet_quadrature(tet, plane, int(signs[0]))
        else:
            volume, integrals = full_volume, np.full(4, full_volume / 4)
        fraction = volume / full_volume
        fractions.append(fraction)
        for attribute in ("tet_mu", "tet_lambda", "tet_damping"):
            getattr(builder, attribute)[element] *= fraction
        np.add.at(masses, indices, density * integrals)
    if np.any(masses <= 0):
        raise ValueError("a virtual node has zero physical mass")
    builder.particle_mass = masses.tolist()
    return np.asarray(fractions)
