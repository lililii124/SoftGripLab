"""Isosurface reconstruction from saved material points (rendering only)."""
import numpy as np
from scipy.ndimage import gaussian_filter


def particle_surface(points, spacing):
    """Marching cubes on a smoothed, conservative trilinear particle deposit.

    The reconstruction has no knowledge of the knife trajectory or expected
    material shape. Coordinates are physical and deformation is never scaled.
    """
    from skimage.measure import marching_cubes

    cell = spacing / 3
    origin = points.min(axis=0) - 2 * spacing
    shape = np.ceil((points.max(axis=0) + 2 * spacing - origin) / cell).astype(int) + 1
    field = np.zeros(shape, dtype=np.float32)
    coords = (points - origin) / cell
    ids = np.floor(coords).astype(int)
    fraction = coords - ids
    for corner in np.ndindex(2, 2, 2):
        corner = np.array(corner)
        weight = np.prod(np.where(corner, fraction, 1 - fraction), axis=1)
        np.add.at(field, tuple((ids + corner).T), weight * (spacing / cell) ** 3)
    density = gaussian_filter(field, sigma=2.25, mode="constant")
    if density.max() <= 0.4:
        return np.empty((0, 3)), np.empty((0, 3), dtype=np.int32)
    vertices, faces, _, _ = marching_cubes(
        density, 0.4, spacing=(cell,) * 3, allow_degenerate=False
    )
    triangle = vertices[faces]
    if (
        np.einsum("ij,ij->", triangle[:, 0], np.cross(triangle[:, 1], triangle[:, 2]))
        < 0
    ):
        faces = faces[:, [0, 2, 1]]
    return vertices + origin, faces
