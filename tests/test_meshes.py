from collections import Counter
import numpy as np
import pytest
from softgriplab.meshes import tetrahedral_block


def test_tetrahedra_fill_volume_and_surface_is_closed_outward():
    points, tets, faces = tetrahedral_block(cells=(3, 4, 3))
    determinants = np.linalg.det(
        (points[tets[:, 1:]] - points[tets[:, 0, None]]).transpose(0, 2, 1)
    )
    assert np.all(determinants > 0)
    assert determinants.sum() / 6 == pytest.approx(0.05 * 0.05 * 0.04)
    vertices = points[faces]
    normals = np.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0])
    assert np.all(np.einsum("ij,ij->i", normals, vertices.mean(axis=1)) > 0)
    edges = Counter(
        tuple(sorted(edge))
        for face in faces
        for edge in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0]))
    )
    assert set(edges.values()) == {2}
