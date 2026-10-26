import numpy as np
import pytest

from softgriplab.cut_quadrature import clipped_tet_quadrature


def test_clipped_mass_and_first_moment_partition_original_tet():
    tet = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    upper_volume, upper_mass = clipped_tet_quadrature(tet, 0.5, 1)
    lower_volume, lower_mass = clipped_tet_quadrature(tet, 0.5, -1)
    assert upper_volume == pytest.approx(1 / 48)
    assert upper_volume + lower_volume == pytest.approx(1 / 6)
    np.testing.assert_allclose(upper_mass, np.array([0.125, 0.625, 0.125, 0.125]) / 48)
    np.testing.assert_allclose(upper_mass + lower_mass, np.full(4, 1 / 24))
    np.testing.assert_allclose(upper_mass @ tet / upper_volume, [0.625, 0.125, 0.125])


def test_disjoint_clip_has_no_material_and_full_clip_preserves_volume():
    tet = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]])
    volume, mass = clipped_tet_quadrature(tet, 3.0, 1)
    assert volume == 0
    assert np.array_equal(mass, np.zeros(4))
    volume, mass = clipped_tet_quadrature(tet, 3.0, -1)
    assert volume == pytest.approx(4.0)
    np.testing.assert_allclose(mass, np.ones(4))
