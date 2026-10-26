import math

import numpy as np
import pytest

from softgriplab.suction import SuctionSpec


def test_pressure_area_capacity_and_scaling():
    spec = SuctionSpec(pressure_pa=30000., radius_m=.012, efficiency=.6)
    assert spec.force_capacity_n == pytest.approx(.6 * 30000 * math.pi * .012 ** 2)
    twice_radius = SuctionSpec(pressure_pa=30000., radius_m=.024, efficiency=.6)
    assert twice_radius.force_capacity_n == pytest.approx(4 * spec.force_capacity_n)
    assert not SuctionSpec(pressure_pa=0).permits_seal([0, 0, 0], 1, .025, touching=True)


@pytest.mark.parametrize("position,alignment,touching", [
    ([0, 0, 0], 1, False), ([0, 0, .003], 1, True),
    ([.02, 0, 0], 1, True), ([0, 0, 0], .9, True),
])
def test_seal_requires_contact_footprint_and_alignment(position, alignment, touching):
    assert not SuctionSpec().permits_seal(position, alignment, .025, touching=touching)


def test_seal_can_hold_static_payload_but_opens_under_large_force_or_moment():
    spec = SuctionSpec()
    assert spec.permits_seal([0, 0, 0], 1, .025, touching=True)
    assert not spec.overloaded([0, 0, .05 * 9.81, 0, 0, 0])
    assert spec.overloaded([0, 0, 2 * spec.force_capacity_n, 0, 0, 0])
    assert spec.overloaded([0, 0, 0, 2 * spec.moment_capacity_nm, 0, 0])
    with pytest.raises(ValueError):
        spec.overloaded(np.full(6, np.nan))
