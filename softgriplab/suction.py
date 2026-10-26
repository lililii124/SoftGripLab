"""Conservative reduced suction capacity for the rigid-body robot demonstration.

This is a seal-gated breakable joint, not a fluid or deformable-cup solver.
"""

from dataclasses import dataclass
import math

import numpy as np

from .materials import positive


@dataclass(frozen=True)
class SuctionSpec:
    pressure_pa: float = 30000.0
    radius_m: float = 0.012
    efficiency: float = 0.6
    seal_tolerance_m: float = 0.0015
    tilt_limit_deg: float = 8.0

    def __post_init__(self):
        positive(radius_m=self.radius_m, efficiency=self.efficiency,
                 seal_tolerance_m=self.seal_tolerance_m, tilt_limit_deg=self.tilt_limit_deg)
        if not np.isfinite(self.pressure_pa) or self.pressure_pa < 0:
            raise ValueError("pressure must be finite and nonnegative")
        if self.efficiency > 1 or self.tilt_limit_deg >= 90:
            raise ValueError("efficiency must be <=1 and tilt limit must be <90 degrees")

    @property
    def force_capacity_n(self):
        return self.efficiency * self.pressure_pa * math.pi * self.radius_m ** 2

    @property
    def moment_capacity_nm(self):
        # Conservative lever-arm limit, a declared design assumption.
        return self.force_capacity_n * self.radius_m / 2

    def permits_seal(self, local_tip_m, normal_alignment, half_width_m, *, touching):
        """Local tip: x/y from top-face center, z gap from that face."""
        tip = np.asarray(local_tip_m, float)
        if tip.shape != (3,) or not np.all(np.isfinite(tip)):
            raise ValueError("local cup tip must contain three finite coordinates")
        positive(half_width_m=half_width_m)
        if not np.isfinite(normal_alignment) or not -1 <= normal_alignment <= 1:
            raise ValueError("normal alignment must lie in [-1,1]")
        return bool(touching and self.force_capacity_n > 0
                    and abs(tip[2]) <= self.seal_tolerance_m
                    and np.max(np.abs(tip[:2])) + self.radius_m <= half_width_m
                    and normal_alignment >= math.cos(math.radians(self.tilt_limit_deg)))

    def overloaded(self, reaction):
        reaction = np.asarray(reaction, float)
        if reaction.shape != (6,) or not np.all(np.isfinite(reaction)):
            raise ValueError("constraint reaction must contain three forces and three moments")
        return bool(np.linalg.norm(reaction[:3]) > self.force_capacity_n
                    or np.linalg.norm(reaction[3:]) > self.moment_capacity_nm)

