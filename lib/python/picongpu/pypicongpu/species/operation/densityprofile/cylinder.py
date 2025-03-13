"""
This file is part of PIConGPU.
Copyright 2023-2024 PIConGPU contributors
Authors: Kristin Tippey, Brian Edward Marre
License: GPLv3+
"""

from .densityprofile import DensityProfile
from .plasmaramp import PlasmaRamp
from .... import util

import typeguard


@typeguard.typechecked
class Cylinder(DensityProfile):
    """
    Directional density profile with thickness and pre- and
    post-plasma lengths and cutoffs
    """

    density_si = util.build_typesafe_property(float)
    """particle number density at at the foil plateau (m^-3)"""

    center_position_si = util.build_typesafe_property(tuple[float, float, float])
    """center of the cylinder [x, y, z], [m]"""

    radius_si = util.build_typesafe_property(float)
    """cylinder radius, [m]"""

    jet_axis = util.build_typesafe_property(tuple[float, float, float])
    """cylinder axis [x, y, z], [unitless]"""

    pre_plasma_ramp = util.build_typesafe_property(PlasmaRamp)
    """pre plasma ramp"""

    def __init__(self):
        # (nothing to do, overwrite from abstract parent)
        pass

    def check(self) -> None:
        if self.density_si <= 0.0:
            raise ValueError("density must be > 0")
        if self.radius_si <= 0.0:
            raise ValueError("radius must be > 0")
        self.pre_plasma_ramp.check()

    def _get_serialized(self) -> dict:
        self.check()

        return {
            "density_si": self.density_si,
            "center_position_si": [{"component": x} for x in self.center_position_si],
            "radius_si": self.radius_si,
            "jet_axis": [{"component": x} for x in self.jet_axis],
            "pre_plasma_ramp": self.pre_plasma_ramp.get_generic_profile_rendering_context(),
        }
