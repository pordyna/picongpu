"""
This file is part of PIConGPU.
Copyright 2024-2024 PIConGPU contributors
Authors: Brian Edward Marre
License: GPLv3+
"""

from ...pypicongpu import species
from ...pypicongpu import util

from .Distribution import Distribution

import typeguard
import typing


@typeguard.typechecked
class CylyndricalDistribution(Distribution):
    """
    Describes a density distribution of particles with gaussian up- and down-ramps with a constant density region in
    between

    Will create the following profile:
    - for y < center_front:                density * exp(factor * abs(y - center_front/sigma_front)**power)
    - for center_front <= y <=center_rear: density
    - for y > center_rear:                 density * exp(factor * abs(y - center_rear/sigma_rear)**power)

    with y being the position in the simulation box
    """

    density: float
    """particle number density, [m^-3]"""

    center_position: tuple[float, float, float]
    """center of the cylinder [x, y, z], [m]"""

    radius: float
    """cylinder radius, [m]"""

    jet_axis: tuple[float, float, float]
    """cylinder axis [x, y, z], [unitless]"""

    exponential_pre_plasma_length: float | None
    """scale length of the exponential pre-plasma ramp, [m]"""
    exponential_pre_plasma_cutoff: float | None
    """cutoff of the exponential pre-plasma ramp, [m]"""

    lower_bound: typing.Tuple[float, float, float] | typing.Tuple[None, None, None] = (None, None, None)
    upper_bound: typing.Tuple[float, float, float] | typing.Tuple[None, None, None] = (None, None, None)

    # @details pydantic provides an automatically generated __init__/constructor method which allows initialization off
    #   all attributes as keyword arguments

    # @note user may add additional attributes by hand, these will be available but not type verified

    def get_as_pypicongpu(self) -> species.operation.densityprofile.Cylinder:
        util.unsupported("fill in not active", self.fill_in, True)

        # @todo support bounds, Brian Marre, 2024
        util.unsupported("lower bound", self.lower_bound, (None, None, None))
        util.unsupported("upper bound", self.upper_bound, (None, None, None))

        profile = species.operation.densityprofile.Cylinder()

        if self.density <= 0.0:
            raise ValueError("density must be > 0")
        if self.radius <= 0.0:
            raise ValueError("radius must be > 0")

        # create prePlasma ramp if indicated by settings
        prePlasma: bool = (self.exponential_pre_plasma_cutoff is not None) and (
            self.exponential_pre_plasma_length is not None
        )
        explicitlyNoPrePlasma: bool = (self.exponential_pre_plasma_cutoff is None) and (
            self.exponential_pre_plasma_length is None
        )

        if prePlasma:
            profile.pre_plasma_ramp = species.operation.densityprofile.plasmaramp.Exponential(
                self.exponential_pre_plasma_length,  # type: ignore
                self.exponential_pre_plasma_cutoff,  # type: ignore
            )
        elif explicitlyNoPrePlasma:
            profile.pre_plasma_ramp = species.operation.densityprofile.plasmaramp.None_()
        else:
            raise ValueError(
                "either both exponential_pre_plasma_length and"
                " exponential_pre_plasma_cutoff must be set to"
                " none or neither!"
            )

        # @todo change to constructor call once we switched PyPIConGPU to use pydantic, Brian Marre, 2024
        profile.density_si = self.density
        profile.center_position_si = self.center_position
        profile.radius_si = self.radius
        profile.jet_axis = self.jet_axis

        return profile
