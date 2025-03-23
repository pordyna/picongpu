"""
This file is part of PIConGPU.
Copyright 2021-2024 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre, Richard Pausch
License: GPLv3+
"""

from . import util
import typeguard
import typing
import enum
from .rendering import RenderedObject


@typeguard.typechecked
class BoundaryCondition(enum.Enum):
    """
    Boundary Condition of PIConGPU

    Defines how particles that pass the simulation bounding box are treated.

    TODO: implement the other methods supported by PIConGPU
    (reflecting, thermal)
    """

    PERIODIC = 1
    ABSORBING = 2

    def get_cfg_str(self) -> str:
        """
        Get string equivalent for cfg files
        :return: string for --periodic
        """
        literal_by_boundarycondition = {
            BoundaryCondition.PERIODIC: "1",
            BoundaryCondition.ABSORBING: "0",
        }
        return literal_by_boundarycondition[self]


@typeguard.typechecked
class Grid3D(RenderedObject):
    """
    PIConGPU 3 dimensional (cartesian) grid

    Defined by the dimensions of each cell and the number of cells per axis.

    The bounding box is implicitly given as TODO.
    """

    cell_size_x_si = util.build_typesafe_property(float)
    """Width of individual cell in X direction"""
    cell_size_y_si = util.build_typesafe_property(float)
    """Width of individual cell in Y direction"""
    cell_size_z_si = util.build_typesafe_property(float)
    """Width of individual cell in Z direction"""

    cell_cnt_x = util.build_typesafe_property(int)
    """total number of cells in X direction"""
    cell_cnt_y = util.build_typesafe_property(int)
    """total number of cells in Y direction"""
    cell_cnt_z = util.build_typesafe_property(int)
    """total number of cells in Z direction"""

    boundary_condition_x = util.build_typesafe_property(BoundaryCondition)
    """behavior towards particles crossing the X boundary"""
    boundary_condition_y = util.build_typesafe_property(BoundaryCondition)
    """behavior towards particles crossing the Y boundary"""
    boundary_condition_z = util.build_typesafe_property(BoundaryCondition)
    """behavior towards particles crossing the Z boundary"""

    n_gpus = util.build_typesafe_property(typing.Tuple[int, int, int])
    """number of GPUs in x y and z direction as 3-integer tuple"""

    grid_dist = util.build_typesafe_property(typing.Tuple[list[int], list[int], list[int]] | None)
    """distribution of grid cells to GPUs for each axis"""

    super_cell_size = util.build_typesafe_property(typing.Tuple[int, int, int])
    """size of super cell in x y and z direction as 3-integer tuple in cells"""

    def _get_serialized(self) -> dict:
        """serialized representation provided for RenderedObject"""
        assert self.cell_cnt_x > 0, "cell_cnt_x must be greater than 0"
        assert self.cell_cnt_y > 0, "cell_cnt_y must be greater than 0"
        assert self.cell_cnt_z > 0, "cell_cnt_z must be greater than 0"
        for i in range(3):
            assert self.n_gpus[i] > 0, "all n_gpus entries must be greater than 0"
        if self.grid_dist is not None:
            assert sum(self.grid_dist[0]) == self.cell_cnt_x, "sum of grid_dists in x must be equal to number_of_cells"
            assert sum(self.grid_dist[1]) == self.cell_cnt_y, "sum of grid_dists in y must be equal to number_of_cells"
            assert sum(self.grid_dist[2]) == self.cell_cnt_z, "sum of grid_dists in z must be equal to number_of_cells"

        result_dict = {
            "cell_size": {
                "x": self.cell_size_x_si,
                "y": self.cell_size_y_si,
                "z": self.cell_size_z_si,
            },
            "cell_cnt": {
                "x": self.cell_cnt_x,
                "y": self.cell_cnt_y,
                "z": self.cell_cnt_z,
            },
            "boundary_condition": {
                "x": self.boundary_condition_x.get_cfg_str(),
                "y": self.boundary_condition_y.get_cfg_str(),
                "z": self.boundary_condition_z.get_cfg_str(),
            },
            "gpu_cnt": {
                "x": self.n_gpus[0],
                "y": self.n_gpus[1],
                "z": self.n_gpus[2],
            },
            "super_cell_size": {
                "x": self.super_cell_size[0],
                "y": self.super_cell_size[1],
                "z": self.super_cell_size[2],
            },
        }
        if self.grid_dist is not None:
            result_dict |= {
                "grid_dist": {
                    "x": list(map(lambda x: {"device_cells": x}, self.grid_dist[0])),
                    "y": list(map(lambda x: {"device_cells": x}, self.grid_dist[1])),
                    "z": list(map(lambda x: {"device_cells": x}, self.grid_dist[2])),
                }
            }

        return result_dict
