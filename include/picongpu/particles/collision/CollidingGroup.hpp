/* Copyright 2014-2020 Pawel Ordyna
 *
 * This file is part of PIConGPU.
 *
 * PIConGPU is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PIConGPU is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "picongpu/simulation_defines.hpp"
namespace picongpu
{
    namespace particles
    {
        namespace collision
        {
            template<typename T_ParBox, typename T_ParCellsList, typename T_DensArray>
            struct CollidingGroup
            {
                // TODO: PMACC align??
                using FramePtrType = typename T_ParBox::FramePtr;
                FramePtrType firstFrame;
                T_ParBox& pb;
                T_ParCellsList& parCellsList;
                T_DensArray& densArray;
                uint32_t const numParticlesInSuperCell;


                HDINLINE CollidingGroup(
                    T_ParBox& pb_p,
                    T_ParCellsList& cellsList_p,
                    T_DensArray& densArray_p,
                    DataSpace<simDim> const superCellIdx)
                    : // firstFrame(firstFrame_p)
                    pb(pb_p)
                    , parCellsList(cellsList_p)
                    , densArray(densArray_p)
                    , firstFrame(pb_p.getFirstFrame(superCellIdx))
                    , numParticlesInSuperCell(pb_p.getSuperCell(superCellIdx).getNumParticles())
                {
                }
            };

            template<typename T_CollidingGroupShort, typename T_CollidingGroupLong>
            struct CollidingGroupPair
            {
                T_CollidingGroupShort& collidingGroupShort;
                T_CollidingGroupLong& collidingGroupLong;

                HDINLINE CollidingGroupPair(
                    T_CollidingGroupShort& collidingGroupShort_p,
                    T_CollidingGroupLong& collidingGroupLong_p)
                    : collidingGroupShort(collidingGroupShort_p)
                    , collidingGroupLong(collidingGroupLong_p)
                {
                }
            };
        } // namespace collision
    } // namespace particles
} // namespace picongpu
