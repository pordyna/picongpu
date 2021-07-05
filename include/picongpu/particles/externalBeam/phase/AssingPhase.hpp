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

#include <pmacc/traits/HasIdentifier.hpp>

namespace picongpu
{
    namespace particles
    {
        namespace externalBeam
        {
            namespace initPhase
            {
                /* Assign particle phase as the startPhase attribute
                 *
                 * @tparam T_Species species type
                 * @tparam usePhaseInsteadOfStartPhase switch between phase and startPhase attribute.
                 *      Default: true when species has the phase attribute.
                 */
                template<typename T_Species>
                struct AssignPhase
                {
                    HDINLINE static void assign(T_Species& particle, float_X const& phase)
                    {
                        particle[startPhase_] = phase;
                    }
                };
            } // namespace initPhase
        } // namespace externalBeam
    } // namespace particles
} // namespace picongpu
