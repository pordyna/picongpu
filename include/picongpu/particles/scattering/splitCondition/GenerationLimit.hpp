/* Copyright 2021 Pawel Ordyna
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

#include "picongpu/particles/scattering/generic/Free.hpp"


namespace picongpu
{
    namespace particles
    {
        namespace scattering
        {
            namespace splitCondition
            {
                namespace acc
                {
                    template<uint32_t maxGenToSplit>
                    struct GenerationLimit
                    {
                        /* decide if a particle should be splitted
                         *
                         * @tparam T_Particle species type
                         *
                         * @param particle particle that could be splitted
                         */
                        template<typename T_Particle>
                        DINLINE bool operator()(T_Particle const& particle) const
                        {
                            return (particle[generation_] <= maxGenToSplit);
                        }
                    };
                } // namespace acc
            } // namespace splitCondition
        } // namespace scattering
    } // namespace particles
} // namespace picongpu
