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

#include "picongpu/particles/particleToGrid/combinedAttributes/AverageAttribute.def"

#include <limits>
#include <string>
#include <vector>
namespace picongpu
{
    namespace particles
    {
        namespace particleToGrid
        {
            namespace combinedAttributes
            {
                struct ScreeningInvSqrtOperation
                {
                    template<typename T_Species>
                    struct apply
                    {
                        using type = ScreeningInvSqrtOperation;
                    };

                    /** Functor implementation
                     *
                     * Result overwrites the chargeDensity value.
                     *
                     * @tparam T_Acc alpaka accelerator type
                     * @param acc alpaka accelerator
                     * @param density charge density value and the result destination
                     * @param energyDensity energy density value
                     */
                    template<typename T_Acc>
                    HDINLINE void operator()(T_Acc const& acc, float1_X& chargeDensity, const float1_X& energyDensity)
                        const
                    {
                        chargeDensity = (1.0_X / EPS0) * chargeDensity * chargeDensity / energyDensity;
                    }
                };


                struct ScreeningInvSqrtDescription
                {
                    HDINLINE float1_64 getUnit() const
                    {
                        // inverse squared screening length has unit:
                        return 1 / (UNIT_LENGTH * UNIT_LENGTH);
                    }

                    HINLINE std::vector<float_64> getUnitDimension() const
                    {
                        /* L, M, T, I, theta, N, J
                         *
                         *  inverse squared meter: m^-2
                         *   -> L^-2
                         */
                        std::vector<float_64> unitDimension(7, 0.0);
                        unitDimension.at(SIBaseUnits::length) = -2.0;

                        return unitDimension;
                    }

                    HINLINE static std::string getName()
                    {
                        return "invSqrtScreenLength";
                    }
                };

            } // namespace combinedAttributes
        } // namespace particleToGrid
    } // namespace particles
} // namespace picongpu
