/* Copyright 2015-2021 Alexander Grund
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "picongpu/particles/boundary/Apply.hpp"
#include "picongpu/particles/boundary/Utility.hpp"

#include <pmacc/Environment.hpp>
#include <pmacc/boundary/Utility.hpp>
#include <pmacc/types.hpp>

namespace picongpu
{
    namespace particles
    {
        namespace boundary
        {
            //! Tells if an exchange is an external boundary. Periodicity is ignored!
            struct IsExternalBoundary
            {
                HINLINE IsExternalBoundary()
                {
                    auto const position = Environment<simDim>::get().GridController().getPosition();
                    auto const grid = Environment<simDim>::get().GridController().getGpuNodes();

                    using namespace pmacc::boundary;
                    for(const auto exchange : particles::boundary::getAllAxisAlignedExchanges())
                    {
                        auto const axis = getAxis(exchange);
                        auto isExternal = false;
                        if(isMinSide(exchange) && position[axis] == 0)
                            isExternal = true;
                        if(isMaxSide(exchange) && position[axis] + 1 == grid[axis])
                            isExternal = true;
                        if(isExternal)
                            axisAlignedExternalExchanges.push_back(exchange);
                    }
                }
                HINLINE bool operator()(uint32_t exchange) const
                {
                    auto const exchangeMask = pmacc::Mask(exchange);
                    for(const auto& axisAlignedExternalExchange : axisAlignedExternalExchanges)
                    {
                        if(exchangeMask.containsExchangeType(axisAlignedExternalExchange))
                            return true;
                    }
                    return false;
                }

            private:
                std::vector<uint32_t> axisAlignedExternalExchanges;
            };

            /**
             * Policy for @see HandleGuardRegion that moves particles from guard cells to exchange buffers
             * and sends those to the correct neighbors
             */
            struct IgnorePeriodicExchangeOrAbsorb
            {
                IsExternalBoundary isExternalBoundary;

                template<class T_Particles>
                void handleOutgoing(T_Particles& par, int32_t direction) const
                {
                    if(isExternalBoundary(direction))
                    {
                        /* Here is the only place in the computational loop where we can call hooks from plugins:
                         *    - we know those particles crossed the active boundary
                         *    - but we didn't yet apply the boundary conditions
                         *      that would normally modify or delete the particles
                         *      (this is done just afterwards in this function)
                         */
                        if(pmacc::boundary::isAxisAligned(direction))
                        {
                            detail::callPluginHooks(par, direction);
                            // Axis aligned boundaries come always first so there are no particles to delete
                            // in other exchanges anyway.
                            par.deleteGuardParticles(direction);
                        }
                    }
                    else
                    {
                        pmacc::Environment<>::get().ParticleFactory().createTaskSendParticlesExchange(par, direction);
                    }
                }

                template<class T_Particles>
                void handleIncoming(T_Particles& par, int32_t direction) const
                {
                    if(!isExternalBoundary(direction))
                    {
                        pmacc::Environment<>::get().ParticleFactory().createTaskReceiveParticlesExchange(
                            par,
                            direction);
                    }
                }
            };

        } // namespace boundary
    } // namespace particles
} // namespace picongpu
