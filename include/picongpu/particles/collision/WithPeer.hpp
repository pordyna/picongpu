/* Copyright 2014-2021 Rene Widera, Pawel Ordyna
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

#include <pmacc/particles/meta/FindByNameOrType.hpp>

#include "picongpu/particles/collision/IntraCollision.hpp"
#include "picongpu/particles/collision/InterCollision.hpp"

#include <boost/mpl/apply.hpp>

#include <cstdio>


namespace picongpu
{
    namespace particles
    {
        namespace collision
        {
            namespace detail
            {
                template<
                    typename T_CollisionFunctor,
                    typename T_Params,
                    typename T_FilterPair,
                    typename T_BaseSpecies,
                    typename T_PeerSpecies>
                struct WithPeer
                {
                    void operator()(const std::shared_ptr<DeviceHeap>& deviceHeap, uint32_t currentStep)
                    {
                        DoInterCollision<T_CollisionFunctor, T_Params, T_FilterPair, T_BaseSpecies, T_PeerSpecies>{}(
                            deviceHeap,
                            currentStep);
                    }
                };

                template<typename T_CollisionFunctor, typename T_Params, typename T_FilterPair, typename T_Species>
                struct WithPeer<T_CollisionFunctor, T_Params, T_FilterPair, T_Species, T_Species>
                {
                    void operator()(const std::shared_ptr<DeviceHeap>& deviceHeap, uint32_t currentStep)
                    {
                        DoIntraCollision<T_CollisionFunctor, T_Params, T_FilterPair, T_Species>{}(
                            deviceHeap,
                            currentStep);
                    }
                };
            } // namespace detail

            /* Runs the binary collision algorithm for a pair of colliding species.
             *
             * These struct chooses the InterCollision algorithm if the colliding
             * species are two different species and the IntraCollision algorithm if
             * they are identical.
             *
             * @tparam T_CollisionFunctor A binary particle functor defining a collision
             *    between two macro particles.
             * @tparam T_BaseSpecies First species in the collision pair.
             * @tparam T_PeerSpecies Second species in the collision pair.
             * @tparam T_Params A struct defining `coulombLog` for the collisions.
             * @tparam T_FilterPair A pair of particle filters, each for each species
             *     in the colliding pair.
             */
            template<
                typename T_CollisionFunctor,
                typename T_BaseSpecies,
                typename T_PeerSpecies,
                typename T_Params,
                typename T_FilterPair>
            struct WithPeer
            {
                void operator()(const std::shared_ptr<DeviceHeap>& deviceHeap, uint32_t currentStep)
                {
                    using BaseSpecies = pmacc::particles::meta::FindByNameOrType_t<VectorAllSpecies, T_BaseSpecies>;

                    using PeerSpecies = pmacc::particles::meta::FindByNameOrType_t<VectorAllSpecies, T_PeerSpecies>;

                    using CollisionFunctor = typename bmpl::apply2<T_CollisionFunctor, BaseSpecies, PeerSpecies>::type;

                    detail::WithPeer<CollisionFunctor, T_Params, T_FilterPair, BaseSpecies, PeerSpecies>{}(
                        deviceHeap,
                        currentStep);
                }
            };

        } // namespace collision
    } // namespace particles
} // namespace picongpu
