/* Copyright 2019-2022 Rene Widera
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


#include "picongpu/fields/FieldJ.hpp"
#include "picongpu/fields/FieldTmp.hpp"

#include "picongpu/particles/collision/collision.hpp"
#include "picongpu/particles/particleToGrid/ComputeFieldValue.hpp"
#include "picongpu/particles/particleToGrid/combinedAttributes/CombinedAttributes.def"

#include <pmacc/Environment.hpp>
#include <pmacc/dataManagement/DataConnector.hpp>
#include <pmacc/meta/ForEach.hpp>
#include <pmacc/particles/traits/FilterByFlag.hpp>
#include <pmacc/type/Area.hpp>

#include <cstdint>


namespace picongpu
{
    namespace simulation
    {
        namespace stage
        {
            //! Functor for the stage of the PIC loop performing particle collision
            class Collision
            {
            public:
                Collision(std::shared_ptr<DeviceHeap>& heap) : m_heap(heap)
                {
                }

                template<typename T_Species>
                struct AddNextField
                {
                    HINLINE void operator()(
                        FieldTmp& fieldTmp1,
                        FieldTmp& fieldTmp2,
                        uint32_t const& currentStep,
                        uint32_t const& extraSlotNr) const
                    {
                        DataConnector& dc = Environment<>::get().DataConnector();
                        using DeriveOperation = particles::particleToGrid::CreateFieldTmpOperation_t<
                            T_Species,
                            particles::particleToGrid::combinedAttributes::ScreeningInvSqrt>;
                        using Solver = typename DeriveOperation::Solver;
                        using Filter = typename DeriveOperation::Filter;
                        auto eventPtr
                            = particles::particleToGrid::ComputeFieldValue<CORE + BORDER, Solver, T_Species, Filter>()(
                                fieldTmp2,
                                currentStep,
                                extraSlotNr);
                        // wait for unfinished asynchronous communication
                        if(eventPtr != nullptr)
                            __setTransactionEvent(*eventPtr);
                        fieldTmp1.template modifyByField<CORE + BORDER, pmacc::math::operation::Add>(fieldTmp2);
                    }
                };

                /** Perform particle particle collision
                 *
                 * @param step index of time iteration
                 */
                void operator()(uint32_t const step) const
                {
                    if constexpr(particles::collision::calculateScreeningLength)
                    {
                        using Species = picongpu::particles::collision::CollisionScreeningSpecies;
                        using FirstSpecies = typename bmpl::at_c<Species, 0>::type;
                        using RemainingSpecies = typename bmpl::pop_front<Species>::type;

                        using DeriveOperation = particles::particleToGrid::CreateFieldTmpOperation_t<
                            FirstSpecies,
                            particles::particleToGrid::combinedAttributes::ScreeningInvSqrt>;
                        using Solver = typename DeriveOperation::Solver;
                        using Filter = typename DeriveOperation::Filter;
                        DataConnector& dc = Environment<>::get().DataConnector();
                        constexpr uint32_t slot = picongpu::particles::collision::screeningLengthSlot;
                        auto fieldTmp1 = dc.get<FieldTmp>(FieldTmp::getUniqueId(slot), true);
                        auto eventPtr = particles::particleToGrid::ComputeFieldValue<
                            CORE + BORDER,
                            Solver,
                            FirstSpecies,
                            Filter>()(*fieldTmp1, step, slot + 1u);
                        // wait for unfinished asynchronous communication
                        if(eventPtr != nullptr)
                            __setTransactionEvent(*eventPtr);

                        if constexpr(!bmpl::empty<RemainingSpecies>::value)
                        {
                            auto fieldTmp2 = dc.get<FieldTmp>(FieldTmp::getUniqueId(slot + 1), true);
                            pmacc::meta::ForEach<RemainingSpecies, AddNextField<bmpl::_1>>{}(
                                *fieldTmp1,
                                *fieldTmp2,
                                step,
                                slot + 2u);
                            fieldTmp2.reset();
                        }
                    }

                    pmacc::meta::ForEach<
                        particles::collision::CollisionPipeline,
                        particles::collision::CallCollider<bmpl::_1>>{}(m_heap, step);
                }

            private:
                std::shared_ptr<DeviceHeap> m_heap;
            };

        } // namespace stage
    } // namespace simulation
} // namespace picongpu
