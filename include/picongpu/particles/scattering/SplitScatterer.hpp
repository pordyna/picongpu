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

#include "picongpu/particles/scattering/SplitAndScatterParticles.kernel"
#include "picongpu/particles/scattering/detail/AdvanceScatteringCounter.hpp"

#include <pmacc/particles/operations/Assign.hpp>
#include <pmacc/particles/operations/Deselect.hpp>

namespace picongpu
{
    namespace particles
    {
        namespace scattering
        {
            namespace acc
            {
                template<
                    typename T_AccConditionFunctor,
                    typename T_AccSplitConditionFunctor,
                    typename T_AccDirectionFunctor,
                    typename T_HandleDistanceDependence,
                    uint32_t t_newParticlesOnSplit>
                struct SplitScatterer
                {
                    static constexpr uint32_t newParticlesOnSplit = t_newParticlesOnSplit;

                    HDINLINE SplitScatterer(
                        T_AccConditionFunctor const& accConditionFunctor,
                        T_AccSplitConditionFunctor const& accSplitConditionFunctor,
                        T_AccDirectionFunctor const& accDirectionFunctor,
                        T_HandleDistanceDependence const& handleDistanceDependence)
                        : accConditionFunctor_m(accConditionFunctor)
                        , accSplitConditionFunctor_m(accSplitConditionFunctor)
                        , accDirectionFunctor_m(accDirectionFunctor)
                        , handleDistanceDependence_m(handleDistanceDependence)
                    {
                    }

                    template<typename T_Acc, typename T_Particle>
                    DINLINE bool scatteringCondition(T_Acc const& acc, T_Particle& particle, float_X const& density)
                    {
                        return accConditionFunctor_m(acc, particle, density);
                    }

                    template<typename T_Acc, typename T_Particle>
                    DINLINE bool splittingCondition(T_Acc const& acc, T_Particle& particle)
                    {
                        return accSplitConditionFunctor_m(acc, particle);
                    }

                    template<typename T_Acc, typename T_Particle>
                    DINLINE void scatterSingle(T_Acc const& acc, T_Particle& particle)
                    {
                        accDirectionFunctor_m(acc, particle);
                        if constexpr(pmacc::traits::HasFlag<typename T_Particle::FrameType, startPhase>::type::value)
                        {
                            const float_X phase = particle[startPhase_] + pmacc::math::Pi<float_X>::value;
                            constexpr float_X twoPi = pmacc::math::Pi<float_X>::doubleValue;
                            particle[startPhase_] = math::fmod(phase, twoPi);
                        }
                        scattering::detail::AdvanceScatteringCounter()(particle);
                        handleDistanceDependence_m(acc, particle);
                    }

                    template<typename T_Acc, typename T_Particle>
                    DINLINE void split(
                        T_Acc const& acc,
                        uint32_t id,
                        T_Particle& sourceParticle,
                        T_Particle& newParticle)

                    {
                        const float_X newWeighting = sourceParticle[weighting_] / (newParticlesOnSplit + 1u);
                        using namespace pmacc::particles::operations;
                        assign(newParticle, deselect<bmpl::vector<particleId, multiMask>>(sourceParticle));
                        scatterSingle(acc, newParticle);
                        newParticle[multiMask_] = 1;
                        newParticle[weighting_] = newWeighting;
                        if constexpr(pmacc::traits::HasIdentifier<typename T_Particle::FrameType, generation>::type::
                                         value)
                        {
                            newParticle[generation_] += 1;
                        }
                    }

                    template<typename T_Acc, typename T_Particle>
                    DINLINE void afterSplit(T_Acc const& acc, T_Particle& sourceParticle)

                    {
                        const float_X newWeighting = sourceParticle[weighting_] / (newParticlesOnSplit + 1u);
                        scatterSingle(acc, sourceParticle);
                        sourceParticle[weighting_] = newWeighting;
                    }

                private:
                    PMACC_ALIGN(accConditionFunctor_m, T_AccConditionFunctor);
                    PMACC_ALIGN(accDirectionFunctor_m, T_AccDirectionFunctor);
                    PMACC_ALIGN(accSplitConditionFunctor_m, T_AccSplitConditionFunctor);
                    PMACC_ALIGN(handleDistanceDependence_m, T_HandleDistanceDependence);
                };

            } // namespace acc

            template<
                typename T_DensityFields,
                typename T_ConditionFunctor,
                typename T_SplitConditionFunctor,
                typename T_DirectionFunctor,
                typename T_ReferenceDistance,
                uint32_t t_maxNewParticlesOnSplit>
            struct SplitScatterer
                : private T_ConditionFunctor
                , private T_DirectionFunctor
                , private T_SplitConditionFunctor
            {
                template<typename T_Species>
                struct apply
                {
                    using type = SplitScatterer<
                        T_DensityFields,
                        T_ConditionFunctor,
                        T_SplitConditionFunctor,
                        T_DirectionFunctor,
                        T_ReferenceDistance,
                        t_maxNewParticlesOnSplit>;
                };
                static constexpr bool needsFillGaps = true;
                static constexpr uint32_t maxNewParticlesOnSplit = t_maxNewParticlesOnSplit;

                using RequiredDerivedFields = T_DensityFields;
                using RequiredNativeFields = MakeSeq_t<>;
                template<uint32_t T_numWorkers>
                using CallingKernel = acc::SplitAndScatterParticlesKernel<T_numWorkers>;

                HINLINE SplitScatterer(uint32_t const& currentStep)
                    : T_ConditionFunctor(currentStep)
                    , T_DirectionFunctor(currentStep)
                    , T_SplitConditionFunctor(currentStep)
                    , handleDistanceDependence(currentStep)
                {
                }

                template<typename T_Acc, typename T_WorkerCfg>
                HDINLINE auto operator()(
                    T_Acc const& acc,
                    DataSpace<simDim> const& localSupercellOffset,
                    T_WorkerCfg const& workerCfg) const
                {
                    return acc::SplitScatterer<
                        typename T_ConditionFunctor::AccFunctorType,
                        typename T_SplitConditionFunctor::AccFunctorType,
                        typename T_DirectionFunctor::AccFunctorType,
                        HandleDistanceDependence<T_ReferenceDistance>,
                        t_maxNewParticlesOnSplit>(
                        T_ConditionFunctor::operator()(acc, localSupercellOffset, workerCfg),
                        T_SplitConditionFunctor::operator()(acc, localSupercellOffset, workerCfg),
                        T_DirectionFunctor::operator()(acc, localSupercellOffset, workerCfg),
                        handleDistanceDependence);
                }
            private:
                PMACC_ALIGN(handleDistanceDependence, HandleDistanceDependence<T_ReferenceDistance>);
            };

            template<
                typename T_DensityFields,
                typename T_ConditionFunctor,
                typename T_SplitConditionFunctor,
                typename T_DirectionFunctor,
                typename T_ReferenceDistance,
                uint32_t t_maxNewParticlesOnSplit>
            constexpr bool SplitScatterer<
                T_DensityFields,
                T_ConditionFunctor,
                T_SplitConditionFunctor,
                T_DirectionFunctor,
                T_ReferenceDistance,
                t_maxNewParticlesOnSplit>::needsFillGaps;

            template<
                typename T_DensityFields,
                typename T_ConditionFunctor,
                typename T_SplitConditionFunctor,
                typename T_DirectionFunctor,
                typename T_ReferenceDistance,
                uint32_t t_maxNewParticlesOnSplit>
            constexpr uint32_t SplitScatterer<
                T_DensityFields,
                T_ConditionFunctor,
                T_SplitConditionFunctor,
                T_DirectionFunctor,
                T_ReferenceDistance,
                t_maxNewParticlesOnSplit>::maxNewParticlesOnSplit;

        } // namespace scattering
    } // namespace particles
} // namespace picongpu
