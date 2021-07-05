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

#include "picongpu/particles/scattering/detail/AdvanceScatteringCounter.hpp"
#include "picongpu/particles/scattering/scattering.kernel"
namespace picongpu
{
    namespace particles
    {
        namespace scattering
        {
            template<typename T_ReferenceDistance>
            struct HandleDistanceDependence
            {
                static constexpr float_X refDist = static_cast<float_X>(T_ReferenceDistance::value / UNIT_LENGTH);
                /* Handle distance dependence in scattering
                 *
                 * When scattering amplitudons, one needs to account for the fact that the amplitude falls with 1/r
                 * and not with 1/r. The amplitudon scattering cross section is proportional to the distance to the
                 * point towards which the particle is scattered. Since this is unknown at the scattering itself
                 * the scattering cross section is set with some reference distance l_0 instead. From the 2nd
                 * scattering on the particle weighting is corrected by r/l_0, where r is the distance to the last
                 * scattering point.
                 */
                HDINLINE HandleDistanceDependence(uint32_t const& currentStep) : currentStep_m(currentStep){};

                //! Operator implementation. Run only if the scattering is happening.
                template<typename T_Acc, typename T_Particle>
                DINLINE void operator()(T_Acc const& acc, T_Particle& particle)
                {
                    // The max value is reserved for when there was no event (scattering) yet.
                    if(particle[lastEventStep_] != std::numeric_limits<uint32_t>::max())
                    {
                        const float_X distance = (particle[lastEventStep_] - currentStep_m) * SPEED_OF_LIGHT;
                        particle[weighting_] *= distance / refDist;
                    }
                    // Update the last event.
                    particle[lastEventStep_] = currentStep_m;
                }

            private:
                PMACC_ALIGN(currentStep_m, uint32_t);
            };

            //! Disable distance handling when T_ReferenceDistance is void.
            template<>
            struct HandleDistanceDependence<void>
            {
                HDINLINE HandleDistanceDependence(uint32_t const& currentStep){};

                //! Empty operator implementation.
                template<typename T_Acc, typename T_Particle>
                DINLINE void operator()(T_Acc const& acc, T_Particle& particle)
                {
                }
            };

            namespace acc
            {
                //! Device side scattering functor
                template<
                    typename T_AccConditionFunctor,
                    typename T_AccDirectionFunctor,
                    typename T_HandleDistanceDependence>
                struct ScatterFunctor
                {
                    /* Constructor
                     *  @param accConditionFunctor device side condition functor
                     *  @param accDirectionFunctor device side direction functor
                     */
                    HDINLINE ScatterFunctor(
                        T_AccConditionFunctor const& accConditionFunctor,
                        T_AccDirectionFunctor const& accDirectionFunctor,
                        T_HandleDistanceDependence const& handleDistanceDependence)
                        : accConditionFunctor_m(accConditionFunctor)
                        , accDirectionFunctor_m(accDirectionFunctor)
                        , handleDistanceDependence_m(handleDistanceDependence)
                    {
                    }


                    /* Functor implementation
                     *
                     * @param acc alpaka accelerator
                     * @param particle particle to scatter
                     * @param density local scatterer density
                     */
                    template<typename T_Acc, typename T_Particle>
                    DINLINE void operator()(T_Acc const& acc, T_Particle& particle, float_X const& density)
                    {
                        const bool condition = accConditionFunctor_m(acc, particle, density);
                        if(condition)
                        {
                            accDirectionFunctor_m(acc, particle);
                            // 180 degree phase shift (thomson scattering)
                            if constexpr(pmacc::traits::HasFlag<typename T_Particle::FrameType, startPhase>::type::
                                             value)
                            {
                                const float_X phase = particle[startPhase_] + pmacc::math::Pi<float_X>::value;
                                constexpr float_X twoPi = pmacc::math::Pi<float_X>::doubleValue;
                                particle[startPhase_] = math::fmod(phase, twoPi);
                            }
                            scattering::detail::AdvanceScatteringCounter()(particle);
                            handleDistanceDependence_m(acc, particle);
                        }
                    }

                private:
                    PMACC_ALIGN(accConditionFunctor_m, T_AccConditionFunctor);
                    PMACC_ALIGN(accDirectionFunctor_m, T_AccDirectionFunctor);
                    PMACC_ALIGN(handleDistanceDependence_m, T_HandleDistanceDependence);
                };
            } // namespace acc

            template<
                typename T_DensityFields,
                typename T_ConditionFunctor,
                typename T_DirectionFunctor,
                typename T_ReferenceDistance>
            struct ScatterFunctor
                : private T_ConditionFunctor
                , private T_DirectionFunctor
            {
                template<typename T_Species>
                struct apply
                {
                    using type
                        = ScatterFunctor<T_DensityFields, T_ConditionFunctor, T_DirectionFunctor, T_ReferenceDistance>;
                };

                // A different scatterer functor may delete particles, if so one would need to call fillGaps
                // after the scatterer kernel is called. Here it is not needed.
                static constexpr bool needsFillGaps = false;

                template<uint32_t T_numWorkers>
                using CallingKernel = acc::ScatterParticlesKernel<T_numWorkers>;

                using RequiredDerivedFields = T_DensityFields;
                using RequiredNativeFields = MakeSeq_t<>;


                HINLINE ScatterFunctor(uint32_t const& currentStep)
                    : T_ConditionFunctor(currentStep)
                    , T_DirectionFunctor(currentStep)
                    , handleDistanceDependence(currentStep)
                {
                }

                template<typename T_Acc, typename T_WorkerCfg>
                HDINLINE auto operator()(
                    T_Acc const& acc,
                    DataSpace<simDim> const& localSupercellOffset,
                    T_WorkerCfg const& workerCfg) const
                {
                    return acc::ScatterFunctor<
                        typename T_ConditionFunctor::AccFunctorType,
                        typename T_DirectionFunctor::AccFunctorType,
                        HandleDistanceDependence<T_ReferenceDistance>>(
                        T_ConditionFunctor::operator()(acc, localSupercellOffset, workerCfg),
                        T_DirectionFunctor::operator()(acc, localSupercellOffset, workerCfg),
                        handleDistanceDependence);
                }

            private:
                PMACC_ALIGN(handleDistanceDependence, HandleDistanceDependence<T_ReferenceDistance>);
            };
            template<
                typename T_DensityFields,
                typename T_ConditionFunctor,
                typename T_DirectionFunctor,
                typename T_ReferenceDistance>
            constexpr bool
                ScatterFunctor<T_DensityFields, T_ConditionFunctor, T_DirectionFunctor, T_ReferenceDistance>::
                    needsFillGaps;
        } // namespace scattering
    } // namespace particles
} // namespace picongpu
