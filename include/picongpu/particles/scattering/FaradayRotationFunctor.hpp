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

#include "picongpu/particles/traits/GetInterpolation.hpp"
#include "picongpu/particles/scattering/scattering.kernel"

namespace picongpu
{
    namespace particles
    {
        namespace scattering
        {
            namespace acc
            {
                struct FaradayRotationFunctor
                {
                    template<typename T_Acc, typename T_Particle>
                    DINLINE void operator()(
                        T_Acc const& acc,
                        T_Particle& particle,
                        float3_X const& bField,
                        float_X const& density)
                    {
                        // electron charge is negative (we need e so -ELECTRON_CHARGE)
                        constexpr float_X integrationConstant = -ELECTRON_CHARGE * ELECTRON_CHARGE * ELECTRON_CHARGE
                            / ELECTRON_MASS / ELECTRON_MASS / SPEED_OF_LIGHT / 2.0_X / EPS0;
                        constexpr float_X propDistance = DELTA_T * SPEED_OF_LIGHT;

                        const float3_X photonMomentum = particle[momentum_];
                        const float3_X dir = photonMomentum / math::abs(photonMomentum);

                        const float_X bFieldDir = pmacc::math::dot(bField, dir);
                        const float_X densityPICUnits = density * particles::TYPICAL_NUM_PARTICLES_PER_MACROPARTICLE;
                        const float_X omega = particles::GetAngFrequency<T_Particle>()(particle);
                        const float_X angleChange
                            = densityPICUnits * bFieldDir / (omega * omega) * integrationConstant * propDistance;
                        float_X newAngle = particle[polarizationAngle_] + angleChange;
                        // keep the angle in the (-PI, PI] range
                        constexpr float_X doublePi{pmacc::math::Pi<float_X>::doubleValue};
                        newAngle = math::fmod(
                                       newAngle + pmacc::math::Pi<float_X>::value,
                                       doublePi)
                            - pmacc::math::Pi<float_X>::value;
                        particle[polarizationAngle_] = newAngle;
                    }
                };
            } // namespace acc

            template<typename T_DensityFields, typename T_Species>
            struct FaradayRotationFunctorImpl
            {
                static constexpr bool needsFillGaps = false;

                using RequiredDerivedFields = T_DensityFields;
                using RequiredNativeFields = MakeSeq_t<FieldB>;


                template<uint32_t T_numWorkers>
                using CallingKernel = typename acc::ScatterParticlesWithBFieldKernel<T_numWorkers, T_Species>;

                HINLINE FaradayRotationFunctorImpl(uint32_t const& currentStep)
                {
                }

                template<typename T_Acc, typename T_WorkerCfg>
                HDINLINE auto operator()(
                    T_Acc const& acc,
                    DataSpace<simDim> const& localSupercellOffset,
                    T_WorkerCfg const& workerCfg) const
                {
                    return acc::FaradayRotationFunctor();
                }
            };
            template<typename T_DensityFields, typename T_Species>
            constexpr bool FaradayRotationFunctorImpl<T_DensityFields, T_Species>::needsFillGaps;

            template<typename T_DensityFields>
            struct FaradayRotationFunctor
            {
                template<typename T_Species>
                struct apply
                {
                    using type = FaradayRotationFunctorImpl<T_DensityFields, T_Species>;
                };
            };
        } // namespace scattering
    } // namespace particles
} // namespace picongpu
