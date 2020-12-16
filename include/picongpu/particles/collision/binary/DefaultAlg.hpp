/* Copyright 2013-2019 Rene Widera, Axel Huebl
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
#include "picongpu/particles/collision/binary/DefaultAlg.def"

#include <pmacc/random/distributions/Uniform.hpp>

#include <utility>
#include <type_traits>
#include <cmath>
#include <stdio.h>
#include <fenv.h>

namespace picongpu
{
    namespace particles
    {
        namespace collision
        {
            namespace binary
            {
                namespace acc
                {
                    using namespace pmacc;

                    /* Perform a single binary collision between two macro particles. (Device side functor)
                     *
                     * This algorithm was described in [Perez 2012] @url www.doi.org/10.1063/1.4742167.
                     * And it incorporates changes suggested in [Higginson 2020]
                     * @url www.doi.org/10.1016/j.jcp.2020.109450
                     */
                    struct DefaultAlg
                    {
                        float_X densitySqCbrt0;
                        float_X densitySqCbrt1;
                        uint32_t duplications;
                        uint32_t potentialPartners;
                        float_X coulombLog;

                        /* Initialize device side functor.
                         *
                         * @param p_densitySqCbrt0 @f[ n_0^{2/3} @f] where @f[ n_0 @f] is the 1st species density.
                         * @param p_densitySqCbrt1 @f[ n_1^{2/3} @f] where @f[ n_1 @f] is the 2nd species density.
                         * @param p_potentialPartners number of potential collision partners for a macro particle in
                         *   the cell.
                         * @param p_coulombLog coulomb logarithm
                         */
                        HDINLINE DefaultAlg(
                            float_X p_densitySqCbrt0,
                            float_X p_densitySqCbrt1,
                            uint32_t p_potentialPartners,
                            float_X p_coulombLog)
                            : densitySqCbrt0(p_densitySqCbrt0)
                            , densitySqCbrt1(p_densitySqCbrt1)
                            , duplications(1u)
                            , potentialPartners(p_potentialPartners)
                            , coulombLog(p_coulombLog){};

                        static constexpr float_X c = SPEED_OF_LIGHT;

                        /* Convert momentum from the lab frame into the COM frame.
                         *
                         * @param labMomentum momentum in the lab frame
                         * @param mass particle mass
                         * @param gamma particle Lorentz factor in the lab frame
                         * @param gammaComs @f[ \gamma_C @f] Lorentz factor of the COM frame in the lab frame
                         * @param factorA @f[ \frac{\gamma_C - 1}{v_C^2} @f]
                         * @param comsVelocity @f[ v_C @f] COM system velocity in the lab frame
                         */
                        static DINLINE float3_X labToComs(
                            float3_X labMomentum,
                            float_X mass,
                            float_X gamma,
                            float_X gammaComs,
                            float_X factorA,
                            float3_X comsVelocity)
                        {
                            float_X dot = pmacc::math::dot(comsVelocity, labMomentum);
                            float_X factorB = mass * gamma * gammaComs;
                            float3_X diff = (factorA * dot - factorB) * comsVelocity;
                            return labMomentum + diff;
                        }

                        /* Convert momentum from the COM frame into the lab frame.
                         *
                         * @param labMomentum momentum in the COM frame
                         * @param mass particle mass
                         * @param gamma particle Lorentz factor in the lab frame
                         * @param gammaComs @f[ \gamma_C @f] Lorentz factor of the COM frame in the lab frame
                         * @param factorA @f[ \frac{\gamma_C - 1}{v_C^2} @f]
                         * @param comsVelocity @f[ v_C @f] COM system velocity in the lab frame
                         */
                        static DINLINE float3_X comsToLab(
                            float3_X comsMomentum,
                            float_X mass,
                            float_X coeff,
                            float_X gammaComs,
                            float_X factorA,
                            float3_X comsVelocity)
                        {
                            // (13) in [Perez 2012]
                            float_X dot = pmacc::math::dot(comsVelocity, comsMomentum);
                            float_X factorB = coeff * gammaComs;
                            float3_X diff = (factorA * dot + factorB) * comsVelocity;
                            return comsMomentum + diff;
                        }

                        /* Calculate relative velocity in the COM system
                         *
                         * @param comsMementumMag0 1st particle momentum (in the COM system) magnitude
                         * @param mass0 1st particle mass
                         * @param mass1 2nd particle mass
                         * @param gamma0 1st particle Lorentz factor
                         * @param gamma1 2nd particle Lorentz factor
                         * @param gammaComs Lorentz factor of the COM frame in the lab frame
                         */
                        static DINLINE float_X calcRelativeComsVelocity(
                            float_X comsMomentumMag0,
                            float_X mass0,
                            float_X mass1,
                            float_X gamma0,
                            float_X gamma1,
                            float_X coeff0,
                            float_X coeff1,
                            float_X gammaComs)
                        {
                            float_X val = (mass0 * gamma0 + mass1 * gamma1) * comsMomentumMag0;
                            val = val / (coeff0 * coeff1 * gammaComs);
                            return val;
                        }

                        /* Calculate @f[ \gamma^* m @f]
                         *
                         * Returns particle mass times its Lorentz factor in the COM frame.
                         *
                         * @param labMomentum particle momentum in the labFrame
                         * @param mass particle mass
                         * @param gammaComs Lorentz factor of the COM frame in the lab frame
                         * @param comsVelocity COM system velocity in the lab frame
                         */
                        //
                        static DINLINE float_X coeff(
                            float3_X labMomentum,
                            float_X mass,
                            float_X gamma,
                            float_X gammaComs,
                            float3_X comsVelocity)
                        {
                            float_X dot = pmacc::math::dot(comsVelocity, labMomentum);
                            float_X val = mass * gamma - dot / (c * c);
                            return gammaComs * val;
                        }

                        /* Calculate the cosine of the scattering angle.
                         *
                         * The probabilty distribution  for the cosine depends on @f[ s_{12} @f]. The returned vale
                         * is determined by a float value between 0 and 1.
                         *
                         * @param s12 @f[ s_{12} @f] parameter. See [Perez 2012].
                         * @param u a usually random generated float between 0 and 1
                         */
                        static DINLINE float_X calcCosXi(float_X const s12, float_X const u)
                        {
                            // TODO: find some better way to restrict the val to [-1, +1].
                            if(s12 < 0.1_X)
                            {
                                float_X cosXi = 1.0_X + s12 * math::log(u);
                                if(cosXi < -1.0_X || cosXi > 1.0_X)
                                {
                                    if(cosXi < -1.0_X)
                                        return -1.0_X;
                                    else
                                        return 1.0_X;
                                }
                                else
                                    return cosXi;
                            }
                            else if(s12 < 6.0_X)
                            {
                                float_X a;
                                if(s12 < 3.0_X)
                                {
                                    float_X s12sq = s12 * s12;
                                    a = 0.0056958_X + 0.9560202_X * s12 - 0.508139_X * s12sq
                                        + 0.47913906_X * s12sq * s12 - 0.12788975_X * s12sq * s12sq
                                        + 0.02389567_X * s12sq * s12sq * s12sq;
                                    a = 1 / a;
                                }
                                else
                                {
                                    a = 3.0_X * math::exp(-1.0_X * s12);
                                }
                                float_X bracket = math::exp(-1.0_X * a) + 2.0_X * u * std::sinh(a);
                                float_X cosXi = math::log(bracket) / a;
                                if(cosXi < -1.0_X || cosXi > 1.0_X)
                                {
                                    if(cosXi < -1.0_X)
                                        return -1.0_X;
                                    else
                                        return 1.0_X;
                                }
                                else
                                    return cosXi;
                            }
                            else
                            {
                                // + 1 in Perez 2012 but cos in [-1,1]. Smilei uses -1.
                                return 2 * u - 1;
                            }
                        }

                        /* Calculate the momentum after the collision in the COM frame
                         *
                         * @param p momentum in the COM frame
                         * @param cosXi cosine of the scattering angle
                         * @param phi azimuthal scattering angle from [0, 2pi]
                         */
                        static DINLINE float3_X
                        calcFinalComsMomentum(float3_X const p, float_X const cosXi, float_X const phi)
                        {
                            float_X sinPhi, cosPhi;
                            pmacc::math::sincos(phi, sinPhi, cosPhi);
                            float_X sinXi = math::sqrt(1 - cosXi * cosXi);

                            // (12) in [Perez 2012]
                            float3_X finalVec;
                            // TODO: sqrt returns a pmacc complex number. cast to float?
                            float_X const pAbs = math::sqrt(pmacc::math::abs2(p));
                            float_X const pPerp = math::sqrt(p.x() * p.x() + p.y() * p.y());
                            // TODO chose a better limmit?
                            if(pPerp > std::max(std::numeric_limits<float_X>::epsilon, 1.0e-10_X) * pAbs)
                            {
                                finalVec[0] = (p.x() * p.z() * sinXi * cosPhi - p.y() * pAbs * sinXi * sinPhi) / pPerp
                                    + p.x() * cosXi;
                                finalVec[1] = (p.y() * p.z() * sinXi * cosPhi + p.x() * pAbs * sinXi * sinPhi) / pPerp
                                    + p.y() * cosXi;
                                finalVec[2] = -1.0_X * pPerp * sinXi * cosPhi + p.z() * cosXi;
                            }
                            else // limit px->0 py=0
                            {
                                finalVec[0] = pAbs * sinXi * cosPhi;
                                finalVec[1] = pAbs * sinXi * sinPhi;
                                finalVec[2] = pAbs * cosXi;
                            }
                            return finalVec;
                        }

                        /** Execute the collision functor
                         *
                         * @param ctx collision context
                         * @param par0 1st colliding macro particle
                         * @param par1 2nd colliding macro particle
                         */
                        template<typename T_Context, typename T_Par0, typename T_Par1>
                        DINLINE void operator()(T_Context const& ctx, T_Par0& par0, T_Par1& par1) const
                        {
                            // feenableexcept(FE_INVALID | FE_OVERFLOW | FE_DIVBYZERO);
                            float_X const weight0
                                = par0[weighting_] / particles::TYPICAL_NUM_PARTICLES_PER_MACROPARTICLE;
                            float_X const weight1
                                = par1[weighting_] / particles::TYPICAL_NUM_PARTICLES_PER_MACROPARTICLE;

                            float3_X const labMomentum0 = par0[momentum_] / weight0;
                            float3_X const labMomentum1 = par1[momentum_] / weight1;

                            float_X const mass0 = picongpu::traits::attribute::getMass(
                                particles::TYPICAL_NUM_PARTICLES_PER_MACROPARTICLE,
                                par0);
                            float_X const mass1 = picongpu::traits::attribute::getMass(
                                particles::TYPICAL_NUM_PARTICLES_PER_MACROPARTICLE,
                                par1);
                            float_X const charge0 = picongpu::traits::attribute::getCharge(
                                particles::TYPICAL_NUM_PARTICLES_PER_MACROPARTICLE,
                                par0);
                            float_X const charge1 = picongpu::traits::attribute::getCharge(
                                particles::TYPICAL_NUM_PARTICLES_PER_MACROPARTICLE,
                                par1);

                            float_X const gamma0 = picongpu::gamma<float_X>(labMomentum0, mass0);
                            float_X const gamma1 = picongpu::gamma<float_X>(labMomentum1, mass1);

                            // [Perez 2012] (1)
                            float3_X const comsVelocity
                                = (labMomentum0 + labMomentum1) / (mass0 * gamma0 + mass1 * gamma1);
                            float_X const comsVelocityAbs2 = pmacc::math::abs2(comsVelocity);
                            float3_X comsMomentum0;
                            float_X gammaComs, factorA, coeff0, coeff1;

                            if(comsVelocityAbs2 != 0.0_X)
                            {
                                gammaComs = 1.0_X / math::sqrt(1 - comsVelocityAbs2 / (c * c));
                                // used later for comsToLab:
                                factorA = (gammaComs - 1.0_X) / comsVelocityAbs2;

                                // Stared gamma times mass, from [Perez 2012].
                                coeff0 = coeff(labMomentum0, mass0, gamma0, gammaComs, comsVelocity);
                                // gamma^* . mass
                                coeff1 = coeff(labMomentum1, mass1, gamma1, gammaComs, comsVelocity);

                                // (2) in [Perez 2012]
                                comsMomentum0
                                    = labToComs(labMomentum0, mass0, gamma0, gammaComs, factorA, comsVelocity);
                            }
                            else
                            {
                                // From smilei implementation, for v_coms^2 = 0
                                gammaComs = 1.0_X;
                                // used later for comsToLab:
                                factorA = 0.5_X;
                                // Stared gamma times mass, from [Perez 2012].
                                coeff0 = mass0 * gamma0;
                                // gamma^* . mass
                                coeff1 = mass1 * gamma1;
                                comsMomentum0 = labMomentum0;
                            }


                            //  f0 * f1 * f2^2
                            //  is equal  s12 * (n12/(n1*n2)) from [Perez 2012]
                            float_X comsMomentum0Abs2 = pmacc::math::abs2(comsMomentum0);

                            if(comsMomentum0Abs2 == 0.0_X && (comsMomentum0 != float3_X{0.0_X, 0.0_X, 0.0_X}))
                                return;

                            float_X s12Factor0 = (DELTA_T * coulombLog * charge0 * charge0 * charge1 * charge1)
                                / (4.0_X * PI * EPS0 * EPS0 * c * c * c * c * mass0 * gamma0 * mass1 * gamma1);
                            s12Factor0 *= 1.0_X / particles::TYPICAL_NUM_PARTICLES_PER_MACROPARTICLE
                                / particles::TYPICAL_NUM_PARTICLES_PER_MACROPARTICLE;
                            float_X const s12Factor1
                                = gammaComs * math::sqrt(comsMomentum0Abs2) / (mass0 * gamma0 + mass1 * gamma1);
                            float_X const s12Factor2 = coeff0 * coeff1 * c * c / comsMomentum0Abs2 + 1.0_X;
                            // Statistical part from [Higginson 2020],
                            // corresponds to n1*n2/n12 in [Perez 2012]:
                            float_X const s12Factor3 = potentialPartners * pmacc::math::max(weight0, weight1)
                                * particles::TYPICAL_NUM_PARTICLES_PER_MACROPARTICLE / duplications / CELL_VOLUME;
                            float_X s12 = s12Factor0 * s12Factor1 * s12Factor2 * s12Factor2 * s12Factor3;

                            // Low Temeprature correction:
                            // [Perez 2012] (8)
                            float_X relativeComsVelocity = calcRelativeComsVelocity(
                                math::sqrt(comsMomentum0Abs2),
                                mass0,
                                mass1,
                                gamma0,
                                gamma1,
                                coeff0,
                                coeff1,
                                gammaComs);
                            // [Perez 2012] (21) ( without n1*n2/n12 )
                            float_X s12Max = math::pow(4 * PI / 3, 1. / 3.) * DELTA_T * (mass0 + mass1)
                                / pmacc::math::max(mass0 * densitySqCbrt0, mass1 * densitySqCbrt1)
                                * relativeComsVelocity;
                            s12Max *= s12Factor3;
                            pmacc::math::min(s12, s12Max);

                            // Get a random float value from 0,1
                            auto const& acc = *ctx.m_acc;
                            auto& rngHandle = *ctx.m_hRng;
                            using UniformFloat = pmacc::random::distributions::Uniform<
                                pmacc::random::distributions::uniform::ExcludeZero<float_X>>;
                            auto rng = rngHandle.template applyDistribution<UniformFloat>();
                            float_X rngValue = rng(acc);

                            float_X const cosXi = calcCosXi(s12, rngValue);
                            // PMACC_ASSERT( cosXi <= 1 && cosXi >= -1 );
                            float_X const phi = 2.0_X * PI * rng(acc);
                            float3_X const finalComs0 = calcFinalComsMomentum(comsMomentum0, cosXi, phi);

                            float3_X finalLab0, finalLab1;
                            if(weight0 > weight1)
                            {
                                finalLab1
                                    = comsToLab(-1.0_X * finalComs0, mass1, coeff1, gammaComs, factorA, comsVelocity);
                                par1[momentum_] = finalLab1 * weight1;
                                if((weight1 / weight0) - rng(acc) > 0)
                                {
                                    finalLab0 = comsToLab(finalComs0, mass0, coeff0, gammaComs, factorA, comsVelocity);
                                    par0[momentum_] = finalLab0 * weight0;
                                }
                            }
                            else
                            {
                                finalLab0 = comsToLab(finalComs0, mass0, coeff0, gammaComs, factorA, comsVelocity);
                                par0[momentum_] = finalLab0 * weight0;
                                if((weight0 / weight1) - rng(acc) >= 0)
                                {
                                    finalLab1 = comsToLab(
                                        -1.0_X * finalComs0,
                                        mass1,
                                        coeff1,
                                        gammaComs,
                                        factorA,
                                        comsVelocity);
                                    par1[momentum_] = finalLab1 * weight1;
                                }
                            }
                        }
                    };
                } // namespace acc

                //! Host side binary collision functor
                struct DefaultAlg
                {
                    template<typename T_Species0, typename T_Species1>
                    struct apply
                    {
                        using type = DefaultAlg;
                    };

                    HINLINE DefaultAlg(uint32_t currentStep){};

                    /** create device manipulator functor
                     *
                     * @param acc alpaka accelerator
                     * @param offset (in supercells, without any guards) to the origin of the local domain
                     * @param workerCfg configuration of the worker
                     * @param density0 cell density of the 1st species
                     * @param density1 cell density of the 2nd species
                     * @param potentialPartners number of potential collision partners for a macro particle in
                     *   the cell.
                     * @param coulombLog Coulomb logarithm
                     */
                    template<typename T_WorkerCfg, typename T_Acc>
                    HDINLINE acc::DefaultAlg operator()(
                        T_Acc const& acc,
                        DataSpace<simDim> const& offset,
                        T_WorkerCfg const& workerCfg,
                        float_X const& density0,
                        float_X const& density1,
                        uint32_t const& potentialPartners,
                        float_X const& coulombLog) const
                    {
                        return acc::DefaultAlg(
                            math::pow(density0, 2. / 3.),
                            math::pow(density1, 2. / 3.),
                            potentialPartners,
                            coulombLog);
                    }

                    //! get the name of the functor
                    static HINLINE std::string getName()
                    {
                        return "DefaultAlg";
                    }
                };
            } // namespace binary
        } // namespace collision
    } // namespace particles
} // namespace picongpu
