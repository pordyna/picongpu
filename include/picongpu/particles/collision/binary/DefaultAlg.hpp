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

                    using float_COLL = float_64;
                    using float3_COLL = float3_64;
                    const float_COLL DELTA_T_COLL = precisionCast<float_COLL>(DELTA_T);
                    const float_COLL EPS0_COLL = precisionCast<float_COLL>(EPS0);
                    const float_COLL WEIGHT_NORM_COLL
                        = precisionCast<float_COLL>(particles::TYPICAL_NUM_PARTICLES_PER_MACROPARTICLE);
                    const float_COLL CELL_VOLUME_COLL = precisionCast<float_COLL>(CELL_VOLUME);

                    // literals for short-hand notations for collisions
                    constexpr float_COLL operator""_COLL(long double x)
                    {
                        return float_COLL(x);
                    }

                    /* Perform a single binary collision between two macro particles. (Device side functor)
                     *
                     * This algorithm was described in [Perez 2012] @url www.doi.org/10.1063/1.4742167.
                     * And it incorporates changes suggested in [Higginson 2020]
                     * @url www.doi.org/10.1016/j.jcp.2020.109450
                     */
                    struct DefaultAlg
                    {
                        float_COLL densitySqCbrt0;
                        float_COLL densitySqCbrt1;
                        uint32_t duplications;
                        uint32_t potentialPartners;
                        float_COLL coulombLog;

                        /* Initialize device side functor.
                         *
                         * @param p_densitySqCbrt0 @f[ n_0^{2/3} @f] where @f[ n_0 @f] is the 1st species density.
                         * @param p_densitySqCbrt1 @f[ n_1^{2/3} @f] where @f[ n_1 @f] is the 2nd species density.
                         * @param p_potentialPartners number of potential collision partners for a macro particle in
                         *   the cell.
                         * @param p_coulombLog coulomb logarithm
                         */
                        HDINLINE DefaultAlg(
                            float_COLL p_densitySqCbrt0,
                            float_COLL p_densitySqCbrt1,
                            uint32_t p_potentialPartners,
                            float_COLL p_coulombLog)
                            : densitySqCbrt0(p_densitySqCbrt0)
                            , densitySqCbrt1(p_densitySqCbrt1)
                            , duplications(1u)
                            , potentialPartners(p_potentialPartners)
                            , coulombLog(p_coulombLog){};

                        static constexpr float_COLL c = static_cast<float_COLL>(SPEED_OF_LIGHT);


                        /* Convert momentum from the lab frame into the COM frame.
                         *
                         * @param labMomentum momentum in the lab frame
                         * @param mass particle mass
                         * @param gamma particle Lorentz factor in the lab frame
                         * @param gammaComs @f[ \gamma_C @f] Lorentz factor of the COM frame in the lab frame
                         * @param factorA @f[ \frac{\gamma_C - 1}{v_C^2} @f]
                         * @param comsVelocity @f[ v_C @f] COM system velocity in the lab frame
                         */
                        static DINLINE float3_COLL labToComs(
                            float3_COLL labMomentum,
                            float_COLL mass,
                            float_COLL gamma,
                            float_COLL gammaComs,
                            float_COLL factorA,
                            float3_COLL comsVelocity)
                        {
                            float3_COLL labVelocity = labMomentum / gamma / mass;
                            float_COLL dot = pmacc::math::dot(comsVelocity, labVelocity);
                            float_COLL factor = (factorA * dot - gammaComs);
                            factor *= mass * gamma;
                            float3_COLL diff = factor * comsVelocity;

                            PMACC_DEVICE_ASSERT(std::isfinite((labMomentum + diff)[0]));
                            PMACC_DEVICE_ASSERT(std::isfinite((labMomentum + diff)[1]));
                            PMACC_DEVICE_ASSERT(std::isfinite((labMomentum + diff)[2]));
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
                        static DINLINE float3_COLL comsToLab(
                            float3_COLL comsMomentum,
                            float_COLL mass,
                            float_COLL coeff,
                            float_COLL gammaComs,
                            float_COLL factorA,
                            float3_COLL comsVelocity)
                        {
                            // (13) in [Perez 2012]
                            // TODO: try upgrading this to 64bit as well
                            float_COLL dot = pmacc::math::dot(comsVelocity, comsMomentum);
                            float_COLL factor = (factorA * dot + coeff * gammaComs);
                            float3_COLL diff = factor * comsVelocity;
                            PMACC_DEVICE_ASSERT(std::isfinite((comsMomentum + diff)[0]));
                            PMACC_DEVICE_ASSERT(std::isfinite((comsMomentum + diff)[1]));
                            PMACC_DEVICE_ASSERT(std::isfinite((comsMomentum + diff)[2]));
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
                        static DINLINE float_COLL calcRelativeComsVelocity(
                            float_COLL comsMomentumMag0,
                            float_COLL mass0,
                            float_COLL mass1,
                            float_COLL gamma0,
                            float_COLL gamma1,
                            float_COLL coeff0,
                            float_COLL coeff1,
                            float_COLL gammaComs)
                        {
                            float_COLL val = (mass0 * gamma0 + mass1 * gamma1) * comsMomentumMag0;
                            val = val / (coeff0 * coeff1 * gammaComs); // TODO: null division?
                            // TODO:
                            // this is actualy no the relative velocity! since we are missing the (1 + v1v2/c^2) factor
                            // this is actually only used in the non-relativistic limit. Should we use
                            // a non-relativistic formula? Propably yes!

                            //                            if(!(std::isfinite(val)) || val>=c)
                            //                            {
                            //                                printf(
                            //                                    "comsMomentumMag0: %e, mass0: %e, mass1: %e, gamma0:
                            //                                    %e, gamma1: %e," " coeff0: %e, coeff1: %e, gammaComs:
                            //                                    %e, val: %e", comsMomentumMag0, mass0, mass1, gamma0,
                            //                                    gamma1,
                            //                                    coeff0,
                            //                                    coeff1,
                            //                                    gammaComs,
                            //                                    val);
                            //                                PMACC_DEVICE_ASSERT0);
                            //}
                            return val;
                        }

                        /* Calculate @f[ \gamma^* m @f]
                         *
                         * Returns particle mass times its Lorentz factor in the COM frame.
                         *
                         * @param labMomentum particle momentum in the labFrame
                         * @param mass particle mass
                         * @param gamma
                         * @param gammaComs Lorentz factor of the COM frame in the lab frame
                         * @param comsVelocity COM system velocity in the lab frame
                         */
                        //
                        static DINLINE float_COLL coeff(
                            float3_COLL labMomentum,
                            float_COLL mass,
                            float_COLL gamma,
                            float_COLL gammaComs,
                            float3_COLL comsVelocity)
                        {
                            float3_COLL labVelocity = labMomentum / gamma / mass;
                            float_COLL dot = pmacc::math::dot(comsVelocity, labVelocity);

                            float_COLL val = gammaComs * gamma - dot * (gammaComs * gamma / (c * c));
                            PMACC_DEVICE_ASSERT_MSG(
                                val > 0.0_COLL,
                                "dot: %e, val: %e, gammaComs: %e, mass: %e, labMomentum0: %e, labMomentum1: %e, "
                                "labMomentum2: %e, gamma: %e, comsVelocity0 %e, comsVelocity1 %e, comsVelocity2 "
                                "%e \n",
                                dot,
                                val,
                                gammaComs,
                                mass,
                                labMomentum[0],
                                labMomentum[1],
                                labMomentum[2],
                                gamma,
                                comsVelocity[0],
                                comsVelocity[1],
                                comsVelocity[2]);
                            val *= mass;
                            PMACC_DEVICE_ASSERT_MSG(
                                (std::isfinite(val) && val > 0.0_COLL),
                                "dot: %e, val: %e, gammaComs: %e, mass: %e, labMomentum0: %e, labMomentum1: %e, "
                                "labMomentum2: %e, gamma: %e, comsVelocity0 %e, comsVelocity1 %e, comsVelocity2 "
                                "%e \n",
                                dot,
                                val,
                                gammaComs,
                                mass,
                                labMomentum[0],
                                labMomentum[1],
                                labMomentum[2],
                                gamma,
                                comsVelocity[0],
                                comsVelocity[1],
                                comsVelocity[2]);
                            return val;
                        }

                        /* Calculate the cosine of the scattering angle.
                         *
                         * The probabilty distribution  for the cosine depends on @f[ s_{12} @f]. The returned vale
                         * is determined by a float value between 0 and 1.
                         *
                         * @param s12 @f[ s_{12} @f] parameter. See [Perez 2012].
                         * @param u a usually random generated float between 0 and 1
                         */
                        static DINLINE float_COLL calcCosXi(float_COLL const s12, float_COLL const u)
                        {
                            // TODO: find some better way to restrict the val to [-1, +1].
                            if(s12 < 0.1_COLL)
                            {
                                float_COLL cosXi = 1.0_COLL + s12 * math::log(u);
                                if(cosXi < -1.0_COLL || cosXi > 1.0_COLL)
                                {
                                    if(cosXi < -1.0_COLL)
                                        return -1.0_COLL;
                                    else
                                        return 1.0_COLL;
                                }
                                else
                                    return cosXi;
                            }
                            else if(s12 < 6.0_COLL)
                            {
                                float_COLL a;
                                if(s12 < 3.0_COLL)
                                {
                                    float_COLL s12sq = s12 * s12;
                                    a = 0.0056958_COLL + 0.9560202_COLL * s12 - 0.508139_COLL * s12sq
                                        + 0.47913906_COLL * s12sq * s12 - 0.12788975_COLL * s12sq * s12sq
                                        + 0.02389567_COLL * s12sq * s12sq
                                            * s12sq; //                                PMACC_DEVICE_ASSERT_MSG(
                                    //                                    std::isfinite(gammaComs),
                                    //                                    "gammaComs: %e, comsVelocityAbs2: %e,
                                    //                                    labMomentum0: %e, labMomentum1 %e, mass0: %e,
                                    //                                    mass1: %e, " "gamma0: %e, gamma1: %e",
                                    //                                    gammaComs,
                                    //                                    comsVelocityAbs2,
                                    //                                    labMomentum0,
                                    //                                    labMomentum1,
                                    //                                    mass0,
                                    //                                    mass1,
                                    //                                    gamma0,
                                    //                                    gamma1);
                                    a = 1 / a;
                                }
                                else
                                {
                                    a = 3.0_COLL * math::exp(-1.0_COLL * s12);
                                }
                                float_COLL bracket = math::exp(-1.0_COLL * a) + 2.0_COLL * u * std::sinh(a);
                                float_COLL cosXi = math::log(bracket) / a;
                                if(cosXi < -1.0_COLL || cosXi > 1.0_COLL)
                                {
                                    if(cosXi < -1.0_COLL)
                                        return -1.0_COLL;
                                    else
                                        return 1.0_COLL;
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
                        static DINLINE float3_COLL
                        calcFinalComsMomentum(float3_COLL const p, float_COLL const cosXi, float_COLL const phi)
                        {
                            float_COLL sinPhi, cosPhi;
                            pmacc::math::sincos(phi, sinPhi, cosPhi);
                            float_COLL sinXi = math::sqrt(1 - cosXi * cosXi);

                            // (12) in [Perez 2012]
                            float3_COLL finalVec;
                            // TODO: sqrt returns a pmacc complex number. cast to float?
                            float_COLL const pAbs = math::sqrt(pmacc::math::abs2(p));
                            float_COLL const pPerp = math::sqrt(p.x() * p.x() + p.y() * p.y());
                            // TODO chose a better limmit?
                            if(pPerp > std::max(std::numeric_limits<float_COLL>::epsilon(), 1.0e-10_COLL) * pAbs)
                            {
                                finalVec[0] = (p.x() * p.z() * sinXi * cosPhi - p.y() * pAbs * sinXi * sinPhi) / pPerp
                                    + p.x() * cosXi;
                                finalVec[1] = (p.y() * p.z() * sinXi * cosPhi + p.x() * pAbs * sinXi * sinPhi) / pPerp
                                    + p.y() * cosXi;
                                finalVec[2] = -1.0_COLL * pPerp * sinXi * cosPhi + p.z() * cosXi;
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
                            if((par0[momentum_] == float3_X{0.0_X, 0.0_X, 0.0_X})
                               && (par1[momentum_] == float3_X{0.0_X, 0.0_X, 0.0_X}))
                                return;
                            // feenableexcept(FE_INVALID | FE_OVERFLOW | FE_DIVBYZERO);
                            float_COLL const normalizedWeight0
                                = precisionCast<float_COLL>(par0[weighting_]) / WEIGHT_NORM_COLL;
                            float_COLL const normalizedWeight1
                                = precisionCast<float_COLL>(par1[weighting_]) / WEIGHT_NORM_COLL;

                            float3_COLL const labMomentum0
                                = precisionCast<float_COLL>(par0[momentum_]) / normalizedWeight0;
                            float3_COLL const labMomentum1
                                = precisionCast<float_COLL>(par1[momentum_]) / normalizedWeight1;
                            float_COLL const mass0 = precisionCast<float_COLL>(picongpu::traits::attribute::getMass(
                                particles::TYPICAL_NUM_PARTICLES_PER_MACROPARTICLE,
                                par0));
                            float_COLL const mass1 = precisionCast<float_COLL>(picongpu::traits::attribute::getMass(
                                particles::TYPICAL_NUM_PARTICLES_PER_MACROPARTICLE,
                                par1));
                            float_COLL const charge0
                                = precisionCast<float_COLL>(picongpu::traits::attribute::getCharge(
                                    particles::TYPICAL_NUM_PARTICLES_PER_MACROPARTICLE,
                                    par0));
                            float_COLL const charge1
                                = precisionCast<float_COLL>(picongpu::traits::attribute::getCharge(
                                    particles::TYPICAL_NUM_PARTICLES_PER_MACROPARTICLE,
                                    par1));
                            float_COLL const gamma0 = picongpu::gamma<float_COLL>(labMomentum0, mass0);
                            float_COLL const gamma1 = picongpu::gamma<float_COLL>(labMomentum1, mass1);

                            //                            PMACC_DEVICE_ASSERT_MSG(
                            //                                math::sqrt(pmacc::math::abs2(labMomentum0)) / gamma0 /
                            //                                mass0 < c, "labVelocity0: %e",
                            //                                sqrt(pmacc::math::abs2(labMomentum0)) / gamma0 / mass0);
                            //                            PMACC_DEVICE_ASSERT_MSG(
                            //                                math::sqrt(pmacc::math::abs2(labMomentum1)) / gamma1 /
                            //                                mass1 < c, "labVelocity1: %e",
                            //                                sqrt(pmacc::math::abs2(labMomentum1)) / gamma1 / mass1);
                            // [Perez 2012] (1)
                            // todo:: maybe this also with doubles?
                            float3_COLL const comsVelocity
                                = (labMomentum0 + labMomentum1) / (mass0 * gamma0 + mass1 * gamma1);
                            float_COLL const comsVelocityAbs2 = pmacc::math::abs2(comsVelocity);
                            float3_COLL comsMomentum0;
                            float_COLL gammaComs, factorA, coeff0, coeff1;

                            if(comsVelocityAbs2 != 0.0_COLL)
                            {
                                gammaComs = 1.0_COLL / math::sqrt(1.0_COLL - comsVelocityAbs2 / c / c);
                                PMACC_DEVICE_ASSERT_MSG(
                                    std::isfinite(gammaComs),
                                    "gammaComs: %e, comsVelocityAbs2: %e, labMomentum0: %e, labMomentum1 %e, mass0: "
                                    "%e, mass1: %e, "
                                    "gamma0: %e, gamma1: %e",
                                    gammaComs,
                                    comsVelocityAbs2,
                                    labMomentum0,
                                    labMomentum1,
                                    mass0,
                                    mass1,
                                    gamma0,
                                    gamma1);
                                // used later for comsToLab:
                                factorA = (gammaComs - 1.0_COLL) / comsVelocityAbs2;

                                // Stared gamma times mass, from [Perez 2012].
                                coeff0 = coeff(labMomentum0, mass0, gamma0, gammaComs, comsVelocity);
                                // PMACC_DEVICE_ASSERT(coeff0 > 0.0_COLL);
                                // gamma^* . mass
                                // if(coeff0 == 0.0_COLL) return;
                                coeff1 = coeff(labMomentum1, mass1, gamma1, gammaComs, comsVelocity);
                                // if(coeff1 == 0.0_COLL) return;
                                // PMACC_DEVICE_ASSERT(coeff1 > 0.0_COLL);
                                // (2) in [Perez 2012]
                                comsMomentum0
                                    = labToComs(labMomentum0, mass0, gamma0, gammaComs, factorA, comsVelocity);
                            }
                            else
                            {
                                // From smilei implementation, for v_coms^2 = 0
                                gammaComs = 1.0_COLL;
                                // used later for comsToLab:
                                factorA = 0.5_COLL;
                                // Stared gamma times mass, from [Perez 2012].
                                coeff0 = mass0 * gamma0;
                                // PMACC_DEVICE_ASSERT(coeff0 > 0.0_COLL);
                                // gamma^* . mass
                                coeff1 = mass1 * gamma1;
                                // PMACC_DEVICE_ASSERT(coeff1 > 0.0_COLL);
                                comsMomentum0 = labMomentum0;
                            }


                            //  f0 * f1 * f2^2
                            //  is equal  s12 * (n12/(n1*n2)) from [Perez 2012]
                            if(comsMomentum0 == float3_COLL{0.0_COLL, 0.0_COLL, 0.0_COLL})
                                return;
                            float_COLL comsMomentum0Abs2 = pmacc::math::abs2(comsMomentum0);
                            if(comsMomentum0Abs2 == 0.0_COLL)
                                return;
                            float_COLL s12Factor0 = (DELTA_T_COLL * coulombLog * charge0 * charge0 * charge1 * charge1)
                                / (4.0_COLL * pmacc::math::Pi<float_COLL>::value * EPS0_COLL * EPS0_COLL * c * c * c
                                   * c * mass0 * gamma0 * mass1 * gamma1);
                            // PMACC_DEVICE_ASSERT(std::isfinite(s12Factor0));
                            s12Factor0 *= 1.0_COLL / WEIGHT_NORM_COLL / WEIGHT_NORM_COLL;
                            float_COLL const s12Factor1
                                = gammaComs * math::sqrt(comsMomentum0Abs2) / (mass0 * gamma0 + mass1 * gamma1);
                            // PMACC_DEVICE_ASSERT(std::isfinite(s12Factor1));
                            float_COLL const s12Factor2 = coeff0 * coeff1 * c * c / comsMomentum0Abs2 + 1.0_COLL;
                            // PMACC_DEVICE_ASSERT(std::isfinite(s12Factor2));
                            // Statistical part from [Higginson 2020],
                            // corresponds to n1*n2/n12 in [Perez 2012]:
                            float_COLL const s12Factor3 = potentialPartners
                                * pmacc::math::max(normalizedWeight0, normalizedWeight1) * WEIGHT_NORM_COLL
                                / static_cast<float_COLL>(duplications) / CELL_VOLUME_COLL;
                            float_COLL s12n = s12Factor0 * s12Factor1 * s12Factor2 * s12Factor2 * s12Factor3;
                            PMACC_DEVICE_ASSERT(!(std::isnan(s12n)));

                            // Low Temeprature correction:
                            // [Perez 2012] (8)
                            // TODO: should we check for the non-relativistic condition? Which gamma should we look at?
                            float_COLL relativeComsVelocity = calcRelativeComsVelocity(
                                math::sqrt(comsMomentum0Abs2),
                                mass0,
                                mass1,
                                gamma0,
                                gamma1,
                                coeff0,
                                coeff1,
                                gammaComs);
                            // [Perez 2012] (21) ( without n1*n2/n12 )
                            float_COLL s12Max
                                = math::pow(4.0_COLL * pmacc::math::Pi<float_COLL>::value / 3._COLL, 1.0 / 3.0)
                                * DELTA_T_COLL * (mass0 + mass1)
                                / pmacc::math::max(mass0 * densitySqCbrt0, mass1 * densitySqCbrt1)
                                * relativeComsVelocity;
                            s12Max *= s12Factor3;
                            float_COLL s12;
                            // PMACC_DEVICE_ASSERT(std::isfinite(s12Max));
                            s12 = pmacc::math::min(s12n, s12Max);
                            PMACC_DEVICE_ASSERT(std::isfinite(s12));

                            // Get a random float value from 0,1
                            auto const& acc = *ctx.m_acc;
                            auto& rngHandle = *ctx.m_hRng;
                            using UniformFloat = pmacc::random::distributions::Uniform<
                                pmacc::random::distributions::uniform::ExcludeZero<float_COLL>>;
                            auto rng = rngHandle.template applyDistribution<UniformFloat>();
                            float_COLL rngValue = rng(acc);

                            float_COLL const cosXi = calcCosXi(s12, rngValue);
                            // PMACC_ASSERT( cosXi <= 1 && cosXi >= -1 );
                            float_COLL const phi = 2.0_COLL * PI * rng(acc);
                            float3_COLL const finalComs0 = calcFinalComsMomentum(comsMomentum0, cosXi, phi);
                            PMACC_DEVICE_ASSERT(std::isfinite(finalComs0[0]));
                            PMACC_DEVICE_ASSERT(std::isfinite(finalComs0[1]));
                            PMACC_DEVICE_ASSERT(std::isfinite(finalComs0[2]));

                            float3_COLL finalLab0, finalLab1;
                            if(normalizedWeight0 > normalizedWeight1)
                            {
                                finalLab1 = comsToLab(
                                    -1.0_COLL * finalComs0,
                                    mass1,
                                    coeff1,
                                    gammaComs,
                                    factorA,
                                    comsVelocity);
                                //                                PMACC_DEVICE_ASSERT_MSG(
                                //                                    (sqrt(pmacc::math::abs2(finalLab1)) /
                                //                                    (picongpu::gamma<float_COLL>(finalLab1, mass1))
                                //                                     / mass1)
                                //                                        < c,
                                //                                    "finalLabVelocity1: %e",
                                //                                    math::sqrt(pmacc::math::abs2(finalLab1))
                                //                                        / (picongpu::gamma<float_COLL>(finalLab1,
                                //                                        mass1)) / mass1);
                                PMACC_DEVICE_ASSERT(std::isfinite(finalLab1[0] * normalizedWeight1));
                                PMACC_DEVICE_ASSERT(std::isfinite(finalLab1[1] * normalizedWeight1));
                                PMACC_DEVICE_ASSERT(std::isfinite(finalLab1[2] * normalizedWeight1));
                                par1[momentum_] = finalLab1 * normalizedWeight1;
                                if((normalizedWeight1 / normalizedWeight0) - rng(acc) > 0)
                                {
                                    finalLab0 = comsToLab(finalComs0, mass0, coeff0, gammaComs, factorA, comsVelocity);
                                    //                                    PMACC_DEVICE_ASSERT_MSG(
                                    //                                        (sqrt(pmacc::math::abs2(finalLab0))
                                    //                                         /
                                    //                                         (picongpu::gamma<float_COLL>(finalLab0,
                                    //                                         mass0)) / mass0)
                                    //                                            < c,
                                    //                                        "finalLabVelocity0: %e",
                                    //                                        math::sqrt(pmacc::math::abs2(finalLab0))
                                    //                                            /
                                    //                                            (picongpu::gamma<float_COLL>(finalLab0,
                                    //                                            mass0)) / mass0);
                                    PMACC_DEVICE_ASSERT(std::isfinite(finalLab0[0] * normalizedWeight0));
                                    PMACC_DEVICE_ASSERT(std::isfinite(finalLab0[1] * normalizedWeight0));
                                    PMACC_DEVICE_ASSERT(std::isfinite(finalLab0[2] * normalizedWeight0));
                                    par0[momentum_] = finalLab0 * normalizedWeight0;
                                }
                            }
                            else
                            {
                                finalLab0 = comsToLab(finalComs0, mass0, coeff0, gammaComs, factorA, comsVelocity);
                                //                                PMACC_DEVICE_ASSERT_MSG(
                                //                                (sqrt(pmacc::math::abs2(finalLab0)) /
                                //                                (picongpu::gamma<float_COLL>(finalLab0, mass0))
                                //                                     / mass0)
                                //                                        < c,
                                //                                    "finalLabVelocity1: %e",
                                //                                    math::sqrt(pmacc::math::abs2(finalLab0))
                                //                                        / (picongpu::gamma<float_COLL>(finalLab0,
                                //                                        mass0)) / mass0);
                                PMACC_DEVICE_ASSERT(std::isfinite(finalLab0[0] * normalizedWeight0));
                                PMACC_DEVICE_ASSERT(std::isfinite(finalLab0[1] * normalizedWeight0));
                                PMACC_DEVICE_ASSERT(std::isfinite(finalLab0[2] * normalizedWeight0));
                                par0[momentum_] = finalLab0 * normalizedWeight0;
                                if((normalizedWeight0 / normalizedWeight1) - rng(acc) >= 0.0_COLL)
                                {
                                    finalLab1 = comsToLab(
                                        -1.0_COLL * finalComs0,
                                        mass1,
                                        coeff1,
                                        gammaComs,
                                        factorA,
                                        comsVelocity);
                                    //                                    PMACC_DEVICE_ASSERT_MSG(
                                    //                                        (sqrt(pmacc::math::abs2(finalLab1))
                                    //                                         /
                                    //                                         (picongpu::gamma<float_COLL>(finalLab1,
                                    //                                         mass1)) / mass1)
                                    //                                            < c,
                                    //                                        "finalLabVelocity0: %e",
                                    //                                        math::sqrt(pmacc::math::abs2(finalLab1))
                                    //                                            /
                                    //                                            (picongpu::gamma<float_COLL>(finalLab1,
                                    //                                            mass1)) / mass1);
                                    PMACC_DEVICE_ASSERT(std::isfinite(finalLab1[0] * normalizedWeight1));
                                    PMACC_DEVICE_ASSERT(std::isfinite(finalLab1[1] * normalizedWeight1));
                                    PMACC_DEVICE_ASSERT(std::isfinite(finalLab1[2] * normalizedWeight1));
                                    par1[momentum_] = finalLab1 * normalizedWeight1;
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
                            math::pow(precisionCast<acc::float_COLL>(density0), 2. / 3.),
                            math::pow(precisionCast<acc::float_COLL>(density1), 2. / 3.),
                            potentialPartners,
                            precisionCast<acc::float_COLL>(coulombLog));
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
