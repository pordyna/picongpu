/* Copyright 2013-2021 Axel Huebl, Heiko Burau, Rene Widera,
 *                     Alexander Grund
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

#include "picongpu/particles/startPosition/detail/WeightMacroParticles.hpp"
#include "picongpu/particles/startPosition/generic/FreeRng.def"

#include <boost/mpl/integral_c.hpp>


namespace picongpu
{
    namespace particles
    {
        namespace externalBeam
        {
            namespace startPosition
            {
                template<typename T_ParamClass>
                struct RandomProbingBeamImpl
                {
                    using ParamClass = T_ParamClass;
                    // Defines from which side the beam enters the simulation box.
                    using SideCfg = typename T_ParamClass::ProbingBeam::SideCfg;

                private:
                    // shorthand for compile-time indices conversion (between the beam and the simulation coordinates)
                    template<uint32_t idx>
                    using BeamToPicIdx_t = typename SideCfg::AxisSwapCT::template BeamToPicIdx<idx>::type;
                    template<uint32_t idx>
                    using PicToBeamIdx_t = typename SideCfg::AxisSwapCT::template PicToBeamIdx<idx>::type;

                    /* compile-time calculation of the in-cell position range. Along the beam propagation direction
                     * ( z in the beam system) particles are created only up to the distance that a particle travels
                     * in one time-step. Here we assume the particles are photons and travel with the speed of light.
                     */
                    static constexpr float_X cellSizeCT[3] = {CELL_WIDTH, CELL_HEIGHT, CELL_DEPTH};
                    static constexpr float_X cellDepth{cellSizeCT[BeamToPicIdx_t<2u>::value]};
                    static constexpr float_X posLimBeam[3] = {1.0_X, 1.0_X, DELTA_T* SPEED_OF_LIGHT / cellDepth};
                    static constexpr float_X posLimPic_x = posLimBeam[PicToBeamIdx_t<0u>::value];
                    static constexpr float_X posLimPic_y = posLimBeam[PicToBeamIdx_t<1u>::value];
                    static constexpr float_X posLimPic_z = posLimBeam[PicToBeamIdx_t<2u>::value];

                    // This is true when the beam travels **against** one of the simulation coordinate system unit
                    // vectors.
                    static constexpr bool reverse = SideCfg::Side::reverse[2];


                public:
                    HINLINE RandomProbingBeamImpl(uint32_t currentStep) : axisSwap()
                    {
                    }

                    /** Set in-cell position and weighting
                     *
                     * @tparam T_MetaData type of the low level functor used to call the StartAttributes functor
                     *      which calls this functor.
                     * @tparam T_Particle pmacc::Particle, particle type
                     * @tparam T_Args pmacc::Particle, arbitrary number of particles types
                     *
                     * @param acc alpaka accelerator
                     * @param meta the instance of T_MetaData, provides domain info and the rng.
                     * @param particle particle to be manipulated
                     * @param ... unused particles
                     */
                    template<typename T_Acc, typename T_MetaData, typename T_Particle, typename... T_Args>
                    HDINLINE void operator()(T_Acc const& acc, T_MetaData& meta, T_Particle& particle, T_Args&&...)
                    {
                        // get the random number generator from the low level functor
                        auto rng = meta.getRngHandle()
                                       .template applyDistribution<pmacc::random::distributions::Uniform<float_X>>();
                        floatD_X tmpPos;

                        // array is not available in device code. even constexpr  array.
                        const float3_X posLimPicVec{posLimPic_x, posLimPic_y, posLimPic_z};

                        // generate a random in-cell position for each coordinate. In the beam propagation direction
                        // the position is limited by the distance a particle can travel in one time-step.
                        for(uint32_t d = 0; d < simDim; ++d)
                            tmpPos[d] = rng(acc) * posLimPicVec[d];


                        // Shift the coordinate along the beam propagation direction towards the external boundary if
                        // the boundary is on the end of the cell. ( The beam enters the simulation from the bottom,
                        // rear or the right side).
                        if(reverse)
                        {
                            tmpPos[BeamToPicIdx_t<2u>::value] = 1.0_X - tmpPos[BeamToPicIdx_t<2u>::value];
                        }

                        particle[position_] = tmpPos;
                        particle[weighting_] = m_weighting;
                    }

                    /* Get the number of particles needed to be created in the simulation cell
                     *
                     * @tparam T_Particle type of the particles that should be created
                     * @tparam T_MetaData type of the low level functor sued to call the StartAttributes functor
                     *      which call this functor.
                     *
                     * @param meta the instance of T_MetaData, provides domain info and the rng.
                     * @param realParticlesPerCell number of new real particles in the cell in which this instance
                     * creates particles
                     *
                     * @return number of macro particles that need to be created in this cell (The operator() will be
                     * called that many times)
                     */
                    template<typename T_Particle, typename T_MetaData>
                    HDINLINE uint32_t
                    numberOfMacroParticles(T_MetaData const& meta, float_X const realParticlesPerCell)
                    {
                        // Check if the beam is coming from this side.
                        DataSpace<simDim> globalDomainOffset = meta.domInfo.global.offset;
                        DataSpace<DIM3> globalDomainOffsetBeamSystem = axisSwap.transformCellIdx(globalDomainOffset);
                        if(globalDomainOffsetBeamSystem.z() != 0)
                            return 0u;

                        // Only create particles if weighting is not below the minimal value.
                        // Notice this is different as in the usual startPosition functor where the number of macro
                        // particles would be reduced to satisfy this condition.
                        m_weighting = realParticlesPerCell / T_ParamClass::numParticlesPerCell;
                        if(m_weighting < T_ParamClass::minWeighting)
                            return 0u;
                        return T_ParamClass::numParticlesPerCell;
                    }

                    float_X m_weighting;
                    PMACC_ALIGN(axisSwap, typename SideCfg::AxisSwapRT);
                };

            } // namespace startPosition
        } // namespace externalBeam
    } // namespace particles
} // namespace picongpu
