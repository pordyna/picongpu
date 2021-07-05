/* Copyright 2013-2021 Axel Huebl, Heiko Burau, Rene Widera, Pawel Ordyna
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

#include "picongpu/particles/manipulators/IUnary.def"
#include "picongpu/plugins/externalBeam/AxisSwap.hpp"
#include "picongpu/plugins/externalBeam/Side.hpp"

#include <boost/mpl/integral_c.hpp>


namespace picongpu
{
    namespace particles
    {
        namespace externalBeam
        {
            namespace acc
            {
                template<
                    typename T_StartPositionFunctor,
                    typename T_MomentumFunctor,
                    typename T_PhaseFunctor,
                    typename T_PolarizationFunctor,
                    typename T_Species>
                struct StartAttributesImpl
                {
                    // sub-functors:
                    using StartPositionFunctor = T_StartPositionFunctor;
                    using MomentumFunctor = T_MomentumFunctor;
                    using PhaseFunctor = typename bmpl::apply1<T_PhaseFunctor, T_Species>::type;
                    using PolarizationFunctor = typename bmpl::apply1<T_PolarizationFunctor, T_Species>::type;
                    using Species = T_Species;

                    template<typename T_SpeciesType>
                    struct apply
                    {
                        using type = StartAttributesImpl<
                            T_StartPositionFunctor,
                            T_MomentumFunctor,
                            T_PhaseFunctor,
                            T_PolarizationFunctor,
                            T_SpeciesType>;
                    };

                public:
                    HINLINE StartAttributesImpl(uint32_t currentStep)
                        : startPositionFunctor(StartPositionFunctor(currentStep))
                        , momentumFunctor(MomentumFunctor(currentStep))
                        , phaseFunctor(PhaseFunctor(currentStep))
                        , polarizationFunctor(PolarizationFunctor(currentStep))
                    {
                    }
                    /** Set in-cell position, weighting, momentum, phase
                     *
                     * @tparam T_Acc alpaka accelerator type
                     * @tparam T_MetaData a generic functor type which inherits from this class and calls this functor
                     * @tparam T_Particle pmacc::Particle, particle type
                     * @tparam T_Args pmacc::Particle, arbitrary number of particles types
                     *
                     * @param acc alpaka accelerator
                     * @param meta the instance of T_MetaData that calls this functor
                     * @param particle particle to be manipulated
                     * @param ... unused particles
                     */
                    template<typename T_Acc, typename T_MetaData, typename T_Particle, typename... T_Args>
                    HDINLINE void operator()(T_Acc const& acc, T_MetaData& meta, T_Particle& particle, T_Args&&...)
                    {
                        // set position and weighting
                        startPositionFunctor(acc, meta, particle);
                        // set momentum (needs weighting to be already set)
                        momentumFunctor(acc, meta, particle);
                        // this does nothing if T_PhaseFunctor = particles::externalBeam::polarization::NoPolarization.
                        polarizationFunctor(acc, meta, particle);
                        // set phase (needs position already set and may need momentum already set)
                        // this does nothing if T_PhaseFunctor = particles::externalBeam::phase::NoPhase .
                        phaseFunctor(acc, meta, particle);
                    }


                    /** Get the number of macro particles that should be created and initialize this functor
                     *
                     * This is called only once before the operator is called. Hence this method is also used to
                     * initialize this functor and the  sub-functors. There is one instance of the low level functor
                     * for each cell in KernelFillGridWithParticles so that the initialization can depend on the cell
                     * position.
                     *
                     * @tparam T_Particle type of the particles that should be created
                     * @tparam T_MetaData a generic functor type which inherits from this class and calls this functor
                     *
                     * @param meta the instance of T_MetaData that calls this functor
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
                        // initialize sub-functors
                        momentumFunctor.template init<T_Particle>(meta);
                        phaseFunctor.template init<T_Particle>(meta);
                        polarizationFunctor.template init<T_Particle>(meta);

                        // numberOfMacroParticles also initializes the start position functor
                        const uint32_t numMacroParticles
                            = startPositionFunctor.template numberOfMacroParticles<T_Particle>(
                                meta,
                                realParticlesPerCell);

                        return numMacroParticles;
                    }

                private:
                    PMACC_ALIGN(startPositionFunctor, StartPositionFunctor);
                    PMACC_ALIGN(momentumFunctor, MomentumFunctor);
                    PMACC_ALIGN(phaseFunctor, PhaseFunctor);
                    PMACC_ALIGN(polarizationFunctor, PolarizationFunctor);
                };
            } // namespace acc
        } // namespace externalBeam
    } // namespace particles
} // namespace picongpu
