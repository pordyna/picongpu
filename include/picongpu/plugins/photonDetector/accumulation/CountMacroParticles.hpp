/* Copyright 2015-2021 Alexander Grund, Pawel Ordyna
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

#include "picongpu/plugins/photonDetector/accumulation/CountParticles.def"


namespace picongpu
{
    namespace plugins
    {
        namespace photonDetector
        {
            namespace accumulation
            {
                namespace acc
                {
                    /** Accumulation device functor for simple particle counting
                     *
                     * @tparam T_Species type of the particle species accumulated on the detector
                     */
                    template<typename T_Species>
                    struct CountMacroParticles
                    {
                    public:
                        //! type used to store detector cell values
                        using Type = float_64;

                        /** CountParticles device functor constructor
                         *
                         * @param currentStep current simulation time step
                         * @param detector detector description
                         * @param simSize simulation size (in the detector coordinate system)
                         */
                        HDINLINE CountMacroParticles(
                            uint32_t const& currentStep,
                            DetectorParams const& detector,
                            DataSpace<simDim> const& simSize)
                            : detector_m(detector)
                            , simSize_m(simSize)
                        {
                        }

                        /** accumulate particle on a detector cell
                         *
                         * @param acc alpaka accelerator
                         * @param detectorBox detector data box
                         * @param targetCellIdx the index of the detector cell hit by the particle
                         * @param particle particle to accumulate
                         * @param globalCellIdx the index of the cell from which the particle has left the simulation
                         *  box (the cell in GUARD where the particle will be deleted). In the detector coordinate
                         *  system.
                         */
                        template<typename T_Acc, typename T_DetectorBox, typename T_Particle>
                        DINLINE void operator()(
                            T_Acc const& acc,
                            T_DetectorBox detectorBox,
                            const DataSpace<DIM2>& targetCellIdx,
                            const T_Particle& particle,
                            const DataSpace<simDim>& globalCellIdx) const
                        {
                            cupla::atomicAdd(
                                acc,
                                &(detectorBox(targetCellIdx)[0]),
                                1.0,
                                alpaka::hierarchy::Blocks{});
                        }

                    private:
                        PMACC_ALIGN(detector_m, const DetectorParams);
                        PMACC_ALIGN(simSize_m, const DataSpace<simDim>);
                    };
                } // namespace acc
                /** Functor factory (host side functor) for the CountParticles device accumulation functor
                 *
                 * @tparam T_Species type of the particle species accumulated on the detector
                 */
                template<typename T_Species>
                struct CountMacroParticles
                {
                public:
                    //! type used to store detector cell values
                    // TODO: switch to uint (MPI reduce needs a new specialization)
                    using ComponentType = float_64;
                    using Type = pmacc::math::Vector<ComponentType, DIM1>;
                    //! the value used to initialize (reset) detector storage
                    const Type initValue = Type::create(0.0);
                    using AccFunctorType = acc::CountMacroParticles<T_Species>;

                    //! Get unit dimension of the values accumulated on the detector
                    HINLINE std::vector<float_64> getUnitDimension()
                    {
                        /*
                         */
                        std::vector<float_64> unitDimension(7, 0.0);
                        unitDimension.at(SIBaseUnits::length) = 0.0;
                        unitDimension.at(SIBaseUnits::mass) = 0.0;
                        unitDimension.at(SIBaseUnits::time) = -0.0;
                        unitDimension.at(SIBaseUnits::electricCurrent) = 0.0;
                        return unitDimension;
                    }

                    //! Get the SI unit conversion for the values accumulated on the detector
                    HDINLINE float_64 getUnit()
                    {
                        return 1.0;
                    }

                    //! Get a descriptive name for the openPMD mesh storing detector output
                    HINLINE static std::string getOpenPMDMeshName()
                    {
                        return "macroParticleCount";
                    }

                    //! Get a descriptive name of this accumulation policy
                    HINLINE static std::string getName()
                    {
                        return "CountMacroParticles";
                    }

                    HINLINE std::string getOpenPMDMeshSuffix(uint32_t component) const
                    {
                        return "";
                    }

                    /** Create a device side particle accumulation functor for the CountParticles policy
                     *
                     * @param currentStep current simulation step
                     * @param detector detector description
                     * @param simSize simulation size (in the detector coordinate system)
                     * @return device side functor
                     */
                    HINLINE acc::CountMacroParticles<T_Species> operator()(
                        uint32_t const& currentStep,
                        DetectorParams const& detector,
                        DataSpace<simDim> const& simSize,
                        externalBeam::AxisSwap const& axisSwap) const
                    {
                        return acc::CountMacroParticles<T_Species>(currentStep, detector, simSize);
                    }
                };
            } // namespace accumulation
        } // namespace photonDetector
    } // namespace plugins
} // namespace picongpu
