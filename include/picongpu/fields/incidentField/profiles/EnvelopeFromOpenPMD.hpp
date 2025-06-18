/* Copyright 2024-2024 Fabia Dietrich
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#if (ENABLE_OPENPMD == 1)

#    pragma once

#    include "picongpu/defines.hpp"
#    include "picongpu/fields/incidentField/Functors.hpp"
#    include "picongpu/fields/incidentField/profiles/FromOpenPMDPulse.def"

#    include <pmacc/math/Complex.hpp>
#    include <pmacc/memory/buffers/HostDeviceBuffer.hpp>

#    include <algorithm>
#    include <array>
#    include <cmath>
#    include <cstdint>
#    include <limits>
#    include <memory>
#    include <string>
#    include <type_traits>
#    include <vector>

#    include <openPMD/openPMD.hpp>

/* REFACTORING IDEAS FOR THIS INCIDENT FIELD PROFILE
 * -------------------------------------------------
 * - make time delay parameter optional
 * - load openPMD file (= call the corresponding singelton) or initialize the
 *   Laser once before timestep 0 (before particle memory allocation)
 * - load just the necessary parts of the measured data if the tranversal
 *   simulation window extent is smaller than the transversal field chunk size
 * - allow diagonal laser propagation instead of just parallel to the axes
 * - every used device will store the whole field data chunk, which consumes
 *   quite some memory. Instead, one could push only those two time slices to
 *   the device which are necessary for the current time step.
 * - get rid of the 'wrong' transformation from time to space (z = c*t) of the
 *   longitudinal axis by using several iterations inside the openPMD file
 *   instead of just one
 * - in 2D simulations, load only the necessary slice into the simulation, not
 *   the whole 3D chunk, and provide the index of the used slice as user parameter
 */

namespace picongpu
{
    namespace fields
    {
        namespace incidentField
        {
            namespace profiles
            {
                namespace detail
                {

                    /** Singleton to load field data from openPMD to device
                     *
                     * The complete dataset will be loaded (equally) to all GPUs, as well as
                     * the necessary attributes (extent, cell size, offset to simulation window).
                     *
                     * Right now, the data will be loaded at timestep 0, which means that the user
                     * has to **increase the reserved GPU memory** in memory.param, since otherwise
                     * the simulation will run into memory issues.
                     *
                     * @tparam T_Params user parameters, providing filename etc.
                     */
                    template<typename T_Params>
                    struct EnvelopeOpenPMDdata : public T_Params
                    {
                        //! Parameters type
                        using Params T_Params;
                        //! field record data type
                        using dataTypeEnvelope = typename Params::dataTypeEnvelope;

                        //! HostDeviceBuffer to store field envelope data
                        std::shared_ptr<pmacc::HostDeviceBuffer<dataTypeEnvelope, 1u>> bufferFieldEnvelope;
                        size_t extent;
                        float_X timeGridOffset;
                        float_X timeGridSize;

                        //! loading data to device
                        static EnvelopeOpenPMDdata& get()
                        {
                            static EnvelopeOpenPMDdata dataBuffers{};
                            return dataBuffers;
                        }

                    private:
                        EnvelopeOpenPMDdata()
                        {
                            /* Open a series (this does not read the dataset itself).
                             * This is MPI collective and so has to be done by all ranks.
                             */
                            auto& gc = Environment<simDim>::get().GridController();
                            eventSystem::getTransactionEvent().waitForFinished();
                            auto series = ::openPMD::Series{
                                Params::filename,
                                ::openPMD::Access::READ_ONLY,
                                gc.getCommunicator().getMPIComm()};
                            ::openPMD::Mesh envelopeMesh
                                = series.iterations[Params::iteration].meshes[Params::datasetEnvelopeName];
                            // check data order
                            if(envelopeMesh.dataOrder() != ::openPMD::Mesh::DataOrder::C)
                                throw std::runtime_error(
                                    "Unsupported dataOrder in openPMD field envelope dataset, only C is supported");

                            ::openPMD::MeshRecordComponent envelopeMeshRecord
                                = envelopeMesh[::openPMD::RecordComponent::SCALAR];

                            if(envelopeMeshRecord.getDimensionality() != 1u)
                                throw std::runtime_error("field envelope dataset has to be 1D");
                            if(envelopeMeshRecord.unitSI() != 1.0)
                                throw std::runtime_error("field envelope's unitSI should be equal 1");
                            if(envelopeMesh.unitDimension != {{0., 0., 0., 0., 0., 0., 0.}})
                                throw std::runtime_error("field envelope has to be dimensionless");

                            ::openPMD::Extent const openPMDExtentEnvelope = envelopeMeshRecord.getExtent();
                            DataSpace<1u> const envelopeExtent{openPMDExtentEnvelope[0]};

                            bufferFieldEnvelope
                                = std::make_shared<pmacc::HostDeviceBuffer<float_X, 1u>>(envelopeExtent);
                            envelopeMeshRecord.loadChunkRaw<dataTypeEnvelope>(
                                bufferFieldEnvelope->getHostBuffer().data());
                            // This is MPI collective and so has to be done by all ranks
                            series.flush();
                            //! Push field data to device
                            envelopeMeshRecord->hostToDevice();
                            eventSystem::getTransactionEvent().waitForFinished();

                            extent = envelopeExtent[0];
                            timeGridOffset = precisionCast<float_X>(
                                envelopeMesh.gridOffset<float_64>()[0] * envelopeMesh.gridUnitSI() / sim.unit.time());
                            timeGridSize = precisionCast<float_X>(
                                envelopeMesh.gridSpacing<float_64>()[0] * envelopeMesh.gridUnitSI() / sim.unit.time());
                        } // OpenPMDdata
                    };
                } // namespace detail

                /** FromOpenPMDPulse incident E functor
                 *
                 * @tparam T_Params parameters
                 */
                template<typename T_Params>
                struct EnvelopeFromOpenPMD : public T_Params
                {
                    using dataTypeEnvelope = typename T_Params::dataTypeEnvelope;

                    static constexpr float_X TIME_SHIFT = 0.0_X;

                    /** Create a functor on the host side for the given time step
                     *
                     * @param currentStep current time step index, note that it is fractional
                     * @param unitField conversion factor from SI to internal units,
                     *                  fieldE_internal = fieldE_SI / unitField
                     */
                    HINLINE EnvelopeFromOpenPMD()
                    {
                        // load data at timestep 0
                        auto& openPMDdata = EnvelopeOpenPMDdata<T_Params>::get();
                        // get field data
                        extent = openPMDdata.bufferFieldData->getDeviceBuffer().getDataBox();
                        extent = openPMDdata.extent;
                        timeGridOffset = openPMDdata.timeGridOffset;
                        timeGridSize = openPMDdata.timeGridSize;
                    }

                    HDINLINE float_X getEnvelope(float_X const time)
                    {
                        auto const timeInTimeCellsFloat = (time - timeGridOffset) / timeGridSize;
                        // exit if simulation time before provided envelope
                        if(timeInTimeCellsFloat < 0.0_X)
                            return 0.0_X;
                        size_t const idx_low = precisionCast<size_t>(math::floor(timeInTimeCellsFloat));
                        size_t idx_high = precisionCast<size_t>(math::ceil(timeInTimeCellsFloat));
                        // exit if simulation time past provided envelope
                        if(idx_low >= extent)
                            return 0.0_X;
                        if(idx_low == extent - 1u)
                            idx_high = idx_low;
                        auto const remainder = math::remainder(timeInTimeCellsFloat, static_cast<float_X>(idx_low));

                        return envelopeDataBox[idx_low] * (1.0_X - remainder) + envelopeDataBox[idx_high] * remainder;
                    }

                    //! Get text name of the incident field profile
                    HINLINE static std::string getName()
                    {
                        return "EnvelopeFromOpenPMD";
                    }

                protected:
                    PMACC_ALIGN(envelopeDataBox, typename pmacc::Buffer<dataTypeEnvelope, 1u>::DataBoxType);
                    PMACC_ALIGN(extent, size_t);
                    PMACC_ALIGN(timeGridOffset, float_X);
                    PMACC_ALIGN(timeGridSize, float_X);
                };
            } // namespace profiles
        } // namespace incidentField
    } // namespace fields
} // namespace picongpu

#endif
