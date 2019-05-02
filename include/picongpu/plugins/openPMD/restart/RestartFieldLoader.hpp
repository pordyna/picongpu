/* Copyright 2014-2019 Axel Huebl, Felix Schmitt, Heiko Burau, Rene Widera
 *                     Benjamin Worpitz, Franz Pöschel
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

#include "picongpu/plugins/openPMD/openPMDWriter.def"
#include "picongpu/simulationControl/MovingWindow.hpp"
#include "picongpu/simulation_defines.hpp"

#include <pmacc/dataManagement/DataConnector.hpp>
#include <pmacc/dimensions/DataSpace.hpp>
#include <pmacc/dimensions/GridLayout.hpp>
#include <pmacc/particles/frame_types.hpp>
#include <pmacc/types.hpp>

#include <openPMD/openPMD.hpp>

#include <adios.h>
#include <sstream>
#include <stdexcept>
#include <string>

#include <adios_error.h>
#include <adios_read.h>


namespace picongpu
{
namespace openPMD
{
    /**
     * Helper class for ADIOS plugin to load fields from parallel ADIOS BP
     * files.
     */
    class RestartFieldLoader
    {
    public:
        template< class Data >
        static void
        loadField( Data & field,
            const uint32_t numComponents,
            std::string objectName,
            ThreadParams * params )
        {
            log< picLog::INPUT_OUTPUT >( "Begin loading field '%1%'" ) %
                objectName;

            const std::string name_lookup_tpl[] = { "x", "y", "z", "w" };
            const DataSpace< simDim > field_guard =
                field.getGridLayout().getGuard();

            const pmacc::Selection< simDim > & localDomain =
                Environment< simDim >::get().SubGrid().getLocalDomain();

            field.getHostBuffer().setValue( float3_X::create( 0.0 ) );

            DataSpace< simDim > domain_offset = localDomain.offset;

            DataSpace< simDim > local_domain_size =
                params->window.localDimensions.size;
            ::openPMD::Series & series = *params->openPMDSeries;
            ::openPMD::Container<::openPMD::Mesh > & meshes =
                series.iterations[ params->currentStep ].meshes;

            auto destBox = field.getHostBuffer().getDataBox();
            for( uint32_t n = 0; n < numComponents; ++n )
            {
                // Read the subdomain which belongs to our mpi position.
                // The total grid size must match the grid size of the stored
                // data.
                log< picLog::INPUT_OUTPUT >(
                    "openPMD: Read from domain: offset=%1% size=%2%" ) %
                    domain_offset % local_domain_size;
                ::openPMD::RecordComponent & rc = numComponents > 1
                    ? meshes[ objectName ][ name_lookup_tpl[ n ] ]
                    : meshes[ objectName ][::openPMD::RecordComponent::SCALAR ];

                log< picLog::INPUT_OUTPUT >(
                    "openPMD: Read from field '%1%'" ) %
                    objectName;

                auto ndim = rc.getDimensionality();
                ::openPMD::Offset start;
                ::openPMD::Extent count;
                start.reserve( ndim );
                count.reserve( ndim );
                for( int d = 0; d < ndim; ++d )
                {
                    /* \see adios_define_var: z,y,x in C-order */
                    start[ d ] = domain_offset.revert()[ d ];
                    count[ d ] = local_domain_size.revert()[ d ];
                }

                /* specify what we want to read, but start reading at below at
                 * `adios_perform_reads` */
                log< picLog::INPUT_OUTPUT >(
                    "openPMD: Allocate %1% elements" ) %
                    local_domain_size.productOfComponents();

                /// \todo float_X should be some kind of gridBuffer's
                /// GetComponentsType<ValueType>::type
                /* magic parameters (0, 1): `from_step` (not used in streams),
                 * `nsteps` to read (must be 1 for stream) */
                // log<picLog::INPUT_OUTPUT > ("ADIOS: Schedule read from field
                // (%1%, %2%, %3%, %4%)") %
                //                             params->fp % fSel %
                //                             datasetName.str() %
                //                             (void*)field_container;

                // avoid deadlock between not finished pmacc tasks and mpi calls
                // in adios
                __getTransactionEvent().waitForFinished();

                std::shared_ptr< float_X > field_container =
                    rc.loadChunk< float_X >( start, count );
                series.flush();

                /* start a blocking read of all scheduled variables */

                int elementCount =
                    params->window.localDimensions.size.productOfComponents();

#pragma omp parallel for
                for( int linearId = 0; linearId < elementCount; ++linearId )
                {
                    /* calculate index inside the moving window domain which is
                     * located on the local grid*/
                    DataSpace< simDim > destIdx =
                        DataSpaceOperations< simDim >::map(
                            params->window.localDimensions.size, linearId );
                    /* jump over guard and local sliding window offset*/
                    destIdx += field_guard + params->localWindowToDomainOffset;

                    destBox( destIdx )[ n ] = field_container.get()[ linearId ];
                }
            }

            field.hostToDevice();

            __getTransactionEvent().waitForFinished();

            log< picLog::INPUT_OUTPUT >(
                "openPMD: Read from domain: offset=%1% size=%2%" ) %
                domain_offset % local_domain_size;
            log< picLog::INPUT_OUTPUT >(
                "openPMD: Finished loading field '%1%'" ) %
                objectName;
        }
    };

    /**
     * Helper class for openPMDWriter (forEach operator) to load a field from
     * ADIOS
     *
     * @tparam FieldType field class to load
     */
    template< typename FieldType >
    struct LoadFields
    {
    public:
        HDINLINE void
        operator()( ThreadParams * params )
        {
#ifndef __CUDA_ARCH__
            DataConnector & dc = Environment<>::get().DataConnector();
            ThreadParams * tp = params;

            /* load field without copying data to host */
            auto field = dc.get< FieldType >( FieldType::getName(), true );

            /* load from ADIOS */
            RestartFieldLoader::loadField( field->getGridBuffer(),
                ( uint32_t )FieldType::numComponents,
                FieldType::getName(),
                tp );

            dc.releaseData( FieldType::getName() );
#endif
        }
    };

    using namespace pmacc;

} /* namespace openPMD */
} /* namespace picongpu */
