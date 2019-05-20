/* Copyright 2013-2019 Axel Huebl, Felix Schmitt, Rene Widera, Franz Poeschel
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
#include "picongpu/simulation_defines.hpp"
#include "picongpu/traits/PICToOpenPMD.hpp"

#include <pmacc/assert.hpp>
#include <pmacc/traits/GetComponentsType.hpp>
#include <pmacc/traits/GetNComponents.hpp>
#include <pmacc/traits/Resolve.hpp>

#include <openPMD/openPMD.hpp>

#include <memory>

namespace picongpu
{
namespace openPMD
{
    using namespace pmacc;

    /** Load attribute of a species from ADIOS checkpoint file
     *
     * @tparam T_Identifier identifier of species attribute
     */
    template< typename T_Identifier >
    struct LoadParticleAttributesFromADIOS
    {
        /** read attributes from ADIOS file
         *
         * @param params thread params with ADIOS_FILE, ...
         * @param frame frame with all particles
         * @param particlePath path to the group in the ADIOS file
         * @param particlesOffset read offset in the attribute array
         * @param elements number of elements which should be read the attribute
         * array
         */
        template< typename FrameType >
        HINLINE void
        operator()( ThreadParams * params,
            FrameType & frame,
            ::openPMD::ParticleSpecies particleSpecies,
            const uint64_t particlesOffset,
            const uint64_t elements )
        {
            std::string particlePath = "";
            typedef T_Identifier Identifier;
            typedef typename pmacc::traits::Resolve< Identifier >::type::type
                ValueType;
            const uint32_t components = GetNComponents< ValueType >::value;
            typedef typename GetComponentsType< ValueType >::type ComponentType;

            log< picLog::INPUT_OUTPUT >(
                "ADIOS: ( begin ) load species attribute: %1%" ) %
                Identifier::getName();

            const std::string name_lookup[] = { "x", "y", "z" };

            std::shared_ptr< ComponentType > loadBfr;
            if( elements > 0 )
            {
                loadBfr = std::shared_ptr< ComponentType >{
                    new ComponentType[ elements ],
                    []( ComponentType * ptr ) { delete[] ptr; }
                };
            }

            // dev assert!
            //        if( elements > 0 )
            //            PMACC_ASSERT(tmpArray);

            for( uint32_t n = 0; n < components; ++n )
            {
                OpenPMDName< T_Identifier > openPMDName;
                ::openPMD::Record & record = particleSpecies[ openPMDName() ];
                ::openPMD::RecordComponent & rc = components > 1
                    ? record[ name_lookup[ n ] ]
                    : record[::openPMD::RecordComponent::SCALAR ];

                ValueType * dataPtr =
                    frame.getIdentifier( Identifier() ).getPointer();
                // it's possible to aquire the local block with that call again
                // and the local elements to-be-read, but the block-ID must be
                // known (MPI rank?) ADIOS_CMD(adios_inq_var_blockinfo(
                // params->fp, varInfo ));


                if( elements > 0 )
                {
                    // avoid deadlock between not finished pmacc tasks and mpi
                    // calls in adios
                    __getTransactionEvent().waitForFinished();
                    rc.loadChunk< ComponentType >( loadBfr,
                        ::openPMD::Offset{ particlesOffset },
                        ::openPMD::Extent{ elements } );
                }

                /** start a blocking read of all scheduled variables
                 *  (this is collective call in many ADIOS methods) */
                params->openPMDSeries->flush();

                log< picLog::INPUT_OUTPUT >(
                    "openPMD:  Did read %1% local of %2% global elements for "
                    "%3%" ) %
                    elements % rc.getDimensionality() % openPMDName();

/* copy component from temporary array to array of structs */
#pragma omp parallel for
                for( size_t i = 0; i < elements; ++i )
                {
                    ComponentType & ref = reinterpret_cast< ComponentType * >(
                        dataPtr )[ i * components + n ];
                    ref = loadBfr.get()[ i ];
                }
            }

            log< picLog::INPUT_OUTPUT >(
                "openPMD:  ( end ) load species attribute: %1%" ) %
                Identifier::getName();
        }
    };

} /* namespace openPMD */

} /* namespace picongpu */
