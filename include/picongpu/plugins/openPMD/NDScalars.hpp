/* Copyright 2016-2018 Alexander Grund
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

#pragma once

#include <pmacc/Environment.hpp>
#include <pmacc/types.hpp>
#include <stdexcept>
#include "picongpu/plugins/openPMD/openPMDWriter.def"

namespace picongpu
{
namespace openPMD
{
template < typename F, typename... Params >
void withAlteredMeshesPath(
    ::openPMD::Series & series,
    std::string const & path,
    F f,
    Params &&... params )
{
    series.flush( );
    std::string oldPath = series.meshesPath( );
    series.setMeshesPath( path );
    f( params... );
    series.flush( );
    series.setMeshesPath( oldPath );
}

/** Functor for writing N-dimensional scalar fields with N=simDim
 * In the current implementation each process (of the ND grid of processes)
 * writes 1 scalar value Optionally the processes can also write an attribute
 * for this dataset by using a non-empty attrName
 *
 * @tparam T_Scalar    Type of the scalar value to write
 * @tparam T_Attribute Type of the attribute (can be omitted if attribute is not
 * written, defaults to uint64_t)
 */
template < typename T_Scalar, typename T_Attribute = uint64_t >
struct WriteNDScalars
{
    WriteNDScalars(
        const std::string & baseName,
        const std::string & group,
        const std::string & dataset,
        const std::string & attrName = "" )
    : baseName( baseName )
    , group( group )
    , dataset( dataset )
    , attrName( attrName )
    {
    }

    /** Prepare the write operation:
     *  Define ADIOS variable, increase params.openPMDGroupSize and write
     * attribute (if attrName is non-empty)
     *
     *  Must be called before executing the functor
     */
    void prepare(
        ThreadParams & params, T_Attribute attribute = T_Attribute( ) )
    {
        auto f = [&params, &attribute, this]( ) {
            auto name = baseName + "/" + group + "/" + dataset;
            const auto openPMDScalarType =
                ::openPMD::determineDatatype< T_Scalar >( );
            typedef pmacc::math::UInt64< simDim > Dimensions;

            log< picLog::INPUT_OUTPUT >(
                "openPMD: prepare write %1%D scalars: %2%" ) %
                simDim % name;

            params.openPMDGroupSize += sizeof( T_Scalar );
            if ( !attrName.empty( ) )
                params.openPMDGroupSize += sizeof( T_Attribute );

            // Size over all processes
            Dimensions globalDomainSize = Dimensions::create( 1 );
            // Offset for this process
            Dimensions localDomainOffset = Dimensions::create( 0 );

            for ( uint32_t d = 0; d < simDim; ++d )
            {
                globalDomainSize[d] = Environment< simDim >::get( )
                                          .GridController( )
                                          .getGpuNodes( )[d];
                localDomainOffset[d] = Environment< simDim >::get( )
                                           .GridController( )
                                           .getPosition( )[d];
            }

            ::openPMD::Series & series = *params.openPMDSeries;
            ::openPMD::MeshRecordComponent & mrc =
                series.iterations[params.currentStep].meshes[group][dataset];

            preparedDataset = std::unique_ptr< WithWindow< RecordComponent > >{
                new WithWindow< RecordComponent >{prepareDataset< simDim >(
                    mrc,
                    openPMDScalarType,
                    globalDomainSize,
                    Dimensions::create( 1 ),
                    localDomainOffset,
                    true,
                    params.adiosCompression )}};

            if ( !attrName.empty( ) )
            {
                log< picLog::INPUT_OUTPUT >(
                    "openPMD: write attribute %1% of %2%D scalars: %3%" ) %
                    attrName % simDim % name;

                mrc.setAttribute( attrName, attribute );
            }
        };

        withAlteredMeshesPath( *params.openPMDSeries, baseName, f );
    }

    void operator( )( ThreadParams & params, T_Scalar value )
    {
        auto name = baseName + "/" + group + "/" + dataset;
        log< picLog::INPUT_OUTPUT >( "openPMD: write %1%D scalars: %2%" ) %
            simDim % name;

        withAlteredMeshesPath(
            *params.openPMDSeries, baseName, [&value, this]( ) {
                preparedDataset->m_data.storeChunk(
                    std::make_shared< T_Scalar >( value ),
                    preparedDataset->m_offset,
                    preparedDataset->m_extent );
            } );
    }

private:
    const std::string baseName, group, dataset, attrName;
    int64_t varId;
    std::unique_ptr< WithWindow< RecordComponent > > preparedDataset;
};

/** Functor for reading ND scalar fields with N=simDim
 * In the current implementation each process (of the ND grid of processes)
 * reads 1 scalar value Optionally the processes can also read an attribute for
 * this dataset by using a non-empty attrName
 *
 * @tparam T_Scalar    Type of the scalar value to read
 * @tparam T_Attribute Type of the attribute (can be omitted if attribute is not
 * read, defaults to uint64_t)
 */
template < typename T_Scalar, typename T_Attribute = uint64_t >
struct ReadNDScalars
{
    /** Read the skalar field and optionally the attribute into the values
     * referenced by the pointers */
    void operator( )(
        ThreadParams & params,
        const std::string & baseName,
        const std::string & group,
        const std::string & dataset,
        T_Scalar * value,
        const std::string & attrName = "",
        T_Attribute * attribute = nullptr )
    {
        auto name = baseName + "/" + group + "/" + dataset;
        log< picLog::INPUT_OUTPUT >( "openPMD: read %1%D scalars: %2%" ) %
            simDim % name;

        auto f = [&params,
                  &baseName,
                  &group,
                  &dataset,
                  value,
                  &attrName,
                  attribute,
                  &name]( ) {
            auto datasetName = baseName + "/" + group + "/" + dataset;
            ::openPMD::Series & series = *params.openPMDSeries;
            ::openPMD::RecordComponent & rc =
                series.iterations[params.currentStep].meshes[group][dataset];
            auto ndim = rc.getDimensionality( );
            if ( ndim != simDim )
            {
                throw std::runtime_error(
                    std::string( "Invalid dimensionality for " ) + name );
            }

            DataSpace< simDim > gridPos =
                Environment< simDim >::get( ).GridController( ).getPosition( );
            ::openPMD::Offset start;  //[varInfo->ndim];
            ::openPMD::Extent count;  //[varInfo->ndim];
            start.reserve( ndim );
            count.reserve( ndim );
            for ( int d = 0; d < ndim; ++d )
            {
                /* \see adios_define_var: z,y,x in C-order */
                start[d] = gridPos.revert( )[d];
                count[d] = 1;
            }

            __getTransactionEvent( ).waitForFinished( );

            log< picLog::INPUT_OUTPUT >(
                "openPMD: Schedule read skalar %1%)" ) %
                datasetName;

            std::shared_ptr< T_Scalar > readValue =
                rc.loadChunk< T_Scalar >( start, count );

            series.flush( );

            *value = *readValue;

            if ( !attrName.empty( ) )
            {
                log< picLog::INPUT_OUTPUT >(
                    "openPMD: read attribute %1% for scalars: %2%" ) %
                    attrName % name;
                *attribute = rc.getAttribute( name ).get< T_Attribute >( );
            }
        };

        withAlteredMeshesPath( *params.openPMDSeries, baseName, f );
    }
};

}  // namespace openPMD
}  // namespace picongpu
