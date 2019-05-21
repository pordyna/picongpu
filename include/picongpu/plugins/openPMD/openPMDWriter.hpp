/* Copyright 2014-2019 Axel Huebl, Felix Schmitt, Heiko Burau, Rene Widera,
 *                     Benjamin Worpitz, Alexander Grund, Franz Poeschel
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

#include "picongpu/fields/FieldB.hpp"
#include "picongpu/fields/FieldE.hpp"
#include "picongpu/fields/FieldJ.hpp"
#include "picongpu/fields/FieldTmp.hpp"
#include "picongpu/particles/filter/filter.hpp"
#include "picongpu/particles/traits/SpeciesEligibleForSolver.hpp"
#include "picongpu/plugins/misc/SpeciesFilter.hpp"
#include "picongpu/plugins/misc/misc.hpp"
#include "picongpu/plugins/multi/Option.hpp"
#include "picongpu/plugins/openPMD/openPMDWriter.def"
#include "picongpu/simulationControl/MovingWindow.hpp"
#include "picongpu/simulation_defines.hpp"

#include <pmacc/assert.hpp>
#include <pmacc/dataManagement/DataConnector.hpp>
#include <pmacc/dimensions/GridLayout.hpp>
#include <pmacc/mappings/simulation/GridController.hpp>
#include <pmacc/mappings/simulation/SubGrid.hpp>
#include <pmacc/math/Vector.hpp>
#include <pmacc/particles/IdProvider.def>
#include <pmacc/particles/frame_types.hpp>
#include <pmacc/particles/operations/CountParticles.hpp>
#include <pmacc/pluginSystem/PluginConnector.hpp>
#include <pmacc/static_assert.hpp>
#if( PMACC_CUDA_ENABLED == 1 )
#    include <pmacc/particles/memory/buffers/MallocMCBuffer.hpp>
#endif
#include "picongpu/plugins/misc/SpeciesFilter.hpp"
#include "picongpu/plugins/openPMD/NDScalars.hpp"
#include "picongpu/plugins/openPMD/WriteMeta.hpp"
#include "picongpu/plugins/openPMD/WriteSpecies.hpp"
#include "picongpu/plugins/openPMD/openPMDCountParticles.hpp"
#include "picongpu/plugins/openPMD/restart/LoadSpecies.hpp"
#include "picongpu/plugins/openPMD/restart/RestartFieldLoader.hpp"
#include "picongpu/plugins/output/IIOBackend.hpp"

#include <pmacc/traits/Limits.hpp>

#include <boost/filesystem.hpp>
#include <boost/mpl/at.hpp>
#include <boost/mpl/begin_end.hpp>
#include <boost/mpl/find.hpp>
#include <boost/mpl/pair.hpp>
#include <boost/mpl/size.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/type_traits.hpp>
#include <boost/type_traits/is_same.hpp>

#include <openPMD/openPMD.hpp>

#if !defined( _WIN32 )
#    include <unistd.h>
#endif

#include <algorithm>
#include <list>
#include <pthread.h>
#include <sstream>
#include <string>
#include <vector>


namespace picongpu
{
namespace openPMD
{
    using namespace pmacc;


    namespace po = boost::program_options;

    template< unsigned DIM >
    ::openPMD::RecordComponent &
    initDataset(
        ::openPMD::RecordComponent & recordComponent,
        ::openPMD::Datatype datatype,
        pmacc::math::UInt64< DIM > const & globalDimensions,
        bool compression,
        std::string const & compressionMethod )
    {
        std::vector< uint64_t > v = asStandardVector< DIM >( globalDimensions );
        ::openPMD::Dataset dataset{ datatype, std::move( v ) };
        if( compression )
        {
            dataset.compression = compressionMethod;
        }
        recordComponent.resetDataset( std::move( dataset ) );
        return recordComponent;
    }

    template< unsigned DIM >
    WithWindow<::openPMD::RecordComponent >
    prepareDataset( ::openPMD::RecordComponent & recordComponent,
        ::openPMD::Datatype datatype,
        pmacc::math::UInt64< DIM > const & globalDimensions,
        pmacc::math::UInt64< DIM > const & localDimensions,
        pmacc::math::UInt64< DIM > const & offset,
        bool compression,
        std::string const & compressionMethod )
    {
        std::vector< uint64_t > v = asStandardVector< DIM >( globalDimensions );
        ::openPMD::Dataset dataset{ datatype, std::move( v ) };
        if( compression )
        {
            dataset.compression = compressionMethod;
        }
        recordComponent.resetDataset( std::move( dataset ) );
        return WithWindow<::openPMD::RecordComponent >::init< DIM >(
            recordComponent, offset, localDimensions );
    }

    template< unsigned DIM >
    void
    pushDataset( ::openPMD::RecordComponent & recordComponent,
        ::openPMD::Datatype datatype,
        pmacc::math::UInt64< DIM > const & globalDimensions,
        pmacc::math::UInt64< DIM > const & localDimensions,
        pmacc::math::UInt64< DIM > const & offset,
        bool compression,
        std::string const & compressionMethod,
        std::list< WithWindow<::openPMD::RecordComponent > > & push_to )
    {
        if( std::all_of( &globalDimensions[ 0 ],
                &globalDimensions[ 0 ] + DIM,
                []( uint64_t i ) { return i > 0; } ) )
        {
            push_to.push_back( prepareDataset< DIM >( recordComponent,
                datatype,
                globalDimensions,
                localDimensions,
                offset,
                compression,
                compressionMethod ) );
        }
    }

    template< unsigned DIM, typename T >
    std::vector< T >
    asStandardVector( pmacc::math::Vector< T, DIM > const & v )
    {
        std::vector< T > res;
        res.reserve( DIM );
        for( unsigned i = 0; i < DIM; ++i )
        {
            res.push_back( v[ i ] );
        }
        return res;
    }

    template< typename T >
    template< unsigned DIM >
    WithWindow< T >
    WithWindow< T >::init( T & data,
        pmacc::math::UInt64< DIM > offset,
        pmacc::math::UInt64< DIM > extent )
    {
        return WithWindow< T >{ data,
            asStandardVector< DIM >( std::move( offset ) ),
            asStandardVector< DIM >( std::move( extent ) ) };
    }

    ::openPMD::Series &
    ThreadParams::openSeries( ::openPMD::AccessType at )
    {
        if( !openPMDSeries )
        {
            log< picLog::INPUT_OUTPUT >( "openPMD: open file: %1%" ) % fileName;
            openPMDSeries = std::unique_ptr<::openPMD::Series >(
                new ::openPMD::Series( fileName, at, communicator ) );
            if( at == ::openPMD::AccessType::CREATE )
            {
                openPMDSeries->setMeshesPath( MESHES_PATH );
            }
            log< picLog::INPUT_OUTPUT >(
                "openPMD: successfully opened file: %1%" ) %
                fileName;
            return *openPMDSeries;
        }
        else
        {
            throw std::runtime_error(
                "openPMD: Tried opening a Series while old Series was still "
                "active" );
        }
    }

    void
    ThreadParams::closeSeries()
    {
        if( openPMDSeries )
        {
            log< picLog::INPUT_OUTPUT >( "openPMD: close file: %1%" ) %
                fileName;
            openPMDSeries.reset();
            MPI_Barrier( this->communicator );
            log< picLog::INPUT_OUTPUT >(
                "openPMD: successfully closed file: %1%" ) %
                fileName;
        }
        else
        {
            throw std::runtime_error(
                "openPMD: Tried closing a Series that was not active" );
        }
    }


    /** Writes simulation data to openPMD.
     *
     * Implements the IIOBackend interface.
     */
    class openPMDWriter : public IIOBackend
    {
    public:
        struct Help : public plugins::multi::IHelp
        {
            /** creates a instance of ISlave
             *
             * @param help plugin defined help
             * @param id index of the plugin, range: [0;help->getNumPlugins())
             */
            std::shared_ptr< ISlave >
            create( std::shared_ptr< IHelp > & help,
                size_t const id,
                MappingDesc * cellDescription )
            {
                return std::shared_ptr< ISlave >(
                    new openPMDWriter( help, id, cellDescription ) );
            }

            plugins::multi::Option< std::string > notifyPeriod = { "period",
                "enable openPMD IO [for each n-th step]" };

            plugins::multi::Option< std::string > source = { "source",
                "data sources: ",
                "species_all, fields_all" };

            plugins::multi::Option< std::string > fileName = { "file",
                "openPMD file basename" };

            plugins::multi::Option< std::string > fileNameExtension = { "ext",
                "openPMD filename extension (this controls the"
                "backend picked by the openPMD API)",
                "bp" };

            std::vector< std::string > allowedDataSources = { "species_all",
                "fields_all" };

            plugins::multi::Option< uint32_t > numAggregators = { "aggregators",
                "Number of aggregators [0 == number of MPI processes] "
                "(currently "
                "controlled by backend-specific environment variables)",
                0u };

            plugins::multi::Option< uint32_t > numOSTs = { "ost",
                "Number of OST (currently controlled by backend-specific "
                "environment variables)",
                1u };

            plugins::multi::Option< uint32_t > disableMeta = { "disable-meta",
                "Disable online gather and write of a global meta file, can be "
                "time consuming (use `bpmeta` post-mortem) (currently "
                "controlled "
                "by backend-specific environment variables)",
                0u };

            /* select MPI method, #OSTs and #aggregators */
            plugins::multi::Option< std::string > transportParams = {
                "transport-params",
                "additional transport parameters, see ADIOS manual chapter "
                "6.1.5, "
                "e.g., 'random_offset=1;stripe_count=4' (currently controlled "
                "by "
                "backend-specific environment variables)",
                ""
            };

            plugins::multi::Option< std::string > compression = { "compression",
                "Backend-specific openPMD compression method, e.g., zlib (see "
                "`adios_config -m` for help)",
                "none" };

            /** defines if the plugin must register itself to the PMacc plugin
             * system
             *
             * true = the plugin is registering it self
             * false = the plugin is not registering itself (plugin is
             * controlled by another class)
             */
            bool selfRegister = false;

            template< typename T_TupleVector >
            struct CreateSpeciesFilter
            {
                using type = plugins::misc::SpeciesFilter<
                    typename pmacc::math::CT::At< T_TupleVector,
                        bmpl::int_< 0 > >::type,
                    typename pmacc::math::CT::At< T_TupleVector,
                        bmpl::int_< 1 > >::type >;
            };

            using AllParticlesTimesAllFilters =
                typename AllCombinations< bmpl::vector< FileOutputParticles,
                    particles::filter::AllParticleFilters > >::type;

            using AllSpeciesFilter =
                typename bmpl::transform< AllParticlesTimesAllFilters,
                    CreateSpeciesFilter< bmpl::_1 > >::type;

            using AllEligibleSpeciesSources = typename bmpl::copy_if<
                AllSpeciesFilter,
                plugins::misc::speciesFilter::IsEligible< bmpl::_1 > >::type;

            using AllFieldSources = FileOutputFields;

            ///! method used by plugin controller to get --help description
            void
            registerHelp( boost::program_options::options_description & desc,
                std::string const & masterPrefix = std::string{} )
            {
                ForEach
                    < AllEligibleSpeciesSources,
                        plugins::misc::AppendName<
                            bmpl::_1 > > getEligibleDataSourceNames;
                getEligibleDataSourceNames( allowedDataSources );

                ForEach
                    < AllFieldSources,
                        plugins::misc::AppendName<
                            bmpl::_1 > > appendFieldSourceNames;
                appendFieldSourceNames( allowedDataSources );

                // string list with all possible particle sources
                std::string concatenatedSourceNames =
                    plugins::misc::concatenateToString(
                        allowedDataSources, ", " );

                notifyPeriod.registerHelp( desc, masterPrefix + prefix );
                source.registerHelp( desc,
                    masterPrefix + prefix,
                    std::string( "[" ) + concatenatedSourceNames + "]" );

                expandHelp( desc, "" );
                selfRegister = true;
            }

            void
            expandHelp( boost::program_options::options_description & desc,
                std::string const & masterPrefix = std::string{} )
            {
                numAggregators.registerHelp( desc, masterPrefix + prefix );
                numOSTs.registerHelp( desc, masterPrefix + prefix );
                disableMeta.registerHelp( desc, masterPrefix + prefix );
                transportParams.registerHelp( desc, masterPrefix + prefix );
                compression.registerHelp( desc, masterPrefix + prefix );
                //fileName.registerHelp( desc, masterPrefix + prefix );
                //fileNameExtension.registerHelp( desc, masterPrefix + prefix );
                fileName.registerHelp( desc, masterPrefix + prefix );
                fileNameExtension.registerHelp( desc, masterPrefix + prefix );
            }

            void
            validateOptions()
            {
                if( selfRegister )
                {
                    if( notifyPeriod.empty() || fileName.empty() )
                        throw std::runtime_error( name +
                            ": parameter period and file must be defined" );

                    // check if user passed data source names are valid
                    for( auto const & dataSourceNames : source )
                    {
                        auto vectorOfDataSourceNames =
                            plugins::misc::splitString(
                                plugins::misc::removeSpaces(
                                    dataSourceNames ) );

                        for( auto const & f : vectorOfDataSourceNames )
                        {
                            if( !plugins::misc::containsObject(
                                    allowedDataSources, f ) )
                            {
                                throw std::runtime_error( name +
                                    ": unknown data source '" + f + "'" );
                            }
                        }
                    }
                }
            }

            size_t
            getNumPlugins() const
            {
                if( selfRegister )
                    return notifyPeriod.size();
                else
                    return 1;
            }

            std::string
            getDescription() const
            {
                return description;
            }

            std::string
            getOptionPrefix() const
            {
                return prefix;
            }

            std::string
            getName() const
            {
                return name;
            }

            std::string const name = "openPMDWriter";
            //! short description of the plugin
            std::string const description = "dump simulation data with openPMD";
            //! prefix used for command line arguments
            std::string const prefix = "openPMD";
        };

        //! must be implemented by the user
        static std::shared_ptr< plugins::multi::IHelp >
        getHelp()
        {
            return std::shared_ptr< plugins::multi::IHelp >( new Help{} );
        }

    private:
        template< typename UnitType >
        static std::vector< float_64 >
        createUnit( UnitType unit, uint32_t numComponents )
        {
            std::vector< float_64 > tmp( numComponents );
            for( uint32_t i = 0; i < numComponents; ++i )
                tmp[ i ] = unit[ i ];
            return tmp;
        }

        /**
         * Write calculated fields to openPMD.
         */
        template< typename T >
        struct GetFields
        {
        private:
            typedef typename T::ValueType ValueType;
            typedef typename GetComponentsType< ValueType >::type ComponentType;
            typedef typename T::UnitValueType UnitType;

        public:
            static std::vector< float_64 >
            getUnit()
            {
                UnitType unit = T::getUnit();
                return createUnit( unit, T::numComponents );
            }

            HDINLINE void
            operator()( ThreadParams * params )
            {
#ifndef __CUDA_ARCH__
                DataConnector & dc =
                    Environment< simDim >::get().DataConnector();

                auto field = dc.get< T >( T::getName() );
                params->gridLayout = field->getGridLayout();

                const traits::FieldPosition<
                    typename fields::Solver::NummericalCellType,
                    T >
                    fieldPos;

                std::vector< std::vector< float_X > > inCellPosition;
                for( uint32_t n = 0; n < T::numComponents; ++n )
                {
                    std::vector< float_X > inCellPositonComponent;
                    for( uint32_t d = 0; d < simDim; ++d )
                        inCellPositonComponent.push_back(
                            fieldPos()[ n ][ d ] );
                    inCellPosition.push_back( inCellPositonComponent );
                }

                /** \todo check if always correct at this point, depends on
                 * solver implementation */
                const float_X timeOffset = 0.0;

                openPMDWriter::template writeField< ComponentType >( params,
                    sizeof( ComponentType ),
                    ::openPMD::determineDatatype< ComponentType >(),
                    GetNComponents< ValueType >::value,
                    T::getName(),
                    field->getHostDataBox().getPointer(),
                    getUnit(),
                    T::getUnitDimension(),
                    std::move( inCellPosition ),
                    timeOffset );

                dc.releaseData( T::getName() );
#endif
            }
        };

        /** Calculate FieldTmp with given solver and particle species
         * and write them to openPMD.
         *
         * FieldTmp is calculated on device and then dumped to openPMD.
         */
        template< typename Solver, typename Species >
        struct GetFields< FieldTmpOperation< Solver, Species > >
        {
            /*
             * This is only a wrapper function to allow disable nvcc warnings.
             * Warning: calling a __host__ function from __host__ __device__
             * function.
             * Use of PMACC_NO_NVCC_HDWARNING is not possible if we call a
             * virtual method inside of the method were we disable the warnings.
             * Therefore we create this method and call a new method were we can
             * call virtual functions.
             */
            PMACC_NO_NVCC_HDWARNING
            HDINLINE void
            operator()( ThreadParams * tparam )
            {
                this->operator_impl( tparam );
            }

        private:
            typedef typename FieldTmp::UnitValueType UnitType;
            typedef typename FieldTmp::ValueType ValueType;
            typedef typename GetComponentsType< ValueType >::type ComponentType;

            /** Get the unit for the result from the solver*/
            static std::vector< float_64 >
            getUnit()
            {
                UnitType unit = FieldTmp::getUnit< Solver >();
                const uint32_t components = GetNComponents< ValueType >::value;
                return createUnit( unit, components );
            }

            /** Create a name for the openPMD identifier.
             */
            static std::string
            getName()
            {
                return FieldTmpOperation< Solver, Species >::getName();
            }

            HINLINE void
            operator_impl( ThreadParams * params )
            {
                DataConnector & dc = Environment<>::get().DataConnector();

                /*## update field ##*/

                /*load FieldTmp without copy data to host*/
                PMACC_CASSERT_MSG(
                    _please_allocate_at_least_one_FieldTmp_in_memory_param,
                    fieldTmpNumSlots > 0 );
                auto fieldTmp =
                    dc.get< FieldTmp >( FieldTmp::getUniqueId( 0 ), true );
                /*load particle without copy particle data to host*/
                auto speciesTmp =
                    dc.get< Species >( Species::FrameType::getName(), true );

                fieldTmp->getGridBuffer().getDeviceBuffer().setValue(
                    ValueType::create( 0.0 ) );
                /*run algorithm*/
                fieldTmp->template computeValue< CORE + BORDER, Solver >(
                    *speciesTmp, params->currentStep );

                EventTask fieldTmpEvent =
                    fieldTmp->asyncCommunication( __getTransactionEvent() );
                __setTransactionEvent( fieldTmpEvent );
                /* copy data to host that we can write same to disk*/
                fieldTmp->getGridBuffer().deviceToHost();
                dc.releaseData( Species::FrameType::getName() );
                /*## finish update field ##*/

                const uint32_t components = GetNComponents< ValueType >::value;

                /*wrap in a one-component vector for writeField API*/
                const traits::FieldPosition<
                    typename fields::Solver::NummericalCellType,
                    FieldTmp >
                    fieldPos;

                std::vector< std::vector< float_X > > inCellPosition;
                std::vector< float_X > inCellPositonComponent;
                for( uint32_t d = 0; d < simDim; ++d )
                    inCellPositonComponent.push_back( fieldPos()[ 0 ][ d ] );
                inCellPosition.push_back( inCellPositonComponent );

                /** \todo check if always correct at this point, depends on
                 * solver implementation */
                const float_X timeOffset = 0.0;

                params->gridLayout = fieldTmp->getGridLayout();
                /*write data to openPMD Series*/
                openPMDWriter::template writeField< ComponentType >( params,
                    sizeof( ComponentType ),
                    ::openPMD::determineDatatype< ComponentType >(),
                    components,
                    getName(),
                    fieldTmp->getHostDataBox().getPointer(),
                    getUnit(),
                    FieldTmp::getUnitDimension< Solver >(),
                    std::move( inCellPosition ),
                    timeOffset );

                dc.releaseData( FieldTmp::getUniqueId( 0 ) );
            }
        };

    public:
        /** constructor
         *
         * @param help instance of the class Help
         * @param id index of this plugin instance within help
         * @param cellDescription PIConGPu cell description information for
         * kernel index mapping
         */
        openPMDWriter( std::shared_ptr< plugins::multi::IHelp > & help,
            size_t const id,
            MappingDesc * cellDescription ) :
            m_help( std::static_pointer_cast< Help >( help ) ),
            m_id( id ),
            m_cellDescription( cellDescription ),
            outputDirectory( "openPMD" ),
            lastSpeciesSyncStep( pmacc::traits::limits::Max< uint32_t >::value )
        {
            mThreadParams.compressionMethod = m_help->compression.get( id );

            GridController< simDim > & gc =
                Environment< simDim >::get().GridController();
            /* It is important that we never change the mpi_pos after this point
             * because we get problems with the restart.
             * Otherwise we do not know which gpu must load the ghost parts
             * around the sliding window.
             */
            mpi_pos = gc.getPosition();
            mpi_size = gc.getGpuNodes();

            if( m_help->selfRegister )
            {
                std::string notifyPeriod = m_help->notifyPeriod.get( id );
                /* only register for notify callback when .period is set on
                 * command line */
                if( !notifyPeriod.empty() )
                {
                    Environment<>::get()
                        .PluginConnector()
                        .setNotificationPeriod( this, notifyPeriod );

                    /** create notify directory */
                    Environment< simDim >::get()
                        .Filesystem()
                        .createDirectoryWithPermissions( outputDirectory );
                }
            }

            // avoid deadlock between not finished pmacc tasks and mpi blocking
            // collectives
            __getTransactionEvent().waitForFinished();
            mThreadParams.communicator = MPI_COMM_NULL;
            MPI_CHECK( MPI_Comm_dup( gc.getCommunicator().getMPIComm(),
                &( mThreadParams.communicator ) ) );

            /* TODO: select MPI method, #OSTs and #aggregators */
        }

        virtual ~openPMDWriter()
        {
            if( mThreadParams.communicator != MPI_COMM_NULL )
            {
                // avoid deadlock between not finished pmacc tasks and mpi
                // blocking collectives
                __getTransactionEvent().waitForFinished();
                MPI_CHECK_NO_EXCEPT(
                    MPI_Comm_free( &( mThreadParams.communicator ) ) );
            }
        }

        void
        notify( uint32_t currentStep )
        {
            // notify is only allowed if the plugin is not controlled by the
            // class Checkpoint
            assert( m_help->selfRegister );

            __getTransactionEvent().waitForFinished();

            std::string filename = m_help->fileName.get( m_id ) + "_%T." +
                m_help->fileNameExtension.get( m_id );

            /* if file name is relative, prepend with common directory */
            if( boost::filesystem::path( filename ).has_root_path() )
                mThreadParams.fileName = filename;
            else
                mThreadParams.fileName = outputDirectory + "/" + filename;

            /* window selection */
            mThreadParams.window =
                MovingWindow::getInstance().getWindow( currentStep );
            mThreadParams.isCheckpoint = false;
            dumpData( currentStep );
        }

        virtual void
        restart( uint32_t restartStep, std::string const & restartDirectory )
        {
            /* ISlave restart interface is not needed becase IIOBackend
             * restart interface is used
             */
        }

        virtual void
        checkpoint( uint32_t currentStep,
            std::string const & checkpointDirectory )
        {
            /* ISlave checkpoint interface is not needed becase IIOBackend
             * checkpoint interface is used
             */
        }

        void
        dumpCheckpoint( const uint32_t currentStep,
            const std::string & checkpointDirectory,
            const std::string & checkpointFilename )
        {
            // checkpointing is only allowed if the plugin is controlled by the
            // class Checkpoint
            assert( !m_help->selfRegister );

            __getTransactionEvent().waitForFinished();
            /* if file name is relative, prepend with common directory */
            if( boost::filesystem::path( checkpointFilename ).has_root_path() )
                mThreadParams.fileName = checkpointFilename;
            else
                mThreadParams.fileName =
                    checkpointDirectory + "/" + checkpointFilename;
            mThreadParams.fileName +=
                    "_%T." + m_help->fileNameExtension.get( m_id );

            mThreadParams.window =
                MovingWindow::getInstance().getDomainAsWindow( currentStep );
            mThreadParams.isCheckpoint = true;

            dumpData( currentStep );
        }

        void
        doRestart( const uint32_t restartStep,
            const std::string & restartDirectory,
            const std::string & constRestartFilename,
            const uint32_t restartChunkSize )
        {
            // restart is only allowed if the plugin is controlled by the class
            // Checkpoint
            assert( !m_help->selfRegister );

            /* if restartFilename is relative, prepend with restartDirectory */
            if( !boost::filesystem::path( constRestartFilename )
                     .has_root_path() )
            {
                mThreadParams.fileName = restartDirectory + std::string( "/" ) +
                    constRestartFilename;
            }
            else
            {
                mThreadParams.fileName = constRestartFilename;
            }

            mThreadParams.fileName +=
                    "_%T." + m_help->fileNameExtension.get( m_id );

            // mThreadParams.isCheckpoint = isCheckpoint;
            mThreadParams.currentStep = restartStep;
            mThreadParams.cellDescription = m_cellDescription;

            mThreadParams.openSeries( ::openPMD::AccessType::READ_ONLY );

            ::openPMD::Iteration & iteration =
                mThreadParams.openPMDSeries
                    ->iterations[ mThreadParams.currentStep ];

            /* load number of slides to initialize MovingWindow */
            log< picLog::INPUT_OUTPUT >(
                "openPMD: (begin) read attr (%1% available)" ) %
                iteration.numAttributes();


            uint32_t slides =
                iteration.getAttribute( "sim_slides" ).get< uint32_t >();
            log< picLog::INPUT_OUTPUT >(
                "openPMD: value of sim_slides = %1%" ) %
                slides;

            uint32_t lastStep =
                iteration.getAttribute( "iteration" ).get< uint32_t >();
            log< picLog::INPUT_OUTPUT >( "openPMD: value of iteration = %1%" ) %
                lastStep;

            PMACC_ASSERT( lastStep == restartStep );

            /* apply slides to set gpus to last/written configuration */
            log< picLog::INPUT_OUTPUT >(
                "openPMD: Setting slide count for moving window to %1%" ) %
                slides;
            MovingWindow::getInstance().setSlideCounter( slides, restartStep );

            /* re-distribute the local offsets in y-direction
             * this will work for restarts with moving window still enabled
             * and restarts that disable the moving window
             * \warning enabling the moving window from a checkpoint that
             *          had no moving window will not work
             */
            GridController< simDim > & gc =
                Environment< simDim >::get().GridController();
            gc.setStateAfterSlides( slides );

            /* set window for restart, complete global domain */
            mThreadParams.window =
                MovingWindow::getInstance().getDomainAsWindow( restartStep );
            mThreadParams.localWindowToDomainOffset =
                DataSpace< simDim >::create( 0 );

            /* load all fields */
            ForEach
                < FileCheckpointFields,
                    LoadFields< bmpl::_1 > > forEachLoadFields;
            forEachLoadFields( &mThreadParams );

            /* load all particles */
            ForEach
                < FileCheckpointParticles,
                    LoadSpecies< bmpl::_1 > > forEachLoadSpecies;
            forEachLoadSpecies( &mThreadParams, restartChunkSize );

            IdProvider< simDim >::State idProvState;
            ReadNDScalars< uint64_t, uint64_t >()( mThreadParams,
                "picongpu",
                "idProvider",
                "startId",
                &idProvState.startId,
                "maxNumProc",
                &idProvState.maxNumProc );
            ReadNDScalars< uint64_t >()( mThreadParams,
                "picongpu",
                "idProvider",
                "nextId",
                &idProvState.nextId );
            log< picLog::INPUT_OUTPUT >(
                "Setting next free id on current rank: %1%" ) %
                idProvState.nextId;
            IdProvider< simDim >::setState( idProvState );

            // avoid deadlock between not finished pmacc tasks and mpi calls in
            // openPMD
            __getTransactionEvent().waitForFinished();

            // Finalize the openPMD Series by calling its destructor
            mThreadParams.closeSeries();
        }

    private:
        void
        endWrite()
        {
            mThreadParams.fieldBuffer.reset();
        }

        void
        initWrite()
        {
            mThreadParams.fieldBuffer = std::shared_ptr< float_X >{
                new float_X[ mThreadParams.window.localDimensions.size
                                 .productOfComponents() ],
                []( float_X * ptr ) { delete[] ptr; }
            };
        }

        /**
         * Notification for dump or checkpoint received
         *
         * @param currentStep current simulation step
         */
        void
        dumpData( uint32_t currentStep )
        {
            // local offset + extent
            const pmacc::Selection< simDim > & localDomain =
                Environment< simDim >::get().SubGrid().getLocalDomain();
            mThreadParams.cellDescription = m_cellDescription;
            mThreadParams.currentStep = currentStep;

            for( uint32_t i = 0; i < simDim; ++i )
            {
                mThreadParams.localWindowToDomainOffset[ i ] = 0;
                if( mThreadParams.window.globalDimensions.offset[ i ] >
                    localDomain.offset[ i ] )
                {
                    mThreadParams.localWindowToDomainOffset[ i ] =
                        mThreadParams.window.globalDimensions.offset[ i ] -
                        localDomain.offset[ i ];
                }
            }

            /* copy species only one time per timestep to the host */
            if( lastSpeciesSyncStep != currentStep )
            {
                DataConnector & dc = Environment<>::get().DataConnector();

#if( PMACC_CUDA_ENABLED == 1 )
                /* synchronizes the MallocMCBuffer to the host side */
                dc.get< MallocMCBuffer< DeviceHeap > >(
                    MallocMCBuffer< DeviceHeap >::getName() );
#endif
                /* here we are copying all species to the host side since we
                 * can not say at this point if this time step will need all of
                 * them for sure (checkpoint) or just some user-defined species
                 * (dump)
                 */
                ForEach
                    < FileCheckpointParticles,
                        CopySpeciesToHost< bmpl::_1 > > copySpeciesToHost;
                copySpeciesToHost();
                lastSpeciesSyncStep = currentStep;
#if( PMACC_CUDA_ENABLED == 1 )
                dc.releaseData( MallocMCBuffer< DeviceHeap >::getName() );
#endif
            }

            initWrite();

            write( &mThreadParams, mpiTransportParams );

            endWrite();
        }

        static void writeFieldAttributes(
            ThreadParams* params,
            std::vector<float_64> const & unitDimension,
            float_X timeOffset,
            ::openPMD::Mesh& mesh
        ){
            static constexpr ::openPMD::UnitDimension openPMDUnitDimensions[ 7 ]
                = {
                ::openPMD::UnitDimension::L,
                ::openPMD::UnitDimension::M,
                ::openPMD::UnitDimension::T,
                ::openPMD::UnitDimension::I,
                ::openPMD::UnitDimension::theta,
                ::openPMD::UnitDimension::N,
                ::openPMD::UnitDimension::J
            };
            std::map<::openPMD::UnitDimension, double > unitMap;
            for( unsigned i = 0; i < 7; ++i )
            {
                unitMap[ openPMDUnitDimensions[ i ] ] = unitDimension[ i ];
            }

            mesh.setUnitDimension( unitMap );
            mesh.setTimeOffset< float_X >(timeOffset);
            mesh.setGeometry( ::openPMD::Mesh::Geometry::cartesian );
            mesh.setDataOrder( ::openPMD::Mesh::DataOrder::C );

            if( simDim == DIM2 )
            {
                std::vector< std::string > axisLabels = { "y",
                "x" }; // 2D: F[y][x]
                mesh.setAxisLabels( axisLabels );
            }
            if( simDim == DIM3 )
            {
                std::vector< std::string > axisLabels = {
                    "z", "y", "x"
                }; // 3D: F[z][y][x]
                mesh.setAxisLabels( axisLabels );
            }

            // cellSize is {x, y, z} but fields are F[z][y][x]
            std::vector< float_X > gridSpacing( simDim, 0.0 );
            for( uint32_t d = 0; d < simDim; ++d )
                gridSpacing.at( simDim - 1 - d ) = cellSize[ d ];

            mesh.setGridSpacing( gridSpacing );

            /* globalSlideOffset due to gpu slides between origin at time step 0
             * and origin at current time step
             * ATTENTION: splash offset are globalSlideOffset + picongpu offsets
             */
            DataSpace< simDim > globalSlideOffset;
            const pmacc::Selection< simDim > & localDomain =
            Environment< simDim >::get().SubGrid().getLocalDomain();
            const uint32_t numSlides =
            MovingWindow::getInstance().getSlideCounter(
            params->currentStep );
            globalSlideOffset.y() += numSlides * localDomain.size.y();

            // globalDimensions is {x, y, z} but fields are F[z][y][x]
            std::vector< float_64 > gridGlobalOffset( simDim, 0.0 );
            for( uint32_t d = 0; d < simDim; ++d )
                gridGlobalOffset.at( simDim - 1 - d ) =
                    float_64( cellSize[ d ] ) *
                    float_64( params->window.globalDimensions.offset[ d ] +
                    globalSlideOffset[ d ] );

            mesh.setGridGlobalOffset( std::move( gridGlobalOffset ) );
            mesh.setGridUnitSI( UNIT_LENGTH );
            mesh.setAttribute( "fieldSmoothing", "none" );
        }

        template< typename ComponentType >
        static void
        writeField( ThreadParams * params,
            const uint32_t sizePtrType,
            ::openPMD::Datatype openPMDType,
            const uint32_t nComponents,
            const std::string name,
            void * ptr,
            std::vector< float_64 > unit,
            std::vector< float_64 > unitDimension,
            std::vector< std::vector< float_X > > inCellPosition,
            float_X timeOffset )
        {
            const std::string name_lookup_tpl[] = { "x", "y", "z", "w" };

            /* parameter checking */
            PMACC_ASSERT( unit.size() == nComponents );
            PMACC_ASSERT( inCellPosition.size() == nComponents );
            for( uint32_t n = 0; n < nComponents; ++n )
                PMACC_ASSERT( inCellPosition.at( n ).size() == simDim );
            PMACC_ASSERT(
                unitDimension.size() == 7 ); // seven openPMD base units

            log< picLog::INPUT_OUTPUT >( "openPMD: write field: %1% %2% %3%" ) %
                name % nComponents % ptr;

            const bool fieldTypeCorrect(
                boost::is_same< ComponentType, float_X >::value );
            PMACC_CASSERT_MSG( Precision_mismatch_in_Field_Components__ADIOS,
                fieldTypeCorrect );

            ::openPMD::Iteration & iteration =
                params->openPMDSeries->iterations[ params->currentStep ];
            ::openPMD::Mesh & mesh = iteration.meshes[ name ];

            /* data to describe source buffer */
            GridLayout< simDim > field_layout = params->gridLayout;
            DataSpace< simDim > field_full = field_layout.getDataSpace();
            DataSpace< simDim > field_no_guard =
                params->window.localDimensions.size;
            DataSpace< simDim > field_guard =
                field_layout.getGuard() + params->localWindowToDomainOffset;

            /* write the actual field data */
            for( uint32_t d = 0; d < nComponents; d++ )
            {

                const size_t plane_full_size =
                    field_full[ 1 ] * field_full[ 0 ] * nComponents;
                const size_t plane_no_guard_size =
                    field_no_guard[ 1 ] * field_no_guard[ 0 ];

                /* copy strided data from source to temporary buffer
                 *
                 * \todo use d1Access as in
                 * `include/plugins/hdf5/writer/Field.hpp`
                 */
                const int maxZ = simDim == DIM3 ? field_no_guard[ 2 ] : 1;
                const int guardZ = simDim == DIM3 ? field_guard[ 2 ] : 0;
                for( int z = 0; z < maxZ; ++z )
                {
                    for( int y = 0; y < field_no_guard[ 1 ]; ++y )
                    {
                        const size_t base_index_src =
                            ( z + guardZ ) * plane_full_size +
                            ( y + field_guard[ 1 ] ) * field_full[ 0 ] *
                                nComponents;

                        const size_t base_index_dst =
                            z * plane_no_guard_size + y * field_no_guard[ 0 ];

                        for( int x = 0; x < field_no_guard[ 0 ]; ++x )
                        {
                            size_t index_src = base_index_src +
                                ( x + field_guard[ 0 ] ) * nComponents + d;
                            size_t index_dst = base_index_dst + x;

                            params->fieldBuffer.get()[ index_dst ] =
                                reinterpret_cast< float_X * >(
                                    ptr )[ index_src ];
                        }
                    }
                }

                /* Write the actual field data. The id is on the front of the
                 * list.
                 */

                ::openPMD::MeshRecordComponent & mrc = mesh[ nComponents > 1
                        ? name_lookup_tpl[ d ]
                        : ::openPMD::RecordComponent::SCALAR ];

                initDataset< simDim >(
                    mrc,
                    openPMDType,
                    params->fieldsGlobalSizeDims,
                    true,
                    params->compressionMethod
                ).template storeChunk< ComponentType >(
                    params->fieldBuffer,
                    asStandardVector< simDim >( params->fieldsOffsetDims),
                    asStandardVector< simDim >( params->fieldsSizeDims )
                );

                params->openPMDSeries->flush();
            }

            // set mesh attributes
            writeFieldAttributes(params, unitDimension, timeOffset, mesh);
        }

        template< typename T_ParticleFilter >
        struct CallCountParticles
        {
            void
            operator()(
                const std::vector< std::string > & vectorOfDataSourceNames,
                ThreadParams * params )
            {
                bool const containsDataSource = plugins::misc::containsObject(
                    vectorOfDataSourceNames, T_ParticleFilter::getName() );

                if( containsDataSource )
                {
                    openPMDCountParticles< T_ParticleFilter > count;
                    count( params );
                }
            }
        };

        template< typename T_ParticleFilter >
        struct CallWriteSpecies
        {
            template< typename Space >
            void
            operator()(
                const std::vector< std::string > & vectorOfDataSourceNames,
                ThreadParams * params,
                const Space domainOffset )
            {
                bool const containsDataSource = plugins::misc::containsObject(
                    vectorOfDataSourceNames, T_ParticleFilter::getName() );

                if( containsDataSource )
                {
                    WriteSpecies< T_ParticleFilter > writeSpecies;
                    writeSpecies( params, domainOffset );
                }
            }
        };

        template< typename T_Fields >
        struct CallGetFields
        {
            void
            operator()(
                const std::vector< std::string > & vectorOfDataSourceNames,
                ThreadParams * params )
            {
                bool const containsDataSource = plugins::misc::containsObject(
                    vectorOfDataSourceNames, T_Fields::getName() );

                if( containsDataSource )
                {
                    GetFields< T_Fields > getFields;
                    getFields( params );
                }
            }
        };

        void
        write( ThreadParams * threadParams, std::string mpiTransportParams )
        {
            /* y direction can be negative for first gpu */
            const pmacc::Selection< simDim > & localDomain =
                Environment< simDim >::get().SubGrid().getLocalDomain();
            DataSpace< simDim > particleOffset( localDomain.offset );
            particleOffset.y() -=
                threadParams->window.globalDimensions.offset.y();

            threadParams->fieldsOffsetDims =
                precisionCast< uint64_t >( localDomain.offset );

            /* write created variable values */
            for( uint32_t d = 0; d < simDim; ++d )
            {
                /* dimension 1 is y and is the direction of the moving window
                 * (if any) */
                if( 1 == d )
                {
                    uint64_t offset = std::max( 0,
                        localDomain.offset.y() -
                            threadParams->window.globalDimensions.offset.y() );
                    threadParams->fieldsOffsetDims[ d ] = offset;
                }

                threadParams->fieldsSizeDims[ d ] =
                    threadParams->window.localDimensions.size[ d ];
                threadParams->fieldsGlobalSizeDims[ d ] =
                    threadParams->window.globalDimensions.size[ d ];
            }

            std::vector< std::string > vectorOfDataSourceNames;
            if( m_help->selfRegister )
            {
                std::string dataSourceNames = m_help->source.get( m_id );

                vectorOfDataSourceNames = plugins::misc::splitString(
                    plugins::misc::removeSpaces( dataSourceNames ) );
            }

            bool dumpFields = plugins::misc::containsObject(
                vectorOfDataSourceNames, "fields_all" );

            log< picLog::INPUT_OUTPUT >( "openPMD: opening Series %1%" ) %
                threadParams->fileName;
            threadParams->openSeries( ::openPMD::AccessType::CREATE );

            /* collect size information for all attributes of all species and
             * define particle variables
             */
            threadParams->particleAttributes.clear();
            threadParams->speciesIndices.clear();

            bool dumpAllParticles = plugins::misc::containsObject(
                vectorOfDataSourceNames, "species_all" );

            log< picLog::INPUT_OUTPUT >(
                "openPMD: (begin) counting particles." );
            if( threadParams->isCheckpoint )
            {
                ForEach
                    < FileCheckpointParticles,
                        openPMDCountParticles< plugins::misc::UnfilteredSpecies<
                            bmpl::_1 > > > countParticles;
                countParticles( threadParams );
            }
            else
            {
                // count particles if data source "species_all" is selected
                if( dumpAllParticles )
                {
                    // move over all species defined in FileOutputParticles
                    ForEach
                        < FileOutputParticles,
                            openPMDCountParticles<
                                plugins::misc::UnfilteredSpecies<
                                    bmpl::_1 > > > countParticles;
                    countParticles( threadParams );
                }

                // move over all species data sources
                ForEach
                    < typename Help::AllEligibleSpeciesSources,
                        CallCountParticles< bmpl::_1 > >{}(
                        vectorOfDataSourceNames, threadParams );
            }
            log< picLog::INPUT_OUTPUT >(
                "openPMD: ( end ) counting particles." );

            auto idProviderState = IdProvider< simDim >::getState();
            WriteNDScalars< uint64_t, uint64_t > writeIdProviderStartId(
                "picongpu", "idProvider", "startId", "maxNumProc" );
            WriteNDScalars< uint64_t, uint64_t > writeIdProviderNextId(
                "picongpu", "idProvider", "nextId" );
            writeIdProviderStartId.prepare(
                *threadParams, idProviderState.maxNumProc );
            writeIdProviderNextId.prepare( *threadParams );

            /* attributes written here are pure meta data */
            WriteMeta writeMetaAttributes;
            writeMetaAttributes( threadParams );

            /* write fields */
            log< picLog::INPUT_OUTPUT >( "openPMD: (begin) writing fields." );
            if( threadParams->isCheckpoint )
            {
                ForEach
                    < FileCheckpointFields,
                        GetFields< bmpl::_1 > > forEachGetFields;
                forEachGetFields( threadParams );
            }
            else
            {
                if( dumpFields )
                {
                    ForEach
                        < FileOutputFields,
                            GetFields< bmpl::_1 > > forEachGetFields;
                    forEachGetFields( threadParams );
                }

                // move over all field data sources
                ForEach
                    < typename Help::AllFieldSources,
                        CallGetFields< bmpl::_1 > >{}(
                        vectorOfDataSourceNames, threadParams );
            }
            log< picLog::INPUT_OUTPUT >( "openPMD: ( end ) writing fields." );

            /* print all particle species */
            log< picLog::INPUT_OUTPUT >(
                "openPMD: (begin) writing particle species." );
            if( threadParams->isCheckpoint )
            {
                ForEach
                    < FileCheckpointParticles,
                        WriteSpecies< plugins::misc::SpeciesFilter<
                            bmpl::_1 > > > writeSpecies;
                writeSpecies( threadParams, particleOffset );
            }
            else
            {
                // dump data if data source "species_all" is selected
                if( dumpAllParticles )
                {
                    // move over all species defined in FileOutputParticles
                    ForEach
                        < FileOutputParticles,
                            WriteSpecies< plugins::misc::UnfilteredSpecies<
                                bmpl::_1 > > > writeSpecies;
                    writeSpecies( threadParams, particleOffset );
                }

                // move over all species data sources
                ForEach
                    < typename Help::AllEligibleSpeciesSources,
                        CallWriteSpecies< bmpl::_1 > >{}(
                        vectorOfDataSourceNames,
                        threadParams,
                        particleOffset );
            }
            log< picLog::INPUT_OUTPUT >(
                "openPMD: ( end ) writing particle species." );

            log< picLog::INPUT_OUTPUT >(
                "openPMD: Writing IdProvider state (StartId: %1%, NextId: %2%, "
                "maxNumProc: %3%)" ) %
                idProviderState.startId % idProviderState.nextId %
                idProviderState.maxNumProc;
            writeIdProviderStartId( *threadParams, idProviderState.startId );
            writeIdProviderNextId( *threadParams, idProviderState.nextId );

            // avoid deadlock between not finished pmacc tasks and mpi calls in
            // openPMD
            __getTransactionEvent().waitForFinished();

            /* close openPMD Series, most likely the actual write point */
            log< picLog::INPUT_OUTPUT >( "openPMD: closing series: %1%" ) %
                threadParams->fileName;
            threadParams->closeSeries();

            return;
        }

        ThreadParams mThreadParams;

        std::shared_ptr< Help > m_help;
        size_t m_id;

        MappingDesc * m_cellDescription;

        std::string outputDirectory;

        /* select MPI method, #OSTs and #aggregators */
        std::string mpiTransportParams;

        uint32_t lastSpeciesSyncStep;

        DataSpace< simDim > mpi_pos;
        DataSpace< simDim > mpi_size;
    };

} // namespace openPMD
} // namespace picongpu
