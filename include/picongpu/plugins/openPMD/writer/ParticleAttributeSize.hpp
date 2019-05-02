/* Copyright 2014-2019 Felix Schmitt, Axel Huebl, Franz Pöschel
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


namespace picongpu
{
namespace openPMD
{
    using namespace pmacc;


    /** collect size of a particle attribute
     *
     * @tparam T_Identifier identifier of a particle attribute
     */
    template< typename T_Identifier >
    struct ParticleAttributeSize
    {
        /** collect size of attribute
         *
         * @param params wrapped params
         * @param elements number of particles for this attribute
         */
        HINLINE void
        operator()( ThreadParams * params,
            const std::string speciesGroup,
            const uint64_t elements,
            const uint64_t globalElements,
            const uint64_t globalOffset )
        {
            typedef T_Identifier Identifier;
            typedef typename pmacc::traits::Resolve< Identifier >::type::type
                ValueType;
            const uint32_t components = GetNComponents< ValueType >::value;
            typedef typename GetComponentsType< ValueType >::type ComponentType;

            const std::string name_lookup[] = { "x", "y", "z" };

            OpenPMDName< T_Identifier > openPMDName;
            ::openPMD::Series & series = *params->openPMDSeries;
            ::openPMD::Iteration & iteration =
                series.iterations[ params->currentStep ];
            ::openPMD::Record & record =
                iteration.particles[ speciesGroup ][ openPMDName() ];

            // get the SI scaling, dimensionality and weighting of the attribute
            OpenPMDUnit< T_Identifier > openPMDUnit;
            std::vector< float_64 > unit = openPMDUnit();
            OpenPMDUnitDimension< T_Identifier > openPMDUnitDimension;
            std::vector< float_64 > unitDimension = openPMDUnitDimension();
            const bool macroWeightedBool = MacroWeighted< T_Identifier >::get();
            const uint32_t macroWeighted = ( macroWeightedBool ? 1 : 0 );
            const float_64 weightingPower =
                WeightingPower< T_Identifier >::get();

            PMACC_ASSERT(
                unit.size() == components ); // unitSI for each component
            PMACC_ASSERT(
                unitDimension.size() == 7 ); // seven openPMD base units

            for( uint32_t d = 0; d < components; d++ )
            {
                ::openPMD::RecordComponent & recordComponent = components > 1
                    ? record[ name_lookup[ d ] ]
                    : record[::openPMD::MeshRecordComponent::SCALAR ];
                ::openPMD::Datatype openPMDType =
                    ::openPMD::determineDatatype< ComponentType >();

                pushDataset< DIM1 >( recordComponent,
                        openPMDType,
                        pmacc::math::UInt64< DIM1 >( globalElements ),
                        pmacc::math::UInt64< DIM1 >( elements ),
                        pmacc::math::UInt64< DIM1 >( globalOffset ),
                        true,
                        params->compressionMethod,
                        params->particleAttributes);


                /* check if this attribute actually has a unit (unit.size() == 0
                 * is no unit) */
                if( unit.size() >= ( d + 1 ) )
                {
                    recordComponent.setUnitSI( unit.at( d ) );
                }
            }

            std::array< double, 7 > unitDimensionArr;
            std::copy_n( unitDimension.begin(), 7, unitDimensionArr.begin() );
            record.setAttribute(
                "unitDimension", std::move( unitDimensionArr ) );
            record.setAttribute( "macroWeighted", macroWeighted );
            record.setAttribute( "weightingPower", weightingPower );

            // /** \todo check if always correct at this point, depends on
            // attribute
            //  *        and MW-solver/pusher implementation */
            const std::vector< float_X > timeOffset( 7, 0.0 );
            record.setAttribute( "timeOffset", timeOffset );
        }
    };

} // namespace openPMD

} // namespace picongpu
