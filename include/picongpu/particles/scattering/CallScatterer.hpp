/* Copyright 2021 Pawel Ordyna
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

#include "picongpu/particles/scattering/SplitAndScatterParticles.kernel"
#include "picongpu/particles/scattering/scattering.kernel"

#include <pmacc/traits/GetFlagType.hpp>

#include <functional>
#include <tuple>
#include <utility>

namespace picongpu
{
    namespace particles
    {
        namespace scattering
        {
            namespace detail
            {
                /* Get a FieldTmp pointer
                 *
                 * @tparam i FieldTmp slot id
                 */
                template<size_t i>
                struct GetFieldTmp
                {
                    /* Functor implementation
                     *
                     * @param dc data connector
                     */
                    template<typename T_Dc>
                    HINLINE std::shared_ptr<FieldTmp> operator()(T_Dc& dc) const
                    {
                        return dc.template get<FieldTmp>(FieldTmp::getUniqueId(i), true);
                    }
                };

                template<typename T_NativeFields, size_t i>
                struct GetNativeField
                {
                    using Field = typename bmpl::at_c<T_NativeFields, i>::type;
                    template<typename T_Dc>
                    HINLINE std::shared_ptr<Field> operator()(T_Dc& dc) const
                    {
                        return dc.template get<Field>(Field::getName(), true);
                    }
                };

                template<typename T_NativeFields>
                struct GetNativeFieldsPointersTupleImpl
                {
                    template<typename T_Dc, std::size_t... I>
                    HINLINE auto operator()(T_Dc& dc, std::index_sequence<I...>) const
                    {
                        return std::make_tuple(GetNativeField<T_NativeFields, I>()(dc)...);
                    }
                };

                template<typename T_Dc, std::size_t... I>
                HINLINE auto getTmpFieldPointersTuple(T_Dc& dc, std::index_sequence<I...>)
                {
                    return std::make_tuple(GetFieldTmp<I>()(dc)...);
                }

                /* Get a std::tuple of device side data boxes
                 *
                 * @tparam T_Tuple field pointers tuple type
                 * @tparam I ids of the fields in the provided tuple for witch a databox should be included
                 *
                 * @param fieldPointers A tuple with field pointers
                 * @param the index sequence containing I
                 */
                template<typename T_Tuple, std::size_t... I>
                HINLINE auto getDataBoxTuple(T_Tuple& fieldPointers, std::index_sequence<I...>)
                {
                    return std::make_tuple(
                        (std::get<I>(fieldPointers)->getGridBuffer().getDeviceBuffer().getDataBox())...);
                }

                /* Compute field values for an element from a list of FieldTmp operations.
                 *
                 * @tparam T_FieldTmpOperations A list FieldTmp operations
                 * @tparam extraTmpSlot fieldTmp memory slot to use for intermediary results
                 * @tparam i The index in the operation list for which the computeValue should be called
                 */
                template<typename T_FieldTmpOperations, uint32_t extraTmpSlot, size_t i>
                struct ComputeFieldValuesImpl
                {
                    using FieldTmpOp = typename bmpl::at_c<T_FieldTmpOperations, i>::type;
                    using Species = typename FieldTmpOp::Species;
                    using Solver = typename FieldTmpOp::Solver;
                    using Filter = typename FieldTmpOp::Filter;

                    /* Functor implementation
                     *
                     * @param dc data conector
                     * @param currentStep current simulation step
                     * @param fieldPointers a tuple of field pointers. The tmp operation is called for the i-th
                     *  field in this tuple.
                     */
                    template<typename T_Dc, typename T_Tuple>
                    HINLINE auto operator()(uint32_t const& currentStep, T_Dc& dc, T_Tuple& fieldPointers) const
                    {
                        return particles::particleToGrid::ComputeFieldValue<CORE + BORDER, Solver, Species, Filter>()(
                            *(std::get<i>(fieldPointers)),
                            currentStep,
                            extraTmpSlot);
                    }
                };

                template<template<size_t> class T_Functor>
                struct CallComputeFieldValues
                {
                    template<typename T_Dc, typename T_Tuple, std::size_t... I>
                    HINLINE auto operator()(
                        std::index_sequence<I...>,
                        uint32_t const& currentStep,
                        T_Dc& dc,
                        T_Tuple& fieldPointers) const
                    {
                        return std::make_tuple(T_Functor<I>()(currentStep, dc, fieldPointers)...);
                    }
                };

                template<size_t i>
                struct SetTransactionEvent
                {
                    template<typename T_Tuple>
                    HINLINE void operator()(T_Tuple& eventPointers) const
                    {
                        auto eventPtr = std::get<i>(eventPointers);
                        if(eventPtr.has_value())
                        {
                            __setTransactionEvent(*eventPtr);
                        }
                    }
                };

                template<template<size_t> class T_Functor>
                struct CallSetTransactionEvent
                {
                    template<typename T_Tuple, std::size_t... I>
                    HINLINE void operator()(std::index_sequence<I...>, T_Tuple& eventPointers) const
                    {
                        [[maybe_unused]] int t[] = {((void) T_Functor<I>()(eventPointers), 1)...};
                    }
                };

                template<typename T_ScatterKernel>
                struct CallKernel
                {
                    template<
                        typename T_Mapper,
                        typename T_ParticleBox,
                        typename T_HostScattererFunctor,
                        typename... T_FieldBoxes>
                    HINLINE void operator()(
                        T_Mapper const& mapper,
                        T_ParticleBox& particleBox,
                        T_HostScattererFunctor& hostScattererFunctor,
                        T_FieldBoxes&... fieldBoxes)
                    {
                        constexpr uint32_t numWorkers
                            = pmacc::traits::GetNumWorkers<pmacc::math::CT::volume<SuperCellSize>::type::value>::value;
                        PMACC_KERNEL(T_ScatterKernel{})
                        (mapper.getGridDim(), numWorkers)(mapper, particleBox, hostScattererFunctor, fieldBoxes...);
                    }
                };

                template<typename T_Func, typename... T_values, std::size_t... I, typename... T_Args>
                HINLINE void callWithTuple(
                    T_Func func,
                    const std::tuple<T_values...>& tuple,
                    std::index_sequence<I...>,
                    T_Args&&... args)
                {
                    func(args..., std::get<I>(tuple)...);
                }

            } // namespace detail

            template<typename T_SpeciesType>
            struct CallScatterer
            {
                using SpeciesType = pmacc::particles::meta::FindByNameOrType_t<VectorAllSpecies, T_SpeciesType>;
                using FrameType = typename SpeciesType::FrameType;

                // For now each species can have just one scatterer. One could implement multiple scatterers per
                // species the same way we call multiple ionizers per species.
                using ScattererUnspecialized =
                    typename pmacc::traits::Resolve<typename GetFlagType<FrameType, scatterer<>>::type>::type;
                using Scatterer = typename ScattererUnspecialized::template apply<T_SpeciesType>::type;

                static constexpr uint32_t numWorkers
                    = pmacc::traits::GetNumWorkers<pmacc::math::CT::volume<SuperCellSize>::type::value>::value;
                using Kernel = typename Scatterer::template CallingKernel<numWorkers>;

                /* List of fields that are required by the functor.
                 *
                 * The Scatterer provides a list of required derived fields, i.e. density or temperature of some
                 * species. This is an mpl sequence of already specialized FieldTmpOperations.
                 */
                using RequiredDerivedFields =
                    typename pmacc::traits::Resolve<typename Scatterer::RequiredDerivedFields>::type;
                static constexpr size_t numTmpFields = bmpl::size<RequiredDerivedFields>::type::value;
                template<size_t i>
                using ComputeFieldValues = detail::ComputeFieldValuesImpl<RequiredDerivedFields, numTmpFields, i>;
                using RequiredNativeFields =
                    typename pmacc::traits::Resolve<typename Scatterer::RequiredNativeFields>::type;
                using GetNativeFieldsPointersTuple = detail::GetNativeFieldsPointersTupleImpl<RequiredNativeFields>;

                template<typename T_FieldTmpOperation>
                struct GetExtraSlots
                {
                    using Solver = typename T_FieldTmpOperation::Solver;
                    using type = bmpl::int_<particles::particleToGrid::RequiredExtraSlots<Solver>::type::value>;
                };

                using ListExtraSlots = typename bmpl::transform<RequiredDerivedFields, GetExtraSlots<bmpl::_1>>::type;

                using RequiredExtraTmpSlots =
                    typename bmpl::accumulate<ListExtraSlots, bmpl::int_<0>, bmpl::max<bmpl::_1, bmpl::_2>>::type;
                static constexpr uint32_t requiredExtraTmpSlots = RequiredExtraTmpSlots::value;


                /** Functor implementation
                 *
                 * @tparam T_CellDescription contains the number of blocks and blocksize
                 *                           that is later passed to the kernel
                 * @param cellDesc logical block information like dimension and cell sizes
                 * @param currentStep The current time step
                 */
                HINLINE void operator()(const uint32_t currentStep) const
                {
                    using RequiredSlots = bmpl::int_<requiredExtraTmpSlots>;
                    PMACC_CASSERT_MSG_TYPE(
                        _please_allocate_at_least_as_many_FieldTmp_slots_in_memory_param_as_fields_required_for_scattering_plus_1_if_combined_attrbiutes_are_required,
                        ListExtraSlots,
                        fieldTmpNumSlots >= numTmpFields + requiredExtraTmpSlots);
                    DataConnector& dc = Environment<>::get().DataConnector();
                    // Use an index sequnce (C++11 feature) to iterate over the fields with compile time loops.
                    std::make_index_sequence<numTmpFields> indexTmpFields{};
                    // Get an emtpy TmpField slot for each field. Store the shared field pointers in an std::tuple.
                    auto tmpFieldPointers = detail::getTmpFieldPointersTuple(dc, indexTmpFields);

                    auto eventTaskPointers = detail::CallComputeFieldValues<ComputeFieldValues>()(
                        indexTmpFields,
                        currentStep,
                        dc,
                        tmpFieldPointers);
                    auto tmpDataBoxes = detail::getDataBoxTuple(tmpFieldPointers, indexTmpFields);
                    // TODO:: Add support for field modifiers. We need to do 1/gamma^2 for Faraday.

                    // get native fields
                    constexpr size_t numNativeFields = bmpl::size<RequiredNativeFields>::type::value;
                    std::make_index_sequence<numNativeFields> indexNativeFields{};
                    auto nativeFieldPointers = GetNativeFieldsPointersTuple()(dc, indexNativeFields);
                    auto nativeDataBoxes = detail::getDataBoxTuple(nativeFieldPointers, indexNativeFields);

                    // combine all fields
                    auto dataBoxes = std::tuple_cat(nativeDataBoxes, tmpDataBoxes);
                    std::make_index_sequence<numNativeFields + numTmpFields> indexAllFields{};

                    // Get particle data.
                    auto species = dc.get<SpeciesType>(FrameType::getName(), true);
                    // Mapping descripiton for the kernel call.
                    AreaMapping<CORE + BORDER, picongpu::MappingDesc> mapper(species->getCellDescription());

                    detail::CallSetTransactionEvent<detail::SetTransactionEvent>()(indexTmpFields, eventTaskPointers);
                    // Call the scattering kernel and pass the field databoxes.
                    detail::callWithTuple(
                        detail::CallKernel<Kernel>(),
                        dataBoxes,
                        indexAllFields,
                        mapper,
                        species->getDeviceParticlesBox(),
                        Scatterer(currentStep));
                    // When the scatterer removes particles fill gaps needs to be called afterwards.
                    // Should be turned into a constexpr if after switching to c++17.
                    if constexpr (Scatterer::needsFillGaps)
                    {
                        // fill Gaps only in CORE + BORDER?
                        species->fillAllGaps();
                    }
                }
            };
        } // namespace scattering
    } // namespace particles
} // namespace picongpu
