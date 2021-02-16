/* Copyright 2014-2019 Rene Widera
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

#include <pmacc/mappings/threads/ForEachIdx.hpp>
#include <pmacc/mappings/threads/IdxConfig.hpp>
#include <pmacc/mappings/threads/WorkerCfg.hpp>
#include <pmacc/particles/meta/FindByNameOrType.hpp>
#include <pmacc/math/Vector.hpp>
#include <pmacc/random/RNGProvider.hpp>
#include <pmacc/random/distributions/Uniform.hpp>

#include "picongpu/particles/collision/detail/ListEntry.hpp"
#include "picongpu/particles/collision/detail/CollisionContext.hpp"
#include "picongpu/particles/collision/detail/cellDensity.hpp"


namespace picongpu
{
    namespace particles
    {
        namespace collision
        {
            using namespace pmacc::mappings;

            template<uint32_t T_numWorkers>
            struct InterCollision
            {
                template<
                    typename T_ParBox0,
                    typename T_ParBox1,
                    typename T_Mapping,
                    typename T_Acc,
                    typename T_DeviceHeapHandle,
                    typename T_RngHandle,
                    typename T_CollisionFunctor,
                    typename T_Filter0,
                    typename T_Filter1>
                DINLINE void operator()(
                    T_Acc const& acc,
                    T_ParBox0 pb0,
                    T_ParBox1 pb1,
                    T_Mapping const mapper,
                    T_DeviceHeapHandle deviceHeapHandle,
                    T_RngHandle rngHandle,
                    T_CollisionFunctor const collisionFunctor,
                    float_X coulombLog,
                    T_Filter0 filter0,
                    T_Filter1 filter1) const
                {
                    using namespace pmacc::particles::operations;
                    using namespace mappings::threads;

                    using SuperCellSize = typename T_ParBox0::FrameType::SuperCellSize;
                    constexpr uint32_t frameSize = pmacc::math::CT::volume<SuperCellSize>::type::value;
                    constexpr uint32_t numWorkers = T_numWorkers;

                    PMACC_SMEM(acc, nppc, memory::Array<uint32_t, frameSize>);
                    PMACC_SMEM(acc, parCellList0, memory::Array<detail::ListEntry, frameSize>);
                    PMACC_SMEM(acc, parCellList1, memory::Array<detail::ListEntry, frameSize>);
                    PMACC_SMEM(acc, densityArray0, memory::Array<float_X, frameSize>);
                    PMACC_SMEM(acc, densityArray1, memory::Array<float_X, frameSize>);

                    uint32_t const workerIdx = cupla::threadIdx(acc).x;
                    using FrameDomCfg = IdxConfig<frameSize, numWorkers>;

                    DataSpace<simDim> const superCellIdx
                        = mapper.getSuperCellIndex(DataSpace<simDim>(cupla::blockIdx(acc)));

                    // offset of the superCell (in cells, without any guards) to the
                    // origin of the local domain
                    DataSpace<simDim> const localSuperCellOffset = superCellIdx - mapper.getGuardingSuperCells();
                    rngHandle.init(
                        localSuperCellOffset * SuperCellSize::toRT()
                        + DataSpaceOperations<simDim>::template map<SuperCellSize>(workerIdx));

                    auto accFilter0 = filter0(acc, localSuperCellOffset, threads::WorkerCfg<T_numWorkers>{workerIdx});
                    auto accFilter1 = filter1(acc, localSuperCellOffset, threads::WorkerCfg<T_numWorkers>{workerIdx});

                    /* loop over all particles in the frame */
                    ForEachIdx<FrameDomCfg> forEachFrameElem(workerIdx);

                    using CollidingGroup0Type
                        = CollidingGroup<T_ParBox0, decltype(parCellList0), decltype(densityArray0)>;
                    using CollidingGroup1Type
                        = CollidingGroup<T_ParBox1, decltype(parCellList1), decltype(densityArray1)>;
                    CollidingGroup0Type collidingGroup0(pb0, parCellList0, densityArray0, superCellIdx);
                    CollidingGroup1Type collidingGroup1(pb1, parCellList1, densityArray1, superCellIdx);

                    prepareList(acc, forEachFrameElem, deviceHeapHandle, collidingGroup0, nppc, accFilter0);

                    prepareList(acc, forEachFrameElem, deviceHeapHandle, collidingGroup1, nppc, accFilter1);

                    cellDensity(acc, forEachFrameElem, collidingGroup0, accFilter0);
                    cellDensity(acc, forEachFrameElem, collidingGroup1, accFilter1);
                    cupla::__syncthreads(acc);

                    memory::CtxArray<
                        decltype(collisionFunctor(
                            acc,
                            alpaka::core::declval<DataSpace<simDim> const>(),
                            /* cellsPerSupercell is used because each virtual worker
                             * is creating **exactly one** functor
                             */
                            alpaka::core::declval<WorkerCfg<frameSize> const>(),
                            alpaka::core::declval<float_X const>(),
                            alpaka::core::declval<float_X const>(),
                            alpaka::core::declval<uint32_t const>(),
                            alpaka::core::declval<float_X const>())),
                        FrameDomCfg>
                        collisionFunctorCtx{};

                    // shuffle indices list of the longest particle list
                    forEachFrameElem([&](uint32_t const linearIdx, uint32_t const idx) {
                        // find longer list
                        auto collidingGroups = (parCellList0[linearIdx].size >= parCellList1[linearIdx].size)
                            ? CollidingGroupPair<CollidingGroup1Type, CollidingGroup0Type>(
                                collidingGroup1,
                                collidingGroup0)
                            : CollidingGroupPair<CollidingGroup0Type, CollidingGroup1Type>(
                                collidingGroup0,
                                collidingGroup1);

                        collidingGroups.collidingGroupLong.parCellsList[linearIdx].shuffle(acc, rngHandle);

                        uint32_t const sizeShort = collidingGroups.collidingGroupShort.parCellsList[linearIdx].size;
                        uint32_t const sizeLong = collidingGroups.collidingGroupLong.parCellsList[linearIdx].size;
                        collisionFunctorCtx[idx] = collisionFunctor(
                            acc,
                            localSuperCellOffset,
                            threads::WorkerCfg<T_numWorkers>{workerIdx},
                            collidingGroups.collidingGroupLong.densArray[linearIdx],
                            collidingGroups.collidingGroupShort.densArray[linearIdx],
                            sizeLong,
                            coulombLog);
                        if(sizeShort == 0u)
                            return;
                        for(uint32_t i = 0; i < sizeLong; ++i)
                        {
                            PMACC_DEVICE_ASSERT(sizeLong >= sizeShort);
                            uint32_t duplications(1u);
                            if(sizeLong > sizeShort)
                            {
                                // integer division: floor(longSize / shortSize)
                                duplications = sizeLong / sizeShort;
                                uint32_t modulo = sizeLong % sizeShort;
                                if((i % sizeShort) < modulo) // TODO:
                                {
                                    duplications += 1;
                                }
                            };
                            auto parLong = detail::getParticle(
                                collidingGroups.collidingGroupLong.pb,
                                collidingGroups.collidingGroupLong.firstFrame,
                                (collidingGroups.collidingGroupLong.parCellsList[linearIdx])[i]);
                            auto parShort = detail::getParticle(
                                collidingGroups.collidingGroupShort.pb,
                                collidingGroups.collidingGroupShort.firstFrame,
                                (collidingGroups.collidingGroupShort.parCellsList[linearIdx])[i % sizeShort]);
                            collisionFunctorCtx[idx].duplications = duplications;
                            (collisionFunctorCtx[idx])(
                                detail::makeCollisionContext(acc, rngHandle),
                                parLong,
                                parShort);
                        }
                    });

                    cupla::__syncthreads(acc);

                    forEachFrameElem([&](uint32_t const linearIdx, uint32_t const) {
                        parCellList0[linearIdx].finalize(acc, deviceHeapHandle);
                        parCellList1[linearIdx].finalize(acc, deviceHeapHandle);
                    });
                }
            };


            /* Run kernel for collisions between two species.
             *
             * @tparam T_CollisionFunctor A binary particle functor defining a single macro particle collision in
             * the binary-collision algorithm.
             * @tparam T_Params A struct defining `coulombLog` for the collisions.
             * @tparam T_FilterPair A pair of particle filters, each for each species
             *     in the colliding pair.
             * @tparam T_Species0 1st colliding species.
             * @tparam T_Species1 2nd colliding species.
             */
            template<
                typename T_CollisionFunctor,
                typename T_Params,
                typename T_FilterPair,
                typename T_Species0,
                typename T_Species1>
            struct DoInterCollision
            {
                /* Run kernel
                 *
                 * @param deviceHeap A pointer to device heap for allocating particle lists.
                 * @param currentStep The current simulation step.
                 */
                HINLINE void operator()(const std::shared_ptr<DeviceHeap>& deviceHeap, uint32_t currentStep)
                {
                    using Species0 = T_Species0;
                    using FrameType0 = typename Species0::FrameType;
                    using Filter0 = typename T_FilterPair::first ::template apply<Species0>::type;

                    using Species1 = T_Species1;
                    using FrameType1 = typename Species1::FrameType;
                    using Filter1 = typename T_FilterPair::second ::template apply<Species1>::type;

                    using CollisionFunctor = T_CollisionFunctor;

                    // Access particle data:
                    DataConnector& dc = Environment<>::get().DataConnector();
                    auto species0 = dc.get<Species0>(FrameType0::getName(), true);
                    auto species1 = dc.get<Species1>(FrameType1::getName(), true);

                    // Use mapping information from the first species:
                    AreaMapping<CORE + BORDER, picongpu::MappingDesc> mapper(species0->getCellDescription());

                    constexpr uint32_t numWorkers
                        = pmacc::traits::GetNumWorkers<pmacc::math::CT::volume<SuperCellSize>::type::value>::value;

                    //! random number generator
                    using RNGFactory = pmacc::random::RNGProvider<simDim, random::Generator>;
                    constexpr float_X coulombLog = T_Params::coulombLog;

                    PMACC_KERNEL(InterCollision<numWorkers>{})
                    (mapper.getGridDim(), numWorkers)(
                        species0->getDeviceParticlesBox(),
                        species1->getDeviceParticlesBox(),
                        mapper,
                        deviceHeap->getAllocatorHandle(),
                        RNGFactory::createHandle(),
                        CollisionFunctor(currentStep),
                        coulombLog,
                        Filter0(),
                        Filter1());

                    // Release particle data:
                    // TODO: can I relase it like this? Or do I have to synchronize, make KERNEL call blocking?
                    dc.releaseData(FrameType0::getName());
                    dc.releaseData(FrameType1::getName());
                }
            };

        } // namespace collision
    } // namespace particles
} // namespace picongpu
