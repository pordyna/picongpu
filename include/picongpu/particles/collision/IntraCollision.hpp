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
            struct IntraCollision
            {
                template<
                    typename T_ParBox,
                    typename T_Mapping,
                    typename T_Acc,
                    typename T_DeviceHeapHandle,
                    typename T_RngHandle,
                    typename T_CollisionFunctor,
                    typename T_Filter>
                DINLINE void operator()(
                    T_Acc const& acc,
                    T_ParBox pb,
                    T_Mapping const mapper,
                    T_DeviceHeapHandle deviceHeapHandle,
                    T_RngHandle rngHandle,
                    T_CollisionFunctor const collisionFunctor,
                    float_X coulombLog,
                    T_Filter filter) const
                {
                    using namespace pmacc::particles::operations;
                    using namespace mappings::threads;

                    using SuperCellSize = typename T_ParBox::FrameType::SuperCellSize;
                    constexpr uint32_t frameSize = pmacc::math::CT::volume<SuperCellSize>::type::value;
                    constexpr uint32_t numWorkers = T_numWorkers;

                    using FramePtr = typename T_ParBox::FramePtr;

                    PMACC_SMEM(acc, nppc, memory::Array<uint32_t, frameSize>);

                    PMACC_SMEM(acc, parCellList, memory::Array<detail::ListEntry, frameSize>);

                    PMACC_SMEM(acc, densityArray, memory::Array<float_X, frameSize>);

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

                    auto& superCell = pb.getSuperCell(superCellIdx);
                    uint32_t numParticlesInSupercell = superCell.getNumParticles();

                    auto accFilter = filter(acc, localSuperCellOffset, threads::WorkerCfg<T_numWorkers>{workerIdx});

                    /* loop over all particles in the frame */
                    ForEachIdx<FrameDomCfg> forEachFrameElem(workerIdx);
                    FramePtr firstFrame = pb.getFirstFrame(superCellIdx);

                    prepareList(
                        acc,
                        forEachFrameElem,
                        deviceHeapHandle,
                        pb,
                        firstFrame,
                        numParticlesInSupercell,
                        parCellList,
                        nppc,
                        accFilter);

                    // TODO: density, debaylength

                    cellDensity(acc, forEachFrameElem, firstFrame, pb, parCellList, densityArray, accFilter);
                    cupla::__syncthreads(acc);

                    // shuffle  indices list
                    forEachFrameElem([&](uint32_t const linearIdx, uint32_t const) {
                        parCellList[linearIdx].shuffle(acc, rngHandle);
                    });

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

                    forEachFrameElem([&](uint32_t const linearIdx, uint32_t const idx) {
                        uint32_t const sizeAll = parCellList[linearIdx].size;
                        if(sizeAll < 2u)
                            return;
                        // skip particle offset counter
                        uint32_t* listAll = parCellList[linearIdx].ptrToIndicies;
                        uint32_t potentialPartners = sizeAll - 1u + sizeAll % 2u;
                        collisionFunctorCtx[idx] = collisionFunctor(
                            acc,
                            localSuperCellOffset,
                            WorkerCfg<T_numWorkers>{workerIdx},
                            densityArray[linearIdx],
                            densityArray[linearIdx],
                            potentialPartners,
                            coulombLog);
                        for(uint32_t i = 0; i < sizeAll; i += 2)
                        {
                            uint32_t const sizeOdd = sizeAll / 2;
                            uint32_t const iOdd = (i / 2) % sizeOdd;
                            uint32_t duplications = (iOdd == 0 && sizeAll % 2 == 1) ? 2 : 1;
                            duplications *= 2u; // TODO move this somewhere else
                            auto parEven = detail::getParticle(pb, firstFrame, listAll[i]);
                            auto parOdd = detail::getParticle(pb, firstFrame, listAll[iOdd * 2 + 1]);
                            collisionFunctorCtx[idx].duplications = duplications;
                            (collisionFunctorCtx[idx])(detail::makeCollisionContext(acc, rngHandle), parEven, parOdd);
                        }
                    });

                    cupla::__syncthreads(acc);

                    forEachFrameElem([&](uint32_t const linearIdx, uint32_t const) {
                        parCellList[linearIdx].finalize(acc, deviceHeapHandle);
                    });
                }
            };

            /* Run kernel for internal collisions of one species.
             *
             * @tparam T_CollisionFunctor A binary particle functor defining a single macro particle collision in the
             *     binary-collision algorithm.
             * @tparam T_Params A struct defining `coulombLog` for the collisions.
             * @tparam T_FilterPair A pair of particle filters, each for each species
             *     in the colliding pair.
             * @tparam T_Species0 Colliding species.
             * @tparam T_Species1 2nd colliding species.
             */
            template<typename T_CollisionFunctor, typename T_Params, typename T_FilterPair, typename T_Species>
            struct DoIntraCollision;

            // A single template specialization. This ensures that the code want compile if the FilterPair contains two
            // different filters. That wouldn't make much sense for internal collisions.
            template<typename T_CollisionFunctor, typename T_Params, typename T_Filter, typename T_Species>
            struct DoIntraCollision<T_CollisionFunctor, T_Params, FilterPair<T_Filter, T_Filter>, T_Species>
            {
                /* Run kernel
                 *
                 * @param deviceHeap A pointer to device heap for allocating particle lists.
                 * @param currentStep The current simulation step.
                 */
                void operator()(const std::shared_ptr<DeviceHeap>& deviceHeap, uint32_t currentStep)
                {
                    using Species = T_Species;
                    using FrameType = typename Species::FrameType;
                    using Filter = typename T_Filter::template apply<Species>::type;
                    using CollisionFunctor = T_CollisionFunctor;

                    DataConnector& dc = Environment<>::get().DataConnector();
                    auto species = dc.get<Species>(FrameType::getName(), true);

                    AreaMapping<CORE + BORDER, picongpu::MappingDesc> mapper(species->getCellDescription());

                    constexpr uint32_t numWorkers
                        = pmacc::traits::GetNumWorkers<pmacc::math::CT::volume<SuperCellSize>::type::value>::value;

                    /* random number generator */
                    using RNGFactory = pmacc::random::RNGProvider<simDim, random::Generator>;
                    constexpr float_X coulombLog = T_Params::coulombLog;
                    PMACC_KERNEL(IntraCollision<numWorkers>{})
                    (mapper.getGridDim(), numWorkers)(
                        species->getDeviceParticlesBox(),
                        mapper,
                        deviceHeap->getAllocatorHandle(),
                        RNGFactory::createHandle(),
                        CollisionFunctor(currentStep),
                        coulombLog,
                        Filter());

                    dc.releaseData(FrameType::getName());
                }
            };

        } // namespace collision
    } // namespace particles
} // namespace picongpu
