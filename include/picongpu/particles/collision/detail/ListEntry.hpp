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
#include <pmacc/random/distributions/Uniform.hpp>

namespace picongpu
{
    namespace particles
    {
        namespace collision
        {
            namespace detail
            {
                struct ListEntry
                {
                    uint32_t size;
                    uint32_t* ptrToIndicies;


                    template<typename T_acc, typename T_DeviceHeapHandle>
                    DINLINE void init(T_acc const& acc, T_DeviceHeapHandle& deviceHeapHandle, uint32_t numPar)
                    {
                        ptrToIndicies = nullptr;
                        if(numPar != 0u)
                        {
                            // printf("alloc %u: %u\n", linearIdx, (nppc[ linearIdx ] + 1) );
#if(PMACC_CUDA_ENABLED == 1)
                            int i = 0;
                            while(ptrToIndicies == nullptr)
                            {
                                ptrToIndicies = (uint32_t*) deviceHeapHandle.malloc(acc, sizeof(uint32_t) * numPar);

                                if(i >= 5)
                                    printf("no memory: %u\n", numPar);
                                ++i;
                            }
#else
                            ptrToIndicies = new uint32_t[numPar];
#endif
                        }
                        // reset counter
                        size = 0u;
                    }


                    template<typename T_acc, typename T_DeviceHeapHandle>
                    DINLINE void finalize(T_acc const& acc, T_DeviceHeapHandle& deviceHeapHandle)
                    {
                        if(ptrToIndicies != nullptr)
                        {
#if(PMACC_CUDA_ENABLED == 1)
                            deviceHeapHandle.free(acc, (void*) ptrToIndicies);
                            ptrToIndicies = nullptr;
#else
                            delete(ptrToIndicies);
#endif
                        }
                    }


                    // non collective
                    template<typename T_Acc, typename T_RngHandle>
                    DINLINE void shuffle(T_Acc const& acc, T_RngHandle& rngHandle)
                    {
                        using UniformUint32_t = pmacc::random::distributions::Uniform<uint32_t>;
                        auto rng = rngHandle.template applyDistribution<UniformUint32_t>();
                        // shuffle the particle lookup table
                        for(uint32_t i = size; i > 1; --i)
                        {
                            /* modulo is not perfect but okish,
                             * because of the loop head mod zero is not possible
                             */
                            int p = rng(acc) % i;
                            if(i - 1 != p)
                                swap(ptrToIndicies[i - 1], ptrToIndicies[p]);
                        }
                    }


                private:
                    DINLINE void swap(uint32_t& v0, uint32_t& v1)
                    {
                        uint32_t tmp = v0;
                        v0 = v1;
                        v1 = tmp;
                    }
                };


                //! Counting particles per grid frame
                template<
                    typename T_Acc,
                    typename T_ForEach,
                    typename T_ParBox,
                    typename T_FramePtr,
                    typename T_Array,
                    typename T_Filter>
                DINLINE void particlesCntHistogram(
                    T_Acc const& acc,
                    T_ForEach forEach,
                    T_ParBox& parBox,
                    T_FramePtr frame,
                    uint32_t const numParticlesInSupercell,
                    T_Array& nppc,
                    T_Filter& filter)
                {
                    using SuperCellSize = typename T_ParBox::FrameType::SuperCellSize;
                    constexpr uint32_t frameSize = pmacc::math::CT::volume<SuperCellSize>::type::value;

                    for(uint32_t i = 0; i < numParticlesInSupercell; i += frameSize)
                    {
                        forEach([&](uint32_t const linearIdx, uint32_t const idx) {
                            if(i + linearIdx < numParticlesInSupercell)
                            {
                                auto particle = frame[linearIdx];
                                if(filter(acc, particle))
                                {
                                    auto parLocalIndex = particle[localCellIdx_];
                                    cupla::atomicAdd(acc, &nppc[parLocalIndex], 1u);
                                }
                            }
                        });
                        frame = parBox.getNextFrame(frame);
                    }
                }

                /* Fills parCellList with new particles.
                 * parCellList stores a list of particles for each grid cell and the
                 * index in the supercell for each particle.
                 */
                template<
                    typename T_Acc,
                    typename T_ForEach,
                    typename T_ParBox,
                    typename T_FramePtr,
                    typename T_EntryListArray,
                    typename T_Filter>
                DINLINE void updateLinkedList(
                    T_Acc const& acc,
                    T_ForEach forEach,
                    T_ParBox& parBox,
                    T_FramePtr frame,
                    uint32_t const numParticlesInSupercell,
                    T_EntryListArray& parCellList,
                    T_Filter& filter)
                {
                    using SuperCellSize = typename T_ParBox::FrameType::SuperCellSize;
                    constexpr uint32_t frameSize = pmacc::math::CT::volume<SuperCellSize>::type::value;
                    for(uint32_t i = 0; i < numParticlesInSupercell; i += frameSize)
                    {
                        forEach([&](uint32_t const linearIdx, uint32_t const idx) {
                            uint32_t const parInSuperCellIdx = i + linearIdx;
                            if(parInSuperCellIdx < numParticlesInSupercell)
                            {
                                auto particle = frame[linearIdx];
                                if(filter(acc, particle))
                                {
                                    auto parLocalIndex = particle[localCellIdx_];
                                    uint32_t parOffset = cupla::atomicAdd(acc, &parCellList[parLocalIndex].size, 1u);
                                    parCellList[parLocalIndex].ptrToIndicies[parOffset] = parInSuperCellIdx;
                                }
                            }
                        });
                        frame = parBox.getNextFrame(frame);
                    }
                }

                template<typename T_ParBox, typename T_FramePtr>
                DINLINE auto getParticle(T_ParBox& parBox, T_FramePtr frame, uint32_t particleId) ->
                    typename T_FramePtr::type::ParticleType
                {
                    constexpr uint32_t frameSize
                        = pmacc::math::CT::volume<typename T_FramePtr::type::SuperCellSize>::type::value;
                    uint32_t const skipFrames = particleId / frameSize;
                    for(uint32_t i = 0; i < skipFrames; ++i)
                        frame = parBox.getNextFrame(frame);
                    return frame[particleId % frameSize];
                }

                template<
                    typename T_Acc,
                    typename T_ForEach,
                    typename T_DeviceHeapHandle,
                    typename T_ParBox,
                    typename T_FramePtr,
                    typename T_EntryListArray,
                    typename T_Array,
                    typename T_Filter>
                DINLINE void prepareList(
                    T_Acc const& acc,
                    T_ForEach forEach,
                    T_DeviceHeapHandle deviceHeapHandle,
                    T_ParBox& parBox,
                    T_FramePtr firstFrame,
                    uint32_t const numParticlesInSupercell,
                    T_EntryListArray& parCellList,
                    T_Array& nppc,
                    T_Filter filter)
                {
                    forEach([&](uint32_t const linearIdx, uint32_t const idx) { nppc[linearIdx] = 0u; });

                    cupla::__syncthreads(acc);

                    particlesCntHistogram(acc, forEach, parBox, firstFrame, numParticlesInSupercell, nppc, filter);

                    cupla::__syncthreads(acc);

                    // memory for particle indices
                    forEach([&](uint32_t const linearIdx, uint32_t const) {
                        parCellList[linearIdx].init(acc, deviceHeapHandle, nppc[linearIdx]);
                    });

                    cupla::__syncthreads(acc);

                    detail::updateLinkedList(
                        acc,
                        forEach,
                        parBox,
                        firstFrame,
                        numParticlesInSupercell,
                        parCellList,
                        filter);
                    cupla::__syncthreads(acc);
                }
            } // namespace detail
        } // namespace collision
    } // namespace particles
} // namespace picongpu
