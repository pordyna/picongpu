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

#include "picongpu/particles/particleToGrid/CombinedDerive.def"
namespace picongpu
{
    namespace particles
    {
        namespace particleToGrid
        {
            /** Derived Attribute as a function of two attributes directly derived from particles
             *
             * @tparam T_BaseDerivedAttribute first parameter (derived attribute)
             * @tparam T_ModifyingDerivedAttribute second parameter (derived attribute)
             * @tparam T_ModifyingOperation functor defining the function of the two parameters
             * @tparam T_AttributeDescription class providing unit and name for the resulting attribute
             */
            template<
                typename T_BaseDerivedAttribute,
                typename T_ModifyingDerivedAttribute,
                typename T_ModifyingOperation,
                typename T_AttributeDescription>
            struct CombinedDeriveAttribute
            {
                float1_64 getUnit() const
                {
                    return T_AttributeDescription().getUnit();
                }

                std::vector<float_64> getUnitDimension() const
                {
                    return T_AttributeDescription().getUnitDimension();
                }

                static std::string getName()
                {
                    return T_AttributeDescription::getName();
                }
            };

            template<typename T_ListFieldOperations, typename T_ListOperations, typename T_FieldDescription>
            struct MultiAttributeFieldOperation
            {
                PMACC_STATIC_ASSERT_MSG(
                    bmpl::size<T_ListFieldOperations>::value == bmpl::size<T_ListOperations>::value + 1u,
                    operationsList_must_have_one_less_element_as_fieldOperationsList);

                float1_64 getUnit() const
                {
                    return T_FieldDescription().getUnit();
                }

                std::vector<float_64> getUnitDimension() const
                {
                    return T_FieldDescription().getUnitDimension();
                }

                static std::string getName()
                {
                    return T_FieldDescription::getName();
                }
            };

            template<typename T_ListFieldOperations, typename T_Name>
            struct MakeSumFieldOperation
            {
                struct FieldDescription
                {
                    using FirstOperation = typename bmpl::first<T_ListFieldOperations>::type;
                    float1_64 getUnit() const
                    {
                        return FirstOperation::Solver().getUnit();
                    }

                    std::vector<float_64> getUnitDimension() const
                    {
                        return FirstOperation::Solver().getUnitDimension();
                    }

                    static std::string getName()
                    {
                        return bmpl::c_str<T_Name>::value;
                    }
                };
                using ListOperations = MakeSeqWithIdenticalElements<
                    bmpl::size<T_ListFieldOperations>::value - 1,
                    pmacc::math::operation::Add>;
                using type = MultiAttributeFieldOperation<T_ListFieldOperations, ListOperations, FieldDescription>;
            };

            template<typename T_ListAttributes, typename T_Name>
            using MakeSumFieldOperation_t = typename MakeSumFieldOperation<T_ListAttributes, T_Name>::type;

        } // namespace particleToGrid
    } // namespace particles
} // namespace picongpu
