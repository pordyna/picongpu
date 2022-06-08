/* Copyright 2013-2022 Rene Widera
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */


#pragma once


#include "pmacc/meta/conversion/MakeSeqFromNestedSeq.hpp"

#include <boost/mpl/range_c.hpp>
#include <boost/mpl/vector.hpp>

namespace pmacc
{
    /** combine all input types to one sequence
     *
     * Note: if the input type is a sequence itself, its elements will be unfolded
     *       and added separately
     *
     * @tparam T_Args a boost mpl sequence or single type
     *
     * @code
     * using MyType = typename MakeSeq< A, B >::type
     * using MyType2 = typename MakeSeq< boost::mpl::vector<A, B>, C >::type
     * @endcode
     *
     */
    template<typename... T_Args>
    struct MakeSeq
    {
        using type = typename MakeSeqFromNestedSeq<bmpl::vector<T_Args...>>::type;
    };

    /** short hand definition for @see MakeSeq<> */
    template<typename... T_Args>
    using MakeSeq_t = typename MakeSeq<T_Args...>::type;

    /** create a vector with n elements of type T
     *
     * @tparam n number of elements
     * @tparam T sequence element
     */
    template<size_t n, typename T>
    struct MakeSeqWithIdenticalElements
    {
        using type = typename bmpl::
            fold<typename bmpl::range_c<size_t, 0, n>::type, bmpl::vector0<>, JoinToSeq<bmpl::_1, T>>::type;
    };
} // namespace pmacc
