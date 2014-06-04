/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef TMEN_UTIL_VECUTIL_HPP
#define TMEN_UTIL_VECUTIL_HPP

#include <vector>
#include <iostream>
#include "tensormental/core/error_decl.hpp"
#include "tensormental/core/types_decl.hpp"

namespace tmen {

template<typename T>
T prod(const std::vector<T>& a, const Unsigned startIndex = 0);

template<typename T>
T prod(const std::vector<T>& a, const Unsigned startIndex, const Unsigned endIndex);

template<typename T>
void ElemwiseSum(const std::vector<T>& src1, const std::vector<T>& src2, std::vector<T>& out);

template<typename T>
bool AnyNonNegativeElem(const std::vector<T>& vec);

template<typename T>
bool AnyNonPositiveElem(const std::vector<T>& vec);

template<typename T>
bool AnyNegativeElem(const std::vector<T>& vec);

template<typename T>
bool ElemwiseLessThan(const std::vector<T>& vec1, const std::vector<T>& vec2);

template<typename T>
bool AnyElemwiseGreaterThan(const std::vector<T>& vec1, const std::vector<T>& vec2);

template<typename T>
bool AnyElemwiseGreaterThanEqualTo(const std::vector<T>& vec1, const std::vector<T>& vec2);

template<typename T>
bool AnyZeroElem(const std::vector<T>& vec);

bool AnyFalseElem(const std::vector<bool>& vec);

template<typename T>
bool AnyElemwiseNotEqual(const std::vector<T>& vec1, const std::vector<T>& vec2);

template<typename T>
bool EqualUnderPermutation(const std::vector<T>& vec1, const std::vector<T>& vec2);

template<typename T>
std::vector<T> FilterVector(const std::vector<T>& vec, const std::vector<Unsigned>& filter);

template<typename T>
std::vector<T> NegFilterVector(const std::vector<T>& vec, const std::vector<Unsigned>& filter);

template<typename T>
bool IsSuffix(const std::vector<T>& vec1, const std::vector<T>& vec2);

template<typename T>
bool IsPrefix(const std::vector<T>& vec1, const std::vector<T>& vec2);

template<typename T>
std::vector<T> ConcatenateVectors(const std::vector<T>& vec1, const std::vector<T>& vec2);

template<typename T>
Permutation DeterminePermutation(const std::vector<T>& ref, const std::vector<T>& vec);

} // namespace tmen

#endif // ifndef TMEN_UTIL_VECUTIL_HPP
