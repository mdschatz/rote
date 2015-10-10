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
T sum(const std::vector<T>& a, const Unsigned startIndex = 0);

template<typename T>
T sum(const std::vector<T>& a, const Unsigned startIndex, const Unsigned nElem);

template<typename T>
T prod(const std::vector<T>& a, const Unsigned startIndex = 0);

template<typename T>
T prod(const std::vector<T>& a, const Unsigned startIndex, const Unsigned nElem);

template<typename T>
void ElemwiseSum(const std::vector<T>& src1, const std::vector<T>& src2, std::vector<T>& out);

template<typename T>
std::vector<T> ElemwiseSum(const std::vector<T>& src1, const std::vector<T>& src2);

template<typename T>
void ElemwiseSubtract(const std::vector<T>& src1, const std::vector<T>& src2, std::vector<T>& out);

template<typename T>
std::vector<T> ElemwiseSubtract(const std::vector<T>& src1, const std::vector<T>& src2);

template<typename T>
void ElemwiseProd(const std::vector<T>& src1, const std::vector<T>& src2, std::vector<T>& out);

template<typename T>
std::vector<T> ElemwiseProd(const std::vector<T>& src1, const std::vector<T>& src2);

template<typename T>
void ElemwiseDivide(const std::vector<T>& src1, const std::vector<T>& src2, std::vector<T>& out);

template<typename T>
std::vector<T> ElemwiseDivide(const std::vector<T>& src1, const std::vector<T>& src2);

template<typename T>
void ElemwiseMod(const std::vector<T>& src1, const std::vector<T>& src2, std::vector<T>& out);

template<typename T>
std::vector<T> ElemwiseMod(const std::vector<T>& src1, const std::vector<T>& src2);

template<typename T>
bool AnyNonNegativeElem(const std::vector<T>& vec);

template<typename T>
bool AnyNonPositiveElem(const std::vector<T>& vec);

template<typename T>
bool AnyPositiveElem(const std::vector<T>& vec);

template<typename T>
bool AnyNegativeElem(const std::vector<T>& vec);

template<typename T>
bool ElemwiseLessThan(const std::vector<T>& vec1, const std::vector<T>& vec2);

template<typename T>
bool ElemwiseLessThanEqualTo(const std::vector<T>& vec1, const std::vector<T>& vec2);

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
std::vector<T> PermuteVector(const std::vector<T>& vec, const Permutation& perm);

template<typename T>
std::vector<T> FilterVector(const std::vector<T>& vec, const std::vector<Unsigned>& filter);

template<typename T>
std::vector<T> NegFilterVector(const std::vector<T>& vec, const std::vector<Unsigned>& filter);

template<typename T>
bool IsSame(const std::vector<T>& vec1, const std::vector<T>& vec2);

template<typename T>
std::vector<T> GetSuffix(const std::vector<T>& vec1, const std::vector<T>& vec2);

template<typename T>
bool IsSuffix(const std::vector<T>& vec1, const std::vector<T>& vec2);

template<typename T>
bool IsPrefix(const std::vector<T>& vec1, const std::vector<T>& vec2);

template<typename T>
T Min(const std::vector<T>& vec);

template<typename T>
T Max(const std::vector<T>& vec);

template<typename T>
std::vector<T> ConcatenateVectors(const std::vector<T>& vec1, const std::vector<T>& vec2);

template<typename T>
Permutation DeterminePermutation(const std::vector<T>& ref, const std::vector<T>& vec);

Permutation DetermineInversePermutation(const Permutation& perm);

} // namespace tmen

#endif // ifndef TMEN_UTIL_VECUTIL_HPP
