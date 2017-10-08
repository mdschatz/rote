/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef ROTE_UTIL_VECTOR_HPP
#define ROTE_UTIL_VECTOR_HPP

namespace rote {

template<typename T>
T sum(const std::vector<T>& a);

template<typename T>
T prod(const std::vector<T>& a, const Unsigned startIndex = 0);

template<typename T>
std::vector<T> ElemwiseSum(const std::vector<T>& src1, const std::vector<T>& src2);

template<typename T>
std::vector<T> ElemwiseSubtract(const std::vector<T>& src1, const std::vector<T>& src2);

template<typename T>
std::vector<T> ElemwiseProd(const std::vector<T>& src1, const std::vector<T>& src2);

template<typename T>
std::vector<T> ElemwiseDivide(const std::vector<T>& src1, const std::vector<T>& src2);

template<typename T>
std::vector<T> ElemwiseMod(const std::vector<T>& src1, const std::vector<T>& src2);

template<typename T>
bool ElemwiseLessThan(const std::vector<T>& vec1, const std::vector<T>& vec2);

template<typename T>
bool ElemwiseLessThanEqualTo(const std::vector<T>& vec1, const std::vector<T>& vec2);

template<typename T>
bool AnyPositiveElem(const std::vector<T>& vec);

template<typename T>
bool AnyNegativeElem(const std::vector<T>& vec);

template<typename T>
bool AnyElemwiseGreaterThan(const std::vector<T>& vec1, const std::vector<T>& vec2);

template<typename T>
bool AnyZeroElem(const std::vector<T>& vec);

template<typename T>
bool AnyElemwiseNotEqual(const std::vector<T>& vec1, const std::vector<T>& vec2);

template<typename T>
std::vector<T> FilterVector(const std::vector<T>& vec, const std::vector<Unsigned>& filter);

template<typename T>
std::vector<T> NegFilterVector(const std::vector<T>& vec, const std::vector<Unsigned>& filter);

template<typename T>
std::vector<T> GetSuffix(const std::vector<T>& vec1, const std::vector<T>& vec2);

template<typename T>
std::vector<T> GetPrefix(const std::vector<T>& vec1, const std::vector<T>& vec2);

template<typename T>
std::vector<T> Unique(const std::vector<T>& vec);

template<typename T>
bool Contains(const std::vector<T>& vec, const T& val);

template<typename T>
int IndexOf(const std::vector<T>& vec, const T& val);

template<typename T>
std::vector<Unsigned> IndicesOf(const std::vector<T>& vec, const std::vector<T>& vals);

template<typename T>
T Min(const std::vector<T>& vec);

template<typename T>
T Max(const std::vector<T>& vec);

template<typename T>
std::vector<T> ConcatenateVectors(const std::vector<T>& vec1, const std::vector<T>& vec2);

template<typename T>
std::vector<T> DiffVector(const std::vector<T>& vec1, const std::vector<T>& vec2);

template<typename T>
std::vector<T> IsectVector(const std::vector<T>& vec1, const std::vector<T>& vec2);

template<typename T>
void SortVector(std::vector<T>& vec1);

} // namespace rote

#endif // ifndef ROTE_UTIL_VECTOR_HPP
