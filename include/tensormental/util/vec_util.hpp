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
namespace elem {

template<typename T>
T prod(const std::vector<T>& a);

template<typename T>
bool AnyNonNegativeElem(const std::vector<T>& vec);

template<typename T>
bool ElemwiseLessThan(const std::vector<T>& vec1, const std::vector<T>& vec2);

template<typename T>
bool AnyZeroElem(const std::vector<T>& vec);

template<typename T>
bool AnyElemwiseNotEqual(const std::vector<T>& vec1, const std::vector<T>& vec2);

} // namespace elem
#endif // ifndef TMEN_UTIL_VECUTIL_HPP
