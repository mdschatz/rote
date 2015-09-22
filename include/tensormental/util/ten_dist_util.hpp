/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef TMEN_UTIL_TENDISTUTIL_HPP
#define TMEN_UTIL_TENDISTUTIL_HPP

#include <vector>
#include <iostream>
#include "tensormental/core/error_decl.hpp"
#include "tensormental/core/types_decl.hpp"

namespace tmen {

bool CheckOrder(const Unsigned& outOrder, const Unsigned& inOrder);

bool CheckPartition(const TensorDistribution& outDist, const TensorDistribution& inDist);

bool CheckSameNonDist(const TensorDistribution& outDist, const TensorDistribution& inDist);

bool CheckSameCommModes(const TensorDistribution& outDist, const TensorDistribution& inDist);

bool CheckInIsPrefix(const TensorDistribution& outDist, const TensorDistribution& inDist);

bool CheckOutIsPrefix(const TensorDistribution& outDist, const TensorDistribution& inDist);

bool CheckSameGridViewShape(const ObjShape& outShape, const ObjShape& inShape);

} // namespace tmen

#endif // ifndef TMEN_UTIL_TENDISTUTIL_HPP
