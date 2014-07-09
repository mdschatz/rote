/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/

#pragma once
#ifndef TMEN_CORE_UTIL_BTAS_UTIL_DECL_HPP
#define TMEN_CORE_UTIL_BTAS_UTIL_DECL_HPP

#include <vector>
#include <iostream>
#include "tensormental/core.hpp"

namespace tmen{

std::vector<ModeArray> DetermineContractModes(const IndexArray& indicesA, const IndexArray& indicesB, const IndexArray& indicesC);

IndexArray DetermineContractIndices(const IndexArray& indicesA, const IndexArray& indicesB);

}
#endif // ifndef TMEN_CORE_UTIL_REDISTRIBUTE_UTIL_DECL_HPP
