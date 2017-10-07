/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/

#pragma once
#ifndef ROTE_CORE_UTIL_BTAS_UTIL_DECL_HPP
#define ROTE_CORE_UTIL_BTAS_UTIL_DECL_HPP

#include "rote.hpp"
#include "rote/core/types_decl.hpp"
#include "rote/util/vec_util.hpp"
#include <iostream>
#include <vector>

namespace rote {

std::vector<ModeArray> DetermineContractModes(const IndexArray &indicesA,
                                              const IndexArray &indicesB,
                                              const IndexArray &indicesC);
IndexArray DetermineContractIndices(const IndexArray &indicesA,
                                    const IndexArray &indicesB);

void SetTensorShapeToMatch(const ObjShape &matchAgainst,
                           const IndexArray &indicesMatchAgainst,
                           ObjShape &toMatch, const IndexArray &indicesToMatch);

} // namespace rote
#endif // ifndef ROTE_CORE_UTIL_REDISTRIBUTE_UTIL_DECL_HPP
