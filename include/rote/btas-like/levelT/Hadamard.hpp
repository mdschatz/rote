/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef ROTE_BTAS_HADAMARD_HPP
#define ROTE_BTAS_HADAMARD_HPP

#include "../level1/Permute.hpp"
#include "rote/util/btas_util.hpp"
#include "rote/util/vec_util.hpp"
#include "rote/util/btas_util.hpp"
#include "rote/core/view_decl.hpp"
#include "rote/io/Print.hpp"

namespace rote{

////////////////////////////////////
// LocalContract Workhorse
////////////////////////////////////

template <typename T>
void LocalHadamard(const Tensor<T>& A, const IndexArray& indicesA, const Tensor<T>& B, const IndexArray& indicesB, Tensor<T>& C, const IndexArray& indicesC);

} // namespace rote

#endif // ifndef ROTE_BTAS_HADAMARD_HPP
