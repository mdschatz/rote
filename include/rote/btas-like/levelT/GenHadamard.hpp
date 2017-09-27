/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef ROTE_BTAS_GEN_HADAMARD_HPP
#define ROTE_BTAS_GEN_HADAMARD_HPP

#include "../level1/Permute.hpp"
#include "rote/util/btas_util.hpp"
#include "rote/util/vec_util.hpp"
#include "rote/util/btas_util.hpp"
#include "rote/core/view_decl.hpp"
#include "rote/io/Print.hpp"

namespace rote{

////////////////////////////////////
// DistHadamard Workhorses
////////////////////////////////////

template <typename T>
void HadamardStatA(
  const DistTensor<T>& A, const IndexArray& indicesA,
  const DistTensor<T>& B, const IndexArray& indicesB,
        DistTensor<T>& C, const IndexArray& indicesC,
  const std::vector<Unsigned>& blkSizes = std::vector<Unsigned>(0)
);

template <typename T>
void HadamardStatC(
  const DistTensor<T>& A, const IndexArray& indicesA,
  const DistTensor<T>& B, const IndexArray& indicesB,
        DistTensor<T>& C, const IndexArray& indicesC,
  const std::vector<Unsigned>& blkSizes = std::vector<Unsigned>(0)
);

////////////////////////////////////
// DistTensor interfaces
////////////////////////////////////

template <typename T>
void GenHadamard(
  const DistTensor<T>& A, const std::string& indicesA,
  const DistTensor<T>& B, const std::string& indicesB,
        DistTensor<T>& C, const std::string& indicesC,
  const std::vector<Unsigned>& blkSizes = std::vector<Unsigned>(0)
);

} // namespace rote

#endif // ifndef ROTE_BTAS_GEN_HADAMARD_HPP
