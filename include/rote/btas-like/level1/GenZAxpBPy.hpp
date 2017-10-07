/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef ROTE_BTAS_GEN_ZAXPBPY_HPP
#define ROTE_BTAS_GEN_ZAXPBPY_HPP

#include "../level1/Permute.hpp"
#include "rote/core/view_decl.hpp"
#include "rote/io/Print.hpp"
#include "rote/util/btas_util.hpp"
#include "rote/util/vec_util.hpp"

namespace rote {

////////////////////////////////////
// DistContract Workhorse
////////////////////////////////////

template <typename T>
bool CheckZAxpBPyArgs(
  const DistTensor<T> &X,
  const DistTensor<T> &Y,
  const Permutation &perm,
  const DistTensor<T> &Z
);

template <typename T>
void GenZAxpBPy(
  T alpha, const DistTensor<T> &X,
  T beta, const DistTensor<T> &Y,
  const Permutation &perm, DistTensor<T> &Z
);

} // namespace rote

#endif // ifndef ROTE_BTAS_GEN_ZAXPBPY_HPP
