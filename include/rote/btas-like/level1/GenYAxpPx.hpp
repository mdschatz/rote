/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef ROTE_BTAS_GEN_YAXPPX_HPP
#define ROTE_BTAS_GEN_YAXPPX_HPP

namespace rote{

////////////////////////////////////
// DistContract Workhorse
////////////////////////////////////

template <typename T>
bool CheckYAxpPxArgs(const DistTensor<T>& X, const Permutation& perm, const DistTensor<T>& Y);

template <typename T>
void GenYAxpPx( T alpha, const DistTensor<T>& X, T beta, const Permutation& perm, DistTensor<T>& Y );

} // namespace rote

#endif // ifndef ROTE_BTAS_GEN_YAXPPX_HPP
