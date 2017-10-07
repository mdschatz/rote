/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef ROTE_BTAS_ZAXPBYPPX_HPP
#define ROTE_BTAS_ZAXPBYPPX_HPP

namespace rote {

template <typename T>
void ZAxpBypPx(T alpha, const Tensor<T> &X, T beta, const Tensor<T> &Y,
               const Tensor<T> &PX, Tensor<T> &Z);

template <typename T>
void ZAxpBypPx(T alpha, const Tensor<T> &X, const Permutation &permXToZ, T beta,
               const Tensor<T> &Y, const Permutation &permYToZ,
               const Tensor<T> &PX, const Permutation &permPXToZ, Tensor<T> &Z);

template <typename T>
void ZAxpBypPx(T alpha, const DistTensor<T> &X, T beta, const DistTensor<T> &Y,
               const DistTensor<T> &PX, const Permutation &perm,
               DistTensor<T> &Z);

////////////////////////////////////
// Workhorse routines
////////////////////////////////////

} // namespace rote

#endif // ifndef ROTE_BTAS_ZAXPBYPPX_HPP
