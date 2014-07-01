/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef TMEN_BTAS_DIFF_DECL_HPP
#define TMEN_BTAS_DIFF_DECL_HPP

namespace tmen{

///////////////////////
// Local routines
///////////////////////

template<typename T>
void Diff(const Tensor<T>& A, const Tensor<T>& B, Tensor<T>& C);

///////////////////////
// Global routines
///////////////////////

template<typename T>
void Diff(const DistTensor<T>& A, const DistTensor<T>& B, DistTensor<T>& C);

} // namespace tmen

#endif // ifndef TMEN_BTAS_DIFF_DECL_HPP
