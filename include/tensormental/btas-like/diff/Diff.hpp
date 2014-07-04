/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef TMEN_BTAS_DIFF_HPP
#define TMEN_BTAS_DIFF_HPP

#include "tensormental/util/vec_util.hpp"
#include "tensormental/util/btas_util.hpp"
#include "tensormental/core/view_decl.hpp"
#include "tensormental/core/tensor_forward_decl.hpp"

namespace tmen{

////////////////////////////////////
// Local routines
////////////////////////////////////

template <typename T>
void Diff(const Tensor<T>& A, const Tensor<T>& B, Tensor<T>& C){
#ifndef RELEASE
    CallStackEntry("Diff");
#endif
    Unsigned i;
    Unsigned order = C.Order();
    //C.ResizeTo(A);
    const T* bufA = A.LockedBuffer();
    const T* bufB = B.LockedBuffer();
    T* bufC = C.Buffer();

    //Only do this if we are sure it's a scalar
    if(order == 0)
        bufC[0] = bufA[0] - bufB[0];
    //If a tensor, could be of size 0, so we have to ignore the diff
    else{
        for(i = 0; i < prod(A.Shape()); i++){
            bufC[i] = bufA[i] - bufB[i];
        }
    }
}

////////////////////////////////////
// Global routines
////////////////////////////////////
template <typename T>
void Diff(const DistTensor<T>& A, const DistTensor<T>& B, DistTensor<T>& C){
#ifndef RELEASE
    CallStackEntry("Diff");
#endif
    //C.ResizeTo(A);
    Diff(A.LockedTensor(), B.LockedTensor(), C.Tensor());
}

} // namespace tmen

#endif // ifndef TMEN_BTAS_DIFF_HPP
