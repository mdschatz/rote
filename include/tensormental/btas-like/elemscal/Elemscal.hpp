/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef TMEN_BTAS_ELEMSCAL_HPP
#define TMEN_BTAS_ELEMSCAL_HPP

#include "tensormental/util/vec_util.hpp"
#include "tensormental/util/btas_util.hpp"
#include "tensormental/core/view_decl.hpp"

namespace tmen{

////////////////////////////////////
// Local routines
////////////////////////////////////

//NOTE: Add checks for conforming shapes
//NOTE: Convert to incorporate blocked tensors.
template <typename T>
void ElemScal(const Tensor<T>& A, const Tensor<T>& B, Tensor<T>& C){
#ifndef RELEASE
    CallStackEntry("Elemscal");
#endif

    Unsigned i;

    const T* bufA = A.LockedBuffer();
    const T* bufB = B.LockedBuffer();
    T* bufC = C.Buffer();

    for(i = 0; i < prod(C.Shape()); i++){
        bufC[i] = bufA[i] * bufB[i];
    }
}

////////////////////////////////////
// Global routines
////////////////////////////////////
//NOTE: Add checks for conforming dists and shapes
//NOTE: Convert to incorporate blocked tensors.
template<typename T>
void ElemScal(const DistTensor<T>& A, const DistTensor<T>& B, DistTensor<T>& C){
#ifndef RELEASE
    CallStackEntry("Elemscal");
#endif
    ElemScal(A.LockedTensor(), B.LockedTensor(), C.Tensor());
}

} // namespace tmen

#endif // ifndef TMEN_BTAS_ELEMSCAL_HPP
