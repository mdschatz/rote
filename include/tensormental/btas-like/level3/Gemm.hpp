/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef TMEN_BLAS_GEMM_HPP
#define TMEN_BLAS_GEMM_HPP

#include "tensormental/core/tensor_forward_decl.hpp"

namespace tmen {

template<typename T>
inline void
Gemm
( T alpha, const Tensor<T>& A, const Tensor<T>& B, T beta, Tensor<T>& C )
{
#ifndef RELEASE
    CallStackEntry entry("Gemm");
    if(A.Order() != 2 || B.Order() != 2 || C.Order() != 2)
        LogicError("Performing Gemm on non-matrix objects");
#endif
    const Int m = C.Dimension(0);
    const Int n = C.Dimension(1);
    const Int k = A.Dimension(1);
//    if( k != 0 )
//    {
        blas::Gemm
        ( 'N', 'N', m, n, k,
          alpha, A.LockedBuffer(), A.LDim(1), B.LockedBuffer(), B.LDim(1),
          beta,  C.Buffer(),       C.LDim(1) );
//    }
//    else
//    {
//        Scale( beta, C );
//    }
}

template<typename T>
inline void
Gemm
( T alpha, const Tensor<T>& A, const Tensor<T>& B, Tensor<T>& C )
{
#ifndef RELEASE
    CallStackEntry entry("Gemm");
#endif
    Zeros( C );
    Gemm( alpha, A, B, T(0), C );
}


} // namespace tmen

#endif // ifndef TMEN_BLAS_GEMM_HPP
