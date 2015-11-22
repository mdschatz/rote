/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef ROTE_BTAS_GEMM_HPP
#define ROTE_BTAS_GEMM_HPP

#include "rote/core/tensor_forward_decl.hpp"
#include "rote/core/imports/blas.hpp"

namespace rote {

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

    blas::Gemm
    ( 'N', 'N', m, n, k,
      alpha, A.LockedBuffer(), A.Stride(1), B.LockedBuffer(), B.Stride(1),
      beta,  C.Buffer(),       C.Stride(1) );
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


} // namespace rote

#endif // ifndef ROTE_BTAS_GEMM_HPP
