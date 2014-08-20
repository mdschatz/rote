/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef TMEN_TENSORS_ZEROS_HPP
#define TMEN_TENSORS_ZEROS_HPP

#include "tensormental/btas-like/level1/Zero.hpp"

namespace tmen {

template<typename T> 
inline void
MakeZeros( Tensor<T>& A )
{
#ifndef RELEASE
    CallStackEntry cse("MakeZeros");
#endif
    Zero( A );
}

template<typename T>
inline void
MakeZeros( DistTensor<T>& A )
{
#ifndef RELEASE
    CallStackEntry cse("MakeZeros");
#endif
    Zero( A.Tensor() );
}

template<typename T>
inline void
Zeros( Tensor<T>& A, const ObjShape& shape )
{
#ifndef RELEASE
    CallStackEntry cse("Zeros");
#endif
    A.ResizeTo( shape );
    MakeZeros( A );
}

template<typename T>
inline Tensor<T>
Zeros( const ObjShape& shape )
{
    Tensor<T> A( shape );
    MakeZeros( A ); 
    return A;
}

template<typename T>
inline void
Zeros( DistTensor<T>& A, const ObjShape& shape )
{
#ifndef RELEASE
    CallStackEntry cse("Zeros");
#endif
    A.ResizeTo( shape );
    MakeZeros( A );
}

template<typename T>
inline DistTensor<T>
Zeros( const Grid& g, const ObjShape& shape )
{
    DistTensor<T> A( shape, g );
    MakeZeros( A );
    return A;
}

} // namespace tmen

#endif // ifndef TMEN_TENSORS_ZEROS_HPP
