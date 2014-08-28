/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef TMEN_BTAS_AXPY_HPP
#define TMEN_BTAS_AXPY_HPP

namespace tmen {

//NOTE: Place appropriate guards
template<typename T>
void
Axpy( T alpha, const Tensor<T>& X, Tensor<T>& Y )
{
#ifndef RELEASE
    CallStackEntry entry("Axpy");
#endif

    Unsigned i;
    const T* bufX = X.LockedBuffer();
    T* bufY = Y.Buffer();

    for(i = 0; i < prod(X.Shape()); i++)
        bufY[i] += alpha*bufX[i];
}

template<typename T>
void
Axpy( T alpha, const DistTensor<T>& X, DistTensor<T>& Y )
{
#ifndef RELEASE
    CallStackEntry entry("Axpy");
    if( X.Grid() != Y.Grid() )
        LogicError
        ("X and Y must be distributed over the same grid");
#endif
    Axpy(alpha, X.LockedTensor(), Y.Tensor());
}

} // namespace tmen

#endif // ifndef TMEN_BTAS_AXPY_HPP
