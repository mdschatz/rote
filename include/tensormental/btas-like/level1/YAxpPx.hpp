/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef TMEN_BTAS_YAXPPX_HPP
#define TMEN_BTAS_YAXPPX_HPP

namespace tmen {

//NOTE: Place appropriate guards
//NOTE: Make this more efficient
template<typename T>
inline void
YAxpPx( T alpha, const Tensor<T>& X, const Permutation& perm, Tensor<T>& Y )
{
#ifndef RELEASE
    CallStackEntry entry("YAxpPx");
#endif

    Tensor<T> PX(FilterVector(X.Shape(), perm));
    Axpy(alpha, X, PX);
    Y.CopyBuffer(PX);
}

template<typename T>
inline void
YAxpPx( T alpha, const DistTensor<T>& X, const Permutation& perm, DistTensor<T>& Y )
{
#ifndef RELEASE
    CallStackEntry entry("YAxpPx");
    if( X.Grid() != Y.Grid() )
        LogicError
        ("X and Y must be distributed over the same grid");
#endif
    YAxpPx(alpha, X.LockedTensor(), perm, Y.Tensor());
}

} // namespace tmen

#endif // ifndef TMEN_BTAS_YAXPPX_HPP
