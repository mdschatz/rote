/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef TMEN_BTAS_ZERO_HPP
#define TMEN_BTAS_ZERO_HPP

namespace tmen {

template<typename T>
inline void
Zero( Tensor<T>& A )
{
#ifndef RELEASE
    CallStackEntry entry("Zero");
#endif
    const Int numElem = prod(A.Shape());
    //PARALLEL_FOR
    MemZero( A.Buffer(), numElem );
}

template<typename T>
inline void
Zero( DistTensor<T>& A )
{
#ifndef RELEASE
    CallStackEntry entry("Zero");
#endif
    Zero( A.Tensor() );
}

} // namespace tmen

#endif // ifndef TMEN_BTAS_ZERO_HPP
