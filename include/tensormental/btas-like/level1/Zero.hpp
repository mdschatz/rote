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
void ZeroHelper(Mode mode, const ObjShape& shape, const std::vector<Unsigned>& strides, T * const buf){
    Unsigned i;
    Unsigned bufPtr = 0;
    if(mode == 0){
        if(strides[mode] == 1)
            MemZero(&(buf[bufPtr]), shape[mode]);
        else{
            for(i = 0; i < shape[mode]; i++){
                buf[bufPtr] = 0;
                bufPtr += strides[mode];
            }
        }
    }else{
        for(i = 0; i < shape[mode]; i++){
            ZeroHelper(mode - 1, shape, strides, &(buf[bufPtr]));
            bufPtr += strides[mode];
        }
    }
}

template<typename T>
inline void
Zero( Tensor<T>& A )
{
#ifndef RELEASE
    CallStackEntry entry("Zero");
#endif
    Unsigned order = A.Order();
    if(order == 0)
        MemZero(A.Buffer(), 1);
    else
        ZeroHelper(order - 1, A.Shape(), A.Strides(), A.Buffer());

    //PARALLEL_FOR
//    MemZero( A.Buffer(), numElem );
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
