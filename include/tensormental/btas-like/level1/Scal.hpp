/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef TMEN_BTAS_SCAL_HPP
#define TMEN_BTAS_SCAL_HPP

namespace tmen {

template<typename T>
inline void
ScalHelper(T alpha, Tensor<T>& X, Mode mode, T * srcBuf, const ScalData& data ){
    Unsigned i;
    const Unsigned loopEnd = data.loopShape[mode];
    const Unsigned srcStride = data.srcStrides[mode];
    Unsigned srcBufPtr = 0;

    if(mode == 0){
        for(i = 0; i < loopEnd; i++){
            srcBuf[srcBufPtr] *= alpha;

            srcBufPtr += srcStride;
        }
    }else{
        for(i = 0; i < loopEnd; i++){
            ScalHelper(alpha, X, mode-1, &(srcBuf[srcBufPtr]), data);
            srcBufPtr += srcStride;
        }
    }
}

//NOTE: Place appropriate guards
template<typename T>
void
Scal( T alpha, Tensor<T>& X )
{
#ifndef RELEASE
    CallStackEntry entry("Axpy");
#endif

    Unsigned order = X.Order();
    ScalData data;
    data.loopShape = X.Shape();
    data.srcStrides = X.Strides();

    T* srcBuf = X.Buffer();

    if(order == 0){
        srcBuf[0] *= alpha;
    }else{
        ScalHelper(alpha, X, order-1, srcBuf, data);
    }
}

template<typename T>
void
Scal( T alpha, DistTensor<T>& X )
{
#ifndef RELEASE
    CallStackEntry entry("Axpy");
#endif
    Scal(alpha, X.Tensor());
}

} // namespace tmen

#endif // ifndef TMEN_BTAS_AXPY_HPP
