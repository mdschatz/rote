/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef TMEN_BTAS_YXPBY_HPP
#define TMEN_BTAS_YXPBY_HPP

namespace tmen {

//Add guards
template<typename T>
inline void
YxpByHelper(const Tensor<T>& X, T beta, Tensor<T>& Y, Mode mode, T const * const srcBuf, T * const dstBuf, const YxpByData& data ){
    Unsigned i;
    const Unsigned loopEnd = data.loopShape[mode];
    const Unsigned srcStride = data.srcStrides[mode];
    const Unsigned dstStride = data.dstStrides[mode];
    Unsigned srcBufPtr = 0;
    Unsigned dstBufPtr = 0;

    if(mode == 0){
        for(i = 0; i < loopEnd; i++){
            dstBuf[dstBufPtr] = srcBuf[srcBufPtr] + beta*dstBuf[dstBufPtr];

            srcBufPtr += srcStride;
            dstBufPtr += dstStride;
        }
    }else{
        for(i = 0; i < loopEnd; i++){
            YxpByHelper(X, beta, Y, mode-1, &(srcBuf[srcBufPtr]), &(dstBuf[dstBufPtr]), data);
            srcBufPtr += srcStride;
            dstBufPtr += dstStride;
        }
    }
}

//NOTE: Place appropriate guards
//NOTE: Make this more efficient
template<typename T>
inline void
YxpBy( const Tensor<T>& X, T beta, Tensor<T>& Y )
{
#ifndef RELEASE
    CallStackEntry entry("YxpBy");
#endif
    Unsigned order = Y.Order();

    YxpByData data;
    data.loopShape = Y.Shape();
    data.srcStrides = X.Strides();
    data.dstStrides = Y.Strides();

    const T* srcBuf = X.LockedBuffer();
    T* dstBuf = Y.Buffer();

    if(order == 0)
        dstBuf[0] = srcBuf[0] + beta*dstBuf[0];
    else
        YxpByHelper(X, beta, Y, order-1, srcBuf, dstBuf, data);

}

template<typename T>
inline void
YxpBy( const DistTensor<T>& X, T beta, DistTensor<T>& Y )
{
#ifndef RELEASE
    CallStackEntry entry("YxpBy");
    if( X.Grid() != Y.Grid() )
        LogicError
        ("X and Y must be distributed over the same grid");
#endif
    YxpBy(X.LockedTensor(), beta, Y.Tensor());
}

} // namespace tmen

#endif // ifndef TMEN_BTAS_YXPBY_HPP
