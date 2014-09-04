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

//Add guards
template<typename T>
inline void
YAxpPxHelper(T alpha, const Tensor<T>& X, T beta, const Tensor<T>& PX, const Permutation& perm, Tensor<T>& Y, Mode mode, T const * const srcBuf, T const * const permSrcBuf, T * const dstBuf, const YAxpPxData& data ){
    Unsigned i;
    const Unsigned loopEnd = data.loopShape[mode];
    const Unsigned srcStride = data.srcStrides[mode];
    const Unsigned permSrcStride = data.permSrcStrides[mode];
    const Unsigned dstStride = data.dstStrides[mode];
    Unsigned srcBufPtr = 0;
    Unsigned permSrcBufPtr = 0;
    Unsigned dstBufPtr = 0;

    if(mode == 0){
        for(i = 0; i < loopEnd; i++){
            dstBuf[dstBufPtr] = alpha*srcBuf[srcBufPtr] + beta*permSrcBuf[permSrcBufPtr];

            srcBufPtr += srcStride;
            permSrcBufPtr += permSrcStride;
            dstBufPtr += dstStride;
        }
    }else{
        for(i = 0; i < loopEnd; i++){
            YAxpPxHelper(alpha, X, beta, PX, perm, Y, mode-1, &(srcBuf[srcBufPtr]), &(permSrcBuf[permSrcBufPtr]), &(dstBuf[dstBufPtr]), data);
            srcBufPtr += srcStride;
            permSrcBufPtr += permSrcStride;
            dstBufPtr += dstStride;
        }
    }
}

//NOTE: Place appropriate guards
//NOTE: Make this more efficient
template<typename T>
inline void
YAxpPx( T alpha, const Tensor<T>& X, T beta, const Tensor<T>& PX, const Permutation& perm, Tensor<T>& Y )
{
#ifndef RELEASE
    CallStackEntry entry("YAxpPx");
#endif
    Unsigned order = X.Order();
    YAxpPxData data;
    data.loopShape = X.Shape();
    data.srcStrides = X.Strides();
    data.permSrcStrides = PX.Strides();
    data.dstStrides = Y.Strides();

    const T* srcBuf = X.LockedBuffer();
    const T* permSrcBuf = PX.LockedBuffer();
    T* dstBuf = Y.Buffer();

    YAxpPxHelper(alpha, X, beta, PX, perm, Y, order-1, srcBuf, permSrcBuf, dstBuf, data);

}

template<typename T>
inline void
YAxpPx( T alpha, const DistTensor<T>& X, T beta, const DistTensor<T>& PX, const Permutation& perm, DistTensor<T>& Y )
{
#ifndef RELEASE
    CallStackEntry entry("YAxpPx");
    if( X.Grid() != Y.Grid() )
        LogicError
        ("X and Y must be distributed over the same grid");
#endif
    YAxpPx(alpha, X.LockedTensor(), beta, PX.LockedTensor(), perm, Y.Tensor());
}

} // namespace tmen

#endif // ifndef TMEN_BTAS_YAXPPX_HPP
