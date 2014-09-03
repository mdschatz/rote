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
ZAxpByHelper(T alpha, const Tensor<T>& X, T beta, const Tensor<T>& Y, Mode mode, T const * const src1Buf, T const * const src2Buf,  T * const dstBuf, const ZAxpByData& data ){
    Unsigned i;
    const Unsigned loopEnd = data.loopShape[mode];
    const Unsigned src1Stride = data.src1Strides[mode];
    const Unsigned src2Stride = data.src2Strides[mode];
    const Unsigned dstStride = data.dstStrides[mode];
    Unsigned src1BufPtr = 0;
    Unsigned src2BufPtr = 0;
    Unsigned dstBufPtr = 0;

    if(mode == 0){
        for(i = 0; i < loopEnd; i++){
            dstBuf[dstBufPtr] = alpha * src1Buf[src1BufPtr] + beta*src2Buf[src2BufPtr];

            src1BufPtr += src1Stride;
            src2BufPtr += src2Stride;
            dstBufPtr += dstStride;
        }
    }else{
        for(i = 0; i < loopEnd; i++){
            ZAxpByHelper(alpha X, beta, Y, mode-1, &(src1Buf[src1BufPtr]), &(src2Buf[src2BufPtr]), &(dstBuf[dstBufPtr]), data);
            src1BufPtr += src1Stride;
            src2BufPtr += src2Stride;
            dstBufPtr += dstStride;
        }
    }
}

//NOTE: Place appropriate guards
//NOTE: Make this more efficient
template<typename T>
inline void
ZAxpBy( T alpha, const Tensor<T>& X, T beta, const Tensor<T>& Y, Tensor<T>& Z )
{
#ifndef RELEASE
    CallStackEntry entry("ZAxpBy");
#endif
    Unsigned order = Z.Order();
    ZAxpByData data;
    data.loopShape = Z.Shape();
    data.src1Strides = X.Strides();
    data.src2Strides = Y.Strides();
    data.dstStrides = Z.Strides();

    const T* src1Buf = X.LockedBuffer();
    const T* src2Buf = Y.LockedBuffer();
    T* dstBuf = Z.Buffer();

    ZAxpByHelper(alpha, X, beta, Y, order-1, src1Buf, src2Buf, dstBuf, data);

}

template<typename T>
inline void
ZAxpBy( T alpha, const DistTensor<T>& X, T beta, const DistTensor<T>& Y, DistTensor<T>& Z )
{
#ifndef RELEASE
    CallStackEntry entry("ZAxpBy");
    if( X.Grid() != Y.Grid() )
        LogicError
        ("X and Y must be distributed over the same grid");
#endif
    ZAxpBy(alpha, X.LockedTensor(), beta, Y.LockedTensor(), Z.Tensor());
}

} // namespace tmen

#endif // ifndef TMEN_BTAS_YXPBY_HPP
