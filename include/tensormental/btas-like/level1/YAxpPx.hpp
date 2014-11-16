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
YAxpPxHelper(T alpha, const Tensor<T>& X, T beta, const Tensor<T>& PX, Tensor<T>& Y, Mode mode, T const * const srcBuf, T const * const permSrcBuf, T * const dstBuf, const YAxpPxData& data ){
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
            YAxpPxHelper(alpha, X, beta, PX, Y, mode-1, &(srcBuf[srcBufPtr]), &(permSrcBuf[permSrcBufPtr]), &(dstBuf[dstBufPtr]), data);
            srcBufPtr += srcStride;
            permSrcBufPtr += permSrcStride;
            dstBufPtr += dstStride;
        }
    }
}

template<typename T>
inline void
YAxpPx_fast(T alpha, T beta, T const * const srcBuf, T const * const permSrcBuf, T * const dstBuf, const YAxpPxData& data ){
    const std::vector<Unsigned> loopEnd = data.loopShape;
    const std::vector<Unsigned> srcBufStrides = data.srcStrides;
    const std::vector<Unsigned> permBufStrides = data.permSrcStrides;
    const std::vector<Unsigned> dstBufStrides = data.dstStrides;
    Unsigned srcBufPtr = 0;
    Unsigned permBufPtr = 0;
    Unsigned dstBufPtr = 0;
    Unsigned order = loopEnd.size();
    Location curLoc(order, 0);
    Unsigned ptr = 0;

//    std::string ident = "";
//    for(i = 0; i < packData.loopShape.size() - packMode; i++)
//        ident += "  ";

    if(loopEnd.size() == 0){
        dstBuf[0] = alpha*srcBuf[0] + beta*permSrcBuf[0];
        return;
    }

    bool done = !ElemwiseLessThan(curLoc, loopEnd);

    while(!done){

        dstBuf[dstBufPtr] = alpha*srcBuf[srcBufPtr] + beta*permSrcBuf[permBufPtr];
        //Update
        curLoc[ptr]++;
        dstBufPtr += dstBufStrides[ptr];
        srcBufPtr += srcBufStrides[ptr];
        permBufPtr += permBufStrides[ptr];
        while(ptr < order && curLoc[ptr] >= loopEnd[ptr]){
            curLoc[ptr] = 0;

            dstBufPtr -= dstBufStrides[ptr] * (loopEnd[ptr]);
            srcBufPtr -= srcBufStrides[ptr] * (loopEnd[ptr]);
            permBufPtr -= permBufStrides[ptr] * (loopEnd[ptr]);
            ptr++;
            if(ptr >= order){
                done = true;
                break;
            }else{
                curLoc[ptr]++;
                dstBufPtr += dstBufStrides[ptr];
                srcBufPtr += srcBufStrides[ptr];
                permBufPtr += permBufStrides[ptr];
            }
        }
        if(done)
            break;
        ptr = 0;
    }
}

//NOTE: Place appropriate guards
template<typename T>
inline void
YAxpPx( T alpha, const Tensor<T>& X, T beta, const Tensor<T>& PX, const Permutation& perm, Tensor<T>& Y )
{
#ifndef RELEASE
    CallStackEntry entry("YAxpPx");
#endif
    Permutation permXToY = DefaultPermutation(X.Order());
    YAxpPx(alpha, X, permXToY, beta, PX, perm, Y);
}

template<typename T>
inline void
YAxpPx( T alpha, const Tensor<T>& X, const Permutation& permXToY, T beta, const Tensor<T>& PX, const Permutation& permPXToY, Tensor<T>& Y )
{
#ifndef RELEASE
    CallStackEntry entry("YAxpPx");
#endif
    Unsigned order = X.Order();

    YAxpPxData data;
    data.loopShape = Y.Shape();
    data.srcStrides = PermuteVector(X.Strides(), permXToY);
    data.permSrcStrides = PermuteVector(PX.Strides(), permPXToY);
    data.dstStrides = Y.Strides();

    const T* srcBuf = X.LockedBuffer();
    const T* permSrcBuf = PX.LockedBuffer();
    T* dstBuf = Y.Buffer();

    if(order == 0){
        dstBuf[0] = alpha*srcBuf[0] + beta*permSrcBuf[0];
    }else{
#ifndef RELEASE
        YAxpPxHelper(alpha, X, beta, PX, Y, order-1, srcBuf, permSrcBuf, dstBuf, data);
#else
        YAxpPx_fast(alpha, beta, srcBuf, permSrcBuf, dstBuf, data);
#endif
    }
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
    Permutation permXToY = DeterminePermutation(X.LocalPermutation(), Y.LocalPermutation());
    Permutation invPermPX = DetermineInversePermutation(PX.LocalPermutation());
    Permutation invPermPXToDefY = PermuteVector(invPermPX, perm);
    Permutation permPXToY = PermuteVector(invPermPXToDefY, Y.LocalPermutation());
    //NOTE: Before change to utilize local permutation
    YAxpPx(alpha, X.LockedTensor(), permXToY, beta, PX.LockedTensor(), permPXToY, Y.Tensor());
}

} // namespace tmen

#endif // ifndef TMEN_BTAS_YAXPPX_HPP
