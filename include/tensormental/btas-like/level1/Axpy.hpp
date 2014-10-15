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

template<typename T>
inline void
AxpyHelper(T alpha, const Tensor<T>& X, const Tensor<T>& Y, Mode mode, T const * const srcBuf,  T * const dstBuf, const AxpyData& data ){
    Unsigned i;
    const Unsigned loopEnd = data.loopShape[mode];
    const Unsigned srcStride = data.srcStrides[mode];
    const Unsigned dstStride = data.dstStrides[mode];
    Unsigned srcBufPtr = 0;
    Unsigned dstBufPtr = 0;

    if(mode == 0){
        for(i = 0; i < loopEnd; i++){
            dstBuf[dstBufPtr] += alpha * srcBuf[srcBufPtr];

            srcBufPtr += srcStride;
            dstBufPtr += dstStride;
        }
    }else{
        for(i = 0; i < loopEnd; i++){
            AxpyHelper(alpha, X, Y, mode-1, &(srcBuf[srcBufPtr]), &(dstBuf[dstBufPtr]), data);
            srcBufPtr += srcStride;
            dstBufPtr += dstStride;
        }
    }
}

template<typename T>
inline void
Axpy_fast(T alpha, const Tensor<T>& X, const Tensor<T>& Y, Mode mode, T const * const srcBuf,  T * const dstBuf, const AxpyData& data){
    const std::vector<Unsigned> loopEnd = data.loopShape;
    const std::vector<Unsigned> srcBufStrides = data.srcStrides;
    const std::vector<Unsigned> dstBufStrides = data.dstStrides;
    Unsigned srcBufPtr = 0;
    Unsigned dstBufPtr = 0;
    Unsigned order = loopEnd.size();
    Location curLoc(order, 0);
    Unsigned ptr = 0;
    Unsigned i;
//    std::string ident = "";
//    for(i = 0; i < packData.loopShape.size() - packMode; i++)
//        ident += "  ";

    if(loopEnd.size() == 0){
        dstBuf[0] = alpha * srcBuf[0];
        return;
    }

    bool done = !ElemwiseLessThan(curLoc, loopEnd);

    while(!done){

        dstBuf[dstBufPtr] = alpha * srcBuf[srcBufPtr];
        //Update
        curLoc[ptr]++;
        dstBufPtr += dstBufStrides[ptr];
        srcBufPtr += srcBufStrides[ptr];
        while(ptr < order && curLoc[ptr] >= loopEnd[ptr]){
            curLoc[ptr] = 0;

            dstBufPtr -= dstBufStrides[ptr] * (loopEnd[ptr]);
            srcBufPtr -= srcBufStrides[ptr] * (loopEnd[ptr]);
            ptr++;
            if(ptr >= order){
                done = true;
                break;
            }else{
                curLoc[ptr]++;
                dstBufPtr += dstBufStrides[ptr];
                srcBufPtr += srcBufStrides[ptr];
            }
        }
        if(done)
            break;
        ptr = 0;
    }
}

//NOTE: Place appropriate guards
template<typename T>
void
Axpy( T alpha, const Tensor<T>& X, Tensor<T>& Y )
{
#ifndef RELEASE
    CallStackEntry entry("Axpy");
#endif
    Permutation perm = DefaultPermutation(X.Order());
    Axpy(alpha, X, perm, Y);
//    Unsigned order = Y.Order();
//    AxpyData data;
//    data.loopShape = Y.Shape();
//    data.srcStrides = X.Strides();
//
//    const T* srcBuf = X.LockedBuffer();
//    T* dstBuf = Y.Buffer();
//
//    AxpyHelper(alpha, X, Y, order-1, srcBuf, dstBuf, data);
}

template<typename T>
void
Axpy( T alpha, const Tensor<T>& X, const Permutation& permXToY, Tensor<T>& Y )
{
#ifndef RELEASE
    CallStackEntry entry("Axpy");
#endif

    Unsigned order = Y.Order();
    AxpyData data;
    data.loopShape = Y.Shape();
    data.srcStrides = PermuteVector(X.Strides(), permXToY);

    const T* srcBuf = X.LockedBuffer();
    T* dstBuf = Y.Buffer();

    if(order == 0){
        dstBuf[0] = alpha * srcBuf[0];
    }else{
#ifndef RELEASE
        AxpyHelper(alpha, X, Y, order-1, srcBuf, dstBuf, data);
#else
        Axpy_fast(alpha, X, Y, order - 1, srcBuf, dstBuf, data);
#endif
    }
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
    Permutation permXToY = DeterminePermutation(X.LocalPermutation(), Y.LocalPermutation());
    Axpy(alpha, X.LockedTensor(), permXToY, Y.Tensor());
}

} // namespace tmen

#endif // ifndef TMEN_BTAS_AXPY_HPP
