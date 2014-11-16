/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef TMEN_BTAS_ELEMSCAL_HPP
#define TMEN_BTAS_ELEMSCAL_HPP

#include "tensormental/util/vec_util.hpp"
#include "tensormental/util/btas_util.hpp"
#include "tensormental/core/view_decl.hpp"

namespace tmen{

////////////////////////////////////
// Local routines
////////////////////////////////////

template<typename T>
inline void
ElemScalHelper(const Tensor<T>& A, const Tensor<T>& B, Mode mode, T const * const src1Buf, T const * const src2Buf,  T * const dstBuf, const ElemScalData& data ){
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
            dstBuf[dstBufPtr] = src1Buf[src1BufPtr] * src2Buf[src2BufPtr];

            src1BufPtr += src1Stride;
            src2BufPtr += src2Stride;
            dstBufPtr += dstStride;
        }
    }else{
        for(i = 0; i < loopEnd; i++){
            ElemScalHelper(A, B, mode-1, &(src1Buf[src1BufPtr]), &(src2Buf[src2BufPtr]), &(dstBuf[dstBufPtr]), data);
            src1BufPtr += src1Stride;
            src2BufPtr += src2Stride;
            dstBufPtr += dstStride;
        }
    }
}

template<typename T>
inline void
ElemScal_fast(T const * const src1Buf, T const * const src2Buf,  T * const dstBuf, const ElemScalData& data ){
    const std::vector<Unsigned> loopEnd = data.loopShape;
    const std::vector<Unsigned> src1BufStrides = data.src1Strides;
    const std::vector<Unsigned> src2BufStrides = data.src2Strides;
    const std::vector<Unsigned> dstBufStrides = data.dstStrides;
    Unsigned src1BufPtr = 0;
    Unsigned src2BufPtr = 0;
    Unsigned dstBufPtr = 0;
    Unsigned order = loopEnd.size();
    Location curLoc(order, 0);
    Unsigned ptr = 0;

    if(loopEnd.size() == 0){
        dstBuf[0] = src1Buf[0] * src2Buf[0];
        return;
    }

    bool done = !ElemwiseLessThan(curLoc, loopEnd);

    while(!done){

        dstBuf[dstBufPtr] = src1Buf[src1BufPtr] * src2Buf[src2BufPtr];
        //Update
        curLoc[ptr]++;
        dstBufPtr += dstBufStrides[ptr];
        src1BufPtr += src1BufStrides[ptr];
        src2BufPtr += src2BufStrides[ptr];
        while(ptr < order && curLoc[ptr] >= loopEnd[ptr]){
            curLoc[ptr] = 0;

            dstBufPtr -= dstBufStrides[ptr] * (loopEnd[ptr]);
            src1BufPtr -= src1BufStrides[ptr] * (loopEnd[ptr]);
            src2BufPtr -= src2BufStrides[ptr] * (loopEnd[ptr]);
            ptr++;
            if(ptr >= order){
                done = true;
                break;
            }else{
                curLoc[ptr]++;
                dstBufPtr += dstBufStrides[ptr];
                src1BufPtr += src1BufStrides[ptr];
                src2BufPtr += src2BufStrides[ptr];
            }
        }
        if(done)
            break;
        ptr = 0;
    }
}

//NOTE: Add checks for conforming shapes
//NOTE: Convert to incorporate blocked tensors.
template <typename T>
void ElemScal(const Tensor<T>& A, const Tensor<T>& B, Tensor<T>& C){
#ifndef RELEASE
    CallStackEntry("Elemscal");
#endif
    ElemScalData data;
    data.loopShape = C.Shape();
    data.src1Strides = A.Strides();
    data.src2Strides = B.Strides();
    data.dstStrides = C.Strides();

    const T* src1Buf = A.LockedBuffer();
    const T* src2Buf = B.LockedBuffer();
    T* dstBuf = C.Buffer();

#ifndef RELEASE
    ElemScalHelper(A, B, C.Order()-1, src1Buf, src2Buf, dstBuf, data);
#else
    ElemScal_fast(src1Buf, src2Buf, dstBuf, data);
#endif
}

////////////////////////////////////
// Global routines
////////////////////////////////////
//NOTE: Add checks for conforming dists and shapes
template<typename T>
void ElemScal(const DistTensor<T>& A, const DistTensor<T>& B, DistTensor<T>& C){
#ifndef RELEASE
    CallStackEntry("Elemscal");
#endif
    ElemScal(A.LockedTensor(), B.LockedTensor(), C.Tensor());
}

} // namespace tmen

#endif // ifndef TMEN_BTAS_ELEMSCAL_HPP
