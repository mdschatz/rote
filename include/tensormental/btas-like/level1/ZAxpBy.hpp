/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef TMEN_BTAS_ZAXPBY_HPP
#define TMEN_BTAS_ZAXPBY_HPP

namespace tmen {

////////////////////////////////////
// Workhorse routines
////////////////////////////////////

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
            ZAxpByHelper(alpha, X, beta, Y, mode-1, &(src1Buf[src1BufPtr]), &(src2Buf[src2BufPtr]), &(dstBuf[dstBufPtr]), data);
            src1BufPtr += src1Stride;
            src2BufPtr += src2Stride;
            dstBufPtr += dstStride;
        }
    }
}

template<typename T>
inline void
ZAxpBy_fast(T alpha, T beta, T const * const src1Buf, T const * const src2Buf,  T * const dstBuf, const ZAxpByData& data ){
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
        dstBuf[0] = alpha * src1Buf[0] + beta * src2Buf[0];
        return;
    }

    bool done = !ElemwiseLessThan(curLoc, loopEnd);

    if(alpha == T(1) && beta == T(1)){

        while(!done){

            dstBuf[dstBufPtr] = src1Buf[src1BufPtr] + src2Buf[src2BufPtr];
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
    }else if(alpha == T(1) && beta != T(1)){
        while(!done){

            dstBuf[dstBufPtr] = src1Buf[src1BufPtr] + beta * src2Buf[src2BufPtr];
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
    }else if(alpha != T(1) && beta == T(1)){
        while(!done){

            dstBuf[dstBufPtr] = alpha * src1Buf[src1BufPtr] + src2Buf[src2BufPtr];
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
    }else{
        while(!done){

            dstBuf[dstBufPtr] = alpha * src1Buf[src1BufPtr] + beta * src2Buf[src2BufPtr];
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

}

////////////////////////////////////
// Local interfaces
////////////////////////////////////

template<typename T>
inline void
Zxpy( const Tensor<T>& X, const Tensor<T>& Y, Tensor<T>& Z )
{
#ifndef RELEASE
    CallStackEntry entry("ZAxpBy");
#endif
    Permutation perm = DefaultPermutation(X.Order());
    Zxpy(X, perm, Y, perm, Z);
}

template<typename T>
inline void
Zxpy( const Tensor<T>& X, const Permutation& permXToZ, const Tensor<T>& Y, const Permutation& permYToZ, Tensor<T>& Z )
{
#ifndef RELEASE
    CallStackEntry entry("ZAxpBy");
#endif
    Unsigned order = Z.Order();
    ZAxpByData data;
    data.loopShape = Z.Shape();
    data.src1Strides = PermuteVector(X.Strides(), permXToZ);
    data.src2Strides = PermuteVector(Y.Strides(), permYToZ);
    data.dstStrides = Z.Strides();

    const T* src1Buf = X.LockedBuffer();
    const T* src2Buf = Y.LockedBuffer();
    T* dstBuf = Z.Buffer();

    if(order == 0){
        dstBuf[0] = src1Buf[0] + src2Buf[0];
    }else{
#ifndef RELEASE
        ZAxpByHelper(T(1), X, T(1), Y, order-1, src1Buf, src2Buf, dstBuf, data);
#else
        ZAxpBy_fast(T(1), T(1), src1Buf, src2Buf, dstBuf, data);
#endif
    }
}

template<typename T>
inline void
ZAxpy( T alpha, const Tensor<T>& X, const Tensor<T>& Y, Tensor<T>& Z )
{
#ifndef RELEASE
    CallStackEntry entry("ZAxpBy");
#endif
    Permutation perm = DefaultPermutation(X.Order());
    ZAxpy(alpha, X, perm, Y, perm, Z);
}

template<typename T>
inline void
ZAxpy( T alpha, const Tensor<T>& X, const Permutation& permXToZ, const Tensor<T>& Y, const Permutation& permYToZ, Tensor<T>& Z )
{
#ifndef RELEASE
    CallStackEntry entry("ZAxpBy");
#endif
    Unsigned order = Z.Order();
    ZAxpByData data;
    data.loopShape = Z.Shape();
    data.src1Strides = PermuteVector(X.Strides(), permXToZ);
    data.src2Strides = PermuteVector(Y.Strides(), permYToZ);
    data.dstStrides = Z.Strides();

    const T* src1Buf = X.LockedBuffer();
    const T* src2Buf = Y.LockedBuffer();
    T* dstBuf = Z.Buffer();

    if(order == 0){
        dstBuf[0] = alpha * src1Buf[0] + src2Buf[0];
    }else{
#ifndef RELEASE
        ZAxpByHelper(alpha, X, T(1), Y, order-1, src1Buf, src2Buf, dstBuf, data);
#else
        ZAxpBy_fast(alpha, T(1), src1Buf, src2Buf, dstBuf, data);
#endif
    }
}

template<typename T>
inline void
ZxpBy( const Tensor<T>& X, T beta, const Tensor<T>& Y, Tensor<T>& Z )
{
#ifndef RELEASE
    CallStackEntry entry("ZAxpBy");
#endif
    Permutation perm = DefaultPermutation(X.Order());
    ZxpBy(X, perm, beta, Y, perm, Z);
}

template<typename T>
inline void
ZxpBy( const Tensor<T>& X, const Permutation& permXToZ, T beta, const Tensor<T>& Y, const Permutation& permYToZ, Tensor<T>& Z )
{
#ifndef RELEASE
    CallStackEntry entry("ZAxpBy");
#endif
    Unsigned order = Z.Order();
    ZAxpByData data;
    data.loopShape = Z.Shape();
    data.src1Strides = PermuteVector(X.Strides(), permXToZ);
    data.src2Strides = PermuteVector(Y.Strides(), permYToZ);
    data.dstStrides = Z.Strides();

    const T* src1Buf = X.LockedBuffer();
    const T* src2Buf = Y.LockedBuffer();
    T* dstBuf = Z.Buffer();

    if(order == 0){
        dstBuf[0] = src1Buf[0] + beta * src2Buf[0];
    }else{
#ifndef RELEASE
        ZAxpByHelper(T(1), X, beta, Y, order-1, src1Buf, src2Buf, dstBuf, data);
#else
        ZAxpBy_fast(T(1), beta, src1Buf, src2Buf, dstBuf, data);
#endif
    }
}

//NOTE: Place appropriate guards
template<typename T>
inline void
ZAxpBy( T alpha, const Tensor<T>& X, T beta, const Tensor<T>& Y, Tensor<T>& Z )
{
#ifndef RELEASE
    CallStackEntry entry("ZAxpBy");
#endif
    Permutation perm = DefaultPermutation(X.Order());
    ZAxpBy(alpha, X, perm, beta, Y, perm, Z);
}

template<typename T>
inline void
ZAxpBy( T alpha, const Tensor<T>& X, const Permutation& permXToZ, T beta, const Tensor<T>& Y, const Permutation& permYToZ, Tensor<T>& Z )
{
#ifndef RELEASE
    CallStackEntry entry("ZAxpBy");
#endif
    Unsigned order = Z.Order();
    ZAxpByData data;
    data.loopShape = Z.Shape();
    data.src1Strides = PermuteVector(X.Strides(), permXToZ);
    data.src2Strides = PermuteVector(Y.Strides(), permYToZ);
    data.dstStrides = Z.Strides();

    const T* src1Buf = X.LockedBuffer();
    const T* src2Buf = Y.LockedBuffer();
    T* dstBuf = Z.Buffer();

    if(order == 0){
        dstBuf[0] = alpha * src1Buf[0] + beta * src2Buf[0];
    }else{
#ifndef RELEASE
        ZAxpByHelper(alpha, X, beta, Y, order-1, src1Buf, src2Buf, dstBuf, data);
#else
        ZAxpBy_fast(alpha, beta, src1Buf, src2Buf, dstBuf, data);
#endif
    }
}

////////////////////////////////////
// Global interfaces
////////////////////////////////////

template<typename T>
inline void
Zxpy( const DistTensor<T>& X, const DistTensor<T>& Y, DistTensor<T>& Z )
{
#ifndef RELEASE
    CallStackEntry entry("ZAxpBy");
    if( X.Grid() != Y.Grid() )
        LogicError
        ("X and Y must be distributed over the same grid");
#endif
    Permutation permXToZ = DeterminePermutation(X.LocalPermutation(), Z.LocalPermutation());
    Permutation permYToZ = DeterminePermutation(Y.LocalPermutation(), Z.LocalPermutation());
    ZAxpBy(X.LockedTensor(), permXToZ, Y.LockedTensor(), permYToZ, Z.Tensor());
}

template<typename T>
inline void
ZAxpy( T alpha, const DistTensor<T>& X, const DistTensor<T>& Y, DistTensor<T>& Z )
{
#ifndef RELEASE
    CallStackEntry entry("ZAxpBy");
    if( X.Grid() != Y.Grid() )
        LogicError
        ("X and Y must be distributed over the same grid");
#endif
    Permutation permXToZ = DeterminePermutation(X.LocalPermutation(), Z.LocalPermutation());
    Permutation permYToZ = DeterminePermutation(Y.LocalPermutation(), Z.LocalPermutation());
    ZAxpy(alpha, X.LockedTensor(), permXToZ, Y.LockedTensor(), permYToZ, Z.Tensor());
}

template<typename T>
inline void
ZxpBy( const DistTensor<T>& X, T beta, const DistTensor<T>& Y, DistTensor<T>& Z )
{
#ifndef RELEASE
    CallStackEntry entry("ZAxpBy");
    if( X.Grid() != Y.Grid() )
        LogicError
        ("X and Y must be distributed over the same grid");
#endif
    Permutation permXToZ = DeterminePermutation(X.LocalPermutation(), Z.LocalPermutation());
    Permutation permYToZ = DeterminePermutation(Y.LocalPermutation(), Z.LocalPermutation());
    ZxpBy(X.LockedTensor(), permXToZ, beta, Y.LockedTensor(), permYToZ, Z.Tensor());
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
    Permutation permXToZ = DeterminePermutation(X.LocalPermutation(), Z.LocalPermutation());
    Permutation permYToZ = DeterminePermutation(Y.LocalPermutation(), Z.LocalPermutation());
    ZAxpBy(alpha, X.LockedTensor(), permXToZ, beta, Y.LockedTensor(), permYToZ, Z.Tensor());
}

} // namespace tmen

#endif // ifndef TMEN_BTAS_ZAXPBY_HPP
