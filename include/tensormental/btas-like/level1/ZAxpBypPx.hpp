/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef TMEN_BTAS_ZAXPBYPPX_HPP
#define TMEN_BTAS_ZAXPBYPPX_HPP

namespace tmen {

////////////////////////////////////
// Workhorse routines
////////////////////////////////////

template<typename T>
inline void
ZAxpBypPx_fast(T alpha, T beta, T const * const src1Buf, T const * const src2Buf,  T const * const permSrcBuf, T * const dstBuf, const ZAxpBypPxData& data ){
    const std::vector<Unsigned> loopEnd = data.loopShape;
    const std::vector<Unsigned> src1BufStrides = data.src1Strides;
    const std::vector<Unsigned> src2BufStrides = data.src2Strides;
    const std::vector<Unsigned> permSrcBufStrides = data.permSrcStrides;
    const std::vector<Unsigned> dstBufStrides = data.dstStrides;
    Unsigned src1BufPtr = 0;
    Unsigned src2BufPtr = 0;
    Unsigned permSrcBufPtr = 0;
    Unsigned dstBufPtr = 0;
    Unsigned order = loopEnd.size();
    Location curLoc(order, 0);
    Unsigned ptr = 0;

    if(loopEnd.size() == 0){
        dstBuf[0] = alpha * src1Buf[0] + beta * src2Buf[0] + permSrcBuf[0];
        return;
    }

    bool done = !ElemwiseLessThan(curLoc, loopEnd);

    if(alpha == T(1) && beta == T(1)){

        while(!done){

            dstBuf[dstBufPtr] = src1Buf[src1BufPtr] + src2Buf[src2BufPtr] + permSrcBuf[permSrcBufPtr];
            //Update
            curLoc[ptr]++;
            dstBufPtr += dstBufStrides[ptr];
            src1BufPtr += src1BufStrides[ptr];
            src2BufPtr += src2BufStrides[ptr];
            permSrcBufPtr += permSrcBufStrides[ptr];
            while(ptr < order && curLoc[ptr] >= loopEnd[ptr]){
                curLoc[ptr] = 0;

                dstBufPtr -= dstBufStrides[ptr] * (loopEnd[ptr]);
                src1BufPtr -= src1BufStrides[ptr] * (loopEnd[ptr]);
                src2BufPtr -= src2BufStrides[ptr] * (loopEnd[ptr]);
                permSrcBufPtr -= permSrcBufStrides[ptr] * (loopEnd[ptr]);
                ptr++;
                if(ptr >= order){
                    done = true;
                    break;
                }else{
                    curLoc[ptr]++;
                    dstBufPtr += dstBufStrides[ptr];
                    src1BufPtr += src1BufStrides[ptr];
                    src2BufPtr += src2BufStrides[ptr];
                    permSrcBufPtr += permSrcBufStrides[ptr];
                }
            }
            if(done)
                break;
            ptr = 0;
        }
    }else if(alpha == T(1) && beta != T(1)){
        while(!done){

            dstBuf[dstBufPtr] = src1Buf[src1BufPtr] + beta * src2Buf[src2BufPtr] + permSrcBuf[permSrcBufPtr];
            //Update
            curLoc[ptr]++;
            dstBufPtr += dstBufStrides[ptr];
            src1BufPtr += src1BufStrides[ptr];
            src2BufPtr += src2BufStrides[ptr];
            permSrcBufPtr += permSrcBufStrides[ptr];
            while(ptr < order && curLoc[ptr] >= loopEnd[ptr]){
                curLoc[ptr] = 0;

                dstBufPtr -= dstBufStrides[ptr] * (loopEnd[ptr]);
                src1BufPtr -= src1BufStrides[ptr] * (loopEnd[ptr]);
                src2BufPtr -= src2BufStrides[ptr] * (loopEnd[ptr]);
                permSrcBufPtr -= permSrcBufStrides[ptr] * (loopEnd[ptr]);
                ptr++;
                if(ptr >= order){
                    done = true;
                    break;
                }else{
                    curLoc[ptr]++;
                    dstBufPtr += dstBufStrides[ptr];
                    src1BufPtr += src1BufStrides[ptr];
                    src2BufPtr += src2BufStrides[ptr];
                    permSrcBufPtr += permSrcBufStrides[ptr];
                }
            }
            if(done)
                break;
            ptr = 0;
        }
    }else if(alpha != T(1) && beta == T(1)){
        while(!done){

            dstBuf[dstBufPtr] = alpha * src1Buf[src1BufPtr] + src2Buf[src2BufPtr] + permSrcBuf[permSrcBufPtr];
            //Update
            curLoc[ptr]++;
            dstBufPtr += dstBufStrides[ptr];
            src1BufPtr += src1BufStrides[ptr];
            src2BufPtr += src2BufStrides[ptr];
            permSrcBufPtr += permSrcBufStrides[ptr];
            while(ptr < order && curLoc[ptr] >= loopEnd[ptr]){
                curLoc[ptr] = 0;

                dstBufPtr -= dstBufStrides[ptr] * (loopEnd[ptr]);
                src1BufPtr -= src1BufStrides[ptr] * (loopEnd[ptr]);
                src2BufPtr -= src2BufStrides[ptr] * (loopEnd[ptr]);
                permSrcBufPtr -= permSrcBufStrides[ptr] * (loopEnd[ptr]);
                ptr++;
                if(ptr >= order){
                    done = true;
                    break;
                }else{
                    curLoc[ptr]++;
                    dstBufPtr += dstBufStrides[ptr];
                    src1BufPtr += src1BufStrides[ptr];
                    src2BufPtr += src2BufStrides[ptr];
                    permSrcBufPtr += permSrcBufStrides[ptr];
                }
            }
            if(done)
                break;
            ptr = 0;
        }
    }else{
        while(!done){

            dstBuf[dstBufPtr] = alpha * src1Buf[src1BufPtr] + beta * src2Buf[src2BufPtr] + permSrcBuf[permSrcBufPtr];
            //Update
            curLoc[ptr]++;
            dstBufPtr += dstBufStrides[ptr];
            src1BufPtr += src1BufStrides[ptr];
            src2BufPtr += src2BufStrides[ptr];
            permSrcBufPtr += permSrcBufStrides[ptr];
            while(ptr < order && curLoc[ptr] >= loopEnd[ptr]){
                curLoc[ptr] = 0;

                dstBufPtr -= dstBufStrides[ptr] * (loopEnd[ptr]);
                src1BufPtr -= src1BufStrides[ptr] * (loopEnd[ptr]);
                src2BufPtr -= src2BufStrides[ptr] * (loopEnd[ptr]);
                permSrcBufPtr -= permSrcBufStrides[ptr] * (loopEnd[ptr]);
                ptr++;
                if(ptr >= order){
                    done = true;
                    break;
                }else{
                    curLoc[ptr]++;
                    dstBufPtr += dstBufStrides[ptr];
                    src1BufPtr += src1BufStrides[ptr];
                    src2BufPtr += src2BufStrides[ptr];
                    permSrcBufPtr += permSrcBufStrides[ptr];
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
ZAxpBypPx( const Tensor<T>& X, const Tensor<T>& Y, const Tensor<T>& PX, Tensor<T>& Z )
{
    ZAxpBypPx(T(1), X, T(1), Y, PX, Z);
}

template<typename T>
inline void
ZAxpBypPx( T alpha, const Tensor<T>& X, const Tensor<T>& Y, const Tensor<T>& PX, Tensor<T>& Z )
{
    ZAxpBypPx(alpha, X, T(1), Y, PX, Z);
}

template<typename T>
inline void
ZAxpBypPx( const Tensor<T>& X, T beta, const Tensor<T>& Y, const Tensor<T>& PX, Tensor<T>& Z )
{
    ZAxpBypPx(T(1), X, beta, Y, PX, Z);
}

//NOTE: Place appropriate guards
template<typename T>
inline void
ZAxpBypPx( T alpha, const Tensor<T>& X, T beta, const Tensor<T>& Y, const Tensor<T>& PX, Tensor<T>& Z )
{
#ifndef RELEASE
    CallStackEntry entry("ZAxpBy");
#endif
    Permutation perm = DefaultPermutation(X.Order());
    ZAxpBypPx(alpha, X, perm, beta, Y, perm, PX, perm, Z);
}

template<typename T>
inline void
ZAxpBypPx( T alpha, const Tensor<T>& X, const Permutation& permXToZ, T beta, const Tensor<T>& Y, const Permutation& permYToZ, const Tensor<T>& PX, const Permutation& permPXToZ, Tensor<T>& Z )
{
#ifndef RELEASE
    CallStackEntry entry("ZAxpBy");
#endif
    Unsigned order = Z.Order();
    ZAxpBypPxData data;
    data.loopShape = Z.Shape();
    data.src1Strides = PermuteVector(X.Strides(), permXToZ);
    data.src2Strides = PermuteVector(Y.Strides(), permYToZ);
    data.permSrcStrides = PermuteVector(PX.Strides(), permPXToZ);
    data.dstStrides = Z.Strides();

    const T* src1Buf = X.LockedBuffer();
    const T* src2Buf = Y.LockedBuffer();
    const T* permSrcBuf = PX.LockedBuffer();
    T* dstBuf = Z.Buffer();

    ZAxpBypPx_fast(alpha, beta, src1Buf, src2Buf, permSrcBuf, dstBuf, data);
}

////////////////////////////////////
// Global interfaces
////////////////////////////////////

template<typename T>
inline void
ZAxpBypPx( const DistTensor<T>& X, const DistTensor<T>& Y, const DistTensor<T>& PX, DistTensor<T>& Z )
{
    ZAxpBypPx(T(1), X, T(1), Y, PX, Z);
}

template<typename T>
inline void
ZAxpBypPx( T alpha, const DistTensor<T>& X, const DistTensor<T>& Y, const DistTensor<T>& PX, DistTensor<T>& Z )
{
    ZAxpBypPx(alpha, X, T(1), Y, PX, Z);
}

template<typename T>
inline void
ZAxpBypPx( const DistTensor<T>& X, T beta, const DistTensor<T>& Y, const DistTensor<T>& PX, DistTensor<T>& Z )
{
    ZAxpBypPx(T(1), X, beta, Y, PX, Z);
}

template<typename T>
inline void
ZAxpBypPx( T alpha, const DistTensor<T>& X, T beta, const DistTensor<T>& Y, const DistTensor<T>& PX, DistTensor<T>& Z )
{
#ifndef RELEASE
    CallStackEntry entry("ZAxpBy");
    if( X.Grid() != Y.Grid() )
        LogicError
        ("X and Y must be distributed over the same grid");
#endif
    Permutation permXToZ = DeterminePermutation(X.LocalPermutation(), Z.LocalPermutation());
    Permutation permYToZ = DeterminePermutation(Y.LocalPermutation(), Z.LocalPermutation());
    Permutation permPXToZ = DeterminePermutation(PX.LocalPermutation(), Z.LocalPermutation());
    ZAxpBypPx(alpha, X.LockedTensor(), permXToZ, beta, Y.LockedTensor(), permYToZ, PX.LockedTensor(), permPXToZ, Z.Tensor());
}

} // namespace tmen

#endif // ifndef TMEN_BTAS_ZAXPBYPPX_HPP
