/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/

#include "rote.hpp"
namespace rote {

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

//NOTE: Place appropriate guards
template<typename T>
inline void
ZAxpBy( T alpha, const Tensor<T>& X, T beta, const Tensor<T>& Y, Tensor<T>& Z )
{
    Permutation perm(X.Order());
    ZAxpBy(alpha, X, perm, beta, Y, perm, Z);
}

template<typename T>
inline void
ZAxpBy( T alpha, const Tensor<T>& X, const Permutation& permXToZ, T beta, const Tensor<T>& Y, const Permutation& permYToZ, Tensor<T>& Z )
{
    ZAxpByData data;
    data.loopShape = Z.Shape();
    data.src1Strides = permXToZ.applyTo(X.Strides());
    data.src2Strides = permYToZ.applyTo(Y.Strides());
    data.dstStrides = Z.Strides();

    const T* src1Buf = X.LockedBuffer();
    const T* src2Buf = Y.LockedBuffer();
    T* dstBuf = Z.Buffer();

    ZAxpBy_fast(alpha, beta, src1Buf, src2Buf, dstBuf, data);
}

////////////////////////////////////
// Global interfaces
////////////////////////////////////

template<typename T>
inline void
ZAxpBy( T alpha, const DistTensor<T>& X, T beta, const DistTensor<T>& Y, DistTensor<T>& Z )
{
#ifndef RELEASE
    if( X.Grid() != Y.Grid() )
        LogicError
        ("X and Y must be distributed over the same grid");
#endif
    Z.ResizeTo(X.Shape());

    Permutation permXToZ = X.LocalPermutation().PermutationTo(Z.LocalPermutation());
    Permutation permYToZ = Y.LocalPermutation().PermutationTo(Z.LocalPermutation());
    ZAxpBy(alpha, X.LockedTensor(), permXToZ, beta, Y.LockedTensor(), permYToZ, Z.Tensor());
}

//Non-template functions
//bool AnyFalseElem(const std::vector<bool>& vec);
#define PROTO(T) \
	template void ZAxpBy( T alpha, const DistTensor<T>& X, T beta, const DistTensor<T>& Y, DistTensor<T>& Z );


//PROTO(Unsigned)
//PROTO(Int)
PROTO(float)
PROTO(double)
//PROTO(char)

#ifndef DISABLE_COMPLEX
#ifndef DISABLE_FLOAT
PROTO(std::complex<float>)
#endif
PROTO(std::complex<double>)
#endif

}
