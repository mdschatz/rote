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

//NOTE: Place appropriate guards
template<typename T>
inline void
ZAxpBypPx( T alpha, const Tensor<T>& X, T beta, const Tensor<T>& Y, const Tensor<T>& PX, Tensor<T>& Z )
{
    Permutation perm(X.Order());
    ZAxpBypPx(alpha, X, perm, beta, Y, perm, PX, perm, Z);
}

template<typename T>
inline void
ZAxpBypPx( T alpha, const Tensor<T>& X, const Permutation& permXToZ, T beta, const Tensor<T>& Y, const Permutation& permYToZ, const Tensor<T>& PX, const Permutation& permPXToZ, Tensor<T>& Z )
{
    ZAxpBypPxData data;
    data.loopShape = Z.Shape();
    data.src1Strides = permXToZ.applyTo(X.Strides());
    data.src2Strides = permYToZ.applyTo(Y.Strides());
    data.permSrcStrides = permPXToZ.applyTo(PX.Strides());
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
ZAxpBypPx( T alpha, const DistTensor<T>& X, T beta, const DistTensor<T>& Y, const DistTensor<T>& PX, const Permutation& perm, DistTensor<T>& Z )
{
#ifndef RELEASE
    if( X.Grid() != Y.Grid() )
        LogicError
        ("X and Y must be distributed over the same grid");
#endif
    Permutation permXToZ = X.LocalPermutation().PermutationTo(Z.LocalPermutation());
    Permutation permYToZ = Y.LocalPermutation().PermutationTo(Z.LocalPermutation());
    Permutation invPermPX = PX.LocalPermutation().InversePermutation();
    Permutation invPermPXToDefZ = invPermPX.PermutationTo(perm);
    Permutation permPXToZ = invPermPXToDefZ.PermutationTo(Z.LocalPermutation());
    ZAxpBypPx(alpha, X.LockedTensor(), permXToZ, beta, Y.LockedTensor(), permYToZ, PX.LockedTensor(), permPXToZ, Z.Tensor());
}

//Non-template functions
//bool AnyFalseElem(const std::vector<bool>& vec);
#define PROTO(T) \
		template void ZAxpBypPx( T alpha, const Tensor<T>& X, T beta, const Tensor<T>& Y, const Tensor<T>& PX, Tensor<T>& Z ); \
		template void ZAxpBypPx( T alpha, const DistTensor<T>& X, T beta, const DistTensor<T>& Y, const DistTensor<T>& PX, const Permutation& perm, DistTensor<T>& Z );

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

} //namespace rote
