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
void
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

////////////////////////////////////
// Local interfaces
////////////////////////////////////

//NOTE: Place appropriate guards
template<typename T>
void
YAxpPx( T alpha, const Tensor<T>& X, T beta, const Tensor<T>& PX, const Permutation& perm, Tensor<T>& Y )
{
#ifndef RELEASE
    CallStackEntry entry("YAxpPx");
#endif
    Permutation permXToY = DefaultPermutation(X.Order());
    YAxpPx(alpha, X, permXToY, beta, PX, perm, Y);
}

template<typename T>
void
YAxpPx( T alpha, const Tensor<T>& X, const Permutation& permXToY, T beta, const Tensor<T>& PX, const Permutation& permPXToY, Tensor<T>& Y )
{
#ifndef RELEASE
    CallStackEntry entry("YAxpPx");
#endif
    YAxpPxData data;
    data.loopShape = Y.Shape();
    data.srcStrides = PermuteVector(X.Strides(), permXToY);
    data.permSrcStrides = PermuteVector(PX.Strides(), permPXToY);
    data.dstStrides = Y.Strides();

    const T* srcBuf = X.LockedBuffer();
    const T* permSrcBuf = PX.LockedBuffer();
    T* dstBuf = Y.Buffer();

//    PrintArray(srcBuf, X.Shape(), X.Strides(), "srcBuf");
//    PrintArray(permSrcBuf, PX.Shape(), PX.Strides(), "permSrcBuf");
    YAxpPx_fast(alpha, beta, srcBuf, permSrcBuf, dstBuf, data);
}

////////////////////////////////////
// Global interfaces
////////////////////////////////////

template<typename T>
void
YAxpPx( T alpha, const DistTensor<T>& X, T beta, const DistTensor<T>& PX, const Permutation& perm, DistTensor<T>& Y )
{
#ifndef RELEASE
    CallStackEntry entry("YAxpPx");
    if( X.Grid() != Y.Grid() )
        LogicError
        ("X and Y must be distributed over the same grid");
#endif
    Permutation permXToY = X.LocalPermutation().PermutationTo(Y.LocalPermutation());
    Permutation invPermPX = PX.LocalPermutation().InversePermutation();
    Permutation invPermPXToDefY = invPermPX.PermutationTo(perm);
    Permutation permPXToY = invPermPXToDefY.PermutationTo(Y.LocalPermutation());
    //NOTE: Before change to utilize local permutation
    YAxpPx(alpha, X.LockedTensor(), permXToY, beta, PX.LockedTensor(), permPXToY, Y.Tensor());
}

//Non-template functions
//bool AnyFalseElem(const std::vector<bool>& vec);
#define PROTO(T) \
	template void YAxpPx( T alpha, const DistTensor<T>& X, T beta, const DistTensor<T>& PX, const Permutation& perm, DistTensor<T>& Y );

//PROTO(Unsigned)
//PROTO(Int)
PROTO(float)
PROTO(double)
//PROTO(char)

} // namespace rote
