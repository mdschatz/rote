/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef ROTE_BTAS_REDUCE_HPP
#define ROTE_BTAS_REDUCE_HPP

#include "rote/util/vec_util.hpp"
#include "rote/util/btas_util.hpp"
#include "rote/core/view_decl.hpp"
#include "rote/io/Print.hpp"
#include "rote/core/time.hpp"

namespace rote{

////////////////////////////////////
// Workhorse routines
////////////////////////////////////

template <typename T>
void LocalReduceElemSelect_merged(const Unsigned nReduceModes, const ObjShape& reduceShape, T const * const srcBuf, const std::vector<Unsigned>& srcStrides, T * const dstBuf, const std::vector<Unsigned>& dstStrides){
    const std::vector<Unsigned> loopEnd = reduceShape;

    Unsigned srcBufPtr = 0;
    Unsigned dstBufPtr = 0;
    Unsigned order = loopEnd.size();
    Location curLoc(order, 0);
    Unsigned ptr = 0;

    bool done = !ElemwiseLessThan(curLoc, loopEnd);

    if(!done && order == 0){
        dstBuf[0] += srcBuf[0];
        return;
    }

//    PrintVector(loopEnd, "loopEnd");
//    PrintVector(srcStrides, "srcBufStrides");
//    PrintVector(dstStrides, "dstBufStrides");
    while(!done){
//    	PrintVector(curLoc, "curLoc");
//    	printf("dstBufPtr: %d srcBufPtr: %d\n", dstBufPtr, srcBufPtr);
        dstBuf[dstBufPtr] += srcBuf[srcBufPtr];
        //Update
        curLoc[ptr]++;
        srcBufPtr += srcStrides[ptr];
        if(ptr >= nReduceModes)
          dstBufPtr += dstStrides[ptr];
        while(ptr < order && curLoc[ptr] >= loopEnd[ptr]){
            curLoc[ptr] = 0;

            srcBufPtr -= srcStrides[ptr] * (loopEnd[ptr]);
            if(ptr >= nReduceModes)
              dstBufPtr -= dstStrides[ptr] * (loopEnd[ptr]);
            ptr++;
            if(ptr >= order){
                done = true;
                break;
            }else{
                curLoc[ptr]++;
                srcBufPtr += srcStrides[ptr];
                if(ptr >= nReduceModes)
                    dstBufPtr += dstStrides[ptr];
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

template <typename T>
void LocalReduce(const Tensor<T>& A, Tensor<T>& B, const Permutation& permBToA, const ModeArray& reduceModes){
#ifndef RELEASE
    CallStackEntry("LocalReduce");
    if(reduceModes.size() > A.Order())
        LogicError("LocalReduce: modes must be of length <= order");

    for(Unsigned i = 0; i < reduceModes.size(); i++)
        if(reduceModes[i] >= A.Order())
            LogicError("LocalReduce: Supplied mode is out of range");
#endif
    Unsigned order = A.Order();

    ModeArray tensorModes = DefaultPermutation(order);
    ModeArray nonReduceModes = NegFilterVector(tensorModes, reduceModes);

    const ObjShape shapeA = A.Shape();
    const std::vector<Unsigned> srcStrides = A.Strides();
    const std::vector<Unsigned> dstStrides = PermuteVector(B.Strides(), permBToA);

//    LocalReduceElemSelect_fast(nonReduceModes.size() - 1, nonReduceModes, reduceModes, A.Shape(), A.LockedBuffer(), srcStrides, B.Buffer(), dstStrides);
    std::vector<Unsigned> srcReduceStrides = FilterVector(srcStrides, reduceModes);
    std::vector<Unsigned> srcNonReduceStrides = NegFilterVector(srcStrides, reduceModes);
    std::vector<Unsigned> useSrcStrides = srcReduceStrides;
    useSrcStrides.insert(useSrcStrides.end(), srcNonReduceStrides.begin(), srcNonReduceStrides.end());

    std::vector<Unsigned> dstReduceStrides = FilterVector(dstStrides, reduceModes);
    std::vector<Unsigned> dstNonReduceStrides = NegFilterVector(dstStrides, reduceModes);
    std::vector<Unsigned> useDstStrides = dstReduceStrides;
    useDstStrides.insert(useDstStrides.end(), dstNonReduceStrides.begin(), dstNonReduceStrides.end());

    std::vector<Unsigned> srcReduceShape = FilterVector(shapeA, reduceModes);
    std::vector<Unsigned> srcNonReduceShape = NegFilterVector(shapeA, reduceModes);
    std::vector<Unsigned> useSrcShape = srcReduceShape;
    useSrcShape.insert(useSrcShape.end(), srcNonReduceShape.begin(), srcNonReduceShape.end());

    LocalReduceElemSelect_merged(reduceModes.size(), useSrcShape, A.LockedBuffer(), useSrcStrides, B.Buffer(), useDstStrides);
}

template <typename T>
void LocalReduce(const Tensor<T>& A, Tensor<T>& B, const ModeArray& reduceModes){
    Permutation perm(A.Order());
    for(Unsigned i = 0; i < A.Order(); i++)
        perm[i] = i;
    LocalReduce(A, B, perm, reduceModes);
}

template <typename T>
void LocalReduce(const Tensor<T>& A, Tensor<T>& B, const Mode& reduceMode){
    ModeArray modeArr(1);
    modeArr[0] = reduceMode;
    LocalReduce(A, B, modeArr);
}

////////////////////////////////////
// Global interfaces
////////////////////////////////////

template <typename T>
void LocalReduce(const DistTensor<T>& A, DistTensor<T>& B, const ModeArray& reduceModes){
    PROFILE_SECTION("LocalReduce");
    Unsigned i;
    ObjShape shapeB = A.Shape();
    for(i = 0; i < reduceModes.size(); i++)
        shapeB[reduceModes[i]] = Min(A.GetGridView().Dimension(reduceModes[i]), A.Dimension(reduceModes[i]));
    B.ResizeTo(shapeB);
    Zero(B);

    if(B.Participating()){
        //Account for the local data being permuted

        Permutation permBToA = DeterminePermutation(B.LocalPermutation(), A.LocalPermutation());
        LocalReduce(A.LockedTensor(), B.Tensor(), permBToA, FilterVector(DetermineInversePermutation(A.LocalPermutation()), reduceModes));
    }
    PROFILE_STOP;
}

template <typename T>
void LocalReduce(const DistTensor<T>& A, DistTensor<T>& B, const Mode& reduceMode){
    if(B.Participating()){
        ModeArray modeArr(1);
        modeArr[0] = reduceMode;
        LocalReduce(A, B, modeArr);
    }
}
} // namespace rote

#endif // ifndef ROTE_BTAS_CONTRACT_HPP
