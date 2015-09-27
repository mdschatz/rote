/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef TMEN_BTAS_PERMUTE_HPP
#define TMEN_BTAS_PERMUTE_HPP

#include "tensormental/io/Print.hpp"
namespace tmen{

////////////////////////////////////
// Workhorse routines
////////////////////////////////////

template<typename T>
void PackCommHelper_fast(const PackData& packData, T const * const srcBuf, T * const dstBuf){
#ifndef RELEASE
    CallStackEntry cse("PackCommHelper_fast");
#endif
    if(packData.loopShape.size() == 0){
        dstBuf[0] = srcBuf[0];
        return;
    }

    const std::vector<Unsigned> loopEnd = packData.loopShape;
    const std::vector<Unsigned> dstBufStrides = packData.dstBufStrides;
    const std::vector<Unsigned> srcBufStrides = packData.srcBufStrides;
    const std::vector<Unsigned> loopStart = packData.loopStarts;
    const std::vector<Unsigned> loopIncs = packData.loopIncs;
    Unsigned order = loopEnd.size();
    Location curLoc = loopStart;
    Unsigned dstBufPtr = 0;
    Unsigned srcBufPtr = 0;
    Unsigned ptr = 0;

    if(loopEnd.size() == 0){
        dstBuf[0] = srcBuf[0];
        return;
    }

    bool done = !ElemwiseLessThan(curLoc, loopEnd);

    if (srcBufStrides[0] == 1 && dstBufStrides[0] == 1) {
        while (!done) {
            MemCopy(&(dstBuf[dstBufPtr]), &(srcBuf[srcBufPtr]), loopEnd[0]);
            curLoc[0] += loopEnd[0];
            srcBufPtr += srcBufStrides[0] * (loopEnd[0]);
            dstBufPtr += dstBufStrides[0] * (loopEnd[0]);

            while (ptr < order && curLoc[ptr] >= loopEnd[ptr]) {
                curLoc[ptr] = 0;

                dstBufPtr -= dstBufStrides[ptr] * loopEnd[ptr];
                srcBufPtr -= srcBufStrides[ptr] * loopEnd[ptr];
                ptr++;
                if (ptr >= order) {
                    done = true;
                    break;
                } else {
                    curLoc[ptr]++;
                    dstBufPtr += dstBufStrides[ptr];
                    srcBufPtr += srcBufStrides[ptr];
                }
            }
            if (done)
                break;
            ptr = 0;
        }
    } else {
        while (!done) {
            dstBuf[dstBufPtr] = srcBuf[srcBufPtr];
            //Update
            curLoc[0]++;
            dstBufPtr += dstBufStrides[0];
            srcBufPtr += srcBufStrides[0];

            while (ptr < order && curLoc[ptr] >= loopEnd[ptr]) {
                curLoc[ptr] = 0;

                dstBufPtr -= dstBufStrides[ptr] * loopEnd[ptr];
                srcBufPtr -= srcBufStrides[ptr] * loopEnd[ptr];
                ptr++;
                if (ptr >= order) {
                    done = true;
                    break;
                } else {
                    curLoc[ptr]++;
                    dstBufPtr += dstBufStrides[ptr];
                    srcBufPtr += srcBufStrides[ptr];
                }
            }
            if (done)
                break;
            ptr = 0;
        }
    }
}

template<typename T>
void PackCommHelper(const PackData& packData, T const * const srcBuf, T * const dstBuf){
#ifndef RELEASE
    CallStackEntry cse("PackCommHelper");
#endif
//    PROFILE_SECTION("Pack");
    PackData modifiedData = packData;
    modifiedData.loopShape = IntCeils(packData.loopShape, packData.loopIncs);
    Location ones(packData.loopStarts.size(), 1);
    Location zeros(packData.loopStarts.size(), 0);
    modifiedData.loopIncs = ones;
    modifiedData.loopStarts = zeros;


    //Attempt to merge modes
    PackData newData;
    Unsigned i;
    Unsigned oldOrder = packData.loopShape.size();

    if(oldOrder == 0){
        newData = modifiedData;
    }else{
        newData.loopShape.push_back(modifiedData.loopShape[0]);
        newData.srcBufStrides.push_back(modifiedData.srcBufStrides[0]);
        newData.dstBufStrides.push_back(modifiedData.dstBufStrides[0]);
        Unsigned srcStrideToMatch = modifiedData.srcBufStrides[0] * modifiedData.loopShape[0];
        Unsigned dstStrideToMatch = modifiedData.dstBufStrides[0] * modifiedData.loopShape[0];

        Unsigned mergeMode = 0;
        for(i = 1; i < oldOrder; i++){
            if(modifiedData.srcBufStrides[i] == srcStrideToMatch &&
               modifiedData.dstBufStrides[i] == dstStrideToMatch){
                newData.loopShape[mergeMode] *= modifiedData.loopShape[i];
                srcStrideToMatch *= modifiedData.loopShape[i];
                dstStrideToMatch *= modifiedData.loopShape[i];
            }else{
                newData.loopShape.push_back(modifiedData.loopShape[i]);
                newData.srcBufStrides.push_back(modifiedData.srcBufStrides[i]);
                newData.dstBufStrides.push_back(modifiedData.dstBufStrides[i]);
                srcStrideToMatch = modifiedData.srcBufStrides[i] * modifiedData.loopShape[i];
                dstStrideToMatch = modifiedData.dstBufStrides[i] * modifiedData.loopShape[i];
                mergeMode++;
            }
        }
        std::vector<Unsigned> newones(newData.loopShape.size(), 1);
        std::vector<Unsigned> newzeros(newData.loopShape.size(), 0);
        newData.loopIncs = newones;
        newData.loopStarts = newzeros;
    }

//    PrintPackData(newData, "packData");
    PackCommHelper_fast(newData, srcBuf, dstBuf);

//    PROFILE_STOP;
}

////////////////////////////////////
// Local interfaces
////////////////////////////////////

template<typename T>
void Permute(const Tensor<T>& A, Tensor<T>& B, const Permutation& perm){
    B.ResizeTo(FilterVector(A.Shape(), perm));
    Unsigned order = A.Order();
    T* dstBuf = B.Buffer();
    const T * srcBuf = A.LockedBuffer();

    Location zeros(order, 0);
    Location ones(order, 1);
    Permutation invperm = DetermineInversePermutation(perm);
    PackData data;
    data.loopShape = A.Shape();
    data.srcBufStrides = A.Strides();
    data.dstBufStrides = PermuteVector(B.Strides(), invperm);
    data.loopStarts = zeros;
    data.loopIncs = ones;

    PackCommHelper(data, &(srcBuf[0]), &(dstBuf[0]));
}

////////////////////////////////////
// Global interfaces
////////////////////////////////////

template<typename T>
void Permute(const DistTensor<T>& A, DistTensor<T>& B){
    PROFILE_SECTION("Permute");
    Permutation perm = DeterminePermutation(A.LocalPermutation(), B.LocalPermutation());
    B.ResizeTo(A.Shape());
    Permute(A.LockedTensor(), B.Tensor(), perm);
    PROFILE_STOP;
}

} // namespace tmen

#endif // ifndef TMEN_BTAS_PERMUTE_HPP
