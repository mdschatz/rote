/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef ROTE_BTAS_PERMUTE_HPP
#define ROTE_BTAS_PERMUTE_HPP

namespace rote{

////////////////////////////////////
// Workhorse routines
////////////////////////////////////

template<typename T>
void PackCommHelper_fast(const PackData& packData, T const * const srcBuf, T * const dstBuf){
    if(packData.loopShape.size() == 0){
        dstBuf[0] = srcBuf[0];
        return;
    }

    const std::vector<Unsigned> loopEnd = packData.loopShape;
    const std::vector<Unsigned> dstBufStrides = packData.dstBufStrides;
    const std::vector<Unsigned> srcBufStrides = packData.srcBufStrides;
    Unsigned order = loopEnd.size();
    Location curLoc(order,0);
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
//    PROFILE_SECTION("Pack");
    PackData modifiedData = packData;
    modifiedData.loopShape = packData.loopShape;


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
    B.ResizeTo(FilterVector(A.Shape(), perm.Entries()));
    Unsigned order = A.Order();
    T* dstBuf = B.Buffer();
    const T * srcBuf = A.LockedBuffer();

    Location zeros(order, 0);
    Location ones(order, 1);
    Permutation invperm = perm.InversePermutation();
    PackData data;
    data.loopShape = A.Shape();
    data.srcBufStrides = A.Strides();
    data.dstBufStrides = invperm.applyTo(B.Strides());

    PackCommHelper(data, &(srcBuf[0]), &(dstBuf[0]));
}

////////////////////////////////////
// Global interfaces
////////////////////////////////////

template<typename T>
void Permute(const DistTensor<T>& A, DistTensor<T>& B){
    PROFILE_SECTION("Permute");
    Permutation perm = A.LocalPermutation().PermutationTo(B.LocalPermutation());
    B.ResizeTo(A.Shape());
    Permute(A.LockedTensor(), B.Tensor(), perm);
    PROFILE_STOP;
}

} // namespace rote

#endif // ifndef ROTE_BTAS_PERMUTE_HPP
