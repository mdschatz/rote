/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef TMEN_BTAS_REDUCE_HPP
#define TMEN_BTAS_REDUCE_HPP

#include "tensormental/util/vec_util.hpp"
#include "tensormental/util/btas_util.hpp"
#include "tensormental/core/view_decl.hpp"

namespace tmen{

////////////////////////////////////
// Local routines
////////////////////////////////////

template <typename T>
void LocalReduceHelper(const Unsigned mode, const ModeArray& reduceModes, const ObjShape& reduceShape, T const * const srcBuf, const std::vector<Unsigned>& srcStrides, T * const dstBuf, const std::vector<Unsigned>& dstStrides){
//    std::cout << "before: " << dstBuf[0] << std::endl;
    Unsigned i;
    //NOTE: What about scalars?  (reduceShape.size() == 0)
    if(reduceModes.size() != 0){
        Mode reduceMode = reduceModes[mode];
        Unsigned srcBufPtr = 0;

        if(mode == 0){
            if(srcStrides[reduceMode] == 1 && dstStrides[reduceMode] == 1){
                for(i = 0; i < reduceShape[reduceMode]; i++){
                    dstBuf[0] += srcBuf[i];
                }
            }else{
                for(i = 0; i < reduceShape[reduceMode]; i++){
                    dstBuf[0] += srcBuf[srcBufPtr];
                    srcBufPtr += srcStrides[reduceMode];
//                    std::cout << "incing by: " << srcStrides[reduceMode] << std::endl;
                }
            }
        }else{
            for(i = 0; i < reduceShape[reduceMode]; i++){
                LocalReduceHelper(mode - 1, reduceModes, reduceShape, &(srcBuf[srcBufPtr]), srcStrides, &(dstBuf[0]), dstStrides);
                srcBufPtr += srcStrides[reduceMode];
            }
        }
    }
//    std::cout << "after: " << dstBuf[0] << std::endl;
}

template <typename T>
void LocalReduceElemSelectHelper(const Unsigned elemMode, const ModeArray& nonReduceModes, const ModeArray& reduceModes, const ObjShape& reduceShape, T const * const srcBuf, const std::vector<Unsigned>& srcStrides, T * const dstBuf, const std::vector<Unsigned>& dstStrides){
    Unsigned i;
    if(nonReduceModes.size() == 0){
        LocalReduceHelper(reduceModes.size() - 1, reduceModes, reduceShape, &(srcBuf[0]), srcStrides, &(dstBuf[0]), dstStrides);
    }else{
        Mode nonReduceMode = nonReduceModes[elemMode];
        Unsigned srcBufPtr = 0;
        Unsigned dstBufPtr = 0;

        for(i = 0; i < reduceShape[nonReduceMode]; i++){
            if(elemMode == 0){
                LocalReduceHelper(reduceModes.size() - 1, reduceModes, reduceShape, &(srcBuf[srcBufPtr]), srcStrides, &(dstBuf[dstBufPtr]), dstStrides);
            }else{
                LocalReduceElemSelectHelper(elemMode - 1, nonReduceModes, reduceModes, reduceShape, &(srcBuf[srcBufPtr]), srcStrides, &(dstBuf[dstBufPtr]), dstStrides);
            }
            srcBufPtr += srcStrides[nonReduceMode];
            dstBufPtr += dstStrides[nonReduceMode];
        }
    }
}

template <typename T>
void LocalReduce(Tensor<T>& B, const Tensor<T>& A, const ModeArray& reduceModes){
#ifndef RELEASE
    CallStackEntry("LocalReduce");
    if(reduceModes.size() > A.Order())
        LogicError("LocalReduce: modes must be of length <= order");

    for(Unsigned i = 0; i < reduceModes.size(); i++)
        if(reduceModes[i] >= A.Order())
            LogicError("LocalReduce: Supplied mode is out of range");
#endif
    Unsigned i;
    Unsigned order = A.Order();
//    MakeZero(B);

    ModeArray tensorModes(order);
    for(i = 0; i < order; i++)
        tensorModes[i] = i;
    ModeArray nonReduceModes = NegFilterVector(tensorModes, reduceModes);

    const std::vector<Unsigned> srcStrides = A.Strides();
    const std::vector<Unsigned> dstStrides = B.Strides();

    LocalReduceElemSelectHelper(nonReduceModes.size() - 1, nonReduceModes, reduceModes, A.Shape(), A.LockedBuffer(), A.Strides(), B.Buffer(), B.Strides());
}

template <typename T>
void LocalReduce(Tensor<T>& B, const Tensor<T>& A, const Mode& reduceMode){
    ModeArray modeArr(1);
    modeArr[0] = reduceMode;
    LocalReduce(B, A, modeArr);
}

////////////////////////////////////
// Global routines
////////////////////////////////////

//NOTE: Should we account for A and B having different local permutations?
template <typename T>
void LocalReduce(DistTensor<T>& B, const DistTensor<T>& A, const ModeArray& reduceModes){
    if(B.Participating()){
        //Account for the local data being permuted
        Permutation invPermA = DetermineInversePermutation(A.LocalPermutation());
        LocalReduce(B.Tensor(), A.LockedTensor(), FilterVector(invPermA, reduceModes));
    }
}

template <typename T>
void LocalReduce(DistTensor<T>& B, const DistTensor<T>& A, const Mode& reduceMode){
    if(B.Participating()){
        ModeArray modeArr(1);
        modeArr[0] = reduceMode;
        LocalReduce(B, A, modeArr);
    }
}
} // namespace tmen

#endif // ifndef TMEN_BTAS_CONTRACT_HPP
