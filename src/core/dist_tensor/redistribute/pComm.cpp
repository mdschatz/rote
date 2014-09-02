/*
   Copyright (c) 2009-2013, Jack Poulson
                      2013, Jeff Hammond
                      2013, Jed Brown
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#include "tensormental.hpp"
#include <algorithm>

namespace tmen{

//TODO: Check all unaffected indices are distributed similarly (Only done for CheckPermutationRedist currently)
template <typename T>
Int DistTensor<T>::CheckPermutationCommRedist(const DistTensor<T>& A, const Mode permuteMode, const ModeArray& redistModes){
    Unsigned i;
    const tmen::GridView gvA = A.GetGridView();

    const Unsigned AOrder = A.Order();
    const Unsigned BOrder = this->Order();

    //Test order retained
    if(BOrder != AOrder){
        LogicError("CheckPermutationRedist: Permutation retains the same order of objects");
    }

    //Test dimension has been resized correctly
    //NOTE: Uses fancy way of performing Ceil() on integer division
    if(this->Dimension(permuteMode) != A.Dimension(permuteMode))
        LogicError("CheckPartialReduceScatterRedist: Permutation retains the same dimension of indices");

    //Make sure all indices are distributed similarly
    for(i = 0; i < BOrder; i++){
        Mode mode = i;
        if(mode == permuteMode){
            if(!EqualUnderPermutation(this->ModeDist(mode), A.ModeDist(mode)))
                LogicError("CheckPermutationRedist: Distribution of permuted mode does not involve same modes of grid as input");
        }else{
            if(AnyElemwiseNotEqual(this->ModeDist(mode), A.ModeDist(mode)))
                LogicError("CheckPartialReduceScatterRedist: All modes must be distributed similarly");
        }
    }
    return 1;

}

template <typename T>
void DistTensor<T>::PermutationCommRedist(const DistTensor<T>& A, const Mode permuteMode, const ModeArray& redistModes){
    if(!this->CheckPermutationCommRedist(A, permuteMode, redistModes))
            LogicError("PermutationRedist: Invalid redistribution request");

    const mpi::Comm comm = this->GetCommunicatorForModes(redistModes, A.Grid());
    if(!A.Participating())
        return;

    Unsigned sendSize, recvSize;

    //Determine buffer sizes for communication
    const ObjShape gridViewSlice = FilterVector(A.GridViewShape(), A.ModeDist(permuteMode));
    const ObjShape maxLocalShapeB = MaxLengths(this->Shape(), GetGridView().ParticipatingShape());
    recvSize = prod(maxLocalShapeB);
    sendSize = recvSize;

    const int myRank = mpi::CommRank(comm);

    Memory<T> auxMemory;
    T* auxBuf = auxMemory.Require(sendSize + recvSize);
    MemZero(&(auxBuf[0]), sendSize + recvSize);
    T* sendBuf = &(auxBuf[0]);
    T* recvBuf = &(auxBuf[sendSize]);

    PackPermutationCommSendBuf(A, permuteMode, sendBuf);

    const GridView gvA = A.GetGridView();
    const GridView gvB = GetGridView();

    const ModeDistribution permuteModeDistA = A.ModeDist(permuteMode);
    const ModeDistribution permuteModeDistB = this->ModeDist(permuteMode);

    ModeDistribution gridModesUsed(permuteModeDistB);
    std::sort(gridModesUsed.begin(), gridModesUsed.end());

    const ObjShape gridSliceShape = FilterVector(A.Grid().Shape(), gridModesUsed);
    const std::vector<Unsigned> gridSliceStridesA = Dimensions2Strides(FilterVector(A.Grid().Shape(), permuteModeDistA));
    const std::vector<Unsigned> gridSliceStridesB = Dimensions2Strides(FilterVector(A.Grid().Shape(), permuteModeDistB));

    const std::vector<Unsigned> permA = DeterminePermutation(gridModesUsed, permuteModeDistA);
    const std::vector<Unsigned> permB = DeterminePermutation(gridModesUsed, permuteModeDistB);

    //Determine sendRank
    const Location sendLoc = LinearLoc2Loc(myRank, gridSliceShape, permB);
    const Unsigned sendRank = Loc2LinearLoc(FilterVector(sendLoc, permA), FilterVector(gridSliceShape, permA));
//    const Unsigned sendRank = Loc2LinearLoc(FilterVector(sendLoc, permA), FilterVector(A.Grid().Shape(), permuteModeDistA));

    //Determine recvRank
    const Location myLoc = LinearLoc2Loc(myRank, gridSliceShape, permA);
    const Unsigned recvLinearLoc = Loc2LinearLoc(FilterVector(myLoc, permB), FilterVector(gridSliceShape, permB));
//    const Unsigned recvLinearLoc = Loc2LinearLoc(FilterVector(myLoc, permB), FilterVector(A.Grid().Shape(), permuteModeDistB));
    const Location recvLoc = LinearLoc2Loc(recvLinearLoc, gridSliceShape, permA);
    const Unsigned recvRank = Loc2LinearLoc(FilterVector(recvLoc, permA), FilterVector(gridSliceShape, permA));
//    const Unsigned recvRank = Loc2LinearLoc(FilterVector(recvLoc, permA), FilterVector(A.Grid().Shape(), permuteModeDistA));

    //printf("myRank: %d sending to rank: %d, receiving from rank: %d\n", myRank, sendRank, recvRank);
    mpi::SendRecv(sendBuf, sendSize, sendRank,
                  recvBuf, recvSize, recvRank, comm);

    if(!(this->Participating()))
        return;
    UnpackPermutationCommRecvBuf(recvBuf, permuteMode, A);
}

template<typename T>
void DistTensor<T>::PackPCommSendBufHelper(const PPackData& packData, const Mode packMode, T const * const dataBuf, T * const sendBuf){

    Unsigned packSlice = packMode;
    Unsigned packSliceMaxDim = packData.sendShape[packSlice];
    Unsigned packSliceLocalDim = packData.localShape[packSlice];
    Unsigned packSliceSendBufStride = packData.sendBufModeStrides[packSlice];
    Unsigned packSliceDataBufStride = packData.dataBufModeStrides[packSlice];
    Unsigned sendBufPtr = 0;
    Unsigned dataBufPtr = 0;

    if(packMode == 0){
        if(packSliceSendBufStride == 1 && packSliceDataBufStride == 1){
            MemCopy(&(sendBuf[0]), &(dataBuf[0]), packSliceLocalDim);
        }else{
            for(packSlice = 0; packSlice < packSliceLocalDim; packSlice++){
                sendBuf[sendBufPtr] = dataBuf[dataBufPtr];
                sendBufPtr += packSliceSendBufStride;
                dataBufPtr += packSliceDataBufStride;
            }
        }
    }else{
        for(packSlice = 0; packSlice < packSliceLocalDim; packSlice++){
            PackPCommSendBufHelper(packData, packMode-1, &(dataBuf[dataBufPtr]), &(sendBuf[sendBufPtr]));
            sendBufPtr += packSliceSendBufStride;
            dataBufPtr += packSliceDataBufStride;
        }
    }
}

//NOTE: This should just be a direct memcopy. But sticking to the same structured code as all other collectives
template <typename T>
void DistTensor<T>::PackPermutationCommSendBuf(const DistTensor<T>& A, const Mode pMode, T * const sendBuf)
{
    Unsigned i;
    const Unsigned order = A.Order();
    const T* dataBuf = A.LockedBuffer();
    const tmen::GridView gvA = A.GetGridView();

    PPackData packData;
    packData.sendShape = MaxLengths(A.Shape(), gvA.ParticipatingShape());
    packData.localShape = A.LocalShape();

    packData.dataBufModeStrides = A.LocalStrides();

    packData.sendBufModeStrides.resize(order);
    packData.sendBufModeStrides = Dimensions2Strides(packData.sendShape);

    PackPCommSendBufHelper(packData, order - 1, &(dataBuf[0]), &(sendBuf[0]));
}

template<typename T>
void DistTensor<T>::UnpackPCommRecvBufHelper(const PUnpackData& unpackData, const Mode unpackMode, T const * const recvBuf, T * const dataBuf){
    Unsigned unpackSlice = unpackMode;
    Unsigned unpackSliceMaxDim = unpackData.recvShape[unpackSlice];
    Unsigned unpackSliceLocalDim = unpackData.localShape[unpackSlice];
    Unsigned unpackSliceRecvBufStride = unpackData.recvBufModeStrides[unpackSlice];
    Unsigned unpackSliceDataBufStride = unpackData.dataBufModeStrides[unpackSlice];
    Unsigned recvBufPtr = 0;
    Unsigned dataBufPtr = 0;

//    std::cout << "Unpacking mode " << unpackMode << std::endl;
//    std::cout << "agMode " << commMode << std::endl;

    if(unpackMode == 0){
        if(unpackSliceRecvBufStride == 1 && unpackSliceDataBufStride == 1){
//            std::cout << "unpacking elems" << unpackSliceLocalDim << std::endl;
            MemCopy(&(dataBuf[0]), &(recvBuf[0]), unpackSliceLocalDim);
        }else{
//            std::cout << "unpackSliceRecvBufStride" << unpackSliceRecvBufStride << std::endl;
//            std::cout << "unpackSliceDataBufStride" << unpackSliceDataBufStride << std::endl;
            for(unpackSlice = 0; unpackSlice < unpackSliceLocalDim; unpackSlice++){
                dataBuf[dataBufPtr] = recvBuf[recvBufPtr];
                recvBufPtr += unpackSliceRecvBufStride;
                dataBufPtr += unpackSliceDataBufStride;
            }
        }
    }else {
//        std::cout << "unpackSliceRecvBufStride" << unpackSliceRecvBufStride << std::endl;
//        std::cout << "unpackSliceDataBufStride" << unpackSliceDataBufStride << std::endl;
        for(unpackSlice = 0; unpackSlice < unpackSliceLocalDim; unpackSlice++){
            UnpackPCommRecvBufHelper(unpackData, unpackMode-1, &(recvBuf[recvBufPtr]), &(dataBuf[dataBufPtr]));
            recvBufPtr += unpackSliceRecvBufStride;
            dataBufPtr += unpackSliceDataBufStride;
        }
    }
}

template <typename T>
void DistTensor<T>::UnpackPermutationCommRecvBuf(const T * const recvBuf, const Mode pMode, const DistTensor<T>& A)
{
    Unsigned order = A.Order();
    T* dataBuf = this->Buffer();

    const tmen::GridView gvA = A.GetGridView();
    const tmen::GridView gvB = GetGridView();

    const ObjShape maxLocalShapeA = MaxLengths(A.Shape(), gvA.ParticipatingShape());
    const ObjShape maxLocalShapeB = MaxLengths(this->Shape(), gvB.ParticipatingShape());

    const ObjShape localShapeB = this->LocalShape();         //Shape of the local tensor we are packing

    PUnpackData unpackData;

    unpackData.recvShape = maxLocalShapeA;
    unpackData.localShape = this->LocalShape();

    unpackData.recvBufModeStrides = Dimensions2Strides(maxLocalShapeA);
    unpackData.dataBufModeStrides = LocalStrides();

    UnpackPCommRecvBufHelper(unpackData, order - 1, &(recvBuf[0]), &(dataBuf[0]));
}

#define PROTO(T) \
        template Int  DistTensor<T>::CheckPermutationCommRedist(const DistTensor<T>& A, const Mode permuteMode, const ModeArray& redistModes); \
        template void DistTensor<T>::PermutationCommRedist(const DistTensor<T>& A, const Mode permuteMode, const ModeArray& redistModes); \
        template void DistTensor<T>::PackPermutationCommSendBuf(const DistTensor<T>& A, const Mode permuteMode, T * const sendBuf); \
        template void DistTensor<T>::UnpackPermutationCommRecvBuf(const T * const recvBuf, const Mode permuteMode, const DistTensor<T>& A);

PROTO(int)
PROTO(float)
PROTO(double)
PROTO(Complex<float>)
PROTO(Complex<double>)

} //namespace tmen
