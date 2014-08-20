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
    if(!A.Participating())
        return;

    Unsigned sendSize, recvSize;

    //Determine buffer sizes for communication
    const ObjShape gridViewSlice = FilterVector(A.GridViewShape(), A.ModeDist(permuteMode));
    const ObjShape maxLocalShapeB = MaxLengths(this->Shape(), GetGridView().ParticipatingShape());
    recvSize = prod(maxLocalShapeB);
    sendSize = recvSize;

    const mpi::Comm comm = this->GetCommunicatorForModes(redistModes, A.Grid());
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
    const Unsigned sendRank = Loc2LinearLoc(FilterVector(sendLoc, permA), FilterVector(A.Grid().Shape(), permuteModeDistA));

    //Determine recvRank
    const Location myLoc = LinearLoc2Loc(myRank, gridSliceShape, permA);
    const Unsigned recvLinearLoc = Loc2LinearLoc(FilterVector(myLoc, permB), FilterVector(A.Grid().Shape(), permuteModeDistB));
    const Location recvLoc = LinearLoc2Loc(recvLinearLoc, gridSliceShape, permA);
    const Unsigned recvRank = Loc2LinearLoc(FilterVector(recvLoc, permA), FilterVector(A.Grid().Shape(), permuteModeDistA));

    //printf("myRank: %d sending to rank: %d, receiving from rank: %d\n", myRank, sendRank, recvRank);
    mpi::SendRecv(sendBuf, sendSize, sendRank,
                  recvBuf, recvSize, recvRank, comm);

    if(!(this->Participating()))
        return;
    UnpackPermutationCommRecvBuf(recvBuf, permuteMode, A);
}

//NOTE: This should just be a direct memcopy. But sticking to the same structured code as all other collectives
template <typename T>
void DistTensor<T>::PackPermutationCommSendBuf(const DistTensor<T>& A, const Mode pMode, T * const sendBuf)
{
    const T* dataBuf = A.LockedBuffer();

    const tmen::GridView gvA = A.GetGridView();

    //Shape of the local tensor we are packing
    const ObjShape maxLocalShapeA = MaxLengths(A.Shape(), gvA.ParticipatingShape());
    const ObjShape localShapeA = A.LocalShape();

    //Calculate number of outer slices to pack
    const Unsigned nMaxOuterSlices = Max(1, prod(maxLocalShapeA, pMode + 1));
    const Unsigned nLocalOuterSlices = prod(localShapeA, pMode + 1);

    //Calculate number of rsMode slices to pack
    const Unsigned nMaxPModeSlices = maxLocalShapeA[pMode];
    const Unsigned nLocalPModeSlices = localShapeA[pMode];
    const Unsigned maxCopySliceSize = Max(1, prod(maxLocalShapeA, 0, pMode));
    const Unsigned copySliceSize = prod(localShapeA, 0, pMode);

    Unsigned outerSliceNum, pModeSliceNum; //Which slice of which wrap of which process are we packing
    Unsigned outerSendBufOff, pModeSendBufOff;
    Unsigned outerDataBufOff, pModeDataBufOff;
    Unsigned startSendBuf, startDataBuf;

    for(outerSliceNum = 0; outerSliceNum < nMaxOuterSlices; outerSliceNum++){
        if(outerSliceNum >= nLocalOuterSlices)
            break;
        outerSendBufOff = maxCopySliceSize * nMaxPModeSlices * outerSliceNum;
        outerDataBufOff = copySliceSize * nLocalPModeSlices * outerSliceNum;

        for(pModeSliceNum = 0; pModeSliceNum < nMaxPModeSlices; pModeSliceNum++){
            if(pModeSliceNum >= nLocalPModeSlices)
                break;
            pModeSendBufOff = maxCopySliceSize * pModeSliceNum;
            pModeDataBufOff = copySliceSize * pModeSliceNum;

            startSendBuf = outerSendBufOff + pModeSendBufOff;
            startDataBuf = outerDataBufOff + pModeDataBufOff;
            MemCopy(&(sendBuf[startSendBuf]), &(dataBuf[startDataBuf]), copySliceSize);
        }
    }
}

template <typename T>
void DistTensor<T>::UnpackPermutationCommRecvBuf(const T * const recvBuf, const Mode pMode, const DistTensor<T>& A)
{
        T* dataBuf = this->Buffer();

        const tmen::GridView gvA = A.GetGridView();
        const tmen::GridView gvB = GetGridView();

        const ObjShape maxLocalShapeA = MaxLengths(A.Shape(), gvA.ParticipatingShape());
        const ObjShape maxLocalShapeB = MaxLengths(this->Shape(), gvB.ParticipatingShape());

        const ObjShape localShapeB = this->LocalShape();         //Shape of the local tensor we are packing

        //Number of outer slices to unpack
        const Unsigned nMaxOuterSlices = Max(1, prod(maxLocalShapeB, pMode + 1));
        const Unsigned nLocalOuterSlices = prod(localShapeB, pMode + 1);

        //Loop packing bounds variables
        const Unsigned nMaxPModeSlices = maxLocalShapeB[pMode];
        const Unsigned nLocalPModeSlices = localShapeB[pMode];

        //Variables for calculating elements to copy
        const Unsigned maxCopySliceSize = Max(1, prod(maxLocalShapeB, 0, pMode));
        const Unsigned copySliceSize = prod(localShapeB, 0, pMode);

        //Loop iteration vars
        Unsigned outerSliceNum, pModeSliceNum;  //Pack data for slice "sliceNum" (<nSlices) of wrap "wrapNum" (<nWraps) for proc "procSendNum" Unsigned offSliceRecvBuf, offWrapRecvBuf;  //Offsets used to index into sendBuf array
        Unsigned outerRecvBufOff, outerDataBufOff;  //Offsets used to index into recvBuf array
        Unsigned pModeRecvBufOff, pModeDataBufOff;  //Offsets used to index into dataBuf array
        Unsigned startRecvBuf, startDataBuf;

        for(outerSliceNum = 0; outerSliceNum < nMaxOuterSlices; outerSliceNum++){
            if(outerSliceNum >= nLocalOuterSlices)
                break;

            outerRecvBufOff = maxCopySliceSize * nMaxPModeSlices * outerSliceNum;
            outerDataBufOff = copySliceSize * nLocalPModeSlices * outerSliceNum;

            for(pModeSliceNum = 0; pModeSliceNum < nMaxPModeSlices; pModeSliceNum++){
                if(pModeSliceNum >= nLocalPModeSlices)
                    break;
                pModeRecvBufOff = maxCopySliceSize * pModeSliceNum;
                pModeDataBufOff = copySliceSize * pModeSliceNum;

                startRecvBuf = outerRecvBufOff + pModeRecvBufOff;
                startDataBuf = outerDataBufOff + pModeDataBufOff;
                //printf("startRecvBuf: %d startDataBuf: %d copySliceSize: %d\n", startRecvBuf, startDataBuf, copySliceSize);
                MemCopy(&(dataBuf[startDataBuf]), &(recvBuf[startRecvBuf]), copySliceSize);
            }
        }
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
