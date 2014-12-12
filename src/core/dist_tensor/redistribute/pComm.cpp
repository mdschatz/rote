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
    const Unsigned BOrder = Order();

    //Test order retained
    if(BOrder != AOrder){
        LogicError("CheckPermutationRedist: Permutation retains the same order of objects");
    }

    //Test dimension has been resized correctly
    //NOTE: Uses fancy way of performing Ceil() on integer division
    if(Dimension(permuteMode) != A.Dimension(permuteMode))
        LogicError("CheckPartialReduceScatterRedist: Permutation retains the same dimension of indices");

    //Make sure all indices are distributed similarly
    for(i = 0; i < BOrder; i++){
        Mode mode = i;
        if(mode == permuteMode){
            if(!EqualUnderPermutation(ModeDist(mode), A.ModeDist(mode)))
                LogicError("CheckPermutationRedist: Distribution of permuted mode does not involve same modes of grid as input");
        }else{
            if(AnyElemwiseNotEqual(ModeDist(mode), A.ModeDist(mode)))
                LogicError("CheckPartialReduceScatterRedist: All modes must be distributed similarly");
        }
    }
    return 1;

}

template <typename T>
void DistTensor<T>::PermutationCommRedist(const DistTensor<T>& A, const ModeArray& commModes){
//    if(!CheckPermutationCommRedist(A, permuteMode, commModes))
//            LogicError("PermutationRedist: Invalid redistribution request");

    const mpi::Comm comm = GetCommunicatorForModes(commModes, A.Grid());
    if(!A.Participating())
        return;

    Unsigned sendSize, recvSize;

    const tmen::Grid& g = A.Grid();
    const tmen::GridView gvA = A.GetGridView();
    const tmen::GridView gvB = GetGridView();

    //Determine buffer sizes for communication
    //NOTE: Next line is example of clang not detecting dead code/unused var.
//    const ObjShape gridViewSlice = FilterVector(A.GridViewShape(), A.ModeDist(permuteMode));
    const ObjShape commDataShape = MaxLocalShape();
    recvSize = prod(commDataShape);
    sendSize = recvSize;

    const int myRank = mpi::CommRank(comm);

    T* auxBuf = this->auxMemory_.Require(sendSize + recvSize);
    T* sendBuf = &(auxBuf[0]);
    T* recvBuf = &(auxBuf[sendSize]);

//    const T* dataBuf = A.LockedBuffer();
//    PrintArray(dataBuf, A.LocalShape(), A.LocalStrides(), "srcBuf");

    //Pack the data
    PROFILE_SECTION("PermutationPack");
    PackAGCommSendBuf(A, sendBuf);
    PROFILE_STOP;

//    ObjShape sendShape = commDataShape;
//    PrintArray(sendBuf, sendShape, "sendBuf");

    //Determine who I send+recv data from
    PROFILE_SECTION("PermutationComm");

    ModeArray sortedCommModes = commModes;
    std::sort(sortedCommModes.begin(), sortedCommModes.end());

    const Location myGridViewLocA = gvA.ParticipatingLoc();
    const Location sendLoc = GridViewLoc2GridLoc(myGridViewLocA, gvB);
//    const Unsigned sendLinLoc = Loc2LinearLoc(FilterVector(sendLoc, sortedCommModes), FilterVector(g.Shape(), sortedCommModes));

    const Location myGridViewLocB = gvB.ParticipatingLoc();
    const Location recvLoc = GridViewLoc2GridLoc(myGridViewLocB, gvA);
//    const Unsigned recvLinLoc = Loc2LinearLoc(FilterVector(recvLoc, sortedCommModes), FilterVector(g.Shape(), sortedCommModes));

    //Make sure we account for alignments
    //Ripped from AlignCommBufRedist
    Location firstOwnerA = GridViewLoc2GridLoc(A.Alignments(), gvA);
    Location firstOwnerB = GridViewLoc2GridLoc(Alignments(), gvB);
    std::vector<Unsigned> alignDiff = ElemwiseSubtract(firstOwnerA, firstOwnerB);

    Location alignedSendGridLoc = ElemwiseMod(ElemwiseSum(ElemwiseSubtract(sendLoc, alignDiff), g.Shape()), g.Shape());
    Location alignedRecvGridLoc = ElemwiseMod(ElemwiseSum(ElemwiseSum(recvLoc, alignDiff), g.Shape()), g.Shape());

    ModeArray misalignedModes;
    for(Unsigned i = 0; i < firstOwnerB.size(); i++){
        if(firstOwnerB[i] != firstOwnerA[i]){
            misalignedModes.insert(misalignedModes.end(), i);
        }
    }

    ModeArray actualCommModes = misalignedModes;
    for(Unsigned i = 0; i < commModes.size(); i++){
        if(std::find(actualCommModes.begin(), actualCommModes.end(), commModes[i]) == actualCommModes.end()){
            actualCommModes.insert(actualCommModes.end(), commModes[i]);
        }
    }
    std::sort(actualCommModes.begin(), actualCommModes.end());

    mpi::Comm sendRecvComm = GetCommunicatorForModes(misalignedModes, g);

    Location alignedSendSliceLoc = FilterVector(alignedSendGridLoc, actualCommModes);
    Location alignedRecvSliceLoc = FilterVector(alignedRecvGridLoc, actualCommModes);
    ObjShape gridSliceShape = FilterVector(g.Shape(), actualCommModes);

    Unsigned sendLinLoc = Loc2LinearLoc(alignedSendSliceLoc, gridSliceShape);
    Unsigned recvLinLoc = Loc2LinearLoc(alignedRecvSliceLoc, gridSliceShape);

    mpi::SendRecv(sendBuf, sendSize, sendLinLoc,
                  recvBuf, recvSize, recvLinLoc, comm);

    PROFILE_STOP;

//    ObjShape recvShape = commDataShape;
//    PrintArray(recvBuf, recvShape, "recvBuf");

    if(!(Participating())){
        this->auxMemory_.Release();
        return;
    }

    //Unpack the data (if participating)
    PROFILE_SECTION("PermutationUnpack");
    UnpackPCommRecvBuf(recvBuf, A);
    PROFILE_STOP;

//    const T* myBuf = LockedBuffer();
//    PrintArray(myBuf, LocalShape(), LocalStrides(), "myBuf");

    this->auxMemory_.Release();
}

template <typename T>
void DistTensor<T>::UnpackPCommRecvBuf(const T * const recvBuf, const DistTensor<T>& A)
{
    const Unsigned order = Order();
    T* dataBuf = Buffer();

    const Location zeros(order, 0);
    const Location ones(order, 1);

    PackData unpackData;
    unpackData.loopShape = LocalShape();
    unpackData.dstBufStrides = LocalStrides();
    unpackData.srcBufStrides = PermuteVector(Dimensions2Strides(MaxLocalShape()), localPerm_);
    unpackData.loopStarts = zeros;
    unpackData.loopIncs = ones;

    PackCommHelper(unpackData, order - 1, &(recvBuf[0]), &(dataBuf[0]));
}

#define PROTO(T) template class DistTensor<T>
#define COPY(T) \
  template DistTensor<T>::DistTensor( const DistTensor<T>& A )
#define FULL(T) \
  PROTO(T);


FULL(Int);
#ifndef DISABLE_FLOAT
FULL(float);
#endif
FULL(double);

#ifndef DISABLE_COMPLEX
#ifndef DISABLE_FLOAT
FULL(std::complex<float>);
#endif
FULL(std::complex<double>);
#endif

} //namespace tmen
