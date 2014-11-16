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
void DistTensor<T>::PermutationCommRedist(const DistTensor<T>& A, const Mode permuteMode, const ModeArray& commModes){
    if(!CheckPermutationCommRedist(A, permuteMode, commModes))
            LogicError("PermutationRedist: Invalid redistribution request");

    const mpi::Comm comm = GetCommunicatorForModes(commModes, A.Grid());
    if(!A.Participating())
        return;

    Unsigned sendSize, recvSize;

    //Determine buffer sizes for communication
    const ObjShape gridViewSlice = FilterVector(A.GridViewShape(), A.ModeDist(permuteMode));
    const ObjShape maxLocalShapeB = MaxLocalShape();
    recvSize = prod(maxLocalShapeB);
    sendSize = recvSize;

    const int myRank = mpi::CommRank(comm);

    T* auxBuf = this->auxMemory_.Require(sendSize + recvSize);
    T* sendBuf = &(auxBuf[0]);
    T* recvBuf = &(auxBuf[sendSize]);

    //Pack the data
    PackAGCommSendBuf(A, sendBuf);

    //Determine who I send+recv data from
    const ModeDistribution permuteModeDistA = A.ModeDist(permuteMode);
    const ModeDistribution permuteModeDistB = ModeDist(permuteMode);

    ModeDistribution gridModesUsed(permuteModeDistB);
    std::sort(gridModesUsed.begin(), gridModesUsed.end());

    const ObjShape gridSliceShape = FilterVector(A.Grid().Shape(), gridModesUsed);

    const std::vector<Unsigned> permA = DeterminePermutation(gridModesUsed, permuteModeDistA);
    const std::vector<Unsigned> permB = DeterminePermutation(gridModesUsed, permuteModeDistB);

    //Determine sendRank
    const Location sendLoc = LinearLoc2Loc(myRank, gridSliceShape, permB);
    const Unsigned sendRank = Loc2LinearLoc(PermuteVector(sendLoc, permA), PermuteVector(gridSliceShape, permA));

    //Determine recvRank
    const Location myLoc = LinearLoc2Loc(myRank, gridSliceShape, permA);
    const Unsigned recvLinearLoc = Loc2LinearLoc(PermuteVector(myLoc, permB), PermuteVector(gridSliceShape, permB));
    const Location recvLoc = LinearLoc2Loc(recvLinearLoc, gridSliceShape, permA);
    const Unsigned recvRank = Loc2LinearLoc(PermuteVector(recvLoc, permA), PermuteVector(gridSliceShape, permA));

    //Communicate the data
    mpi::SendRecv(sendBuf, sendSize, sendRank,
                  recvBuf, recvSize, recvRank, comm);

    if(!(Participating())){
        this->auxMemory_.Release();
        return;
    }

    //Unpack the data (if participating)
    UnpackRSCommRecvBuf(recvBuf, A);
    this->auxMemory_.Release();
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
FULL(Complex<float>);
#endif
FULL(Complex<double>);
#endif

} //namespace tmen
