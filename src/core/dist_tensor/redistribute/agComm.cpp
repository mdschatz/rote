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

template<typename T>
Int
DistTensor<T>::CheckAllGatherCommRedist(const DistTensor<T>& A, const Mode& allGatherMode, const ModeArray& redistModes){
    if(A.Order() != this->Order()){
        LogicError("CheckAllGatherRedist: Objects being redistributed must be of same order");
    }

    ModeDistribution allGatherDistA = A.ModeDist(allGatherMode);

    const ModeDistribution check = ConcatenateVectors(this->ModeDist(allGatherMode), redistModes);
    if(AnyElemwiseNotEqual(check, allGatherDistA)){
        LogicError("CheckAllGatherRedist: [Output distribution ++ redistModes] does not match Input distribution");
    }

    return true;
}

template<typename T>
void
DistTensor<T>::AllGatherCommRedist(const DistTensor<T>& A, const Mode& agMode, const ModeArray& gridModes){
#ifndef RELEASE
    CallStackEntry entry("DistTensor::AllGatherCommRedist");
    if(!CheckAllGatherCommRedist(A, agMode, gridModes))
        LogicError("AllGatherRedist: Invalid redistribution request");
#endif

    const mpi::Comm comm = this->GetCommunicatorForModes(gridModes, A.Grid());

    if(!A.Participating())
        return;

    //NOTE: Fix to handle strides in Tensor data
    if(gridModes.size() == 0){
        this->CopyLocalBuffer(A);
        return;
    }
    Unsigned sendSize, recvSize;

    //Determine buffer sizes for communication
    const Unsigned nRedistProcs = Max(1, prod(FilterVector(A.Grid().Shape(), gridModes)));
    const ObjShape maxLocalShapeA = A.MaxLocalShape();

    sendSize = prod(maxLocalShapeA);
    recvSize = sendSize * nRedistProcs;

    Memory<T> auxMemory;
    T* auxBuf = auxMemory.Require(sendSize + recvSize);
    MemZero(&(auxBuf[0]), sendSize + recvSize);

    T* sendBuf = &(auxBuf[0]);
    T* recvBuf = &(auxBuf[sendSize]);

    //printf("Alloc'd %d elems to send and %d elems to receive\n", sendSize, recvSize);
    PackAGCommSendBuf(A, agMode, sendBuf, gridModes);

    //printf("Allgathering %d elements\n", sendSize);
    mpi::AllGather(sendBuf, sendSize, recvBuf, sendSize, comm);

    if(!(this->Participating()))
        return;
    UnpackAGCommRecvBuf(recvBuf, agMode, gridModes, A);
    //Print(B.LockedTensor(), "A's local tensor after allgathering:");
}

template <typename T>
void DistTensor<T>::PackAGCommSendBuf(const DistTensor<T>& A, const Mode& agMode, T * const sendBuf, const ModeArray& redistModes)
{
  const Unsigned order = A.Order();
  const T* dataBuf = A.LockedBuffer();

  const Location zeros(order, 0);
  const Location ones(order, 1);

  const tmen::GridView gvA = A.GetGridView();

  PackData packData;
  packData.loopShape = A.LocalShape();
  packData.srcBufStrides = A.LocalStrides();

  packData.dstBufStrides = Dimensions2Strides(A.MaxLocalShape());

  packData.loopStarts = zeros;
  packData.loopIncs = ones;

  PackCommHelper(packData, order - 1, &(dataBuf[0]), &(sendBuf[0]));
}

template <typename T>
void DistTensor<T>::UnpackAGCommRecvBuf(const T * const recvBuf, const Mode& agMode, const ModeArray& redistModes, const DistTensor<T>& A)
{
    Unsigned order = A.Order();
    Unsigned i;
    T* dataBuf = this->Buffer();

    const tmen::Grid& g = A.Grid();
    const tmen::GridView gvA = A.GetGridView();
    const tmen::GridView gvB = GetGridView();
    const Unsigned nRedistProcs = prod(FilterVector(g.Shape(), redistModes));
    const ObjShape recvShape = A.MaxLocalShape();

    ModeArray commModes = redistModes;
    std::sort(commModes.begin(), commModes.end());
    const ObjShape redistShape = FilterVector(Grid().Shape(), redistModes);
    const ObjShape commShape = FilterVector(Grid().Shape(), commModes);
    const Permutation redistPerm = DeterminePermutation(commModes, redistModes);

    const Unsigned nCommElemsPerProc = prod(recvShape);
    const Unsigned agModeStride = LocalModeStride(agMode);
//    printf("recvBuf:");
//    for(Unsigned i = 0; i < nCommElemsPerProc * nRedistProcs; i++){
//        printf(" %d", recvBuf[i]);
//    }
//    printf("\n");

    const Location zeros(order, 0);
    const Location ones(order, 1);

    PackData unpackData;
    unpackData.loopShape = LocalShape();
    unpackData.dstBufStrides = LocalStrides();
    unpackData.dstBufStrides[agMode] *= nRedistProcs;

    unpackData.srcBufStrides = Dimensions2Strides(recvShape);

    unpackData.loopStarts = zeros;
    unpackData.loopIncs = ones;
    unpackData.loopIncs[agMode] = nRedistProcs;

    //NOTE: Check
    for(i = 0; i < nRedistProcs; i++){
        const Location elemCommLoc = LinearLoc2Loc(i, commShape);
        const Unsigned elemRedistLinLoc = Loc2LinearLoc(FilterVector(elemCommLoc, redistPerm), redistShape);

        if(elemRedistLinLoc >= LocalDimension(agMode))
            continue;
        unpackData.loopStarts[agMode] = elemRedistLinLoc;

        PackCommHelper(unpackData, order - 1, &(recvBuf[i * nCommElemsPerProc]), &(dataBuf[elemRedistLinLoc * agModeStride]));

    }

//    printf("dataBuf:");
//    for(Unsigned i = 0; i < prod(LocalShape()); i++)
//        printf(" %d", dataBuf[i]);
//    printf("\n");
}

#define PROTO(T) \
        template void DistTensor<T>::AllGatherCommRedist(const DistTensor<T>& A, const Mode& agMode, const ModeArray& gridModes); \
        template Int  DistTensor<T>::CheckAllGatherCommRedist(const DistTensor<T>& A, const Mode& allGatherMode, const ModeArray& redistModes); \
        template void DistTensor<T>::PackAGCommSendBuf(const DistTensor<T>& A, const Mode& allGatherMode, T * const sendBuf, const ModeArray& redistModes); \
        template void DistTensor<T>::UnpackAGCommRecvBuf(const T * const recvBuf, const Mode& allGatherMode, const ModeArray& redistModes, const DistTensor<T>& A);

//template Int CheckAllGatherCommRedist(const DistTensor<T>& B, const DistTensor<T>& A, const Mode& allGatherMode, const ModeArray& redistModes);
//template void AllGatherCommRedist(DistTensor<T>& B, const DistTensor<T>& A, const Mode& allGatherMode, const ModeArray& redistModes );



PROTO(int)
PROTO(float)
PROTO(double)
PROTO(Complex<double>)
PROTO(Complex<float>)

} //namespace tmen
