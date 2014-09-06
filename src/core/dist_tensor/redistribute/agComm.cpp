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
    if(A.Order() != Order()){
        LogicError("CheckAllGatherRedist: Objects being redistributed must be of same order");
    }

    ModeDistribution allGatherDistA = A.ModeDist(allGatherMode);

    const ModeDistribution check = ConcatenateVectors(ModeDist(allGatherMode), redistModes);
    if(AnyElemwiseNotEqual(check, allGatherDistA)){
        LogicError("CheckAllGatherRedist: [Output distribution ++ redistModes] does not match Input distribution");
    }

    return true;
}

template<typename T>
void
DistTensor<T>::AllGatherCommRedist(const DistTensor<T>& A, const ModeArray& agModes, const ModeArray& commModes){
#ifndef RELEASE
    CallStackEntry entry("DistTensor::AllGatherCommRedist");
//    if(!CheckAllGatherCommRedist(A, agMode, gridModes))
//        LogicError("AllGatherRedist: Invalid redistribution request");
#endif
    Unsigned i;
    const tmen::Grid& g = A.Grid();

    const mpi::Comm comm = GetCommunicatorForModes(commModes, A.Grid());

    if(!A.Participating())
        return;

    //NOTE: Fix to handle strides in Tensor data
    if(agModes.size() == 0){
        CopyLocalBuffer(A);
        return;
    }
    Unsigned sendSize, recvSize;

    //Determine buffer sizes for communication
    const Unsigned nRedistProcs = Max(1, prod(FilterVector(g.Shape(), commModes)));
    const ObjShape maxLocalShapeA = A.MaxLocalShape();

    sendSize = prod(maxLocalShapeA);
    recvSize = sendSize * nRedistProcs;

    Memory<T> auxMemory;
    T* auxBuf = auxMemory.Require(sendSize + recvSize);
    MemZero(&(auxBuf[0]), sendSize + recvSize);

    T* sendBuf = &(auxBuf[0]);
    T* recvBuf = &(auxBuf[sendSize]);

    //printf("Alloc'd %d elems to send and %d elems to receive\n", sendSize, recvSize);
    PackAGCommSendBuf(A, sendBuf);

    //printf("Allgathering %d elements\n", sendSize);
    mpi::AllGather(sendBuf, sendSize, recvBuf, sendSize, comm);

    if(!(Participating()))
        return;

    //NOTE: AG and A2A unpack routines are the exact same
    UnpackA2ACommRecvBuf(recvBuf, agModes, commModes, maxLocalShapeA, A);
    //Print(B.LockedTensor(), "A's local tensor after allgathering:");
}

template<typename T>
void
DistTensor<T>::AllGatherCommRedist(const DistTensor<T>& A, const ModeArray& agModes, const std::vector<ModeArray>& gridGroups){
#ifndef RELEASE
    CallStackEntry entry("DistTensor::AllGatherCommRedist");
//    if(!CheckAllGatherCommRedist(A, agMode, gridModes))
//        LogicError("AllGatherRedist: Invalid redistribution request");
#endif
    Unsigned i;
    const tmen::Grid& g = A.Grid();

    ModeArray commModes;
    for(i = 0; i < gridGroups.size(); i++)
        commModes.insert(commModes.end(), gridGroups[i].begin(), gridGroups[i].end());
    std::sort(commModes.begin(), commModes.end());

    const mpi::Comm comm = GetCommunicatorForModes(commModes, A.Grid());

    if(!A.Participating())
        return;

    //NOTE: Fix to handle strides in Tensor data
    if(agModes.size() == 0){
        CopyLocalBuffer(A);
        return;
    }
    Unsigned sendSize, recvSize;

    //Determine buffer sizes for communication
    const Unsigned nRedistProcs = Max(1, prod(FilterVector(g.Shape(), commModes)));
    const ObjShape maxLocalShapeA = A.MaxLocalShape();

    sendSize = prod(maxLocalShapeA);
    recvSize = sendSize * nRedistProcs;

    Memory<T> auxMemory;
    T* auxBuf = auxMemory.Require(sendSize + recvSize);
    MemZero(&(auxBuf[0]), sendSize + recvSize);

    T* sendBuf = &(auxBuf[0]);
    T* recvBuf = &(auxBuf[sendSize]);

    //printf("Alloc'd %d elems to send and %d elems to receive\n", sendSize, recvSize);
    PackAGCommSendBuf(A, sendBuf);

    //printf("Allgathering %d elements\n", sendSize);
    mpi::AllGather(sendBuf, sendSize, recvBuf, sendSize, comm);

    if(!(Participating()))
        return;

    //NOTE: AG and A2A unpack routines are the exact same
    UnpackA2ACommRecvBuf(recvBuf, agModes, commModes, maxLocalShapeA, A);
    //Print(B.LockedTensor(), "A's local tensor after allgathering:");
}

template <typename T>
void DistTensor<T>::PackAGCommSendBuf(const DistTensor<T>& A, T * const sendBuf)
{
  const Unsigned order = A.Order();
  const T* dataBuf = A.LockedBuffer();

  const Location zeros(order, 0);
  const Location ones(order, 1);

  PackData packData;
  packData.loopShape = A.LocalShape();
  packData.srcBufStrides = A.LocalStrides();

  packData.dstBufStrides = Dimensions2Strides(A.MaxLocalShape());

  packData.loopStarts = zeros;
  packData.loopIncs = ones;

  PackCommHelper(packData, order - 1, &(dataBuf[0]), &(sendBuf[0]));
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
