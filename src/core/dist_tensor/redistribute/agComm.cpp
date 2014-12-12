/*
x   Copyright (c) 2009-2013, Jack Poulson
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
DistTensor<T>::AllGatherCommRedist(const DistTensor<T>& A, const ModeArray& commModes){
#ifndef RELEASE
    CallStackEntry entry("DistTensor::AllGatherCommRedist");
//    if(!CheckAllGatherCommRedist(A, agMode, gridModes))
//        LogicError("AllGatherRedist: Invalid redistribution request");
#endif
    const tmen::Grid& g = A.Grid();

    const mpi::Comm comm = GetCommunicatorForModes(commModes, g);

    if(!A.Participating())
        return;

    Unsigned sendSize, recvSize;

    //For unaligned communications
    const tmen::GridView gvA = A.GetGridView();
    const tmen::GridView gvB = GetGridView();
    const std::vector<Unsigned> alignments = Alignments();
    const std::vector<Unsigned> alignmentsA = A.Alignments();
    const TensorDistribution tensorDist = A.TensorDist();

    //Determine buffer sizes for communication
    const Unsigned nRedistProcs = Max(1, prod(FilterVector(g.Shape(), commModes)));
    const ObjShape commDataShape = A.MaxLocalShape();
//    PrintVector(commDataShape, "commDataShape");

    sendSize = prod(commDataShape);
    recvSize = sendSize * nRedistProcs;

    T* auxBuf;
    auxBuf = this->auxMemory_.Require(sendSize + recvSize);

    T* sendBuf = &(auxBuf[0]);
    T* recvBuf = &(auxBuf[sendSize]);

//    const T* dataBuf = A.LockedBuffer();
//    PrintArray(dataBuf, A.LocalShape(), A.LocalStrides(), "srcBuf");
//    std::cout << "srcBuf:";
//    for(Unsigned i = 0; i < prod(A.LocalShape()); i++){
//      std::cout << " " <<  dataBuf[i];
//    }
//    std::cout << std::endl;

    //Pack the data
    PROFILE_SECTION("AGPack");
    PROFILE_FLOPS(prod(commDataShape));
    PackAGCommSendBuf(A, sendBuf);
    PROFILE_STOP;

//    PrintArray(sendBuf, commDataShape, "sendBuf");

    //Communicate the data
    PROFILE_SECTION("AGComm");
    //If unaligned, realign with send/recv BEFORE Allgather (ensures data arrives in correct place)
    Location firstOwnerA = GridViewLoc2GridLoc(A.Alignments(), gvA);
    Location firstOwnerB = GridViewLoc2GridLoc(Alignments(), gvB);
    if(AnyElemwiseNotEqual(firstOwnerA, firstOwnerB)){
//        PrintVector(firstOwnerA, "firstOwnerA");
//        PrintVector(firstOwnerB, "firstOwnerB");
        T* alignSendBuf = &(auxBuf[0]);
        T* alignRecvBuf = &(auxBuf[sendSize * nRedistProcs]);

        AlignCommBufRedist(A, alignSendBuf, sendSize, alignRecvBuf, sendSize);

        sendBuf = &(alignRecvBuf[0]);
        recvBuf = &(alignSendBuf[0]);
//        PrintArray(sendBuf, commDataShape, "postsendBuf");
    }

//    PrintArray(sendBuf, commDataShape, "sendBuf before ag");
    mpi::AllGather(sendBuf, sendSize, recvBuf, sendSize, comm);
    PROFILE_STOP;

    if(!(Participating())){
        this->auxMemory_.Release();
        return;
    }

//    ObjShape recvShape = commDataShape;
//    recvShape.insert(recvShape.end(), nRedistProcs);
//    PrintArray(recvBuf, recvShape, "recvBuf");

    //Unpack the data (if participating)
    PROFILE_SECTION("AGUnpack");
    PROFILE_FLOPS(prod(MaxLocalShape()));
    UnpackA2ACommRecvBuf(recvBuf, commModes, commDataShape, A);
    PROFILE_STOP;

//    PrintData(*this, "data after AGComm");
//    const T* myBuf = LockedBuffer();
//    PrintArray(myBuf, LocalShape(), "myBuf");

    this->auxMemory_.Release();
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

  //Pack into permuted form to minimize striding when unpacking
  ObjShape finalShape = PermuteVector(A.MaxLocalShape(), localPerm_);
  std::vector<Unsigned> finalStrides = Dimensions2Strides(finalShape);

  //Determine permutation from local output to local input
  Permutation out2in = DetermineInversePermutation(DeterminePermutation(A.localPerm_, localPerm_));

  //Permute pack strides to match input local permutation (for correct packing)
  packData.dstBufStrides = PermuteVector(finalStrides, out2in);

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
FULL(std::complex<float>);
#endif
FULL(std::complex<double>);
#endif

} //namespace tmen
