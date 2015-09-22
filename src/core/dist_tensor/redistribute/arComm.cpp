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

template <typename T>
Int DistTensor<T>::CheckAllReduceCommRedist(const DistTensor<T>& A, const ModeArray& reduceModes){
	const TensorDistribution outDist = TensorDist();
	const TensorDistribution inDist = A.TensorDist();

	bool ret = true;
	ret &= CheckOrder(Order(), A.Order());
	ret &= CheckOutIsPrefix(outDist, inDist);
	ret &= CheckSameNonDist(outDist, inDist);

    return ret;
}

template <typename T>
void DistTensor<T>::AllReduceUpdateCommRedist(const T alpha, const DistTensor<T>& A, const T beta, const ModeArray& reduceModes, const ModeArray& commModes){
    if(!CheckAllReduceCommRedist(A, reduceModes))
      LogicError("AllReduceRedist: Invalid redistribution request");
    const tmen::Grid& g = A.Grid();

    const mpi::Comm comm = GetCommunicatorForModes(commModes, g);
    const tmen::GridView gvA = A.GetGridView();
    const tmen::GridView gvB = GetGridView();

    if(!A.Participating())
        return;
    Unsigned sendSize, recvSize;

    //Determine buffer sizes for communication
    const Unsigned nRedistProcs = Max(1, prod(FilterVector(g.Shape(), commModes)));
    const ObjShape commDataShape = MaxLocalShape();
//    PrintVector(commDataShape, "commDataShape");
//    printf("nRedistProcs\n", nRedistProcs);
    recvSize = prod(commDataShape);
    sendSize = recvSize;
//    printf("sendSize: %d\n", sendSize);

    T* auxBuf;

    //TODO: Figure out how to nicely do this alloc for Alignment
    Location firstOwnerA = GridViewLoc2GridLoc(A.Alignments(), gvA);
    Location firstOwnerB = GridViewLoc2GridLoc(Alignments(), gvB);
    if(AnyElemwiseNotEqual(firstOwnerA, firstOwnerB)){
        auxBuf = this->auxMemory_.Require(sendSize + sendSize);
        MemZero(&(auxBuf[0]), sendSize + sendSize);
//        printf("required: %d elems\n", sendSize + sendSize);
    }else{
        auxBuf = this->auxMemory_.Require(sendSize + recvSize);
        //First, set all entries of sendBuf to zero so we don't accumulate garbage
        MemZero(&(auxBuf[0]), sendSize + recvSize);
    }
    T* sendBuf = &(auxBuf[0]);
    T* recvBuf = &(auxBuf[sendSize]);

//    const T* dataBuf = A.LockedBuffer();
//    PrintArray(dataBuf, A.LocalShape(), A.LocalStrides(), "srcBuf");

    //Pack the data
    PROFILE_SECTION("ARPack");
    PackAGCommSendBuf(A, sendBuf);
    PROFILE_STOP;

//    ObjShape sendShape = commDataShape;
//    sendShape.insert(sendShape.end(), nRedistProcs);
//    PrintVector(sendShape, "sendShape");
//    PrintArray(sendBuf, sendShape, "sendBuf");

    //Communicate the data
    PROFILE_SECTION("ARComm");

    if(AnyElemwiseNotEqual(firstOwnerA, firstOwnerB)){
//        printf("aligningAR\n");
//        PrintData(A, "A");
//        PrintData(*this, "*this");
        T* alignSendBuf = &(sendBuf[0]);
        T* alignRecvBuf = &(recvBuf[0]);

        AlignCommBufRedist(A, alignSendBuf, sendSize, alignRecvBuf, sendSize);
        sendBuf = &(alignRecvBuf[0]);
        recvBuf = &(alignSendBuf[0]);
//        PrintArray(alignRecvBuf, sendShape, "recvBuf from SendRecv");
    }

    mpi::AllReduce(sendBuf, recvBuf, recvSize, comm);
    PROFILE_STOP;

//    ObjShape recvShape = commDataShape;
//    PrintArray(recvBuf, recvShape, "recvBuf");

    if(!Participating()){
        this->auxMemory_.Release();
        return;
    }

    //Unpack the data (if participating)
    PROFILE_SECTION("ARUnpack");
    UnpackRSUCommRecvBuf(recvBuf, alpha, A, beta);
    PROFILE_STOP;

//    const T* myBuf = LockedBuffer();
//    PrintArray(myBuf, LocalShape(), LocalStrides(), "myBuf");

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
FULL(std::complex<float>);
#endif
FULL(std::complex<double>);
#endif

} //namespace tmen
