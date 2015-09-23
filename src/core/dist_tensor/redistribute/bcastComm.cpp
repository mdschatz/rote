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
bool DistTensor<T>::CheckBroadcastCommRedist(const DistTensor<T>& A){
	const TensorDistribution outDist = TensorDist();
	const TensorDistribution inDist = A.TensorDist();

	bool ret = true;
	ret &= CheckOrder(Order(), A.Order());
	ret &= CheckOutIsPrefix(outDist, inDist);
	ret &= CheckNonDistOutIsPrefix(outDist, inDist);

    return ret;
}

template<typename T>
void
DistTensor<T>::BroadcastCommRedist(const DistTensor<T>& A, const ModeArray& commModes){
#ifndef RELEASE
    CallStackEntry entry("DistTensor::BroadcastCommRedist");
    if(!CheckBroadcastCommRedist(A))
        LogicError("BroadcastRedist: Invalid redistribution request");
#endif
    const tmen::Grid& g = A.Grid();

    const mpi::Comm comm = GetCommunicatorForModes(commModes, g);

//    if(!A.Participating())
//        return;

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

    sendSize = prod(commDataShape);
    recvSize = sendSize;

    T* auxBuf;
    auxBuf = this->auxMemory_.Require(sendSize + recvSize);

    T* sendBuf = &(auxBuf[0]);
    T* recvBuf = sendBuf;

//    const T* dataBuf = A.LockedBuffer();
//    PrintArray(dataBuf, A.LocalShape(), A.LocalStrides(), "srcBuf");

    //Pack the data
    PROFILE_SECTION("BCastPack");
    PROFILE_MEMOPS(prod(commDataShape));
    if(A.Participating())
    	PackAGCommSendBuf(A, sendBuf);
    PROFILE_STOP;

//    PrintArray(sendBuf, commDataShape, "sendBuf");

    //Communicate the data
    PROFILE_SECTION("BCastComm");
    //If unaligned, realign with send/recv BEFORE Allgather (ensures data arrives in correct place)
    Location firstOwnerA = GridViewLoc2GridLoc(A.Alignments(), gvA);
    Location firstOwnerB = GridViewLoc2GridLoc(Alignments(), gvB);
    if(AnyElemwiseNotEqual(firstOwnerA, firstOwnerB)){
        T* alignSendBuf = &(auxBuf[0]);
        T* alignRecvBuf = &(auxBuf[sendSize]);

        AlignCommBufRedist(A, alignSendBuf, sendSize, alignRecvBuf, sendSize);

        sendBuf = &(alignRecvBuf[0]);
        recvBuf = sendBuf;
//        PrintArray(sendBuf, commDataShape, "postsendBuf");
    }

//    PrintArray(sendBuf, commDataShape, "sendBuf before ag");
    mpi::Broadcast(sendBuf, sendSize, 0, comm);
    PROFILE_STOP;

    if(!(Participating())){
        this->auxMemory_.Release();
        return;
    }

//    ObjShape recvShape = commDataShape;
//    recvShape.insert(recvShape.end(), nRedistProcs);
//    PrintArray(recvBuf, recvShape, "recvBuf");

    //Unpack the data (if participating)
    PROFILE_SECTION("BCastUnpack");
    PROFILE_MEMOPS(prod(MaxLocalShape()));
    if(Participating())
    	UnpackA2ACommRecvBuf(recvBuf, commModes, commDataShape, A);
    PROFILE_STOP;

//    const T* myBuf = LockedBuffer();
//    PrintArray(myBuf, LocalShape(), "myBuf");

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
