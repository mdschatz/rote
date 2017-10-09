/*
   Copyright (c) 2009-2013, Jack Poulson
                      2013, Jeff Hammond
                      2013, Jed Brown
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#include "rote.hpp"

namespace rote{

//TODO: Properly Check indices and distributions match between input and output
//TODO: FLESH OUT THIS CHECK
template <typename T>
bool DistTensor<T>::CheckReduceToOneCommRedist(const DistTensor<T>& A){
	return CheckGatherToOneCommRedist(A);
}

template <typename T>
void DistTensor<T>::ReduceToOneUpdateCommRedist(const T alpha, const DistTensor<T>& A, const T beta, const ModeArray& commModes){
    if(!CheckReduceToOneCommRedist(A))
      LogicError("ReduceToOneRedist: Invalid redistribution request");

    const rote::Grid& g = A.Grid();
    const mpi::Comm comm = this->GetCommunicatorForModes(commModes, g);

    if(!A.Participating())
        return;

    //Determine buffer sizes for communication
    const ObjShape commDataShape = A.MaxLocalShape();
    const Unsigned sendSize = prod(commDataShape);
    const Unsigned recvSize = sendSize;

    T* auxBuf = this->auxMemory_.Require(sendSize + recvSize);
    T* sendBuf = &(auxBuf[0]);
    T* recvBuf = &(auxBuf[sendSize]);

//    const T* dataBuf = A.LockedBuffer();
//    PrintArray(dataBuf, A.LocalShape(), A.LocalStrides(), "srcBuf");

    MemZero(&(sendBuf[0]), sendSize);

    //Pack the data
    PROFILE_SECTION("RTOPack");
    this->PackAGCommSendBuf(A, sendBuf);
    PROFILE_STOP;

//    ObjShape sendShape = commDataShape;
//    PrintArray(sendBuf, sendShape, "sendBuf");

    //Communicate the data
    PROFILE_SECTION("RTOComm");
    T* alignSendBuf = &(sendBuf[0]);
    T* alignRecvBuf = &(recvBuf[0]);

    bool didAlign = this->AlignCommBufRedist(A, alignSendBuf, sendSize, alignRecvBuf, sendSize);
    if(didAlign){
		sendBuf = &(alignRecvBuf[0]);
		recvBuf = &(alignSendBuf[0]);
    }

    mpi::Reduce(sendBuf, recvBuf, sendSize, mpi::SUM, 0, comm);
    PROFILE_STOP;

//    PrintVector(commDataShape, "commDataShape");
//    ObjShape recvShape = commDataShape;
//    PrintArray(recvBuf, recvShape, "recvBuf");

    //Unpack the data (if participating)
    PROFILE_SECTION("RTOUnpack");
    if(this->Participating())
    	this->UnpackRSUCommRecvBuf(recvBuf, alpha, beta);
    PROFILE_STOP;

//    const T* myBuf = LockedBuffer();
//    PrintArray(myBuf, LocalShape(), LocalStrides(), "myBuf");

    this->auxMemory_.Release();
}

#define FULL(T) \
    template class DistTensor<T>;

FULL(Int)
#ifndef DISABLE_FLOAT
FULL(float)
#endif
FULL(double)

#ifndef DISABLE_COMPLEX
#ifndef DISABLE_FLOAT
FULL(std::complex<float>)
#endif
FULL(std::complex<double>)
#endif

} //namespace rote
