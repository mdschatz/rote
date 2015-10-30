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

//TODO: Properly Check indices and distributions match between input and output
//TODO: FLESH OUT THIS CHECK
template <typename T>
bool DistTensor<T>::CheckGatherToOneCommRedist(const DistTensor<T>& A){
	const TensorDistribution outDist = TensorDist();
	const TensorDistribution inDist = A.TensorDist();

	bool ret = true;
	ret &= CheckOrder(Order(), A.Order());
	ret &= CheckOutIsPrefix(outDist, inDist);
	ret &= CheckSameCommModes(outDist, inDist);

    return ret;
}

template <typename T>
void DistTensor<T>::GatherToOneCommRedist(const DistTensor<T>& A, const ModeArray& commModes){
    if(!CheckGatherToOneCommRedist(A))
      LogicError("GatherToOneRedist: Invalid redistribution request");

    const mpi::Comm comm = GetCommunicatorForModes(commModes, A.Grid());

    if(!A.Participating())
        return;

    //Determine buffer sizes for communication
    const Unsigned nRedistProcs = Max(1, prod(FilterVector(A.Grid().Shape(), commModes)));
    const ObjShape commDataShape = A.MaxLocalShape();

    const Unsigned sendSize = prod(commDataShape);
    const Unsigned recvSize = sendSize;

    T* auxBuf = this->auxMemory_.Require(sendSize + nRedistProcs*recvSize);
    T* sendBuf = &(auxBuf[0]);
    T* recvBuf = &(auxBuf[sendSize]);

    //Pack the data
    PROFILE_SECTION("GTOPack");
    PackAGCommSendBuf(A, sendBuf);
    PROFILE_STOP;

    //Communicate the data
    PROFILE_SECTION("GTOComm");
    //Realignment
    T* alignSendBuf = &(auxBuf[0]);
    T* alignRecvBuf = &(auxBuf[sendSize]);

    bool didAlign = AlignCommBufRedist(A, alignSendBuf, sendSize, alignRecvBuf, sendSize);

    if(didAlign){
		sendBuf = &(alignRecvBuf[0]);
		recvBuf = &(alignSendBuf[0]);
    }

    mpi::Gather(sendBuf, sendSize, recvBuf, recvSize, 0, comm);
    PROFILE_STOP;

    //Unpack the data (if participating)
    PROFILE_SECTION("GTOUnpack");
    if(Participating())
    	UnpackA2ACommRecvBuf(recvBuf, commModes, commDataShape, A);
    PROFILE_STOP;

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

} //namespace tmen
