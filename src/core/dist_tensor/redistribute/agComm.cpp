/*
x   Copyright (c) 2009-2013, Jack Poulson
                      2013, Jeff Hammond
                      2013, Jed Brown
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#include "rote.hpp"
#include <algorithm>

namespace rote{

template<typename T>
bool DistTensor<T>::CheckAllGatherCommRedist(const DistTensor<T>& A){
	const TensorDistribution outDist = TensorDist();
	const TensorDistribution inDist = A.TensorDist();

	bool ret = true;
	ret &= CheckOrder(Order(), A.Order());
	ret &= CheckOutIsPrefix(outDist, inDist);
	ret &= CheckSameNonDist(outDist, inDist);

    return ret;
}

template<typename T>
void
DistTensor<T>::AllGatherCommRedist(const DistTensor<T>& A, const ModeArray& commModes, const T alpha){
#ifndef RELEASE
    CallStackEntry entry("DistTensor::AllGatherCommRedist");
    if(!CheckAllGatherCommRedist(A))
        LogicError("AllGatherRedist: Invalid redistribution request");
#endif
    const rote::Grid& g = A.Grid();
    const mpi::Comm comm = GetCommunicatorForModes(commModes, g);

    if(!A.Participating())
        return;

    //Determine buffer sizes for communication
    const Unsigned nRedistProcs = Max(1, prod(FilterVector(g.Shape(), commModes)));
    const ObjShape commDataShape = A.MaxLocalShape();

    const Unsigned sendSize = prod(commDataShape);
    const Unsigned recvSize = sendSize * nRedistProcs;

    T* auxBuf = this->auxMemory_.Require(sendSize + recvSize);

    T* sendBuf = &(auxBuf[0]);
    T* recvBuf = &(auxBuf[sendSize]);

//    const T* dataBuf = A.LockedBuffer();
//    PrintArray(dataBuf, A.LocalShape(), A.LocalStrides(), "srcBuf");

    //Pack the data
    PROFILE_SECTION("AGPack");
    PackAGCommSendBuf(A, sendBuf);
    PROFILE_STOP;

//    PrintArray(sendBuf, commDataShape, "sendBuf");

    //Communicate the data
    PROFILE_SECTION("AGComm");
    //Realignment
    T* alignSendBuf = &(auxBuf[0]);
	T* alignRecvBuf = &(auxBuf[sendSize * nRedistProcs]);

	bool didAlign = AlignCommBufRedist(A, alignSendBuf, sendSize, alignRecvBuf, sendSize);
	if(didAlign){
        sendBuf = &(alignRecvBuf[0]);
        recvBuf = &(alignSendBuf[0]);
	}

	mpi::AllGather(sendBuf, sendSize, recvBuf, sendSize, comm);
    PROFILE_STOP;

//    ObjShape recvShape = commDataShape;
//    recvShape.insert(recvShape.end(), nRedistProcs);
//    PrintArray(recvBuf, recvShape, "recvBuf");

    PROFILE_SECTION("AGUnpack");
    UnpackA2ACommRecvBuf(recvBuf, commModes, commDataShape, A, alpha);
    PROFILE_STOP;

//    const T* myBuf = LockedBuffer();
//    PrintArray(myBuf, LocalShape(), "myBuf");

    this->auxMemory_.Release();
}

template <typename T>
void DistTensor<T>::PackAGCommSendBuf(const DistTensor<T>& A, T * const sendBuf)
{
  const T* dataBuf = A.LockedBuffer();

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

  PackCommHelper(packData, &(dataBuf[0]), &(sendBuf[0]));
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
