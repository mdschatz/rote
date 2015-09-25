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
#include "tensormental/core/tensor.hpp"
namespace tmen{

template <typename T>
bool DistTensor<T>::CheckScatterCommRedist(const DistTensor<T>& A){
	const TensorDistribution outDist = TensorDist();
	const TensorDistribution inDist = A.TensorDist();

	bool ret = true;
	ret &= CheckOrder(Order(), A.Order());
	ret &= CheckSameCommModes(outDist, inDist);
	ret &= CheckInIsPrefix(outDist, inDist);
	ret &= CheckNonDistOutIsPrefix(outDist, inDist);

    return ret;
}

template <typename T>
void DistTensor<T>::ScatterCommRedist(const DistTensor<T>& A, const ModeArray& commModes){
	if(!CheckScatterCommRedist(A))
		LogicError("ScatterRedist: Invalid redistribution request");

	const tmen::Grid& g = A.Grid();

	const mpi::Comm comm = GetCommunicatorForModes(commModes, g);

	if(!Participating())
		return;

	//Determine buffer sizes for communication
	const Unsigned nRedistProcs = Max(1, prod(FilterVector(g.Shape(), commModes)));

	const ObjShape gvAShape = A.GetGridView().ParticipatingShape();
	const ObjShape gvBShape = GetGridView().ParticipatingShape();

	std::vector<Unsigned> localPackStrides = ElemwiseDivide(LCMs(gvBShape, gvAShape), gvAShape);
	ObjShape commDataShape = IntCeils(A.MaxLocalShape(), localPackStrides);

	const Unsigned sendSize = prod(commDataShape);
	const Unsigned recvSize = sendSize;

	T* auxBuf = this->auxMemory_.Require((sendSize + recvSize) * nRedistProcs);

	T* sendBuf = &(auxBuf[0]);
	T* recvBuf = &(auxBuf[sendSize * nRedistProcs]);

//	const T* dataBuf = A.LockedBuffer();
//	PrintArray(dataBuf, A.LocalShape(), A.LocalStrides(), "srcBuf");

	//Pack the data
	PROFILE_SECTION("ScatterPack");
	if(A.Participating())
		PackA2ACommSendBuf(A, commModes, commDataShape, sendBuf);
	PROFILE_STOP;

//	ObjShape sendShape = commDataShape;
//	sendShape.insert(sendShape.end(), nRedistProcs);
//	PrintArray(sendBuf, sendShape, "sendBuf");

	//Communicate the data
	PROFILE_SECTION("ScatterComm");
	//Realignment
	const tmen::GridView gvA = A.GetGridView();
	const tmen::GridView gvB = GetGridView();
	const Location firstOwnerA = GridViewLoc2GridLoc(A.Alignments(), gvA);
	const Location firstOwnerB = GridViewLoc2GridLoc(Alignments(), gvB);

	if(AnyElemwiseNotEqual(firstOwnerA, firstOwnerB)){
		T* alignSendBuf = &(sendBuf[0]);
		T* alignRecvBuf = &(sendBuf[sendSize * nRedistProcs]);
		AlignCommBufRedist(A, alignSendBuf, sendSize * nRedistProcs, alignRecvBuf, sendSize * nRedistProcs);
		sendBuf = &(alignRecvBuf[0]);
		recvBuf = &(alignSendBuf[0]);
//		PrintArray(alignRecvBuf, sendShape, "recvBuf from SendRecv");
	}

	mpi::Scatter(sendBuf, sendSize, recvBuf, recvSize, 0, comm);
	PROFILE_STOP;

//	ObjShape recvShape = commDataShape;
//	PrintArray(recvBuf, recvShape, "recvBuf");

	//Unpack the data (if participating)
	PROFILE_SECTION("ScatterUnpack");
	PROFILE_MEMOPS(prod(MaxLocalShape()));
	UnpackPCommRecvBuf(recvBuf, A);
	PROFILE_STOP;

//	const T* myBuf = LockedBuffer();
//	PrintArray(myBuf, LocalShape(), LocalStrides(), "myBuf");

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
