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

//	if(!A.Participating())
//		return;

	//Determine buffer sizes for communication
	Unsigned sendSize, recvSize;
	const Unsigned nRedistProcs = Max(1, prod(FilterVector(g.Shape(), commModes)));
	const ObjShape maxLocalShapeA = A.MaxLocalShape();
	const ObjShape maxLocalShapeB = MaxLocalShape();

	const tmen::GridView gvA = A.GetGridView();
	const tmen::GridView gvB = GetGridView();
	const ObjShape gvAShape = gvA.ParticipatingShape();
	const ObjShape gvBShape = gvB.ParticipatingShape();

	//For unaligned communications
	const std::vector<Unsigned> alignments = Alignments();
	const std::vector<Unsigned> alignmentsA = A.Alignments();
	const TensorDistribution tensorDist = A.TensorDist();


	std::vector<Unsigned> localPackStrides(maxLocalShapeA.size());
	localPackStrides = ElemwiseDivide(LCMs(gvBShape, gvAShape), gvAShape);
	ObjShape commDataShape(maxLocalShapeA.size());
	commDataShape = IntCeils(maxLocalShapeA, localPackStrides);
//        PrintVector(commDataShape, "commDataShape");
//        printf("nRedistProcs: %d\n", nRedistProcs);

	sendSize = prod(commDataShape);
	recvSize = sendSize;

	T* auxBuf;
	auxBuf = this->auxMemory_.Require(sendSize + recvSize);

	T* sendBuf = &(auxBuf[0]);
	T* recvBuf = sendBuf;

//        const T* dataBuf = A.LockedBuffer();
//        PrintArray(dataBuf, A.LocalShape(), A.LocalStrides(), "srcBuf");

	//Pack the data
	PROFILE_SECTION("ScatterPack");
	PROFILE_MEMOPS(prod(maxLocalShapeA));
	if(A.Participating())
		PackA2ACommSendBuf(A, commModes, commDataShape, sendBuf);
	PROFILE_STOP;


	ObjShape sendShape = commDataShape;
	sendShape.insert(sendShape.end(), nRedistProcs);
//        PrintVector(sendShape, "sendShape");
//        PrintArray(sendBuf, sendShape, "sendBuf");

	//Communicate the data
	PROFILE_SECTION("ScatterComm");
	Location firstOwnerA = GridViewLoc2GridLoc(A.Alignments(), gvA);
	Location firstOwnerB = GridViewLoc2GridLoc(Alignments(), gvB);
//        printf("distA: %s\n", tmen::TensorDistToString(gvA.Distribution()).c_str());
//        PrintVector(A.Alignments(), "A.Alignments()");
//        PrintVector(firstOwnerA, "firstOwnerA");

//        printf("distB: %s\n", tmen::TensorDistToString(gvB.Distribution()).c_str());
//        PrintVector(Alignments(), "B.Alignments()");
//        PrintVector(firstOwnerB, "firstOwnerB");
	if(AnyElemwiseNotEqual(firstOwnerA, firstOwnerB)){
//                PrintVector(g.Loc(), "myGridLoc");
//                PrintVector(firstOwnerA, "firstOwnerA");
//                PrintVector(firstOwnerB, "firstOwnerB");
		T* alignSendBuf = &(sendBuf[0]);
		T* alignRecvBuf = &(sendBuf[sendSize]);
		AlignCommBufRedist(A, alignSendBuf, sendSize * nRedistProcs, alignRecvBuf, sendSize * nRedistProcs);
		sendBuf = &(alignRecvBuf[0]);
		recvBuf = sendBuf;
//            PrintArray(alignRecvBuf, sendShape, "recvBuf from SendRecv");
	}

	mpi::Scatter(sendBuf, sendSize, recvSize, 0, comm);
	//Perform a send/recv to realign the data (if needed)
	PROFILE_STOP;

	if(!(Participating())){
		this->auxMemory_.Release();
		return;
	}

//        ObjShape recvShape = commDataShape;
//        recvShape.insert(recvShape.end(), nRedistProcs);
//        PrintArray(recvBuf, recvShape, "recvBuf");

	//Unpack the data (if participating)
	PROFILE_SECTION("ScatterUnpack");
	PROFILE_MEMOPS(prod(MaxLocalShape()));
	if(Participating())
		UnpackA2ACommRecvBuf(recvBuf, commModes, commDataShape, A);
	PROFILE_STOP;

//        const T* myBuf = LockedBuffer();
//        PrintArray(myBuf, LocalShape(), LocalStrides(), "myBuf");

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
