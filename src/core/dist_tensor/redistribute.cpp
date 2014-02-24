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
#include "tensormental/core/dist_tensor/redistribute.hpp"
#include "tensormental/core/dist_tensor/pack.hpp"
#include "tensormental/util/vec_util.hpp"
#include <algorithm>

namespace tmen{

template <typename T>
void ReduceScatterRedist(DistTensor<T>& A, const DistTensor<T>& B, int reduceIndex, int scatterIndex){
	//if(CheckReduceScatterRedist(A, B, reduceIndex, scatterIndex))
	//	LogicError("ReduceScatterRedist: Invalid redistribution request");

	int sendSize, recvSize;
	DetermineRSCommunicateDataSize(B, reduceIndex, recvSize, sendSize);
	const mpi::Comm comm = B.GetCommunicator(reduceIndex);

	Memory<T> auxMemory;
	T* auxBuf = auxMemory.Require(sendSize);
	T* sendBuf = &(auxBuf[0]);
	T* recvBuf = &(auxBuf[sendSize]);

	PackRSSendBuf(B, reduceIndex, scatterIndex, sendBuf);

	mpi::ReduceScatter(sendBuf, recvBuf, recvSize, comm);

	UnpackRSRecvBuf(recvBuf, reduceIndex, scatterIndex, A);
}

template <typename T>
void AllGatherRedist(DistTensor<T>& A, const DistTensor<T>& B, int allGatherIndex){
	if(!CheckAllGatherRedist(A, B, allGatherIndex))
		LogicError("AllGatherRedist: Invalid redistribution request");

	int sendSize, recvSize;
	DetermineAGCommunicateDataSize(B, allGatherIndex, recvSize, sendSize);
	const mpi::Comm comm = B.GetCommunicator(allGatherIndex);

	Memory<T> auxMemory;
	T* auxBuf = auxMemory.Require(sendSize);
	T* sendBuf = &(auxBuf[0]);
	T* recvBuf = &(auxBuf[sendSize]);

	PackAGSendBuf(B, allGatherIndex, sendBuf);

	mpi::AllGather(sendBuf, sendSize, recvBuf, sendSize, comm);

	UnpackAGRecvBuf(recvBuf, allGatherIndex, B.GridView(), A);
}


#define PROTO(T) \
	template void ReduceScatterRedist(DistTensor<T>& A, const DistTensor<T>& B, int reduceIndex, int scatterIndex); \
	template void AllGatherRedist(DistTensor<T>& A, const DistTensor<T>& B, int allGatherIndex);

PROTO(int)
PROTO(float)
PROTO(double)

} // namespace tmen
