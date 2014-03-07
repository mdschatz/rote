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
void ReduceScatterRedist(DistTensor<T>& B, const DistTensor<T>& A, const int reduceIndex, const int scatterIndex){
    if(!CheckReduceScatterRedist(B, A, reduceIndex, scatterIndex))
      LogicError("ReduceScatterRedist: Invalid redistribution request");

    int sendSize, recvSize;
    DetermineRSCommunicateDataSize(B, A, reduceIndex, recvSize, sendSize);
    const mpi::Comm comm = A.GetCommunicator(reduceIndex);
    const int myRank = mpi::CommRank(comm);

    Memory<T> auxMemory;
    T* auxBuf = auxMemory.Require(sendSize + recvSize);
    MemZero(&(auxBuf[0]), sendSize + recvSize);
    T* sendBuf = &(auxBuf[0]);
    T* recvBuf = &(auxBuf[sendSize]);

    PackRSSendBuf(B, A, reduceIndex, scatterIndex, sendBuf);

    mpi::ReduceScatter(sendBuf, recvBuf, recvSize, comm);

    UnpackRSRecvBuf(recvBuf, reduceIndex, scatterIndex, A, B);
}

template <typename T>
void PartialReduceScatterRedist(DistTensor<T>& B, const DistTensor<T>& A, const int reduceScatterIndex){
	PartialReduceScatterRedist(B, A, reduceScatterIndex, A.ModeDist(reduceScatterIndex));
}

template <typename T>
void PartialReduceScatterRedist(DistTensor<T>& B, const DistTensor<T>& A, const int reduceScatterIndex, const std::vector<Int>& rsGridModes){
	if(CheckPartialReduceScatterRedist(A, B, reduceScatterIndex, rsGridModes))
		LogicError("ReduceScatterRedist: Invalid redistribution request");

	int sendSize, recvSize;
	DetermineRSCommunicateDataSize(B, A, reduceScatterIndex, recvSize, sendSize);
	const mpi::Comm comm = A.GetCommunicator(reduceScatterIndex);

	Memory<T> auxMemory;
	T* auxBuf = auxMemory.Require(sendSize + recvSize);
	T* sendBuf = &(auxBuf[0]);
	T* recvBuf = &(auxBuf[sendSize]);

	PackRSSendBuf(A, reduceScatterIndex, sendBuf);

	mpi::ReduceScatter(sendBuf, recvBuf, recvSize, comm);

	UnpackRSRecvBuf(recvBuf, reduceScatterIndex, B);
}

template <typename T>
void AllGatherRedist(DistTensor<T>& B, const DistTensor<T>& A, const int allGatherIndex){
	if(!CheckAllGatherRedist(B, A, allGatherIndex))
		LogicError("AllGatherRedist: Invalid redistribution request");

	int sendSize, recvSize;
	DetermineAGCommunicateDataSize(A, allGatherIndex, recvSize, sendSize);
	const mpi::Comm comm = A.GetCommunicator(allGatherIndex);

	Memory<T> auxMemory;
	T* auxBuf = auxMemory.Require(sendSize + recvSize);
	MemZero(&(auxBuf[0]), sendSize + recvSize);
	printf("alloc'd sendSize: %d recvSize: %d\n", sendSize, recvSize);
	T* sendBuf = &(auxBuf[0]);
	T* recvBuf = &(auxBuf[sendSize]);

	//printf("Alloc'd %d elems to send and %d elems to receive\n", sendSize, recvSize);
	PackAGSendBuf(A, allGatherIndex, sendBuf);

	//printf("Allgathering %d elements\n", sendSize);
	mpi::AllGather(sendBuf, sendSize, recvBuf, sendSize, comm);

	UnpackAGRecvBuf(recvBuf, allGatherIndex, A, B);
	//Print(B.LockedTensor(), "A's local tensor after allgathering:");
}



#define PROTO(T) \
	template void ReduceScatterRedist(DistTensor<T>& B, const DistTensor<T>& A, const int reduceIndex, const int scatterIndex); \
    template void PartialReduceScatterRedist(DistTensor<T>& B, const DistTensor<T>& A, const int reduceScatterIndex); \
    template void PartialReduceScatterRedist(DistTensor<T>& B, const DistTensor<T>& A, const int reduceScatterIndex, const std::vector<Int>& rsGridModes); \
	template void AllGatherRedist(DistTensor<T>& B, const DistTensor<T>& A, const int allGatherIndex);

PROTO(int)
PROTO(float)
PROTO(double)

} // namespace tmen
