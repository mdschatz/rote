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
#include "tensormental/core/dist_tensor/pack.hpp"
#include <algorithm>

namespace tmen{

template <typename T>
void ReduceScatterRedist(DistTensor<T>& A, const DistTensor<T>& B, int reduceIndex, int scatterIndex){
	if(CheckReduceScatterRedist(A, B, reduceIndex, scatterIndex))
		LogicError("ReduceScatterRedist: Invalid Redistribution request");

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

#define PROTO(T) \
	template void ReduceScatterRedist(DistTensor<T>& A, const DistTensor<T>& B, int reduceIndex, int scatterIndex);

PROTO(int)
PROTO(float)
PROTO(double)

} // namespace tmen
