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
void PermutationRedist(DistTensor<T>& B, const DistTensor<T>& A, const int permuteIndex){
    if(!CheckPermutationRedist(B, A, permuteIndex))
            LogicError("PermutationRedist: Invalid redistribution request");

        int sendSize, recvSize;
        DeterminePermCommunicateDataSize(B, A, permuteIndex, recvSize, sendSize);
        const mpi::Comm comm = A.GetCommunicator(permuteIndex);
        const int myRank = mpi::CommRank(comm);

        Memory<T> auxMemory;
        T* auxBuf = auxMemory.Require(sendSize + recvSize);
        MemZero(&(auxBuf[0]), sendSize + recvSize);
        T* sendBuf = &(auxBuf[0]);
        T* recvBuf = &(auxBuf[sendSize]);

        PackPermutationSendBuf(B, A, permuteIndex, sendBuf);

        const GridView gvA = A.GridView();
        const GridView gvB = B.GridView();

        const ModeDistribution permuteIndexDistA = A.IndexDist(permuteIndex);
        const ModeDistribution permuteIndexDistB = B.IndexDist(permuteIndex);

        ModeDistribution gridModesUsed(permuteIndexDistB);
        std::sort(gridModesUsed.begin(), gridModesUsed.end());

        const std::vector<int> gridSliceShape = FilterVector(A.Grid().Shape(), gridModesUsed);
        const std::vector<int> gridSliceStridesA = Dimensions2Strides(FilterVector(A.Grid().Shape(), permuteIndexDistA));
        const std::vector<int> gridSliceStridesB = Dimensions2Strides(FilterVector(A.Grid().Shape(), permuteIndexDistB));

        const std::vector<int> permA = DeterminePermutation(gridModesUsed, permuteIndexDistA);
        const std::vector<int> permB = DeterminePermutation(gridModesUsed, permuteIndexDistB);

        //Determine sendRank
        const std::vector<int> sendLoc = LinearLoc2Loc(myRank, gridSliceShape, permB);
        const int sendRank = LinearIndex(FilterVector(sendLoc, permA), gridSliceStridesA);

        //Determine recvRank
        const std::vector<int> myLoc = LinearLoc2Loc(myRank, gridSliceShape, permA);
        const int recvLinearLoc = LinearIndex(FilterVector(myLoc, permB), gridSliceStridesB);
        const std::vector<int> recvLoc = LinearLoc2Loc(recvLinearLoc, gridSliceShape, permA);
        const int recvRank = LinearIndex(FilterVector(recvLoc, permA), gridSliceStridesA);

        //printf("myRank: %d sending to rank: %d, receiving from rank: %d\n", myRank, sendRank, recvRank);
        mpi::SendRecv(sendBuf, sendSize, sendRank,
                      recvBuf, recvSize, recvRank, comm);

        UnpackPermutationRecvBuf(recvBuf, permuteIndex, A, B);
}

template <typename T>
void PartialReduceScatterRedist(DistTensor<T>& B, const DistTensor<T>& A, const int reduceScatterIndex){
    if(!CheckPartialReduceScatterRedist(B, A, reduceScatterIndex))
        LogicError("PartialReduceScatterRedist: Invalid redistribution request");

    int sendSize, recvSize;
    DeterminePartialRSCommunicateDataSize(B, A, reduceScatterIndex, recvSize, sendSize);
    const mpi::Comm comm = A.GetCommunicator(reduceScatterIndex);

    Memory<T> auxMemory;
    T* auxBuf = auxMemory.Require(sendSize + recvSize);
    MemZero(&(auxBuf[0]), sendSize + recvSize);
    T* sendBuf = &(auxBuf[0]);
    T* recvBuf = &(auxBuf[sendSize]);

    PackPartialRSSendBuf(B, A, reduceScatterIndex, sendBuf);

    mpi::ReduceScatter(sendBuf, recvBuf, recvSize, comm);

    UnpackPartialRSRecvBuf(recvBuf, reduceScatterIndex, A, B);

}

template <typename T>
void ReduceScatterRedist(DistTensor<T>& B, const DistTensor<T>& A, const int reduceIndex, const int scatterIndex){
    if(!CheckReduceScatterRedist(B, A, reduceIndex, scatterIndex))
      LogicError("ReduceScatterRedist: Invalid redistribution request");

    int sendSize, recvSize;
    DetermineRSCommunicateDataSize(B, A, reduceIndex, recvSize, sendSize);
    const mpi::Comm comm = A.GetCommunicator(reduceIndex);

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
void AllGatherRedist(DistTensor<T>& B, const DistTensor<T>& A, const int allGatherIndex){
	if(!CheckAllGatherRedist(B, A, allGatherIndex))
		LogicError("AllGatherRedist: Invalid redistribution request");

	int sendSize, recvSize;
	DetermineAGCommunicateDataSize(A, allGatherIndex, recvSize, sendSize);
	const mpi::Comm comm = A.GetCommunicator(allGatherIndex);

	Memory<T> auxMemory;
	T* auxBuf = auxMemory.Require(sendSize + recvSize);
	MemZero(&(auxBuf[0]), sendSize + recvSize);

	T* sendBuf = &(auxBuf[0]);
	T* recvBuf = &(auxBuf[sendSize]);

	//printf("Alloc'd %d elems to send and %d elems to receive\n", sendSize, recvSize);
	PackAGSendBuf(A, allGatherIndex, sendBuf);

	//printf("Allgathering %d elements\n", sendSize);
	mpi::AllGather(sendBuf, sendSize, recvBuf, sendSize, comm);

	UnpackAGRecvBuf(recvBuf, allGatherIndex, A, B);
	//Print(B.LockedTensor(), "A's local tensor after allgathering:");
}

template <typename T>
void AllToAllDoubleIndexRedist(DistTensor<T>& B, const DistTensor<T>& A, const std::pair<int, int>& a2aIndices, const std::pair<std::vector<int>, std::vector<int> >& a2aCommGroups){
    if(!CheckAllToAllDoubleIndexRedist(B, A, a2aIndices, a2aCommGroups))
        LogicError("AllToAllSingleIndexRedist: Invalid redistribution request");

    int sendSize, recvSize;

    std::vector<int> commModes = a2aCommGroups.first;
    commModes.insert(commModes.end(), a2aCommGroups.second.begin(), a2aCommGroups.second.end());
    std::sort(commModes.begin(), commModes.end());

    const mpi::Comm comm = A.GetCommunicatorForModes(commModes);
    DetermineA2ADoubleIndexCommunicateDataSize(B, A, a2aIndices, a2aCommGroups, recvSize, sendSize);

    Memory<T> auxMemory;
    T* auxBuf = auxMemory.Require(sendSize + recvSize);
    MemZero(&(auxBuf[0]), sendSize + recvSize);

    T* sendBuf = &(auxBuf[0]);
    T* recvBuf = &(auxBuf[sendSize]);

    PackA2ADoubleIndexSendBuf(B, A, a2aIndices, a2aCommGroups, sendBuf);

    mpi::AllToAll(sendBuf, sendSize, recvBuf, recvSize, comm);

    UnpackA2ADoubleIndexRecvBuf(recvBuf, a2aIndices, commModes, A, B);
}

template <typename T>
void AllToAllRedist(DistTensor<T>& B, const DistTensor<T>& A){
//NOTE: All2All is valid if both distributions are valid
//	if(!CheckAllToAllRedist(B, A))
//		LogicError("AllToAllRedist: Invalid redistribution request");

	int sendSize, recvSize;
	//Figure out which modes we have to communicate along (to save some pain)
	const std::vector<int> commModes = DetermineA2ACommunicateModes(B, A);
//	DetermineA2ACommunicateDataSize(B, A, recvSize, sendSize);
//	//Get one big communicator for the modes we need to communicate along
//	const mpi::Comm comm = A.GetCommunicatorForModes(commModes);
//
//	Memory<T> auxMemory;
//	T* auxBuf = auxMemory.Require(sendSize + recvSize);
//	MemZero(&(auxBuf[0]), sendSize + recvSize);
//
//	T* sendBuf = &(auxBuf[0]);
//	T* recvBuf = &(auxBuf[sendSize]);
//
//	//printf("Alloc'd %d elems to send and %d elems to receive\n", sendSize, recvSize);
//	PackA2ASendBuf(B, A, sendBuf);
//
//	//printf("Allgathering %d elements\n", sendSize);
//	mpi::AllToAll(sendBuf, sendSize, recvBuf, sendSize, comm);
//
//	UnpackA2ARecvBuf(recvBuf, A, B);
}

#define PROTO(T) \
    template void PermutationRedist(DistTensor<T>& B, const DistTensor<T>& A, const int permuteIndex); \
    template void ReduceScatterRedist(DistTensor<T>& B, const DistTensor<T>& A, const int reduceIndex, const int scatterIndex); \
    template void PartialReduceScatterRedist(DistTensor<T>& B, const DistTensor<T>& A, const int reduceScatterIndex); \
	template void AllGatherRedist(DistTensor<T>& B, const DistTensor<T>& A, const int allGatherIndex); \
	template void AllToAllDoubleIndexRedist(DistTensor<T>& B, const DistTensor<T>& A, const std::pair<int, int>& a2aIndices, const std::pair<std::vector<int>, std::vector<int> >& commGroups);
PROTO(int)
PROTO(float)
PROTO(double)

} // namespace tmen
