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
void PermutationRedist(DistTensor<T>& B, const DistTensor<T>& A, const Mode permuteMode){
    if(!CheckPermutationRedist(B, A, permuteMode))
            LogicError("PermutationRedist: Invalid redistribution request");

        Unsigned sendSize, recvSize;
        DeterminePermCommunicateDataSize(B, A, permuteMode, recvSize, sendSize);

        //NOTE: Hack for testing.  We actually need to let the user specify the commModes
        const ModeArray commModes = A.ModeDist(permuteMode);
        const mpi::Comm comm = A.GetCommunicatorForModes(commModes);
        const int myRank = mpi::CommRank(comm);

        Memory<T> auxMemory;
        T* auxBuf = auxMemory.Require(sendSize + recvSize);
        MemZero(&(auxBuf[0]), sendSize + recvSize);
        T* sendBuf = &(auxBuf[0]);
        T* recvBuf = &(auxBuf[sendSize]);

        PackPermutationSendBuf(B, A, permuteMode, sendBuf);

        const GridView gvA = A.GridView();
        const GridView gvB = B.GridView();

        const ModeDistribution permuteModeDistA = A.ModeDist(permuteMode);
        const ModeDistribution permuteModeDistB = B.ModeDist(permuteMode);

        ModeDistribution gridModesUsed(permuteModeDistB);
        std::sort(gridModesUsed.begin(), gridModesUsed.end());

        const ObjShape gridSliceShape = FilterVector(A.Grid().Shape(), gridModesUsed);
        const std::vector<Unsigned> gridSliceStridesA = Dimensions2Strides(FilterVector(A.Grid().Shape(), permuteModeDistA));
        const std::vector<Unsigned> gridSliceStridesB = Dimensions2Strides(FilterVector(A.Grid().Shape(), permuteModeDistB));

        const std::vector<Unsigned> permA = DeterminePermutation(gridModesUsed, permuteModeDistA);
        const std::vector<Unsigned> permB = DeterminePermutation(gridModesUsed, permuteModeDistB);

        //Determine sendRank
        const Location sendLoc = LinearLoc2Loc(myRank, gridSliceShape, permB);
        const Unsigned sendRank = Loc2LinearLoc(FilterVector(sendLoc, permA), FilterVector(A.Grid().Shape(), permuteModeDistA));

        //Determine recvRank
        const Location myLoc = LinearLoc2Loc(myRank, gridSliceShape, permA);
        const Unsigned recvLinearLoc = Loc2LinearLoc(FilterVector(myLoc, permB), FilterVector(A.Grid().Shape(), permuteModeDistB));
        const Location recvLoc = LinearLoc2Loc(recvLinearLoc, gridSliceShape, permA);
        const Unsigned recvRank = Loc2LinearLoc(FilterVector(recvLoc, permA), FilterVector(A.Grid().Shape(), permuteModeDistA));

        //printf("myRank: %d sending to rank: %d, receiving from rank: %d\n", myRank, sendRank, recvRank);
        mpi::SendRecv(sendBuf, sendSize, sendRank,
                      recvBuf, recvSize, recvRank, comm);

        UnpackPermutationRecvBuf(recvBuf, permuteMode, A, B);
}

template <typename T>
void PartialReduceScatterRedist(DistTensor<T>& B, const DistTensor<T>& A, const Mode reduceScatterMode){
    if(!CheckPartialReduceScatterRedist(B, A, reduceScatterMode))
        LogicError("PartialReduceScatterRedist: Invalid redistribution request");

    Unsigned sendSize, recvSize;
    DeterminePartialRSCommunicateDataSize(B, A, reduceScatterMode, recvSize, sendSize);
    //NOTE: Hack for testing.  We actually need to let the user specify the commModes
    const ModeArray commModes = A.ModeDist(reduceScatterMode);
    const mpi::Comm comm = A.GetCommunicatorForModes(commModes);

    Memory<T> auxMemory;
    T* auxBuf = auxMemory.Require(sendSize + recvSize);
    MemZero(&(auxBuf[0]), sendSize + recvSize);
    T* sendBuf = &(auxBuf[0]);
    T* recvBuf = &(auxBuf[sendSize]);

    PackPartialRSSendBuf(B, A, reduceScatterMode, sendBuf);

    mpi::ReduceScatter(sendBuf, recvBuf, recvSize, comm);

    UnpackPartialRSRecvBuf(recvBuf, reduceScatterMode, A, B);

}

template <typename T>
void ReduceScatterRedist(DistTensor<T>& B, const DistTensor<T>& A, const Mode reduceMode, const Mode scatterMode){
    if(!CheckReduceScatterRedist(B, A, reduceMode, scatterMode))
      LogicError("ReduceScatterRedist: Invalid redistribution request");

    Unsigned sendSize, recvSize;
    DetermineRSCommunicateDataSize(B, A, reduceMode, recvSize, sendSize);
    //NOTE: Hack for testing.  We actually need to let the user specify the commModes
    const ModeArray commModes = A.ModeDist(reduceMode);
    const mpi::Comm comm = A.GetCommunicatorForModes(commModes);

    Memory<T> auxMemory;
    T* auxBuf = auxMemory.Require(sendSize + recvSize);
    MemZero(&(auxBuf[0]), sendSize + recvSize);
    T* sendBuf = &(auxBuf[0]);
    T* recvBuf = &(auxBuf[sendSize]);

    PackRSSendBuf(B, A, reduceMode, scatterMode, sendBuf);

    mpi::ReduceScatter(sendBuf, recvBuf, recvSize, comm);

    UnpackRSRecvBuf(recvBuf, reduceMode, scatterMode, A, B);
}

template <typename T>
void AllGatherRedist(DistTensor<T>& B, const DistTensor<T>& A, const Mode allGatherMode, const ModeArray& redistModes ){
    if(!CheckAllGatherRedist(B, A, allGatherMode, redistModes))
        LogicError("AllGatherRedist: Invalid redistribution request");

    //NOTE: Fix to handle strides in Tensor data
    if(redistModes.size() == 0){
        const Location start(A.Order(), 0);
        T* dst = B.Buffer(start);
        const T* src = A.LockedBuffer(start);
        MemCopy(&(dst[0]), &(src[0]), prod(A.LocalShape()));
        return;
    }
    Unsigned sendSize, recvSize;
    DetermineAGCommunicateDataSize(A, allGatherMode, redistModes, recvSize, sendSize);

    const mpi::Comm comm = A.GetCommunicatorForModes(redistModes);

    Memory<T> auxMemory;
    T* auxBuf = auxMemory.Require(sendSize + recvSize);
    MemZero(&(auxBuf[0]), sendSize + recvSize);

    T* sendBuf = &(auxBuf[0]);
    T* recvBuf = &(auxBuf[sendSize]);

    //printf("Alloc'd %d elems to send and %d elems to receive\n", sendSize, recvSize);
    PackAGSendBuf(A, allGatherMode, sendBuf, redistModes);

    //printf("Allgathering %d elements\n", sendSize);
    mpi::AllGather(sendBuf, sendSize, recvBuf, sendSize, comm);

    UnpackAGRecvBuf(recvBuf, allGatherMode, redistModes, A, B);
    //Print(B.LockedTensor(), "A's local tensor after allgathering:");
}

template <typename T>
void AllGatherRedist(DistTensor<T>& B, const DistTensor<T>& A, const Mode allGatherMode){
	if(!CheckAllGatherRedist(B, A, allGatherMode))
		LogicError("AllGatherRedist: Invalid redistribution request");

	Unsigned sendSize, recvSize;
	DetermineAGCommunicateDataSize(A, allGatherMode, recvSize, sendSize);
	//NOTE: Hack for testing.  We actually need to let the user specify the commModes
	const ModeArray commModes = A.ModeDist(allGatherMode);
	const mpi::Comm comm = A.GetCommunicatorForModes(commModes);

	Memory<T> auxMemory;
	T* auxBuf = auxMemory.Require(sendSize + recvSize);
	MemZero(&(auxBuf[0]), sendSize + recvSize);

	T* sendBuf = &(auxBuf[0]);
	T* recvBuf = &(auxBuf[sendSize]);

	//printf("Alloc'd %d elems to send and %d elems to receive\n", sendSize, recvSize);
	PackAGSendBuf(A, allGatherMode, sendBuf);

	//printf("Allgathering %d elements\n", sendSize);
	mpi::AllGather(sendBuf, sendSize, recvBuf, sendSize, comm);

	UnpackAGRecvBuf(recvBuf, allGatherMode, A, B);
	//Print(B.LockedTensor(), "A's local tensor after allgathering:");
}

template <typename T>
void AllToAllDoubleModeRedist(DistTensor<T>& B, const DistTensor<T>& A, const std::pair<Mode, Mode>& a2aModes, const std::pair<ModeArray, ModeArray >& a2aCommGroups){
    if(!CheckAllToAllDoubleModeRedist(B, A, a2aModes, a2aCommGroups))
        LogicError("AllToAllDoubleModeRedist: Invalid redistribution request");

    Unsigned sendSize, recvSize;

    ModeArray commModes = a2aCommGroups.first;
    commModes.insert(commModes.end(), a2aCommGroups.second.begin(), a2aCommGroups.second.end());
    std::sort(commModes.begin(), commModes.end());

    const mpi::Comm comm = A.GetCommunicatorForModes(commModes);
    DetermineA2ADoubleModeCommunicateDataSize(B, A, a2aModes, a2aCommGroups, recvSize, sendSize);

    Memory<T> auxMemory;
    T* auxBuf = auxMemory.Require(sendSize + recvSize);
    MemZero(&(auxBuf[0]), sendSize + recvSize);

    T* sendBuf = &(auxBuf[0]);
    T* recvBuf = &(auxBuf[sendSize]);

    Unsigned nRedistProcs = prod(FilterVector(A.Grid().Shape(), commModes));

    PackA2ADoubleModeSendBuf(B, A, a2aModes, a2aCommGroups, sendBuf);

    mpi::AllToAll(sendBuf, sendSize/nRedistProcs, recvBuf, recvSize/nRedistProcs, comm);

    UnpackA2ADoubleModeRecvBuf(recvBuf, a2aModes, a2aCommGroups, A, B);
}

template<typename T>
void LocalRedist(DistTensor<T>& B, const DistTensor<T>& A, const Mode localMode, const ModeArray& gridRedistModes){
    if(!CheckLocalRedist(B, A, localMode, gridRedistModes))
        LogicError("LocalRedist: Invalid redistribution request");

    //Packing is what is stored in memory
    UnpackLocalRedist(B, A, localMode, gridRedistModes);
}

//NOTE: Assuming everything is correct, this is just a straight memcopy
template<typename T>
void RemoveUnitModesRedist(DistTensor<T>& B, const DistTensor<T>& A, const ModeArray& unitModes){
    if(!CheckRemoveUnitModesRedist(B, A, unitModes))
        LogicError("RemoveUnitModesRedist: Invalid redistribution request");

    const Unsigned order = A.Order();
    const Location start(order, 0);
    T* dst = B.Buffer(start);
    const T* src = A.LockedBuffer(start);
    MemCopy(&(dst[0]), &(src[0]), prod(A.LocalShape()));
}

//NOTE: Assuming everything is correct, this is just a straight memcopy
template<typename T>
void IntroduceUnitModesRedist(DistTensor<T>& B, const DistTensor<T>& A, const std::vector<Unsigned>& newModePositions){
    if(!CheckIntroduceUnitModesRedist(B, A, newModePositions))
        LogicError("IntroduceUnitModesRedist: Invalid redistribution request");

    const Unsigned order = A.Order();
    const Location start(order, 0);
    T* dst = B.Buffer(start);
    const T* src = A.LockedBuffer(start);
    MemCopy(&(dst[0]), &(src[0]), prod(A.LocalShape()));
}

template <typename T>
void AllToAllRedist(DistTensor<T>& B, const DistTensor<T>& A){
//NOTE: All2All is valid if both distributions are valid
//	if(!CheckAllToAllRedist(B, A))
//		LogicError("AllToAllRedist: Invalid redistribution request");

//	int sendSize, recvSize;
	//Figure out which modes we have to communicate along (to save some pain)
	const ModeArray commModes = DetermineA2ACommunicateModes(B, A);
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
    template void PermutationRedist(DistTensor<T>& B, const DistTensor<T>& A, const Mode permuteMode); \
    template void ReduceScatterRedist(DistTensor<T>& B, const DistTensor<T>& A, const Mode reduceMode, const Mode scatterMode); \
    template void PartialReduceScatterRedist(DistTensor<T>& B, const DistTensor<T>& A, const Mode reduceScatterMode); \
    template void AllGatherRedist(DistTensor<T>& B, const DistTensor<T>& A, const Mode allGatherMode, const ModeArray& redistModes ); \
	template void AllGatherRedist(DistTensor<T>& B, const DistTensor<T>& A, const Mode allGatherMode); \
	template void LocalRedist(DistTensor<T>& B, const DistTensor<T>& A, const Mode localMode, const ModeArray& gridRedistModes); \
	template void AllToAllDoubleModeRedist(DistTensor<T>& B, const DistTensor<T>& A, const std::pair<Mode, Mode>& a2aIndices, const std::pair<ModeArray, ModeArray >& commGroups); \
	template void RemoveUnitModesRedist(DistTensor<T>& B, const DistTensor<T>& A, const ModeArray& unitModes); \
	template void IntroduceUnitModesRedist(DistTensor<T>& B, const DistTensor<T>& A, const std::vector<Unsigned>& newModePositions); \


PROTO(int)
PROTO(float)
PROTO(double)

} // namespace tmen
