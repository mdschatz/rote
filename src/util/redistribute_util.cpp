#include "tensormental/util/redistribute_util.hpp"


namespace tmen{

//NOTE: B is the output DistTensor, A is the input (consistency among the redistribution routines
template <typename T>
void DeterminePermCommunicateDataSize(const DistTensor<T>& B, const DistTensor<T>& A, const Unsigned permuteIndex, Unsigned& recvSize, Unsigned& sendSize){
    if(!B.Participating())
        return;

    const ModeDistribution indexDist = A.IndexDist(permuteIndex);
    const ObjShape gridViewSlice = FilterVector(A.GridViewShape(), indexDist);

    const ObjShape maxLocalShapeB = MaxLengths(B.Shape(), B.GridView().Shape());

    recvSize = prod(maxLocalShapeB);
    sendSize = recvSize;
}

template <typename T>
void DeterminePartialRSCommunicateDataSize(const DistTensor<T>& B, const DistTensor<T>& A, const Index reduceScatterIndex, Unsigned& recvSize, Unsigned& sendSize){
    if(!B.Participating())
        return;

    const ModeDistribution indexDist = A.IndexDist(reduceScatterIndex);

    const Unsigned nRedistProcs = Max(1, prod(FilterVector(A.Grid().Shape(), indexDist)));
    const ObjShape maxLocalShapeA = MaxLengths(A.Shape(), A.GridView().Shape());

    recvSize = prod(maxLocalShapeA);
    sendSize = recvSize * nRedistProcs;
}

//NOTE: B is the output DistTensor, A is the input (consistency among the redistribution routines
template <typename T>
void DetermineRSCommunicateDataSize(const DistTensor<T>& B, const DistTensor<T>& A, const Index reduceIndex, Unsigned& recvSize, Unsigned& sendSize){
	if(!B.Participating())
		return;

	const ModeDistribution indexDist = A.IndexDist(reduceIndex);
	const ObjShape gridViewSlice = FilterVector(A.GridViewShape(), indexDist);

	const Unsigned nRedistProcs = Max(1, prod(FilterVector(A.Grid().Shape(), indexDist)));
	const ObjShape maxLocalShapeA = MaxLengths(A.Shape(), A.GridView().Shape());

	recvSize = prod(maxLocalShapeA);
	sendSize = recvSize * nRedistProcs;
}

template <typename T>
void DetermineAGCommunicateDataSize(const DistTensor<T>& A, const Index allGatherIndex, const ModeArray& redistModes, Unsigned& recvSize, Unsigned& sendSize){
    if(!A.Participating())
        return;

    const Unsigned nRedistProcs = Max(1, prod(FilterVector(A.Grid().Shape(), redistModes)));
    const ObjShape maxLocalShapeA = MaxLengths(A.Shape(), A.GridView().Shape());

    sendSize = prod(maxLocalShapeA);
    recvSize = sendSize * nRedistProcs;
}

template <typename T>
void DetermineAGCommunicateDataSize(const DistTensor<T>& A, const Index allGatherIndex, Unsigned& recvSize, Unsigned& sendSize){
	if(!A.Participating())
		return;

	const Mode allGatherMode = A.ModeOfIndex(allGatherIndex);

	const Unsigned nRedistProcs = A.GridView().Dimension(allGatherMode);
	const ObjShape maxLocalShapeA = MaxLengths(A.Shape(), A.GridView().Shape());

	sendSize = prod(maxLocalShapeA);
	recvSize = sendSize * nRedistProcs;
}

template <typename T>
void DetermineA2ADoubleIndexCommunicateDataSize(const DistTensor<T>& B, const DistTensor<T>& A, const std::pair<Index, Index>& a2aIndices, const std::pair<ModeArray, ModeArray >& a2aCommModes, Unsigned& recvSize, Unsigned& sendSize){
    if(!A.Participating())
        return;

    ModeArray commModes = a2aCommModes.first;
    commModes.insert(commModes.end(), a2aCommModes.second.begin(), a2aCommModes.second.end());
    std::sort(commModes.begin(), commModes.end());

    const ObjShape commGridSlice = FilterVector(B.Grid().Shape(), commModes);
    const Unsigned nRedistProcs = prod(commGridSlice);
    const ObjShape maxLocalShape = MaxLengths(A.Shape(), A.GridView().Shape());

    sendSize = prod(maxLocalShape) * nRedistProcs;
    recvSize = sendSize;
}

#define PROTO(T) \
    template void DeterminePermCommunicateDataSize(const DistTensor<T>& B, const DistTensor<T>& A, const Index permuteIndex, Unsigned& recvSize, Unsigned& sendSize); \
    template void DeterminePartialRSCommunicateDataSize(const DistTensor<T>& B, const DistTensor<T>& A, const Index reduceScatterIndex, Unsigned& recvSize, Unsigned& sendSize); \
	template void DetermineRSCommunicateDataSize(const DistTensor<T>& B, const DistTensor<T>& A, const Index reduceIndex, Unsigned& recvSize, Unsigned& sendSize); \
	template void DetermineAGCommunicateDataSize(const DistTensor<T>& A, const Index allGatherIndex, const ModeArray& redistModes, Unsigned& recvSize, Unsigned& sendSize); \
	template void DetermineAGCommunicateDataSize(const DistTensor<T>& A, const Index allGatherIndex, Unsigned& recvSize, Unsigned& sendSize); \
	template void DetermineA2ADoubleIndexCommunicateDataSize(const DistTensor<T>& B, const DistTensor<T>& A, const std::pair<Index, Index>& a2aIndices, const std::pair<ModeArray, ModeArray >& a2aCommModes, Unsigned& recvSize, Unsigned& sendSize);

PROTO(int)
PROTO(float)
PROTO(double)
}
