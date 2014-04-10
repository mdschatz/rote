	#include "tensormental/util/redistribute_util.hpp"


namespace tmen{

//NOTE: B is the output DistTensor, A is the input (consistency among the redistribution routines
template <typename T>
void DeterminePermCommunicateDataSize(const DistTensor<T>& B, const DistTensor<T>& A, const int permuteIndex, int& recvSize, int& sendSize){
    if(!B.Participating())
        return;

    ModeDistribution indexDist = A.IndexDist(permuteIndex);
    std::vector<Int> gridViewSlice = FilterVector(A.GridViewShape(), indexDist);

    std::vector<Int> maxLocalShapeB = MaxLengths(B.Shape(), B.GridView().Shape());

    recvSize = prod(maxLocalShapeB);
    sendSize = recvSize;
}

template <typename T>
void DeterminePartialRSCommunicateDataSize(const DistTensor<T>& B, const DistTensor<T>& A, const int reduceScatterIndex, int& recvSize, int& sendSize){
    if(!B.Participating())
        return;

    ModeDistribution indexDist = A.IndexDist(reduceScatterIndex);

    const int nRedistProcs = Max(1, prod(FilterVector(A.Grid().Shape(), indexDist)));
    std::vector<Int> maxLocalShapeA = MaxLengths(A.Shape(), A.GridView().Shape());

    recvSize = prod(maxLocalShapeA);
    sendSize = recvSize * nRedistProcs;
}

//NOTE: B is the output DistTensor, A is the input (consistency among the redistribution routines
template <typename T>
void DetermineRSCommunicateDataSize(const DistTensor<T>& B, const DistTensor<T>& A, const int reduceIndex, int& recvSize, int& sendSize){
	if(!B.Participating())
		return;

	ModeDistribution indexDist = A.IndexDist(reduceIndex);
	std::vector<Int> gridViewSlice = FilterVector(A.GridViewShape(), indexDist);

	const int nRedistProcs = Max(1, prod(FilterVector(A.Grid().Shape(), indexDist)));
	std::vector<Int> maxLocalShapeA = MaxLengths(A.Shape(), A.GridView().Shape());

	recvSize = prod(maxLocalShapeA);
	sendSize = recvSize * nRedistProcs;
}

template <typename T>
void DetermineAGCommunicateDataSize(const DistTensor<T>& A, const int allGatherIndex, int& recvSize, int& sendSize){
	if(!A.Participating())
		return;

	const int allGatherMode = A.ModeOfIndex(allGatherIndex);

	const int nRedistProcs = A.GridView().Dimension(allGatherMode);
	std::vector<Int> maxLocalShapeA = MaxLengths(A.Shape(), A.GridView().Shape());

	sendSize = prod(maxLocalShapeA);
	recvSize = sendSize * nRedistProcs;
}

template <typename T>
void DetermineA2ADoubleIndexCommunicateDataSize(const DistTensor<T>& B, const DistTensor<T>& A, const std::pair<int, int>& a2aIndices, const std::pair<std::vector<int>, std::vector<int> >& a2aCommModes, int& recvSize, int& sendSize){
    if(!A.Participating())
        return;

    std::vector<int> commModes = a2aCommModes.first;
    commModes.insert(commModes.end(), a2aCommModes.second.begin(), a2aCommModes.second.end());
    std::sort(commModes.begin(), commModes.end());

    const std::vector<int> commGridSlice = FilterVector(B.Grid().Shape(), commModes);
    const int nRedistProcs = prod(commGridSlice);
    std::vector<Int> maxLocalShape = MaxLengths(A.Shape(), A.GridView().Shape());

    sendSize = prod(maxLocalShape) * nRedistProcs;
    recvSize = sendSize;
}

#define PROTO(T) \
    template void DeterminePermCommunicateDataSize(const DistTensor<T>& B, const DistTensor<T>& A, const int permuteIndex, int& recvSize, int& sendSize); \
    template void DeterminePartialRSCommunicateDataSize(const DistTensor<T>& B, const DistTensor<T>& A, const int reduceScatterIndex, int& recvSize, int& sendSize); \
	template void DetermineRSCommunicateDataSize(const DistTensor<T>& B, const DistTensor<T>& A, const int reduceIndex, int& recvSize, int& sendSize); \
	template void DetermineAGCommunicateDataSize(const DistTensor<T>& A, const int allGatherIndex, int& recvSize, int& sendSize); \
	template void DetermineA2ADoubleIndexCommunicateDataSize(const DistTensor<T>& B, const DistTensor<T>& A, const std::pair<int, int>& a2aIndices, const std::pair<std::vector<int>, std::vector<int> >& a2aCommModes, int& recvSize, int& sendSize);

PROTO(int)
PROTO(float)
PROTO(double)
}
