#include "tensormental/util/redistribute_util.hpp"


namespace tmen{

//NOTE: B is the output DistTensor, A is the input (consistency among the redistribution routines
template <typename T>
void DeterminePermCommunicateDataSize(const DistTensor<T>& B, const DistTensor<T>& A, const int permuteIndex, int& recvSize, int& sendSize){
    if(!B.Participating())
        return;

    ModeDistribution indexDist = A.IndexDist(permuteIndex);
    std::vector<Int> gridViewSlice = FilterVector(A.GridViewShape(), indexDist);

    const int nRedistProcs = Max(1, prod(gridViewSlice));
    std::vector<Int> maxLocalShapeB = MaxLengths(B.Shape(), B.GridView().Shape());

    recvSize = prod(maxLocalShapeB);
    sendSize = recvSize;
}

template <typename T>
void DeterminePartialRSCommunicateDataSize(const DistTensor<T>& B, const DistTensor<T>& A, const int reduceScatterIndex, int& recvSize, int& sendSize){
    return DetermineRSCommunicateDataSize(B, A, reduceScatterIndex, recvSize, sendSize);
}

//NOTE: B is the output DistTensor, A is the input (consistency among the redistribution routines
template <typename T>
void DetermineRSCommunicateDataSize(const DistTensor<T>& B, const DistTensor<T>& A, const int reduceIndex, int& recvSize, int& sendSize){
	if(!B.Participating())
		return;

	ModeDistribution indexDist = A.IndexDist(reduceIndex);
	std::vector<Int> gridViewSlice = FilterVector(A.GridViewShape(), indexDist);

	const int nRedistProcs = Max(1, prod(gridViewSlice));
	std::vector<Int> maxLocalShapeB = MaxLengths(B.Shape(), B.GridView().Shape());

	recvSize = prod(maxLocalShapeB);
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

#define PROTO(T) \
    template void DeterminePermCommunicateDataSize(const DistTensor<T>& B, const DistTensor<T>& A, const int permuteIndex, int& recvSize, int& sendSize); \
    template void DeterminePartialRSCommunicateDataSize(const DistTensor<T>& B, const DistTensor<T>& A, const int reduceScatterIndex, int& recvSize, int& sendSize); \
	template void DetermineRSCommunicateDataSize(const DistTensor<T>& B, const DistTensor<T>& A, const int reduceIndex, int& recvSize, int& sendSize); \
	template void DetermineAGCommunicateDataSize(const DistTensor<T>& A, const int allGatherIndex, int& recvSize, int& sendSize);

PROTO(int)
PROTO(float)
PROTO(double)
}
