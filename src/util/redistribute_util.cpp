#include "tensormental/util/redistribute_util.hpp"


namespace tmen{

template <typename T>
void DetermineRSCommunicateDataSize(const DistTensor<T>& A, const int reduceIndex, int& recvSize, int& sendSize){
	if(!A.Participating())
		return;

	ModeDistribution indexDist = A.ModeDist(reduceIndex);
	std::vector<Int> gridViewSlice = FilterVector(A.GridViewShape(), indexDist);

	const int nRedistProcs = prod(indexDist);
	std::vector<Int> maxLocalShape = MaxLengths(A.Shape(), A.GridViewShape());

	std::vector<Int> maxLocalShapeAfterReduce = maxLocalShape;
	maxLocalShapeAfterReduce.erase(std::find(maxLocalShapeAfterReduce.begin(), maxLocalShapeAfterReduce.end(), reduceIndex));

	recvSize = prod(maxLocalShapeAfterReduce) / nRedistProcs;
	sendSize = prod(maxLocalShape);
}

template <typename T>
void DetermineAGCommunicateDataSize(const DistTensor<T>& A, const int allGatherIndex, int& recvSize, int& sendSize){
	if(!A.Participating())
		return;

	ModeDistribution indexDist = A.ModeDist(allGatherIndex);
	std::vector<Int> gridViewSlice = FilterVector(A.GridViewShape(), indexDist);

	const int nRedistProcs = prod(gridViewSlice);
	std::vector<Int> maxLocalShape = MaxLengths(A.Shape(), A.GridView().Shape());
	const int nLocalElems = prod(maxLocalShape);
	recvSize = nLocalElems * nRedistProcs;
	sendSize = nLocalElems;
}

#define PROTO(T) \
	template void DetermineRSCommunicateDataSize(const DistTensor<T>& B, const int reduceIndex, int& recvSize, int& sendSize); \
	template void DetermineAGCommunicateDataSize(const DistTensor<T>& A, const int allGatherIndex, int& recvSize, int& sendSize);

PROTO(int)
PROTO(float)
PROTO(double)
}
