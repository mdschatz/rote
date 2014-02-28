#include "tensormental/util/redistribute_util.hpp"


namespace tmen{

template <typename T>
void DetermineRSCommunicateDataSize(const DistTensor<T>& A, const int reduceIndex, int& recvSize, int& sendSize){
	if(!A.Participating())
		return;

	const int reduceIndexMode = A.ModeOfIndex(reduceIndex);
	ModeDistribution reduceIndexDist = A.IndexDist(reduceIndex);
	std::vector<Int> gridViewSlice = FilterVector(A.GridViewShape(), reduceIndexDist);

	const int nRedistProcs = prod(gridViewSlice);
	std::vector<Int> maxLocalShape = MaxLengths(A.Shape(), A.GridViewShape());
	const int nElemsPerProc = prod(maxLocalShape);

	//NOTE: For now we are testing functionality of ReduceScatter to figure out what is appropriate size
	//recvSize = nElemsPerProc / maxLocalShape[reduceIndexMode];
	recvSize = nElemsPerProc;
	sendSize = nElemsPerProc;
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
