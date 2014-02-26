#include "tensormental/util/redistribute_util.hpp"


namespace tmen{

template <typename T>
void DetermineRSCommunicateDataSize(const DistTensor<T>& A, const int reduceIndex, int& recvSize, int& sendSize){
	const tmen::Grid& grid = A.Grid();
	if(!A.Participating())
		return;

	std::vector<Int> myGridLoc = grid.Loc();
	ModeDistribution indexDist = A.ModeDist(reduceIndex);
	const int nRedistProcs = prod(indexDist);
	std::vector<Int> localShape = A.LocalShape();
	const int nLocalElems = prod(localShape);
	recvSize = nLocalElems;
	sendSize = nLocalElems * nRedistProcs;
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
