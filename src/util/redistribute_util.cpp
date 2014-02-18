#include "tensormental/util/redistribute/redistribute_util.hpp"


namespace tmen{

template <typename T>
void DetermineRSCommunicateDataSize(const DistTensor<T>& A, const int reduceIndex, int& recvSize, int& sendSize){
	const tmen::Grid& grid = A->Grid();
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

#define PROTO(T) \
	template void DetermineRSCommunicateDataSize(const DistTensor<T>& B, const int reduceIndex, int& recvSize, int& sendSize);

PROTO(int)
PROTO(float)
PROTO(double)
}
