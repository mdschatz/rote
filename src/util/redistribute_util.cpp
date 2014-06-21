#include "tensormental/util/redistribute_util.hpp"
#include "tensormental.hpp"

namespace tmen{

//NOTE: B is the output DistTensor, A is the input (consistency among the redistribution routines
template <typename T>
void DistTensor<T>::DeterminePermCommunicateDataSize(const DistTensor<T>& A, const Mode permuteMode, Unsigned& recvSize, Unsigned& sendSize){
    if(!this->Participating())
        return;

    const ModeDistribution modeDist = A.ModeDist(permuteMode);
    const ObjShape gridViewSlice = FilterVector(A.GridViewShape(), modeDist);

    const ObjShape maxLocalShapeB = MaxLengths(this->Shape(), this->GridView().Shape());

    recvSize = prod(maxLocalShapeB);
    sendSize = recvSize;
}

template <typename T>
void DistTensor<T>::DeterminePartialRSCommunicateDataSize(const DistTensor<T>& A, const Mode reduceScatterMode, Unsigned& recvSize, Unsigned& sendSize){
    if(!this->Participating())
        return;

    const ModeDistribution modeDist = A.ModeDist(reduceScatterMode);

    const Unsigned nRedistProcs = Max(1, prod(FilterVector(A.Grid().Shape(), modeDist)));
    const ObjShape maxLocalShapeA = MaxLengths(A.Shape(), A.GridView().Shape());

    recvSize = prod(maxLocalShapeA);
    sendSize = recvSize * nRedistProcs;
}

//NOTE: B is the output DistTensor, A is the input (consistency among the redistribution routines
template <typename T>
void DistTensor<T>::DetermineRSCommunicateDataSize(const DistTensor<T>& A, const Mode reduceMode, Unsigned& recvSize, Unsigned& sendSize){
	if(!this->Participating())
		return;

	const ModeDistribution modeDist = A.ModeDist(reduceMode);
	const ObjShape gridViewSlice = FilterVector(A.GridViewShape(), modeDist);

	const Unsigned nRedistProcs = Max(1, prod(FilterVector(A.Grid().Shape(), modeDist)));
	const ObjShape maxLocalShapeA = MaxLengths(A.Shape(), A.GridView().Shape());

	recvSize = prod(maxLocalShapeA);
	sendSize = recvSize * nRedistProcs;
}

template <typename T>
void DistTensor<T>::DetermineAGCommunicateDataSize(const DistTensor<T>& A, const Mode allGatherMode, const ModeArray& redistModes, Unsigned& recvSize, Unsigned& sendSize){
    if(!A.Participating())
        return;

    const Unsigned nRedistProcs = Max(1, prod(FilterVector(A.Grid().Shape(), redistModes)));
    const ObjShape maxLocalShapeA = MaxLengths(A.Shape(), A.GridView().Shape());

    sendSize = prod(maxLocalShapeA);
    recvSize = sendSize * nRedistProcs;
}

template <typename T>
void DistTensor<T>::DetermineAGCommunicateDataSize(const DistTensor<T>& A, const Mode allGatherMode, Unsigned& recvSize, Unsigned& sendSize){
	if(!A.Participating())
		return;

	const Unsigned nRedistProcs = A.GridView().Dimension(allGatherMode);
	const ObjShape maxLocalShapeA = MaxLengths(A.Shape(), A.GridView().Shape());

	sendSize = prod(maxLocalShapeA);
	recvSize = sendSize * nRedistProcs;
}

template <typename T>
void DistTensor<T>::DetermineA2ADoubleModeCommunicateDataSize(const DistTensor<T>& A, const std::pair<Mode, Mode>& a2aModes, const std::pair<ModeArray, ModeArray >& a2aCommModes, Unsigned& recvSize, Unsigned& sendSize){
    if(!A.Participating())
        return;

    ModeArray commModes = a2aCommModes.first;
    commModes.insert(commModes.end(), a2aCommModes.second.begin(), a2aCommModes.second.end());
    std::sort(commModes.begin(), commModes.end());

    const ObjShape commGridSlice = FilterVector(this->Grid().Shape(), commModes);
    const Unsigned nRedistProcs = prod(commGridSlice);
    const ObjShape maxLocalShape = MaxLengths(A.Shape(), A.GridView().Shape());

    sendSize = prod(maxLocalShape) * nRedistProcs;
    recvSize = sendSize;
}

#define PROTO(T) \
    template void DistTensor<T>::DeterminePermCommunicateDataSize(const DistTensor<T>& A, const Mode permuteMode, Unsigned& recvSize, Unsigned& sendSize); \
    template void DistTensor<T>::DeterminePartialRSCommunicateDataSize(const DistTensor<T>& A, const Mode reduceScatterMode, Unsigned& recvSize, Unsigned& sendSize); \
	template void DistTensor<T>::DetermineRSCommunicateDataSize(const DistTensor<T>& A, const Mode reduceMode, Unsigned& recvSize, Unsigned& sendSize); \
	template void DistTensor<T>::DetermineAGCommunicateDataSize(const DistTensor<T>& A, const Mode allGatherMode, const ModeArray& redistModes, Unsigned& recvSize, Unsigned& sendSize); \
	template void DistTensor<T>::DetermineAGCommunicateDataSize(const DistTensor<T>& A, const Mode allGatherMode, Unsigned& recvSize, Unsigned& sendSize); \
	template void DistTensor<T>::DetermineA2ADoubleModeCommunicateDataSize(const DistTensor<T>& A, const std::pair<Mode, Mode>& a2aModes, const std::pair<ModeArray, ModeArray >& a2aCommModes, Unsigned& recvSize, Unsigned& sendSize);

PROTO(int)
PROTO(float)
PROTO(double)
PROTO(Complex<double>)
PROTO(Complex<float>)
}
