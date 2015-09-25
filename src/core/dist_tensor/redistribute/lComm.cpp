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
#include <algorithm>

namespace tmen{

template<typename T>
bool DistTensor<T>::CheckLocalCommRedist(const DistTensor<T>& A){
	const TensorDistribution outDist = TensorDist();
	const TensorDistribution inDist = A.TensorDist();

	bool ret = true;
	ret &= CheckOrder(Order(), A.Order());
	ret &= CheckInIsPrefix(outDist, inDist);
	ret &= CheckSameNonDist(outDist, inDist);

    return ret;
}

template<typename T>
void DistTensor<T>::LocalCommRedist(const DistTensor<T>& A){
    if(!CheckLocalCommRedist(A))
        LogicError("LocalRedist: Invalid redistribution request");

    if(!(Participating()))
        return;

//    const T* dataBuf = A.LockedBuffer();
//    PrintArray(dataBuf, A.LocalShape(), A.LocalStrides(), "srcBuf");

    const ObjShape commDataShape = A.MaxLocalShape();
    const Unsigned sendSize = prod(commDataShape);
    const Unsigned recvSize = sendSize;

    T* auxBuf = this->auxMemory_.Require(sendSize + recvSize);
    const T* recvBuf = A.LockedBuffer();

//        PrintArray(sendBuf, commDataShape, "sendBuf");

    //Realignment
    const tmen::GridView gvA = A.GetGridView();
    const tmen::GridView gvB = GetGridView();
    const Location firstOwnerA = GridViewLoc2GridLoc(A.Alignments(), gvA);
    const Location firstOwnerB = GridViewLoc2GridLoc(Alignments(), gvB);
    if(AnyElemwiseNotEqual(firstOwnerA, firstOwnerB)){
        T* sendBuf = &(auxBuf[0]);
        PackAGCommSendBuf(A, sendBuf);

    	T* alignSendBuf = &(auxBuf[0]);
    	T* alignRecvBuf = &(auxBuf[sendSize]);
        AlignCommBufRedist(A, alignSendBuf, sendSize, alignRecvBuf, sendSize);

        recvBuf = &(alignRecvBuf[0]);
    }
        //Packing is what is stored in memory
	PROFILE_SECTION("LocalUnpack");
	UnpackLocalCommRedist(A, recvBuf);
	PROFILE_STOP;

//    const T* myBuf = LockedBuffer();
//    PrintArray(myBuf, LocalShape(), LocalStrides(), "myBuf");
	this->auxMemory_.Release();
}


//TODO: Optimize strides when unpacking
//TODO: Check that logic works out (modeStrides being global info applied to local info)
template <typename T>
void DistTensor<T>::UnpackLocalCommRedist(const DistTensor<T>& A, const T* unpackBuf)
{
    Unsigned order = A.Order();
    T* dataBuf = Buffer();

    //GridView information
    const ObjShape gvAShape = A.GetGridView().ParticipatingShape();
    const ObjShape gvBShape = GetGridView().ParticipatingShape();

    //Different striding information
    std::vector<Unsigned> commLCMs = tmen::LCMs(gvAShape, gvBShape);
    std::vector<Unsigned> modeStrideFactor = ElemwiseDivide(commLCMs, gvAShape);

    //Grid information
    const tmen::Grid& g = Grid();

    const Location zeros(order, 0);
    const Location ones(order, 1);

    Permutation in2OutPerm = DeterminePermutation(A.localPerm_, localPerm_);
    PackData unpackData;
    unpackData.loopShape = LocalShape();
    unpackData.dstBufStrides = LocalStrides();
    unpackData.srcBufStrides = PermuteVector(ElemwiseProd(A.LocalStrides(), PermuteVector(modeStrideFactor, A.localPerm_)), in2OutPerm);
    unpackData.loopStarts = zeros;
    unpackData.loopIncs = ones;

    const Location myFirstElemLoc = DetermineFirstElem(GetGridView().ParticipatingLoc());

    if(ElemwiseLessThan(myFirstElemLoc, A.Shape())){
        const Location firstLocInA = A.Global2LocalIndex(myFirstElemLoc);
        Unsigned srcBufPtr = Loc2LinearLoc(PermuteVector(firstLocInA, A.localPerm_), A.LocalShape(), A.LocalStrides());
        PackCommHelper(unpackData, order - 1, &(unpackBuf[srcBufPtr]), &(dataBuf[0]));
    }
}

#define PROTO(T) template class DistTensor<T>
#define COPY(T) \
  template DistTensor<T>::DistTensor( const DistTensor<T>& A )
#define FULL(T) \
  PROTO(T);


FULL(Int);
#ifndef DISABLE_FLOAT
FULL(float);
#endif
FULL(double);

#ifndef DISABLE_COMPLEX
#ifndef DISABLE_FLOAT
FULL(std::complex<float>);
#endif
FULL(std::complex<double>);
#endif

} //namespace tmen
