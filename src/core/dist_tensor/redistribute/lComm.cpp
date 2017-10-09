/*
   Copyright (c) 2009-2013, Jack Poulson
                      2013, Jeff Hammond
                      2013, Jed Brown
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#include "rote.hpp"

namespace rote{

template<typename T>
bool DistTensor<T>::CheckLocalCommRedist(const DistTensor<T>& A){
	const TensorDistribution outDist = this->TensorDist();
	const TensorDistribution inDist = A.TensorDist();

	bool ret = true;
	ret &= CheckOrder(this->Order(), A.Order());
	ret &= CheckInIsPrefix(outDist, inDist);
	ret &= CheckSameNonDist(outDist, inDist);

    return ret;
}

template<typename T>
void DistTensor<T>::LocalCommRedist(const DistTensor<T>& A, const T alpha){
    if(!this->CheckLocalCommRedist(A))
        LogicError("LocalRedist: Invalid redistribution request");

    if(!(this->Participating()))
        return;

//    const T* dataBuf = A.LockedBuffer();
//    PrintArray(dataBuf, A.LocalShape(), A.LocalStrides(), "srcBuf");

    const ObjShape commDataShape = A.MaxLocalShape();
    const Unsigned sendSize = prod(commDataShape);
    const Unsigned recvSize = sendSize;

    T* auxBuf = this->auxMemory_.Require(sendSize + recvSize);
    T* sendBuf = &(auxBuf[0]);
    T* recvBuf = &(auxBuf[sendSize]);

    //Pack the data
    PROFILE_SECTION("LPack");
    this->PackAGCommSendBuf(A, sendBuf);
    PROFILE_STOP;

//	PrintArray(sendBuf, commDataShape, "sendBuf");

    PROFILE_SECTION("LComm");
    //Realignment
    T* alignSendBuf = &(auxBuf[0]);
	T* alignRecvBuf = &(auxBuf[sendSize]);

	bool didAlign = this->AlignCommBufRedist(A, alignSendBuf, sendSize, alignRecvBuf, sendSize);
	if(didAlign){
        sendBuf = &(alignSendBuf[0]);
        recvBuf = &(alignRecvBuf[0]);
	}else{
		recvBuf = sendBuf;
	}

	PROFILE_STOP;

//    PrintArray(recvBuf, commDataShape, "recvBuf");

        //Packing is what is stored in memory
	PROFILE_SECTION("LocalUnpack");
	this->UnpackLocalCommRecvBuf(A, recvBuf, alpha);
	PROFILE_STOP;

//    const T* myBuf = LockedBuffer();
//    PrintArray(myBuf, LocalShape(), LocalStrides(), "myBuf");
	this->auxMemory_.Release();
}

//TODO: Optimize strides when unpacking
//TODO: Check that logic works out (modeStrides being global info applied to local info)
template <typename T>
void DistTensor<T>::UnpackLocalCommRecvBuf(const DistTensor<T>& A, const T* recvBuf, const T alpha)
{

	const Location myFirstLocB = this->DetermineFirstElem(this->GetGridView().ParticipatingLoc());
	const Location myFirstLocA = A.DetermineFirstElem(A.GetGridView().ParticipatingLoc());

	const rote::Grid& g = this->Grid();
    const rote::GridView gvA = A.GetGridView();
    const rote::GridView& gvB = this->GetGridView();
    const ObjShape gvAShape = gvA.ParticipatingShape();
    const ObjShape gvBShape = gvB.ParticipatingShape();
	const Location myGridLoc = g.Loc();
	Location firstOwnerB = gvB.ToGridLoc(this->Alignments());
	Location unpackProcGVA = g.ToParticipatingGridViewLoc(myGridLoc, gvA);
    std::vector<Unsigned> alignBinA = g.ToParticipatingGridViewLoc(firstOwnerB, gvA);
    Location myFirstElemLocAligned = A.DetermineFirstUnalignedElem(unpackProcGVA, alignBinA);

//    PrintVector(myFirstLocB, "firstLocB");
//    PrintVector(myFirstElemLocAligned, "firstAlignedALoc");

    Unsigned dataBufPtr = LinearLocFromStrides(ElemwiseDivide(ElemwiseSubtract(myFirstLocB, myFirstElemLocAligned), gvAShape), Dimensions2Strides(A.MaxLocalShape()));


    const std::vector<Unsigned> commLCMs = LCMs(gvAShape, gvBShape);
    const std::vector<Unsigned> modeStrideFactor = ElemwiseDivide(commLCMs, gvAShape);

    T* dataBuf = this->Buffer();

    PackData unpackData;
    unpackData.loopShape = this->LocalShape();
    unpackData.srcBufStrides = ElemwiseProd(Dimensions2Strides(A.MaxLocalShape()), modeStrideFactor);
    unpackData.dstBufStrides = this->LocalStrides();

//    PrintPackData(unpackData, "unpacking local");
    if(alpha == T(0))
    	PackCommHelper(unpackData, &(recvBuf[dataBufPtr]), &(dataBuf[0]));
    else{
    	YAxpByData data;
    	data.loopShape = unpackData.loopShape;
    	data.dstStrides = unpackData.dstBufStrides;
    	data.srcStrides = unpackData.srcBufStrides;
    	YAxpBy_fast(T(1), alpha, &(recvBuf[dataBufPtr]), &(dataBuf[0]), data);
    }
}

#define FULL(T) \
    template class DistTensor<T>;

FULL(Int)
#ifndef DISABLE_FLOAT
FULL(float)
#endif
FULL(double)

#ifndef DISABLE_COMPLEX
#ifndef DISABLE_FLOAT
FULL(std::complex<float>)
#endif
FULL(std::complex<double>)
#endif

} //namespace rote
