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

//TODO: Properly Check indices and distributions match between input and output
//TODO: FLESH OUT THIS CHECK
template <typename T>
Int DistTensor<T>::CheckReduceToOneCommRedist(const DistTensor<T>& A, const ModeArray& reduceModes){
	Unsigned i;
	if(A.Order() != Order()){
        LogicError("CheckReduceToOneRedist: Objects being redistributed must be of same order");
    }

    const TensorDistribution outDist = TensorDist();
    const TensorDistribution inDist = A.TensorDist();
    ModeDistribution commModes;
    for(i = 0; i < Order(); i++){
    	if(std::find(reduceModes.begin(), reduceModes.end(), i) != reduceModes.end()){
			if(!(IsPrefix(outDist[i], inDist[i]))){
				std::stringstream msg;
				msg << "Invalid Reduce-to-one redistribution\n"
					<< tmen::TensorDistToString(outDist)
					<< " <-- "
					<< tmen::TensorDistToString(inDist)
					<< std::endl
					<< "Output mode-" << i << " mode distribution must be prefix of input mode distribution"
					<< std::endl;
				LogicError(msg.str());
			}
			commModes = ConcatenateVectors(commModes, GetSuffix(outDist[i], inDist[i]));
    	}else{
    		if(outDist[i].size() != inDist[i].size() || !(IsSame(outDist[i], inDist[i]))){
				std::stringstream msg;
				msg << "Invalid Reduce-to-one redistribution\n"
					<< tmen::TensorDistToString(outDist)
					<< " <-- "
					<< tmen::TensorDistToString(inDist)
					<< std::endl
					<< "Output mode-" << i << " mode distribution must be same as input mode distribution"
					<< std::endl;
				LogicError(msg.str());
    		}
    	}
    }

    if(!IsPrefix(inDist[Order()], outDist[Order()])){
    	std::stringstream msg;
		msg << "Invalid Reduce-to-one redistribution\n"
			<< tmen::TensorDistToString(outDist)
			<< " <-- "
			<< tmen::TensorDistToString(inDist)
			<< std::endl
			<< "Output Non-distributed mode distribution cannot be formed"
			<< std::endl;
		LogicError(msg.str());
    }

    const ModeDistribution nonDistSuffix = GetSuffix(outDist[Order()], inDist[Order()]);
    if(nonDistSuffix.size() != commModes.size() || !EqualUnderPermutation(nonDistSuffix, commModes)){
    	std::stringstream msg;
    	msg << "Invalid Reduce-to-one redistribution\n"
			<< tmen::TensorDistToString(outDist)
			<< " <-- "
			<< tmen::TensorDistToString(inDist)
			<< std::endl
			<< "Output Non-distributed mode distribution cannot be formed"
			<< std::endl;
    	LogicError(msg.str());
    }

    return true;
}

template <typename T>
void DistTensor<T>::ReduceToOneUpdateCommRedist(const T alpha, const DistTensor<T>& A, const T beta, const ModeArray& reduceModes, const ModeArray& commModes){
    if(!CheckReduceToOneCommRedist(A, reduceModes))
      LogicError("ReduceToOneRedist: Invalid redistribution request");

    const tmen::Grid& g = A.Grid();
    const tmen::GridView gvA = A.GetGridView();
    const tmen::GridView gvB = GetGridView();

    const mpi::Comm comm = GetCommunicatorForModes(commModes, g);

    if(!A.Participating())
        return;
    Unsigned sendSize, recvSize;

    //Determine buffer sizes for communication
    const ObjShape commDataShape = A.MaxLocalShape();
    sendSize = prod(commDataShape);
    recvSize = sendSize;

    T* auxBuf = this->auxMemory_.Require(sendSize + recvSize);
    T* sendBuf = &(auxBuf[0]);
    T* recvBuf = &(auxBuf[sendSize]);

//    const T* dataBuf = A.LockedBuffer();
//    PrintArray(dataBuf, A.LocalShape(), A.LocalStrides(), "srcBuf");

    MemZero(&(sendBuf[0]), sendSize);

    //Pack the data
    PROFILE_SECTION("RTOPack");
    PackAGCommSendBuf(A, sendBuf);
    PROFILE_STOP;

//    ObjShape sendShape = commDataShape;
//    PrintArray(sendBuf, sendShape, "sendBuf");

    //Communicate the data
    PROFILE_SECTION("RTOComm");
    Location firstOwnerA = GridViewLoc2GridLoc(A.Alignments(), gvA);
    Location firstOwnerB = GridViewLoc2GridLoc(Alignments(), gvB);
    if(AnyElemwiseNotEqual(firstOwnerA, firstOwnerB)){
        T* alignSendBuf = &(sendBuf[0]);
        T* alignRecvBuf = &(recvBuf[0]);

        AlignCommBufRedist(A, alignSendBuf, sendSize, alignRecvBuf, sendSize);
        sendBuf = &(alignRecvBuf[0]);
        recvBuf = &(alignSendBuf[0]);
    }

    mpi::Reduce(sendBuf, recvBuf, sendSize, mpi::SUM, 0, comm);
    PROFILE_STOP;

//    ObjShape recvShape = commDataShape;
//    PrintArray(recvBuf, recvShape, "recvBuf");

    if(!(Participating())){
        this->auxMemory_.Release();
        return;
    }

    //Unpack the data (if participating)
    PROFILE_SECTION("RTOUnpack");
    UnpackRSUCommRecvBuf(recvBuf, alpha, A, beta);
    PROFILE_STOP;

//    const T* myBuf = LockedBuffer();
//    PrintArray(myBuf, LocalShape(), LocalStrides(), "myBuf");

    this->auxMemory_.Release();
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
