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
Int DistTensor<T>::CheckGatherToOneCommRedist(const DistTensor<T>& A){
    if(A.Order() != Order()){
        LogicError("CheckGatherToOneRedist: Objects being redistributed must be of same order");
    }

    const TensorDistribution outDist = TensorDist();
    const TensorDistribution inDist = A.TensorDist();
    ModeDistribution commModes;
    for(Unsigned i = 0; i < Order(); i++){
    	if(!(IsPrefix(outDist[i], inDist[i]))){
    		std::stringstream msg;
    		msg << "Invalid Gather-to-one redistribution\n"
    		    << tmen::TensorDistToString(outDist)
    		    << " <-- "
    		    << tmen::TensorDistToString(inDist)
    		    << std::endl
    		    << "Output mode-" << i << " mode distribution must be prefix of input mode distribution"
    			<< std::endl;
    		LogicError(msg.str());
    	}
    	commModes = ConcatenateVectors(commModes, GetSuffix(outDist[i], inDist[i]));
    }

    if(!IsPrefix(inDist[Order()], outDist[Order()])){
    	std::stringstream msg;
		msg << "Invalid Gather-to-one redistribution\n"
			<< tmen::TensorDistToString(outDist)
			<< " <-- "
			<< tmen::TensorDistToString(inDist)
			<< std::endl
			<< "Output Non-distributed mode distribution cannot be formed"
			<< std::endl;
		LogicError(msg.str());
    }

    ModeDistribution nonDistModes;
    ModeDistribution outNonDist = outDist[Order()];
    ModeDistribution inNonDist = inDist[Order()];
    std::sort(outNonDist.begin(), outNonDist.end());
    std::sort(inNonDist.begin(), inNonDist.end());
    std::set_difference(outNonDist.begin(), outNonDist.end(), inNonDist.begin(), inNonDist.end(), std::inserter(nonDistModes, nonDistModes.end()));

    if(nonDistModes.size() != commModes.size() || !EqualUnderPermutation(nonDistModes, commModes)){
    	std::stringstream msg;
    	msg << "Invalid Gather-to-one redistribution\n"
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
void DistTensor<T>::GatherToOneCommRedist(const DistTensor<T>& A, const ModeArray& commModes){
    if(!CheckGatherToOneCommRedist(A))
      LogicError("GatherToOneRedist: Invalid redistribution request");

    const mpi::Comm comm = GetCommunicatorForModes(commModes, A.Grid());

    if(!A.Participating())
        return;
    Unsigned sendSize, recvSize;
    const tmen::GridView gvA = A.GetGridView();
    const tmen::GridView gvB = GetGridView();

    //Determine buffer sizes for communication
    const Unsigned nRedistProcs = Max(1, prod(FilterVector(A.Grid().Shape(), commModes)));
    const ObjShape maxLocalShapeA = A.MaxLocalShape();
    sendSize = prod(maxLocalShapeA);
    recvSize = sendSize;

    T* auxBuf = this->auxMemory_.Require(sendSize + nRedistProcs*recvSize);
    T* sendBuf = &(auxBuf[0]);
    T* recvBuf = &(auxBuf[sendSize]);

    //Pack the data
    PROFILE_SECTION("GTOPack");
    PackAGCommSendBuf(A, sendBuf);
    PROFILE_STOP;

    //Communicate the data
    PROFILE_SECTION("GTOComm");
    //If unaligned, realign with send/recv BEFORE Allgather (ensures data arrives in correct place)
    Location firstOwnerA = GridViewLoc2GridLoc(A.Alignments(), gvA);
    Location firstOwnerB = GridViewLoc2GridLoc(Alignments(), gvB);
    if(AnyElemwiseNotEqual(firstOwnerA, firstOwnerB)){
//        PrintVector(firstOwnerA, "firstOwnerA");
//        PrintVector(firstOwnerB, "firstOwnerB");
        T* alignSendBuf = &(auxBuf[0]);
        T* alignRecvBuf = &(auxBuf[sendSize]);

        AlignCommBufRedist(A, alignSendBuf, sendSize, alignRecvBuf, sendSize);

        sendBuf = &(alignRecvBuf[0]);
        recvBuf = &(alignSendBuf[0]);
//        PrintArray(sendBuf, commDataShape, "postsendBuf");
    }
    mpi::Gather(sendBuf, sendSize, recvBuf, recvSize, 0, comm);
    PROFILE_STOP;

    if(!(Participating())){
        this->auxMemory_.Release();
        return;
    }

    //Unpack the data (if participating)
    PROFILE_SECTION("GTOUnpack");
    UnpackA2ACommRecvBuf(recvBuf, commModes, maxLocalShapeA, A);
    PROFILE_STOP;

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
