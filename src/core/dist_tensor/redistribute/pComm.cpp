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

//TODO: Check all unaffected indices are distributed similarly (Only done for CheckPermutationRedist currently)
template <typename T>
Int DistTensor<T>::CheckPermutationCommRedist(const DistTensor<T>& A){
    if(A.Order() != Order()){
        LogicError("CheckAllGatherRedist: Objects being redistributed must be of same order");
    }

    const ObjShape gvOutShape = GetGridView().ParticipatingShape();
    const ObjShape gvInShape = A.GetGridView().ParticipatingShape();

    if(AnyElemwiseNotEqual(gvOutShape, gvInShape)){
		std::stringstream msg;
		msg << "Invalid Permutation redistribution\n"
		    << tmen::TensorDistToString(TensorDist())
		    << " <-- "
		    << tmen::TensorDistToString(A.TensorDist())
		    << std::endl
		    << "Mode distributions must correspond to equal logical grid dimensions"
			<< std::endl;
		LogicError(msg.str());
    }

    const tmen::TensorDistribution outDist = TensorDist();
    const tmen::TensorDistribution inDist = TensorDist();
    if(outDist[Order()].size() != inDist[Order()].size() || !EqualUnderPermutation(outDist[Order()], inDist[Order()])){
    	std::stringstream msg;
		msg << "Invalid Permutation redistribution\n"
		    << tmen::TensorDistToString(TensorDist())
		    << " <-- "
		    << tmen::TensorDistToString(A.TensorDist())
		    << std::endl
		    << "Non-distributed mode distribution must shate same modes"
			<< std::endl;
		LogicError(msg.str());
    }
    return true;

}

template <typename T>
void DistTensor<T>::PermutationCommRedist(const DistTensor<T>& A, const ModeArray& commModes){
    if(!CheckPermutationCommRedist(A))
		LogicError("PermutationRedist: Invalid redistribution request");

    const tmen::Grid& g = A.Grid();
    const tmen::GridView gvA = A.GetGridView();
    const tmen::GridView gvB = GetGridView();

    //Ripped from AlignCommBufRedist
    Location firstOwnerA = GridViewLoc2GridLoc(A.Alignments(), gvA);
    Location firstOwnerB = GridViewLoc2GridLoc(Alignments(), gvB);

    std::vector<Unsigned> alignA = A.Alignments();
    std::vector<Unsigned> alignB = Alignments();

    ModeArray misalignedModes;
    for(Unsigned i = 0; i < alignA.size(); i++){
        if(alignA[i] != alignB[i]){
            ModeDistribution modeDist = A.ModeDist(i);
            misalignedModes.insert(misalignedModes.end(), modeDist.begin(), modeDist.end());
        }
    }

    ModeArray actualCommModes = misalignedModes;
    for(Unsigned i = 0; i < commModes.size(); i++){
        if(std::find(actualCommModes.begin(), actualCommModes.end(), commModes[i]) == actualCommModes.end()){
            actualCommModes.insert(actualCommModes.end(), commModes[i]);
        }
    }
    std::sort(actualCommModes.begin(), actualCommModes.end());

//    PrintVector(actualCommModes, "actualCommModes");
    mpi::Comm sendRecvComm = GetCommunicatorForModes(actualCommModes, g);

    //Skip if we aren't participating
    if(!A.Participating())
        return;

    Unsigned sendSize, recvSize;

    //Determine buffer sizes for communication
    //NOTE: Next line is example of clang not detecting dead code/unused var.
//    const ObjShape gridViewSlice = FilterVector(A.GridViewShape(), A.ModeDist(permuteMode));
    const ObjShape commDataShape = MaxLocalShape();
    recvSize = prod(commDataShape);
    sendSize = recvSize;

    T* auxBuf = this->auxMemory_.Require(sendSize + recvSize);
    T* sendBuf = &(auxBuf[0]);
    T* recvBuf = &(auxBuf[sendSize]);

//    const T* dataBuf = A.LockedBuffer();
//    PrintArray(dataBuf, A.LocalShape(), A.LocalStrides(), "srcBuf");

    //Pack the data
    PROFILE_SECTION("PermutationPack");
    PackAGCommSendBuf(A, sendBuf);
    PROFILE_STOP;

//    ObjShape sendShape = commDataShape;
//    PrintArray(sendBuf, sendShape, "sendBuf");

    //Determine who I send+recv data from
    PROFILE_SECTION("PermutationComm");

    ModeArray sortedCommModes = commModes;
    std::sort(sortedCommModes.begin(), sortedCommModes.end());

    const Location myFirstElemA = A.DetermineFirstElem(gvA.ParticipatingLoc());
    const Location ownergvB = DetermineOwner(myFirstElemA);

    const Location myFirstElemB = DetermineFirstElem(gvB.ParticipatingLoc());
    const Location ownergvA = A.DetermineOwner(myFirstElemB);


//    const Location myGridViewLocA = gvA.ParticipatingLoc();
    const Location sendLoc = GridViewLoc2GridLoc(ownergvB, gvB);
    const Unsigned sendLinLoc = Loc2LinearLoc(FilterVector(sendLoc, actualCommModes), FilterVector(g.Shape(), actualCommModes));

//    const Location myGridViewLocB = gvB.ParticipatingLoc();
    const Location recvLoc = GridViewLoc2GridLoc(ownergvA, gvA);
    const Unsigned recvLinLoc = Loc2LinearLoc(FilterVector(recvLoc, actualCommModes), FilterVector(g.Shape(), actualCommModes));

    //Make sure we account for alignments

//    PrintData(A, "Adata");
//    PrintData(*this, "Bdata");
//
//    const Location myLoc = g.Loc();
//    PrintVector(myLoc, "myLoc");
//    PrintVector(gvA.ParticipatingLoc(), "gvALoc");
//    PrintVector(gvB.ParticipatingLoc(), "gvBLoc");
//
//    PrintVector(myFirstElemA, "myFirstElemA");
//    PrintVector(ownergvB, "ownergvB");
//    PrintVector(myFirstElemB, "myFirstElemB");
//    PrintVector(ownergvA, "ownergvA");
//
//    PrintVector(sendLoc, "sendLoc");
//    PrintVector(recvLoc, "recvLoc");
//    printf("myRank: %d, sendRank: %d, recvRank: %d\n", mpi::CommRank(sendRecvComm), sendLinLoc, recvLinLoc);
	mpi::SendRecv(sendBuf, sendSize, sendLinLoc,
				  recvBuf, recvSize, recvLinLoc, sendRecvComm);

    PROFILE_STOP;

//    ObjShape recvShape = commDataShape;
//    PrintArray(recvBuf, recvShape, "recvBuf");

    if(!(Participating())){
        this->auxMemory_.Release();
        return;
    }

    //Unpack the data (if participating)
    PROFILE_SECTION("PermutationUnpack");
    UnpackPCommRecvBuf(recvBuf, A);
    PROFILE_STOP;

//    const T* myBuf = LockedBuffer();
//    PrintArray(myBuf, LocalShape(), LocalStrides(), "myBuf");

    this->auxMemory_.Release();
}

template <typename T>
void DistTensor<T>::UnpackPCommRecvBuf(const T * const recvBuf, const DistTensor<T>& A)
{
    const Unsigned order = Order();
    T* dataBuf = Buffer();

    const Location zeros(order, 0);
    const Location ones(order, 1);

    PackData unpackData;
    unpackData.loopShape = LocalShape();
    unpackData.dstBufStrides = LocalStrides();
    unpackData.srcBufStrides = PermuteVector(Dimensions2Strides(MaxLocalShape()), localPerm_);
    unpackData.loopStarts = zeros;
    unpackData.loopIncs = ones;

    PackCommHelper(unpackData, order - 1, &(recvBuf[0]), &(dataBuf[0]));
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
