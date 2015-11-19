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
#include <algorithm>

namespace rote{

//TODO: Properly Check indices and distributions match between input and output
//TODO: FLESH OUT THIS CHECK
template <typename T>
bool DistTensor<T>::CheckReduceScatterCommRedist(const DistTensor<T>& A){
	return CheckAllToAllCommRedist(A);
}

template <typename T>
void DistTensor<T>::ReduceScatterUpdateCommRedist(const T alpha, const DistTensor<T>& A, const T beta, const ModeArray& reduceModes, const ModeArray& commModes){
    if(!CheckReduceScatterCommRedist(A))
      LogicError("ReduceScatterRedist: Invalid redistribution request");
    const rote::Grid& g = A.Grid();

    const mpi::Comm comm = this->GetCommunicatorForModes(commModes, g);
    const rote::GridView gvA = A.GetGridView();
    const rote::GridView gvB = this->GetGridView();

    if(!A.Participating())
        return;

    //Determine buffer sizes for communication
    const Unsigned nRedistProcs = Max(1, prod(FilterVector(g.Shape(), commModes)));
    const ObjShape commDataShape = this->MaxLocalShape();

    const Unsigned recvSize = prod(commDataShape);
    const Unsigned sendSize = recvSize * nRedistProcs;

    //NOTE: requiring 2*sendSize in case we realign
	T* auxBuf = this->auxMemory_.Require(sendSize + sendSize);
	MemZero(&(auxBuf[0]), sendSize + sendSize);

	T* sendBuf = &(auxBuf[0]);
    T* recvBuf = &(auxBuf[sendSize]);

//    const T* dataBuf = A.LockedBuffer();
//    PrintArray(dataBuf, A.LocalShape(), A.LocalStrides(), "srcBuf");

    //Pack the data
    PROFILE_SECTION("RSPack");
    PackRSCommSendBuf(A, reduceModes, commModes, sendBuf);
    PROFILE_STOP;

//    ObjShape sendShape = commDataShape;
//    sendShape.insert(sendShape.end(), nRedistProcs);
//    PrintArray(sendBuf, sendShape, "sendBuf");

    //Communicate the data
    PROFILE_SECTION("RSComm");

    T* alignSendBuf = &(sendBuf[0]);
    T* alignRecvBuf = &(recvBuf[0]);

    bool didAlign = AlignCommBufRedist(A, alignSendBuf, sendSize, alignRecvBuf, sendSize);
    if(didAlign){
		sendBuf = &(alignRecvBuf[0]);
		recvBuf = &(alignSendBuf[0]);
    }

    mpi::ReduceScatter(sendBuf, recvBuf, recvSize, comm);
    PROFILE_STOP;

//    ObjShape recvShape = commDataShape;
//    PrintArray(recvBuf, recvShape, "recvBuf");

    //Unpack the data (if participating)
    PROFILE_SECTION("RSUnpack");
    UnpackRSUCommRecvBuf(recvBuf, alpha, A, beta);
    PROFILE_STOP;

//    const T* myBuf = LockedBuffer();
//    PrintArray(myBuf, LocalShape(), LocalStrides(), "myBuf");

    this->auxMemory_.Release();
}

template <typename T>
void DistTensor<T>::PackRSCommSendBuf(const DistTensor<T>& A, const ModeArray& rModes, const ModeArray& commModes, T * const sendBuf)
{
    const Unsigned order = A.Order();
    const T* dataBuf = A.LockedBuffer();

    const Location zeros(order, 0);
    const Location ones(order, 1);

    //GridView information
    const rote::GridView gvA = A.GetGridView();
    const rote::GridView gvB = this->GetGridView();
    const ObjShape gvAShape = gvA.ParticipatingShape();
    const ObjShape gvBShape = gvB.ParticipatingShape();

    //TODO: I know I made a method to get this
    ModeArray nonRModes;
    for(Unsigned i = 0; i < order; i++){
        if(!Contains(rModes, i))
            nonRModes.insert(nonRModes.end(), i);
    }

    //Different striding information
    const std::vector<Unsigned> commLCMs = LCMs(gvAShape, gvBShape);
    std::vector<Unsigned> modeStrideFactor = ElemwiseDivide(commLCMs, gvAShape);
    for(Unsigned i = 0; i < rModes.size(); i++)
        modeStrideFactor[rModes[i]] = 1;

    const ObjShape sendShape = this->MaxLocalShape();
    const Unsigned nElemsPerProc = prod(sendShape);

    //Grid information
    const rote::Grid& g = this->Grid();
    const ObjShape gridShape = g.Shape();
    const Location myGridLoc = g.Loc();

    const Unsigned nRedistProcsAll = prod(FilterVector(gridShape, commModes));

    //Redistribute information
    ModeArray sortedCommModes = commModes;
    SortVector(sortedCommModes);
    const ObjShape commShape = FilterVector(gridShape, sortedCommModes);

//    PrintData(A, "AData");
//    PrintData(*this, "thisData");
    //For each process we send to, we need to determine the first element we need to send them
    PARALLEL_FOR
    for(Unsigned i = 0; i < nRedistProcsAll; i++){
        Unsigned j;
        //Invert the process order based on the communicator used, to the actual process location
        Location sortedCommLoc = LinearLoc2Loc(i, commShape);
//        Location procGridLoc = myGridLoc;
        Location myFirstElemLocA = A.DetermineFirstElem(gvA.ParticipatingLoc());
        Location firstOwnerB = GridViewLoc2GridLoc(this->Alignments(), gvB);
        std::vector<Unsigned> alignBinA = GridLoc2ParticipatingGridViewLoc(firstOwnerB, g.Shape(), A.TensorDist());
        Location sendGridLoc = GridViewLoc2GridLoc(A.DetermineOwnerNewAlignment(myFirstElemLocA, alignBinA), gvA);
        Location procGridLoc = sendGridLoc;

        for(j = 0; j < sortedCommModes.size(); j++){
            procGridLoc[sortedCommModes[j]] = sortedCommLoc[j];
        }
//        PrintVector(procGridLoc, "procGridLoc");
        //This is the first element p_i needs using A's alignment for B.
        //Note the following 4 work, but accidentally assign a valid firstLoc no matter what.
//        Location myFirstLoc = A.DetermineFirstElem(A.GetGridView().ParticipatingLoc());
//        Location myFirstLocUnaligned = ElemwiseMod(ElemwiseSum(ElemwiseSubtract(myFirstLoc, A.Alignments()), A.Shape()), A.Shape());
//        Location procFirstLoc = DetermineFirstElem(GridLoc2ParticipatingGridViewLoc(procGridLoc, gridShape, TensorDist()));
//        Location procFirstLocUnaligned = ElemwiseMod(ElemwiseSum(ElemwiseSubtract(procFirstLoc, Alignments()), Shape()), Shape());

        //Get my first elem location
        Location myFirstLoc = A.DetermineFirstElem(A.GetGridView().ParticipatingLoc());

        //Determine what grid location p_i corresponds to AFTER alignment has been performed
        //so we correctly determine what elements to pack
//        Location firstElemOwnerA = GridViewLoc2GridLoc(A.Alignments(), gvA);
//        Location firstElemOwnerB = GridViewLoc2GridLoc(Alignments(), gvB);
//        Location alignDiff = ElemwiseSubtract(firstElemOwnerB, firstElemOwnerA);
//        Location procLocAfterRealign = ElemwiseMod(ElemwiseSum(ElemwiseSum(procGridLoc, alignDiff), g.Shape()), g.Shape());
        Location procFirstLoc = this->DetermineFirstElem(GridLoc2ParticipatingGridViewLoc(procGridLoc, gridShape, this->TensorDist()));

        //Determine the first element I need to send to p_i
        //The first element I own is
//        Location myFirstLoc = A.DetermineFirstElem(A.GetGridView().ParticipatingLoc());

        //Iterate to figure out the first elem I need to send p_i
        Location firstSendLoc = myFirstLoc;

//        PrintVector(myFirstLoc, "myFirstLoc");
//        PrintVector(adjustedProcFirstElemLoc, "adjustedProcFirstElemLoc");

        bool found = true;
        for(j = 0; j < nonRModes.size(); j++){
            Mode nonRMode = nonRModes[j];
            Unsigned myFirstIndex = myFirstLoc[nonRMode];
            Unsigned sendFirstIndex = procFirstLoc[nonRMode];
//            Unsigned sendFirstIndex = adjustedProcFirstElemLoc[nonRMode];
            Unsigned myModeStride = A.ModeStride(nonRMode);
            Unsigned sendProcModeStride = this->ModeStride(nonRMode);

            while(myFirstIndex != sendFirstIndex && myFirstIndex < this->Dimension(nonRMode)){
                if(myFirstIndex < sendFirstIndex)
                    myFirstIndex += myModeStride;
                else
                    sendFirstIndex += sendProcModeStride;
            }
            if(myFirstIndex >= this->Dimension(nonRMode)){
                found &= false;
                break;
            }
            firstSendLoc[nonRMode] = myFirstIndex;
        }

//        if(found)
//            PrintVector(firstSendLoc, "haha sending first loc");
//        else
//            PrintVector(firstSendLoc, "nope sending first loc");

        //Convert this value back to a Global Loc I own.
//        Location actualFirstSendLoc = ElemwiseMod(ElemwiseSum(firstSendLoc, A.Alignments()), A.Shape());

        //Check this is a valid location to pack
        Location firstRecvLoc = firstSendLoc;
        for(j = 0; j < rModes.size(); j++)
            firstRecvLoc[rModes[j]] = 0;

        //Pack the data if we need to send data to p_i
        if(found && ElemwiseLessThan(firstRecvLoc, this->Shape()) && ElemwiseLessThan(firstSendLoc, A.Shape())){
            //Determine where the initial piece of data is located.
            const Location localLoc = A.Global2LocalIndex(firstSendLoc);
//            PrintVector(localLoc, "localLoc");
            Unsigned dataBufPtr = LinearLocFromStrides(PermuteVector(localLoc, A.localPerm_), A.LocalStrides());

            PackData packData;
            packData.loopShape = MaxLengths(ElemwiseSubtract(A.LocalShape(), PermuteVector(localLoc, A.localPerm_)), PermuteVector(modeStrideFactor, A.localPerm_));
            packData.srcBufStrides = ElemwiseProd(A.LocalStrides(), PermuteVector(modeStrideFactor, A.localPerm_));

            //Pack into permuted form to minimize striding when unpacking
            ObjShape finalShape = PermuteVector(sendShape, this->localPerm_);
            std::vector<Unsigned> finalStrides = Dimensions2Strides(finalShape);

            //Determine permutation from local output to local input
            Permutation out2in = DetermineInversePermutation(DeterminePermutation(A.localPerm_, this->localPerm_));

            //Permute pack strides to match input local permutation (for correct packing)
            packData.dstBufStrides = PermuteVector(finalStrides, out2in);

//            PrintPackData(packData, "rsPackData");
            PackCommHelper(packData, &(dataBuf[dataBufPtr]), &(sendBuf[i * nElemsPerProc]));
        }
    }
}

template <typename T>
void DistTensor<T>::UnpackRSUCommRecvBuf(const T * const recvBuf, const T alpha, const DistTensor<T>& A, const T beta)
{
    const Unsigned order = this->Order();
    T* dataBuf = this->Buffer();

    const Location zeros(order, 0);
    const Location ones(order, 1);

    YAxpByData data;
    data.loopShape = this->LocalShape();
    data.srcStrides = Dimensions2Strides(PermuteVector(this->MaxLocalShape(), this->localPerm_));
    data.dstStrides = this->LocalStrides();

    YAxpBy_fast(alpha, beta, &(recvBuf[0]), &(dataBuf[0]), data);
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
