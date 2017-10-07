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
#include "rote/core/tensor.hpp"
namespace rote{

//TODO: Check that allToAllIndices and commGroups are valid
template <typename T>
bool DistTensor<T>::CheckAllToAllCommRedist(const DistTensor<T>& A){
	const TensorDistribution outDist = this->TensorDist();
	const TensorDistribution inDist = A.TensorDist();

	bool ret = true;
	ret &= CheckOrder(this->Order(), A.Order());
	ret &= CheckPartition(outDist, inDist);
	ret &= CheckSameCommModes(outDist, inDist);
	ret &= CheckSameNonDist(outDist, inDist);

    return ret;
}

template <typename T>
void DistTensor<T>::AllToAllCommRedist(const DistTensor<T>& A, const ModeArray& commModes, const T alpha){
        if(!this->CheckAllToAllCommRedist(A))
            LogicError("AllToAllDoubleModeRedist: Invalid redistribution request");

        const rote::Grid& g = A.Grid();
        const mpi::Comm comm = this->GetCommunicatorForModes(commModes, g);

        if(!A.Participating())
            return;

        //Determine buffer sizes for communication
        const Unsigned nRedistProcs = Max(1, prod(FilterVector(g.Shape(), commModes)));
        const ObjShape maxLocalShapeA = A.MaxLocalShape();

        const ObjShape gvAShape = A.GridViewShape();
        const ObjShape gvBShape = this->GridViewShape();

        const std::vector<Unsigned> localPackStrides = ElemwiseDivide(LCMs(gvBShape, gvAShape), gvAShape);
        const ObjShape commDataShape = IntCeils(maxLocalShapeA, localPackStrides);

        const Unsigned sendSize = prod(commDataShape);
        const Unsigned recvSize = sendSize;

        T* auxBuf = this->auxMemory_.Require((sendSize + recvSize) * nRedistProcs);

        T* sendBuf = &(auxBuf[0]);
        T* recvBuf = &(auxBuf[sendSize*nRedistProcs]);

//        const T* dataBuf = A.LockedBuffer();
//        PrintArray(dataBuf, A.LocalShape(), A.LocalStrides(), "srcBuf");

        //Pack the data
        PROFILE_SECTION("A2APack");
        this->PackA2ACommSendBuf(A, commModes, commDataShape, sendBuf);
        PROFILE_STOP;

//        ObjShape sendShape = commDataShape;
//        sendShape.insert(sendShape.end(), nRedistProcs);
//        PrintArray(sendBuf, sendShape, "sendBuf");

        //Communicate the data
        PROFILE_SECTION("A2AComm");
        //Realignment
        T* alignSendBuf = &(sendBuf[0]);
        T* alignRecvBuf = &(recvBuf[0]);
        bool didAlign = this->AlignCommBufRedist(A, alignSendBuf, sendSize * nRedistProcs, alignRecvBuf, sendSize * nRedistProcs);
        if(didAlign){
            sendBuf = &(alignRecvBuf[0]);
            recvBuf = &(alignSendBuf[0]);
        }

        mpi::AllToAll(sendBuf, sendSize, recvBuf, recvSize, comm);
        PROFILE_STOP;

//        ObjShape recvShape = commDataShape;
//        recvShape.insert(recvShape.end(), nRedistProcs);
//        PrintArray(recvBuf, recvShape, "recvBuf");

        //Unpack the data (if participating)
        PROFILE_SECTION("A2AUnpack");
        this->UnpackA2ACommRecvBuf(recvBuf, commModes, commDataShape, A, alpha);
        PROFILE_STOP;

//        const T* myBuf = LockedBuffer();
//        PrintArray(myBuf, LocalShape(), LocalStrides(), "myBuf");

        this->auxMemory_.Release();
}

template <typename T>
void DistTensor<T>::PackA2ACommSendBuf(const DistTensor<T>& A, const ModeArray& commModes, const ObjShape& sendShape, T * const sendBuf){
    const Unsigned order = A.Order();
    const T* dataBuf = A.LockedBuffer();

    const Location zeros(order, 0);
    const Location ones(order, 1);

    //GridView information
    const rote::GridView gvA = A.GetGridView();
    const rote::GridView gvB = this->GetGridView();
    const ObjShape gvAShape = gvA.ParticipatingShape();
    const ObjShape gvBShape = gvB.ParticipatingShape();

    //Different striding information
    const std::vector<Unsigned> commLCMs = LCMs(gvAShape, gvBShape);
    const std::vector<Unsigned> modeStrideFactor = ElemwiseDivide(commLCMs, gvAShape);

    const Unsigned nElemsPerProc = prod(sendShape);

    //Grid information
    const rote::Grid& g = this->Grid();
    const ObjShape gridShape = g.Shape();
    const Location myGridLoc = g.Loc();

    const Unsigned nRedistProcsAll = Max(1, prod(FilterVector(gridShape, commModes)));

    //Redistribute information
    ModeArray sortedCommModes = commModes;
    SortVector(sortedCommModes);
    const ObjShape commShape = FilterVector(gridShape, sortedCommModes);

    //For each process we send to, we need to determine the first element we need to send them
    PARALLEL_FOR
    for(Unsigned i = 0; i < nRedistProcsAll; i++){
        Unsigned j;
        //Invert the process order based on the communicator used, to the actual process location
        Location sortedCommLoc = LinearLoc2Loc(i, commShape);
        Location myFirstElemLocA = A.DetermineFirstElem(gvA.ParticipatingLoc());
        Location firstOwnerB = gvB.ToGridLoc(this->Alignments());
        std::vector<Unsigned> alignBinA = g.ToParticipatingGridViewLoc(firstOwnerB, gvA);
        Location sendGridLoc = gvA.ToGridLoc(A.DetermineOwnerNewAlignment(myFirstElemLocA, alignBinA));
        Location procGridLoc = sendGridLoc;

        for(j = 0; j < sortedCommModes.size(); j++){
            procGridLoc[sortedCommModes[j]] = sortedCommLoc[j];
        }

        //Get my first elem location
        Location myFirstLoc = A.DetermineFirstElem(A.GetGridView().ParticipatingLoc());
        Location procFirstLoc = this->DetermineFirstElem(g.ToParticipatingGridViewLoc(procGridLoc, gvB));

        //Iterate to figure out the first elem I need to send p_i
        Location firstSendLoc(order,-1);

        bool found = true;
        for(j = 0; j < myFirstLoc.size(); j++){
            Unsigned myFirstIndex = myFirstLoc[j];
            Unsigned sendFirstIndex = procFirstLoc[j];
            Unsigned myModeStride = A.ModeStride(j);
            Unsigned sendProcModeStride = this->ModeStride(j);

            while(myFirstIndex != sendFirstIndex && myFirstIndex < this->Dimension(j)){
                if(myFirstIndex < sendFirstIndex)
                    myFirstIndex += myModeStride;
                else
                    sendFirstIndex += sendProcModeStride;
            }
            if(myFirstIndex >= this->Dimension(j)){
                found &= false;
                break;
            }
            firstSendLoc[j] = myFirstIndex;
        }

        //Pack the data if we need to send data to p_i
        if(found && ElemwiseLessThan(firstSendLoc, this->Shape())){
            //Determine where the initial piece of data is located.
            const Location localLoc = A.Global2LocalIndex(firstSendLoc);
            Unsigned dataBufPtr = LinearLocFromStrides(PermuteVector(localLoc, A.localPerm_), A.LocalStrides());

            PackData packData;
            packData.srcBufStrides = ElemwiseProd(A.LocalStrides(), PermuteVector(modeStrideFactor, A.localPerm_));

            //Pack into permuted form to minimize striding when unpacking
            ObjShape finalShape = PermuteVector(sendShape, this->localPerm_);
            std::vector<Unsigned> finalStrides = Dimensions2Strides(finalShape);

            //Determine permutation from local output to local input
            Permutation out2in = A.localPerm_.PermutationTo(this->localPerm_).InversePermutation();//DetermineInversePermutation(DeterminePermutation(A.localPerm_, this->localPerm_));

            //Permute pack strides to match input local permutation (for correct packing)
            packData.dstBufStrides = PermuteVector(finalStrides, out2in);

            packData.loopShape = MaxLengths(ElemwiseSubtract(A.LocalShape(), PermuteVector(localLoc, A.localPerm_)), PermuteVector(modeStrideFactor, A.localPerm_));

            PackCommHelper(packData, &(dataBuf[dataBufPtr]), &(sendBuf[i * nElemsPerProc]));
        }
    }
}

template<typename T>
void DistTensor<T>::UnpackA2ACommRecvBuf(const T * const recvBuf, const ModeArray& commModes, const ObjShape& recvShape, const DistTensor<T>& A, const T alpha){

    const Unsigned order = A.Order();
    T* dataBuf = this->Buffer();

    const Location zeros(order, 0);
    const Location ones(order, 1);

    //GridView information
    const rote::GridView gvA = A.GetGridView();
    const rote::GridView gvB = this->GetGridView();
    const ObjShape gvAShape = gvA.ParticipatingShape();
    const ObjShape gvBShape = gvB.ParticipatingShape();

    //Different striding information
    std::vector<Unsigned> commLCMs = rote::LCMs(gvAShape, gvBShape);
    std::vector<Unsigned> modeStrideFactor = ElemwiseDivide(commLCMs, gvBShape);

    const Unsigned nElemsPerProc = prod(recvShape);

    //Grid information
    const rote::Grid& g = this->Grid();
    const ObjShape gridShape = g.Shape();
    const Location myGridLoc = g.Loc();

    const Unsigned nRedistProcsAll = Max(1, prod(FilterVector(gridShape, commModes)));

    //Redistribute information
    ModeArray sortedCommModes = commModes;
    SortVector(sortedCommModes);
    const ObjShape commShape = FilterVector(gridShape, sortedCommModes);

    //For each process we recv from, we need to determine the first element we get from them
    PARALLEL_FOR
    for(Unsigned i = 0; i < nRedistProcsAll; i++){
        Unsigned j;
        //Invert the process order based on the communicator used, to the actual process location
        Location sortedCommLoc = LinearLoc2Loc(i, commShape);
        Location procGridLoc = myGridLoc;

        for(j = 0; j < sortedCommModes.size(); j++){
            procGridLoc[sortedCommModes[j]] = sortedCommLoc[j];
        }

        //Get my first elem location
        Location myFirstLoc = this->DetermineFirstElem(this->GetGridView().ParticipatingLoc());

        //Determine what grid location p_i corresponds to BEFORE alignment has been performed
        //so we correctly determine what elements to unpack
        Location firstOwnerB = gvB.ToGridLoc(this->Alignments());
        Location unpackProcGVA = g.ToParticipatingGridViewLoc(procGridLoc, gvA);
        std::vector<Unsigned> alignBinA = g.ToParticipatingGridViewLoc(firstOwnerB, gvA);
        Location myFirstElemLocAligned = A.DetermineFirstUnalignedElem(unpackProcGVA, alignBinA);
        Location procLocBeforeRealign = gvA.ToGridLoc(A.DetermineOwner(myFirstElemLocAligned));

        Location procFirstLoc = A.DetermineFirstElem(g.ToParticipatingGridViewLoc(procLocBeforeRealign, gvA));

        //Iterate to figure out the first elem I need from p_i
        Location firstRecvLoc(order,-1);

        bool found = true;
        for(j = 0; j < myFirstLoc.size(); j++){
            Unsigned myFirstIndex = myFirstLoc[j];
            Unsigned recvFirstIndex = procFirstLoc[j];
            Unsigned myModeStride = this->ModeStride(j);
            Unsigned recvProcModeStride = A.ModeStride(j);

            while(myFirstIndex != recvFirstIndex && myFirstIndex < this->Dimension(j)){
                if(myFirstIndex < recvFirstIndex)
                    myFirstIndex += myModeStride;
                else
                    recvFirstIndex += recvProcModeStride;
            }
            if(myFirstIndex >= this->Dimension(j)){
                found &= false;
                break;
            }
            firstRecvLoc[j] = myFirstIndex;
        }

        //Unpack the data if we need to recv data from p_i
        if(found && ElemwiseLessThan(firstRecvLoc, this->Shape())){
            //Determine where to place the initial piece of data.
            const Location localLoc = this->Global2LocalIndex(firstRecvLoc);
            Unsigned dataBufPtr = LinearLocFromStrides(PermuteVector(localLoc, this->localPerm_), this->LocalStrides());

            PackData unpackData;
            unpackData.dstBufStrides = ElemwiseProd(this->LocalStrides(), PermuteVector(modeStrideFactor, this->localPerm_));

            //Recv data is permuted the same way our local data is permuted
            ObjShape actualRecvShape = PermuteVector(recvShape, this->localPerm_);
            unpackData.srcBufStrides = Dimensions2Strides(actualRecvShape);

            //Test to fix bug
            unpackData.loopShape = MaxLengths(ElemwiseSubtract(this->LocalShape(), PermuteVector(localLoc, this->localPerm_)), PermuteVector(modeStrideFactor, this->localPerm_));

            if(alpha == T(0))
            	PackCommHelper(unpackData, &(recvBuf[i * nElemsPerProc]), &(dataBuf[dataBufPtr]));
            else{
            	YAxpByData data;
            	data.loopShape = unpackData.loopShape;
            	data.dstStrides = unpackData.dstBufStrides;
            	data.srcStrides = unpackData.srcBufStrides;
            	YAxpBy_fast(T(1), alpha, &(recvBuf[i * nElemsPerProc]), &(dataBuf[dataBufPtr]), data);
            }
        }
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
