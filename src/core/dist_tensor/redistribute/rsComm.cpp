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
Int DistTensor<T>::CheckReduceScatterCommRedist(const DistTensor<T>& A, const Mode reduceMode, const Mode scatterMode){
//    Unsigned i;
//    const tmen::GridView gvA = A.GetGridView();
//
//  //Test elimination of mode
//  const Unsigned AOrder = A.Order();
//  const Unsigned BOrder = B.Order();
//
//  //Check that redist modes are assigned properly on input and output
//  ModeDistribution BScatterModeDist = B.ModeDist(scatterMode);
//  ModeDistribution AReduceModeDist = A.ModeDist(reduceMode);
//  ModeDistribution AScatterModeDist = A.ModeDist(scatterMode);
//
//  //Test elimination of mode
//  if(BOrder != AOrder - 1){
//      LogicError("CheckReduceScatterRedist: Full Reduction requires elimination of mode being reduced");
//  }
//
//  //Test no wrapping of mode to reduce
//  if(A.Dimension(reduceMode) > gvA.Dimension(reduceMode))
//      LogicError("CheckReduceScatterRedist: Full Reduction requires global mode dimension <= gridView dimension");

    //Make sure all indices are distributed similarly between input and output (excluding reduce+scatter indices)

//  for(i = 0; i < BOrder; i++){
//      Mode mode = i;
//      if(mode == scatterMode){
//          ModeDistribution check(BScatterModeDist.end() - AReduceModeDist.size(), BScatterModeDist.end());
//            if(AnyElemwiseNotEqual(check, AReduceModeDist))
//                LogicError("CheckReduceScatterRedist: Reduce mode distribution of A must be a suffix of Scatter mode distribution of B");
//      }
//  }

    return 1;
}

template <typename T>
void DistTensor<T>::ReduceScatterUpdateCommRedist(const T alpha, const DistTensor<T>& A, const T beta, const ModeArray& reduceModes, const ModeArray& commModes){
//    if(!CheckReduceScatterCommRedist(A, reduceMode, scatterMode))
//      LogicError("ReduceScatterRedist: Invalid redistribution request");
    const tmen::Grid& g = A.Grid();

    const mpi::Comm comm = GetCommunicatorForModes(commModes, g);
    const tmen::GridView gvA = A.GetGridView();
    const tmen::GridView gvB = GetGridView();

    if(!A.Participating())
        return;
    Unsigned sendSize, recvSize;

    //Determine buffer sizes for communication
    const Unsigned nRedistProcs = Max(1, prod(FilterVector(g.Shape(), commModes)));
    const ObjShape commDataShape = MaxLocalShape();
//    PrintVector(commDataShape, "commDataShape");
//    printf("nRedistProcs\n", nRedistProcs);
    recvSize = prod(commDataShape);
    sendSize = recvSize * nRedistProcs;
//    printf("sendSize: %d\n", sendSize);

    T* auxBuf;

    //TODO: Figure out how to nicely do this alloc for Alignment
    Location firstOwnerA = GridViewLoc2GridLoc(A.Alignments(), gvA);
    Location firstOwnerB = GridViewLoc2GridLoc(Alignments(), gvB);
    if(AnyElemwiseNotEqual(firstOwnerA, firstOwnerB)){
        auxBuf = this->auxMemory_.Require(sendSize + sendSize);
        MemZero(&(auxBuf[0]), sendSize + sendSize);
//        printf("required: %d elems\n", sendSize + sendSize);
    }else{
        auxBuf = this->auxMemory_.Require(sendSize + recvSize);
        //First, set all entries of sendBuf to zero so we don't accumulate garbage
        MemZero(&(auxBuf[0]), sendSize + recvSize);
    }
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

    Location firstOwnerA = GridViewLoc2GridLoc(A.Alignments(), gvA);
    Location firstOwnerB = GridViewLoc2GridLoc(Alignments(), gvB);
    if(AnyElemwiseNotEqual(firstOwnerA, firstOwnerB)){
//        printf("aligningRS\n");
//        PrintData(A, "A");
//        PrintData(*this, "*this");
        T* alignSendBuf = &(sendBuf[0]);
        T* alignRecvBuf = &(recvBuf[0]);

        AlignCommBufRedist(A, alignSendBuf, sendSize, alignRecvBuf, sendSize);
        sendBuf = &(alignRecvBuf[0]);
        recvBuf = &(alignSendBuf[0]);
    }

    mpi::ReduceScatter(sendBuf, recvBuf, recvSize, comm);
    PROFILE_STOP;

//    ObjShape recvShape = commDataShape;
//    PrintArray(recvBuf, recvShape, "recvBuf");

    if(!Participating()){
        this->auxMemory_.Release();
        return;
    }

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
    Unsigned i,j;
    const Unsigned order = A.Order();
    const T* dataBuf = A.LockedBuffer();

    const Location zeros(order, 0);
    const Location ones(order, 1);

    //GridView information
    const tmen::GridView gvA = A.GetGridView();
    const tmen::GridView gvB = GetGridView();
    const ObjShape gvAShape = gvA.ParticipatingShape();
    const ObjShape gvBShape = gvB.ParticipatingShape();

    //TODO: I know I made a method to get this
    ModeArray nonRModes;
    for(i = 0; i < order; i++){
        if(std::find(rModes.begin(), rModes.end(), i) == rModes.end())
            nonRModes.insert(nonRModes.end(), i);
    }

    //Different striding information
    const std::vector<Unsigned> commLCMs = LCMs(gvAShape, gvBShape);
    std::vector<Unsigned> modeStrideFactor = ElemwiseDivide(commLCMs, gvAShape);
    for(i = 0; i < rModes.size(); i++)
        modeStrideFactor[rModes[i]] = 1;

    const ObjShape sendShape = MaxLocalShape();
    const Unsigned nElemsPerProc = prod(sendShape);

    //Grid information
    const tmen::Grid& g = Grid();
    const ObjShape gridShape = g.Shape();
    const Location myGridLoc = g.Loc();

    const Unsigned nRedistProcsAll = prod(FilterVector(gridShape, commModes));

    //Redistribute information
    ModeArray sortedCommModes = commModes;
    std::sort(sortedCommModes.begin(), sortedCommModes.end());
    const ObjShape commShape = FilterVector(gridShape, sortedCommModes);

//    PrintData(A, "AData");
//    PrintData(*this, "thisData");
    //For each process we send to, we need to determine the first element we need to send them
    PARALLEL_FOR
    for(i = 0; i < nRedistProcsAll; i++){

        //Invert the process order based on the communicator used, to the actual process location
        Location sortedCommLoc = LinearLoc2Loc(i, commShape);
        Location procGridLoc = myGridLoc;

        for(j = 0; j < sortedCommModes.size(); j++){
            procGridLoc[sortedCommModes[j]] = sortedCommLoc[j];
        }

        //This is the first element p_i needs using A's alignment for B.
        Location procGridViewLocB = GridLoc2ParticipatingGridViewLoc(procGridLoc, gridShape, TensorDist());
        Location firstElemOwnerA = GridViewLoc2GridLoc(A.Alignments(), gvA);
        Location alignAinB = ElemwiseMod(GridLoc2ParticipatingGridViewLoc(firstElemOwnerA, gridShape, TensorDist()), ModeStrides());

        Location procFirstElemLoc = DetermineFirstUnalignedElem(procGridViewLocB, alignAinB);
        //Determine where to pack this information as B's alignment may differ from A's (meaning we pack data in different process order).
        Location firstElemOwnerB = GridViewLoc2GridLoc(Alignments(), gvB);
        std::vector<Unsigned> alignDiff = ElemwiseSubtract(firstElemOwnerA, firstElemOwnerB);

        Location adjustedProcGridLoc = ElemwiseMod(ElemwiseSum(ElemwiseSubtract(procGridLoc, alignDiff), gridShape), gridShape);
        Location adjustedProcGridLocSlice = FilterVector(adjustedProcGridLoc, sortedCommModes);
        Unsigned adjustedProcLinLoc = Loc2LinearLoc(adjustedProcGridLocSlice, FilterVector(gridShape, sortedCommModes));
//        printf("i: %d adjustedLoc: %d\n", i, adjustedProcLinLoc);
//        PrintVector(sortedCommModes, "sortedModes");
//        PrintVector(g.Loc(), "myLoc");
//        PrintVector(procGridLoc, "procGridLoc");
//        PrintVector(procFirstElemLoc, "sendBuf first elem is");


        //Determine the first element I need to send to p_i
        //The first element I own is
        Location myFirstLoc = A.DetermineFirstElem(A.GetGridView().ParticipatingLoc());

        //Iterate to figure out the first elem I need to send p_i
        Location firstSendLoc = myFirstLoc;

//        PrintVector(myFirstLoc, "myFirstLoc");
//        PrintVector(procFirstElemLoc, "procFirstElemLoc");
        bool found = true;
        for(j = 0; j < nonRModes.size(); j++){
            Mode nonRMode = nonRModes[j];
            Unsigned myFirstIndex = myFirstLoc[nonRMode];
            Unsigned sendFirstIndex = procFirstElemLoc[nonRMode];
            Unsigned myModeStride = A.ModeStride(nonRMode);
            Unsigned sendProcModeStride = ModeStride(nonRMode);

            while(myFirstIndex != sendFirstIndex && myFirstIndex < Dimension(nonRMode)){
                if(myFirstIndex < sendFirstIndex)
                    myFirstIndex += myModeStride;
                else
                    sendFirstIndex += sendProcModeStride;
            }
            if(myFirstIndex >= Dimension(nonRMode)){
                found &= false;
                break;
            }
            firstSendLoc[nonRMode] = myFirstIndex;
        }

//        if(found)
//            PrintVector(firstSendLoc, "haha sending first loc");
//        else
//            PrintVector(firstSendLoc, "nope sending first loc");

        //Check this is a valid location to pack
        Location firstRecvLoc = firstSendLoc;
        for(j = 0; j < rModes.size(); j++)
            firstRecvLoc[rModes[j]] = 0;

        //Pack the data if we need to send data to p_i
        if(found && ElemwiseLessThan(firstRecvLoc, Shape())){
            //Determine where the initial piece of data is located.
            const Location localLoc = A.Global2LocalIndex(firstSendLoc);
//            PrintVector(localLoc, "localLoc");
            Unsigned dataBufPtr = LinearLocFromStrides(PermuteVector(localLoc, A.localPerm_), A.LocalStrides());
//            printf("dataBufPtr: %d\n", dataBufPtr);

            PackData packData;
            packData.loopShape = ElemwiseSubtract(A.LocalShape(), PermuteVector(localLoc, A.localPerm_));
            packData.srcBufStrides = ElemwiseProd(A.LocalStrides(), PermuteVector(modeStrideFactor, A.localPerm_));

            //Pack into permuted form to minimize striding when unpacking
            ObjShape finalShape = PermuteVector(sendShape, localPerm_);
            std::vector<Unsigned> finalStrides = Dimensions2Strides(finalShape);

            //Determine permutation from local output to local input
            Permutation out2in = DetermineInversePermutation(DeterminePermutation(A.localPerm_, localPerm_));

            //Permute pack strides to match input local permutation (for correct packing)
            packData.dstBufStrides = PermuteVector(finalStrides, out2in);

            packData.loopStarts = zeros;
            //ModeStrideFactor is global information, we need to permute it to match locally
            packData.loopIncs = PermuteVector(modeStrideFactor, localPerm_);

//            PrintPackData(packData, "rsPackData");
            PackCommHelper(packData, order - 1, &(dataBuf[dataBufPtr]), &(sendBuf[adjustedProcLinLoc * nElemsPerProc]));
        }
    }
}

template <typename T>
void DistTensor<T>::UnpackRSUCommRecvBuf(const T * const recvBuf, const T alpha, const DistTensor<T>& A, const T beta)
{
    const Unsigned order = Order();
    T* dataBuf = Buffer();

    const Location zeros(order, 0);
    const Location ones(order, 1);

    YAxpByData data;
    data.loopShape = LocalShape();
    data.srcStrides = PermuteVector(Dimensions2Strides(MaxLocalShape()), localPerm_);
    data.dstStrides = LocalStrides();

    YAxpBy_fast(alpha, beta, &(recvBuf[0]), &(dataBuf[0]), data);
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
