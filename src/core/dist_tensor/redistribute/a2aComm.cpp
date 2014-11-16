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
#include "tensormental/core/tensor.hpp"
namespace tmen{

//TODO: Check that allToAllIndices and commGroups are valid
template <typename T>
Int DistTensor<T>::CheckAllToAllDoubleModeCommRedist(const DistTensor<T>& A, const std::pair<Mode, Mode>& allToAllModes, const std::pair<ModeArray, ModeArray >& a2aCommGroups){
    if(A.Order() != Order())
        LogicError("CheckAllToAllDoubleModeRedist: Objects being redistributed must be of same order");
    Unsigned i;
    for(i = 0; i < A.Order(); i++){
        if(i != allToAllModes.first && i != allToAllModes.second){
            if(ModeDist(i) != A.ModeDist(i))
                LogicError("CheckAllToAllDoubleModeRedist: Non-redist modes must have same distribution");
        }
    }
    return 1;
}

template <typename T>
void DistTensor<T>::AllToAllCommRedist(const DistTensor<T>& A, const ModeArray& changedA2AModes, const ModeArray& commModes){
    //    if(!CheckAllToAllDoubleModeCommRedist(A, a2aModes, a2aCommGroups))
    //        LogicError("AllToAllDoubleModeRedist: Invalid redistribution request");

        const tmen::Grid& g = A.Grid();

        const mpi::Comm comm = GetCommunicatorForModes(commModes, g);

        if(!A.Participating())
            return;

        //Determine buffer sizes for communication
        Unsigned sendSize, recvSize;
        const Unsigned nRedistProcs = Max(1, prod(FilterVector(g.Shape(), commModes)));
        const ObjShape maxLocalShapeA = A.MaxLocalShape();
        const ObjShape maxLocalShapeB = MaxLocalShape();

        const tmen::GridView gvA = A.GetGridView();
        const tmen::GridView gvB = GetGridView();
        const ObjShape gvAShape = gvA.ParticipatingShape();
        const ObjShape gvBShape = gvB.ParticipatingShape();

        std::vector<Unsigned> localPackStrides(maxLocalShapeA.size());
        localPackStrides = ElemwiseDivide(LCMs(gvBShape, gvAShape), gvAShape);
        ObjShape commDataShape(maxLocalShapeA.size());
        commDataShape = IntCeils(maxLocalShapeA, localPackStrides);

        sendSize = prod(commDataShape);
        recvSize = sendSize;

        T* auxBuf;
        PROFILE_SECTION("A2ARequire");
        PROFILE_FLOPS((sendSize + recvSize) * nRedistProcs);
        auxBuf = this->auxMemory_.Require((sendSize + recvSize) * nRedistProcs);
        PROFILE_STOP;

        T* sendBuf = &(auxBuf[0]);
        T* recvBuf = &(auxBuf[sendSize*nRedistProcs]);

        //Pack the data
        PROFILE_SECTION("A2APack");
        PROFILE_FLOPS(prod(maxLocalShapeA));
        PackA2ACommSendBuf(A, changedA2AModes, commModes, commDataShape, sendBuf);
        PROFILE_STOP;

        //Communicate the data
        PROFILE_SECTION("A2AComm");
        mpi::AllToAll(sendBuf, sendSize, recvBuf, recvSize, comm);
        PROFILE_STOP;

        if(!(Participating())){
            this->auxMemory_.Release();
            return;
        }

        //Unpack the data (if participating)
        PROFILE_SECTION("A2AUnpack");
        PROFILE_FLOPS(prod(MaxLocalShape()));
        UnpackA2ACommRecvBuf(recvBuf, changedA2AModes, commModes, commDataShape, A);
        PROFILE_STOP;
        this->auxMemory_.Release();
}

template <typename T>
void DistTensor<T>::PackA2ACommSendBuf(const DistTensor<T>& A, const ModeArray& changedA2AModes, const ModeArray& commModes, const ObjShape& sendShape, T * const sendBuf){
    Unsigned i,j;
    const Unsigned order = A.Order();
    const T* dataBuf = A.LockedBuffer();

//    std::cout << "dataBuf:";
//    for(Unsigned i = 0; i < prod(A.LocalShape()); i++){
//        std::cout << " " <<  dataBuf[i];
//    }
//    std::cout << std::endl;

    const Location zeros(order, 0);
    const Location ones(order, 1);

    //GridView information
    const tmen::GridView gvA = A.GetGridView();
    const tmen::GridView gvB = GetGridView();
    const ObjShape gvAShape = gvA.ParticipatingShape();
    const ObjShape gvBShape = gvB.ParticipatingShape();

    //Different striding information
    const std::vector<Unsigned> commLCMs = LCMs(gvAShape, gvBShape);
    const std::vector<Unsigned> modeStrideFactor = ElemwiseDivide(commLCMs, gvAShape);

    const Unsigned nElemsPerProc = prod(sendShape);

    //Grid information
    const tmen::Grid& g = Grid();
    const Location myGridLoc = g.Loc();

    const Unsigned nRedistProcsAll = prod(FilterVector(g.Shape(), commModes));

    //Redistribute information
    ModeArray sortedCommModes = commModes;
    std::sort(sortedCommModes.begin(), sortedCommModes.end());
    const ObjShape commShape = FilterVector(g.Shape(), sortedCommModes);



    const Location myFirstElemLoc = A.ModeShifts();

    PARALLEL_FOR
    for(i = 0; i < nRedistProcsAll; i++){
        Location sortedCommLoc = LinearLoc2Loc(i, commShape);
        Location procGridLoc = myGridLoc;

        for(j = 0; j < sortedCommModes.size(); j++){
            procGridLoc[sortedCommModes[j]] = sortedCommLoc[j];
        }

        Location procGridViewLoc = GridLoc2ParticipatingGridViewLoc(procGridLoc, g.Shape(), TensorDist());
        Location procFirstElemLoc = DetermineFirstElem(procGridViewLoc);

        //Determine the first element I own of recvProc
        //My first loc under B is:
        Location myFirstLoc = A.ModeShifts();
        //Iterate to figure out the first elem I got from recvProc
        Location firstSendLoc(order,0);

        bool found = true;
        for(j = 0; j < myFirstLoc.size(); j++){
            Unsigned myFirstIndex = myFirstLoc[j];
            Unsigned sendFirstIndex = procFirstElemLoc[j];
            Unsigned myModeStride = A.ModeStride(j);
            Unsigned sendProcModeStride = ModeStride(j);

            while(myFirstIndex != sendFirstIndex && myFirstIndex < Dimension(j)){
                if(myFirstIndex < sendFirstIndex)
                    myFirstIndex += myModeStride;
                else
                    sendFirstIndex += sendProcModeStride;
            }
            if(myFirstIndex >= Dimension(j)){
                found &= false;
                break;
            }
            firstSendLoc[j] = myFirstIndex;
        }

        //Cobble into .Buffer()
        if(found && ElemwiseLessThan(firstSendLoc, Shape())){
            const Location localLoc = A.Global2LocalIndex(firstSendLoc);

            Unsigned dataBufPtr = LinearLocFromStrides(PermuteVector(localLoc, A.localPerm_), A.LocalStrides());

            PackData newpackData;
            newpackData.loopShape = ElemwiseSubtract(A.LocalShape(), PermuteVector(localLoc, A.localPerm_));
            newpackData.srcBufStrides = ElemwiseProd(A.LocalStrides(), PermuteVector(modeStrideFactor, A.localPerm_));

            ObjShape finalShape = PermuteVector(sendShape, localPerm_);
            std::vector<Unsigned> finalStrides = Dimensions2Strides(finalShape);

            //Permute the finalStrides to match local permutation strides
            Permutation out2in = DetermineInversePermutation(DeterminePermutation(A.localPerm_, localPerm_));
            newpackData.dstBufStrides = PermuteVector(finalStrides, out2in);

            newpackData.loopStarts = zeros;
            newpackData.loopIncs = PermuteVector(modeStrideFactor, localPerm_);

            PackCommHelper(newpackData, order - 1, &(dataBuf[dataBufPtr]), &(sendBuf[i * nElemsPerProc]));
        }
    }

//    printf("sendBuf:");
//    for(i = 0; i < prod(sendShape)*nRedistProcsAll; i++)
//        std::cout << " " << sendBuf[i];
//    printf("\n");
//
}

template<typename T>
void DistTensor<T>::UnpackA2ACommRecvBuf(const T * const recvBuf, const ModeArray& changedA2AModes, const ModeArray& commModesAll, const ObjShape& recvShape, const DistTensor<T>& A){
    Unsigned i, j;
    const Unsigned order = A.Order();
    T* dataBuf = Buffer();

    const tmen::GridView gvA = A.GetGridView();
    const tmen::GridView gvB = GetGridView();

    const Location zeros(order, 0);
    const Location ones(order, 1);

    std::vector<Unsigned> commLCMs = tmen::LCMs(gvA.ParticipatingShape(), gvB.ParticipatingShape());
    std::vector<Unsigned> modeStrideFactor = ElemwiseDivide(commLCMs, gvB.ParticipatingShape());

    const tmen::Grid& g = Grid();
    const Location myGridLoc = g.Loc();
    const Unsigned nRedistProcsAll = prod(FilterVector(g.Shape(), commModesAll));

//    std::cout << "recvBuf:";
//    for(i = 0; i < nRedistProcsAll * prod(recvShape); i++){
//        std::cout << " " << recvBuf[i];
//    }
//    std::cout << std::endl;

    const Location myFirstElemLoc = ModeShifts();

    const Unsigned nElemsPerProc = prod(recvShape);
    ModeArray sortedCommModes = commModesAll;
    std::sort(sortedCommModes.begin(), sortedCommModes.end());
    const ObjShape commShape = FilterVector(g.Shape(), sortedCommModes);
    Permutation commModesPerm = DeterminePermutation(sortedCommModes, commModesAll);
    const ObjShape distCommShape = PermuteVector(commShape, commModesPerm);

    PARALLEL_FOR
    for(i = 0; i < nRedistProcsAll; i++){
        Location sortedCommLoc = LinearLoc2Loc(i, commShape);
        Location procGridLoc = myGridLoc;

        for(j = 0; j < sortedCommModes.size(); j++){
            procGridLoc[sortedCommModes[j]] = sortedCommLoc[j];
        }

//        PrintVector(procGridLoc, "procGridLoc");
        Location procGridViewLoc = GridLoc2ParticipatingGridViewLoc(procGridLoc, g.Shape(), A.TensorDist());
        Location procFirstElemLoc = A.DetermineFirstElem(procGridViewLoc);

        //Determine the first element I own of recvProc
        //My first loc under B is:
        Location myFirstLoc = ModeShifts();
        //Iterate to figure out the first elem I got from recvProc
        Location firstRecvLoc(order,0);

        bool found = true;
        for(j = 0; j < myFirstLoc.size(); j++){
            Unsigned myFirstIndex = myFirstLoc[j];
            Unsigned recvFirstIndex = procFirstElemLoc[j];
            Unsigned myModeStride = ModeStride(j);
            Unsigned recvProcModeStride = A.ModeStride(j);

            while(myFirstIndex != recvFirstIndex && myFirstIndex < Dimension(j)){
                if(myFirstIndex < recvFirstIndex)
                    myFirstIndex += myModeStride;
                else
                    recvFirstIndex += recvProcModeStride;
            }
            if(myFirstIndex >= Dimension(j)){
                found &= false;
                break;
            }
            firstRecvLoc[j] = myFirstIndex;
        }


        //Cobble into .Buffer()
        if(found && ElemwiseLessThan(firstRecvLoc, Shape())){
            const Location localLoc = Global2LocalIndex(firstRecvLoc);
            Unsigned dataBufPtr = LinearLocFromStrides(PermuteVector(localLoc, localPerm_), LocalStrides());

            PackData unpackData;
            unpackData.loopShape = ElemwiseSubtract(LocalShape(), PermuteVector(localLoc, localPerm_));

            ObjShape actualRecvShape = PermuteVector(recvShape, localPerm_);
            unpackData.srcBufStrides = Dimensions2Strides(actualRecvShape);
            unpackData.dstBufStrides = ElemwiseProd(LocalStrides(), PermuteVector(modeStrideFactor, localPerm_));

            unpackData.loopStarts = zeros;
            unpackData.loopIncs = PermuteVector(modeStrideFactor, localPerm_);

            PackCommHelper(unpackData, order - 1, &(recvBuf[i * nElemsPerProc]), &(dataBuf[dataBufPtr]));
        }
    }

//    printf("dataBuf:");
//    for(i = 0; i < prod(LocalShape()); i++)
//        std::cout << " " << dataBuf[i];
//    printf("\n");
//
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
FULL(Complex<float>);
#endif
FULL(Complex<double>);
#endif

} //namespace tmen
