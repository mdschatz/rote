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
Int
DistTensor<T>::CheckAllGatherCommRedist(const DistTensor<T>& A, const Mode& allGatherMode, const ModeArray& redistModes){
    if(A.Order() != Order()){
        LogicError("CheckAllGatherRedist: Objects being redistributed must be of same order");
    }

    ModeDistribution allGatherDistA = A.ModeDist(allGatherMode);

    const ModeDistribution check = ConcatenateVectors(ModeDist(allGatherMode), redistModes);
    if(AnyElemwiseNotEqual(check, allGatherDistA)){
        LogicError("CheckAllGatherRedist: [Output distribution ++ redistModes] does not match Input distribution");
    }

    return true;
}

template<typename T>
void
DistTensor<T>::AllGatherCommRedist(const DistTensor<T>& A, const Mode& agMode, const ModeArray& gridModes){
#ifndef RELEASE
    CallStackEntry entry("DistTensor::AllGatherCommRedist");
    if(!CheckAllGatherCommRedist(A, agMode, gridModes))
        LogicError("AllGatherRedist: Invalid redistribution request");
#endif

    const mpi::Comm comm = GetCommunicatorForModes(gridModes, A.Grid());

    if(!A.Participating())
        return;

    //NOTE: Fix to handle strides in Tensor data
    if(gridModes.size() == 0){
        CopyLocalBuffer(A);
        return;
    }
    Unsigned sendSize, recvSize;

    //Determine buffer sizes for communication
    const Unsigned nRedistProcs = Max(1, prod(FilterVector(A.Grid().Shape(), gridModes)));
    const ObjShape maxLocalShapeA = A.MaxLocalShape();

    sendSize = prod(maxLocalShapeA);
    recvSize = sendSize * nRedistProcs;

    Memory<T> auxMemory;
    T* auxBuf = auxMemory.Require(sendSize + recvSize);
    MemZero(&(auxBuf[0]), sendSize + recvSize);

    T* sendBuf = &(auxBuf[0]);
    T* recvBuf = &(auxBuf[sendSize]);

    //printf("Alloc'd %d elems to send and %d elems to receive\n", sendSize, recvSize);
    PackAGCommSendBuf(A, sendBuf);

    //printf("Allgathering %d elements\n", sendSize);
    mpi::AllGather(sendBuf, sendSize, recvBuf, sendSize, comm);

    if(!(Participating()))
        return;
    UnpackAGCommRecvBuf(recvBuf, agMode, gridModes, A);
    //Print(B.LockedTensor(), "A's local tensor after allgathering:");
}

template <typename T>
void DistTensor<T>::PackAGCommSendBuf(const DistTensor<T>& A, T * const sendBuf)
{
  const Unsigned order = A.Order();
  const T* dataBuf = A.LockedBuffer();

  const Location zeros(order, 0);
  const Location ones(order, 1);

  PackData packData;
  packData.loopShape = A.LocalShape();
  packData.srcBufStrides = A.LocalStrides();

  packData.dstBufStrides = Dimensions2Strides(A.MaxLocalShape());

  packData.loopStarts = zeros;
  packData.loopIncs = ones;

  PackCommHelper(packData, order - 1, &(dataBuf[0]), &(sendBuf[0]));
}

template <typename T>
void DistTensor<T>::UnpackAGCommRecvBuf(const T * const recvBuf, const Mode& agMode, const ModeArray& redistModes, const DistTensor<T>& A)
{
    Unsigned order = A.Order();
    Unsigned i;
    T* dataBuf = Buffer();

    const ObjShape gridShape = Grid().Shape();

    const Unsigned nRedistProcs = prod(FilterVector(gridShape, redistModes));
    const ObjShape recvShape = A.MaxLocalShape();

    ModeArray commModes = redistModes;
    std::sort(commModes.begin(), commModes.end());
    const ObjShape redistShape = FilterVector(gridShape, redistModes);
    const ObjShape commShape = FilterVector(gridShape, commModes);
    const Permutation redistPerm = DeterminePermutation(commModes, redistModes);

    const Unsigned nCommElemsPerProc = prod(recvShape);
    const Unsigned agModeStride = LocalModeStride(agMode);
//    printf("recvBuf:");
//    for(Unsigned i = 0; i < nCommElemsPerProc * nRedistProcs; i++){
//        printf(" %d", recvBuf[i]);
//    }
//    printf("\n");

    const Location zeros(order, 0);
    const Location ones(order, 1);

    PackData unpackData;
    unpackData.loopShape = LocalShape();
    unpackData.dstBufStrides = LocalStrides();
    unpackData.dstBufStrides[agMode] *= nRedistProcs;

    unpackData.srcBufStrides = Dimensions2Strides(recvShape);

    unpackData.loopStarts = zeros;
    unpackData.loopIncs = ones;
    unpackData.loopIncs[agMode] = nRedistProcs;

    //NOTE: Check
    for(i = 0; i < nRedistProcs; i++){
        const Location elemCommLoc = LinearLoc2Loc(i, commShape);
        const Unsigned elemRedistLinLoc = Loc2LinearLoc(FilterVector(elemCommLoc, redistPerm), redistShape);

        if(elemRedistLinLoc >= LocalDimension(agMode))
            continue;
        unpackData.loopStarts[agMode] = elemRedistLinLoc;

        PackCommHelper(unpackData, order - 1, &(recvBuf[i * nCommElemsPerProc]), &(dataBuf[elemRedistLinLoc * agModeStride]));
    }

//    printf("dataBuf:");
//    for(Unsigned i = 0; i < prod(LocalShape()); i++)
//        printf(" %d", dataBuf[i]);
//    printf("\n");
}

template<typename T>
void DistTensor<T>::UnpackAGCommRecvBuf(const T * const recvBuf, const ModeArray& changedA2AModes, const ModeArray& commModesAll, const ObjShape& recvShape, const DistTensor<T>& A){
    Unsigned i;
    const Unsigned order = A.Order();
    T* dataBuf = Buffer();

    const tmen::GridView gvA = A.GetGridView();
    const tmen::GridView gvB = GetGridView();

    const Location zeros(order, 0);
    const Location ones(order, 1);

    //------------------------------------
    //------------------------------------
    //------------------------------------

//    ModeArray changedA2AModes = a2aModesFrom;
//    changedA2AModes.insert(changedA2AModes.end(), a2aModesTo.begin(), a2aModesTo.end());
//    std::sort(changedA2AModes.begin(), changedA2AModes.end());
//    changedA2AModes.erase(std::unique(changedA2AModes.begin(), changedA2AModes.end()), changedA2AModes.end());

//    printf("unpack size changedModes: %d\n", changedA2AModes.size());
//    ModeArray commModesAll;
//    for(i = 0; i < commGroups.size(); i++)
//        commModesAll.insert(commModesAll.end(), commGroups[i].begin(), commGroups[i].end());
//    std::sort(commModesAll.begin(), commModesAll.end());

    std::vector<Unsigned> commLCMs = tmen::LCMs(gvA.ParticipatingShape(), gvB.ParticipatingShape());
    std::vector<Unsigned> modeStrideFactor = ElemwiseDivide(commLCMs, gvB.ParticipatingShape());

//    const ObjShape maxLocalShapeA = A.MaxLocalShape();
//    const ObjShape maxLocalShapeB = MaxLocalShape();
//    const ObjShape recvShape = prod(maxLocalShapeA) < prod(maxLocalShapeB) ? maxLocalShapeA : maxLocalShapeB;

//    const Unsigned nRedistProcsAll = prod(FilterVector(g.Shape(), commModesAll));
//    std::cout << "recvBuf:";
//    for(Unsigned i = 0; i < nRedistProcsAll * prod(recvShape); i++){
//        std::cout << " " << recvBuf[i];
//    }
//    std::cout << std::endl;

//    PrintVector(modeStrideFactor, "modeStrideFactor");
    PackData unpackData;
    unpackData.loopShape = LocalShape();
    unpackData.dstBufStrides = ElemwiseProd(LocalStrides(), modeStrideFactor);
    unpackData.srcBufStrides = Dimensions2Strides(recvShape);

    unpackData.loopStarts = zeros;
    unpackData.loopIncs = modeStrideFactor;

//    Unsigned a2aMode2Stride = LocalModeStride(a2aMode2);
//    Unsigned a2aMode1Stride = LocalModeStride(a2aMode1);

    const Location myFirstElemLoc = ModeShifts();

//    PrintVector(unpackElem, "unpackElem");
//    PrintVector(changedA2AModes, "uniqueA2AModesTo");
//    PrintVector(Shape(), "shapeA");
//    PrintVector(LocalShape(), "localShapeA");

    if(ElemwiseLessThan(myFirstElemLoc, A.Shape())){
//        A2AUnpackTestHelper(unpackData, changedA2AModes.size() - 1, commModesAll, changedA2AModes, myFirstElemLoc, myFirstElemLoc, modeStrideFactor, prod(recvShape), A, &(recvBuf[0]), &(dataBuf[0]));
        ElemSelectData elemData;
        elemData.commModes = commModesAll;
        elemData.changedModes = changedA2AModes;
        elemData.packElem = myFirstElemLoc;
        elemData.loopShape = modeStrideFactor;
        elemData.nElemsPerProc = prod(recvShape);

        ElemSelectUnpackHelper(unpackData, elemData, changedA2AModes.size() - 1, A, &(recvBuf[0]), &(dataBuf[0]));
    }
}

#define PROTO(T) \
        template void DistTensor<T>::AllGatherCommRedist(const DistTensor<T>& A, const Mode& agMode, const ModeArray& gridModes); \
        template Int  DistTensor<T>::CheckAllGatherCommRedist(const DistTensor<T>& A, const Mode& allGatherMode, const ModeArray& redistModes); \
        template void DistTensor<T>::PackAGCommSendBuf(const DistTensor<T>& A, T * const sendBuf); \
        template void DistTensor<T>::UnpackAGCommRecvBuf(const T * const recvBuf, const Mode& allGatherMode, const ModeArray& redistModes, const DistTensor<T>& A); \
        template void DistTensor<T>::UnpackAGCommRecvBuf(const T * const recvBuf, const ModeArray& changedA2AModes, const ModeArray& commModesAll, const ObjShape& recvShape, const DistTensor<T>& A);

//template Int CheckAllGatherCommRedist(const DistTensor<T>& B, const DistTensor<T>& A, const Mode& allGatherMode, const ModeArray& redistModes);
//template void AllGatherCommRedist(DistTensor<T>& B, const DistTensor<T>& A, const Mode& allGatherMode, const ModeArray& redistModes );



PROTO(int)
PROTO(float)
PROTO(double)
PROTO(Complex<double>)
PROTO(Complex<float>)

} //namespace tmen
