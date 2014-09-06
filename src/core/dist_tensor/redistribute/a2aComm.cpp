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
void DistTensor<T>::A2APackTestHelper(const PackData& packData, const Mode mode, const ModeArray& commModes, const ModeArray& changedA2AModes, const Location& packElem, const Location& myFirstLoc, const std::vector<Unsigned>& nProcsPerA2AMode, const Unsigned& nElemsPerProc, const DistTensor<T>& A, T const * const dataBuf, T * const sendBuf){

    Unsigned order = A.Order();
    PackData data = packData;
    Location elem = packElem;
    Unsigned i;
    const tmen::GridView gvA = A.GetGridView();
    const tmen::GridView gvB = GetGridView();
    const tmen::Grid& g = Grid();
    const Mode changedA2AMode = changedA2AModes[mode];

//    PrintVector(nProcsPerA2AMode, "nProcsPerA2AMode");
    Unsigned startLoc = packElem[changedA2AMode];
    for(i = 0; i < nProcsPerA2AMode[changedA2AMode]; i++){
        elem[changedA2AMode] = startLoc + i * gvA.ModeWrapStride(changedA2AMode);
//        std::cout << "PackTestHelper mode: " << mode << std::endl;
//        PrintVector(elem, "elem is now");
        if(elem[changedA2AMode] >= A.Dimension(changedA2AMode)){
//            printf("continuing\n");
            continue;
        }
        data.loopStarts[changedA2AMode] = i;

        if(mode == 0){
//            printf("hmm\n");
            Location ownerB = DetermineOwner(elem);
            Location ownerGridLoc = GridViewLoc2GridLoc(ownerB, gvB);
            Unsigned commLinLoc = Loc2LinearLoc(FilterVector(ownerGridLoc, commModes), FilterVector(g.Shape(), commModes));

//            printf("sMode: %d\n", a2aModeTo);
//            PrintVector(ownerB, "ownerB");
//            PrintVector(ownerGridLoc, "ownerGridLoc");
//            PrintVector(commModes, "commModes");
//            printf("commLinLoc: %d\n", commLinLoc);

//            PrintVector(elem, "pack Global elem");
//            PrintVector(data.loopStarts, "local location");

//            std::cout << "offsetting dataBuf by: " << i * A.LocalModeStride(a2aModeTo) << std::endl;
//            std::cout << "offsetting sendBuf by: " << commLinLoc * nElemsPerProc << std::endl;
//            printf("pack data:\n");
//            PrintVector(data.loopShape, "  loop shape");
//            PrintVector(data.loopStarts, "  loop starts");
//            PrintVector(data.loopIncs, "  loop incs");
//            PrintVector(data.srcBufStrides, "  srcBufStrides");
//            PrintVector(data.dstBufStrides, "  dstBufStrides");
            PackCommHelper(data, order - 1, &(dataBuf[i * A.LocalModeStride(changedA2AMode)]), &(sendBuf[commLinLoc * nElemsPerProc]));
//            std::cout << "procs: " << prod(nProcsPerA2AMode) << std::endl;
        }else{

            A2APackTestHelper(data, mode - 1, commModes, changedA2AModes, elem, myFirstLoc, nProcsPerA2AMode, nElemsPerProc, A, &(dataBuf[i * A.LocalModeStride(changedA2AMode)]), &(sendBuf[0]));
        }
    }
}

template <typename T>
void DistTensor<T>::AllToAllCommRedist(const DistTensor<T>& A, const ModeArray& a2aModesFrom, const ModeArray& a2aModesTo, const std::vector<ModeArray>& a2aCommGroups){
//    if(!CheckAllToAllDoubleModeCommRedist(A, a2aModes, a2aCommGroups))
//        LogicError("AllToAllDoubleModeRedist: Invalid redistribution request");

    Unsigned i;
    const tmen::Grid& g = A.Grid();

    //Determine buffer sizes for communication
    ModeArray commModes;
    for(i = 0; i < a2aCommGroups.size(); i++)
        commModes.insert(commModes.end(), a2aCommGroups[i].begin(), a2aCommGroups[i].end());
    std::sort(commModes.begin(), commModes.end());

    ModeArray changedA2AModes = ConcatenateVectors(a2aModesFrom, a2aModesTo);
    std::sort(changedA2AModes.begin(), changedA2AModes.end());
    changedA2AModes.erase(std::unique(changedA2AModes.begin(), changedA2AModes.end()), changedA2AModes.end());

//    PrintVector(commModes, "commModes");
    const mpi::Comm comm = GetCommunicatorForModes(commModes, g);

    if(!A.Participating())
        return;
//    printf("Participating\n");
//    printf("commRank: %d\n", mpi::CommRank(comm));
//    printf("commSize: %d\n", mpi::CommSize(comm));
    Unsigned sendSize, recvSize;

    const Unsigned nRedistProcs = Max(1, prod(FilterVector(g.Shape(), commModes)));
    const ObjShape maxLocalShapeA = A.MaxLocalShape();
    const ObjShape maxLocalShapeB = MaxLocalShape();
    const ObjShape commDataShape = prod(maxLocalShapeA) < prod(maxLocalShapeB) ? maxLocalShapeA : maxLocalShapeB;

    sendSize = prod(commDataShape);
    recvSize = sendSize;

    Memory<T> auxMemory;
    T* auxBuf = auxMemory.Require((sendSize + recvSize) * nRedistProcs);
    MemZero(&(auxBuf[0]), (sendSize + recvSize) * nRedistProcs);

    T* sendBuf = &(auxBuf[0]);
    T* recvBuf = &(auxBuf[sendSize*nRedistProcs]);

    PackA2ACommSendBuf(A, changedA2AModes, commModes, commDataShape, sendBuf);

    mpi::AllToAll(sendBuf, sendSize, recvBuf, recvSize, comm);

    if(!(Participating()))
        return;
    UnpackA2ACommRecvBuf(recvBuf, changedA2AModes, commModes, commDataShape, A);
}

template <typename T>
void DistTensor<T>::PackA2ACommSendBuf(const DistTensor<T>& A, const ModeArray& changedA2AModes, const ModeArray& commModes, const ObjShape& sendShape, T * const sendBuf){
    const Unsigned order = A.Order();
    const T* dataBuf = A.LockedBuffer();

//    std::cout << "dataBuf:";
//    for(Unsigned i = 0; i < prod(A.LocalShape()); i++){
//        std::cout << " " <<  dataBuf[i];
//    }
//    std::cout << std::endl;

    const Location zeros(order, 0);

    //----------------------------------------
    //----------------------------------------
    //----------------------------------------

    const tmen::GridView gvA = A.GetGridView();
    const tmen::GridView gvB = GetGridView();
    const ObjShape gvAShape = gvA.ParticipatingShape();
    const ObjShape gvBShape = gvB.ParticipatingShape();

    const std::vector<Unsigned> commLCMs = LCMs(gvAShape, gvBShape);
    const std::vector<Unsigned> modeStrideFactor = ElemwiseDivide(commLCMs, gvAShape);

    PackData packData;
    packData.loopShape = A.LocalShape();
    packData.srcBufStrides = ElemwiseProd(A.LocalStrides(), modeStrideFactor);

    packData.dstBufStrides = Dimensions2Strides(sendShape);

    packData.loopStarts = zeros;
    packData.loopIncs = modeStrideFactor;

    const Location myFirstElemLoc = A.ModeShifts();

//    PrintVector(packElem, "packElem");
//    PrintVector(changedA2AModes, "changedA2AModes");
//    PrintVector(A.Shape(), "shapeA");
//    PrintVector(A.LocalShape(), "localShapeA");
//    PrintVector(modeStrideFactor, "modeStrideFactor");

    if(ElemwiseLessThan(myFirstElemLoc, A.Shape())){
        A2APackTestHelper(packData, changedA2AModes.size() - 1, commModes, changedA2AModes, myFirstElemLoc, myFirstElemLoc, modeStrideFactor, prod(sendShape), A, &(dataBuf[0]), &(sendBuf[0]));
    }
}

template <typename T>
void DistTensor<T>::A2AUnpackTestHelper(const PackData& packData, const Mode mode, const ModeArray& commModes, const ModeArray& changedA2AModes, const Location& packElem, const Location& myFirstLoc, const std::vector<Unsigned>& nProcsPerA2AMode, const Unsigned& nElemsPerProc, const DistTensor<T>& A, T const * const recvBuf, T * const dataBuf){

    Unsigned order = A.Order();
    PackData data = packData;
    Location elem = packElem;
    Unsigned i;
    const tmen::GridView gvA = A.GetGridView();
    const tmen::GridView gvB = GetGridView();
    const tmen::Grid& g = Grid();
    const Mode changedA2AMode = changedA2AModes[mode];

//    PrintVector(nProcsPerA2AMode, "nProcsPerA2AMode");
    for(i = 0; i < nProcsPerA2AMode[changedA2AMode]; i++){
        elem[changedA2AMode] = myFirstLoc[changedA2AMode] + i * gvB.ModeWrapStride(changedA2AMode);
//        std::cout << "UnpackTestHelper mode: " << mode << std::endl;
//        PrintVector(elem, "elem is now");
        if(elem[changedA2AMode] >= Dimension(changedA2AMode))
            continue;
        data.loopStarts[changedA2AMode] = i;

//        printf("a2aModeTo: %d\n", a2aModeTo);
//        std::cout << "offsetting dataBuf by: " << i * LocalModeStride(a2aModeTo) << std::endl;
        if(mode == 0){
            Location ownerA = A.DetermineOwner(elem);
            Location ownerGridLoc = GridViewLoc2GridLoc(ownerA, gvA);
            Unsigned commLinLoc = Loc2LinearLoc(FilterVector(ownerGridLoc, commModes), FilterVector(g.Shape(), commModes));

//            PrintVector(ownerA, "ownerA");
//            PrintVector(ownerGridLoc, "ownerGridLoc");
//            PrintVector(commModes, "commModes");
//            printf("commLinLoc: %d\n", commLinLoc);

//            PrintVector(elem, "pack Global elem");
//            PrintVector(data.loopStarts, "local location");

//            std::cout << "offsetting recvBuf by: " << commLinLoc * nElemsPerProc << std::endl;
//            printf("pack data:\n");
//            PrintVector(data.loopShape, "  loop shape");
//            PrintVector(data.loopStarts, "  loop starts");
//            PrintVector(data.loopIncs, "  loop incs");
//            PrintVector(data.srcBufStrides, "  srcBufStrides");
//            PrintVector(data.dstBufStrides, "  dstBufStrides");
            PackCommHelper(data, order - 1, &(recvBuf[commLinLoc * nElemsPerProc]), &(dataBuf[i * LocalModeStride(changedA2AMode)]));
//            std::cout << "dataBuf:";
//            for(Unsigned j = 0; j < prod(LocalShape()); j++)
//                std::cout << " " << dataBufOrig[j];
//            std::cout << std::endl;
        }else{
            A2AUnpackTestHelper(data, mode - 1, commModes, changedA2AModes, elem, myFirstLoc, nProcsPerA2AMode, nElemsPerProc, A, &(recvBuf[0]), &(dataBuf[i * LocalModeStride(changedA2AMode)]));
        }
    }
}

template<typename T>
void DistTensor<T>::UnpackA2ACommRecvBuf(const T * const recvBuf, const ModeArray& changedA2AModes, const ModeArray& commModesAll, const ObjShape& recvShape, const DistTensor<T>& A){
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
        A2AUnpackTestHelper(unpackData, changedA2AModes.size() - 1, commModesAll, changedA2AModes, myFirstElemLoc, myFirstElemLoc, modeStrideFactor, prod(recvShape), A, &(recvBuf[0]), &(dataBuf[0]));
    }
}

#define PROTO(T) \
        template Int  DistTensor<T>::CheckAllToAllDoubleModeCommRedist(const DistTensor<T>& A, const std::pair<Mode, Mode>& a2aModes, const std::pair<ModeArray, ModeArray >& a2aCommGroups); \
        template void DistTensor<T>::AllToAllCommRedist(const DistTensor<T>& A, const ModeArray& a2aIndicesFrom, const ModeArray& a2aIndicesTo, const std::vector<ModeArray >& commGroups); \
        template void DistTensor<T>::PackA2ACommSendBuf(const DistTensor<T>& A, const ModeArray& changedA2AModes, const ModeArray& commGroups, const ObjShape& sendShape, T * const sendBuf); \
        template void DistTensor<T>::UnpackA2ACommRecvBuf(const T * const recvBuf, const ModeArray& changedA2AModes, const ModeArray& commGroups, const ObjShape& recvShape, const DistTensor<T>& A); \
        template void DistTensor<T>::A2APackTestHelper(const PackData& packData, const Mode mode, const ModeArray& commModes, const ModeArray& changedA2AModes, const Location& packElem, const Location& myFirstLoc, const std::vector<Unsigned>& nProcsPerA2AMode, const Unsigned& nElemsPerProc, const DistTensor<T>& A, T const * const dataBuf, T * const sendBuf); \
        template void DistTensor<T>::A2AUnpackTestHelper(const PackData& packData, const Mode mode, const ModeArray& commModes, const ModeArray& changedA2AModes, const Location& packElem, const Location& myFirstLoc, const std::vector<Unsigned>& nProcsPerA2AMode, const Unsigned& nElemsPerProc, const DistTensor<T>& A, T const * const recvBuf, T * const dataBuf);

PROTO(int)
PROTO(float)
PROTO(double)
PROTO(Complex<double>)
PROTO(Complex<float>)

} //namespace tmen
