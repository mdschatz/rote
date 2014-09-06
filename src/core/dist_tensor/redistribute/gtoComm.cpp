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
Int DistTensor<T>::CheckGatherToOneCommRedist(const DistTensor<T>& A, const Mode rMode, const ModeArray& gridModes){
//    Unsigned i;
//    const tmen::GridView gvA = A.GetGridView();
//
//  //Test elimination of mode
//  const Unsigned AOrder = A.Order();
//  const Unsigned BOrder = B.Order();
//
//  //Check that redist modes are assigned properly on input and output
//  ModeDistribution BScatterModeDist = B.ModeDist(scatterMode);
//  ModeDistribution AGatherModeDist = A.ModeDist(reduceMode);
//  ModeDistribution AScatterModeDist = A.ModeDist(scatterMode);
//
//  //Test elimination of mode
//  if(BOrder != AOrder - 1){
//      LogicError("CheckGatherScatterRedist: Full Reduction requires elimination of mode being reduced");
//  }
//
//  //Test no wrapping of mode to reduce
//  if(A.Dimension(reduceMode) > gvA.Dimension(reduceMode))
//      LogicError("CheckGatherScatterRedist: Full Reduction requires global mode dimension <= gridView dimension");

    //Make sure all indices are distributed similarly between input and output (excluding reduce+scatter indices)

//  for(i = 0; i < BOrder; i++){
//      Mode mode = i;
//      if(mode == scatterMode){
//          ModeDistribution check(BScatterModeDist.end() - AGatherModeDist.size(), BScatterModeDist.end());
//            if(AnyElemwiseNotEqual(check, AGatherModeDist))
//                LogicError("CheckGatherScatterRedist: Gather mode distribution of A must be a suffix of Scatter mode distribution of B");
//      }
//  }

    return 1;
}

template <typename T>
void DistTensor<T>::GatherToOneCommRedist(const DistTensor<T>& A, const ModeArray& gatherModes, const std::vector<ModeArray>& commGroups){
//    if(!CheckGatherToOneCommRedist(A, gatherMode, commGroups))
//      LogicError("GatherToOneRedist: Invalid redistribution request");

    Unsigned i;
    const tmen::Grid& g = A.Grid();

    ModeArray commModes;
    for(i = 0; i < commGroups.size(); i++)
        commModes.insert(commModes.end(), commGroups[i].begin(), commGroups[i].end());
    std::sort(commModes.begin(), commModes.end());

    //NOTE: Hack for testing.  We actually need to let the user specify the commModes
    //NOTE: THIS NEEDS TO BE BEFORE Participating() OTHERWISE PROCESSES GET OUT OF SYNC
    const mpi::Comm comm = GetCommunicatorForModes(commModes, A.Grid());

    if(!A.Participating())
        return;
    Unsigned sendSize, recvSize;

    //Determine buffer sizes for communication
    const Unsigned nRedistProcs = Max(1, prod(FilterVector(A.Grid().Shape(), commModes)));
    const ObjShape maxLocalShapeA = A.MaxLocalShape();
    sendSize = prod(maxLocalShapeA);
    recvSize = sendSize;

    Memory<T> auxMemory;
    T* auxBuf = auxMemory.Require(sendSize + nRedistProcs*recvSize);
    MemZero(&(auxBuf[0]), sendSize + recvSize);
    T* sendBuf = &(auxBuf[0]);
    T* recvBuf = &(auxBuf[sendSize]);

    PackGTOCommSendBuf(A, sendBuf);

    mpi::Gather(sendBuf, sendSize, recvBuf, recvSize, 0, comm);

    if(!(Participating()))
        return;
    UnpackGTOCommRecvBuf(recvBuf, gatherModes, commModes, maxLocalShapeA, A);
}

template <typename T>
void DistTensor<T>::PackGTOCommSendBuf(const DistTensor<T>& A, T * const sendBuf)
{
    Unsigned order = A.Order();
    const T* dataBuf = A.LockedBuffer();

//    printf("dataBuf: ");
//    for(Unsigned i = 0; i < prod(A.LocalShape()); i++){
//        printf("%d ", dataBuf[i]);
//    }
//    printf("\n");

    const Location zeros(order, 0);
    const Location ones(order, 1);

    PackData packData;
    packData.loopShape = A.LocalShape();
    packData.srcBufStrides = A.LocalStrides();

    packData.dstBufStrides.resize(order);
    packData.dstBufStrides = Dimensions2Strides(A.MaxLocalShape());

    packData.loopStarts = zeros;
    packData.loopIncs = ones;

    PackCommHelper(packData, order - 1, &(dataBuf[0]), &(sendBuf[0]));

//    printf("packed sendBuf: ");
//    for(Unsigned i = 0; i < prod(packData.sendShape); i++)
//        printf("%d ", sendBuf[i]);
//    printf("\n");
}

template<typename T>
void DistTensor<T>::UnpackGTOCommRecvBuf(const T * const recvBuf, const ModeArray& changedGTOModes, const ModeArray& commModesAll, const ObjShape& recvShape, const DistTensor<T>& A){
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
        elemData.changedModes = changedGTOModes;
        elemData.packElem = myFirstElemLoc;
        elemData.loopShape = modeStrideFactor;
        elemData.nElemsPerProc = prod(recvShape);

        ElemSelectUnpackHelper(unpackData, elemData, changedGTOModes.size() - 1, A, &(recvBuf[0]), &(dataBuf[0]));
    }
}

#define PROTO(T) \
        template Int  DistTensor<T>::CheckGatherToOneCommRedist(const DistTensor<T>& A, const Mode gMode, const ModeArray& gridModes); \
        template void DistTensor<T>::GatherToOneCommRedist(const DistTensor<T>& A, const ModeArray& gatherModes, const std::vector<ModeArray>& commGroups); \
        template void DistTensor<T>::PackGTOCommSendBuf(const DistTensor<T>& A, T * const sendBuf); \
        template void DistTensor<T>::UnpackGTOCommRecvBuf(const T * const recvBuf, const ModeArray& changedGTOModes, const ModeArray& commModesAll, const ObjShape& recvShape, const DistTensor<T>& A);

PROTO(int)
PROTO(float)
PROTO(double)
PROTO(Complex<float>)
PROTO(Complex<double>)

} //namespace tmen
