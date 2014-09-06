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
void DistTensor<T>::PackTestHelper(const PackData& packData, const Mode mode, const ModeArray& commModes, const ModeArray& sModes, const Location& packElem, const Location& myFirstLoc, const std::vector<Unsigned>& nProcsPerRMode, const Unsigned& nElemsPerProc, const DistTensor<T>& A, T const * const dataBuf, T * const sendBuf, T* const sendBufOrig){

    Unsigned order = A.Order();
    PackData data = packData;
    Location elem = packElem;
    Unsigned i;
    const tmen::GridView gvA = A.GetGridView();
    const tmen::GridView gvB = GetGridView();
    const tmen::Grid& g = Grid();
    const Mode sMode = sModes[mode];

//    PrintVector(nProcsPerRMode, "nProcsPerRMode");
    for(i = 0; i < nProcsPerRMode[mode]; i++){
        elem[sMode] = myFirstLoc[sMode] + i * gvA.ModeWrapStride(sMode);
//        std::cout << "PackTestHelper mode: " << mode << std::endl;
//        PrintVector(elem, "elem is now");
        if(elem[sMode] >= A.Dimension(sMode))
            continue;
        data.loopStarts[sMode] = i;

//        std::cout << "offsetting dataBuf by: " << i * A.LocalModeStride(sMode) << std::endl;
        if(mode == 0){
            Location ownerB = DetermineOwner(elem);
            Location ownerGridLoc = GridViewLoc2GridLoc(ownerB, gvB);
            Unsigned commLinLoc = Loc2LinearLoc(FilterVector(ownerGridLoc, commModes), FilterVector(g.Shape(), commModes));

//            printf("sMode: %d\n", sMode);
//            PrintVector(ownerB, "ownerB");
//            PrintVector(ownerGridLoc, "ownerGridLoc");
//            PrintVector(commModes, "commModes");
//            printf("commLinLoc: %d\n", commLinLoc);

//            PrintVector(elem, "pack Global elem");
//            PrintVector(data.loopStarts, "local location");

//            printf("pack data:\n");
//            PrintVector(data.loopShape, "  loop shape");
//            PrintVector(data.loopStarts, "  loop starts");
//            PrintVector(data.loopIncs, "  loop incs");
//            PrintVector(data.srcBufStrides, "  srcBufStrides");
//            PrintVector(data.dstBufStrides, "  dstBufStrides");
//            std::cout << "offsetting sendBuf by: " << commLinLoc * nElemsPerProc << std::endl;
            PackCommHelper(data, order - 1, &(dataBuf[i * A.LocalModeStride(sMode)]), &(sendBuf[commLinLoc * nElemsPerProc]));
//            std::cout << "packed sendBuf:";
//            for(Unsigned i = 0; i < nElemsPerProc * prod(nProcsPerRMode); i++)
//                std::cout << " " << sendBufOrig[i];
//            std::cout << std::endl;
        }else{

            PackTestHelper(data, mode - 1, commModes, sModes, elem, myFirstLoc, nProcsPerRMode, nElemsPerProc, A, &(dataBuf[i * A.LocalModeStride(sMode)]), &(sendBuf[0]), sendBufOrig);
        }
    }
}

template <typename T>
void DistTensor<T>::ReduceScatterCommRedist(const DistTensor<T>& A, const ModeArray& reduceModes, const ModeArray& scatterModes){
//    if(!CheckReduceScatterCommRedist(A, reduceMode, scatterMode))
//      LogicError("ReduceScatterRedist: Invalid redistribution request");
    Unsigned i;
    const tmen::Grid& g = A.Grid();
    //NOTE: Hack for testing.  We actually need to let the user specify the commModes
    ModeArray commModes;
    for(i = 0; i < reduceModes.size(); i++){
        ModeDistribution modeDist = A.ModeDist(reduceModes[i]);
        commModes.insert(commModes.end(), modeDist.begin(), modeDist.end());
    }

    const mpi::Comm comm = GetCommunicatorForModes(commModes, g);

    if(!A.Participating())
        return;
    Unsigned sendSize, recvSize;

    //Determine buffer sizes for communication
    const Unsigned nRedistProcs = Max(1, prod(FilterVector(g.Shape(), commModes)));
    const ObjShape maxLocalShapeB = MaxLocalShape();
    recvSize = prod(maxLocalShapeB);
    sendSize = recvSize * nRedistProcs;

//    PrintVector(maxLocalShapeB, "sendShape");
    Memory<T> auxMemory;
    T* auxBuf = auxMemory.Require(sendSize + recvSize);
    MemZero(&(auxBuf[0]), sendSize + recvSize);
    T* sendBuf = &(auxBuf[0]);
    T* recvBuf = &(auxBuf[sendSize]);

    PackRSCommSendBuf(A, reduceModes, scatterModes, sendBuf);

    mpi::ReduceScatter(sendBuf, recvBuf, recvSize, comm);

    if(!(Participating()))
        return;
    UnpackRSCommRecvBuf(recvBuf, reduceModes, scatterModes, A);
}

template <typename T>
void DistTensor<T>::PackRSCommSendBuf(const DistTensor<T>& A, const ModeArray& rModes, const ModeArray& sModes, T * const sendBuf)
{
    Unsigned i, j;
    Unsigned order = A.Order();
    const T* dataBuf = A.LockedBuffer();

    const tmen::GridView gvA = A.GetGridView();
    const tmen::GridView gvB = GetGridView();

//    std::cout << "dataBuf:";
//    for(Unsigned i = 0; i < prod(A.LocalShape()); i++){
//        std::cout << " " << dataBuf[i];
//    }
//    std::cout << std::endl;

    std::vector<Unsigned> commLCMs = tmen::LCMs(gvA.ParticipatingShape(), gvB.ParticipatingShape());
    std::vector<Unsigned> modeStrideFactor = ElemwiseDivide(commLCMs, gvA.ParticipatingShape());

    const ObjShape gridShape = Grid().Shape();
    ModeArray uniqueSModes = sModes;
    std::sort(uniqueSModes.begin(), uniqueSModes.end());
    uniqueSModes.erase(std::unique(uniqueSModes.begin(), uniqueSModes.end()), uniqueSModes.end());

    std::vector<Unsigned> nProcsForSMode(uniqueSModes.size(), 1);
    for(i = 0; i < uniqueSModes.size(); i++){
        for(j = 0; j < sModes.size(); j++)
            if(sModes[j] == uniqueSModes[i])
                nProcsForSMode[i] *= Max(1, gvA.Dimension(rModes[j]));
    }

    ModeArray redistModes;
    for(i = 0; i < rModes.size(); i++){
        ModeDistribution modeDist = A.ModeDist(rModes[i]);
        redistModes.insert(redistModes.end(), modeDist.begin(), modeDist.end());
    }

    ModeArray commModes = redistModes;
    std::sort(commModes.begin(), commModes.end());
    const ObjShape redistShape = FilterVector(gridShape, redistModes);
    const ObjShape commShape = FilterVector(gridShape, commModes);
    const Permutation commPerm = DeterminePermutation(redistModes, commModes);

    const ObjShape sendShape = MaxLocalShape();

    const Location zeros(order, 0);
    const Location ones(order, 1);
    PackData packData;
    packData.loopShape = A.LocalShape();
    packData.srcBufStrides = A.LocalStrides();
    for(i = 0; i < uniqueSModes.size(); i++){
        packData.srcBufStrides[uniqueSModes[i]] *= modeStrideFactor[uniqueSModes[i]];
    }
    packData.dstBufStrides = Dimensions2Strides(sendShape);
    packData.loopStarts = zeros;
    packData.loopIncs = modeStrideFactor;
    const Unsigned nCommElemsPerProc = prod(sendShape);

    const std::vector<Unsigned> sModeStrides = LocalStrides();
    const Location myFirstElemLoc = A.ModeShifts();
    Location packElem = myFirstElemLoc;

    for(i = 0; i < rModes.size(); i++)
        packElem[rModes[i]] = 0;

//    PrintVector(packElem, "packElem");
//    PrintVector(uniqueSModes, "uniqueSModes");
//    PrintVector(nProcsForSMode, "procsForSMode");
//    PrintVector(A.Shape(), "shapeA");
//    PrintVector(A.LocalShape(), "localShapeA");

    if(ElemwiseLessThan(packElem, A.Shape())){
        PackTestHelper(packData, uniqueSModes.size() - 1, commModes, uniqueSModes, packElem, myFirstElemLoc, nProcsForSMode, nCommElemsPerProc, A, &(dataBuf[0]), &(sendBuf[0]), &(sendBuf[0]));
    }
//    std::cout << "packed sendBuf:";
//    for(Unsigned i = 0; i < nCommElemsPerProc * nRedistProcs; i++)
//        std::cout << " " << sendBuf[i];
//    std::cout << std::endl;
//    for(i = 0; i < nRedistProcs; i++){
//        const Location elemCommLoc = LinearLoc2Loc(i, redistShape);
//        const Unsigned elemRedistLinLoc = Loc2LinearLoc(FilterVector(elemCommLoc, commPerm), commShape);
//
//        packData.loopStarts[sMode] = i;
//        PackCommHelper(packData, order - 1, &(dataBuf[i*sModeStride]), &(sendBuf[elemRedistLinLoc * nCommElemsPerProc]));
////        std::cout << "pack slice:" << i << std::endl;
////        std::cout << "packed sendBuf:";
////        for(Unsigned i = 0; i < nCommElemsPerProc * nRedistProcs; i++)
////            std::cout << " " << sendBuf[i];
////        std::cout << std::endl;
//    }
}

template <typename T>
void DistTensor<T>::UnpackRSCommRecvBuf(const T * const recvBuf, const ModeArray& rModes, const ModeArray& sModes, const DistTensor<T>& A)
{
    const Unsigned order = A.Order();
    T* dataBuf = Buffer();

//    std::cout << "recvBuf:";
//    for(Unsigned i = 0; i < prod(MaxLocalShape()); i++){
//        std::cout << " " << recvBuf[i];
//    }
//    std::cout << std::endl;

    const Location zeros(order, 0);
    const Location ones(order, 1);
    PackData unpackData;
    unpackData.loopShape = LocalShape();
    unpackData.dstBufStrides = LocalStrides();
    unpackData.srcBufStrides = Dimensions2Strides(MaxLocalShape());
    unpackData.loopStarts = zeros;
    unpackData.loopIncs = ones;

    PackCommHelper(unpackData, order - 1, &(recvBuf[0]), &(dataBuf[0]));
}

#define PROTO(T) \
        template Int  DistTensor<T>::CheckReduceScatterCommRedist(const DistTensor<T>& A, const Mode reduceMode, const Mode scatterMode); \
        template void DistTensor<T>::ReduceScatterCommRedist(const DistTensor<T>& A, const ModeArray& reduceModes, const ModeArray& scatterModes); \
        template void DistTensor<T>::PackRSCommSendBuf(const DistTensor<T>& A, const ModeArray& rModes, const ModeArray& sModes, T * const sendBuf); \
        template void DistTensor<T>::UnpackRSCommRecvBuf(const T * const recvBuf, const ModeArray& rModes, const ModeArray& sModes, const DistTensor<T>& A);

PROTO(int)
PROTO(float)
PROTO(double)
PROTO(Complex<float>)
PROTO(Complex<double>)

} //namespace tmen
