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
void DistTensor<T>::GatherToOneCommRedist(const DistTensor<T>& A, const Mode gatherMode, const ModeArray& gridModes){
    if(!this->CheckGatherToOneCommRedist(A, gatherMode, gridModes))
      LogicError("GatherToOneRedist: Invalid redistribution request");

    //NOTE: Hack for testing.  We actually need to let the user specify the commModes
    //NOTE: THIS NEEDS TO BE BEFORE Participating() OTHERWISE PROCESSES GET OUT OF SYNC
    const mpi::Comm comm = this->GetCommunicatorForModes(gridModes, A.Grid());

    if(!A.Participating())
        return;
    Unsigned sendSize, recvSize;

    //Determine buffer sizes for communication
    const ObjShape gridViewSlice = FilterVector(A.GridViewShape(), A.ModeDist(gatherMode));
    const Unsigned nRedistProcs = Max(1, prod(FilterVector(A.Grid().Shape(), gridModes)));
    const ObjShape maxLocalShapeA = MaxLengths(A.Shape(), A.GetGridView().ParticipatingShape());
    sendSize = prod(maxLocalShapeA);
    recvSize = sendSize;

    Memory<T> auxMemory;
    T* auxBuf = auxMemory.Require(sendSize + nRedistProcs*recvSize);
    MemZero(&(auxBuf[0]), sendSize + recvSize);
    T* sendBuf = &(auxBuf[0]);
    T* recvBuf = &(auxBuf[sendSize]);

    PackGTOCommSendBuf(A, gatherMode, gridModes, sendBuf);

    mpi::Gather(sendBuf, sendSize, recvBuf, recvSize, 0, comm);

    if(!(this->Participating()))
        return;
    UnpackGTOCommRecvBuf(recvBuf, gatherMode, gridModes, A);
}

template<typename T>
void DistTensor<T>::PackGTOCommSendBufHelper(const GTOPackData& packData, const Mode packMode, T const * const dataBuf, T * const sendBuf){

    Unsigned packSlice = packMode;
    Unsigned packSliceMaxDim = packData.sendShape[packSlice];
    Unsigned packSliceLocalDim = packData.localShape[packSlice];
    Unsigned packSliceSendBufStride = packData.sendBufModeStrides[packSlice];
    Unsigned packSliceDataBufStride = packData.dataBufModeStrides[packSlice];
    Unsigned sendBufPtr = 0;
    Unsigned dataBufPtr = 0;

    if(packMode == 0){
        if(packSliceSendBufStride == 1 && packSliceDataBufStride == 1){
            MemCopy(&(sendBuf[0]), &(dataBuf[0]), packSliceLocalDim);
        }else{
            for(packSlice = 0; packSlice < packSliceMaxDim && packSlice < packSliceLocalDim; packSlice++){
                sendBuf[sendBufPtr] = dataBuf[dataBufPtr];
                sendBufPtr += packSliceSendBufStride;
                dataBufPtr += packSliceDataBufStride;
            }
        }
    }else{
        for(packSlice = 0; packSlice < packSliceMaxDim && packSlice < packSliceLocalDim; packSlice++){
            PackGTOCommSendBufHelper(packData, packMode-1, &(dataBuf[dataBufPtr]), &(sendBuf[sendBufPtr]));
            sendBufPtr += packSliceSendBufStride;
            dataBufPtr += packSliceDataBufStride;
        }
    }
}

template <typename T>
void DistTensor<T>::PackGTOCommSendBuf(const DistTensor<T>& A, const Mode gMode, const ModeArray& gridModes, T * const sendBuf)
{
    Unsigned order = A.Order();
    const T* dataBuf = A.LockedBuffer();

//    printf("dataBuf: ");
//    for(Unsigned i = 0; i < prod(A.LocalShape()); i++){
//        printf("%d ", dataBuf[i]);
//    }
//    printf("\n");

    const tmen::GridView gvA = A.GetGridView();
    const tmen::GridView gvB = GetGridView();

    GTOPackData packData;
    packData.sendShape = MaxLengths(A.Shape(), gvA.ParticipatingShape());
    packData.localShape = A.LocalShape();

    packData.dataBufModeStrides = A.LocalStrides();

    packData.sendBufModeStrides.resize(order);
    packData.sendBufModeStrides = Dimensions2Strides(packData.sendShape);

    PackGTOCommSendBufHelper(packData, order - 1, &(dataBuf[0]), &(sendBuf[0]));

//    printf("packed sendBuf: ");
//    for(Unsigned i = 0; i < prod(packData.sendShape); i++)
//        printf("%d ", sendBuf[i]);
//    printf("\n");
    //-------------------------------------------------
    //-------------------------------------------------
    //-------------------------------------------------

    //Shape of the local tensor we are packing
//    const ObjShape maxLocalShapeA = MaxLengths(A.Shape(), gvA.ParticipatingShape());
//    const ObjShape localShapeA = A.LocalShape();
//
//    //Calculate number of outer slices to pack
//    const Unsigned nMaxOuterSlices = Max(1, prod(maxLocalShapeA, gMode + 1));
//    const Unsigned nLocalOuterSlices = prod(localShapeA, gMode + 1);
//
//    //Calculate number of sMode slices to pack
//    const Unsigned nMaxGModeSlices = maxLocalShapeA[gMode];
//    const Unsigned nLocalGModeSlices = localShapeA[gMode];
//
//    const Unsigned maxCopySliceSize = Max(1, prod(maxLocalShapeA, 0, gMode));
//    const Unsigned copySliceSize = prod(localShapeA, 0, gMode);
//
//    Unsigned outerSliceNum, gModeSliceNum; //Which slice of which wrap of which process are we packing
//    Unsigned outerSendBufOff, gModeSendBufOff;
//    Unsigned outerDataBufOff, gModeDataBufOff;
//    Unsigned startSendBuf, startDataBuf;
//
//
////    printf("MemCopy info:\n");
////    printf("    nMaxOuterSlices: %d\n", nMaxOuterSlices);
////    printf("    gMaxSModeSlices: %d\n", nMaxGModeSlices);
////    printf("    maxCopySliceSize: %d\n", maxCopySliceSize);
////    printf("    copySliceSize: %d\n", copySliceSize);
//
//    for(outerSliceNum = 0; outerSliceNum < nMaxOuterSlices; outerSliceNum++ ){
//        if(outerSliceNum >= nLocalOuterSlices)
//            break;
//        outerSendBufOff = maxCopySliceSize * nMaxGModeSlices * outerSliceNum;
//        outerDataBufOff = copySliceSize * nLocalGModeSlices * outerSliceNum;
//
////        printf("        outerSliceNum: %d\n", outerSliceNum);
////        printf("        outerSendBufOff: %d\n", outerSendBufOff);
////        printf("        outerDataBufOff: %d\n", outerDataBufOff);
//
//        for(gModeSliceNum = 0; gModeSliceNum < nMaxGModeSlices; gModeSliceNum++){
//            if(gModeSliceNum >= nLocalGModeSlices)
//                break;
//            gModeSendBufOff = maxCopySliceSize * gModeSliceNum;
//            gModeDataBufOff = copySliceSize * gModeSliceNum;
//
////            printf("          gModeSliceNum: %d\n", gModeSliceNum);
////            printf("          gModeSendBufOff: %d\n", gModeSendBufOff);
////            printf("          gModeDataBufOff: %d\n", gModeDataBufOff);
//            startSendBuf = outerSendBufOff + gModeSendBufOff;
//            startDataBuf = outerDataBufOff + gModeDataBufOff;
//
////            printf("          startSendBuf: %d\n", startSendBuf);
////            printf("          startDataBuf: %d\n", startDataBuf);
//            MemCopy(&(sendBuf[startSendBuf]), &(dataBuf[startDataBuf]), copySliceSize);
//        }
//    }

//    printf("packed sendBuf: ");
//    for(Unsigned i = 0; i < prod(maxLocalShapeA); i++)
//        printf("%d ", sendBuf[i]);
//    printf("\n");
}

template<typename T>
void DistTensor<T>::UnpackGTOCommRecvBufHelper(const GTOUnpackData& unpackData, const Mode unpackMode, T const * const recvBuf, T * const dataBuf){
    Unsigned unpackSlice = unpackMode;
    Unsigned unpackSliceMaxDim = unpackData.recvShape[unpackSlice];
    Unsigned unpackSliceLocalDim = unpackData.localShape[unpackSlice];
    Unsigned unpackSliceRecvBufStride = unpackData.recvBufModeStrides[unpackSlice];
    Unsigned unpackSliceDataBufStride = unpackData.dataBufModeStrides[unpackSlice];
    Mode commMode = unpackData.commMode;
    Unsigned recvBufPtr = 0;
    Unsigned dataBufPtr = 0;
    Unsigned pRecvBufPtr = 0;
    Unsigned pDataBufPtr = 0;

//    Unsigned order = Order();
//    Unsigned i;
//    std::string ident = "";
//    for(i = 0; i < order - unpackMode; i++)
//        ident += "  ";
//    std::cout << ident << "Unpacking mode " << unpackMode << std::endl;

    if(unpackMode == commMode){
            Unsigned elemSlice = unpackData.elemSlice;
            Unsigned maxElemSlice = unpackData.maxElemSlices;

            //NOTE: Each tensor data we unpack is strided by nRedistProcs (maxElemSlice) entries away
            unpackSliceDataBufStride *= maxElemSlice;

            if(unpackMode == 0){
//                std::cout << ident << "recvBuf loc " << &(recvBuf[pRecvBufPtr]) << std::endl;
//                std::cout << ident << "dataBuf loc " << &(dataBuf[pDataBufPtr]) << std::endl;
//                std::cout << ident << "unpacking recv data:";
//                for(unpackSlice = 0; unpackSlice < unpackSliceMaxDim && (unpackSlice + elemSlice) < unpackSliceLocalDim; unpackSlice += maxElemSlice){
//                    std::cout << " " << recvBuf[pRecvBufPtr];
//
//                    std::cout << "recvBuf inc by " << unpackSliceRecvBufStride << std::endl;
//                    std::cout << "dataBuf inc by " << unpackSliceDataBufStride << std::endl;
//                    pRecvBufPtr += unpackSliceRecvBufStride;
//                    pDataBufPtr += unpackSliceDataBufStride;
//                }
//                std::cout << std::endl;

                for(unpackSlice = 0; unpackSlice < unpackSliceMaxDim && (unpackSlice + elemSlice) < unpackSliceLocalDim; unpackSlice += maxElemSlice){
                    dataBuf[dataBufPtr] = recvBuf[recvBufPtr];

//                    std::cout << "recvBuf inc by " << unpackSliceRecvBufStride << std::endl;
//                    std::cout << "dataBuf inc by " << unpackSliceDataBufStride << std::endl;
                    recvBufPtr += unpackSliceRecvBufStride;
                    dataBufPtr += unpackSliceDataBufStride;
                }
            }else{
                for(unpackSlice = 0; unpackSlice < unpackSliceMaxDim && (unpackSlice + elemSlice) < unpackSliceLocalDim; unpackSlice += maxElemSlice){
                    UnpackGTOCommRecvBufHelper(unpackData, unpackMode-1, &(recvBuf[recvBufPtr]), &(dataBuf[dataBufPtr]));

//                    std::cout << "recvBuf inc by " << unpackSliceRecvBufStride << std::endl;
//                    std::cout << "dataBuf inc by " << unpackSliceDataBufStride << std::endl;
                    recvBufPtr += unpackSliceRecvBufStride;
                    dataBufPtr += unpackSliceDataBufStride;
                }
            }
    }else if(unpackMode == 0){
//        std::cout << ident << "recvBuf loc " << &(recvBuf[pRecvBufPtr]) << std::endl;
//        std::cout << ident << "dataBuf loc " << &(dataBuf[pDataBufPtr]) << std::endl;
//        std::cout << ident << "unpacking recv data:";
//        for(unpackSlice = 0; unpackSlice < unpackSliceMaxDim && unpackSlice < unpackSliceLocalDim; unpackSlice++){
//            std::cout << " " << recvBuf[pRecvBufPtr];
//            pRecvBufPtr += unpackSliceRecvBufStride;
//            pDataBufPtr += unpackSliceDataBufStride;
//        }
//        std::cout << std::endl;

        if(unpackSliceRecvBufStride == 1 && unpackSliceDataBufStride == 1){
            MemCopy(&(dataBuf[0]), &(recvBuf[0]), unpackSliceLocalDim);
        }else{
            for(unpackSlice = 0; unpackSlice < unpackSliceMaxDim && unpackSlice < unpackSliceLocalDim; unpackSlice++){
                dataBuf[dataBufPtr] = recvBuf[recvBufPtr];

//                std::cout << "recvBuf inc by " << unpackSliceRecvBufStride << std::endl;
//                std::cout << "dataBuf inc by " << unpackSliceDataBufStride << std::endl;
                recvBufPtr += unpackSliceRecvBufStride;
                dataBufPtr += unpackSliceDataBufStride;
            }
        }
    }else {
        for(unpackSlice = 0; unpackSlice < unpackSliceMaxDim && unpackSlice < unpackSliceLocalDim; unpackSlice++){
            UnpackGTOCommRecvBufHelper(unpackData, unpackMode-1, &(recvBuf[recvBufPtr]), &(dataBuf[dataBufPtr]));
//            std::cout << "recvBuf inc by " << unpackSliceRecvBufStride << std::endl;
//            std::cout << "dataBuf inc by " << unpackSliceDataBufStride << std::endl;
            recvBufPtr += unpackSliceRecvBufStride;
            dataBufPtr += unpackSliceDataBufStride;
        }
    }
}

template <typename T>
void DistTensor<T>::UnpackGTOCommRecvBuf(const T * const recvBuf, const Mode gMode, const ModeArray& gridModes, const DistTensor<T>& A)
{
    Unsigned i;
    Unsigned order = Order();
    T* dataBuf = this->Buffer();

    const tmen::Grid& g = A.Grid();
    const tmen::GridView gvA = A.GetGridView();
    const tmen::GridView gvB = GetGridView();

    const Unsigned nRedistProcs = Max(1, prod(FilterVector(g.Shape(), gridModes)));

    const ObjShape maxLocalShapeA = MaxLengths(A.Shape(), gvA.ParticipatingShape());

    ModeArray commModes = gridModes;
    std::sort(commModes.begin(), commModes.end());
    const ObjShape redistShape = FilterVector(Grid().Shape(), gridModes);
    const ObjShape commShape = FilterVector(Grid().Shape(), commModes);
    const Permutation redistPerm = DeterminePermutation(commModes, gridModes);

    const Unsigned nCommElemsPerProc = prod(maxLocalShapeA);
    const Unsigned gModeStride = LocalModeStride(gMode);
    //    printf("recvBuf:");
    //    for(Unsigned i = 0; i < nCommElemsPerProc * nRedistProcs; i++){
    //        printf(" %d", recvBuf[i]);
    //    }
    //    printf("\n");

    GTOUnpackData unpackData;

    unpackData.recvShape = MaxLengths(this->Shape(), gvB.ParticipatingShape());
    unpackData.localShape = this->LocalShape();

    unpackData.recvBufModeStrides = Dimensions2Strides(maxLocalShapeA);
    unpackData.dataBufModeStrides = LocalStrides();
    unpackData.commMode = gMode;
    unpackData.maxElemSlices = nRedistProcs;

    //NOTE: Check
    for(i = 0; i < nRedistProcs; i++){
        const Location elemCommLoc = LinearLoc2Loc(i, commShape);
        const Unsigned elemRedistLinLoc = Loc2LinearLoc(FilterVector(elemCommLoc, redistPerm), redistShape);
        if(elemRedistLinLoc >= LocalDimension(gMode))
            continue;
        unpackData.elemSlice = elemRedistLinLoc;

//        printf("elemSlice: %d\n", i);
//        printf("elemRedistLinLoc: %d\n", elemRedistLinLoc);
//        printf("dataBufPtr: %d\n", elemRedistLinLoc * gModeStride);
//        printf("recvBufPtr: %d\n", i * nCommElemsPerProc);
//        printf("nCommElemsPerProc: %d\n", nCommElemsPerProc);

        UnpackGTOCommRecvBufHelper(unpackData, order - 1, &(recvBuf[i * nCommElemsPerProc]), &(dataBuf[elemRedistLinLoc * gModeStride]));
//        printf("dataBuf:");
//        for(Unsigned i = 0; i < prod(this->LocalShape()); i++)
//            printf(" %d", dataBuf[i]);
//        printf("\n");
    }

//    printf("dataBuf:");
//    for(Unsigned i = 0; i < prod(this->LocalShape()); i++)
//        printf(" %d", dataBuf[i]);
//    printf("\n");

    //-----------------------------------------
    //-----------------------------------------
    //-----------------------------------------

    //Only unpack if we are the root (everyone else gets nothing)
    //if(gvB.ModeLoc(gMode) == 0){
//        const ObjShape maxLocalShapeA = MaxLengths(A.Shape(), gvA.ParticipatingShape());
//        const ObjShape maxLocalShapeB = MaxLengths(this->Shape(), gvB.ParticipatingShape());
//
////        const Unsigned maxRecvElem = prod(maxLocalShapeA) * nRedistProcs;
////        printf("maxRecvElem: %d\n", maxRecvElem);
////        printf("recvBuf:");
////        for(Unsigned i = 0; i < maxRecvElem; i++){
////            printf(" %d", recvBuf[i]);
////        }
////        printf("\n");
//
//        const ObjShape localShapeB = this->LocalShape();         //Shape of the local tensor we are packing
//
//        //Number of outer slices to unpack
//        const Unsigned nMaxOuterSlices = Max(1, prod(maxLocalShapeB, gMode + 1));
//        const Unsigned nLocalOuterSlices = prod(localShapeB, gMode + 1);
//
//        //Loop packing bounds variables
//        const Unsigned nMaxGModeSlices = maxLocalShapeB[gMode];
//        const Unsigned nLocalGModeSlices = localShapeB[gMode];
//        const Unsigned gModeUnpackStride = nRedistProcs;
//
//        //Each wrap is copied contiguously because the distribution of reduce-to-one mode does not change
//
//        //Variables for calculating elements to copy
//        const Unsigned maxCopySliceSize = Max(1, prod(maxLocalShapeB, 0, gMode));
//        const Unsigned copySliceSize = prod(localShapeB, 0, gMode);
//
//        //Number of processes we have to unpack from
//        const Unsigned nElemSlices = nRedistProcs;
//
//        //Loop iteration vars
//        Unsigned elemSliceNum, outerSliceNum, gModeSliceNum;  //Pack data for slice "sliceNum" (<nSlices) of wrap "wrapNum" (<nWraps) for proc "procSendNum" int offSliceRecvBuf, offWrapRecvBuf;  //Offsets used to index into sendBuf array
//        Unsigned elemRecvBufOff, elemDataBufOff;
//        Unsigned outerRecvBufOff, outerDataBufOff;  //Offsets used to index into recvBuf array
//        Unsigned gModeRecvBufOff, gModeDataBufOff;  //Offsets used to index into dataBuf array
//        Unsigned startRecvBuf, startDataBuf;
//
//        ModeArray commModes = gridModes;
//        std::sort(commModes.begin(), commModes.end());
//        const ObjShape redistShape = FilterVector(Grid().Shape(), gridModes);
//        const ObjShape commShape = FilterVector(Grid().Shape(), commModes);
//        const Permutation redistPerm = DeterminePermutation(commModes, gridModes);
//
////        printf("MemCopy info:\n");
////        printf("    nMaxOuterSlices: %d\n", nMaxOuterSlices);
////        printf("    nMaxGModeSlices: %d\n", nMaxGModeSlices);
////        printf("    maxCopySliceSize: %d\n", maxCopySliceSize);
////        printf("    copySliceSize: %d\n", copySliceSize);
//        for(elemSliceNum = 0; elemSliceNum < nElemSlices; elemSliceNum++){
//            const Location elemCommLoc = LinearLoc2Loc(elemSliceNum, commShape);
//            const Unsigned elemRedistLinLoc = Loc2LinearLoc(FilterVector(elemCommLoc, redistPerm), redistShape);
//
//            elemRecvBufOff = prod(maxLocalShapeA) * elemSliceNum;
//            elemDataBufOff = copySliceSize * elemRedistLinLoc;
//
//            for(outerSliceNum = 0; outerSliceNum < nMaxOuterSlices; outerSliceNum++){
//                if(outerSliceNum >= nLocalOuterSlices)
//                    break;
//                outerRecvBufOff = maxCopySliceSize * Max(1, (nMaxGModeSlices - 1) / gModeUnpackStride + 1) * outerSliceNum;
//                outerDataBufOff = copySliceSize * nLocalGModeSlices * outerSliceNum;
//
////                printf("        outerSliceNum: %d\n", outerSliceNum);
////                printf("        outerRecvBufOff: %d\n", outerRecvBufOff);
////                printf("        outerDataBufOff: %d\n", outerDataBufOff);
//
//                for(gModeSliceNum = 0; gModeSliceNum < nMaxGModeSlices; gModeSliceNum+= gModeUnpackStride){
//                    if(gModeSliceNum + elemSliceNum >= nLocalGModeSlices)
//                        break;
//
//                    gModeRecvBufOff = maxCopySliceSize * (gModeSliceNum / gModeUnpackStride);
//                    gModeDataBufOff = (copySliceSize * gModeSliceNum);
//
//                    startRecvBuf = elemRecvBufOff + outerRecvBufOff + gModeRecvBufOff;
//                    startDataBuf = elemDataBufOff + outerDataBufOff + gModeDataBufOff;
//
////                    printf("          startRecvBuf: %d\n", startRecvBuf);
////                    printf("          startDataBuf: %d\n", startDataBuf);
//                    MemCopy(&(dataBuf[startDataBuf]), &(recvBuf[startRecvBuf]), copySliceSize);
//                }
//            }
//        }

//        printf("dataBuf:");
//        for(Unsigned i = 0; i < prod(this->LocalShape()); i++)
//            printf(" %d", dataBuf[i]);
//        printf("\n");
   // }else{
     //   MemZero(&(dataBuf[0]), prod(this->LocalShape()));
    //}
}

#define PROTO(T) \
        template Int  DistTensor<T>::CheckGatherToOneCommRedist(const DistTensor<T>& A, const Mode gMode, const ModeArray& gridModes); \
        template void DistTensor<T>::GatherToOneCommRedist(const DistTensor<T>& A, const Mode gatherMode, const ModeArray& gridModes); \
        template void DistTensor<T>::PackGTOCommSendBuf(const DistTensor<T>& A, const Mode gatherMode, const ModeArray& gridModes, T * const sendBuf); \
        template void DistTensor<T>::UnpackGTOCommRecvBuf(const T * const recvBuf, const Mode gatherMode, const ModeArray& gridModes, const DistTensor<T>& A);

PROTO(int)
PROTO(float)
PROTO(double)
PROTO(Complex<float>)
PROTO(Complex<double>)

} //namespace tmen
