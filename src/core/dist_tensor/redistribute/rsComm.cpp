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
void DistTensor<T>::ReduceScatterCommRedist(const DistTensor<T>& A, const Mode reduceMode, const Mode scatterMode){
    if(!this->CheckReduceScatterCommRedist(A, reduceMode, scatterMode))
      LogicError("ReduceScatterRedist: Invalid redistribution request");

    //NOTE: Hack for testing.  We actually need to let the user specify the commModes
    const ModeArray commModes = A.ModeDist(reduceMode);
    const mpi::Comm comm = this->GetCommunicatorForModes(commModes, A.Grid());

    if(!A.Participating())
        return;
    Unsigned sendSize, recvSize;

    //Determine buffer sizes for communication
    const ObjShape gridViewSlice = FilterVector(A.GridViewShape(), A.ModeDist(reduceMode));
    const Unsigned nRedistProcs = Max(1, prod(FilterVector(A.Grid().Shape(), A.ModeDist(reduceMode))));
    const ObjShape maxLocalShapeB = MaxLengths(Shape(), GetGridView().ParticipatingShape());
    recvSize = prod(maxLocalShapeB);
    sendSize = recvSize * nRedistProcs;



    Memory<T> auxMemory;
    T* auxBuf = auxMemory.Require(sendSize + recvSize);
    MemZero(&(auxBuf[0]), sendSize + recvSize);
    T* sendBuf = &(auxBuf[0]);
    T* recvBuf = &(auxBuf[sendSize]);

    PackRSCommSendBuf(A, reduceMode, scatterMode, sendBuf);

    mpi::ReduceScatter(sendBuf, recvBuf, recvSize, comm);

    if(!(this->Participating()))
        return;
    UnpackRSCommRecvBuf(recvBuf, reduceMode, scatterMode, A);
}

template<typename T>
void DistTensor<T>::PackRSCommSendBufHelper(const RSPackData& packData, const Mode packMode, T const * const dataBuf, T * const sendBuf){
    Unsigned packSlice = packMode;
    Unsigned packSliceMaxDim = packData.sendShape[packSlice];
    Unsigned packSliceLocalDim = packData.localShape[packSlice];
    Unsigned packSliceSendBufStride = packData.sendBufModeStrides[packSlice];
    Unsigned packSliceDataBufStride = packData.dataBufModeStrides[packSlice];
    Mode sMode = packData.sMode;
    Unsigned sendBufPtr = 0;
    Unsigned dataBufPtr = 0;

    //    Unsigned order = Order();
    //    Unsigned i;
    //    std::string ident = "";
    //    for(i = 0; i < order - unpackMode; i++)
    //        ident += "  ";
    //    std::cout << ident << "Unpacking mode " << unpackMode << std::endl;

    if(packMode == sMode){
            Unsigned elemSlice = packData.elemSlice;
            Unsigned maxElemSlice = packData.maxElemSlices;

            //NOTE: Each tensor data we unpack is strided by nRedistProcs (maxElemSlice) entries away
            packSliceDataBufStride *= maxElemSlice;

            if(packMode == 0){
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

                for(packSlice = 0; (packSlice + elemSlice) < packSliceLocalDim; packSlice += maxElemSlice){
                    sendBuf[sendBufPtr] = dataBuf[dataBufPtr];

//                    std::cout << "recvBuf inc by " << unpackSliceRecvBufStride << std::endl;
//                    std::cout << "dataBuf inc by " << unpackSliceDataBufStride << std::endl;
                    sendBufPtr += packSliceSendBufStride;
                    dataBufPtr += packSliceDataBufStride;
                }
            }else{
                for(packSlice = 0; (packSlice + elemSlice) < packSliceLocalDim; packSlice += maxElemSlice){
                    PackRSCommSendBufHelper(packData, packMode-1, &(dataBuf[dataBufPtr]), &(sendBuf[sendBufPtr]));

//                    std::cout << "recvBuf inc by " << unpackSliceRecvBufStride << std::endl;
//                    std::cout << "dataBuf inc by " << unpackSliceDataBufStride << std::endl;
                    sendBufPtr += packSliceSendBufStride;
                    dataBufPtr += packSliceDataBufStride;
                }
            }
    }else if(packMode == 0){
//        std::cout << ident << "recvBuf loc " << &(recvBuf[pRecvBufPtr]) << std::endl;
//        std::cout << ident << "dataBuf loc " << &(dataBuf[pDataBufPtr]) << std::endl;
//        std::cout << ident << "unpacking recv data:";
//        for(unpackSlice = 0; unpackSlice < unpackSliceMaxDim && unpackSlice < unpackSliceLocalDim; unpackSlice++){
//            std::cout << " " << recvBuf[pRecvBufPtr];
//            pRecvBufPtr += unpackSliceRecvBufStride;
//            pDataBufPtr += unpackSliceDataBufStride;
//        }
//        std::cout << std::endl;

        if(packSliceSendBufStride == 1 && packSliceDataBufStride == 1){
            MemCopy(&(sendBuf[0]), &(dataBuf[0]), packSliceLocalDim);
        }else{
            for(packSlice = 0; packSlice < packSliceMaxDim && packSlice < packSliceLocalDim; packSlice++){
                sendBuf[sendBufPtr] = dataBuf[dataBufPtr];

//                std::cout << "recvBuf inc by " << unpackSliceRecvBufStride << std::endl;
//                std::cout << "dataBuf inc by " << unpackSliceDataBufStride << std::endl;
                sendBufPtr += packSliceSendBufStride;
                dataBufPtr += packSliceDataBufStride;
            }
        }
    }else {
        for(packSlice = 0; packSlice < packSliceMaxDim && packSlice < packSliceLocalDim; packSlice++){
            PackRSCommSendBufHelper(packData, packMode-1, &(dataBuf[dataBufPtr]), &(sendBuf[sendBufPtr]));
//            std::cout << "recvBuf inc by " << unpackSliceRecvBufStride << std::endl;
//            std::cout << "dataBuf inc by " << unpackSliceDataBufStride << std::endl;
            sendBufPtr += packSliceSendBufStride;
            dataBufPtr += packSliceDataBufStride;
        }
    }
}

template <typename T>
void DistTensor<T>::PackRSCommSendBuf(const DistTensor<T>& A, const Mode rMode, const Mode sMode, T * const sendBuf)
{
    Unsigned i;
    Unsigned order = A.Order();
    const T* dataBuf = A.LockedBuffer();

//    std::cout << "dataBuf:";
//    for(Unsigned i = 0; i < prod(A.LocalShape()); i++){
//        std::cout << " " << dataBuf[i];
//    }
//    std::cout << std::endl;

    const tmen::GridView gvA = A.GetGridView();
    const tmen::GridView gvB = GetGridView();
    const Unsigned nRedistProcs = Max(1, gvA.Dimension(rMode));

    const ModeArray redistModes = A.ModeDist(rMode);
    ModeArray commModes = redistModes;
    std::sort(commModes.begin(), commModes.end());
    const ObjShape redistShape = FilterVector(Grid().Shape(), redistModes);
    const ObjShape commShape = FilterVector(Grid().Shape(), commModes);
    const Permutation commPerm = DeterminePermutation(redistModes, commModes);

    RSPackData packData;
    packData.sendShape = MaxLengths(Shape(), gvB.ParticipatingShape());
    packData.localShape = A.LocalShape();

    packData.dataBufModeStrides = A.LocalStrides();
    packData.sMode = sMode;
    packData.sendBufModeStrides.resize(order);
    packData.sendBufModeStrides = Dimensions2Strides(packData.sendShape);
    packData.maxElemSlices = nRedistProcs;

    const Unsigned nCommElemsPerProc = prod(packData.sendShape);
    const Unsigned sModeStride = LocalModeStride(sMode);

    for(i = 0; i < nRedistProcs; i++){
        const Location elemCommLoc = LinearLoc2Loc(i, redistShape);
        const Unsigned elemRedistLinLoc = Loc2LinearLoc(FilterVector(elemCommLoc, commPerm), commShape);
        packData.elemSlice = i;

        PackRSCommSendBufHelper(packData, order - 1, &(dataBuf[i*sModeStride]), &(sendBuf[elemRedistLinLoc * nCommElemsPerProc]));
//        std::cout << "pack slice:" << i << std::endl;
//        std::cout << "packed sendBuf:";
//        for(Unsigned i = 0; i < nCommElemsPerProc * nRedistProcs; i++)
//            std::cout << " " << sendBuf[i];
//        std::cout << std::endl;
    }

//    std::cout << "packed sendBuf:";
//    for(Unsigned i = 0; i < nCommElemsPerProc * nRedistProcs; i++)
//        std::cout << " " << sendBuf[i];
//    std::cout << std::endl;

    //-------------------------------------------
    //-------------------------------------------
    //-------------------------------------------
//    const Unsigned nRedistProcs = Max(1, gvA.Dimension(rMode));
//
//    //Shape of the local tensor we are packing
//    const ObjShape maxLocalShapeA = MaxLengths(A.Shape(), gvA.ParticipatingShape());
//    const ObjShape localShapeA = A.LocalShape();
//
//    //Calculate number of outer slices to pack
//    const Unsigned nMaxOuterSlices = Max(1, prod(maxLocalShapeA, sMode + 1));
//    const Unsigned nLocalOuterSlices = prod(localShapeA, sMode + 1);
//
//    //Calculate number of sMode slices to pack
//    const Unsigned nMaxSModeSlices = maxLocalShapeA[sMode];
//    const Unsigned nLocalSModeSlices = localShapeA[sMode];
//    const Unsigned sModePackStride = nRedistProcs;
//    const Unsigned nMaxPackSModeSlices = MaxLength(maxLocalShapeA[sMode], sModePackStride);
//
//    //Number of processes we have to pack for
//    const Unsigned nElemSlices = nRedistProcs;
//
//    const Unsigned maxCopySliceSize = Max(1, prod(maxLocalShapeA, 0, sMode));
//    const Unsigned copySliceSize = prod(localShapeA, 0, sMode);
//
//
//    Unsigned outerSliceNum, sModeSliceNum, elemSliceNum; //Which slice of which wrap of which process are we packing
//    Unsigned elemSendBufOff, elemDataBufOff;
//    Unsigned outerSendBufOff, sModeSendBufOff;
//    Unsigned outerDataBufOff, sModeDataBufOff;
//    Unsigned startSendBuf, startDataBuf;
//
//    const ModeArray redistModes = A.ModeDist(rMode);
//    ModeArray commModes = redistModes;
//    std::sort(commModes.begin(), commModes.end());
//    const ObjShape redistShape = FilterVector(Grid().Shape(), redistModes);
//    const ObjShape commShape = FilterVector(Grid().Shape(), commModes);
//    const Permutation commPerm = DeterminePermutation(redistModes, commModes);
//
////    printf("MemCopy info:\n");
////    printf("    prod(maxLocalShapeA): %d\n", prod(maxLocalShapeA));
////    printf("    nMaxOuterSlices: %d\n", nMaxOuterSlices);
////    printf("    nLocalOuterSlices: %d\n", nLocalOuterSlices);
////    printf("    nMaxSModeSlices: %d\n", nMaxSModeSlices);
////    printf("    nLocalSModeSlices: %d\n", nLocalSModeSlices);
////    printf("    sModePackStride: %d\n", sModePackStride);
////    printf("    maxCopySliceSize: %d\n", maxCopySliceSize);
////    printf("    copySliceSize: %d\n", copySliceSize);
//    for(elemSliceNum = 0; elemSliceNum < nElemSlices; elemSliceNum++){
//        const Location elemRedistLoc = LinearLoc2Loc(elemSliceNum, redistShape);
//        const Unsigned elemCommLinLoc = Loc2LinearLoc(FilterVector(elemRedistLoc, commPerm), commShape);
//
//
////        PrintVector(redistShape, "redistShape");
////        PrintVector(redistModes, "redistModes");
////        PrintVector(elemRedistLoc, "elemRedistLoc");
////        PrintVector(commPerm, "commPerm");
////        PrintVector(FilterVector(elemRedistLoc, commPerm), "redistLoc");
////        std::cout << "elemCommLinLoc" << elemCommLinLoc << std::endl;
//
//        elemSendBufOff = prod(maxLocalShapeA) * elemCommLinLoc;
//        elemDataBufOff = copySliceSize * elemSliceNum;
//
////        printf("      elemSliceNum:   %d\n", elemSliceNum);
////        printf("      elemSendBufOff: %d\n", elemSendBufOff);
////        printf("      elemDataBufOff: %d\n", elemDataBufOff);
//
//        for(outerSliceNum = 0; outerSliceNum < nMaxOuterSlices; outerSliceNum++ ){
//            if(outerSliceNum >= nLocalOuterSlices)
//                break;
//            outerSendBufOff = maxCopySliceSize * Max(1, nMaxPackSModeSlices) * outerSliceNum;
//            outerDataBufOff = copySliceSize * nLocalSModeSlices * outerSliceNum;
//
////            printf("        outerSliceNum:   %d\n", outerSliceNum);
////            printf("        outerSendBufOff: %d\n", outerSendBufOff);
////            printf("        outerDataBufOff: %d\n", outerDataBufOff);
//
//                for(sModeSliceNum = 0; sModeSliceNum < nMaxSModeSlices; sModeSliceNum += sModePackStride){
//                    if(sModeSliceNum + elemSliceNum >= nLocalSModeSlices)
//                        break;
//                    sModeSendBufOff = maxCopySliceSize * (sModeSliceNum / sModePackStride);
//                    sModeDataBufOff = copySliceSize * sModeSliceNum;
//
////                    printf("          sModeSliceNum:   %d\n", sModeSliceNum);
////                    printf("          sModeSendBufOff: %d\n", sModeSendBufOff);
////                    printf("          sModeDataBufOff: %d\n", sModeDataBufOff);
//                    startSendBuf = elemSendBufOff + outerSendBufOff + sModeSendBufOff;
//                    startDataBuf = elemDataBufOff + outerDataBufOff + sModeDataBufOff;
//
////                    printf("            startSendBuf: %d\n", startSendBuf);
////                    printf("            startDataBuf: %d\n", startDataBuf);
//                    MemCopy(&(sendBuf[startSendBuf]), &(dataBuf[startDataBuf]), copySliceSize);
//                }
//        }
//    }

//    std::cout << "packed sendBuf:";
//    for(Unsigned i = 0; i < prod(maxLocalShapeA) * nRedistProcs; i++)
//        std::cout << " " << sendBuf[i];
//    std::cout << std::endl;
}

template <typename T>
void DistTensor<T>::UnpackRSCommRecvBufHelper(const RSUnpackData& unpackData, const Mode unpackMode, T const * const recvBuf, T * const dataBuf){
    Unsigned unpackSlice = unpackMode;
    Unsigned unpackSliceMaxDim = unpackData.recvShape[unpackSlice];
    Unsigned unpackSliceLocalDim = unpackData.localShape[unpackSlice];
    Unsigned unpackSliceRecvBufStride = unpackData.recvBufModeStrides[unpackSlice];
    Unsigned unpackSliceDataBufStride = unpackData.dataBufModeStrides[unpackSlice];
    Unsigned recvBufPtr = 0;
    Unsigned dataBufPtr = 0;

    if(unpackMode == 0){
        if(unpackSliceRecvBufStride == 1 && unpackSliceDataBufStride == 1){
            MemCopy(&(dataBuf[0]), &(recvBuf[0]), unpackSliceLocalDim);
        }else{
            for(unpackSlice = 0; unpackSlice < unpackSliceMaxDim && unpackSlice < unpackSliceLocalDim; unpackSlice++){
                dataBuf[dataBufPtr] = recvBuf[recvBufPtr];
                recvBufPtr += unpackSliceRecvBufStride;
                dataBufPtr += unpackSliceDataBufStride;
            }
        }
    }else{
        for(unpackSlice = 0; unpackSlice < unpackSliceMaxDim && unpackSlice < unpackSliceLocalDim; unpackSlice++){
            UnpackRSCommRecvBufHelper(unpackData, unpackMode-1, &(recvBuf[recvBufPtr]), &(dataBuf[dataBufPtr]));
            recvBufPtr += unpackSliceRecvBufStride;
            dataBufPtr += unpackSliceDataBufStride;
        }
    }
}

template <typename T>
void DistTensor<T>::UnpackRSCommRecvBuf(const T * const recvBuf, const Mode rMode, const Mode sMode, const DistTensor<T>& A)
{
    Unsigned i;
    const Unsigned order = A.Order();
    T* dataBuf = this->Buffer();

    const tmen::GridView gvA = A.GetGridView();
    const tmen::GridView gvB = GetGridView();

    const ObjShape maxLocalShapeB = MaxLengths(this->Shape(), gvB.ParticipatingShape());

//    std::cout << "recvBuf:";
//    for(Unsigned i = 0; i < prod(maxLocalShapeB); i++){
//        std::cout << " " << recvBuf[i];
//    }
//    std::cout << std::endl;

    RSUnpackData unpackData;
    unpackData.recvShape = MaxLengths(Shape(), gvB.ParticipatingShape());
    unpackData.localShape = LocalShape();

    unpackData.dataBufModeStrides = LocalStrides();

    unpackData.recvBufModeStrides.resize(order);
    unpackData.recvBufModeStrides = Dimensions2Strides(unpackData.recvShape);

    UnpackRSCommRecvBufHelper(unpackData, order - 1, &(recvBuf[0]), &(dataBuf[0]));

    //-------------------------------------------
    //-------------------------------------------
    //-------------------------------------------

//    const ObjShape localShapeB = this->LocalShape();         //Shape of the local tensor we are packing
//
//    //Number of outer slices to unpack
//    const Unsigned nMaxOuterSlices = Max(1, prod(maxLocalShapeB, sMode + 1));
//    const Unsigned nLocalOuterSlices = prod(localShapeB, sMode + 1);
//
//    //Loop packing bounds variables
//    const Unsigned nMaxSModeSlices = maxLocalShapeB[sMode];
//    const Unsigned nLocalSModeSlices = localShapeB[sMode];
//
//    //Each wrap is copied contiguously because the distribution of reduceScatter index does not change
//
//    //Variables for calculating elements to copy
//    const Unsigned maxCopySliceSize = Max(1, prod(maxLocalShapeB, 0, sMode));
//    const Unsigned copySliceSize = prod(localShapeB, 0, sMode);
//
//    //Loop iteration vars
//    Unsigned outerSliceNum, sModeSliceNum;  //Pack data for slice "sliceNum" (<nSlices) of wrap "wrapNum" (<nWraps) for proc "procSendNum" int offSliceRecvBuf, offWrapRecvBuf;  //Offsets used to index into sendBuf array
//    Unsigned outerRecvBufOff, outerDataBufOff;  //Offsets used to index into recvBuf array
//    Unsigned sModeRecvBufOff, sModeDataBufOff;  //Offsets used to index into dataBuf array
//    Unsigned startRecvBuf, startDataBuf;
//
////    printf("MemCopy info:\n");
////    printf("    nMaxOuterSlices: %d\n", nMaxOuterSlices);
////    printf("    nMaxSModeSlices: %d\n", nMaxSModeSlices);
////    printf("    maxCopySliceSize: %d\n", maxCopySliceSize);
////    printf("    copySliceSize: %d\n", copySliceSize);
//    for(outerSliceNum = 0; outerSliceNum < nMaxOuterSlices; outerSliceNum++){
//        if(outerSliceNum >= nLocalOuterSlices)
//            break;
//        outerRecvBufOff = maxCopySliceSize * nMaxSModeSlices * outerSliceNum;
//        outerDataBufOff = copySliceSize * nLocalSModeSlices * outerSliceNum;
//
////        printf("      outerSliceNum: %d\n", outerSliceNum);
////        printf("      outerRecvBufOff: %d\n", outerRecvBufOff);
////        printf("      outerDataBufOff: %d\n", outerDataBufOff);
//
//        for(sModeSliceNum = 0; sModeSliceNum < nMaxSModeSlices; sModeSliceNum++){
//            if(sModeSliceNum >= nLocalSModeSlices)
//                break;
//
//            sModeRecvBufOff = (maxCopySliceSize * sModeSliceNum);
//            sModeDataBufOff = (copySliceSize * sModeSliceNum);
//
//            startRecvBuf = outerRecvBufOff + sModeRecvBufOff;
//            startDataBuf = outerDataBufOff + sModeDataBufOff;
//
////            printf("        startRecvBuf: %d\n", startRecvBuf);
////            printf("        startDataBuf: %d\n", startDataBuf);
//            MemCopy(&(dataBuf[startDataBuf]), &(recvBuf[startRecvBuf]), copySliceSize);
//        }
//    }

//    std::cout << "dataBuf:";
//    for(Unsigned i = 0; i < prod(this->LocalShape()); i++)
//        std::cout << " " << dataBuf[i];
//    std::cout << std::endl;
}

#define PROTO(T) \
        template Int  DistTensor<T>::CheckReduceScatterCommRedist(const DistTensor<T>& A, const Mode reduceMode, const Mode scatterMode); \
        template void DistTensor<T>::ReduceScatterCommRedist(const DistTensor<T>& A, const Mode reduceMode, const Mode scatterMode); \
        template void DistTensor<T>::PackRSCommSendBuf(const DistTensor<T>& A, const Mode reduceMode, const Mode scatterMode, T * const sendBuf); \
        template void DistTensor<T>::UnpackRSCommRecvBuf(const T * const recvBuf, const Mode reduceMode, const Mode scatterMode, const DistTensor<T>& A);

PROTO(int)
PROTO(float)
PROTO(double)
PROTO(Complex<float>)
PROTO(Complex<double>)

} //namespace tmen
