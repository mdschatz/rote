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
    if(A.Order() != this->Order())
        LogicError("CheckAllToAllDoubleModeRedist: Objects being redistributed must be of same order");
    Unsigned i;
    for(i = 0; i < A.Order(); i++){
        if(i != allToAllModes.first && i != allToAllModes.second){
            if(this->ModeDist(i) != A.ModeDist(i))
                LogicError("CheckAllToAllDoubleModeRedist: Non-redist modes must have same distribution");
        }
    }
    return 1;
}

template <typename T>
void DistTensor<T>::AllToAllDoubleModeCommRedist(const DistTensor<T>& A, const std::pair<Mode, Mode>& a2aModes, const std::pair<ModeArray, ModeArray >& a2aCommGroups){
    if(!this->CheckAllToAllDoubleModeCommRedist(A, a2aModes, a2aCommGroups))
        LogicError("AllToAllDoubleModeRedist: Invalid redistribution request");

    //Determine buffer sizes for communication
    //NOTE: Swap to concatenate vectors
    ModeArray commModes = a2aCommGroups.first;
    commModes.insert(commModes.end(), a2aCommGroups.second.begin(), a2aCommGroups.second.end());
    std::sort(commModes.begin(), commModes.end());

    const mpi::Comm comm = this->GetCommunicatorForModes(commModes, A.Grid());

    if(!A.Participating())
        return;
    Unsigned sendSize, recvSize;



    const Unsigned nRedistProcs = prod(FilterVector(A.Grid().Shape(), commModes));
    const ObjShape maxLocalShape = MaxLengths(A.Shape(), A.GetGridView().ParticipatingShape());

    sendSize = prod(maxLocalShape) * Max(1, nRedistProcs);
    recvSize = sendSize;



    Memory<T> auxMemory;
    T* auxBuf = auxMemory.Require(sendSize + recvSize);
    MemZero(&(auxBuf[0]), sendSize + recvSize);

    T* sendBuf = &(auxBuf[0]);
    T* recvBuf = &(auxBuf[sendSize]);

    PackA2ADoubleModeCommSendBuf(A, a2aModes, a2aCommGroups, sendBuf);

    mpi::AllToAll(sendBuf, sendSize/Max(1, nRedistProcs), recvBuf, recvSize/Max(1, nRedistProcs), comm);

    if(!(this->Participating()))
        return;
    UnpackA2ADoubleModeCommRecvBuf(recvBuf, a2aModes, a2aCommGroups, A);
}

template <typename T>
void DistTensor<T>::PackA2ACommSendBufHelper(const A2APackData& packData, const Mode packMode, T const * const dataBuf, T * const sendBuf){
    Unsigned packSlice = packMode;
    Unsigned packSliceMaxDim = packData.sendShape[packSlice];
    Unsigned packSliceLocalDim = packData.localShape[packSlice];
    Unsigned packSliceSendBufStride = packData.sendBufModeStrides[packSlice];
    Unsigned packSliceDataBufStride = packData.dataBufModeStrides[packSlice];
    Unsigned elemSlice2 = packData.elemSlice2;
    Unsigned maxElemSlice2 = packData.maxElemSlices2;
    Unsigned elemSlice1 = packData.elemSlice1;
    Unsigned maxElemSlice1 = packData.maxElemSlices1;
    Unsigned commMode1 = packData.commMode1;
    Unsigned commMode2 = packData.commMode2;

    Unsigned sendBufPtr = 0;
    Unsigned dataBufPtr = 0;

    //    Unsigned order = Order();
    //    Unsigned i;
    //    std::string ident = "";
    //    for(i = 0; i < order - unpackMode; i++)
    //        ident += "  ";
    //    std::cout << ident << "Unpacking mode " << unpackMode << std::endl;
    if(packMode == 0){
//        std::cout << ident << "recvBuf loc " << &(recvBuf[pRecvBufPtr]) << std::endl;
//        std::cout << ident << "dataBuf loc " << &(dataBuf[pDataBufPtr]) << std::endl;
//        std::cout << ident << "unpacking recv data:";
//        for(unpackSlice = 0; unpackSlice < unpackSliceMaxDim && unpackSlice < unpackSliceLocalDim; unpackSlice++){
//            std::cout << " " << recvBuf[pRecvBufPtr];
//            pRecvBufPtr += unpackSliceRecvBufStride;
//            pDataBufPtr += unpackSliceDataBufStride;
//        }
//        std::cout << std::endl;

        if(packMode == commMode1 || packMode == commMode2){
            const Unsigned maxElemSlice = (packMode == commMode1) ? maxElemSlice1 : maxElemSlice2;
            const Unsigned elemSlice = (packMode == commMode1) ? elemSlice1 : elemSlice2;
            //NOTE: Each tensor data we unpack is strided by nRedistProcs (maxElemSlice) entries away
            packSliceDataBufStride *= maxElemSlice;

            for(packSlice = 0; (packSlice + elemSlice) < packSliceLocalDim; packSlice += maxElemSlice){
                sendBuf[sendBufPtr] = dataBuf[dataBufPtr];

//                    std::cout << "recvBuf inc by " << unpackSliceRecvBufStride << std::endl;
//                    std::cout << "dataBuf inc by " << unpackSliceDataBufStride << std::endl;
                sendBufPtr += packSliceSendBufStride;
                dataBufPtr += packSliceDataBufStride;
            }
        }else if(packSliceSendBufStride == 1 && packSliceDataBufStride == 1){
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
        if(packMode == commMode1 || packMode == commMode2){
            const Unsigned maxElemSlice = (packMode == commMode1) ? maxElemSlice1 : maxElemSlice2;
            const Unsigned elemSlice = (packMode == commMode1) ? elemSlice1 : elemSlice2;
            //NOTE: Each tensor data we unpack is strided by nRedistProcs (maxElemSlice) entries away
            packSliceDataBufStride *= maxElemSlice;

            for(packSlice = 0; (packSlice + elemSlice) < packSliceLocalDim; packSlice += maxElemSlice){
                PackA2ACommSendBufHelper(packData, packMode-1, &(dataBuf[dataBufPtr]), &(sendBuf[sendBufPtr]));

//                    std::cout << "recvBuf inc by " << unpackSliceRecvBufStride << std::endl;
//                    std::cout << "dataBuf inc by " << unpackSliceDataBufStride << std::endl;
                sendBufPtr += packSliceSendBufStride;
                dataBufPtr += packSliceDataBufStride;
            }
        }else{
            for(packSlice = 0; packSlice < packSliceMaxDim && packSlice < packSliceLocalDim; packSlice++){
                PackA2ACommSendBufHelper(packData, packMode-1, &(dataBuf[dataBufPtr]), &(sendBuf[sendBufPtr]));
    //            std::cout << "recvBuf inc by " << unpackSliceRecvBufStride << std::endl;
    //            std::cout << "dataBuf inc by " << unpackSliceDataBufStride << std::endl;
                sendBufPtr += packSliceSendBufStride;
                dataBufPtr += packSliceDataBufStride;
            }
        }
    }
}

template <typename T>
void DistTensor<T>::PackA2ADoubleModeCommSendBuf(const DistTensor<T>& A, const std::pair<Mode, Mode>& a2aModes, const std::pair<ModeArray, ModeArray >& commGroups, T * const sendBuf){
    Unsigned i, j;
    const Unsigned order = A.Order();
    const T* dataBuf = A.LockedBuffer();

//    const tmen::GridView gvA = A.GetGridView();
//    const tmen::GridView gvB = GetGridView();

//    std::cout << "dataBuf:";
//    for(Unsigned i = 0; i < prod(A.LocalShape()); i++){
//        std::cout << " " <<  dataBuf[i];
//    }
//    std::cout << std::endl;

    const tmen::Grid& g = A.Grid();

    Mode a2aMode1 = a2aModes.first;
    Mode a2aMode2 = a2aModes.second;

    ModeArray commGroup1 = commGroups.first;
    ModeArray commGroup2 = commGroups.second;

    //For convenience make sure that a2aMode1 is earlier in the packing
    if(a2aMode1 > a2aMode2){
        std::swap(a2aMode1, a2aMode2);
        std::swap(commGroup1, commGroup2);
    }

    //----------------------------------------
    //----------------------------------------
    //----------------------------------------

    const tmen::GridView gvA = A.GetGridView();
    const tmen::GridView gvB = GetGridView();

    ModeArray commModesAll = commGroup1;
    commModesAll.insert(commModesAll.end(), commGroup2.begin(), commGroup2.end());
    std::sort(commModesAll.begin(), commModesAll.end());
    const ObjShape commShapeAll = FilterVector(Grid().Shape(), commModesAll);
    const Unsigned nRedistProcsAll = Max(1, prod(commShapeAll));

    const Unsigned comm1LCM = tmen::LCM(gvA.Dimension(a2aMode1), gvB.Dimension(a2aMode1));
    const Unsigned comm2LCM = tmen::LCM(gvA.Dimension(a2aMode2), gvB.Dimension(a2aMode2));
    A2APackData packData;
    packData.sendShape = MaxLengths(A.Shape(), gvA.ParticipatingShape());
    packData.localShape = A.LocalShape();

    packData.dataBufModeStrides = A.LocalStrides();

    packData.sendBufModeStrides.resize(order);
    packData.sendBufModeStrides = Dimensions2Strides(packData.sendShape);
    packData.maxElemSlices1 = comm1LCM/gvA.Dimension(a2aMode1);
    packData.commMode1 = a2aMode1;

    packData.maxElemSlices2 = comm2LCM/gvA.Dimension(a2aMode2);
    packData.commMode2 = a2aMode2;

    Unsigned a2aMode2Stride = A.LocalModeStride(a2aMode2);
    Unsigned a2aMode1Stride = A.LocalModeStride(a2aMode1);

    const Unsigned nCommElemsPerProc = prod(packData.sendShape);
    const Location myFirstElemLoc = A.ModeShifts();
    Location packElem = myFirstElemLoc;
    //Pack only if we can
    if(ElemwiseLessThan(packElem, A.Shape())){
        //NOTE: Need to make sure this packs correct global element
    //    std::cout << "loop bounds: " << nRedistProcs2 << " " << nRedistProcs1 << std::endl;
        for(i = 0; i < packData.maxElemSlices2; i++){
            packElem[a2aMode2] = myFirstElemLoc[a2aMode2] + i * gvA.ModeWrapStride(a2aMode2);
    //        const Location elemCommLoc2 = LinearLoc2Loc(i, redistShape2);
    //        const Unsigned elemRedistLinLoc2 = Loc2LinearLoc(FilterVector(elemCommLoc2, commPerm2), commShape2);

            if(packElem[a2aMode2] >= Dimension(a2aMode2))
                continue;
            packData.elemSlice2 = i;

    //        const Location procOwner2 = LinearLoc2Loc(elemRedistLinLoc2, redistShape1);

            for(j = 0; j < packData.maxElemSlices1; j++){
                packElem[a2aMode1] = myFirstElemLoc[a2aMode1] + j * gvA.ModeWrapStride(a2aMode1);
    //            const Location elemCommLoc1 = LinearLoc2Loc(j, redistShape1);
    //            const Unsigned elemRedistLinLoc1 = Loc2LinearLoc(FilterVector(elemCommLoc1, commPerm1), commShape1);

                if(packElem[a2aMode1] >= Dimension(a2aMode1))
                    continue;

                packData.elemSlice1 = j;

    //            const Location procOwner1 = LinearLoc2Loc(elemRedistLinLoc1, redistShape2);
    //            Location procOwnerAll = procOwner2;
    //            procOwnerAll.insert(procOwnerAll.end(), procOwner1.begin(), procOwner1.end());
    //            const Unsigned commLinLoc = Loc2LinearLoc(FilterVector(procOwnerAll, commPermAll), commShapeAll);

//                PrintVector(packElem, "packElem");
                Location ownerB = DetermineOwner(packElem);
                Location ownerGridLoc = GridViewLoc2GridLoc(ownerB, gvB);
                Unsigned commLinLoc = Loc2LinearLoc(FilterVector(ownerGridLoc, commModesAll), FilterVector(g.Shape(), commModesAll));


//                std::cout << "startDataBufLoc: " << (i * a2aMode2Stride) + (j * a2aMode1Stride) << std::endl;
//                std::cout << "startSendBufLoc: " << commLinLoc * nCommElemsPerProc << std::endl;
                PackA2ACommSendBufHelper(packData, order - 1, &(dataBuf[(i * a2aMode2Stride) + (j * a2aMode1Stride)]), &(sendBuf[commLinLoc * nCommElemsPerProc]));
//                std::cout << "packed sendBuf:";
//                for(Unsigned i = 0; i < nCommElemsPerProc * nRedistProcsAll; i++)
//                    std::cout << " " << sendBuf[i];
//                std::cout << std::endl;
            }
        }
    }

//    std::cout << "packed sendBuf:";
//    for(Unsigned i = 0; i < nCommElemsPerProc * nRedistProcsAll; i++)
//        std::cout << " " << sendBuf[i];
//    std::cout << std::endl;

    //------------------------------------
    //------------------------------------
    //------------------------------------

//    ModeArray commModes  = commGroup1;
//    commModes.insert(commModes.end(), commGroup2.begin(), commGroup2.end());
//    ModeArray sortedCommModes = commModes;
//    std::sort(sortedCommModes.begin(), sortedCommModes.end());
//
//    ObjShape gridShape = g.Shape();
//
//    std::vector<Unsigned> wrapLCMs(order);
//    for(i = 0; i < order; i++)
//        wrapLCMs[i] = tmen::LCM(gvA.ModeWrapStride(i), gvB.ModeWrapStride(i));
//
//    //Number of entries to skip when packing the specified mode
//    std::vector<Unsigned> modePackStrides(order);
//    for(i = 0; i < order; i++){
//        modePackStrides[i] = MaxLength(wrapLCMs[i], gvA.ModeWrapStride(i));
//    }
//
//    const ObjShape localShape = A.LocalShape();
//    //The shape we assume each process is packing into
//    const ObjShape packLocalShape = MaxLengths(A.Shape(), gvA.ParticipatingShape());
//
//    //Slices of a2aMode1
//    const Unsigned nMaxA2AMode1Slices = packLocalShape[a2aMode1];
//    const Unsigned nLocalA2AMode1Slices = localShape[a2aMode1];
//
//    //Slices between a2aMode1 and a2aMode2
//    const Unsigned nMaxMidSlices = Max(1, prod(packLocalShape, a2aMode1 + 1, a2aMode2));
//    const Unsigned nLocalMidSlices = prod(localShape, a2aMode1 + 1, a2aMode2);
//
//    //Slices of a2aMode2
//    const Unsigned nMaxA2AMode2Slices = packLocalShape[a2aMode2];
//    const Unsigned nLocalA2AMode2Slices = localShape[a2aMode2];
//
//    //All remaining slices
//    const Unsigned nMaxOuterSlices = Max(1, prod(packLocalShape, a2aMode2 + 1));
//    const Unsigned nLocalOuterSlices = prod(localShape, a2aMode2 + 1);
//
//    const Unsigned copySliceSize = prod(localShape, 0, a2aMode1);
////    const Unsigned maxCopySliceSize = Max(1, prod(packLocalShape, 0, a2aMode1));
//    const Unsigned nElemsPerProc = prod(packLocalShape);
//
//    //Various counters used to offset in data arrays
//    Unsigned a2aMode1SliceNum, midSliceNum, a2aMode2SliceNum, outerSliceNum;  //Which slice we are packing for modeK
//    Unsigned a2aMode1SendBufOff, midSendBufOff, a2aMode2SendBufOff, outerSendBufOff;  //Offsets used to mode into data arrays
//    Unsigned a2aMode1DataBufOff, midDataBufOff, a2aMode2DataBufOff, outerDataBufOff;  //Offsets used to mode into data arrays
//    Unsigned packElemSendBufOff, packElemDataBufOff;
//    Unsigned startSendBuf, startDataBuf;
//
//    //a2aMode1 and a2aMode2 have different increments per slice because their distributions change
//    const Unsigned a2aMode1PackStride = modePackStrides[a2aMode1];
//    const Unsigned a2aMode2PackStride = modePackStrides[a2aMode2];
//
//    //The number of times we will pack
//    const Unsigned nPackA2AMode1Slices = Max(1, MaxLength(nMaxA2AMode1Slices, a2aMode1PackStride));
//    const Unsigned nPackA2AMode2Slices = Max(1, MaxLength(nMaxA2AMode2Slices, a2aMode2PackStride));
//
//    Location myFirstLoc = A.ModeShifts();
//
//    Unsigned packElemNum;
//    const Unsigned nPackElems = prod(modePackStrides);
//
////    printf("MemCopy info:\n");
////    printf("    nPackElems: %d\n", nPackElems);
////    printf("    nMaxOuterSlices: %d\n", nMaxOuterSlices);
////    printf("    nA2AMode2Slices: %d\n", nMaxA2AMode2Slices);
////    printf("    nMaxMidSlices: %d\n", nMaxMidSlices);
////    printf("    nA2AMode1Slices: %d\n", nMaxA2AMode1Slices);
////    printf("    copySliceSize: %d\n", copySliceSize);
//    for(packElemNum = 0; packElemNum < nPackElems; packElemNum++){
//        Location packElemMultiLoc = LinearLoc2Loc(packElemNum, modePackStrides);
//
//        //Determine the global mode of this first element we are packing
//        Location startPackElemLoc = myFirstLoc;
//        for(i = 0; i < order; i++){
//            startPackElemLoc[i] += packElemMultiLoc[i] * gvA.ModeWrapStride(i);
//        }
//
//        //If we run over the edge, don't try to pack the global element
//        if(AnyElemwiseGreaterThanEqualTo(startPackElemLoc, A.Shape()))
//            continue;
//
//        //Determine the Multiloc of the process that owns this element
//        Location owningProcGVB = this->DetermineOwner(startPackElemLoc);
//        Location owningProcG = GridViewLoc2GridLoc(owningProcGVB, gvB);
//        Unsigned owningProc = Loc2LinearLoc(FilterVector(owningProcG, sortedCommModes), FilterVector(gridShape, sortedCommModes));
//
//        //Find the local location of the global starting element we are now packing
//        Location localLoc = A.Global2LocalIndex(startPackElemLoc);
//
//        //Update the corresponding offsets
//        packElemSendBufOff = nElemsPerProc * owningProc;
//        packElemDataBufOff = Loc2LinearLoc(localLoc, localShape);
//
////        printf("        packElemSendBufOff: %d\n", packElemSendBufOff);
////        printf("        packElemDataBufOff: %d\n", packElemDataBufOff);
//        //Now that we have figured out the starting point, begin copying the entire slice from this element
//        for(outerSliceNum = 0; outerSliceNum < nMaxOuterSlices; outerSliceNum++){
//            if(outerSliceNum >= nLocalOuterSlices)
//                break;
//            outerSendBufOff = copySliceSize * nPackA2AMode1Slices * nMaxMidSlices * nPackA2AMode2Slices * outerSliceNum;
//            outerDataBufOff = copySliceSize * nLocalA2AMode1Slices * nLocalMidSlices * nLocalA2AMode2Slices * outerSliceNum;
//
////            printf("          outerSliceNum: %d\n", outerSliceNum);
////            printf("          outerSendBufOff: %d\n", outerSendBufOff);
////            printf("          outerDataBufOff: %d\n", outerDataBufOff);
//            for(a2aMode2SliceNum = 0; a2aMode2SliceNum < nMaxA2AMode2Slices; a2aMode2SliceNum += a2aMode2PackStride){
//                if(a2aMode2SliceNum >= nLocalA2AMode2Slices)
//                    break;
//                a2aMode2SendBufOff = copySliceSize * nPackA2AMode1Slices * nMaxMidSlices * (a2aMode2SliceNum / a2aMode2PackStride);
//                a2aMode2DataBufOff = copySliceSize * nLocalA2AMode1Slices * nLocalMidSlices * a2aMode2SliceNum;
//
////                printf("            a2aMode2SliceNum: %d\n", a2aMode2SliceNum);
////                printf("            a2aMode2SendBufOff: %d\n", a2aMode2SendBufOff);
////                printf("            a2aMode2DataBufOff: %d\n", a2aMode2DataBufOff);
//                for(midSliceNum = 0; midSliceNum < nMaxMidSlices; midSliceNum++){
//                    if(midSliceNum >= nLocalMidSlices)
//                        break;
//                    midSendBufOff = copySliceSize * nPackA2AMode1Slices * midSliceNum;
//                    midDataBufOff = copySliceSize * nLocalA2AMode1Slices * midSliceNum;
//
////                    printf("              midSliceNum: %d\n", midSliceNum);
////                    printf("              midSendBufOff: %d\n", midSendBufOff);
////                    printf("              midDataBufOff: %d\n", midDataBufOff);
//                    for(a2aMode1SliceNum = 0; a2aMode1SliceNum < nMaxA2AMode1Slices; a2aMode1SliceNum += a2aMode1PackStride){
//                        if(a2aMode1SliceNum >= nLocalA2AMode1Slices)
//                            break;
//                        a2aMode1SendBufOff = copySliceSize * (a2aMode1SliceNum / a2aMode1PackStride);
//                        a2aMode1DataBufOff = copySliceSize * a2aMode1SliceNum;
//
////                        printf("                a2aMode1SliceNum: %d\n", a2aMode1SliceNum);
////                        printf("                a2aMode1SendBufOff: %d\n", a2aMode1SendBufOff);
////                        printf("                a2aMode1DataBufOff: %d\n", a2aMode1DataBufOff);
//                        //Down to all contiguous slices, so just copy
//
//                        startSendBuf = packElemSendBufOff + outerSendBufOff + a2aMode2SendBufOff + midSendBufOff + a2aMode1SendBufOff;
//                        startDataBuf = packElemDataBufOff + outerDataBufOff + a2aMode2DataBufOff + midDataBufOff + a2aMode1DataBufOff;
//
////                        printf("                  startSendBuf: %d\n", startSendBuf);
////                        printf("                  startDataBuf: %d\n", startDataBuf);
//                        MemCopy(&(sendBuf[startSendBuf]), &(dataBuf[startDataBuf]), copySliceSize);
//                    }
//                }
//            }
//        }
//    }

//    const ObjShape commGridSlice = FilterVector(this->Grid().Shape(), commModes);
//    const Unsigned nRedistProcs = prod(commGridSlice);
//    std::cout << "packed sendBuf:";
//    for(Unsigned i = 0; i < prod(packLocalShape) * nRedistProcs; i++)
//        std::cout << " " << sendBuf[i];
//    std::cout << std::endl;
}

template<typename T>
void DistTensor<T>::UnpackA2ACommRecvBufHelper(const A2AUnpackData& unpackData, const Mode unpackMode, T const * const recvBuf, T * const dataBuf){
    Unsigned unpackSlice = unpackMode;
    Unsigned unpackSliceMaxDim = unpackData.recvShape[unpackSlice];
    Unsigned unpackSliceLocalDim = unpackData.localShape[unpackSlice];
    Unsigned unpackSliceRecvBufStride = unpackData.recvBufModeStrides[unpackSlice];
    Unsigned unpackSliceDataBufStride = unpackData.dataBufModeStrides[unpackSlice];
    Unsigned elemSlice2 = unpackData.elemSlice2;
    Unsigned maxElemSlice2 = unpackData.elemSlice2Stride;
    Unsigned elemSlice1 = unpackData.elemSlice1;
    Unsigned maxElemSlice1 = unpackData.elemSlice1Stride;
    Unsigned commMode1 = unpackData.commMode1;
    Unsigned commMode2 = unpackData.commMode2;
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

    if(unpackMode == 0){
        if(unpackMode == commMode1 || unpackMode == commMode2){
            Unsigned elemSlice = (unpackMode == commMode1) ? unpackData.elemSlice1 : unpackData.elemSlice2;
            Unsigned maxElemSlice = (unpackMode == commMode1) ? unpackData.elemSlice1Stride : unpackData.elemSlice2Stride;

            //NOTE: Each tensor data we unpack is strided by nRedistProcs (maxElemSlice) entries away
            unpackSliceDataBufStride *= maxElemSlice;
//            std::cout << ident << "recvBuf loc " << &(recvBuf[pRecvBufPtr]) << std::endl;
//            std::cout << ident << "dataBuf loc " << &(dataBuf[pDataBufPtr]) << std::endl;
//            std::cout << ident << "unpacking recv data:";
//            for(unpackSlice = 0; unpackSlice < unpackSliceMaxDim && (unpackSlice + elemSlice) < unpackSliceLocalDim; unpackSlice += maxElemSlice){
//                std::cout << " " << recvBuf[pRecvBufPtr];
//
//                pRecvBufPtr += unpackSliceRecvBufStride;
//                pDataBufPtr += unpackSliceDataBufStride;
//            }
//            std::cout << std::endl;

            for(unpackSlice = 0; (unpackSlice + elemSlice) < unpackSliceLocalDim; unpackSlice += maxElemSlice){
                dataBuf[dataBufPtr] = recvBuf[recvBufPtr];

//                std::cout << ident << "recvBuf inc by " << unpackSliceRecvBufStride << std::endl;
//                std::cout << ident << "dataBuf inc by " << unpackSliceDataBufStride << std::endl;
                recvBufPtr += unpackSliceRecvBufStride;
                dataBufPtr += unpackSliceDataBufStride;
            }
        }else if(unpackSliceRecvBufStride == 1 && unpackSliceDataBufStride == 1){
//            std::cout << ident << "unpacking recv data:";
//            for(unpackSlice = 0; unpackSlice < unpackSliceLocalDim; unpackSlice++){
//                std::cout << " " << recvBuf[pRecvBufPtr];
//                pRecvBufPtr += unpackSliceRecvBufStride;
//                pDataBufPtr += unpackSliceDataBufStride;
//            }
//            std::cout << std::endl;
            MemCopy(&(dataBuf[0]), &(recvBuf[0]), unpackSliceLocalDim);
        }else{
//            std::cout << ident << "unpacking recv data:";
//            for(unpackSlice = 0; unpackSlice < unpackSliceLocalDim; unpackSlice++){
//                std::cout << " " << recvBuf[pRecvBufPtr];
//                pRecvBufPtr += unpackSliceRecvBufStride;
//                pDataBufPtr += unpackSliceDataBufStride;
//            }
//            std::cout << std::endl;
            for(unpackSlice = 0; unpackSlice < unpackSliceLocalDim; unpackSlice++){
                dataBuf[dataBufPtr] = recvBuf[recvBufPtr];

//                std::cout << ident << "recvBuf inc by " << unpackSliceRecvBufStride << std::endl;
//                std::cout << ident << "dataBuf inc by " << unpackSliceDataBufStride << std::endl;
                recvBufPtr += unpackSliceRecvBufStride;
                dataBufPtr += unpackSliceDataBufStride;
            }
        }
    }else {
        if(unpackMode == commMode1 || unpackMode == commMode2){
            Unsigned elemSlice = (unpackMode == commMode1) ? unpackData.elemSlice1 : unpackData.elemSlice2;
            Unsigned maxElemSlice = (unpackMode == commMode1) ? unpackData.elemSlice1Stride : unpackData.elemSlice2Stride;

            //NOTE: Each tensor data we unpack is strided by nRedistProcs (maxElemSlice) entries away
            unpackSliceDataBufStride *= maxElemSlice;

            for(unpackSlice = 0; (unpackSlice + elemSlice) < unpackSliceLocalDim; unpackSlice += maxElemSlice){
                UnpackA2ACommRecvBufHelper(unpackData, unpackMode-1, &(recvBuf[recvBufPtr]), &(dataBuf[dataBufPtr]));

//                std::cout << ident << "recvBuf inc by " << unpackSliceRecvBufStride << std::endl;
//                std::cout << ident << "dataBuf inc by " << unpackSliceDataBufStride << std::endl;
                recvBufPtr += unpackSliceRecvBufStride;
                dataBufPtr += unpackSliceDataBufStride;
            }
        }else{
            for(unpackSlice = 0; unpackSlice < unpackSliceLocalDim; unpackSlice++){
                UnpackA2ACommRecvBufHelper(unpackData, unpackMode-1, &(recvBuf[recvBufPtr]), &(dataBuf[dataBufPtr]));
//                std::cout << ident << "recvBuf inc by " << unpackSliceRecvBufStride << std::endl;
//                std::cout << ident << "dataBuf inc by " << unpackSliceDataBufStride << std::endl;
                recvBufPtr += unpackSliceRecvBufStride;
                dataBufPtr += unpackSliceDataBufStride;
            }
        }
    }
}

template<typename T>
void DistTensor<T>::UnpackA2ADoubleModeCommRecvBuf(const T * const recvBuf, const std::pair<Mode, Mode>& a2aModes, const std::pair<ModeArray, ModeArray >& commGroups, const DistTensor<T>& A){
    Unsigned i, j;
    const Unsigned order = A.Order();
    T* dataBuf = this->Buffer();



    const tmen::GridView gvA = A.GetGridView();
    const tmen::GridView gvB = GetGridView();

    const tmen::Grid& g = A.Grid();

    Mode a2aMode1 = a2aModes.first;
    Mode a2aMode2 = a2aModes.second;

    ModeArray commGroup1 = commGroups.first;
    ModeArray commGroup2 = commGroups.second;

    //For convenience make sure that a2aMode1 is earlier in the packing
    if(a2aMode1 > a2aMode2){
        std::swap(a2aMode1, a2aMode2);
        std::swap(commGroup1, commGroup2);
    }

    //------------------------------------
    //------------------------------------
    //------------------------------------

    ModeArray commModesAll = commGroup1;
    commModesAll.insert(commModesAll.end(), commGroup2.begin(), commGroup2.end());
    std::sort(commModesAll.begin(), commModesAll.end());
    const ObjShape commShapeAll = FilterVector(Grid().Shape(), commModesAll);
    const Unsigned nRedistProcsAll = Max(1, prod(commShapeAll));

    const Unsigned comm1LCM = tmen::LCM(gvA.Dimension(a2aMode1), gvB.Dimension(a2aMode1));
    const Unsigned comm2LCM = tmen::LCM(gvA.Dimension(a2aMode2), gvB.Dimension(a2aMode2));
    A2AUnpackData unpackData;
    unpackData.recvShape = MaxLengths(A.Shape(), gvA.ParticipatingShape());
    unpackData.localShape = LocalShape();

    unpackData.dataBufModeStrides = LocalStrides();

    unpackData.recvBufModeStrides.resize(order);
    unpackData.recvBufModeStrides = Dimensions2Strides(unpackData.recvShape);
    unpackData.elemSlice1Stride = comm1LCM/gvB.Dimension(a2aMode1);
    unpackData.commMode1 = a2aMode1;

    unpackData.elemSlice2Stride = comm2LCM/gvB.Dimension(a2aMode2);
    unpackData.commMode2 = a2aMode2;

    Unsigned a2aMode2Stride = LocalModeStride(a2aMode2);
    Unsigned a2aMode1Stride = LocalModeStride(a2aMode1);

    const Unsigned nCommElemsPerProc = prod(unpackData.recvShape);
    const Location myFirstElemLoc = ModeShifts();

//    std::cout << "recvBuf:";
//    for(Unsigned i = 0; i < nRedistProcsAll * nCommElemsPerProc; i++){
//        std::cout << " " << recvBuf[i];
//    }
//    std::cout << std::endl;

    Location unpackElem = myFirstElemLoc;
    //NOTE: Need to make sure this packs correct global element
//    std::cout << "loop bounds: " << nRedistProcs2 << " " << nRedistProcs1 << std::endl;
    //Unpack only if we can
    if(ElemwiseLessThan(unpackElem, Shape())){
        for(i = 0; i < unpackData.elemSlice2Stride; i++){
            unpackElem[a2aMode2] = myFirstElemLoc[a2aMode2] + i*gvB.ModeWrapStride(a2aMode2);
    //        const Location elemCommLoc2 = LinearLoc2Loc(i, redistShape2);
    //        const Unsigned elemRedistLinLoc2 = Loc2LinearLoc(FilterVector(elemCommLoc2, commPerm2), commShape2);

            if(unpackElem[a2aMode2] >= Dimension(a2aMode2))
                continue;
            unpackData.elemSlice2 = i;

    //        const Location procOwner2 = LinearLoc2Loc(elemRedistLinLoc2, redistShape1);

            for(j = 0; j < unpackData.elemSlice1Stride; j++){
                unpackElem[a2aMode1] = myFirstElemLoc[a2aMode1] + j*gvB.ModeWrapStride(a2aMode1);
    //            const Location elemCommLoc1 = LinearLoc2Loc(j, redistShape1);
    //            const Unsigned elemRedistLinLoc1 = Loc2LinearLoc(FilterVector(elemCommLoc1, commPerm1), commShape1);

                if(unpackElem[a2aMode1] >= Dimension(a2aMode1))
                    continue;

                unpackData.elemSlice1 = j;

    //            const Location procOwner1 = LinearLoc2Loc(elemRedistLinLoc1, redistShape2);
    //            Location procOwnerAll = procOwner2;
    //            procOwnerAll.insert(procOwnerAll.end(), procOwner1.begin(), procOwner1.end());
    //            const Unsigned commLinLoc = Loc2LinearLoc(FilterVector(procOwnerAll, commPermAll), commShapeAll);

                Location ownerA = A.DetermineOwner(unpackElem);
                Location ownerGridLoc = GridViewLoc2GridLoc(ownerA, gvA);
                Unsigned commLinLoc = Loc2LinearLoc(FilterVector(ownerGridLoc, commModesAll), FilterVector(g.Shape(), commModesAll));

//                PrintVector(unpackElem, "unpackElem");
//                std::cout << "startDataBufLoc: " << (i * a2aMode2Stride) + (j * a2aMode1Stride) << std::endl;
//                std::cout << "startRecvBufLoc: " << commLinLoc * nCommElemsPerProc << std::endl;
                UnpackA2ACommRecvBufHelper(unpackData, order - 1, &(recvBuf[commLinLoc * nCommElemsPerProc]), &(dataBuf[(i * a2aMode2Stride) + (j * a2aMode1Stride)]));
//                std::cout << "data:";
//                for(Unsigned k = 0; k < prod(unpackData.localShape); k++)
//                    std::cout << " " << dataBuf[k];
//                std::cout << std::endl;
            }
        }
    }
    //------------------------------------
    //------------------------------------
    //------------------------------------

//    ModeArray commModes  = commGroup1;
//    commModes.insert(commModes.end(), commGroup2.begin(), commGroup2.end());
//    ModeArray sortedCommModes = commModes;
//    std::sort(sortedCommModes.begin(), sortedCommModes.end());
//
//    ModeArray nonCommModes;
//    for(i = 0; i < g.Order(); i++){
//        if(std::find(commModes.begin(), commModes.end(), i) == commModes.end()){
//            nonCommModes.push_back(i);
//        }
//    }
//
////    const ObjShape commGridSlice = FilterVector(this->Grid().Shape(), commModes);
////    const Unsigned nRedistProcs = prod(commGridSlice);
////    std::cout << "recvBuf:";
////    for(Unsigned i = 0; i < prod(MaxLengths(A.Shape(), gvA.ParticipatingShape())) * nRedistProcs; i++)
////        std::cout << " " << recvBuf[i];
////    std::cout << std::endl;
//
//    ObjShape tensorShape = this->Shape();
//    Location myGridLoc = g.Loc();
//    ObjShape gridShape = g.Shape();
//
//    std::vector<Unsigned> modeLCMs(order);
//    for(Unsigned i = 0; i < order; i++)
//        modeLCMs[i] = tmen::LCM(gvA.ModeWrapStride(i), gvB.ModeWrapStride(i));
//
//    //Stride taken to unpack into databuf per mode
//    std::vector<Unsigned> modeUnpackStrides(order);
//    for(Unsigned i = 0; i < order; i++)
//        modeUnpackStrides[i] = modeLCMs[i] / gvB.ModeWrapStride(i);
//
//    std::vector<Unsigned> modePackStrides(order);
//    for(Unsigned i = 0; i < order; i++)
//        modePackStrides[i] = modeLCMs[i] / gvA.ModeWrapStride(i);
//
//    const ObjShape localShape = this->LocalShape();
//    ObjShape packedLocalShape = MaxLengths(A.Shape(), gvA.ParticipatingShape());
//
//    //Slices of a2aMode1
//    const Unsigned nLocalA2AMode1Slices = localShape[a2aMode1];
//
//    //Slices between a2aMode1 and a2aMode2
//    const Unsigned nLocalMidSlices = Max(1, prod(localShape, a2aMode1 + 1, a2aMode2));
//
//    //Slices of a2aMode2
//    const Unsigned nLocalA2AMode2Slices = localShape[a2aMode2];
//
//    const Unsigned copySliceSize = prod(localShape, 0, a2aMode1);
//    const Unsigned nElemsPerProc = prod(packedLocalShape);
//
//    //Various counters used to offset in data arrays
//    Unsigned a2aMode1SliceNum, midSliceNum, a2aMode2SliceNum, outerSliceNum;  //Which slice we are packing for indexK
//    Unsigned a2aMode1RecvBufOff, midRecvBufOff, a2aMode2RecvBufOff, outerRecvBufOff;  //Offsets used to index into data arrays
//    Unsigned a2aMode1DataBufOff, midDataBufOff, a2aMode2DataBufOff, outerDataBufOff;  //Offsets used to index into data arrays
//    Unsigned unpackElemRecvBufOff, unpackElemDataBufOff;
//    Unsigned startRecvBuf, startDataBuf;
//
//    const Unsigned a2aMode1UnpackStride = modeUnpackStrides[a2aMode1];
//    const Unsigned a2aMode2UnpackStride = modeUnpackStrides[a2aMode2];
//
//    const Unsigned a2aMode1PackStride = modePackStrides[a2aMode1];
//    const Unsigned a2aMode2PackStride = modePackStrides[a2aMode2];
//
//    Location myFirstLoc = this->ModeShifts();
//
//    Unsigned unpackElemNum;
//    const Unsigned nUnpackElems = prod(modeUnpackStrides);
//
//    for(unpackElemNum = 0; unpackElemNum < nUnpackElems; unpackElemNum++){
//        Location unpackElemMultiLoc = LinearLoc2Loc(unpackElemNum, modeUnpackStrides);
//
//        //Determine the global index of this first element we are packing
//        Location startUnpackElemLoc = myFirstLoc;
//        for(Unsigned i = 0; i < order; i++){
//            startUnpackElemLoc[i] += unpackElemMultiLoc[i] * gvB.ModeWrapStride(i);
//        }
//
//        //If we run over the edge, don't try to unpack the global element
//        if(AnyElemwiseGreaterThanEqualTo(startUnpackElemLoc, this->Shape()))
//            continue;
//
//        //Determine the Multiloc of the process that sent this element
//        Location owningProcGVA = A.DetermineOwner(startUnpackElemLoc);
//        Location owningProcG = GridViewLoc2GridLoc(owningProcGVA, gvA);
//        Unsigned owningProc = Loc2LinearLoc(FilterVector(owningProcG, sortedCommModes), FilterVector(gridShape, sortedCommModes));
//
//        //Find the local location of the global starting element we are now unpacking
//        Location localLoc = this->Global2LocalIndex(startUnpackElemLoc);
//
//        //Now that we know the local loc of the element to unpack, we know how many iterations of unpacking to perform per mode
//        const ObjShape tensorShape = this->Shape();
//        const ObjShape gvAShape = gvA.ParticipatingShape();
//
//        const ObjShape outerSliceShape(tensorShape.begin() + a2aMode2 + 1, tensorShape.end());
//        const Location outerSliceMultiLoc(startUnpackElemLoc.begin() + a2aMode2 + 1, startUnpackElemLoc.end());
//        const ObjShape outerSlicePackShape(gvAShape.begin() + a2aMode2 + 1, gvAShape.end());
//        std::vector<Unsigned> nPackedOuterWraps(outerSliceShape.size());
//        for(i = 0; i < nPackedOuterWraps.size(); i++){
//            nPackedOuterWraps[i] = MaxLength(outerSliceShape[i] - outerSliceMultiLoc[i],outerSlicePackShape[i]);
//        }
//        const Unsigned nPackedOuterSlices = Max(1, prod(nPackedOuterWraps));
//
//        const Unsigned nMaxPackedA2AMode2Slices = Max(1, MaxLength(tensorShape[a2aMode2], gvAShape[a2aMode2] * a2aMode2PackStride));
//        const Unsigned nPackedA2AMode2Slices = Max(1, MaxLength(tensorShape[a2aMode2] - startUnpackElemLoc[a2aMode2], gvAShape[a2aMode2] * a2aMode2PackStride));
//
//        const ObjShape midSliceShape(tensorShape.begin() + a2aMode1 + 1, tensorShape.begin() + a2aMode2);
//        const Location midSliceMultiLoc(startUnpackElemLoc.begin() + a2aMode1 + 1, startUnpackElemLoc.begin() + a2aMode2);
//        const ObjShape midSlicePackShape(gvAShape.begin() + a2aMode1 + 1, gvAShape.begin() + a2aMode2);
//        std::vector<Unsigned> nMaxPackedMidWraps(midSliceShape.size());
//        std::vector<Unsigned> nPackedMidWraps(midSliceShape.size());
//        for(i = 0; i < nPackedMidWraps.size(); i++){
//            nPackedMidWraps[i] = MaxLength(midSliceShape[i] - midSliceMultiLoc[i], midSlicePackShape[i] );
//            nMaxPackedMidWraps[i] = MaxLength(midSliceShape[i], midSlicePackShape[i]);
//        }
//        const Unsigned nMaxPackedMidSlices = Max(1, prod(nMaxPackedMidWraps));
//        const Unsigned nPackedMidSlices = Max(1, prod(nPackedMidWraps));
//
//        const Unsigned nMaxPackedA2AMode1Slices = Max(1, MaxLength(tensorShape[a2aMode1], gvAShape[a2aMode1] * a2aMode1PackStride));
//        const Unsigned nPackedA2AMode1Slices = Max(1, MaxLength(tensorShape[a2aMode1] - startUnpackElemLoc[a2aMode1], gvAShape[a2aMode1] * a2aMode1PackStride));
//
//        //Update the corresponding offsets
//        unpackElemRecvBufOff = nElemsPerProc * owningProc;
//        unpackElemDataBufOff = Loc2LinearLoc(localLoc, localShape);
//
//
////        printf("MemCopy info:\n");
////        printf("    unpackElemRecvBufOff: %d\n", unpackElemRecvBufOff);
////        printf("    unpackElemDataBufOff: %d\n", unpackElemDataBufOff);
////        printf("    nPackElems: %d\n", nUnpackElems);
////        printf("    nPackedOuterSlices: %d\n", nPackedOuterSlices);
////        printf("    nPackedA2AMode2Slices: %d\n", nPackedA2AMode2Slices);
////        printf("    nPackedMidSlices: %d\n", nPackedMidSlices);
////        printf("    nPackedA2AMode1Slices: %d\n", nPackedA2AMode1Slices);
////        printf("    copySliceSize: %d\n", copySliceSize);
//        //Now that we have figured out the starting point, begin copying the entire slice from this element
//        for(outerSliceNum = 0; outerSliceNum < nPackedOuterSlices; outerSliceNum++){
//
//            outerRecvBufOff = copySliceSize * nMaxPackedA2AMode1Slices * nMaxPackedMidSlices * nMaxPackedA2AMode2Slices * outerSliceNum;
//            outerDataBufOff = copySliceSize * nLocalA2AMode1Slices * nLocalMidSlices * nLocalA2AMode2Slices * outerSliceNum;
//
////            printf("      outerSliceNum: %d\n", outerSliceNum);
////            printf("      outerRecvBufOff: %d\n", outerRecvBufOff);
////            printf("      outerDataBufOff: %d\n", outerDataBufOff);
//            for(a2aMode2SliceNum = 0; a2aMode2SliceNum < nPackedA2AMode2Slices; a2aMode2SliceNum++){
//
//                a2aMode2RecvBufOff = copySliceSize * nMaxPackedA2AMode1Slices * nMaxPackedMidSlices * a2aMode2SliceNum;
//                a2aMode2DataBufOff = copySliceSize * nLocalA2AMode1Slices * nLocalMidSlices * (a2aMode2SliceNum * a2aMode2UnpackStride);
//
////                printf("        a2aMode2SliceNum: %d\n", a2aMode2SliceNum);
////                printf("        a2aMode2RecvBufOff: %d\n", a2aMode2RecvBufOff);
////                printf("        a2aMode2DataBufOff: %d\n", a2aMode2DataBufOff);
//                for(midSliceNum = 0; midSliceNum < nPackedMidSlices; midSliceNum++){
//                    midRecvBufOff = copySliceSize * nMaxPackedA2AMode1Slices * midSliceNum;
//                    midDataBufOff = copySliceSize * nLocalA2AMode1Slices * midSliceNum;
//
////                    printf("          midSliceNum: %d\n", midSliceNum);
////                    printf("          midRecvBufOff: %d\n", midRecvBufOff);
////                    printf("          midDataBufOff: %d\n", midDataBufOff);
//                    for(a2aMode1SliceNum = 0; a2aMode1SliceNum < nPackedA2AMode1Slices; a2aMode1SliceNum++){
//                        a2aMode1RecvBufOff = copySliceSize * a2aMode1SliceNum;
//                        a2aMode1DataBufOff = copySliceSize * (a2aMode1SliceNum * a2aMode1UnpackStride);
//
////                        printf("            a2aMode1SliceNum: %d\n", a2aMode1SliceNum);
////                        printf("            a2aMode1RecvBufOff: %d\n", a2aMode1RecvBufOff);
////                        printf("            a2aMode1DataBufOff: %d\n", a2aMode1DataBufOff);
//                        //Down to all contiguous slices, so just copy
//                        startRecvBuf = unpackElemRecvBufOff + outerRecvBufOff + a2aMode2RecvBufOff + midRecvBufOff + a2aMode1RecvBufOff;
//                        startDataBuf = unpackElemDataBufOff + outerDataBufOff + a2aMode2DataBufOff + midDataBufOff + a2aMode1DataBufOff;
//
////                        printf("                startRecvBuf: %d\n", startRecvBuf);
////                        printf("                startDataBuf: %d\n", startDataBuf);
//                        MemCopy(&(dataBuf[startDataBuf]), &(recvBuf[startRecvBuf]), copySliceSize);
//
//                    }
//                }
//            }
//        }
//    }
//    std::cout << "dataBuf:";
//    for(Unsigned i = 0; i < prod(LocalShape()); i++)
//        std::cout << " " << dataBuf[i];
//    std::cout << std::endl;
}


#define PROTO(T) \
        template Int  DistTensor<T>::CheckAllToAllDoubleModeCommRedist(const DistTensor<T>& A, const std::pair<Mode, Mode>& a2aModes, const std::pair<ModeArray, ModeArray >& a2aCommGroups); \
        template void DistTensor<T>::AllToAllDoubleModeCommRedist(const DistTensor<T>& A, const std::pair<Mode, Mode>& a2aIndices, const std::pair<ModeArray, ModeArray >& commGroups); \
        template void DistTensor<T>::PackA2ADoubleModeCommSendBuf(const DistTensor<T>& A, const std::pair<Mode, Mode>& a2aModes, const std::pair<ModeArray, ModeArray >& commGroups, T * const sendBuf); \
        template void DistTensor<T>::UnpackA2ADoubleModeCommRecvBuf(const T * const recvBuf, const std::pair<Mode, Mode>& a2aModes, const std::pair<ModeArray, ModeArray >& commGroups, const DistTensor<T>& A);

PROTO(int)
PROTO(float)
PROTO(double)
PROTO(Complex<double>)
PROTO(Complex<float>)

} //namespace tmen
