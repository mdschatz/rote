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
    if(!A.Participating())
        return;
    Unsigned sendSize, recvSize;

    //Determine buffer sizes for communication
    //NOTE: Swap to concatenate vectors
    ModeArray commModes = a2aCommGroups.first;
    commModes.insert(commModes.end(), a2aCommGroups.second.begin(), a2aCommGroups.second.end());
    std::sort(commModes.begin(), commModes.end());

    const Unsigned nRedistProcs = prod(FilterVector(A.Grid().Shape(), commModes));
    const ObjShape maxLocalShape = MaxLengths(A.Shape(), A.GetGridView().Shape());

    sendSize = prod(maxLocalShape) * nRedistProcs;
    recvSize = sendSize;

    const mpi::Comm comm = A.GetCommunicatorForModes(commModes);

    Memory<T> auxMemory;
    T* auxBuf = auxMemory.Require(sendSize + recvSize);
    MemZero(&(auxBuf[0]), sendSize + recvSize);

    T* sendBuf = &(auxBuf[0]);
    T* recvBuf = &(auxBuf[sendSize]);

    PackA2ADoubleModeCommSendBuf(A, a2aModes, a2aCommGroups, sendBuf);

    mpi::AllToAll(sendBuf, sendSize/nRedistProcs, recvBuf, recvSize/nRedistProcs, comm);

    UnpackA2ADoubleModeCommRecvBuf(recvBuf, a2aModes, a2aCommGroups, A);
}

template <typename T>
void DistTensor<T>::PackA2ADoubleModeCommSendBuf(const DistTensor<T>& A, const std::pair<Mode, Mode>& a2aModes, const std::pair<ModeArray, ModeArray >& commGroups, T * const sendBuf){
    Unsigned i;
    const Unsigned order = A.Order();
    const T* dataBuf = A.LockedBuffer();

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

    ModeArray commModes  = commGroup1;
    commModes.insert(commModes.end(), commGroup2.begin(), commGroup2.end());

    ObjShape gridShape = g.Shape();

    std::vector<Unsigned> wrapLCMs(order);
    for(i = 0; i < order; i++)
        wrapLCMs[i] = tmen::LCM(gvA.ModeWrapStride(i), gvB.ModeWrapStride(i));

    //Number of entries to skip when packing the specified mode
    std::vector<Unsigned> modePackStrides(order);
    for(i = 0; i < order; i++){
        modePackStrides[i] = wrapLCMs[i] / gvA.ModeWrapStride(i);
    }

    const ObjShape localShape = A.LocalShape();
    //The shape we assume each process is packing into
    const ObjShape packLocalShape = MaxLengths(A.Shape(), gvA.Shape());

    //Slices of a2aMode1
    const Unsigned nMaxA2AMode1Slices = packLocalShape[a2aMode1];
    const Unsigned nLocalA2AMode1Slices = localShape[a2aMode1];

    //Slices between a2aMode1 and a2aMode2
    const Unsigned nMaxMidSlices = Max(1, prod(packLocalShape, a2aMode1 + 1, a2aMode2));
    const Unsigned nLocalMidSlices = prod(localShape, a2aMode1 + 1, a2aMode2);

    //Slices of a2aMode2
    const Unsigned nMaxA2AMode2Slices = packLocalShape[a2aMode2];
    const Unsigned nLocalA2AMode2Slices = localShape[a2aMode2];

    //All remaining slices
    const Unsigned nMaxOuterSlices = Max(1, prod(packLocalShape, a2aMode2 + 1));
    const Unsigned nLocalOuterSlices = prod(localShape, a2aMode2 + 1);

    const Unsigned copySliceSize = A.LocalModeStride(a2aMode1);
    const Unsigned nElemsPerProc = prod(packLocalShape);

    //Various counters used to offset in data arrays
    Unsigned a2aMode1SliceNum, midSliceNum, a2aMode2SliceNum, outerSliceNum;  //Which slice we are packing for modeK
    Unsigned a2aMode1SendBufOff, midSendBufOff, a2aMode2SendBufOff, outerSendBufOff;  //Offsets used to mode into data arrays
    Unsigned a2aMode1DataBufOff, midDataBufOff, a2aMode2DataBufOff, outerDataBufOff;  //Offsets used to mode into data arrays
    Unsigned packElemSendBufOff, packElemDataBufOff;
    Unsigned startSendBuf, startDataBuf;

    //a2aMode1 and a2aMode2 have different increments per slice because their distributions change
    const Unsigned a2aMode1PackStride = modePackStrides[a2aMode1];
    const Unsigned a2aMode2PackStride = modePackStrides[a2aMode2];

    //The number of times we will pack
    const Unsigned nPackA2AMode1Slices = Max(1, ((nMaxA2AMode1Slices - 1) / a2aMode1PackStride + 1));
    const Unsigned nPackA2AMode2Slices = Max(1, ((nMaxA2AMode2Slices - 1) / a2aMode2PackStride + 1));

    Location myFirstLoc = A.ModeShifts();

    Unsigned packElemNum;
    const Unsigned nPackElems = prod(modePackStrides);

//    printf("MemCopy info:\n");
//    printf("    nPackElems: %d\n", nPackElems);
//    printf("    nMaxOuterSlices: %d\n", nMaxOuterSlices);
//    printf("    nA2AMode2Slices: %d\n", nMaxA2AMode2Slices);
//    printf("    nMaxMidSlices: %d\n", nMaxMidSlices);
//    printf("    nA2AMode1Slices: %d\n", nMaxA2AMode1Slices);
//    printf("    copySliceSize: %d\n", copySliceSize);
    for(packElemNum = 0; packElemNum < nPackElems; packElemNum++){
        Location packElemMultiLoc = LinearLoc2Loc(packElemNum, modePackStrides);

        //Determine the global mode of this first element we are packing
        Location startPackElemLoc = myFirstLoc;
        for(i = 0; i < order; i++){
            startPackElemLoc[i] += packElemMultiLoc[i] * gvA.ModeWrapStride(i);
        }

        //If we run over the edge, don't try to pack the global element
        if(AnyElemwiseGreaterThanEqualTo(startPackElemLoc, A.Shape()))
            continue;

        //Determine the Multiloc of the process that owns this element
        Location owningProcGVB = this->DetermineOwner(startPackElemLoc);
        Location owningProcG = GridViewLoc2GridLoc(owningProcGVB, gvB);
        Unsigned owningProc = Loc2LinearLoc(FilterVector(owningProcG, commModes), FilterVector(gridShape, commModes));

        //Find the local location of the global starting element we are now packing
        Location localLoc = A.Global2LocalIndex(startPackElemLoc);

        //Update the corresponding offsets
        packElemSendBufOff = nElemsPerProc * owningProc;
        packElemDataBufOff = Loc2LinearLoc(localLoc, localShape);

//        printf("        packElemSendBufOff: %d\n", packElemSendBufOff);
//        printf("        packElemDataBufOff: %d\n", packElemDataBufOff);
        //Now that we have figured out the starting point, begin copying the entire slice from this element
        for(outerSliceNum = 0; outerSliceNum < nMaxOuterSlices; outerSliceNum++){
            if(outerSliceNum >= nLocalOuterSlices)
                break;
            outerSendBufOff = copySliceSize * nPackA2AMode1Slices * nMaxMidSlices * nPackA2AMode2Slices * outerSliceNum;
            outerDataBufOff = copySliceSize * nLocalA2AMode1Slices * nLocalMidSlices * nLocalA2AMode2Slices * outerSliceNum;

//            printf("        outerSliceNum: %d\n", outerSliceNum);
//            printf("        outerSendBufOff: %d\n", outerSendBufOff);
//            printf("        outerDataBufOff: %d\n", outerDataBufOff);
            for(a2aMode2SliceNum = 0; a2aMode2SliceNum < nMaxA2AMode2Slices; a2aMode2SliceNum += a2aMode2PackStride){
                if(a2aMode2SliceNum >= nLocalA2AMode2Slices)
                    break;
                a2aMode2SendBufOff = copySliceSize * nPackA2AMode1Slices * nMaxMidSlices * (a2aMode2SliceNum / a2aMode2PackStride);
                a2aMode2DataBufOff = copySliceSize * nLocalA2AMode1Slices * nLocalMidSlices * a2aMode2SliceNum;

//                printf("        a2aMode2SliceNum: %d\n", a2aMode2SliceNum);
//                printf("        a2aMode2SendBufOff: %d\n", a2aMode2SendBufOff);
//                printf("        a2aMode2DataBufOff: %d\n", a2aMode2DataBufOff);
                for(midSliceNum = 0; midSliceNum < nMaxMidSlices; midSliceNum++){
                    if(midSliceNum >= nLocalMidSlices)
                        break;
                    midSendBufOff = copySliceSize * nPackA2AMode1Slices * midSliceNum;
                    midDataBufOff = copySliceSize * nLocalA2AMode1Slices * midSliceNum;

//                    printf("        midSliceNum: %d\n", midSliceNum);
//                    printf("        midSendBufOff: %d\n", midSendBufOff);
//                    printf("        midDataBufOff: %d\n", midDataBufOff);
                    for(a2aMode1SliceNum = 0; a2aMode1SliceNum < nMaxA2AMode1Slices; a2aMode1SliceNum += a2aMode1PackStride){
                        if(a2aMode1SliceNum >= nLocalA2AMode1Slices)
                            break;
                        a2aMode1SendBufOff = copySliceSize * (a2aMode1SliceNum / a2aMode1PackStride);
                        a2aMode1DataBufOff = copySliceSize * a2aMode1SliceNum;

//                        printf("        a2aMode1SliceNum: %d\n", a2aMode1SliceNum);
//                        printf("        a2aMode1SendBufOff: %d\n", a2aMode1SendBufOff);
//                        printf("        a2aMode1DataBufOff: %d\n", a2aMode1DataBufOff);
                        //Down to all contiguous slices, so just copy

                        startSendBuf = packElemSendBufOff + outerSendBufOff + a2aMode2SendBufOff + midSendBufOff + a2aMode1SendBufOff;
                        startDataBuf = packElemDataBufOff + outerDataBufOff + a2aMode2DataBufOff + midDataBufOff + a2aMode1DataBufOff;

//                        printf("        startSendBuf: %d\n", startSendBuf);
//                        printf("        startDataBuf: %d\n", startDataBuf);
                        MemCopy(&(sendBuf[startSendBuf]), &(dataBuf[startDataBuf]), copySliceSize);
                    }
                }
            }
        }
    }

    const ObjShape commGridSlice = FilterVector(this->Grid().Shape(), commModes);
    //const Unsigned nRedistProcs = prod(commGridSlice);

//    printf("packed sendBuf: ");
//    for(Unsigned i = 0; i < prod(packLocalShape) * nRedistProcs; i++)
//        printf("%d ", sendBuf[i]);
//    printf("\n");
}

template<typename T>
void DistTensor<T>::UnpackA2ADoubleModeCommRecvBuf(const T * const recvBuf, const std::pair<Mode, Mode>& a2aModes, const std::pair<ModeArray, ModeArray >& commGroups, const DistTensor<T>& A){
    Unsigned i;
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

    ModeArray commModes  = commGroup1;
    commModes.insert(commModes.end(), commGroup2.begin(), commGroup2.end());

    ModeArray nonCommModes;
    for(i = 0; i < g.Order(); i++){
        if(std::find(commModes.begin(), commModes.end(), i) == commModes.end()){
            nonCommModes.push_back(i);
        }
    }

    ObjShape tensorShape = this->Shape();
    Location myGridLoc = g.Loc();
    ObjShape gridShape = g.Shape();

    std::vector<Unsigned> modeLCMs(order);
    for(Unsigned i = 0; i < order; i++)
        modeLCMs[i] = tmen::LCM(gvA.ModeWrapStride(i), gvB.ModeWrapStride(i));

    //Stride taken to unpack into databuf per mode
    std::vector<Unsigned> modeUnpackStrides(order);
    for(Unsigned i = 0; i < order; i++)
        modeUnpackStrides[i] = modeLCMs[i] / gvB.ModeWrapStride(i);

    std::vector<Unsigned> modePackStrides(order);
    for(Unsigned i = 0; i < order; i++)
        modePackStrides[i] = modeLCMs[i] / gvA.ModeWrapStride(i);

    const ObjShape localShape = this->LocalShape();
    ObjShape packedLocalShape = MaxLengths(A.Shape(), gvA.Shape());

    //Slices of a2aMode1
    const Unsigned nLocalA2AMode1Slices = localShape[a2aMode1];

    //Slices between a2aMode1 and a2aMode2
    const Unsigned nLocalMidSlices = Max(1, prod(localShape, a2aMode1 + 1, a2aMode2));

    //Slices of a2aMode2
    const Unsigned nLocalA2AMode2Slices = localShape[a2aMode2];

    const Unsigned copySliceSize = this->LocalModeStride(a2aMode1);
    const Unsigned nElemsPerProc = prod(packedLocalShape);

    //Various counters used to offset in data arrays
    Unsigned a2aMode1SliceNum, midSliceNum, a2aMode2SliceNum, outerSliceNum;  //Which slice we are packing for indexK
    Unsigned a2aMode1RecvBufOff, midRecvBufOff, a2aMode2RecvBufOff, outerRecvBufOff;  //Offsets used to index into data arrays
    Unsigned a2aMode1DataBufOff, midDataBufOff, a2aMode2DataBufOff, outerDataBufOff;  //Offsets used to index into data arrays
    Unsigned unpackElemRecvBufOff, unpackElemDataBufOff;
    Unsigned startRecvBuf, startDataBuf;

    const Unsigned a2aMode1UnpackStride = modeUnpackStrides[a2aMode1];
    const Unsigned a2aMode2UnpackStride = modeUnpackStrides[a2aMode2];

    const Unsigned a2aMode1PackStride = modePackStrides[a2aMode1];
    const Unsigned a2aMode2PackStride = modePackStrides[a2aMode2];

    Location myFirstLoc = this->ModeShifts();

    Unsigned unpackElemNum;
    const Unsigned nUnpackElems = prod(modeUnpackStrides);

    for(unpackElemNum = 0; unpackElemNum < nUnpackElems; unpackElemNum++){
        Location unpackElemMultiLoc = LinearLoc2Loc(unpackElemNum, modeUnpackStrides);

        //Determine the global index of this first element we are packing
        Location startUnpackElemLoc = myFirstLoc;
        for(Unsigned i = 0; i < order; i++){
            startUnpackElemLoc[i] += unpackElemMultiLoc[i] * gvB.ModeWrapStride(i);
        }

        //If we run over the edge, don't try to unpack the global element
        if(AnyElemwiseGreaterThanEqualTo(startUnpackElemLoc, this->Shape()))
            continue;

        //Determine the Multiloc of the process that sent this element
        Location owningProcGVA = A.DetermineOwner(startUnpackElemLoc);
        Location owningProcG = GridViewLoc2GridLoc(owningProcGVA, gvA);
        Unsigned owningProc = Loc2LinearLoc(FilterVector(owningProcG, commModes), FilterVector(gridShape, commModes));

        //Find the local location of the global starting element we are now unpacking
        Location localLoc = this->Global2LocalIndex(startUnpackElemLoc);

        //Now that we know the local loc of the element to unpack, we know how many iterations of unpacking to perform per mode
        const ObjShape tensorShape = this->Shape();
        const ObjShape gvAShape = gvA.Shape();

        const ObjShape outerSliceShape(tensorShape.begin() + a2aMode2 + 1, tensorShape.end());
        const Location outerSliceMultiLoc(startUnpackElemLoc.begin() + a2aMode2 + 1, startUnpackElemLoc.end());
        const ObjShape outerSlicePackShape(gvAShape.begin() + a2aMode2 + 1, gvAShape.end());
        std::vector<Unsigned> nPackedOuterWraps(outerSliceShape.size());
        for(i = 0; i < nPackedOuterWraps.size(); i++){
            nPackedOuterWraps[i] = (outerSliceShape[i] - outerSliceMultiLoc[i] - 1) / outerSlicePackShape[i] + 1;
        }
        const Unsigned nPackedOuterSlices = Max(1, prod(nPackedOuterWraps));

        const Unsigned nMaxPackedA2AMode2Slices = Max(1, (tensorShape[a2aMode2] - 1) / gvAShape[a2aMode2] / a2aMode2PackStride + 1);
        const Unsigned nPackedA2AMode2Slices = Max(1, (tensorShape[a2aMode2] - startUnpackElemLoc[a2aMode2] - 1) / gvAShape[a2aMode2] / a2aMode2PackStride + 1);

        const ObjShape midSliceShape(tensorShape.begin() + a2aMode1 + 1, tensorShape.begin() + a2aMode2);
        const Location midSliceMultiLoc(startUnpackElemLoc.begin() + a2aMode1 + 1, startUnpackElemLoc.begin() + a2aMode2);
        const ObjShape midSlicePackShape(gvAShape.begin() + a2aMode1 + 1, gvAShape.begin() + a2aMode2);
        std::vector<Unsigned> nMaxPackedMidWraps(midSliceShape.size());
        std::vector<Unsigned> nPackedMidWraps(midSliceShape.size());
        for(i = 0; i < nPackedMidWraps.size(); i++){
            nPackedMidWraps[i] = (midSliceShape[i] - midSliceMultiLoc[i] - 1) / midSlicePackShape[i] + 1;
            nMaxPackedMidWraps[i] = (midSliceShape[i] - 1) / midSlicePackShape[i] + 1;
        }
        const Unsigned nMaxPackedMidSlices = Max(1, prod(nMaxPackedMidWraps));
        const Unsigned nPackedMidSlices = Max(1, prod(nPackedMidWraps));

        const Unsigned nMaxPackedA2AMode1Slices = Max(1, (tensorShape[a2aMode1] - 1) / gvAShape[a2aMode1] / a2aMode1PackStride + 1);
        const Unsigned nPackedA2AMode1Slices = Max(1, (tensorShape[a2aMode1] - startUnpackElemLoc[a2aMode1] - 1) / gvAShape[a2aMode1] / a2aMode1PackStride + 1);

        //Update the corresponding offsets
        unpackElemRecvBufOff = nElemsPerProc * owningProc;
        unpackElemDataBufOff = Loc2LinearLoc(localLoc, localShape);


//        printf("MemCopy info:\n");
//        printf("    unpackElemRecvBufOff: %d\n", unpackElemRecvBufOff);
//        printf("    unpackElemDataBufOff: %d\n", unpackElemDataBufOff);
//        printf("    nPackElems: %d\n", nUnpackElems);
//        printf("    nPackedOuterSlices: %d\n", nPackedOuterSlices);
//        printf("    nPackedA2AMode2Slices: %d\n", nPackedA2AMode2Slices);
//        printf("    nPackedMidSlices: %d\n", nPackedMidSlices);
//        printf("    nPackedA2AMode1Slices: %d\n", nPackedA2AMode1Slices);
//        printf("    copySliceSize: %d\n", copySliceSize);
        //Now that we have figured out the starting point, begin copying the entire slice from this element
        for(outerSliceNum = 0; outerSliceNum < nPackedOuterSlices; outerSliceNum++){

            outerRecvBufOff = copySliceSize * nMaxPackedA2AMode1Slices * nMaxPackedMidSlices * nMaxPackedA2AMode2Slices * outerSliceNum;
            outerDataBufOff = copySliceSize * nLocalA2AMode1Slices * nLocalMidSlices * nLocalA2AMode2Slices * outerSliceNum;

//            printf("        outerSliceNum: %d\n", outerSliceNum);
//            printf("        outerRecvBufOff: %d\n", outerRecvBufOff);
//            printf("        outerDataBufOff: %d\n", outerDataBufOff);
            for(a2aMode2SliceNum = 0; a2aMode2SliceNum < nPackedA2AMode2Slices; a2aMode2SliceNum++){

                a2aMode2RecvBufOff = copySliceSize * nMaxPackedA2AMode1Slices * nMaxPackedMidSlices * a2aMode2SliceNum;
                a2aMode2DataBufOff = copySliceSize * nLocalA2AMode1Slices * nLocalMidSlices * (a2aMode2SliceNum * a2aMode2UnpackStride);

//                printf("        a2aMode2SliceNum: %d\n", a2aMode2SliceNum);
//                printf("        a2aMode2RecvBufOff: %d\n", a2aMode2RecvBufOff);
//                printf("        a2aMode2DataBufOff: %d\n", a2aMode2DataBufOff);
                for(midSliceNum = 0; midSliceNum < nPackedMidSlices; midSliceNum++){
                    midRecvBufOff = copySliceSize * nMaxPackedA2AMode1Slices * midSliceNum;
                    midDataBufOff = copySliceSize * nLocalA2AMode1Slices * midSliceNum;

//                    printf("        midSliceNum: %d\n", midSliceNum);
//                    printf("        midRecvBufOff: %d\n", midRecvBufOff);
//                    printf("        midDataBufOff: %d\n", midDataBufOff);
                    for(a2aMode1SliceNum = 0; a2aMode1SliceNum < nPackedA2AMode1Slices; a2aMode1SliceNum++){
                        a2aMode1RecvBufOff = copySliceSize * a2aMode1SliceNum;
                        a2aMode1DataBufOff = copySliceSize * (a2aMode1SliceNum * a2aMode1UnpackStride);

//                        printf("        a2aMode1SliceNum: %d\n", a2aMode1SliceNum);
//                        printf("        a2aMode1RecvBufOff: %d\n", a2aMode1RecvBufOff);
//                        printf("        a2aMode1DataBufOff: %d\n", a2aMode1DataBufOff);
                        //Down to all contiguous slices, so just copy
                        startRecvBuf = unpackElemRecvBufOff + outerRecvBufOff + a2aMode2RecvBufOff + midRecvBufOff + a2aMode1RecvBufOff;
                        startDataBuf = unpackElemDataBufOff + outerDataBufOff + a2aMode2DataBufOff + midDataBufOff + a2aMode1DataBufOff;

//                        printf("          startRecvBuf: %d\n", startRecvBuf);
//                        printf("          startDataBuf: %d\n", startDataBuf);
                        MemCopy(&(dataBuf[startDataBuf]), &(recvBuf[startRecvBuf]), copySliceSize);

                    }
                }
            }
        }
    }
//    printf("dataBuf:");
//    for(Unsigned i = 0; i < prod(B.LocalShape()); i++)
//        printf(" %d", dataBuf[i]);
//    printf("\n");
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
