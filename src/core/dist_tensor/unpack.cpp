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

template <typename T>
void UnpackPermutationRecvBuf(const T * const recvBuf, const Int permuteIndex, const DistTensor<T>& A, DistTensor<T>& B)
{
        const std::vector<Int> start(B.Order(), 0);
        T* dataBuf = B.Buffer(start);

        const int pModeB = B.ModeOfIndex(permuteIndex);

        const tmen::GridView gvA = A.GridView();
        const tmen::GridView gvB = B.GridView();

        const std::vector<Int> maxLocalShapeA = MaxLengths(A.Shape(), gvA.Shape());
        const std::vector<Int> maxLocalShapeB = MaxLengths(B.Shape(), gvB.Shape());

        const std::vector<Int> localShapeB = B.LocalShape();         //Shape of the local tensor we are packing

        //Number of outer slices to unpack
        const int nMaxOuterSlices = Max(1, prod(maxLocalShapeB, pModeB + 1));
        const int nLocalOuterSlices = prod(localShapeB, pModeB + 1);

        //Loop packing bounds variables
        const int nMaxPModeSlices = maxLocalShapeB[pModeB];
        const int nLocalPModeSlices = localShapeB[pModeB];

        //Variables for calculating elements to copy
        const int maxCopySliceSize = Max(1, prod(maxLocalShapeB, 0, pModeB));
        const int copySliceSize = B.LocalModeStride(pModeB);

        //Loop iteration vars
        int outerSliceNum, pModeSliceNum;  //Pack data for slice "sliceNum" (<nSlices) of wrap "wrapNum" (<nWraps) for proc "procSendNum" int offSliceRecvBuf, offWrapRecvBuf;  //Offsets used to index into sendBuf array
        int outerRecvBufOff, outerDataBufOff;  //Offsets used to index into recvBuf array
        int pModeRecvBufOff, pModeDataBufOff;  //Offsets used to index into dataBuf array
        int startRecvBuf, startDataBuf;

        for(outerSliceNum = 0; outerSliceNum < nMaxOuterSlices; outerSliceNum++){
            if(outerSliceNum >= nLocalOuterSlices)
                break;

            outerRecvBufOff = maxCopySliceSize * nMaxPModeSlices * outerSliceNum;
            outerDataBufOff = copySliceSize * nLocalPModeSlices * outerSliceNum;

            for(pModeSliceNum = 0; pModeSliceNum < nMaxPModeSlices; pModeSliceNum++){
                if(pModeSliceNum >= nLocalPModeSlices)
                    break;
                pModeRecvBufOff = maxCopySliceSize * pModeSliceNum;
                pModeDataBufOff = copySliceSize * pModeSliceNum;

                startRecvBuf = outerRecvBufOff + pModeRecvBufOff;
                startDataBuf = outerDataBufOff + pModeDataBufOff;
                //printf("startRecvBuf: %d startDataBuf: %d copySliceSize: %d\n", startRecvBuf, startDataBuf, copySliceSize);
                MemCopy(&(dataBuf[startDataBuf]), &(recvBuf[startRecvBuf]), copySliceSize);
            }
        }
}

//NOTE: Should be equivalent to UnpackRSRecvBuf(recvBuf, reduceScatterIndex, reduceScatterIndex, A, B);
template <typename T>
void UnpackPartialRSRecvBuf(const T * const recvBuf, const Int reduceScatterIndex, const DistTensor<T>& A, DistTensor<T>& B)
{
    const std::vector<Int> start(B.Order(), 0);
    T* dataBuf = B.Buffer(start);

    const int rsModeB = B.ModeOfIndex(reduceScatterIndex);

    const tmen::GridView gvA = A.GridView();
    const tmen::GridView gvB = B.GridView();

    const std::vector<Int> maxLocalShapeA = MaxLengths(A.Shape(), gvA.Shape());
    const std::vector<Int> maxLocalShapeB = MaxLengths(B.Shape(), gvB.Shape());

//    printf("recvBuf:");
//    for(int i = 0; i < prod(maxLocalShapeA) / nRedistProcs; i++){
//        printf(" %d", recvBuf[i]);
//    }
//    printf("\n");

    const std::vector<Int> localShapeB = B.LocalShape();         //Shape of the local tensor we are packing

    //Number of outer slices to unpack
    const int nMaxOuterSlices = Max(1, prod(maxLocalShapeB, rsModeB + 1));
    const int nLocalOuterSlices = prod(localShapeB, rsModeB + 1);

    //Loop packing bounds variables
    const int nMaxRSModeSlices = maxLocalShapeB[rsModeB];
    const int nLocalRSModeSlices = localShapeB[rsModeB];

    //Each wrap is copied contiguously because the distribution of reduceScatter index does not change

    //Variables for calculating elements to copy
    const int maxCopySliceSize = Max(1, prod(maxLocalShapeB, 0, rsModeB));
    const int copySliceSize = B.LocalModeStride(rsModeB);

    //Loop iteration vars
    int outerSliceNum, rsModeSliceNum;  //Pack data for slice "sliceNum" (<nSlices) of wrap "wrapNum" (<nWraps) for proc "procSendNum" int offSliceRecvBuf, offWrapRecvBuf;  //Offsets used to index into sendBuf array
    int outerRecvBufOff, outerDataBufOff;  //Offsets used to index into recvBuf array
    int rsModeRecvBufOff, rsModeDataBufOff;  //Offsets used to index into dataBuf array
    int startRecvBuf, startDataBuf;

//    printf("MemCopy info:\n");
//    printf("    nMaxOuterSlices: %d\n", nMaxOuterSlices);
//    printf("    nMaxRSModeSlices: %d\n", nMaxRSModeSlices);
//    printf("    maxCopySliceSize: %d\n", maxCopySliceSize);
//    printf("    copySliceSize: %d\n", copySliceSize);
    for(outerSliceNum = 0; outerSliceNum < nMaxOuterSlices; outerSliceNum++){
        if(outerSliceNum >= nLocalOuterSlices)
            break;
        outerRecvBufOff = maxCopySliceSize * nMaxRSModeSlices * outerSliceNum;
        outerDataBufOff = copySliceSize * nLocalRSModeSlices * outerSliceNum;

//        printf("        outerSliceNum: %d\n", outerSliceNum);
//        printf("        outerRecvBufOff: %d\n", outerRecvBufOff);
//        printf("        outerDataBufOff: %d\n", outerDataBufOff);

        for(rsModeSliceNum = 0; rsModeSliceNum < nMaxRSModeSlices; rsModeSliceNum++){
            if(rsModeSliceNum >= nLocalRSModeSlices)
                break;

            rsModeRecvBufOff = (maxCopySliceSize * rsModeSliceNum);
            rsModeDataBufOff = (copySliceSize * rsModeSliceNum);

            startRecvBuf = outerRecvBufOff + rsModeRecvBufOff;
            startDataBuf = outerDataBufOff + rsModeDataBufOff;

//            printf("          startRecvBuf: %d\n", startRecvBuf);
//            printf("          startDataBuf: %d\n", startDataBuf);
            MemCopy(&(dataBuf[startDataBuf]), &(recvBuf[startRecvBuf]), copySliceSize);
        }
    }

//    printf("dataBuf:");
//    for(int i = 0; i < prod(B.LocalShape()); i++)
//        printf(" %d", dataBuf[i]);
//    printf("\n");
}

//Only called when fully reducing an index
//NOTE: Looks an awful lot like PackAGSendBuf...
template <typename T>
void UnpackRSRecvBuf(const T * const recvBuf, const Int reduceIndex, const Int scatterIndex, const DistTensor<T>& A, DistTensor<T>& B)
{
    const std::vector<Int> start(B.Order(), 0);
    T* dataBuf = B.Buffer(start);

    const int sModeB = B.ModeOfIndex(scatterIndex);

    const tmen::GridView gvA = A.GridView();
    const tmen::GridView gvB = B.GridView();

    const std::vector<Int> maxLocalShapeA = MaxLengths(A.Shape(), gvA.Shape());
    const std::vector<Int> maxLocalShapeB = MaxLengths(B.Shape(), gvB.Shape());

//    printf("recvBuf:");
//    for(int i = 0; i < prod(maxLocalShapeA) / nRedistProcs; i++){
//        printf(" %d", recvBuf[i]);
//    }
//    printf("\n");

    const std::vector<Int> localShapeB = B.LocalShape();         //Shape of the local tensor we are packing

    //Number of outer slices to unpack
    const int nMaxOuterSlices = Max(1, prod(maxLocalShapeB, sModeB + 1));
    const int nLocalOuterSlices = prod(localShapeB, sModeB + 1);

    //Loop packing bounds variables
    const int nMaxSModeSlices = maxLocalShapeB[sModeB];
    const int nLocalSModeSlices = localShapeB[sModeB];

    //Each wrap is copied contiguously because the distribution of reduceScatter index does not change

    //Variables for calculating elements to copy
    const int maxCopySliceSize = Max(1, prod(maxLocalShapeB, 0, sModeB));
    const int copySliceSize = B.LocalModeStride(sModeB);

    //Loop iteration vars
    int outerSliceNum, sModeSliceNum;  //Pack data for slice "sliceNum" (<nSlices) of wrap "wrapNum" (<nWraps) for proc "procSendNum" int offSliceRecvBuf, offWrapRecvBuf;  //Offsets used to index into sendBuf array
    int outerRecvBufOff, outerDataBufOff;  //Offsets used to index into recvBuf array
    int sModeRecvBufOff, sModeDataBufOff;  //Offsets used to index into dataBuf array
    int startRecvBuf, startDataBuf;

//    printf("MemCopy info:\n");
//    printf("    nMaxOuterSlices: %d\n", nMaxOuterSlices);
//    printf("    nMaxSModeSlices: %d\n", nMaxSModeSlices);
//    printf("    maxCopySliceSize: %d\n", maxCopySliceSize);
//    printf("    copySliceSize: %d\n", copySliceSize);
    for(outerSliceNum = 0; outerSliceNum < nMaxOuterSlices; outerSliceNum++){
        if(outerSliceNum >= nLocalOuterSlices)
            break;
        outerRecvBufOff = maxCopySliceSize * nMaxSModeSlices * outerSliceNum;
        outerDataBufOff = copySliceSize * nLocalSModeSlices * outerSliceNum;

//        printf("        outerSliceNum: %d\n", outerSliceNum);
//        printf("        outerRecvBufOff: %d\n", outerRecvBufOff);
//        printf("        outerDataBufOff: %d\n", outerDataBufOff);

        for(sModeSliceNum = 0; sModeSliceNum < nMaxSModeSlices; sModeSliceNum++){
            if(sModeSliceNum >= nLocalSModeSlices)
                break;

            sModeRecvBufOff = (maxCopySliceSize * sModeSliceNum);
            sModeDataBufOff = (copySliceSize * sModeSliceNum);

            startRecvBuf = outerRecvBufOff + sModeRecvBufOff;
            startDataBuf = outerDataBufOff + sModeDataBufOff;

//            printf("          startRecvBuf: %d\n", startRecvBuf);
//            printf("          startDataBuf: %d\n", startDataBuf);
            MemCopy(&(dataBuf[startDataBuf]), &(recvBuf[startRecvBuf]), copySliceSize);
        }
    }

//    printf("dataBuf:");
//    for(int i = 0; i < prod(B.LocalShape()); i++)
//        printf(" %d", dataBuf[i]);
//    printf("\n");
}

template <typename T>
void UnpackAGRecvBuf(const T * const recvBuf, const Int allGatherIndex, const DistTensor<T>& A, DistTensor<T>& B)
{
    const std::vector<Int> start(B.Order(), 0);
    T* dataBuf = B.Buffer(start);

    const int agModeA = A.ModeOfIndex(allGatherIndex);
    const int agModeB = B.ModeOfIndex(allGatherIndex);

    const tmen::GridView gvA = A.GridView();
    const tmen::GridView gvB = B.GridView();

    const int nRedistProcs = gvA.Dimension(agModeA);

    const std::vector<Int> maxLocalShapeA = MaxLengths(A.Shape(), gvA.Shape());
    const std::vector<Int> maxLocalShapeB = MaxLengths(B.Shape(), gvB.Shape());

//    printf("recvBuf:");
//    for(int i = 0; i < prod(maxLocalShapeA) * nRedistProcs; i++){
//        printf(" %d", recvBuf[i]);
//    }
//    printf("\n");

    const std::vector<Int> localShapeB = B.LocalShape();

    //Number of outer slices to unpack
    const int nMaxOuterSlices = Max(1, prod(maxLocalShapeB, agModeB + 1));
    const int nLocalOuterSlices = prod(localShapeB, agModeB + 1);

    //Loop packing bounds variables
    const int nMaxAGModeSlices = maxLocalShapeB[agModeB];
    const int nLocalAGModeSlices = localShapeB[agModeB];
    const int agModeUnpackStride = nRedistProcs;

    //Variables for calculating elements to copy
    const int maxCopySliceSize = Max(1, prod(maxLocalShapeB, 0, agModeB));
    const int copySliceSize = B.LocalModeStride(agModeB);

    //Number of processes we have to unpack from
    const int nElemSlices = nRedistProcs;

    //Loop iteration vars
    int outerSliceNum, agModeSliceNum, elemSliceNum;  //Pack data for slice "sliceNum" (<nSlices) of wrap "wrapNum" (<nWraps) for proc "procSendNum" int offSliceRecvBuf, offWrapRecvBuf;  //Offsets used to index into sendBuf array
    int elemRecvBufOff, elemDataBufOff;
    int outerRecvBufOff, outerDataBufOff;  //Offsets used to index into recvBuf array
    int agModeRecvBufOff, agModeDataBufOff;  //Offsets used to index into dataBuf array
    int startRecvBuf, startDataBuf;

//    printf("MemCopy info:\n");
//    printf("    nMaxOuterSlices: %d\n", nMaxOuterSlices);
//    printf("    nMaxAGModeSlices: %d\n", nMaxAGModeSlices);
//    printf("    maxCopySliceSize: %d\n", maxCopySliceSize);
//    printf("    copySliceSize: %d\n", copySliceSize);
//    printf("    agModeUnpackStride: %d\n", agModeUnpackStride);
    for(elemSliceNum = 0; elemSliceNum < nElemSlices; elemSliceNum++){
        elemRecvBufOff = prod(maxLocalShapeA) * elemSliceNum;
        elemDataBufOff = copySliceSize * elemSliceNum;

//        printf("      elemSliceNum: %d\n", elemSliceNum);
//        printf("      elemRecvBufOff: %d\n", elemRecvBufOff);
//        printf("      elemDataBufOff: %d\n", elemDataBufOff);
        for(outerSliceNum = 0; outerSliceNum < nMaxOuterSlices; outerSliceNum++){
            if(outerSliceNum >= nLocalOuterSlices)
                break;
            //NOTE: the weird Max() function ensures we increment the recvBuf correctly
            //e.g. we need to ensure that we jump over all slices packed by the pack routine.  Which should be maxLocalShapeA[agModeA];
            //For consistency, kept same structure as in PackPartialRSSendBuf
            outerRecvBufOff = maxCopySliceSize * Max(1, (nMaxAGModeSlices - 1) / agModeUnpackStride + 1) * outerSliceNum;
            outerDataBufOff = copySliceSize * nLocalAGModeSlices * outerSliceNum;

//            printf("        outerSliceNum: %d\n", outerSliceNum);
//            printf("        outerRecvBufOff: %d\n", outerRecvBufOff);
//            printf("        outerDataBufOff: %d\n", outerDataBufOff);
            for(agModeSliceNum = 0; agModeSliceNum < nMaxAGModeSlices; agModeSliceNum += agModeUnpackStride){
                if(agModeSliceNum + elemSliceNum >= nLocalAGModeSlices)
                    break;
                agModeRecvBufOff = maxCopySliceSize * (agModeSliceNum / agModeUnpackStride);
                agModeDataBufOff = copySliceSize * agModeSliceNum;

//                printf("          agModeSliceNum: %d\n", agModeSliceNum);
//                printf("          agModeRecvBufOff: %d\n", agModeRecvBufOff);
//                printf("          agModeDataBufOff: %d\n", agModeDataBufOff);
                startRecvBuf = elemRecvBufOff + outerRecvBufOff + agModeRecvBufOff;
                startDataBuf = elemDataBufOff + outerDataBufOff + agModeDataBufOff;

//                printf("          startRecvBuf: %d\n", startRecvBuf);
//                printf("          startDataBuf: %d\n", startDataBuf);
                MemCopy(&(dataBuf[startDataBuf]), &(recvBuf[startRecvBuf]), copySliceSize);
            }
        }
    }
//    printf("dataBuf:");
//    for(int i = 0; i < prod(B.LocalShape()); i++)
//        printf(" %d", dataBuf[i]);
//    printf("\n");
}

template<typename T>
void UnpackA2ADoubleIndexRecvBuf(const T * const recvBuf, const std::pair<int, int>& a2aIndices, const std::pair<std::vector<int>, std::vector<int> >& commGroups, const DistTensor<T>& A, DistTensor<T>& B){
    const int order = A.Order();
    const std::vector<int> start(order, 0);
    T* dataBuf = B.Buffer(start);

    const tmen::GridView gvA = A.GridView();
    const tmen::GridView gvB = B.GridView();

    const tmen::Grid& g = A.Grid();

    int a2aMode1 = A.ModeOfIndex(a2aIndices.first);
    int a2aMode2 = A.ModeOfIndex(a2aIndices.second);

    std::vector<int> commGroup1 = commGroups.first;
    std::vector<int> commGroup2 = commGroups.second;

    //For convenience make sure that a2aMode1 is earlier in the packing
    if(a2aMode1 > a2aMode2){
        std::swap(a2aMode1, a2aMode2);
        std::swap(commGroup1, commGroup2);
    }

    std::vector<int> commModes  = commGroup1;
    commModes.insert(commModes.end(), commGroup2.begin(), commGroup2.end());

    std::vector<int> nonCommModes;
    for(int i = 0; i < g.Order(); i++){
        if(std::find(commModes.begin(), commModes.end(), i) == commModes.end()){
            nonCommModes.push_back(i);
        }
    }

    std::vector<int> tensorShape = B.Shape();
    std::vector<int> myGridLoc = g.Loc();
    std::vector<int> gridShape = g.Shape();

    std::vector<int> modeLCMs(order);
    for(int i = 0; i < order; i++)
    	modeLCMs[i] = tmen::LCM(gvA.ModeWrapStride(i), gvB.ModeWrapStride(i));

    //Stride taken to unpack into databuf per mode
    std::vector<int> modeUnpackStrides(order);
    for(int i = 0; i < order; i++)
    	modeUnpackStrides[i] = modeLCMs[i] / gvB.ModeWrapStride(i);

    std::vector<int> modePackStrides(order);
    for(int i = 0; i < order; i++)
        modePackStrides[i] = modeLCMs[i] / gvA.ModeWrapStride(i);

    const std::vector<Int> localShape = B.LocalShape();
    std::vector<Int> packedLocalShape = MaxLengths(A.Shape(), gvA.Shape());

    //Slices of a2aMode1
    const int nLocalA2AMode1Slices = localShape[a2aMode1];

    //Slices between a2aMode1 and a2aMode2
    const int nLocalMidSlices = Max(1, prod(localShape, a2aMode1 + 1, a2aMode2));

    //Slices of a2aMode2
    const int nLocalA2AMode2Slices = localShape[a2aMode2];

    const int copySliceSize = B.LocalModeStride(a2aMode1);
    const int nElemsPerProc = prod(packedLocalShape);

    //Various counters used to offset in data arrays
    int a2aMode1SliceNum, midSliceNum, a2aMode2SliceNum, outerSliceNum;  //Which slice we are packing for indexK
    int a2aMode1RecvBufOff, midRecvBufOff, a2aMode2RecvBufOff, outerRecvBufOff;  //Offsets used to index into data arrays
    int a2aMode1DataBufOff, midDataBufOff, a2aMode2DataBufOff, outerDataBufOff;  //Offsets used to index into data arrays
    int unpackElemRecvBufOff, unpackElemDataBufOff;
    int startRecvBuf, startDataBuf;

    const int a2aMode1UnpackStride = modeUnpackStrides[a2aMode1];
    const int a2aMode2UnpackStride = modeUnpackStrides[a2aMode2];

    const int a2aMode1PackStride = modePackStrides[a2aMode1];
    const int a2aMode2PackStride = modePackStrides[a2aMode2];

    std::vector<int> myFirstLoc = B.ModeShifts();

    int unpackElemNum;
    const int nUnpackElems = prod(modeUnpackStrides);

    for(unpackElemNum = 0; unpackElemNum < nUnpackElems; unpackElemNum++){
        std::vector<int> unpackElemMultiLoc = LinearLoc2Loc(unpackElemNum, modeUnpackStrides);

        //Determine the global index of this first element we are packing
        std::vector<int> startUnpackElemLoc = myFirstLoc;
        for(int i = 0; i < order; i++){
            startUnpackElemLoc[i] += unpackElemMultiLoc[i] * gvB.ModeWrapStride(i);
        }

        //If we run over the edge, don't try to unpack the global element
        if(AnyElemwiseGreaterThanEqualTo(startUnpackElemLoc, B.Shape()))
            continue;

        //Determine the Multiloc of the process that sent this element
        std::vector<int> owningProcGVA = A.DetermineOwner(startUnpackElemLoc);
        std::vector<int> owningProcG = GridViewLoc2GridLoc(owningProcGVA, gvA);
        int owningProc = LinearIndex(FilterVector(owningProcG, commModes), Dimensions2Strides(FilterVector(gridShape, commModes)));

        //Find the local location of the global starting element we are now unpacking
        std::vector<int> localLoc = B.Global2LocalIndex(startUnpackElemLoc);

        //Now that we know the local loc of the element to unpack, we know how many iterations of unpacking to perform per mode
        const std::vector<int> tensorShape = B.Shape();
        const std::vector<int> gvAShape = gvA.Shape();

        const std::vector<int> outerSliceShape(tensorShape.begin() + a2aMode2 + 1, tensorShape.end());
        const std::vector<int> outerSliceMultiLoc(startUnpackElemLoc.begin() + a2aMode2 + 1, startUnpackElemLoc.end());
        const std::vector<int> outerSlicePackShape(gvAShape.begin() + a2aMode2 + 1, gvAShape.end());
        std::vector<int> nPackedOuterWraps(outerSliceShape.size());
        for(int i = 0; i < nPackedOuterWraps.size(); i++){
            nPackedOuterWraps[i] = (outerSliceShape[i] - outerSliceMultiLoc[i] - 1) / outerSlicePackShape[i] + 1;
        }
        const int nPackedOuterSlices = Max(1, prod(nPackedOuterWraps));

        const int nMaxPackedA2AMode2Slices = Max(1, (tensorShape[a2aMode2] - 1) / gvAShape[a2aMode2] / a2aMode2PackStride + 1);
        const int nPackedA2AMode2Slices = Max(1, (tensorShape[a2aMode2] - startUnpackElemLoc[a2aMode2] - 1) / gvAShape[a2aMode2] / a2aMode2PackStride + 1);

        const std::vector<int> midSliceShape(tensorShape.begin() + a2aMode1 + 1, tensorShape.begin() + a2aMode2);
        const std::vector<int> midSliceMultiLoc(startUnpackElemLoc.begin() + a2aMode1 + 1, startUnpackElemLoc.begin() + a2aMode2);
        const std::vector<int> midSlicePackShape(gvAShape.begin() + a2aMode1 + 1, gvAShape.begin() + a2aMode2);
        std::vector<int> nMaxPackedMidWraps(midSliceShape.size());
        std::vector<int> nPackedMidWraps(midSliceShape.size());
        for(int i = 0; i < nPackedMidWraps.size(); i++){
            nPackedMidWraps[i] = (midSliceShape[i] - midSliceMultiLoc[i] - 1) / midSlicePackShape[i] + 1;
            nMaxPackedMidWraps[i] = (midSliceShape[i] - 1) / midSlicePackShape[i] + 1;
        }
        const int nMaxPackedMidSlices = Max(1, prod(nMaxPackedMidWraps));
        const int nPackedMidSlices = Max(1, prod(nPackedMidWraps));

        const int nMaxPackedA2AMode1Slices = Max(1, (tensorShape[a2aMode1] - 1) / gvAShape[a2aMode1] / a2aMode1PackStride + 1);
        const int nPackedA2AMode1Slices = Max(1, (tensorShape[a2aMode1] - startUnpackElemLoc[a2aMode1] - 1) / gvAShape[a2aMode1] / a2aMode1PackStride + 1);

        //Update the corresponding offsets
        unpackElemRecvBufOff = nElemsPerProc * owningProc;
        unpackElemDataBufOff = LinearIndex(localLoc, Dimensions2Strides(localShape));

        //Now that we have figured out the starting point, begin copying the entire slice from this element
        for(outerSliceNum = 0; outerSliceNum < nPackedOuterSlices; outerSliceNum++){

            outerRecvBufOff = copySliceSize * nMaxPackedA2AMode1Slices * nMaxPackedMidSlices * nMaxPackedA2AMode2Slices * outerSliceNum;
            outerDataBufOff = copySliceSize * nLocalA2AMode1Slices * nLocalMidSlices * nLocalA2AMode2Slices * outerSliceNum;

            for(a2aMode2SliceNum = 0; a2aMode2SliceNum < nPackedA2AMode2Slices; a2aMode2SliceNum++){

                a2aMode2RecvBufOff = copySliceSize * nMaxPackedA2AMode1Slices * nMaxPackedMidSlices * a2aMode2SliceNum;
                a2aMode2DataBufOff = copySliceSize * nLocalA2AMode1Slices * nLocalMidSlices * (a2aMode2SliceNum * a2aMode2UnpackStride);

                for(midSliceNum = 0; midSliceNum < nPackedMidSlices; midSliceNum++){
                    midRecvBufOff = copySliceSize * nMaxPackedA2AMode1Slices * midSliceNum;
                    midDataBufOff = copySliceSize * nLocalA2AMode1Slices * midSliceNum;

                    for(a2aMode1SliceNum = 0; a2aMode1SliceNum < nPackedA2AMode1Slices; a2aMode1SliceNum++){
                        a2aMode1RecvBufOff = copySliceSize * a2aMode1SliceNum;
                        a2aMode1DataBufOff = copySliceSize * (a2aMode1SliceNum * a2aMode1UnpackStride);

                        //Down to all contiguous slices, so just copy
                        startRecvBuf = unpackElemRecvBufOff + outerRecvBufOff + a2aMode2RecvBufOff + midRecvBufOff + a2aMode1RecvBufOff;
                        startDataBuf = unpackElemDataBufOff + outerDataBufOff + a2aMode2DataBufOff + midDataBufOff + a2aMode1DataBufOff;

                        MemCopy(&(dataBuf[startDataBuf]), &(recvBuf[startRecvBuf]), copySliceSize);

                    }
                }
            }
        }
    }
}

#define PROTO(T) \
        template void UnpackPermutationRecvBuf(const T * const recvBuf, const Int permuteIndex, const DistTensor<T>& A, DistTensor<T>& B); \
        template void UnpackPartialRSRecvBuf(const T * const recvBuf, const Int reduceScatterIndex, const DistTensor<T>& A, DistTensor<T>& B); \
        template void UnpackRSRecvBuf(const T * const recvBuf, const Int reduceIndex, const Int scatterIndex, const DistTensor<T>& A, DistTensor<T>& B); \
        template void UnpackAGRecvBuf(const T * const recvBuf, const Int allGatherIndex, const DistTensor<T>& A, DistTensor<T>& B); \
        template void UnpackA2ADoubleIndexRecvBuf(const T * const recvBuf, const std::pair<int, int>& a2aIndices, const std::pair<std::vector<int>, std::vector<int> >& commGroups, const DistTensor<T>& A, DistTensor<T>& B);

PROTO(int)
PROTO(float)
PROTO(double)

} // namespace tmen
