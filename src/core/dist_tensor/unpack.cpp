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
        const tmen::GridView gv = A.GridView();

        const int permuteModeA = A.ModeOfIndex(permuteIndex);

        const std::vector<Int> maxRecvLocalShape = MaxLengths(A.Shape(), gv.Shape());

        const std::vector<Int> localShape = B.LocalShape();         //Shape of the local tensor we are packing
        const int pModeLocalDim = B.LocalDimension(permuteModeA); //Local version
        const int pModeLocalStride = B.LocalModeStride(permuteModeA);

        //Number of local slices and slice size we must pack per proc per wrap
        const int nLocalSlices = Max(1, prod(localShape, permuteModeA + 1));
        const int nMaxSlices = Max(1, prod(maxRecvLocalShape, permuteModeA + 1));

        //Variables for calculating elements to copy
        const int maxCopySliceSize = Max(1, prod(maxRecvLocalShape, 0, permuteModeA + 1));
        const int copySliceSize = pModeLocalStride * pModeLocalDim;

        //Loop iteration vars
        int sliceNum;  //Pack data for slice "sliceNum" (<nSlices) of wrap "wrapNum" (<nWraps) for proc "procSendNum" int offSliceRecvBuf, offWrapRecvBuf;  //Offsets used to index into sendBuf array
        int offSliceRecvBuf;  //Offsets used to index into recvBuf array
        int offSliceDataBuf;  //Offsets used to index into dataBuf array
        int startRecvBuf, startDataBuf;

        for(sliceNum = 0; sliceNum < nMaxSlices; sliceNum++){
            offSliceRecvBuf = maxCopySliceSize * sliceNum;
            offSliceDataBuf = copySliceSize * sliceNum;
            if(sliceNum >= nLocalSlices){
                break;
            }
            startRecvBuf = offSliceRecvBuf;
            startDataBuf = offSliceDataBuf;
            //printf("startRecvBuf: %d startDataBuf: %d copySliceSize: %d\n", startRecvBuf, startDataBuf, copySliceSize);
            MemCopy(&(dataBuf[startDataBuf]), &(recvBuf[startRecvBuf]), copySliceSize);
        }
}

//NOTE: Should be equivalent to UnpackRSRecvBuf(recvBuf, reduceScatterIndex, reduceScatterIndex, A, B);
template <typename T>
void UnpackPartialRSRecvBuf(const T * const recvBuf, const Int reduceScatterIndex, const DistTensor<T>& A, DistTensor<T>& B)
{
    const std::vector<Int> start(B.Order(), 0);
    T* dataBuf = B.Buffer(start);

    const int rsModeA = A.ModeOfIndex(reduceScatterIndex);
    const int rsModeB = B.ModeOfIndex(reduceScatterIndex);

    const tmen::GridView gvA = A.GridView();
    const tmen::GridView gvB = B.GridView();

    const int nModeProcs = gvB.Dimension(rsModeB);   //In PRS, scatter into same index
    const int rsModeGlobalDimB = B.Dimension(rsModeB); //Number of indices in the mode we are redistributing
    const int rsModeGlobalDimA = A.Dimension(rsModeA); //Number of indices in the mode we are redistributing

    const std::vector<Int> maxLocalShapeA = MaxLengths(A.Shape(), gvA.Shape());
    const std::vector<Int> maxLocalShapeB = MaxLengths(B.Shape(), gvB.Shape());

    printf("recvBuf:");
    for(int i = 0; i < prod(maxLocalShapeA) / nModeProcs; i++){
        printf(" %d", recvBuf[i]);
    }
    printf("\n");

    const std::vector<Int> localShapeB = B.LocalShape();         //Shape of the local tensor we are packing

    //Number of outer slices to unpack
    const int nMaxOuterSlices = Max(1, prod(maxLocalShapeB, rsModeB + 1));
    const int nLocalOuterSlices = Max(1, prod(localShapeB, rsModeB + 1));

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

    printf("MemCopy info:\n");
    printf("    nMaxOuterSlices: %d\n", nMaxOuterSlices);
    printf("    nMaxRSModeSlices: %d\n", nMaxRSModeSlices);
    printf("    maxCopySliceSize: %d\n", maxCopySliceSize);
    printf("    copySliceSize: %d\n", copySliceSize);
    for(outerSliceNum = 0; outerSliceNum < nMaxOuterSlices; outerSliceNum++){
        if(outerSliceNum >= nLocalOuterSlices)
            break;
        outerRecvBufOff = maxCopySliceSize * nMaxRSModeSlices * outerSliceNum;
        outerDataBufOff = copySliceSize * nLocalRSModeSlices * outerSliceNum;

        printf("        outerSliceNum: %d\n", outerSliceNum);
        printf("        outerRecvBufOff: %d\n", outerRecvBufOff);
        printf("        outerDataBufOff: %d\n", outerDataBufOff);

        for(rsModeSliceNum = 0; rsModeSliceNum < nMaxRSModeSlices; rsModeSliceNum++){
            if(rsModeSliceNum >= nLocalRSModeSlices)
                break;

            rsModeRecvBufOff = (maxCopySliceSize * rsModeSliceNum);
            rsModeDataBufOff = (copySliceSize * rsModeSliceNum);

            startRecvBuf = outerRecvBufOff + rsModeRecvBufOff;
            startDataBuf = outerDataBufOff + rsModeDataBufOff;

            printf("          startRecvBuf: %d\n", startRecvBuf);
            printf("          startDataBuf: %d\n", startDataBuf);
            MemCopy(&(dataBuf[startDataBuf]), &(recvBuf[startRecvBuf]), copySliceSize);
        }
    }

    printf("dataBuf:");
    for(int i = 0; i < prod(B.LocalShape()); i++)
        printf(" %d", dataBuf[i]);
    printf("\n");
}

//Only called when fully reducing an index
//NOTE: Looks an awful lot like PackAGSendBuf...
template <typename T>
void UnpackRSRecvBuf(const T * const recvBuf, const Int reduceIndex, const Int scatterIndex, const DistTensor<T>& A, DistTensor<T>& B)
{
    const std::vector<Int> start(B.Order(), 0);
    T* dataBuf = B.Buffer(start);

    const int rModeA = A.ModeOfIndex(reduceIndex);
    const int sModeA = A.ModeOfIndex(scatterIndex);
    const int sModeB = B.ModeOfIndex(scatterIndex);

    const tmen::GridView gvA = A.GridView();
    const tmen::GridView gvB = B.GridView();

    const int nRedistProcs = gvA.Dimension(rModeA);   //In PRS, scatter into same index

    const std::vector<Int> maxLocalShapeA = MaxLengths(A.Shape(), gvA.Shape());
    const std::vector<Int> maxLocalShapeB = MaxLengths(B.Shape(), gvB.Shape());

    printf("recvBuf:");
    for(int i = 0; i < prod(maxLocalShapeA) / nRedistProcs; i++){
        printf(" %d", recvBuf[i]);
    }
    printf("\n");

    const std::vector<Int> localShapeB = B.LocalShape();         //Shape of the local tensor we are packing

    //Number of outer slices to unpack
    const int nMaxOuterSlices = Max(1, prod(maxLocalShapeB, sModeB + 1));
    const int nLocalOuterSlices = Max(1, prod(localShapeB, sModeB + 1));

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

    printf("MemCopy info:\n");
    printf("    nMaxOuterSlices: %d\n", nMaxOuterSlices);
    printf("    nMaxSModeSlices: %d\n", nMaxSModeSlices);
    printf("    maxCopySliceSize: %d\n", maxCopySliceSize);
    printf("    copySliceSize: %d\n", copySliceSize);
    for(outerSliceNum = 0; outerSliceNum < nMaxOuterSlices; outerSliceNum++){
        if(outerSliceNum >= nLocalOuterSlices)
            break;
        outerRecvBufOff = maxCopySliceSize * nMaxSModeSlices * outerSliceNum;
        outerDataBufOff = copySliceSize * nLocalSModeSlices * outerSliceNum;

        printf("        outerSliceNum: %d\n", outerSliceNum);
        printf("        outerRecvBufOff: %d\n", outerRecvBufOff);
        printf("        outerDataBufOff: %d\n", outerDataBufOff);

        for(sModeSliceNum = 0; sModeSliceNum < nMaxSModeSlices; sModeSliceNum++){
            if(sModeSliceNum >= nLocalSModeSlices)
                break;

            sModeRecvBufOff = (maxCopySliceSize * sModeSliceNum);
            sModeDataBufOff = (copySliceSize * sModeSliceNum);

            startRecvBuf = outerRecvBufOff + sModeRecvBufOff;
            startDataBuf = outerDataBufOff + sModeDataBufOff;

            printf("          startRecvBuf: %d\n", startRecvBuf);
            printf("          startDataBuf: %d\n", startDataBuf);
            MemCopy(&(dataBuf[startDataBuf]), &(recvBuf[startRecvBuf]), copySliceSize);
        }
    }

    printf("dataBuf:");
    for(int i = 0; i < prod(B.LocalShape()); i++)
        printf(" %d", dataBuf[i]);
    printf("\n");
}

//Given following set of strides in a tensor (s1, s2, ..., sm) and mode x we wish to redistribute
//Pack according to following scheme:
//ptr = 0  <-- Ptr into the sendBuf
//foreach proc in nProcs:
//  foreach wrapNum in ceil((mx-procNum)/nProcs)  <-- Number of wraps we have to send to each proc
//    startIndex = sx * (wrapNum* nProcs + proc)  <-- Sets the first index we should copy from, i.e, (0,...,0,wrapNum*nProcs + proc, 0, ..., 0)
//    foreach prod(m(x+1), ..., m(m)):  <-- Defines the number of slices we have to copy over per wrap
//      memcpy(dataBuf[startIndex], sendBuf[ptr], max(1, prod(s1,..., s(x-1)))) <-- Copy the slice (0,..., 0, proc, 0, ..., 0) to (m1, m2, ..., m(x-1), proc, 0, ..., 0)
//      startIndex += sx*mx  <-- Put us in the next slice , i.e., (0,..., 0, proc, 1, ..., 0) to (m1, m2, ..., m(x-1), proc, 1, ..., 0)
//      ptr += max(1, prod(s1,..., s(x-1))) <-- Increment ptr in the sendBuf
//TODO: Make this work with blocks
//TODO: Swap nWrap test to be more like sliceNum test (Refer to local information)
template <typename T>
void UnpackAGRecvBuf(const T * const recvBuf, const Int allGatherIndex, const DistTensor<T>& A, DistTensor<T>& B)
{
//  printf("B can unpack %d elems\n", prod(B.LocalShape()));
    const std::vector<Int> start(B.Order(), 0);
    T* dataBuf = B.Buffer(start);
    const tmen::GridView gv = A.GridView();

    const int allGatherMode = A.ModeOfIndex(allGatherIndex);
    const int nModeProcs = gv.Dimension(allGatherMode);   //Number of procs per wrap
    const int agModeGlobalDim = B.Dimension(allGatherMode);           //Number of indices in the mode we are redistributing

    const std::vector<Int> maxRecvLocalShape = MaxLengths(A.Shape(), gv.Shape());
    const int agModeMaxLocalDim = maxRecvLocalShape[allGatherMode];

    const std::vector<Int> localShape = B.LocalShape();         //Shape of the local tensor we are packing
    const int agModeLocalDim = B.LocalDimension(allGatherMode); //Local version
    const int agModeLocalStride = B.LocalModeStride(allGatherMode);

    //Loop packing bounds variables
    const int nWraps = MaxLength(agModeGlobalDim, nModeProcs);
    //Number of local slices and slice size we must pack per proc per wrap
    const int nLocalSlices = Max(1, prod(localShape, allGatherMode + 1));
    const int nMaxSlices = Max(1, prod(maxRecvLocalShape, allGatherMode + 1));

    //Variables for calculating elements to copy
    const int nMaxElemsPerProc = prod(maxRecvLocalShape);
    const int copySliceSize = agModeLocalStride;

    //Loop iteration vars
    int procRecvNum, wrapNum, sliceNum;  //Pack data for slice "sliceNum" (<nSlices) of wrap "wrapNum" (<nWraps) for proc "procSendNum" int offSliceRecvBuf, offWrapRecvBuf;  //Offsets used to index into sendBuf array
    int offSliceRecvBuf, offWrapRecvBuf;  //Offsets used to index into recvBuf array
    int offSliceDataBuf, offWrapDataBuf;  //Offsets used to index into dataBuf array
    int startRecvBuf, startDataBuf;

    for(sliceNum = 0; sliceNum < nMaxSlices; sliceNum++){
        offSliceRecvBuf = copySliceSize * agModeMaxLocalDim * sliceNum;
        offSliceDataBuf = copySliceSize * agModeGlobalDim * sliceNum;
        if(sliceNum >= nLocalSlices){
            break;
        }
        for(wrapNum = 0; wrapNum < nWraps; wrapNum++){
            offWrapRecvBuf = copySliceSize * wrapNum;
            offWrapDataBuf = copySliceSize * nModeProcs * wrapNum;
            for(procRecvNum = 0; procRecvNum < nModeProcs; procRecvNum++){
                startRecvBuf = offSliceRecvBuf + offWrapRecvBuf + (nMaxElemsPerProc * procRecvNum);
                startDataBuf = offSliceDataBuf + offWrapDataBuf + (copySliceSize * procRecvNum);
                if(wrapNum * nModeProcs + procRecvNum >= agModeLocalDim){
                    break;
                }
                MemCopy(&(dataBuf[startDataBuf]), &(recvBuf[startRecvBuf]), copySliceSize);
            }
        }
    }
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
