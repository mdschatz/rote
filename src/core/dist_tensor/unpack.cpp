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
void UnpackPermutationRecvBuf(const T * const recvBuf, const Index permuteIndex, const DistTensor<T>& A, DistTensor<T>& B)
{
        const Location start(B.Order(), 0);
        T* dataBuf = B.Buffer(start);

        const Mode pModeB = B.ModeOfIndex(permuteIndex);

        const tmen::GridView gvA = A.GridView();
        const tmen::GridView gvB = B.GridView();

        const ObjShape maxLocalShapeA = MaxLengths(A.Shape(), gvA.Shape());
        const ObjShape maxLocalShapeB = MaxLengths(B.Shape(), gvB.Shape());

        const ObjShape localShapeB = B.LocalShape();         //Shape of the local tensor we are packing

        //Number of outer slices to unpack
        const Unsigned nMaxOuterSlices = Max(1, prod(maxLocalShapeB, pModeB + 1));
        const Unsigned nLocalOuterSlices = prod(localShapeB, pModeB + 1);

        //Loop packing bounds variables
        const Unsigned nMaxPModeSlices = maxLocalShapeB[pModeB];
        const Unsigned nLocalPModeSlices = localShapeB[pModeB];

        //Variables for calculating elements to copy
        const Unsigned maxCopySliceSize = Max(1, prod(maxLocalShapeB, 0, pModeB));
        const Unsigned copySliceSize = B.LocalModeStride(pModeB);

        //Loop iteration vars
        Unsigned outerSliceNum, pModeSliceNum;  //Pack data for slice "sliceNum" (<nSlices) of wrap "wrapNum" (<nWraps) for proc "procSendNum" Unsigned offSliceRecvBuf, offWrapRecvBuf;  //Offsets used to index into sendBuf array
        Unsigned outerRecvBufOff, outerDataBufOff;  //Offsets used to index into recvBuf array
        Unsigned pModeRecvBufOff, pModeDataBufOff;  //Offsets used to index into dataBuf array
        Unsigned startRecvBuf, startDataBuf;

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
void UnpackPartialRSRecvBuf(const T * const recvBuf, const Index reduceScatterIndex, const DistTensor<T>& A, DistTensor<T>& B)
{
    const Location start(B.Order(), 0);
    T* dataBuf = B.Buffer(start);

    const Mode rsModeB = B.ModeOfIndex(reduceScatterIndex);

    const tmen::GridView gvA = A.GridView();
    const tmen::GridView gvB = B.GridView();

    const ObjShape maxLocalShapeA = MaxLengths(A.Shape(), gvA.Shape());
    const ObjShape maxLocalShapeB = MaxLengths(B.Shape(), gvB.Shape());

//    printf("recvBuf:");
//    for(Unsigned i = 0; i < prod(maxLocalShapeA) / nRedistProcs; i++){
//        printf(" %d", recvBuf[i]);
//    }
//    printf("\n");

    const ObjShape localShapeB = B.LocalShape();         //Shape of the local tensor we are packing

    //Number of outer slices to unpack
    const Unsigned nMaxOuterSlices = Max(1, prod(maxLocalShapeB, rsModeB + 1));
    const Unsigned nLocalOuterSlices = prod(localShapeB, rsModeB + 1);

    //Loop packing bounds variables
    const Unsigned nMaxRSModeSlices = maxLocalShapeB[rsModeB];
    const Unsigned nLocalRSModeSlices = localShapeB[rsModeB];

    //Each wrap is copied contiguously because the distribution of reduceScatter index does not change

    //Variables for calculating elements to copy
    const Unsigned maxCopySliceSize = Max(1, prod(maxLocalShapeB, 0, rsModeB));
    const Unsigned copySliceSize = B.LocalModeStride(rsModeB);

    //Loop iteration vars
    Unsigned outerSliceNum, rsModeSliceNum;  //Pack data for slice "sliceNum" (<nSlices) of wrap "wrapNum" (<nWraps) for proc "procSendNum" into offSliceRecvBuf, offWrapRecvBuf;  //Offsets used to index into sendBuf array
    Unsigned outerRecvBufOff, outerDataBufOff;  //Offsets used to index into recvBuf array
    Unsigned rsModeRecvBufOff, rsModeDataBufOff;  //Offsets used to index into dataBuf array
    Unsigned startRecvBuf, startDataBuf;

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
//    for(Unsigned i = 0; i < prod(B.LocalShape()); i++)
//        printf(" %d", dataBuf[i]);
//    printf("\n");
}

//Only called when fully reducing an index
//NOTE: Looks an awful lot like PackAGSendBuf...
template <typename T>
void UnpackRSRecvBuf(const T * const recvBuf, const Index reduceIndex, const Index scatterIndex, const DistTensor<T>& A, DistTensor<T>& B)
{
    const Location start(B.Order(), 0);
    T* dataBuf = B.Buffer(start);

    const Mode sModeB = B.ModeOfIndex(scatterIndex);

    const tmen::GridView gvA = A.GridView();
    const tmen::GridView gvB = B.GridView();

    const ObjShape maxLocalShapeA = MaxLengths(A.Shape(), gvA.Shape());
    const ObjShape maxLocalShapeB = MaxLengths(B.Shape(), gvB.Shape());

//    printf("recvBuf:");
//    for(Unsigned i = 0; i < prod(maxLocalShapeA) / nRedistProcs; i++){
//        printf(" %d", recvBuf[i]);
//    }
//    printf("\n");

    const ObjShape localShapeB = B.LocalShape();         //Shape of the local tensor we are packing

    //Number of outer slices to unpack
    const Unsigned nMaxOuterSlices = Max(1, prod(maxLocalShapeB, sModeB + 1));
    const Unsigned nLocalOuterSlices = prod(localShapeB, sModeB + 1);

    //Loop packing bounds variables
    const Unsigned nMaxSModeSlices = maxLocalShapeB[sModeB];
    const Unsigned nLocalSModeSlices = localShapeB[sModeB];

    //Each wrap is copied contiguously because the distribution of reduceScatter index does not change

    //Variables for calculating elements to copy
    const Unsigned maxCopySliceSize = Max(1, prod(maxLocalShapeB, 0, sModeB));
    const Unsigned copySliceSize = B.LocalModeStride(sModeB);

    //Loop iteration vars
    Unsigned outerSliceNum, sModeSliceNum;  //Pack data for slice "sliceNum" (<nSlices) of wrap "wrapNum" (<nWraps) for proc "procSendNum" int offSliceRecvBuf, offWrapRecvBuf;  //Offsets used to index into sendBuf array
    Unsigned outerRecvBufOff, outerDataBufOff;  //Offsets used to index into recvBuf array
    Unsigned sModeRecvBufOff, sModeDataBufOff;  //Offsets used to index into dataBuf array
    Unsigned startRecvBuf, startDataBuf;

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
//    for(Unsigned i = 0; i < prod(B.LocalShape()); i++)
//        printf(" %d", dataBuf[i]);
//    printf("\n");
}

template <typename T>
void UnpackAGRecvBuf(const T * const recvBuf, const Index allGatherIndex, const DistTensor<T>& A, DistTensor<T>& B)
{
    const Location start(B.Order(), 0);
    T* dataBuf = B.Buffer(start);

    const Mode agModeA = A.ModeOfIndex(allGatherIndex);
    const Mode agModeB = B.ModeOfIndex(allGatherIndex);

    const tmen::GridView gvA = A.GridView();
    const tmen::GridView gvB = B.GridView();

    const Unsigned nRedistProcs = gvA.Dimension(agModeA);

    const ObjShape maxLocalShapeA = MaxLengths(A.Shape(), gvA.Shape());
    const ObjShape maxLocalShapeB = MaxLengths(B.Shape(), gvB.Shape());

//    printf("recvBuf:");
//    for(Unsigned i = 0; i < prod(maxLocalShapeA) * nRedistProcs; i++){
//        printf(" %d", recvBuf[i]);
//    }
//    printf("\n");

    const ObjShape localShapeB = B.LocalShape();

    //Number of outer slices to unpack
    const Unsigned nMaxOuterSlices = Max(1, prod(maxLocalShapeB, agModeB + 1));
    const Unsigned nLocalOuterSlices = prod(localShapeB, agModeB + 1);

    //Loop packing bounds variables
    const Unsigned nMaxAGModeSlices = maxLocalShapeB[agModeB];
    const Unsigned nLocalAGModeSlices = localShapeB[agModeB];
    const Unsigned agModeUnpackStride = nRedistProcs;

    //Variables for calculating elements to copy
    const Unsigned maxCopySliceSize = Max(1, prod(maxLocalShapeB, 0, agModeB));
    const Unsigned copySliceSize = B.LocalModeStride(agModeB);

    //Number of processes we have to unpack from
    const Unsigned nElemSlices = nRedistProcs;

    //Loop iteration vars
    Unsigned outerSliceNum, agModeSliceNum, elemSliceNum;  //Pack data for slice "sliceNum" (<nSlices) of wrap "wrapNum" (<nWraps) for proc "procSendNum" int offSliceRecvBuf, offWrapRecvBuf;  //Offsets used to index into sendBuf array
    Unsigned elemRecvBufOff, elemDataBufOff;
    Unsigned outerRecvBufOff, outerDataBufOff;  //Offsets used to index into recvBuf array
    Unsigned agModeRecvBufOff, agModeDataBufOff;  //Offsets used to index into dataBuf array
    Unsigned startRecvBuf, startDataBuf;

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
//    for(Unsigned i = 0; i < prod(B.LocalShape()); i++)
//        printf(" %d", dataBuf[i]);
//    printf("\n");
}

template <typename T>
void UnpackLocalRedist(DistTensor<T>& B, const DistTensor<T>& A, const Index localIndex, const ModeArray& gridRedistModes)
{
    Unsigned i;
    const Unsigned order = A.Order();
    const Location start(order, 0);
    T* dstBuf = B.Buffer(start);
    const T* srcBuf = A.LockedBuffer(start);

    const tmen::GridView gvA = A.GridView();
    const tmen::GridView gvB = B.GridView();

    const tmen::Grid& g = A.Grid();

    Mode localModeA = A.ModeOfIndex(localIndex);
    Mode localModeB = B.ModeOfIndex(localIndex);

    ModeDistribution lIndexDistA = A.ModeDist(localModeA);
    ModeDistribution lIndexDistB = B.ModeDist(localModeB);

    ModeArray commModes(lIndexDistB.begin() + lIndexDistA.size(), lIndexDistB.end());

    Location myGridLoc = g.Loc();
    ObjShape gridShape = g.Shape();

    Location myCommLoc = FilterVector(myGridLoc, commModes);
    ObjShape commShape = FilterVector(gridShape, commModes);
    Unsigned myCommLinLoc = Loc2LinearLoc(myCommLoc, commShape);

    //NOTE: CHECK THIS IS CORRECT
    Unsigned modeUnpackStride = prod(commShape);

    //Number of slices after the mode to redist
    const ObjShape localShape = A.LocalShape();
    const ObjShape outerSliceShape(localShape.begin() + localModeA + 1, localShape.end());
    const Unsigned nOuterSlices = prod(outerSliceShape);

    //Number of slices represented by the mode
    const Unsigned nLModeSlices = localShape[localModeA];

    //Size of slice to copy
    const ObjShape copySliceShape(localShape.begin(), localShape.begin() + localModeA);
    const Unsigned copySliceSize = prod(copySliceShape);

    //Where we start copying
    const Unsigned elemStartLoc = myCommLinLoc * copySliceSize;

    Unsigned lModeSliceNum, outerSliceNum;
    Unsigned lModeDstOff, outerDstOff;
    Unsigned lModeSrcOff, outerSrcOff;
    Unsigned startDstBuf, startSrcBuf;


//        printf("MemCopy info:\n");
//        printf("    nOuterSlices: %d\n", nOuterSlices);
//        printf("    nLModeSlices: %d\n", nLModeSlices);
//        printf("    copySliceSize: %d\n", copySliceSize);
//        printf("    modeUnpackStride: %d\n", modeUnpackStride);
    for(outerSliceNum = 0; outerSliceNum < nOuterSlices; outerSliceNum++){
        //NOTE: FIX THIS, WE NEED TO SEE HOW MANY TIMES WE RUN THROUGH THE lModeSliceNum loop (similar to some other unpack routine)
        outerDstOff = copySliceSize * (nLModeSlices / modeUnpackStride) * outerSliceNum;
        outerSrcOff = copySliceSize * nLModeSlices * outerSliceNum;

//      printf("        outerSliceNum: %d\n", outerSliceNum);
//      printf("        outerDstOff: %d\n", outerDstOff);
//      printf("        outerSrcOff: %d\n", outerSrcOff);
        for(lModeSliceNum = 0; lModeSliceNum < nLModeSlices; lModeSliceNum += modeUnpackStride){
            lModeDstOff = copySliceSize * lModeSliceNum / modeUnpackStride;
            lModeSrcOff = copySliceSize * lModeSliceNum;

//          printf("          lModeSliceNum: %d\n", lModeSliceNum);
//          printf("          lModeDstOff: %d\n", lModeDstOff);
//          printf("          lModeSrcOff: %d\n", lModeSrcOff);
            startDstBuf = outerDstOff + lModeDstOff;
            startSrcBuf = outerSrcOff + lModeSrcOff + elemStartLoc;

//          printf("          startDstBuf: %d\n", startDstBuf);
//          printf("          startSrcBuf: %d\n", startSrcBuf);
            MemCopy(&(dstBuf[startDstBuf]), &(srcBuf[startSrcBuf]), copySliceSize);
        }
    }
}

template<typename T>
void UnpackA2ADoubleIndexRecvBuf(const T * const recvBuf, const std::pair<Index, Index>& a2aIndices, const std::pair<ModeArray, ModeArray >& commGroups, const DistTensor<T>& A, DistTensor<T>& B){
    Unsigned i;
    const Unsigned order = A.Order();
    const Location  start(order, 0);
    T* dataBuf = B.Buffer(start);

    const tmen::GridView gvA = A.GridView();
    const tmen::GridView gvB = B.GridView();

    const tmen::Grid& g = A.Grid();

    Mode a2aMode1 = A.ModeOfIndex(a2aIndices.first);
    Mode a2aMode2 = A.ModeOfIndex(a2aIndices.second);

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

    ObjShape tensorShape = B.Shape();
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

    const ObjShape localShape = B.LocalShape();
    ObjShape packedLocalShape = MaxLengths(A.Shape(), gvA.Shape());

    //Slices of a2aMode1
    const Unsigned nLocalA2AMode1Slices = localShape[a2aMode1];

    //Slices between a2aMode1 and a2aMode2
    const Unsigned nLocalMidSlices = Max(1, prod(localShape, a2aMode1 + 1, a2aMode2));

    //Slices of a2aMode2
    const Unsigned nLocalA2AMode2Slices = localShape[a2aMode2];

    const Unsigned copySliceSize = B.LocalModeStride(a2aMode1);
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

    Location myFirstLoc = B.ModeShifts();

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
        if(AnyElemwiseGreaterThanEqualTo(startUnpackElemLoc, B.Shape()))
            continue;

        //Determine the Multiloc of the process that sent this element
        Location owningProcGVA = A.DetermineOwner(startUnpackElemLoc);
        Location owningProcG = GridViewLoc2GridLoc(owningProcGVA, gvA);
        Unsigned owningProc = Loc2LinearLoc(FilterVector(owningProcG, commModes), FilterVector(gridShape, commModes));

        //Find the local location of the global starting element we are now unpacking
        Location localLoc = B.Global2LocalIndex(startUnpackElemLoc);

        //Now that we know the local loc of the element to unpack, we know how many iterations of unpacking to perform per mode
        const ObjShape tensorShape = B.Shape();
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
//    printf("dataBuf:");
//    for(Unsigned i = 0; i < prod(B.LocalShape()); i++)
//        printf(" %d", dataBuf[i]);
//    printf("\n");
}

#define PROTO(T) \
        template void UnpackPermutationRecvBuf(const T * const recvBuf, const Index permuteIndex, const DistTensor<T>& A, DistTensor<T>& B); \
        template void UnpackPartialRSRecvBuf(const T * const recvBuf, const Index reduceScatterIndex, const DistTensor<T>& A, DistTensor<T>& B); \
        template void UnpackRSRecvBuf(const T * const recvBuf, const Index reduceIndex, const Index scatterIndex, const DistTensor<T>& A, DistTensor<T>& B); \
        template void UnpackAGRecvBuf(const T * const recvBuf, const Index allGatherIndex, const DistTensor<T>& A, DistTensor<T>& B); \
        template void UnpackLocalRedist(DistTensor<T>& B, const DistTensor<T>& A, const Index localIndex, const ModeArray& gridRedistModes); \
        template void UnpackA2ADoubleIndexRecvBuf(const T * const recvBuf, const std::pair<Index, Index>& a2aIndices, const std::pair<ModeArray, ModeArray >& commGroups, const DistTensor<T>& A, DistTensor<T>& B);

PROTO(int)
PROTO(float)
PROTO(double)

} // namespace tmen
