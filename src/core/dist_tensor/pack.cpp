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

//NOTE: This should just be a direct memcopy. But sticking to the same structured code as all other collectives
template <typename T>
void PackPermutationSendBuf(const DistTensor<T>& B, const DistTensor<T>& A, const Index permuteIndex, T * const sendBuf)
{
    const Location start(A.Order(), 0);
    const T* dataBuf = A.LockedBuffer(start);

    const Mode pModeA = A.ModeOfIndex(permuteIndex);

    const tmen::GridView gvA = A.GridView();

    //Shape of the local tensor we are packing
    const ObjShape maxLocalShapeA = MaxLengths(A.Shape(), gvA.Shape());
    const ObjShape localShapeA = A.LocalShape();

    //Calculate number of outer slices to pack
    const Unsigned nMaxOuterSlices = Max(1, prod(maxLocalShapeA, pModeA + 1));
    const Unsigned nLocalOuterSlices = prod(localShapeA, pModeA + 1);

    //Calculate number of rsMode slices to pack
    const Unsigned nMaxPModeSlices = maxLocalShapeA[pModeA];
    const Unsigned nLocalPModeSlices = localShapeA[pModeA];
    const Unsigned maxCopySliceSize = Max(1, prod(maxLocalShapeA, 0, pModeA));
    const Unsigned copySliceSize = prod(localShapeA, 0, pModeA);

    Unsigned outerSliceNum, pModeSliceNum; //Which slice of which wrap of which process are we packing
    Unsigned outerSendBufOff, pModeSendBufOff;
    Unsigned outerDataBufOff, pModeDataBufOff;
    Unsigned startSendBuf, startDataBuf;

    for(outerSliceNum = 0; outerSliceNum < nMaxOuterSlices; outerSliceNum++){
        if(outerSliceNum >= nLocalOuterSlices)
            break;
        outerSendBufOff = maxCopySliceSize * nMaxPModeSlices * outerSliceNum;
        outerDataBufOff = copySliceSize * nLocalPModeSlices * outerSliceNum;

        for(pModeSliceNum = 0; pModeSliceNum < nMaxPModeSlices; pModeSliceNum++){
            if(pModeSliceNum >= nLocalPModeSlices)
                break;
            pModeSendBufOff = maxCopySliceSize * pModeSliceNum;
            pModeDataBufOff = copySliceSize * pModeSliceNum;

            startSendBuf = outerSendBufOff + pModeSendBufOff;
            startDataBuf = outerDataBufOff + pModeDataBufOff;
            MemCopy(&(sendBuf[startSendBuf]), &(dataBuf[startDataBuf]), copySliceSize);
        }
    }
}

//NOTE: Exactly the same code as PackRSSendBuf(B, A, reduceScatterIndex, reduceScatterIndex, sendBuf);
template <typename T>
void PackPartialRSSendBuf(const DistTensor<T>& B, const DistTensor<T>& A, const Index reduceScatterIndex, T * const sendBuf)
{
    const Location start(A.Order(), 0);
    const T* dataBuf = A.LockedBuffer(start);

//    printf("dataBuf: ");
//    for(Unsigned i = 0; i < prod(A.LocalShape()); i++){
//        printf("%d ", dataBuf[i]);
//    }
//    printf("\n");

    const Mode rsModeA = A.ModeOfIndex(reduceScatterIndex);

    const tmen::GridView gvA = A.GridView();
    const tmen::GridView gvB = B.GridView();

    const Unsigned nRedistProcs = gvA.Dimension(rsModeA);

    //Shape of the local tensor we are packing
    const ObjShape maxLocalShapeA = MaxLengths(A.Shape(), gvA.Shape());
    const ObjShape localShapeA = A.LocalShape();

    //Calculate number of outer slices to pack
    const Unsigned nMaxOuterSlices = Max(1, prod(maxLocalShapeA, rsModeA + 1));
    const Unsigned nLocalOuterSlices = prod(localShapeA, rsModeA + 1);

    //Calculate number of rsMode slices to pack
    const Unsigned nMaxRSModeSlices = maxLocalShapeA[rsModeA];
    const Unsigned nLocalRSModeSlices = localShapeA[rsModeA];
    const Unsigned rsModePackStride = nRedistProcs;

    //Number of processes we have to pack for
    const Unsigned nElemSlices = nRedistProcs;

    const Unsigned maxCopySliceSize = Max(1, prod(maxLocalShapeA, 0, rsModeA));
    const Unsigned copySliceSize = prod(localShapeA, 0, rsModeA);

    Unsigned outerSliceNum, rsModeSliceNum, elemSliceNum; //Which slice of which wrap of which process are we packing
    Unsigned elemSendBufOff, elemDataBufOff;
    Unsigned outerSendBufOff, rsModeSendBufOff;
    Unsigned outerDataBufOff, rsModeDataBufOff;
    Unsigned startSendBuf, startDataBuf;

//    printf("MemCopy info:\n");
//    printf("    nMaxOuterSlices: %d\n", nMaxOuterSlices);
//    printf("    nMaxRSModeSlices: %d\n", nMaxRSModeSlices);
//    printf("    rsModePackStride: %d\n", rsModePackStride);
//    printf("    maxCopySliceSize: %d\n", maxCopySliceSize);
//    printf("    copySliceSize: %d\n", copySliceSize);
    for(elemSliceNum = 0; elemSliceNum < nElemSlices; elemSliceNum++){
        elemSendBufOff = prod(maxLocalShapeA) * elemSliceNum;
        elemDataBufOff = copySliceSize * elemSliceNum;

//        printf("      elemSliceNum: %d\n", elemSliceNum);
//        printf("      elemSendBufOff: %d\n", elemSendBufOff);
//        printf("      elemDataBufOff: %d\n", elemDataBufOff);

        for(outerSliceNum = 0; outerSliceNum < nMaxOuterSlices; outerSliceNum++ ){
            if(outerSliceNum >= nLocalOuterSlices)
                break;

            //NOTE: We need to make sure that we correctly set the outerSendBufOff
            //Meaning we need to skip over the maximum number of packed modes per process
            //thus we need an nMaxRSModeSlices / rsModePackStride rounded up
            //i.e we wish to pack 3 rsMode slices where the packStride is 2.  We will iterate in the rsModeSlice loop twice max
            outerSendBufOff = maxCopySliceSize * Max(1, (nMaxRSModeSlices - 1) / rsModePackStride + 1) * outerSliceNum;
            outerDataBufOff = copySliceSize * nLocalRSModeSlices * outerSliceNum;

//            printf("        outerSliceNum: %d\n", outerSliceNum);
//            printf("        outerSendBufOff: %d\n", outerSendBufOff);
//            printf("        outerDataBufOff: %d\n", outerDataBufOff);

            for(rsModeSliceNum = 0; rsModeSliceNum < nMaxRSModeSlices; rsModeSliceNum += rsModePackStride){
                if(rsModeSliceNum + elemSliceNum >= nLocalRSModeSlices)
                    break;
                rsModeSendBufOff = maxCopySliceSize * (rsModeSliceNum / rsModePackStride);
                rsModeDataBufOff = copySliceSize * rsModeSliceNum;

//                printf("          rsModeSliceNum: %d\n", rsModeSliceNum);
//                printf("          rsModeSendBufOff: %d\n", rsModeSendBufOff);
//                printf("          rsModeDataBufOff: %d\n", rsModeDataBufOff);
                startSendBuf = elemSendBufOff + outerSendBufOff + rsModeSendBufOff;
                startDataBuf = elemDataBufOff + outerDataBufOff + rsModeDataBufOff;

//                printf("          startSendBuf: %d\n", startSendBuf);
//                printf("          startDataBuf: %d\n", startDataBuf);
                MemCopy(&(sendBuf[startSendBuf]), &(dataBuf[startDataBuf]), copySliceSize);
            }
        }
    }
//    printf("packed sendBuf: ");
//    for(Unsigned i = 0; i < prod(maxLocalShapeA) * nRedistProcs; i++)
//        printf("%d ", sendBuf[i]);
//    printf("\n");
}

//Only called when fully reducing an index
//NOTE: Looks an awful lot like PackAGSendBuf...
//TODO: Merge with PackAGSendBuf?
template <typename T>
void PackRSSendBuf(const DistTensor<T>& B, const DistTensor<T>& A, const Index reduceIndex, const Index scatterIndex, T * const sendBuf)
{
    const Location start(A.Order(), 0);
    const T* dataBuf = A.LockedBuffer(start);

//    printf("dataBuf: ");
//    for(Unsigned i = 0; i < prod(A.LocalShape()); i++){
//        printf("%d ", dataBuf[i]);
//    }
//    printf("\n");

    const Mode rModeA = A.ModeOfIndex(reduceIndex);
    const Mode sModeA = A.ModeOfIndex(scatterIndex);

    const tmen::GridView gvA = A.GridView();
    const tmen::GridView gvB = B.GridView();

    const Unsigned nRedistProcs = gvA.Dimension(rModeA);

    //Shape of the local tensor we are packing
    const ObjShape maxLocalShapeA = MaxLengths(A.Shape(), gvA.Shape());
    const ObjShape localShapeA = A.LocalShape();

    //Calculate number of outer slices to pack
    const Unsigned nMaxOuterSlices = Max(1, prod(maxLocalShapeA, sModeA + 1));
    const Unsigned nLocalOuterSlices = prod(localShapeA, sModeA + 1);

    //Calculate number of sMode slices to pack
    const Unsigned nMaxSModeSlices = maxLocalShapeA[sModeA];
    const Unsigned nLocalSModeSlices = localShapeA[sModeA];
    const Unsigned sModePackStride = nRedistProcs;

    //Number of processes we have to pack for
    const Unsigned nElemSlices = nRedistProcs;

    const Unsigned maxCopySliceSize = Max(1, prod(maxLocalShapeA, 0, sModeA));
    const Unsigned copySliceSize = prod(localShapeA, 0, sModeA);

    Unsigned outerSliceNum, sModeSliceNum, elemSliceNum; //Which slice of which wrap of which process are we packing
    Unsigned elemSendBufOff, elemDataBufOff;
    Unsigned outerSendBufOff, sModeSendBufOff;
    Unsigned outerDataBufOff, sModeDataBufOff;
    Unsigned startSendBuf, startDataBuf;


//    printf("MemCopy info:\n");
//    printf("    nMaxOuterSlices: %d\n", nMaxOuterSlices);
//    printf("    nMaxSModeSlices: %d\n", nMaxSModeSlices);
//    printf("    sModePackStride: %d\n", sModePackStride);
//    printf("    maxCopySliceSize: %d\n", maxCopySliceSize);
//    printf("    copySliceSize: %d\n", copySliceSize);
    for(elemSliceNum = 0; elemSliceNum < nElemSlices; elemSliceNum++){
        elemSendBufOff = prod(maxLocalShapeA) * elemSliceNum;
        elemDataBufOff = copySliceSize * elemSliceNum;

//        printf("      elemSliceNum: %d\n", elemSliceNum);
//        printf("      elemSendBufOff: %d\n", elemSendBufOff);
//        printf("      elemDataBufOff: %d\n", elemDataBufOff);

        for(outerSliceNum = 0; outerSliceNum < nMaxOuterSlices; outerSliceNum++ ){
            if(outerSliceNum >= nLocalOuterSlices)
                break;
            outerSendBufOff = maxCopySliceSize * Max(1, nMaxSModeSlices / sModePackStride) * outerSliceNum;
            outerDataBufOff = copySliceSize * nLocalSModeSlices * outerSliceNum;

//            printf("        outerSliceNum: %d\n", outerSliceNum);
//            printf("        outerSendBufOff: %d\n", outerSendBufOff);
//            printf("        outerDataBufOff: %d\n", outerDataBufOff);

                for(sModeSliceNum = 0; sModeSliceNum < nMaxSModeSlices; sModeSliceNum += sModePackStride){
                    if(sModeSliceNum + elemSliceNum >= nLocalSModeSlices)
                        break;
                    sModeSendBufOff = maxCopySliceSize * (sModeSliceNum / sModePackStride);
                    sModeDataBufOff = copySliceSize * sModeSliceNum;

//                    printf("          sModeSliceNum: %d\n", sModeSliceNum);
//                    printf("          sModeSendBufOff: %d\n", sModeSendBufOff);
//                    printf("          sModeDataBufOff: %d\n", sModeDataBufOff);
                    startSendBuf = elemSendBufOff + outerSendBufOff + sModeSendBufOff;
                    startDataBuf = elemDataBufOff + outerDataBufOff + sModeDataBufOff;

//                    printf("          startSendBuf: %d\n", startSendBuf);
//                    printf("          startDataBuf: %d\n", startDataBuf);
                    MemCopy(&(sendBuf[startSendBuf]), &(dataBuf[startDataBuf]), copySliceSize);
                }
        }
    }

//    printf("packed sendBuf: ");
//    for(Unsigned i = 0; i < prod(maxLocalShapeA) * nRedistProcs; i++)
//        printf("%d ", sendBuf[i]);
//    printf("\n");
}

template <typename T>
void PackAGSendBuf(const DistTensor<T>& A, const Index allGatherIndex, T * const sendBuf, const ModeArray& redistModes)
{
  const Location start(A.Order(), 0);
  const T* dataBuf = A.LockedBuffer(start);

  const Mode agModeA = A.ModeOfIndex(allGatherIndex);

  const tmen::GridView gvA = A.GridView();

  //Shape of the local tensor we are packing
  const ObjShape maxLocalShapeA = MaxLengths(A.Shape(), gvA.Shape());
  const ObjShape localShapeA = A.LocalShape();

  //Calculate number of outer slices to pack
  const Unsigned nMaxOuterSlices = Max(1, prod(maxLocalShapeA, agModeA + 1));
  const Unsigned nLocalOuterSlices = prod(localShapeA, agModeA + 1);

  //Calculate number of agMode slices to pack
  const Unsigned nMaxAGModeSlices = maxLocalShapeA[agModeA];
  const Unsigned nLocalAGModeSlices = localShapeA[agModeA];

  const Unsigned maxCopySliceSize = Max(1, prod(maxLocalShapeA, 0, agModeA));
  const Unsigned copySliceSize = prod(localShapeA, 0, agModeA);

  Unsigned outerSliceNum, agModeSliceNum; //Which slice of which wrap of which process are we packing
  Unsigned outerSendBufOff, agModeSendBufOff;
  Unsigned outerDataBufOff, agModeDataBufOff;
  Unsigned startSendBuf, startDataBuf;

  printf("MemCopy info:\n");
  printf("    nMaxOuterSlices: %d\n", nMaxOuterSlices);
  printf("    nMaxAGModeSlices: %d\n", nMaxAGModeSlices);
  printf("    maxCopySliceSize: %d\n", maxCopySliceSize);
  printf("    copySliceSize: %d\n", copySliceSize);
  for(outerSliceNum = 0; outerSliceNum < nMaxOuterSlices; outerSliceNum++){
      if(outerSliceNum >= nLocalOuterSlices)
          break;
      outerSendBufOff = maxCopySliceSize * nMaxAGModeSlices * outerSliceNum;
      outerDataBufOff = copySliceSize * nLocalAGModeSlices * outerSliceNum;

      printf("        outerSliceNum: %d\n", outerSliceNum);
      printf("        outerSendBufOff: %d\n", outerSendBufOff);
      printf("        outerDataBufOff: %d\n", outerDataBufOff);

      for(agModeSliceNum = 0; agModeSliceNum < nMaxAGModeSlices; agModeSliceNum++){
          if(agModeSliceNum >= nLocalAGModeSlices)
              break;
          agModeSendBufOff = maxCopySliceSize * agModeSliceNum;
          agModeDataBufOff = copySliceSize * agModeSliceNum;

          printf("          agModeSliceNum: %d\n", agModeSliceNum);
          printf("          agModeSendBufOff: %d\n", agModeSendBufOff);
          printf("          agModeDataBufOff: %d\n", agModeDataBufOff);
          startSendBuf = outerSendBufOff + agModeSendBufOff;
          startDataBuf = outerDataBufOff + agModeDataBufOff;

          printf("          startSendBuf: %d\n", startSendBuf);
          printf("          startDataBuf: %d\n", startDataBuf);
          MemCopy(&(sendBuf[startSendBuf]), &(dataBuf[startDataBuf]), copySliceSize);
      }
  }
  printf("packed sendBuf: ");
  for(Unsigned i = 0; i < prod(maxLocalShapeA); i++)
      printf("%d ", sendBuf[i]);
  printf("\n");
}

template <typename T>
void PackAGSendBuf(const DistTensor<T>& A, const Index allGatherIndex, T * const sendBuf)
{
  const Location start(A.Order(), 0);
  const T* dataBuf = A.LockedBuffer(start);

  const Mode agModeA = A.ModeOfIndex(allGatherIndex);

  const tmen::GridView gvA = A.GridView();

  //Shape of the local tensor we are packing
  const ObjShape maxLocalShapeA = MaxLengths(A.Shape(), gvA.Shape());
  const ObjShape localShapeA = A.LocalShape();

  //Calculate number of outer slices to pack
  const Unsigned nMaxOuterSlices = Max(1, prod(maxLocalShapeA, agModeA + 1));
  const Unsigned nLocalOuterSlices = prod(localShapeA, agModeA + 1);

  //Calculate number of rsMode slices to pack
  const Unsigned nMaxAGModeSlices = maxLocalShapeA[agModeA];
  const Unsigned nLocalAGModeSlices = localShapeA[agModeA];

  const Unsigned maxCopySliceSize = Max(1, prod(maxLocalShapeA, 0, agModeA));
  const Unsigned copySliceSize = prod(localShapeA, 0, agModeA);

  Unsigned outerSliceNum, agModeSliceNum; //Which slice of which wrap of which process are we packing
  Unsigned outerSendBufOff, agModeSendBufOff;
  Unsigned outerDataBufOff, agModeDataBufOff;
  Unsigned startSendBuf, startDataBuf;

//  printf("MemCopy info:\n");
//  printf("    nMaxOuterSlices: %d\n", nMaxOuterSlices);
//  printf("    nMaxAGModeSlices: %d\n", nMaxAGModeSlices);
//  printf("    maxCopySliceSize: %d\n", maxCopySliceSize);
//  printf("    copySliceSize: %d\n", copySliceSize);
  for(outerSliceNum = 0; outerSliceNum < nMaxOuterSlices; outerSliceNum++){
      if(outerSliceNum >= nLocalOuterSlices)
          break;
      outerSendBufOff = maxCopySliceSize * nMaxAGModeSlices * outerSliceNum;
      outerDataBufOff = copySliceSize * nLocalAGModeSlices * outerSliceNum;

//      printf("        outerSliceNum: %d\n", outerSliceNum);
//      printf("        outerSendBufOff: %d\n", outerSendBufOff);
//      printf("        outerDataBufOff: %d\n", outerDataBufOff);

      for(agModeSliceNum = 0; agModeSliceNum < nMaxAGModeSlices; agModeSliceNum++){
          if(agModeSliceNum >= nLocalAGModeSlices)
              break;
          agModeSendBufOff = maxCopySliceSize * agModeSliceNum;
          agModeDataBufOff = copySliceSize * agModeSliceNum;

//          printf("          agModeSliceNum: %d\n", agModeSliceNum);
//          printf("          agModeSendBufOff: %d\n", agModeSendBufOff);
//          printf("          agModeDataBufOff: %d\n", agModeDataBufOff);
          startSendBuf = outerSendBufOff + agModeSendBufOff;
          startDataBuf = outerDataBufOff + agModeDataBufOff;

//          printf("          startSendBuf: %d\n", startSendBuf);
//          printf("          startDataBuf: %d\n", startDataBuf);
          MemCopy(&(sendBuf[startSendBuf]), &(dataBuf[startDataBuf]), copySliceSize);
      }
  }
//  printf("packed sendBuf: ");
//  for(Unsigned i = 0; i < prod(maxLocalShapeA); i++)
//      printf("%d ", sendBuf[i]);
//  printf("\n");
}

template <typename T>
void PackA2ADoubleIndexSendBuf(const DistTensor<T>& B, const DistTensor<T>& A, const std::pair<Index, Index>& a2aIndices, const std::pair<ModeArray, ModeArray >& commGroups, T * const sendBuf){
    Unsigned i;
    const Unsigned order = A.Order();
    const Location start(order, 0);
    const T* dataBuf = A.LockedBuffer(start);

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
    Unsigned a2aMode1SliceNum, midSliceNum, a2aMode2SliceNum, outerSliceNum;  //Which slice we are packing for indexK
    Unsigned a2aMode1SendBufOff, midSendBufOff, a2aMode2SendBufOff, outerSendBufOff;  //Offsets used to index into data arrays
    Unsigned a2aMode1DataBufOff, midDataBufOff, a2aMode2DataBufOff, outerDataBufOff;  //Offsets used to index into data arrays
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

    	//Determine the global index of this first element we are packing
    	Location startPackElemLoc = myFirstLoc;
    	for(i = 0; i < order; i++){
    		startPackElemLoc[i] += packElemMultiLoc[i] * gvA.ModeWrapStride(i);
    	}

    	//If we run over the edge, don't try to pack the global element
    	if(AnyElemwiseGreaterThanEqualTo(startPackElemLoc, A.Shape()))
    	    continue;

    	//Determine the Multiloc of the process that owns this element
    	Location owningProcGVB = B.DetermineOwner(startPackElemLoc);
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

    const ObjShape commGridSlice = FilterVector(B.Grid().Shape(), commModes);
    const Unsigned nRedistProcs = prod(commGridSlice);

//    printf("packed sendBuf: ");
//    for(Unsigned i = 0; i < prod(packLocalShape) * nRedistProcs; i++)
//        printf("%d ", sendBuf[i]);
//    printf("\n");
}

#define PROTO(T) \
        template void PackPermutationSendBuf(const DistTensor<T>& B, const DistTensor<T>& A, const Index permuteIndex, T * const sendBuf); \
		template void PackPartialRSSendBuf(const DistTensor<T>& B, const DistTensor<T>& A, const Index reduceScatterIndex, T * const sendBuf); \
        template void PackRSSendBuf(const DistTensor<T>& B, const DistTensor<T>& A, const Index reduceIndex, const Index scatterIndex, T * const sendBuf); \
        template void PackAGSendBuf(const DistTensor<T>& A, const Index allGatherIndex, T * const sendBuf, const ModeArray& redistModes); \
        template void PackAGSendBuf(const DistTensor<T>& A, const Index allGatherIndex, T * const sendBuf); \
        template void PackA2ADoubleIndexSendBuf(const DistTensor<T>& B, const DistTensor<T>& A, const std::pair<Index, Index>& a2aIndices, const std::pair<ModeArray, ModeArray >& commGroups, T * const sendBuf);

PROTO(int)
PROTO(float)
PROTO(double)

} // namespace tmen
