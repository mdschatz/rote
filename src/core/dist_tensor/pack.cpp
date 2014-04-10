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
void PackPermutationSendBuf(const DistTensor<T>& B, const DistTensor<T>& A, const Int permuteIndex, T * const sendBuf)
{
    const std::vector<Int> start(A.Order(), 0);
    const T* dataBuf = A.LockedBuffer(start);

    const int pModeA = A.ModeOfIndex(permuteIndex);

    const tmen::GridView gvA = A.GridView();

    const int nRedistProcs = gvA.Dimension(pModeA);

    //Shape of the local tensor we are packing
    const std::vector<Int> maxLocalShapeA = MaxLengths(A.Shape(), gvA.Shape());
    const std::vector<Int> localShapeA = A.LocalShape();

    //Calculate number of outer slices to pack
    const int nMaxOuterSlices = Max(1, prod(maxLocalShapeA, pModeA + 1));
    const int nLocalOuterSlices = prod(localShapeA, pModeA + 1);

    //Calculate number of rsMode slices to pack
    const int nMaxPModeSlices = maxLocalShapeA[pModeA];
    const int nLocalPModeSlices = localShapeA[pModeA];
    const int pModePackStride = nRedistProcs;


    const int maxCopySliceSize = Max(1, prod(maxLocalShapeA, 0, pModeA));
    const int copySliceSize = prod(localShapeA, 0, pModeA);

    const int nMaxElemsPerProc = prod(maxLocalShapeA);

    int outerSliceNum, pModeSliceNum, elemSliceNum; //Which slice of which wrap of which process are we packing
    int elemSendBufOff, elemDataBufOff;
    int outerSendBufOff, pModeSendBufOff;
    int outerDataBufOff, pModeDataBufOff;
    int startSendBuf, startDataBuf;

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
void PackPartialRSSendBuf(const DistTensor<T>& B, const DistTensor<T>& A, const Int reduceScatterIndex, T * const sendBuf)
{
    const std::vector<Int> start(A.Order(), 0);
    const T* dataBuf = A.LockedBuffer(start);

    printf("dataBuf: ");
    for(int i = 0; i < prod(A.LocalShape()); i++){
        printf("%d ", dataBuf[i]);
    }
    printf("\n");

    const int rsModeA = A.ModeOfIndex(reduceScatterIndex);
    const int rsModeB = B.ModeOfIndex(reduceScatterIndex);

    const tmen::GridView gvA = A.GridView();
    const tmen::GridView gvB = B.GridView();

    const int nRedistProcs = gvA.Dimension(rsModeA);

    //Shape of the local tensor we are packing
    const std::vector<Int> maxLocalShapeA = MaxLengths(A.Shape(), gvA.Shape());
    const std::vector<Int> localShapeA = A.LocalShape();

    //Calculate number of outer slices to pack
    const int nMaxOuterSlices = Max(1, prod(maxLocalShapeA, rsModeA + 1));
    const int nLocalOuterSlices = prod(localShapeA, rsModeA + 1);

    //Calculate number of rsMode slices to pack
    const int nMaxRSModeSlices = maxLocalShapeA[rsModeA];
    const int nLocalRSModeSlices = localShapeA[rsModeA];
    const int rsModePackStride = nRedistProcs;

    //Number of processes we have to pack for
    const int nElemSlices = nRedistProcs;

    const int maxCopySliceSize = Max(1, prod(maxLocalShapeA, 0, rsModeA));
    const int copySliceSize = prod(localShapeA, 0, rsModeA);

    const int nMaxElemsPerProc = prod(maxLocalShapeA) / nRedistProcs;

    int outerSliceNum, rsModeSliceNum, elemSliceNum; //Which slice of which wrap of which process are we packing
    int elemSendBufOff, elemDataBufOff;
    int outerSendBufOff, rsModeSendBufOff;
    int outerDataBufOff, rsModeDataBufOff;
    int startSendBuf, startDataBuf;

    printf("MemCopy info:\n");
    printf("    nMaxOuterSlices: %d\n", nMaxOuterSlices);
    printf("    nMaxRSModeSlices: %d\n", nMaxRSModeSlices);
    printf("    rsModePackStride: %d\n", rsModePackStride);
    printf("    maxCopySliceSize: %d\n", maxCopySliceSize);
    printf("    copySliceSize: %d\n", copySliceSize);
    for(elemSliceNum = 0; elemSliceNum < nElemSlices; elemSliceNum++){
        elemSendBufOff = prod(maxLocalShapeA) * elemSliceNum;
        elemDataBufOff = copySliceSize * elemSliceNum;

        printf("      elemSliceNum: %d\n", elemSliceNum);
        printf("      elemSendBufOff: %d\n", elemSendBufOff);
        printf("      elemDataBufOff: %d\n", elemDataBufOff);

        for(outerSliceNum = 0; outerSliceNum < nMaxOuterSlices; outerSliceNum++ ){
            if(outerSliceNum >= nLocalOuterSlices)
                break;
            outerSendBufOff = maxCopySliceSize * Max(1, nMaxRSModeSlices / rsModePackStride) * outerSliceNum;
            outerDataBufOff = copySliceSize * nLocalRSModeSlices * outerSliceNum;

            printf("        outerSliceNum: %d\n", outerSliceNum);
            printf("        outerSendBufOff: %d\n", outerSendBufOff);
            printf("        outerDataBufOff: %d\n", outerDataBufOff);

                for(rsModeSliceNum = 0; rsModeSliceNum < nMaxRSModeSlices; rsModeSliceNum += rsModePackStride){
                    if(rsModeSliceNum + elemSliceNum >= nLocalRSModeSlices)
                        break;
                    rsModeSendBufOff = maxCopySliceSize * (rsModeSliceNum / rsModePackStride);
                    rsModeDataBufOff = copySliceSize * rsModeSliceNum;

                    printf("          rsModeSliceNum: %d\n", rsModeSliceNum);
                    printf("          rsModeSendBufOff: %d\n", rsModeSendBufOff);
                    printf("          rsModeDataBufOff: %d\n", rsModeDataBufOff);
                    startSendBuf = elemSendBufOff + outerSendBufOff + rsModeSendBufOff;
                    startDataBuf = elemDataBufOff + outerDataBufOff + rsModeDataBufOff;

                    printf("          startSendBuf: %d\n", startSendBuf);
                    printf("          startDataBuf: %d\n", startDataBuf);
                    MemCopy(&(sendBuf[startSendBuf]), &(dataBuf[startDataBuf]), copySliceSize);
                }
        }
    }
    printf("packed sendBuf: ");
    for(int i = 0; i < prod(maxLocalShapeA) * nRedistProcs; i++)
        printf("%d ", sendBuf[i]);
    printf("\n");
}

//Only called when fully reducing an index
//NOTE: Looks an awful lot like PackAGSendBuf...
//TODO: Merge with PackAGSendBuf?
template <typename T>
void PackRSSendBuf(const DistTensor<T>& B, const DistTensor<T>& A, const Int reduceIndex, const Int scatterIndex, T * const sendBuf)
{
    const std::vector<Int> start(A.Order(), 0);
    const T* dataBuf = A.LockedBuffer(start);

    printf("dataBuf: ");
    for(int i = 0; i < prod(A.LocalShape()); i++){
        printf("%d ", dataBuf[i]);
    }
    printf("\n");

    const int rModeA = A.ModeOfIndex(reduceIndex);
    const int sModeA = A.ModeOfIndex(scatterIndex);

    const tmen::GridView gvA = A.GridView();
    const tmen::GridView gvB = B.GridView();

    const int nRedistProcs = gvA.Dimension(rModeA);

    //Shape of the local tensor we are packing
    const std::vector<Int> maxLocalShapeA = MaxLengths(A.Shape(), gvA.Shape());
    const std::vector<Int> localShapeA = A.LocalShape();

    //Calculate number of outer slices to pack
    const int nMaxOuterSlices = Max(1, prod(maxLocalShapeA, sModeA + 1));
    const int nLocalOuterSlices = prod(localShapeA, sModeA + 1);

    //Calculate number of sMode slices to pack
    const int nMaxSModeSlices = maxLocalShapeA[sModeA];
    const int nLocalSModeSlices = localShapeA[sModeA];
    const int sModePackStride = nRedistProcs;

    //Number of processes we have to pack for
    const int nElemSlices = nRedistProcs;

    const int maxCopySliceSize = Max(1, prod(maxLocalShapeA, 0, sModeA));
    const int copySliceSize = prod(localShapeA, 0, sModeA);

    const int nMaxElemsPerProc = prod(maxLocalShapeA) / nRedistProcs;

    int outerSliceNum, sModeSliceNum, elemSliceNum; //Which slice of which wrap of which process are we packing
    int elemSendBufOff, elemDataBufOff;
    int outerSendBufOff, sModeSendBufOff;
    int outerDataBufOff, sModeDataBufOff;
    int startSendBuf, startDataBuf;


    printf("MemCopy info:\n");
    printf("    nMaxOuterSlices: %d\n", nMaxOuterSlices);
    printf("    nMaxSModeSlices: %d\n", nMaxSModeSlices);
    printf("    sModePackStride: %d\n", sModePackStride);
    printf("    maxCopySliceSize: %d\n", maxCopySliceSize);
    printf("    copySliceSize: %d\n", copySliceSize);
    for(elemSliceNum = 0; elemSliceNum < nElemSlices; elemSliceNum++){
        elemSendBufOff = prod(maxLocalShapeA) * elemSliceNum;
        elemDataBufOff = copySliceSize * elemSliceNum;

        printf("      elemSliceNum: %d\n", elemSliceNum);
        printf("      elemSendBufOff: %d\n", elemSendBufOff);
        printf("      elemDataBufOff: %d\n", elemDataBufOff);

        for(outerSliceNum = 0; outerSliceNum < nMaxOuterSlices; outerSliceNum++ ){
            if(outerSliceNum >= nLocalOuterSlices)
                break;
            outerSendBufOff = maxCopySliceSize * Max(1, nMaxSModeSlices / sModePackStride) * outerSliceNum;
            outerDataBufOff = copySliceSize * nLocalSModeSlices * outerSliceNum;

            printf("        outerSliceNum: %d\n", outerSliceNum);
            printf("        outerSendBufOff: %d\n", outerSendBufOff);
            printf("        outerDataBufOff: %d\n", outerDataBufOff);

                for(sModeSliceNum = 0; sModeSliceNum < nMaxSModeSlices; sModeSliceNum += sModePackStride){
                    if(sModeSliceNum + elemSliceNum >= nLocalSModeSlices)
                        break;
                    sModeSendBufOff = maxCopySliceSize * (sModeSliceNum / sModePackStride);
                    sModeDataBufOff = copySliceSize * sModeSliceNum;

                    printf("          sModeSliceNum: %d\n", sModeSliceNum);
                    printf("          sModeSendBufOff: %d\n", sModeSendBufOff);
                    printf("          sModeDataBufOff: %d\n", sModeDataBufOff);
                    startSendBuf = elemSendBufOff + outerSendBufOff + sModeSendBufOff;
                    startDataBuf = elemDataBufOff + outerDataBufOff + sModeDataBufOff;

                    printf("          startSendBuf: %d\n", startSendBuf);
                    printf("          startDataBuf: %d\n", startDataBuf);
                    MemCopy(&(sendBuf[startSendBuf]), &(dataBuf[startDataBuf]), copySliceSize);
                }
        }
    }

    printf("packed sendBuf: ");
    for(int i = 0; i < prod(maxLocalShapeA) * nRedistProcs; i++)
        printf("%d ", sendBuf[i]);
    printf("\n");
}

//TODO: Adjust this for blocks (not contiguous tensors)
//For Allgather (without blocks) we just need to directly copy the data
//Pack according to the following scheme:
//foreach sliceNum in nSlices:
//  offSliceSendBuf = sliceNum * maxLocalDim
//  offSliceDataBuf = sliceNum * localDim
//  memcpy(sendBuf[offSliceSendBuf], dataBuf[offSliceDataBuf], sliceSize)
//TODO: Make this work with blocks
template <typename T>
void PackAGSendBuf(const DistTensor<T>& A, const Int allGatherIndex, T * const sendBuf)
{
  const tmen::GridView gridView = A.GridView();

  const std::vector<Int> start(A.Order(), 0);
  const T* dataBuf = A.LockedBuffer(start);

  const tmen::GridView gv = A.GridView();

  const int allGatherMode = A.ModeOfIndex(allGatherIndex);

  const std::vector<Int> localShape = A.LocalShape();         //Shape of the local tensor we are packing
  const std::vector<Int> maxLocalShape = MaxLengths(A.Shape(), gv.Shape());

  const int agModeLocalDim = A.LocalDimension(allGatherMode); //Local version
  const int agModeMaxLocalDim = maxLocalShape[allGatherMode];
  const int agModeLocalStride = A.LocalModeStride(allGatherMode);

  //Calculate number of local slices and slice size we must pack per proc per wrap
  const int nLocalSlices = Max(1, prod(localShape, allGatherMode + 1));
  const int nMaxSlices = Max(1, prod(maxLocalShape, allGatherMode + 1));    //Cover the case where allGatherMode is the last one (in which case we do need 1 slice)

  const int copySliceSize = agModeLocalStride * agModeLocalDim;

  int sliceNum;  //Which slice we are packing
  int offSliceSendBuf, offSliceDataBuf;  //Offsets used to index into data arrays

  for(sliceNum = 0; sliceNum < nMaxSlices; sliceNum++){
	  offSliceSendBuf = sliceNum * agModeMaxLocalDim;
	  offSliceDataBuf = sliceNum * agModeLocalDim;
	  if(sliceNum >= nLocalSlices){
		  break;
	  }
	  MemCopy(&(sendBuf[offSliceSendBuf]), &(dataBuf[offSliceDataBuf]), copySliceSize);
  }
}

template <typename T>
void PackA2ADoubleIndexSendBuf(const DistTensor<T>& B, const DistTensor<T>& A, const std::pair<int, int>& a2aIndices, const std::pair<std::vector<int>, std::vector<int> >& commGroups, T * const sendBuf){
    const int order = A.Order();
    const std::vector<int> start(order, 0);
    const T* dataBuf = A.LockedBuffer(start);

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

    std::vector<int> myGridLoc = g.Loc();
    std::vector<int> gridShape = g.Shape();

    std::vector<int> modeLCMs(order);
    for(int i = 0; i < order; i++)
    	modeLCMs[i] = tmen::LCM(gvA.ModeWrapStride(i), gvB.ModeWrapStride(i));

    std::vector<int> modePackStrides(order);
    for(int i = 0; i < order; i++){
    	modePackStrides[i] = modeLCMs[i] / gvA.ModeWrapStride(i);
    }

    const std::vector<Int> localShape = A.LocalShape();
    const std::vector<Int> packLocalShape = MaxLengths(A.Shape(), gvA.Shape());

    //Slices of a2aMode1
    const int nMaxA2AMode1Slices = packLocalShape[a2aMode1];
    const int nLocalA2AMode1Slices = localShape[a2aMode1];

    //Slices between a2aMode1 and a2aMode2
    const int nMaxMidSlices = Max(1, prod(packLocalShape, a2aMode1 + 1, a2aMode2));
    const int nLocalMidSlices = prod(localShape, a2aMode1 + 1, a2aMode2);

    //Slices of a2aMode2
    const int nMaxA2AMode2Slices = packLocalShape[a2aMode2];
    const int nLocalA2AMode2Slices = localShape[a2aMode2];

    //All remaining slices
    const int nMaxOuterSlices = Max(1, prod(packLocalShape, a2aMode2 + 1));
    const int nLocalOuterSlices = prod(localShape, a2aMode2 + 1);

    const int copySliceSize = A.LocalModeStride(a2aMode1);
    const int nElemsPerProc = prod(packLocalShape);

    //Various counters used to offset in data arrays
    int a2aMode1SliceNum, midSliceNum, a2aMode2SliceNum, outerSliceNum;  //Which slice we are packing for indexK
    int a2aMode1SendBufOff, midSendBufOff, a2aMode2SendBufOff, outerSendBufOff;  //Offsets used to index into data arrays
    int a2aMode1DataBufOff, midDataBufOff, a2aMode2DataBufOff, outerDataBufOff;  //Offsets used to index into data arrays
    int packElemSendBufOff, packElemDataBufOff;
    int startSendBuf, startDataBuf;

    //a2aMode1 and a2aMode2 have different increments per slice because their distributions change
    const int a2aMode1PackStride = modePackStrides[a2aMode1];
    const int a2aMode2PackStride = modePackStrides[a2aMode2];

    std::vector<int> myFirstLoc = A.ModeShifts();

    int packElemNum;
    const int nPackElems = prod(modePackStrides);

    for(packElemNum = 0; packElemNum < nPackElems; packElemNum++){
    	std::vector<int> packElemMultiLoc = LinearLoc2Loc(packElemNum, modePackStrides);

    	//Determine the global index of this first element we are packing
    	std::vector<int> startPackElemLoc = myFirstLoc;
    	for(int i = 0; i < order; i++){
    		startPackElemLoc[i] += packElemMultiLoc[i] * gvA.ModeWrapStride(i);
    	}

    	//If we run over the edge, don't try to pack the global element
    	if(AnyElemwiseGreaterThanEqualTo(startPackElemLoc, A.Shape()))
    	    continue;

    	//Determine the Multiloc of the process that owns this element
    	std::vector<int> owningProcGVB = B.DetermineOwner(startPackElemLoc);
    	std::vector<int> owningProcG = GridViewLoc2GridLoc(owningProcGVB, gvB);
    	int owningProc = LinearIndex(FilterVector(owningProcG, commModes), Dimensions2Strides(FilterVector(gridShape, commModes)));

        //Find the local location of the global starting element we are now packing
        std::vector<int> localLoc = A.Global2LocalIndex(startPackElemLoc);

        //Update the corresponding offsets
        packElemSendBufOff = nElemsPerProc * owningProc;
        packElemDataBufOff = LinearIndex(localLoc, Dimensions2Strides(localShape));

        //Now that we have figured out the starting point, begin copying the entire slice from this element
        for(outerSliceNum = 0; outerSliceNum < nMaxOuterSlices; outerSliceNum++){
            if(outerSliceNum >= nLocalOuterSlices)
                break;
            outerSendBufOff = copySliceSize * Max(1, (nMaxA2AMode1Slices / a2aMode1PackStride)) * nMaxMidSlices * Max(1, (nMaxA2AMode2Slices / a2aMode2PackStride)) * outerSliceNum;
            outerDataBufOff = copySliceSize * nLocalA2AMode1Slices * nLocalMidSlices * nLocalA2AMode2Slices * outerSliceNum;

            for(a2aMode2SliceNum = 0; a2aMode2SliceNum < nMaxA2AMode2Slices; a2aMode2SliceNum += a2aMode2PackStride){
                if(a2aMode2SliceNum >= nLocalA2AMode2Slices)
                    break;
                a2aMode2SendBufOff = copySliceSize * Max(1, (nMaxA2AMode1Slices / a2aMode1PackStride)) * nMaxMidSlices * (a2aMode2SliceNum / a2aMode2PackStride);
                a2aMode2DataBufOff = copySliceSize * nLocalA2AMode1Slices * nLocalMidSlices * a2aMode2SliceNum;

                for(midSliceNum = 0; midSliceNum < nMaxMidSlices; midSliceNum++){
                    if(midSliceNum >= nLocalMidSlices)
                        break;
                    midSendBufOff = copySliceSize * Max(1, (nMaxA2AMode1Slices / a2aMode1PackStride)) * midSliceNum;
                    midDataBufOff = copySliceSize * nLocalA2AMode1Slices * midSliceNum;

                    for(a2aMode1SliceNum = 0; a2aMode1SliceNum < nMaxA2AMode1Slices; a2aMode1SliceNum += a2aMode1PackStride){
                        if(a2aMode1SliceNum >= nLocalA2AMode1Slices)
                            break;
                        a2aMode1SendBufOff = copySliceSize * (a2aMode1SliceNum / a2aMode1PackStride);
                        a2aMode1DataBufOff = copySliceSize * a2aMode1SliceNum;

                        //Down to all contiguous slices, so just copy
                        startSendBuf = packElemSendBufOff + outerSendBufOff + a2aMode2SendBufOff + midSendBufOff + a2aMode1SendBufOff;
                        startDataBuf = packElemDataBufOff + outerDataBufOff + a2aMode2DataBufOff + midDataBufOff + a2aMode1DataBufOff;

                        MemCopy(&(sendBuf[startSendBuf]), &(dataBuf[startDataBuf]), copySliceSize);
                    }
                }
            }
        }
    }
}

#define PROTO(T) \
        template void PackPermutationSendBuf(const DistTensor<T>& B, const DistTensor<T>& A, const Int permuteIndex, T * const sendBuf); \
		template void PackPartialRSSendBuf(const DistTensor<T>& B, const DistTensor<T>& A, const int reduceScatterIndex, T * const sendBuf); \
        template void PackRSSendBuf(const DistTensor<T>& B, const DistTensor<T>& A, const int reduceIndex, const int scatterIndex, T * const sendBuf); \
        template void PackAGSendBuf(const DistTensor<T>& A, const int allGatherIndex, T * const sendBuf); \
        template void PackA2ADoubleIndexSendBuf(const DistTensor<T>& B, const DistTensor<T>& A, const std::pair<int, int>& a2aIndices, const std::pair<std::vector<int>, std::vector<int> >& commGroups, T * const sendBuf);

PROTO(int)
PROTO(float)
PROTO(double)

} // namespace tmen
