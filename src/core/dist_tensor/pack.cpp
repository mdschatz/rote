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
void PackPermutationSendBuf(const DistTensor<T>& B, const DistTensor<T>& A, const Int permuteIndex, T * const sendBuf)
{
    const std::vector<Int> start(A.Order(), 0);
    const T* dataBuf = A.LockedBuffer(start);

    const int permuteModeA = A.ModeOfIndex(permuteIndex);
    const int permuteModeB = B.ModeOfIndex(permuteIndex);

    const tmen::GridView gvA = A.GridView();
    const tmen::GridView gvB = B.GridView();

    const std::vector<Int> localShapeA = A.LocalShape(); //Shape of the local tensor we are packing
    const std::vector<Int> maxLocalShapeA = MaxLengths(A.Shape(), gvA.Shape());
    const std::vector<Int> maxLocalShapeB = MaxLengths(B.Shape(), gvB.Shape());

    const int pModeLocalDim = A.LocalDimension(permuteModeA); //Local version
    const int pModeLocalStride = A.LocalModeStride(permuteModeA);

    //Calculate number of local slices and slice size we must pack per proc per wrap
    const int nLocalSlices = Max(1, prod(localShapeA, permuteModeA + 1));
    const int nMaxSlices = Max(1, prod(maxLocalShapeA, permuteModeA + 1)); //Cover the case where scatterMode is the last one (in which case we do need 1 slice)

    const int maxCopySliceSize = Max(1, prod(maxLocalShapeA, 0, permuteModeA + 1));
    const int copySliceSize = pModeLocalStride * pModeLocalDim;
    const int nMaxElemsPerProc = prod(maxLocalShapeB);

    int sliceNum; //Which slice of which wrap of which process are we packing
    int offSliceSendBuf;  //Offsets used to index into send buf
    int offSliceDataBuf;  //Offsets used to index into data buf
    int startSendBuf, startDataBuf;

    for(sliceNum = 0; sliceNum < nMaxSlices; sliceNum++){
        offSliceSendBuf = maxCopySliceSize * sliceNum;
        offSliceDataBuf = copySliceSize * sliceNum;
        if(sliceNum >= nLocalSlices)
            break;
        startSendBuf = offSliceSendBuf;
        startDataBuf = offSliceDataBuf;
        //printf("startSendBuf: %d startDataBuf: %d copySliceSize: %d\n", startSendBuf, startDataBuf, copySliceSize);
        MemCopy(&(sendBuf[startSendBuf]), &(dataBuf[startDataBuf]), copySliceSize);
    }
//        printf("packing %d elems\n", nModeProcs * nMaxElemsPerProc);
//        std::ostringstream msg;
//        msg << "send'd data: [" << sendBuf[0];
//        for (int i = 1; i < nMaxElemsPerProc * nModeProcs; i++)
//            msg << ", " << sendBuf[i];
//        msg << "]" << std::endl;
//        std::cout << msg.str();
}

template <typename T>
void PackPartialRSSendBuf(const DistTensor<T>& B, const DistTensor<T>& A, const Int reduceScatterIndex, T * const sendBuf)
{
    PackRSSendBuf(B, A, reduceScatterIndex, reduceScatterIndex, sendBuf);
}

//Only called when fully reducing an index
//NOTE: Looks an awful lot like PackAGSendBuf...
//TODO: Merge with PackAGSendBuf?
//TODO: Make this work with blocks (more general is commented out code
template <typename T>
void PackRSSendBuf(const DistTensor<T>& B, const DistTensor<T>& A, const Int reduceIndex, const Int scatterIndex, T * const sendBuf)
{
//    printf("A has %d elems to pack\n", prod(A.LocalShape()));
    const std::vector<Int> start(A.Order(), 0);
    const T* dataBuf = A.LockedBuffer(start);

    const int reduceModeA = A.ModeOfIndex(reduceIndex);
    const int scatterModeA = A.ModeOfIndex(scatterIndex);

    const int scatterModeB = B.ModeOfIndex(scatterIndex);

    const tmen::GridView gvA = A.GridView();
    const tmen::GridView gvB = B.GridView();

    const int nModeProcs = gvA.Dimension(reduceModeA);
    const int sModeGlobalDim = B.Dimension(scatterModeB);

    const std::vector<Int> localShapeA = A.LocalShape(); //Shape of the local tensor we are packing
    const std::vector<Int> maxLocalShapeA = MaxLengths(A.Shape(), gvA.Shape());
    const std::vector<Int> maxLocalShapeB = MaxLengths(B.Shape(), gvB.Shape());

    const int sModeLocalDim = A.LocalDimension(scatterModeA); //Local version
    const int sModeLocalStride = A.LocalModeStride(scatterModeA);

    //Calculate number of local slices and slice size we must pack per proc per wrap
    const int nLocalSlices = Max(1, prod(localShapeA, scatterModeA + 1));
    const int nMaxSlices = Max(1, prod(maxLocalShapeA, scatterModeA + 1)); //Cover the case where scatterMode is the last one (in which case we do need 1 slice)

    const int nMaxWraps = MaxLength(sModeGlobalDim, nModeProcs);
    const int maxCopySliceSize = Max(1, prod(maxLocalShapeA, 0, scatterModeA));
    const int copySliceSize = sModeLocalStride;
    const int nMaxElemsPerProc = prod(maxLocalShapeB);

    int procNum, wrapNum, sliceNum; //Which slice of which wrap of which process are we packing
    int offSliceSendBuf, offWrapSendBuf;  //Offsets used to index into send buf
    int offSliceDataBuf, offWrapDataBuf;  //Offsets used to index into data buf
    int startSendBuf, startDataBuf;

    for(sliceNum = 0; sliceNum < nMaxSlices; sliceNum++){
    	offSliceSendBuf = maxCopySliceSize * nMaxWraps * sliceNum;
    	offSliceDataBuf = copySliceSize * sModeLocalDim * sliceNum;
    	if(sliceNum >= nLocalSlices)
    		break;
    	for(wrapNum = 0; wrapNum < nMaxWraps; wrapNum++){
    		offWrapSendBuf = maxCopySliceSize * wrapNum;
    		offWrapDataBuf = copySliceSize * nModeProcs * wrapNum;
    		for(procNum = 0; procNum < nModeProcs; procNum++){
    			if(wrapNum * nModeProcs + procNum >= sModeLocalDim)
    				break;
    			startSendBuf = offSliceSendBuf + offWrapSendBuf + (nMaxElemsPerProc * procNum);
    			startDataBuf = offSliceDataBuf + offWrapDataBuf + (procNum * copySliceSize);
    			//printf("startSendBuf: %d startDataBuf: %d copySliceSize: %d\n", startSendBuf, startDataBuf, copySliceSize);
    			MemCopy(&(sendBuf[startSendBuf]), &(dataBuf[startDataBuf]), copySliceSize);
    		}
    	}
    }

//    printf("packing %d elems\n", nModeProcs * nMaxElemsPerProc);
//    std::ostringstream msg;
//    msg << "send'd data: [" << sendBuf[0];
//    for (int i = 1; i < nMaxElemsPerProc * nModeProcs; i++)
//        msg << ", " << sendBuf[i];
//    msg << "]" << std::endl;
//    std::cout << msg.str();
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
//  printf("A has %d elems to pack\n", prod(A.LocalShape()));
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
	  //printf("offSliceSendBuf: %d offSliceDataBuf: %d copySliceSize: %d\n", offSliceSendBuf, offSliceDataBuf, copySliceSize);
	  MemCopy(&(sendBuf[offSliceSendBuf]), &(dataBuf[offSliceDataBuf]), copySliceSize);
  }
//  printf("packing %d elems\n", prod(maxLocalShape));
//  std::ostringstream msg;
//  msg << "send'd data: [" << sendBuf[0];
//  for (int i = 1; i < prod(maxLocalShape); i++)
//      msg << ", " << sendBuf[i];
//  msg << "]" << std::endl;
//  std::cout << msg.str();
}

//TODO: HACK HACK HACK THIS IS A HACK. I SEND THE GLOBAL LOC OF THE FIRST ELEMENT I SEND TO EACH PROCESS TO MAKE LIFE EASIER
template <typename T>
void PackA2ADoubleIndexSendBuf(const DistTensor<T>& B, const DistTensor<T>& A, const std::pair<int, int>& a2aIndices, const std::pair<std::vector<int>, std::vector<int> >& commGroups, T * const sendBuf, std::vector<std::vector<int> >& sendFirstLocs){
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

    const int nRedistProcs = prod(FilterVector(g.Shape(), commModes));

    const std::vector<Int> localShape = A.LocalShape();
    const std::vector<Int> maxLocalShape = MaxLengths(A.Shape(), gvA.Shape());

    //Slices we can directly copy
    const int nMaxContigSlices = Max(1, prod(maxLocalShape) / prod(maxLocalShape, a2aMode1));
    const int nLocalContigSlices = Max(1, prod(localShape) / prod(localShape, a2aMode1));

    //Slices of a2aMode1
    const int nMaxA2AMode1Slices = maxLocalShape[a2aMode1];
    const int nLocalA2AMode1Slices = localShape[a2aMode1];

    //Slices between a2aMode1 and a2aMode2
    const int nMaxMidSlices = Max(1, prod(maxLocalShape, a2aMode1 + 1) / prod(maxLocalShape, a2aMode2));
    const int nLocalMidSlices = Max(1, prod(localShape, a2aMode1 + 1) / prod(localShape, a2aMode2));

    //Slices of a2aMode2
    const int nMaxA2AMode2Slices = maxLocalShape[a2aMode2];
    const int nLocalA2AMode2Slices = localShape[a2aMode2];

    //All remaining slices
    const int nMaxOuterSlices = Max(1, prod(maxLocalShape, a2aMode2 + 1));
    const int nLocalOuterSlices = Max(1, prod(localShape, a2aMode1 + 1));

    const int copySliceSize = A.LocalModeStride(a2aMode1);
    const int nElemsPerProc = prod(maxLocalShape);

    //Various counters used to offset in data arrays
    int contigSliceNum, a2aMode1SliceNum, midSliceNum, a2aMode2SliceNum, outerSliceNum;  //Which slice we are packing for indexK
    int contigSendBufOff, a2aMode1SendBufOff, midSendBufOff, a2aMode2SendBufOff, outerSendBufOff;  //Offsets used to index into data arrays
    int contigDataBufOff, a2aMode1DataBufOff, midDataBufOff, a2aMode2DataBufOff, outerDataBufOff;  //Offsets used to index into data arrays
    int packElemSendBufOff, packElemDataBufOff;
    int startSendBuf, startDataBuf;

    //a2aMode1 and a2aMode2 have different increments per slice because their distributions change
    const int a2aMode1PackStride = modePackStrides[a2aMode1];
    const int a2aMode2PackStride = modePackStrides[a2aMode2];

    //Stores the first packed multi-loc sent to each process in communicator
    sendFirstLocs.reserve(nRedistProcs);
    sendFirstLocs.resize(nRedistProcs);
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

    	//Determine the Multiloc of the process that owns this element
    	std::vector<int> owningProcGVB = B.DetermineOwner(startPackElemLoc);
    	std::vector<int> owningProcG = GridViewLoc2GridLoc(owningProcGVB, gvB);
    	int owningProc = LinearIndex(FilterVector(owningProcG, commModes), Dimensions2Strides(FilterVector(gridShape, commModes)));

        //Record the Multiloc of this first element for the associated process
        sendFirstLocs[owningProc].reserve(order);
        sendFirstLocs[owningProc].resize(order);
        sendFirstLocs[owningProc] = startPackElemLoc;

        //Find the local location of the global starting element we are now packing
        std::vector<int> localLoc = A.Global2LocalIndex(sendFirstLocs[owningProc]);

        //Update the corresponding offsets
        packElemSendBufOff = nElemsPerProc * owningProc;
        packElemDataBufOff = LinearIndex(localLoc, Dimensions2Strides(localShape));

        //Now that we have figured out the starting point, begin copying the entire slice from this element
        for(outerSliceNum = 0; outerSliceNum < nMaxOuterSlices; outerSliceNum++){
            if(outerSliceNum >= nLocalOuterSlices)
                break;
            outerSendBufOff = copySliceSize * (nMaxA2AMode1Slices / a2aMode1PackStride) * nMaxMidSlices * (nMaxA2AMode2Slices / a2aMode2PackStride) * outerSliceNum;
            outerDataBufOff = copySliceSize * nLocalA2AMode1Slices * nLocalMidSlices * nLocalA2AMode2Slices * outerSliceNum;

            //offSendBufSlice2 = copySliceSize * nMaxSlices1 * sliceNum2;
            //offDataBufSlice2 = copySliceSize * nLocalSlices1 * a2aMode1PackWrapStride * a2aMode2PackWrapStride * sliceNum2;
            for(a2aMode2SliceNum = 0; a2aMode2SliceNum < nMaxA2AMode2Slices; a2aMode2SliceNum += a2aMode2PackStride){
                if(a2aMode2SliceNum >= nLocalA2AMode2Slices)
                    break;
                a2aMode2SendBufOff = copySliceSize * (nMaxA2AMode1Slices / a2aMode1PackStride) * nMaxMidSlices * (a2aMode2SliceNum / a2aMode2PackStride);
                a2aMode2DataBufOff = copySliceSize * nLocalA2AMode1Slices * nLocalMidSlices * a2aMode2SliceNum;

                for(midSliceNum = 0; midSliceNum < nMaxMidSlices; midSliceNum++){
                    if(midSliceNum >= nLocalMidSlices)
                        break;
                    midSendBufOff = copySliceSize * (nMaxA2AMode1Slices / a2aMode1PackStride) * midSliceNum;
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

    printf("dataBuf:");
    for(int i = 0; i < prod(localShape); i++){
        printf(" %d", dataBuf[i]);
    }
    printf("\n");

    printf("Packed sendBuf:");
    for(int i = 0; i < prod(localShape) * nRedistProcs; i++){
        printf(" %d", sendBuf[i]);
    }
    printf("\n");
}

#define PROTO(T) \
        template void PackPermutationSendBuf(const DistTensor<T>& B, const DistTensor<T>& A, const Int permuteIndex, T * const sendBuf); \
		template void PackPartialRSSendBuf(const DistTensor<T>& B, const DistTensor<T>& A, const int reduceScatterIndex, T * const sendBuf); \
        template void PackRSSendBuf(const DistTensor<T>& B, const DistTensor<T>& A, const int reduceIndex, const int scatterIndex, T * const sendBuf); \
        template void PackAGSendBuf(const DistTensor<T>& A, const int allGatherIndex, T * const sendBuf); \
        template void PackA2ADoubleIndexSendBuf(const DistTensor<T>& B, const DistTensor<T>& A, const std::pair<int, int>& a2aIndices, const std::pair<std::vector<int>, std::vector<int> >& commGroups, T * const sendBuf, std::vector<std::vector<int> >& sendFirstLocs);

PROTO(int)
PROTO(float)
PROTO(double)

} // namespace tmen
