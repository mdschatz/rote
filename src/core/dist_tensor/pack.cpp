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

template <typename T>
void PackA2ADoubleIndexSendBuf(const DistTensor<T>& B, const DistTensor<T>& A, const std::pair<int, int>& a2aIndices, const std::pair<std::vector<int>, std::vector<int> >& commGroups, T * const sendBuf){
    const tmen::GridView gridView = A.GridView();

    const std::vector<int> start(A.Order(), 0);
    const T* dataBuf = A.LockedBuffer(start);

    const tmen::GridView gvA = A.GridView();
    const tmen::GridView gvB = B.GridView();

    int a2aMode1 = A.ModeOfIndex(a2aIndices.first);
    int a2aMode2 = A.ModeOfIndex(a2aIndices.second);


    std::vector<int> commGroup1 = commGroups.first;
    std::vector<int> commGroup2 = commGroups.second;

    std::vector<int> distAa2aMode1 = A.ModeDist(a2aMode1);
    std::vector<int> distAa2aMode2 = A.ModeDist(a2aMode2);

    std::vector<int> distBa2aMode1 = B.ModeDist(a2aMode1);
    std::vector<int> distBa2aMode2 = B.ModeDist(a2aMode2);

    //For convenience make sure that a2aMode1 is earlier in the packing
    if(a2aMode1 > a2aMode2){
        std::swap(a2aMode1, a2aMode2);
        std::swap(commGroup1, commGroup2);
    }

    std::vector<int> commModes  = commGroup1;
    commModes.insert(commModes.end(), commGroup2.begin(), commGroup2.end());

    const int a2aMode2LCM = tmen::LCM(gvA.ModeWrapStride(a2aMode2), gvB.ModeWrapStride(a2aMode2));
    const int a2aMode1LCM = tmen::LCM(gvA.ModeWrapStride(a2aMode1), gvB.ModeWrapStride(a2aMode1));

    const int nRedistProcs = prod(FilterVector(gvA.Shape(), commModes));

    const std::vector<Int> localShape = A.LocalShape();
    std::vector<Int> maxLocalShapeB = MaxLengths(B.Shape(), gvA.Shape());

    const int a2aMode1LocalDim = A.LocalDimension(a2aMode1);
    const int a2aMode1MaxLocalDim = maxLocalShapeB[a2aMode1];
    const int a2aMode1LocalStride = A.LocalModeStride(a2aMode1);

    const int a2aMode2LocalDim = A.LocalDimension(a2aMode2);
    const int a2aMode2MaxLocalDim = maxLocalShapeB[a2aMode2];
    const int a2aMode2LocalStride = A.LocalModeStride(a2aMode2);

    const int nLocalSlices2 = Max(1, prod(localShape, a2aMode2 + 1));
    const int nMaxSlices2 = Max(1, prod(maxLocalShapeB, a2aMode2 + 1));

    //Slices1 only counts up to next a2aIndex
    const int nLocalSlices1 = Max(1, prod(localShape, a2aMode1 + 1)) / nLocalSlices2 / a2aMode2LocalDim;
    const int nMaxSlices1 = Max(1, prod(maxLocalShapeB, a2aMode1 + 1)) / nMaxSlices2 / a2aMode2MaxLocalDim;

    const int copySliceSize = a2aMode1LocalStride * a2aMode1LocalDim;

    int sliceNum1, sliceNum2;  //Which slice we are packing for indexK
    int offSendBuf, offDataBuf;  //Offsets used to index into data arrays

    int procNum;

    offSendBuf = 0;
    offDataBuf = 0;
    for(procNum = 0; procNum < nRedistProcs; procNum++){
        for(sliceNum2 = 0; sliceNum2 < nMaxSlices2; sliceNum2++){
            if(sliceNum2 >= nLocalSlices2)
                break;
            offSendBuf += copySliceSize * a2aMode1LocalDim * nLocalSlices1;
            offDataBuf += copySliceSize * a2aMode1LocalDim * nLocalSlices1 * a2aMode2LCM / a2aMode2LocalDim;
            for(sliceNum1 = 0; sliceNum1 < nMaxSlices1; sliceNum1++){
                if(sliceNum1 >= nLocalSlices1)
                    break;
                offSendBuf += copySliceSize;
                offDataBuf += copySliceSize * a2aMode1LCM / a2aMode1LocalDim;
                memcpy(&(sendBuf[offSendBuf]), &(dataBuf[offDataBuf]), copySliceSize);
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
