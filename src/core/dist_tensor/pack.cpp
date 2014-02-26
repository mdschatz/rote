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
void PackRSSendBuf(const DistTensor<T>& A, const Int reduceScatterIndex, T * const sendBuf)
{

}

template <typename T>
void PackRSSendBuf(const DistTensor<T>& A, const Int reduceIndex, const Int scatterIndex, T * const sendBuf)
{

}

template <typename T>
void UnpackRSRecvBuf(const T * const recvBuf, const Int reduceScatterIndex, DistTensor<T>& A)
{

}

template <typename T>
void UnpackRSRecvBuf(const T * const recvBuf, const Int reduceIndex, const Int scatterIndex, DistTensor<T>& A)
{

}

//TODO: Adjust this for blocks (not contiguous tensors)
//For Allgather (without blocks) we just need to directly copy the data
//Pack according to the following scheme:
//foreach sliceNum in nSlices:
//  offSliceSendBuf = sliceNum * maxLocalDim
//  offSliceDataBuf = sliceNum * localDim
//  memcpy(sendBuf[offSliceSendBuf], dataBuf[offSliceDataBuf], sliceSize)
template <typename T>
void PackAGSendBuf(const DistTensor<T>& A, const Int allGatherMode, T * const sendBuf)
{
  printf("A has %d elems to pack\n", prod(A.LocalShape()));
  const tmen::GridView gridView = A.GridView();


  std::vector<Int> start(A.Order(), 0);
  const T* dataBuf = A.LockedBuffer(start);

  const tmen::GridView gv = A.GridView();

  int agModeLocalDim = A.LocalDimension(allGatherMode); //Local version
  std::vector<Int> localShape = A.LocalShape();         //Shape of the local tensor we are packing
  int agModeLocalStride = A.LocalModeStride(allGatherMode);
  int sliceNum, nMaxSlices, nLocalSlices;  //Pack data for slice "sliceNum" (<nSlices) of wrap "wrapNum" (<nWraps) for proc "procSendNum"
  int copySliceSize;

  std::vector<Int> maxLocalShape = MaxLengths(A.Shape(), gv.Shape());
  const int agModeMaxLocalDim = maxLocalShape[allGatherMode];

  int offSliceSendBuf;  //Offsets used to index into sendBuf array
  int offSliceDataBuf;  //Offsets used to index into dataBuf array

  //Calculate number of local slices and slice size we must pack per proc per wrap
  std::vector<Int> tmp(maxLocalShape.begin() + allGatherMode + 1, maxLocalShape.end());
  std::vector<Int> tmp2(localShape.begin() + allGatherMode + 1, localShape.end());
  nLocalSlices = Max(1, prod(tmp2));
  nMaxSlices = Max(1, prod(tmp));    //Cover the case where allGatherMode is the last one (in which case we do need 1 slice)
  //TODO: Make this work with blocks (more general is commented out code
  //std::vector<Int> tmp2(localShape.begin(), localShape.begin() + allGatherMode);
  //sliceSize = prod(tmp2);
  copySliceSize = agModeLocalStride * agModeLocalDim;

  for(sliceNum = 0; sliceNum < nMaxSlices; sliceNum++){
	  offSliceSendBuf = sliceNum * agModeMaxLocalDim;
	  offSliceDataBuf = sliceNum * agModeLocalDim;
	  if(sliceNum >= nLocalSlices){
		  break;
	  }
	  printf("offSliceSendBuf: %d offSliceDataBuf: %d copySliceSize: %d\n", offSliceSendBuf, offSliceDataBuf, copySliceSize);
	  MemCopy(&(sendBuf[offSliceSendBuf]), &(dataBuf[offSliceDataBuf]), copySliceSize);
  }
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
//		ptr += max(1, prod(s1,..., s(x-1))) <-- Increment ptr in the sendBuf
template <typename T>
void UnpackAGRecvBuf(const T * const recvBuf, const Int allGatherMode, const DistTensor<T>& A, DistTensor<T>& B)
{
	printf("B can unpack %d elems\n", prod(B.LocalShape()));
    std::vector<Int> start(B.Order(), 0);
    T* dataBuf = B.Buffer(start);
    const tmen::GridView gv = A.GridView();

	int nModeProcs = gv.Dimension(allGatherMode);   //Number of procs per wrap
	int agModeGlobalDim = B.Dimension(allGatherMode);           //Number of indices in the mode we are redistributing
	int agModeLocalDim = B.LocalDimension(allGatherMode); //Local version
	std::vector<Int> localShape = B.LocalShape();         //Shape of the local tensor we are packing
	int agModeLocalStride = B.LocalModeStride(allGatherMode);
	int procRecvNum, nWraps, wrapNum, sliceNum, nMaxSlices, nLocalSlices;  //Pack data for slice "sliceNum" (<nSlices) of wrap "wrapNum" (<nWraps) for proc "procSendNum"
	int copySliceSize;

	std::vector<Int> maxRecvLocalShape = MaxLengths(A.Shape(), gv.Shape());
	const int agModeMaxLocalDim = maxRecvLocalShape[allGatherMode];
	int nElemsPerProc = prod(maxRecvLocalShape);

	int offSliceRecvBuf, offWrapRecvBuf;  //Offsets used to index into sendBuf array
	int offSliceDataBuf, offWrapDataBuf;  //Offsets used to index into dataBuf array
	int startRecvBuf, startDataBuf;

	nWraps = MaxLength(agModeGlobalDim, nModeProcs);
	//Calculate number of local slices and slice size we must pack per proc per wrap
	std::vector<Int> tmp(maxRecvLocalShape.begin() + allGatherMode + 1, maxRecvLocalShape.end());
	nMaxSlices = Max(1, prod(tmp));    //Cover the case where allGatherMode is the last one (in which case we do need 1 slice)
	std::vector<Int> tmp2(localShape.begin() + allGatherMode + 1, localShape.end());
	nLocalSlices = Max(1, prod(tmp2));
	//TODO: Make this work with blocks (more general is commented out code
	//TODO: Swap nWrap test to be more like sliceNum test (Refer to local information)
	//std::vector<Int> tmp2(localShape.begin(), localShape.begin() + allGatherMode);
	//sliceSize = prod(tmp2);
	copySliceSize = agModeLocalStride;

	printf("alloced %d local elems for output\n", prod(B.LocalShape()));
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
                startRecvBuf = offSliceRecvBuf + offWrapRecvBuf + (nElemsPerProc * procRecvNum);
                startDataBuf = offSliceDataBuf + offWrapDataBuf + (copySliceSize * procRecvNum);
                if(wrapNum * nModeProcs + procRecvNum >= agModeLocalDim){
                    break;
                }
                printf("startRecvBuf: %d startDataBuf: %d copySliceSize: %d\n", startRecvBuf, startDataBuf, copySliceSize);
                MemCopy(&(dataBuf[startDataBuf]), &(recvBuf[startRecvBuf]), copySliceSize);
            }
        }
	}

	//
	/*
	for(procSendNum = 0; procSendNum < nModeProcs; procSendNum++){
		offSliceRecvBuf = procSendNum * nWraps * copySliceSize * nSlices;
		offSliceDataBuf = procSendNum * agModeLocalStride;  //TODO: Adjust for Alignment
		for(wrapNum = 0; wrapNum < nWraps; wrapNum++){
			offWrapRecvBuf = wrapNum * copySliceSize * nSlices;
			offWrapDataBuf = wrapNum * nModeProcs * agModeLocalStride;

			for(sliceNum = 0; sliceNum < nSlices; sliceNum++){
				startSendBuf = offSliceRecvBuf + offWrapRecvBuf + copySliceSize * sliceNum;
				startDataBuf = offSliceDataBuf + offWrapDataBuf + (agModeLocalStride * agModeLocalDim * sliceNum);
				MemCopy(&(recvBuf[startSendBuf]), &(dataBuf[startDataBuf]), copySliceSize);
			}
		}
	}
	*/
}

#define PROTO(T) \
		template void PackRSSendBuf(const DistTensor<T>& A, const int reduceScatterIndex, T * const sendBuf); \
        template void PackRSSendBuf(const DistTensor<T>& A, const int reduceIndex, const int scatterIndex, T * const sendBuf); \
        template void UnpackRSRecvBuf(const T * const recvBuf, const Int reduceScatterIndex, DistTensor<T>& A); \
        template void UnpackRSRecvBuf(const T * const recvBuf, const Int reduceIndex, const Int scatterIndex, DistTensor<T>& A); \
        template void PackAGSendBuf(const DistTensor<T>& A, const int allGatherIndex, T * const sendBuf); \
		template void UnpackAGRecvBuf(const T * const recvBuf, const Int allGatherIndex, const DistTensor<T>& A, DistTensor<T>& B);

PROTO(int)
PROTO(float)
PROTO(double)

} // namespace tmen
