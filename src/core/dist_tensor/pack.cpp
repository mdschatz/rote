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
void PackRSSendBuf(const DistTensor<T>& A, const Int reduceScatterMode, T * const sendBuf)
{

}

//Only called when fully reducing an index
//NOTE: Looks an awful lot like PackAGSendBuf...
//TODO: Merge with PackAGSendBuf?
template <typename T>
void PackRSSendBuf(const DistTensor<T>& A, const Int reduceMode, const Int scatterMode, T * const sendBuf)
{
	  printf("A has %d elems to pack\n", prod(A.LocalShape()));
	  const tmen::GridView gridView = A.GridView();

	  std::vector<Int> start(A.Order(), 0);
	  const T* dataBuf = A.LockedBuffer(start);

	  const tmen::GridView gv = A.GridView();

	  int sModeLocalDim = A.LocalDimension(scatterMode); //Local version
	  std::vector<Int> localShape = A.LocalShape();         //Shape of the local tensor we are packing
	  int sModeLocalStride = A.LocalModeStride(scatterMode);
	  int sliceNum, nMaxSlices, nLocalSlices;  //Pack data for slice "sliceNum" (<nSlices) of wrap "wrapNum" (<nWraps) for proc "procSendNum"
	  int copySliceSize;

	  const std::vector<Int> maxLocalShape = MaxLengths(A.Shape(), gv.Shape());
	  const int sModeMaxLocalDim = maxLocalShape[scatterMode];

	  int offSliceSendBuf;  //Offsets used to index into sendBuf array
	  int offSliceDataBuf;  //Offsets used to index into dataBuf array

	  //Calculate number of local slices and slice size we must pack per proc per wrap
	  nLocalSlices = Max(1, prod(localShape, scatterMode + 1));
	  nMaxSlices = Max(1, prod(maxLocalShape, scatterMode + 1));    //Cover the case where scatterMode is the last one (in which case we do need 1 slice)
	  //TODO: Make this work with blocks (more general is commented out code
	  //std::vector<Int> tmp2(localShape.begin(), localShape.begin() + reduceMode);
	  //sliceSize = prod(tmp2);
	  copySliceSize = sModeLocalStride * sModeLocalDim;

	  for(sliceNum = 0; sliceNum < nMaxSlices; sliceNum++){
		  offSliceSendBuf = sliceNum * sModeMaxLocalDim;
		  offSliceDataBuf = sliceNum * sModeLocalDim;
		  if(sliceNum >= nLocalSlices){
			  break;
		  }
		  printf("offSliceSendBuf: %d offSliceDataBuf: %d copySliceSize: %d\n", offSliceSendBuf, offSliceDataBuf, copySliceSize);
		  MemCopy(&(sendBuf[offSliceSendBuf]), &(dataBuf[offSliceDataBuf]), copySliceSize);
	  }
}

template <typename T>
void UnpackRSRecvBuf(const T * const recvBuf, const Int reduceMode, const Int scatterMode, const DistTensor<T>& A, DistTensor<T>& B)
{
}

template <typename T>
void UnpackRSRecvBuf(const T * const recvBuf, const Int reduceScatterMode, DistTensor<T>& A)
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

  const std::vector<Int> start(A.Order(), 0);
  const T* dataBuf = A.LockedBuffer(start);

  const tmen::GridView gv = A.GridView();

  const std::vector<Int> localShape = A.LocalShape();         //Shape of the local tensor we are packing
  const std::vector<Int> maxLocalShape = MaxLengths(A.Shape(), gv.Shape());

  const int agModeLocalDim = A.LocalDimension(allGatherMode); //Local version
  const int agModeMaxLocalDim = maxLocalShape[allGatherMode];

  int agModeLocalStride = A.LocalModeStride(allGatherMode);
  int sliceNum, nMaxSlices, nLocalSlices;  //Pack data for slice "sliceNum" (<nSlices) of wrap "wrapNum" (<nWraps) for proc "procSendNum"
  int copySliceSize;

  int offSliceSendBuf;  //Offsets used to index into sendBuf array
  int offSliceDataBuf;  //Offsets used to index into dataBuf array

  //Calculate number of local slices and slice size we must pack per proc per wrap
  nLocalSlices = Max(1, prod(localShape, allGatherMode + 1));
  nMaxSlices = Max(1, prod(maxLocalShape, allGatherMode + 1));    //Cover the case where allGatherMode is the last one (in which case we do need 1 slice)
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
//TODO: Make this work with blocks (more general is commented out code
//TODO: Swap nWrap test to be more like sliceNum test (Refer to local information)
template <typename T>
void UnpackAGRecvBuf(const T * const recvBuf, const Int allGatherMode, const DistTensor<T>& A, DistTensor<T>& B)
{
	printf("B can unpack %d elems\n", prod(B.LocalShape()));
    const std::vector<Int> start(B.Order(), 0);
    T* dataBuf = B.Buffer(start);
    const tmen::GridView gv = A.GridView();

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
	int procRecvNum, wrapNum, sliceNum;  //Pack data for slice "sliceNum" (<nSlices) of wrap "wrapNum" (<nWraps) for proc "procSendNum"	int offSliceRecvBuf, offWrapRecvBuf;  //Offsets used to index into sendBuf array
	int offSliceRecvBuf, offWrapRecvBuf;  //Offsets used to index into recvBuf array
	int offSliceDataBuf, offWrapDataBuf;  //Offsets used to index into dataBuf array
	int startRecvBuf, startDataBuf;

	printf("alloced %d local elems for output\n", prod(B.LocalShape()));
	printf("data: [%.0f", (double)(recvBuf[0]));
	for(int i = 1; i < nMaxElemsPerProc * nModeProcs; i++)
		printf(", %.0f", (double)(recvBuf[i]));
	printf("]\n");

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
                printf("startRecvBuf: %d startDataBuf: %d copySliceSize: %d\n", startRecvBuf, startDataBuf, copySliceSize);
                MemCopy(&(dataBuf[startDataBuf]), &(recvBuf[startRecvBuf]), copySliceSize);
            }
        }
	}
}

#define PROTO(T) \
		template void PackRSSendBuf(const DistTensor<T>& A, const int reduceScatterMode, T * const sendBuf); \
        template void PackRSSendBuf(const DistTensor<T>& A, const int reduceMode, const int scatterMode, T * const sendBuf); \
        template void UnpackRSRecvBuf(const T * const recvBuf, const Int reduceScatterMode, DistTensor<T>& A); \
        template void UnpackRSRecvBuf(const T * const recvBuf, const Int reduceMode, const Int scatterMode, const DistTensor<T>& A, DistTensor<T>& B); \
        template void PackAGSendBuf(const DistTensor<T>& A, const int allGatherMode, T * const sendBuf); \
		template void UnpackAGRecvBuf(const T * const recvBuf, const Int allGatherMode, const DistTensor<T>& A, DistTensor<T>& B);

PROTO(int)
PROTO(float)
PROTO(double)

} // namespace tmen
