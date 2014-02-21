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
void PackRSSendBuf(const DistTensor<T>& A, const Int reduceIndex, const Int scatterIndex, T * const sendBuf)
{

}

template <typename T>
void UnpackRSRecvBuf(const T * const recvBuf, const Int reduceIndex, const Int scatterIndex, DistTensor<T>& A)
{

}


//TODO: Adjust this for blocks (not contiguous tensors)
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
void PackAGSendBuf(const DistTensor<T>& A, const Int allGatherMode, T * const sendBuf)
{
  const tmen::GridView gridView = A.GridView();

  int nModeProcs = gridView.Dimension(allGatherMode);   //Number of procs per wrap
  int myProcNum = gridView.ModeLoc(allGatherMode);      //My index in the wrapping
  int agModeDim = A.Dimension(allGatherMode);           //Number of indices in the mode we are redistributing
  std::vector<Int> localShape = A.LocalShape();         //Shape of the local tensor we are packing
  int localAGModeStride = A.LocalModeStride(allGatherMode);
  int procSendNum, nWraps, wrapNum, sliceNum, nSlices;  //Pack data for slice "sliceNum" (<nSlices) of wrap "wrapNum" (<nWraps) for proc "procSendNum"
  int sliceSize;

  int offProcSendBuf, offWrapSendBuf, offSliceSendBuf;  //Offsets used to index into sendBuf array
  int offProcDataBuf, offWrapDataBuf, offSliceDataBuf;  //Offsets used to index into dataBuf array
  int startSendBuf, startDataBuf;

  std::vector<Int> start(A.Order(), 0);
  const T* dataBuf = A.LockedBuffer(start);

  nWraps = tmen::Ceil(agModeDim / nModeProcs);
  //Calculate number of local slices and slice size we must pack per proc per wrap
  std::vector<Int> tmp(localShape.begin() + allGatherMode + 1, localShape.end());
  nSlices = Max(1, prod(tmp));    //Cover the case where allGatherMode is the last one (in which case we do need 1 slice)
  //TODO: Make this work with blocks (more general is commented out code
  //std::vector<Int> tmp2(localShape.begin(), localShape.begin() + allGatherMode);
  //sliceSize = prod(tmp2);
  sliceSize = localAGModeStride;

  for(procSendNum = 0; procSendNum < nModeProcs; procSendNum++){
      offProcSendBuf = procSendNum * nWraps * sliceSize * nSlices;
      offProcDataBuf = procSendNum * localAGModeStride;  //TODO: Adjust for Alignment
	  for(wrapNum = 0; wrapNum < nWraps; wrapNum++){
	      offWrapSendBuf = wrapNum * sliceSize * nSlices;
	      offWrapDataBuf = wrapNum * nModeProcs * localAGModeStride;

	      for(sliceNum = 0; sliceNum < nSlices; sliceNum++){
	          startSendBuf = offProcSendBuf + offWrapSendBuf + sliceSize * sliceNum;
	          startDataBuf = offProcDataBuf + offWrapDataBuf + (localAGModeStride * agModeDim);
	          MemCopy(&(sendBuf[startSendBuf]), &(dataBuf[startDataBuf]), sliceSize);
	      }
	  }
  }
}

template <typename T>
void UnpackAGRecvBuf(const T * const recvBuf, const Int allGatherIndex, DistTensor<T>& A)
{

}

#define PROTO(T) \
		template void PackRSSendBuf(const DistTensor<T>& A, const int reduceIndex, const int scatterIndex, T * const sendBuf); \
		template void UnpackRSRecvBuf(const T * const recvBuf, const Int reduceIndex, const Int scatterIndex, DistTensor<T>& A); \
        template void PackAGSendBuf(const DistTensor<T>& A, const int allGatherIndex, T * const sendBuf); \
		template void UnpackAGRecvBuf(const T * const recvBuf, const Int allGatherIndex, DistTensor<T>& A);

PROTO(int)
PROTO(float)
PROTO(double)

} // namespace tmen
