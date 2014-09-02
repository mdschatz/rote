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

//TODO: Properly Check indices and distributions match between input and output
//TODO: FLESH OUT THIS CHECK
template <typename T>
Int DistTensor<T>::CheckReduceToOneCommRedist(const DistTensor<T>& A, const Mode rMode){
//    Unsigned i;
//    const tmen::GridView gvA = A.GetGridView();
//
//  //Test elimination of mode
//  const Unsigned AOrder = A.Order();
//  const Unsigned BOrder = B.Order();
//
//  //Check that redist modes are assigned properly on input and output
//  ModeDistribution BScatterModeDist = B.ModeDist(scatterMode);
//  ModeDistribution AReduceModeDist = A.ModeDist(reduceMode);
//  ModeDistribution AScatterModeDist = A.ModeDist(scatterMode);
//
//  //Test elimination of mode
//  if(BOrder != AOrder - 1){
//      LogicError("CheckReduceScatterRedist: Full Reduction requires elimination of mode being reduced");
//  }
//
//  //Test no wrapping of mode to reduce
//  if(A.Dimension(reduceMode) > gvA.Dimension(reduceMode))
//      LogicError("CheckReduceScatterRedist: Full Reduction requires global mode dimension <= gridView dimension");

    //Make sure all indices are distributed similarly between input and output (excluding reduce+scatter indices)

//  for(i = 0; i < BOrder; i++){
//      Mode mode = i;
//      if(mode == scatterMode){
//          ModeDistribution check(BScatterModeDist.end() - AReduceModeDist.size(), BScatterModeDist.end());
//            if(AnyElemwiseNotEqual(check, AReduceModeDist))
//                LogicError("CheckReduceScatterRedist: Reduce mode distribution of A must be a suffix of Scatter mode distribution of B");
//      }
//  }

    return 1;
}

template <typename T>
void DistTensor<T>::ReduceToOneCommRedist(const DistTensor<T>& A, const Mode reduceMode){
    if(!this->CheckReduceToOneCommRedist(A, reduceMode))
      LogicError("ReduceToOneRedist: Invalid redistribution request");

    //NOTE: Hack for testing.  We actually need to let the user specify the commModes
    //NOTE: THIS NEEDS TO BE BEFORE Participating() OTHERWISE PROCESSES GET OUT OF SYNC
    const ModeArray commModes = A.ModeDist(reduceMode);
    const mpi::Comm comm = this->GetCommunicatorForModes(commModes, A.Grid());

    if(!A.Participating())
        return;
    Unsigned sendSize, recvSize;

    //Determine buffer sizes for communication
    const ObjShape gridViewSlice = FilterVector(A.GridViewShape(), A.ModeDist(reduceMode));
    const ObjShape maxLocalShapeA = A.MaxLocalShape();
    sendSize = prod(maxLocalShapeA);
    recvSize = sendSize;

    Memory<T> auxMemory;
    T* auxBuf = auxMemory.Require(sendSize + recvSize);
    MemZero(&(auxBuf[0]), sendSize + recvSize);
    T* sendBuf = &(auxBuf[0]);
    T* recvBuf = &(auxBuf[sendSize]);

    PackRTOCommSendBuf(A, reduceMode, sendBuf);

    mpi::Reduce(sendBuf, recvBuf, sendSize, mpi::SUM, 0, comm);

    if(!(this->Participating()))
        return;
    UnpackRTOCommRecvBuf(recvBuf, reduceMode, A);
}

template<typename T>
void DistTensor<T>::PackRTOCommHelper(const RTOData& packData, const Mode packMode, T const * const srcBuf, T * const dstBuf){
    Unsigned packSlice;
    const Unsigned loopEnd = packData.loopShape[packMode];
    const Unsigned dstBufStride = packData.dstBufStrides[packMode];
    const Unsigned srcBufStride = packData.srcBufStrides[packMode];
    const Unsigned loopStart = packData.loopStarts[packMode];
    const Unsigned loopInc = packData.loopIncs[packMode];
    Unsigned dstBufPtr = 0;
    Unsigned srcBufPtr = 0;

    if(packMode == 0){
        if(dstBufStride == 1 && srcBufStride == 1){
            MemCopy(&(dstBuf[0]), &(srcBuf[0]), loopEnd);
        }else{
            for(packSlice = loopStart; packSlice < loopEnd; packSlice += loopInc){
                dstBuf[dstBufPtr] = srcBuf[srcBufPtr];
                dstBufPtr += dstBufStride;
                srcBufPtr += srcBufStride;
            }
        }
    }else{
        for(packSlice = loopStart; packSlice < loopEnd; packSlice += loopInc){
            PackRTOCommHelper(packData, packMode-1, &(srcBuf[srcBufPtr]), &(dstBuf[dstBufPtr]));
            dstBufPtr += dstBufStride;
            srcBufPtr += srcBufStride;
        }
    }
}

template <typename T>
void DistTensor<T>::PackRTOCommSendBuf(const DistTensor<T>& A, const Mode rMode, T * const sendBuf)
{
    const Unsigned order = A.Order();
    const T* dataBuf = A.LockedBuffer();

//    printf("dataBuf: ");
//    for(Unsigned i = 0; i < prod(A.LocalShape()); i++){
//        printf("%d ", dataBuf[i]);
//    }
//    printf("\n");

    const tmen::GridView gvA = A.GetGridView();
    const tmen::GridView gvB = GetGridView();

    const Location zeros(order, 0);
    const Location ones(order, 1);

    RTOData packData;
    packData.loopShape = A.LocalShape();
    packData.srcBufStrides = A.LocalStrides();
    packData.dstBufStrides = Dimensions2Strides(A.MaxLocalShape());

    packData.loopStarts = zeros;
    packData.loopIncs = ones;

    PackRTOCommHelper(packData, order - 1, &(dataBuf[0]), &(sendBuf[0]));
}

template <typename T>
void DistTensor<T>::UnpackRTOCommRecvBuf(const T * const recvBuf, const Mode rMode, const DistTensor<T>& A)
{
    T* dataBuf = this->Buffer();
    const Unsigned order = this->Order();

    const tmen::GridView gvA = A.GetGridView();
    const tmen::GridView gvB = GetGridView();

    const Location zeros(order, 0);
    const Location ones(order, 1);

    RTOData unpackData;
    unpackData.loopShape = this->LocalShape();
    unpackData.dstBufStrides = LocalStrides();
    unpackData.srcBufStrides = Dimensions2Strides(MaxLocalShape());

    unpackData.loopStarts = zeros;
    unpackData.loopIncs = ones;

    PackRTOCommHelper(unpackData, order - 1, &(recvBuf[0]), &(dataBuf[0]));

}

#define PROTO(T) \
        template Int  DistTensor<T>::CheckReduceToOneCommRedist(const DistTensor<T>& A, const Mode rMode); \
        template void DistTensor<T>::ReduceToOneCommRedist(const DistTensor<T>& A, const Mode reduceMode); \
        template void DistTensor<T>::PackRTOCommSendBuf(const DistTensor<T>& A, const Mode reduceMode, T * const sendBuf); \
        template void DistTensor<T>::UnpackRTOCommRecvBuf(const T * const recvBuf, const Mode reduceMode, const DistTensor<T>& A);

PROTO(int)
PROTO(float)
PROTO(double)
PROTO(Complex<float>)
PROTO(Complex<double>)

} //namespace tmen
