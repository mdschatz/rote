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
    const ObjShape maxLocalShapeA = MaxLengths(A.Shape(), A.GetGridView().ParticipatingShape());
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
void DistTensor<T>::PackRTOCommSendBufHelper(const RTOPackData& packData, const Mode packMode, T const * const dataBuf, T * const sendBuf){

    Unsigned packSlice = packMode;
    Unsigned packSliceMaxDim = packData.sendShape[packSlice];
    Unsigned packSliceLocalDim = packData.localShape[packSlice];
    Unsigned packSliceSendBufStride = packData.sendBufModeStrides[packSlice];
    Unsigned packSliceDataBufStride = packData.dataBufModeStrides[packSlice];
    Unsigned sendBufPtr = 0;
    Unsigned dataBufPtr = 0;

    if(packMode == 0){
        if(packSliceSendBufStride == 1 && packSliceDataBufStride == 1){
            MemCopy(&(sendBuf[0]), &(dataBuf[0]), packSliceLocalDim);
        }else{
            for(packSlice = 0; packSlice < packSliceLocalDim; packSlice++){
                sendBuf[sendBufPtr] = dataBuf[dataBufPtr];
                sendBufPtr += packSliceSendBufStride;
                dataBufPtr += packSliceDataBufStride;
            }
        }
    }else{
        for(packSlice = 0; packSlice < packSliceLocalDim; packSlice++){
            PackRTOCommSendBufHelper(packData, packMode-1, &(dataBuf[dataBufPtr]), &(sendBuf[sendBufPtr]));
            sendBufPtr += packSliceSendBufStride;
            dataBufPtr += packSliceDataBufStride;
        }
    }
}

template <typename T>
void DistTensor<T>::PackRTOCommSendBuf(const DistTensor<T>& A, const Mode rMode, T * const sendBuf)
{
    const Unsigned orderA = A.Order();
    const Unsigned order = A.Order();
    const T* dataBuf = A.LockedBuffer();

//    printf("dataBuf: ");
//    for(Unsigned i = 0; i < prod(A.LocalShape()); i++){
//        printf("%d ", dataBuf[i]);
//    }
//    printf("\n");

    const tmen::GridView gvA = A.GetGridView();
    const tmen::GridView gvB = GetGridView();

    RTOPackData packData;
    packData.sendShape = MaxLengths(A.Shape(), gvA.ParticipatingShape());
    packData.localShape = A.LocalShape();

    packData.dataBufModeStrides = A.LocalStrides();

    packData.sendBufModeStrides.resize(order);
    packData.sendBufModeStrides = Dimensions2Strides(packData.sendShape);

    PackRTOCommSendBufHelper(packData, order - 1, &(dataBuf[0]), &(sendBuf[0]));
}

template <typename T>
void DistTensor<T>::UnpackRTOCommRecvBufHelper(const RTOUnpackData& unpackData, const Mode unpackMode, T const * const recvBuf, T * const dataBuf){
    Unsigned unpackSlice = unpackMode;
    Unsigned unpackSliceMaxDim = unpackData.recvShape[unpackSlice];
    Unsigned unpackSliceLocalDim = unpackData.localShape[unpackSlice];
    Unsigned unpackSliceRecvBufStride = unpackData.recvBufModeStrides[unpackSlice];
    Unsigned unpackSliceDataBufStride = unpackData.dataBufModeStrides[unpackSlice];
    Unsigned recvBufPtr = 0;
    Unsigned dataBufPtr = 0;

//    std::cout << "Unpacking mode " << unpackMode << std::endl;
//    std::cout << "agMode " << commMode << std::endl;

    if(unpackMode == 0){
        if(unpackSliceRecvBufStride == 1 && unpackSliceDataBufStride == 1){
//            std::cout << "unpacking elems" << unpackSliceLocalDim << std::endl;
            MemCopy(&(dataBuf[0]), &(recvBuf[0]), unpackSliceLocalDim);
        }else{
//            std::cout << "unpackSliceRecvBufStride" << unpackSliceRecvBufStride << std::endl;
//            std::cout << "unpackSliceDataBufStride" << unpackSliceDataBufStride << std::endl;
            for(unpackSlice = 0; unpackSlice < unpackSliceLocalDim; unpackSlice++){
                dataBuf[dataBufPtr] = recvBuf[recvBufPtr];
                recvBufPtr += unpackSliceRecvBufStride;
                dataBufPtr += unpackSliceDataBufStride;
            }
        }
    }else {
//        std::cout << "unpackSliceRecvBufStride" << unpackSliceRecvBufStride << std::endl;
//        std::cout << "unpackSliceDataBufStride" << unpackSliceDataBufStride << std::endl;
        for(unpackSlice = 0; unpackSlice < unpackSliceLocalDim; unpackSlice++){
            UnpackRTOCommRecvBufHelper(unpackData, unpackMode-1, &(recvBuf[recvBufPtr]), &(dataBuf[dataBufPtr]));
            recvBufPtr += unpackSliceRecvBufStride;
            dataBufPtr += unpackSliceDataBufStride;
        }
    }
}

template <typename T>
void DistTensor<T>::UnpackRTOCommRecvBuf(const T * const recvBuf, const Mode rMode, const DistTensor<T>& A)
{
    T* dataBuf = this->Buffer();
    const Unsigned order = this->Order();

    const tmen::GridView gvA = A.GetGridView();
    const tmen::GridView gvB = GetGridView();

    //Only unpack if we are the root (everyone else gets nothing)
    if(gvB.ModeLoc(rMode) == 0){
        //NOTE: RTO will reduce the dimension of rMode by gv.Dim(rMode)
        ObjShape recvShape = MaxLengths(A.Shape(), gvA.ParticipatingShape());
        //NOTE: MaxLength used here as a Ceil
        recvShape[rMode] = MaxLength(recvShape[rMode], gvA.ParticipatingShape()[rMode]);

        RTOUnpackData unpackData;

//        unpackData.recvShape = recvShape;
        unpackData.recvShape = MaxLengths(Shape(), gvB.ParticipatingShape());
        unpackData.localShape = this->LocalShape();

        unpackData.recvBufModeStrides = Dimensions2Strides(unpackData.recvShape);
        unpackData.dataBufModeStrides = LocalStrides();

        UnpackRTOCommRecvBufHelper(unpackData, order - 1, &(recvBuf[0]), &(dataBuf[0]));

    }else{
        MemZero(&(dataBuf[0]), prod(this->LocalShape()));
    }
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
