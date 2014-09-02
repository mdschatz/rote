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

template<typename T>
Int
DistTensor<T>::CheckAllGatherCommRedist(const DistTensor<T>& A, const Mode& allGatherMode, const ModeArray& redistModes){
    if(A.Order() != this->Order()){
        LogicError("CheckAllGatherRedist: Objects being redistributed must be of same order");
    }

    ModeDistribution allGatherDistA = A.ModeDist(allGatherMode);

    const ModeDistribution check = ConcatenateVectors(this->ModeDist(allGatherMode), redistModes);
    if(AnyElemwiseNotEqual(check, allGatherDistA)){
        LogicError("CheckAllGatherRedist: [Output distribution ++ redistModes] does not match Input distribution");
    }

    return true;
}

template<typename T>
void
DistTensor<T>::AllGatherCommRedist(const DistTensor<T>& A, const Mode& agMode, const ModeArray& gridModes){
#ifndef RELEASE
    CallStackEntry entry("DistTensor::AllGatherCommRedist");
    if(!CheckAllGatherCommRedist(A, agMode, gridModes))
        LogicError("AllGatherRedist: Invalid redistribution request");
#endif

    const mpi::Comm comm = this->GetCommunicatorForModes(gridModes, A.Grid());

    if(!A.Participating())
        return;

    //NOTE: Fix to handle strides in Tensor data
    if(gridModes.size() == 0){
        this->CopyLocalBuffer(A);
        return;
    }
    Unsigned sendSize, recvSize;

    //Determine buffer sizes for communication
    const Unsigned nRedistProcs = Max(1, prod(FilterVector(A.Grid().Shape(), gridModes)));
    const ObjShape maxLocalShapeA = MaxLengths(A.Shape(), A.GetGridView().ParticipatingShape());

    sendSize = prod(maxLocalShapeA);
    recvSize = sendSize * nRedistProcs;

    Memory<T> auxMemory;
    T* auxBuf = auxMemory.Require(sendSize + recvSize);
    MemZero(&(auxBuf[0]), sendSize + recvSize);

    T* sendBuf = &(auxBuf[0]);
    T* recvBuf = &(auxBuf[sendSize]);

    //printf("Alloc'd %d elems to send and %d elems to receive\n", sendSize, recvSize);
    PackAGCommSendBuf(A, agMode, sendBuf, gridModes);

    //printf("Allgathering %d elements\n", sendSize);
    mpi::AllGather(sendBuf, sendSize, recvBuf, sendSize, comm);

    if(!(this->Participating()))
        return;
    UnpackAGCommRecvBuf(recvBuf, agMode, gridModes, A);
    //Print(B.LockedTensor(), "A's local tensor after allgathering:");
}

template<typename T>
void DistTensor<T>::PackAGCommSendBufHelper(const AGPackData& packData, const Mode packMode, T const * const dataBuf, T * const sendBuf){

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
            PackAGCommSendBufHelper(packData, packMode-1, &(dataBuf[dataBufPtr]), &(sendBuf[sendBufPtr]));
            sendBufPtr += packSliceSendBufStride;
            dataBufPtr += packSliceDataBufStride;
        }
    }
}

template <typename T>
void DistTensor<T>::PackAGCommSendBuf(const DistTensor<T>& A, const Mode& agMode, T * const sendBuf, const ModeArray& redistModes)
{
  Unsigned i;
  const Unsigned order = A.Order();
  const T* dataBuf = A.LockedBuffer();

  const tmen::GridView gvA = A.GetGridView();

  AGPackData packData;
  packData.sendShape = MaxLengths(A.Shape(), gvA.ParticipatingShape());
  packData.localShape = A.LocalShape();

  packData.dataBufModeStrides = A.LocalStrides();

  packData.sendBufModeStrides.resize(order);
  packData.sendBufModeStrides = Dimensions2Strides(packData.sendShape);

  PackAGCommSendBufHelper(packData, order - 1, &(dataBuf[0]), &(sendBuf[0]));
}

template<typename T>
void DistTensor<T>::UnpackAGCommRecvBufHelper(const AGUnpackData& unpackData, const Mode unpackMode, T const * const recvBuf, T * const dataBuf){
    Unsigned unpackSlice = unpackMode;
    Unsigned unpackSliceMaxDim = unpackData.recvShape[unpackSlice];
    Unsigned unpackSliceLocalDim = unpackData.localShape[unpackSlice];
    Unsigned unpackSliceRecvBufStride = unpackData.recvBufModeStrides[unpackSlice];
    Unsigned unpackSliceDataBufStride = unpackData.dataBufModeStrides[unpackSlice];
    Mode commMode = unpackData.commMode;
    Unsigned recvBufPtr = 0;
    Unsigned dataBufPtr = 0;
    Unsigned pRecvBufPtr = 0;
    Unsigned pDataBufPtr = 0;

//    Unsigned order = Order();
//    Unsigned i;
//    std::string ident = "";
//    for(i = 0; i < order - unpackMode; i++)
//        ident += "  ";
//    std::cout << ident << "Unpacking mode " << unpackMode << std::endl;

    if(unpackMode == 0){
//        std::cout << ident << "recvBuf loc " << &(recvBuf[pRecvBufPtr]) << std::endl;
//        std::cout << ident << "dataBuf loc " << &(dataBuf[pDataBufPtr]) << std::endl;
//        std::cout << ident << "unpacking recv data:";
//        for(unpackSlice = 0; unpackSlice < unpackSliceMaxDim && unpackSlice < unpackSliceLocalDim; unpackSlice++){
//            std::cout << " " << recvBuf[pRecvBufPtr];
//            pRecvBufPtr += unpackSliceRecvBufStride;
//            pDataBufPtr += unpackSliceDataBufStride;
//        }
//        std::cout << std::endl;
        if(unpackMode == commMode){
            Unsigned elemSlice = unpackData.elemSlice;
            Unsigned elemSliceStride = unpackData.elemSliceStride;

            //NOTE: Each tensor data we unpack is strided by nRedistProcs (maxElemSlice) entries away
            unpackSliceDataBufStride *= elemSliceStride;
//            std::cout << ident << "recvBuf loc " << &(recvBuf[pRecvBufPtr]) << std::endl;
//            std::cout << ident << "dataBuf loc " << &(dataBuf[pDataBufPtr]) << std::endl;
//            std::cout << ident << "unpacking recv data:";
//            for(unpackSlice = 0; unpackSlice < unpackSliceMaxDim && (unpackSlice + elemSlice) < unpackSliceLocalDim; unpackSlice += maxElemSlice){
//                std::cout << " " << recvBuf[pRecvBufPtr];
//
//                std::cout << "recvBuf inc by " << unpackSliceRecvBufStride << std::endl;
//                std::cout << "dataBuf inc by " << unpackSliceDataBufStride << std::endl;
//                pRecvBufPtr += unpackSliceRecvBufStride;
//                pDataBufPtr += unpackSliceDataBufStride;
//            }
//            std::cout << std::endl;

            for(unpackSlice = 0; (unpackSlice + elemSlice) < unpackSliceLocalDim; unpackSlice += elemSliceStride){
                dataBuf[dataBufPtr] = recvBuf[recvBufPtr];

//                std::cout << "recvBuf inc by " << unpackSliceRecvBufStride << std::endl;
//                std::cout << "dataBuf inc by " << unpackSliceDataBufStride << std::endl;
                recvBufPtr += unpackSliceRecvBufStride;
                dataBufPtr += unpackSliceDataBufStride;
            }
        }else{
            if(unpackSliceRecvBufStride == 1 && unpackSliceDataBufStride == 1){
                MemCopy(&(dataBuf[0]), &(recvBuf[0]), unpackSliceLocalDim);
            }else{
                for(unpackSlice = 0; unpackSlice < unpackSliceLocalDim; unpackSlice++){
                    dataBuf[dataBufPtr] = recvBuf[recvBufPtr];

    //                std::cout << "recvBuf inc by " << unpackSliceRecvBufStride << std::endl;
    //                std::cout << "dataBuf inc by " << unpackSliceDataBufStride << std::endl;
                    recvBufPtr += unpackSliceRecvBufStride;
                    dataBufPtr += unpackSliceDataBufStride;
                }
            }
        }
    }else {
        if(unpackMode == commMode){
            Unsigned elemSlice = unpackData.elemSlice;
            Unsigned maxElemSlice = unpackData.elemSliceStride;

            //NOTE: Each tensor data we unpack is strided by nRedistProcs (maxElemSlice) entries away
            unpackSliceDataBufStride *= maxElemSlice;
            for(unpackSlice = 0; (unpackSlice + elemSlice) < unpackSliceLocalDim; unpackSlice += maxElemSlice){
                UnpackAGCommRecvBufHelper(unpackData, unpackMode-1, &(recvBuf[recvBufPtr]), &(dataBuf[dataBufPtr]));

//                    std::cout << "recvBuf inc by " << unpackSliceRecvBufStride << std::endl;
//                    std::cout << "dataBuf inc by " << unpackSliceDataBufStride << std::endl;
                recvBufPtr += unpackSliceRecvBufStride;
                dataBufPtr += unpackSliceDataBufStride;
            }
        }else{
            for(unpackSlice = 0; unpackSlice < unpackSliceLocalDim; unpackSlice++){
                UnpackAGCommRecvBufHelper(unpackData, unpackMode-1, &(recvBuf[recvBufPtr]), &(dataBuf[dataBufPtr]));
    //            std::cout << "recvBuf inc by " << unpackSliceRecvBufStride << std::endl;
    //            std::cout << "dataBuf inc by " << unpackSliceDataBufStride << std::endl;
                recvBufPtr += unpackSliceRecvBufStride;
                dataBufPtr += unpackSliceDataBufStride;
            }
        }
    }
}

template <typename T>
void DistTensor<T>::UnpackAGCommRecvBuf(const T * const recvBuf, const Mode& agMode, const ModeArray& redistModes, const DistTensor<T>& A)
{
    Unsigned order = A.Order();
    Unsigned i;
    T* dataBuf = this->Buffer();

    const tmen::Grid& g = A.Grid();
    const tmen::GridView gvA = A.GetGridView();
    const tmen::GridView gvB = GetGridView();
    const Unsigned nRedistProcs = prod(FilterVector(g.Shape(), redistModes));
    const ObjShape maxLocalShapeA = MaxLengths(A.Shape(), gvA.ParticipatingShape());

    ModeArray commModes = redistModes;
    std::sort(commModes.begin(), commModes.end());
    const ObjShape redistShape = FilterVector(Grid().Shape(), redistModes);
    const ObjShape commShape = FilterVector(Grid().Shape(), commModes);
    const Permutation redistPerm = DeterminePermutation(commModes, redistModes);

    const Unsigned nCommElemsPerProc = prod(maxLocalShapeA);
    const Unsigned agModeStride = LocalModeStride(agMode);
//    printf("recvBuf:");
//    for(Unsigned i = 0; i < nCommElemsPerProc * nRedistProcs; i++){
//        printf(" %d", recvBuf[i]);
//    }
//    printf("\n");

    AGUnpackData unpackData;

    unpackData.recvShape = MaxLengths(this->Shape(), gvB.ParticipatingShape());
    unpackData.localShape = this->LocalShape();

    unpackData.recvBufModeStrides = Dimensions2Strides(maxLocalShapeA);
    unpackData.dataBufModeStrides = LocalStrides();
    unpackData.commMode = agMode;
    unpackData.elemSliceStride = nRedistProcs;

    //NOTE: Check
    for(i = 0; i < nRedistProcs; i++){
        const Location elemCommLoc = LinearLoc2Loc(i, commShape);
        const Unsigned elemRedistLinLoc = Loc2LinearLoc(FilterVector(elemCommLoc, redistPerm), redistShape);

        if(elemRedistLinLoc >= LocalDimension(agMode))
            continue;
        unpackData.elemSlice = elemRedistLinLoc;

        UnpackAGCommRecvBufHelper(unpackData, order - 1, &(recvBuf[i * nCommElemsPerProc]), &(dataBuf[elemRedistLinLoc * agModeStride]));

    }

//    printf("dataBuf:");
//    for(Unsigned i = 0; i < prod(LocalShape()); i++)
//        printf(" %d", dataBuf[i]);
//    printf("\n");
}

#define PROTO(T) \
        template void DistTensor<T>::AllGatherCommRedist(const DistTensor<T>& A, const Mode& agMode, const ModeArray& gridModes); \
        template Int  DistTensor<T>::CheckAllGatherCommRedist(const DistTensor<T>& A, const Mode& allGatherMode, const ModeArray& redistModes); \
        template void DistTensor<T>::PackAGCommSendBuf(const DistTensor<T>& A, const Mode& allGatherMode, T * const sendBuf, const ModeArray& redistModes); \
        template void DistTensor<T>::UnpackAGCommRecvBuf(const T * const recvBuf, const Mode& allGatherMode, const ModeArray& redistModes, const DistTensor<T>& A);

//template Int CheckAllGatherCommRedist(const DistTensor<T>& B, const DistTensor<T>& A, const Mode& allGatherMode, const ModeArray& redistModes);
//template void AllGatherCommRedist(DistTensor<T>& B, const DistTensor<T>& A, const Mode& allGatherMode, const ModeArray& redistModes );



PROTO(int)
PROTO(float)
PROTO(double)
PROTO(Complex<double>)
PROTO(Complex<float>)

} //namespace tmen
