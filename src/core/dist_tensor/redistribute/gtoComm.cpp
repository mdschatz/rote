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
Int DistTensor<T>::CheckGatherToOneCommRedist(const DistTensor<T>& A, const Mode rMode, const ModeArray& gridModes){
//    Unsigned i;
//    const tmen::GridView gvA = A.GetGridView();
//
//  //Test elimination of mode
//  const Unsigned AOrder = A.Order();
//  const Unsigned BOrder = B.Order();
//
//  //Check that redist modes are assigned properly on input and output
//  ModeDistribution BScatterModeDist = B.ModeDist(scatterMode);
//  ModeDistribution AGatherModeDist = A.ModeDist(reduceMode);
//  ModeDistribution AScatterModeDist = A.ModeDist(scatterMode);
//
//  //Test elimination of mode
//  if(BOrder != AOrder - 1){
//      LogicError("CheckGatherScatterRedist: Full Reduction requires elimination of mode being reduced");
//  }
//
//  //Test no wrapping of mode to reduce
//  if(A.Dimension(reduceMode) > gvA.Dimension(reduceMode))
//      LogicError("CheckGatherScatterRedist: Full Reduction requires global mode dimension <= gridView dimension");

    //Make sure all indices are distributed similarly between input and output (excluding reduce+scatter indices)

//  for(i = 0; i < BOrder; i++){
//      Mode mode = i;
//      if(mode == scatterMode){
//          ModeDistribution check(BScatterModeDist.end() - AGatherModeDist.size(), BScatterModeDist.end());
//            if(AnyElemwiseNotEqual(check, AGatherModeDist))
//                LogicError("CheckGatherScatterRedist: Gather mode distribution of A must be a suffix of Scatter mode distribution of B");
//      }
//  }

    return 1;
}

template <typename T>
void DistTensor<T>::GatherToOneCommRedist(const DistTensor<T>& A, const Mode gatherMode, const ModeArray& gridModes){
    if(!this->CheckGatherToOneCommRedist(A, gatherMode, gridModes))
      LogicError("GatherToOneRedist: Invalid redistribution request");

    //NOTE: Hack for testing.  We actually need to let the user specify the commModes
    //NOTE: THIS NEEDS TO BE BEFORE Participating() OTHERWISE PROCESSES GET OUT OF SYNC
    const mpi::Comm comm = this->GetCommunicatorForModes(gridModes, A.Grid());

    if(!A.Participating())
        return;
    Unsigned sendSize, recvSize;

    //Determine buffer sizes for communication
    const ObjShape gridViewSlice = FilterVector(A.GridViewShape(), A.ModeDist(gatherMode));
    const Unsigned nRedistProcs = Max(1, prod(FilterVector(A.Grid().Shape(), gridModes)));
    const ObjShape maxLocalShapeA = MaxLengths(A.Shape(), A.GetGridView().ParticipatingShape());
    sendSize = prod(maxLocalShapeA);
    recvSize = sendSize;

    Memory<T> auxMemory;
    T* auxBuf = auxMemory.Require(sendSize + nRedistProcs*recvSize);
    MemZero(&(auxBuf[0]), sendSize + recvSize);
    T* sendBuf = &(auxBuf[0]);
    T* recvBuf = &(auxBuf[sendSize]);

    PackGTOCommSendBuf(A, gatherMode, gridModes, sendBuf);

    mpi::Gather(sendBuf, sendSize, recvBuf, recvSize, 0, comm);

    if(!(this->Participating()))
        return;
    UnpackGTOCommRecvBuf(recvBuf, gatherMode, gridModes, A);
}

template<typename T>
void DistTensor<T>::PackGTOCommSendBufHelper(const GTOPackData& packData, const Mode packMode, T const * const dataBuf, T * const sendBuf){

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
            PackGTOCommSendBufHelper(packData, packMode-1, &(dataBuf[dataBufPtr]), &(sendBuf[sendBufPtr]));
            sendBufPtr += packSliceSendBufStride;
            dataBufPtr += packSliceDataBufStride;
        }
    }
}

template <typename T>
void DistTensor<T>::PackGTOCommSendBuf(const DistTensor<T>& A, const Mode gMode, const ModeArray& gridModes, T * const sendBuf)
{
    Unsigned order = A.Order();
    const T* dataBuf = A.LockedBuffer();

//    printf("dataBuf: ");
//    for(Unsigned i = 0; i < prod(A.LocalShape()); i++){
//        printf("%d ", dataBuf[i]);
//    }
//    printf("\n");

    const tmen::GridView gvA = A.GetGridView();
    const tmen::GridView gvB = GetGridView();

    GTOPackData packData;
    packData.sendShape = MaxLengths(A.Shape(), gvA.ParticipatingShape());
    packData.localShape = A.LocalShape();

    packData.dataBufModeStrides = A.LocalStrides();

    packData.sendBufModeStrides.resize(order);
    packData.sendBufModeStrides = Dimensions2Strides(packData.sendShape);

    PackGTOCommSendBufHelper(packData, order - 1, &(dataBuf[0]), &(sendBuf[0]));

//    printf("packed sendBuf: ");
//    for(Unsigned i = 0; i < prod(packData.sendShape); i++)
//        printf("%d ", sendBuf[i]);
//    printf("\n");
}

template<typename T>
void DistTensor<T>::UnpackGTOCommRecvBufHelper(const GTOUnpackData& unpackData, const Mode unpackMode, T const * const recvBuf, T * const dataBuf){
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
//                std::cout << ident << "recvBuf loc " << &(recvBuf[pRecvBufPtr]) << std::endl;
//                std::cout << ident << "dataBuf loc " << &(dataBuf[pDataBufPtr]) << std::endl;
//                std::cout << ident << "unpacking recv data:";
//                for(unpackSlice = 0; unpackSlice < unpackSliceMaxDim && (unpackSlice + elemSlice) < unpackSliceLocalDim; unpackSlice += maxElemSlice){
//                    std::cout << " " << recvBuf[pRecvBufPtr];
//
//                    std::cout << "recvBuf inc by " << unpackSliceRecvBufStride << std::endl;
//                    std::cout << "dataBuf inc by " << unpackSliceDataBufStride << std::endl;
//                    pRecvBufPtr += unpackSliceRecvBufStride;
//                    pDataBufPtr += unpackSliceDataBufStride;
//                }
//                std::cout << std::endl;

            for(unpackSlice = 0; (unpackSlice + elemSlice) < unpackSliceLocalDim; unpackSlice += elemSliceStride){
                dataBuf[dataBufPtr] = recvBuf[recvBufPtr];

//                    std::cout << "recvBuf inc by " << unpackSliceRecvBufStride << std::endl;
//                    std::cout << "dataBuf inc by " << unpackSliceDataBufStride << std::endl;
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
            Unsigned elemSliceStride = unpackData.elemSliceStride;

            //NOTE: Each tensor data we unpack is strided by nRedistProcs (maxElemSlice) entries away
            unpackSliceDataBufStride *= elemSliceStride;
            for(unpackSlice = 0; (unpackSlice + elemSlice) < unpackSliceLocalDim; unpackSlice += elemSliceStride){
                UnpackGTOCommRecvBufHelper(unpackData, unpackMode-1, &(recvBuf[recvBufPtr]), &(dataBuf[dataBufPtr]));

//                    std::cout << "recvBuf inc by " << unpackSliceRecvBufStride << std::endl;
//                    std::cout << "dataBuf inc by " << unpackSliceDataBufStride << std::endl;
                recvBufPtr += unpackSliceRecvBufStride;
                dataBufPtr += unpackSliceDataBufStride;
            }
        }else{
            for(unpackSlice = 0; unpackSlice < unpackSliceLocalDim; unpackSlice++){
                UnpackGTOCommRecvBufHelper(unpackData, unpackMode-1, &(recvBuf[recvBufPtr]), &(dataBuf[dataBufPtr]));
    //            std::cout << "recvBuf inc by " << unpackSliceRecvBufStride << std::endl;
    //            std::cout << "dataBuf inc by " << unpackSliceDataBufStride << std::endl;
                recvBufPtr += unpackSliceRecvBufStride;
                dataBufPtr += unpackSliceDataBufStride;
            }
        }
    }
}

template <typename T>
void DistTensor<T>::UnpackGTOCommRecvBuf(const T * const recvBuf, const Mode gMode, const ModeArray& gridModes, const DistTensor<T>& A)
{
    Unsigned i;
    Unsigned order = Order();
    T* dataBuf = this->Buffer();

    const tmen::Grid& g = A.Grid();
    const tmen::GridView gvA = A.GetGridView();
    const tmen::GridView gvB = GetGridView();

    const Unsigned nRedistProcs = Max(1, prod(FilterVector(g.Shape(), gridModes)));

    const ObjShape maxLocalShapeA = MaxLengths(A.Shape(), gvA.ParticipatingShape());

    ModeArray commModes = gridModes;
    std::sort(commModes.begin(), commModes.end());
    const ObjShape redistShape = FilterVector(Grid().Shape(), gridModes);
    const ObjShape commShape = FilterVector(Grid().Shape(), commModes);
    const Permutation redistPerm = DeterminePermutation(commModes, gridModes);

    const Unsigned nCommElemsPerProc = prod(maxLocalShapeA);
    const Unsigned gModeStride = LocalModeStride(gMode);
    //    printf("recvBuf:");
    //    for(Unsigned i = 0; i < nCommElemsPerProc * nRedistProcs; i++){
    //        printf(" %d", recvBuf[i]);
    //    }
    //    printf("\n");

    GTOUnpackData unpackData;

    unpackData.recvShape = MaxLengths(this->Shape(), gvB.ParticipatingShape());
    unpackData.localShape = this->LocalShape();

    unpackData.recvBufModeStrides = Dimensions2Strides(maxLocalShapeA);
    unpackData.dataBufModeStrides = LocalStrides();
    unpackData.commMode = gMode;
    unpackData.elemSliceStride = nRedistProcs;

    //NOTE: Check
    for(i = 0; i < nRedistProcs; i++){
        const Location elemCommLoc = LinearLoc2Loc(i, commShape);
        const Unsigned elemRedistLinLoc = Loc2LinearLoc(FilterVector(elemCommLoc, redistPerm), redistShape);
        if(elemRedistLinLoc >= LocalDimension(gMode))
            continue;
        unpackData.elemSlice = elemRedistLinLoc;

//        printf("elemSlice: %d\n", i);
//        printf("elemRedistLinLoc: %d\n", elemRedistLinLoc);
//        printf("dataBufPtr: %d\n", elemRedistLinLoc * gModeStride);
//        printf("recvBufPtr: %d\n", i * nCommElemsPerProc);
//        printf("nCommElemsPerProc: %d\n", nCommElemsPerProc);

        UnpackGTOCommRecvBufHelper(unpackData, order - 1, &(recvBuf[i * nCommElemsPerProc]), &(dataBuf[elemRedistLinLoc * gModeStride]));
//        printf("dataBuf:");
//        for(Unsigned i = 0; i < prod(this->LocalShape()); i++)
//            printf(" %d", dataBuf[i]);
//        printf("\n");
    }

//    printf("dataBuf:");
//    for(Unsigned i = 0; i < prod(this->LocalShape()); i++)
//        printf(" %d", dataBuf[i]);
//    printf("\n");
}

#define PROTO(T) \
        template Int  DistTensor<T>::CheckGatherToOneCommRedist(const DistTensor<T>& A, const Mode gMode, const ModeArray& gridModes); \
        template void DistTensor<T>::GatherToOneCommRedist(const DistTensor<T>& A, const Mode gatherMode, const ModeArray& gridModes); \
        template void DistTensor<T>::PackGTOCommSendBuf(const DistTensor<T>& A, const Mode gatherMode, const ModeArray& gridModes, T * const sendBuf); \
        template void DistTensor<T>::UnpackGTOCommRecvBuf(const T * const recvBuf, const Mode gatherMode, const ModeArray& gridModes, const DistTensor<T>& A);

PROTO(int)
PROTO(float)
PROTO(double)
PROTO(Complex<float>)
PROTO(Complex<double>)

} //namespace tmen
