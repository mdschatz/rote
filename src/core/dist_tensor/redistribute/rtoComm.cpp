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
    if(!this->Participating())
        return;
    Unsigned sendSize, recvSize;

    //Determine buffer sizes for communication
    const ObjShape gridViewSlice = FilterVector(A.GridViewShape(), A.ModeDist(reduceMode));
    const Unsigned nRedistProcs = Max(1, prod(FilterVector(A.Grid().Shape(), A.ModeDist(reduceMode))));
    const ObjShape maxLocalShapeA = MaxLengths(A.Shape(), A.GetGridView().Shape());
    sendSize = prod(maxLocalShapeA);
    recvSize = sendSize;

    //NOTE: Hack for testing.  We actually need to let the user specify the commModes
    const ModeArray commModes = A.ModeDist(reduceMode);
    const mpi::Comm comm = A.GetCommunicatorForModes(commModes);

    Memory<T> auxMemory;
    T* auxBuf = auxMemory.Require(sendSize + recvSize);
    MemZero(&(auxBuf[0]), sendSize + recvSize);
    T* sendBuf = &(auxBuf[0]);
    T* recvBuf = &(auxBuf[sendSize]);

    PackRTOCommSendBuf(A, reduceMode, sendBuf);

    mpi::Reduce(sendBuf, recvBuf, sendSize, mpi::SUM, 0, comm);

    UnpackRTOCommRecvBuf(recvBuf, reduceMode, A);
}

template <typename T>
void DistTensor<T>::PackRTOCommSendBuf(const DistTensor<T>& A, const Mode rMode, T * const sendBuf)
{
    const T* dataBuf = A.LockedBuffer();

    printf("dataBuf: ");
    for(Unsigned i = 0; i < prod(A.LocalShape()); i++){
        printf("%d ", dataBuf[i]);
    }
    printf("\n");

    const tmen::GridView gvA = A.GetGridView();
    const tmen::GridView gvB = GetGridView();

    const Unsigned nRedistProcs = gvA.Dimension(rMode);

    //Shape of the local tensor we are packing
    const ObjShape maxLocalShapeA = MaxLengths(A.Shape(), gvA.Shape());
    const ObjShape localShapeA = A.LocalShape();

    //Calculate number of outer slices to pack
    const Unsigned nMaxOuterSlices = Max(1, prod(maxLocalShapeA, rMode + 1));
    const Unsigned nLocalOuterSlices = prod(localShapeA, rMode + 1);

    //Calculate number of sMode slices to pack
    const Unsigned nMaxRModeSlices = maxLocalShapeA[rMode];
    const Unsigned nLocalRModeSlices = localShapeA[rMode];
    const Unsigned rModePackStride = nRedistProcs;

    //Number of processes we have to pack for
    const Unsigned nElemSlices = nRedistProcs;

    const Unsigned maxCopySliceSize = Max(1, prod(maxLocalShapeA, 0, rMode));
    const Unsigned copySliceSize = prod(localShapeA, 0, rMode);

    Unsigned outerSliceNum, rModeSliceNum, elemSliceNum; //Which slice of which wrap of which process are we packing
    Unsigned outerSendBufOff, rModeSendBufOff;
    Unsigned outerDataBufOff, rModeDataBufOff;
    Unsigned startSendBuf, startDataBuf;


    printf("MemCopy info:\n");
    printf("    nMaxOuterSlices: %d\n", nMaxOuterSlices);
    printf("    nMaxSModeSlices: %d\n", nMaxRModeSlices);
    printf("    sModePackStride: %d\n", rModePackStride);
    printf("    maxCopySliceSize: %d\n", maxCopySliceSize);
    printf("    copySliceSize: %d\n", copySliceSize);

    for(outerSliceNum = 0; outerSliceNum < nMaxOuterSlices; outerSliceNum++ ){
        if(outerSliceNum >= nLocalOuterSlices)
            break;
        outerSendBufOff = maxCopySliceSize * nMaxRModeSlices * outerSliceNum;
        outerDataBufOff = copySliceSize * nLocalRModeSlices * outerSliceNum;

        printf("        outerSliceNum: %d\n", outerSliceNum);
        printf("        outerSendBufOff: %d\n", outerSendBufOff);
        printf("        outerDataBufOff: %d\n", outerDataBufOff);

        for(rModeSliceNum = 0; rModeSliceNum < nMaxRModeSlices; rModeSliceNum++){
            if(rModeSliceNum >= nLocalRModeSlices)
                break;
            rModeSendBufOff = maxCopySliceSize * rModeSliceNum;
            rModeDataBufOff = copySliceSize * rModeSliceNum;

            printf("          rModeSliceNum: %d\n", rModeSliceNum);
            printf("          rModeSendBufOff: %d\n", rModeSendBufOff);
            printf("          rModeDataBufOff: %d\n", rModeDataBufOff);
            startSendBuf = outerSendBufOff + rModeSendBufOff;
            startDataBuf = outerDataBufOff + rModeDataBufOff;

            printf("          startSendBuf: %d\n", startSendBuf);
            printf("          startDataBuf: %d\n", startDataBuf);
            MemCopy(&(sendBuf[startSendBuf]), &(dataBuf[startDataBuf]), copySliceSize);
        }
    }

    printf("packed sendBuf: ");
    for(Unsigned i = 0; i < prod(maxLocalShapeA); i++)
        printf("%d ", sendBuf[i]);
    printf("\n");
}

template <typename T>
void DistTensor<T>::UnpackRTOCommRecvBuf(const T * const recvBuf, const Mode rMode, const DistTensor<T>& A)
{
    T* dataBuf = this->Buffer();

    const tmen::GridView gvA = A.GetGridView();
    const tmen::GridView gvB = GetGridView();

    //Only unpack if we are the root (everyone else gets nothing)
    if(gvB.ModeLoc(rMode) == 0){
        const ObjShape maxLocalShapeA = MaxLengths(A.Shape(), gvA.Shape());
        const ObjShape maxLocalShapeB = MaxLengths(this->Shape(), gvB.Shape());

        const Unsigned maxRecvElem = prod(maxLocalShapeB);
        printf("maxRecvElem: %d\n", maxRecvElem);
        printf("recvBuf:");
        for(Unsigned i = 0; i < maxRecvElem; i++){
            printf(" %d", recvBuf[i]);
        }
        printf("\n");

        const ObjShape localShapeB = this->LocalShape();         //Shape of the local tensor we are packing

        //Number of outer slices to unpack
        const Unsigned nMaxOuterSlices = Max(1, prod(maxLocalShapeB, rMode + 1));
        const Unsigned nLocalOuterSlices = prod(localShapeB, rMode + 1);

        //Loop packing bounds variables
        const Unsigned nMaxRModeSlices = maxLocalShapeB[rMode];
        const Unsigned nLocalRModeSlices = localShapeB[rMode];

        //Each wrap is copied contiguously because the distribution of reduce-to-one mode does not change

        //Variables for calculating elements to copy
        const Unsigned maxCopySliceSize = Max(1, prod(maxLocalShapeB, 0, rMode));
        const Unsigned copySliceSize = this->LocalModeStride(rMode);

        //Loop iteration vars
        Unsigned outerSliceNum, rModeSliceNum;  //Pack data for slice "sliceNum" (<nSlices) of wrap "wrapNum" (<nWraps) for proc "procSendNum" int offSliceRecvBuf, offWrapRecvBuf;  //Offsets used to index into sendBuf array
        Unsigned outerRecvBufOff, outerDataBufOff;  //Offsets used to index into recvBuf array
        Unsigned rModeRecvBufOff, rModeDataBufOff;  //Offsets used to index into dataBuf array
        Unsigned startRecvBuf, startDataBuf;

        printf("MemCopy info:\n");
        printf("    nMaxOuterSlices: %d\n", nMaxOuterSlices);
        printf("    nMaxRModeSlices: %d\n", nMaxRModeSlices);
        printf("    maxCopySliceSize: %d\n", maxCopySliceSize);
        printf("    copySliceSize: %d\n", copySliceSize);
        for(outerSliceNum = 0; outerSliceNum < nMaxOuterSlices; outerSliceNum++){
            if(outerSliceNum >= nLocalOuterSlices)
                break;
            outerRecvBufOff = maxCopySliceSize * nMaxRModeSlices * outerSliceNum;
            outerDataBufOff = copySliceSize * nLocalRModeSlices * outerSliceNum;

            printf("        outerSliceNum: %d\n", outerSliceNum);
            printf("        outerRecvBufOff: %d\n", outerRecvBufOff);
            printf("        outerDataBufOff: %d\n", outerDataBufOff);

            for(rModeSliceNum = 0; rModeSliceNum < nMaxRModeSlices; rModeSliceNum++){
                if(rModeSliceNum >= nLocalRModeSlices)
                    break;

                rModeRecvBufOff = (maxCopySliceSize * rModeSliceNum);
                rModeDataBufOff = (copySliceSize * rModeSliceNum);

                startRecvBuf = outerRecvBufOff + rModeRecvBufOff;
                startDataBuf = outerDataBufOff + rModeDataBufOff;

                printf("          startRecvBuf: %d\n", startRecvBuf);
                printf("          startDataBuf: %d\n", startDataBuf);
                MemCopy(&(dataBuf[startDataBuf]), &(recvBuf[startRecvBuf]), copySliceSize);
            }
        }

        printf("dataBuf:");
        for(Unsigned i = 0; i < prod(this->LocalShape()); i++)
            printf(" %d", dataBuf[i]);
        printf("\n");
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
