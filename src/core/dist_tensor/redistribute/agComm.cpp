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

    const mpi::Comm comm = A.GetCommunicatorForModes(gridModes);

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

template <typename T>
void DistTensor<T>::PackAGCommSendBuf(const DistTensor<T>& A, const Mode& agMode, T * const sendBuf, const ModeArray& redistModes)
{
  const T* dataBuf = A.LockedBuffer();

  const tmen::GridView gvA = A.GetGridView();

  //Shape of the local tensor we are packing
  const ObjShape maxLocalShapeA = MaxLengths(A.Shape(), gvA.ParticipatingShape());
  const ObjShape localShapeA = A.LocalShape();

  //Calculate number of outer slices to pack
  const Unsigned nMaxOuterSlices = Max(1, prod(maxLocalShapeA, agMode + 1));
  const Unsigned nLocalOuterSlices = prod(localShapeA, agMode + 1);

  //Calculate number of agMode slices to pack
  const Unsigned nMaxAGModeSlices = maxLocalShapeA[agMode];
  const Unsigned nLocalAGModeSlices = localShapeA[agMode];

  const Unsigned maxCopySliceSize = Max(1, prod(maxLocalShapeA, 0, agMode));
  const Unsigned copySliceSize = prod(localShapeA, 0, agMode);

  Unsigned outerSliceNum, agModeSliceNum; //Which slice of which wrap of which process are we packing
  Unsigned outerSendBufOff, agModeSendBufOff;
  Unsigned outerDataBufOff, agModeDataBufOff;
  Unsigned startSendBuf, startDataBuf;

//  printf("MemCopy info:\n");
//  printf("    nMaxOuterSlices: %d\n", nMaxOuterSlices);
//  printf("    nMaxAGModeSlices: %d\n", nMaxAGModeSlices);
//  printf("    maxCopySliceSize: %d\n", maxCopySliceSize);
//  printf("    copySliceSize: %d\n", copySliceSize);
  for(outerSliceNum = 0; outerSliceNum < nMaxOuterSlices; outerSliceNum++){
      if(outerSliceNum >= nLocalOuterSlices)
          break;
      outerSendBufOff = maxCopySliceSize * nMaxAGModeSlices * outerSliceNum;
      outerDataBufOff = copySliceSize * nLocalAGModeSlices * outerSliceNum;

//      printf("        outerSliceNum: %d\n", outerSliceNum);
//      printf("        outerSendBufOff: %d\n", outerSendBufOff);
//      printf("        outerDataBufOff: %d\n", outerDataBufOff);

      for(agModeSliceNum = 0; agModeSliceNum < nMaxAGModeSlices; agModeSliceNum++){
          if(agModeSliceNum >= nLocalAGModeSlices)
              break;
          agModeSendBufOff = maxCopySliceSize * agModeSliceNum;
          agModeDataBufOff = copySliceSize * agModeSliceNum;

//          printf("          agModeSliceNum: %d\n", agModeSliceNum);
//          printf("          agModeSendBufOff: %d\n", agModeSendBufOff);
//          printf("          agModeDataBufOff: %d\n", agModeDataBufOff);
          startSendBuf = outerSendBufOff + agModeSendBufOff;
          startDataBuf = outerDataBufOff + agModeDataBufOff;

//          printf("          startSendBuf: %d\n", startSendBuf);
//          printf("          startDataBuf: %d\n", startDataBuf);
          MemCopy(&(sendBuf[startSendBuf]), &(dataBuf[startDataBuf]), copySliceSize);
      }
  }
//  printf("packed sendBuf: ");
//  for(Unsigned i = 0; i < prod(maxLocalShapeA); i++)
//      printf("%d ", sendBuf[i]);
//  printf("\n");
}

template <typename T>
void DistTensor<T>::UnpackAGCommRecvBuf(const T * const recvBuf, const Mode& agMode, const ModeArray& redistModes, const DistTensor<T>& A)
{
    T* dataBuf = this->Buffer();

    const tmen::Grid& g = A.Grid();
    const tmen::GridView gvA = A.GetGridView();
    const tmen::GridView gvB = GetGridView();

    const ObjShape commShape = FilterVector(g.Shape(), redistModes);
    const Unsigned nRedistProcs = prod(commShape);

    const ObjShape maxLocalShapeA = MaxLengths(A.Shape(), gvA.ParticipatingShape());
    const ObjShape maxLocalShapeB = MaxLengths(this->Shape(), gvB.ParticipatingShape());

//    printf("recvBuf:");
//    for(Unsigned i = 0; i < prod(maxLocalShapeA) * nRedistProcs; i++){
//        printf(" %d", recvBuf[i]);
//    }
//    printf("\n");

    const ObjShape localShapeB = this->LocalShape();

    //Number of outer slices to unpack
    const Unsigned nMaxOuterSlices = Max(1, prod(maxLocalShapeB, agMode + 1));
    const Unsigned nLocalOuterSlices = prod(localShapeB, agMode + 1);

    //Loop packing bounds variables
    const Unsigned nMaxAGModeSlices = maxLocalShapeB[agMode];
    const Unsigned nLocalAGModeSlices = localShapeB[agMode];
    const Unsigned agModeUnpackStride = nRedistProcs;
    const Unsigned nMaxAGModePackSlices = MaxLength(nMaxAGModeSlices, agModeUnpackStride);

    //Variables for calculating elements to copy
    const Unsigned maxCopySliceSize = Max(1, prod(maxLocalShapeB, 0, agMode));
    const Unsigned copySliceSize = prod(localShapeB, 0, agMode);

    //Number of processes we have to unpack from
    const Unsigned nElemSlices = nRedistProcs;

    //Loop iteration vars
    Unsigned outerSliceNum, agModeSliceNum, elemSliceNum;  //Pack data for slice "sliceNum" (<nSlices) of wrap "wrapNum" (<nWraps) for proc "procSendNum" int offSliceRecvBuf, offWrapRecvBuf;  //Offsets used to index into sendBuf array
    Unsigned elemRecvBufOff, elemDataBufOff;
    Unsigned outerRecvBufOff, outerDataBufOff;  //Offsets used to index into recvBuf array
    Unsigned agModeRecvBufOff, agModeDataBufOff;  //Offsets used to index into dataBuf array
    Unsigned startRecvBuf, startDataBuf;

//    printf("MemCopy info:\n");
//    printf("    nMaxOuterSlices: %d\n", nMaxOuterSlices);
//    printf("    nMaxAGModeSlices: %d\n", nMaxAGModeSlices);
//    printf("    maxCopySliceSize: %d\n", maxCopySliceSize);
//    printf("    copySliceSize: %d\n", copySliceSize);
//    printf("    agModeUnpackStride: %d\n", agModeUnpackStride);
    for(elemSliceNum = 0; elemSliceNum < nElemSlices; elemSliceNum++){
        elemRecvBufOff = prod(maxLocalShapeA) * elemSliceNum;
        elemDataBufOff = copySliceSize * elemSliceNum;

//        printf("      elemSliceNum: %d\n", elemSliceNum);
//        printf("      elemRecvBufOff: %d\n", elemRecvBufOff);
//        printf("      elemDataBufOff: %d\n", elemDataBufOff);
        for(outerSliceNum = 0; outerSliceNum < nMaxOuterSlices; outerSliceNum++){
            if(outerSliceNum >= nLocalOuterSlices)
                break;
            //NOTE: the weird Max() function ensures we increment the recvBuf correctly
            //e.g. we need to ensure that we jump over all slices packed by the pack routine.  Which should be maxLocalShapeA[agModeA];
            //For consistency, kept same structure as in PackPartialRSSendBuf
            outerRecvBufOff = maxCopySliceSize * Max(1, nMaxAGModePackSlices) * outerSliceNum;
            outerDataBufOff = copySliceSize * nLocalAGModeSlices * outerSliceNum;

//            printf("        outerSliceNum: %d\n", outerSliceNum);
//            printf("        outerRecvBufOff: %d\n", outerRecvBufOff);
//            printf("        outerDataBufOff: %d\n", outerDataBufOff);
            for(agModeSliceNum = 0; agModeSliceNum < nMaxAGModeSlices; agModeSliceNum += agModeUnpackStride){
                if(agModeSliceNum + elemSliceNum >= nLocalAGModeSlices)
                    break;
                agModeRecvBufOff = maxCopySliceSize * (agModeSliceNum / agModeUnpackStride);
                agModeDataBufOff = copySliceSize * agModeSliceNum;

//                printf("          agModeSliceNum: %d\n", agModeSliceNum);
//                printf("          agModeRecvBufOff: %d\n", agModeRecvBufOff);
//                printf("          agModeDataBufOff: %d\n", agModeDataBufOff);
                startRecvBuf = elemRecvBufOff + outerRecvBufOff + agModeRecvBufOff;
                startDataBuf = elemDataBufOff + outerDataBufOff + agModeDataBufOff;

//                printf("          startRecvBuf: %d\n", startRecvBuf);
//                printf("          startDataBuf: %d\n", startDataBuf);
                MemCopy(&(dataBuf[startDataBuf]), &(recvBuf[startRecvBuf]), copySliceSize);
            }
        }
    }
//    printf("dataBuf:");
//    for(Unsigned i = 0; i < prod(this->LocalShape()); i++)
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
