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
Int DistTensor<T>::CheckReduceScatterCommRedist(const DistTensor<T>& A, const Mode reduceMode, const Mode scatterMode){
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
void DistTensor<T>::ReduceScatterCommRedist(const DistTensor<T>& A, const Mode reduceMode, const Mode scatterMode){
    if(!this->CheckReduceScatterCommRedist(A, reduceMode, scatterMode))
      LogicError("ReduceScatterRedist: Invalid redistribution request");
    if(!A.Participating())
        return;
    Unsigned sendSize, recvSize;

    //Determine buffer sizes for communication
    const ObjShape gridViewSlice = FilterVector(A.GridViewShape(), A.ModeDist(reduceMode));
    const Unsigned nRedistProcs = Max(1, prod(FilterVector(A.Grid().Shape(), A.ModeDist(reduceMode))));
    const ObjShape maxLocalShapeA = MaxLengths(A.Shape(), A.GetGridView().ParticipatingShape());
    recvSize = prod(maxLocalShapeA);
    sendSize = recvSize * nRedistProcs;

    //NOTE: Hack for testing.  We actually need to let the user specify the commModes
    const ModeArray commModes = A.ModeDist(reduceMode);
    const mpi::Comm comm = A.GetCommunicatorForModes(commModes);

    Memory<T> auxMemory;
    T* auxBuf = auxMemory.Require(sendSize + recvSize);
    MemZero(&(auxBuf[0]), sendSize + recvSize);
    T* sendBuf = &(auxBuf[0]);
    T* recvBuf = &(auxBuf[sendSize]);

    PackRSCommSendBuf(A, reduceMode, scatterMode, sendBuf);

    mpi::ReduceScatter(sendBuf, recvBuf, recvSize, comm);

    if(!(this->Participating()))
        return;
    UnpackRSCommRecvBuf(recvBuf, reduceMode, scatterMode, A);
}

template <typename T>
void DistTensor<T>::PackRSCommSendBuf(const DistTensor<T>& A, const Mode rMode, const Mode sMode, T * const sendBuf)
{
    const T* dataBuf = A.LockedBuffer();

//    std::cout << "dataBuf:";
//    for(Unsigned i = 0; i < prod(A.LocalShape()); i++){
//        std::cout << " " << dataBuf[i];
//    }
//    std::cout << std::endl;

    const tmen::GridView gvA = A.GetGridView();
    const tmen::GridView gvB = GetGridView();

    const Unsigned nRedistProcs = Max(1, gvA.Dimension(rMode));

    //Shape of the local tensor we are packing
    const ObjShape maxLocalShapeA = MaxLengths(A.Shape(), gvA.ParticipatingShape());
    const ObjShape localShapeA = A.LocalShape();

    //Calculate number of outer slices to pack
    const Unsigned nMaxOuterSlices = Max(1, prod(maxLocalShapeA, sMode + 1));
    const Unsigned nLocalOuterSlices = prod(localShapeA, sMode + 1);

    //Calculate number of sMode slices to pack
    const Unsigned nMaxSModeSlices = maxLocalShapeA[sMode];
    const Unsigned nLocalSModeSlices = localShapeA[sMode];
    const Unsigned sModePackStride = nRedistProcs;
    const Unsigned nMaxPackSModeSlices = MaxLength(maxLocalShapeA[sMode], sModePackStride);

    //Number of processes we have to pack for
    const Unsigned nElemSlices = nRedistProcs;

    const Unsigned maxCopySliceSize = Max(1, prod(maxLocalShapeA, 0, sMode));
    const Unsigned copySliceSize = prod(localShapeA, 0, sMode);


    Unsigned outerSliceNum, sModeSliceNum, elemSliceNum; //Which slice of which wrap of which process are we packing
    Unsigned elemSendBufOff, elemDataBufOff;
    Unsigned outerSendBufOff, sModeSendBufOff;
    Unsigned outerDataBufOff, sModeDataBufOff;
    Unsigned startSendBuf, startDataBuf;


//    printf("MemCopy info:\n");
//    printf("    nMaxOuterSlices: %d\n", nMaxOuterSlices);
//    printf("    nLocalOuterSlices: %d\n", nLocalOuterSlices);
//    printf("    nMaxSModeSlices: %d\n", nMaxSModeSlices);
//    printf("    nLocalSModeSlices: %d\n", nLocalSModeSlices);
//    printf("    sModePackStride: %d\n", sModePackStride);
//    printf("    maxCopySliceSize: %d\n", maxCopySliceSize);
//    printf("    copySliceSize: %d\n", copySliceSize);
    for(elemSliceNum = 0; elemSliceNum < nElemSlices; elemSliceNum++){
        elemSendBufOff = prod(maxLocalShapeA) * elemSliceNum;
        elemDataBufOff = copySliceSize * elemSliceNum;

//        printf("      elemSliceNum:   %d\n", elemSliceNum);
//        printf("      elemSendBufOff: %d\n", elemSendBufOff);
//        printf("      elemDataBufOff: %d\n", elemDataBufOff);

        for(outerSliceNum = 0; outerSliceNum < nMaxOuterSlices; outerSliceNum++ ){
            if(outerSliceNum >= nLocalOuterSlices)
                break;
            outerSendBufOff = maxCopySliceSize * Max(1, nMaxPackSModeSlices) * outerSliceNum;
            outerDataBufOff = copySliceSize * nLocalSModeSlices * outerSliceNum;

//            printf("        outerSliceNum:   %d\n", outerSliceNum);
//            printf("        outerSendBufOff: %d\n", outerSendBufOff);
//            printf("        outerDataBufOff: %d\n", outerDataBufOff);

                for(sModeSliceNum = 0; sModeSliceNum < nMaxSModeSlices; sModeSliceNum += sModePackStride){
                    if(sModeSliceNum + elemSliceNum >= nLocalSModeSlices)
                        break;
                    sModeSendBufOff = maxCopySliceSize * (sModeSliceNum / sModePackStride);
                    sModeDataBufOff = copySliceSize * sModeSliceNum;

//                    printf("          sModeSliceNum:   %d\n", sModeSliceNum);
//                    printf("          sModeSendBufOff: %d\n", sModeSendBufOff);
//                    printf("          sModeDataBufOff: %d\n", sModeDataBufOff);
                    startSendBuf = elemSendBufOff + outerSendBufOff + sModeSendBufOff;
                    startDataBuf = elemDataBufOff + outerDataBufOff + sModeDataBufOff;

//                    printf("            startSendBuf: %d\n", startSendBuf);
//                    printf("            startDataBuf: %d\n", startDataBuf);
                    MemCopy(&(sendBuf[startSendBuf]), &(dataBuf[startDataBuf]), copySliceSize);
                }
        }
    }

//    std::cout << "packed sendBuf:";
//    for(Unsigned i = 0; i < prod(maxLocalShapeA) * nRedistProcs; i++)
//        std::cout << " " << sendBuf[i];
//    std::cout << std::endl;
}

template <typename T>
void DistTensor<T>::UnpackRSCommRecvBuf(const T * const recvBuf, const Mode rMode, const Mode sMode, const DistTensor<T>& A)
{
    T* dataBuf = this->Buffer();

    const tmen::GridView gvA = A.GetGridView();
    const tmen::GridView gvB = GetGridView();

    const ObjShape maxLocalShapeA = MaxLengths(A.Shape(), gvA.ParticipatingShape());
    const ObjShape maxLocalShapeB = MaxLengths(this->Shape(), gvB.ParticipatingShape());

//    const Unsigned maxRecvElem = prod(maxLocalShapeA) / (gvA.ParticipatingShape()[rMode]);
//    std::cout << "maxRecvElem: " << maxRecvElem << std::endl;
//    std::cout << "recvBuf:";
//    for(Unsigned i = 0; i < maxRecvElem; i++){
//        std::cout << " " << recvBuf[i];
//    }
//    std::cout << std::endl;

    const ObjShape localShapeB = this->LocalShape();         //Shape of the local tensor we are packing

    //Number of outer slices to unpack
    const Unsigned nMaxOuterSlices = Max(1, prod(maxLocalShapeB, sMode + 1));
    const Unsigned nLocalOuterSlices = prod(localShapeB, sMode + 1);

    //Loop packing bounds variables
    const Unsigned nMaxSModeSlices = maxLocalShapeB[sMode];
    const Unsigned nLocalSModeSlices = localShapeB[sMode];

    //Each wrap is copied contiguously because the distribution of reduceScatter index does not change

    //Variables for calculating elements to copy
    const Unsigned maxCopySliceSize = Max(1, prod(maxLocalShapeB, 0, sMode));
    const Unsigned copySliceSize = prod(localShapeB, 0, sMode);

    //Loop iteration vars
    Unsigned outerSliceNum, sModeSliceNum;  //Pack data for slice "sliceNum" (<nSlices) of wrap "wrapNum" (<nWraps) for proc "procSendNum" int offSliceRecvBuf, offWrapRecvBuf;  //Offsets used to index into sendBuf array
    Unsigned outerRecvBufOff, outerDataBufOff;  //Offsets used to index into recvBuf array
    Unsigned sModeRecvBufOff, sModeDataBufOff;  //Offsets used to index into dataBuf array
    Unsigned startRecvBuf, startDataBuf;

//    printf("MemCopy info:\n");
//    printf("    nMaxOuterSlices: %d\n", nMaxOuterSlices);
//    printf("    nMaxSModeSlices: %d\n", nMaxSModeSlices);
//    printf("    maxCopySliceSize: %d\n", maxCopySliceSize);
//    printf("    copySliceSize: %d\n", copySliceSize);
    for(outerSliceNum = 0; outerSliceNum < nMaxOuterSlices; outerSliceNum++){
        if(outerSliceNum >= nLocalOuterSlices)
            break;
        outerRecvBufOff = maxCopySliceSize * nMaxSModeSlices * outerSliceNum;
        outerDataBufOff = copySliceSize * nLocalSModeSlices * outerSliceNum;

//        printf("      outerSliceNum: %d\n", outerSliceNum);
//        printf("      outerRecvBufOff: %d\n", outerRecvBufOff);
//        printf("      outerDataBufOff: %d\n", outerDataBufOff);

        for(sModeSliceNum = 0; sModeSliceNum < nMaxSModeSlices; sModeSliceNum++){
            if(sModeSliceNum >= nLocalSModeSlices)
                break;

            sModeRecvBufOff = (maxCopySliceSize * sModeSliceNum);
            sModeDataBufOff = (copySliceSize * sModeSliceNum);

            startRecvBuf = outerRecvBufOff + sModeRecvBufOff;
            startDataBuf = outerDataBufOff + sModeDataBufOff;

//            printf("        startRecvBuf: %d\n", startRecvBuf);
//            printf("        startDataBuf: %d\n", startDataBuf);
            MemCopy(&(dataBuf[startDataBuf]), &(recvBuf[startRecvBuf]), copySliceSize);
        }
    }

//    std::cout << "dataBuf:";
//    for(Unsigned i = 0; i < prod(this->LocalShape()); i++)
//        std::cout << " " << dataBuf[i];
//    std::cout << std::endl;
}

#define PROTO(T) \
        template Int  DistTensor<T>::CheckReduceScatterCommRedist(const DistTensor<T>& A, const Mode reduceMode, const Mode scatterMode); \
        template void DistTensor<T>::ReduceScatterCommRedist(const DistTensor<T>& A, const Mode reduceMode, const Mode scatterMode); \
        template void DistTensor<T>::PackRSCommSendBuf(const DistTensor<T>& A, const Mode reduceMode, const Mode scatterMode, T * const sendBuf); \
        template void DistTensor<T>::UnpackRSCommRecvBuf(const T * const recvBuf, const Mode reduceMode, const Mode scatterMode, const DistTensor<T>& A);

PROTO(int)
PROTO(float)
PROTO(double)
PROTO(Complex<float>)
PROTO(Complex<double>)

} //namespace tmen
