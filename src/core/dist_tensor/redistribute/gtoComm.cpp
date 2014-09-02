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
    const ObjShape maxLocalShapeA = A.MaxLocalShape();
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
void DistTensor<T>::PackGTOCommHelper(const GTOData& packData, const Mode packMode, T const * const dataBuf, T * const sendBuf){

    Unsigned packSlice;
    Unsigned loopEnd = packData.loopShape[packMode];
    const Unsigned dstBufStride = packData.dstBufStrides[packMode];
    const Unsigned srcBufStride = packData.srcBufStrides[packMode];
    const Unsigned loopStart = packData.loopStarts[packMode];
    const Unsigned loopInc = packData.loopIncs[packMode];
    Unsigned dstBufPtr = 0;
    Unsigned srcBufPtr = 0;

    if(packMode == 0){
        if(dstBufStride == 1 && srcBufStride == 1){
            MemCopy(&(sendBuf[0]), &(dataBuf[0]), loopEnd);
        }else{
            for(packSlice = loopStart; packSlice < loopEnd; packSlice += loopInc){
                sendBuf[dstBufPtr] = dataBuf[srcBufPtr];
                dstBufPtr += dstBufStride;
                srcBufPtr += srcBufStride;
            }
        }
    }else{
        for(packSlice = loopStart; packSlice < loopEnd; packSlice += loopInc){
            PackGTOCommHelper(packData, packMode-1, &(dataBuf[srcBufPtr]), &(sendBuf[dstBufPtr]));
            dstBufPtr += dstBufStride;
            srcBufPtr += srcBufStride;
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

    const Location zeros(order, 0);
    const Location ones(order, 1);

    GTOData packData;
    packData.loopShape = A.LocalShape();
    packData.srcBufStrides = A.LocalStrides();

    packData.dstBufStrides.resize(order);
    packData.dstBufStrides = Dimensions2Strides(A.MaxLocalShape());

    packData.loopStarts = zeros;
    packData.loopIncs = ones;

    PackGTOCommHelper(packData, order - 1, &(dataBuf[0]), &(sendBuf[0]));

//    printf("packed sendBuf: ");
//    for(Unsigned i = 0; i < prod(packData.sendShape); i++)
//        printf("%d ", sendBuf[i]);
//    printf("\n");
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

    const ObjShape recvShape = A.MaxLocalShape();

    ModeArray commModes = gridModes;
    std::sort(commModes.begin(), commModes.end());
    const ObjShape redistShape = FilterVector(Grid().Shape(), gridModes);
    const ObjShape commShape = FilterVector(Grid().Shape(), commModes);
    const Permutation redistPerm = DeterminePermutation(commModes, gridModes);

    const Unsigned nCommElemsPerProc = prod(recvShape);
    const Unsigned gModeStride = LocalModeStride(gMode);
    //    printf("recvBuf:");
    //    for(Unsigned i = 0; i < nCommElemsPerProc * nRedistProcs; i++){
    //        printf(" %d", recvBuf[i]);
    //    }
    //    printf("\n");

    const Location zeros(order, 0);
    const Location ones(order, 1);

    GTOData unpackData;
    unpackData.loopShape = this->LocalShape();
    unpackData.dstBufStrides = LocalStrides();
    unpackData.dstBufStrides[gMode] *= nRedistProcs;

    unpackData.srcBufStrides = Dimensions2Strides(recvShape);

    unpackData.loopStarts = zeros;
    unpackData.loopIncs = ones;
    unpackData.loopIncs[gMode] = nRedistProcs;

    //NOTE: Check
    for(i = 0; i < nRedistProcs; i++){
        const Location elemCommLoc = LinearLoc2Loc(i, commShape);
        const Unsigned elemRedistLinLoc = Loc2LinearLoc(FilterVector(elemCommLoc, redistPerm), redistShape);
        if(elemRedistLinLoc >= LocalDimension(gMode))
            continue;
        unpackData.loopStarts[gMode] = elemRedistLinLoc;

//        printf("elemSlice: %d\n", i);
//        printf("elemRedistLinLoc: %d\n", elemRedistLinLoc);
//        printf("dataBufPtr: %d\n", elemRedistLinLoc * gModeStride);
//        printf("recvBufPtr: %d\n", i * nCommElemsPerProc);
//        printf("nCommElemsPerProc: %d\n", nCommElemsPerProc);

        PackGTOCommHelper(unpackData, order - 1, &(recvBuf[i * nCommElemsPerProc]), &(dataBuf[elemRedistLinLoc * gModeStride]));
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
