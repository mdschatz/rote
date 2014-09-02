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

    //NOTE: Hack for testing.  We actually need to let the user specify the commModes
    const ModeArray commModes = A.ModeDist(reduceMode);
    const mpi::Comm comm = this->GetCommunicatorForModes(commModes, A.Grid());

    if(!A.Participating())
        return;
    Unsigned sendSize, recvSize;

    //Determine buffer sizes for communication
    const ObjShape gridViewSlice = FilterVector(A.GridViewShape(), A.ModeDist(reduceMode));
    const Unsigned nRedistProcs = Max(1, prod(FilterVector(A.Grid().Shape(), A.ModeDist(reduceMode))));
    const ObjShape maxLocalShapeB = MaxLocalShape();
    recvSize = prod(maxLocalShapeB);
    sendSize = recvSize * nRedistProcs;



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
    Unsigned i;
    Unsigned order = A.Order();
    const T* dataBuf = A.LockedBuffer();

//    std::cout << "dataBuf:";
//    for(Unsigned i = 0; i < prod(A.LocalShape()); i++){
//        std::cout << " " << dataBuf[i];
//    }
//    std::cout << std::endl;

    const tmen::GridView gvA = A.GetGridView();
    const tmen::GridView gvB = GetGridView();
    const Unsigned nRedistProcs = Max(1, gvA.Dimension(rMode));

    const ModeArray redistModes = A.ModeDist(rMode);
    ModeArray commModes = redistModes;
    std::sort(commModes.begin(), commModes.end());
    const ObjShape redistShape = FilterVector(Grid().Shape(), redistModes);
    const ObjShape commShape = FilterVector(Grid().Shape(), commModes);
    const Permutation commPerm = DeterminePermutation(redistModes, commModes);

    const ObjShape sendShape = MaxLocalShape();

    const Location zeros(order, 0);
    const Location ones(order, 1);
    PackData packData;
    packData.loopShape = A.LocalShape();
    packData.srcBufStrides = A.LocalStrides();
    packData.srcBufStrides[sMode] *= nRedistProcs;
    packData.dstBufStrides = Dimensions2Strides(sendShape);
    packData.loopStarts = zeros;
    packData.loopIncs = ones;
    packData.loopIncs[sMode] = nRedistProcs;
    const Unsigned nCommElemsPerProc = prod(sendShape);
    const Unsigned sModeStride = LocalModeStride(sMode);

    for(i = 0; i < nRedistProcs; i++){
        const Location elemCommLoc = LinearLoc2Loc(i, redistShape);
        const Unsigned elemRedistLinLoc = Loc2LinearLoc(FilterVector(elemCommLoc, commPerm), commShape);

        packData.loopStarts[sMode] = i;
        PackCommHelper(packData, order - 1, &(dataBuf[i*sModeStride]), &(sendBuf[elemRedistLinLoc * nCommElemsPerProc]));
//        std::cout << "pack slice:" << i << std::endl;
//        std::cout << "packed sendBuf:";
//        for(Unsigned i = 0; i < nCommElemsPerProc * nRedistProcs; i++)
//            std::cout << " " << sendBuf[i];
//        std::cout << std::endl;
    }

//    std::cout << "packed sendBuf:";
//    for(Unsigned i = 0; i < nCommElemsPerProc * nRedistProcs; i++)
//        std::cout << " " << sendBuf[i];
//    std::cout << std::endl;
}

template <typename T>
void DistTensor<T>::UnpackRSCommRecvBuf(const T * const recvBuf, const Mode rMode, const Mode sMode, const DistTensor<T>& A)
{
    const Unsigned order = A.Order();
    T* dataBuf = this->Buffer();

    const tmen::GridView gvA = A.GetGridView();
    const tmen::GridView gvB = GetGridView();

//    std::cout << "recvBuf:";
//    for(Unsigned i = 0; i < prod(maxLocalShapeB); i++){
//        std::cout << " " << recvBuf[i];
//    }
//    std::cout << std::endl;

    const Location zeros(order, 0);
    const Location ones(order, 1);
    PackData unpackData;
    unpackData.loopShape = LocalShape();
    unpackData.dstBufStrides = LocalStrides();
    unpackData.srcBufStrides = Dimensions2Strides(MaxLocalShape());
    unpackData.loopStarts = zeros;
    unpackData.loopIncs = ones;

    PackCommHelper(unpackData, order - 1, &(recvBuf[0]), &(dataBuf[0]));
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
