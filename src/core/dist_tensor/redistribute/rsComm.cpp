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
void DistTensor<T>::ReduceScatterCommRedist(const DistTensor<T>& A, const ModeArray& reduceModes, const ModeArray& scatterModes, const ModeArray& commModes){
//    if(!CheckReduceScatterCommRedist(A, reduceMode, scatterMode))
//      LogicError("ReduceScatterRedist: Invalid redistribution request");
    const tmen::Grid& g = A.Grid();

    const mpi::Comm comm = GetCommunicatorForModes(commModes, g);

    if(!A.Participating())
        return;
    Unsigned sendSize, recvSize;

    //Determine buffer sizes for communication
    const Unsigned nRedistProcs = Max(1, prod(FilterVector(g.Shape(), commModes)));
    const ObjShape maxLocalShapeB = MaxLocalShape();
    recvSize = prod(maxLocalShapeB);
    sendSize = recvSize * nRedistProcs;

    T* auxBuf = this->auxMemory_.Require(sendSize + recvSize);
    T* sendBuf = &(auxBuf[0]);
    T* recvBuf = &(auxBuf[sendSize]);

    //Pack the data
    PROFILE_SECTION("RSPack");
    PackRSCommSendBuf(A, reduceModes, scatterModes, commModes, sendBuf);
    PROFILE_STOP;

    //Communicate the data
    PROFILE_SECTION("RSComm");
    mpi::ReduceScatter(sendBuf, recvBuf, recvSize, comm);
    PROFILE_STOP;

    if(!Participating()){
        this->auxMemory_.Release();
        return;
    }

    //Unpack the data (if participating)
    PROFILE_SECTION("RSUnpack");
    UnpackRSCommRecvBuf(recvBuf, A);
    PROFILE_STOP;
    this->auxMemory_.Release();
}

//TODO: Optimize striding when packing
//TODO: Ensure modeStrideFactor is being correctly applied (global to local information)
template <typename T>
void DistTensor<T>::PackRSCommSendBuf(const DistTensor<T>& A, const ModeArray& rModes, const ModeArray& sModes, const ModeArray& commModes, T * const sendBuf)
{
    Unsigned i;
    Unsigned order = A.Order();
    const T* dataBuf = A.LockedBuffer();

//    std::cout << "dataBuf:";
//    for(Unsigned i = 0; i < prod(A.LocalShape()); i++){
//        std::cout << " " << dataBuf[i];
//    }
//    std::cout << std::endl;

    const Location zeros(order, 0);
    const Location ones(order, 1);

    //GridView information
    const tmen::GridView gvA = A.GetGridView();
    const tmen::GridView gvB = GetGridView();
    const ObjShape gvAShape = gvA.ParticipatingShape();
    const ObjShape gvBShape = gvB.ParticipatingShape();

    //Different striding information
    std::vector<Unsigned> commLCMs = tmen::LCMs(gvA.ParticipatingShape(), gvB.ParticipatingShape());
    std::vector<Unsigned> modeStrideFactor = ElemwiseDivide(commLCMs, gvA.ParticipatingShape());
    //Set the mode stride factor to 1 for all reduceModes
    for(i = 0; i < rModes.size(); i++)
        modeStrideFactor[rModes[i]] = 1;

    const ObjShape sendShape = MaxLocalShape();

    Permutation invPerm = DetermineInversePermutation(A.localPerm_);

    PackData packData;
    packData.loopShape = PermuteVector(A.LocalShape(), invPerm);
    packData.srcBufStrides = ElemwiseProd(PermuteVector(A.LocalStrides(), invPerm), modeStrideFactor);

    packData.dstBufStrides = Dimensions2Strides(sendShape);
    packData.loopStarts = zeros;
    packData.loopIncs = modeStrideFactor;

    //Determine the first element we will accumulate to
    Location packElem = A.ModeShifts();
    for(i = 0; i < rModes.size(); i++)
        packElem[rModes[i]] = 0;

    if(ElemwiseLessThan(packElem, A.Shape())){
        ElemSelectData elemData;
        elemData.commModes = commModes;
        elemData.packElem = packElem;
        elemData.loopShape = modeStrideFactor;
        elemData.nElemsPerProc = prod(sendShape);
        elemData.srcElem = zeros;
        elemData.srcStrides = A.LocalStrides();
        elemData.permutation = A.localPerm_;

        ElemSelectHelper(packData, elemData, order - 1, A, &(dataBuf[0]), &(sendBuf[0]));
    }

//    std::cout << "sendBuf:";
//    for(Unsigned i = 0; i < prod(sendShape)* nRedistProcs; i++){
//        std::cout << " " << sendBuf[i];
//    }
//    std::cout << std::endl;
}

//TODO: Optimize stride when unpacking
template <typename T>
void DistTensor<T>::UnpackRSCommRecvBuf(const T * const recvBuf, const DistTensor<T>& A)
{
    const Unsigned order = Order();
    T* dataBuf = Buffer();

    const Location zeros(order, 0);
    const Location ones(order, 1);

//    std::cout << "recvBuf:";
//    for(Unsigned i = 0; i < prod(A.MaxLocalShape()); i++){
//        std::cout << " " << recvBuf[i];
//    }
//    std::cout << std::endl;


    PackData unpackData;
    unpackData.loopShape = LocalShape();
    unpackData.dstBufStrides = LocalStrides();
    unpackData.srcBufStrides = PermuteVector(Dimensions2Strides(MaxLocalShape()), localPerm_);
    unpackData.loopStarts = zeros;
    unpackData.loopIncs = ones;

    PackCommHelper(unpackData, order - 1, &(recvBuf[0]), &(dataBuf[0]));
}

#define PROTO(T) template class DistTensor<T>
#define COPY(T) \
  template DistTensor<T>::DistTensor( const DistTensor<T>& A )
#define FULL(T) \
  PROTO(T);


FULL(Int);
#ifndef DISABLE_FLOAT
FULL(float);
#endif
FULL(double);

#ifndef DISABLE_COMPLEX
#ifndef DISABLE_FLOAT
FULL(std::complex<float>);
#endif
FULL(std::complex<double>);
#endif

} //namespace tmen
