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
//TODO: Make sure outgoing reduce Mode differs from incoming (partial reduction forms a new Mode)
template <typename T>
Int CheckPartialReduceScatterRedist(const DistTensor<T>& B, const DistTensor<T>& A, const Mode reduceScatterMode){
    Unsigned i;
    const tmen::GridView gvA = A.GridView();

    const Unsigned AOrder = A.Order();
    const Unsigned BOrder = B.Order();

    //Test order retained
    if(BOrder != AOrder){
        LogicError("CheckPartialReduceScatterRedist: Partial Reduction must retain mode being reduced");
    }

    //Test dimension has been resized correctly
    //NOTE: Uses fancy way of performing Ceil() on integer division
    if(B.Dimension(reduceScatterMode) != Max(1,MaxLength(A.Dimension(reduceScatterMode), gvA.Dimension(reduceScatterMode))))
        LogicError("CheckPartialReduceScatterRedist: Partial Reduction must reduce mode dimension by factor Dimension/Grid Dimension");

    //Make sure all indices are distributed similarly between input and output (excluding reduce+scatter indices)
    for(i = 0; i < BOrder; i++){
        Mode mode = i;
        if(AnyElemwiseNotEqual(B.ModeDist(mode), A.ModeDist(mode)))
            LogicError("CheckPartialReduceScatterRedist: All modes must be distributed similarly");
    }
    return 1;
}

//TODO: Properly Check indices and distributions match between input and output
//TODO: FLESH OUT THIS CHECK
template <typename T>
Int CheckReduceScatterRedist(const DistTensor<T>& B, const DistTensor<T>& A, const Mode reduceMode, const Mode scatterMode){
//    Unsigned i;
//    const tmen::GridView gvA = A.GridView();
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
void PartialReduceScatterRedist(DistTensor<T>& B, const DistTensor<T>& A, const Mode reduceScatterMode){
    if(!CheckPartialReduceScatterRedist(B, A, reduceScatterMode))
        LogicError("PartialReduceScatterRedist: Invalid redistribution request");

    ReduceScatterCommRedist(B, A, reduceScatterMode, reduceScatterMode);
}

template <typename T>
void ReduceScatterRedist(DistTensor<T>& B, const DistTensor<T>& A, const Mode reduceMode, const Mode scatterMode){
    if(!CheckReduceScatterRedist(B, A, reduceMode, scatterMode))
      LogicError("ReduceScatterRedist: Invalid redistribution request");

    ObjShape tmpShape = A.Shape();
    tmpShape[reduceMode] = A.GridView().Dimension(reduceMode);
    DistTensor<T> tmp(tmpShape, A.TensorDist(), A.Grid());

    TensorDistribution dist = A.TensorDist();
    dist[scatterMode] = ConcatenateVectors(dist[scatterMode], dist[reduceMode]);
    ModeDistribution blank(0);
    dist[reduceMode] = blank;
    ObjShape tmp2Shape = A.Shape();
    tmp2Shape[reduceMode] = 1;
    DistTensor<T> tmp2(tmp2Shape, dist, A.Grid());

    LocalReduce(tmp, A, reduceMode);
    Print(tmp, "tmp after local reduce");
    ReduceScatterCommRedist(tmp2, tmp, reduceMode, scatterMode);
    Print(tmp2, "tmp2 after global reduce");

    //B.RemoveUnitMode(reduceMode);
    T* BBuf = B.Buffer();
    const T* tmp2Buf = tmp2.LockedBuffer();
    MemCopy(&(BBuf[0]), &(tmp2Buf[0]), prod(B.LocalShape()));
    Print(B, "B after full reduce");
}

#define PROTO(T) \
        template Int CheckPartialReduceScatterRedist(const DistTensor<T>& B, const DistTensor<T>& A, const Mode reduceScatterMode); \
        template Int CheckReduceScatterRedist(const DistTensor<T>& B, const DistTensor<T>& A, const Mode reduceMode, const Mode scatterMode); \
        template void PartialReduceScatterRedist(DistTensor<T>& B, const DistTensor<T>& A, const Mode reduceScatterMode); \
        template void ReduceScatterRedist(DistTensor<T>& B, const DistTensor<T>& A, const Mode reduceMode, const Mode scatterMode);

PROTO(int)
PROTO(float)
PROTO(double)

} //namespace tmen
