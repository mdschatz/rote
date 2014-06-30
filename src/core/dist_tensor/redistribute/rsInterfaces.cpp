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

template <typename T>
void DistTensor<T>::PartialReduceScatterRedistFrom(const DistTensor<T>& A, const Mode reduceScatterMode){

    ObjShape tmpShape = A.Shape();
    tmpShape[reduceScatterMode] = A.GetGridView().Dimension(reduceScatterMode);
    this->ResizeTo(tmpShape);
    ReduceScatterCommRedist(A, reduceScatterMode, reduceScatterMode);
}

template <typename T>
void DistTensor<T>::ReduceScatterRedistFrom(const DistTensor<T>& A, const Mode reduceMode, const Mode scatterMode){

    ObjShape tmpShape = A.Shape();
    tmpShape[reduceMode] = A.GetGridView().Dimension(reduceMode);
    DistTensor<T> tmp(tmpShape, A.TensorDist(), A.Grid());

    TensorDistribution dist = A.TensorDist();
    dist[scatterMode] = ConcatenateVectors(dist[scatterMode], dist[reduceMode]);
    ModeDistribution blank(0);
    dist[reduceMode] = blank;
    ObjShape tmp2Shape = A.Shape();
    tmp2Shape[reduceMode] = 1;
    DistTensor<T> tmp2(tmp2Shape, dist, A.Grid());

    LocalReduce(tmp, A, reduceMode);
//    Print(tmp, "tmp after local reduce");
    tmp2.ReduceScatterCommRedist(tmp, reduceMode, scatterMode);
//    Print(tmp2, "tmp2 after global reduce");

    //B.RemoveUnitMode(reduceMode);
    ObjShape BShape = tmp2Shape;
    BShape.erase(BShape.begin() + reduceMode);
    this->ResizeTo(BShape);
    T* BBuf = this->Buffer();
    const T* tmp2Buf = tmp2.LockedBuffer();
    MemCopy(&(BBuf[0]), &(tmp2Buf[0]), prod(this->LocalShape()));
//    Print(*this, "B after full reduce");
}

#define PROTO(T) \
        template void DistTensor<T>::PartialReduceScatterRedistFrom(const DistTensor<T>& A, const Mode reduceScatterMode); \
        template void DistTensor<T>::ReduceScatterRedistFrom(const DistTensor<T>& A, const Mode reduceMode, const Mode scatterMode);

PROTO(int)
PROTO(float)
PROTO(double)
PROTO(Complex<float>)
PROTO(Complex<double>)

} //namespace tmen
