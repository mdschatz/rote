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
void DistTensor<T>::PartialReduceToOneRedistFrom(const DistTensor<T>& A, const Mode rMode){
//    ObjShape tmpShape = A.Shape();
//    tmpShape[rMode] = A.GetGridView().Dimension(rMode);
//    ResizeTo(tmpShape);
//    ReduceToOneCommRedist(A, rMode);
    ModeArray rModes(1);
    rModes[0] = rMode;
    ObjShape tmpShape = A.Shape();
    tmpShape[rMode] = A.GetGridView().Dimension(rMode);
    ResizeTo(tmpShape);
    ReduceToOneRedistFrom(A, rModes);
}

template <typename T>
void DistTensor<T>::ReduceToOneRedistFrom(const DistTensor<T>& A, const Mode rMode){
    ModeArray rModes(1);
    rModes[0] = rMode;

    ReduceToOneRedistFrom(A, rModes);
}

template <typename T>
void DistTensor<T>::ReduceToOneRedistFrom(const DistTensor<T>& A, const ModeArray& rModes){
    Unsigned i;
    const tmen::GridView gv = A.GetGridView();
    const tmen::Grid& g = A.Grid();
    TensorDistribution dist = A.TensorDist();
    ModeDistribution blank(0);

    ObjShape tmpShape = A.Shape();
    for(i = 0; i < rModes.size(); i++)
        tmpShape[rModes[i]] = gv.Dimension(rModes[i]);
    DistTensor<T> tmp(tmpShape, A.TensorDist(), g);

    LocalReduce(tmp, A, rModes);

    ObjShape tmp2Shape = A.Shape();
    for(i = 0; i < rModes.size(); i++)
        tmp2Shape[rModes[i]] = 1;
    DistTensor<T> tmp2(tmp2Shape, A.TensorDist(), g);

    tmp2.ReduceToOneCommRedist(tmp, rModes);

    ModeArray sortedRModes = rModes;
    std::sort(sortedRModes.begin(), sortedRModes.end());
    ObjShape BShape = tmp2Shape;
    for(i = sortedRModes.size() - 1; i < sortedRModes.size(); i--){
        BShape.erase(BShape.begin() + sortedRModes[i]);
    }

    ResizeTo(BShape);
    T* thisBuf = Buffer();
    const T* tmp2Buf = tmp2.LockedBuffer();

    //Only do this if we know we are copying into a scalar
    if(Order() == 0)
        MemCopy(&(thisBuf[0]), &(tmp2Buf[0]), 1);
    else
        MemCopy(&(thisBuf[0]), &(tmp2Buf[0]), prod(tmp2.LocalShape()));
}

#define PROTO(T) \
        template void DistTensor<T>::PartialReduceToOneRedistFrom(const DistTensor<T>& A, const Mode reduceMode); \
        template void DistTensor<T>::ReduceToOneRedistFrom(const DistTensor<T>& A, const Mode reduceMode); \
        template void DistTensor<T>::ReduceToOneRedistFrom(const DistTensor<T>& A, const ModeArray& rModes);

PROTO(int)
PROTO(float)
PROTO(double)
PROTO(Complex<float>)
PROTO(Complex<double>)

} //namespace tmen
