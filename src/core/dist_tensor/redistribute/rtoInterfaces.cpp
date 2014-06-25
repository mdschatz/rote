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

    ReduceToOneCommRedist(A, rMode);
}

template <typename T>
void DistTensor<T>::ReduceToOneRedistFrom(const DistTensor<T>& A, const Mode rMode){
    ObjShape tmpShape = A.Shape();
    tmpShape[rMode] = A.GetGridView().Dimension(rMode);
    DistTensor<T> tmp(tmpShape, A.TensorDist(), A.Grid());

    printf("pre lreduce\n");
    LocalReduce(tmp, A, rMode);

    ObjShape tmp2Shape = A.Shape();
    tmp2Shape[rMode] = 1;
    DistTensor<T> tmp2(tmp2Shape, A.TensorDist(), A.Grid());

    printf("pre rto\n");
    tmp2.ReduceToOneCommRedist(tmp, rMode);
    printf("pre copy\n");
    T* thisBuf = this->Buffer();
    const T* tmp2Buf = tmp2.LockedBuffer();

    MemCopy(&(thisBuf[0]), &(tmp2Buf[0]), prod(tmp2.LocalShape()));
    PrintData(*this, "this");
}

#define PROTO(T) \
        template void DistTensor<T>::PartialReduceToOneRedistFrom(const DistTensor<T>& A, const Mode reduceMode); \
        template void DistTensor<T>::ReduceToOneRedistFrom(const DistTensor<T>& A, const Mode reduceMode);

PROTO(int)
PROTO(float)
PROTO(double)
PROTO(Complex<float>)
PROTO(Complex<double>)

} //namespace tmen
