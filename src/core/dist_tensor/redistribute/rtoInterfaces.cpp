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
    this->SetAlignmentsAndResize(A.Alignments(), A.Shape());

    ObjShape tmpShape = A.Shape();
    tmpShape[rMode] = A.GetGridView().Dimension(rMode);
    DistTensor<T> tmp(tmpShape, A.TensorDist(), A.Grid());

    LocalReduce(tmp, A, rMode);

    this->shape_[rMode] = 1;

    this->ReduceToOneCommRedist(tmp, rMode);

    this->RemoveUnitModeRedist(rMode);
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
