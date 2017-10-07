/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#include "rote.hpp"

namespace rote {

template <typename T>
DistTensor<T>::DistTensor(const rote::Grid &grid) : DistTensorBase<T>(grid) {}

template <typename T>
DistTensor<T>::DistTensor(const Unsigned order, const rote::Grid &grid)
    : DistTensorBase<T>(order, grid) {}

template <typename T>
DistTensor<T>::DistTensor(const TensorDistribution &dist,
                          const rote::Grid &grid)
    : DistTensorBase<T>(dist, grid) {}

template <typename T>
DistTensor<T>::DistTensor(const ObjShape &shape, const TensorDistribution &dist,
                          const rote::Grid &grid)
    : DistTensorBase<T>(shape, dist, grid) {}

template <typename T>
DistTensor<T>::DistTensor(const ObjShape &shape, const TensorDistribution &dist,
                          const std::vector<Unsigned> &modeAlignments,
                          const rote::Grid &g)
    : DistTensorBase<T>(shape, dist, modeAlignments, g) {}

template <typename T>
DistTensor<T>::DistTensor(const ObjShape &shape, const TensorDistribution &dist,
                          const std::vector<Unsigned> &modeAlignments,
                          const std::vector<Unsigned> &strides,
                          const rote::Grid &g)
    : DistTensorBase<T>(shape, dist, modeAlignments, strides, g) {}

template <typename T>
DistTensor<T>::DistTensor(const ObjShape &shape, const TensorDistribution &dist,
                          const std::vector<Unsigned> &modeAlignments,
                          const T *buffer, const std::vector<Unsigned> &strides,
                          const rote::Grid &g)
    : DistTensorBase<T>(shape, dist, modeAlignments, buffer, strides, g) {}

template <typename T>
DistTensor<T>::DistTensor(const ObjShape &shape, const TensorDistribution &dist,
                          const std::vector<Unsigned> &modeAlignments,
                          T *buffer, const std::vector<Unsigned> &strides,
                          const rote::Grid &g)
    : DistTensorBase<T>(shape, dist, modeAlignments, buffer, strides, g) {}

//////////////////////////////////
/// String distribution versions
//////////////////////////////////

template <typename T>
DistTensor<T>::DistTensor(const std::string &dist, const rote::Grid &grid)
    : DistTensorBase<T>(dist, grid) {}

template <typename T>
DistTensor<T>::DistTensor(const ObjShape &shape, const std::string &dist,
                          const rote::Grid &grid)
    : DistTensorBase<T>(shape, dist, grid) {}

template <typename T>
DistTensor<T>::DistTensor(const ObjShape &shape, const std::string &dist,
                          const std::vector<Unsigned> &modeAlignments,
                          const rote::Grid &g)
    : DistTensorBase<T>(shape, dist, modeAlignments, g) {}

template <typename T>
DistTensor<T>::DistTensor(const ObjShape &shape, const std::string &dist,
                          const std::vector<Unsigned> &modeAlignments,
                          const std::vector<Unsigned> &strides,
                          const rote::Grid &g)
    : DistTensorBase<T>(shape, dist, modeAlignments, strides, g) {}

template <typename T>
DistTensor<T>::DistTensor(const ObjShape &shape, const std::string &dist,
                          const std::vector<Unsigned> &modeAlignments,
                          const T *buffer, const std::vector<Unsigned> &strides,
                          const rote::Grid &g)
    : DistTensorBase<T>(shape, dist, modeAlignments, buffer, strides, g) {}

template <typename T>
DistTensor<T>::DistTensor(const ObjShape &shape, const std::string &dist,
                          const std::vector<Unsigned> &modeAlignments,
                          T *buffer, const std::vector<Unsigned> &strides,
                          const rote::Grid &g)
    : DistTensorBase<T>(shape, dist, modeAlignments, buffer, strides, g) {}

template <typename T> DistTensor<T>::~DistTensor() {}

#define FULL(T) template class DistTensor<T>;

FULL(Int)
#ifndef DISABLE_FLOAT
FULL(float)
#endif
FULL(double)

#ifndef DISABLE_COMPLEX
#ifndef DISABLE_FLOAT
FULL(std::complex<float>)
#endif
FULL(std::complex<double>)
#endif

} // namespace rote
