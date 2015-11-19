/*
   Copyright (c) 2009-2013, Jack Poulson
                      2013, Jeff Hammond
                      2013, Jed Brown
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#include "rote.hpp"
#include <algorithm>

namespace rote{

////////////////////////////////
// Workhorse interface
////////////////////////////////

template<typename T>
void
DistTensor<T>::ReduceScatterUpdateRedistFrom(const T alpha, const DistTensor<T>& A, const T beta, const ModeArray& rModes)
{
#ifndef RELEASE
    CallStackEntry cse("DistTensor::ReduceScatterUpdateRedistFrom");
#endif
    PROFILE_SECTION("RSRedist");
    ReduceUpdateRedistFrom(RS, alpha, A, beta, rModes);
    PROFILE_STOP;
}

////////////////////////////////
// Set Wrappers
////////////////////////////////

template <typename T>
void DistTensor<T>::ReduceScatterRedistFrom(const DistTensor<T>& A, const ModeArray& rModes){
    ObjShape newShape = NegFilterVector(A.Shape(), rModes);
    this->ResizeTo(newShape);
    ReduceScatterUpdateRedistFrom(T(1), A, T(0), rModes);
}

template <typename T>
void DistTensor<T>::ReduceScatterRedistFrom(const DistTensor<T>& A, const Mode reduceMode){
    ModeArray reduceModes(1);
    reduceModes[0] = reduceMode;
    ReduceScatterRedistFrom(A, reduceModes);
}

template <typename T>
void DistTensor<T>::ReduceScatterRedistFrom(const T alpha, const DistTensor<T>& A, const ModeArray& rModes){
    ObjShape newShape = NegFilterVector(A.Shape(), rModes);
    ResizeTo(newShape);
    ReduceScatterUpdateRedistFrom(alpha, A, T(0), rModes);
}

template <typename T>
void DistTensor<T>::ReduceScatterRedistFrom(const T alpha, const DistTensor<T>& A, const Mode reduceMode){
    ModeArray reduceModes(1);
    reduceModes[0] = reduceMode;
    ReduceScatterRedistFrom(alpha, A, reduceModes);
}

////////////////////////////////
// Update Wrappers
////////////////////////////////

template<typename T>
void
DistTensor<T>::ReduceScatterUpdateRedistFrom(const DistTensor<T>& A, const T beta, const ModeArray& reduceModes)
{
#ifndef RELEASE
    CallStackEntry cse("DistTensor::ReduceScatterUpdateRedistFrom");
#endif
    ReduceScatterUpdateRedistFrom(T(1), A, beta, reduceModes);
}

template<typename T>
void
DistTensor<T>::ReduceScatterUpdateRedistFrom(const DistTensor<T>& A, const T beta, const Mode reduceMode)
{
#ifndef RELEASE
    CallStackEntry cse("DistTensor::ReduceScatterUpdateRedistFrom");
#endif
    ModeArray reduceModes(1);
    reduceModes[0] = reduceMode;
    ReduceScatterUpdateRedistFrom(A, beta, reduceModes);
}

template<typename T>
void
DistTensor<T>::ReduceScatterUpdateRedistFrom(const DistTensor<T>& A, const ModeArray& reduceModes)
{
#ifndef RELEASE
    CallStackEntry cse("DistTensor::ReduceScatterUpdateRedistFrom");
#endif
    ReduceScatterUpdateRedistFrom(A, T(1), reduceModes);
}

#define FULL(T) \
    template class DistTensor<T>;

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

} //namespace rote
