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
DistTensor<T>::AllReduceUpdateRedistFrom(const T alpha, const DistTensor<T>& A, const T beta, const ModeArray& rModes)
{
    PROFILE_SECTION("ARRedist");
    ReduceUpdateRedistFrom(AR, alpha, A, beta, rModes);
    PROFILE_STOP;
}

////////////////////////////////
// Set Wrappers
////////////////////////////////

template <typename T>
void DistTensor<T>::AllReduceRedistFrom(const DistTensor<T>& A, const ModeArray& rModes){
    ObjShape newShape = NegFilterVector(A.Shape(), rModes);
    this->ResizeTo(newShape);
    AllReduceUpdateRedistFrom(T(1), A, T(0), rModes);
}

template <typename T>
void DistTensor<T>::AllReduceRedistFrom(const DistTensor<T>& A, const Mode reduceMode){
    ModeArray reduceModes(1);
    reduceModes[0] = reduceMode;
    AllReduceRedistFrom(A, reduceModes);
}

////////////////////////////////
// Update Wrappers
////////////////////////////////

template<typename T>
void
DistTensor<T>::AllReduceUpdateRedistFrom(const DistTensor<T>& A, const T beta, const ModeArray& reduceModes)
{
    AllReduceUpdateRedistFrom(T(1), A, beta, reduceModes);
}

template<typename T>
void
DistTensor<T>::AllReduceUpdateRedistFrom(const DistTensor<T>& A, const T beta, const Mode reduceMode)
{
    ModeArray reduceModes(1);
    reduceModes[0] = reduceMode;
    AllReduceUpdateRedistFrom(A, beta, reduceModes);
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
