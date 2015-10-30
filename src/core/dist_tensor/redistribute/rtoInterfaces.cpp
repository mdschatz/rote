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

////////////////////////////////
// Workhorse interface
////////////////////////////////

template<typename T>
void
DistTensor<T>::ReduceToOneUpdateRedistFrom(const T alpha, const DistTensor<T>& A, const T beta, const ModeArray& rModes)
{
#ifndef RELEASE
    CallStackEntry cse("DistTensor::ReduceToOneUpdateRedistFrom");
#endif
    PROFILE_SECTION("RTORedist");
    ReduceUpdateRedistFrom(RTO, alpha, A, beta, rModes);
    PROFILE_STOP;
}

////////////////////////////////
// Set interfaces
////////////////////////////////

template <typename T>
void DistTensor<T>::ReduceToOneRedistFrom(const DistTensor<T>& A, const ModeArray& rModes){

    ReduceToOneUpdateRedistFrom(T(1), A, T(0), rModes);
}

template <typename T>
void DistTensor<T>::ReduceToOneRedistFrom(const DistTensor<T>& A, const Mode rMode){
    ModeArray rModes(1);
    rModes[0] = rMode;

    ReduceToOneRedistFrom(A, rModes);
}

template<typename T>
void
DistTensor<T>::ReduceToOneUpdateRedistFrom(const DistTensor<T>& A, const T beta, const ModeArray& reduceModes)
{
#ifndef RELEASE
    CallStackEntry cse("DistTensor::ReduceToOneUpdateRedistFrom");
#endif

    ReduceToOneUpdateRedistFrom(T(1), A, beta, reduceModes);
}

template <typename T>
void DistTensor<T>::ReduceToOneUpdateRedistFrom(const DistTensor<T>& A, const T beta, const Mode rMode){
    ModeArray rModes(1);
    rModes[0] = rMode;

    ReduceToOneUpdateRedistFrom(A, beta, rModes);
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

} //namespace tmen
