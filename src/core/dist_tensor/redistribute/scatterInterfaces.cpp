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

namespace rote{

////////////////////////////////
// Workhorse interface
////////////////////////////////

//TODO: MAKE SURE ALL REDISTS WORK WITH blank commModes (size=0)
template <typename T>
void DistTensor<T>::ScatterRedistFrom(const DistTensor<T>& A, const ModeArray& commModes, const T alpha, const T beta){
    PROFILE_SECTION("ScatterRedist");
    this->ResizeTo(A);
    ModeArray sortedCommModes = commModes;
    SortVector(sortedCommModes);

	ScatterCommRedist(A, sortedCommModes, alpha, beta);

    PROFILE_STOP;
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
