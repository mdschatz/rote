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

template <typename T>
void DistTensor<T>::PermutationRedistFrom(const DistTensor<T>& A, const ModeArray& redistModes, const T alpha){
	NOT_USED(redistModes);
    PROFILE_SECTION("PermuteRedist");
    this->ResizeTo(A);
    ModeArray commModes = GetCommonSuffix(A.TensorDist(), this->TensorDist()).UsedModes().Entries();

    commModes = Unique(commModes);
    PermutationCommRedist(A, commModes, alpha);
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
