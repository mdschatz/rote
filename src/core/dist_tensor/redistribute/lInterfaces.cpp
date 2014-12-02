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

template<typename T>
void DistTensor<T>::LocalRedistFrom(const DistTensor<T>& A, const Mode localMode, const ModeArray& gridRedistModes){
    ModeArray lModes(1);
    lModes[0] = localMode;
    std::vector<ModeArray> commGroups(1);
    commGroups[0] = gridRedistModes;
    LocalRedistFrom(A, lModes, commGroups);
}

template<typename T>
void DistTensor<T>::LocalRedistFrom(const DistTensor<T>& A, const ModeArray& localModes, const std::vector<ModeArray>& gridRedistModes){
    ResizeTo(A);
    LocalCommRedist(A, localModes);
}

#define PROTO(T) template class DistTensor<T>
#define COPY(T) \
  template DistTensor<T>::DistTensor( const DistTensor<T>& A )
#define FULL(T) \
  PROTO(T);


FULL(Int);
#ifndef DISABLE_FLOAT
FULL(float);
#endif
FULL(double);

#ifndef DISABLE_COMPLEX
#ifndef DISABLE_FLOAT
FULL(std::complex<float>);
#endif
FULL(std::complex<double>);
#endif

} //namespace tmen
