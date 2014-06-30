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
void
DistTensor<T>::GatherToOneRedistFrom(const DistTensor<T>& A, const Mode gMode)
{
#ifndef RELEASE
    CallStackEntry cse("DistTesnor::GatherToOneRedistFrom");
#endif
    ModeArray gModeDist = A.ModeDist(gMode);
    GatherToOneRedistFrom(A, gMode, gModeDist);
}

template <typename T>
void DistTensor<T>::GatherToOneRedistFrom(const DistTensor<T>& A, const Mode gMode, const ModeArray& gridModes){
    this->ResizeTo(A);
    GatherToOneCommRedist(A, gMode, gridModes);
}



#define PROTO(T) \
        template void DistTensor<T>::GatherToOneRedistFrom(const DistTensor<T>& A, const Mode gMode); \
        template void DistTensor<T>::GatherToOneRedistFrom(const DistTensor<T>& A, const Mode gMode, const ModeArray& gridModes);

PROTO(int)
PROTO(float)
PROTO(double)
PROTO(Complex<float>)
PROTO(Complex<double>)

} //namespace tmen
