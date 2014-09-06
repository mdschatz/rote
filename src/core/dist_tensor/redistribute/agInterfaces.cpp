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
//#include "/tensormental/mc_mr.hpp"
#include <algorithm>

namespace tmen{

template <typename T>
void
DistTensor<T>::AllGatherRedistFrom(const DistTensor<T>& A, const Mode& allGatherMode, const ModeArray& redistModes ){
    ResizeTo(A);
    ModeArray agModes(1);
    agModes[0] = allGatherMode;
    std::vector<ModeArray> commGroups(1);
    commGroups[0] = redistModes;
    AllGatherRedistFrom(A, agModes, commGroups);
}

template <typename T>
void
DistTensor<T>::AllGatherRedistFrom(const DistTensor<T>& A, const ModeArray& allGatherModes, const std::vector<ModeArray>& redistGroups ){
    ResizeTo(A);
    AllGatherCommRedist(A, allGatherModes, redistGroups);
}

#define PROTO(T) \
        template void DistTensor<T>::AllGatherRedistFrom(const DistTensor<T>& A, const Mode& allGatherMode, const ModeArray& redistModes ); \
        template void DistTensor<T>::AllGatherRedistFrom(const DistTensor<T>& A, const ModeArray& allGatherModes, const std::vector<ModeArray>& redistGroups );

PROTO(int)
PROTO(float)
PROTO(double)
PROTO(Complex<double>)
PROTO(Complex<float>)

} //namespace tmen
