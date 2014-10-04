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
    ModeArray agModes(1);
    agModes[0] = allGatherMode;
    std::vector<ModeArray> commGroups(1);
    commGroups[0] = redistModes;
    AllGatherRedistFrom(A, agModes, commGroups);
}

template <typename T>
void
DistTensor<T>::AllGatherRedistFrom(const DistTensor<T>& A, const ModeArray& allGatherModes, const std::vector<ModeArray>& redistGroups ){
    Unsigned i;
    ResizeTo(A);
    ModeArray commModes;
    for(i = 0; i < redistGroups.size(); i++)
        commModes.insert(commModes.end(), redistGroups[i].begin(), redistGroups[i].end());
    std::sort(commModes.begin(), commModes.end());
    AllGatherCommRedist(A, allGatherModes, commModes);
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
FULL(Complex<float>);
#endif
FULL(Complex<double>);
#endif

} //namespace tmen
