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

template <typename T>
void DistTensor<T>::AllToAllDoubleModeRedistFrom(const DistTensor<T>& A, const std::pair<Mode, Mode>& a2aModes, const std::pair<ModeArray, ModeArray >& a2aCommGroups){
    ModeArray a2aModesFrom(2);
    a2aModesFrom[0] = a2aModes.first;
    a2aModesFrom[1] = a2aModes.second;
    ModeArray a2aModesTo(2);
    a2aModesTo[0] = a2aModes.second;
    a2aModesTo[1] = a2aModes.first;

    std::vector<ModeArray > commGroups(2);
    commGroups[0] = a2aCommGroups.first;
    commGroups[1] = a2aCommGroups.second;

    ModeArray commModes;
    for(Unsigned i = 0; i < commGroups.size(); i++)
        commModes.insert(commModes.end(), commGroups[i].begin(), commGroups[i].end());

    AllToAllRedistFrom(A, commModes);
}

template <typename T>
void DistTensor<T>::AllToAllRedistFrom(const DistTensor<T>& A, const ModeArray& commModes){
    PROFILE_SECTION("A2ARedist");
    ResizeTo(A);
    ModeArray sortedCommModes = commModes;

    std::sort(sortedCommModes.begin(), sortedCommModes.end());

    AllToAllCommRedist(A, sortedCommModes);
    PROFILE_STOP;
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
