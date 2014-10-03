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
    ResizeTo(A);

    ModeArray a2aModesFrom(2);
    a2aModesFrom[0] = a2aModes.first;
    a2aModesFrom[1] = a2aModes.second;
    ModeArray a2aModesTo(2);
    a2aModesTo[0] = a2aModes.second;
    a2aModesTo[1] = a2aModes.first;

    std::vector<ModeArray > commGroups(2);
    commGroups[0] = a2aCommGroups.first;
    commGroups[1] = a2aCommGroups.second;

    AllToAllRedistFrom(A, a2aModesFrom, a2aModesTo, commGroups);
}

template <typename T>
void DistTensor<T>::AllToAllRedistFrom(const DistTensor<T>& A, const ModeArray& a2aModesFrom, const ModeArray& a2aModesTo, const std::vector<ModeArray >& a2aCommGroups){
    Unsigned i;
    ResizeToUnderPerm(A);
    ModeArray commModes;
    for(i = 0; i < a2aCommGroups.size(); i++)
        commModes.insert(commModes.end(), a2aCommGroups[i].begin(), a2aCommGroups[i].end());
    std::sort(commModes.begin(), commModes.end());

    ModeArray changedA2AModes = ConcatenateVectors(a2aModesFrom, a2aModesTo);
    std::sort(changedA2AModes.begin(), changedA2AModes.end());
    changedA2AModes.erase(std::unique(changedA2AModes.begin(), changedA2AModes.end()), changedA2AModes.end());

    AllToAllCommRedist(A, changedA2AModes, commModes);
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
