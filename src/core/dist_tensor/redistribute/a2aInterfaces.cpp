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
    AllToAllDoubleModeCommRedist(A, a2aModes, a2aCommGroups);
}

template <typename T>
void DistTensor<T>::AllToAllRedistFrom(const DistTensor<T>& A, const ModeArray& a2aModesFrom, const ModeArray& a2aModesTo, const std::vector<ModeArray >& a2aCommGroups){
    ResizeTo(A);
    AllToAllCommRedist(A, a2aModesFrom, a2aModesTo, a2aCommGroups);
}

#define PROTO(T) \
        template void DistTensor<T>::AllToAllDoubleModeRedistFrom(const DistTensor<T>& A, const std::pair<Mode, Mode>& a2aIndices, const std::pair<ModeArray, ModeArray >& commGroups); \
        template void DistTensor<T>::AllToAllRedistFrom(const DistTensor<T>& A, const ModeArray& a2aModesFrom, const ModeArray& a2aModesTo, const std::vector<ModeArray >& a2aCommGroups);


PROTO(int)
PROTO(float)
PROTO(double)
PROTO(Complex<double>)
PROTO(Complex<float>)

} //namespace tmen
