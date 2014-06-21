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

//TODO: Check that allToAllIndices and commGroups are valid
template <typename T>
Int DistTensor<T>::CheckAllToAllDoubleModeRedist(const DistTensor<T>& A, const std::pair<Mode, Mode>& allToAllModes, const std::pair<ModeArray, ModeArray >& a2aCommGroups){
    if(A.Order() != this->Order())
        LogicError("CheckAllToAllDoubleModeRedist: Objects being redistributed must be of same order");
    Unsigned i;
    for(i = 0; i < A.Order(); i++){
        if(i != allToAllModes.first && i != allToAllModes.second){
            if(this->ModeDist(i) != A.ModeDist(i))
                LogicError("CheckAlLToAllDoubleModeRedist: Non-redist modes must have same distribution");
        }
    }
    return 1;
}

template <typename T>
void DistTensor<T>::AllToAllDoubleModeRedist(const DistTensor<T>& A, const std::pair<Mode, Mode>& a2aModes, const std::pair<ModeArray, ModeArray >& a2aCommGroups){
    if(!this->CheckAllToAllDoubleModeRedist(A, a2aModes, a2aCommGroups))
        LogicError("AllToAllDoubleModeRedist: Invalid redistribution request");

    this->AllToAllDoubleModeCommRedist(A, a2aModes, a2aCommGroups);
}

#define PROTO(T) \
        template Int  DistTensor<T>::CheckAllToAllDoubleModeRedist(const DistTensor<T>& A, const std::pair<Mode, Mode>& a2aModes, const std::pair<ModeArray, ModeArray >& a2aCommGroups); \
        template void DistTensor<T>::AllToAllDoubleModeRedist(const DistTensor<T>& A, const std::pair<Mode, Mode>& a2aIndices, const std::pair<ModeArray, ModeArray >& commGroups);


PROTO(int)
PROTO(float)
PROTO(double)
PROTO(Complex<double>)
PROTO(Complex<float>)

} //namespace tmen
