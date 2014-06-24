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

//NOTE: Assuming everything is correct, this is just a straight memcopy
template<typename T>
void DistTensor<T>::RemoveUnitModesRedist(const ModeArray& unitModes){
    Unsigned i;
    ModeArray sorted = unitModes;
    std::sort(sorted.begin(), sorted.end());
    for(i = sorted.size() - 1; i >= 0; i--){
        this->shape_.erase(this->shape_.begin() + sorted[i]);
    }
}

template<typename T>
void DistTensor<T>::RemoveUnitModeRedist(const Mode& unitMode){
    ModeArray modeArr(1);
    modeArr[0] = unitMode;
    RemoveUnitModesRedist(modeArr);
}

//NOTE: Assuming everything is correct, this is just a straight memcopy
template<typename T>
void DistTensor<T>::IntroduceUnitModesRedist(const std::vector<Unsigned>& newModePositions){
//    if(!CheckIntroduceUnitModesRedist(B, A, newModePositions))
//        LogicError("IntroduceUnitModesRedist: Invalid redistribution request");
//
//    const Unsigned order = A.Order();
//    const Location start(order, 0);
//    T* dst = B.Buffer(start);
//    const T* src = A.LockedBuffer(start);
//    MemCopy(&(dst[0]), &(src[0]), prod(A.LocalShape()));
}

template<typename T>
void DistTensor<T>::IntroduceUnitModeRedist(const Unsigned& newModePosition){
//    ModeArray modeArr(1);
//    modeArr[0] = newModePosition;
//    IntroduceUnitModesRedist(B, A, modeArr);
}

#define PROTO(T) \
        template void DistTensor<T>::RemoveUnitModesRedist(const ModeArray& unitModes); \
        template void DistTensor<T>::RemoveUnitModeRedist(const Mode& unitMode); \
        template void DistTensor<T>::IntroduceUnitModesRedist(const std::vector<Unsigned>& newModePositions); \
        template void DistTensor<T>::IntroduceUnitModeRedist(const Unsigned& newModePosition);

PROTO(int)
PROTO(float)
PROTO(double)
PROTO(Complex<float>)
PROTO(Complex<double>)
} //namespace tmen
