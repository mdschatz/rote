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

//TODO: Make sure these checks are correct (look at LDim, strides, distributions, etc).
template <typename T>
Int CheckRemoveUnitModesRedist(const DistTensor<T>& B, const ModeArray& unitModes){
    return 1;
}

template <typename T>
Int CheckRemoveUnitModeRedist(const DistTensor<T>& B, const Mode& unitMode){
    ModeArray modeArr(1);
    modeArr[0] = unitMode;
    CheckRemoveUnitModesRedist(B, modeArr);
}

template <typename T>
Int CheckIntroduceUnitModesRedist(const DistTensor<T>& B, const std::vector<Unsigned>& newModePositions){

    return 1;
}

template <typename T>
Int CheckIntroduceUnitModeRedist(const DistTensor<T>& B, const Unsigned& newModePosition){
    std::vector<Unsigned> modeArr(1);
    modeArr[0] = newModePosition;
    CheckIntroduceUnitModesRedist(B, modeArr);
}

//NOTE: Assuming everything is correct, this is just a straight memcopy
template<typename T>
void RemoveUnitModesRedist(DistTensor<T>& B, const ModeArray& unitModes){
//    if(!CheckRemoveUnitModesRedist(B, unitModes))
//        LogicError("RemoveUnitModesRedist: Invalid redistribution request");
//
//    Unsigned i;
//    std::sort(unitModes.begin(), unitModes.end());
//    for(i = unitModes.size() - 1; i >= 0; i--){
//        B.shape_.
//    }
//
//    const Location start(order, 0);
//    T* dst = B.Buffer(start);
//    const T* src = A.LockedBuffer(start);
//    MemCopy(&(dst[0]), &(src[0]), prod(A.LocalShape()));
}

template<typename T>
void RemoveUnitModeRedist(DistTensor<T>& B, const Mode& unitMode){
//    ModeArray modeArr(1);
//    modeArr[0] = unitMode;
//    RemoveUnitModesRedist(B, A, modeArr);
}

//NOTE: Assuming everything is correct, this is just a straight memcopy
template<typename T>
void IntroduceUnitModesRedist(DistTensor<T>& B, const std::vector<Unsigned>& newModePositions){
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
void IntroduceUnitModeRedist(DistTensor<T>& B, const Unsigned& newModePosition){
//    ModeArray modeArr(1);
//    modeArr[0] = newModePosition;
//    IntroduceUnitModesRedist(B, A, modeArr);
}
#define PROTO(T) \
        template Int CheckRemoveUnitModesRedist(const DistTensor<T>& B, const ModeArray& unitModes); \
        template Int CheckRemoveUnitModeRedist(const DistTensor<T>& B, const Mode& unitMode); \
        template Int CheckIntroduceUnitModesRedist(const DistTensor<T>& B, const std::vector<Unsigned>& newModePositions); \
        template Int CheckIntroduceUnitModeRedist(const DistTensor<T>& B, const Unsigned& newModePosition); \
        template void RemoveUnitModesRedist(DistTensor<T>& B, const ModeArray& unitModes); \
        template void RemoveUnitModeRedist(DistTensor<T>& B, const Mode& unitMode); \
        template void IntroduceUnitModesRedist(DistTensor<T>& B, const std::vector<Unsigned>& newModePositions); \
        template void IntroduceUnitModeRedist(DistTensor<T>& B, const Unsigned& newModePosition);

PROTO(int)
PROTO(float)
PROTO(double)
} //namespace tmen
