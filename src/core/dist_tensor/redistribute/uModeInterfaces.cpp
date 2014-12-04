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
    Unsigned i, j;
    ModeArray sorted = unitModes;
    std::sort(sorted.begin(), sorted.end());

    Permutation invPerm = DetermineInversePermutation(localPerm_);
    for(i = sorted.size() - 1; i < sorted.size(); i--){
        shape_.erase(shape_.begin() + sorted[i]);
        dist_.erase(dist_.begin() + sorted[i]);
        constrainedModeAlignments_.erase(constrainedModeAlignments_.begin() + sorted[i]);
        modeAlignments_.erase(modeAlignments_.begin() + sorted[i]);
        modeShifts_.erase(modeShifts_.begin() + sorted[i]);
        for(j = 0; j < localPerm_.size(); j++)
            if(localPerm_[j] > localPerm_[sorted[i]])
                localPerm_[j] -= 1;
        localPerm_.erase(localPerm_.begin() + sorted[i]);
    }
    tensor_.RemoveUnitModes(FilterVector(invPerm, unitModes));
    gridView_.RemoveUnitModes(unitModes);
    ResizeTo(Shape());
}

template<typename T>
void DistTensor<T>::RemoveUnitModeRedist(const Mode& unitMode){
    ModeArray modeArr(1);
    modeArr[0] = unitMode;
    RemoveUnitModesRedist(modeArr);
}

//NOTE: Assuming everything is correct, this is just a straight memcopy
template<typename T>
void DistTensor<T>::IntroduceUnitModesRedist(const std::vector<Unsigned>& unitModes){
//    if(!CheckIntroduceUnitModesRedist(B, A, newModePositions))
//        LogicError("IntroduceUnitModesRedist: Invalid redistribution request");

    Unsigned i, j;
    ModeArray sorted = unitModes;
    ModeArray blank(0);
    std::sort(sorted.begin(), sorted.end());
    shape_.reserve(shape_.size() + sorted.size());
    dist_.reserve(dist_.size() + sorted.size());
    constrainedModeAlignments_.reserve(constrainedModeAlignments_.size() + sorted.size());
    modeAlignments_.reserve(modeAlignments_.size() + sorted.size());
    modeShifts_.reserve(modeShifts_.size() + sorted.size());
    localPerm_.reserve(localPerm_.size() + sorted.size());
    Permutation invPerm = DetermineInversePermutation(localPerm_);
    for(i = 0; i < sorted.size(); i++){
        shape_.insert(shape_.begin() + sorted[i], 1);
        dist_.insert(dist_.begin() + sorted[i], blank);
        constrainedModeAlignments_.insert(constrainedModeAlignments_.begin() + sorted[i], false);
        modeAlignments_.insert(modeAlignments_.begin() + sorted[i], 0);
        modeShifts_.insert(modeShifts_.begin() + sorted[i], 0);
        for(j = 0; j < localPerm_.size(); j++)
            if(localPerm_[j] >= localPerm_[sorted[i]])
                localPerm_[j] += 1;
        localPerm_.insert(localPerm_.begin() + sorted[i], sorted[i]);
    }
    tensor_.IntroduceUnitModes(FilterVector(invPerm, unitModes));
    gridView_.IntroduceUnitModes(unitModes);
    ResizeTo(Shape());
}

template<typename T>
void DistTensor<T>::IntroduceUnitModeRedist(const Unsigned& newModePosition){
//    ModeArray modeArr(1);
//    modeArr[0] = newModePosition;
//    IntroduceUnitModesRedist(B, A, modeArr);
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
