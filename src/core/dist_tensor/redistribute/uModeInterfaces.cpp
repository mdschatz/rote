/*
   Copyright (c) 2009-2013, Jack Poulson
                      2013, Jeff Hammond
                      2013, Jed Brown
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#include "rote.hpp"
#include <algorithm>

namespace rote{

//NOTE: Assuming everything is correct, this is just a straight memcopy
template<typename T>
void DistTensor<T>::RemoveUnitModesRedist(const ModeArray& unitModes){
    Unsigned i, j;
    ModeArray sorted = unitModes;
    SortVector(sorted);

    Permutation invPerm = DetermineInversePermutation(this->localPerm_);
    for(i = sorted.size() - 1; i < sorted.size(); i--){
        this->shape_.erase(this->shape_.begin() + sorted[i]);
        this->dist_.erase(this->dist_.begin() + sorted[i]);
        this->constrainedModeAlignments_.erase(this->constrainedModeAlignments_.begin() + sorted[i]);
        this->modeAlignments_.erase(this->modeAlignments_.begin() + sorted[i]);
        this->modeShifts_.erase(this->modeShifts_.begin() + sorted[i]);
        for(j = 0; j < this->localPerm_.size(); j++)
            if(this->localPerm_[j] > this->localPerm_[sorted[i]])
            	this->localPerm_[j] -= 1;
        this->localPerm_.erase(this->localPerm_.begin() + sorted[i]);
    }
    this->tensor_.RemoveUnitModes(FilterVector(invPerm, unitModes));
    this->gridView_.RemoveUnitModes(unitModes);
    this->ResizeTo(this->Shape());
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
    SortVector(sorted);
    this->shape_.reserve(this->shape_.size() + sorted.size());
    this->dist_.reserve(this->dist_.size() + sorted.size());
    this->constrainedModeAlignments_.reserve(this->constrainedModeAlignments_.size() + sorted.size());
    this->modeAlignments_.reserve(this->modeAlignments_.size() + sorted.size());
    this->modeShifts_.reserve(this->modeShifts_.size() + sorted.size());
    this->localPerm_.reserve(this->localPerm_.size() + sorted.size());
    Permutation invPerm = DetermineInversePermutation(this->localPerm_);
    for(i = 0; i < sorted.size(); i++){
    	this->shape_.insert(this->shape_.begin() + sorted[i], 1);
    	this->dist_.insert(this->dist_.begin() + sorted[i], blank);
    	this->constrainedModeAlignments_.insert(this->constrainedModeAlignments_.begin() + sorted[i], false);
    	this->modeAlignments_.insert(this->modeAlignments_.begin() + sorted[i], 0);
    	this->modeShifts_.insert(this->modeShifts_.begin() + sorted[i], 0);
        for(j = 0; j < this->localPerm_.size(); j++)
            if(this->localPerm_[j] >= this->localPerm_[sorted[i]])
            	this->localPerm_[j] += 1;
        this->localPerm_.insert(this->localPerm_.begin() + sorted[i], sorted[i]);
    }
    this->tensor_.IntroduceUnitModes(FilterVector(invPerm, unitModes));
    this->gridView_.IntroduceUnitModes(unitModes);
    this->ResizeTo(this->Shape());
}

template<typename T>
void DistTensor<T>::IntroduceUnitModeRedist(const Unsigned& newModePosition){
//    ModeArray modeArr(1);
//    modeArr[0] = newModePosition;
//    IntroduceUnitModesRedist(B, A, modeArr);
}

#define FULL(T) \
    template class DistTensor<T>;

FULL(Int)
#ifndef DISABLE_FLOAT
FULL(float)
#endif
FULL(double)

#ifndef DISABLE_COMPLEX
#ifndef DISABLE_FLOAT
FULL(std::complex<float>)
#endif
FULL(std::complex<double>)
#endif
} //namespace rote
