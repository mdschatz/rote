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
    for(i = sorted.size() - 1; i < sorted.size(); i--){
        shape_.erase(shape_.begin() + sorted[i]);
        dist_.erase(dist_.begin() + sorted[i]);
        constrainedModeAlignments_.erase(constrainedModeAlignments_.begin() + sorted[i]);
        modeAlignments_.erase(modeAlignments_.begin() + sorted[i]);
        modeShifts_.erase(modeShifts_.begin() + sorted[i]);
    }
    std::cout << "Removing unit mode of distTensor" << std::endl;
    tensor_.RemoveUnitModes(unitModes);
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
