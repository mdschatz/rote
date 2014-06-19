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
Int CheckRemoveUnitModesRedist(const DistTensor<T>& B, const DistTensor<T>& A, const ModeArray& unitModes){
    const Unsigned orderA = A.Order();
    const Unsigned orderB = B.Order();
    const Unsigned nModesRemove = unitModes.size();

    if(orderB != orderA - nModesRemove)
        LogicError("CheckRemoveUnitIndicesRedist: Object being redistributed must be of correct order");

    return 1;
}

template <typename T>
Int CheckIntroduceUnitModesRedist(const DistTensor<T>& B, const DistTensor<T>& A, const std::vector<Unsigned>& newModePositions){
    const Unsigned orderA = A.Order();
    const Unsigned orderB = B.Order();
    const Unsigned nModesIntroduce = newModePositions.size();

    if(orderB != orderA + nModesIntroduce)
        LogicError("CheckIntroduceUnitIndicesRedist: Object being redistributed must be of correct order");
    return 1;
}

//NOTE: Assuming everything is correct, this is just a straight memcopy
template<typename T>
void RemoveUnitModesRedist(DistTensor<T>& B, const DistTensor<T>& A, const ModeArray& unitModes){
    if(!CheckRemoveUnitModesRedist(B, A, unitModes))
        LogicError("RemoveUnitModesRedist: Invalid redistribution request");

    const Unsigned order = A.Order();
    const Location start(order, 0);
    T* dst = B.Buffer(start);
    const T* src = A.LockedBuffer(start);
    MemCopy(&(dst[0]), &(src[0]), prod(A.LocalShape()));
}

//NOTE: Assuming everything is correct, this is just a straight memcopy
template<typename T>
void IntroduceUnitModesRedist(DistTensor<T>& B, const DistTensor<T>& A, const std::vector<Unsigned>& newModePositions){
    if(!CheckIntroduceUnitModesRedist(B, A, newModePositions))
        LogicError("IntroduceUnitModesRedist: Invalid redistribution request");

    const Unsigned order = A.Order();
    const Location start(order, 0);
    T* dst = B.Buffer(start);
    const T* src = A.LockedBuffer(start);
    MemCopy(&(dst[0]), &(src[0]), prod(A.LocalShape()));
}

#define PROTO(T) \
        template Int CheckRemoveUnitModesRedist(const DistTensor<T>& B, const DistTensor<T>& A, const ModeArray& unitIndices); \
        template Int CheckIntroduceUnitModesRedist(const DistTensor<T>& B, const DistTensor<T>& A, const std::vector<Unsigned>& newModePositions); \
        template void RemoveUnitModesRedist(DistTensor<T>& B, const DistTensor<T>& A, const ModeArray& unitModes); \
        template void IntroduceUnitModesRedist(DistTensor<T>& B, const DistTensor<T>& A, const std::vector<Unsigned>& newModePositions);

PROTO(int)
PROTO(float)
PROTO(double)
} //namespace tmen
