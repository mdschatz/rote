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

template<typename T>
Int DistTensor<T>::CheckLocalRedist(const DistTensor<T>& A, const Mode localMode, const ModeArray& gridRedistModes){
    if(A.Order() != this->Order())
        LogicError("CheckLocalRedist: Objects being redistributed must be of same order");

    Unsigned i, j;
    TensorDistribution distA = A.TensorDist();
    ModeDistribution localModeDistA = A.ModeDist(localMode);
    ModeDistribution localModeDistB = this->ModeDist(localMode);

    if(localModeDistB.size() != localModeDistA.size() + gridRedistModes.size())
        LogicError("CheckLocalReist: Input object cannot be redistributed to output object");

    ModeArray check(localModeDistB);
    for(i = 0; i < localModeDistA.size(); i++)
        check[i] = localModeDistA[i];
    for(i = 0; i < gridRedistModes.size(); i++)
        check[localModeDistA.size() + i] = gridRedistModes[i];

    for(i = 0; i < check.size(); i++){
        if(check[i] != localModeDistB[i])
            LogicError("CheckLocalRedist: Output distribution cannot be formed from supplied parameters");
    }

    ModeArray boundModes;
    for(i = 0; i < distA.size(); i++){
        for(j = 0; j < distA[i].size(); j++){
            boundModes.push_back(distA[i][j]);
        }
    }

    for(i = 0; i < gridRedistModes.size(); i++)
        if(std::find(boundModes.begin(), boundModes.end(), gridRedistModes[i]) != boundModes.end())
            LogicError("CheckLocalRedist: Attempting to redistribute with already bound mode of the grid");

    return 1;
}

template<typename T>
void DistTensor<T>::LocalRedist(const DistTensor<T>& A, const Mode localMode, const ModeArray& gridRedistModes){
    if(!this->CheckLocalRedist(A, localMode, gridRedistModes))
        LogicError("LocalRedist: Invalid redistribution request");

    LocalCommRedist(A, localMode, gridRedistModes);
}

#define PROTO(T) \
        template Int  DistTensor<T>::CheckLocalRedist(const DistTensor<T>& A, const Mode localMode, const ModeArray& gridRedistModes); \
        template void DistTensor<T>::LocalRedist(const DistTensor<T>& A, const Mode localMode, const ModeArray& gridRedistModes);

PROTO(int)
PROTO(float)
PROTO(double)
PROTO(Complex<float>)
PROTO(Complex<double>)

} //namespace tmen
