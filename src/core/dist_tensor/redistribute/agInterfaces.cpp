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
//#include "/tensormental/mc_mr.hpp"
#include <algorithm>

namespace tmen{

template<typename T>
Int
DistTensor<T>::CheckAllGatherRedist(const DistTensor<T>& A, const Mode& allGatherMode){
    if(A.Order() != this->Order()){
        LogicError("CheckAllGatherRedist: Objects being redistributed must be of same order");
    }

    ModeDistribution AAllGatherModeDist = A.ModeDist(allGatherMode);
    if(AAllGatherModeDist.size() != 0)
        LogicError("CheckAllGatherRedist: Allgather only redistributes to * (for now)");

    return true;
}

template<typename T>
Int
DistTensor<T>::CheckAllGatherRedist(const DistTensor<T>& A, const Mode& allGatherMode, const ModeArray& redistModes){
    if(A.Order() != this->Order()){
        LogicError("CheckAllGatherRedist: Objects being redistributed must be of same order");
    }

    ModeDistribution allGatherDistA = A.ModeDist(allGatherMode);

    const ModeDistribution check = ConcatenateVectors(this->ModeDist(allGatherMode), redistModes);
    if(AnyElemwiseNotEqual(check, allGatherDistA)){
        LogicError("CheckAllGatherRedist: [Output distribution ++ redistModes] does not match Input distribution");
    }

    return true;
}

template <typename T>
void
DistTensor<T>::AllGatherRedistFrom(const DistTensor<T>& A, const Mode& allGatherMode, const ModeArray& redistModes ){
//    if(!CheckAllGatherRedist(B, A, allGatherMode, redistModes))
//        LogicError("AllGatherRedist: Invalid redistribution request");

    AllGatherCommRedist(A, allGatherMode, redistModes);
}

#define PROTO(T) \
        template Int DistTensor<T>::CheckAllGatherRedist(const DistTensor<T>& A, const Mode& allGatherMode); \
        template Int DistTensor<T>::CheckAllGatherRedist(const DistTensor<T>& A, const Mode& allGatherMode, const ModeArray& redistModes); \
        template void DistTensor<T>::AllGatherRedistFrom(const DistTensor<T>& A, const Mode& allGatherMode, const ModeArray& redistModes );

PROTO(int)
PROTO(float)
PROTO(double)
PROTO(Complex<double>)
PROTO(Complex<float>)

} //namespace tmen
