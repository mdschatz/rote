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
Int CheckAllGatherRedist(const DistTensor<T>& B, const DistTensor<T>& A, const Mode allGatherMode, const ModeArray& redistModes){
    if(A.Order() != B.Order()){
        LogicError("CheckAllGatherRedist: Objects being redistributed must be of same order");
    }

    ModeDistribution allGatherDistA = A.ModeDist(allGatherMode);
    ModeDistribution allGatherDistB = B.ModeDist(allGatherMode);

    const ModeDistribution check = ConcatenateVectors(allGatherDistB, redistModes);
    if(AnyElemwiseNotEqual(check, allGatherDistA)){
        LogicError("CheckAllGatherRedist: [Output distribution ++ redistModes] does not match Input distribution");
    }

    return true;
}

template<typename T>
Int CheckAllGatherRedist(const DistTensor<T>& B, const DistTensor<T>& A, const Mode allGatherMode){
    if(A.Order() != B.Order()){
        LogicError("CheckAllGatherRedist: Objects being redistributed must be of same order");
    }

    ModeDistribution AAllGatherModeDist = A.ModeDist(allGatherMode);
    if(AAllGatherModeDist.size() != 0)
        LogicError("CheckAllGatherRedist: Allgather only redistributes to * (for now)");

    return true;
}

template <typename T>
void AllGatherRedist(DistTensor<T>& B, const DistTensor<T>& A, const Mode allGatherMode, const ModeArray& redistModes ){
    if(!CheckAllGatherRedist(B, A, allGatherMode, redistModes))
        LogicError("AllGatherRedist: Invalid redistribution request");

    AllGatherCommRedist(B, A, allGatherMode, redistModes);
}

template <typename T>
void AllGatherRedist(DistTensor<T>& B, const DistTensor<T>& A, const Mode allGatherMode){
    if(!CheckAllGatherRedist(B, A, allGatherMode))
        LogicError("AllGatherRedist: Invalid redistribution request");

    ModeDistribution modeDist = A.ModeDist(allGatherMode);

    AllGatherCommRedist(B, A, allGatherMode, modeDist);
}

#define PROTO(T) \
        template Int CheckAllGatherRedist(const DistTensor<T>& B, const DistTensor<T>& A, const Mode allGatherMode, const ModeArray& redistModes); \
        template Int CheckAllGatherRedist(const DistTensor<T>& B, const DistTensor<T>& A, const Mode allGatherMode); \
        template void AllGatherRedist(DistTensor<T>& B, const DistTensor<T>& A, const Mode allGatherMode, const ModeArray& redistModes ); \
        template void AllGatherRedist(DistTensor<T>& B, const DistTensor<T>& A, const Mode allGatherMode);

PROTO(int)
PROTO(float)
PROTO(double)

} //namespace tmen
