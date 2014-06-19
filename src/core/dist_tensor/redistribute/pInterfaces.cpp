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

//TODO: Check all unaffected indices are distributed similarly (Only done for CheckPermutationRedist currently)
template <typename T>
Int CheckPermutationRedist(const DistTensor<T>& B, const DistTensor<T>& A, const Mode permuteMode, const ModeArray& redistModes){
    Unsigned i;
    const tmen::GridView gvA = A.GridView();

    const Unsigned AOrder = A.Order();
    const Unsigned BOrder = B.Order();

    //Test order retained
    if(BOrder != AOrder){
        LogicError("CheckPermutationRedist: Permutation retains the same order of objects");
    }

    //Test dimension has been resized correctly
    //NOTE: Uses fancy way of performing Ceil() on integer division
    if(B.Dimension(permuteMode) != A.Dimension(permuteMode))
        LogicError("CheckPartialReduceScatterRedist: Permutation retains the same dimension of indices");

    //Make sure all indices are distributed similarly
    for(i = 0; i < BOrder; i++){
        Mode mode = i;
        if(mode == permuteMode){
            if(!EqualUnderPermutation(B.ModeDist(mode), A.ModeDist(mode)))
                LogicError("CheckPermutationRedist: Distribution of permuted mode does not involve same modes of grid as input");
        }else{
            if(AnyElemwiseNotEqual(B.ModeDist(mode), A.ModeDist(mode)))
                LogicError("CheckPartialReduceScatterRedist: All modes must be distributed similarly");
        }
    }
    return 1;

}

template <typename T>
void PermutationRedist(DistTensor<T>& B, const DistTensor<T>& A, const Mode permuteMode, const ModeArray& redistModes){
    if(!CheckPermutationRedist(B, A, permuteMode, redistModes))
            LogicError("PermutationRedist: Invalid redistribution request");

    PermutationCommRedist(B, A, permuteMode, redistModes);
}

#define PROTO(T) \
        template Int CheckPermutationRedist(const DistTensor<T>& B, const DistTensor<T>& A, const Mode permuteMode, const ModeArray& redistModes); \
        template void PermutationRedist(DistTensor<T>& B, const DistTensor<T>& A, const Mode permuteMode, const ModeArray& redistModes);

PROTO(int)
PROTO(float)
PROTO(double)

} //namespace tmen
