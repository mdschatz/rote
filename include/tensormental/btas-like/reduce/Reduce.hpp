/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef TMEN_BTAS_REDUCE_HPP
#define TMEN_BTAS_REDUCE_HPP

#include "tensormental/util/vec_util.hpp"
#include "tensormental/util/btas_util.hpp"
#include "tensormental/core/view_decl.hpp"

namespace tmen{

////////////////////////////////////
// Local routines
////////////////////////////////////

template <typename T>
void LocalReduce(Tensor<T>& B, const Tensor<T>& A, const ModeArray& reduceModes){
#ifndef RELEASE
    CallStackEntry("LocalReduce");
    if(reduceModes.size() > A.Order())
        LogicError("LocalReduce: modes must be of length <= order");

    for(Unsigned i = 0; i < reduceModes.size(); i++)
        if(reduceModes[i] >= A.Order())
            LogicError("LocalReduce: Supplied mode is out of range");
#endif
    T* BBuf = B.Buffer();
    MemZero(&(BBuf[0]), prod(B.Shape()));
    Unsigned i, j;
    const Unsigned order = A.Order();

    ModeArray nonReduceModes;
    for(i = 0; i < order; i++){
        if(std::find(reduceModes.begin(), reduceModes.end(), i) == reduceModes.end())
            nonReduceModes.push_back(i);
    }

    ModeArray reduceOrder = ConcatenateVectors(reduceModes, nonReduceModes);

    ModeArray origOrder(A.Order());
    for(i = 0; i < order; i++)
        origOrder[i] = i;

     //Determine the permutations for each Tensor
     const std::vector<Unsigned> perm(reduceOrder);
     const std::vector<Unsigned> invPerm(DeterminePermutation(reduceOrder, origOrder));

     Tensor<T> PA(FilterVector(A.Shape(), perm));
     Tensor<T> PB(FilterVector(B.Shape(), perm));
     Tensor<T> MPA, MPB;

     //Permute A, B
//     printf("\n\nPermuting A: [%d", perm[0]);
//     for(i = 1; i < perm.size(); i++)
//         printf(" %d", perm[i]);
//     printf("]\n");
     Permute(PA, A, perm);

//     printf("\n\nPermuting B: [%d", perm[0]);
//     for(i = 1; i < perm.size(); i++)
//         printf(" %d", perm[i]);
//     printf("]\n");
     Permute(PB, B, perm);

     //View as matrices
     ModeArray mergeModes0(reduceModes.size());
     for(i = 0; i < mergeModes0.size(); i++)
         mergeModes0[i] = i;

     ModeArray mergeModes1(nonReduceModes.size());
     for(i = 0; i < mergeModes1.size(); i++)
         mergeModes1[i] = mergeModes0.size() + i;

     std::vector<ModeArray> MPAOldModes(2);
     MPAOldModes[0] = mergeModes0;
     MPAOldModes[1] = mergeModes1;

     std::vector<ModeArray> MPBOldModes(2);
     MPBOldModes[0] = mergeModes0;
     MPBOldModes[1] = mergeModes1;


     ViewAsMatrix(MPA, PA, MPAOldModes );

//     Print(PB, "PB");
     ViewAsMatrix(MPB, PB, MPBOldModes );

//     Print(MPA, "MPA");
//     Print(MPB, "MPB");

     const T* MPAData = MPA.LockedBuffer();
     T* MPBData = MPB.Buffer();
     if(A.Order() == 0)
         MPBData[0] = MPAData[0];
     else{
         for(i = 0; i < MPA.Dimension(1); i++){
             for(j = 0; j < MPA.Dimension(0); j++){
    //             printf("MPBDataptr: %d, MPADataptr: %d\n", i, j + i*MPA.Dimension(0));
                 MPBData[i] += MPAData[j + i*MPA.Dimension(0)];
             }
         }
     }

//     Print(MPB, "MPB after reduce");
     //View as tensor
     ModeArray MPBModes(2);
     MPBModes[0] = 0;
     MPBModes[1] = 1;

     std::vector<ObjShape> PBShape(2);
     PBShape[0] = FilterVector(PB.Shape(), reduceModes);
     PBShape[1] = FilterVector(PB.Shape(), nonReduceModes);

     //ViewAsHigherOrder(PB, MPB, MPBModes, PBShape);
//     Print(PB, "PB after higher order view");

     //Permute back the data
//     printf("\n\nPermuting PB: [%d", invPerm[0]);
//     for(i = 1; i < invPerm.size(); i++)
//         printf(" %d", invPerm[i]);
//     printf("]\n");
     Permute(B, PB, invPerm);
//     Print(B, "B after final permute");
}

template <typename T>
void LocalReduce(Tensor<T>& B, const Tensor<T>& A, const Mode& reduceMode){
    ModeArray modeArr(1);
    modeArr[0] = reduceMode;
    LocalReduce(B, A, modeArr);
}

////////////////////////////////////
// Global routines
////////////////////////////////////

template <typename T>
void LocalReduce(DistTensor<T>& B, const DistTensor<T>& A, const ModeArray& reduceModes){
    if(B.Participating())
        LocalReduce(B.Tensor(), A.LockedTensor(), reduceModes);
}

template <typename T>
void LocalReduce(DistTensor<T>& B, const DistTensor<T>& A, const Mode& reduceMode){
    if(B.Participating()){
        ModeArray modeArr(1);
        modeArr[0] = reduceMode;
        LocalReduce(B, A, modeArr);
    }
}

} // namespace tmen

#endif // ifndef TMEN_BTAS_CONTRACT_HPP
