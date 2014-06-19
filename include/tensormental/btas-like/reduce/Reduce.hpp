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
void LocalReduce(const Tensor<T>& A, const Tensor<T>& B, const ModeArray& reduceModes){
#ifndef RELEASE
    CallStackEntry("LocalReduce");
    if(reduceModes.size() > A.Order())
        LogicError("LocalReduce: modes must be of length <= order");

    for(Unsigned i = 0; i < reduceModes.size(); i++)
        if(reduceModes[i] >= A.Order())
            LogicError("LocalReduce: Supplied mode is out of range");
#endif
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
     const std::vector<Unsigned> permA(reduceOrder);
     const std::vector<Unsigned> permB(reduceOrder);
     const std::vector<Unsigned> invPermB(DeterminePermutation(reduceOrder, origOrder));

     Tensor<T> PA(FilterVector(A.Shape(), permA));
     Tensor<T> PB(FilterVector(B.Shape(), permB));
     Tensor<T> MPA, MPB;

     //Permute A, B, C
     printf("\n\nPermuting A: [%d", permA[0]);
     for(i = 1; i < permA.size(); i++)
         printf(" %d", permA[i]);
     printf("]\n");
     Permute(PA, A, permA);

     printf("\n\nPermuting B: [%d", permB[0]);
     for(i = 1; i < permB.size(); i++)
         printf(" %d", permB[i]);
     printf("]\n");
     Permute(PB, B, permB);

     //View as matrices
     std::vector<ModeArray> MPAOldModes(2);
     MPAOldModes[0] = reduceModes;
     MPAOldModes[1] = contractModes[1];

     std::vector<ModeArray> MPBOldModes(2);
     MPBOldModes[0] = contractModes[1];
     MPBOldModes[1] = contractModes[2];


     ViewAsLowerOrder(MPA, PA, MPAOldModes );

     Print(PB, "PB");
     ViewAsLowerOrder(MPB, PB, MPBOldModes );
     ViewAsLowerOrder(MPC, PC, MPCOldModes );

     Print(MPA, "MPA");
     Print(MPB, "MPB");
     Print(MPC, "MPC");
     Gemm(alpha, MPA, MPB, beta, MPC);
     Print(MPC, "PostMult");
     //View as tensor

     std::vector<ObjShape> newShape(MPCOldModes.size());
     for(i = 0; i < newShape.size(); i++){
         ModeArray oldModes(MPCOldModes[i].size());
         for(j = 0; j < MPCOldModes[i].size(); j++){
             oldModes[j] = MPCOldModes[i][j];
         }
         newShape[i] = FilterVector(PC.Shape(), oldModes);
     }
     ModeArray MPCModes(2);
     MPCModes[0] = 0;
     MPCModes[1] = 1;
     ViewAsHigherOrder(PC, MPC, MPCModes, newShape);

     //Permute back the data
     printf("\n\nPermuting PC: [%d", invPermC[0]);
     for(i = 1; i < invPermC.size(); i++)
         printf(" %d", invPermC[i]);
     printf("]\n");
     Permute(C, PC, invPermC);
}

} // namespace tmen

#endif // ifndef TMEN_BTAS_CONTRACT_HPP
