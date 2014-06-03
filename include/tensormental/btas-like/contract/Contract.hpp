/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef TMEN_BTAS_CONTRACT_HPP
#define TMEN_BTAS_CONTRACT_HPP

#include "tensormental/util/vec_util.hpp"
#include "tensormental/util/btas_util.hpp"
#include "tensormental/core/view_decl.hpp"

namespace tmen{

////////////////////////////////////
// Local routines
////////////////////////////////////

template <typename T>
void LocalContract(T alpha, const Tensor<T>& A, const Tensor<T>& B, T beta, Tensor<T>& C){
#ifndef RELEASE
    CallStackEntry("LocalContract");
    IndexArray checkIndicesA = A.Indices();
    IndexArray checkIndicesB = B.Indices();
    IndexArray checkIndicesC = C.Indices();

    for(Unsigned i = 0; i < checkIndicesA.size(); i++){
        Index index = checkIndicesA[i];
        if(std::find(checkIndicesB.begin(), checkIndicesB.end(), index) != checkIndicesB.end()){
            if(A.IndexDimension(index) != B.IndexDimension(index))
                LogicError("Indices must have conforming dimensions");
        }
    }

    for(Unsigned i = 0; i < checkIndicesA.size(); i++){
        Index index = checkIndicesA[i];
        if(std::find(checkIndicesC.begin(), checkIndicesC.end(), index) != checkIndicesC.end()){
            if(A.IndexDimension(index) != C.IndexDimension(index))
                LogicError("Indices must have conforming dimensions");
        }
    }

    for(Unsigned i = 0; i < checkIndicesB.size(); i++){
        Index index = checkIndicesB[i];
        if(std::find(checkIndicesC.begin(), checkIndicesC.end(), index) != checkIndicesC.end()){
            if(B.IndexDimension(index) != C.IndexDimension(index))
                LogicError("Indices must have conforming dimensions");
        }
    }
#endif
    Unsigned i, j;
    const std::vector<IndexArray> contractIndices(DetermineContractIndices(A, B, C));

    const IndexArray indicesA = A.Indices();
    const IndexArray indicesB = B.Indices();
    const IndexArray indicesC = C.Indices();

    //Determine the permutations for each Tensor
    const IndexArray permA(DeterminePermutation(indicesA, ConcatenateVectors(contractIndices[0], contractIndices[1])));
    const IndexArray permB(DeterminePermutation(indicesB, ConcatenateVectors(contractIndices[1], contractIndices[2])));
    const IndexArray permC(DeterminePermutation(indicesC, ConcatenateVectors(contractIndices[0], contractIndices[2])));
    const IndexArray invPermC(DeterminePermutation(ConcatenateVectors(contractIndices[0], contractIndices[2]), indicesC));

    Tensor<T> PA(FilterVector(indicesA, permA), FilterVector(A.Shape(), permA));
    Tensor<T> PB(FilterVector(indicesB, permB), FilterVector(B.Shape(), permB));
    Tensor<T> PC(FilterVector(indicesC, permC), FilterVector(C.Shape(), permC));
    Tensor<T> MPA, MPB, MPC;

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

    printf("\n\nPermuting C: [%d", permC[0]);
    for(i = 1; i < permC.size(); i++)
        printf(" %d", permC[i]);
    printf("]\n");
    Permute(PC, C, permC);

    //View as matrices
    std::vector<IndexArray> MPAOldIndices(2);
    MPAOldIndices[0] = contractIndices[0];
    MPAOldIndices[1] = contractIndices[1];

    std::vector<IndexArray> MPBOldIndices(2);
    MPBOldIndices[0] = contractIndices[1];
    MPBOldIndices[1] = contractIndices[2];
    
    std::vector<IndexArray> MPCOldIndices(2);
    MPCOldIndices[0] = contractIndices[0];
    MPCOldIndices[1] = contractIndices[2];

    IndexArray MPANewIndices(2);
    MPANewIndices[0] = 0;
    MPANewIndices[1] = 1;

    IndexArray MPBNewIndices(2);
    MPBNewIndices[0] = 1;
    MPBNewIndices[1] = 2;
    
    IndexArray MPCNewIndices(2);
    MPCNewIndices[0] = 0;
    MPCNewIndices[1] = 2;

    ViewAsLowerOrder(MPA, PA, MPANewIndices, MPAOldIndices );

    Print(PB, "PB");
    ViewAsLowerOrder(MPB, PB, MPBNewIndices, MPBOldIndices );
    ViewAsLowerOrder(MPC, PC, MPCNewIndices, MPCOldIndices );

    Print(MPA, "MPA");
    Print(MPB, "MPB");
    Print(MPC, "MPC");
    Gemm(alpha, MPA, MPB, beta, MPC);
    Print(MPC, "PostMult");
    //View as tensor

    std::vector<ObjShape> newShape(MPCOldIndices.size());
    for(i = 0; i < newShape.size(); i++){
        ModeArray oldModes(MPCOldIndices[i].size());
        for(j = 0; j < MPCOldIndices[i].size(); j++){
            oldModes[j] = PC.ModeOfIndex(MPCOldIndices[i][j]);
        }
        newShape[i] = FilterVector(PC.Shape(), oldModes);
    }
    ViewAsHigherOrder(PC, MPC, MPCOldIndices, MPCNewIndices, newShape);

    //Permute back the data
    printf("\n\nPermuting PC: [%d", invPermC[0]);
    for(i = 1; i < invPermC.size(); i++)
        printf(" %d", invPermC[i]);
    printf("]\n");
    Permute(C, PC, invPermC);
}

} // namespace tmen

#endif // ifndef TMEN_BTAS_CONTRACT_HPP
