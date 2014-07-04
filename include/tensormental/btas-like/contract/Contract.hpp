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
void LocalContract(T alpha, const Tensor<T>& A, const Tensor<T>& B, T beta, Tensor<T>& C, const std::vector<IndexArray>& indices){
#ifndef RELEASE
    CallStackEntry("LocalContract");
    if(indices.size() != 3)
        LogicError("LocalContract: indices vector must be of length 3 (A indices, B indices, C indices)");

    IndexArray indA = indices[0];
    IndexArray indB = indices[1];
    IndexArray indC = indices[2];

    if(indA.size() != A.Order() || indB.size() != B.Order() || indC.size() != C.Order())
        LogicError("LocalContract: number of indices assigned to each tensor must be of same order");

    //Check conformal modes
//    for(Unsigned i = 0; i < indA.size(); i++){
//        Index index = indA[i];
//        //Check A,B
//        for(Unsigned j = 0; j < indB.size(); j++){
//            if(indB[j] == index){
//                if(A.Dimension(i) != B.Dimension(j))
//                    LogicError("LocalContract: Modes assigned same indices must be conformal");
//            }
//        }
//        //Check A,C
//        for(Unsigned j = 0; j < indC.size(); j++){
//            if(indC[j] == index){
//                if(A.Dimension(i) != C.Dimension(j))
//                    LogicError("LocalContract: Modes assigned same indices must be conformal");
//            }
//        }
//    }
//    for(Unsigned i = 0; i < indB.size(); i++){
//        Index index = indB[i];
//        //Check B,C
//        for(Unsigned j = 0; j < indC.size(); j++){
//            if(indC[j] == index){
//                if(B.Dimension(i) != C.Dimension(j))
//                    LogicError("LocalContract: Modes assigned same indices must be conformal");
//            }
//        }
//    }
#endif
    Unsigned i, j;
    const std::vector<ModeArray> contractPerms(DetermineContractModes(A, B, C, indices));

    Unsigned nIndicesContract = 0;
    for(i = 0; i < indices[0].size(); i++)
        if(std::find(indices[1].begin(), indices[1].end(), indices[0][i]) != indices[1].end())
            nIndicesContract++;
    //TODO: Implement invPerm routine to get the invPermC variable (only reason I form CModes)
    ModeArray CModes(C.Order());
    for(i = 0; i < C.Order(); i++)
        CModes[i] = i;

    //Determine the permutations for each Tensor
    const std::vector<Unsigned> permA = contractPerms[0];
    const std::vector<Unsigned> permB = contractPerms[1];
    const std::vector<Unsigned> permC = contractPerms[2];
    const std::vector<Unsigned> invPermC = DeterminePermutation(permC, CModes);

    Tensor<T> PA(FilterVector(A.Shape(), permA));
    Tensor<T> PB(FilterVector(B.Shape(), permB));
    Tensor<T> PC(FilterVector(C.Shape(), permC));
    Tensor<T> MPA, MPB, MPC;

    //Permute A, B, C
//    printf("\n\nPermuting A: [%d", permA[0]);
//    for(i = 1; i < permA.size(); i++)
//        printf(" %d", permA[i]);
//    printf("]\n");

//    printf("\n\nPermuting B: [%d", permB[0]);
//    for(i = 1; i < permB.size(); i++)
//        printf(" %d", permB[i]);
//    printf("]\n");

//    printf("\n\nPermuting C: [%d", permC[0]);
//    for(i = 1; i < permC.size(); i++)
//        printf(" %d", permC[i]);
//    printf("]\n");

    Permute(PA, A, permA);
    Permute(PB, B, permB);
    Permute(PC, C, permC);

//    Print(PA, "PA");
//    Print(PB, "PB");
//    Print(PC, "PC");

    const Unsigned maxOrder = Max(Max(PA.Order(), PB.Order()), PC.Order());
    std::vector<Mode> tensorModes(maxOrder);

    for(i = 0; i < maxOrder; i++)
        tensorModes[i] = i;

    const Unsigned nIndicesM = permA.size() - nIndicesContract;
    const Unsigned nIndicesN = permB.size() - nIndicesContract;

    //View as matrices
    std::vector<ModeArray> MPAOldModes(2);
    MPAOldModes[0].insert(MPAOldModes[0].end(), tensorModes.begin(), tensorModes.begin() + nIndicesM);
    MPAOldModes[1].insert(MPAOldModes[1].end(), tensorModes.begin() + nIndicesM, tensorModes.begin() + A.Order());

    std::vector<ModeArray> MPBOldModes(2);
    MPBOldModes[0].insert(MPBOldModes[0].end(), tensorModes.begin(), tensorModes.begin() + nIndicesContract);
    MPBOldModes[1].insert(MPBOldModes[1].end(), tensorModes.begin() + nIndicesContract, tensorModes.begin() + B.Order());

    
    std::vector<ModeArray> MPCOldModes(2);
    MPCOldModes[0].insert(MPCOldModes[0].end(), tensorModes.begin(), tensorModes.begin() + nIndicesM);
    MPCOldModes[1].insert(MPCOldModes[1].end(), tensorModes.begin() + nIndicesM, tensorModes.begin() + C.Order());

    ViewAsMatrix(MPA, PA, MPAOldModes );
    ViewAsMatrix(MPB, PB, MPBOldModes );
    ViewAsMatrix(MPC, PC, MPCOldModes );

//    Print(MPA, "MPA");
//    Print(MPB, "MPB");
//    Print(MPC, "MPC");
    Gemm(alpha, MPA, MPB, beta, MPC);
//    Print(MPC, "PostMult");
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
//    printf("\n\nPermuting PC: [%d", invPermC[0]);
//    for(i = 1; i < invPermC.size(); i++)
//        printf(" %d", invPermC[i]);
//    printf("]\n");
    Permute(C, PC, invPermC);
//    Print(C, "result C");
}

template <typename T>
void LocalContract(T alpha, const Tensor<T>& A, const IndexArray& indicesA, const Tensor<T>& B, const IndexArray& indicesB, T beta, Tensor<T>& C, const IndexArray& indicesC){

    std::vector<IndexArray> indices;
    indices.push_back(indicesA);
    indices.push_back(indicesB);
    indices.push_back(indicesC);
    LocalContract(alpha, A, B, beta, C, indices);
}

//NOTE: Get rid of memcopy
template <typename T>
void LocalContractAndLocalEliminate(T alpha, const Tensor<T>& A, const IndexArray& indicesA, const Tensor<T>& B, const IndexArray& indicesB, T beta, Tensor<T>& C, const IndexArray& indicesC){
    Unsigned i;
    IndexArray contractIndices;
    for(i = 0; i < indicesA.size(); i++)
        if(std::find(indicesB.begin(), indicesB.end(), indicesA[i]) != indicesB.end())
            contractIndices.push_back(indicesA[i]);
    std::sort(contractIndices.begin(), contractIndices.end());

    ObjShape tmpShape = C.Shape();
    IndexArray tmpIndices = ConcatenateVectors(indicesC, contractIndices);
    for(i = 0; i < contractIndices.size(); i++){
        tmpShape.push_back(1);
    }

    Tensor<T> tmp(tmpShape);
    T* CBuf = C.Buffer();
    T* tmpBuf = tmp.Buffer();
    MemCopy(&(tmpBuf[0]), &(CBuf[0]), prod(tmp.Shape()));
    LocalContract(alpha, A, indicesA, B, indicesB, beta, tmp, tmpIndices);
    if(C.Order() == 0)
        MemCopy(&(CBuf[0]), &(tmpBuf[0]),Max(1, prod(C.Shape())));
    else
        MemCopy(&(CBuf[0]), &(tmpBuf[0]), prod(C.Shape()));
}

} // namespace tmen

#endif // ifndef TMEN_BTAS_CONTRACT_HPP
