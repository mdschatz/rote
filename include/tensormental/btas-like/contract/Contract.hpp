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
void LocalContract(T alpha, const Tensor<T>& A, const IndexArray& indicesA, const Tensor<T>& B, const IndexArray& indicesB, T beta, Tensor<T>& C, const IndexArray& indicesC){
#ifndef RELEASE
    CallStackEntry("LocalContract");

    if(indicesA.size() != A.Order() || indicesB.size() != B.Order() || indicesC.size() != C.Order())
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
    Unsigned i;
    const std::vector<ModeArray> contractPerms(DetermineContractModes(indicesA, indicesB, indicesC));
    const IndexArray contractIndices = DetermineContractIndices(indicesA, indicesB);
    const Unsigned nIndicesContract = contractIndices.size();

    //Determine the permutations for each Tensor
    const Permutation permA = contractPerms[0];
    const Permutation permB = contractPerms[1];
    const Permutation permC = contractPerms[2];
    const Permutation invPermC = DetermineInversePermutation(permC);

    Tensor<T> PA(PermuteVector(A.Shape(), permA));
    Tensor<T> PB(PermuteVector(B.Shape(), permB));
    Tensor<T> PC(PermuteVector(C.Shape(), permC));
    Tensor<T> IPC;
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

//    PrintData(PA, "PA");
//    PrintData(PB, "PB");
//    PrintData(PC, "PC");
//    Print(PA, "PA");
//    Print(PB, "PB");
//    Print(PC, "PC");

    const Unsigned nIndicesM = permA.size() - nIndicesContract;

    //View as matrices
//    printf("MPA Merged %d indices left and %d right\n", nIndicesM, PA.Order() - nIndicesM);
//    printf("MPB Merged %d indices left and %d right\n", nIndicesContract, PB.Order() - nIndicesContract);
//    printf("MPC Merged %d indices left and %d right\n", nIndicesM, PC.Order() - nIndicesM);
    ViewAsMatrix(MPA, PA, nIndicesM );
    ViewAsMatrix(MPB, PB, nIndicesContract );
    ViewAsMatrix(MPC, PC, nIndicesM );

//    PrintData(MPA, "MPA");
//    PrintData(MPB, "MPB");
//    PrintData(MPC, "MPC");
//    Print(MPA, "MPA");
//    Print(MPB, "MPB");
//    Print(MPC, "MPC");
    Gemm(alpha, MPA, MPB, beta, MPC);
//    Print(MPC, "PostMult");
    //View as tensor

    ObjShape splitColModes(nIndicesM);
    for(i = 0; i < splitColModes.size(); i++)
        splitColModes[i] = i;

    std::vector<ObjShape> newShape(2);
    newShape[0] = FilterVector(PC.Shape(), splitColModes);
    newShape[1] = NegFilterVector(PC.Shape(), splitColModes);

    ViewAsHigherOrder(IPC, MPC, newShape);

    //Permute back the data
//    printf("\n\nPermuting PC: [%d", invPermC[0]);
//    for(i = 1; i < invPermC.size(); i++)
//        printf(" %d", invPermC[i]);
//    printf("]\n");
    Permute(C, IPC, invPermC);
//    Print(C, "result C");
}

//NOTE: Get rid of memcopy
template <typename T>
void LocalContractAndLocalEliminate(T alpha, const Tensor<T>& A, const IndexArray& indicesA, const Tensor<T>& B, const IndexArray& indicesB, T beta, Tensor<T>& C, const IndexArray& indicesC){

    Unsigned i;
    Unsigned order = C.Order();
    IndexArray contractIndices = DetermineContractIndices(indicesA, indicesB);

    ModeArray uModes(contractIndices.size());
    for(i = 0; i < uModes.size(); i++)
        uModes[i] = i + order;
    ObjShape tmpShape(C.Shape());
    IndexArray tmpIndices = ConcatenateVectors(indicesC, contractIndices);

    Tensor<T> tmp(tmpShape);
    tmp.CopyBuffer(C);
    tmp.IntroduceUnitModes(uModes);

    LocalContract(alpha, A, indicesA, B, indicesB, beta, tmp, tmpIndices);

    tmp.RemoveUnitModes(uModes);
    C.CopyBuffer(tmp);

//    ObjShape unitModes(contractIndices.size(), 1);
//    ObjShape tmpShape = ConcatenateVectors(C.Shape(), unitModes);
//    IndexArray tmpIndices = ConcatenateVectors(indicesC, contractIndices);
//
//    Tensor<T> tmp(tmpShape);
//    //Cannot simply be an attach as will not have correct ldims_ set
//    tmp.CopyBuffer(C);
//    LocalContract(alpha, A, indicesA, B, indicesB, beta, tmp, tmpIndices);
//    C.CopyBuffer(tmp);
}

////////////////////////////////////
// Local routines without repacking
////////////////////////////////////

//NOTE: Assumes A, B, C are all tightly packed tensors (stride[i] = stride[i-1] * size[i-1] and stride[0] = 1;
template <typename T>
void LocalContractDirect(T alpha, const Tensor<T>& A, const IndexArray& indicesA, const Tensor<T>& B, const IndexArray& indicesB, T beta, Tensor<T>& C, const IndexArray& indicesC){
#ifndef RELEASE
    CallStackEntry("LocalContract");

    if(indicesA.size() != A.Order() || indicesB.size() != B.Order() || indicesC.size() != C.Order())
        LogicError("LocalContract: number of indices assigned to each tensor must be of same order");

#endif
    Unsigned i;
    const std::vector<ModeArray> contractPerms(DetermineContractModes(indicesA, indicesB, indicesC));
    const IndexArray contractIndices = DetermineContractIndices(indicesA, indicesB);
    const Unsigned nIndicesContract = contractIndices.size();

    //Determine the permutations for each Tensor
    const Permutation permA = contractPerms[0];
    const Permutation permB = contractPerms[1];
    const Permutation permC = contractPerms[2];
    const Permutation invPermC = DetermineInversePermutation(permC);

    Tensor<T> MPA, MPB, MPC;

    const Unsigned nIndicesM = permA.size() - nIndicesContract;

    //View as matrices
//    printf("MPA Merged %d indices left and %d right\n", nIndicesM, PA.Order() - nIndicesM);
//    printf("MPB Merged %d indices left and %d right\n", nIndicesContract, PB.Order() - nIndicesContract);
//    printf("MPC Merged %d indices left and %d right\n", nIndicesM, PC.Order() - nIndicesM);
    ViewAsMatrix(MPA, A, nIndicesM );
    ViewAsMatrix(MPB, B, nIndicesContract );
    ViewAsMatrix(MPC, C, nIndicesM );

//    PrintData(MPA, "MPA");
//    PrintData(MPB, "MPB");
//    PrintData(MPC, "MPC");
//    Print(MPA, "MPA");
//    Print(MPB, "MPB");
//    Print(MPC, "MPC");
    Gemm(alpha, MPA, MPB, beta, MPC);
//    Print(MPC, "PostMult");
    //View as tensor

    ObjShape splitColModes(nIndicesM);
    for(i = 0; i < splitColModes.size(); i++)
        splitColModes[i] = i;

    std::vector<ObjShape> newShape(2);
    newShape[0] = FilterVector(C.Shape(), splitColModes);
    newShape[1] = NegFilterVector(C.Shape(), splitColModes);

    ViewAsHigherOrder(C, MPC, newShape);

//    Print(C, "result C");
}

//NOTE: Assumes Local data of A, B, C are all tightly packed tensors (stride[i] = stride[i-1] * size[i-1] and stride[0] = 1;
template <typename T>
void LocalContractAndLocalEliminateDirect(T alpha, const Tensor<T>& A, const IndexArray& indicesA, const Tensor<T>& B, const IndexArray& indicesB, T beta, Tensor<T>& C, const IndexArray& indicesC){

    Unsigned i;
    Unsigned order = C.Order();
    IndexArray contractIndices = DetermineContractIndices(indicesA, indicesB);

    ModeArray uModes(contractIndices.size());
    for(i = 0; i < uModes.size(); i++)
        uModes[i] = i + order;

    IndexArray CIndices = ConcatenateVectors(indicesC, contractIndices);

    C.IntroduceUnitModes(uModes);

    LocalContractDirect(alpha, A, indicesA, B, indicesB, beta, C, CIndices);

    C.RemoveUnitModes(uModes);

}

} // namespace tmen

#endif // ifndef TMEN_BTAS_CONTRACT_HPP
