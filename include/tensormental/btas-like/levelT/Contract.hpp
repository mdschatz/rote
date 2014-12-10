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

#include "../level1/Permute.hpp"
#include "tensormental/util/vec_util.hpp"
#include "tensormental/util/btas_util.hpp"
#include "tensormental/core/view_decl.hpp"
#include "tensormental/io/Print.hpp"

namespace tmen{

////////////////////////////////////
// LocalContract Workhorse
////////////////////////////////////

//NOTE: Assumes A, B, C are all tightly packed tensors (stride[i] = stride[i-1] * size[i-1] and stride[0] = 1;
template <typename T>
void LocalContract(T alpha, const Tensor<T>& A, const IndexArray& indicesA, const bool permuteA, const Tensor<T>& B, const IndexArray& indicesB, const bool permuteB, T beta, Tensor<T>& C, const IndexArray& indicesC, const bool permuteC){
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
    PROFILE_SECTION("Contract");

    Unsigned i;
    const std::vector<ModeArray> contractPerms(DetermineContractModes(indicesA, indicesB, indicesC));
    const IndexArray contractIndices = DetermineContractIndices(indicesA, indicesB);
    const Unsigned nIndicesContract = contractIndices.size();

    //Determine the permutations for each Tensor
    const Permutation permA = contractPerms[0];
    const Permutation permB = contractPerms[1];
    const Permutation permC = contractPerms[2];
    const Unsigned nIndicesM = permA.size() - nIndicesContract;

    Tensor<T> PA(A.Order());
    Tensor<T> PB(B.Order());
    Tensor<T> PC(C.Order());

    Tensor<T> MPA, MPB, MPC;

    //Create a matrix view of the tensors (permute if needed)
    //Then call Gemm
    if(permuteA){
        PA.ResizeTo(PermuteVector(A.Shape(), permA));
        Permute(PA, A, permA);
        ViewAsMatrix(MPA, PA, nIndicesM);
    }else{
        ViewAsMatrix(MPA, A, nIndicesM);
    }
    if(permuteB){
        PB.ResizeTo(PermuteVector(B.Shape(), permB));
        Permute(PB, B, permB);
        ViewAsMatrix(MPB, PB, nIndicesContract);
    }else{
        ViewAsMatrix(MPB, B, nIndicesContract);
    }
    if(permuteC){
        PC.ResizeTo(PermuteVector(C.Shape(), permC));
        Permute(PC, C, permC);
        ViewAsMatrix(MPC, PC, nIndicesM);

//        PrintArray(MPA.LockedBuffer(), MPA.Shape(), "MPA in");
//        PrintArray(MPB.LockedBuffer(), MPB.Shape(), "MPB in");
//        const T* MPBBuf = MPB.LockedBuffer();
//        printf("mpb buf:");
//        for(i = 0; i < prod(MPB.Shape())*2; i++)
//            printf(" %e", MPBBuf[i]);
//        printf("\n");
//        PrintArray(MPC.LockedBuffer(), MPC.Shape(), "MPC in");
        Gemm(alpha, MPA, MPB, beta, MPC);
//        PrintArray(MPC.LockedBuffer(), MPC.Shape(), "MPC out");

        Tensor<T> IPC;
        const Permutation invPermC = DetermineInversePermutation(permC);
        ObjShape splitColModes(nIndicesM);
        for(i = 0; i < splitColModes.size(); i++)
            splitColModes[i] = i;

        std::vector<ObjShape> newShape(2);
        newShape[0] = FilterVector(PC.Shape(), splitColModes);
        newShape[1] = NegFilterVector(PC.Shape(), splitColModes);

        ViewAsHigherOrder(IPC, MPC, newShape);

        Permute(C, IPC, invPermC);
    }else{
        ViewAsMatrix(MPC, C, nIndicesM);
        PrintArray(MPA.LockedBuffer(), MPA.Shape(), "MPA in");
        PrintArray(MPB.LockedBuffer(), MPB.Shape(), "MPB in");
        PrintArray(MPC.LockedBuffer(), MPC.Shape(), "MPC in");
        Gemm(alpha, MPA, MPB, beta, MPC);
        PrintArray(MPC.LockedBuffer(), MPC.Shape(), "MPC out");
    }
    PROFILE_STOP;
}

////////////////////////////////////
// Local Interfaces
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

    LocalContract(alpha, A, indicesA, true, B, indicesB, true, beta, C, indicesC, true);
}

template <typename T>
void LocalContractAndLocalEliminate(T alpha, const Tensor<T>& A, const IndexArray& indicesA, const Tensor<T>& B, const IndexArray& indicesB, T beta, Tensor<T>& C, const IndexArray& indicesC){
    LocalContractAndLocalEliminate(alpha, A, indicesA, true, B, indicesB, true, beta, C, indicesC, true);
}

//NOTE: Assumes Local data of A, B, C are all tightly packed tensors (stride[i] = stride[i-1] * size[i-1] and stride[0] = 1;
template <typename T>
void LocalContractAndLocalEliminate(T alpha, const Tensor<T>& A, const IndexArray& indicesA, const bool permuteA, const Tensor<T>& B, const IndexArray& indicesB, const bool permuteB, T beta, Tensor<T>& C, const IndexArray& indicesC, const bool permuteC){
    Unsigned i;
    Unsigned order = C.Order();
    IndexArray contractIndices = DetermineContractIndices(indicesA, indicesB);

    ModeArray uModes(contractIndices.size());
    for(i = 0; i < uModes.size(); i++)
        uModes[i] = i + order;

    IndexArray CIndices = ConcatenateVectors(indicesC, contractIndices);

    //LocalContract leaves unit modes in result, so introduce them here
    C.IntroduceUnitModes(uModes);

    LocalContract(alpha, A, indicesA, permuteA, B, indicesB, permuteB, beta, C, CIndices, permuteC);

    //Remove the unit modes
    C.RemoveUnitModes(uModes);
}

} // namespace tmen

#endif // ifndef TMEN_BTAS_CONTRACT_HPP
