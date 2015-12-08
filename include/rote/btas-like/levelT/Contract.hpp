/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef ROTE_BTAS_CONTRACT_HPP
#define ROTE_BTAS_CONTRACT_HPP

#include "../level1/Permute.hpp"
#include "rote/util/vec_util.hpp"
#include "rote/util/btas_util.hpp"
#include "rote/core/view_decl.hpp"
#include "rote/io/Print.hpp"
#include "rote/util/btas_util.hpp"

namespace rote{

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

    //Check that the strides of each tensor are amenable to a direct Gemm call, and if not, permute
    //NOTE: I think the check is currently overly conservative.
    bool reallyPermuteA = permuteA;
    bool reallyPermuteB = permuteB;
    bool reallyPermuteC = permuteC;
    if(!reallyPermuteA){
        const ObjShape pShapeA = PermuteVector(A.Shape(), permA);
        const std::vector<Unsigned> pStridesA = PermuteVector(A.Strides(), permA);
        if(pStridesA[0] != 1)
            reallyPermuteA = true;
        for(i = 1; i < A.Order(); i++)
            reallyPermuteA |= (pStridesA[i] != (pShapeA[i-1] * pStridesA[i-1]));
    }

    if(!reallyPermuteB){
        const ObjShape pShapeB = PermuteVector(B.Shape(), permB);
        const std::vector<Unsigned> pStridesB = PermuteVector(B.Strides(), permB);
        if(pStridesB[0] != 1)
            reallyPermuteB = true;
        for(i = 1; i < B.Order(); i++)
            reallyPermuteB |= (pStridesB[i] != (pShapeB[i-1] * pStridesB[i-1]));
    }

    if(!reallyPermuteC){
        const ObjShape pShapeC = PermuteVector(C.Shape(), permC);
        const std::vector<Unsigned> pStridesC = PermuteVector(C.Strides(), permC);
        if(pStridesC[0] != 1)
            reallyPermuteC = true;
        for(i = 1; i < C.Order(); i++)
            reallyPermuteC |= (pStridesC[i] != (pShapeC[i-1] * pStridesC[i-1]));
    }

    printf("reallyPermuteA: %s\n", reallyPermuteA ? "True" : "False");
    printf("reallyPermuteB: %s\n", reallyPermuteB ? "True" : "False");
    printf("reallyPermuteC: %s\n", reallyPermuteC ? "True" : "False");
    //Create a matrix view of the tensors (permute if needed)
    //Then call Gemm
    if(reallyPermuteA){
//        PA.ResizeTo(PermuteVector(A.Shape(), permA));
        Permute(A, PA, permA);
        ViewAsMatrix(MPA, PA, nIndicesM);
    }else{
        ViewAsMatrix(MPA, A, nIndicesM);
    }
    if(reallyPermuteB){
//        PB.ResizeTo(PermuteVector(B.Shape(), permB));
        Permute(B, PB, permB);
        ViewAsMatrix(MPB, PB, nIndicesContract);
    }else{
        ViewAsMatrix(MPB, B, nIndicesContract);
    }
    if(reallyPermuteC){
//        PC.ResizeTo(PermuteVector(C.Shape(), permC));
        Permute(C, PC, permC);
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

        Permute(IPC, C, invPermC);
    }else{
        ViewAsMatrix(MPC, C, nIndicesM);
        PrintData(A, "A in");
        PrintData(B, "B in");
        PrintData(C, "C in");
        PrintData(MPA, "MPA in");
        PrintData(MPB, "MPB in");
        PrintData(MPC, "MPC in");
        Gemm(alpha, MPA, MPB, beta, MPC);
        PrintData(MPC, "MPC out");
    }
    PROFILE_STOP;
}

template <typename T>
void LocalContractNoReallyPerm(T alpha, const Tensor<T>& A, const IndexArray& indicesA, const bool permuteA, const Tensor<T>& B, const IndexArray& indicesB, const bool permuteB, T beta, Tensor<T>& C, const IndexArray& indicesC, const bool permuteC){
#ifndef RELEASE
    CallStackEntry("LocalContract");

    if(indicesA.size() != A.Order() || indicesB.size() != B.Order() || indicesC.size() != C.Order())
        LogicError("LocalContract: number of indices assigned to each tensor must be of same order");
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
//        PA.ResizeTo(PermuteVector(A.Shape(), permA));
        Permute(A, PA, permA);
        ViewAsMatrix(MPA, PA, nIndicesM);
    }else{
        ViewAsMatrix(MPA, A, nIndicesM);
    }
    if(permuteB){
//        PB.ResizeTo(PermuteVector(B.Shape(), permB));
        Permute(B, PB, permB);
        ViewAsMatrix(MPB, PB, nIndicesContract);
    }else{
        ViewAsMatrix(MPB, B, nIndicesContract);
    }
    if(permuteC){
//        PC.ResizeTo(PermuteVector(C.Shape(), permC));
        Permute(C, PC, permC);
        ViewAsMatrix(MPC, PC, nIndicesM);

        PrintData(A, "A in");
        PrintData(B, "B in");
        PrintData(C, "C in");
        PrintData(MPA, "MPA in");
        PrintData(MPB, "MPB in");
        PrintData(MPC, "MPC in");
        Print(MPA, "MPA data");
        Print(MPB, "MPB data");
        Print(MPC, "MPC data");
        Gemm(alpha, MPA, MPB, beta, MPC);
        PrintData(MPC, "MPC out");
        Print(MPC, "MPC out data");


        Tensor<T> IPC;
        const Permutation invPermC = DetermineInversePermutation(permC);
        ObjShape splitColModes(nIndicesM);
        for(i = 0; i < splitColModes.size(); i++)
            splitColModes[i] = i;

        std::vector<ObjShape> newShape(2);
        newShape[0] = FilterVector(PC.Shape(), splitColModes);
        newShape[1] = NegFilterVector(PC.Shape(), splitColModes);

        ViewAsHigherOrder(IPC, MPC, newShape);

        Permute(IPC, C, invPermC);
    }else{
        ViewAsMatrix(MPC, C, nIndicesM);
        PrintData(A, "A in");
        PrintData(B, "B in");
        PrintData(C, "C in");
        Print(A, "A in");
        Print(B, "B in");
        Print(C, "C in");
        PrintData(MPA, "MPA in");
        PrintData(MPB, "MPB in");
        PrintData(MPC, "MPC in");
        Print(MPA, "MPA in");
        Print(MPB, "MPB in");
        Print(MPC, "MPC in");
        Gemm(alpha, MPA, MPB, beta, MPC);
        PrintData(MPC, "MPC out");
        Print(MPC, "MPC out");
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

    LocalContractNoReallyPerm(alpha, A, indicesA, permuteA, B, indicesB, permuteB, beta, C, CIndices, permuteC);

    //Remove the unit modes
    C.RemoveUnitModes(uModes);
}

} // namespace rote

#endif // ifndef ROTE_BTAS_CONTRACT_HPP
