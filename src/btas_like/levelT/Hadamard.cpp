/*
   Copyright (c) 2009-2013, Jack Poulson
                      2013, Jeff Hammond
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#include "rote.hpp"

namespace rote {

// Assumes Tensor indices are arranged as follows:
template <typename T>
void LocalHadamard(const Tensor<T>& A, const IndexArray& indicesA, const Tensor<T>& B, const IndexArray& indicesB, Tensor<T>& C, const IndexArray& indicesC) {
#ifndef RELEASE
    CallStackEntry("LocalHadamard");

    if(indicesA.size() != A.Order() || indicesB.size() != B.Order() || indicesC.size() != C.Order())
        LogicError("LocalHadamard: number of indices assigned to each tensor must be of same order");
#endif
    PROFILE_SECTION("Hadamard");
    // TODO: Support Stationary variants.
    // Reference indices object needs to be updated
    // (As implemented, the indices will force a permutation for stat-A)

    // Check if permutation needed:
    // indicesA = indicesAC U indicesABC
    // indicesB = indicesBC U indicesABC
    // indicesC = indicesAC U indicesBC U indicesABC
    const IndexArray indicesAC = DiffVector(IsectVector(indicesC, indicesA), indicesB);
    const IndexArray indicesBC = DiffVector(IsectVector(indicesC, indicesB), indicesA);
    const IndexArray indicesABC = IsectVector(IsectVector(indicesC, indicesB), indicesA);
    const Permutation permA = DeterminePermutation(indicesA, ConcatenateVectors(indicesAC, indicesABC));
    const Permutation permB = DeterminePermutation(indicesB, ConcatenateVectors(indicesABC, indicesBC));
    const Permutation permC = DeterminePermutation(indicesC, ConcatenateVectors(indicesAC, ConcatenateVectors(indicesBC, indicesABC)));

    HadamardScalData data;
    data.loopShapeAC = FilterVector(C.Shape(), IndicesOf(indicesC, indicesAC));
    data.stridesACA = FilterVector(A.Strides(), IndicesOf(indicesA, indicesAC));
    data.stridesACC = FilterVector(C.Strides(), IndicesOf(indicesC, indicesAC));

    data.loopShapeBC = FilterVector(C.Shape(), IndicesOf(indicesC, indicesBC));
    data.stridesBCB = FilterVector(B.Strides(), IndicesOf(indicesB, indicesBC));
    data.stridesBCC = FilterVector(C.Strides(), IndicesOf(indicesC, indicesBC));

    data.loopShapeABC = FilterVector(C.Shape(), IndicesOf(indicesC, indicesABC));
    data.stridesABCA = FilterVector(A.Strides(), IndicesOf(indicesA, indicesABC));
    data.stridesABCB = FilterVector(B.Strides(), IndicesOf(indicesB, indicesABC));
    data.stridesABCC = FilterVector(C.Strides(), IndicesOf(indicesC, indicesABC));

    std::cout << "Compute\n";
    PrintData(A, "A");
    PrintData(B, "B");
    PrintData(C, "C");
    HadamardScal(A, B, C, data);
    PrintData(C, "Cafter");

    PROFILE_STOP;
}

//Non-template functions
//bool AnyFalseElem(const std::vector<bool>& vec);
#define PROTO(T) \
	template void LocalHadamard(const Tensor<T>& A, const IndexArray& indicesA, const Tensor<T>& B, const IndexArray& indicesB, Tensor<T>& C, const IndexArray& indicesC);

//PROTO(Unsigned)
//PROTO(Int)
PROTO(float)
PROTO(double)
//PROTO(char)

} // namespace rote
