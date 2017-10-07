/*
   Copyright (c) 2009-2013, Jack Poulson
                      2013, Jeff Hammond
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#include "rote.hpp"
#include "rote/core/dist_tensor_forward_decl.hpp"

namespace rote {

// Main interface
template <typename T>
void Hadamard<T>::run(const DistTensor<T> &A, const std::string &indicesA,
                      const DistTensor<T> &B, const std::string &indicesB,
                      DistTensor<T> &C, const std::string &indicesC,
                      const std::vector<Unsigned> &blkSizes) {
  // Convert to index array
  IndexArray indA(indicesA.size());
  for (int i = 0; i < indicesA.size(); i++)
    indA[i] = indicesA[i];

  IndexArray indB(indicesB.size());
  for (int i = 0; i < indicesB.size(); i++)
    indB[i] = indicesB[i];

  IndexArray indC(indicesC.size());
  for (int i = 0; i < indicesC.size(); i++)
    indC[i] = indicesC[i];

  // Determine Stationary variant.
  const Unsigned numElemA = prod(A.Shape());
  const Unsigned numElemB = prod(B.Shape());
  const Unsigned numElemC = prod(C.Shape());

  bool isBiggerAB = numElemA > numElemB;
  bool isBiggerAC = numElemA > numElemC;
  bool isBiggerBC = numElemB > numElemC;

  bool isEqualAB = numElemA == numElemB;

  bool isSmallerAB = numElemA < numElemB;
  bool isSmallerAC = numElemA < numElemC;

  bool isSmallerEqualAC = numElemA <= numElemC;
  bool isSmallerEqualBC = numElemB <= numElemC;

  bool isBiggerEqualAB = numElemA >= numElemB;
  bool isBiggerEqualAC = numElemA >= numElemC;

  if (isBiggerEqualAB && isBiggerAC) {
    Hadamard::run(A, indA, B, indB, C, indC, blkSizes, false);
  } else if ((isSmallerAB && isBiggerEqualAC) ||
             (isSmallerAB && isSmallerAC && isBiggerBC)) {
    Hadamard::run(B, indB, A, indA, C, indC, blkSizes, false);
  } else if ((isBiggerAB && isSmallerEqualAC) ||
             (isEqualAB && isSmallerEqualAC) ||
             (isSmallerAB && isSmallerAC && isSmallerEqualBC)) {
    Hadamard::run(A, indA, B, indB, C, indC, blkSizes, true);
  } else {
    LogicError("Should never occur");
  }
}

// Internal interface
template <typename T>
void Hadamard<T>::run(const DistTensor<T> &A, const IndexArray &indicesA,
                      const DistTensor<T> &B, const IndexArray &indicesB,
                      DistTensor<T> &C, const IndexArray &indicesC,
                      const std::vector<Unsigned> &blkSizes, bool isStatC) {
  // Determine how to partition
  BlkHadamardStatCInfo hadamardInfo;
  Hadamard<T>::setHadamardInfo(A, indicesA, B, indicesB, C, indicesC, blkSizes,
                               isStatC, hadamardInfo);

  if (isStatC) {
    DistTensor<T> tmpC(C.TensorDist(), C.Grid());
    tmpC.SetLocalPermutation(hadamardInfo.permC);
    Permute(C, tmpC);

    // TODO: Hack need to reset local permutation in permute function
    Permutation defaultPerm(tmpC.Order());
    tmpC.SetLocalPermutation(defaultPerm);

    Hadamard<T>::runHelperPartitionAC(0, hadamardInfo, A, indicesA, B, indicesB,
                                      tmpC, indicesC);
    Permute(tmpC, C);
  } else {
    DistTensor<T> tmpA(A.TensorDist(), A.Grid());
    tmpA.SetLocalPermutation(hadamardInfo.permA);
    Permute(A, tmpA);
    Hadamard<T>::runHelperPartitionAC(0, hadamardInfo, A, indicesA, B, indicesB,
                                      C, indicesC);
  }
}

#define PROTO(T) template class Hadamard<T>;

// PROTO(Unsigned)
// PROTO(Int)
PROTO(float)
PROTO(double)
// PROTO(char)

#ifndef DISABLE_COMPLEX
#ifndef DISABLE_FLOAT
PROTO(std::complex<float>)
#endif
PROTO(std::complex<double>)
#endif

} // namespace rote
