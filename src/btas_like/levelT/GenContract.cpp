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

// TODO: Handle updates
// Note: StatA equivalent to StatB with rearranging operands
template <typename T>
void GenContract(T alpha, const DistTensor<T> &A, const IndexArray &indicesA,
                 const DistTensor<T> &B, const IndexArray &indicesB, T beta,
                 DistTensor<T> &C, const IndexArray &indicesC,
                 const std::vector<Unsigned> &blkSizes) {
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
    ContractStatA(alpha, A, indicesA, B, indicesB, beta, C, indicesC, blkSizes);
  } else if ((isSmallerAB && isBiggerEqualAC) ||
             (isSmallerAB && isSmallerAC && isBiggerBC)) {
    ContractStatA(alpha, B, indicesB, A, indicesA, beta, C, indicesC, blkSizes);
  } else if ((isBiggerAB && isSmallerEqualAC) ||
             (isEqualAB && isSmallerEqualAC) ||
             (isSmallerAB && isSmallerAC && isSmallerEqualBC)) {
    ContractStatC(alpha, B, indicesB, A, indicesA, beta, C, indicesC, blkSizes);
  } else {
    LogicError("Should never occur");
  }
}

template <typename T>
void GenContract(T alpha, const DistTensor<T> &A, const std::string &indicesA,
                 const DistTensor<T> &B, const std::string &indicesB, T beta,
                 DistTensor<T> &C, const std::string &indicesC,
                 const std::vector<Unsigned> &blkSizes) {
  IndexArray indA(indicesA.size());
  for (int i = 0; i < indicesA.size(); i++)
    indA[i] = indicesA[i];

  IndexArray indB(indicesB.size());
  for (int i = 0; i < indicesB.size(); i++)
    indB[i] = indicesB[i];

  IndexArray indC(indicesC.size());
  for (int i = 0; i < indicesC.size(); i++)
    indC[i] = indicesC[i];

  GenContract(alpha, A, indA, B, indB, beta, C, indC, blkSizes);
}

// Non-template functions
// bool AnyFalseElem(const std::vector<bool>& vec);
#define PROTO(T)                                                               \
  template void GenContract(                                                   \
      T alpha, const DistTensor<T> &A, const IndexArray &indicesA,             \
      const DistTensor<T> &B, const IndexArray &indicesB, T beta,              \
      DistTensor<T> &C, const IndexArray &indicesC,                            \
      const std::vector<Unsigned> &blkSizes);                                  \
  template void GenContract(                                                   \
      T alpha, const DistTensor<T> &A, const std::string &indicesA,            \
      const DistTensor<T> &B, const std::string &indicesB, T beta,             \
      DistTensor<T> &C, const std::string &indicesC,                           \
      const std::vector<Unsigned> &blkSizes);

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
