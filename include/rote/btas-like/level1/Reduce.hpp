/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef ROTE_BTAS_REDUCE_HPP
#define ROTE_BTAS_REDUCE_HPP

namespace rote{

////////////////////////////////////
// Workhorse routines
////////////////////////////////////

template <typename T>
void LocalReduceElemSelect_merged(const T alpha, const ObjShape& sB, T const * const a, const std::vector<Unsigned>& stA, T * const b, const std::vector<Unsigned>& stB){
  Unsigned o = sB.size();
  Location l(o, 0);
  Unsigned p = 0;
  Unsigned pA = 0;
  Unsigned pB = 0;

  if(o == 0){
      b[0] += alpha * a[0];
      return;
  }

  while(p < o){
    b[pB] += alpha * a[pA];

    //Update
    l[p]++;
    pA += stA[p];
    pB += stB[p];
    while(l[p] >= sB[p]){
      l[p] = 0;
      pA -= stA[p] * sB[p];
      pB -= stB[p] * sB[p];

      p++;
      if(p >= o){
        break;
      }
      l[p]++;
      pA += stA[p];
      pB += stB[p];
    }
    if (p == o) {
      break;
    }
    p = 0;
  }
}

////////////////////////////////////
// Local interfaces
////////////////////////////////////

template <typename T>
void LocalReduce(const T alpha, const Tensor<T>& A, Tensor<T>& B, const Permutation& permBToA, const ModeArray& reduceModes){
#ifndef RELEASE
  if(reduceModes.size() > A.Order())
    LogicError("LocalReduce: modes must be of length <= order");

  for(Unsigned i = 0; i < reduceModes.size(); i++)
    if(reduceModes[i] >= A.Order())
      LogicError("LocalReduce: Supplied mode is out of range");
#endif

  const ObjShape sA = A.Shape();
  const std::vector<Unsigned> stA = A.Strides();
  const std::vector<Unsigned> stB = permBToA.applyTo(B.Strides());
  const std::vector<Unsigned> zero(reduceModes.size(), 0);
  Location stUseA = ConcatenateVectors(FilterVector(stA, reduceModes), NegFilterVector(stA, reduceModes));
  Location stUseB = ConcatenateVectors(zero, NegFilterVector(stB, reduceModes));
  ObjShape sUseA = ConcatenateVectors(FilterVector(sA, reduceModes), NegFilterVector(sA, reduceModes));

  LocalReduceElemSelect_merged(alpha, sUseA, A.LockedBuffer(), stUseA, B.Buffer(), stUseB);
}

template <typename T>
void LocalReduce(const T alpha, const Tensor<T>& A, Tensor<T>& B, const ModeArray& reduceModes){
  Permutation perm(A.Order());
  LocalReduce(alpha, A, B, perm, reduceModes);
}

template <typename T>
void LocalReduce(const T alpha, const Tensor<T>& A, Tensor<T>& B, const Mode& reduceMode){
  ModeArray modeArr(1);
  modeArr[0] = reduceMode;
  LocalReduce(alpha, A, B, modeArr);
}

////////////////////////////////////
// Global interfaces
////////////////////////////////////

template <typename T>
void LocalReduce(const T alpha, const DistTensor<T>& A, DistTensor<T>& B, const ModeArray& reduceModes){
  PROFILE_SECTION("LocalReduce");
  Unsigned i;
  ObjShape shapeB = A.Shape();
  for(i = 0; i < reduceModes.size(); i++)
    shapeB[reduceModes[i]] = Min(A.GetGridView().Dimension(reduceModes[i]), A.Dimension(reduceModes[i]));
  B.ResizeTo(shapeB);
  Zero(B);

  if(B.Participating()){
    //Account for the local data being permuted

    Permutation permBToA = B.LocalPermutation().PermutationTo(A.LocalPermutation());
    LocalReduce(alpha, A.LockedTensor(), B.Tensor(), permBToA, FilterVector(A.LocalPermutation().InversePermutation().Entries(), reduceModes));
  }
  PROFILE_STOP;
}

template <typename T>
void LocalReduce(const T alpha, const DistTensor<T>& A, DistTensor<T>& B, const Mode& reduceMode){
  if(B.Participating()){
    ModeArray modeArr(1);
    modeArr[0] = reduceMode;
    LocalReduce(alpha, A, B, modeArr);
  }
}
} // namespace rote

#endif // ifndef ROTE_BTAS_CONTRACT_HPP
