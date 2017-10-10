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

namespace rote{

////////////////////////////////////
// LocalContract Workhorse
////////////////////////////////////

template <typename T>
void LocalContract(
  T alpha,
  const Tensor<T>& A, const IndexArray& indicesA, const bool permuteA,
  const Tensor<T>& B, const IndexArray& indicesB, const bool permuteB,
  T beta,
        Tensor<T>& C, const IndexArray& indicesC, const bool permuteC
);

template <typename T>
void LocalContractForRun(
  T alpha,
  const Tensor<T>& A, const IndexArray& indicesA,
  const Tensor<T>& B, const IndexArray& indicesB,
  T beta,
        Tensor<T>& C, const IndexArray& indicesC,
  bool doEliminate, bool doPermute
);

////////////////////////////////////
// Local Interfaces
////////////////////////////////////
// TODO: Deprecate. in doing so, deprecate 'doPermute' param from Contract::run (local)
template <typename T>
void LocalContract(T alpha, const Tensor<T>& A, const IndexArray& indicesA, const Tensor<T>& B, const IndexArray& indicesB, T beta, Tensor<T>& C, const IndexArray& indicesC);

template <typename T>
void LocalContractAndLocalEliminate(T alpha, const Tensor<T>& A, const IndexArray& indicesA, const Tensor<T>& B, const IndexArray& indicesB, T beta, Tensor<T>& C, const IndexArray& indicesC);

template <typename T>
void LocalContractAndLocalEliminate(T alpha, const Tensor<T>& A, const IndexArray& indicesA, const bool permuteA, const Tensor<T>& B, const IndexArray& indicesB, const bool permuteB, T beta, Tensor<T>& C, const IndexArray& indicesC, const bool permuteC);

////////////////////////////////////
// DistContract Workhorse
////////////////////////////////////

template <typename T>
void ContractStat(T alpha, const DistTensor<T>& A, const IndexArray& indicesA, const DistTensor<T>& B, const IndexArray& indicesB, T beta, DistTensor<T>& C, const IndexArray& indicesC, bool isStatC, const std::vector<Unsigned>& blkSizes = std::vector<Unsigned>(0));

template <typename T>
void GenContract(T alpha, const DistTensor<T>& A, const IndexArray& indicesA, const DistTensor<T>& B, const IndexArray& indicesB, T beta, DistTensor<T>& C, const IndexArray& indicesC, const std::vector<Unsigned>& blkSizes = std::vector<Unsigned>(0));

template <typename T>
void GenContract(T alpha, const DistTensor<T>& A, const std::string& indicesA, const DistTensor<T>& B, const std::string& indicesB, T beta, DistTensor<T>& C, const std::string& indicesC, const std::vector<Unsigned>& blkSizes = std::vector<Unsigned>(0));
} // namespace rote

#endif // ifndef ROTE_BTAS_CONTRACT_HPP
