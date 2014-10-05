/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef TMEN_BTAS_CONTRACT_DECL_HPP
#define TMEN_BTAS_CONTRACT_DECL_HPP

namespace tmen{

///////////////////////
// Local routines
///////////////////////

template <typename T>
void LocalContract(T alpha, const Tensor<T>& A, const IndexArray& indicesA, const Tensor<T>& B, const IndexArray& indicesB, T beta, Tensor<T>& C, const IndexArray& indicesC);

template <typename T>
void LocalContractAndLocalEliminate(T alpha, const Tensor<T>& A, const IndexArray& indicesA, const Tensor<T>& B, const IndexArray& indicesB, T beta, Tensor<T>& C, const IndexArray& indicesC);

///////////////////////
// Direct routines
///////////////////////

template <typename T>
void LocalContractDirect(T alpha, const Tensor<T>& A, const IndexArray& indicesA, const bool permuteA, const Tensor<T>& B, const IndexArray& indicesB, const bool permuteB, T beta, Tensor<T>& C, const IndexArray& indicesC, const bool permuteC);

template <typename T>
void LocalContractAndLocalEliminateDirect(T alpha, const Tensor<T>& A, const IndexArray& indicesA, const bool permuteA, const Tensor<T>& B, const IndexArray& indicesB, const bool permuteB, T beta, Tensor<T>& C, const IndexArray& indicesC, const bool permuteC);

} // namespace tmen

#endif // ifndef TMEN_BTAS_CONTRACT_DECL_HPP
