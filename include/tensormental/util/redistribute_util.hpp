/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/

#pragma once
#ifndef TMEN_CORE_UTIL_REDISTRIBUTE_UTIL_DECL_HPP
#define TMEN_CORE_UTIL_REDISTRIBUTE_UTIL_DECL_HPP

#include "tensormental/core/imports/mpi.hpp"
#include "tensormental/core/imports/choice.hpp"
#include "tensormental/core/imports/mpi_choice.hpp"

#include <vector>
#include <iostream>
#include "tensormental/core.hpp"

namespace tmen{

template <typename T>
void DeterminePermCommunicateDataSize(const DistTensor<T>& B, const DistTensor<T>& A, const Index permuteIndex, Unsigned& recvSize, Unsigned& sendSize);

template <typename T>
void DeterminePartialRSCommunicateDataSize(const DistTensor<T>& B, const DistTensor<T>& A, const Index reduceScatterIndex, Unsigned& recvSize, Unsigned& sendSize);

template <typename T>
void DetermineRSCommunicateDataSize(const DistTensor<T>& B, const DistTensor<T>& A, const Index reduceIndex, Unsigned& recvSize, Unsigned& sendSize);

template <typename T>
void DetermineAGCommunicateDataSize(const DistTensor<T>& A, const Index allGatherIndex, Unsigned& recvSize, Unsigned& sendSize);

template <typename T>
void DetermineA2ADoubleIndexCommunicateDataSize(const DistTensor<T>& B, const DistTensor<T>& A, const std::pair<Index, Index>& a2aIndices, const std::pair<ModeArray, ModeArray >& a2aCommModes, Unsigned& recvSize, Unsigned& sendSize);

}
#endif // ifndef TMEN_CORE_UTIL_REDISTRIBUTE_UTIL_DECL_HPP
