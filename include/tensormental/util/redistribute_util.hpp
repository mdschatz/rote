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
void DeterminePermCommunicateDataSize(const DistTensor<T>& B, const DistTensor<T>& A, const Mode permuteMode, Unsigned& recvSize, Unsigned& sendSize);

template <typename T>
void DeterminePartialRSCommunicateDataSize(const DistTensor<T>& B, const DistTensor<T>& A, const Mode reduceScatterMode, Unsigned& recvSize, Unsigned& sendSize);

template <typename T>
void DetermineRSCommunicateDataSize(const DistTensor<T>& B, const DistTensor<T>& A, const Mode reduceMode, Unsigned& recvSize, Unsigned& sendSize);

template <typename T>
void DetermineAGCommunicateDataSize(const DistTensor<T>& A, const Mode allGatherMode, const ModeArray& redistModes, Unsigned& recvSize, Unsigned& sendSize);

template <typename T>
void DetermineAGCommunicateDataSize(const DistTensor<T>& A, const Mode allGatherMode, Unsigned& recvSize, Unsigned& sendSize);

template <typename T>
void DetermineA2ADoubleModeCommunicateDataSize(const DistTensor<T>& B, const DistTensor<T>& A, const std::pair<Mode, Mode>& a2aModes, const std::pair<ModeArray, ModeArray >& a2aCommModes, Unsigned& recvSize, Unsigned& sendSize);

}
#endif // ifndef TMEN_CORE_UTIL_REDISTRIBUTE_UTIL_DECL_HPP
