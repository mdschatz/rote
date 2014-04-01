/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/

#pragma once
#ifndef TMEN_CORE_DISTTENSOR_PACK_DECL_HPP
#define TMEN_CORE_DISTTENSOR_PACK_DECL_HPP

#include "tensormental/core/imports/mpi.hpp"
#include "tensormental/core/imports/choice.hpp"
#include "tensormental/core/imports/mpi_choice.hpp"

#include <vector>
#include <iostream>
#include "tensormental/core/dist_tensor/mc_mr.hpp"

namespace tmen{

////////////////////
///  Pack routines
////////////////////

template <typename T>
void PackPermutationSendBuf(const DistTensor<T>& B, const DistTensor<T>& A, const Int permuteIndex, T * const sendBuf);

template <typename T>
void PackPartialRSSendBuf(const DistTensor<T>& B, const DistTensor<T>& A, const Int reduceScatterIndex, T * const sendBuf);

template <typename T>
void PackRSSendBuf(const DistTensor<T>& B, const DistTensor<T>& A, const Int reduceIndex, const Int scatterIndex, T * const sendBuf);

template <typename T>
void PackAGSendBuf(const DistTensor<T>& A, const Int allGatherIndex, T * const sendBuf);

template <typename T>
void PackA2ADoubleIndexSendBuf(const DistTensor<T>& B, const DistTensor<T>& A, const std::pair<int, int>& a2aIndices, const std::pair<std::vector<int>, std::vector<int> >& commGroups, T * const sendBuf);

////////////////////
///  Unpack routines
////////////////////

template <typename T>
void UnpackPermutationRecvBuf(const T * const recvBuf, const Int permuteIndex, const DistTensor<T>& A, DistTensor<T>& B);

template <typename T>
void UnpackPartialRSRecvBuf(const T * const recvBuf, const Int reduceScatterIndex, const DistTensor<T>& A, DistTensor<T>& B);

template <typename T>
void UnpackRSRecvBuf(const T* const recvBuf, const Int reduceIndex,
        const Int scatterIndex, const DistTensor<T>& A, DistTensor<T>& B);

template <typename T>
void UnpackAGRecvBuf(const T * const recvBuf, const Int allGatherIndex, const DistTensor<T>& A, DistTensor<T>& B);

template <typename T>
void UnpackA2ADoubleIndexRecvBuf(const T * const recvBuf, const std::pair<int, int>& a2aIndices, const std::vector<int>& commModes, const DistTensor<T>& A, DistTensor<T>& B);

}
#endif // ifndef TMEN_CORE_DISTTENSOR_REDISTRIBUTE_UTIL_DECL_HPP
