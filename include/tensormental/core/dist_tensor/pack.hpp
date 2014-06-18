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
void PackPermutationSendBuf(const DistTensor<T>& B, const DistTensor<T>& A, const Mode permuteMode, T * const sendBuf);

template <typename T>
void PackPartialRSSendBuf(const DistTensor<T>& B, const DistTensor<T>& A, const Mode reduceScatterMode, T * const sendBuf);

template <typename T>
void PackRSSendBuf(const DistTensor<T>& B, const DistTensor<T>& A, const Mode reduceMode, const Mode scatterMode, T * const sendBuf);

template <typename T>
void PackAGSendBuf(const DistTensor<T>& A, const Mode allGatherMode, T * const sendBuf, const ModeArray& redistModes);

template <typename T>
void PackAGSendBuf(const DistTensor<T>& A, const Mode allGatherMode, T * const sendBuf);

template <typename T>
void PackA2ADoubleModeSendBuf(const DistTensor<T>& B, const DistTensor<T>& A, const std::pair<Mode, Mode>& a2aModes, const std::pair<ModeArray, ModeArray >& commGroups, T * const sendBuf);

////////////////////
///  Unpack routines
////////////////////

template <typename T>
void UnpackPermutationRecvBuf(const T * const recvBuf, const Mode permuteMode, const DistTensor<T>& A, DistTensor<T>& B);

template <typename T>
void UnpackPartialRSRecvBuf(const T * const recvBuf, const Mode reduceScatterMode, const DistTensor<T>& A, DistTensor<T>& B);

template <typename T>
void UnpackRSRecvBuf(const T* const recvBuf, const Mode reduceMode,
        const Mode scatterMode, const DistTensor<T>& A, DistTensor<T>& B);

template <typename T>
void UnpackAGRecvBuf(const T * const recvBuf, const Mode allGatherMode, const ModeArray& redistModes, const DistTensor<T>& A, DistTensor<T>& B);

template <typename T>
void UnpackAGRecvBuf(const T * const recvBuf, const Mode allGatherMode, const DistTensor<T>& A, DistTensor<T>& B);

template <typename T>
void UnpackLocalRedist(DistTensor<T>& B, const DistTensor<T>& A, const Mode localMode, const ModeArray& gridRedistModes);

template <typename T>
void UnpackA2ADoubleModeRecvBuf(const T * const recvBuf, const std::pair<Mode, Mode>& a2aModes, const std::pair<ModeArray, ModeArray >& commGroups, const DistTensor<T>& A, DistTensor<T>& B);

}
#endif // ifndef TMEN_CORE_DISTTENSOR_REDISTRIBUTE_UTIL_DECL_HPP
