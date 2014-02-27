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

template <typename T>
void PackRSSendBuf(const DistTensor<T>& A, const Int reduceScatterMode, T * const sendBuf);

template <typename T>
void PackRSSendBuf(const DistTensor<T>& A, const Int reduceMode, const Int scatterMode, T * const sendBuf);

template <typename T>
void UnpackRSRecvBuf(const T * const recvBuf, const Int reduceScatterMode, DistTensor<T>& A);

template <typename T>
void UnpackRSRecvBuf(const T * const recvBuf, const Int reduceMode, const Int scatterMode, const DistTensor<T>& A, DistTensor<T>& B);

template <typename T>
void PackAGSendBuf(const DistTensor<T>& A, const Int allGatherMode, T * const sendBuf);

template <typename T>
void UnpackAGRecvBuf(const T * const recvBuf, const Int allGatherMode, const DistTensor<T>& A, DistTensor<T>& B);

}
#endif // ifndef TMEN_CORE_DISTTENSOR_REDISTRIBUTE_UTIL_DECL_HPP
