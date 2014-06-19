/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef TMEN_CORE_DISTTENSOR_REDISTRIBUTE_PCOMM_HPP
#define TMEN_CORE_DISTTENSOR_REDISTRIBUTE_PCOMM_HPP

#include "tensormental/core/imports/mpi.hpp"
#include "tensormental/core/imports/choice.hpp"
#include "tensormental/core/imports/mpi_choice.hpp"

#include <vector>
#include <iostream>
#include "tensormental/core/dist_tensor/mc_mr.hpp"

namespace tmen{

/////////////////
//Check routines
/////////////////
template<typename T>
Int CheckPermutationCommRedist(const DistTensor<T>& B, const DistTensor<T>& A, const Mode permuteMode, const ModeArray& redistModes);

/////////////////
//Redist routines
/////////////////
template<typename T>
void PermutationCommRedist(DistTensor<T>& B, const DistTensor<T>& A, const Mode permuteMode, const ModeArray& redistModes);

/////////////////
//Pack routine
/////////////////
template <typename T>
void PackPermutationCommSendBuf(const DistTensor<T>& B, const DistTensor<T>& A, const Mode permuteMode, T * const sendBuf);

/////////////////
//Unpack routine
/////////////////
template <typename T>
void UnpackPermutationCommRecvBuf(const T * const recvBuf, const Mode permuteMode, const DistTensor<T>& A, DistTensor<T>& B);

}
#endif // ifndef TMEN_CORE_DISTTENSOR_REDISTRIBUTE_PCOMM_HPP
