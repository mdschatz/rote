/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef TMEN_CORE_DISTTENSOR_REDISTRIBUTE_A2ACOMM_HPP
#define TMEN_CORE_DISTTENSOR_REDISTRIBUTE_A2ACOMM_HPP

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
Int CheckAllToAllDoubleModeCommRedist(const DistTensor<T>& B, const DistTensor<T>& A, const std::pair<Mode, Mode>& a2aModes, const std::pair<ModeArray, ModeArray >& a2aCommGroups);


/////////////////
//Redist routines
/////////////////
template<typename T>
void AllToAllDoubleModeCommRedist(DistTensor<T>& B, const DistTensor<T>& A, const std::pair<Mode, Mode>& a2aIndices, const std::pair<ModeArray, ModeArray >& a2aCommGroups);

/////////////////
//Pack routine
/////////////////
template <typename T>
void PackA2ADoubleModeCommSendBuf(const DistTensor<T>& B, const DistTensor<T>& A, const std::pair<Mode, Mode>& a2aModes, const std::pair<ModeArray, ModeArray >& commGroups, T * const sendBuf);

/////////////////
//Unpack routine
/////////////////
template <typename T>
void UnpackA2ADoubleModeCommRecvBuf(const T * const recvBuf, const std::pair<Mode, Mode>& a2aModes, const std::pair<ModeArray, ModeArray >& commGroups, const DistTensor<T>& A, DistTensor<T>& B);


}
#endif // ifndef TMEN_CORE_DISTTENSOR_REDISTRIBUTE_A2ACOMM_HPP
