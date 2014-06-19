/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef TMEN_CORE_DISTTENSOR_REDISTRIBUTE_AGCOMM_HPP
#define TMEN_CORE_DISTTENSOR_REDISTRIBUTE_AGCOMM_HPP

#include "tensormental/core/imports/mpi.hpp"
#include "tensormental/core/imports/choice.hpp"
#include "tensormental/core/imports/mpi_choice.hpp"

#include <vector>
#include <iostream>
#include "tensormental/core/dist_tensor/mc_mr.hpp"

namespace tmen{

/////////////////
//Check routine
/////////////////
template<typename T>
Int CheckAllGatherCommRedist(const DistTensor<T>& B, const DistTensor<T>& A, const Mode allGatherMode, const ModeArray& redistModes);

/////////////////
//Redist routine
/////////////////
template<typename T>
void AllGatherCommRedist(DistTensor<T>& B, const DistTensor<T>& A, const Mode allGatherMode, const ModeArray& redistModes);

/////////////////
//Pack routine
/////////////////
template <typename T>
void PackAGCommSendBuf(const DistTensor<T>& A, const Mode allGatherMode, T * const sendBuf, const ModeArray& redistModes);

/////////////////
//Unpack routine
/////////////////
template <typename T>
void UnpackCommAGRecvBuf(const T * const recvBuf, const Mode allGatherMode, const ModeArray& redistModes, const DistTensor<T>& A, DistTensor<T>& B);

}
#endif // ifndef TMEN_CORE_DISTTENSOR_REDISTRIBUTE_AGCOMM_HPP
