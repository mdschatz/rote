/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef TMEN_CORE_DISTTENSOR_REDISTRIBUTE_UMODEINTERFACES_HPP
#define TMEN_CORE_DISTTENSOR_REDISTRIBUTE_UMODEINTERFACES_HPP

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
Int CheckRemoveUnitModesCommRedist(const DistTensor<T>& B, const ModeArray& unitModes);

template<typename T>
Int CheckRemoveUnitModeCommRedist(const DistTensor<T>& B, const Mode& unitMode);

template<typename T>
Int CheckIntroduceUnitModesCommRedist(const DistTensor<T>& B, const ModeArray& newModePositions);

template<typename T>
Int CheckIntroduceUnitModeCommRedist(const DistTensor<T>& B, const Mode& newModePosition);

/////////////////
//Redist routines
/////////////////
template<typename T>
void RemoveUnitModesCommRedist(DistTensor<T>& B, const ModeArray& unitModes);

template<typename T>
void RemoveUnitModeCommRedist(DistTensor<T>& B, const Mode& unitMode);

template<typename T>
void IntroduceUnitModesCommRedist(const DistTensor<T>& B, const std::vector<Unsigned>& newModePositions);

template<typename T>
void IntroduceUnitModeCommRedist(const DistTensor<T>& B, const Unsigned& newModePosition);

}
#endif // ifndef TMEN_CORE_DISTTENSOR_REDISTRIBUTE_UMODEINTERFACES_HPP
