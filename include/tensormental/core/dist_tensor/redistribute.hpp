/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef TMEN_CORE_DISTTENSOR_REDISTRIBUTE_DECL_HPP
#define TMEN_CORE_DISTTENSOR_REDISTRIBUTE_DECL_HPP

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
Int CheckPermutationRedist(const DistTensor<T>& B, const DistTensor<T>& A, const Mode permuteMode);

template<typename T>
Int CheckPartialReduceScatterRedist(const DistTensor<T>& B, const DistTensor<T>& A, const Mode reduceScatterMode);

template<typename T>
Int CheckReduceScatterRedist(const DistTensor<T>& B, const DistTensor<T>& A, const Mode reduceMode, const Mode scatterMode);

template<typename T>
Int CheckAllGatherRedist(const DistTensor<T>& B, const DistTensor<T>& A, const Mode allGatherMode, const ModeArray& redistModes);

template<typename T>
Int CheckAllGatherRedist(const DistTensor<T>& B, const DistTensor<T>& A, const Mode allGatherMode);

template<typename T>
Int CheckAllToAllDoubleModeRedist(const DistTensor<T>& B, const DistTensor<T>& A, const std::pair<Mode, Mode>& a2aModes, const std::pair<ModeArray, ModeArray >& a2aCommGroups);

template<typename T>
Int CheckLocalRedist(DistTensor<T>& B, const DistTensor<T>& A, const Mode localMode, const ModeArray& gridRedistModes);

template<typename T>
Int CheckRemoveUnitModesRedist(const DistTensor<T>& B, const DistTensor<T>& A, const ModeArray& unitModes);

template<typename T>
Int CheckIntroduceUnitModesRedist(const DistTensor<T>& B, const DistTensor<T>& A, const ModeArray& newModePositions);

//template<typename T>
//int CheckPartialReduceScatterRedist(const DistTensor<T>& B, const DistTensor<T>& A, const Mode mode, const ModeArray& rsGridModes);

//TODO: Not entirely correct definition
template<typename T>
Int CheckAllToAllRedist(const DistTensor<T>& B, const DistTensor<T>& A, const Mode allToAllMode);


/////////////////
//Redist routines
/////////////////

template<typename T>
void PermutationRedist(DistTensor<T>& B, const DistTensor<T>& A, const Mode permuteMode);

template<typename T>
void PartialReduceScatterRedist(DistTensor<T>& B, const DistTensor<T>& A, const Mode reduceScatterMode);

template<typename T>
void ReduceScatterRedist(DistTensor<T>& B, const DistTensor<T>& A, const Mode reduceMode, const Mode scatterMode);

template<typename T>
void AllGatherRedist(DistTensor<T>& B, const DistTensor<T>& A, const Mode allGatherMode, const ModeArray& redistModes);

template<typename T>
void AllGatherRedist(DistTensor<T>& B, const DistTensor<T>& A, const Mode allGatherMode);

template<typename T>
void AllToAllDoubleModeRedist(DistTensor<T>& B, const DistTensor<T>& A, const std::pair<Mode, Mode>& a2aIndices, const std::pair<ModeArray, ModeArray >& a2aCommGroups);

template<typename T>
void LocalRedist(DistTensor<T>& B, const DistTensor<T>& A, const Mode localMode, const ModeArray& gridRedistModes);

template<typename T>
void RemoveUnitModesRedist(DistTensor<T>& B, const DistTensor<T>& A, const ModeArray& unitModes);

template<typename T>
void IntroduceUnitModesRedist(const DistTensor<T>& B, const DistTensor<T>& A, const std::vector<Unsigned>& newModePositions);

//template<typename T>
//void PartialReduceScatterRedist(DistTensor<T>& B, const DistTensor<T>& A, const Mode reduceScatterMode, const ModeArray& rsGridModes);

//TODO: Not entirely correct definition
template<typename T>
void AllToAllRedist(DistTensor<T>& B, const DistTensor<T>& A, const Mode mode);


}
#endif // ifndef TMEN_CORE_DISTTENSOR_REDISTRIBUTE_DECL_HPP
