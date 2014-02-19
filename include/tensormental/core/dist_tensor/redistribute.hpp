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
int CheckReduceScatterRedist(DistTensor<T>& A, const DistTensor<T>& B, int reduceIndex, int scatterIndex);

template<typename T>
int CheckPartialReduceScatterRedist(DistTensor<T>& A, const DistTensor<T>& B, int index, std::vector<int> redistModes);

template<typename T>
int CheckAllGatherRedist(DistTensor<T>& A, const DistTensor<T>& B, int allGatherIndex);

//TODO: Not entirely correct definition
template<typename T>
int CheckAllToAllRedist(DistTensor<T>& A, const DistTensor<T>& B, int index);

template<typename T>
int CheckPermutationRedist(DistTensor<T>& A, const DistTensor<T>& B, int index);

/////////////////
//Redist routines
/////////////////
template<typename T>
void ReduceScatterRedist(DistTensor<T>& A, const DistTensor<T>& B, int reduceIndex, int scatterIndex);

template<typename T>
void PartialReduceScatterRedist(DistTensor<T>& A, const DistTensor<T>& B, int index, std::vector<int> redistModes);

template<typename T>
void AllGatherRedist(DistTensor<T>& A, const DistTensor<T>& B, int allGatherIndex);

//TODO: Not entirely correct definition
template<typename T>
void AllToAllRedist(DistTensor<T>& A, const DistTensor<T>& B, int index);

template<typename T>
void PermutationRedist(DistTensor<T>& A, const DistTensor<T>& B, int index);

}
#endif // ifndef TMEN_CORE_DISTTENSOR_REDISTRIBUTE_DECL_HPP
