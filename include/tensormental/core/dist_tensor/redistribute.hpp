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
int CheckReduceScatterRedist(const DistTensor<T>& B, const DistTensor<T>& A, const int reduceIndex, const int scatterIndex);

template<typename T>
int CheckPartialReduceScatterRedist(const DistTensor<T>& B, const DistTensor<T>& A, int index, std::vector<int> redistModes);

template<typename T>
int CheckAllGatherRedist(const DistTensor<T>& B, const DistTensor<T>& A, const int allGatherIndex);

//TODO: Not entirely correct definition
template<typename T>
int CheckAllToAllRedist(DistTensor<T>& B, const DistTensor<T>& A, int index);

template<typename T>
int CheckPermutationRedist(DistTensor<T>& B, const DistTensor<T>& A, int index);

/////////////////
//Redist routines
/////////////////
template<typename T>
void ReduceScatterRedist(DistTensor<T>& B, const DistTensor<T>& A, int reduceIndex, int scatterIndex);

template<typename T>
void PartialReduceScatterRedist(DistTensor<T>& B, const DistTensor<T>& A, int index, std::vector<int> redistModes);

template<typename T>
void AllGatherRedist(DistTensor<T>& B, const DistTensor<T>& A, int allGatherIndex);

//TODO: Not entirely correct definition
template<typename T>
void AllToAllRedist(DistTensor<T>& B, const DistTensor<T>& A, int index);

template<typename T>
void PermutationRedist(DistTensor<T>& B, const DistTensor<T>& A, int index);

}
#endif // ifndef TMEN_CORE_DISTTENSOR_REDISTRIBUTE_DECL_HPP
