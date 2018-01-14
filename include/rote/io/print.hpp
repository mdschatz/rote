/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef ROTE_IO_PRINT_HPP
#define ROTE_IO_PRINT_HPP

namespace rote {

template<typename T>
void
Print( const Tensor<T>& A, std::string title="", bool all = false );

template<typename T>
void
PrintVector
( const std::vector<T>& vec, std::string title="", bool all = false );

template<typename T>
void
Print
( const DistTensor<T>& A, std::string title="");

template<typename T>
void
PrintData
( const Tensor<T>& A, std::string title="", bool all=false);

template<typename T>
void
PrintData
( const DistTensor<T>& A, std::string title="", bool all=false);

template<typename T>
void
PrintArray
( const T* dataBuf, const ObjShape& shape, const ObjShape strides, std::string title="");

template<typename T>
void
PrintArray
( const T* dataBuf, const ObjShape& loopShape, std::string title="");

void
PrintPackData
( const PackData& packData, std::string title="");

void
PrintElemScalData
( const ElemScalData& elemScalData, std::string title="", bool all=false);

void
PrintHadamardStatCData
( const BlkHadamardStatCInfo& hadamardInfo, std::string title="", bool all=false);

void
PrintHadamardScalData
( const HadamardScalData& hadamardInfo, std::string title="", bool all=false);

void
PrintRedistPlan
( const TensorDistribution& startDist, const RedistPlan& redistPlan, std::string title="");

void
PrintRedistPlanInfo
( const RedistPlanInfo& redistPlanInfo, std::string title="", bool all=false);

} // namespace rote

#endif // ifndef ROTE_IO_PRINT_HPP
