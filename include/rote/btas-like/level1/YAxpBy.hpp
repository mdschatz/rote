/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef ROTE_BTAS_YAXPBY_HPP
#define ROTE_BTAS_YAXPBY_HPP

#include "rote.hpp"

namespace rote {

////////////////////////////////////
// Global interfaces
////////////////////////////////////

template<typename T>
void YAxpBy_fast(T alpha, T beta, T const * const srcBuf, T * const dstBuf, const YAxpByData& data );

template<typename T>
void YAxpy( T alpha, const DistTensor<T>& X, DistTensor<T>& Y );

template<typename T>
void YxpBy( const DistTensor<T>& X, T beta, DistTensor<T>& Y );

template<typename T>
void Yxpy( const DistTensor<T>& X, DistTensor<T>& Y );

template<typename T>
void YAxpBy( T alpha, const DistTensor<T>& X, T beta, DistTensor<T>& Y );

} // namespace rote

#endif // ifndef ROTE_BTAS_YAXPBY_HPP
