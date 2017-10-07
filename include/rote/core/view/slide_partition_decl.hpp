/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef ROTE_CORE_SLIDEPARTITION_DECL_HPP
#define ROTE_CORE_SLIDEPARTITION_DECL_HPP

namespace rote {

// To make our life easier. Undef'd at the bottom of the header
#define TEN  Tensor<T>
#define DTEN DistTensor<T>

//
// SlidePartitionUp
//

template<typename T>
void SlidePartitionUp
( TEN& AT, TEN& A0,
         TEN& A1,
  TEN& AB, TEN& A2, Mode mode );

template<typename T>
void SlidePartitionUp
( DTEN& AT, DTEN& A0,
          DTEN& A1,
  DTEN& AB, DTEN& A2, Mode mode );

template<typename T>
void SlideLockedPartitionUp
( TEN& AT, const TEN& A0,
         const TEN& A1,
  TEN& AB, const TEN& A2, Mode mode );

template<typename T>
void SlideLockedPartitionUp
( DTEN& AT, const DTEN& A0,
          const DTEN& A1,
  DTEN& AB, const DTEN& A2, Mode mode );

//
// SlidePartitionDown
//

template<typename T>
void SlidePartitionDown
( TEN& AT, TEN& A0,
         TEN& A1,
  TEN& AB, TEN& A2, Mode mode );

template<typename T>
void SlidePartitionDown
( DTEN& AT, DTEN& A0,
          DTEN& A1,
  DTEN& AB, DTEN& A2, Mode mode );

template<typename T>
void SlideLockedPartitionDown
( TEN& AT, const TEN& A0,
         const TEN& A1,
  TEN& AB, const TEN& A2, Mode mode );

template<typename T>
void SlideLockedPartitionDown
( DTEN& AT, const DTEN& A0,
          const DTEN& A1,
  DTEN& AB, const DTEN& A2, Mode mode );

#undef DTEN
#undef TEN

} // namespace rote

#endif // ifndef TTENEN_CORE_SLIDEPARTITION_DECL_HPP
