/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef ROTE_CORE_PARTITION_DECL_HPP
#define ROTE_CORE_PARTITION_DECL_HPP

namespace rote {

// To make our life easier. Undef'd at the bottom of the header
#define TEN Tensor<T>
#define DTEN DistTensor<T>

//
// PartitionUp
//

template <typename T>
void PartitionUp(TEN &A, TEN &AT, TEN &AB, Mode mode,
                 Unsigned dimensionAB = Blocksize());

template <typename T>
void PartitionUp(DTEN &A, DTEN &AT, DTEN &AB, Mode mode,
                 Unsigned dimensionAB = Blocksize());

template <typename T>
void LockedPartitionUp(const TEN &A, TEN &AT, TEN &AB, Mode mode,
                       Unsigned dimensionAB = Blocksize());

template <typename T>
void LockedPartitionUp(const DTEN &A, DTEN &AT, DTEN &AB, Mode mode,
                       Unsigned dimensionAB = Blocksize());

//
// PartitionDown
//

template <typename T>
void PartitionDown(TEN &A, TEN &AT, TEN &AB, Mode mode,
                   Unsigned dimensionAT = Blocksize());

template <typename T>
void PartitionDown(DTEN &A, DTEN &AT, DTEN &AB, Mode mode,
                   Unsigned dimensionAT = Blocksize());

template <typename T>
void LockedPartitionDown(const TEN &A, TEN &AT, TEN &AB, Mode mode,
                         Unsigned dimensionAT = Blocksize());

template <typename T>
void LockedPartitionDown(const DTEN &A, DTEN &AT, DTEN &AB, Mode mode,
                         Unsigned dimensionAT = Blocksize());

#undef DTEN
#undef TEN

} // namespace rote

#endif // ifndef ROTE_CORE_PARTITION_DECL_HPP
