/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef ROTE_CORE_PARTITION_IMPL_HPP
#define ROTE_CORE_PARTITION_IMPL_HPP

namespace rote {

// To make our life easier. Undef'd at the bottom of the header
#define TEN Tensor<T>
#define DTEN DistTensor<T>

//
// PartitionUp
//

template <typename T>
inline void PartitionUp(TEN &A, TEN &AT, TEN &AB, Mode mode,
                        Unsigned dimensionAB) {
#ifndef RELEASE
  CallStackEntry entry("PartitionUp [Tensor]");
#endif
  PartitionDown(A, AT, AB, A.Dimension(mode) - dimensionAB);
}

template <typename T>
inline void PartitionUp(DTEN &A, DTEN &AT, DTEN &AB, Mode mode,
                        Unsigned dimensionAB) {
#ifndef RELEASE
  CallStackEntry entry("PartitionUp [DistTensor]");
#endif
  PartitionDown(A, AT, AB, A.Dimension(mode) - dimensionAB);
}

template <typename T>
inline void LockedPartitionUp(const TEN &A, TEN &AT, TEN &AB, Mode mode,
                              Unsigned dimensionAB) {
#ifndef RELEASE
  CallStackEntry entry("LockedPartitionUp [Tensor]");
#endif
  LockedPartitionDown(A, AT, AB, A.Dimension(mode) - dimensionAB);
}

template <typename T>
inline void LockedPartitionUp(const DTEN &A, DTEN &AT, DTEN &AB, Mode mode,
                              Unsigned dimensionAB) {
#ifndef RELEASE
  CallStackEntry entry("LockedPartitionUp [DistTensor]");
#endif
  LockedPartitionDown(A, AT, AB, A.Dimension(mode) - dimensionAB);
}

//
// PartitionDown
//

template <typename T>
inline void PartitionDown(TEN &A, TEN &AT, TEN &AB, Mode mode,
                          Unsigned dimensionAT) {
#ifndef RELEASE
  CallStackEntry entry("PartitionDown [Tensor]");
#endif
  ObjShape viewShape = A.Shape();
  Location viewLoc(A.Order());

  dimensionAT = Max(Min(dimensionAT, A.Dimension(mode)), 0);
  const Unsigned dimensionAB = A.Dimension(mode) - dimensionAT;

  viewShape[mode] = dimensionAT;
  std::fill(viewLoc.begin(), viewLoc.end(), 0);
  View(AT, A, viewLoc, viewShape);

  viewLoc[mode] = dimensionAT;
  viewShape[mode] = dimensionAB;
  View(AB, A, viewLoc, viewShape);
}

template <typename T>
inline void PartitionDown(DTEN &A, DTEN &AT, DTEN &AB, Mode mode,
                          Unsigned dimensionAT) {
#ifndef RELEASE
  CallStackEntry entry("PartitionDown [DistTensor]");
#endif
  ObjShape viewShape = A.Shape();
  Location viewLoc(A.Order());

  dimensionAT = Max(Min(dimensionAT, A.Dimension(mode)), 0);
  const Unsigned dimensionAB = A.Dimension(mode) - dimensionAT;

  viewShape[mode] = dimensionAT;
  std::fill(viewLoc.begin(), viewLoc.end(), 0);
  View(AT, A, viewLoc, viewShape);

  viewLoc[mode] = dimensionAT;
  viewShape[mode] = dimensionAB;
  View(AB, A, viewLoc, viewShape);
}

template <typename T>
inline void LockedPartitionDown(const TEN &A, TEN &AT, TEN &AB, Mode mode,
                                Unsigned dimensionAT) {
#ifndef RELEASE
  CallStackEntry entry("LockedPartitionDown [Tensor]");
#endif
  ObjShape viewShape = A.Shape();
  Location viewLoc(A.Order());

  dimensionAT = Max(Min(dimensionAT, A.Dimension(mode)), 0);
  const Unsigned dimensionAB = A.Dimension(mode) - dimensionAT;

  viewShape[mode] = dimensionAT;
  std::fill(viewLoc.begin(), viewLoc.end(), 0);
  LockedView(AT, A, viewLoc, viewShape);

  viewLoc[mode] = dimensionAT;
  viewShape[mode] = dimensionAB;
  LockedView(AB, A, viewLoc, viewShape);
}

template <typename T>
inline void LockedPartitionDown(const DTEN &A, DTEN &AT, DTEN &AB, Mode mode,
                                Unsigned dimensionAT) {
#ifndef RELEASE
  CallStackEntry entry("LockedPartitionDown [DistTensor]");
#endif
  ObjShape viewShape = A.Shape();
  Location viewLoc(A.Order());

  dimensionAT = Max(Min(dimensionAT, A.Dimension(mode)), 0);
  const Unsigned dimensionAB = A.Dimension(mode) - dimensionAT;

  viewShape[mode] = dimensionAT;
  std::fill(viewLoc.begin(), viewLoc.end(), 0);
  LockedView(AT, A, viewLoc, viewShape);

  viewLoc[mode] = dimensionAT;
  viewShape[mode] = dimensionAB;
  LockedView(AB, A, viewLoc, viewShape);
}

#undef DTEN
#undef TEN

} // namespace rote

#endif // ifndef ROTE_CORE_PARTITION_IMPL_HPP
