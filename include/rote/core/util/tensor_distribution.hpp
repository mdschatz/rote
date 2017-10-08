/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef ROTE_UTIL_TENDIST_HPP
#define ROTE_UTIL_TENDIST_HPP

namespace rote {

//
// Check
//
bool CheckOrder(const Unsigned& outOrder, const Unsigned& inOrder);
bool CheckPartition(const TensorDistribution& outDist, const TensorDistribution& inDist);
bool CheckNonDistOutIsPrefix(const TensorDistribution& outDist, const TensorDistribution& inDist);
bool CheckNonDistInIsPrefix(const TensorDistribution& outDist, const TensorDistribution& inDist);
bool CheckSameNonDist(const TensorDistribution& outDist, const TensorDistribution& inDist);
bool CheckSameCommModes(const TensorDistribution& outDist, const TensorDistribution& inDist);
bool CheckInIsPrefix(const TensorDistribution& outDist, const TensorDistribution& inDist);
bool CheckOutIsPrefix(const TensorDistribution& outDist, const TensorDistribution& inDist);
bool CheckSameGridViewShape(const ObjShape& outShape, const ObjShape& inShape);
bool CheckIsValidPermutation(const Unsigned& order, const Permutation& perm);

//
// Util
//
ModeArray GetCommonSuffixModes(const ModeDistribution& lhs, const ModeDistribution& rhs);
ModeArray GetCommonSuffixModes(const TensorDistribution& lhs, const TensorDistribution& rhs);
ModeDistribution GetCommonPrefix(const ModeDistribution& lhs, const ModeDistribution& rhs);
TensorDistribution GetCommonPrefix(const TensorDistribution& lhs, const TensorDistribution& rhs);

} // namespace rote

#endif // ifndef ROTE_UTIL_TENDIST_HPP
