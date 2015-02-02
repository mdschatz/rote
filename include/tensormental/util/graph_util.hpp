/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef TMEN_UTIL_GRAPHUTIL_HPP
#define TMEN_UTIL_GRAPHUTIL_HPP

#include <vector>
#include <map>
#include <utility>
#include <iostream>
#include "tensormental/core/environment_decl.hpp"
#include "tensormental/core/error_decl.hpp"
#include "tensormental/core/types_decl.hpp"

namespace tmen {

//Move these utils to a better location
void DetermineSCC(const ModeArray& changedModes, const std::vector<std::pair<Mode, Mode> >& tensorModeFromTo, ModeArray& p2pCommModes);
void StrongConnect(Unsigned& minIndex, TarjanVertex& v, std::vector<TarjanVertex>& S,
                   std::map<Mode, TarjanVertex>& mode2TarjanVertexMap, const ModeArray& commModes, const std::vector<std::pair<Mode, Mode> >& tensorModeFromTo,
                   ModeArray& p2pModes);
TensorDistribution CreatePrefixDistribution(const TensorDistribution& inDist, const TensorDistribution& outDist);
TensorDistribution CreatePrefixA2ADistribution(const TensorDistribution& prefixDist, const TensorDistribution& inDist, const ModeArray& a2aModes);
TensorDistribution CreateA2AOptDist1(const TensorDistribution& prefixA2ADist, const TensorDistribution& inDist, const ModeArray& p2pModes);
TensorDistribution CreateA2AOptDist2(const TensorDistribution& prefixA2ADist, const TensorDistribution& outDist, const ModeArray& p2pModes);
TensorDistribution CreatePrefixP2PDistribution(const TensorDistribution& prefixDist, const TensorDistribution& outDist, const ModeArray& p2pModes);
TensorDistribution CreateA2AOptDist3(const TensorDistribution& prefixP2PDist, const TensorDistribution& inDist, const ModeArray& a2aModes);
TensorDistribution CreateA2AOptDist4(const TensorDistribution& prefixP2PDist, const TensorDistribution& outDist, const ModeArray& a2aModes);

} // namespace tmen

#endif // ifndef TMEN_UTIL_GRAPHUTIL_HPP
