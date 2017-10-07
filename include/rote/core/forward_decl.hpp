/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef ROTE_CORE_FORWARD_DECL_HPP
#define ROTE_CORE_FORWARD_DECL_HPP

namespace rote {

template<typename T>
class DistTensorBase;

template<typename T>
class DistTensor;

// TODO: Move this
template<typename T>
class Hadamard;

class GridView;
class Grid;

class TensorDistribution;
class ModeDistribution;
class Permutation;

template<typename T>
class Memory;

template<typename T>
class Tensor;

} // namespace rote

#endif // ifndef ROTE_CORE_FORWARD_DECL_HPP
