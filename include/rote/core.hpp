/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef ROTE_CORE_HPP
#define ROTE_CORE_HPP

#if defined(BLAS_POST)
#define BLAS(name) name ## _
#else
#define BLAS(name) name
#endif

#ifdef A2A_OPT
#define OPT true
#else
#define OPT false
#endif

// Declare the intertwined core parts of our library
#include "rote/core/types.hpp"
#include "rote/core/error.hpp"
#include "rote/core/imports.hpp"

#include "rote/core/environment.hpp"

#include "rote/core/memory.hpp"
#include "rote/core/complex.hpp"
#include "rote/core/permutation.hpp"

#include "rote/core/mode_distribution.hpp"
#include "rote/core/tensor_distribution.hpp"
#include "rote/core/util.hpp"
#include "rote/core/structs.hpp"
#include "rote/core/forward_decl.hpp"
#include "rote/core/grid.hpp"
#include "rote/core/grid_view.hpp"
// TODO: Fix indexing_* headers
#include "rote/core/indexing_decl.hpp"
#include "rote/core/indexing_impl.hpp"
// TODO: Fix view headers
#include "rote/core/view.hpp"
#include "rote/core/random.hpp"
#include "rote/core/tensor.hpp"

#include "rote/core/dist_tensor.hpp"

#include "rote/core/time.hpp"

#endif // ifndef ROTE_CORE_HPP
