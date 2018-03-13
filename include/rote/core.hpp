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

// Dependencies
#include "builtin.hpp"
#include "forward_decl.hpp"
#include "types.hpp"

// Safe to reorder
#include "core/error.hpp"
#include "core/indexing.hpp"
#include "core/imports.hpp"
#include "core/environment.hpp"
#include "core/memory.hpp"
#include "core/complex.hpp"
#include "core/mode_distribution.hpp"
#include "core/tensor_distribution.hpp"
#include "core/redist_plan.hpp"
#include "core/util.hpp"
#include "core/permutation.hpp"
#include "core/structs.hpp"
#include "core/grid.hpp"
#include "core/grid_view.hpp"
// TODO: Fix view headers
#include "core/view.hpp"
#include "core/random.hpp"
#include "core/tensor.hpp"
#include "core/dist_tensor.hpp"
#include "core/time.hpp"

#endif // ifndef ROTE_CORE_HPP
