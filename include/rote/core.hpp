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

//Define the max tensor order for this library build
#ifndef ROTE_MAX_ORDER
# define ROTE_MAX_ORDER 10
#endif

// If defined, the _OPENMP macro contains the date of the specification
#ifdef HAVE_OPENMP
# include <omp.h>
# if _OPENMP >= 200805
#  define COLLAPSE(N) collapse(N)
# else
#  define COLLAPSE(N)
# endif
# define PARALLEL_FOR _Pragma("omp parallel for")
#else
# define PARALLEL_FOR
# define COLLAPSE(N)
#endif

#ifdef AVOID_OMP_FMA
# define FMA_PARALLEL_FOR
#else
# define FMA_PARALLEL_FOR PARALLEL_FOR
#endif
#ifdef PARALLELIZE_INNER_LOOPS
# define INNER_PARALLEL_FOR PARALLEL_FOR
# define OUTER_PARALLEL_FOR
#else
# define INNER_PARALLEL_FOR
# define OUTER_PARALLEL_FOR PARALLEL_FOR
#endif

#if defined(BLAS_POST)
#define BLAS(name) name ## _
#else
#define BLAS(name) name
#endif

// The DEBUG_ONLY macro is, to the best of my knowledge, the only preprocessor
// name defined by Elemental that is not namespaced with "ELEM". Given how
// frequently it is used, I will leave it as-is unless/until a user/developer
// complains.
#ifdef PURE_RELEASE
# define DEBUG_ONLY(cmd)
#else
# define DEBUG_ONLY(cmd) cmd;
#endif

#ifdef A2A_OPT
#define OPT true
#else
#define OPT false
#endif

// Declare the intertwined core parts of our library
#include "rote/core/types_decl.hpp"
#include "rote/core/error_decl.hpp"
#include "rote/core/imports/mpi.hpp"
#include "rote/core/memory_decl.hpp"
#include "rote/core/complex_decl.hpp"
#include "rote/core/permutation.hpp"
#include "rote/core/mode_distribution.hpp"
#include "rote/core/tensor_distribution.hpp"
#include "rote/core/structs_decl.hpp"
#include "rote/core/tensor_forward_decl.hpp"
#include "rote/core/dist_tensor_forward_decl.hpp"
#include "rote/core/view_decl.hpp"
#include "rote/core/tensor.hpp"
#include "rote/core/util.hpp"
// #include "rote/util/vec_util.hpp"
// #include "rote/util/ten_dist_util.hpp"
#include "rote/core/imports.hpp"
#include "rote/core/grid_decl.hpp"
#include "rote/core/grid_view_decl.hpp"
#include "rote/core/dist_tensor.hpp"
#include "rote/core/environment_decl.hpp"
#include "rote/core/indexing_decl.hpp"

#include "rote/core/memory_impl.hpp"
#include "rote/core/complex_impl.hpp"
#include "rote/core/grid_impl.hpp"
#include "rote/core/grid_view_impl.hpp"
#include "rote/core/environment_impl.hpp"
#include "rote/core/indexing_impl.hpp"

#include "rote/core/view_impl.hpp"
#include "rote/core/partition_decl.hpp"
#include "rote/core/partition_impl.hpp"
#include "rote/core/repartition_decl.hpp"
#include "rote/core/repartition_impl.hpp"
#include "rote/core/slide_partition_decl.hpp"
#include "rote/core/slide_partition_impl.hpp"
#include "rote/core/random_decl.hpp"
#include "rote/core/random_impl.hpp"

#include "rote/core/time.hpp"
#endif // ifndef ROTE_CORE_HPP
