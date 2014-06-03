/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef TMEN_CORE_HPP
#define TMEN_CORE_HPP

#include "mpi.h"
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stack>
#include <stdexcept>
#include <string>
#include <random>
#include <vector>

//Define the max tensor order for this library build
#ifndef TMEN_MAX_ORDER
# define TMEN_MAX_ORDER 10
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

// Declare the intertwined core parts of our library
//#include "tensormental/core/timer_decl.hpp"
#include "tensormental/core/memory_decl.hpp"
#include "tensormental/core/error_decl.hpp"
#include "tensormental/core/complex_decl.hpp"
#include "tensormental/core/types_decl.hpp"
#include "tensormental/core/tensor_forward_decl.hpp"
#include "tensormental/core/dist_tensor_forward_decl.hpp"
#include "tensormental/core/view_decl.hpp"
#include "tensormental/core/tensor.hpp"
#include "tensormental/core/imports/mpi.hpp"
#include "tensormental/util/vec_util.hpp"
#include "tensormental/core/grid_decl.hpp"
#include "tensormental/core/grid_view_decl.hpp"
#include "tensormental/core/dist_tensor.hpp"
#include "tensormental/core/imports/choice.hpp"
#include "tensormental/core/imports/mpi_choice.hpp"
#include "tensormental/core/environment_decl.hpp"
#include "tensormental/core/indexing_decl.hpp"

#include "tensormental/core/imports/blas.hpp"
//#include "tensormental/core/imports/lapack.hpp"
#include "tensormental/core/imports/flame.hpp"
//#include "tensormental/core/imports/pmrrr.hpp"

// Implement the intertwined parts of the library
//#include "tensormental/core/timer_impl.hpp"
#include "tensormental/util/redistribute_util.hpp"
#include "tensormental/core/memory_impl.hpp"
#include "tensormental/core/complex_impl.hpp"
#include "tensormental/core/types_impl.hpp"
#include "tensormental/core/grid_impl.hpp"
#include "tensormental/core/grid_view_impl.hpp"
#include "tensormental/core/environment_impl.hpp"
#include "tensormental/core/indexing_impl.hpp"

// Declare and implement the decoupled parts of the core of the library
// (perhaps these should be moved into their own directory?)
#include "tensormental/core/view_impl.hpp"
#include "tensormental/core/partition_decl.hpp"
#include "tensormental/core/partition_impl.hpp"
#include "tensormental/core/repartition_decl.hpp"
#include "tensormental/core/repartition_impl.hpp"
#include "tensormental/core/slide_partition_decl.hpp"
#include "tensormental/core/slide_partition_impl.hpp"
//#include "tensormental/core/random_decl.hpp"
//#include "tensormental/core/random_impl.hpp"
//#include "tensormental/core/axpy_interface_decl.hpp"
//#include "tensormental/core/axpy_interface_impl.hpp"

//#include "tensormental/core/ReduceComm.hpp"

#endif // ifndef TMEN_CORE_HPP
