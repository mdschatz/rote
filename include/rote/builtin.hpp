/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef ROTE_BUILTIN_HPP
#define ROTE_BUILTIN_HPP

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

#include "mpi.h"
#include <map>
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
#include <numeric>
#include <sstream>
#include <stack>
#include <stdexcept>
#include <string>
#include <stdlib.h>
#include <vector>
#include <list>
#include <time.h>

#ifdef __MACH__
#include <mach/mach_time.h>
#endif
#include <sys/time.h>

#endif // ifndef ROTE_BUILTIN_HPP
