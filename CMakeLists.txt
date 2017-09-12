cmake_minimum_required(VERSION 2.8.8)

project(ROTE)
set(ROTE_VERSION_MAJOR 0)
set(ROTE_VERSION_MINOR 0-dev)

################################################################################
# Common build options                                                         #
################################################################################

# By default, Rote builds static libraries, but shared libraries are
# also supported through this option.
option(SHARED_LIBRARIES "Switch to shared libraries" OFF)

# Whether or not to build a collection of simple example drivers.
option(ROTE_EXAMPLES "A collection of simple examples" OFF)

# Whether or not to build a collection of performance and correctness tests
option(ROTE_TESTS "A collection of performance and correctness tests" ON)

# Whether or not to build current 'experimental' code, such as:
# 1) prototype C and F90 interfaces
# 2) Martin's 3D Gemm code
option(ROTE_EXPERIMENTAL "Build experimental code" OFF)

################################################################################
# Advanced build options                                                       #
################################################################################

# Since it is surprisingly common for MPI libraries to have bugs in their
# support for complex data, the following option forces Elemental to cast
# all possible MPI communications in terms of twice as many real units of data.
option(AVOID_COMPLEX_MPI "Avoid potentially buggy complex MPI routines" OFF)
mark_as_advanced(AVOID_COMPLEX_MPI)

# At one point, a bug was found in IBM's C++ compiler for Blue Gene/P,
# where OpenMP statements of the form a[i] += alpha b[i], with complex data,
# would segfault and/or return incorrect results
option(AVOID_OMP_FMA "Avoid a bug in the IBM compilers." OFF)
mark_as_advanced(AVOID_OMP_FMA)

# Due to a subtle flaw in the Blue Gene/P extensions for MPICH2, treating
# floating-point data as a collection of byte-sized objects results in a
# better algorithm being chosen for MPI_Allgather. This should not effect
# performance on most machines.
option(USE_BYTE_ALLGATHERS "Avoid BG/P allgather performance bug." OFF)
mark_as_advanced(USE_BYTE_ALLGATHERS)

# If MPI_Reduce_scatter_block doesn't exist, perform it by composing
# MPI_Allreduce and std::memcpy rather than MPI_Reduce and MPI_Scatter
option(REDUCE_SCATTER_BLOCK_VIA_ALLREDUCE
       "AllReduce based block MPI_Reduce_scatter" OFF)
mark_as_advanced(REDUCE_SCATTER_BLOCK_VIA_ALLREDUCE)

# This can easily be performed, but is likely not a good idea
option(POOL_MEMORY "Make Memory class accumulate memory until destruction" OFF)
mark_as_advanced(POOL_MEMORY)

################################################################################
# Development build options                                                    #
################################################################################

# Print a warning any time a redistribution is performed which unpacks a
# large amount of data with a non-unit stride
option(CACHE_WARNINGS "Warns when using cache-unfriendly routines" OFF)
mark_as_advanced(CACHE_WARNINGS)

# Print a warning when an improperly aligned redistribution is performed,
# i.e., if an unnecessary permutation communication stage must take place
option(UNALIGNED_WARNINGS "Warn when performing unaligned redistributions" OFF)
mark_as_advanced(UNALIGNED_WARNINGS)

# Print a warning if an opportunity was missed to implement a redistribution
# approach specifically for vectors (instead of matrices)
option(VECTOR_WARNINGS "Warn when vector redistribution chances are missed" OFF)
mark_as_advanced(CACHE_WARNINGS UNALIGNED_WARNINGS VECTOR_WARNINGS)

################################################################################
# Significant command-line variable definitions                                #
################################################################################

# "CMAKE_BUILD_TYPE"
#   Elemental requires it to be one of the following choices:
#     1. "PureDebug": Vanilla MPI build meant for development and debugging
#     2. "PureRelease": Vanilla MPI build meant for production runs
#     3. "HybridDebug": MPI+OpenMP build meant for development and debugging
#     4. "HybridRelease": MPI+OpenMP build meant for production runs
#   If "CMAKE_BUILD_TYPE" is not defined, the default is "PureRelease".

# "CXX_FLAGS"
#   Override the default optimization (and debugging/symbol) flags

# "OpenMP_CXX_FLAGS"
#   Overrides the default compile flags for adding OpenMP support to CXX code

# TODO: Add list of MPI variables here, such as "MPI_CXX_COMPILE_FLAGS"

# TODO: Add discussion of "MATH_LIBS"

# TODO: Add discussion of "REFERENCE_ROOT"

# TODO: Check to see if there are any others...

################################################################################
# The build logic starts here                                                  #
################################################################################

# Elemental must be built "out-of-source", so we start by ensuring that the
# source and build directories are different.
if("${PROJECT_SOURCE_DIR}" STREQUAL "${PROJECT_BINARY_DIR}")
  message(FATAL_ERROR "In-source build attempted; please clean the CMake cache and then switch to an out-of-source build, e.g., rm CMakeCache.txt && rm -Rf CMakeFiles/ && mkdir build/ && cd build/ && cmake ..")
endif()

# Get the Git revision
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")
include(GetGitRevisionDescription)
get_git_head_revision(GIT_REFSPEC GIT_SHA1)

# Extract a few booleans from the build type and default to PureRelease
if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "Build mode not specified, defaulting to PureRelease build.")
  set(CMAKE_BUILD_TYPE PureRelease)
endif()
if(CMAKE_BUILD_TYPE STREQUAL "HybridDebug")
  set(HYBRID TRUE)
  set(DEBUG TRUE)
elseif(CMAKE_BUILD_TYPE STREQUAL "HybridRelease")
  set(HYBRID TRUE)
  set(RELEASE TRUE)
elseif(CMAKE_BUILD_TYPE STREQUAL "PureDebug")
  set(PURE TRUE)
  set(DEBUG TRUE)
elseif(CMAKE_BUILD_TYPE STREQUAL "PureRelease")
  set(PURE TRUE)
  set(RELEASE TRUE)
else()
  message(FATAL_ERROR "CMAKE_BUILD_TYPE must be [Hybrid,Pure][Debug,Release]")
endif()
string(TOUPPER ${CMAKE_BUILD_TYPE} UPPER_BUILD_TYPE)

message(STATUS ${CMAKE_PREFIX_PATH})

# Set the basic compile flags from the build type
if(NOT WIN32) # I forget why we need this guard
  set(LANGUAGES CXX C)
  foreach(LANG ${LANGUAGES})
    message("flags" ${${LANG}_FLAGS_${UPPER_BUILD_TYPE}})
    if(NOT ${LANG}_FLAGS_${UPPER_BUILD_TYPE})
      if(DEBUG)
	  message("debug flags set")
          set(${LANG}_FLAGS_${UPPER_BUILD_TYPE} "-O0 -g -Wall -Wunreachable-code -Wunused-parameter -std=c++0x" CACHE STRING
          "${LANG} optimization/debug flags for ${UPPER_BUILD_TYPE} mode")
      else()
        set(${LANG}_FLAGS_${UPPER_BUILD_TYPE} "-O3 -Wall -Wunused-parameter -std=c++0x" CACHE STRING
          "${LANG} optimization flags for ${UPPER_BUILD_TYPE} mode")
      endif()
    endif()
  endforeach()
endif()

# Go ahead and check for Fortran, but keep in mind that CMake's 'OPTIONAL'
# argument for enable_language is still completely broken as of 2.8.8
#include(./cmake/language_support_v2.cmake)
#workaround_9220(Fortran FORTRAN_WORKS)
#if(FORTRAN_WORKS)
#  enable_language(Fortran)
#  set(HAVE_F90_INTERFACE FALSE)
#  if(CMAKE_Fortran_COMPILER_SUPPORTS_F90)
#    include(FortranCInterface)
#    FortranCInterface_VERIFY(CXX)
#    if(FortranCInterface_VERIFIED_CXX)
#      set(HAVE_F90_INTERFACE TRUE)
#      FortranCInterface_HEADER(
#        ${CMAKE_CURRENT_BINARY_DIR}/include/elemental/FCMangle.h
#        MACRO_NAMESPACE "FC_")
#      install(FILES ${PROJECT_BINARY_DIR}/include/elemental/FCMangle.h
#              DESTINATION include/elemental/)
#    endif()
#  endif()
#endif()

# Make sure that MPI is available
find_package(MPI)
if(NOT MPI_CXX_FOUND)
  message(FATAL_ERROR "MPI C++ compiler was not found and is required")
endif()
include_directories(${MPI_CXX_INCLUDE_PATH})
set(EXTRA_FLAGS "${MPI_CXX_COMPILE_FLAGS}")

# Search for OpenMP if necessary
if(HYBRID)
  if(OpenMP_CXX_FLAGS)
    set(HAVE_OPENMP TRUE)
    set(OpenMP_CXX_FLAGS ${OpenMP_CXX_FLAGS})
    message(STATUS "Using user-defined OpenMP_CXX_FLAGS=${OpenMP_CXX_FLAGS}")
  else()
    find_package(OpenMP)
    if(OPENMP_FOUND)
      set(HAVE_OPENMP TRUE)
    else()
      set(OpenMP_CXX_FLAGS "" CACHE STRING "OpenMP CXX FLAGS")
      message(FATAL_ERROR
        "Hybrid build failed because OpenMP support not detected. Please specify OpenMP_CXX_FLAGS.")
    endif()
  endif()
  set(EXTRA_FLAGS "${OpenMP_CXX_FLAGS} ${EXTRA_FLAGS}")
endif()

# Now append the extra flags
set(CMAKE_CXX_FLAGS_${UPPER_BUILD_TYPE} "${CXX_FLAGS} ${CXX_FLAGS_${UPPER_BUILD_TYPE}} ${EXTRA_FLAGS}")

# Set the library type from the 'SHARED_LIBRARIES' option
if(SHARED_LIBRARIES OR HAVE_SWIG)
  if(NOT SHARED_LIBRARIES)
    message(WARNING "Switched to a dynamic build since SWIG is enabled")
  endif()
  set(LIBRARY_TYPE SHARED)
else()
  set(LIBRARY_TYPE STATIC)
endif()

# Since the CMake find_blas for some reason requires a Fortran compiler, we
# must build our own.
if(PURE)
  set(MATH_DESC "Unthreaded BLAS/LAPACK link flags")
else()
  set(MATH_DESC "Threaded BLAS/LAPACK link flags")
endif()
if(MATH_LIBS)
  message(STATUS "Using user-defined MATH_LIBS=${MATH_LIBS}")
elseif(APPLE)
  set(MATH_LIBS "-framework accelerate" CACHE STRING ${MATH_DESC})
 message(STATUS "Using Apple accelerate framework.")
else()
  # Look for default BLAS and LAPACK
  if(REFERENCE_ROOT)
    message(STATUS "Searching REFERENCE_ROOT=${REFERENCE_ROOT} for math libs")
  endif()
  set(REFERENCE_REQUIRED LAPACK BLAS)
  find_library(BLAS_LIB NAMES blas PATHS ${REFERENCE_ROOT})
  find_library(LAPACK_LIB NAMES lapack reflapack PATHS ${REFERENCE_ROOT})
  set(REFERENCE_FOUND TRUE)
  foreach(NAME ${REFERENCE_REQUIRED})
    if(${NAME}_LIB)
      message(STATUS "Found ${NAME}_LIB: ${${NAME}_LIB}")
      list(APPEND MATH_LIBS ${${NAME}_LIB})
    else()
      message(STATUS "Could not find ${NAME}_LIB")
      set(MATH_LIBS "")
      set(REFERENCE_FOUND FALSE)
    endif()
  endforeach()
  if(REFERENCE_FOUND)
    message(WARNING "Using reference BLAS/LAPACK.")
  else()
    message(FATAL_ERROR "Could not find BLAS/LAPACK. Please specify MATH_LIBS")
  endif()
endif()

# Attempt to detect the BLAS and LAPACK underscore conventions.
# We currently only handle whether or not there is an underscore appended.
include(CheckFunctionExists)
set(CMAKE_REQUIRED_LIBRARIES ${MATH_LIBS})
check_function_exists(daxpy  HAVE_DAXPY)
check_function_exists(daxpy_ HAVE_DAXPY_POST)
if(HAVE_DAXPY)
  set(BLAS_POST FALSE)
  set(BLAS_DEFS "")
elseif(HAVE_DAXPY_POST)
  set(BLAS_POST TRUE)
  set(BLAS_DEFS "-DBLAS_POST")
else()
  message(FATAL_ERROR "Could not determine BLAS format.")
endif()
#check_function_exists(dpotrf  HAVE_DPOTRF)
#check_function_exists(dpotrf_ HAVE_DPOTRF_POST)
#if(HAVE_DPOTRF)
#  set(LAPACK_POST FALSE)
#  set(LAPACK_DEFS "")
#elseif(HAVE_DPOTRF_POST)
#  set(LAPACK_POST TRUE)
#  set(LAPACK_DEFS "-DLAPACK_POST")
#else()
#  message(FATAL_ERROR "Could not determine LAPACK format.")
#endif()

# Look for MPI_Reduce_scatter_block (and MPI_Reduce_scatter as sanity check)
set(CMAKE_REQUIRED_FLAGS "${MPI_C_COMPILE_FLAGS} ${MPI_C_LINK_FLAGS}")
set(CMAKE_REQUIRED_INCLUDES ${MPI_C_INCLUDE_PATH})
set(CMAKE_REQUIRED_LIBRARIES ${MPI_C_LIBRARIES})
check_function_exists(MPI_Reduce_scatter       HAVE_MPI_REDUCE_SCATTER)
if(NOT HAVE_MPI_REDUCE_SCATTER)
  message(FATAL_ERROR "Could not find MPI_Reduce_scatter")
endif()
check_function_exists(MPI_Reduce_scatter_block HAVE_MPI_REDUCE_SCATTER_BLOCK)
check_function_exists(MPI_Iallgather HAVE_MPI3_NONBLOCKING_COLLECTIVES)
check_function_exists(MPIX_Iallgather HAVE_MPIX_NONBLOCKING_COLLECTIVES)
check_function_exists(MPI_Init_thread HAVE_MPI_INIT_THREAD)
check_function_exists(MPI_Query_thread HAVE_MPI_QUERY_THREAD)
check_function_exists(MPI_Comm_set_errhandler HAVE_MPI_COMM_SET_ERRHANDLER)
check_function_exists(MPI_Type_create_struct HAVE_MPI_TYPE_CREATE_STRUCT)
if(NOT HAVE_MPI_TYPE_CREATE_STRUCT)
  message(FATAL_ERROR "Could not find MPI_Type_create_struct")
endif()

# Check for C++11 support
#include(CheckCXXSourceCompiles)
#set(CMAKE_REQUIRED_FLAGS ${CXX_FLAGS})
#set(RANDOM_CODE
#    "#include <random>
#     int main()
#     {
#         std::random_device rd;
#         std::mt19937 mt(rd());
#         std::normal_distribution<double> dist(0,1);
#         const double x = dist(mt);
#     }")
#check_cxx_source_compiles("${RANDOM_CODE}" HAVE_NORMAL_DIST)
#if(NOT HAVE_NORMAL_DIST)
#  message(FATAL_ERROR "C++11 random number generation not found. You may want to add -std=c++11 to your build flags, e.g., via -D CXX_FLAGS=\"-std=c++11\"")
#endif()

# Check for MPI_IN_PLACE (essentially MPI2 support)
#include(CheckCSourceCompiles)
#set(MPI_IN_PLACE_CODE
#    "#include \"mpi.h\"
#     int main( int argc, char* argv[] )
#     {
#         MPI_Init( &argc, &argv );
#         float a;
#         MPI_Allreduce
#         ( MPI_IN_PLACE, &a, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD );
#         MPI_Finalize();
#         return 0;
#     }")
#set(MPI_LONG_LONG_CODE
#    "#include \"mpi.h\"
#     int main( int argc, char* argv[] )
#     {
#         MPI_Init( &argc, &argv );
#         MPI_Datatype lli = MPI_LONG_LONG_INT;
#         MPI_Datatype llu = MPI_UNSIGNED_LONG_LONG;
#         MPI_Finalize();
#         return 0;
#     }")
#check_c_source_compiles("${MPI_IN_PLACE_CODE}" HAVE_MPI_IN_PLACE)
#check_c_source_compiles("${MPI_LONG_LONG_CODE}" HAVE_MPI_LONG_LONG)

# Look for restrict support
include(CheckCXXSourceCompiles)
set(RESTRICT_CODE "int main() { int* RESTRICT a; return 0; }")
set(CMAKE_REQUIRED_DEFINITIONS "-DRESTRICT=__restrict__")
check_cxx_source_compiles("${RESTRICT_CODE}" HAVE___restrict__)
set(CMAKE_REQUIRED_DEFINITIONS "-DRESTRICT=__restrict")
check_cxx_source_compiles("${RESTRICT_CODE}" HAVE___restrict)
set(CMAKE_REQUIRED_DEFINITIONS "-DRESTRICT=restrict")
check_cxx_source_compiles("${RESTRICT_CODE}" HAVE_restrict)
if(HAVE___restrict__)
  set(RESTRICT "__restrict__")
  message(STATUS "Using __restrict__ keyword.")
elseif(HAVE___restrict)
  set(RESTRICT "__restrict")
  message(STATUS "Using __restrict keyword.")
elseif(HAVE_restrict)
  set(RESTRICT "restrict")
  message(STATUS "Using restrict keyword.")
else()
  set(RESTRICT "")
  message(STATUS "Could not find a restrict keyword.")
endif()

# Create the Rote configuration header
configure_file(${PROJECT_SOURCE_DIR}/cmake/config.h.cmake
               ${PROJECT_BINARY_DIR}/include/rote/config.h)
install(FILES ${PROJECT_BINARY_DIR}/include/rote/config.h
        DESTINATION include/rote)

# Create a file which can be included in Makefile's.
# This is meant to be analogous to PETSc's 'conf/petscvariables' file
set(MPI_CXX_INCLUDE_STRING)
foreach(INC_PATH ${MPI_CXX_INCLUDE_PATH})
  set(MPI_CXX_INCLUDE_STRING "${MPI_CXX_INCLUDE_STRING} -I${INC_PATH}")
endforeach()
set(MATH_LIBS_STRING)
foreach(LIB ${MATH_LIBS})
  set(MATH_LIBS_STRING "${MATH_LIBS_STRING} ${LIB}")
endforeach()
set(MPI_CXX_LIBS_STRING)
foreach(LIB ${MPI_CXX_LIBRARIES})
  set(MPI_CXX_LIBS "${MPI_CXX_LIBS} ${LIB}")
endforeach()
set(LAPACK_ADDONS_STRING)
foreach(LIB ${LAPACK_ADDONS_LIBS})
  set(LAPACK_ADDONS_STRING "${LAPACK_ADDONS_STRING} ${LIB}")
endforeach()
configure_file(${PROJECT_SOURCE_DIR}/cmake/rotevariables.cmake
               ${PROJECT_BINARY_DIR}/conf/rotevariables @ONLY)
install(FILES ${PROJECT_BINARY_DIR}/conf/rotevariables DESTINATION conf)

# Grab all of the .c, .cpp, .h, and .hpp Rote files
file(GLOB_RECURSE ROTE_CPP RELATIVE ${PROJECT_SOURCE_DIR} "src/*.c" "src/*.cpp")
file(GLOB_RECURSE ROTE_HEADERS RELATIVE ${PROJECT_SOURCE_DIR}
  "include/*.h" "include/*.hpp")
if(ROTE_EXPERIMENTAL)
  if(NOT MPI_C_FOUND)
    message(FATAL_ERROR "Could not build C interface without C MPI")
  endif()
  include_directories(${MPI_C_INCLUDE_PATH})
  set(CMAKE_C_FLAGS_${UPPER_BUILD_TYPE} "${MPI_C_COMPILE_FLAGS}")
  set(ROTE_CPP "${ROTE_CPP};experimental/c/interface.cpp")
  set(ROTE_HEADERS "${ROTE_HEADERS};experimental/c/elemental.h")

#  if(HAVE_F90_INTERFACE)
#    if(NOT MPI_Fortran_FOUND)
#      message(FATAL_ERROR "Could not build F90 interface without Fortran MPI")
#    endif()
    # We cannot simply use check_function_exists, since MPI_Comm_f2c is
    # a macro in some MPI implementations
#    set(MPI_COMM_F2C_CODE
#    "#include \"mpi.h\"
#     int main( int argc, char* argv[] )
#     {
#         MPI_Init( &argc, &argv );
#         MPI_Fint fint;
#         MPI_Comm comm = MPI_Comm_f2c( fint );
#         MPI_Finalize();
#         return 0;
#     }")
#    check_c_source_compiles("${MPI_COMM_F2C_CODE}" HAVE_MPI_COMM_F2C)
#    if(NOT HAVE_MPI_COMM_F2C)
#      message(FATAL_ERROR
#        "Could not find MPI_Comm_f2c, but it is needed for the F90 interface")
#    endif()
#    include_directories(${MPI_Fortran_INCLUDE_PATH})
#    set(CMAKE_Fortran_FLAGS_${UPPER_BUILD_TYPE} "${MPI_Fortran_COMPILE_FLAGS}")
#    set(ROTE_CPP "${ROTE_CPP};experimental/f90/interface.cpp")
#  endif()
endif()
set(ROTE_SRC "${ROTE_CPP};${ROTE_HEADERS};${ROTE_MOC_SRC}")

# The main library
set(LINK_LIBS ${MATH_LIBS} ${MPI_CXX_LIBRARIES})
add_library(rote ${LIBRARY_TYPE} ${ROTE_SRC})
target_link_libraries(rote ${LINK_LIBS})
if(MPI_CXX_LINK_FLAGS)
  set(MPI_LINK_FLAGS "${MPI_CXX_LINK_FLAGS}")
endif()
install(TARGETS rote DESTINATION lib)

# Define the header-file preparation rules
set(PREPARED_HEADERS)
foreach(HEADER ${ROTE_HEADERS})
  add_custom_command(OUTPUT ${PROJECT_BINARY_DIR}/${HEADER}
    COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_SOURCE_DIR}/${HEADER}
            ${PROJECT_BINARY_DIR}/${HEADER}
    DEPENDS "${PROJECT_SOURCE_DIR}/${HEADER}")
  list(APPEND PREPARED_HEADERS ${PROJECT_BINARY_DIR}/${HEADER})

  get_filename_component(HEADER_PATH ${HEADER} PATH)
  install(FILES ${PROJECT_BINARY_DIR}/${HEADER} DESTINATION ${HEADER_PATH})
endforeach()
add_custom_target(prepare_rote_headers DEPENDS ${PREPARED_HEADERS})
add_dependencies(rote prepare_rote_headers)

# Make sure the Rote headers can be found
include_directories("${PROJECT_BINARY_DIR}/include")

# Build the test drivers if necessary
if(ROTE_TESTS)
  set(TEST_DIR ${PROJECT_SOURCE_DIR}/tests)
#  set(TEST_TYPES core btas applications ccsd)
  set(TEST_TYPES core ccsd ccsd_gen gunnels)
  set(core_TESTS Views Grids GenContractTest )#RedistCheckTest GenRedistTest )
  set(gunnels_TESTS example)
#  set(btas_TESTS Level1 Contract)
  set(ccsd_TESTS ccsd ccsd_gen)#term_Tau term_W term_X term_U term_P term_Q term_H term_F term_G term_z_small term_Z ccsd)
#  set(ccsd_TESTS term_Tau term_H term_U)
  set(applications_TESTS BlockMP3DoublePartPerm MP3new MP3Newer)

  foreach(TYPE ${TEST_TYPES})
    set(OUTPUT_DIR "${PROJECT_BINARY_DIR}/bin/tests/${TYPE}")
    foreach(TEST ${${TYPE}_TESTS})
      add_executable(tests-${TYPE}-${TEST} ${TEST_DIR}/${TYPE}/${TEST}.cpp)
      target_link_libraries(tests-${TYPE}-${TEST} rote)
      set_target_properties(tests-${TYPE}-${TEST} PROPERTIES
        OUTPUT_NAME ${TEST} RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_DIR})
      if(MPI_LINK_FLAGS)
        set_target_properties(tests-${TYPE}-${TEST} PROPERTIES
          LINK_FLAGS ${MPI_LINK_FLAGS})
      endif()
      install(TARGETS tests-${TYPE}-${TEST} DESTINATION bin/tests/${TYPE})
    endforeach()
  endforeach()
endif()

# Build the example drivers if necessary
if(ROTE_EXAMPLES)
  set(EXAMPLE_DIR ${PROJECT_SOURCE_DIR}/examples)
  set(EXAMPLE_TYPES core compute tensors)

  set(core_EXAMPLES Constructors)
  set(compute_EXAMPLES Contract)
  set(tensors_EXAMPLES Ones Zeros)

  foreach(TYPE ${EXAMPLE_TYPES})
    set(OUTPUT_DIR "${PROJECT_BINARY_DIR}/bin/examples/${TYPE}")
    foreach(EXAMPLE ${${TYPE}_EXAMPLES})
      add_executable(examples-${TYPE}-${EXAMPLE}
        ${EXAMPLE_DIR}/${TYPE}/${EXAMPLE}.cpp)
      target_link_libraries(examples-${TYPE}-${EXAMPLE} rote)
      set_target_properties(examples-${TYPE}-${EXAMPLE} PROPERTIES
        OUTPUT_NAME ${EXAMPLE} RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_DIR})
      if(MPI_LINK_FLAGS)
        set_target_properties(examples-${TYPE}-${EXAMPLE} PROPERTIES
          LINK_FLAGS ${MPI_LINK_FLAGS})
      endif()
      install(TARGETS examples-${TYPE}-${EXAMPLE}
        DESTINATION bin/examples/${TYPE})
    endforeach()
  endforeach()
endif()

# Build experimental drivers that do not require an eigensolver
if(ROTE_EXPERIMENTAL)
  set(EXPERIMENTAL_DIR ${PROJECT_SOURCE_DIR}/experimental)
  #set(G3D_EXPERS G3DGemm)

  # Build the G3D example(s)
  #set(OUTPUT_DIR "${PROJECT_BINARY_DIR}/bin/experimental/g3d")
  #foreach(EXPER ${G3D_EXPERS})
  #  add_executable(experimental-g3d-${EXPER}
  #    ${EXPERIMENTAL_DIR}/g3d/${EXPER}.cpp)
  #  target_link_libraries(experimental-g3d-${EXPER} elemental)
  #  set_target_properties(experimental-g3d-${EXPER} PROPERTIES
  #    OUTPUT_NAME ${EXPER} RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_DIR})
  #  if(MPI_LINK_FLAGS)
  #    set_target_properties(experimental-g3d-${EXPER} PROPERTIES
  #      LINK_FLAGS ${MPI_LINK_FLAGS})
  #  endif()
  #  install(TARGETS experimental-g3d-${EXPER} DESTINATION bin/experimental/g3d)
  #endforeach()
endif()

# If Rote is a subproject, then pass some variables to the parent
if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  include(./cmake/rote_sub.cmake)
endif()
