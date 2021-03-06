# Set the serial GCC compilers
set(COMPILER_DIR /opt/apps/gcc_amd/4.4.5/bin)
set(CMAKE_C_COMPILER       ${COMPILER_DIR}/gcc)
set(CMAKE_CXX_COMPILER     ${COMPILER_DIR}/g++)
set(CMAKE_Fortran_COMPILER ${COMPILER_DIR}/gfortran)

# Set the MPI wrappers for the C and C++ compilers
set(MPI_COMPILER_DIR /opt/apps/gcc4_4/mvapich/1.0.1/bin)
set(MPI_C_COMPILER       ${MPI_COMPILER_DIR}/mpicc)
set(MPI_CXX_COMPILER     ${MPI_COMPILER_DIR}/mpicxx)
set(MPI_Fortran_COMPILER ${MPI_COMPILER_DIR}/mpif90)

if(CMAKE_BUILD_TYPE MATCHES PureDebug OR
   CMAKE_BUILD_TYPE MATCHES HybridDebug)
  set(CXX_FLAGS "-g -std=c++11")
else()
  set(CXX_FLAGS "-O3 -std=c++11")
endif()

set(OpenMP_CXX_FLAGS "-fopenmp")

set(MATH_LIBS 
    "-L/opt/apps/intel/mkl/10.0.1.014/lib/em64t -lmkl_em64t -lmkl -lguide -lpthread /opt/apps/gcc_amd/4.4.5/lib64/libgfortran.a -lm")

