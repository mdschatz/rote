#set(CMAKE_SYSTEM_NAME BlueGeneQ-static)

# The serial compilers are typically set here, but, at least at some point,
# it was easier to set them to the MPI compilers.
set(MPI_ROOT  "/bgsys/drivers/ppcfloor/comm/gcc")
set(CMAKE_C_COMPILER       ${MPI_ROOT}/bin/mpicc)
set(CMAKE_CXX_COMPILER     ${MPI_ROOT}/bin/mpicxx)
set(CMAKE_Fortran_COMPILER ${MPI_ROOT}/bin/mpif77)

# The MPI wrappers for the C and C++ compilers
set(MPI_C_COMPILER   ${MPI_ROOT}/bin/mpicc)
set(MPI_CXX_COMPILER ${MPI_ROOT}/bin/mpicxx)

set(PAMI_ROOT "/bgsys/drivers/ppcfloor/comm/sys")
set(SPI_ROOT  "/bgsys/drivers/ppcfloor/spi")
set(MPI_C_COMPILE_FLAGS   "")
set(MPI_CXX_COMPILE_FLAGS "")
set(MPI_C_INCLUDE_PATH   "${MPI_ROOT}/include")
set(MPI_CXX_INCLUDE_PATH "${MPI_ROOT}/include")
set(MPI_C_LINK_FLAGS   "-L${MPI_ROOT}/lib -L${PAMI_ROOT}/lib -L${SPI_ROOT}/lib")
set(MPI_CXX_LINK_FLAGS ${MPI_C_LINK_FLAGS})
set(MPI_C_LIBRARIES "-lmpich -lopa -lmpl -ldl -lpami -lSPI -lSPI_cnk -lpthread -lrt -lstdc++")
set(MPI_CXX_LIBRARIES "-lcxxmpich ${MPI_C_LIBRARIES}")

if(CMAKE_BUILD_TYPE MATCHES PureDebug OR
   CMAKE_BUILD_TYPE MATCHES HybridDebug)
  set(CXX_FLAGS "-g -std=c++11")
else()
  set(CXX_FLAGS "-g -O3 -std=c++11")
endif()

set(CMAKE_THREAD_LIBS_INIT "-fopenmp")
set(OpenMP_CXX_FLAGS "-fopenmp")

##############################################################

# set the search path for the environment coming with the compiler
# and a directory where you can install your own compiled software
set(CMAKE_FIND_ROOT_PATH
    /bgsys/drivers/ppcfloor/
    /bgsys/drivers/ppcfloor/gnu-linux/powerpc64-bgq-linux
    /bgsys/drivers/ppcfloor/comm/gcc
    /bgsys/drivers/ppcfloor/comm/sys/
    /bgsys/drivers/ppcfloor/spi/)

# adjust the default behaviour of the FIND_XXX() commands:
# search headers and libraries in the target environment, search
# programs in the host environment
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

##############################################################

#set(CMAKE_FIND_LIBRARY_SUFFIXES .a ${CMAKE_FIND_LIBRARY_SUFFIXES})
#set(CMAKE_EXE_LINKER_FLAGS "-static")
set(MATH_LIBS "-L/soft/libraries/alcf/current/gcc/LAPACK/lib -llapack -L/soft/libraries/alcf/current/gcc/BLAS/lib -lblas -lgfortran -lm")
