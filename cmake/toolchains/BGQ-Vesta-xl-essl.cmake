#set(CMAKE_SYSTEM_NAME BlueGeneQ-static)

set(IBMCMP_ROOT "$ENV{IBM_MAIN_DIR}")

set(ESSL_ROOT "/soft/libraries/essl/current/")
set(ESSL_LIB "${ESSL_ROOT}/lib64")
set(MASS_LIB "${IBMCMP_ROOT}/xlmass/bg/7.3/bglib64")
set(XLF_LIB "${IBMCMP_ROOT}/xlf/bg/14.1/bglib64")
set(XLSMP_LIB "${IBMCMP_ROOT}/xlsmp/bg/3.1/bglib64")

# V1R2M0
#set(MPI_ROOT  "/bgsys/drivers/ppcfloor/comm/gcc")
#set(PAMI_ROOT "/bgsys/drivers/ppcfloor/comm/sys")
# V1R2M1
set(MPI_ROOT  "/bgsys/drivers/ppcfloor/comm")
set(PAMI_ROOT "/bgsys/drivers/ppcfloor/comm")
set(SPI_ROOT  "/bgsys/drivers/ppcfloor/spi")

# The serial compilers
set(CMAKE_C_COMPILER       "mpixlcxx_r")
set(CMAKE_CXX_COMPILER     "mpixlcxx_r")
set(CMAKE_Fortran_COMPILER "mpixlcxx_r")

# The MPI wrappers for the C and C++ compilers
set(MPI_C_COMPILER   "mpixlcxx_r")
set(MPI_CXX_COMPILER "mpixlcxx_r")

set(MPI_C_COMPILE_FLAGS    "-qlanglvl=extended0x -qarch=qp -qtune=qp -qsimd=auto -qhot=level=1 -qprefetch -qunroll=yes -qrepor")
set(MPI_CXX_COMPILE_FLAGS  "-qlanglvl=extended0x -qarch=qp -qtune=qp -qsimd=auto -qhot=level=1 -qprefetch -qunroll=yes -qrepor")
set(MPI_C_INCLUDE_PATH     "${MPI_ROOT}/include")
set(MPI_CXX_INCLUDE_PATH   "${MPI_ROOT}/include")
set(MPI_C_LINK_FLAGS       "-L${MPI_ROOT}/lib -L${PAMI_ROOT}/lib -L${SPI_ROOT}/lib")
set(MPI_CXX_LINK_FLAGS     "${MPI_C_LINK_FLAGS}")

# V1R2M0
#set(MPI_C_LIBRARIES       "-lmpich -lopa -lmpl -ldl -lpami -lSPI -lSPI_cnk -lpthread -lrt -lstdc++")
#set(MPI_CXX_LIBRARIES     "-lcxxmpich ${MPI_C_LIBRARIES}")
# V1R2M1
set(MPI_C_LIBRARIES       "${MPI_C_LINK_FLAGS}   -lmpich-xl -lopa-xl -lmpl-xl -lpami-gcc -lSPI -lSPI_cnk -lrt -lpthread -lstdc++ -lpthread")
set(MPI_CXX_LIBRARIES     "${MPI_CXX_LINK_FLAGS} -lmpichcxx-xl ${MPI_C_LIBRARIES}")

if(CMAKE_BUILD_TYPE MATCHES PureDebug OR
   CMAKE_BUILD_TYPE MATCHES HybridDebug)
  set(CXX_FLAGS "${CXX_FLAGS} -g -DBGQ")
else()
  set(CXX_FLAGS "${CXX_FLAGS} -g -O2 -DBGQ -qlanglvl=extended0x")
endif()

set(CMAKE_THREAD_LIBS_INIT "-qsmp=omp")
set(OpenMP_CXX_FLAGS "-qsmp=omp")

##############################################################

# set the search path for the environment coming with the compiler
# and a directory where you can install your own compiled software
set(CMAKE_FIND_ROOT_PATH
    /bgsys/drivers/ppcfloor/
    ${MPI_ROOT}
    ${PAMI_ROOT}
    ${SPI_ROOT}
    ${IBMCMP_ROOT}
    ${ESSL_ROOT})

# adjust the default behaviour of the FIND_XXX() commands:
# search headers and libraries in the target environment, search
# programs in the host environment
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

##############################################################

set(XLF_FLAGS "-L${XLF_LIB} -lxlf90_r")
set(MASS_FLAGS "-L${MASS_LIB} -lmassv -lmass")
if(CMAKE_BUILD_TYPE MATCHES PureDebug OR
   CMAKE_BUILD_TYPE MATCHES PureRelease OR 
   NOT CMAKE_BUILD_TYPE)
  set(ESSL_FLAGS "-L${ESSL_LIB} -lesslbg")
  set(XL_FLAGS "-L${XLSMP_LIB} -lxlomp_ser")
else()
  set(ESSL_FLAGS "-L${ESSL_LIB} -lesslsmpbg")
  set(XL_FLAGS "-L${XLSMP_LIB} -lxlsmp")
endif()

#set(CMAKE_FIND_LIBRARY_SUFFIXES .a ${CMAKE_FIND_LIBRARY_SUFFIXES})
#set(CMAKE_EXE_LINKER_FLAGS "-static")
#set(MATH_LIBS "${LAPACK_FLAGS} ${ESSL_FLAGS} ${XLF_FLAGS} ${XL_FLAGS} -lxlopt -lxlfmath -lxl -lgfortran -lm -lpthread -ldl -Wl,--allow-multiple-definition")

# NOTE: It is apparently important that MATH_LIBS not begin with a full path
#       to a particular file, e.g., /path/to/libname.a, as CMake is 
#       prepending -L for some reason.
set(MATH_LIBS "${ESSL_FLAGS} ${MASS_FLAGS} ${XLF_FLAGS} ${XL_FLAGS} -lxlopt -lxlfmath -lxl -lpthread -ldl -Wl,--allow-multiple-definition")

