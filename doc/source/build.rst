Build system
************
|projectName|'s build system relies on `CMake <http://www.cmake.org>`__ 
in order to manage a large number of configuration options in a 
platform-independent manner; it can be easily configured to build on Linux and 
Unix environments (including Darwin) as well as various versions of 
Microsoft Windows.

|projectName|'s main dependencies are

1. `CMake <http://www.cmake.org/>`__ (required)
2. `MPI <http://en.wikipedia.org/wiki/Message_Passing_Interface>`_ (required) 
3. `BLAS <http://netlib.org/blas>`__ and `LAPACK <http://netlib.org/lapack>`__ (required)

Each of these dependencies is discussed in detail below.

Dependencies
============

CMake
-----
|projectName| uses several new CMake modules, so it is important to ensure that 
version 2.8.5 or later is installed. Thankfully the 
`installation process <http://www.cmake.org/cmake/help/install.html>`_
is extremely straightforward: either download a platform-specific binary from
the `downloads page <http://www.cmake.org/cmake/resources/software.html>`_,
or instead grab the most recent stable tarball and have CMake bootstrap itself.
In the simplest case, the bootstrap process is as simple as running the 
following commands::

    ./bootstrap
    make
    make install

Note that recent versions of `Ubuntu <http://www.ubuntu.com/>`__ (e.g., version 12.04) have sufficiently up-to-date
versions of CMake, and so the following command is sufficient for installation::

    sudo apt-get install cmake

If you do install from source, there are two important issues to consider

1. By default, ``make install`` attempts a system-wide installation 
   (e.g., into ``/usr/bin``) and will likely require administrative privileges.
   A different installation folder may be specified with the ``--prefix`` 
   option to the ``bootstrap`` script, e.g.,::

    ./bootstrap --prefix=/home/your_username
    make
    make install

   Afterwards, it is a good idea to make sure that the environment variable 
   ``PATH`` includes the ``bin`` subdirectory of the installation folder, e.g.,
   ``/home/your_username/bin``.

2. Some highly optimizing compilers will not correctly build CMake, but the GNU
   compilers nearly always work. You can specify which compilers to use by
   setting the environment variables ``CC`` and ``CXX`` to the full paths to 
   your preferred C and C++ compilers before running the ``bootstrap`` script.

Basic usage
^^^^^^^^^^^
Though many configuration utilities, like 
`autoconf <http://www.gnu.org/software/autoconf/>`_, are designed such that
the user need only invoke ``./configure && make && make install`` from the
top-level source directory, CMake targets *out-of-source* builds, which is 
to say that the build process occurs away from the source code. The 
out-of-source build approach is ideal for projects that offer several 
different build modes, as each version of the project can be built in a 
separate folder.

A common approach is to create a folder named ``build`` in the top-level of 
the source directory and to invoke CMake from within it::

    mkdir build
    cd build
    cmake ..

The last line calls the command line version of CMake, ``cmake``,
and tells it that it should look in the parent directory for the configuration
instructions, which should be in a file named ``CMakeLists.txt``. Users that 
would prefer a graphical interface from the terminal (through ``curses``)
should instead use ``ccmake`` (on Unix platforms) or ``CMakeSetup`` 
(on Windows platforms). In addition, a GUI version is available through 
``cmake-gui``. 

Though running ``make clean`` will remove all files generated from running 
``make``, it will not remove configuration files. Thus, the best approach for
completely cleaning a build is to remove the entire build folder. On \*nix 
machines, this is most easily accomplished with::

    cd .. 
    rm -rf build

This is a better habit than simply running ``rm -rf *`` since, 
if accidentally run from the wrong directory, the former will most likely fail.

MPI
---
An implementation of the Message Passing Interface (MPI) is required for 
building |projectName|. The two most commonly used implementations are

1. `MPICH2 <http://www.mcs.anl.gov/research/projects/mpich2/>`_
2. `OpenMPI <http://www.open-mpi.org/>`_

If your cluster uses `InfiniBand <http://en.wikipedia.org/wiki/InfiniBand>`_ as its interconnect, you may want to look into 
`MVAPICH2 <http://mvapich.cse.ohio-state.edu/overview/mvapich2/>`_.

Each of the respective websites contains installation instructions, but, on recent versions of `Ubuntu <http://www.ubuntu.com/>`__ (such as version 12.04), 
MPICH2 can be installed with ::

    sudo apt-get install libmpich2-dev

and OpenMPI can be installed with ::

    sudo apt-get install libopenmpi-dev

BLAS
---------------
The Basic Linear Algebra Subprograms (BLAS) is used within |projectName|. 
On most installations of `Ubuntu <http://www.ubuntu.com>`__, the following command should suffice for their installation::

    sudo apt-get install libblas-dev

The reference implementation of BLAS can be found at

    http://www.netlib.org/blas/

However, it is better to install an optimized version of these libraries,
especialy for the BLAS. The most commonly used open source versions are 
`ATLAS <http://math-atlas.sourceforge.net/>`__ and `OpenBLAS <https://github.com/xianyi/OpenBLAS>`__.

Getting |projectName|'s source 
==========================
There are two basic approaches:

1. Download a tarball of the most recent version from 
   `>`_. 

2. Install `git <http://git-scm.com/>`_ and check out a copy of 
   the repository by running ::

    git clone 

Building |projectName|
==================
On \*nix machines with `BLAS <http://www.netlib.org/blas/>`__, and
`MPI <http://en.wikipedia.org/wiki/Message_Passing_Interface>`__ installed in 
standard locations, building |projectName| can be as simple as::

    cd tensormental
    mkdir build
    cd build
    cmake ..
    make
    make install

As with the installation of CMake, the default install location is 
system-wide, e.g., ``/usr/local``. The installation directory can be changed
at any time by running::

    cmake -D CMAKE_INSTALL_PREFIX=/your/desired/install/path ..
    make install


Though the above instructions will work on many systems, it is common to need
to manually specify several build options, especially when multiple versions of
libraries or several different compilers are available on your system. For 
instance, any C++, C, or Fortran compiler can respectively be set with the 
``CMAKE_CXX_COMPILER``, ``CMAKE_C_COMPILER``, and ``CMAKE_Fortran_COMPILER`` 
variables, e.g., ::

    cmake -D CMAKE_CXX_COMPILER=/usr/bin/g++ \
          -D CMAKE_C_COMPILER=/usr/bin/gcc   \
          -D CMAKE_Fortran_COMPILER=/usr/bin/gfortran ..
    
It is also common to need to specify which libraries need to be linked in order
to provide serial BLAS routines.
The ``MATH_LIBS`` variable was introduced for this purpose and an example 
usage for configuring with BLAS and LAPACK libraries in ``/usr/lib`` would be ::

    cmake -D MATH_LIBS="-L/usr/lib -lblas -lm" ..

It is important to ensure that if library A depends upon library B, A should 
be specified to the left of B.

If `libFLAME <http://www.cs.utexas.edu/users/flame/>`__ is 
available at ``/path/to/libflame.a``, then the above link line should be changed
to ::

    cmake -D MATH_LIBS="/path/to/libflame.a;-L/usr/lib -lblas -lm" ..

Build Modes
-----------
|projectName| currently has four different build modes:

* **PureDebug** - An MPI-only build that maintains a call stack and provides 
  more error checking.
* **PureRelease** - An optimized MPI-only build suitable for production use.
* **HybridDebug** - An MPI+OpenMP build that maintains a call stack and provides
  more error checking.
* **HybridRelease** - An optimized MPI+OpenMP build suitable for production use.

The build mode can be specified with the ``CMAKE_BUILD_TYPE`` option, e.g., 
``-D CMAKE_BUILD_TYPE=PureDebug``. If this option is not specified, |projectName|
defaults to the **PureRelease** build mode.

Testing the installation
========================
Once |projectName| has been installed, it is a good idea to verify that it is 
functioning properly. 

|projectName| as a subproject
=========================
Adding |projectName| as a dependency into a project which uses CMake for its build 
system is relatively straightforward: simply put an entire copy of the 
Elemental source tree in a subdirectory of your main project folder, say 
``external/elemental``, and then create a ``CMakeLists.txt`` file in your main 
project folder that builds off of the following snippet::

    cmake_minimum_required(VERSION 2.8.5) 
    project(Foo)

    add_subdirectory(external/elemental)
    include_directories("${PROJECT_BINARY_DIR}/external/tensormental/include")
    include_directories(${MPI_CXX_INCLUDE_PATH})

    # Build your project here
    # e.g., 
    #   add_library(foo ${LIBRARY_TYPE} ${FOO_SRC})
    #   target_link_libraries(foo tensormental)

Troubleshooting
===============
If you run into build problems, please email 
`martin.schatz@utexas.edu <mailto:martin.schatz@utexas.edu>`_ 
and make sure to attach the file ``include/tensormental/config.h``, which should 
be generated within your build directory. 
Please only direct usage questions to 
`martin.schatz@utexas.edu <mailto:martin.schatz@utexas.edu>`_, 
and development questions to 
`martin.schatz@utexas.edu <mailto:martin.schatz@utexas.edu>`_.
