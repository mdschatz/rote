Environment
===========

This section describes the routines and data structures which help set up 
|projectName|'s programming environment: it discusses initialization of |projectName|,
call stack manipulation, a custom data structure for complex data, many routines
for manipulating real and complex data, and a few 
useful routines for simplifying index calculations.

Build and version information
-----------------------------
FIX THIS THSI THIT THSI

Every |projectName| driver with proper command-line argument processing will run
`PrintVersion` if the ``--version`` argument is used. If ``--build`` is used,
then all of the below information is reported.

.. cpp:function:: void PrintVersion( std::ostream& os=std::cout )

   Prints the Git revision, (pre-)release version, and build type. 
   For example::

    Tensormental version information:
      Git revision: 3c6fbdaad901a554fc27a83378d63dab55af0dd3
      Version:      0.01-dev
      Build type:   PureDebug
   
.. cpp:function:: void PrintConfig( std::ostream& os=std::cout )

   Prints the relevant configuration details. For example::

    Tensormental configuration:
      Math libraries: /usr/lib/liblapack.so;/usr/lib/libblas.so
      HAVE_F90_INTERFACE
      HAVE_MPI_REDUCE_SCATTER_BLOCK
      HAVE_MPI_IN_PLACE
      USE_BYTE_ALLGATHERS

.. cpp:function:: void PrintCCompilerInfo( std::ostream& os=std::cout )

   Prints the relevant C compilation information. For example::

    Tensormental's C compiler info:
      CMAKE_C_COMPILER:    /usr/local/bin/gcc
      MPI_C_COMPILER:      /home/poulson/Install/bin/mpicc
      MPI_C_INCLUDE_PATH:  /home/poulson/Install/include
      MPI_C_COMPILE_FLAGS: 
      MPI_C_LINK_FLAGS:     -Wl,-rpath  -Wl,/home/poulson/Install/lib
      MPI_C_LIBRARIES:     /Users/schatz/openmpi/lib/libopenmpi.a

.. cpp:function:: void PrintCxxCompilerInfo( std::ostream& os=std::cout )

   Prints the relevant C++ compilation information. For example::

    Tensormental's C++ compiler info:
      CMAKE_CXX_COMPILER:    /usr/local/bin/g++
      CXX_FLAGS:             -Wall
      MPI_CXX_COMPILER:      /home/poulson/Install/bin/mpicxx
      MPI_CXX_INCLUDE_PATH:  /home/poulson/Install/include
      MPI_CXX_COMPILE_FLAGS: 
      MPI_CXX_LINK_FLAGS:     -Wl,-rpath  -Wl,/home/poulson/Install/lib
      MPI_CXX_LIBRARIES:     

Set up and clean up
-------------------

.. cpp:function:: void Initialize( int& argc, char**& argv )

   Initializes |projectName| and (if necessary) MPI. The usage is very similar to 
   ``MPI_Init``, but the `argc` and `argv` can be directly passed in.

   .. code-block:: cpp

      #include "tensormental.hpp"
      int main( int argc, char* argv[] )
      {
          tmen::Initialize( argc, argv );
          // ...
          tmen::Finalize();
          return 0;
      }

.. cpp:function:: void Finalize()

   Frees all resources allocated by |projectName| and (if necessary) MPI.

.. cpp:function:: bool Initialized()

   Returns whether or not |projectName| is currently initialized.

.. cpp:function:: void ReportException( std::exception& e )

   Used for handling |projectName|'s various exceptions, e.g.,

   .. code-block:: cpp

      #include "tensormental.hpp"
      int main( int argc, char* argv[] )
      {
          tmen::Initialize( argc, argv );
          try {
              // ...
          } catch( std::exception& e ) { ReportException(e); }
          tmen::Finalize();
          return 0;
      }

Blocksize manipulation
----------------------

.. cpp:function:: int Blocksize()

   Return the currently chosen algorithmic blocksize. The optimal value 
   depends on the problem size, algorithm, and architecture; the default value
   is 128.

.. cpp:function:: void SetBlocksize( int blocksize )

   Change the algorithmic blocksize to the specified value.

.. cpp:function:: void PushBlocksizeStack( int blocksize )

   It is frequently useful to temporarily change the algorithmic blocksize, so 
   rather than having to manually store and reset the current state, one can 
   simply push a new value onto a stack 
   (and later pop the stack to reset the value).

.. cpp:function:: void PopBlocksizeStack() 

   Pops the stack of blocksizes. See above.

Default process grid
--------------------

.. cpp:function:: Grid& DefaultGrid()

   Return a process grid built over :cpp:type:`mpi::COMM_WORLD`. This is 
   typically used as a means of allowing instances of the 
   :cpp:type:`DistTensor\<T>` class to be constructed without having to 
   manually specify a process grid, e.g., 

   .. code-block:: cpp

      // Build a 10 x 10 x 10 distributed tensor over mpi::COMM_WORLD
      ObjShape shape(3);
      shape[0] = 10;
      shape[1] = 10;
      shape[2] = 10;
      tmen::DistMatrix<T,MC,MR> A( shape );

Call stack manipulation
-----------------------

.. note::

   The following call stack manipulation routines are only available in 
   non-release builds (i.e., PureDebug and HybridDebug) and are meant to allow 
   for the call stack to be printed (via :cpp:func:`DumpCallStack`) when an 
   exception is caught.

.. cpp:function:: void PushCallStack( std::string s )

   Push the given routine name onto the call stack.

.. cpp:function:: void PopCallStack()

   Remove the routine name at the top of the call stack.

.. cpp:function:: void DumpCallStack()

   Print (and empty) the contents of the call stack.

Complex data
------------

.. cpp:type:: struct Base<F>

   .. cpp:type:: type

      The underlying real datatype of the (potentially complex) datatype `F`.
      For example, ``typename Base<std::complex<double> >::type`` and 
      ``typename Base<double>::type`` are both equivalent to ``double``.
      This is often extremely useful in implementing routines which are 
      templated over real and complex datatypes but still make use of real 
      datatypes.

.. cpp:function:: std::ostream& operator<<( std::ostream& os, Complex<R> alpha )

   Pretty prints `alpha` in the form ``a+bi``.

.. cpp:type:: scomplex

   ``typedef Complex<float> scomplex;``

.. cpp:type:: dcomplex

   ``typedef Complex<double> dcomplex;``

Scalar manipulation
-------------------

.. cpp:function:: typename Base<F>::type Abs( const F& alpha )

   Return the absolute value of the real or complex variable :math:`\alpha`.

.. cpp:function:: F FastAbs( const F& alpha )

   Return a cheaper norm of the real or complex :math:`\alpha`:

   .. math::
   
      |\alpha|_{\mbox{fast}} = |\mathcal{R}(\alpha)| + |\mathcal{I}(\alpha)|

.. cpp:function:: F RealPart( const F& alpha )
.. cpp:function:: F ImagPart( const F& alpha )

   Return the real (imaginary) part of the real or complex variable 
   :math:`\alpha`.

.. cpp:function:: void SetRealPart( F& alpha, typename Base<F>::type& beta )
.. cpp:function:: void SetImagPart( F& alpha, typename Base<F>::type& beta )

   Set the real (imaginary) part of the real or complex variable 
   :math:`\alpha` to :math:`\beta`. 
   If :math:`\alpha` has a real type, an error is thrown when an attempt is
   made to set the imaginary component.

.. cpp:function:: void UpdateRealPart( F& alpha, typename Base<F>::type& beta )
.. cpp:function:: void UpdateImagPart( F& alpha, typename Base<F>::type& beta )

   Update the real (imaginary) part of the real or complex variable 
   :math:`\alpha` to :math:`\beta`.
   If :math:`\alpha` has a real type, an error is thrown when an attempt is
   made to update the imaginary component.

.. cpp:function:: F Conj( const F& alpha )

   Return the complex conjugate of the real or complex variable :math:`\alpha`.

.. cpp:function:: F Sqrt( const F& alpha )

   Returns the square root or the real or complex variable :math:`\alpha`.

.. cpp:function:: F Cos( const F& alpha )

   Returns the cosine of the real or complex variable :math:`\alpha`.

.. cpp:function:: F Sin( const F& alpha )

   Returns the sine of the real or complex variable :math:`\alpha`.

.. cpp:function:: F Tan( const F& alpha )

   Returns the tangent of the real or complex variable :math:`\alpha`.

.. cpp:function:: F Cosh( const F& alpha )

   Returns the hyperbolic cosine of the real or complex variable :math:`\alpha`.

.. cpp:function:: F Sinh( const F& alpha )

   Returns the hyperbolic sine of the real or complex variable :math:`\alpha`.

.. cpp:function:: typename Base<F>::type Arg( const F& alpha )

   Returns the argument of the real or complex variable :math:`\alpha`.

.. cpp:function:: Complex<R> Polar( const R& r, const R& theta=0 )

   Returns the complex variable constructed from the polar coordinates
   :math:`(r,\theta)`.

.. cpp:function:: F Exp( const F& alpha )

   Returns the exponential of the real or complex variable :math:`\alpha`.

.. cpp:function:: F Pow( const F& alpha, const F& beta )

   Returns :math:`\alpha^\beta` for real or complex :math:`\alpha` and 
   :math:`\beta`.

.. cpp:function:: F Log( const F& alpha )

   Returns the logarithm of the real or complex variable :math:`\alpha`.

Other typedefs and enums
------------------------

.. cpp:type:: byte

   ``typedef unsigned char byte;``

Indexing utilities
------------------

.. cpp:function:: Int Shift( Int rank, Int firstRank, Int numProcs )

   Given a element-wise cyclic distribution over `numProcs` processes, 
   where the first entry is owned by the process with rank `firstRank`, 
   this routine returns the first entry owned by the process with rank
   `rank`.

.. cpp:function:: Int LocalLength( Int n, Int shift, Int numProcs )

   Given a vector with :math:`n` entries distributed over `numProcs` 
   processes with shift as defined above, this routine returns the number of 
   entries of the vector which are owned by this process.

.. cpp:function:: Int LocalLength( Int n, Int rank, Int firstRank, Int numProcs )

   Given a vector with :math:`n` entries distributed over `numProcs` 
   processes, with the first entry owned by process `firstRank`, this routine
   returns the number of entries locally owned by the process with rank 
   `rank`.
