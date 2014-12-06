The Grid class
==============

This class is responsible for converting MPI communicators into an 
order-:math:`\gridOrder` process grid meant for distributing tensors (ala the 
soon-to-be-discussed :cpp:type:`DistTensor\<T>` class).

.. cpp:type:: class Grid

   .. cpp:function:: Grid( mpi::Comm comm, const ObjShape& shape )

      Construct a process grid of shape `shape` over the specified communicator
      If no communicator is specified, mpi::COMM_WORLD is used.

   .. rubric:: Simple interface (simpler version of distribution-based interface)

   .. cpp:function:: Unsigned Order() const

      Return the order of the process grid the processes are arranged as.

   .. cpp:function:: Unsigned ModeLoc(Mode mode) const

      Return the location within mode-`mode` of the process grid that this process lies in.

   .. cpp:function:: Location Loc() const

      Return the location of this process in the process grid.

   .. cpp:function:: Unsigned Dimension(Mode mode) const

      Return the dimension of mode-`mode` of the process grid.

   .. cpp:function:: ObjShape Shape() const

      Return the shape of the process grid.

   .. cpp:function:: Unsigned LinearRank() const

      Return our process's rank in the grid. The result is equivalent to this 
      process's index within a column-major ordering of process locations.

   .. cpp:function:: int Size() const

      Return the number of active processes in the process grid. This number 
      is equal to ``Height()`` :math:`\times` ``Width()``.

   .. rubric:: Advanced routines

   .. cpp:function:: bool InGrid() const

      Return whether or not our process is actively participating in the process
      grid.

   .. cpp:function:: mpi::Comm OwningComm() const

      Return the communicator for the set of processes actively participating
      in the grid. Note that this can only be valid if the calling process
      is an active member of the grid!

.. rubric:: Grid comparison functions

.. cpp:function:: bool operator==( const Grid& A, const Grid& B )

   Returns whether or not `A` and `B` are the same process grid.

.. cpp:function:: bool operator!=( const Grid& A, const Grid& B )

   Returns whether or not `A` and `B` are different process grids.
