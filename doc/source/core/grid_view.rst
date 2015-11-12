The GridView class
==============

This class is responsible for representing the logical process grid based on the distribution used to distribute a tensor onto a process grid.

.. cpp:type:: class GridView

   .. cpp:function:: GridView( const rote::Grid* g, const TensorDistribution& dist)

      Construct a view of the grid `g` under the distribution `dist`.

   .. rubric:: Simple interface (simpler version of distribution-based interface)

   .. cpp:function:: int ParticipatingOrder() const

      Return the order of the process grid view considering only modes participating in communication.

   .. cpp:function:: ObjShape ParticipatingShape() const

      Return the shape of the process grid view considering only modes participating in communication.

   .. cpp:function:: ObjShape Dimension(Mode mode) const

      Return the dimension of mode-`mode` of the process grid view.

   .. cpp:function:: Location ParticipatingLoc() const

      Return the location of this process in the grid view considering only modes participating in communication.

   .. cpp:function:: Location Loc() const

      Return the location of this process in the grid view considering all modes (even those not participating in communication).

   .. cpp:function:: Unsigned ModeLoc(Mode mode) const

      Return the location of this process in the grid view.

   .. cpp:function:: Location GridLoc() const

      Return the location of this process in the process grid.

   .. cpp:function:: Location Loc() const

      Return the location of this process in the grid view considering all modes (even those not participating in communication).

   .. cpp:function:: TensorDistribution TensorDist() const

      Return the tensor distribution used to create this grid view.

   .. cpp:function:: const rote::Grid* Grid() const

      Return the underlying Grid object.



   .. rubric:: Distribution-based interface

   .. cpp:function:: ModeArray BoundModes() const

      Return set of modes used to distribute modes of the tensor.

   .. cpp:function:: ModeArray FreeModes() const

      Return set of modes which have not been bound to a mode of the tensor for distribution.

   .. cpp:function:: ModeArray UnusedModes() const

      Return set of modes which are not participating in communication.

   .. cpp:function:: bool IsBound(Mode mode) const

      Return whether the specified mode is bound to some mode of the tensor for distribution.

   .. cpp:function:: bool IsFree(Mode mode) const

      Return whether the specified mode is unbound.

   .. cpp:function:: bool IsUnused(Mode mode) const

      Return whether the specified mode is not used in communciation.

   .. cpp:function:: Unsigned LinearRank() const

      Return this process's linear rank within a column-major ordering of processes in the grid view.

   .. cpp:function:: Participating() const;

      Return whether this process is participating in communications.

.. rubric:: GridView comparison functions

.. cpp:function:: bool operator==( const GridView& A, const GridView& B )

   Returns whether or not `A` and `B` are the same grid view.

.. cpp:function:: bool operator!=( const GridView& A, const GridView& B )

   Returns whether or not `A` and `B` are different grid views.
