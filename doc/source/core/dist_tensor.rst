The DistTensor class
====================
The :cpp:type:`DistTensor\<T>` class is meant to provide a 
distributed-memory analogue of the :cpp:type:`Tensor\<T>` class. 
Because |projectName| is designed to support order-arbitrary objects distributed 
on order-arbitrary grids, general redistributions must need to be explicitly
performed using the provided methods for redistribution. Each redistribution can 
redistribute according to a specific pattern detailed in the redistribution section.

Since it is crucial to know not only how many 
processes to distribute the data over, but *which* processes, and in what 
manner they should be decomposed into a logical process grid, an 
instance of the :cpp:type:`Grid` class must be passed into the constructor of 
the :cpp:type:`DistTensor\<T>` class along with the desired distribution.

For more information on the valid distributions, please refer to (proposal).

To facilitate arbitrary storage formats for local data, the :cpp:type:`DistTensor\<T>` 
class maintains meta-data about how the underlying local tensor is permuted when stored.

.. note:: 
   
   Since the :cpp:type:`DistTensor\<T>` class makes use of MPI for 
   message passing, custom interfaces must be written for nonstandard datatypes.
   As of now, the following datatypes are fully supported for 
   :cpp:type:`DistTensor\<T>`:
   ``int``, ``float``, ``double``, ``Complex<float>``, and ``Complex<double>``.

.. cpp:type:: class DistTensor<T>

   The most general case, where the underlying datatype `T` is only assumed 
   to be a ring; that is, it supports multiplication and addition and has the 
   appropriate identities.

   .. rubric:: Constructors

   .. cpp:function:: DistTensor( const Grid& grid=DefaultGrid() )
      
      Create an order-0 distributed tensor over the specified grid.

   .. cpp:function:: DistTensor( const Unsigned order, const Grid& grid=DefaultGrid() )
      
      Create an order-`order` distributed tensor over the specified grid.
      The permutation of the underlying local data is set to the identity permutation.

   .. cpp:function:: DistTensor( const TensorDistribution& dist, const Grid& grid=DefaultGrid() )
      
      Create a distributed tensor with the specified distribution over the specified grid.
      The order of the tensor is inferred from the distribution.
      The permutation of the underlying local data is set to the identity permutation.

   .. cpp:function:: DistTensor( const ObjShape& shape, const TensorDistribution& dist, const Grid& grid=DefaultGrid() )

      Create a distributed tensor of shape `shape` with the specified distribution over the specified grid.
      The order of the tensor is inferred from the distribution.
      The permutation of the underlying local data is set to the identity permutation.

   .. cpp:function:: DistTensor( const ObjShape& shape, const TensorDistribution& dist, const std::vector<Unsigned>& modeAligns, const Grid& grid )

      Create a distributed tensor of shape `shape` with the specified distribution over the specified grid but 
      with the first element (element whose location only contains zeros) owned by the process at location specified by `modeAligns`.
      The order of the tensor is inferred from the distribution.
      The permutation of the underlying local data is set to the identity permutation.

   .. cpp:function:: DistTensor( const ObjShape& shape, const TensorDistribution& dist, const std::vector<Unsigned>& modeAligns, const std::vector<Unsigned>& strides, const Grid& grid )

      Same as above, but the local strides are also specified.

   .. cpp:function:: DistTensor( const ObjShape& shape, const TensorDistribution& dist, const std::vector<Unsigned>& modeAligns, const T* buffer, const std::vector<Unsigned>& strides, const Grid& grid )

      View a constant distributed tensor's buffer; the buffer must correspond 
      to the local portion of a distributed tensor with the 
      specified row and column alignments and strides, `strides`.

   .. cpp:function:: DistTensor( const ObjShape& shape, const TensorDistribution& dist, const std::vector<Unsigned>& modeAligns, T* buffer, const std::vector<Unsigned>& strides, const Grid& grid )

      Same as above, but the contents of the tensor are modifiable.


   .. cpp:function:: DistTensor( const TensorDistribution& dist, const Grid& grid=DefaultGrid() )
   .. cpp:function:: DistTensor( const ObjShape& shape, const TensorDistribution& dist, const Grid& grid=DefaultGrid() )
   .. cpp:function:: DistTensor( const ObjShape& shape, const TensorDistribution& dist, const std::vector<Unsigned>& modeAligns, const Grid& grid )
   .. cpp:function:: DistTensor( const ObjShape& shape, const TensorDistribution& dist, const std::vector<Unsigned>& modeAligns, const std::vector<Unsigned>& strides, const Grid& grid )
   .. cpp:function:: DistTensor( const ObjShape& shape, const TensorDistribution& dist, const std::vector<Unsigned>& modeAligns, const T* buffer, const std::vector<Unsigned>& strides, const Grid& grid )
   .. cpp:function:: DistTensor( const ObjShape& shape, const TensorDistribution& dist, const std::vector<Unsigned>& modeAligns, T* buffer, const std::vector<Unsigned>& strides, const Grid& grid )

      Variants of the above constructors which take strings representing the tensor distributions instead of the internal :cpp:type:`TensorDistribution` type.

   .. cpp:function:: DistTensor( const DistTensor<T>& A )

      Build a copy of the distributed tensor `A`, but force it to be in the distribution specified by `A`.

   .. rubric:: Basic information

   .. cpp:function:: Unsigned Order() const

      Return the order of the tensor.

   .. cpp:function:: Unsigned Dimension(Mode mode) const

      Return the dimension of the specified mode of the tensor.

   .. cpp:function:: ObjShape Shape() const

      Return the shape of the tensor.

   .. cpp:function:: Unsigned Dimension() const

      Return the local dimension of the specified mode of the tensor.
      This routine respects the permutation applied to the local tensor.

   .. cpp:function:: ObjShape LocalShape() const

      Return the shape of the local tensor.

   .. cpp:function:: ObjShape MaxLocalShape() const

      Return the maximum shape of the local tensor any process in the process grid can store.

   .. cpp:function:: Unsigned LocalModeStride(Mode mode) const

      Return the stride of the specified mode of the local tensor.
      This routine respects the permutation applied to the local tensor.

   .. cpp:function:: Unsigned LocalStrides() const

      Return the strides of the local tensor.

   .. cpp:function:: size_t AllocatedMemory() const

      Return the number of entries of type `T` that we have locally allocated
      space for.

   .. cpp:function:: const rote::Grid& Grid() const

      Return the grid that this distributed tensor is distributed over.

   .. cpp:function:: const rote::GridView GetGridView() const

      Return the logical grid (GridView) that this distributed tensor is distributed over.

   .. cpp:function:: T* Buffer()

      Return a pointer to the portion of the local buffer that stores the first element (location only contains zeros).

   .. cpp:function:: const T* LockedBuffer() const

      Return a pointer to the portion of the local buffer that stores 
      the first element (location only contains zeros), but do not allow for the data to be modified through
      the returned pointer.


   .. cpp:function:: T* Buffer( const Location& loc )

      Return a pointer to the portion of the local buffer that stores entry 
      at location `loc`.

   .. cpp:function:: const T* LockedBuffer( const Location& loc ) const

      Return a pointer to the portion of the local buffer that stores entry
      at location `loc`, but do not allow for the data to be modified through
      the returned pointer.

   .. cpp:function:: Tensor<T>& Tensor()

      Return a reference to the local tensor.

   .. cpp:function:: const Tensor<T>& LockedTensor() const

      Return an unmodifiable reference to the local tensor.

   .. rubric:: Distribution details

   .. cpp:function:: void FreeAlignments()

      Free all alignment constaints.

   .. cpp:function:: bool ConstrainedModeAlignment(Mode mode) const

      Return whether or not the specified mode alignment is constrained.

   .. cpp:function:: Unsigned ModeAlignment() const

      Return the alignment of the specified mode of the tensor.

   .. cpp:function:: std::vector<Unsigned> Alignments() const

      Return the alignments of the tensor.

   .. cpp:function:: Unsigned ModeShift(Mode mode) const

      Return the first global location in the specified mode that our process owns.

   .. cpp:function:: std::vector<Unsigned> ModeShifts() const

      Return the first global location that our process owns.

   .. cpp:function:: Unsigned ModeStride(Mode mode) const

      Return the number of locations in the specified mode between locally owned entries.

   .. cpp:function:: std::vector<Unsigned> ModeStrides() const

      Return the number of locations between locally owned entries.



   .. rubric:: Entry manipulation

   .. cpp:function:: T Get( const Location& loc ) const

      Return the entry at location `loc` of the global tensor. This operation is 
      collective.  This routine respects the permutation applied to the local tensor.

   .. cpp:function:: void Set( const Location& loc, T alpha )

      Set the entry at location `loc` of the global tensor to :math:`\alpha`. This 
      operation is collective.  This routine respects the permutation applied to the local tensor.

   .. cpp:function:: void Update( const Location& loc, T alpha )

      Add :math:`\alpha` to the entry at location `loc` of the global tensor. This 
      operation is collective.  This routine respects the permutation applied to the local tensor.

   .. note::
      Check if the following routines should respect permutation or not

   .. cpp:function:: T GetLocal( const Location& loc ) const

      Return the entry at location `loc` of our local tensor.

   .. cpp:function:: void SetLocal( const Location& loc, T alpha )

      Set the entry at location `loc` of our local tensor to :math:`\alpha`.

   .. cpp:function:: void UpdateLocal( const Location& loc, T alpha )

      Add :math:`\alpha` to the entry at location `loc` of our local tensor.

   .. note::

      Many of the following routines are only valid for complex datatypes.

   .. cpp:function:: typename Base<T>::type GetRealPart( const Location& loc ) const
   .. cpp:function:: typename Base<T>::type GetImagPart( const Location& loc ) const

      Return the real (imaginary) part of the entry at location `loc` of the global 
      tensor. This operation is collective.  This routine respects the permutation applied to the local tensor.

   .. cpp:function:: void SetRealPart( const Location& loc, typename Base<T>::type alpha )
   .. cpp:function:: void SetImagPart( const Location& loc, typename Base<T>::type alpha )

      Set the real (imaginary) part of the entry at location `loc` of the global tensor to
      :math:`\alpha`.  This routine respects the permutation applied to the local tensor.

   .. cpp:function:: void UpdateRealPart( const Location& loc, typename Base<T>::type alpha )
   .. cpp:function:: void UpdateImagPart( const Location& loc, typename Base<T>::type alpha )

      Add :math:`\alpha` to the real (imaginary) part of the entry at location `loc` of 
      the global tensor.  This routine respects the permutation applied to the local tensor.

   .. cpp:function:: typename Base<T>::type GetLocalRealPart( const Location& loc ) const
   .. cpp:function:: typename Base<T>::type GetLocalImagPart( const Location& loc ) const

      Return the real (imaginary) part of the entry at location `loc` of our 
      local tensor.

   .. cpp:function:: void SetLocalRealPart( const Location& loc, typename Base<T>::type alpha )
   .. cpp:function:: void SetLocalImagPart( const Location& loc, typename Base<T>::type alpha )

      Set the real (imaginary) part of the entry at location `loc` of our local 
      tensor.

   .. cpp:function:: void UpdateRealPartLocal( const Location& loc, typename Base<T>::type alpha )
   .. cpp:function:: void UpdateLocalImagPart( const Location& loc, typename Base<T>::type alpha )

      Add :math:`\alpha` to the real (imaginary) part of the  
      entry at location `loc` of our local tensor.

   .. rubric:: Viewing

   .. cpp:function:: bool Viewing() const

      Return whether or not this tensor is viewing another.

   .. cpp:function:: bool Locked() const

      Return whether or not this tensor is viewing another in a manner
      that does not allow for modifying the viewed data.

   .. rubric:: Utilities

   .. cpp:function:: void Empty()

      Resize the distributed tensor so that it is :math:`0 \times 0` and free 
      all allocated storage.

   .. cpp:function:: void ResizeTo( int height, int width )

      Reconfigure the tensor so that it is `height` :math:`\times` `width`.

   .. cpp:function:: void ResizeTo( int height, int width, int ldim )

      Same as above, but the local leading dimension is also specified.

   .. cpp:function:: void SetGrid( const Grid& grid )

      Clear the distributed tensor's contents and reconfigure for the new 
      process grid.

   .. cpp:function:: Location DetermineOwner(const Location& loc) const

      Return the location of the process in the process grid owning the entry at location `loc`.

   .. cpp:function:: Location DetermineFirstElem(const Location& gridLoc) const

      Return the location of first entry owned by the process at location `gridLoc` in the process grid.

   .. cpp:function:: Location Global2LocalIndex(const Location& loc) const

      Return the location in our tensor corresponding to the global entry at location `loc`.  This 
      routine respects the permutation applied to the local tensor.



   .. rubric:: Alignment

   All of the following clear the distributed tensor's contents and then 
   reconfigure the alignments as described.

   .. cpp:function:: void AlignWith( const DistTensor<T>& A )

      Force the alignments to match those of `A`.

   .. cpp:function:: void AlignModeWith( Mode mode, const DistTensor<T>& A )

      Force the alignment of mode-`mode` to match that of mode-`mode` of `A`.

   .. cpp:function:: void AlignModeWith( Mode mode, const DistTensor<T>& A, Mode modeA )

      Force the alignment of mode-`mode` to match that of mode-`modeA` of `A`.

   .. rubric:: Views

   .. cpp:function:: void Attach( const ObjShape& shape, const std::vector<Unsigned>& modeAligns, T* buffer, const std::vector<Unsigned>& strides, const Grid& grid )

      Reconfigure this distributed tensor to a distributed tensor of the same distribution with the 
      specified dimensions, alignments, local buffer, local strides, and process grid.

   .. cpp:function:: void LockedAttach( const ObjShape& shape, const std::vector<Unsigned>& modeAligns, const T* buffer, const std::vector<Unsigned>& strides, const Grid& grid )

      Same as above, but the resulting tensor is "locked", meaning that it 
      cannot modify the underlying local data.
