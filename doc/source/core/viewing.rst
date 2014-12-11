Viewing
=======

View a full tensor
------------------

.. cpp:function:: void View( Tensor<T>& A, Tensor<T>& B )
.. cpp:function:: void View( DistTensor<T>& A, DistTensor<T>& B )

   Make `A` a view of the tensor `B`.

.. cpp:function:: Tensor<T> View( Tensor<T>& B )
.. cpp:function:: DistTensor<T> View( DistTensor<T>& B )

   Return a view of the tensor `B`.

.. cpp:function:: void LockedView( Tensor<T>& A, const Tensor<T>& B )
.. cpp:function:: void LockedView( DistTensor<T>& A, const DistTensor<T>& B )

   Make `A` a non-mutable view of the tensor `B`.

.. cpp:function:: Tensor<T> LockedView( const Tensor<T>& B )
.. cpp:function:: DistTensor<T> LockedView( const DistTensor<T>& B )

   Return a view of the tensor `B`.

View a subtensor
----------------

.. cpp:function:: void View( Tensor<T>& A, Tensor<T>& B, const Location& loc, const ObjShape& shape )
.. cpp:function:: void View( DistTensor<T>& A, DistTensor<T>& B, const Location& loc, const ObjShape& shape )

   Make `A` a view of the subtensor of `B` starting at the
   coordinate specified by `loc` with shape specified by `shape`.

.. cpp:function:: Tensor<T> View( Tensor<T>& B, const Location& loc, const ObjShape& shape )
.. cpp:function:: DistTensor<T> View( DistTensor<T>& B, const Location& loc, const ObjShape& shape )

   Return a view of the specified subtensor of `B`.

.. cpp:function:: void LockedView( Tensor<T>& A, const Tensor<T>& B, const Location& loc, const ObjShape& shape )
.. cpp:function:: void LockedView( DistTensor<T>& A, const DistTensor<T>& B, const Location& loc, const ObjShape& shape )

   Make `A` a non-mutable view of the subtensor of `B` starting at the
   coordinate specified by `loc` with shape specified by `shape`.

.. cpp:function:: Tensor<T> LockedView( const Tensor<T>& B, const Location& loc, const ObjShape& shape )
.. cpp:function:: DistTensor<T> LockedView( const DistTensor<T>& B, const Location& loc, const ObjShape& shape )

   Return an immutable view of the specified subtensor of `B`.

View 2x1 tensors
-----------------

.. cpp:function:: void View2x1( Tensor<T>& A, Tensor<T>& BT, Tensor<T>& BB, Mode mode )
.. cpp:function:: void View2x1( DistTensor<T>& A, DistTensor<T>& BT, DistTensor<T>& BB, Mode mode )

   Make `A` a view of the tensor 
   :math:`\left(\begin{array}{c} B_T \\ B_B \end{array}\right)` partitioned along mode `mode`.

.. cpp:function:: Tensor<T> View2x1( Tensor<T>& BT, Tensor<T>& BB, Mode mode )
.. cpp:function:: DistTensor<T> View2x1( DistTensor<T>& BT, DistTensor<T>& BB, Mode mode )

   Return a view of the merged tensor.

.. cpp:function:: void LockedView2x1( Tensor<T>& A, const Tensor<T>& BT, const Tensor<T>& BB, Mode mode )
.. cpp:function:: void LockedView2x1( DistTensor<T>& A, const DistTensor<T>& BT, const DistTensor<T>& BB, Mode mode )

   Make `A` a non-mutable view of the tensor 
   :math:`\left(\begin{array}{c} B_T \\ B_B \end{array}\right)` partitioned along mode `mode`.

.. cpp:function:: Tensor<T> LockedView2x1( const Tensor<T>& BT, const Tensor<T>& BB, Mode mode )
.. cpp:function:: DistTensor<T> LockedView2x1( const DistTensor<T>& BT, const DistTensor<T>& BB, Mode mode )

   Return a view of the merged tensor.
