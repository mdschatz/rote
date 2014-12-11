Partitioning
============

PartitionUp
-----------
Given an :math:`\tenDim{0} \times \cdots \times \tenDim{\tenOrder-1}` tensor `A`, configure `AT` and `AB` to view the local data of `A` corresponding to the partition

.. math::

   A = \left(\begin{array}{c}A_T \\ A_B \end{array}\right), 

where :math:`A_B` is of a specified dimension along a specified mode. 

.. cpp:function:: void PartitionUp( Tensor<T>& A, Tensor<T>& AT, Tensor<T>& AB, Mode mode, int heightAB=Blocksize() )
.. cpp:function:: void LockedPartitionUp( const Tensor<T>& A, Tensor<T>& AT, Tensor<T>& AB, Mode mode, int heightAB=Blocksize() )

.. cpp:function:: void PartitionUp( DistTensor<T>& A, DistTensor<T>& AT, DistTensor<T>& AB, Mode mode, int heightAB=Blocksize() )
.. cpp:function:: void LockedPartitionUp( const DistTensor<T>& A, DistTensor<T>& AT, DistTensor<T>& AB, Mode mode, int heightAB=Blocksize() )


PartitionDown
-------------
Given an :math:`\tenDim{0} \times \cdots \times \tenDim{\tenOrder-1}` tensor `A`, configure `AT` and `AB` to view the local data of `A` corresponding to the partition

.. math::

   A = \left(\begin{array}{c}A_T \\ A_B \end{array}\right),

where :math:`A_T` is of a specified dimension along a specified mode.

.. cpp:function:: void PartitionDown( Tensor<T>& A, Tensor<T>& AT, Tensor<T>& AB, Mode mode, int heightAT=Blocksize() )
.. cpp:function:: void LockedPartitionDown( const Tensor<T>& A, Tensor<T>& AT, Tensor<T>& AB, Mode mode, int heightAT=Blocksize() )

.. cpp:function:: void PartitionDown( DistTensor<T>& A, DistTensor<T>& AT, DistTensor<T>& AB, Mode mode, int heightAT=Blocksize() )
.. cpp:function:: void LockedPartitionDown( const DistTensor<T>& A, DistTensor<T>& AT, DistTensor<T>& AB, Mode mode, int heightAT=Blocksize() )

