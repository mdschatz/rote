Sliding partitions
==================

SlidePartitionUp
----------------
Simultaneously slide and merge the partition

.. math::

   A = \left(\begin{array}{c} A_0 \\ A_1 \\ \hline A_2 \end{array}\right),

into

.. math::

   \left(\begin{array}{c} A_T \\ \hline A_B \end{array}\right) = 
   \left(\begin{array}{c} A_0 \\ \hline A_1 \\ A_2 \end{array}\right).

along a specified mode.

.. cpp:function:: void SlidePartitionUp( Tensor<T>& AT, Tensor<T>& A0, Tensor<T>& A1, Tensor<T>& AB, Tensor<T>& A2, Mode mode )

.. cpp:function:: void SlideLockedPartitionUp( Tensor<T>& AT, const Tensor<T>& A0, const Tensor<T>& A1, Tensor<T>& AB, const Tensor<T>& A2, Mode mode )

.. cpp:function:: void SlidePartitionUp( DistTensor<T>& AT, DistTensor<T>& A0, DistTensor<T>& A1, DistTensor<T>& AB, DistTensor<T>& A2, Mode mode )

.. cpp:function:: void SlideLockedPartitionUp( DistTensor<T>& AT, const DistTensor<T>& A0, const DistTensor<T>& A1, DistTensor<T>& AB, const DistTensor<T>& A2, Mode mode )

Note that each of the above routines is meant to be used in a manner similar 
to the following:

.. code-block:: cpp

   SlidePartitionUp( AT,  A0,
                    /**/ /**/
                          A1,
                     AB,  A2, mode );

SlidePartitionDown
------------------
Simultaneously slide and merge the partition

.. math::

   A = \left(\begin{array}{c} A_0 \\ \hline A_1 \\ A_2 \end{array}\right),

into

.. math::

   \left(\begin{array}{c} A_T \\ \hline A_B \end{array}\right) = 
   \left(\begin{array}{c} A_0 \\ A_1 \\ \hline A_2 \end{array}\right).

along a specified mode.

.. cpp:function:: void SlidePartitionDown( Tensor<T>& AT, Tensor<T>& A0, Tensor<T>& A1, Tensor<T>& AB, Tensor<T>& A2, Mode mode )

.. cpp:function:: void SlideLockedPartitionDown( Tensor<T>& AT, const Tensor<T>& A0, const Tensor<T>& A1, Tensor<T>& AB, const Tensor<T>& A2, Mode mode )

.. cpp:function:: void SlidePartitionDown( DistTensor<T>& AT, DistTensor<T>& A0, DistTensor<T>& A1, DistTensor<T>& AB, DistTensor<T>& A2, Mode mode )

.. cpp:function:: void SlideLockedPartitionDown( DistTensor<T>& AT, const DistTensor<T>& A0, const DistTensor<T>& A1, DistTensor<T>& AB, const DistTensor<T>& A2, Mode mode )

Note that each of the above routines is meant to be used in a manner similar 
to the following:

.. code-block:: cpp

   SlidePartitionDown( AT,  A0,
                            A1,
                      /**/ /**/
                       AB,  A2, mode );
