Repartitioning
==============

RepartitionUp
-------------
Given the partition

.. math::

   A = \left(\begin{array}{c} A_T \\ \hline A_B \end{array}\right),

and a blocksize, :math:`n_b`, turn the two-way partition into the three-way
partition 

.. math::

   \left(\begin{array}{c} A_T \\ \hline A_B \end{array}\right) = 
   \left(\begin{array}{c} A_0 \\ A_1 \\ \hline A_2 \end{array}\right),

where :math:`A_1` is of height :math:`n_b` and :math:`A_2 = A_B` along a specified mode.

.. cpp:function:: void RepartitionUp( Tensor<T>& AT, Tensor<T>& A0, Tensor<T>& A1, Tensor<T>& AB, Tensor<T>& A2, Mode mode, int bsize=Blocksize() )

.. cpp:function:: void LockedRepartitionUp( const Tensor<T>& AT, Tensor<T>& A0, Tensor<T>& A1, const Tensor<T>& AB, Tensor<T>& A2, Mode mode, int bsize=Blocksize() )

.. cpp:function:: void RepartitionUp( DistTensor<T>& AT, DistTensor<T>& A0, DistTensor<T>& A1, DistTensor<T>& AB, DistTensor<T>& A2, Mode mode, int bsize=Blocksize() )

.. cpp:function:: void LockedRepartitionUp( const DistTensor<T>& AT, DistTensor<T>& A0, DistTensor<T>& A1, const DistTensor<T>& AB, DistTensor<T>& A2, Mode mode, int bsize=Blocksize() )

Note that each of the above routines is meant to be used in a manner similar 
to the following:

.. code-block:: cpp

   RepartitionUp( AT,  A0,
                       A1,
                 /**/ /**/
                  AB,  A2, mode, blocksize );

RepartitionDown
---------------
Given the partition

.. math::

   A = \left(\begin{array}{c} A_T \\ \hline A_B \end{array}\right),

and a blocksize, :math:`n_b`, turn the two-way partition into the three-way
partition 

.. math::

   \left(\begin{array}{c} A_T \\ \hline A_B \end{array}\right) = 
   \left(\begin{array}{c} A_0 \\ \hline A_1 \\ A_2 \end{array}\right),

where :math:`A_1` is of height :math:`n_b` and :math:`A_0 = A_T` along a specified mode.

.. cpp:function:: void RepartitionDown( Tensor<T>& AT, Tensor<T>& A0, Tensor<T>& A1, Tensor<T>& AB, Tensor<T>& A2, Mode mode, int bsize=Blocksize() )

.. cpp:function:: void LockedRepartitionDown( const Tensor<T>& AT, Tensor<T>& A0, Tensor<T>& A1, const Tensor<T>& AB, Tensor<T>& A2, Mode mode, int bsize=Blocksize() )

.. cpp:function:: void RepartitionDown( DistTensor<T>& AT, DistTensor<T>& A0, DistTensor<T>& A1, DistTensor<T>& AB, DistTensor<T>& A2, Mode mode, int bsize=Blocksize() )

.. cpp:function:: void LockedRepartitionDown( const DistTensor<T>& AT, DistTensor<T>& A0, DistTensor<T>& A1, const DistTensor<T>& AB, DistTensor<T>& A2, Mode mode, int bsize=Blocksize() )

   Templated over the datatype, `T`, and distribution scheme, `(U,V)`.

Note that each of the above routines is meant to be used in a manner similar 
to the following:

.. code-block:: cpp

   RepartitionDown( AT,  A0,
                   /**/ /**/
                         A1,
                    AB,  A2, mode, blocksize );
