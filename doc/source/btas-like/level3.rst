Level 3
=======

The prototypes for the following routines can be found at          
`include/rote/btas-like_decl.hpp <>`_, while the
implementations are in 
`include/rote/btas-like/level3/ <>`_.


Gemm
----
General matrix-matrix multiplication: updates
:math:`\M{C} := \alpha \M{A} \M{B} + \beta \M{C}`.

.. cpp:function:: void Gemm( T alpha, const Tensor<T>& A, const Tensor<T>& B, T beta, Tensor<T>& C )
.. cpp:function:: void Gemm( T alpha, const DistTensor<T>& A, const DistTensor<T>& B, T beta, DistTensor<T>& C )

