Level 1
=======

The prototypes for the following routines can be found at 
`include/tensormental/btas-like_decl.hpp`, while the
implementations are in `include/tensormental/btas-like/level1/`.

Diff
----
Performs :math:`\T{C} = \T{A} - \T{B}`

.. cpp:function:: void Diff(const Tensor<T>& A, const Tensor<T>& B, Tensor<T>& C)
.. cpp:function:: void Diff(const DistTensor<T>& A, const DistTensor<T>& B, DistTensor<T>& C)

Elemscal
--------
Performs :math:`\T{C} = \T{A} \dotstar \T{B}`

.. cpp:function:: void ElemScal(const Tensor<T>& A, const Tensor<T>& B, Tensor<T>& C)
.. cpp:function:: void ElemScal(const DistTensor<T>& A, const DistTensor<T>& B, DistTensor<T>& C)

Norm
----
Performs :math:`||\T{A}||_2`

.. cpp:function:: Base<T>::type Norm(const Tensor<T>& A)
.. cpp:function:: Base<T>::type Norm(const DistTensor<T>& A)

Permute
-------
Locally performs :math:`\T{B} = \permuteVec\left(\T{A}\right)`

.. cpp:function:: void Permute(Tensor<T>& B, const Tensor<T>& A, const Permutation& perm)
.. cpp:function:: void Permute(DistTensor<T>& B, const DistTensor<T>& A)

Reduce
------
Locally performes :math:`\T{B} = \displaystyle\sum_\SetL{} \T{A}` for the set of modes :math:`\SetL{S}`

.. cpp:function:: void LocalReduce(Tensor<T>& B, const Tensor<T>& A, const ModeArray& reduceModes)
.. cpp:function:: void LocalReduce(Tensor<T>& B, const Tensor<T>& A, const Mode& reduceMode)
.. cpp:function:: void LocalReduce(DistTensor<T>& B, const DistTensor<T>& A, const ModeArray& reduceModes)
.. cpp:function:: void LocalReduce(DistTensor<T>& B, const DistTensor<T>& A, const Mode& reduceMode)

Scal
----
Performs :math:`\T{X} = \alpha \T{X}`

.. cpp:function:: void Scal( T alpha, Tensor<T>& X )
.. cpp:function:: void Scal( T alpha, DistTensor<T>& X )

YaxpBy
------
Performs :math:`\T{Y} = \alpha*\T{X} + \beta*\T{Y}`.  Variants where combinations of :math:`\alpha` and :math:`\beta` are removed from the argument list exist.

.. cpp:function:: void YAxpBy( T alpha, const Tensor<T>& X, T beta, Tensor<T>& Y )
.. cpp:function:: void YAxpBy( T alpha, const DistTensor<T>& X, T beta, DistTensor<T>& Y )

YAxpPx
------
.. note::

   The distributions of X and PX must be conformal

Performs :math:`\T{Y} = \alpha*\T{X} + \beta*\permuteVec\left(\T{X}\right)`. Variants where combinations of :math:`\alpha` and :math:`\beta` are removed from the argument list exist.

.. cpp:function:: void YAxpPx( T alpha, const Tensor<T>& X, T beta, const Tensor<T>& PX, const Permutation& perm, Tensor<T>& Y )
.. cpp:function:: void YAxpPx( T alpha, const DistTensor<T>& X, T beta, const DistTensor<T>& PX, const Permutation& perm, DistTensor<T>& Y )


ZAxpBy
------
Performs :math:`\T{Z} = \alpha * \T{X} + \beta * \T{Y}`. Variants where combinations of :math:`\alpha` and :math:`\beta` are removed from the argument list exist.

.. cpp:function:: void ZAxpBy( T alpha, const Tensor<T>& X, T beta, const Tensor<T>& Y, Tensor<T>& Z )
.. cpp:function:: void ZAxpBy( T alpha, const DistTensor<T>& X, T beta, const DistTensor<T>& Y, DistTensor<T>& Z )

Zero
----
Sets all entries of the input tensor to zero.

.. cpp:function:: void Zero( Tensor<T>& A )
.. cpp:function:: void Zero( DistTensor<T>& A )
