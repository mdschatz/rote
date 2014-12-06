Level T
=======

The prototypes for the following routines can be found at          
`include/elemental/btas-like_decl.hpp <>`_, while the
implementations are in `include/elemental/blas-like/levelT/ <>`_.

LocalContract
-------------
Locally performs a general tensor contractioni retaining indices involved; i.e., :math:`\T{C}^{abijck} = \T{A}^{akic} \T{B}^{cjbk} + \T{C}^{abijck}`.

.. cpp:function:: void LocalContract(T alpha, const Tensor<T>& A, const IndexArray& indicesA, const Tensor<T>& B, const IndexArray& indicesB, T beta, Tensor<T>& C, const IndexArray& indicesC)
.. cpp:function:: void LocalContract(T alpha, const DistTensor<T>& A, const IndexArray& indicesA, const DistTensor<T>& B, const IndexArray& indicesB, T beta, DistTensor<T>& C, const IndexArray& indicesC)

Tensor contractions are currently implemented as calls to Gemm which involves permutations of data.  To specify which permutations are necessary and which are not, the following prototype is available.
When using this interface, it is assumed that tensors which will not be permuted are tightly packed in memory.

.. cpp:function:: void LocalContract(T alpha, const Tensor<T>& A, const IndexArray& indicesA, const bool permuteA, const Tensor<T>& B, const IndexArray& indicesB, const bool permuteB, T beta, Tensor<T>& C, const IndexArray& indicesC, const bool permuteC)

LocalContractAndLocalEliminate
------------------------------
Same as LocalContract, but eliminates the indices involved in the contraction from the output; i.e., :math:`\T{C}^{abij} = \T{A}^{akic} \T{B}^{cjbk} + \T{C}^{abij}`

.. cpp:function:: void LocalContractAndLocalEliminate(T alpha, const Tensor<T>& A, const IndexArray& indicesA, const Tensor<T>& B, const IndexArray& indicesB, T beta, Tensor<T>& C, const IndexArray& indicesC)
.. cpp:function:: void LocalContractAndLocalEliminate(T alpha, const Tensor<T>& A, const IndexArray& indicesA, const bool permuteA, const Tensor<T>& B, const IndexArray& indicesB, const bool permuteB, T beta, Tensor<T>& C, const IndexArray& indicesC, const bool permuteC)
