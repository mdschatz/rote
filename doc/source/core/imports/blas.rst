BLAS
----
The Basic Linear Algebra Subprograms (BLAS) are heavily exploited within 
|projectName| in order to achieve high performance whenever possible.  A thin interface layer has been created 
to accommodate the type-dependent naming scheme of various BLAS libraries. 

The prototypes can be found in 
`include/tensormental/core/imports/blas.hpp <>`_,
while the implementations are in 
`src/imports/blas.cpp <>`_.

Level 3
^^^^^^^

..  cpp:function:: void blas::Gemm( char transA, char transB, int m, int n, int k, T alpha, const T* A, int lda, const T* B, int ldb, T beta, T* C, int ldc )

    Perform the update 
    :math:`C := \alpha \mbox{op}_A(A) \mbox{op}_B(B) + \beta C`, 
    where :math:`\mbox{op}(A) \in \left\{A,A^T,A^H\right\}`
    is determined by `trans` being chosen as 'N', 'T', or 'C', respectively.; it is required that :math:`C \in T^{m \times n}` and that
    the inner dimension of :math:`\mbox{op}_A(A) \mbox{op}_B(B)` is `k`.
