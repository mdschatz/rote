Deterministic
=============

Zeros
-----
Create an :math:`\tenDim{0} \times \cdots \times \tenDim{\tenOrder-1}` tensor of all zeros.

.. cpp:function:: void Zeros( Tensor<T>& A, const ObjShape& objShape )
.. cpp:function:: void Zeros( DistTensor<T>& A, const ObjShape& objShape )

Set the tensor :math:`\T{A}` to be an :math:`\tenDim{0} \times \cdots \times \tenDim{\tenOrder-1}` tensor of all zeros. 

.. cpp:function:: void MakeZeros( Tensor<T>& A )
.. cpp:function:: void MakeZeros( DistTensor<T>& A )

Change all entries of the tensor :math:`\T{A}` to zero.

