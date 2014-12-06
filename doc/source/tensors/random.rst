Random
======

Uniform
-------
Creates an :math:`\tenDim{0} \times \cdots \times \tenDim{\tenOrder-1}` tensor with entries randomly drawn  
from a uniform distribution.

.. cpp:function:: void Uniform( Tensor<T>& A, const ObjShape& objShape )
.. cpp:function:: void Uniform( DistTensor<T>& A, const ObjShape& objShape )

   Set the tensor :math:`\T{A}` to an :math:`\tenDim{0} \times \cdots \times \tenDim{\tenOrder-1}` tensor with each entry drawn from a uniform distribution.

.. cpp:function:: void MakeUniform( Tensor<T>& A )
.. cpp:function:: void MakeUniform( DistTensor<T>& A )

   Sample each entry of :math:`\T{A}` from a uniform distribution.
