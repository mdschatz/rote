Input/output
************

.. cpp:function::

PrintData
---------

.. cpp:function:: void PrintData( const Tensor<T>& A, std::string title="", std::ostream& os = std::cout)
.. cpp:function:: void PrintData( const DistTensor<T>& A, std::string title="", std::ostream& os = std::cout)

   Prints the tensor meta-data to the console.

Print
-----

.. cpp:function:: void Print( const Tensor<T>& A, std::string title="", std::ostream& os=std::cout )
.. cpp:function:: void Print( const DistTensor<T>& A, std::string title="", std::ostream& os=std::cout )

   Prints the tensor to the console.

Write
-----

.. cpp:function:: void Write( const Tensor<T>& A, std::string title="", std::string filename="Tensor" )
.. cpp:function:: void Write( const DistTensor<T>& A, std::string title="", std::string filename="DistTensor" )

   Write the tensor to the specified file.
