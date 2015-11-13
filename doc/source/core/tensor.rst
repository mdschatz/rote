The Tensor class
================
This is the basic building block of the library: its purpose it to provide 
convenient mechanisms for performing basic tensor operations, 
such as setting and querying individual tensor entries, without giving up 
compatibility with interfaces such as BLAS, which assume column-major
storage.


An example of generating an :math:`m \times n` tensor of real double-precision 
numbers where the :math:`(i,j)` entry is equal to :math:`i-j` would be:

  .. code-block:: cpp

     #include "rote.hpp"
     using namespace rote;
     ...
     ObjShape shape(2);
     shape[0] = m;
     shape[1] = n;
     Tensor<double> A( shape );
     Location loc(2, 0);
     for( int j=0; j<n; ++j )
         for( int i=0; i<m; ++i ){
             loc[0] = i;
             loc[1] = j;
             A.Set( loc, double(i-j) );
         }

.. note::
   A scalar can be specified by providing a shape vector of size 0.  This is interpreted as an order-0 tensor (scalar).
     
The underlying data storage is simply a contiguous buffer that stores entries 
in a column-major fashion with a arbitrary strides for each mode (generalization of "leading dimensions"). 
For modifiable instances of the :cpp:type:`Tensor\<T>` class, the routine
:cpp:type:`Tensor\<T>::Buffer` returns a pointer to the underlying 
buffer, while :cpp:type:`Tensor\<T>::Stride` returns the stride of the specified mode; 
these two routines could be used to directly perform the equivalent
of the first code sample as follows:

  .. code-block:: cpp
     
     #include "rote.hpp"
     using namespace rote;
     ...
     ObjShape shape(2);
     shape[0] = m;
     shape[1] = n;
     Tensor<double> A( shape );
     Location loc(2, 0);
     double* buffer = A.Buffer();
     const int stride0 = A.Stride(0);
     const int stride1 = A.Stride(1);
     for( int j=0; j<n; ++j )
         for( int i=0; i<m; ++i ){
             buffer[i*stride0+j*stride1] = double(i-j);
         }

For constant instances of the :cpp:type:`Tensor\<T>` class, a ``const`` pointer
to the underlying data can similarly be returned with a call to 
:cpp:func:`Tensor\<T>::LockedBuffer`.
In addition, a (``const``) pointer to the place in the 
(``const``) buffer where the entry at a specified location resides can be easily retrieved
with a call to :cpp:func:`Tensor\<T>::Buffer` or 
:cpp:func:`Tensor\<T>::LockedBuffer`.

It is also important to be able to create tensors which are simply *views* 
of existing (sub)tensors. For example, if `A` is a :math:`10 \times 10` 
tensor of complex doubles, then a tensor :math:`A_{BR}` can easily be created 
to view the bottom-right :math:`6 \times 7` subtensor using

  .. code-block:: cpp

     #include "rote.hpp"
     ...
     ObjShape shape(2);
     shape[0] = 6;
     shape[1] = 7;
     Location loc(2);
     loc[0] = 4;
     loc[1] = 3;
     Tensor<T> ABR = View( A, loc, shape );

since the bottom-right :math:`6 \times 7` subtensor beings at index 
:math:`(4,3)`. In general, to view the :math:`\tenDim{0} \times \cdots \times \tenDim{\tenOrder-1}` subtensor starting
at entry :math:`(\tenLoc{0}, \ldots, \tenLoc{\tenOrder-1})`, one would set the shape to a variables such as `shape` and
 the location to a variable such as `loc` and the ncall ``View( ABR, A, loc, shape );``.

.. cpp:type:: class Tensor<T>

   The most general case, where the underlying datatype `T` is only assumed to 
   be a ring; that is, it supports multiplication and addition and has the 
   appropriate identities.

   .. rubric:: Constructors

   .. cpp:function:: Tensor()

      This simply creates a default order-0 tensor.

   .. cpp:function:: Tensor( const Unsigned order )

      This simply creates a default order-`order` tensor.

   .. cpp:function:: Tensor( ObjShape& shape )

      An order-`M` tensor of shape `shape[0] \times \cdots \times shape[M-1]` is created (where :math:`M = size(shape)`) 
      with mode strides represent the data being tightly packed in memory; i.e., :math:`stride[0] = 1` and :math:`stride[i] = stride[i-1]*dimension[i-1]`.

   .. cpp:function:: Tensor( const ObjShape shape, const std::vector<Unsigned>& strides )

      An order-`M` tensor of shape `shape[0] \times \cdots \times shape[M-1]` is created (where :math:`M = size(shape)`) 
      with the specified mode strides.

   .. cpp:function:: Tensor( const ObjShape shape, const T* buffer, const std::vector<Unsigned>& strides )

      An order-`M` tensor of shape `shape[0] \times \cdots \times shape[M-1]` is created (where :math:`M = size(shape)`) 
      with the specified mode strides used to view the underlying non-modifiable buffer ``T* buffer``.  The memory pointed to by `buffer` 
      should not be freed until after th :cpp:type:`Tensor\<T>` object is destructed.

   .. cpp:function:: Tensor( const ObjShape shape, T* buffer, const std::vector<Unsigned>& strides )

      An order-`M` tensor of shape `shape[0] \times \cdots \times shape[M-1]` is created (where :math:`M = size(shape)`) 
      with the specified mode strides used to view the underlying modifiable buffer ``T* buffer``.  The memory pointed to by `buffer` 
      should not be freed until after th :cpp:type:`Tensor\<T>` object is destructed.

   .. cpp:function:: Tensor( const Tensor<T>& A )

      A copy (not a view) of the tensor :math:`A` is built.

   .. rubric:: Basic information

   .. cpp:function:: Unsigned Order() const

      Return the order of the tensor.

   .. cpp:function:: ObjShape Shape() const

      Return the shape of the tensor.

   .. cpp:function:: Unsigned Dimension(Mode mode) const

      Return the dimension of the specified mode of the tensor.

   .. cpp:function:: Unsigned Stride(Mode mode) const

      Return the stride of the underlying buffer along the specified mode.

   .. cpp:function:: std::vector<Unsigned> Strides() const

      Return the strides of the underlying buffer.

   .. cpp:function:: Unsigned MemorySize() const

      Return the number of entries of type `T` that this :cpp:type:`Tensor\<T>`
      instance has allocated space for.

   .. cpp:function:: T* Buffer()

      Return a pointer to the underlying buffer.

   .. cpp:function:: const T* LockedBuffer() const

      Return a pointer to the underlying buffer that does not allow for 
      modifying the data.

   .. cpp:function:: T* Buffer( const Location& loc )

      Return a pointer to the portion of the buffer that holds entry 
      at location `loc`.

   .. cpp:function:: const T* LockedBuffer( const Location& loc ) const

      Return a pointer to the portion of the buffer that holds entry
      at location `loc` that does not allow for modifying the data.

   .. rubric:: Entry manipulation

   .. cpp:function:: T Get( const Location& loc ) const

      Return entry at location `loc`.

   .. cpp:function:: void Set( const Location& loc, T alpha )

      Set entry at location `loc` to :math:`\alpha`.

   .. cpp:function:: void Update( const Location& loc, T alpha )

      Add :math:`\alpha` to entry at location `loc`.

   .. note::

      Many of the following routines are only valid for complex datatypes.

   .. cpp:function:: typename Base<T>::type GetRealPart( const Location& loc ) const
   .. cpp:function:: typename Base<T>::type GetImagPart( const Location& loc ) const

      Return the real (imaginary) part of entry at location `loc`.

   .. cpp:function:: void SetRealPart( const Location& loc, typename Base<T>::type alpha )
   .. cpp:function:: void SetImagPart( const Location& loc, typename Base<T>::type alpha )

      Set the real (imaginary) part of entry at location `loc` to :math:`\alpha`.

   .. cpp:function:: void UpdateRealPart( const Location& loc, typename Base<T>::type alpha )
   .. cpp:function:: void UpdateImagPart( const Location& loc, typename Base<T>::type alpha ) 

      Add :math:`\alpha` to the real (imaginary) part of entry at location `loc`.

   .. rubric:: Views

   .. cpp:function:: bool Viewing() const

      Return whether or not this tensor is currently viewing another tensor.

   .. cpp:function:: bool Locked() const

      Return whether or not we can modify the data we are viewing.

   .. cpp:function:: void Attach( const ObjShape& shape, T* buffer, const std::vector<Unsigned>& strides )

      Reconfigure the tensor around the specified buffer.

   .. cpp:function:: void LockedAttach( const ObjShape& shape, const T* buffer, const std::vector<Unsigned>& strides )

      Reconfigure the tensor around the specified unmodifiable buffer.

   .. rubric:: Utilities

   .. cpp:function:: const Tensor<T>& operator=( const Tensor<T>& A )

      Create a copy of tensor :math:`A`.

   .. cpp:function:: void Empty()

      Sets the tensor to an order-0 tensor and frees the underlying buffer.

   .. cpp:function:: void ResizeTo( const ObjShape& shape )

      Reconfigures the tensor to be of shape `shape`.

   .. cpp:function:: void ResizeTo( const ObjShape& shape, const std::vector<Unsigned>& strides )

      Reconfigures the tensor to be of shape `shape`, but with 
      strides equal to `strides`.

Advanced Uses
-------------
Under certain circumstances, it is necessary to view an order-:math:`\tenOrder` tensor as a higher-order 
tensor with additional dimensions of unit size (and vice versa).  The following routines accomplish this functionality

.. cpp:function:: void IntroduceUnitModes(const Mode& mode);
.. cpp:function:: void RemoveUnitModes(const Mode& mode);

Introduce/Remove unit-dimension modes at the mode position specified by `mode`.

.. cpp:function:: void IntroduceUnitModes(const ModeArray& modes);
.. cpp:function:: void RemoveUnitModes(const ModeArray& modes);

Introduce/Remove unit-dimension modes at the mode positions specified by `modes`.


Special cases used in |projectName|
-----------------------------------
This list of special cases is here to help clarify the notation used throughout
|projectName|'s source (as well as this documentation). These are all special
cases of :cpp:type:`Tensor\<T>`.

.. cpp:type:: class Tensor<R>

   Used to denote that the underlying datatype `R` is real.

.. cpp:type:: class Tensor<Complex<R> >

   Used to denote that the underlying datatype :cpp:type:`Complex\<R>` is
   complex with base type `R`.

.. cpp:type:: class Tensor<F>

   Used to denote that the underlying datatype `F` is a field.

