/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#include "tensormental.hpp"
#include "tensormental/util/vec_util.hpp"

namespace tmen {

//
// Assertions
//

template<typename T>
void
Tensor<T>::AssertValidDimensions( const std::vector<Int>& dims ) const
{
#ifndef RELEASE
    CallStackEntry cse("Tensor::AssertValidDimensions");
#endif
        
    if( AnyNegativeElem(dims) )
        LogicError("Dimensions must be non-negative");
}

template<typename T>
void
Tensor<T>::AssertValidDimensions( const std::vector<Int>& dims, const std::vector<Int>& ldims ) const
{
#ifndef RELEASE
    CallStackEntry cse("Tensor::AssertValidDimensions");
#endif
    AssertValidDimensions( dims );
    if( !ElemwiseLessThan(dims, ldims) )
        LogicError("Leading dimensions must be no less than dimensions");
    if( AnyZeroElem(ldims) )
        LogicError("Leading dimensions cannot be zero (for BLAS compatibility)");
}

template<typename T>
void
Tensor<T>::AssertValidEntry( const std::vector<Int>& index ) const
{
#ifndef RELEASE
    CallStackEntry cse("Tensor::AssertValidEntry");
#endif
    if( AnyNegativeElem(index) )
        LogicError("Indices must be non-negative");
    if( !ElemwiseLessThan(index, shape_) )
    {
	int i;
        std::ostringstream msg;
        msg << "Out of bounds: "
            << "(";
	if(index.size() > 0)
		msg << index[0];
	for(i = 1; i < index.size(); i++)
		msg << ", " << index[i];
	msg << ") of ";
	if(index.size() > 0)
		msg << index[0];
	for(i = 1; i < shape_.size(); i++)
		msg << " x " << shape_[i];
	msg << "Tensor.";
        LogicError( msg.str() );
    }
}

//
// Constructors
//

template<typename T>
Tensor<T>::Tensor( bool fixed )
: viewType_( fixed ? OWNER_FIXED : OWNER ),
  order_(), shape_(), ldims_(), strides_(), indices_(),
  data_(nullptr)
{ }

template<typename T>
Tensor<T>::Tensor( const std::vector<Int>& indices, bool fixed )
: viewType_( fixed ? OWNER_FIXED : OWNER ),
  order_(indices.size()), shape_(indices.size()), ldims_(indices.size()), strides_(indices.size()), indices_(indices),
  data_(nullptr)
{
#ifndef RELEASE
    CallStackEntry cse("Tensor::Tensor");
#endif
}

template<typename T>
Tensor<T>::Tensor( const std::vector<Int>& indices, const std::vector<Int>& shape, bool fixed )
: viewType_( fixed ? OWNER_FIXED : OWNER ),
  order_(shape.size()), shape_(shape), strides_(Dimensions2Strides(shape)), indices_(indices)
{
#ifndef RELEASE
    CallStackEntry cse("Tensor::Tensor");
    AssertValidDimensions( shape );
#endif
    SetLDims(shape);
    memory_.Require( prod(ldims_) * shape[shape_.size()-1] );
    data_ = memory_.Buffer();
}

template<typename T>
Tensor<T>::Tensor
( const std::vector<Int>& indices, const std::vector<Int>& shape, const std::vector<Int>& ldims, bool fixed )
: viewType_( fixed ? OWNER_FIXED : OWNER ),
  order_(shape.size()), shape_(shape), strides_(Dimensions2Strides(shape)), indices_(indices)
{
#ifndef RELEASE
    CallStackEntry cse("Tensor::Tensor");
    AssertValidDimensions( shape, ldims );
#endif

    SetLDims(shape);
    Int ldimProd = prod(ldims);
    if(ldimProd > 0){
      memory_.Require( ldimProd*shape[shape.size()-1] );
    }
    else{
      memory_.Require( 0 );
    }
    data_ = memory_.Buffer();
}

template<typename T>
Tensor<T>::Tensor
( const std::vector<Int>& indices, const std::vector<Int>& shape, const T* buffer, const std::vector<Int>& ldims, bool fixed )
: viewType_( fixed ? LOCKED_VIEW_FIXED: LOCKED_VIEW ),
  order_(shape.size()), shape_(shape), ldims_(ldims), strides_(Dimensions2Strides(shape)), indices_(indices),
  data_(buffer)
{
#ifndef RELEASE
    CallStackEntry cse("Tensor::Tensor");
    AssertValidDimensions( shape, ldims );
#endif
}

template<typename T>
Tensor<T>::Tensor
( const std::vector<Int>& indices, const std::vector<Int>& shape, T* buffer, const std::vector<Int>& ldims, bool fixed )
: viewType_( fixed ? VIEW_FIXED: VIEW ),
  order_(shape.size()), shape_(shape), ldims_(ldims), strides_(Dimensions2Strides(shape)), indices_(indices),
  data_(buffer)
{
#ifndef RELEASE
    CallStackEntry cse("Tensor::Tensor");
    AssertValidDimensions( shape, ldims );
#endif
}

template<typename T>
Tensor<T>::Tensor( const Tensor<T>& A )
: viewType_( OWNER ),
  order_(), shape_(), ldims_(), strides_(),
  data_(nullptr)
{
#ifndef RELEASE
    CallStackEntry cse("Tensor::Tensor( const Tensor& )");
#endif
    if( &A != this )
        *this = A;
    else
        LogicError("You just tried to construct a Tensor with itself!");
}

template<typename T>
void
Tensor<T>::Swap( Tensor<T>& A )
{
    memory_.Swap( A.memory_ );
    std::swap( data_, A.data_ );
    std::swap( viewType_, A.viewType_ );
    std::swap( shape_, A.shape_ );
    std::swap( ldims_, A.ldims_ );
}

//
// Destructor
//

template<typename T>
Tensor<T>::~Tensor()
{ }

//
// Basic information
//

template<typename T>
void
Tensor<T>::SetLDims(const std::vector<Int>& dims)
{
  ldims_.resize(dims.size());
  int stride = 1;
  if(ldims_.size() > 0){
    ldims_[0] = 1;
    for(int i = 1; i < dims.size(); i++)
      ldims_[i] = ldims_[i-1]*dims[i-1];
  }
}

template<typename T>
Int
Tensor<T>::Order() const
{
	return order_;
}

template<typename T>
std::vector<Int>
Tensor<T>::Shape() const
{
	std::vector<Int> shape = shape_;
	return shape;
}

template<typename T>
Int 
Tensor<T>::Dimension(Int mode) const
{ 
  if( shape_.size() <= mode){
    LogicError("Requested mode dimension out of range.");
    return 0;
  }else
    return shape_[mode];
}

template<typename T>
std::vector<Int>
Tensor<T>::Indices() const
{
    return indices_;
}

template<typename T>
Int
Tensor<T>::ModeStride(Int mode) const
{
    if( strides_.size() <= mode){
      LogicError("Requested mode dimension out of range.");
      return 0;
    }else
      return strides_[mode];
}


template<typename T>
Int
Tensor<T>::LDim(Int mode) const
{ 
  if( shape_.size() < mode){
    LogicError("Requested mode leading dimension out of range.");
    return 0;
  }else
    return ldims_[mode]; 
}

template<typename T>
Int
Tensor<T>::MemorySize() const
{ return memory_.Size(); }

template<typename T>
bool
Tensor<T>::Owner() const
{ return IsOwner( viewType_ ); }

template<typename T>
bool
Tensor<T>::Viewing() const
{ return !IsOwner( viewType_ ); }

template<typename T>
bool
Tensor<T>::Shrinkable() const
{ return IsShrinkable( viewType_ ); }

template<typename T>
bool
Tensor<T>::FixedSize() const
{ return !IsShrinkable( viewType_ ); }

template<typename T>
bool
Tensor<T>::Locked() const
{ return IsLocked( viewType_ ); }

template<typename T>
T*
Tensor<T>::Buffer()
{
#ifndef RELEASE
    CallStackEntry cse("Tensor::Buffer");
    if( Locked() )
        LogicError("Cannot return non-const buffer of locked Tensor");
#endif
    // NOTE: This const_cast has been carefully considered and should be safe
    //       since the underlying data should be non-const if this is called.
    return const_cast<T*>(data_);
}

template<typename T>
Int
Tensor<T>::LinearOffset(const std::vector<int>& index) const
{
  //TODO: Error checking
  Int offset = 0;
  for(int i = 0; i < index.size(); i++)
    offset += index[i] * strides_[i];
  return offset;
}

template<typename T>
const T*
Tensor<T>::LockedBuffer() const
{ return data_; }

template<typename T>
T*
Tensor<T>::Buffer( const std::vector<Int>& index )
{
#ifndef RELEASE
    CallStackEntry cse("Tensor::Buffer");
    if( Locked() )
        LogicError("Cannot return non-const buffer of locked Tensor");
#endif
    // NOTE: This const_cast has been carefully considered and should be safe
    //       since the underlying data should be non-const if this is called.
    int linearOffset = LinearOffset(index);
    return &const_cast<T*>(data_)[linearOffset];
}

template<typename T>
const T*
Tensor<T>::LockedBuffer( const std::vector<Int>& index ) const
{
#ifndef RELEASE
    CallStackEntry cse("Tensor::LockedBuffer");
#endif
    int linearOffset = LinearOffset(index);
    return &data_[linearOffset];
}

//
// Entry manipulation
//

template<typename T>
const T&
Tensor<T>::Get_( const std::vector<Int>& index ) const
{ 
    int linearOffset = LinearOffset(index);
    return data_[linearOffset]; 
}

template<typename T>
T&
Tensor<T>::Set_( const std::vector<Int>& index ) 
{
    // NOTE: This const_cast has been carefully considered and should be safe
    //       since the underlying data should be non-const if this is called.
    int linearOffset = LinearOffset(index);
    return (const_cast<T*>(data_))[linearOffset];
}

template<typename T>
T
Tensor<T>::Get( const std::vector<Int>& index ) const
{
#ifndef RELEASE
    CallStackEntry cse("Tensor::Get");
    AssertValidEntry( index );
#endif
    return Get_( index );
}

template<typename T>
void
Tensor<T>::Set( const std::vector<Int>& index, T alpha ) 
{
#ifndef RELEASE
    CallStackEntry cse("Tensor::Set");
    AssertValidEntry( index );
    if( Locked() )
        LogicError("Cannot modify data of locked matrices");
#endif
    Set_( index ) = alpha;
}

template<typename T>
void
Tensor<T>::Update( const std::vector<Int>& index, T alpha ) 
{
#ifndef RELEASE
    CallStackEntry cse("Tensor::Update");
    AssertValidEntry( index );
    if( Locked() )
        LogicError("Cannot modify data of locked matrices");
#endif
    Set_( index ) += alpha;
}

//template<typename T>
//void
//Tensor<T>::GetDiagonal( Tensor<T>& d, Int offset ) const
//{ 
//#ifndef RELEASE
//    CallStackEntry cse("Tensor::GetDiagonal");
//    if( d.Locked() )
//        LogicError("d must not be a locked view");
//#endif
//    const Int diagLength = DiagonalLength(offset);
//    d.ResizeTo( diagLength, 1 );
//    if( offset >= 0 )
//        for( Int j=0; j<diagLength; ++j )
//            d.Set_( j, 0 ) = Get_(j,j+offset);
//    else
//        for( Int j=0; j<diagLength; ++j )
//            d.Set_( j, 0 ) = Get_(j-offset,j);
//}

//template<typename T>
//Tensor<T>
//Tensor<T>::GetDiagonal( Int offset ) const
//{ 
//    Tensor<T> d;
//    GetDiagonal( d, offset );
//    return d;
//}

//template<typename T>
//void
//Tensor<T>::SetDiagonal( const Tensor<T>& d, Int offset )
//{ 
//#ifndef RELEASE
//    CallStackEntry cse("Tensor::SetDiagonal");
//    if( d.Height() != DiagonalLength(offset) || d.Width() != 1 )
//        LogicError("d is not a column-vector of the right length");
//#endif
//    const Int diagLength = DiagonalLength(offset);
//    if( offset >= 0 )
//        for( Int j=0; j<diagLength; ++j )
//            Set_( j, j+offset ) = d.Get_(j,0);
//    else
//        for( Int j=0; j<diagLength; ++j )
//            Set_( j-offset, j ) = d.Get_(j,0);
//}
//
//template<typename T>
//void
//Tensor<T>::UpdateDiagonal( const Tensor<T>& d, Int offset )
//{ 
//#ifndef RELEASE
//    CallStackEntry cse("Tensor::UpdateDiagonal");
//    if( d.Height() != DiagonalLength(offset) || d.Width() != 1 )
//        LogicError("d is not a column-vector of the right length");
//#endif
//    const Int diagLength = DiagonalLength(offset);
//    if( offset >= 0 )
//        for( Int j=0; j<diagLength; ++j )
//            Set_( j, j+offset ) += d.Get(j,0);
//    else
//        for( Int j=0; j<diagLength; ++j )
//            Set_( j-offset, j ) += d.Get(j,0);
//}

template<typename T>
void
Tensor<T>::ComplainIfReal() const
{ 
    if( !IsComplex<T>::val )
        LogicError("Called complex-only routine with real data");
}

template<typename T>
BASE(T)
Tensor<T>::GetRealPart( const std::vector<Int>& index ) const
{
#ifndef RELEASE
    CallStackEntry cse("Tensor::GetRealPart");
    AssertValidEntry( index );
#endif
    return tmen::RealPart( Get_( index ) );
}

template<typename T>
BASE(T)
Tensor<T>::GetImagPart( const std::vector<Int>& index ) const
{
#ifndef RELEASE
    CallStackEntry cse("Tensor::GetImagPart");
    AssertValidEntry( index );
#endif
    return tmen::ImagPart( Get_( index ) );
}

template<typename T>
void 
Tensor<T>::SetRealPart( const std::vector<Int>& index, BASE(T) alpha )
{
#ifndef RELEASE
    CallStackEntry cse("Tensor::SetRealPart");
    AssertValidEntry( index );
    if( Locked() )
        LogicError("Cannot modify data of locked matrices");
#endif
    tmen::SetRealPart( Set_( index ), alpha );
}

template<typename T>
void 
Tensor<T>::SetImagPart( const std::vector<Int>& index, BASE(T) alpha )
{
#ifndef RELEASE
    CallStackEntry cse("Tensor::SetImagPart");
    AssertValidEntry( index );
    if( Locked() )
        LogicError("Cannot modify data of locked matrices");
#endif
    ComplainIfReal();
    tmen::SetImagPart( Set_( index ), alpha );
}

template<typename T>
void 
Tensor<T>::UpdateRealPart( const std::vector<Int>& index, BASE(T) alpha )
{
#ifndef RELEASE
    CallStackEntry cse("Tensor::UpdateRealPart");
    AssertValidEntry( index );
    if( Locked() )
        LogicError("Cannot modify data of locked matrices");
#endif
    tmen::UpdateRealPart( Set_( index ), alpha );
}

template<typename T>
void 
Tensor<T>::UpdateImagPart( const std::vector<Int>& index, BASE(T) alpha )
{
#ifndef RELEASE
    CallStackEntry cse("Tensor::UpdateImagPart");
    AssertValidEntry( index );
    if( Locked() )
        LogicError("Cannot modify data of locked matrices");
#endif
    ComplainIfReal();
    tmen::UpdateImagPart( Set_( index ), alpha );
}
   
//template<typename T>
//void
//Tensor<T>::GetRealPartOfDiagonal( Tensor<BASE(T)>& d, Int offset ) const
//{ 
//#ifndef RELEASE
//    CallStackEntry cse("Tensor::GetRealPartOfDiagonal");
//    if( d.Locked() )
//        LogicError("d must not be a locked view");
//#endif
//    const Int diagLength = DiagonalLength(offset);
//    d.ResizeTo( diagLength, 1 );
//    if( offset >= 0 )
//        for( Int j=0; j<diagLength; ++j )
//            d.Set_( j, 0 ) = tmen::RealPart( Get_(j,j+offset) );
//    else
//        for( Int j=0; j<diagLength; ++j )
//            d.Set_( j, 0 ) = tmen::RealPart( Get_(j-offset,j) );
//}
//
//template<typename T>
//void
//Tensor<T>::GetImagPartOfDiagonal( Tensor<BASE(T)>& d, Int offset ) const
//{ 
//#ifndef RELEASE
//    CallStackEntry cse("Tensor::GetImagPartOfDiagonal");
//    if( d.Locked() )
//        LogicError("d must not be a locked view");
//#endif
//    const Int diagLength = DiagonalLength(offset);
//    d.ResizeTo( diagLength, 1 );
//    if( offset >= 0 )
//        for( Int j=0; j<diagLength; ++j )
//            d.Set_( j, 0 ) = tmen::ImagPart( Get_(j,j+offset) );
//    else
//        for( Int j=0; j<diagLength; ++j )
//            d.Set_( j, 0 ) = tmen::ImagPart( Get_(j-offset,j) );
//}
//
//template<typename T>
//Tensor<BASE(T)>
//Tensor<T>::GetRealPartOfDiagonal( Int offset ) const
//{ 
//    Tensor<BASE(T)> d;
//    GetRealPartOfDiagonal( d, offset );
//    return d;
//}
//
//template<typename T>
//Tensor<BASE(T)>
//Tensor<T>::GetImagPartOfDiagonal( Int offset ) const
//{ 
//    Tensor<BASE(T)> d;
//    GetImagPartOfDiagonal( d, offset );
//    return d;
//}
//
//template<typename T>
//void
//Tensor<T>::SetRealPartOfDiagonal( const Tensor<BASE(T)>& d, Int offset )
//{ 
//#ifndef RELEASE
//    CallStackEntry cse("Tensor::SetRealPartOfDiagonal");
//    if( d.Height() != DiagonalLength(offset) || d.Width() != 1 )
//        LogicError("d is not a column-vector of the right length");
//#endif
//    const Int diagLength = DiagonalLength(offset);
//    if( offset >= 0 )
//        for( Int j=0; j<diagLength; ++j )
//            tmen::SetRealPart( Set_(j,j+offset), d.Get_(j,0) );
//    else
//        for( Int j=0; j<diagLength; ++j )
//            tmen::SetRealPart( Set_(j-offset,j), d.Get_(j,0) );
//}
//
//template<typename T>
//void
//Tensor<T>::SetImagPartOfDiagonal( const Tensor<BASE(T)>& d, Int offset )
//{ 
//#ifndef RELEASE
//    CallStackEntry cse("Tensor::SetImagPartOfDiagonal");
//    if( d.Height() != DiagonalLength(offset) || d.Width() != 1 )
//        LogicError("d is not a column-vector of the right length");
//#endif
//    ComplainIfReal();
//    const Int diagLength = DiagonalLength(offset);
//    if( offset >= 0 )
//        for( Int j=0; j<diagLength; ++j )
//            tmen::SetImagPart( Set_(j,j+offset), d.Get_(j,0) );
//    else
//        for( Int j=0; j<diagLength; ++j )
//            tmen::SetImagPart( Set_(j-offset,j), d.Get_(j,0) );
//}
//
//template<typename T>
//void
//Tensor<T>::UpdateRealPartOfDiagonal( const Tensor<BASE(T)>& d, Int offset )
//{ 
//#ifndef RELEASE
//    CallStackEntry cse("Tensor::UpdateRealPartOfDiagonal");
//    if( d.Height() != DiagonalLength(offset) || d.Width() != 1 )
//        LogicError("d is not a column-vector of the right length");
//#endif
//    const Int diagLength = DiagonalLength(offset);
//    if( offset >= 0 )
//        for( Int j=0; j<diagLength; ++j )
//            tmen::UpdateRealPart( Set_(j,j+offset), d.Get_(j,0) );
//    else
//        for( Int j=0; j<diagLength; ++j )
//            tmen::UpdateRealPart( Set_(j-offset,j), d.Get_(j,0) );
//}
//
//template<typename T>
//void
//Tensor<T>::UpdateImagPartOfDiagonal( const Tensor<BASE(T)>& d, Int offset )
//{ 
//#ifndef RELEASE
//    CallStackEntry cse("Tensor::UpdateImagPartOfDiagonal");
//    if( d.Height() != DiagonalLength(offset) || d.Width() != 1 )
//        LogicError("d is not a column-vector of the right length");
//#endif
//    ComplainIfReal();
//    const Int diagLength = DiagonalLength(offset);
//    if( offset >= 0 )
//        for( Int j=0; j<diagLength; ++j )
//            tmen::UpdateImagPart( Set_(j,j+offset), d.Get_(j,0) );
//    else
//        for( Int j=0; j<diagLength; ++j )
//            tmen::UpdateImagPart( Set_(j-offset,j), d.Get_(j,0) );
//}
//
//template<typename T>
//void
//Tensor<T>::Control_( Int height, Int width, T* buffer, Int ldim )
//{
//    memory_.Empty();
//    height_ = height;
//    width_ = width;
//    ldims_ = ldim;
//    data_ = buffer;
//    viewType_ = (ViewType)( viewType_ & ~LOCKED_VIEW );
//}
//
//template<typename T>
//void
//Tensor<T>::Control( Int height, Int width, T* buffer, Int ldim )
//{
//#ifndef RELEASE
//    CallStackEntry cse("Tensor::Control");
//    if( FixedSize() )
//        LogicError( "Cannot attach a new buffer to a view with fixed size" );
//#endif
//    Control_( height, width, buffer, ldim );
//}

template<typename T>
void
Tensor<T>::Attach_( const std::vector<Int>& dims, T* buffer, const std::vector<Int>& ldims )
{
    memory_.Empty();
    shape_ = dims;
    ldims_ = ldims;
    data_ = buffer;
    viewType_ = (ViewType)( ( viewType_ & ~LOCKED_OWNER ) | VIEW );
}

template<typename T>
void
Tensor<T>::Attach( const std::vector<Int>& dims, T* buffer, const std::vector<Int>& ldims )
{
#ifndef RELEASE
    CallStackEntry cse("Tensor::Attach");
    if( FixedSize() )
        LogicError( "Cannot attach a new buffer to a view with fixed size" );
#endif
    Attach_( dims, buffer, ldims );
}

template<typename T>
void
Tensor<T>::LockedAttach_( const std::vector<Int>& dims, const T* buffer, const std::vector<Int>& ldims )
{
    memory_.Empty();
    shape_ = dims;
    ldims_ = ldims;
    data_ = buffer;
    viewType_ = (ViewType)( viewType_ | VIEW );
}

template<typename T>
void
Tensor<T>::LockedAttach
( const std::vector<Int>& dims, const T* buffer, const std::vector<Int>& ldims )
{
#ifndef RELEASE
    CallStackEntry cse("Tensor::LockedAttach");
    if( FixedSize() )
        LogicError( "Cannot attach a new buffer to a view with fixed size" );
#endif
    LockedAttach_( dims, buffer, ldims );
}

//
// Utilities
//

template<typename T>
const Tensor<T>&
Tensor<T>::operator=( const Tensor<T>& A )
{
#ifndef RELEASE
    CallStackEntry cse("Tensor::operator=");
    if( Locked() )
        LogicError("Cannot assign to a locked view");
    if( viewType_ != OWNER && AnyElemwiseNotEqual(A.shape_, shape_) )
        LogicError
        ("Cannot assign to a view of different dimensions");
#endif
//    if( viewType_ == OWNER )
//        ResizeTo( A.dims_ );
    //TODO: IMPLEMENT CORRECTLY
//    const Int height = Height();
//    const Int width = Width();
//    const Int ldim = LDim();
//    const Int ldimOfA = A.LDim();
//    const T* src = A.LockedBuffer();
//    T* dst = this->Buffer();
//    PARALLEL_FOR
//    for( Int j=0; j<width; ++j )
//        MemCopy( &dst[j*ldim], &src[j*ldimOfA], height );
    return *this;
}

template<typename T>
void
Tensor<T>::Empty_()
{
    memory_.Empty();
    std::fill(shape_.begin(), shape_.end(), 0);
    std::fill(ldims_.begin(), shape_.end(), 0);
 
    data_ = nullptr;
    viewType_ = (ViewType)( viewType_ & ~LOCKED_VIEW );
}

template<typename T>
void
Tensor<T>::Empty()
{
#ifndef RELEASE
    CallStackEntry cse("Tensor::Empty()");
    if ( FixedSize() )
        LogicError("Cannot empty a fixed-size matrix" );
#endif
    Empty_();
}

template<typename T>
void
Tensor<T>::ResizeTo_( const std::vector<Int>& dimensions )
{
	//TODO: Implement general stride
	bool reallocate = AnyElemwiseGreaterThan(dimensions, shape_);
	shape_ = dimensions;
	if(reallocate){
		ldims_ = Dimensions2Strides(dimensions);
		strides_ = Dimensions2Strides(dimensions);
		memory_.Require(prod(dimensions));
		data_ = memory_.Buffer();
	}
    //TODO: IMPLEMENT CORRECTLY
//    bool reallocate = height > ldims_ || width > width_;
//    height_ = height;
//    width_ = width;
//    // Only change the ldim when necessary. Simply 'shrink' our view if 
//    // possible.
//    if( reallocate )
//    {
//        ldims_ = Max( height, 1 );
//        memory_.Require( ldims_ * width );
//        data_ = memory_.Buffer();
//    }
}

template<typename T>
void
Tensor<T>::ResizeTo( const std::vector<Int>& dimensions )
{
#ifndef RELEASE
    CallStackEntry cse("Tensor::ResizeTo(dimensions)");
    AssertValidDimensions( dimensions );
    if ( FixedSize() && ( AnyElemwiseNotEqual(dimensions, shape_) ) )
        LogicError("Cannot change the size of this tensor");
    if ( Viewing() && ( AnyElemwiseNotEqual(dimensions, shape_ ) ) )
        LogicError("Cannot increase the size of this tensor");
#endif
    ResizeTo_( dimensions );
}

template<typename T>
void
Tensor<T>::ResizeTo_( const std::vector<Int>& dimensions, const std::vector<Int>& ldims )
{
	//TODO: Implement general stride
	bool reallocate = AnyElemwiseGreaterThan(dimensions, shape_) || AnyElemwiseGreaterThan(ldims, ldims_);
	shape_ = dimensions;
	if(reallocate){
		ldims_ = ldims;
		memory_.Require(prod(dimensions));
		data_ = memory_.Buffer();
	}
    //TODO: IMPLEMENT CORRECTLY
//    bool reallocate = height > ldims_ || width > width_ || ldim != ldims_;
//    height_ = height;
//    width_ = width;
//    if( reallocate )
//    {
//        ldims_ = ldim;
//        memory_.Require(ldim*width);
//        data_ = memory_.Buffer();
//    }
}

template<typename T>
void
Tensor<T>::ResizeTo( const std::vector<Int>& dims, const std::vector<Int>& ldims )
{
    //TODO: IMPLEMENT CORRECTLY
#ifndef RELEASE
    CallStackEntry cse("Tensor::ResizeTo(dims,ldims)");
    AssertValidDimensions( dims, ldims );
    if( FixedSize() &&
        ( AnyElemwiseNotEqual(dims, shape_) || AnyElemwiseNotEqual(ldims, ldims_) ) )
        LogicError("Cannot change the size of this tensor");
    if( Viewing() && ( AnyElemwiseNotEqual(dims, shape_) || AnyElemwiseNotEqual(ldims, ldims_) ) )
        LogicError("Cannot increase the size of this matrix");
#endif
    ResizeTo_( dims, ldims );
}

template class Tensor<Int>;
#ifndef DISABLE_FLOAT
template class Tensor<float>;
#endif // ifndef DISABLE_FLOAT
template class Tensor<double>;
#ifndef DISABLE_COMPLEX
#ifndef DISABLE_FLOAT
template class Tensor<Complex<float> >;
#endif // ifndef DISABLE_FLOAT
template class Tensor<Complex<double> >;
#endif // ifndef DISABLE_COMPLEX

} // namespace tmen
