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
Tensor<T>::AssertValidDimensions( const std::vector<Int>& shape ) const
{
#ifndef RELEASE
    CallStackEntry cse("Tensor::AssertValidDimensions");
#endif
        
    if( AnyNegativeElem(shape) )
        LogicError("Dimensions must be non-negative");
}

template<typename T>
void
Tensor<T>::AssertValidDimensions( const std::vector<Int>& shape, const std::vector<Int>& ldims ) const
{
#ifndef RELEASE
    CallStackEntry cse("Tensor::AssertValidDimensions");
#endif
    AssertValidDimensions( shape );
    if(shape.size() != ldims.size())
        LogicError("shape order must match ldims order");
    if( !ElemwiseLessThan(shape, ldims) )
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
    const int order = this->Order();
    if(order != index.size())
        LogicError("Index must be of same order as object");

    if( AnyNegativeElem(index) )
        LogicError("Indices must be non-negative");

    if( !ElemwiseLessThan(index, shape_) )
    {
        int i;
        std::ostringstream msg;
        msg << "Out of bounds: "
            << "(";
        if(order > 0)
            msg << index[0];
        for(i = 1; i < index.size(); i++)
            msg << ", " << index[i];
        msg << ") of ";
        if(index.size() > 0)
            msg << index[0];
        for(i = 1; i < order; i++)
            msg << " x " << shape_[i];
        msg << "Tensor.";
            LogicError( msg.str() );
    }
}

template<typename T>
void
Tensor<T>::AssertValidIndices() const
{
#ifndef RELEASE
    CallStackEntry cse("Tensor::AssertValidIndices");
#endif
    std::set<Int> uniques(indices_.begin(), indices_.end());
    const int order = this->Order();

    if(uniques.size() != order){
        LogicError("Indices of a tensor must all be unique and number of same order as tensor");
    }
}

//
// Constructors
//

template<typename T>
Tensor<T>::Tensor( bool fixed )
: indices_(), shape_(), strides_(), ldims_(),
  index2modeMap_(), mode2indexMap_(),
  viewType_( fixed ? OWNER_FIXED : OWNER ),
  data_(nullptr), memory_()
{ }

template<typename T>
Tensor<T>::Tensor( const std::vector<Int>& indices, bool fixed )
: indices_(indices), shape_(indices.size()), strides_(indices.size()), ldims_(indices.size()),
  viewType_( fixed ? OWNER_FIXED : OWNER ),
  data_(nullptr), memory_()
{
#ifndef RELEASE
    CallStackEntry cse("Tensor::Tensor");
    AssertValidIndices();
#endif
    SetIndexMaps();
}

//TODO: Check for valid set of indices
template<typename T>
Tensor<T>::Tensor( const std::vector<Int>& indices, const std::vector<Int>& shape, bool fixed )
: indices_(indices), shape_(shape), strides_(Dimensions2Strides(shape)), ldims_(indices.size()),
  viewType_( fixed ? OWNER_FIXED : OWNER )
{
#ifndef RELEASE
    CallStackEntry cse("Tensor::Tensor");
    AssertValidDimensions( shape );
    AssertValidIndices();
#endif
    const int order = this->Order();
    SetLDims(shape_);
    SetIndexMaps();
    memory_.Require( prod(ldims_) * shape_[order-1] );
    data_ = memory_.Buffer();
}

//TODO: Check for valid set of indices
template<typename T>
Tensor<T>::Tensor
( const std::vector<Int>& indices, const std::vector<Int>& shape, const std::vector<Int>& ldims, bool fixed )
: indices_(indices), shape_(shape), strides_(Dimensions2Strides(shape)),
  viewType_( fixed ? OWNER_FIXED : OWNER )
{
#ifndef RELEASE
    CallStackEntry cse("Tensor::Tensor");
    AssertValidDimensions( shape, ldims );
    AssertValidIndices();
#endif
    const int order = this->Order();
    SetIndexMaps();
    SetLDims(shape);
    memory_.Require( prod(ldims_) * shape_[order-1] );

    data_ = memory_.Buffer();
}

//TODO: Check for valid set of indices
template<typename T>
Tensor<T>::Tensor
( const std::vector<Int>& indices, const std::vector<Int>& shape, const T* buffer, const std::vector<Int>& ldims, bool fixed )
: indices_(indices), shape_(shape), strides_(Dimensions2Strides(shape)), ldims_(ldims),
  viewType_( fixed ? LOCKED_VIEW_FIXED: LOCKED_VIEW ),
  data_(buffer), memory_()
{
#ifndef RELEASE
    CallStackEntry cse("Tensor::Tensor");
    AssertValidDimensions( shape, ldims );
    AssertValidIndices();
#endif
    SetIndexMaps();
}

//TODO: Check for valid set of indices
template<typename T>
Tensor<T>::Tensor
( const std::vector<Int>& indices, const std::vector<Int>& shape, T* buffer, const std::vector<Int>& ldims, bool fixed )
: indices_(indices), shape_(shape), strides_(Dimensions2Strides(shape)), ldims_(ldims),
  viewType_( fixed ? VIEW_FIXED: VIEW ),
  data_(buffer), memory_()
{
#ifndef RELEASE
    CallStackEntry cse("Tensor::Tensor");
    AssertValidDimensions( shape, ldims );
    AssertValidIndices();
#endif
    SetIndexMaps();
}

template<typename T>
Tensor<T>::Tensor( const Tensor<T>& A )
: indices_(), shape_(), strides_(), ldims_(),
  index2modeMap_(), mode2indexMap_(),
  viewType_( OWNER ),
  data_(nullptr), memory_()
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
    std::swap( indices_, A.indices_);
    std::swap( shape_, A.shape_ );
    std::swap( strides_, A.strides_ );
    std::swap( ldims_, A.ldims_ );
    std::swap( viewType_, A.viewType_ );
    std::swap( index2modeMap_, A.index2modeMap_ );
    std::swap( mode2indexMap_, A.mode2indexMap_ );
    std::swap( data_, A.data_ );
    memory_.Swap( A.memory_ );
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
Tensor<T>::SetIndexMaps(){

    const int order = this->Order();
    int i;
    for(i = 0; i < order; i++){
        index2modeMap_[indices_[i]] = i;
        mode2indexMap_[i] = indices_[i];
    }
}

template<typename T>
void
Tensor<T>::SetLDims(const std::vector<Int>& shape)
{

  const int order = this->Order();
  if(shape.size() != order){
      LogicError("SetLDims requires that shape order matches object order");
  }
  if(order > 0){
    ldims_[0] = 1;
    for(int i = 1; i < order; i++)
      ldims_[i] = ldims_[i-1]*shape[i-1];
  }
}

template<typename T>
Int
Tensor<T>::Order() const
{
	return shape_.size();
}

template<typename T>
std::vector<Int>
Tensor<T>::Shape() const
{
	return shape_;
}

template<typename T>
Int 
Tensor<T>::Dimension(Int mode) const
{ 
  const int order = this->Order();
  if( mode < 0 || mode >= order){
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
    const int order = this->Order();
    if( mode < 0 || mode >= order){
      LogicError("Requested mode dimension out of range.");
      return 0;
    }else
      return strides_[mode];
}

template<typename T>
Int
Tensor<T>::ModeOfIndex(Int index) const
{
    Int mode = index2modeMap_.at(index);
    return mode;
}

template<typename T>
Int
Tensor<T>::IndexOfMode(Int mode) const
{
    Int index = mode2indexMap_.at(mode);
    return index;
}

template<typename T>
Int
Tensor<T>::LDim(Int mode) const
{ 
  const int order = this->Order();
  if( mode < 0 || mode >= order){
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
  const int order = this->Order();
  if(index.size() != order){
      LogicError("index must be of same order as tensor");
  }
  Int offset = 0;
  for(int i = 0; i < order; i++)
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
Tensor<T>::Attach_( const std::vector<Int>& shape, T* buffer, const std::vector<Int>& ldims )
{
    memory_.Empty();
    shape_ = shape;
    ldims_ = ldims;
    data_ = buffer;
    viewType_ = (ViewType)( ( viewType_ & ~LOCKED_OWNER ) | VIEW );
}

template<typename T>
void
Tensor<T>::Attach( const std::vector<Int>& shape, T* buffer, const std::vector<Int>& ldims )
{
#ifndef RELEASE
    CallStackEntry cse("Tensor::Attach");
    if( FixedSize() )
        LogicError( "Cannot attach a new buffer to a view with fixed size" );
#endif
    Attach_( shape, buffer, ldims );
}

template<typename T>
void
Tensor<T>::LockedAttach_( const std::vector<Int>& shape, const T* buffer, const std::vector<Int>& ldims )
{
    memory_.Empty();
    shape_ = shape;
    ldims_ = ldims;
    data_ = buffer;
    viewType_ = (ViewType)( viewType_ | VIEW );
}

template<typename T>
void
Tensor<T>::LockedAttach
( const std::vector<Int>& shape, const T* buffer, const std::vector<Int>& ldims )
{
#ifndef RELEASE
    CallStackEntry cse("Tensor::LockedAttach");
    if( FixedSize() )
        LogicError( "Cannot attach a new buffer to a view with fixed size" );
#endif
    LockedAttach_( shape, buffer, ldims );
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
    std::fill(indices_.begin(), indices_.end(), 0);
    std::fill(shape_.begin(), shape_.end(), 0);
    std::fill(strides_.begin(), strides_.end(), 0);
    std::fill(ldims_.begin(), shape_.end(), 0);

    viewType_ = (ViewType)( viewType_ & ~LOCKED_VIEW );

    data_ = nullptr;
    memory_.Empty();
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
Tensor<T>::ResizeTo_( const std::vector<Int>& shape )
{
	//TODO: Implement general stride
	bool reallocate = AnyElemwiseGreaterThan(shape, shape_);
	shape_ = shape;
	if(reallocate){
		ldims_ = Dimensions2Strides(shape);
		strides_ = Dimensions2Strides(shape);
		memory_.Require(prod(shape));
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
Tensor<T>::ResizeTo( const std::vector<Int>& shape )
{
#ifndef RELEASE
    CallStackEntry cse("Tensor::ResizeTo(dimensions)");
    AssertValidDimensions( shape );
    if ( FixedSize() && ( AnyElemwiseNotEqual(shape, shape_) ) )
        LogicError("Cannot change the size of this tensor");
    if ( Viewing() && ( AnyElemwiseNotEqual(shape, shape_ ) ) )
        LogicError("Cannot increase the size of this tensor");
#endif
    ResizeTo_( shape );
}

template<typename T>
void
Tensor<T>::ResizeTo_( const std::vector<Int>& shape, const std::vector<Int>& ldims )
{
	//TODO: Implement general stride
	bool reallocate = AnyElemwiseGreaterThan(shape, shape_) || AnyElemwiseGreaterThan(ldims, ldims_);
	shape_ = shape;
	if(reallocate){
		ldims_ = ldims;
		memory_.Require(prod(shape));
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
Tensor<T>::ResizeTo( const std::vector<Int>& shape, const std::vector<Int>& ldims )
{
    //TODO: IMPLEMENT CORRECTLY
#ifndef RELEASE
    CallStackEntry cse("Tensor::ResizeTo(dims,ldims)");
    AssertValidDimensions( shape, ldims );
    if( FixedSize() &&
        ( AnyElemwiseNotEqual(shape, shape_) || AnyElemwiseNotEqual(ldims, ldims_) ) )
        LogicError("Cannot change the size of this tensor");
    if( Viewing() && ( AnyElemwiseNotEqual(shape, shape_) || AnyElemwiseNotEqual(ldims, ldims_) ) )
        LogicError("Cannot increase the size of this matrix");
#endif
    ResizeTo_( shape, ldims );
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
