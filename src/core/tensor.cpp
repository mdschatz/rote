/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#include "tensormental.hpp"
#include "tensormental/util/vec_util.hpp"

namespace rote {

//
// Assertions
//

template<typename T>
void
Tensor<T>::AssertValidDimensions( const ObjShape& shape, const std::vector<Unsigned>& strides ) const
{
#ifndef RELEASE
    CallStackEntry cse("Tensor::AssertValidDimensions");
#endif
    if(shape.size() != strides.size())
        LogicError("shape order must match strides order");
    if( !ElemwiseLessThan(shape, strides) )
        LogicError("Leading dimensions must be no less than dimensions");
    if( AnyZeroElem(strides) )
        LogicError("Leading dimensions cannot be zero (for BLAS compatibility)");
}

template<typename T>
void
Tensor<T>::AssertValidEntry( const Location& loc ) const
{
#ifndef RELEASE
    CallStackEntry cse("Tensor::AssertValidEntry");
#endif
    const Unsigned order = Order();
    if(order != loc.size())
        LogicError("Index must be of same order as object");

     if( !ElemwiseLessThan(loc, shape_) )
    {
        Unsigned i;
        std::ostringstream msg;
        msg << "Out of bounds: (";
        if(order > 0)
            msg << loc[0];
        for(i = 1; i < loc.size(); i++)
            msg << ", " << loc[i];
        msg << ") of ";
        if(loc.size() > 0)
            msg << shape_[0];
        for(i = 1; i < order; i++)
            msg << " x " << shape_[i];
        msg << "Tensor.";
            LogicError( msg.str() );
    }
}

template<typename T>
void
Tensor<T>::AssertMergeableModes(const std::vector<ModeArray>& oldModes) const
{
#ifndef RELEASE
    CallStackEntry cse("Tensor::AssertMergeableModes");
#endif
    Unsigned i, j;
    for(i = 0; i < oldModes.size(); i++){
        for(j = 0; j < oldModes[i].size(); j++){
            if(oldModes[i][j] >= Order())
                LogicError("Specified mode out of range");
        }
    }
}

//
// Constructors
//

template<typename T>
Tensor<T>::Tensor( bool fixed )
: shape_(), strides_(),
  viewType_( fixed ? OWNER_FIXED : OWNER ),
  data_(0), memory_()
{ data_ = memory_.Buffer(); }

template<typename T>
Tensor<T>::Tensor( const Unsigned order, bool fixed )
: shape_(order, 0), strides_(order, 1),
  viewType_( fixed ? OWNER_FIXED : OWNER ),
  data_(0), memory_()
{ data_ = memory_.Buffer(); }

//TODO: Check for valid set of indices
template<typename T>
Tensor<T>::Tensor( const ObjShape& shape, bool fixed )
: shape_(shape), strides_(Dimensions2Strides(shape)),
  viewType_( fixed ? OWNER_FIXED : OWNER )
{
#ifndef RELEASE
    CallStackEntry cse("Tensor::Tensor");
#endif
    const Unsigned order = Order();
    Unsigned numElem = order > 0 ? strides_[order-1] * shape_[order-1] : 1;

    memory_.Require( numElem );
    data_ = memory_.Buffer();
}

//TODO: Check for valid set of indices
template<typename T>
Tensor<T>::Tensor
( const ObjShape& shape, const std::vector<Unsigned>& strides, bool fixed )
: shape_(shape), strides_(Dimensions2Strides(shape)),
  viewType_( fixed ? OWNER_FIXED : OWNER )
{
#ifndef RELEASE
    CallStackEntry cse("Tensor::Tensor");
    AssertValidDimensions( shape, strides );
#endif
    const Unsigned order = Order();
    Unsigned numElem = order > 0 ? strides_[order-1] * shape_[order-1] : 1;

    memory_.Require( numElem );
    data_ = memory_.Buffer();
}

template<typename T>
Tensor<T>::Tensor
( const ObjShape& shape, const T* buffer, const std::vector<Unsigned>& strides, bool fixed )
: shape_(shape), strides_(Dimensions2Strides(shape)),
  viewType_( fixed ? LOCKED_VIEW_FIXED: LOCKED_VIEW ),
  data_(buffer), memory_()
{
#ifndef RELEASE
    CallStackEntry cse("Tensor::Tensor");
#endif
    //Nothing left to do
}

template<typename T>
Tensor<T>::Tensor
( const ObjShape& shape, T* buffer, const std::vector<Unsigned>& strides, bool fixed )
: shape_(shape), strides_(Dimensions2Strides(shape)),
  viewType_( fixed ? VIEW_FIXED: VIEW ),
  data_(buffer), memory_()
{
#ifndef RELEASE
    CallStackEntry cse("Tensor::Tensor");
    AssertValidDimensions( shape, strides );
#endif
    //Nothing left to do
}

template<typename T>
Tensor<T>::Tensor( const Tensor<T>& A )
: shape_(), strides_(),
  viewType_( OWNER ),
  data_(0), memory_()
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
    std::swap( shape_, A.shape_ );
    std::swap( strides_, A.strides_ );
    std::swap( viewType_, A.viewType_ );
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
Unsigned
Tensor<T>::Order() const
{
	return shape_.size();
}

template<typename T>
ObjShape
Tensor<T>::Shape() const
{
	return shape_;
}

template<typename T>
Unsigned
Tensor<T>::Dimension(Mode mode) const
{ 
  const Unsigned order = Order();
  if( mode >= order){
    LogicError("Requested mode dimension out of range.");
    return 0;
  }

  return shape_[mode];
}

template<typename T>
std::vector<Unsigned>
Tensor<T>::Strides() const
{
    return strides_;
}

template<typename T>
Unsigned
Tensor<T>::Stride(Mode mode) const
{
    const Unsigned order = Order();
    if( mode >= order){
      LogicError("Requested mode dimension out of range.");
      return 0;
    }

    return strides_[mode];
}

template<typename T>
Unsigned
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
const T*
Tensor<T>::LockedBuffer() const
{ return data_; }

template<typename T>
T*
Tensor<T>::Buffer( const Location& loc )
{
#ifndef RELEASE
    CallStackEntry cse("Tensor::Buffer");
    if( Locked() )
        LogicError("Cannot return non-const buffer of locked Tensor");
#endif
    // NOTE: This const_cast has been carefully considered and should be safe
    //       since the underlying data should be non-const if this is called.
    Unsigned linearOffset = LinearLocFromStrides(loc, strides_);
    return &const_cast<T*>(data_)[linearOffset];
}

template<typename T>
const T*
Tensor<T>::LockedBuffer( const Location& loc ) const
{
#ifndef RELEASE
    CallStackEntry cse("Tensor::LockedBuffer");
#endif
    Unsigned linearOffset = LinearLocFromStrides(loc, strides_);
    return &data_[linearOffset];
}

//
// Unit mode info
//

template<typename T>
void
Tensor<T>::RemoveUnitModes(const ModeArray& modes){
#ifndef RELEASE
    CallStackEntry cse("Tensor::LockedBuffer");
#endif
    Unsigned i;
    ModeArray sorted = modes;
    SortVector(sorted);

    for(i = sorted.size() - 1; i < sorted.size(); i--){
        shape_.erase(shape_.begin() + sorted[i]);
        strides_.erase(strides_.begin() + sorted[i]);
    }
    ResizeTo(shape_);
}

template<typename T>
void
Tensor<T>::IntroduceUnitModes(const ModeArray& modes){
#ifndef RELEASE
    CallStackEntry cse("Tensor::IntroduceUnitModes");
#endif
    Unsigned i;
    ModeArray sorted = modes;
    SortVector(sorted);
    shape_.reserve(shape_.size() + sorted.size());
    strides_.reserve(strides_.size() + sorted.size());
    for(i = 0; i < sorted.size(); i++){
        Unsigned newStrideVal;
        if(sorted[i] == strides_.size()){
            if(sorted[i] == 0){
                newStrideVal = 1;
            }else{
                newStrideVal = strides_[shape_.size() - 1] * shape_[shape_.size() - 1];
            }
        }else{
            newStrideVal = strides_[sorted[i]];
        }

        strides_.insert(strides_.begin() + sorted[i], newStrideVal);
        shape_.insert(shape_.begin() + sorted[i], 1);
    }
    ResizeTo(shape_);
}

//
// Entry manipulation
//

template<typename T>
const T&
Tensor<T>::Get_( const Location& loc ) const
{ 
    return data_[LinearLocFromStrides(loc, strides_)];
}

template<typename T>
T&
Tensor<T>::Set_( const Location& loc )
{
    // NOTE: This const_cast has been carefully considered and should be safe
    //       since the underlying data should be non-const if this is called.
    return (const_cast<T*>(data_))[LinearLocFromStrides(loc, strides_)];
}

template<typename T>
T
Tensor<T>::Get( const Location& loc ) const
{
#ifndef RELEASE
    CallStackEntry cse("Tensor::Get");
    AssertValidEntry( loc );
#endif
    return Get_( loc );
}

template<typename T>
void
Tensor<T>::Set( const Location& loc, T alpha )
{
#ifndef RELEASE
    CallStackEntry cse("Tensor::Set");
    AssertValidEntry( loc );
    if( Locked() )
        LogicError("Cannot modify data of locked matrices");
#endif
    Set_( loc ) = alpha;
}

template<typename T>
void
Tensor<T>::Update( const Location& loc, T alpha )
{
#ifndef RELEASE
    CallStackEntry cse("Tensor::Update");
    AssertValidEntry( loc );
    if( Locked() )
        LogicError("Cannot modify data of locked matrices");
#endif
    Set_( loc ) += alpha;
}

template<typename T>
void
Tensor<T>::ComplainIfReal() const
{ 
    if( !IsComplex<T>::val )
        LogicError("Called complex-only routine with real data");
}

template<typename T>
BASE(T)
Tensor<T>::GetRealPart( const Location& loc ) const
{
#ifndef RELEASE
    CallStackEntry cse("Tensor::GetRealPart");
    AssertValidEntry( loc );
#endif
    return rote::RealPart( Get_( loc ) );
}

template<typename T>
BASE(T)
Tensor<T>::GetImagPart( const Location& loc ) const
{
#ifndef RELEASE
    CallStackEntry cse("Tensor::GetImagPart");
    AssertValidEntry( loc );
#endif
    return rote::ImagPart( Get_( loc ) );
}

template<typename T>
void 
Tensor<T>::SetRealPart( const Location& loc, BASE(T) alpha )
{
#ifndef RELEASE
    CallStackEntry cse("Tensor::SetRealPart");
    AssertValidEntry( loc );
    if( Locked() )
        LogicError("Cannot modify data of locked matrices");
#endif
    rote::SetRealPart( Set_( loc ), alpha );
}

template<typename T>
void 
Tensor<T>::SetImagPart( const Location& loc, BASE(T) alpha )
{
#ifndef RELEASE
    CallStackEntry cse("Tensor::SetImagPart");
    AssertValidEntry( loc );
    if( Locked() )
        LogicError("Cannot modify data of locked matrices");
#endif
    ComplainIfReal();
    rote::SetImagPart( Set_( loc ), alpha );
}

template<typename T>
void 
Tensor<T>::UpdateRealPart( const Location& loc, BASE(T) alpha )
{
#ifndef RELEASE
    CallStackEntry cse("Tensor::UpdateRealPart");
    AssertValidEntry( loc );
    if( Locked() )
        LogicError("Cannot modify data of locked matrices");
#endif
    rote::UpdateRealPart( Set_( loc ), alpha );
}

template<typename T>
void 
Tensor<T>::UpdateImagPart( const Location& loc, BASE(T) alpha )
{
#ifndef RELEASE
    CallStackEntry cse("Tensor::UpdateImagPart");
    AssertValidEntry( loc );
    if( Locked() )
        LogicError("Cannot modify data of locked matrices");
#endif
    ComplainIfReal();
    rote::UpdateImagPart( Set_( loc ), alpha );
}

template<typename T>
void
Tensor<T>::Attach_( const ObjShape& shape, T* buffer, const std::vector<Unsigned>& strides )
{
    memory_.Empty();
    shape_ = shape;
    strides_ = strides;
    data_ = buffer;
    viewType_ = (ViewType)( ( viewType_ & ~LOCKED_OWNER ) | VIEW );
}

template<typename T>
void
Tensor<T>::Attach( const ObjShape& shape, T* buffer, const std::vector<Unsigned>& strides )
{
#ifndef RELEASE
    CallStackEntry cse("Tensor::Attach");
    if( FixedSize() )
        LogicError( "Cannot attach a new buffer to a view with fixed size" );
#endif
    Attach_( shape, buffer, strides );
}

template<typename T>
void
Tensor<T>::LockedAttach_( const ObjShape& shape, const T* buffer, const std::vector<Unsigned>& strides )
{
    memory_.Empty();
    shape_ = shape;
    strides_ = strides;
    data_ = buffer;
    viewType_ = (ViewType)( viewType_ | VIEW );
}

template<typename T>
void
Tensor<T>::LockedAttach
( const ObjShape& shape, const T* buffer, const std::vector<Unsigned>& strides )
{
#ifndef RELEASE
    CallStackEntry cse("Tensor::LockedAttach");
    if( FixedSize() )
        LogicError( "Cannot attach a new buffer to a view with fixed size" );
#endif
    LockedAttach_( shape, buffer, strides );
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
    if( viewType_ == OWNER )
        ResizeTo( A );

    CopyBuffer(A);

    return *this;
}

template<typename T>
void
Tensor<T>::Empty_()
{
    std::fill(shape_.begin(), shape_.end(), 0);
    std::fill(strides_.begin(), strides_.end(), 1);

    viewType_ = (ViewType)( viewType_ & ~LOCKED_VIEW );

    data_ = 0;
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
Tensor<T>::ResizeTo_( const ObjShape& shape )
{
	ResizeTo_(shape, Dimensions2Strides(shape));
}

template<typename T>
void
Tensor<T>::ResizeTo( const Tensor<T>& A )
{
#ifndef RELEASE
    CallStackEntry cse("Tensor::ResizeTo(Tensor)");

    if ( FixedSize() && AnyElemwiseNotEqual(A.shape_, shape_) )
        LogicError("Cannot change the size of this tensor");
    if ( Viewing() && AnyElemwiseNotEqual(A.shape_, shape_) )
        LogicError("Cannot increase the size of this tensor");
#endif
    shape_.resize(A.shape_.size(), 0);
    strides_.resize(A.strides_.size(), 1);
    ResizeTo(A.shape_);
}

template<typename T>
void
Tensor<T>::ResizeTo( const ObjShape& shape )
{
#ifndef RELEASE
    CallStackEntry cse("Tensor::ResizeTo(dimensions)");

    if ( FixedSize() && AnyElemwiseNotEqual(shape, shape_) )
        LogicError("Cannot change the size of this tensor");
    if ( Viewing() && AnyElemwiseNotEqual(shape, shape_) )
        LogicError("Cannot increase the size of this tensor");
#endif
    ResizeTo_( shape );
}

template<typename T>
void
Tensor<T>::ResizeTo_( const ObjShape& shape, const std::vector<Unsigned>& strides )
{
	Unsigned oldOrder = shape_.size();
	Unsigned curMaxElem = (oldOrder == 0) ? 1 : shape_[oldOrder - 1] * strides_[oldOrder - 1];

	Unsigned newOrder = shape.size();
	Unsigned newMaxElem = (newOrder == 0) ? 1 : shape[newOrder - 1] * strides[newOrder - 1];

	shape_ = shape;
	strides_ = strides;

	//Guard has newOrder == 0 in case we resize scalar to scalar (ensures at least one data entry is created)
	if(newOrder == 0 || (curMaxElem < newMaxElem)){
		memory_.Require(Max(1,newMaxElem));
		data_ = memory_.Buffer();
	}
}

template<typename T>
void
Tensor<T>::ResizeTo( const ObjShape& shape, const std::vector<Unsigned>& strides )
{
    //TODO: IMPLEMENT CORRECTLY
#ifndef RELEASE
    CallStackEntry cse("Tensor::ResizeTo(dims,strides)");
    AssertValidDimensions( shape, strides );
    if( FixedSize() &&
        ( AnyElemwiseNotEqual(shape, shape_) || AnyElemwiseNotEqual(strides, strides_) ) )
        LogicError("Cannot change the size of this tensor");
    if( Viewing() && ( AnyElemwiseNotEqual(shape, shape_) || AnyElemwiseNotEqual(strides, strides_) ) )
        LogicError("Cannot increase the size of this matrix");
#endif
    shape_.resize(shape.size(), 0);
    strides_.resize(strides.size(), 1);
    ResizeTo_( shape, strides );
}

template<typename T>
void
Tensor<T>::CopyBuffer(const Tensor<T>& A)
{
#ifndef RELEASE
    CallStackEntry cse("Tensor::CopyBuffer");
#endif
    Permutation perm = DefaultPermutation(A.Order());
    CopyBuffer(A, perm, perm);
}

template<typename T>
void
Tensor<T>::CopyBuffer(const Tensor<T>& A, const Permutation& srcPerm, const Permutation& dstPerm)
{
#ifndef RELEASE
    CallStackEntry cse("Tensor::CopyBuffer");
#endif
    const T* srcBuf = A.LockedBuffer();
    T* thisBuf = Buffer();

    Permutation out2inPerm = DeterminePermutation(dstPerm, srcPerm);
    PackData packData;
    packData.loopShape = A.Shape();
	packData.srcBufStrides = A.Strides();
    packData.dstBufStrides = PermuteVector(Strides(), out2inPerm);

    PackCommHelper(packData, &(srcBuf[0]), &(thisBuf[0]));
}

template class Tensor<Int>;
#ifndef DISABLE_FLOAT
template class Tensor<float>;
#endif // ifndef DISABLE_FLOAT
template class Tensor<double>;
#ifndef DISABLE_COMPLEX
#ifndef DISABLE_FLOAT
template class Tensor<std::complex<float> >;
#endif // ifndef DISABLE_FLOAT
template class Tensor<std::complex<double> >;
#endif // ifndef DISABLE_COMPLEX

} // namespace tmen
