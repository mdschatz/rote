/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#include "tensormental.hpp"

namespace tmen {

template<typename T>
AbstractDistTensor<T>::AbstractDistTensor( const tmen::Grid& grid )
: viewType_(OWNER),
  order_(),
  dims_(), 
  auxMemory_(),
  dist_(),
  indices_(),
  tensor_(dims_,dims_,true), 
  constrainedModeAlignments_(), 
  modeAlignments_(),
  modeShifts_(),
  grid_(&grid)
{ 

}

template<typename T>
AbstractDistTensor<T>::AbstractDistTensor( Int order, const tmen::Grid& grid )
: viewType_(OWNER),
  order_(order),
  dims_(order),
  auxMemory_(),
  dist_(order),
  indices_(order),
  tensor_(order),
  constrainedModeAlignments_(order),
  modeAlignments_(order),
  modeShifts_(order),
  grid_(&grid)
{

}
/*
template<typename T>
AbstractDistTensor<T>::AbstractDistTensor( AbstractDistTensor<T>&& A )
: viewType_(A.viewType_),
  dims_(A.dims_), 
  constrainedModeAlignments_(A.constrainedModeAlignments_), 
  modeAlignments_(A.modeAlignments_),
  modeShifts_(A.modeShifts_),
  grid_(A.grid_)
{ 
    tensor_.Swap( A.tensor_ );
    auxMemory_.Swap( A.auxMemory_ );
}

template<typename T>
AbstractDistTensor<T>& 
AbstractDistTensor<T>::operator=( AbstractDistTensor<T>&& A )
{
    auxMemory_.Swap( A.auxMemory_ );
    tensor_.Swap( A.tensor_ );
    viewType_ = A.viewType_;
    dims_ = A.dims_;
    constrainedModeAlignments_ = A.constrainedModeAlignments_;
    modeAlignments_ = A.modeAlignments_;
    modeShifts_ = A.modeShifts_;
    grid_ = A.grid_;
    return *this;
}
*/

template<typename T>
AbstractDistTensor<T>::~AbstractDistTensor() 
{ }

template<typename T>
void 
AbstractDistTensor<T>::Swap( AbstractDistTensor<T>& A )
{
    tensor_.Swap( A.tensor_ );
    auxMemory_.Swap( A.auxMemory_ );
    std::swap( viewType_, A.viewType_ );
    std::swap( dims_ , A.dims_ );
    std::swap( dist_, A.dist_ );
    std::swap( indices_, A.indices_ );
    std::swap( constrainedModeAlignments_, A.constrainedModeAlignments_ );
    std::swap( modeAlignments_, A.modeAlignments_ );
    std::swap( modeShifts_, A.modeShifts_ );
    std::swap( grid_, A.grid_ );
}

#ifndef RELEASE
template<typename T>
void
AbstractDistTensor<T>::AssertNotLocked() const
{
    if( Locked() )
        LogicError("Assertion that tensor not be a locked view failed");
}

template<typename T>
void
AbstractDistTensor<T>::AssertNotStoringData() const
{
    if( tensor_.MemorySize() > 0 )
        LogicError("Assertion that tensor not be storing data failed");
}

template<typename T>
void
AbstractDistTensor<T>::AssertValidEntry( const std::vector<Int>& loc ) const
{
    if(loc.size() != dims_.size() )
    {
      std::ostringstream msg;
      msg << "Entry is of incorrect order.\n";
      LogicError( msg.str() );
    }
    if(!ElemwiseLessThan(loc, dims_))
    {
        std::ostringstream msg;
        msg << "Entry (";
        for(int i = 0; i < loc.size() - 1; i++)
          msg << loc[i] << ", ";
        msg << loc[loc.size()-1] << ") is out of bounds of ";
        
	for(int i = 0; i < dims_.size() - 1; i++)
          msg << dims_[i] << " x ";
	msg << dims_[dims_.size()-1] << " tensor.";
        LogicError( msg.str() );
    }
}

//TODO: FIX ASSERTIONS
template<typename T>
void
AbstractDistTensor<T>::AssertValidSubtensor
( const std::vector<Int>& index, const std::vector<Int>& dims ) const
{
    if( AnyNegativeElem(index) )
        LogicError("Indices of subtensor were negative");
    if( AnyNegativeElem(dims) )
        LogicError("Dimensions of subtensor were negative");
    std::vector<Int> maxIndex(index.size());
    ElemwiseSum(index, dims, maxIndex);

    if( !ElemwiseLessThan(maxIndex, dims_) )
    {
        std::ostringstream msg;
        msg << "Subtensor is out of bounds: accessing up to (";
        for(int i = 0; i < order_ - 1; i++)
          msg << maxIndex[i] << ",";
        msg << maxIndex[order_ - 1] << ") of ";

        for(int i = 0; i < order_ - 1; i++)
          msg << Dimension(i) << " x ";
        msg << Dimension(order_ - 1) << " tensor.";
        LogicError( msg.str() );
    }
}

template<typename T> 
void
AbstractDistTensor<T>::AssertSameGrid( const tmen::Grid& grid ) const
{
    if( Grid() != grid )
        LogicError("Assertion that grids match failed");
}

template<typename T> 
void
AbstractDistTensor<T>::AssertSameSize( const std::vector<Int>& dims ) const
{
    if( dims.size() != order_)
      LogicError("Assertion that tensors be the same order failed");
    if( AnyElemwiseNotEqual(dims, dims_) )
        LogicError("Assertion that tensors be the same size failed");
}

//template<typename T> 
//void
//AssertConforming1x2
//( const AbstractDistTensor<T>& AL, const AbstractDistTensor<T>& AR )
//{
//    if( AL.Height() != AR.Height() )    
//    {
//        std::ostringstream msg;
//        msg << "1x2 not conformant. Left is " << AL.Height() << " x " 
//            << AL.Width() << ", right is " << AR.Height() << " x " 
//            << AR.Width();
//        LogicError( msg.str() );
//    }
//    if( AL.ColAlignment() != AR.ColAlignment() )
//        LogicError("1x2 is misaligned");
//}
//
//template<typename T> 
//void
//AssertConforming2x1
//( const AbstractDistTensor<T>& AT, const AbstractDistTensor<T>& AB )
//{
//    if( AT.Width() != AB.Width() )
//    {
//        std::ostringstream msg;        
//        msg << "2x1 is not conformant. Top is " << AT.Height() << " x " 
//            << AT.Width() << ", bottom is " << AB.Height() << " x " 
//            << AB.Width();
//        LogicError( msg.str() );
//    }
//    if( AT.RowAlignment() != AB.RowAlignment() )
//        LogicError("2x1 is not aligned");
//}
//
//template<typename T> 
//void
//AssertConforming2x2
//( const AbstractDistTensor<T>& ATL, const AbstractDistTensor<T>& ATR,
//  const AbstractDistTensor<T>& ABL, const AbstractDistTensor<T>& ABR ) 
//{
//    if( ATL.Width() != ABL.Width() || ATR.Width() != ABR.Width() ||
//        ATL.Height() != ATR.Height() || ABL.Height() != ABR.Height() )
//    {
//        std::ostringstream msg;
//        msg << "2x2 is not conformant: " << std::endl
//            << "  TL is " << ATL.Height() << " x " << ATL.Width() << std::endl
//            << "  TR is " << ATR.Height() << " x " << ATR.Width() << std::endl
//            << "  BL is " << ABL.Height() << " x " << ABL.Width() << std::endl
//            << "  BR is " << ABR.Height() << " x " << ABR.Width();
//        LogicError( msg.str() );
//    }
//    if( ATL.ColAlignment() != ATR.ColAlignment() ||
//        ABL.ColAlignment() != ABR.ColAlignment() ||
//        ATL.RowAlignment() != ABL.RowAlignment() ||
//        ATR.RowAlignment() != ABR.RowAlignment() )
//        LogicError("2x2 set of matrices must aligned to combine");
//}
#endif // RELEASE

template<typename T>
void
AbstractDistTensor<T>::Align( const std::vector<Int>& modeAlignments )
{ 
#ifndef RELEASE
    CallStackEntry entry("AbstractDistTensor::Align");    
#endif
    Empty();
    modeAlignments_ = modeAlignments;
    for(int i = 0; i < modeAlignments.size(); i++)
      constrainedModeAlignments_[i] = true;
    SetShifts();
}

template<typename T>
void
AbstractDistTensor<T>::AlignMode( Int mode, Int modeAlignment )
{ 
#ifndef RELEASE
    CallStackEntry entry("AbstractDistTensor::AlignCols"); 
#endif
    EmptyData();
    modeAlignments_[mode] = modeAlignment;
    constrainedModeAlignments_[mode] = true;
    SetShifts();
}

template<typename T>
void
AbstractDistTensor<T>::AlignWith( const tmen::DistData& data )
{ SetGrid( *data.grid ); }

template<typename T>
void
AbstractDistTensor<T>::AlignWith( const AbstractDistTensor<T>& A )
{ AlignWith( A.DistData() ); }

template<typename T>
void
AbstractDistTensor<T>::AlignModeWith( Int mode, const tmen::DistData& data )
{ 
    EmptyData(); 
    modeAlignments_[mode] = 0; 
    constrainedModeAlignments_[mode] = false; 
    SetShifts(); 
}

template<typename T>
void
AbstractDistTensor<T>::AlignModeWith( Int mode, const AbstractDistTensor<T>& A )
{ AlignModeWith( mode, A.DistData() ); }

template<typename T>
void
AbstractDistTensor<T>::SetAlignmentsAndResize
( const std::vector<Int>& modeAligns, const std::vector<Int>& modeDims )
{
#ifndef RELEASE
    CallStackEntry cse("AbstractDistTensor::SetAlignmentsAndResize");
#endif
    if( !Viewing() )
    {
        for(int i = 0; i < order_; i++){
          if(!ConstrainedModeAlignment(i)){
		modeAlignments_[i] = modeAligns[i];
		SetModeShift(i);
	  }
        }
    }
    ResizeTo( modeDims );
}

template<typename T>
void
AbstractDistTensor<T>::ForceAlignmentsAndResize
( const std::vector<Int>& modeAligns, const std::vector<Int>& modeDims )
{
#ifndef RELEASE
    CallStackEntry cse("AbstractDistTensor::ForceAlignmentsAndResize");
#endif
    SetAlignmentsAndResize( modeAligns, modeDims );
    if(AnyElemwiseNotEqual(modeAlignments_, modeAligns))
        LogicError("Could not set alignments"); 
}

template<typename T>
void
AbstractDistTensor<T>::SetModeAlignmentAndResize
( Int mode, Int modeAlign, const std::vector<Int>& modeDims )
{
#ifndef RELEASE
    CallStackEntry cse("AbstractDistTensor::SetColAlignmentAndResize");
#endif
    if( !Viewing() && !ConstrainedModeAlignment(mode) )
    {
        modeAlignments_[mode] = modeAlign;
        SetModeShift(mode); 
    }
    ResizeTo( modeDims );
}

template<typename T>
void
AbstractDistTensor<T>::ForceModeAlignmentAndResize
(Int mode, Int modeAlign, const std::vector<Int>& modeDims  )
{
#ifndef RELEASE
    CallStackEntry cse("AbstractDistTensor::ForceColAlignmentAndResize");
#endif
    SetModeAlignmentAndResize( mode, modeAlign, modeDims );
    if( modeAlignments_[mode] != modeAlign )
        LogicError("Could not set mode alignment");
}

template<typename T>
bool
AbstractDistTensor<T>::Viewing() const
{ return !IsOwner( viewType_ ); }

template<typename T>
bool
AbstractDistTensor<T>::Locked() const
{ return IsLocked( viewType_ ); }

template<typename T>
Int
AbstractDistTensor<T>::Dimension(Int mode) const
{ return dims_[mode]; }

template<typename T>
Int
AbstractDistTensor<T>::Order() const
{ return order_; }

template<typename T>
std::vector<Int>
AbstractDistTensor<T>::Indices() const
{ std::vector<Int> indices = this->indices_; return indices; }

template<typename T>
TensorDistribution
AbstractDistTensor<T>::TensorDist() const
{
	TensorDistribution dist = dist_;
	return dist;
}

template<typename T>
ModeDistribution
AbstractDistTensor<T>::ModeDist(Int mode) const
{
	if(mode < 0 || mode >= tensor_.Order())
		LogicError("Requesting distribution of invalid mode");
	return dist_[mode];
}

/*
template<typename T>
Int
AbstractDistTensor<T>::DiagonalLength( Int offset ) const
{ return tmen::DiagonalLength(height_,width_,offset); }
*/

template<typename T>
void
AbstractDistTensor<T>::FreeAlignments() 
{
    for(int i = 0; i < order_; i++)
      constrainedModeAlignments_[i] = false; 
}
    
template<typename T>
bool
AbstractDistTensor<T>::ConstrainedModeAlignment(Int mode) const
{ return constrainedModeAlignments_[mode]; }

template<typename T>
Int
AbstractDistTensor<T>::ModeAlignment(Int mode) const
{ return modeAlignments_[mode]; }

template<typename T>
Int
AbstractDistTensor<T>::ModeShift(Int mode) const
{ return modeShifts_[mode]; }

template<typename T>
const tmen::Grid&
AbstractDistTensor<T>::Grid() const
{ return *grid_; }

template<typename T>
size_t
AbstractDistTensor<T>::AllocatedMemory() const
{ return tensor_.MemorySize(); }

template<typename T>
std::vector<Int>
AbstractDistTensor<T>::LocalShape() const
{
	return tensor_.Shape();
}
template<typename T>
Int
AbstractDistTensor<T>::LocalDimension(Int mode) const
{ return tensor_.Dimension(mode); }

template<typename T>
Int
AbstractDistTensor<T>::LDim(Int mode) const
{ return tensor_.LDim(mode); }

template<typename T>
T
AbstractDistTensor<T>::GetLocal( const std::vector<Int>& loc ) const
{ return tensor_.Get(loc); }

template<typename T>
void
AbstractDistTensor<T>::SetLocal( const std::vector<Int>& loc, T alpha )
{ tensor_.Set(loc,alpha); }

template<typename T>
void
AbstractDistTensor<T>::UpdateLocal( const std::vector<Int>& loc, T alpha )
{ tensor_.Update(loc,alpha); }

template<typename T>
T*
AbstractDistTensor<T>::Buffer( const std::vector<Int>& loc )
{ return tensor_.Buffer(loc); }

template<typename T>
const T*
AbstractDistTensor<T>::LockedBuffer( const std::vector<Int>& loc ) const
{ return tensor_.LockedBuffer(loc); }

template<typename T>
tmen::Tensor<T>&
AbstractDistTensor<T>::Tensor()
{ return tensor_; }

template<typename T>
const tmen::Tensor<T>&
AbstractDistTensor<T>::LockedTensor() const
{ return tensor_; }

template<typename T>
std::vector<Int> LGridLoc() const
{
	return lGridLoc_;
}

template<typename T>
std::vector<Int> LGridShape() const
{
	return lGridShape_;
}

template<typename T>
mpi::Comm
AbstractDistTensor<T>::GetCommunicator(int index) const
{
	const int rank = mpi::CommRank( mpi::COMM_WORLD);
	mpi::Comm comm;
	std::vector<Int> logicalGridShape = this->LogicalGridShape();
	std::vector<Int> logicalGridLoc = this->LogicalGridLoc();
	const int commKey = logicalGridLoc[index];

	//Color is defined by the linear index into the logical grid EXCLUDING the index being distributed
	std::vector<Int> lGridSlice = logicalGridShape;
	lGridSlice.erase(lGridSlice.begin() + index);
	const int commColor = Dimensions2Strides(lGridSlice);

	mpi::CommSplit(mpi::COMM_WORLD, commColor, commKey, comm);
	return comm;
}

template<typename T>
void
AbstractDistTensor<T>::Empty()
{
    tensor_.Empty_();
    viewType_ = OWNER;
    dims_.clear();
    modeAlignments_.clear();
    for(int i = 0; i < order_; i++)
      constrainedModeAlignments_[i] = false;

}

template<typename T>
void
AbstractDistTensor<T>::EmptyData()
{
    tensor_.Empty_();
    viewType_ = OWNER;
    dims_.clear();
}

template<typename T>
bool
AbstractDistTensor<T>::Participating() const
{ return grid_->InGrid(); }

//
// Complex-only specializations
//

template<typename T>
BASE(T)
AbstractDistTensor<T>::GetLocalRealPart( const std::vector<Int>& loc ) const
{ return tensor_.GetRealPart(loc); }

template<typename T>
BASE(T)
AbstractDistTensor<T>::GetLocalImagPart( const std::vector<Int>& loc ) const
{ return tensor_.GetImagPart(loc); }

template<typename T>
void
AbstractDistTensor<T>::SetLocalRealPart
( const std::vector<Int>& loc, BASE(T) alpha )
{ tensor_.SetRealPart(loc,alpha); }

// HERE

template<typename T>
void
AbstractDistTensor<T>::SetLocalImagPart
( const std::vector<Int>& loc, BASE(T) alpha )
{ tensor_.SetImagPart(loc,alpha); }

template<typename T>
void
AbstractDistTensor<T>::UpdateLocalRealPart
( const std::vector<Int>& loc, BASE(T) alpha )
{ tensor_.UpdateRealPart(loc,alpha); }

template<typename T>
void
AbstractDistTensor<T>::UpdateLocalImagPart
( const std::vector<Int>& loc, BASE(T) alpha )
{ tensor_.UpdateImagPart(loc,alpha); }

template<typename T>
void
AbstractDistTensor<T>::SetShifts()
{
	//TODO: Figure out participating
    for(int i = 0; i < order_; i++)
          modeShifts_[i] = Shift(ModeRank(i), modeAlignments_[i], ModeStride(i));
/*
    if( Participating() )
    {
        for(int i = 0; i < order_; i++)
          modeShifts_[i] = Shift(ModeRank(i), modeAlignments_[i], ModeStride(i));
    }
    else
    {
        for(int i = 0; i < order_; i++)
          modeShifts_[i] = 0;
    }
*/
}

template<typename T>
void
AbstractDistTensor<T>::SetModeShift(Int mode)
{
    if( Participating() )
        modeShifts_[mode] = Shift(ModeRank(mode),modeAlignments_[mode],ModeStride(mode));
    else
        modeShifts_[mode] = 0;
}

template<typename T>
void
AbstractDistTensor<T>::SetGrid( const tmen::Grid& grid )
{
    Empty();
    grid_ = &grid; 
    SetShifts();
}

template<typename T>
void
AbstractDistTensor<T>::ComplainIfReal() const
{ 
    if( !IsComplex<T>::val )
        LogicError("Called complex-only routine with real data");
}

template<typename T>
Int
AbstractDistTensor<T>::DetermineLinearIndexOwner(const std::vector<Int>& index) const
{
#ifndef RELEASE
    CallStackEntry entry("AbstractDistTensor::DetermineLinearIndexOwner");
    this->AssertValidEntry( index );
#endif
    const tmen::Grid& g = this->Grid();
    std::vector<Int> ownerLoc(g.Order());

    for(Int i = 0; i < g.Order(); i++){
    	ownerLoc[i] = (index[i] + this->ModeAlignment(i)) % this->ModeStride(i);
    }

    return LinearIndex(ownerLoc, Dimensions2Strides(g.Shape()));
}

template<typename T>
std::vector<Int>
AbstractDistTensor<T>::Global2LocalIndex(const std::vector<Int>& index) const
{
#ifndef RELEASE
    CallStackEntry entry("AbstractDistTensor::Global2LocalIndex");
    this->AssertValidEntry( index );
#endif
    const tmen::Grid& g = this->Grid();
    std::vector<Int> loc(index.size());
    for(int i = 0; i < index.size(); i++){
    	loc[i] = (index[i]-this->ModeShift(i)) / this->ModeStride(i);
    }
    return loc;
}

template<typename T>
BASE(T)
AbstractDistTensor<T>::GetRealPart( const std::vector<Int>& loc ) const
{ return RealPart(Get(loc)); }

template<typename T>
BASE(T)
AbstractDistTensor<T>::GetImagPart( const std::vector<Int>& loc ) const
{ return ImagPart(Get(loc)); }


//TODO: Figure out how to extend this
template<typename T>
void
AbstractDistTensor<T>::MakeConsistent()
{
/*
#ifndef RELEASE
    CallStackEntry cse("AbstractDistTensor::MakeConsistent");
#endif
    const tmen::Grid& g = this->Grid();
    const Int root = g.VCToViewingMap(0);
    Int message[7];
    if( g.ViewingRank() == root )
    {
        message[0] = viewType_;
        message[1] = height_;
        message[2] = width_;
        message[3] = constrainedColAlignment_;
        message[4] = constrainedRowAlignment_;
        message[5] = colAlignment_;
        message[6] = rowAlignment_;
    }
    mpi::Broadcast( message, 7, root, g.ViewingComm() );
    const ViewType newViewType = static_cast<ViewType>(message[0]);
    const Int newHeight = message[1]; 
    const Int newWidth = message[2];
    const bool newConstrainedCol = message[3];
    const bool newConstrainedRow = message[4];
    const Int newColAlignment = message[5];
    const Int newRowAlignment = message[6];
    if( !this->Participating() )
    {
        viewType_ = newViewType;
        height_ = newHeight;
        width_ = newWidth;
        constrainedColAlignment_ = newConstrainedCol;
        constrainedRowAlignment_ = newConstrainedRow;
        colAlignment_ = newColAlignment;
        rowAlignment_ = newRowAlignment;
        colShift_ = 0;
        rowShift_ = 0;
    }
#ifndef RELEASE
    else
    {
        if( viewType_ != newViewType )
            LogicError("Inconsistent ViewType");
        if( height_ != newHeight )
            LogicError("Inconsistent height");
        if( width_ != newWidth )
            LogicError("Inconsistent width");
        if( constrainedColAlignment_ != newConstrainedCol || 
            colAlignment_ != newColAlignment )
            LogicError("Inconsistent column constraint");
        if( constrainedRowAlignment_ != newConstrainedRow ||
            rowAlignment_ != newRowAlignment )
            LogicError("Inconsistent row constraint");
    }
#endif
*/
}

#define PROTO(T) template class AbstractDistTensor<T>

PROTO(Int);
#ifndef DISABLE_FLOAT
PROTO(float);
#endif // ifndef DISABLE_FLOAT
PROTO(double);
#ifndef DISABLE_COMPLEX
#ifndef DISABLE_FLOAT
PROTO(Complex<float>);
#endif // ifndef DISABLE_FLOAT
PROTO(Complex<double>);
#endif // ifndef DISABLE_COMPLEX

#ifndef RELEASE

/*
#define CONFORMING(T) \
  template void AssertConforming1x2( const AbstractDistTensor<T>& AL, const AbstractDistTensor<T>& AR ); \
  template void AssertConforming2x1( const AbstractDistTensor<T>& AT, const AbstractDistTensor<T>& AB ); \
  template void AssertConforming2x2( const AbstractDistTensor<T>& ATL, const AbstractDistTensor<T>& ATR, const AbstractDistTensor<T>& ABL, const AbstractDistTensor<T>& ABR )
*/

/*
CONFORMING(Int);
#ifndef DISABLE_FLOAT
CONFORMING(float);
#endif // ifndef DISABLE_FLOAT
CONFORMING(double);
#ifndef DISABLE_COMPLEX
#ifndef DISABLE_FLOAT
CONFORMING(Complex<float>);
#endif // ifndef DISABLE_FLOAT
CONFORMING(Complex<double>);
#endif // ifndef DISABLE_COMPLEX
*/
#endif // ifndef RELEASE

} // namespace tmen
