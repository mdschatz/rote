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
: shape_(),
  dist_(),

  constrainedModeAlignments_(), 
  modeAlignments_(),
  modeShifts_(),

  tensor_(true),

  grid_(&grid),
  gridView_(grid_, dist_),

  viewType_(OWNER),
  auxMemory_()
{ 

}

template<typename T>
AbstractDistTensor<T>::AbstractDistTensor( const Unsigned order, const tmen::Grid& grid )
: shape_(order, 0),
  dist_(order),

  constrainedModeAlignments_(order, 0),
  modeAlignments_(order, 0),
  modeShifts_(order, 0),

  tensor_(order, false),

  grid_(&grid),
  gridView_(grid_, dist_),

  viewType_(OWNER),
  auxMemory_()
{
}

template<typename T>
AbstractDistTensor<T>::AbstractDistTensor( const TensorDistribution& dist, const tmen::Grid& grid )
: shape_(dist.size(), 0),
  dist_(dist),

  constrainedModeAlignments_(dist_.size(), 0),
  modeAlignments_(dist_.size(), 0),
  modeShifts_(dist_.size(), 0),

  tensor_(dist_.size(), false),

  grid_(&grid),
  gridView_(grid_, dist_),

  viewType_(OWNER),
  auxMemory_()
{
}

template<typename T>
AbstractDistTensor<T>::AbstractDistTensor( const ObjShape& shape, const TensorDistribution& dist, const tmen::Grid& grid )
: shape_(shape),
  dist_(dist),

  constrainedModeAlignments_(shape.size(), 0),
  modeAlignments_(shape.size(), 0),
  modeShifts_(shape.size(), 0),

  tensor_(shape.size(), false),

  grid_(&grid),
  gridView_(grid_, dist_),

  viewType_(OWNER),
  auxMemory_()
{
}

template<typename T>
AbstractDistTensor<T>::~AbstractDistTensor() 
{ }

template<typename T>
void 
AbstractDistTensor<T>::Swap( AbstractDistTensor<T>& A )
{
    std::swap( shape_ , A.shape_ );
    std::swap( dist_, A.dist_ );

    std::swap( constrainedModeAlignments_, A.constrainedModeAlignments_ );
    std::swap( modeAlignments_, A.modeAlignments_ );
    std::swap( modeShifts_, A.modeShifts_ );

    tensor_.Swap( A.tensor_ );

    std::swap( grid_, A.grid_ );
    std::swap( gridView_, A.gridView_ );

    std::swap( viewType_, A.viewType_ );
    auxMemory_.Swap( A.auxMemory_ );
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
AbstractDistTensor<T>::AssertValidEntry( const Location& loc ) const
{
    const Unsigned order = this->Order();
    if(loc.size() != order )
    {
        LogicError("Index must be of same order as object");
    }
    if(!ElemwiseLessThan(loc, shape_))
    {
        Unsigned i;
        std::ostringstream msg;
        msg << "Entry (";
        for(i = 0; i < order - 1; i++)
          msg << loc[i] << ", ";
        msg << loc[order - 1] << ") is out of bounds of ";
        
        for(i = 0; i < order - 1; i++)
              msg << shape_[i] << " x ";
        msg << shape_[order - 1] << " tensor.";
        LogicError( msg.str() );
    }
}

//TODO: FIX ASSERTIONS
template<typename T>
void
AbstractDistTensor<T>::AssertValidSubtensor
( const Location& loc, const ObjShape& shape ) const
{
    const Unsigned order = this->Order();
    if(shape.size() != order)
        LogicError("Shape must be of same order as object");
    if(loc.size() != order)
        LogicError("Indices must be of same order as object");
    if( AnyNegativeElem(loc) )
        LogicError("Indices of subtensor must not be negative");
    if( AnyNegativeElem(shape) )
        LogicError("Dimensions of subtensor must not be negative");

    Location maxLoc(order);
    ElemwiseSum(loc, shape, maxLoc);

    if( AnyElemwiseGreaterThan(maxLoc, shape_) )
    {
        Unsigned i;
        std::ostringstream msg;
        msg << "Subtensor is out of bounds: accessing up to (";
        for(i = 0; i < order - 1; i++)
          msg << maxLoc[i] << ",";
        msg << maxLoc[order - 1] << ") of ";

        for(i = 0; i < order - 1; i++)
          msg << Dimension(i) << " x ";
        msg << Dimension(order - 1) << " tensor.";
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
AbstractDistTensor<T>::AssertSameSize( const ObjShape& shape ) const
{
    const Unsigned order = this->Order();
    if( shape.size() != order)
      LogicError("Argument must be of same order as object");
    if( AnyElemwiseNotEqual(shape, shape_) )
        LogicError("Argument must match shape of this object");
}

template<typename T>
void
AbstractDistTensor<T>::AssertMergeableModes(const std::vector<ModeArray>& oldModes) const
{
    tensor_.AssertMergeableModes(oldModes);
}

template<typename T>
void
AssertConforming2x1
( const AbstractDistTensor<T>& AT, const AbstractDistTensor<T>& AB, Mode mode )
{
    std::vector<Mode> negFilterAT(1);
    std::vector<Mode> negFilterAB(1);
    negFilterAT[0] = mode;
    negFilterAB[0] = mode;

    if( AnyElemwiseNotEqual(NegFilterVector(AT.Shape(), negFilterAT), NegFilterVector(AB.Shape(), negFilterAB)) )
    {
        Unsigned i;
        std::ostringstream msg;
        msg << "2x1 is not conformant. Top is ";
        if(AT.Order() > 0)
            msg << AT.Dimension(0);
        for(i = 1; i < AT.Order(); i++)
            msg << " x " << AT.Dimension(i);
        msg << ", bottom is ";
        if(AB.Order() > 0)
            msg << AB.Dimension(0);
        for(i = 1; i < AB.Order(); i++)
            msg << " x " << AB.Dimension(i);
        LogicError( msg.str() );
    }
    if( AnyElemwiseNotEqual(NegFilterVector(AT.Alignments(), negFilterAT), NegFilterVector(AB.Alignments(), negFilterAB)) )
        LogicError("2x1 is not aligned");
}

#endif // RELEASE

//TODO: Check if this should retain order of object
template<typename T>
void
AbstractDistTensor<T>::Align( const std::vector<Unsigned>& modeAlignments )
{ 
#ifndef RELEASE
    CallStackEntry entry("AbstractDistTensor::Align");
#endif
    Unsigned i;
    Empty();
    modeAlignments_ = modeAlignments;
    for(i = 0; i < modeAlignments.size(); i++)
      constrainedModeAlignments_[i] = true;
    SetShifts();
}

template<typename T>
void
AbstractDistTensor<T>::AlignMode( Mode mode, Unsigned modeAlignment )
{ 
#ifndef RELEASE
    CallStackEntry entry("AbstractDistTensor::AlignMode");
    const Unsigned order = this->Order();
    if(mode < 0 || mode >= order)
        LogicError("0 <= mode < object order must be true");
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
AbstractDistTensor<T>::AlignModeWith( Mode mode, const tmen::DistData& data )
{ 
#ifndef RELEASE
    CallStackEntry entry("AbstractDistTensor::AlignModeWith");
    const Unsigned order = this->Order();
    if(mode < 0 || mode >= order)
        LogicError("0 <= mode < object order must be true");
#endif
    EmptyData(); 
    modeAlignments_[mode] = 0; 
    constrainedModeAlignments_[mode] = false; 
    SetShifts(); 
}

template<typename T>
void
AbstractDistTensor<T>::AlignModeWith( Mode mode, const AbstractDistTensor<T>& A )
{ AlignModeWith( mode, A.DistData() ); }

template<typename T>
void
AbstractDistTensor<T>::SetAlignmentsAndResize
( const std::vector<Unsigned>& modeAligns, const ObjShape& shape )
{
    const Unsigned order = this->Order();
#ifndef RELEASE
    CallStackEntry cse("AbstractDistTensor::SetAlignmentsAndResize");

    if(modeAligns.size() != order)
        LogicError("modeAligns must be of same order as object");
    if(shape.size() != order)
        LogicError("shape must be of same order as object");
#endif
    if( !Viewing() )
    {
        Unsigned i;
        for(i = 0; i < order; i++){
          if(!ConstrainedModeAlignment(i)){
              modeAlignments_[i] = modeAligns[i];
              SetModeShift(i);
          }
        }
    }
    ResizeTo( shape );
}

template<typename T>
void
AbstractDistTensor<T>::ForceAlignmentsAndResize
( const std::vector<Unsigned>& modeAligns, const ObjShape& shape )
{
#ifndef RELEASE
    CallStackEntry cse("AbstractDistTensor::ForceAlignmentsAndResize");
#endif
    SetAlignmentsAndResize( modeAligns, shape );
    if(AnyElemwiseNotEqual(modeAlignments_, modeAligns))
        LogicError("Could not set alignments"); 
}

template<typename T>
void
AbstractDistTensor<T>::SetModeAlignmentAndResize
( Mode mode, Unsigned modeAlign, const ObjShape& shape )
{
#ifndef RELEASE
    CallStackEntry cse("AbstractDistTensor::SetModeAlignmentAndResize");
    const Unsigned order = this->Order();
    if(mode < 0 || mode >= order)
        LogicError("0 <= mode < object order must be true");
#endif
    if( !Viewing() && !ConstrainedModeAlignment(mode) )
    {
        modeAlignments_[mode] = modeAlign;
        SetModeShift(mode); 
    }
    ResizeTo( shape );
}

template<typename T>
void
AbstractDistTensor<T>::ForceModeAlignmentAndResize
(Mode mode, Unsigned modeAlign, const ObjShape& shape  )
{
#ifndef RELEASE
    CallStackEntry cse("AbstractDistTensor::ForceColAlignmentAndResize");
    const Unsigned order = this->Order();
    if(mode < 0 || mode >= order)
        LogicError("0 <= mode < object order must be true");
#endif
    SetModeAlignmentAndResize( mode, modeAlign, shape );
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
Unsigned
AbstractDistTensor<T>::Dimension(Mode mode) const
{ return shape_[mode]; }

template<typename T>
ObjShape
AbstractDistTensor<T>::Shape() const
{ return shape_; }

template<typename T>
Unsigned
AbstractDistTensor<T>::Order() const
{ return shape_.size(); }

template<typename T>
TensorDistribution
AbstractDistTensor<T>::TensorDist() const
{
	TensorDistribution dist = dist_;
	return dist;
}

template<typename T>
ModeDistribution
AbstractDistTensor<T>::ModeDist(Mode mode) const
{
#ifndef RELEASE
    CallStackEntry cse("AbstractDistTensor::ModeDist");
	if(mode < 0 || mode >= tensor_.Order())
		LogicError("0 <= mode < object order must be true");
#endif
	return dist_[mode];
}

template<typename T>
void
AbstractDistTensor<T>::FreeAlignments() 
{
    Unsigned i;
    const Unsigned order = this->Order();
    for(i = 0; i < order; i++)
      constrainedModeAlignments_[i] = false; 
}
    
template<typename T>
bool
AbstractDistTensor<T>::ConstrainedModeAlignment(Mode mode) const
{
#ifndef RELEASE
    CallStackEntry cse("AbstractDistTensor::ConstrainedModeAlignment");
    const Unsigned order = this->Order();
    if(mode < 0 || mode >= order)
        LogicError("0 <= mode < object order must be true");
#endif
    return constrainedModeAlignments_[mode];
}

template<typename T>
std::vector<Unsigned>
AbstractDistTensor<T>::Alignments() const
{
    return modeAlignments_;
}
template<typename T>
Unsigned
AbstractDistTensor<T>::ModeAlignment(Mode mode) const
{
#ifndef RELEASE
    CallStackEntry cse("AbstractDistTensor::ModeAlignment");
    const Unsigned order = this->Order();
    if(mode < 0 || mode >= order)
        LogicError("0 <= mode < object order must be true");
#endif
    return modeAlignments_[mode];
}

template<typename T>
Unsigned
AbstractDistTensor<T>::ModeShift(Mode mode) const
{
#ifndef RELEASE
    CallStackEntry cse("AbstractDistTensor::ModeShift");
    const Unsigned order = this->Order();
    if(mode < 0 || mode >= order)
        LogicError("0 <= mode < object order must be true");
#endif
    return modeShifts_[mode];
}

template<typename T>
std::vector<Unsigned>
AbstractDistTensor<T>::ModeShifts() const
{
    return modeShifts_;
}

template<typename T>
const tmen::Grid&
AbstractDistTensor<T>::Grid() const
{ return *grid_; }

template<typename T>
const tmen::GridView
AbstractDistTensor<T>::GridView() const
{
	return gridView_;
}

template<typename T>
size_t
AbstractDistTensor<T>::AllocatedMemory() const
{ return tensor_.MemorySize(); }

template<typename T>
ObjShape
AbstractDistTensor<T>::LocalShape() const
{ return tensor_.Shape(); }

template<typename T>
Unsigned
AbstractDistTensor<T>::LocalDimension(Mode mode) const
{ return tensor_.Dimension(mode); }

template<typename T>
Unsigned
AbstractDistTensor<T>::LocalModeStride(Mode mode) const
{ return tensor_.ModeStride(mode); }

template<typename T>
std::vector<Unsigned>
AbstractDistTensor<T>::LDims() const
{ return tensor_.LDims(); }

template<typename T>
Unsigned
AbstractDistTensor<T>::LDim(Mode mode) const
{ return tensor_.LDim(mode); }

template<typename T>
T
AbstractDistTensor<T>::GetLocal( const Location& loc ) const
{ return tensor_.Get(loc); }

template<typename T>
void
AbstractDistTensor<T>::SetLocal( const Location& loc, T alpha )
{ tensor_.Set(loc,alpha); }

template<typename T>
void
AbstractDistTensor<T>::UpdateLocal( const Location& loc, T alpha )
{ tensor_.Update(loc,alpha); }

template<typename T>
T*
AbstractDistTensor<T>::Buffer()
{ return tensor_.Buffer(); }

template<typename T>
T*
AbstractDistTensor<T>::Buffer( const Location& loc )
{ return tensor_.Buffer(loc); }

template<typename T>
const T*
AbstractDistTensor<T>::LockedBuffer( ) const
{ return tensor_.LockedBuffer(); }

template<typename T>
const T*
AbstractDistTensor<T>::LockedBuffer( const Location& loc ) const
{ return tensor_.LockedBuffer(loc); }

template<typename T>
tmen::Tensor<T>&
AbstractDistTensor<T>::Tensor()
{ return tensor_; }

template<typename T>
const tmen::Tensor<T>&
AbstractDistTensor<T>::LockedTensor() const
{ return tensor_; }
//
//template<typename T>
//void
//AbstractDistTensor<T>::RemoveUnitModes(const ModeArray& modes)
//{
//#ifndef RELEASE
//    CallStackEntry cse("AbstractDistTensor::RemoveUnitModes");
//#endif
//    Unsigned i;
//    ModeArray sorted = modes;
//    std::sort(sorted.begin(), sorted.end());
//    for(i = sorted.size() - 1; i < sorted.size(); i--){
//        shape_.erase(shape_.begin() + sorted[i]);
//        dist_.erase(dist_.begin() + sorted[i]);
//        constrainedModeAlignments_.erase(constrainedModeAlignments_.begin() + sorted[i]);
//        modeAlignments_.erase(modeAlignments_.begin() + sorted[i]);
//        modeShifts_.erase(modeShifts_.begin() + sorted[i]);
//    }
//    tensor_.RemoveUnitModes(sorted);
//    //gridView_.RemoveUnitModes(sorted);
//}
//
//template<typename T>
//void
//AbstractDistTensor<T>::RemoveUnitMode(const Mode& mode)
//{
//#ifndef RELEASE
//    CallStackEntry cse("AbstractDistTensor::RemoveUnitMode");
//#endif
//    shape_.erase(shape_.begin() + mode);
//    dist_.erase(dist_.begin() + mode);
//    constrainedModeAlignments_.erase(constrainedModeAlignments_.begin() + mode);
//    modeAlignments_.erase(modeAlignments_.begin() + mode);
//    modeShifts_.erase(modeShifts_.begin() + mode);
//
//    tensor_.RemoveUnitMode(mode);
//    //gridView_.RemoveUnitMode(mode);
//}
//
//template<typename T>
//void
//AbstractDistTensor<T>::IntroduceUnitMode(const Mode& mode)
//{
//#ifndef RELEASE
//    CallStackEntry cse("AbstractDistTensor::IntroduceUnitMode");
//#endif
//    shape_.insert(shape_.begin() + mode, 1);
//    ModeDistribution newDist(0);
//    dist_.insert(dist_.begin() + mode, newDist);
//    constrainedModeAlignments_.insert(constrainedModeAlignments_.begin() + mode, true);
//    modeAlignments_.insert(modeAlignments_.begin() + mode, 0);
//    modeShifts_.insert(modeShifts_.begin() + mode, 0);
//
//    tensor_.IntroduceUnitMode(mode);
//    //gridView_.IntroduceUnitMode(mode);
//}

template<typename T>
Location AbstractDistTensor<T>::GridViewLoc() const
{ return gridView_.Loc(); }

template<typename T>
ObjShape AbstractDistTensor<T>::GridViewShape() const
{ return gridView_.Shape(); }

//TODO: Differentiate between index and mode
template<typename T>
mpi::Comm
AbstractDistTensor<T>::GetCommunicator(Mode mode) const
{
	mpi::Comm comm;
	ObjShape gridViewSliceShape = this->GridViewShape();
	Location gridViewSliceLoc = this->GridViewLoc();
	const Unsigned commKey = gridViewSliceLoc[mode];

	//Color is defined by the linear index into the logical grid EXCLUDING the index being distributed
	gridViewSliceShape.erase(gridViewSliceShape.begin() + mode);
	gridViewSliceLoc.erase(gridViewSliceLoc.begin() + mode);
	const Unsigned commColor = Loc2LinearLoc(gridViewSliceLoc, gridViewSliceShape);

	mpi::CommSplit(mpi::COMM_WORLD, commColor, commKey, comm);
	return comm;
}

template<typename T>
mpi::Comm
AbstractDistTensor<T>::GetCommunicatorForModes(const ModeArray& commModes) const
{
    mpi::Comm comm;
    const Location gridLoc = grid_->Loc();
    const ObjShape gridShape = grid_->Shape();

    ObjShape gridSliceShape = FilterVector(gridShape, commModes);
    ObjShape gridSliceNegShape = NegFilterVector(gridShape, commModes);
    Location gridSliceLoc = FilterVector(gridLoc, commModes);
    Location gridSliceNegLoc = NegFilterVector(gridLoc, commModes);

    const Unsigned commKey = Loc2LinearLoc(gridSliceLoc, gridSliceShape);
    const Unsigned commColor = Loc2LinearLoc(gridSliceNegLoc, gridSliceNegShape);

    mpi::CommSplit(mpi::COMM_WORLD, commColor, commKey, comm);
    return comm;
}

//TODO: Figure out how to clear grid and gridView
template<typename T>
void
AbstractDistTensor<T>::Empty()
{
    std::fill(shape_.begin(), shape_.end(), 0);
    std::fill(dist_.begin(), dist_.end(), ModeArray());

    std::fill(modeAlignments_.begin(), modeAlignments_.end(), 0);
    //NOTE: C++ complains if I fill with 'false' for the boolean vector
    std::fill(constrainedModeAlignments_.begin(), constrainedModeAlignments_.end(), 0);
    std::fill(modeShifts_.begin(), modeShifts_.end(), 0);

    tensor_.Empty_();

    viewType_ = OWNER;
}

//TODO: Figure out if this is fully correct
template<typename T>
void
AbstractDistTensor<T>::EmptyData()
{
    std::fill(shape_.begin(), shape_.end(), 0);
    std::fill(dist_.begin(), dist_.end(), ModeArray());

    tensor_.Empty_();
    viewType_ = OWNER;
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
AbstractDistTensor<T>::GetLocalRealPart( const Location& loc ) const
{ return tensor_.GetRealPart(loc); }

template<typename T>
BASE(T)
AbstractDistTensor<T>::GetLocalImagPart( const Location& loc ) const
{ return tensor_.GetImagPart(loc); }

template<typename T>
void
AbstractDistTensor<T>::SetLocalRealPart
( const Location& loc, BASE(T) alpha )
{ tensor_.SetRealPart(loc,alpha); }


template<typename T>
void
AbstractDistTensor<T>::SetLocalImagPart
( const Location& loc, BASE(T) alpha )
{ tensor_.SetImagPart(loc,alpha); }

template<typename T>
void
AbstractDistTensor<T>::UpdateLocalRealPart
( const Location& loc, BASE(T) alpha )
{ tensor_.UpdateRealPart(loc,alpha); }

template<typename T>
void
AbstractDistTensor<T>::UpdateLocalImagPart
( const Location& loc, BASE(T) alpha )
{ tensor_.UpdateImagPart(loc,alpha); }

//TODO: Figure out participating logic
template<typename T>
void
AbstractDistTensor<T>::SetShifts()
{

    const Unsigned order = this->Order();
    for(Unsigned i = 0; i < order; i++)
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
AbstractDistTensor<T>::SetModeShift(Mode mode)
{
    const Unsigned order = this->Order();
    if(mode < 0 || mode >= order)
        LogicError("0 < mode <= object order must be true");

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
Location
AbstractDistTensor<T>::DetermineOwner(const Location& loc) const
{
#ifndef RELEASE
    CallStackEntry entry("AbstractDistTensor::DetermineLinearIndexOwner");
    this->AssertValidEntry( loc );
#endif
    const tmen::GridView gv = this->GridView();
    Location ownerLoc(gv.Order());

    for(Int i = 0; i < gv.Order(); i++){
    	ownerLoc[i] = (loc[i] + this->ModeAlignment(i)) % this->ModeStride(i);
    }
    return ownerLoc;
}

template<typename T>
Location
AbstractDistTensor<T>::Global2LocalIndex(const Location& globalLoc) const
{
#ifndef RELEASE
    CallStackEntry entry("AbstractDistTensor::Global2LocalIndex");
    this->AssertValidEntry( globalLoc );
#endif
    Unsigned i;
    Location localLoc(globalLoc.size());
    for(i = 0; i < globalLoc.size(); i++){
    	localLoc[i] = (globalLoc[i]-this->ModeShift(i)) / this->ModeStride(i);
    }
    return localLoc;
}

template<typename T>
BASE(T)
AbstractDistTensor<T>::GetRealPart( const Location& loc ) const
{ return RealPart(Get(loc)); }

template<typename T>
BASE(T)
AbstractDistTensor<T>::GetImagPart( const Location& loc ) const
{ return ImagPart(Get(loc)); }

//
// Redist routines
//

//template<typename T>
//void
//AbstractDistTensor<T>::UnpackAGCommRecvBuf(const T * const recvBuf, const Mode agMode, const ModeArray& redistModes, const DistTensor<T>& A)
//{
//#ifndef RELEASE
//    CallStackEntry entry("AbstractDistTensor::UnpackAGCommRecvBuf");
//#endif
//
//    T* dataBuf = Buffer();
//
//    const tmen::Grid& g = A.Grid();
//    const tmen::GridView gvA = A.GridView();
//    const tmen::GridView gvB = GridView();
//
//    const ObjShape commShape = FilterVector(g.Shape(), redistModes);
//    const Unsigned nRedistProcs = prod(commShape);
//
//    const ObjShape maxLocalShapeA = MaxLengths(A.Shape(), gvA.Shape());
//    const ObjShape maxLocalShapeB = MaxLengths(Shape(), gvB.Shape());
//
//    printf("recvBuf:");
//    for(Unsigned i = 0; i < prod(maxLocalShapeA) * nRedistProcs; i++){
//        printf(" %d", recvBuf[i]);
//    }
//    printf("\n");
//
//    const ObjShape localShapeB = LocalShape();
//
//    //Number of outer slices to unpack
//    const Unsigned nMaxOuterSlices = Max(1, prod(maxLocalShapeB, agMode + 1));
//    const Unsigned nLocalOuterSlices = prod(localShapeB, agMode + 1);
//
//    //Loop packing bounds variables
//    const Unsigned nMaxAGModeSlices = maxLocalShapeB[agMode];
//    const Unsigned nLocalAGModeSlices = localShapeB[agMode];
//    const Unsigned agModeUnpackStride = nRedistProcs;
//
//    //Variables for calculating elements to copy
//    const Unsigned maxCopySliceSize = Max(1, prod(maxLocalShapeB, 0, agMode));
//    const Unsigned copySliceSize = LocalModeStride(agMode);
//
//    //Number of processes we have to unpack from
//    const Unsigned nElemSlices = nRedistProcs;
//
//    //Loop iteration vars
//    Unsigned outerSliceNum, agModeSliceNum, elemSliceNum;  //Pack data for slice "sliceNum" (<nSlices) of wrap "wrapNum" (<nWraps) for proc "procSendNum" int offSliceRecvBuf, offWrapRecvBuf;  //Offsets used to index into sendBuf array
//    Unsigned elemRecvBufOff, elemDataBufOff;
//    Unsigned outerRecvBufOff, outerDataBufOff;  //Offsets used to index into recvBuf array
//    Unsigned agModeRecvBufOff, agModeDataBufOff;  //Offsets used to index into dataBuf array
//    Unsigned startRecvBuf, startDataBuf;
//
//    printf("MemCopy info:\n");
//    printf("    nMaxOuterSlices: %d\n", nMaxOuterSlices);
//    printf("    nMaxAGModeSlices: %d\n", nMaxAGModeSlices);
//    printf("    maxCopySliceSize: %d\n", maxCopySliceSize);
//    printf("    copySliceSize: %d\n", copySliceSize);
//    printf("    agModeUnpackStride: %d\n", agModeUnpackStride);
//    for(elemSliceNum = 0; elemSliceNum < nElemSlices; elemSliceNum++){
//        elemRecvBufOff = prod(maxLocalShapeA) * elemSliceNum;
//        elemDataBufOff = copySliceSize * elemSliceNum;
//
//        printf("      elemSliceNum: %d\n", elemSliceNum);
//        printf("      elemRecvBufOff: %d\n", elemRecvBufOff);
//        printf("      elemDataBufOff: %d\n", elemDataBufOff);
//        for(outerSliceNum = 0; outerSliceNum < nMaxOuterSlices; outerSliceNum++){
//            if(outerSliceNum >= nLocalOuterSlices)
//                break;
//            //NOTE: the weird Max() function ensures we increment the recvBuf correctly
//            //e.g. we need to ensure that we jump over all slices packed by the pack routine.  Which should be maxLocalShapeA[agModeA];
//            //For consistency, kept same structure as in PackPartialRSSendBuf
//            outerRecvBufOff = maxCopySliceSize * Max(1, (nMaxAGModeSlices - 1) / agModeUnpackStride + 1) * outerSliceNum;
//            outerDataBufOff = copySliceSize * nLocalAGModeSlices * outerSliceNum;
//
//            printf("        outerSliceNum: %d\n", outerSliceNum);
//            printf("        outerRecvBufOff: %d\n", outerRecvBufOff);
//            printf("        outerDataBufOff: %d\n", outerDataBufOff);
//            for(agModeSliceNum = 0; agModeSliceNum < nMaxAGModeSlices; agModeSliceNum += agModeUnpackStride){
//                if(agModeSliceNum + elemSliceNum >= nLocalAGModeSlices)
//                    break;
//                agModeRecvBufOff = maxCopySliceSize * (agModeSliceNum / agModeUnpackStride);
//                agModeDataBufOff = copySliceSize * agModeSliceNum;
//
//                printf("          agModeSliceNum: %d\n", agModeSliceNum);
//                printf("          agModeRecvBufOff: %d\n", agModeRecvBufOff);
//                printf("          agModeDataBufOff: %d\n", agModeDataBufOff);
//                startRecvBuf = elemRecvBufOff + outerRecvBufOff + agModeRecvBufOff;
//                startDataBuf = elemDataBufOff + outerDataBufOff + agModeDataBufOff;
//
//                printf("          startRecvBuf: %d\n", startRecvBuf);
//                printf("          startDataBuf: %d\n", startDataBuf);
//                MemCopy(&(dataBuf[startDataBuf]), &(recvBuf[startRecvBuf]), copySliceSize);
//            }
//        }
//    }
//    printf("dataBuf:");
//    for(Unsigned i = 0; i < prod(LocalShape()); i++)
//        printf(" %d", dataBuf[i]);
//    printf("\n");
//}




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


#define CONFORMING(T) \
  template void AssertConforming2x1( const AbstractDistTensor<T>& AT, const AbstractDistTensor<T>& AB, Mode mode ); \

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

#endif // ifndef RELEASE

} // namespace tmen
