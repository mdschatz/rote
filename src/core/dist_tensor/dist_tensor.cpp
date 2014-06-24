/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#include "tensormental.hpp"

namespace tmen {

#ifndef RELEASE
template<typename T>
void
DistTensor<T>::AssertNotLocked() const
{
    if( Locked() )
        LogicError("Assertion that tensor not be a locked view failed");
}

template<typename T>
void
DistTensor<T>::AssertNotStoringData() const
{
    if( tensor_.MemorySize() > 0 )
        LogicError("Assertion that tensor not be storing data failed");
}

template<typename T>
void
DistTensor<T>::AssertValidEntry( const Location& loc ) const
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
DistTensor<T>::AssertValidSubtensor
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
DistTensor<T>::AssertSameGrid( const tmen::Grid& grid ) const
{
    if( Grid() != grid )
        LogicError("Assertion that grids match failed");
}

template<typename T>
void
DistTensor<T>::AssertSameSize( const ObjShape& shape ) const
{
    const Unsigned order = this->Order();
    if( shape.size() != order)
      LogicError("Argument must be of same order as object");
    if( AnyElemwiseNotEqual(shape, shape_) )
        LogicError("Argument must match shape of this object");
}

template<typename T>
void
DistTensor<T>::AssertMergeableModes(const std::vector<ModeArray>& oldModes) const
{
    tensor_.AssertMergeableModes(oldModes);
}

template<typename T>
void
AssertConforming2x1
( const DistTensor<T>& AT, const DistTensor<T>& AB, Mode mode )
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

template<typename T>
DistTensor<T>::DistTensor( const tmen::Grid& grid )
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
{ this->SetShifts(); }

template<typename T>
DistTensor<T>::DistTensor( const Unsigned order, const tmen::Grid& grid )
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
{ this->SetShifts(); }

template<typename T>
DistTensor<T>::DistTensor( const TensorDistribution& dist, const tmen::Grid& grid )
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
{ this->SetShifts(); }

template<typename T>
DistTensor<T>::DistTensor
( const ObjShape& shape, const TensorDistribution& dist, const tmen::Grid& grid )
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
	if(shape.size() != dist.size())
		LogicError("Error: Distribution must be of same order as object");

	this->SetShifts();
	this->ResizeTo( shape );
}

template<typename T>
DistTensor<T>::DistTensor
( const ObjShape& shape, const TensorDistribution& dist, const std::vector<Unsigned>& modeAlignments,
  const tmen::Grid& g )
: shape_(shape),
  dist_(dist),

  constrainedModeAlignments_(shape.size(), 0),
  modeAlignments_(shape.size(), 0),
  modeShifts_(shape.size(), 0),

  tensor_(shape.size(), false),

  grid_(&g),
  gridView_(grid_, dist_),

  viewType_(OWNER),
  auxMemory_()
{
	if(shape.size() != dist.size())
		LogicError("Error: Distribution must be of same order as object");
    this->Align( modeAlignments );
    this->ResizeTo( shape );
}

template<typename T>
DistTensor<T>::DistTensor
( const ObjShape& shape, const TensorDistribution& dist, const std::vector<Unsigned>& modeAlignments,
  const std::vector<Unsigned>& ldims, const tmen::Grid& g )
: shape_(shape),
  dist_(dist),

  constrainedModeAlignments_(shape.size(), 0),
  modeAlignments_(shape.size(), 0),
  modeShifts_(shape.size(), 0),

  tensor_(shape.size(), false),

  grid_(&g),
  gridView_(grid_, dist_),

  viewType_(OWNER),
  auxMemory_()
{ 
	if(shape.size() != dist.size())
		LogicError("Error: Distribution must be of same order as object");
    this->Align( modeAlignments );
    this->ResizeTo( shape, ldims );
}

template<typename T>
DistTensor<T>::DistTensor
( const ObjShape& shape, const TensorDistribution& dist, const std::vector<Unsigned>& modeAlignments,
  const T* buffer, const std::vector<Unsigned>& ldims, const tmen::Grid& g )
: shape_(shape),
  dist_(dist),

  constrainedModeAlignments_(shape.size(), 0),
  modeAlignments_(shape.size(), 0),
  modeShifts_(shape.size(), 0),

  tensor_(shape.size(), false),

  grid_(&g),
  gridView_(grid_, dist_),

  viewType_(OWNER),
  auxMemory_()
{
	if(shape.size() != dist.size())
		LogicError("Error: Distribution must be of same order as object");

    this->LockedAttach
    ( shape, modeAlignments, buffer, ldims, g );
}

template<typename T>
DistTensor<T>::DistTensor
( const ObjShape& shape, const TensorDistribution& dist, const std::vector<Unsigned>& modeAlignments,
  T* buffer, const std::vector<Unsigned>& ldims, const tmen::Grid& g )
: shape_(shape),
  dist_(dist),

  constrainedModeAlignments_(shape.size(), 0),
  modeAlignments_(shape.size(), 0),
  modeShifts_(shape.size(), 0),

  tensor_(shape.size(), false),

  grid_(&g),
  gridView_(grid_, dist_),

  viewType_(OWNER),
  auxMemory_()
{
	if(shape.size() != dist.size())
		LogicError("Error: Distribution must be of same order as object");

    this->Attach
    ( shape, modeAlignments, buffer, ldims, g );
}

template<typename T>
DistTensor<T>::DistTensor( const DistTensor<T>& A )
: shape_(),
  dist_(),

  constrainedModeAlignments_(),
  modeAlignments_(),
  modeShifts_(),

  tensor_(true),

  grid_(&(A.Grid())),
  gridView_(grid_, dist_),

  viewType_(OWNER),
  auxMemory_()
{
#ifndef RELEASE
    CallStackEntry entry("DistTensor[MC,MR]::DistTensor");
#endif
    this->SetShifts();
    if( &A != this )
        *this = A;
    else
        LogicError("Tried to construct [MC,MR] with itself");
}

template<typename T>
DistTensor<T>::~DistTensor()
{ }

template<typename T>
void
DistTensor<T>::Swap( DistTensor<T>& A )
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

template<typename T>
tmen::DistData
DistTensor<T>::DistData() const
{
    tmen::DistData data;
    //data.colDist = MC;
    //data.rowDist = MR;
    data.modeAlignments = this->modeAlignments_;
    data.distribution = this->dist_;
    data.grid = this->grid_;
    return data;
}


//NOTE: This refers to the stride within grid mode.  NOT the stride through index of elements of tensor
template<typename T>
Unsigned
DistTensor<T>::ModeStride(Mode mode) const
{
	return this->gridView_.ModeWrapStride(mode);
}

template<typename T>
Unsigned
DistTensor<T>::ModeRank(Mode mode) const
{ return this->gridView_.ModeLoc(mode); }

//TODO: Check if this should retain order of object
template<typename T>
void
DistTensor<T>::Align( const std::vector<Unsigned>& modeAlignments )
{
#ifndef RELEASE
    CallStackEntry entry("DistTensor::Align");
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
DistTensor<T>::AlignMode( Mode mode, Unsigned modeAlignment )
{
#ifndef RELEASE
    CallStackEntry entry("DistTensor::AlignMode");
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
DistTensor<T>::AlignWith( const tmen::DistData& data )
{
/*
#ifndef RELEASE
    CallStackEntry entry("[MC,MR]::AlignWith");
#endif
    const Grid& grid = *data.grid;
    this->SetGrid( grid );
    if( data.colDist == MC && data.rowDist == MR )
    {
        this->colAlignment_ = data.colAlignment;
        this->rowAlignment_ = data.rowAlignment;
        this->constrainedColAlignment_ = true;
        this->constrainedRowAlignment_ = true;
    }
    else if( data.colDist == MC && data.rowDist == STAR )
    {
        this->colAlignment_ = data.colAlignment;
        this->constrainedColAlignment_ = true; 
    }
    else if( data.colDist == MR && data.rowDist == MC )
    {
        this->colAlignment_ = data.rowAlignment;
        this->rowAlignment_ = data.colAlignment;
        this->constrainedColAlignment_ = true;
        this->constrainedRowAlignment_ = true;
    }
    else if( data.colDist == MR && data.rowDist == STAR )
    {
        this->rowAlignment_ = data.colAlignment;
        this->constrainedRowAlignment_ = true;
    }
    else if( data.colDist == STAR && data.rowDist == MC )
    {
        this->colAlignment_ = data.rowAlignment;
        this->constrainedColAlignment_ = true;
    }
    else if( data.colDist == STAR && data.rowDist == MR )
    {
        this->rowAlignment_ = data.rowAlignment;
        this->constrainedRowAlignment_ = true;
    }
    else if( data.colDist == STAR && data.rowDist == VC )
    {
        this->colAlignment_ = data.rowAlignment % this->ColStride();
        this->constrainedColAlignment_ = true;
    }
    else if( data.colDist == STAR && data.rowDist == VR )
    {
        this->rowAlignment_ = data.rowAlignment % this->RowStride();
        this->constrainedRowAlignment_ = true;
    }
    else if( data.colDist == VC && data.rowDist == STAR )
    {
        this->colAlignment_ = data.colAlignment % this->ColStride();
        this->constrainedColAlignment_ = true;
    }
    else if( data.colDist == VR && data.rowDist == STAR )
    {
        this->rowAlignment_ = data.colAlignment % this->RowStride();
        this->constrainedRowAlignment_ = true;
    }
#ifndef RELEASE
    else LogicError("Nonsensical alignment");
#endif
    this->SetShifts();
*/
}

template<typename T>
void
DistTensor<T>::AlignWith( const DistTensor<T>& A )
{ this->AlignWith( A.DistData() ); }

template<typename T>
void
DistTensor<T>::AlignModeWith( Mode mode, const tmen::DistData& data )
{
/*
#ifndef RELEASE
    CallStackEntry entry("[MC,MR]::AlignColsWith");
    if( *this->grid_ != *data.grid )
        LogicError("Grids do not match");
#endif
    if( data.colDist == MC )
        this->colAlignment_ = data.colAlignment;
    else if( data.rowDist == MC )
        this->colAlignment_ = data.rowAlignment;
    else if( data.colDist == VC )
        this->colAlignment_ = data.colAlignment % this->ColStride();
    else if( data.rowDist == VC )
        this->colAlignment_ = data.rowAlignment % this->ColStride();
#ifndef RELEASE
    else LogicError("Nonsensical alignment");
#endif
    this->constrainedColAlignment_ = true;
    this->SetShifts();
*/
}

template<typename T>
void
DistTensor<T>::AlignModeWith( Mode mode, const DistTensor<T>& A )
{ this->AlignModeWith( mode, A.DistData() ); }

template<typename T>
void
DistTensor<T>::Attach
( const ObjShape& shape, const std::vector<Unsigned>& modeAlignments,
  T* buffer, const std::vector<Unsigned>& ldims, const tmen::Grid& g )
{
/*
#ifndef RELEASE
    CallStackEntry entry("[MC,MR]::Attach");
#endif
    this->Empty();

    this->grid_ = &g;
    this->shape_ = dims;
    this->modeAlignments_ = modeAligns;
    this->viewType_ = VIEW;
    this->SetShifts();
    if( this->Participating() )
    {
        Int localHeight = Length(height,this->colShift_,this->ColStride());
        Int localWidth = Length(width,this->rowShift_,this->RowStride());
        this->matrix_.Attach_( localHeight, localWidth, buffer, ldim );
    }
*/
}

template<typename T>
void
DistTensor<T>::LockedAttach
( const ObjShape& shape, const std::vector<Unsigned>& modeAlignments,
  const T* buffer, const std::vector<Unsigned>& ldims, const tmen::Grid& g )
{
#ifndef RELEASE
    CallStackEntry entry("[MC,MR]::LockedAttach");
#endif
    this->grid_ = &g;
    this->shape_ = shape;
    this->modeAlignments_ = modeAlignments;
    this->SetShifts();
/*
    this->Empty();

    this->grid_ = &g;
    this->height_ = height;
    this->width_ = width;
    this->colAlignment_ = colAlignment;
    this->rowAlignment_ = rowAlignment;
    this->viewType_ = LOCKED_VIEW;
    this->SetShifts();
    if( this->Participating() )
    {
        Int localHeight = Length(height,this->colShift_,this->ColStride());
        Int localWidth = Length(width,this->rowShift_,this->RowStride());
        this->matrix_.LockedAttach_( localHeight, localWidth, buffer, ldim );
    }
*/
}

template<typename T>
void DistTensor<T>::ResizeTo( const DistTensor<T>& A)
{
#ifndef RELEASE
    CallStackEntry entry("[MC,MR]::ResizeTo");
    this->AssertNotLocked();
#endif
    ResizeTo(A.Shape());
}

//TODO: FIX Participating
template<typename T>
void
DistTensor<T>::ResizeTo( const ObjShape& shape )
{
#ifndef RELEASE
    CallStackEntry entry("[MC,MR]::ResizeTo");
    this->AssertNotLocked();
#endif
    this->shape_ = shape;
    this->tensor_.ResizeTo(Lengths(shape, this->modeShifts_, this->gridView_.Shape()));
/*
    this->height_ = height;
    this->width_ = width;
    if( this->Participating() )
        this->matrix_.ResizeTo_
        ( Length(height,this->ColShift(),this->ColStride()),
          Length(width, this->RowShift(),this->RowStride()) );
*/
}

template<typename T>
void
DistTensor<T>::ResizeTo( const ObjShape& shape, const std::vector<Unsigned>& ldims )
{
#ifndef RELEASE
    CallStackEntry entry("[MC,MR]::ResizeTo");
    this->AssertNotLocked();
#endif
    this->shape_ = shape;
/*
    this->height_ = height;
    this->width_ = width;
    if( this->Participating() )
        this->matrix_.ResizeTo_
        ( Length(height,this->ColShift(),this->ColStride()),
          Length(width, this->RowShift(),this->RowStride()), ldim );
*/
}


//NOTE: INCREDIBLY INEFFICIENT (RECREATING A COMMUNICATOR ON EVERY REQUEST!!!
//TODO: FIX THIS
template<typename T>
T
DistTensor<T>::Get( const Location& loc ) const
{
#ifndef RELEASE
    CallStackEntry entry("[MC,MR]::Get");
    this->AssertValidEntry( loc );
#endif
    const Location owningProc = this->DetermineOwner(loc);

    const tmen::GridView& gv = GetGridView();

    T u;
    if(!AnyElemwiseNotEqual(gv.Loc(), owningProc)){
    	const Location localLoc = this->Global2LocalIndex(loc);
    	u = this->GetLocal(localLoc);
    }

    const int ownerLinearLoc = GridViewLoc2GridLinearLoc(owningProc, gv);
    mpi::Broadcast( u, ownerLinearLoc, mpi::COMM_WORLD);
    return u;
}

template<typename T>
void
DistTensor<T>::Set( const Location& loc, T u )
{
#ifndef RELEASE
    CallStackEntry entry("[MC,MR]::Set");
    this->AssertValidEntry( loc );
#endif
    const Location owningProc = this->DetermineOwner(loc);
    const GridView gv = GetGridView();

    if(!AnyElemwiseNotEqual(gv.Loc(), owningProc)){
    	const Location localLoc = this->Global2LocalIndex(loc);
        //std::ostringstream msg;
        /*
        msg << "GlobalIndex: [" << index[0];
        for(int i = 1; i < index.size(); i++)
            msg << ", " << index[i];
        msg << "] is owned by proc" << owningProc;
    	msg << " at local index [" << localLoc[0];
    	for(int i = 1; i < localLoc.size(); i++)
    	    msg << ", " << localLoc[i];
    	msg << "]\n";
    	printf("%s", msg.str().c_str());
    	*/
    	this->SetLocal(localLoc, u);
    }

}

template<typename T>
void
DistTensor<T>::Update( const Location& loc, T u )
{
#ifndef RELEASE
    CallStackEntry entry("[MC,MR]::Update");
    this->AssertValidEntry( loc );
#endif
    const GridView gv = GetGridView();
    const Location owningProc = this->DetermineOwner(loc);
    if(!AnyElemwiseNotEqual(gv.Loc(), owningProc)){
    	const Location localLoc = this->Global2LocalIndex(loc);
    	this->UpdateLocal(localLoc, u);
    }
}

template<typename T>
void
DistTensor<T>::GetDiagonal
( DistTensor<T>& d, Int offset ) const
{
/*
#ifndef RELEASE
    CallStackEntry entry("[MC,MR]::GetDiagonal");
    if( d.Viewing() )
        this->AssertSameGrid( d.Grid() );
    if( ( d.Viewing() || d.ConstrainedColAlignment() ) &&
        !d.AlignedWithDiagonal( *this, offset ) )
    {
        std::ostringstream os;
        os << mpi::WorldRank() << "\n"
           << "offset:         " << offset << "\n"
           << "colAlignment:   " << this->colAlignment_ << "\n"
           << "rowAlignment:   " << this->rowAlignment_ << "\n"
           << "d.diagPath:     " << d.diagPath_ << "\n"
           << "d.colAlignment: " << d.colAlignment_ << std::endl;
        std::cerr << os.str();
        LogicError("d must be aligned with the 'offset' diagonal");
    }
#endif

    const tmen::Grid& g = this->Grid();
    if( !d.Viewing() )
    {
        d.SetGrid( g );
        if( !d.ConstrainedColAlignment() )
            d.AlignWithDiagonal( *this, offset );
    }
    const Int diagLength = this->DiagonalLength(offset);
    d.ResizeTo( diagLength, 1 );
    if( !d.Participating() )
        return;

    Int iStart, jStart;
    const Int diagShift = d.ColShift();
    if( offset >= 0 )
    {
        iStart = diagShift;
        jStart = diagShift+offset;
    }
    else
    {
        iStart = diagShift-offset;
        jStart = diagShift;
    }

    const Int colStride = this->ColStride();
    const Int rowStride = this->RowStride();
    const Int colShift = this->ColShift();
    const Int rowShift = this->RowShift();
    const Int iLocStart = (iStart-colShift) / colStride;
    const Int jLocStart = (jStart-rowShift) / rowStride;

    const Int lcm = g.LCM();
    const Int localDiagLength = d.LocalHeight();
    T* dBuf = d.Buffer();
    const T* buffer = this->LockedBuffer();
    const Int ldim = this->LDim();

    PARALLEL_FOR
    for( Int k=0; k<localDiagLength; ++k )
    {
        const Int iLoc = iLocStart + k*(lcm/colStride);
        const Int jLoc = jLocStart + k*(lcm/rowStride);
        dBuf[k] = buffer[iLoc+jLoc*ldim];
    }
*/
}

template<typename T>
DistTensor<T>
DistTensor<T>::GetDiagonal( Int offset ) const
{
    DistTensor<T> d( this->Grid() );
    GetDiagonal( d, offset );
    return d;
}

template<typename T>
void
DistTensor<T>::SetDiagonal
( const DistTensor<T>& d, Int offset )
{
/*
#ifndef RELEASE
    CallStackEntry entry("[MC,MR]::SetDiagonal");
    this->AssertSameGrid( d.Grid() );
    if( d.Width() != 1 )
        LogicError("d must be a column vector");
    const Int diagLength = this->DiagonalLength(offset);
    if( diagLength != d.Height() )
    {
        std::ostringstream msg;
        msg << "d is not of the same length as the diagonal:\n"
            << "  A ~ " << this->Height() << " x " << this->Width() << "\n"
            << "  d ~ " << d.Height() << " x " << d.Width() << "\n"
            << "  A diag length: " << diagLength << "\n";
        LogicError( msg.str() );
    }
    if( !d.AlignedWithDiagonal( *this, offset ) )
        LogicError("d must be aligned with the 'offset' diagonal");
#endif

    if( !d.Participating() )
        return;

    Int iStart,jStart;
    const Int diagShift = d.ColShift();
    if( offset >= 0 )
    {
        iStart = diagShift;
        jStart = diagShift+offset;
    }
    else
    {
        iStart = diagShift-offset;
        jStart = diagShift;
    }

    const Int colStride = this->ColStride();
    const Int rowStride = this->RowStride();
    const Int colShift = this->ColShift();
    const Int rowShift = this->RowShift();
    const Int iLocStart = (iStart-colShift) / colStride;
    const Int jLocStart = (jStart-rowShift) / rowStride;

    const Int localDiagLength = d.LocalHeight();
    const T* dBuf = d.LockedBuffer();
    T* buffer = this->Buffer();
    const Int ldim = this->LDim();
    const Int lcm = this->Grid().LCM();
    PARALLEL_FOR
    for( Int k=0; k<localDiagLength; ++k )
    {
        const Int iLoc = iLocStart + k*(lcm/colStride);
        const Int jLoc = jLocStart + k*(lcm/rowStride);
        buffer[iLoc+jLoc*ldim] = dBuf[k];
    }
*/
}

template<typename T>
bool
DistTensor<T>::Viewing() const
{ return !IsOwner( viewType_ ); }

template<typename T>
bool
DistTensor<T>::Locked() const
{ return IsLocked( viewType_ ); }

template<typename T>
void
DistTensor<T>::SetAlignmentsAndResize
( const std::vector<Unsigned>& modeAligns, const ObjShape& shape )
{
    const Unsigned order = this->Order();
#ifndef RELEASE
    CallStackEntry cse("DistTensor::SetAlignmentsAndResize");

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
DistTensor<T>::ForceAlignmentsAndResize
( const std::vector<Unsigned>& modeAligns, const ObjShape& shape )
{
#ifndef RELEASE
    CallStackEntry cse("DistTensor::ForceAlignmentsAndResize");
#endif
    SetAlignmentsAndResize( modeAligns, shape );
    if(AnyElemwiseNotEqual(modeAlignments_, modeAligns))
        LogicError("Could not set alignments");
}

template<typename T>
void
DistTensor<T>::SetModeAlignmentAndResize
( Mode mode, Unsigned modeAlign, const ObjShape& shape )
{
#ifndef RELEASE
    CallStackEntry cse("DistTensor::SetModeAlignmentAndResize");
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
DistTensor<T>::ForceModeAlignmentAndResize
(Mode mode, Unsigned modeAlign, const ObjShape& shape  )
{
#ifndef RELEASE
    CallStackEntry cse("DistTensor::ForceColAlignmentAndResize");
    const Unsigned order = this->Order();
    if(mode < 0 || mode >= order)
        LogicError("0 <= mode < object order must be true");
#endif
    SetModeAlignmentAndResize( mode, modeAlign, shape );
    if( modeAlignments_[mode] != modeAlign )
        LogicError("Could not set mode alignment");
}

//
// Utility functions, e.g., TransposeFrom
//

template<typename T>
const DistTensor<T>&
DistTensor<T>::operator=( const DistTensor<T>& A )
{
/*
#ifndef RELEASE
    CallStackEntry entry("[MC,MR] = [MC,MR]");
    this->AssertNotLocked();
#endif
    if( this->Grid() == A.Grid() )
    {
        this->SetAlignmentsAndResize
        ( A.ColAlignment(), A.RowAlignment(), A.Height(), A.Width() );
        if( !this->Participating() && !A.Participating() )
            return *this;
        if( this->ColAlignment() == A.ColAlignment() &&
            this->RowAlignment() == A.RowAlignment() )
        {
            this->matrix_ = A.LockedTensor();
        }
        else
        {
            const tmen::Grid& g = this->Grid();
#ifdef UNALIGNED_WARNINGS
            if( g.Rank() == 0 )
                std::cerr << "Unaligned [MC,MR] <- [MC,MR]." << std::endl;
#endif
            const Int colRank = this->ColRank();
            const Int rowRank = this->RowRank();
            const Int colStride = this->ColStride();
            const Int rowStride = this->RowStride();
            const Int colAlignment = this->ColAlignment();
            const Int rowAlignment = this->RowAlignment();
            const Int colAlignmentA = A.ColAlignment();
            const Int rowAlignmentA = A.RowAlignment();
            const Int colDiff = colAlignment - colAlignmentA;
            const Int rowDiff = rowAlignment - rowAlignmentA;
            const Int sendRow = (colRank + colStride + colDiff) % colStride;
            const Int recvRow = (colRank + colStride - colDiff) % colStride;
            const Int sendCol = (rowRank + rowStride + rowDiff) % rowStride;
            const Int recvCol = (rowRank + rowStride - rowDiff) % rowStride;
            const Int sendRank = sendRow + sendCol*colStride;
            const Int recvRank = recvRow + recvCol*colStride;

            const Int localHeight = this->LocalHeight();
            const Int localWidth = this->LocalWidth();
            const Int localHeightA = A.LocalHeight();
            const Int localWidthA = A.LocalWidth();
            const Int sendSize = localHeightA*localWidthA;
            const Int recvSize = localHeight*localWidth;
            T* auxBuf = this->auxMemory_.Require( sendSize + recvSize );
            T* sendBuf = &auxBuf[0];
            T* recvBuf = &auxBuf[sendSize];

            // Pack
            const Int ALDim = A.LDim();
            const T* ABuffer = A.LockedBuffer();
            PARALLEL_FOR
            for( Int jLoc=0; jLoc<localWidthA; ++jLoc )
                MemCopy
                ( &sendBuf[jLoc*localHeightA], 
                  &ABuffer[jLoc*ALDim], localHeightA );

            // Communicate
            mpi::SendRecv
            ( sendBuf, sendSize, sendRank, 
              recvBuf, recvSize, recvRank, g.VCComm() );

            // Unpack
            T* buffer = this->Buffer();
            const Int ldim = this->LDim();
            PARALLEL_FOR
            for( Int jLoc=0; jLoc<localWidth; ++jLoc )
                MemCopy
                ( &buffer[jLoc*ldim], 
                  &recvBuf[jLoc*localHeight], localHeight );
            this->auxMemory_.Release();
        }
    }
    else // the grids don't match
    {
        CopyFromDifferentGrid( A );
    }
*/
    return *this;

}


/*
template<typename T>
void DistTensor<T>::CopyFromDifferentGrid( const DistTensor<T>& A )
{
#ifndef RELEASE
    CallStackEntry cse("[MC,MR]::CopyFromDifferentGrid");
#endif
    this->ResizeTo( A.Height(), A.Width() ); 
    // Just need to ensure that each viewing comm contains the other team's
    // owning comm. Congruence is too strong.

    // Compute the number of process rows and columns that each process 
    // needs to send to.
    const Int colStride = this->ColStride();
    const Int rowStride = this->RowStride();
    const Int colRank = this->ColRank();
    const Int rowRank = this->RowRank();
    const Int colStrideA = A.ColStride();
    const Int rowStrideA = A.RowStride();
    const Int colRankA = A.ColRank();
    const Int rowRankA = A.RowRank();
    const Int colGCD = GCD( colStride, colStrideA );
    const Int rowGCD = GCD( rowStride, rowStrideA );
    const Int colLCM = colStride*colStrideA / colGCD;
    const Int rowLCM = rowStride*rowStrideA / rowGCD;
    const Int numColSends = colStride / colGCD;
    const Int numRowSends = rowStride / rowGCD;
    const Int localColStride = colLCM / colStride;
    const Int localRowStride = rowLCM / rowStride;
    const Int localColStrideA = numColSends;
    const Int localRowStrideA = numRowSends;

    const Int colAlign = this->ColAlignment();
    const Int rowAlign = this->RowAlignment();
    const Int colAlignA = A.ColAlignment();
    const Int rowAlignA = A.RowAlignment();

    const bool inThisGrid = this->Participating();
    const bool inAGrid = A.Participating();
    if( !inThisGrid && !inAGrid )
        return;

    const Int maxSendSize = 
        (A.Height()/(colStrideA*localColStrideA)+1) * 
        (A.Width()/(rowStrideA*localRowStrideA)+1);

    // Translate the ranks from A's VC communicator to this's viewing so that
    // we can match send/recv communicators
    const int sizeA = A.Grid().Size();
    std::vector<int> rankMap(sizeA), ranks(sizeA);
    for( int j=0; j<sizeA; ++j )
        ranks[j] = j;
    mpi::Group viewingGroup;
    mpi::CommGroup( this->Grid().ViewingComm(), viewingGroup );
    mpi::GroupTranslateRanks
    ( A.Grid().OwningGroup(), sizeA, &ranks[0], viewingGroup, &rankMap[0] );

    // Have each member of A's grid individually send to all numRow x numCol
    // processes in order, while the members of this grid receive from all 
    // necessary processes at each step.
    Int requiredMemory = 0;
    if( inAGrid )
        requiredMemory += maxSendSize;
    if( inThisGrid )
        requiredMemory += maxSendSize;
    T* auxBuf = this->auxMemory_.Require( requiredMemory );
    Int offset = 0;
    T* sendBuf = &auxBuf[offset];
    if( inAGrid )
        offset += maxSendSize;
    T* recvBuf = &auxBuf[offset];

    Int recvRow = 0; // avoid compiler warnings...
    if( inAGrid )
        recvRow = (((colRankA+colStrideA-colAlignA)%colStrideA)+colAlign) % 
                  colStride;
    for( Int colSend=0; colSend<numColSends; ++colSend )
    {
        Int recvCol = 0; // avoid compiler warnings...
        if( inAGrid )
            recvCol = (((rowRankA+rowStrideA-rowAlignA)%rowStrideA)+rowAlign) % 
                      rowStride;
        for( Int rowSend=0; rowSend<numRowSends; ++rowSend )
        {
            mpi::Request sendRequest;
            // Fire off this round of non-blocking sends
            if( inAGrid )
            {
                // Pack the data
                Int sendHeight = Length(A.LocalHeight(),colSend,numColSends);
                Int sendWidth = Length(A.LocalWidth(),rowSend,numRowSends);
                const T* ABuffer = A.LockedBuffer();
                const Int ALDim = A.LDim();
                PARALLEL_FOR
                for( Int jLoc=0; jLoc<sendWidth; ++jLoc )
                {
                    const Int j = rowSend+jLoc*localRowStrideA;
                    for( Int iLoc=0; iLoc<sendHeight; ++iLoc )
                    {
                        const Int i = colSend+iLoc*localColStrideA;
                        sendBuf[iLoc+jLoc*sendHeight] = ABuffer[i+j*ALDim];
                    }
                }
                // Send data
                const Int recvVCRank = recvRow + recvCol*colStride;
                const Int recvViewingRank = 
                    this->Grid().VCToViewingMap( recvVCRank );
                mpi::ISend
                ( sendBuf, sendHeight*sendWidth, recvViewingRank,
                  this->Grid().ViewingComm(), sendRequest );
            }
            // Perform this round of recv's
            if( inThisGrid )
            {
                const Int sendColOffset = (colSend*colStrideA+colAlignA) % colStrideA;
                const Int recvColOffset = (colSend*colStrideA+colAlign) % colStride;
                const Int sendRowOffset = (rowSend*rowStrideA+rowAlignA) % rowStrideA;
                const Int recvRowOffset = (rowSend*rowStrideA+rowAlign) % rowStride;

                const Int firstSendRow = (((colRank+colStride-recvColOffset)%colStride)+sendColOffset)%colStrideA;
                const Int firstSendCol = (((rowRank+rowStride-recvRowOffset)%rowStride)+sendRowOffset)%rowStrideA;

                const Int colShift = (colRank+colStride-recvColOffset)%colStride;
                const Int rowShift = (rowRank+rowStride-recvRowOffset)%rowStride;
                const Int numColRecvs = Length( colStrideA, colShift, colStride ); 
                const Int numRowRecvs = Length( rowStrideA, rowShift, rowStride );

                // Recv data
                // For now, simply receive sequentially. Until we switch to 
                // nonblocking recv's, we won't be using much of the 
                // recvBuf
                Int sendRow = firstSendRow;
                for( Int colRecv=0; colRecv<numColRecvs; ++colRecv )
                {
                    const Int sendColShift = Shift( sendRow, colAlignA, colStrideA ) + colSend*colStrideA;
                    const Int sendHeight = Length( A.Height(), sendColShift, colLCM );
                    const Int localColOffset = (sendColShift-this->ColShift()) / colStride;

                    Int sendCol = firstSendCol;
                    for( Int rowRecv=0; rowRecv<numRowRecvs; ++rowRecv )
                    {
                        const Int sendRowShift = Shift( sendCol, rowAlignA, rowStrideA ) + rowSend*rowStrideA;
                        const Int sendWidth = Length( A.Width(), sendRowShift, rowLCM );
                        const Int localRowOffset = (sendRowShift-this->RowShift()) / rowStride;

                        const Int sendVCRank = sendRow+sendCol*colStrideA;
                        mpi::Recv
                        ( recvBuf, sendHeight*sendWidth, rankMap[sendVCRank],
                          this->Grid().ViewingComm() );
                        
                        // Unpack the data
                        T* buffer = this->Buffer();
                        const Int ldim = this->LDim();
                        PARALLEL_FOR
                        for( Int jLoc=0; jLoc<sendWidth; ++jLoc )
                        {
                            const Int j = localRowOffset+jLoc*localRowStride;
                            for( Int iLoc=0; iLoc<sendHeight; ++iLoc )
                            {
                                const Int i = localColOffset+iLoc*localColStride;
                                buffer[i+j*ldim] = recvBuf[iLoc+jLoc*sendHeight];
                            }
                        }
                        // Set up the next send col
                        sendCol = (sendCol + rowStride) % rowStrideA;
                    }
                    // Set up the next send row
                    sendRow = (sendRow + colStride) % colStrideA;
                }
            }
            // Ensure that this round of non-blocking sends completes
            if( inAGrid )
            {
                mpi::Wait( sendRequest );
                recvCol = (recvCol + rowStrideA) % rowStride;
            }
        }
        if( inAGrid )
            recvRow = (recvRow + colStrideA) % colStride;
    }
    this->auxMemory_.Release();
}
*/
// PAUSED PASS HERE

//
// Functions which explicitly work in the complex plane
//

template<typename T>
void
DistTensor<T>::SetRealPart( const Location& loc, BASE(T) u )
{
/*
#ifndef RELEASE
    CallStackEntry entry("[MC,MR]::SetRealPart");
    this->AssertValidEntry( i, j );
#endif
    const tmen::Grid& g = this->Grid(); 
    const Int ownerRow = (i + this->ColAlignment()) % g.Height();
    const Int ownerCol = (j + this->RowAlignment()) % g.Width();
    const Int ownerRank = ownerRow + ownerCol*g.Height();
    if( g.VCRank() == ownerRank )
    {
        const Int iLoc = (i-this->ColShift()) / g.Height();
        const Int jLoc = (j-this->RowShift()) / g.Width();
        this->SetLocalRealPart( iLoc, jLoc, u );
    }
*/
}

template<typename T>
void
DistTensor<T>::SetImagPart( const Location& loc, BASE(T) u )
{
/*
#ifndef RELEASE
    CallStackEntry entry("[MC,MR]::SetImagPart");
    this->AssertValidEntry( i, j );
#endif
    this->ComplainIfReal();
    const tmen::Grid& g = this->Grid(); 
    const Int ownerRow = (i + this->ColAlignment()) % g.Height();
    const Int ownerCol = (j + this->RowAlignment()) % g.Width();
    const Int ownerRank = ownerRow + ownerCol*g.Height();
    if( g.VCRank() == ownerRank )
    {
        const Int iLoc = (i-this->ColShift()) / g.Height();
        const Int jLoc = (j-this->RowShift()) / g.Width();
        this->SetLocalImagPart( iLoc, jLoc, u );
    }
*/
}

template<typename T>
void
DistTensor<T>::UpdateRealPart( const Location& loc, BASE(T) u )
{
/*
#ifndef RELEASE
    CallStackEntry entry("[MC,MR]::UpdateRealPart");
    this->AssertValidEntry( i, j );
#endif
    const tmen::Grid& g = this->Grid(); 
    const Int ownerRow = (i + this->ColAlignment()) % g.Height();
    const Int ownerCol = (j + this->RowAlignment()) % g.Width();
    const Int ownerRank = ownerRow + ownerCol*g.Height();
    if( g.VCRank() == ownerRank )
    {
        const Int iLoc = (i-this->ColShift()) / g.Height();
        const Int jLoc = (j-this->RowShift()) / g.Width();
        this->UpdateLocalRealPart( iLoc, jLoc, u );
    }
*/
}

template<typename T>
void
DistTensor<T>::UpdateImagPart( const Location& loc, BASE(T) u )
{
/*
#ifndef RELEASE
    CallStackEntry entry("[MC,MR]::UpdateImagPart");
    this->AssertValidEntry( i, j );
#endif
    this->ComplainIfReal();
    const tmen::Grid& g = this->Grid(); 
    const Int ownerRow = (i + this->ColAlignment()) % g.Height();
    const Int ownerCol = (j + this->RowAlignment()) % g.Width();
    const Int ownerRank = ownerRow + ownerCol*g.Height();
    if( g.VCRank() == ownerRank )
    {
        const Int iLoc = (i-this->ColShift()) / g.Height();
        const Int jLoc = (j-this->RowShift()) / g.Width();
        this->UpdateLocalImagPart( iLoc, jLoc, u );
    }
*/
}

template<typename T>
void
DistTensor<T>::GetRealPartOfDiagonal
( DistTensor<BASE(T)>& d, Int offset ) const
{
/*
#ifndef RELEASE
    CallStackEntry entry("[MC,MR]::GetRealPartOfDiagonal");
    if( d.Viewing() )
        this->AssertSameGrid( d.Grid() );
#endif
    typedef BASE(T) R;
    const tmen::Grid& g = this->Grid();
    if( !d.Viewing() )
    {
        d.SetGrid( g );
        if( !d.ConstrainedColAlignment() )
            d.AlignWithDiagonal( this->DistData(), offset );
    }
    const Int length = this->DiagonalLength( offset );
    d.ResizeTo( length, 1 );
    if( !d.Participating() )
        return;

    const Int r = g.Height();
    const Int c = g.Width();
    const Int lcm = g.LCM();
    const Int colShift = this->ColShift();
    const Int rowShift = this->RowShift();
    const Int diagShift = d.ColShift();

    Int iStart, jStart;
    if( offset >= 0 )
    {
        iStart = diagShift;
        jStart = diagShift+offset;
    }
    else
    {
        iStart = diagShift-offset;
        jStart = diagShift;
    }

    const Int iLocStart = (iStart-colShift) / r;
    const Int jLocStart = (jStart-rowShift) / c;

    const Int localDiagLength = d.LocalHeight();

    const T* thisBuffer = this->LockedBuffer();
    const Int thisLDim = this->LDim();
    R* dBuf = d.Buffer();
    PARALLEL_FOR
    for( Int k=0; k<localDiagLength; ++k )
    {
        const Int iLoc = iLocStart + k*(lcm/r);
        const Int jLoc = jLocStart + k*(lcm/c);
        dBuf[k] = RealPart(thisBuffer[iLoc+jLoc*thisLDim]);
    }
*/
}

template<typename T>
void
DistTensor<T>::GetImagPartOfDiagonal
( DistTensor<BASE(T)>& d, Int offset ) const
{
/*
#ifndef RELEASE
    CallStackEntry entry("[MC,MR]::GetImagPartOfDiagonal");
    if( d.Viewing() )
        this->AssertSameGrid( d.Grid() );
#endif
    typedef BASE(T) R;
    const tmen::Grid& g = this->Grid();
    if( !d.Viewing() )
    {
        d.SetGrid( g );
        if( !d.ConstrainedColAlignment() )
            d.AlignWithDiagonal( this->DistData(), offset );
    }
    const Int length = this->DiagonalLength( offset );
    d.ResizeTo( length, 1 );
    if( !d.Participating() )
        return;

    const Int r = g.Height();
    const Int c = g.Width();
    const Int lcm = g.LCM();
    const Int colShift = this->ColShift();
    const Int rowShift = this->RowShift();
    const Int diagShift = d.ColShift();

    Int iStart, jStart;
    if( offset >= 0 )
    {
        iStart = diagShift;
        jStart = diagShift+offset;
    }
    else
    {
        iStart = diagShift-offset;
        jStart = diagShift;
    }

    const Int iLocStart = (iStart-colShift) / r;
    const Int jLocStart = (jStart-rowShift) / c;

    const Int localDiagLength = d.LocalHeight();

    const T* thisBuffer = this->LockedBuffer();
    const Int thisLDim = this->LDim();
    R* dBuf = d.Buffer();
    PARALLEL_FOR
    for( Int k=0; k<localDiagLength; ++k )
    {
        const Int iLoc = iLocStart + k*(lcm/r);
        const Int jLoc = jLocStart + k*(lcm/c);
        dBuf[k] = ImagPart(thisBuffer[iLoc+jLoc*thisLDim]);
    }
*/
}

/*
template<typename T>
DistTensor<BASE(T)>
DistTensor<T>::GetRealPartOfDiagonal( Int offset ) const
{
    DistTensor<BASE(T)> d( this->Grid() );
    GetRealPartOfDiagonal( d, offset );
    return d;
}

template<typename T>
DistTensor<BASE(T)>
DistTensor<T>::GetImagPartOfDiagonal( Int offset ) const
{
    DistTensor<BASE(T)> d( this->Grid() );
    GetImagPartOfDiagonal( d, offset );
    return d;
}
*/

template<typename T>
void
DistTensor<T>::SetRealPartOfDiagonal
( const DistTensor<BASE(T)>& d, Int offset )
{
/*
#ifndef RELEASE
    CallStackEntry entry("[MC,MR]::SetRealPartOfDiagonal");
    this->AssertSameGrid( d.Grid() );
    if( d.Width() != 1 )
        LogicError("d must be a column vector");
    const Int length = this->DiagonalLength( offset );
    if( length != d.Height() )
    {
        std::ostringstream msg;
        msg << "d is not of the same length as the diagonal:\n"
            << "  A ~ " << this->Height() << " x " << this->Width() << "\n"
            << "  d ~ " << d.Height() << " x " << d.Width() << "\n"
            << "  A diag length: " << length << "\n";
        LogicError( msg.str() );
    }
#endif
    typedef BASE(T) R;
    if( !d.Participating() )
        return;

    const tmen::Grid& g = this->Grid();
    const Int r = g.Height();
    const Int c = g.Width();
    const Int lcm = g.LCM();
    const Int colShift = this->ColShift();
    const Int rowShift = this->RowShift();
    const Int diagShift = d.ColShift();

    Int iStart,jStart;
    if( offset >= 0 )
    {
        iStart = diagShift;
        jStart = diagShift+offset;
    }
    else
    {
        iStart = diagShift-offset;
        jStart = diagShift;
    }

    const Int iLocStart = (iStart-colShift) / r;
    const Int jLocStart = (jStart-rowShift) / c;

    const Int localDiagLength = d.LocalHeight();
    const R* dBuf = d.LockedBuffer();
    PARALLEL_FOR
    for( Int k=0; k<localDiagLength; ++k )
    {
        const Int iLoc = iLocStart + k*(lcm/r);
        const Int jLoc = jLocStart + k*(lcm/c);
        this->SetLocalRealPart( iLoc, jLoc, dBuf[k] );
    }
*/
}

template<typename T>
void
DistTensor<T>::SetImagPartOfDiagonal
( const DistTensor<BASE(T)>& d, Int offset )
{
/*
#ifndef RELEASE
    CallStackEntry entry("[MC,MR]::SetImagPartOfDiagonal");
    this->AssertSameGrid( d.Grid() );
    if( d.Width() != 1 )
        LogicError("d must be a column vector");
    const Int length = this->DiagonalLength( offset );
    if( length != d.Height() )
    {
        std::ostringstream msg;
        msg << "d is not of the same length as the diagonal:\n"
            << "  A ~ " << this->Height() << " x " << this->Width() << "\n"
            << "  d ~ " << d.Height() << " x " << d.Width() << "\n"
            << "  A diag length: " << length << "\n";
        LogicError( msg.str() );
    }
#endif
    this->ComplainIfReal();
    typedef BASE(T) R;
    if( !d.Participating() )
        return;

    const tmen::Grid& g = this->Grid();
    const Int r = g.Height();
    const Int c = g.Width();
    const Int lcm = g.LCM();
    const Int colShift = this->ColShift();
    const Int rowShift = this->RowShift();
    const Int diagShift = d.ColShift();

    Int iStart,jStart;
    if( offset >= 0 )
    {
        iStart = diagShift;
        jStart = diagShift+offset;
    }
    else
    {
        iStart = diagShift-offset;
        jStart = diagShift;
    }

    const Int iLocStart = (iStart-colShift) / r;
    const Int jLocStart = (jStart-rowShift) / c;

    const Int localDiagLength = d.LocalHeight();
    const R* dBuf = d.LockedBuffer();
    PARALLEL_FOR
    for( Int k=0; k<localDiagLength; ++k )
    {
        const Int iLoc = iLocStart + k*(lcm/r);
        const Int jLoc = jLocStart + k*(lcm/c);
        this->SetLocalImagPart( iLoc, jLoc, dBuf[k] );
    }
*/
}


template<typename T>
Unsigned
DistTensor<T>::Dimension(Mode mode) const
{ return shape_[mode]; }

template<typename T>
ObjShape
DistTensor<T>::Shape() const
{ return shape_; }

template<typename T>
Unsigned
DistTensor<T>::Order() const
{ return shape_.size(); }

template<typename T>
TensorDistribution
DistTensor<T>::TensorDist() const
{
    TensorDistribution dist = dist_;
    return dist;
}

template<typename T>
ModeDistribution
DistTensor<T>::ModeDist(Mode mode) const
{
#ifndef RELEASE
    CallStackEntry cse("DistTensor::ModeDist");
    if(mode < 0 || mode >= tensor_.Order())
        LogicError("0 <= mode < object order must be true");
#endif
    return dist_[mode];
}

template<typename T>
void
DistTensor<T>::FreeAlignments()
{
    Unsigned i;
    const Unsigned order = this->Order();
    for(i = 0; i < order; i++)
      constrainedModeAlignments_[i] = false;
}

template<typename T>
bool
DistTensor<T>::ConstrainedModeAlignment(Mode mode) const
{
#ifndef RELEASE
    CallStackEntry cse("DistTensor::ConstrainedModeAlignment");
    const Unsigned order = this->Order();
    if(mode < 0 || mode >= order)
        LogicError("0 <= mode < object order must be true");
#endif
    return constrainedModeAlignments_[mode];
}

template<typename T>
std::vector<Unsigned>
DistTensor<T>::Alignments() const
{
    return modeAlignments_;
}
template<typename T>
Unsigned
DistTensor<T>::ModeAlignment(Mode mode) const
{
#ifndef RELEASE
    CallStackEntry cse("DistTensor::ModeAlignment");
    const Unsigned order = this->Order();
    if(mode < 0 || mode >= order)
        LogicError("0 <= mode < object order must be true");
#endif
    return modeAlignments_[mode];
}

template<typename T>
Unsigned
DistTensor<T>::ModeShift(Mode mode) const
{
#ifndef RELEASE
    CallStackEntry cse("DistTensor::ModeShift");
    const Unsigned order = this->Order();
    if(mode < 0 || mode >= order)
        LogicError("0 <= mode < object order must be true");
#endif
    return modeShifts_[mode];
}

template<typename T>
std::vector<Unsigned>
DistTensor<T>::ModeShifts() const
{
    return modeShifts_;
}

template<typename T>
const tmen::Grid&
DistTensor<T>::Grid() const
{ return *grid_; }

template<typename T>
const tmen::GridView
DistTensor<T>::GetGridView() const
{
    return gridView_;
}

template<typename T>
size_t
DistTensor<T>::AllocatedMemory() const
{ return tensor_.MemorySize(); }

template<typename T>
ObjShape
DistTensor<T>::LocalShape() const
{ return tensor_.Shape(); }

template<typename T>
Unsigned
DistTensor<T>::LocalDimension(Mode mode) const
{ return tensor_.Dimension(mode); }

template<typename T>
Unsigned
DistTensor<T>::LocalModeStride(Mode mode) const
{ return tensor_.ModeStride(mode); }

template<typename T>
std::vector<Unsigned>
DistTensor<T>::LDims() const
{ return tensor_.LDims(); }

template<typename T>
Unsigned
DistTensor<T>::LDim(Mode mode) const
{ return tensor_.LDim(mode); }

template<typename T>
T
DistTensor<T>::GetLocal( const Location& loc ) const
{ return tensor_.Get(loc); }

template<typename T>
void
DistTensor<T>::SetLocal( const Location& loc, T alpha )
{ tensor_.Set(loc,alpha); }

template<typename T>
void
DistTensor<T>::UpdateLocal( const Location& loc, T alpha )
{ tensor_.Update(loc,alpha); }

template<typename T>
T*
DistTensor<T>::Buffer()
{ return tensor_.Buffer(); }

template<typename T>
T*
DistTensor<T>::Buffer( const Location& loc )
{ return tensor_.Buffer(loc); }

template<typename T>
const T*
DistTensor<T>::LockedBuffer( ) const
{ return tensor_.LockedBuffer(); }

template<typename T>
const T*
DistTensor<T>::LockedBuffer( const Location& loc ) const
{ return tensor_.LockedBuffer(loc); }

template<typename T>
tmen::Tensor<T>&
DistTensor<T>::Tensor()
{ return tensor_; }

template<typename T>
const tmen::Tensor<T>&
DistTensor<T>::LockedTensor() const
{ return tensor_; }
//
//template<typename T>
//void
//DistTensor<T>::RemoveUnitModes(const ModeArray& modes)
//{
//#ifndef RELEASE
//    CallStackEntry cse("DistTensor::RemoveUnitModes");
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
//DistTensor<T>::RemoveUnitMode(const Mode& mode)
//{
//#ifndef RELEASE
//    CallStackEntry cse("DistTensor::RemoveUnitMode");
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
//DistTensor<T>::IntroduceUnitMode(const Mode& mode)
//{
//#ifndef RELEASE
//    CallStackEntry cse("DistTensor::IntroduceUnitMode");
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
Location DistTensor<T>::GridViewLoc() const
{ return gridView_.Loc(); }

template<typename T>
ObjShape DistTensor<T>::GridViewShape() const
{ return gridView_.Shape(); }

//TODO: Differentiate between index and mode
template<typename T>
mpi::Comm
DistTensor<T>::GetCommunicator(Mode mode) const
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
DistTensor<T>::GetCommunicatorForModes(const ModeArray& commModes) const
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
DistTensor<T>::Empty()
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
DistTensor<T>::EmptyData()
{
    std::fill(shape_.begin(), shape_.end(), 0);
    std::fill(dist_.begin(), dist_.end(), ModeArray());

    tensor_.Empty_();
    viewType_ = OWNER;
}

template<typename T>
bool
DistTensor<T>::Participating() const
{ return grid_->InGrid(); }

//
// Complex-only specializations
//

template<typename T>
BASE(T)
DistTensor<T>::GetLocalRealPart( const Location& loc ) const
{ return tensor_.GetRealPart(loc); }

template<typename T>
BASE(T)
DistTensor<T>::GetLocalImagPart( const Location& loc ) const
{ return tensor_.GetImagPart(loc); }

template<typename T>
void
DistTensor<T>::SetLocalRealPart
( const Location& loc, BASE(T) alpha )
{ tensor_.SetRealPart(loc,alpha); }


template<typename T>
void
DistTensor<T>::SetLocalImagPart
( const Location& loc, BASE(T) alpha )
{ tensor_.SetImagPart(loc,alpha); }

template<typename T>
void
DistTensor<T>::UpdateLocalRealPart
( const Location& loc, BASE(T) alpha )
{ tensor_.UpdateRealPart(loc,alpha); }

template<typename T>
void
DistTensor<T>::UpdateLocalImagPart
( const Location& loc, BASE(T) alpha )
{ tensor_.UpdateImagPart(loc,alpha); }

//TODO: Figure out participating logic
template<typename T>
void
DistTensor<T>::SetShifts()
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
DistTensor<T>::SetModeShift(Mode mode)
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
DistTensor<T>::SetGrid( const tmen::Grid& grid )
{
    Empty();
    grid_ = &grid;
    SetShifts();
}

template<typename T>
void
DistTensor<T>::ComplainIfReal() const
{
    if( !IsComplex<T>::val )
        LogicError("Called complex-only routine with real data");
}

template<typename T>
Location
DistTensor<T>::DetermineOwner(const Location& loc) const
{
#ifndef RELEASE
    CallStackEntry entry("DistTensor::DetermineLinearIndexOwner");
    this->AssertValidEntry( loc );
#endif
    const tmen::GridView gv = GetGridView();
    Location ownerLoc(gv.Order());

    for(Int i = 0; i < gv.Order(); i++){
        ownerLoc[i] = (loc[i] + this->ModeAlignment(i)) % this->ModeStride(i);
    }
    return ownerLoc;
}

template<typename T>
Location
DistTensor<T>::Global2LocalIndex(const Location& globalLoc) const
{
#ifndef RELEASE
    CallStackEntry entry("DistTensor::Global2LocalIndex");
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
DistTensor<T>::GetRealPart( const Location& loc ) const
{ return RealPart(Get(loc)); }

template<typename T>
BASE(T)
DistTensor<T>::GetImagPart( const Location& loc ) const
{ return ImagPart(Get(loc)); }

//
// Redist routines
//

//template<typename T>
//void
//DistTensor<T>::UnpackAGCommRecvBuf(const T * const recvBuf, const Mode agMode, const ModeArray& redistModes, const DistTensor<T>& A)
//{
//#ifndef RELEASE
//    CallStackEntry entry("DistTensor::UnpackAGCommRecvBuf");
//#endif
//
//    T* dataBuf = Buffer();
//
//    const tmen::Grid& g = A.Grid();
//    const tmen::GridView gvA = A.GetGridView();
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
DistTensor<T>::MakeConsistent()
{
/*
#ifndef RELEASE
    CallStackEntry cse("DistTensor::MakeConsistent");
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

#define PROTO(T) template class DistTensor<T>
#define COPY(T) \
  template DistTensor<T>::DistTensor( const DistTensor<T>& A )
#define FULL(T) \
  PROTO(T);


FULL(Int);
#ifndef DISABLE_FLOAT
FULL(float);
#endif
FULL(double);

#ifndef DISABLE_COMPLEX
#ifndef DISABLE_FLOAT
FULL(Complex<float>);
#endif
FULL(Complex<double>);
#endif 

#ifndef RELEASE

#define CONFORMING(T) \
  template void AssertConforming2x1( const DistTensor<T>& AT, const DistTensor<T>& AB, Mode mode ); \

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

#endif

} // namespace tmen
