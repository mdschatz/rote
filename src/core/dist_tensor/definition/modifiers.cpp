/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#include "tensormental.hpp"

namespace tmen {


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
    CallStackEntry entry("DistTensor::AlignWith");
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
    CallStackEntry entry("DistTensor::AlignColsWith");
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
#ifndef RELEASE
    CallStackEntry entry("DistTensor::Attach");
#endif
    this->Empty();

    this->grid_ = &g;
    this->shape_ = shape;
    this->modeAlignments_ = modeAlignments;
    this->viewType_ = VIEW;
    this->SetShifts();
    if( this->Participating() )
    {
        ObjShape localShape = Lengths(shape, this->ModeShifts(), this->ModeStrides());
        this->tensor_.Attach_( localShape, buffer, ldims );
    }
}

template<typename T>
void
DistTensor<T>::LockedAttach
( const ObjShape& shape, const std::vector<Unsigned>& modeAlignments,
  const T* buffer, const std::vector<Unsigned>& ldims, const tmen::Grid& g )
{
#ifndef RELEASE
    CallStackEntry entry("DistTensor::LockedAttach");
#endif
    this->grid_ = &g;
    this->shape_ = shape;
    this->modeAlignments_ = modeAlignments;
    this->SetShifts();
    if(this->Participating() ){
        ObjShape localShape = Lengths(shape, this->ModeShifts(), this->ModeStrides());
        this->tensor_.LockedAttach(localShape, buffer, ldims);
    }
}

template<typename T>
void DistTensor<T>::ResizeTo( const DistTensor<T>& A)
{
#ifndef RELEASE
    CallStackEntry entry("DistTensor::ResizeTo");
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
    CallStackEntry entry("DistTensor::ResizeTo");
    this->AssertNotLocked();
#endif
    this->shape_ = shape;
    SetShifts();
    if(this->Participating()){
        this->tensor_.ResizeTo(Lengths(shape, this->modeShifts_, this->gridView_.Shape()));
    }
}

template<typename T>
void
DistTensor<T>::ResizeTo( const ObjShape& shape, const std::vector<Unsigned>& ldims )
{
#ifndef RELEASE
    CallStackEntry entry("DistTensor::ResizeTo");
    this->AssertNotLocked();
#endif
    this->shape_ = shape;
    if(this->Participating()){
        this->tensor_.ResizeTo(Lengths(shape, this->ModeShifts(), this->ModeStrides()), ldims);
    }
}

template<typename T>
void
DistTensor<T>::Set( const Location& loc, T u )
{
#ifndef RELEASE
    CallStackEntry entry("DistTensor::Set");
    this->AssertValidEntry( loc );
#endif
    const Location owningProc = this->DetermineOwner(loc);
    const GridView gv = GetGridView();

    if(!AnyElemwiseNotEqual(gv.Loc(), owningProc)){
        const Location localLoc = this->Global2LocalIndex(loc);
        this->SetLocal(localLoc, u);
    }

}

template<typename T>
void
DistTensor<T>::Update( const Location& loc, T u )
{
#ifndef RELEASE
    CallStackEntry entry("DistTensor::Update");
    this->AssertValidEntry( loc );
#endif
    const GridView gv = GetGridView();
    const Location owningProc = this->DetermineOwner(loc);
    if(!AnyElemwiseNotEqual(gv.Loc(), owningProc)){
        const Location localLoc = this->Global2LocalIndex(loc);
        this->UpdateLocal(localLoc, u);
    }
}

//
// Functions which explicitly work in the complex plane
//

template<typename T>
void
DistTensor<T>::SetRealPart( const Location& loc, BASE(T) u )
{
/*
#ifndef RELEASE
    CallStackEntry entry("DistTensor::SetRealPart");
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
    CallStackEntry entry("DistTensor::SetImagPart");
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
    CallStackEntry entry("DistTensor::UpdateRealPart");
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
    CallStackEntry entry("DistTensor::UpdateImagPart");
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
DistTensor<T>::SetRealPartOfDiagonal
( const DistTensor<BASE(T)>& d, Int offset )
{
/*
#ifndef RELEASE
    CallStackEntry entry("DistTensor::SetRealPartOfDiagonal");
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
    CallStackEntry entry("DistTensor::SetImagPartOfDiagonal");
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
void
DistTensor<T>::SetAlignmentsAndResize
( const std::vector<Unsigned>& modeAligns, const ObjShape& shape )
{
#ifndef RELEASE
    CallStackEntry cse("DistTensor::SetAlignmentsAndResize");
#endif
    if( !Viewing() )
    {
        modeAlignments_ = modeAligns;
        SetShifts();
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

template<typename T>
void
DistTensor<T>::FreeAlignments()
{
    Unsigned i;
    const Unsigned order = this->Order();
    for(i = 0; i < order; i++)
      constrainedModeAlignments_[i] = false;
}

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
    if(Participating() ){
        for(Unsigned i = 0; i < order; i++)
              modeShifts_[i] = Shift(ModeRank(i), modeAlignments_[i], ModeStride(i));
    }else
    {
        for(Unsigned i = 0; i < order; i++)
            modeShifts_[i] = 0;
    }
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
void
DistTensor<T>::SetLocal( const Location& loc, T alpha )
{ tensor_.Set(loc,alpha); }

template<typename T>
void
DistTensor<T>::UpdateLocal( const Location& loc, T alpha )
{ tensor_.Update(loc,alpha); }

#define FULL(T) \
    template class DistTensor<T>;

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

}
