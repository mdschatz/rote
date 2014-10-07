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
    const Unsigned order = Order();
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
#ifndef RELEASE
    CallStackEntry entry("DistTensor::AlignWith");
#endif
/*
#ifndef RELEASE
    CallStackEntry entry("DistTensor::AlignWith");
#endif
    const Grid& grid = *data.grid;
    SetGrid( grid );
    if( data.colDist == MC && data.rowDist == MR )
    {
        colAlignment_ = data.colAlignment;
        rowAlignment_ = data.rowAlignment;
        constrainedColAlignment_ = true;
        constrainedRowAlignment_ = true;
    }
    else if( data.colDist == MC && data.rowDist == STAR )
    {
        colAlignment_ = data.colAlignment;
        constrainedColAlignment_ = true;
    }
    else if( data.colDist == MR && data.rowDist == MC )
    {
        colAlignment_ = data.rowAlignment;
        rowAlignment_ = data.colAlignment;
        constrainedColAlignment_ = true;
        constrainedRowAlignment_ = true;
    }
    else if( data.colDist == MR && data.rowDist == STAR )
    {
        rowAlignment_ = data.colAlignment;
        constrainedRowAlignment_ = true;
    }
    else if( data.colDist == STAR && data.rowDist == MC )
    {
        colAlignment_ = data.rowAlignment;
        constrainedColAlignment_ = true;
    }
    else if( data.colDist == STAR && data.rowDist == MR )
    {
        rowAlignment_ = data.rowAlignment;
        constrainedRowAlignment_ = true;
    }
    else if( data.colDist == STAR && data.rowDist == VC )
    {
        colAlignment_ = data.rowAlignment % ColStride();
        constrainedColAlignment_ = true;
    }
    else if( data.colDist == STAR && data.rowDist == VR )
    {
        rowAlignment_ = data.rowAlignment % RowStride();
        constrainedRowAlignment_ = true;
    }
    else if( data.colDist == VC && data.rowDist == STAR )
    {
        colAlignment_ = data.colAlignment % ColStride();
        constrainedColAlignment_ = true;
    }
    else if( data.colDist == VR && data.rowDist == STAR )
    {
        rowAlignment_ = data.colAlignment % RowStride();
        constrainedRowAlignment_ = true;
    }
#ifndef RELEASE
    else LogicError("Nonsensical alignment");
#endif
    SetShifts();
*/
}

//NOTE: This needs to be generalized
template<typename T>
void
DistTensor<T>::AlignWith( const DistTensor<T>& A )
{
    Unsigned i;
    Unsigned order = A.Order();
    const tmen::Grid& grid = A.Grid();
    SetGrid( grid );

    for(i = 0; i < order; i++){
        modeAlignments_[i] = A.modeAlignments_[i] % ModeStride(i);
        constrainedModeAlignments_[i] = true;
    }
    SetShifts();
}

template<typename T>
void
DistTensor<T>::SetDistribution( const TensorDistribution& tenDist)
{
    dist_ = tenDist;
}

template<typename T>
void
DistTensor<T>::AlignModeWith( Mode mode, const tmen::DistData& data )
{
/*
#ifndef RELEASE
    CallStackEntry entry("DistTensor::AlignColsWith");
    if( *grid_ != *data.grid )
        LogicError("Grids do not match");
#endif
    if( data.colDist == MC )
        colAlignment_ = data.colAlignment;
    else if( data.rowDist == MC )
        colAlignment_ = data.rowAlignment;
    else if( data.colDist == VC )
        colAlignment_ = data.colAlignment % ColStride();
    else if( data.rowDist == VC )
        colAlignment_ = data.rowAlignment % ColStride();
#ifndef RELEASE
    else LogicError("Nonsensical alignment");
#endif
    constrainedColAlignment_ = true;
    SetShifts();
*/
}

template<typename T>
void
DistTensor<T>::AlignModeWith( Mode mode, const DistTensor<T>& A )
{ AlignModeWith( mode, A.DistData() ); }

template<typename T>
void
DistTensor<T>::AlignModeWith(Mode mode, const DistTensor<T>& A, Mode modeA)
{
#ifndef RELEASE
    CallStackEntry cse("DistTensor::AlignModeWith");
#endif
    modeAlignments_[mode] = A.modeAlignments_[modeA] % ModeStride(mode);
    constrainedModeAlignments_[mode] = true;
    SetModeShift(mode);
}

template<typename T>
void
DistTensor<T>::Attach
( const ObjShape& shape, const std::vector<Unsigned>& modeAlignments,
  T* buffer, const std::vector<Unsigned>& ldims, const tmen::Grid& g )
{
#ifndef RELEASE
    CallStackEntry entry("DistTensor::Attach");
#endif
    Empty();

    grid_ = &g;
    shape_ = shape;
    modeAlignments_ = modeAlignments;
    viewType_ = VIEW;
    SetShifts();
    if( Participating() )
    {
        ObjShape localShape = Lengths(shape, ModeShifts(), ModeStrides());
        tensor_.Attach_( localShape, buffer, ldims );
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
    grid_ = &g;
    shape_ = shape;
    modeAlignments_ = modeAlignments;
    SetShifts();
    if(Participating() ){
        ObjShape localShape = Lengths(shape, ModeShifts(), ModeStrides());
        tensor_.LockedAttach(localShape, buffer, ldims);
    }
}

template<typename T>
void DistTensor<T>::ResizeTo( const DistTensor<T>& A)
{
#ifndef RELEASE
    CallStackEntry entry("DistTensor::ResizeTo");
    AssertNotLocked();
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
    AssertNotLocked();
#endif
    shape_ = shape;
    SetShifts();
    if(Participating()){
        //Account for local permutation
        tensor_.ResizeTo(PermuteVector(Lengths(shape, modeShifts_, gridView_.ParticipatingShape()), localPerm_));
    }
}

template<typename T>
void
DistTensor<T>::ResizeTo( const ObjShape& shape, const std::vector<Unsigned>& ldims )
{
#ifndef RELEASE
    CallStackEntry entry("DistTensor::ResizeTo");
    AssertNotLocked();
#endif
    shape_ = shape;
    if(Participating()){
        tensor_.ResizeTo(Lengths(shape, ModeShifts(), ModeStrides()), ldims);
    }
}

template<typename T>
void
DistTensor<T>::Set( const Location& loc, T u )
{
#ifndef RELEASE
    CallStackEntry entry("DistTensor::Set");
    AssertValidEntry( loc );
#endif
    if(!Participating())
        return;
    const Location owningProc = DetermineOwner(loc);
    const GridView gv = GetGridView();

//    printf("Setting\n");
//    printf("val: %.3f\n", u);
//    PrintVector(owningProc, "owner");
    if(!AnyElemwiseNotEqual(gv.ParticipatingLoc(), owningProc)){
//        printf("I'm the owner\n");
        const Location localLoc = Global2LocalIndex(loc);
        SetLocal(localLoc, u);
    }
//    printf("Exiting\n");
}

template<typename T>
void
DistTensor<T>::Update( const Location& loc, T u )
{
#ifndef RELEASE
    CallStackEntry entry("DistTensor::Update");
    AssertValidEntry( loc );
#endif
    const GridView gv = GetGridView();
    const Location owningProc = DetermineOwner(loc);
    if(!AnyElemwiseNotEqual(gv.ParticipatingLoc(), owningProc)){
        const Location localLoc = Global2LocalIndex(loc);
        UpdateLocal(localLoc, u);
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
    AssertValidEntry( i, j );
#endif
    const tmen::Grid& g = Grid();
    const Int ownerRow = (i + ColAlignment()) % g.Height();
    const Int ownerCol = (j + RowAlignment()) % g.Width();
    const Int ownerRank = ownerRow + ownerCol*g.Height();
    if( g.VCRank() == ownerRank )
    {
        const Int iLoc = (i-ColShift()) / g.Height();
        const Int jLoc = (j-RowShift()) / g.Width();
        SetLocalRealPart( iLoc, jLoc, u );
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
    AssertValidEntry( i, j );
#endif
    ComplainIfReal();
    const tmen::Grid& g = Grid();
    const Int ownerRow = (i + ColAlignment()) % g.Height();
    const Int ownerCol = (j + RowAlignment()) % g.Width();
    const Int ownerRank = ownerRow + ownerCol*g.Height();
    if( g.VCRank() == ownerRank )
    {
        const Int iLoc = (i-ColShift()) / g.Height();
        const Int jLoc = (j-RowShift()) / g.Width();
        SetLocalImagPart( iLoc, jLoc, u );
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
    AssertValidEntry( i, j );
#endif
    const tmen::Grid& g = Grid();
    const Int ownerRow = (i + ColAlignment()) % g.Height();
    const Int ownerCol = (j + RowAlignment()) % g.Width();
    const Int ownerRank = ownerRow + ownerCol*g.Height();
    if( g.VCRank() == ownerRank )
    {
        const Int iLoc = (i-ColShift()) / g.Height();
        const Int jLoc = (j-RowShift()) / g.Width();
        UpdateLocalRealPart( iLoc, jLoc, u );
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
    AssertValidEntry( i, j );
#endif
    ComplainIfReal();
    const tmen::Grid& g = Grid();
    const Int ownerRow = (i + ColAlignment()) % g.Height();
    const Int ownerCol = (j + RowAlignment()) % g.Width();
    const Int ownerRank = ownerRow + ownerCol*g.Height();
    if( g.VCRank() == ownerRank )
    {
        const Int iLoc = (i-ColShift()) / g.Height();
        const Int jLoc = (j-RowShift()) / g.Width();
        UpdateLocalImagPart( iLoc, jLoc, u );
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
    AssertSameGrid( d.Grid() );
    if( d.Width() != 1 )
        LogicError("d must be a column vector");
    const Int length = DiagonalLength( offset );
    if( length != d.Height() )
    {
        std::ostringstream msg;
        msg << "d is not of the same length as the diagonal:\n"
            << "  A ~ " << Height() << " x " << Width() << "\n"
            << "  d ~ " << d.Height() << " x " << d.Width() << "\n"
            << "  A diag length: " << length << "\n";
        LogicError( msg.str() );
    }
#endif
    typedef BASE(T) R;
    if( !d.Participating() )
        return;

    const tmen::Grid& g = Grid();
    const Int r = g.Height();
    const Int c = g.Width();
    const Int lcm = g.LCM();
    const Int colShift = ColShift();
    const Int rowShift = RowShift();
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
        SetLocalRealPart( iLoc, jLoc, dBuf[k] );
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
    AssertSameGrid( d.Grid() );
    if( d.Width() != 1 )
        LogicError("d must be a column vector");
    const Int length = DiagonalLength( offset );
    if( length != d.Height() )
    {
        std::ostringstream msg;
        msg << "d is not of the same length as the diagonal:\n"
            << "  A ~ " << Height() << " x " << Width() << "\n"
            << "  d ~ " << d.Height() << " x " << d.Width() << "\n"
            << "  A diag length: " << length << "\n";
        LogicError( msg.str() );
    }
#endif
    ComplainIfReal();
    typedef BASE(T) R;
    if( !d.Participating() )
        return;

    const tmen::Grid& g = Grid();
    const Int r = g.Height();
    const Int c = g.Width();
    const Int lcm = g.LCM();
    const Int colShift = ColShift();
    const Int rowShift = RowShift();
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
        SetLocalImagPart( iLoc, jLoc, dBuf[k] );
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
    const Unsigned order = Order();
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
    const Unsigned order = Order();
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
    const Unsigned order = Order();
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
    const tmen::Grid& g = Grid();
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
    if( !Participating() )
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

    const Unsigned order = Order();
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
    const Unsigned order = Order();
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
{ tensor_.Set(PermuteVector(loc, localPerm_), alpha); }

template<typename T>
void
DistTensor<T>::UpdateLocal( const Location& loc, T alpha )
{ tensor_.Update(loc,alpha); }

template<typename T>
void
DistTensor<T>::SetLocalPermutation(const Permutation& perm)
{
#ifndef RELEASE
    CallStackEntry cse("DistTensor::SetLocalPermutation");
#endif
    localPerm_ = perm;
}

template<typename T>
void
DistTensor<T>::SetDefaultPermutation()
{
#ifndef RELEASE
    CallStackEntry cse("DistTensor::SetDefaultPermutation");
#endif
    localPerm_ = DefaultPermutation(Order());
}

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
