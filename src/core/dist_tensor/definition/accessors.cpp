/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#include "tensormental.hpp"

namespace tmen {

///////////////////////////////
// DistTensor information
///////////////////////////////

template<typename T>
Unsigned
DistTensor<T>::Order() const
{ return shape_.size(); }

template<typename T>
Unsigned
DistTensor<T>::Dimension(Mode mode) const
{ return shape_[mode]; }

template<typename T>
ObjShape
DistTensor<T>::Shape() const
{ return shape_; }

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
TensorDistribution
DistTensor<T>::TensorDist() const
{
    TensorDistribution dist = dist_;
    return dist;
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
std::vector<Unsigned>
DistTensor<T>::Alignments() const
{
    return modeAlignments_;
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
bool
DistTensor<T>::Viewing() const
{ return !IsOwner( viewType_ ); }

template<typename T>
bool
DistTensor<T>::Locked() const
{ return IsLocked( viewType_ ); }

///////////////////////////////
// GridView pass-through
///////////////////////////////

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

//NOTE: This refers to the stride within grid mode.  NOT the stride through index of elements of tensor
template<typename T>
Unsigned
DistTensor<T>::ModeStride(Mode mode) const
{
    return this->gridView_.ModeWrapStride(mode);
}

//NOTE: This refers to the stride within grid mode.  NOT the stride through index of elements of tensor
template<typename T>
std::vector<Unsigned>
DistTensor<T>::ModeStrides() const
{
    return this->gridView_.ParticipatingModeWrapStrides();
}

template<typename T>
Unsigned
DistTensor<T>::ModeRank(Mode mode) const
{ return this->gridView_.ModeLoc(mode); }

template<typename T>
Location DistTensor<T>::GridViewLoc() const
{ return gridView_.ParticipatingLoc(); }

template<typename T>
ObjShape DistTensor<T>::GridViewShape() const
{ return gridView_.ParticipatingShape(); }

template<typename T>
bool
DistTensor<T>::Participating() const
{ return gridView_.Participating(); }

///////////////////////////////
// Tensor pass-through
///////////////////////////////

template<typename T>
size_t
DistTensor<T>::AllocatedMemory() const
{ return tensor_.MemorySize(); }

template<typename T>
Unsigned
DistTensor<T>::LocalDimension(Mode mode) const
{ return tensor_.Dimension(mode); }

template<typename T>
ObjShape
DistTensor<T>::LocalShape() const
{ return tensor_.Shape(); }

template<typename T>
Unsigned
DistTensor<T>::LocalModeStride(Mode mode) const
{ return tensor_.ModeStride(mode); }

template<typename T>
Unsigned
DistTensor<T>::LDim(Mode mode) const
{ return tensor_.LDim(mode); }

template<typename T>
std::vector<Unsigned>
DistTensor<T>::LDims() const
{ return tensor_.LDims(); }

template<typename T>
T
DistTensor<T>::GetLocal( const Location& loc ) const
{
#ifndef RELEASE
    CallStackEntry entry("[MC,MR]::GetLocal");
    this->AssertValidEntry( loc );
#endif
    return tensor_.Get(loc);
}

template<typename T>
T*
DistTensor<T>::Buffer( const Location& loc )
{ return tensor_.Buffer(loc); }

template<typename T>
T*
DistTensor<T>::Buffer()
{ return tensor_.Buffer(); }

template<typename T>
const T*
DistTensor<T>::LockedBuffer( const Location& loc ) const
{ return tensor_.LockedBuffer(loc); }

template<typename T>
const T*
DistTensor<T>::LockedBuffer( ) const
{ return tensor_.LockedBuffer(); }

template<typename T>
tmen::Tensor<T>&
DistTensor<T>::Tensor()
{ return tensor_; }

template<typename T>
const tmen::Tensor<T>&
DistTensor<T>::LockedTensor() const
{ return tensor_; }

///////////////////////////////
// Element access routines
///////////////////////////////

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
//    PrintVector(loc, "loc");
//    PrintVector(owningProc, "owner");

    const tmen::GridView& gv = GetGridView();
    Location gvLoc = gv.ParticipatingLoc();
    T u = T(0);
    if(Participating()){
        if(!AnyElemwiseNotEqual(gv.ParticipatingLoc(), owningProc)){
            const Location localLoc = this->Global2LocalIndex(loc);
            u = this->GetLocal(localLoc);
        }

        //Get the lin loc of the owner
        Unsigned i, j;
        int ownerLinearLoc = 0;
        const TensorDistribution dist = gv.Distribution();
        const tmen::Grid* g = gv.Grid();
        const Unsigned participatingOrder = gv.ParticipatingOrder();
        ModeArray participatingComms = ConcatenateVectors(gv.FreeModes(), gv.BoundModes());
        std::sort(participatingComms.begin(), participatingComms.end());
        const Location gvParticipatingLoc = gv.ParticipatingLoc();

        ObjShape gridSlice = FilterVector(g->Shape(), participatingComms);
        Location participatingGridLoc(gridSlice.size());

        for(i = 0; i < participatingOrder; i++){
            ModeDistribution modeDist = dist[i];
            ObjShape modeSliceShape = FilterVector(g->Shape(), modeDist);
            const Location modeSliceLoc = LinearLoc2Loc(owningProc[i], modeSliceShape);

            for(j = 0; j < modeDist.size(); j++){
                int indexOfMode = std::find(participatingComms.begin(), participatingComms.end(), modeDist[j]) - participatingComms.begin();
                participatingGridLoc[indexOfMode] = modeSliceLoc[j];
            }
        }
        ownerLinearLoc = Loc2LinearLoc(participatingGridLoc, gridSlice);
        //

        //const int ownerLinearLoc = GridViewLoc2GridLinearLoc(owningProc, gv);
//        std::cout << "owner linloc" << ownerLinearLoc << std::endl;
//        std::cout << "bcastComm size" << mpi::CommSize(participatingComm_) << std::endl;
        mpi::Broadcast( u, ownerLinearLoc, participatingComm_);
    }

    return u;
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
BASE(T)
DistTensor<T>::GetRealPart( const Location& loc ) const
{ return RealPart(Get(loc)); }

template<typename T>
BASE(T)
DistTensor<T>::GetImagPart( const Location& loc ) const
{ return ImagPart(Get(loc)); }

template<typename T>
BASE(T)
DistTensor<T>::GetLocalRealPart( const Location& loc ) const
{ return tensor_.GetRealPart(loc); }

template<typename T>
BASE(T)
DistTensor<T>::GetLocalImagPart( const Location& loc ) const
{ return tensor_.GetImagPart(loc); }

template<typename T>
mpi::Comm
DistTensor<T>::GetParticipatingComm() const
{ return participatingComm_; }

template<typename T>
Unsigned
DistTensor<T>::NumElem() const
{
#ifndef RELEASE
    CallStackEntry cse("DistTensor::NumElem");
#endif
    return prod(shape_);
}

template<typename T>
Unsigned
DistTensor<T>::NumLocalElem() const
{
#ifndef RELEASE
    CallStackEntry cse("DistTensor::NumLocalElem");
#endif
    return tensor_.NumElem();
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
