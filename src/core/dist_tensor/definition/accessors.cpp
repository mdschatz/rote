/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#include "rote.hpp"

namespace rote {

///////////////////////////////
// DistTensor information
///////////////////////////////

template<typename T>
Unsigned
DistTensorBase<T>::Order() const
{ return shape_.size(); }

template<typename T>
Unsigned
DistTensorBase<T>::Dimension(Mode mode) const
{ return shape_[mode]; }

template<typename T>
ObjShape
DistTensorBase<T>::Shape() const
{ return shape_; }

template<typename T>
ObjShape
DistTensorBase<T>::MaxLocalShape() const
{ return MaxLengths(Shape(), gridView_.ParticipatingShape()); }

template<typename T>
ModeDistribution
DistTensorBase<T>::ModeDist(Mode mode) const
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
DistTensorBase<T>::TensorDist() const
{
    TensorDistribution dist = dist_;
    return dist;
}

template<typename T>
bool
DistTensorBase<T>::ConstrainedModeAlignment(Mode mode) const
{
#ifndef RELEASE
    CallStackEntry cse("DistTensor::ConstrainedModeAlignment");
    const Unsigned order = Order();
    if(mode < 0 || mode >= order)
        LogicError("0 <= mode < object order must be true");
#endif
    return constrainedModeAlignments_[mode];
}

template<typename T>
Unsigned
DistTensorBase<T>::ModeAlignment(Mode mode) const
{
#ifndef RELEASE
    CallStackEntry cse("DistTensor::ModeAlignment");
    const Unsigned order = Order();
    if(mode < 0 || mode >= order)
        LogicError("0 <= mode < object order must be true");
#endif
    return modeAlignments_[mode];
}

template<typename T>
std::vector<Unsigned>
DistTensorBase<T>::Alignments() const
{
    return modeAlignments_;
}

template<typename T>
Unsigned
DistTensorBase<T>::ModeShift(Mode mode) const
{
#ifndef RELEASE
    CallStackEntry cse("DistTensor::ModeShift");
    const Unsigned order = Order();
    if(mode < 0 || mode >= order)
        LogicError("0 <= mode < object order must be true");
#endif
    return modeShifts_[mode];
}

template<typename T>
std::vector<Unsigned>
DistTensorBase<T>::ModeShifts() const
{
    return modeShifts_;
}

template<typename T>
bool
DistTensorBase<T>::Viewing() const
{ return !IsOwner( viewType_ ); }

template<typename T>
bool
DistTensorBase<T>::Locked() const
{ return IsLocked( viewType_ ); }

///////////////////////////////
// GridView pass-through
///////////////////////////////

template<typename T>
const rote::Grid&
DistTensorBase<T>::Grid() const
{ return *grid_; }

template<typename T>
const rote::GridView
DistTensorBase<T>::GetGridView() const
{
    return gridView_;
}

//NOTE: This refers to the stride within grid mode.  NOT the stride through index of elements of tensor
template<typename T>
Unsigned
DistTensorBase<T>::ModeStride(Mode mode) const
{
    return gridView_.ModeWrapStride(mode);
}

//NOTE: This refers to the stride within grid mode.  NOT the stride through index of elements of tensor
template<typename T>
std::vector<Unsigned>
DistTensorBase<T>::ModeStrides() const
{
    return gridView_.ParticipatingModeWrapStrides();
}

template<typename T>
Unsigned
DistTensorBase<T>::ModeRank(Mode mode) const
{ return gridView_.ModeLoc(mode); }

template<typename T>
Location DistTensorBase<T>::GridViewLoc() const
{ return gridView_.ParticipatingLoc(); }

template<typename T>
ObjShape DistTensorBase<T>::GridViewShape() const
{ return gridView_.ParticipatingShape(); }

template<typename T>
bool
DistTensorBase<T>::Participating() const
{ return gridView_.Participating(); }

///////////////////////////////
// Tensor pass-through
///////////////////////////////

template<typename T>
size_t
DistTensorBase<T>::AllocatedMemory() const
{ return tensor_.MemorySize(); }

template<typename T>
Unsigned
DistTensorBase<T>::LocalDimension(Mode mode) const
{ return tensor_.Dimension(mode); }

template<typename T>
ObjShape
DistTensorBase<T>::LocalShape() const
{ return tensor_.Shape(); }

template<typename T>
std::vector<Unsigned>
DistTensorBase<T>::LocalStrides() const
{ return tensor_.Strides(); }

template<typename T>
Unsigned
DistTensorBase<T>::LocalModeStride(Mode mode) const
{ return tensor_.Stride(mode); }

template<typename T>
Unsigned
DistTensorBase<T>::Stride(Mode mode) const
{ return tensor_.Stride(mode); }

template<typename T>
std::vector<Unsigned>
DistTensorBase<T>::Strides() const
{ return tensor_.Strides(); }

template<typename T>
T
DistTensorBase<T>::GetLocal( const Location& loc ) const
{
#ifndef RELEASE
    CallStackEntry entry("[MC,MR]::GetLocal");
//    AssertValidEntry( loc );
#endif
    return tensor_.Get(loc);
}

template<typename T>
T*
DistTensorBase<T>::Buffer( const Location& loc )
{ return tensor_.Buffer(loc); }

template<typename T>
T*
DistTensorBase<T>::Buffer()
{ return tensor_.Buffer(); }

template<typename T>
const T*
DistTensorBase<T>::LockedBuffer( const Location& loc ) const
{ return tensor_.LockedBuffer(loc); }

template<typename T>
const T*
DistTensorBase<T>::LockedBuffer( ) const
{ return tensor_.LockedBuffer(); }

template<typename T>
rote::Tensor<T>&
DistTensorBase<T>::Tensor()
{ return tensor_; }

template<typename T>
const rote::Tensor<T>&
DistTensorBase<T>::LockedTensor() const
{ return tensor_; }

///////////////////////////////
// Element access routines
///////////////////////////////

template<typename T>
T
DistTensorBase<T>::Get( const Location& loc ) const
{
#ifndef RELEASE
    CallStackEntry entry("[MC,MR]::Get");
    AssertValidEntry( loc );
#endif
    const rote::Grid& g = Grid();
    const Location owningProc = DetermineOwner(loc);

    const rote::GridView& gv = GetGridView();
    T u = T(0);
    if(Participating()){
        const Location ownerGridLoc = gv.ToGridLoc(owningProc);

        if(!AnyElemwiseNotEqual(g.Loc(), ownerGridLoc)){
            const Location localLoc = Global2LocalIndex(loc);
            u = GetLocal(PermuteVector(localLoc, localPerm_));
        }

        int ownerLinearLoc = gv.ToParticipatingLinearLoc(owningProc);
        mpi::Broadcast( u, ownerLinearLoc, participatingComm_);
    }

    return u;
}

template<typename T>
void
DistTensorBase<T>::GetDiagonal
( DistTensorBase<T>& d, Int offset ) const
{
	NOT_USED(d); NOT_USED(offset);
}

template<typename T>
DistTensorBase<T>
DistTensorBase<T>::GetDiagonal( Int offset ) const
{
    DistTensorBase<T> d( Grid() );
    GetDiagonal( d, offset );
    return d;
}

template<typename T>
void
DistTensorBase<T>::GetRealPartOfDiagonal
( DistTensor<BASE(T)>& d, Int offset ) const
{
	NOT_USED(d); NOT_USED(offset);
}

template<typename T>
void
DistTensorBase<T>::GetImagPartOfDiagonal
( DistTensor<BASE(T)>& d, Int offset ) const
{
	NOT_USED(d); NOT_USED(offset);
}

template<typename T>
BASE(T)
DistTensorBase<T>::GetRealPart( const Location& loc ) const
{ return RealPart(Get(loc)); }

template<typename T>
BASE(T)
DistTensorBase<T>::GetImagPart( const Location& loc ) const
{ return ImagPart(Get(loc)); }

template<typename T>
BASE(T)
DistTensorBase<T>::GetLocalRealPart( const Location& loc ) const
{ return tensor_.GetRealPart(loc); }

template<typename T>
BASE(T)
DistTensorBase<T>::GetLocalImagPart( const Location& loc ) const
{ return tensor_.GetImagPart(loc); }

template<typename T>
mpi::Comm
DistTensorBase<T>::GetParticipatingComm() const
{ return participatingComm_; }

template<typename T>
Permutation
DistTensorBase<T>::LocalPermutation() const
{
#ifndef RELEASE
    CallStackEntry cse("DistTensor::LocalPermutation");
#endif
    return localPerm_;
}

#define FULL(T) \
    template class DistTensorBase<T>;

FULL(Int)
#ifndef DISABLE_FLOAT
FULL(float)
#endif
FULL(double)

#ifndef DISABLE_COMPLEX
#ifndef DISABLE_FLOAT
FULL(std::complex<float>)
#endif
FULL(std::complex<double>)
#endif

}
