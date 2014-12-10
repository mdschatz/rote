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
ObjShape
DistTensor<T>::MaxLocalShape() const
{ return MaxLengths(Shape(), gridView_.ParticipatingShape()); }

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
    const Unsigned order = Order();
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
    const Unsigned order = Order();
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
    const Unsigned order = Order();
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
    return gridView_.ModeWrapStride(mode);
}

//NOTE: This refers to the stride within grid mode.  NOT the stride through index of elements of tensor
template<typename T>
std::vector<Unsigned>
DistTensor<T>::ModeStrides() const
{
    return gridView_.ParticipatingModeWrapStrides();
}

template<typename T>
Unsigned
DistTensor<T>::ModeRank(Mode mode) const
{ return gridView_.ModeLoc(mode); }

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
std::vector<Unsigned>
DistTensor<T>::LocalStrides() const
{ return tensor_.Strides(); }

template<typename T>
Unsigned
DistTensor<T>::LocalModeStride(Mode mode) const
{ return tensor_.Stride(mode); }

template<typename T>
Unsigned
DistTensor<T>::Stride(Mode mode) const
{ return tensor_.Stride(mode); }

template<typename T>
std::vector<Unsigned>
DistTensor<T>::Strides() const
{ return tensor_.Strides(); }

template<typename T>
T
DistTensor<T>::GetLocal( const Location& loc ) const
{
#ifndef RELEASE
    CallStackEntry entry("[MC,MR]::GetLocal");
    AssertValidEntry( loc );
#endif
    return tensor_.Get(PermuteVector(loc, localPerm_));
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

template<typename T>
T
DistTensor<T>::Get( const Location& loc ) const
{
#ifndef RELEASE
    CallStackEntry entry("[MC,MR]::Get");
    AssertValidEntry( loc );
#endif
    const tmen::Grid& g = Grid();
    const Location owningProc = DetermineOwner(loc);

    const tmen::GridView& gv = GetGridView();
    T u = T(0);
    if(Participating()){
        const Location ownerGridLoc = GridViewLoc2GridLoc(owningProc, gv);

        if(!AnyElemwiseNotEqual(g.Loc(), ownerGridLoc)){
            const Location localLoc = Global2LocalIndex(loc);
            u = GetLocal(localLoc);
        }

        int ownerLinearLoc = GridViewLoc2ParticipatingLinearLoc(owningProc, gv);
        mpi::Broadcast( u, ownerLinearLoc, participatingComm_);
    }

    return u;
}

template<typename T>
void
DistTensor<T>::GetDiagonal
( DistTensor<T>& d, Int offset ) const
{
}

template<typename T>
DistTensor<T>
DistTensor<T>::GetDiagonal( Int offset ) const
{
    DistTensor<T> d( Grid() );
    GetDiagonal( d, offset );
    return d;
}

template<typename T>
void
DistTensor<T>::GetRealPartOfDiagonal
( DistTensor<BASE(T)>& d, Int offset ) const
{
}

template<typename T>
void
DistTensor<T>::GetImagPartOfDiagonal
( DistTensor<BASE(T)>& d, Int offset ) const
{
}

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

template<typename T>
Permutation
DistTensor<T>::LocalPermutation() const
{
#ifndef RELEASE
    CallStackEntry cse("DistTensor::LocalPermutation");
#endif
    return localPerm_;
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
FULL(std::complex<float>);
#endif
FULL(std::complex<double>);
#endif

}
