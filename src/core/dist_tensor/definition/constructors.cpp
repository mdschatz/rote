/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#include "rote.hpp"

namespace rote {

template<typename T>
DistTensorBase<T>::DistTensorBase( const rote::Grid& grid )
: dist_(),
  shape_(),

  constrainedModeAlignments_(),
  modeAlignments_(),
  modeShifts_(),

  tensor_(true),
  localPerm_(),

  grid_(&grid),
  commMap_(&(DefaultCommMap())),
  gridView_(grid_, dist_),
  participatingComm_(),

  viewType_(OWNER),
  auxMemory_()
{
  SetShifts(); SetParticipatingComm(); SetDefaultPermutation();}

template<typename T>
DistTensorBase<T>::DistTensorBase( const Unsigned order, const rote::Grid& grid )
: dist_(order+1),
  shape_(order, 0),

  constrainedModeAlignments_(order, 0),
  modeAlignments_(order, 0),
  modeShifts_(order, 0),

  tensor_(order, false),
  localPerm_(order),

  grid_(&grid),
  commMap_(&(DefaultCommMap())),
  gridView_(grid_, dist_),
  participatingComm_(),

  viewType_(OWNER),
  auxMemory_()
{ SetShifts(); SetParticipatingComm(); SetDefaultPermutation();}

template<typename T>
DistTensorBase<T>::DistTensorBase( const TensorDistribution& dist, const rote::Grid& grid )
: dist_(dist),
  shape_(dist_.size()-1, 0),

  constrainedModeAlignments_(shape_.size(), 0),
  modeAlignments_(shape_.size(), 0),
  modeShifts_(shape_.size(), 0),

  tensor_(shape_.size(), false),
  localPerm_(shape_.size()),

  grid_(&grid),
  commMap_(&(DefaultCommMap())),
  gridView_(grid_, dist_),
  participatingComm_(),

  viewType_(OWNER),
  auxMemory_()
{ SetShifts(); SetParticipatingComm(); SetDefaultPermutation();}

template<typename T>
DistTensorBase<T>::DistTensorBase
( const ObjShape& shape, const TensorDistribution& dist, const rote::Grid& grid )
: dist_(dist),
  shape_(shape),

  constrainedModeAlignments_(shape_.size(), 0),
  modeAlignments_(shape_.size(), 0),
  modeShifts_(shape_.size(), 0),

  tensor_(shape_.size(), false),
  localPerm_(shape_.size()),

  grid_(&grid),
  commMap_(&(DefaultCommMap())),
  gridView_(grid_, dist_),
  participatingComm_(),

  viewType_(OWNER),
  auxMemory_()
{
    if(shape_.size() + 1 != dist_.size())
        LogicError("Error: Distribution must be of same order as object");

    SetDefaultPermutation();
    SetShifts();
    ResizeTo( shape );
    SetParticipatingComm();
}

template<typename T>
DistTensorBase<T>::DistTensorBase
( const ObjShape& shape, const TensorDistribution& dist, const std::vector<Unsigned>& modeAlignments,
  const rote::Grid& g )
: dist_(dist),
  shape_(shape),

  constrainedModeAlignments_(shape_.size(), 0),
  modeAlignments_(shape_.size(), 0),
  modeShifts_(shape_.size(), 0),

  tensor_(shape_.size(), false),
  localPerm_(shape_.size()),

  grid_(&g),
  commMap_(&(DefaultCommMap())),
  gridView_(grid_, dist_),
  participatingComm_(),

  viewType_(OWNER),
  auxMemory_()
{
    if(shape_.size() + 1 != dist_.size())
        LogicError("Error: Distribution must be of same order as object");
    Align( modeAlignments );
    SetDefaultPermutation();
    dist_ = dist;
    ResizeTo( shape );
    SetParticipatingComm();
}

template<typename T>
DistTensorBase<T>::DistTensorBase
( const ObjShape& shape, const TensorDistribution& dist, const std::vector<Unsigned>& modeAlignments,
  const std::vector<Unsigned>& strides, const rote::Grid& g )
: dist_(dist),
  shape_(shape),

  constrainedModeAlignments_(shape_.size(), 0),
  modeAlignments_(shape_.size(), 0),
  modeShifts_(shape_.size(), 0),

  tensor_(shape_.size(), false),
  localPerm_(shape_.size()),

  grid_(&g),
  commMap_(&(DefaultCommMap())),
  gridView_(grid_, dist_),
  participatingComm_(),

  viewType_(OWNER),
  auxMemory_()
{
    if(shape_.size() + 1 != dist_.size())
        LogicError("Error: Distribution must be of same order as object");
    Align( modeAlignments );
    SetDefaultPermutation();
    ResizeTo( shape, strides );
    SetParticipatingComm();
}

template<typename T>
DistTensorBase<T>::DistTensorBase
( const ObjShape& shape, const TensorDistribution& dist, const std::vector<Unsigned>& modeAlignments,
  const T* buffer, const std::vector<Unsigned>& strides, const rote::Grid& g )
: dist_(dist),
  shape_(shape),

  constrainedModeAlignments_(shape_.size(), 0),
  modeAlignments_(shape_.size(), 0),
  modeShifts_(shape_.size(), 0),

  tensor_(shape_.size(), false),
  localPerm_(shape_.size()),

  grid_(&g),
  commMap_(&(DefaultCommMap())),
  gridView_(grid_, dist_),
  participatingComm_(),

  viewType_(OWNER),
  auxMemory_()
{
    if(shape_.size() + 1 != dist_.size())
        LogicError("Error: Distribution must be of same order as object");

    SetDefaultPermutation();
    LockedAttach
    ( shape, modeAlignments, buffer, strides, g );
    SetParticipatingComm();
}

template<typename T>
DistTensorBase<T>::DistTensorBase
( const ObjShape& shape, const TensorDistribution& dist, const std::vector<Unsigned>& modeAlignments,
  T* buffer, const std::vector<Unsigned>& strides, const rote::Grid& g )
: dist_(dist),
  shape_(shape),

  constrainedModeAlignments_(shape_.size(), 0),
  modeAlignments_(shape_.size(), 0),
  modeShifts_(shape_.size(), 0),

  tensor_(shape_.size(), false),
  localPerm_(shape_.size()),

  grid_(&g),
  commMap_(&(DefaultCommMap())),
  gridView_(grid_, dist_),
  participatingComm_(),

  viewType_(OWNER),
  auxMemory_()
{
    if(shape_.size() + 1 != dist_.size())
        LogicError("Error: Distribution must be of same order as object");

    SetDefaultPermutation();
    Attach
    ( shape, modeAlignments, buffer, strides, g );
    SetParticipatingComm();
}

//////////////////////////////////
/// String distribution versions
//////////////////////////////////

template<typename T>
DistTensorBase<T>::DistTensorBase( const std::string& dist, const rote::Grid& grid )
: dist_(StringToTensorDist(dist)),
  shape_(dist_.size()-1, 0),

  constrainedModeAlignments_(shape_.size(), 0),
  modeAlignments_(shape_.size(), 0),
  modeShifts_(shape_.size(), 0),

  tensor_(shape_.size(), false),
  localPerm_(shape_.size()),

  grid_(&grid),
  commMap_(&(DefaultCommMap())),
  gridView_(grid_, dist_),
  participatingComm_(),

  viewType_(OWNER),
  auxMemory_()
{ SetShifts(); SetParticipatingComm(); SetDefaultPermutation();}

template<typename T>
DistTensorBase<T>::DistTensorBase
( const ObjShape& shape, const std::string& dist, const rote::Grid& grid )
: dist_(StringToTensorDist(dist)),
  shape_(shape),

  constrainedModeAlignments_(shape_.size(), 0),
  modeAlignments_(shape_.size(), 0),
  modeShifts_(shape_.size(), 0),

  tensor_(shape_.size(), false),
  localPerm_(shape_.size()),

  grid_(&grid),
  commMap_(&(DefaultCommMap())),
  gridView_(grid_, dist_),
  participatingComm_(),

  viewType_(OWNER),
  auxMemory_()
{
    if(shape_.size() + 1 != dist_.size())
        LogicError("Error: Distribution must be of same order as object");

    SetDefaultPermutation();
    SetShifts();
    ResizeTo( shape );
    SetParticipatingComm();
}

template<typename T>
DistTensorBase<T>::DistTensorBase
( const ObjShape& shape, const std::string& dist, const std::vector<Unsigned>& modeAlignments,
  const rote::Grid& g )
: dist_(StringToTensorDist(dist)),
  shape_(shape),

  constrainedModeAlignments_(shape_.size(), 0),
  modeAlignments_(shape_.size(), 0),
  modeShifts_(shape_.size(), 0),

  tensor_(shape_.size(), false),
  localPerm_(shape_.size()),

  grid_(&g),
  commMap_(&(DefaultCommMap())),
  gridView_(grid_, dist_),
  participatingComm_(),

  viewType_(OWNER),
  auxMemory_()
{
    if(shape_.size() + 1 != dist_.size())
        LogicError("Error: Distribution must be of same order as object");
    Align( modeAlignments );
    SetDefaultPermutation();
    ResizeTo( shape );
    SetParticipatingComm();
}

template<typename T>
DistTensorBase<T>::DistTensorBase
( const ObjShape& shape, const std::string& dist, const std::vector<Unsigned>& modeAlignments,
  const std::vector<Unsigned>& strides, const rote::Grid& g )
: dist_(StringToTensorDist(dist)),
  shape_(shape),

  constrainedModeAlignments_(shape_.size(), 0),
  modeAlignments_(shape_.size(), 0),
  modeShifts_(shape_.size(), 0),

  tensor_(shape_.size(), false),
  localPerm_(shape_.size()),

  grid_(&g),
  commMap_(&(DefaultCommMap())),
  gridView_(grid_, dist_),
  participatingComm_(),

  viewType_(OWNER),
  auxMemory_()
{
    if(shape_.size() + 1 != dist_.size())
        LogicError("Error: Distribution must be of same order as object");
    Align( modeAlignments );
    SetDefaultPermutation();
    ResizeTo( shape, strides );
    SetParticipatingComm();
}

template<typename T>
DistTensorBase<T>::DistTensorBase
( const ObjShape& shape, const std::string& dist, const std::vector<Unsigned>& modeAlignments,
  const T* buffer, const std::vector<Unsigned>& strides, const rote::Grid& g )
: dist_(StringToTensorDist(dist)),
  shape_(shape),

  constrainedModeAlignments_(shape_.size(), 0),
  modeAlignments_(shape_.size(), 0),
  modeShifts_(shape_.size(), 0),

  tensor_(shape_.size(), false),
  localPerm_(shape_.size()),

  grid_(&g),
  commMap_(&(DefaultCommMap())),
  gridView_(grid_, dist_),
  participatingComm_(),

  viewType_(OWNER),
  auxMemory_()
{
    if(shape_.size() + 1 != dist_.size())
        LogicError("Error: Distribution must be of same order as object");

    SetDefaultPermutation();
    LockedAttach
    ( shape, modeAlignments, buffer, strides, g );
    SetParticipatingComm();
}

template<typename T>
DistTensorBase<T>::DistTensorBase
( const ObjShape& shape, const std::string& dist, const std::vector<Unsigned>& modeAlignments,
  T* buffer, const std::vector<Unsigned>& strides, const rote::Grid& g )
: dist_(StringToTensorDist(dist)),
  shape_(shape),

  constrainedModeAlignments_(shape_.size(), 0),
  modeAlignments_(shape_.size(), 0),
  modeShifts_(shape_.size(), 0),

  tensor_(shape_.size(), false),
  localPerm_(shape_.size()),

  grid_(&g),
  commMap_(&(DefaultCommMap())),
  gridView_(grid_, dist_),
  participatingComm_(),

  viewType_(OWNER),
  auxMemory_()
{
    if(shape_.size() + 1 != dist_.size())
        LogicError("Error: Distribution must be of same order as object");

    SetDefaultPermutation();
    Attach
    ( shape, modeAlignments, buffer, strides, g );
    SetParticipatingComm();
}

template<typename T>
DistTensorBase<T>::DistTensorBase( const DistTensorBase<T>& A )
: dist_(),
  shape_(),

  constrainedModeAlignments_(),
  modeAlignments_(),
  modeShifts_(),

  tensor_(true),
  localPerm_(shape_.size()),

  grid_(&(A.Grid())),
  commMap_(&(DefaultCommMap())),
  gridView_(grid_, dist_),
  participatingComm_(),

  viewType_(OWNER),
  auxMemory_()
{
    SetShifts();
    if( &A != this )
        *this = A;
    else
        LogicError("Tried to construct [MC,MR] with itself");
    SetDefaultPermutation();
    SetParticipatingComm();
}

template<typename T>
DistTensorBase<T>::~DistTensorBase()
{ }

template<typename T>
void
DistTensorBase<T>::Swap( DistTensorBase<T>& A )
{
    std::swap( shape_ , A.shape_ );
    std::swap( dist_, A.dist_ );

    std::swap( constrainedModeAlignments_, A.constrainedModeAlignments_ );
    std::swap( modeAlignments_, A.modeAlignments_ );
    std::swap( modeShifts_, A.modeShifts_ );

    tensor_.Swap( A.tensor_ );
    std::swap( localPerm_, A.localPerm_ );

    std::swap( grid_, A.grid_ );
    std::swap( gridView_, A.gridView_ );

    std::swap( viewType_, A.viewType_ );
    auxMemory_.Swap( A.auxMemory_ );
}

//
// Utility functions, e.g., TransposeFrom
//

//NOTE: No check for equal distributions
//NOTE: Generalize alignments mismatched case
//NOTE: Generalize CopyFromDifferentGrid case
template<typename T>
const DistTensorBase<T>&
DistTensorBase<T>::operator=( const DistTensorBase<T>& A )
{
#ifndef RELEASE
    AssertNotLocked();
#endif
    if( Grid() == A.Grid() )
    {
        if(A.Order() != Order()){
        	shape_.resize(A.Order());

            constrainedModeAlignments_.resize(shape_.size());
            modeAlignments_.resize(shape_.size());
            modeShifts_.resize(shape_.size());
            gridView_ = A.gridView_;
//            localPerm_.resize(shape_.size()); //Resizing handled in operator=
        }
        ResizeTo(A);
        if( !Participating() && !A.Participating() )
            return *this;

        if( !AnyElemwiseNotEqual(Alignments(), A.Alignments()) )
        {
            dist_ = A.TensorDist();
            tensor_ = A.LockedTensor();
            localPerm_ = A.localPerm_;
            gridView_ = A.gridView_;
            participatingComm_ = A.participatingComm_;
            grid_ = A.grid_;
            SetShifts();
        }
    }
    return *this;
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
