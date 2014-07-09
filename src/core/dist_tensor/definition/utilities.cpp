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
    CallStackEntry entry("DistTensor::DetermineOwner");
    this->AssertValidEntry( loc );
#endif
    const tmen::GridView gv = GetGridView();
    Location ownerLoc(gv.ParticipatingOrder());

    for(Int i = 0; i < gv.ParticipatingOrder(); i++){
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

    mpi::CommSplit(participatingComm_, commColor, commKey, comm);
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

    mpi::CommSplit(participatingComm_, commColor, commKey, comm);
    return comm;
}

template<typename T>
void
DistTensor<T>::SetParticipatingComm()
{
#ifndef RELEASE
    CallStackEntry cse("DistTensor::GetParticipatingComm");
#endif
    ModeArray commModes = ConcatenateVectors(gridView_.FreeModes(), gridView_.BoundModes());
    std::sort(commModes.begin(), commModes.end());

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
    participatingComm_ = comm;
}

template<typename T>
void
DistTensor<T>::CopyLocalBuffer(const DistTensor<T>& A)
{
#ifndef RELEASE
    CallStackEntry cse("DistTensor::CopyBuffer");
#endif
    tensor_.CopyBuffer(A.LockedTensor());
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


} // namespace tmen
