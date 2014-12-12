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
    AssertValidEntry( loc );
#endif
    Unsigned i;
    const tmen::GridView gv = GetGridView();
    Location ownerLoc(gv.ParticipatingOrder());

    for(i = 0; i < gv.ParticipatingOrder(); i++){
        ownerLoc[i] = (loc[i] + ModeAlignment(i)) % ModeStride(i);
    }
    return ownerLoc;
}

//TODO: Change Global2LocalIndex to incorporate localPerm_ info
template<typename T>
Location
DistTensor<T>::Global2LocalIndex(const Location& globalLoc) const
{
#ifndef RELEASE
    CallStackEntry entry("DistTensor::Global2LocalIndex");
    AssertValidEntry( globalLoc );
#endif
    Unsigned i;
    Location localLoc(globalLoc.size());
    for(i = 0; i < globalLoc.size(); i++){
        localLoc[i] = (globalLoc[i]-ModeShift(i) + ModeAlignment(i)) / ModeStride(i);
    }
    return localLoc;
}

template<typename T>
mpi::Comm
DistTensor<T>::GetCommunicatorForModes(const ModeArray& commModes, const tmen::Grid& grid)
{
    ModeArray sortedCommModes = commModes;
    std::sort(sortedCommModes.begin(), sortedCommModes.end());

    if(commMap_->count(sortedCommModes) == 0){
        mpi::Comm comm;
        const Location gridLoc = grid.Loc();
        const ObjShape gridShape = grid.Shape();

        //Determine which communicator subgroup I belong to
        ObjShape gridSliceNegShape = NegFilterVector(gridShape, sortedCommModes);
        Location gridSliceNegLoc = NegFilterVector(gridLoc, sortedCommModes);

        //Determine my rank within the communicator subgroup I belong to
        ObjShape gridSliceShape = FilterVector(gridShape, sortedCommModes);
        Location gridSliceLoc = FilterVector(gridLoc, sortedCommModes);

        //Set the comm key and color for splitting
        const Unsigned commKey = Loc2LinearLoc(gridSliceLoc, gridSliceShape);
        const Unsigned commColor = Loc2LinearLoc(gridSliceNegLoc, gridSliceNegShape);

        mpi::CommSplit(grid.OwningComm(), commColor, commKey, comm);
        (*commMap_)[sortedCommModes] = comm;
    }
    return (*commMap_)[sortedCommModes];
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

    const tmen::Grid& grid = Grid();
    participatingComm_ = GetCommunicatorForModes(commModes, grid);
}

template<typename T>
void
DistTensor<T>::CopyLocalBuffer(const DistTensor<T>& A)
{
#ifndef RELEASE
    CallStackEntry cse("DistTensor::CopyBuffer");
#endif
    tensor_.CopyBuffer(A.LockedTensor(), A.localPerm_, localPerm_);
}

template<typename T>
void
DistTensor<T>::ClearCommMap()
{
#ifndef RELEASE
    CallStackEntry cse("DistTensor::ClearCommMap");
#endif
    tmen::mpi::CommMap::iterator it;
    for(it = commMap_->begin(); it != commMap_->end(); it++){
        mpi::CommFree(it->second);
    }
    commMap_->clear();
}

template<typename T>
Unsigned
DistTensor<T>::CommMapSize()
{
#ifndef RELEASE
    CallStackEntry cse("DistTensor::ClearCommMap");
#endif
    return commMap_->size();
}

template<typename T>
Location
DistTensor<T>::DetermineFirstElem(const Location& gridViewLoc) const
{
#ifndef RELEASE
    CallStackEntry cse("DistTensor::DetermineFirstElem");
#endif
    Unsigned i;
    Location ret(gridViewLoc.size());
    for(i = 0; i < gridViewLoc.size(); i++){
        ret[i] = Shift(gridViewLoc[i], modeAlignments_[i], ModeStride(i));
    }

    return ret;
}

template<typename T>
Location
DistTensor<T>::DetermineFirstUnalignedElem(const Location& gridViewLoc, const std::vector<Unsigned>& alignmentDiff) const
{
#ifndef RELEASE
    CallStackEntry cse("DistTensor::DetermineFirstUnalignedElem");
#endif
    Unsigned i;
    Location ret(gridViewLoc.size());
    for(i = 0; i < gridViewLoc.size(); i++){
        ret[i] = Shift(gridViewLoc[i], alignmentDiff[i], ModeStride(i));
    }
    return ret;
}

template<typename T>
void
DistTensor<T>::AlignCommBufRedist(const DistTensor<T>& A, const T* unalignedSendBuf, const Unsigned sendSize, T* alignedSendBuf, const Unsigned recvSize)
{
#ifndef RELEASE
    CallStackEntry cse("DistTensor::AlignCommBufRedist");
#endif
    const tmen::Grid& g = Grid();
    GridView gvA = A.GetGridView();
    GridView gvB = GetGridView();

    Location firstOwnerA = GridViewLoc2GridLoc(A.Alignments(), gvA);
    Location firstOwnerB = GridViewLoc2GridLoc(Alignments(), gvB);

    std::vector<Unsigned> alignDiff = ElemwiseSubtract(firstOwnerA, firstOwnerB);
//            PrintVector(alignDiff, "alignDiff");
//            PrintVector(ElemwiseSum(g.Loc(), alignDiff), "alignDiff + myLoc");
    Location sendGridLoc = ElemwiseMod(ElemwiseSum(ElemwiseSubtract(g.Loc(), alignDiff), g.Shape()), g.Shape());
    Location recvGridLoc = ElemwiseMod(ElemwiseSum(ElemwiseSum(g.Loc(), alignDiff), g.Shape()), g.Shape());

    //Create the communicator to involve all processes we need to fix misalignment
    ModeArray misalignedModes;
    for(Unsigned i = 0; i < firstOwnerB.size(); i++){
        if(firstOwnerB[i] != firstOwnerA[i]){
            misalignedModes.insert(misalignedModes.end(), i);
        }
    }
    std::sort(misalignedModes.begin(), misalignedModes.end());
//            PrintVector(misalignedModes, "misalignedModes");
    mpi::Comm sendRecvComm = GetCommunicatorForModes(misalignedModes, g);
//            printf("myRank is: %d\n", mpi::CommRank(sendRecvComm));

    Location sendSliceLoc = FilterVector(sendGridLoc, misalignedModes);
    Location recvSliceLoc = FilterVector(recvGridLoc, misalignedModes);
    ObjShape gridSliceShape = FilterVector(g.Shape(), misalignedModes);

    Unsigned sendLinLoc = Loc2LinearLoc(sendSliceLoc, gridSliceShape);
    Unsigned recvLinLoc = Loc2LinearLoc(recvSliceLoc, gridSliceShape);


//            PrintVector(sendGridLoc, "sendLoc");
//            printf("sendLinloc: %d\n", sendLinLoc);
//            PrintVector(recvGridLoc, "recvLoc");
//            printf("recvLinloc: %d\n", recvLinLoc);

//            PrintArray(alignSendBuf, sendShape, "sendBuf to SendRecv");
    mpi::SendRecv(unalignedSendBuf, sendSize, sendLinLoc,
                  alignedSendBuf, recvSize, recvLinLoc, sendRecvComm);

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
FULL(std::complex<float>);
#endif
FULL(std::complex<double>);
#endif 


} // namespace tmen
