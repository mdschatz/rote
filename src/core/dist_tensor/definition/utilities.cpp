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

////////////////////////////////////////////

template<typename T>
void DistTensor<T>::ElemSelectHelper(const PackData& packData, const ElemSelectData& elemData, const Mode mode, const DistTensor<T>& A, T const * const dataBuf, T * const sendBuf){
    Unsigned order = A.Order();
        PackData data = packData;
        Location elem = elemData.packElem;
        Location srcElem = elemData.srcElem;
        std::vector<Unsigned> srcStrides = elemData.srcStrides;

        std::vector<Unsigned> loopShape = elemData.loopShape;
        ModeArray commModes = elemData.commModes;
        Unsigned nElemsPerProc = elemData.nElemsPerProc;
        Unsigned i;
        const tmen::GridView gvA = A.GetGridView();
        const tmen::GridView gvB = GetGridView();
        const tmen::Grid& g = Grid();
        const Mode changedA2AMode = mode;

        Unsigned startLoc = elemData.packElem[changedA2AMode];
        for(i = 0; i < loopShape[changedA2AMode]; i++){
            elem[changedA2AMode] = startLoc + i * gvA.ModeWrapStride(changedA2AMode);
            srcElem[changedA2AMode] = i;
            if(elem[changedA2AMode] >= A.Dimension(changedA2AMode)){
                continue;
            }
            data.loopShape[changedA2AMode] = packData.loopShape[changedA2AMode] - i;

            if(mode == 0){
                Location ownerB = DetermineOwner(elem);
                Location ownerGridLoc = GridViewLoc2GridLoc(ownerB, gvB);
                Unsigned commLinLoc = Loc2LinearLoc(FilterVector(ownerGridLoc, commModes), FilterVector(g.Shape(), commModes));

                Unsigned dataBufPtr = LinearLocFromStrides(PermuteVector(srcElem, elemData.permutation), srcStrides);

                PackCommHelper(data, order - 1, &(dataBuf[dataBufPtr]), &(sendBuf[commLinLoc * nElemsPerProc]));
            }else{
                ElemSelectData newData = elemData;
                newData.packElem = elem;
                newData.srcElem = srcElem;
                ElemSelectHelper(data, newData, mode - 1, A, &(dataBuf[0]), &(sendBuf[0]));
            }
        }
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
    CallStackEntry cse("DistTensor::DetermineFirstElem");
#endif
    Unsigned i;
    Location ret(gridViewLoc.size());
    for(i = 0; i < gridViewLoc.size(); i++){
        ret[i] = Shift(gridViewLoc[i], alignmentDiff[i], ModeStride(i));
    }
//    PrintVector(ModeStrides(), "gridStrides");
//    PrintVector(alignmentDiff, "supplied align");
//    PrintVector(ret, "firstUnalignedElem");
    return ret;
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
