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
    const tmen::GridView gv = GetGridView();
    Location ownerLoc(gv.ParticipatingOrder());

    for(Int i = 0; i < gv.ParticipatingOrder(); i++){
        ownerLoc[i] = (loc[i] + ModeAlignment(i)) % ModeStride(i);
    }
    return ownerLoc;
}

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
        localLoc[i] = (globalLoc[i]-ModeShift(i)) / ModeStride(i);
    }
    return localLoc;
}

//TODO: Differentiate between index and mode
template<typename T>
mpi::Comm
DistTensor<T>::GetCommunicator(Mode mode) const
{
    mpi::Comm comm;
    ObjShape gridViewSliceShape = GridViewShape();
    Location gridViewSliceLoc = GridViewLoc();
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
DistTensor<T>::GetCommunicatorForModes(const ModeArray& commModes, const tmen::Grid& grid)
{
    ModeArray sortedCommModes = commModes;
    std::sort(sortedCommModes.begin(), sortedCommModes.end());
//    return grid_->GetCommunicatorForModes(commModes);
//    mpi::Comm comm;
//    const Location gridLoc = grid_->Loc();
//    const ObjShape gridShape = grid_->Shape();
//
//    ObjShape gridSliceShape = FilterVector(gridShape, commModes);
//    ObjShape gridSliceNegShape = NegFilterVector(gridShape, commModes);
//    Location gridSliceLoc = FilterVector(gridLoc, commModes);
//    Location gridSliceNegLoc = NegFilterVector(gridLoc, commModes);
//
//    const Unsigned commKey = Loc2LinearLoc(gridSliceLoc, gridSliceShape);
//    const Unsigned commColor = Loc2LinearLoc(gridSliceNegLoc, gridSliceNegShape);
//
//    mpi::CommSplit(participatingComm_, commColor, commKey, comm);
    if(commMap_->count(sortedCommModes) == 0){
        mpi::Comm comm;
        const Location gridLoc = grid.Loc();
        const ObjShape gridShape = grid.Shape();

        ObjShape gridSliceShape = FilterVector(gridShape, sortedCommModes);
        ObjShape gridSliceNegShape = NegFilterVector(gridShape, sortedCommModes);
        Location gridSliceLoc = FilterVector(gridLoc, sortedCommModes);
        Location gridSliceNegLoc = NegFilterVector(gridLoc, sortedCommModes);

//        PrintVector(gridSliceShape, "gridSliceShape");
//        PrintVector(gridSliceNegShape, "gridSliceNegShape");
//        PrintVector(gridSliceLoc, "gridSliceLoc");
//        PrintVector(gridSliceNegLoc, "gridSliceNegLoc");
        const Unsigned commKey = Loc2LinearLoc(gridSliceLoc, gridSliceShape);
        const Unsigned commColor = Loc2LinearLoc(gridSliceNegLoc, gridSliceNegShape);

        //Check this, original was commented line with participating
        mpi::CommSplit(grid.OwningComm(), commColor, commKey, comm);
//        std::cout << "made size " << mpi::CommSize(comm) << " comm\n";
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

    mpi::Comm comm;
    const Location gridLoc = grid_->Loc();
    const ObjShape gridShape = grid_->Shape();

    ObjShape gridSliceShape = FilterVector(gridShape, commModes);
    ObjShape gridSliceNegShape = NegFilterVector(gridShape, commModes);
    Location gridSliceLoc = FilterVector(gridLoc, commModes);
    Location gridSliceNegLoc = NegFilterVector(gridLoc, commModes);

    const Unsigned commKey = Loc2LinearLoc(gridSliceLoc, gridSliceShape);
    const Unsigned commColor = Loc2LinearLoc(gridSliceNegLoc, gridSliceNegShape);

    mpi::CommSplit(Grid().OwningComm(), commColor, commKey, comm);
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


template<typename T>
void DistTensor<T>::PackCommHelper(const PackData& packData, const Mode packMode, T const * const srcBuf, T * const dstBuf){
    Unsigned packSlice;
    const Unsigned loopEnd = packData.loopShape[packMode];
    const Unsigned dstBufStride = packData.dstBufStrides[packMode];
    const Unsigned srcBufStride = packData.srcBufStrides[packMode];
    const Unsigned loopStart = packData.loopStarts[packMode];
    const Unsigned loopInc = packData.loopIncs[packMode];
    Unsigned dstBufPtr = 0;
    Unsigned srcBufPtr = 0;

    if(packMode == 0){
        if(dstBufStride == 1 && srcBufStride == 1){
            MemCopy(&(dstBuf[0]), &(srcBuf[0]), loopEnd);
        }else{
            for(packSlice = loopStart; packSlice < loopEnd; packSlice += loopInc){
                dstBuf[dstBufPtr] = srcBuf[srcBufPtr];
                dstBufPtr += dstBufStride;
                srcBufPtr += srcBufStride;
            }
        }
    }else{
        for(packSlice = loopStart; packSlice < loopEnd; packSlice += loopInc){
            PackCommHelper(packData, packMode-1, &(srcBuf[srcBufPtr]), &(dstBuf[dstBufPtr]));
            dstBufPtr += dstBufStride;
            srcBufPtr += srcBufStride;
        }
    }
}

template<typename T>
void DistTensor<T>::ElemSelectPackHelper(const PackData& packData, const ElemSelectData& elemData, const Mode mode, const GridView& gvSrc, const GridView& gvDst, const DistTensor<T>& A, T const * const dataBuf, T * const sendBuf){
    Unsigned order = A.Order();
        PackData data = packData;
        Location elem = elemData.packElem;
        ModeArray changedA2AModes = elemData.changedModes;
        std::vector<Unsigned> nProcsPerA2AMode = elemData.loopShape;
        ModeArray commModes = elemData.commModes;
        Unsigned nElemsPerProc = elemData.nElemsPerProc;
        Unsigned i;
        const tmen::GridView gvA = gvSrc;
        const tmen::GridView gvB = gvDst;
        const tmen::Grid& g = Grid();
        const Mode changedA2AMode = changedA2AModes[mode];
//
//    //    PrintVector(nProcsPerA2AMode, "nProcsPerA2AMode");
        Unsigned startLoc = elemData.packElem[changedA2AMode];
        for(i = 0; i < nProcsPerA2AMode[changedA2AMode]; i++){
            elem[changedA2AMode] = startLoc + i * gvA.ModeWrapStride(changedA2AMode);
//    //        std::cout << "PackTestHelper mode: " << mode << std::endl;
//    //        PrintVector(elem, "elem is now");
            if(elem[changedA2AMode] >= A.Dimension(changedA2AMode)){
//    //            printf("continuing\n");
                continue;
            }
            data.loopStarts[changedA2AMode] = i;
//
            if(mode == 0){
//    //            printf("hmm\n");
                Location ownerB = DetermineOwner(elem);
                Location ownerGridLoc = GridViewLoc2GridLoc(ownerB, gvB);
                Unsigned commLinLoc = Loc2LinearLoc(FilterVector(ownerGridLoc, commModes), FilterVector(g.Shape(), commModes));

    //            printf("sMode: %d\n", a2aModeTo);
    //            PrintVector(ownerB, "ownerB");
    //            PrintVector(ownerGridLoc, "ownerGridLoc");
    //            PrintVector(commModes, "commModes");
    //            printf("commLinLoc: %d\n", commLinLoc);

    //            PrintVector(elem, "pack Global elem");
    //            PrintVector(data.loopStarts, "local location");

    //            std::cout << "offsetting dataBuf by: " << i * A.LocalModeStride(a2aModeTo) << std::endl;
    //            std::cout << "offsetting sendBuf by: " << commLinLoc * nElemsPerProc << std::endl;
    //            printf("pack data:\n");
    //            PrintVector(data.loopShape, "  loop shape");
    //            PrintVector(data.loopStarts, "  loop starts");
    //            PrintVector(data.loopIncs, "  loop incs");
    //            PrintVector(data.srcBufStrides, "  srcBufStrides");
    //            PrintVector(data.dstBufStrides, "  dstBufStrides");
                PackCommHelper(data, order - 1, &(dataBuf[i * A.LocalModeStride(changedA2AMode)]), &(sendBuf[commLinLoc * nElemsPerProc]));
    //            std::cout << "procs: " << prod(nProcsPerA2AMode) << std::endl;
            }else{
                ElemSelectData newData = elemData;
                newData.packElem = elem;
                ElemSelectPackHelper(data, newData, mode - 1, gvSrc, gvDst, A, &(dataBuf[i * A.LocalModeStride(changedA2AMode)]), &(sendBuf[0]));
            }
        }
}

template<typename T>
void DistTensor<T>::ElemSelectHelper(const PackData& packData, const Mode mode, const ModeArray& commModes, const ModeArray& changedA2AModes, const Location& packElem, const std::vector<Unsigned>& nProcsPerA2AMode, const Unsigned& nElemsPerProc, const DistTensor<T>& A, T const * const dataBuf, T * const sendBuf){
    Unsigned order = A.Order();
        PackData data = packData;
        Location elem = packElem;
        Unsigned i;
        const tmen::GridView gvA = A.GetGridView();
        const tmen::GridView gvB = GetGridView();
        const tmen::Grid& g = Grid();
        const Mode changedA2AMode = changedA2AModes[mode];

    //    PrintVector(nProcsPerA2AMode, "nProcsPerA2AMode");
        Unsigned startLoc = packElem[changedA2AMode];
        for(i = 0; i < nProcsPerA2AMode[changedA2AMode]; i++){
            elem[changedA2AMode] = startLoc + i * gvA.ModeWrapStride(changedA2AMode);
    //        std::cout << "PackTestHelper mode: " << mode << std::endl;
    //        PrintVector(elem, "elem is now");
            if(elem[changedA2AMode] >= A.Dimension(changedA2AMode)){
    //            printf("continuing\n");
                continue;
            }
            data.loopStarts[changedA2AMode] = i;

            if(mode == 0){
    //            printf("hmm\n");
                Location ownerB = DetermineOwner(elem);
                Location ownerGridLoc = GridViewLoc2GridLoc(ownerB, gvB);
                Unsigned commLinLoc = Loc2LinearLoc(FilterVector(ownerGridLoc, commModes), FilterVector(g.Shape(), commModes));

    //            printf("sMode: %d\n", a2aModeTo);
    //            PrintVector(ownerB, "ownerB");
    //            PrintVector(ownerGridLoc, "ownerGridLoc");
    //            PrintVector(commModes, "commModes");
    //            printf("commLinLoc: %d\n", commLinLoc);

    //            PrintVector(elem, "pack Global elem");
    //            PrintVector(data.loopStarts, "local location");

    //            std::cout << "offsetting dataBuf by: " << i * A.LocalModeStride(a2aModeTo) << std::endl;
    //            std::cout << "offsetting sendBuf by: " << commLinLoc * nElemsPerProc << std::endl;
    //            printf("pack data:\n");
    //            PrintVector(data.loopShape, "  loop shape");
    //            PrintVector(data.loopStarts, "  loop starts");
    //            PrintVector(data.loopIncs, "  loop incs");
    //            PrintVector(data.srcBufStrides, "  srcBufStrides");
    //            PrintVector(data.dstBufStrides, "  dstBufStrides");
                PackCommHelper(data, order - 1, &(dataBuf[i * A.LocalModeStride(changedA2AMode)]), &(sendBuf[commLinLoc * nElemsPerProc]));
    //            std::cout << "procs: " << prod(nProcsPerA2AMode) << std::endl;
            }else{

                ElemSelectHelper(data, mode - 1, commModes, changedA2AModes, elem, nProcsPerA2AMode, nElemsPerProc, A, &(dataBuf[i * A.LocalModeStride(changedA2AMode)]), &(sendBuf[0]));
            }
        }
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
