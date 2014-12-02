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
        localLoc[i] = (globalLoc[i]-ModeShift(i)) / ModeStride(i);
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
void DistTensor<T>::PackCommHelper(const PackData& packData, const Mode packMode, T const * const srcBuf, T * const dstBuf){
#ifndef RELEASE
    CallStackEntry cse("DistTensor::PackCommHelper");
#endif
//    PROFILE_SECTION("DistTensor Pack");
    PackData modifiedData = packData;
    modifiedData.loopShape = IntCeils(packData.loopShape, packData.loopIncs);
    Location ones(packData.loopStarts.size(), 1);
    Location zeros(packData.loopStarts.size(), 0);
    modifiedData.loopIncs = ones;
    modifiedData.loopStarts = zeros;


    //Attempt to merge modes
    PackData newData;
    Unsigned i;
    Unsigned oldOrder = packData.loopShape.size();

    if(oldOrder == 0){
        newData = modifiedData;
    }else{
        newData.loopShape.push_back(modifiedData.loopShape[0]);
        newData.srcBufStrides.push_back(modifiedData.srcBufStrides[0]);
        newData.dstBufStrides.push_back(modifiedData.dstBufStrides[0]);
        Unsigned srcStrideToMatch = modifiedData.srcBufStrides[0] * modifiedData.loopShape[0];
        Unsigned dstStrideToMatch = modifiedData.dstBufStrides[0] * modifiedData.loopShape[0];

//        PrintPackData(modifiedData, "moded data");

        Unsigned mergeMode = 0;
        for(i = 1; i < oldOrder; i++){
            if(modifiedData.srcBufStrides[i] == srcStrideToMatch &&
               modifiedData.dstBufStrides[i] == dstStrideToMatch){
                newData.loopShape[mergeMode] *= modifiedData.loopShape[i];
                srcStrideToMatch *= modifiedData.loopShape[i];
                dstStrideToMatch *= modifiedData.loopShape[i];
            }else{
                newData.loopShape.push_back(modifiedData.loopShape[i]);
                newData.srcBufStrides.push_back(modifiedData.srcBufStrides[i]);
                newData.dstBufStrides.push_back(modifiedData.dstBufStrides[i]);
                srcStrideToMatch = modifiedData.srcBufStrides[i] * modifiedData.loopShape[i];
                dstStrideToMatch = modifiedData.dstBufStrides[i] * modifiedData.loopShape[i];
                mergeMode++;
            }
        }
        std::vector<Unsigned> newones(newData.loopShape.size(), 1);
        std::vector<Unsigned> newzeros(newData.loopShape.size(), 0);
        newData.loopIncs = newones;
        newData.loopStarts = newzeros;
    }

#ifndef RELEASE
    PackCommHelper_ref(newData, newData.loopShape.size() - 1, srcBuf, dstBuf);
#else
    PackCommHelper_fast(newData, packMode, srcBuf, dstBuf);
#endif
//    PROFILE_STOP;
}

template<typename T>
void DistTensor<T>::PackCommHelper_fast(const PackData& packData, const Mode packMode, T const * const srcBuf, T * const dstBuf){
#ifndef RELEASE
    CallStackEntry cse("DistTensor::PackCommHelper_fast");
#endif
    if(packData.loopShape.size() == 0){
        dstBuf[0] = srcBuf[0];
        return;
    }

    const std::vector<Unsigned> loopEnd = packData.loopShape;
    const std::vector<Unsigned> dstBufStrides = packData.dstBufStrides;
    const std::vector<Unsigned> srcBufStrides = packData.srcBufStrides;
    const std::vector<Unsigned> loopStart = packData.loopStarts;
    const std::vector<Unsigned> loopIncs = packData.loopIncs;
    Unsigned order = loopEnd.size();
    Location curLoc = loopStart;
    Unsigned dstBufPtr = 0;
    Unsigned srcBufPtr = 0;
    Unsigned ptr = 0;

    if(loopEnd.size() == 0){
        dstBuf[0] = srcBuf[0];
        return;
    }

    bool done = !ElemwiseLessThan(curLoc, loopEnd);

    if (srcBufStrides[0] == 1 && dstBufStrides[0] == 1) {
        while (!done) {
            MemCopy(&(dstBuf[dstBufPtr]), &(srcBuf[srcBufPtr]), loopEnd[0]);
            curLoc[0] += loopEnd[0];
            srcBufPtr += srcBufStrides[0] * (loopEnd[0]);
            dstBufPtr += dstBufStrides[0] * (loopEnd[0]);

            while (ptr < order && curLoc[ptr] >= loopEnd[ptr]) {
                curLoc[ptr] = 0;

                dstBufPtr -= dstBufStrides[ptr] * loopEnd[ptr];
                srcBufPtr -= srcBufStrides[ptr] * loopEnd[ptr];
                ptr++;
                if (ptr >= order) {
                    done = true;
                    break;
                } else {
                    curLoc[ptr]++;
                    dstBufPtr += dstBufStrides[ptr];
                    srcBufPtr += srcBufStrides[ptr];
                }
            }
            if (done)
                break;
            ptr = 0;
        }
    } else {
        while (!done) {
            dstBuf[dstBufPtr] = srcBuf[srcBufPtr];
            //Update
            curLoc[0]++;
            dstBufPtr += dstBufStrides[0];
            srcBufPtr += srcBufStrides[0];

            while (ptr < order && curLoc[ptr] >= loopEnd[ptr]) {
                curLoc[ptr] = 0;

                dstBufPtr -= dstBufStrides[ptr] * loopEnd[ptr];
                srcBufPtr -= srcBufStrides[ptr] * loopEnd[ptr];
                ptr++;
                if (ptr >= order) {
                    done = true;
                    break;
                } else {
                    curLoc[ptr]++;
                    dstBufPtr += dstBufStrides[ptr];
                    srcBufPtr += srcBufStrides[ptr];
                }
            }
            if (done)
                break;
            ptr = 0;
        }
    }

//    while(!done){
//
//        if(srcBufStrides[0] == 1 && dstBufStrides[0] == 1){
//            MemCopy(&(dstBuf[dstBufPtr]), &(srcBuf[srcBufPtr]), loopEnd[0]);
//            curLoc[0] += loopEnd[0];
//            srcBufPtr += srcBufStrides[0] * (loopEnd[0]);
//            dstBufPtr += dstBufStrides[0] * (loopEnd[0]);
//        }else{
//            dstBuf[dstBufPtr] = srcBuf[srcBufPtr];
//            //Update
//            curLoc[0]++;
//            dstBufPtr += dstBufStrides[0];
//            srcBufPtr += srcBufStrides[0];
//        }
//        while(ptr < order && curLoc[ptr] >= loopEnd[ptr]){
//            curLoc[ptr] = 0;
//
//            dstBufPtr -= dstBufStrides[ptr] * loopEnd[ptr];
//            srcBufPtr -= srcBufStrides[ptr] * loopEnd[ptr];
//            ptr++;
//            if(ptr >= order){
//                done = true;
//                break;
//            }else{
//                curLoc[ptr]++;
//                dstBufPtr += dstBufStrides[ptr];
//                srcBufPtr += srcBufStrides[ptr];
//            }
//        }
//        if(done)
//            break;
//        ptr = 0;
//    }
}

template<typename T>
void DistTensor<T>::PackCommHelper_ref(const PackData& packData, const Mode packMode, T const * const srcBuf, T * const dstBuf){
#ifndef RELEASE
    CallStackEntry cse("DistTensor::PackCommHelper_ref");
#endif
    Unsigned packSlice;
    if(packData.loopShape.size() == 0){
        dstBuf[0] = srcBuf[0];
        return;
    }

    const Unsigned loopEnd = packData.loopShape[packMode];
    const Unsigned dstBufStride = packData.dstBufStrides[packMode];
    const Unsigned srcBufStride = packData.srcBufStrides[packMode];
    const Unsigned loopStart = packData.loopStarts[packMode];
    const Unsigned loopInc = packData.loopIncs[packMode];
    Unsigned dstBufPtr = 0;
    Unsigned srcBufPtr = 0;

    if(packMode == 0){
        if(dstBufStride == 1 && srcBufStride == 1){
            MemCopy(&(dstBuf[0]), &(srcBuf[0]), loopEnd - loopStart);
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

////////////////////////////////////////////

template<typename T>
void DistTensor<T>::ElemSelectHelper(const PackData& packData, const ElemSelectData& elemData, const Mode mode, const DistTensor<T>& A, T const * const dataBuf, T * const sendBuf){
    Unsigned order = A.Order();
        PackData data = packData;
        Location elem = elemData.packElem;
        Location srcElem = elemData.srcElem;
        std::vector<Unsigned> srcStrides = elemData.srcStrides;
        ModeArray changedA2AModes = elemData.changedModes;
        std::vector<Unsigned> nProcsPerA2AMode = elemData.loopShape;
        ModeArray commModes = elemData.commModes;
        Unsigned nElemsPerProc = elemData.nElemsPerProc;
        Unsigned i;
        const tmen::GridView gvA = A.GetGridView();
        const tmen::GridView gvB = GetGridView();
        const tmen::Grid& g = Grid();
        const Mode changedA2AMode = mode;

    //    PrintVector(nProcsPerA2AMode, "nProcsPerA2AMode");
        Unsigned startLoc = elemData.packElem[changedA2AMode];
        for(i = 0; i < nProcsPerA2AMode[changedA2AMode]; i++){
            elem[changedA2AMode] = startLoc + i * gvA.ModeWrapStride(changedA2AMode);
            srcElem[changedA2AMode] = i;
    //        std::cout << "PackTestHelper mode: " << mode << std::endl;
    //        PrintVector(elem, "elem is now");
            if(elem[changedA2AMode] >= A.Dimension(changedA2AMode)){
    //            printf("continuing\n");
                continue;
            }
//            data.loopStarts[changedA2AMode] = i;
            data.loopShape[changedA2AMode] = packData.loopShape[changedA2AMode] - i;

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
                Unsigned dataBufPtr = LinearLocFromStrides(PermuteVector(srcElem, elemData.permutation), srcStrides);

                PackCommHelper(data, order - 1, &(dataBuf[dataBufPtr]), &(sendBuf[commLinLoc * nElemsPerProc]));
    //            std::cout << "procs: " << prod(nProcsPerA2AMode) << std::endl;
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
