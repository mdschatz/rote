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
//    PrintVector(localLoc, "Unpermuted loc");
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
//        printf("myKey: %d myColor: %d\n", commKey, commColor);

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

    const tmen::Grid& grid = Grid();
    participatingComm_ = GetCommunicatorForModes(commModes, grid);
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
//    mpi::CommSplit(Grid().OwningComm(), commColor, commKey, comm);
//    participatingComm_ = comm;
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
    PROFILE_SECTION("DistTensorPACK");
    Unsigned commRank = mpi::CommRank(MPI_COMM_WORLD);
    //Make loopIncs 1s
    std::vector<Unsigned> ones(packData.loopIncs.size(), 1);
    PackData modified = packData;
    modified.loopShape = IntCeils(packData.loopShape, packData.loopIncs);
    modified.loopIncs = ones;

    //Attempt to merge modes
    PackData newData;
    Unsigned i;
    Unsigned oldOrder = packData.loopShape.size();

    Unsigned oldSrcStride = modified.srcBufStrides[0];
    Unsigned oldDstStride = modified.dstBufStrides[0];
    newData.loopShape.push_back(modified.loopShape[0]);
    newData.srcBufStrides.push_back(modified.srcBufStrides[0]);
    newData.dstBufStrides.push_back(modified.dstBufStrides[0]);

    Unsigned mergeMode = 0;
    for(i = 1; i < oldOrder; i++){
        if(modified.srcBufStrides[i] == modified.loopShape[i-1] * oldSrcStride &&
           modified.dstBufStrides[i] == modified.loopShape[i-1] * oldDstStride){
            newData.loopShape[mergeMode] *= modified.loopShape[i];
            oldSrcStride *= modified.srcBufStrides[i];
            oldDstStride *= modified.dstBufStrides[i];
        }else{
            oldSrcStride = modified.srcBufStrides[i];
            oldDstStride = modified.dstBufStrides[i];
            newData.loopShape.push_back(modified.loopShape[i]);
            newData.srcBufStrides.push_back(oldSrcStride);
            newData.dstBufStrides.push_back(oldDstStride);
            mergeMode++;
        }
    }
    std::vector<Unsigned> newones(newData.loopShape.size(), 1);
    std::vector<Unsigned> zeros(newData.loopShape.size(), 0);
    newData.loopIncs = newones;
    newData.loopStarts = zeros;


//    if(commRank == 0){
////        PrintPackData(packData, "orig");
//        PrintPackData(modified, "modified");
//        PrintPackData(newData, "new");
//    }
#ifndef RELEASE
    PackCommHelper_ref(newData, newData.loopShape.size()-1, srcBuf, dstBuf);
#else
    PackCommHelper_fast(newData, packMode, srcBuf, dstBuf);
#endif
    PROFILE_STOP;
}

template<typename T>
void DistTensor<T>::PackCommHelper_fast(const PackData& packData, const Mode packMode, T const * const srcBuf, T * const dstBuf){
#ifndef RELEASE
    CallStackEntry cse("DistTensor::PackCommHelper_fast");
#endif

    Unsigned packSlice;
//    printf("ping packcommHelper\n");
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
    Unsigned i;
//    std::string ident = "";
//    for(i = 0; i < packData.loopShape.size() - packMode; i++)
//        ident += "  ";

    if(loopEnd.size() == 0){
        dstBuf[0] = srcBuf[0];
        return;
    }

    bool done = !ElemwiseLessThan(curLoc, loopEnd);

    while(!done){
        if(srcBufStrides[0] == 1 && dstBufStrides[0] == 1){
            MemCopy(&(dstBuf[dstBufPtr]), &(srcBuf[srcBufPtr]), loopEnd[0]);
            curLoc[0] += loopEnd[0];
            dstBufPtr += dstBufStrides[0] * loopEnd[0];
            srcBufPtr += srcBufStrides[0] * loopEnd[0];
        }else{
            dstBuf[dstBufPtr] = srcBuf[srcBufPtr];
            //Update
    //        curLoc[ptr]+= loopIncs[ptr];
    //        dstBufPtr += dstBufStrides[ptr];
    //        srcBufPtr += srcBufStrides[ptr];
            curLoc[0]++;
            dstBufPtr += dstBufStrides[0];
            srcBufPtr += srcBufStrides[0];
        }
        while(ptr < order && curLoc[ptr] >= loopEnd[ptr]){
//            curLoc[ptr] = loopStart[ptr];
//            dstBufPtr -= dstBufStrides[ptr] * (IntCeil(loopEnd[ptr] - loopStart[ptr], loopIncs[ptr]));
//            srcBufPtr -= srcBufStrides[ptr] * (IntCeil(loopEnd[ptr] - loopStart[ptr], loopIncs[ptr]));
            curLoc[ptr] = 0;
            dstBufPtr -= dstBufStrides[ptr] * loopEnd[ptr];
            srcBufPtr -= srcBufStrides[ptr] * loopEnd[ptr];
            ptr++;
            if(ptr >= order){
                done = true;
                break;
            }else{
//                curLoc[ptr] += loopIncs[ptr];
                curLoc[ptr]++;
                dstBufPtr += dstBufStrides[ptr];
                srcBufPtr += srcBufStrides[ptr];
            }
        }
        if(done)
            break;
        ptr = 0;
    }
}

template<typename T>
void DistTensor<T>::PackCommHelper_ref(const PackData& packData, const Mode packMode, T const * const srcBuf, T * const dstBuf){
#ifndef RELEASE
    CallStackEntry cse("DistTensor::PackCommHelper_ref");
#endif
    Unsigned commRank = mpi::CommRank(MPI_COMM_WORLD);
    Unsigned packSlice;
//    printf("ping packcommHelper\n");
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


//    Unsigned i;
//    std::string ident = "";
//    for(i = 0; i < packData.loopShape.size() - packMode; i++)
//        ident += "  ";

//    if(commRank == 0){
//        std::cout << ident << " investing mode: " << packMode << "\n";
//        PrintPackData(packData, "packData");
//    }
//    if(packMode == 3){
//        PrintVector(packData.loopShape, "loopEnd");
//        PrintVector(packData.loopStarts, "loopStart");
//        PrintVector(packData.loopIncs, "loopInc");
//    }

    if(packMode == 0){
        if(dstBufStride == 1 && srcBufStride == 1){
//            if(commRank == 0){
//                std::cout << ident << "copying " << loopEnd - loopStart << "elements" << std::endl;
//            }

            MemCopy(&(dstBuf[0]), &(srcBuf[0]), loopEnd - loopStart);
        }else{
//            PrintVector(packData.loopStarts, "loopStarts");
//            if(commRank == 0){
//                printf("loopStart: %d, loopInc: %d, loopEnd: %d\n", loopStart, loopInc, loopEnd);
//            }
            for(packSlice = loopStart; packSlice < loopEnd; packSlice += loopInc){
                dstBuf[dstBufPtr] = srcBuf[srcBufPtr];

//                std::cout << ident << "Packing mode: " << packMode << "iteration: " << packSlice << "of: " << loopEnd << "by: " << loopInc << std::endl;
//                std::cout << ident << "copying elem " << srcBuf[srcBufPtr] << std::endl;
//                std::cout << ident << "incrementing dstBuf by " << dstBufStride << std::endl;
//                std::cout << ident << "incrementing srcBuf by " << srcBufStride << std::endl;
                dstBufPtr += dstBufStride;
                srcBufPtr += srcBufStride;
            }
        }
    }else{
        for(packSlice = loopStart; packSlice < loopEnd; packSlice += loopInc){
//            if(commRank == 0){
//                std::cout << ident << " recurring on mode: " << packMode-1 << "\n";
//            }
            PackCommHelper_ref(packData, packMode-1, &(srcBuf[srcBufPtr]), &(dstBuf[dstBufPtr]));

//            std::cout << ident << "incrementing dstBuf by " << dstBufStride << std::endl;
//            std::cout << ident << "incrementing srcBuf by " << srcBufStride << std::endl;
            dstBufPtr += dstBufStride;
            srcBufPtr += srcBufStride;
        }
    }
}

////////////////////////////////////////////

template<typename T>
void DistTensor<T>::ElemSelectPackHelper(const PackData& packData, const ElemSelectData& elemData, const Mode mode, const DistTensor<T>& A, T const * const dataBuf, T * const sendBuf){
    Unsigned order = A.Order();
    if(order == 0){
        PackCommHelper(packData, order - 1, &(dataBuf[0]), &(sendBuf[0]));
        return;
    }
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
        data.loopShape[changedA2AMode] = data.loopShape[changedA2AMode] - i;

        if(mode == 0){
//            printf("hmm\n");
//            PrintVector(elem, "packing elem");
            Location ownerB;
            PROFILE_SECTION("DETERMINE OWNER");
            ownerB = DetermineOwner(elem);
            PROFILE_STOP;
            Location ownerGridLoc = GridViewLoc2GridLoc(ownerB, gvB);
//            PrintVector(ownerB, "owner loc");
            Unsigned commLinLoc;
            PROFILE_SECTION("LINLOC");
            commLinLoc = Loc2LinearLoc(FilterVector(ownerGridLoc, commModes), FilterVector(g.Shape(), commModes));
            PROFILE_STOP;
//            printf("owner lin loc %d\n", commLinLoc);
//            printf("sMode: %d\n", a2aModeTo);
//            PrintVector(ownerB, "ownerB");
//            PrintVector(ownerGridLoc, "ownerGridLoc");
//            PrintVector(commModes, "commModes");
//            printf("commLinLoc: %d\n", commLinLoc);

//            PrintVector(elem, "pack Global elem");
//            PrintVector(data.loopStarts, "local location");


//            printf("pack data:\n");
//            PrintVector(data.loopShape, "  loop shape");
//            PrintVector(data.loopStarts, "  loop starts");
//            PrintVector(data.loopIncs, "  loop incs");
//            PrintVector(data.srcBufStrides, "  srcBufStrides");
//            PrintVector(data.dstBufStrides, "  dstBufStrides");
            Unsigned dataBufPtr = LinearLocFromStrides(PermuteVector(srcElem, elemData.permutation), srcStrides);
//            std::cout << "offsetting dataBuf by: " << dataBufPtr << std::endl;
//            std::cout << "offsetting sendBuf by: " << commLinLoc * nElemsPerProc << std::endl;
//            data.loopStarts = PermuteVector(data.loopStarts, elemData.permutation);

            PROFILE_SECTION("ELEMSELECT PACK");
            PackCommHelper(data, order - 1, &(dataBuf[dataBufPtr]), &(sendBuf[commLinLoc * nElemsPerProc]));
            PROFILE_STOP;
//            printf("sendBuf:");
//            for(i = 0; i < nElemsPerProc*prod(FilterVector(g.Shape(), commModes)); i++)
//                printf(" %d", sendBuf[i]);
//            printf("\n");
        }else{
            ElemSelectData newData = elemData;
            newData.packElem = elem;
            newData.srcElem = srcElem;
            ElemSelectPackHelper(data, newData, mode - 1, A, &(dataBuf[0]), &(sendBuf[0]));
        }
    }
}

template<typename T>
void DistTensor<T>::ElemSelectUnpackHelper(const PackData& packData, const ElemSelectData& elemData, const Mode mode, const DistTensor<T>& A, T const * const recvBuf, T * const dataBuf){
    Unsigned order = A.Order();
    if(order == 0){
        PackCommHelper(packData, order - 1, &(recvBuf[0]), &(dataBuf[0]));
    }
    PackData data = packData;
    Location elem = elemData.packElem;
    Location dstElem = elemData.dstElem;
    std::vector<Unsigned> srcStrides = elemData.srcStrides;
    std::vector<Unsigned> dstStrides = elemData.dstStrides;
    ModeArray changedA2AModes = elemData.changedModes;
    std::vector<Unsigned> nProcsPerA2AMode = elemData.loopShape;
    ModeArray commModes = elemData.commModes;
    Unsigned nElemsPerProc = elemData.nElemsPerProc;
    Unsigned i, j;
    const tmen::GridView gvA = A.GetGridView();
    const tmen::GridView gvB = GetGridView();
    const tmen::Grid& g = Grid();
    const Mode changedA2AMode = mode;

//    PrintVector(nProcsPerA2AMode, "nProcsPerA2AMode");
    Unsigned startLoc = elemData.packElem[changedA2AMode];
    for(i = 0; i < nProcsPerA2AMode[changedA2AMode]; i++){
        elem[changedA2AMode] = startLoc + i * gvB.ModeWrapStride(changedA2AMode);
//        std::cout << "PackTestHelper mode: " << mode << std::endl;
//        PrintVector(elem, "elem is now");
        if(elem[changedA2AMode] >= Dimension(changedA2AMode)){
//            printf("continuing\n");
            continue;
        }
        data.loopShape[changedA2AMode] = data.loopShape[changedA2AMode] - i;
        dstElem[changedA2AMode] = i;

        if(mode == 0){
//            printf("hmm\n");
            Location ownerA = A.DetermineOwner(elem);
            Location ownerGridLoc = GridViewLoc2GridLoc(ownerA, gvA);
            Unsigned commLinLoc = Loc2LinearLoc(FilterVector(ownerGridLoc, commModes), FilterVector(g.Shape(), commModes));

//            printf("sMode: %d\n", a2aModeTo);
//            PrintVector(elem, "unpacking elem");
//            PrintVector(ownerA, "ownerA");
//            PrintVector(ownerGridLoc, "ownerGridLoc");
//            printf("commLinLoc: %d\n", commLinLoc);

//            PrintVector(elem, "pack Global elem");
//            PrintVector(data.loopStarts, "local location");

//            printf("unpack data:\n");
//            PrintVector(data.loopShape, "  loop shape");
//            PrintVector(data.loopStarts, "  loop starts");
//            PrintVector(data.loopIncs, "  loop incs");
//            PrintVector(data.srcBufStrides, "  srcBufStrides");
//            PrintVector(data.dstBufStrides, "  dstBufStrides");
//            PrintVector(dstElem, "dstElem");
//            PrintVector(PermuteVector(dstElem, elemData.permutation), "permuted dstElem");
            Unsigned dataBufPtr = LinearLocFromStrides(PermuteVector(dstElem, elemData.permutation), dstStrides);
//            data.loopStarts = PermuteVector(data.loopStarts, elemData.permutation);
//            printf("unpacking from recvBuf loc: %d\n", commLinLoc * nElemsPerProc);
//            printf("starting unpack at dataBuf loc: %d\n", dataBufPtr);
            PackCommHelper(data, order - 1, &(recvBuf[commLinLoc * nElemsPerProc]), &(dataBuf[dataBufPtr]));
//            printf("dataBuf:");
//            for(j = 0; j < prod(LocalShape()); j++)
//                printf(" %d", dataBuf[j]);
//            printf("\n");
        }else{
//            printf("ping\n");
//            printf("order: %d\n", order);
//            printf("mode: %d\n", mode);
            ElemSelectData newData = elemData;
            newData.packElem = elem;
            newData.dstElem = dstElem;
            ElemSelectUnpackHelper(data, newData, mode - 1, A, &(recvBuf[0]), &(dataBuf[0]));
//            if(mode == order-1){
//                printf("dataBuf:");
//                for(j = 0; j < prod(LocalShape()); j++)
//                    printf(" %d", dataBuf[j]);
//                printf("\n");
//            }
        }
    }
}

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
            data.loopShape[changedA2AMode] = data.loopShape[changedA2AMode] - i;

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
