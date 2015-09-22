/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#include "tensormental.hpp"
#include "tensormental/util/graph_util.hpp"
namespace tmen {

template<typename T>
void
DistTensor<T>::ComplainIfReal() const
{
    if( !IsComplex<T>::val )
        LogicError("Called complex-only routine with real data");
}

//TODO: FIX THIS CHECK
template<typename T>
Location
DistTensor<T>::DetermineOwner(const Location& loc) const
{
#ifndef RELEASE
    CallStackEntry entry("DistTensor::DetermineOwner");
//    AssertValidEntry( loc );
#endif
    Unsigned i;
    const tmen::GridView gv = GetGridView();
    Location ownerLoc = Alignments();

    for(i = 0; i < gv.ParticipatingOrder(); i++){
        ownerLoc[i] = (loc[i] + ModeAlignment(i)) % ModeStride(i);
    }
    return ownerLoc;
}

template<typename T>
Location
DistTensor<T>::DetermineOwnerNewAlignment(const Location& loc, std::vector<Unsigned>& newAlignment) const
{
#ifndef RELEASE
    CallStackEntry entry("DistTensor::DetermineOwner");
    AssertValidEntry( loc );
#endif
    Unsigned i;
    const tmen::GridView gv = GetGridView();
    Location ownerLoc = Alignments();

    for(i = 0; i < gv.ParticipatingOrder(); i++){
        ownerLoc[i] = (loc[i] + newAlignment[i]) % ModeStride(i);
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

    const GridView gv = GetGridView();
    const ObjShape participatingShape = gv.ParticipatingShape();
    Location ret(gridViewLoc.size());
    for(i = 0; i < gridViewLoc.size(); i++){
        ret[i] = gridViewLoc[i] - modeAlignments_[i];

        if(gridViewLoc[i] < modeAlignments_[i])
            ret[i] += ModeStride(i);
    }
//    Location ret(gridViewLoc.size());
//    for(i = 0; i < gridViewLoc.size(); i++){
//        ret[i] = Shift(gridViewLoc[i], modeAlignments_[i], ModeStride(i));
//    }

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
//    printf("aligning\n");
//    PrintData(A, "A");
//    PrintData(*this, "*this");
    const tmen::Grid& g = Grid();
    GridView gvA = A.GetGridView();
    GridView gvB = GetGridView();

    Location firstOwnerA = GridViewLoc2GridLoc(A.Alignments(), gvA);
    Location firstOwnerB = GridViewLoc2GridLoc(Alignments(), gvB);



    std::vector<Unsigned> alignA = A.Alignments();
    std::vector<Unsigned> alignB = Alignments();

    std::vector<Unsigned> alignBinA = GridLoc2ParticipatingGridViewLoc(firstOwnerB, g.Shape(), A.TensorDist());
//    std::vector<Unsigned> alignDiff = ElemwiseSubtract(firstOwnerA, firstOwnerB);
//    PrintVector(alignDiff, "alignDiff");
//    PrintVector(g.Loc(), "myLoc");
//    PrintVector(ElemwiseSum(g.Loc(), alignDiff), "alignDiff + myLoc");
    Location alignedFirstOwnerA = GridLoc2GridViewLoc(firstOwnerB, g.Shape(), A.TensorDist());
    Location myFirstElemLocA = A.DetermineFirstElem(gvA.ParticipatingLoc());
    Location myFirstElemLocAligned = A.DetermineFirstUnalignedElem(gvA.ParticipatingLoc(), alignBinA);

    Location sendGridLoc = GridViewLoc2GridLoc(A.DetermineOwnerNewAlignment(myFirstElemLocA, alignBinA), gvA);
    Location recvGridLoc = GridViewLoc2GridLoc(A.DetermineOwner(myFirstElemLocAligned), gvA);
//    Location sendGridLoc = ElemwiseMod(ElemwiseSum(ElemwiseSubtract(g.Loc(), alignDiff), g.Shape()), g.Shape());
//    Location recvGridLoc = ElemwiseMod(ElemwiseSum(ElemwiseSum(g.Loc(), alignDiff), g.Shape()), g.Shape());

//    PrintVector(firstOwnerA, "firstOwnerA");
//    PrintVector(firstOwnerB, "firstOwnerB");
//    PrintVector(alignA, "alignA");
//    PrintVector(alignB, "alignB");
//    PrintVector(alignBinA, "alignBinA");
//    PrintVector(myFirstElemLocA, "myFirstElemLocA");
//    PrintVector(myFirstElemLocAligned, "myFirstElemLocAligned");
//    PrintVector(sendGridLoc, "sendGridLoc");
//    PrintVector(recvGridLoc, "recvGridLoc");
//    PrintVector(GridLoc2ParticipatingGridViewLoc(sendGridLoc, g.Shape(), A.TensorDist()), "gvASendLoc");
//    PrintVector(GridLoc2ParticipatingGridViewLoc(recvGridLoc, g.Shape(), A.TensorDist()), "gvARecvLoc");

    //Create the communicator to involve all processes we need to fix misalignment
    ModeArray misalignedModes;
    for(Unsigned i = 0; i < alignA.size(); i++){
        if(alignBinA[i] != alignA[i]){
            ModeDistribution modeDist = A.ModeDist(i);
            misalignedModes.insert(misalignedModes.end(), modeDist.begin(), modeDist.end());
        }
    }
    std::sort(misalignedModes.begin(), misalignedModes.end());
//    PrintVector(misalignedModes, "misalignedModes");
    mpi::Comm sendRecvComm = GetCommunicatorForModes(misalignedModes, g);

    Location sendSliceLoc = FilterVector(sendGridLoc, misalignedModes);
    Location recvSliceLoc = FilterVector(recvGridLoc, misalignedModes);
    ObjShape gridSliceShape = FilterVector(g.Shape(), misalignedModes);

    Unsigned sendLinLoc = Loc2LinearLoc(sendSliceLoc, gridSliceShape);
    Unsigned recvLinLoc = Loc2LinearLoc(recvSliceLoc, gridSliceShape);


//            PrintVector(sendGridLoc, "sendLoc");
//            printf("sendLinloc: %d\n", sendLinLoc);
//            PrintVector(recvGridLoc, "recvLoc");
//            printf("recvLinloc: %d\n", recvLinLoc);
//            printf("sending %d unaligned elems\n", sendSize);
//            printf("recving %d aligned elems\n", recvSize);

//            PrintArray(alignSendBuf, sendShape, "sendBuf to SendRecv");
    mpi::SendRecv(unalignedSendBuf, sendSize, sendLinLoc,
                  alignedSendBuf, recvSize, recvLinLoc, sendRecvComm);

}

//Creates information for optimizing A2A routines as P2P calls when applicable
//Idea is this:
//For all modes of grid we're communicating over that share the same dimension
//Exchange these as a P2P instead of an A2A
//Only perform the A2A exchange on modes we absolutely have to.
//These are modes that don't form a dimension-invariant permutation cycle with any other modes.
template<typename T>
A2AP2PData
DistTensor<T>::DetermineA2AP2POptData(const DistTensor<T>& A, const ModeArray& commModes)
{
    const tmen::Grid& g = A.Grid();
    ModeArray sortedCommModes = commModes;
    std::sort(sortedCommModes.begin(), sortedCommModes.end());

    Unsigned i, j;
    Unsigned order = A.Order();
    TensorDistribution distA = A.TensorDist();
    TensorDistribution distB = TensorDist();

//    PrintData(A, "inData");
//    PrintData(*this, "outData");
    //Determine which modes communicating over that share the same dimension
    ObjShape gShape = g.Shape();
    std::vector<ModeArray> symGridModes;
    ModeArray oldCommModes = commModes;
    while(oldCommModes.size() > 0){
        ModeArray newGroup;
        Mode matchMode = oldCommModes[oldCommModes.size() - 1];
        newGroup.push_back(matchMode);
        oldCommModes.erase(oldCommModes.end() - 1);
        for(i = oldCommModes.size() - 1; i < oldCommModes.size(); i--){
            if(gShape[oldCommModes[i]] == gShape[matchMode]){
                newGroup.push_back(oldCommModes[i]);
                oldCommModes.erase(oldCommModes.begin() + i);
            }
        }
        symGridModes.push_back(newGroup);
    }

    //Create arrays representing how each commMode is exchanged in the redistribution
    ModeArray tenModesFrom(g.Order(), -1);
    ModeArray tenModesTo(g.Order(), -1);
    for(i = 0; i < commModes.size(); i++){
        Mode commMode = commModes[i];
        for(j = 0; j < distA.size(); j++){
            ModeDistribution modeDist = distA[j];
            if(std::find(modeDist.begin(), modeDist.end(), commMode) != modeDist.end()){
                tenModesFrom[commMode] = j;
                break;
            }
        }
        for(j = 0; j < distB.size(); j++){
            ModeDistribution modeDist = distB[j];
            if(std::find(modeDist.begin(), modeDist.end(), commMode) != modeDist.end()){
                tenModesTo[commMode] = j;
                break;
            }
        }
    }
    //Form the fromTo list defining how commModes move
    std::vector<std::pair<Mode, Mode> > tensorModeFromTo(g.Order());
    for(i = 0; i < tenModesFrom.size(); i++){
        std::pair<Mode, Mode> newPair(tenModesFrom[i], tenModesTo[i]);
        tensorModeFromTo[i] = newPair;
    }

    int commRank = mpi::CommRank(MPI_COMM_WORLD);
//    for(i = 0; i < symGridModes.size(); i++){
//        printf("[%d] ", i);
//        PrintVector(symGridModes[i], "symGroup");
//    }
//    printf("commMode changes\n");
//    for(i = 0; i < tensorModeFromTo.size(); i++){
//        printf("%d --> %d\n", tensorModeFromTo[i].first, tensorModeFromTo[i].second);
//    }


    //Using the information regarding how each mode is exchanged, create a list of
    //p2p-redistributable commModes and a2a-redistributable commModes
    ModeArray p2pCommModes;
    ModeArray a2aCommModes;
    for(i = 0; i < symGridModes.size(); i++){
        ModeArray symGroup = symGridModes[i];
//        PrintVector(symGroup, "Finding SCC for group");

        std::vector<std::pair<Mode, Mode> > tensorModeFromToSubset(symGroup.size());
        for(j = 0; j < symGroup.size(); j++){
            tensorModeFromToSubset[j] = tensorModeFromTo[symGroup[j]];
        }

//        printf("Considering edges:\n");
//        for(j = 0; j < tensorModeFromToSubset.size(); j++)
//            printf("%d --> %d\n", tensorModeFromToSubset[j].first, tensorModeFromToSubset[j].second);
//        printf("\n");

        DetermineSCC(symGroup, tensorModeFromToSubset, p2pCommModes);

        //a2aModes are those that weren't added to p2pModes by the DetermineSCC function
        for(j = 0; j < symGroup.size(); j++){
            Mode commMode = symGroup[j];
            if(std::find(p2pCommModes.begin(), p2pCommModes.end(), commMode) == p2pCommModes.end())
                a2aCommModes.push_back(commMode);
        }
    }

    A2AP2PData ret;
    if(p2pCommModes.size() != 0){
    //Determine the distribution prefixes shared by input and output distributions
    TensorDistribution prefixDist = CreatePrefixDistribution(distA, distB);
    //(Prefix | A2AModesIn)
//    TensorDistribution prefixA2ADist = CreatePrefixA2ADistribution(prefixDist, distA, a2aCommModes);
    //(Prefix | A2AModesIn | P2PModesIn)
//    TensorDistribution A2AOpt1Dist = CreatePrefixA2ADistribution(prefixA2ADist, distA, p2pCommModes);
    //(Prefix | A2AModesIn | P2PModesOut)
//    TensorDistribution A2AOpt2Dist = CreatePrefixA2ADistribution(prefixA2ADist, distB, p2pCommModes);
    //(Prefix | P2PModesOut)
    TensorDistribution prefixP2PDist = CreatePrefixP2PDistribution(prefixDist, distB, p2pCommModes);
    //(Prefix | P2PModesOut | A2AModesIn)
    TensorDistribution A2AOpt3Dist = CreateA2AOptDist3(prefixP2PDist, distA, a2aCommModes);
    //(Prefix | P2PModesOut | A2AModesOut)
    TensorDistribution A2AOpt4Dist = CreateA2AOptDist4(prefixP2PDist, distB, a2aCommModes);

    ret.doOpt = true;
    ret.a2aModes = a2aCommModes;
    ret.p2pModes = p2pCommModes;
//    ret.opt1Dist = A2AOpt1Dist;
//    ret.opt2Dist = A2AOpt2Dist;
    ret.opt3Dist = A2AOpt3Dist;
    ret.opt4Dist = A2AOpt4Dist;
    ret.p2pModes = p2pCommModes;
    ret.a2aModes = a2aCommModes;
    }else{
        ret.doOpt = false;
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
