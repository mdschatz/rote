/*
   Copyright (c) 2009-2013, Jack Poulson
                      2013, Jeff Hammond
                      2013, Jed Brown
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#include "tensormental.hpp"
#include <algorithm>

namespace tmen{

template<typename T>
Int DistTensor<T>::CheckLocalCommRedist(const DistTensor<T>& A, const Mode localMode, const ModeArray& gridRedistModes){
    if(A.Order() != Order())
        LogicError("CheckLocalRedist: Objects being redistributed must be of same order");

    Unsigned i, j;
    TensorDistribution distA = A.TensorDist();
    ModeDistribution localModeDistA = A.ModeDist(localMode);
    ModeDistribution localModeDistB = ModeDist(localMode);

    if(localModeDistB.size() != localModeDistA.size() + gridRedistModes.size())
        LogicError("CheckLocalReist: Input object cannot be redistributed to output object");

    ModeArray check(localModeDistB);
    for(i = 0; i < localModeDistA.size(); i++)
        check[i] = localModeDistA[i];
    for(i = 0; i < gridRedistModes.size(); i++)
        check[localModeDistA.size() + i] = gridRedistModes[i];

    for(i = 0; i < check.size(); i++){
        if(check[i] != localModeDistB[i])
            LogicError("CheckLocalRedist: Output distribution cannot be formed from supplied parameters");
    }

    ModeArray boundModes;
    for(i = 0; i < distA.size(); i++){
        for(j = 0; j < distA[i].size(); j++){
            boundModes.push_back(distA[i][j]);
        }
    }

    for(i = 0; i < gridRedistModes.size(); i++)
        if(std::find(boundModes.begin(), boundModes.end(), gridRedistModes[i]) != boundModes.end())
            LogicError("CheckLocalRedist: Attempting to redistribute with already bound mode of the grid");

    return 1;
}

template<typename T>
void DistTensor<T>::LocalCommRedist(const DistTensor<T>& A){
//    if(!CheckLocalCommRedist(A, localMode, gridRedistModes))
//        LogicError("LocalRedist: Invalid redistribution request");
    if(!(Participating()))
        return;

    const tmen::GridView gvA = A.GetGridView();
    const tmen::GridView gvB = GetGridView();

    Location firstOwnerA = GridViewLoc2GridLoc(A.Alignments(), gvA);
    Location firstOwnerB = GridViewLoc2GridLoc(Alignments(), gvB);

//    const T* dataBuf = A.LockedBuffer();
//    PrintArray(dataBuf, A.LocalShape(), A.LocalStrides(), "srcBuf");

    if(AnyElemwiseNotEqual(firstOwnerA, firstOwnerB)){
        const ObjShape commDataShape = MaxLocalShape();
        const Unsigned sendSize = prod(commDataShape);
        const Unsigned recvSize = prod(commDataShape);

        T* auxBuf = this->auxMemory_.Require(sendSize + recvSize);

//        printf("aligningRS\n");
//        PrintData(A, "A");
//        PrintData(*this, "*this");
        T* sendBuf = &(auxBuf[0]);
        T* recvBuf = &(auxBuf[sendSize]);

        PackAGCommSendBuf(A, sendBuf);

        AlignCommBufRedist(A, sendBuf, sendSize, recvBuf, sendSize);

        //Packing is what is stored in memory
        PROFILE_SECTION("LocalUnpack");
        UnpackLocalCommRedist(A, recvBuf);
        PROFILE_STOP;
        this->auxMemory_.Release();

//        PrintArray(alignRecvBuf, sendShape, "recvBuf from SendRecv");
    }else{
        //Packing is what is stored in memory
        PROFILE_SECTION("LocalUnpack");
        UnpackLocalCommRedist(A, A.LockedBuffer());
        PROFILE_STOP;
    }

//    const T* myBuf = LockedBuffer();
//    PrintArray(myBuf, LocalShape(), LocalStrides(), "myBuf");
}


//TODO: Optimize strides when unpacking
//TODO: Check that logic works out (modeStrides being global info applied to local info)
template <typename T>
void DistTensor<T>::UnpackLocalCommRedist(const DistTensor<T>& A, const T* unpackBuf)
{
    Unsigned order = A.Order();
    T* dataBuf = Buffer();
//    const T* srcBuf = A.LockedBuffer();

    //GridView information
    const tmen::GridView gvA = A.GetGridView();
    const tmen::GridView gvB = GetGridView();
    const ObjShape gvAShape = gvA.ParticipatingShape();
    const ObjShape gvBShape = gvB.ParticipatingShape();

    //Different striding information
    std::vector<Unsigned> commLCMs = tmen::LCMs(gvAShape, gvBShape);
    std::vector<Unsigned> modeStrideFactor = ElemwiseDivide(commLCMs, gvAShape);

    //Grid information
    const tmen::Grid& g = Grid();
    const ObjShape gridShape = g.Shape();
    const Location gridLoc = g.Loc();

    const Location zeros(order, 0);
    const Location ones(order, 1);
    Permutation invPermB = DetermineInversePermutation(localPerm_);
    Permutation invPermA = DetermineInversePermutation(A.localPerm_);

    PackData unpackData;
    unpackData.loopShape = PermuteVector(LocalShape(), invPermB);
    unpackData.dstBufStrides = PermuteVector(LocalStrides(), invPermB);
    unpackData.srcBufStrides = PermuteVector(ElemwiseProd(A.LocalStrides(), modeStrideFactor), invPermA);
    unpackData.loopStarts = zeros;
    unpackData.loopIncs = ones;

    const Location myFirstElemLoc = DetermineFirstElem(GetGridView().ParticipatingLoc());

    if(ElemwiseLessThan(myFirstElemLoc, A.Shape())){
        const Location firstLocInA = A.Global2LocalIndex(myFirstElemLoc);
        Unsigned srcBufPtr = Loc2LinearLoc(firstLocInA, A.LocalShape(), A.LocalStrides());
        PackCommHelper(unpackData, order - 1, &(unpackBuf[srcBufPtr]), &(dataBuf[0]));
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
FULL(std::complex<float>);
#endif
FULL(std::complex<double>);
#endif

} //namespace tmen
