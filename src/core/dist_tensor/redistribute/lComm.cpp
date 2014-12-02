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
void DistTensor<T>::LocalCommRedist(const DistTensor<T>& A, const ModeArray& localModes){
//    if(!CheckLocalCommRedist(A, localMode, gridRedistModes))
//        LogicError("LocalRedist: Invalid redistribution request");
    if(!(Participating()))
        return;
    //Packing is what is stored in memory
    UnpackLocalCommRedist(A, localModes);
}


//TODO: Optimize strides when unpacking
//TODO: Check that logic works out (modeStrides being global info applied to local info)
template <typename T>
void DistTensor<T>::UnpackLocalCommRedist(const DistTensor<T>& A, const ModeArray& lModes)
{
    Unsigned i;
    Unsigned order = A.Order();
    T* dataBuf = Buffer();
    const T* srcBuf = A.LockedBuffer();

//    printf("srcBuf:");
//    for(Unsigned i = 0; i < prod(A.LocalShape()); i++){
//        printf(" %d", srcBuf[i]);
//    }
//    printf("\n");

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
//    unpackData.dstBufStrides = PermuteVector(LocalStrides(), invPerm);
    unpackData.dstBufStrides = PermuteVector(LocalStrides(), invPermB);
    unpackData.srcBufStrides = PermuteVector(ElemwiseProd(A.LocalStrides(), modeStrideFactor), invPermA);
//    unpackData.srcBufStrides[lMode] *= nRedistProcs;
    unpackData.loopStarts = zeros;
    unpackData.loopIncs = ones;

    const Location myFirstElemLoc = ModeShifts();

    if(ElemwiseLessThan(myFirstElemLoc, A.Shape())){
        const Location firstLocInA = A.Global2LocalIndex(myFirstElemLoc);
        Unsigned srcBufPtr = 0;
        for(i = 0; i < lModes.size(); i++)
            srcBufPtr += firstLocInA[lModes[i]] * A.LocalModeStride(lModes[i]);
        PackCommHelper(unpackData, order - 1, &(srcBuf[srcBufPtr]), &(dataBuf[0]));
    }

//    printf("dataBuf:");
//    for(Unsigned i = 0; i < prod(LocalShape()); i++){
//        printf(" %d", dataBuf[i]);
//    }
//    printf("\n");
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
