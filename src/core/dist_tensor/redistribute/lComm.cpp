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
    if(A.Order() != this->Order())
        LogicError("CheckLocalRedist: Objects being redistributed must be of same order");

    Unsigned i, j;
    TensorDistribution distA = A.TensorDist();
    ModeDistribution localModeDistA = A.ModeDist(localMode);
    ModeDistribution localModeDistB = this->ModeDist(localMode);

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
void DistTensor<T>::LocalCommRedist(const DistTensor<T>& A, const Mode localMode, const ModeArray& gridRedistModes){
    if(!this->CheckLocalCommRedist(A, localMode, gridRedistModes))
        LogicError("LocalRedist: Invalid redistribution request");
    if(!(this->Participating()))
        return;
    //Packing is what is stored in memory
    UnpackLocalCommRedist(A, localMode, gridRedistModes);
}

template <typename T>
void DistTensor<T>::UnpackLocalCommRedist(const DistTensor<T>& A, const Mode lMode, const ModeArray& gridRedistModes)
{
    Unsigned order = A.Order();
    T* dataBuf = this->Buffer();
    const T* srcBuf = A.LockedBuffer();

//    printf("srcBuf:");
//    for(Unsigned i = 0; i < prod(A.LocalShape()); i++){
//        printf(" %d", srcBuf[i]);
//    }
//    printf("\n");

    const tmen::Grid& g = A.Grid();
    const tmen::GridView gvA = A.GetGridView();
    const tmen::GridView gvB = GetGridView();
    const Unsigned nRedistProcs = prod(FilterVector(g.Shape(), gridRedistModes));

    const Location zeros(order, 0);
    const Location ones(order, 1);
    PackData unpackData;
    unpackData.loopShape = LocalShape();
    unpackData.dstBufStrides = LocalStrides();
    unpackData.srcBufStrides = A.LocalStrides();
    unpackData.srcBufStrides[lMode] = nRedistProcs;
    unpackData.loopStarts = zeros;
    unpackData.loopIncs = ones;

//    ModeArray commModes = gridRedistModes;
//    std::sort(commModes.begin(), commModes.end());
    const ObjShape redistShape = FilterVector(Grid().Shape(), gridRedistModes);
//    const ObjShape commShape = FilterVector(Grid().Shape(), commModes);
//    const Permutation redistPerm = DeterminePermutation(commModes, gridRedistModes);

    Location myCommLoc = FilterVector(g.Loc(), gridRedistModes);
    Unsigned myCommLinLoc = Loc2LinearLoc(myCommLoc, redistShape);

//    PrintVector(myCommLoc, "commLoc");
//    std::cout << "commLinLoc: " << myCommLinLoc << std::endl;
    PackCommHelper(unpackData, order - 1, &(srcBuf[myCommLinLoc * A.LocalModeStride(lMode)]), &(dataBuf[0]));

//    printf("dataBuf:");
//    for(Unsigned i = 0; i < prod(this->LocalShape()); i++){
//        printf(" %d", dataBuf[i]);
//    }
//    printf("\n");
}

#define PROTO(T) \
        template Int  DistTensor<T>::CheckLocalCommRedist(const DistTensor<T>& A, const Mode localMode, const ModeArray& gridRedistModes); \
        template void DistTensor<T>::LocalCommRedist(const DistTensor<T>& A, const Mode localMode, const ModeArray& gridRedistModes); \
        template void DistTensor<T>::UnpackLocalCommRedist(const DistTensor<T>& A, const Mode localMode, const ModeArray& gridRedistModes);

PROTO(int)
PROTO(float)
PROTO(double)
PROTO(Complex<float>)
PROTO(Complex<double>)

} //namespace tmen
