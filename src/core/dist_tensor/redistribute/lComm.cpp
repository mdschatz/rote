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
Int CheckLocalCommRedist(DistTensor<T>& B, const DistTensor<T>& A, const Mode localMode, const ModeArray& gridRedistModes){
    if(A.Order() != B.Order())
        LogicError("CheckLocalRedist: Objects being redistributed must be of same order");

    Unsigned i, j;
    TensorDistribution distA = A.TensorDist();
    TensorDistribution distB = B.TensorDist();
    ModeDistribution localModeDistA = A.ModeDist(localMode);
    ModeDistribution localModeDistB = B.ModeDist(localMode);

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
void LocalCommRedist(DistTensor<T>& B, const DistTensor<T>& A, const Mode localMode, const ModeArray& gridRedistModes){
    if(!CheckLocalRedist(B, A, localMode, gridRedistModes))
        LogicError("LocalRedist: Invalid redistribution request");

    //Packing is what is stored in memory
    UnpackLocalCommRedist(B, A, localMode, gridRedistModes);
}

template <typename T>
void UnpackLocalCommRedist(DistTensor<T>& B, const DistTensor<T>& A, const Mode lMode, const ModeArray& gridRedistModes)
{
    const Unsigned order = A.Order();
    const Location start(order, 0);
    T* dstBuf = B.Buffer(start);
    const T* srcBuf = A.LockedBuffer(start);



    const tmen::GridView gvA = A.GridView();
    const tmen::GridView gvB = B.GridView();

    const tmen::Grid& g = A.Grid();

    ModeDistribution lModeDistA = A.ModeDist(lMode);
    ModeDistribution lModeDistB = B.ModeDist(lMode);

    ModeArray commModes(lModeDistB.begin() + lModeDistA.size(), lModeDistB.end());

    Location myGridLoc = g.Loc();
    ObjShape gridShape = g.Shape();

    Location myCommLoc = FilterVector(myGridLoc, commModes);
    ObjShape commShape = FilterVector(gridShape, commModes);
    Unsigned myCommLinLoc = Loc2LinearLoc(myCommLoc, commShape);

    //NOTE: CHECK THIS IS CORRECT
    Unsigned modeUnpackStride = prod(commShape);

    //Number of slices after the mode to redist
    const ObjShape localShape = A.LocalShape();
    const ObjShape outerSliceShape(localShape.begin() + lMode + 1, localShape.end());
    const Unsigned nOuterSlices = Max(1, prod(outerSliceShape));

    //Number of slices represented by the mode
    const Unsigned nLModeSlices = localShape[lMode];

    //Size of slice to copy
    const ObjShape copySliceShape(localShape.begin(), localShape.begin() + lMode);
    //NOTE: This is based on modeA, different from all other unpacks
    const Unsigned copySliceSize = B.LocalModeStride(lMode);

    //Where we start copying
    const Unsigned elemStartLoc = myCommLinLoc;

    Unsigned lModeSliceNum, outerSliceNum;
    Unsigned lModeDstOff, outerDstOff;
    Unsigned lModeSrcOff, outerSrcOff;
    Unsigned startDstBuf, startSrcBuf;

//    printf("srcBuf:");
//    for(Unsigned i = 0; i < prod(localShape); i++){
//        printf(" %d", srcBuf[i]);
//    }
//    printf("\n");

//    printf("MemCopy info:\n");
//    printf("    elemStartLoc: %d\n", elemStartLoc);
//    printf("    nOuterSlices: %d\n", nOuterSlices);
//    printf("    nLModeSlices: %d\n", nLModeSlices);
//    printf("    copySliceSize: %d\n", copySliceSize);
//    printf("    modeUnpackStride: %d\n", modeUnpackStride);
    for(outerSliceNum = 0; outerSliceNum < nOuterSlices; outerSliceNum++){
        //NOTE: FIX THIS, WE NEED TO SEE HOW MANY TIMES WE RUN THROUGH THE lModeSliceNum loop (similar to some other unpack routine)
        outerDstOff = copySliceSize * ((nLModeSlices - elemStartLoc - 1) / modeUnpackStride + 1) * outerSliceNum;
        outerSrcOff = copySliceSize * nLModeSlices * outerSliceNum;

//        printf("        outerSliceNum: %d\n", outerSliceNum);
//        printf("        outerDstOff: %d\n", outerDstOff);
//        printf("        outerSrcOff: %d\n", outerSrcOff);
        for(lModeSliceNum = elemStartLoc; lModeSliceNum < nLModeSlices; lModeSliceNum += modeUnpackStride){
            lModeDstOff = copySliceSize * (lModeSliceNum - elemStartLoc) / modeUnpackStride;
            lModeSrcOff = copySliceSize * lModeSliceNum;

//            printf("          lModeSliceNum: %d\n", lModeSliceNum);
//            printf("          lModeDstOff: %d\n", lModeDstOff);
//            printf("          lModeSrcOff: %d\n", lModeSrcOff);
            startDstBuf = outerDstOff + lModeDstOff;
            startSrcBuf = outerSrcOff + lModeSrcOff;

//            printf("          startDstBuf: %d\n", startDstBuf);
//            printf("          startSrcBuf: %d\n", startSrcBuf);
            MemCopy(&(dstBuf[startDstBuf]), &(srcBuf[startSrcBuf]), copySliceSize);
        }
    }
//    printf("dstBuf:");
//    for(Unsigned i = 0; i < prod(B.LocalShape()); i++){
//        printf(" %d", dstBuf[i]);
//    }
//    printf("\n");
}

#define PROTO(T) \
        template Int CheckLocalCommRedist(DistTensor<T>& B, const DistTensor<T>& A, const Mode localMode, const ModeArray& gridRedistModes); \
        template void LocalCommRedist(DistTensor<T>& B, const DistTensor<T>& A, const Mode localMode, const ModeArray& gridRedistModes); \
        template void UnpackLocalCommRedist(DistTensor<T>& B, const DistTensor<T>& A, const Mode localMode, const ModeArray& gridRedistModes);

PROTO(int)
PROTO(float)
PROTO(double)

} //namespace tmen
