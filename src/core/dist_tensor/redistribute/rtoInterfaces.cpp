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

template <typename T>
void DistTensor<T>::PartialReduceToOneRedistFrom(const DistTensor<T>& A, const Mode rMode){
    Unsigned i;
//    ObjShape tmpShape = A.Shape();
//    tmpShape[rMode] = A.GetGridView().Dimension(rMode);
//    ResizeTo(tmpShape);
//    ReduceToOneCommRedist(A, rMode);
    ModeArray rModes(1);
    rModes[0] = rMode;
    ObjShape tmpShape = A.Shape();
    tmpShape[rMode] = A.GetGridView().Dimension(rMode);
    ResizeTo(tmpShape);

    ModeArray commModes;
    for(i = 0; i < rModes.size(); i++){
        ModeDistribution modeDist = A.ModeDist(rModes[i]);
        commModes.insert(commModes.end(), modeDist.begin(), modeDist.end());
    }
    std::sort(commModes.begin(), commModes.end());

    ReduceToOneCommRedist(A, rModes, commModes);
}

template <typename T>
void DistTensor<T>::ReduceToOneRedistFrom(const DistTensor<T>& A, const Mode rMode){
    ModeArray rModes(1);
    rModes[0] = rMode;

    ReduceToOneRedistFrom(A, rModes);
}

template <typename T>
void DistTensor<T>::ReduceToOneRedistFrom(const DistTensor<T>& A, const ModeArray& rModes){
    Unsigned i;
    const tmen::GridView gv = A.GetGridView();
    const tmen::Grid& g = A.Grid();
    TensorDistribution dist = A.TensorDist();
    ModeDistribution blank(0);

    ObjShape tmpShape = A.Shape();
    for(i = 0; i < rModes.size(); i++)
        if(tmpShape[rModes[i]] != 0)
            tmpShape[rModes[i]] = Min(gv.Dimension(rModes[i]), A.Dimension(rModes[i]));
    PrintVector(tmpShape, "tmpShape");
    DistTensor<T> tmp(tmpShape, A.TensorDist(), g);
    tmp.AlignWith(A);
    std::cout << "tmp tenDist after alignWith" << tmen::TensorDistToString(tmp.TensorDist()) << std::endl;
    tmp.SetDistribution(A.TensorDist());
    tmp.ResizeTo(tmpShape);
    Zero(tmp);

//    PrintVector(A.Shape(), "shapeA");
//    Print(A.LockedTensor(), "A.Local");
//    PrintVector(A.LocalShape(), "A.LocalShape");
    Print(tmp.Tensor(), "tmp.Local");
    PrintVector(tmp.LocalShape(), "tmp.LocalShape");
    LocalReduce(tmp, A, rModes);

    Print(tmp, "tmp after reduce");

    const Unsigned order = A.Order();
    ObjShape tmp2Shape = A.Shape();

    TensorDistribution tmp2Dist =   A.TensorDist();
    for(i = 0; i < rModes.size(); i++){
        if(tmp2Shape[rModes[i]] != 0)
            tmp2Shape[rModes[i]] = 1;
        ModeDistribution rModeDist = A.ModeDist(rModes[i]);
        tmp2Dist[order].insert(tmp2Dist[order].end(), rModeDist.begin(), rModeDist.end());
        tmp2Dist[rModes[i]] = blank;
    }
    std::sort(tmp2Dist[order].begin(), tmp2Dist[order].end());

    //--------------------------------
    //--------------------------------

    PrintVector(tmp2Shape, "tmp2.shape_ should be");
    DistTensor<T> tmp2(tmp2Shape, tmp2Dist, g);

    PrintVector(tmp2.Alignments(), "aligning tmp2 with alignments");
    PrintVector(tmp.Alignments(), "to tmp with alignments");
    tmp2.AlignWith(tmp);
    tmp2.ResizeTo(tmp2Shape);
    PrintVector(tmp2.Shape(), "tmp2Shape");

    ModeArray commModes;
    std::cout << "tmp tenDist before redist" << tmen::TensorDistToString(tmp.TensorDist()) << std::endl;
    for(i = 0; i < rModes.size(); i++){
        ModeDistribution modeDist = tmp.ModeDist(rModes[i]);
        commModes.insert(commModes.end(), modeDist.begin(), modeDist.end());
    }
    std::sort(commModes.begin(), commModes.end());

    tmp2.ReduceToOneCommRedist(tmp, rModes, commModes);

    PrintVector(tmp2.Shape(), "tmp2Shape after redist");

    Print(tmp2, "tmp2 after redist");
//    Print(tmp2.Tensor(), "tmp2.Local pre removeUnitModes");
//    PrintVector(tmp2.LocalShape(), "tmp2.LocalShape pre removeUnitModes");
//    PrintVector(tmp2.Shape(), "tmp2 shape before remove Unit");
//    PrintVector(tmp2.LocalShape(), "tmp2 localShape before remove Unit");
//    PrintVector(tmp2.LocalStrides(), "tmp2 strides before remove Unit");

    tmp2.RemoveUnitModesRedist(rModes);
//    tmp2.SetAlignmentsAndResize(Alignments(), tmp2.Shape());
//    Print(tmp2.Tensor(), "tmp2.Local post removeUnitModes");
//    PrintVector(tmp2.LocalShape(), "tmp2.LocalShape post removeUnitModes");

    Print(tmp2, "tmp2 after setAlign");
    PrintVector(tmp2.Shape(), "tmp2 shape after remove Unit");
    PrintVector(tmp2.LocalShape(), "tmp2 localShape after remove Unit");
    PrintVector(tmp2.LocalStrides(), "tmp2 strides after remove Unit");

//    ModeArray sortedRModes = rModes;
//    std::sort(sortedRModes.begin(), sortedRModes.end());
//    ObjShape BShape = tmp2Shape;
//    for(i = sortedRModes.size() - 1; i < sortedRModes.size(); i--){
//        BShape.erase(BShape.begin() + sortedRModes[i]);
//    }
//
//    ResizeTo(BShape);

    PrintVector(Shape(), "output Shape");
    PrintVector(LocalShape(), "output LocalShape");

    if(Participating())
        CopyLocalBuffer(tmp2);


    //--------------------------------
    //--------------------------------


//    DistTensor<T> tmp2(tmp2Shape, tmp2Dist, g);
//
//    ModeArray commModes;
//    for(i = 0; i < rModes.size(); i++){
//        ModeDistribution modeDist = tmp.ModeDist(rModes[i]);
//        commModes.insert(commModes.end(), modeDist.begin(), modeDist.end());
//    }
//    std::sort(commModes.begin(), commModes.end());
//
//    tmp2.ReduceToOneCommRedist(tmp, rModes, commModes);
//
//    Print(tmp2, "tmp2 after redist");
////    Print(tmp2.Tensor(), "tmp2.Local pre removeUnitModes");
////    PrintVector(tmp2.LocalShape(), "tmp2.LocalShape pre removeUnitModes");
////    PrintVector(tmp2.Shape(), "tmp2 shape before remove Unit");
////    PrintVector(tmp2.LocalShape(), "tmp2 localShape before remove Unit");
////    PrintVector(tmp2.LocalStrides(), "tmp2 strides before remove Unit");
//
//    tmp2.RemoveUnitModesRedist(rModes);
//    tmp2.SetAlignmentsAndResize(Alignments(), tmp2.Shape());
////    Print(tmp2.Tensor(), "tmp2.Local post removeUnitModes");
////    PrintVector(tmp2.LocalShape(), "tmp2.LocalShape post removeUnitModes");
//
//    Print(tmp2, "tmp2 after setAlign");
//    PrintVector(tmp2.Shape(), "tmp2 shape after remove Unit");
//    PrintVector(tmp2.LocalShape(), "tmp2 localShape after remove Unit");
//    PrintVector(tmp2.LocalStrides(), "tmp2 strides after remove Unit");
//
////    ModeArray sortedRModes = rModes;
////    std::sort(sortedRModes.begin(), sortedRModes.end());
////    ObjShape BShape = tmp2Shape;
////    for(i = sortedRModes.size() - 1; i < sortedRModes.size(); i--){
////        BShape.erase(BShape.begin() + sortedRModes[i]);
////    }
////
////    ResizeTo(BShape);
//
//    PrintVector(Shape(), "output Shape");
//    PrintVector(LocalShape(), "output LocalShape");
//
//    if(Participating())
//        CopyLocalBuffer(tmp2);
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

} //namespace tmen
