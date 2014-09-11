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
void DistTensor<T>::PartialReduceScatterRedistFrom(const DistTensor<T>& A, const Mode reduceScatterMode){

    Unsigned i;
    //ObjShape tmpShape = A.Shape();
    //tmpShape[reduceScatterMode] = A.GetGridView().Dimension(reduceScatterMode);
    //ResizeTo(tmpShape);
//    ModeArray reduceModes(1);
//    ModeArray scatterModes(1);
//
//    reduceModes[0] = reduceScatterMode;
//    scatterModes[0] = reduceScatterMode;
//
//    ModeArray commModes;
//    for(i = 0; i < reduceModes.size(); i++){
//        ModeDistribution modeDist = A.ModeDist(reduceModes[i]);
//        commModes.insert(commModes.end(), modeDist.begin(), modeDist.end());
//    }
//    std::sort(commModes.begin(), commModes.end());
//
//    ReduceScatterCommRedist(A, reduceModes, scatterModes, commModes);
}

template <typename T>
void DistTensor<T>::ReduceScatterRedistFrom(const DistTensor<T>& A, const ModeArray& rModes, const ModeArray& sModes){
    Unsigned i;
    const tmen::GridView gv = A.GetGridView();
    const tmen::Grid& g = A.Grid();
    TensorDistribution dist = A.TensorDist();
    ModeDistribution blank(0);

    ObjShape tmpShape = A.Shape();
    for(i = 0; i < rModes.size(); i++){
        tmpShape[rModes[i]] = Min(gv.Dimension(rModes[i]), A.Dimension(rModes[i]));
    }

    DistTensor<T> tmp(tmpShape, A.TensorDist(), g);
    tmp.AlignWith(A);
    tmp.SetDistribution(A.TensorDist());
    tmp.ResizeTo(tmpShape);
    Zero(tmp);

    LocalReduce(tmp, A, rModes);

    ObjShape tmp2Shape = A.Shape();
    TensorDistribution tmp2Dist = A.TensorDist();
    for(i = 0; i < sModes.size(); i++){
        tmp2Shape[rModes[i]] = Min(1, A.Dimension(rModes[i]));
        ModeDistribution rModeDist = A.ModeDist(rModes[i]);
        tmp2Dist[sModes[i]].insert(tmp2Dist[sModes[i]].end(), rModeDist.begin(), rModeDist.end());
        tmp2Dist[rModes[i]] = blank;
    }

    DistTensor<T> tmp2(tmp2Shape, tmp2Dist, g);
    tmp2.AlignWith(tmp);
    tmp2.ResizeTo(tmp2Shape);
    tmp2.SetDistribution(tmp2Dist);


    ModeArray commModes;
    for(i = 0; i < rModes.size(); i++){
        ModeDistribution modeDist = tmp.ModeDist(rModes[i]);
        commModes.insert(commModes.end(), modeDist.begin(), modeDist.end());
    }
    std::sort(commModes.begin(), commModes.end());

    tmp2.ReduceScatterCommRedist(tmp, rModes, sModes, commModes);

    tmp2.RemoveUnitModesRedist(rModes);

//    ObjShape BShape = tmp2Shape;
//    ModeArray rModes = reduceModes;
//    std::sort(rModes.begin(), rModes.end());
//    for(i = rModes.size() - 1; i < rModes.size(); i--){
//        BShape.erase(BShape.begin() + rModes[i]);
//    }
//
//    ResizeTo(BShape);
    if(Participating())
        CopyLocalBuffer(tmp2);
}

template<typename T>
void
DistTensor<T>::ReduceScatterUpdateRedistFrom(const DistTensor<T>& A, const T beta, const ModeArray& reduceModes, const ModeArray& scatterModes)
{
#ifndef RELEASE
    CallStackEntry cse("DistTensor::ReduceScatterUpdateRedistFrom");
#endif

    ObjShape tmpShape = Shape();
    DistTensor<T> tmp(tmpShape, TensorDist(), Grid());
    T* tmpBuf = tmp.Buffer();
    MemZero(&(tmpBuf[0]), prod(tmp.LocalShape()));

    tmp.ReduceScatterRedistFrom(A, reduceModes, scatterModes);

    ResizeTo(tmpShape);

    YxpBy(tmp, beta, *this);
}

template <typename T>
void DistTensor<T>::ReduceScatterRedistFrom(const DistTensor<T>& A, const Mode reduceMode, const Mode scatterMode){

    ModeArray reduceModes(1);
    ModeArray scatterModes(1);

    reduceModes[0] = reduceMode;
    scatterModes[0] = scatterMode;

    ReduceScatterRedistFrom(A, reduceModes, scatterModes);
}

template<typename T>
void
DistTensor<T>::ReduceScatterUpdateRedistFrom(const DistTensor<T>& A, const T beta, const Mode reduceMode, const Mode scatterMode)
{
#ifndef RELEASE
    CallStackEntry cse("DistTensor::ReduceScatterUpdateRedistFrom");
#endif

    ModeArray reduceModes(1);
    ModeArray scatterModes(1);

    reduceModes[0] = reduceMode;
    scatterModes[0] = scatterMode;

    ReduceScatterUpdateRedistFrom(A, beta, reduceModes, scatterModes);
}

template <typename T>
void DistTensor<T>::ReduceScatterRedistFromWithPermutation(const DistTensor<T>& A, const ModeArray& rModes, const ModeArray& sModes, const Permutation& perm){
    Unsigned i;
    const tmen::GridView gv = A.GetGridView();
    const tmen::Grid& g = A.Grid();
    TensorDistribution dist = A.TensorDist();
    ModeDistribution blank(0);

    ObjShape tmpShape = A.Shape();
    for(i = 0; i < rModes.size(); i++){
        tmpShape[rModes[i]] = Min(gv.Dimension(rModes[i]), A.Dimension(rModes[i]));
    }

    DistTensor<T> tmp(tmpShape, A.TensorDist(), g);
    tmp.AlignWith(A);
    tmp.SetDistribution(A.TensorDist());
    tmp.ResizeTo(tmpShape);
    Zero(tmp);

    LocalReduce(tmp, A, rModes);

    ObjShape tmp2Shape = A.Shape();
    TensorDistribution tmp2Dist = A.TensorDist();
    for(i = 0; i < sModes.size(); i++){
        tmp2Shape[rModes[i]] = Min(1, A.Dimension(rModes[i]));
        ModeDistribution rModeDist = A.ModeDist(rModes[i]);
        tmp2Dist[sModes[i]].insert(tmp2Dist[sModes[i]].end(), rModeDist.begin(), rModeDist.end());
        tmp2Dist[rModes[i]] = blank;
    }

    DistTensor<T> tmp2(tmp2Shape, tmp2Dist, g);
    tmp2.AlignWith(tmp);
    tmp2.ResizeTo(tmp2Shape);
    tmp2.SetDistribution(tmp2Dist);


    ModeArray commModes;
    for(i = 0; i < rModes.size(); i++){
        ModeDistribution modeDist = tmp.ModeDist(rModes[i]);
        commModes.insert(commModes.end(), modeDist.begin(), modeDist.end());
    }
    std::sort(commModes.begin(), commModes.end());

    tmp2.ReduceScatterCommRedistWithPermutation(tmp, rModes, sModes, commModes, perm);

    tmp2.RemoveUnitModesRedist(rModes);

//    ObjShape BShape = tmp2Shape;
//    ModeArray rModes = reduceModes;
//    std::sort(rModes.begin(), rModes.end());
//    for(i = rModes.size() - 1; i < rModes.size(); i--){
//        BShape.erase(BShape.begin() + rModes[i]);
//    }
//
//    ResizeTo(BShape);
    if(Participating())
        CopyLocalBuffer(tmp2);
}

template<typename T>
void
DistTensor<T>::ReduceScatterUpdateRedistFromWithPermutation(const DistTensor<T>& A, const T beta, const ModeArray& reduceModes, const ModeArray& scatterModes, const Permutation& perm)
{
#ifndef RELEASE
    CallStackEntry cse("DistTensor::ReduceScatterUpdateRedistFrom");
#endif

    ObjShape tmpShape = Shape();
    DistTensor<T> tmp(tmpShape, TensorDist(), Grid());
    T* tmpBuf = tmp.Buffer();
    MemZero(&(tmpBuf[0]), prod(tmp.LocalShape()));

    tmp.ReduceScatterRedistFromWithPermutation(A, reduceModes, scatterModes, perm);

    ResizeTo(tmpShape);

    YxpBy(tmp, beta, *this);
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
