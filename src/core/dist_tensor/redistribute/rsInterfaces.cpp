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
    ModeArray reduceModes(1);
    ModeArray scatterModes(1);

    reduceModes[0] = reduceScatterMode;
    scatterModes[0] = reduceScatterMode;

    ModeArray commModes;
    for(i = 0; i < reduceModes.size(); i++){
        ModeDistribution modeDist = A.ModeDist(reduceModes[i]);
        commModes.insert(commModes.end(), modeDist.begin(), modeDist.end());
    }
    std::sort(commModes.begin(), commModes.end());

    ReduceScatterCommRedist(A, reduceModes, scatterModes, commModes);
}

template <typename T>
void DistTensor<T>::ReduceScatterRedistFrom(const DistTensor<T>& A, const ModeArray& reduceModes, const ModeArray& scatterModes){
    Unsigned i;
    const tmen::GridView gv = A.GetGridView();
    const tmen::Grid& g = A.Grid();
    TensorDistribution dist = A.TensorDist();
    ModeDistribution blank(0);

    ObjShape tmpShape = A.Shape();
    for(i = 0; i < reduceModes.size(); i++){
        tmpShape[reduceModes[i]] = gv.Dimension(reduceModes[i]);
    }
    DistTensor<T> tmp(tmpShape, dist, g);
    T* tmpBuf = tmp.Buffer();
    MemZero(&(tmpBuf[0]), prod(tmp.LocalShape()));

    ObjShape tmp2Shape = A.Shape();
    TensorDistribution tmp2Dist = dist;
    for(i = 0; i < scatterModes.size(); i++){
        tmp2Dist[scatterModes[i]] = ConcatenateVectors(tmp2Dist[scatterModes[i]], tmp2Dist[reduceModes[i]]);
        tmp2Dist[reduceModes[i]] = blank;
        tmp2Shape[reduceModes[i]] = 1;
    }

    DistTensor<T> tmp2(tmp2Shape, tmp2Dist, g);
    T* tmp2Buf = tmp2.Buffer();
    MemZero(&(tmp2Buf[0]), prod(tmp2.LocalShape()));

    LocalReduce(tmp, A, reduceModes);

    ModeArray commModes;
    for(i = 0; i < reduceModes.size(); i++){
        ModeDistribution modeDist = A.ModeDist(reduceModes[i]);
        commModes.insert(commModes.end(), modeDist.begin(), modeDist.end());
    }
    std::sort(commModes.begin(), commModes.end());

    tmp2.ReduceScatterCommRedist(tmp, reduceModes, scatterModes, commModes);

    ObjShape BShape = tmp2Shape;
    ModeArray rModes = reduceModes;
    std::sort(rModes.begin(), rModes.end());
    for(i = rModes.size() - 1; i < rModes.size(); i--){
        BShape.erase(BShape.begin() + rModes[i]);
    }

    ResizeTo(BShape);
    CopyLocalBuffer(tmp2);
//    Print(*this, "B after full reduce");
}

template<typename T>
void
DistTensor<T>::ReduceScatterUpdateRedistFrom(const DistTensor<T>& A, const T beta, const ModeArray& reduceModes, const ModeArray& scatterModes)
{
#ifndef RELEASE
    CallStackEntry cse("DistTensor::ReduceScatterUpdateRedistFrom");
#endif
    Unsigned i;

    ObjShape tmpShape = Shape();
    DistTensor<T> tmp(tmpShape, TensorDist(), Grid());
    T* tmpBuf = tmp.Buffer();
    MemZero(&(tmpBuf[0]), prod(tmp.LocalShape()));

    tmp.ReduceScatterRedistFrom(A, reduceModes, scatterModes);

    ResizeTo(tmpShape);
    T* BBuf = Buffer();
    const T* tmpLockedBuf = tmp.LockedBuffer();
    for(i = 0; i < prod(LocalShape()); i++)
        BBuf[i] = beta * BBuf[i] + tmpLockedBuf[i];
//    Print(*this, "B after full reduce");
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
