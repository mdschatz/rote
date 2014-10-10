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
void DistTensor<T>::ReduceToOneUpdateRedistFrom(const DistTensor<T>& A, const T beta, const Mode rMode){
    ModeArray rModes(1);
    rModes[0] = rMode;

    ReduceToOneUpdateRedistFrom(A, beta, rModes);
}

template <typename T>
void DistTensor<T>::ReduceToOneRedistFrom(const DistTensor<T>& A, const ModeArray& rModes){
    Unsigned i;
    Unsigned order = A.Order();
    const tmen::GridView gv = A.GetGridView();
    const tmen::Grid& g = A.Grid();
    TensorDistribution dist = A.TensorDist();
    ModeDistribution blank(0);

    ObjShape tmpShape = A.Shape();
    for(i = 0; i < rModes.size(); i++)
        tmpShape[rModes[i]] = Min(gv.Dimension(rModes[i]), A.Dimension(rModes[i]));

    //NOTE: Cannot write (Investigate)
    // DistTensor<T> tmp(order, g);
    // tmp.AlignWith(A);
    // tmp.SetDistribution(A.TensorDist());
    // tmp.ResizeTo(tmpShape);
    DistTensor<T> tmp(tmpShape, A.TensorDist(), g);
    tmp.AlignWith(A);
    tmp.SetDistribution(A.TensorDist());
    tmp.SetLocalPermutation(A.localPerm_);
    tmp.ResizeTo(tmpShape);
    Zero(tmp);

    LocalReduce(tmp, A, rModes);


    ObjShape tmp2Shape = A.Shape();

    TensorDistribution tmp2Dist =   A.TensorDist();
    for(i = 0; i < rModes.size(); i++){
        tmp2Shape[rModes[i]] = Min(1, A.Dimension(rModes[i]));
        ModeDistribution rModeDist = A.ModeDist(rModes[i]);
        tmp2Dist[order].insert(tmp2Dist[order].end(), rModeDist.begin(), rModeDist.end());
        tmp2Dist[rModes[i]] = blank;
    }
    std::sort(tmp2Dist[order].begin(), tmp2Dist[order].end());

    //--------------------------------
    //--------------------------------

    DistTensor<T> tmp2(tmp2Dist, g);

    Permutation tmp2Perm = DefaultPermutation(tmp2.Order());
    for(i = 0; i < localPerm_.size(); i++)
        tmp2Perm[i] = localPerm_[i];
    tmp2.SetLocalPermutation(tmp2Perm);


    std::vector<Unsigned> tmp2Strides = LocalStrides();
    ModeArray sortedRModes = rModes;
    std::sort(sortedRModes.begin(), sortedRModes.end());

    Unsigned val;
    if(sortedRModes.size() != 0){
        if(sortedRModes[0] == 0){
            if(tmp2Strides.size() == 0){
                val = 1;
            }else{
                val = tmp2Strides[sortedRModes[0]];
            }
        }else{
            val = tmp2Strides[sortedRModes[0]-1];
        }
        tmp2Strides.insert(tmp2Strides.begin() + sortedRModes[0], val);
        for(i = 1; i < sortedRModes.size(); i++){
            tmp2Strides.insert(tmp2Strides.begin() + sortedRModes[i], tmp2Strides[sortedRModes[i]-1]);
        }
    }
    tmp2.Attach(tmp2Shape, tmp.Alignments(), Buffer(), tmp2Strides, g);

    ModeArray commModes;
    for(i = 0; i < rModes.size(); i++){
        ModeDistribution modeDist = tmp.ModeDist(rModes[i]);
        commModes.insert(commModes.end(), modeDist.begin(), modeDist.end());
    }
    std::sort(commModes.begin(), commModes.end());

    tmp2.ReduceToOneCommRedist(tmp, rModes, commModes);

    /////////////////////////////////////////
    /////////////////////////////////////////
    //OLD CODE
//    DistTensor<T> tmp2(tmp2Shape, tmp2Dist, g);
//    tmp2.AlignWith(tmp);
//    tmp2.SetLocalPermutation(A.localPerm_);
//    tmp2.ResizeTo(tmp2Shape);
//    tmp2.SetDistribution(tmp2Dist);
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
//    tmp2.RemoveUnitModesRedist(rModes);
//
//    SetAlignmentsAndResize(tmp2.Alignments(), tmp2.Shape());
//    //NOTE: Permutation already performed in unpack of tmp2
//    if(Participating()){
//        CopyLocalBuffer(tmp2);
//    }
}

template<typename T>
void
DistTensor<T>::ReduceToOneUpdateRedistFrom(const DistTensor<T>& A, const T beta, const ModeArray& reduceModes)
{
#ifndef RELEASE
    CallStackEntry cse("DistTensor::ReduceToOneUpdateRedistFrom");
#endif

    ObjShape tmpShape = Shape();
    DistTensor<T> tmp(tmpShape, TensorDist(), Grid());
    Zero(tmp);

    tmp.ReduceToOneRedistFrom(A, reduceModes);

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
