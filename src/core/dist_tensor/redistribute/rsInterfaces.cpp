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

//    Unsigned i;
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
void DistTensor<T>::ReduceScatterRedistFrom(const DistTensor<T>& A, const ModeArray& rModes, const ModeArray& sModes){
    PROFILE_SECTION("RSRedist");
    Unsigned i;
    const tmen::GridView gv = A.GetGridView();
    const tmen::Grid& g = A.Grid();
    TensorDistribution dist = A.TensorDist();
    ModeDistribution blank(0);

    ObjShape tmpShape = A.Shape();
    for(i = 0; i < rModes.size(); i++){
        tmpShape[rModes[i]] = Min(gv.Dimension(rModes[i]), A.Dimension(rModes[i]));
    }

//    PrintVector(tmpShape, "tmpShape");
    DistTensor<T> tmp(tmpShape, A.TensorDist(), g);
    tmp.AlignWith(A);
    tmp.SetDistribution(A.TensorDist());
    tmp.SetLocalPermutation(A.localPerm_);
    tmp.ResizeTo(tmpShape);

//    PrintVector(tmp.Shape(), "tmpShapeAfterPerm");
    Zero(tmp);

//    const T* dataBuf = A.LockedBuffer();
//    std::cout << "dataBuf:";
//    for(Unsigned i = 0; i < prod(A.LocalShape()); i++){
//        std::cout << " " << dataBuf[i];
//    }
//    std::cout << std::endl;

    //Account for permuted local storage

    LocalReduce(tmp, A, rModes);

//    PrintVector(rModes, "reducing rModes");
//    Print(tmp, "after localReduce");

    ObjShape tmp2Shape = A.Shape();
    TensorDistribution tmp2Dist = A.TensorDist();
    for(i = 0; i < sModes.size(); i++){
        tmp2Shape[rModes[i]] = Min(1, A.Dimension(rModes[i]));
        ModeDistribution rModeDist = A.ModeDist(rModes[i]);
        tmp2Dist[sModes[i]].insert(tmp2Dist[sModes[i]].end(), rModeDist.begin(), rModeDist.end());
        tmp2Dist[rModes[i]] = blank;
    }

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
            if(tmp2Strides.size() == 0)
                val = 1;
            else
                val = tmp2Strides[sortedRModes[0]];
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

    tmp2.ReduceScatterCommRedist(tmp, rModes, sModes, commModes);

    /////////////////////////////////////////
    /////////////////////////////////////////
    //OLD CODE
//    DistTensor<T> tmp2(tmp2Shape, tmp2Dist, g);
//    tmp2.AlignWith(tmp);
//    tmp2.SetLocalPermutation(A.localPerm_);
//    tmp2.ResizeTo(tmp2Shape);
//    tmp2.SetDistribution(tmp2Dist);
//
////    printf("tmpDist: %s\n", tmen::TensorDistToString(tmp.TensorDist()).c_str());
//    ModeArray commModes;
//    for(i = 0; i < rModes.size(); i++){
//        ModeDistribution modeDist = tmp.ModeDist(rModes[i]);
//        commModes.insert(commModes.end(), modeDist.begin(), modeDist.end());
//    }
//    std::sort(commModes.begin(), commModes.end());
//
//    tmp2.ReduceScatterCommRedist(tmp, rModes, sModes, commModes);
//
//    tmp2.RemoveUnitModesRedist(rModes);
//
//    Permutation permB = localPerm_;
//
//    SetAlignmentsAndResize(tmp2.Alignments(), tmp2.Shape());
//    if(Participating())
//        CopyLocalBuffer(tmp2);
    PROFILE_STOP;
}

template<typename T>
void
DistTensor<T>::ReduceScatterUpdateRedistFrom(const DistTensor<T>& A, const T beta, const ModeArray& reduceModes, const ModeArray& scatterModes)
{
#ifndef RELEASE
    CallStackEntry cse("DistTensor::ReduceScatterUpdateRedistFrom");
#endif

    PROFILE_SECTION("RSURedist");
    ObjShape tmpShape = Shape();
    DistTensor<T> tmp(tmpShape, TensorDist(), Grid());
    T* tmpBuf = tmp.Buffer();
    MemZero(&(tmpBuf[0]), prod(tmp.LocalShape()));

    PROFILE_SECTION("RSURSRedist");
    tmp.ReduceScatterRedistFrom(A, reduceModes, scatterModes);
    PROFILE_STOP;

    ResizeTo(tmpShape);

    YxpBy(tmp, beta, *this);
    PROFILE_STOP;
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
