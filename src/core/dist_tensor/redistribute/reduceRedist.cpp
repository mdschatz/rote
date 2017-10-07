/*
   Copyright (c) 2009-2013, Jack Poulson
                      2013, Jeff Hammond
                      2013, Jed Brown
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#include "rote.hpp"
#include <algorithm>

namespace rote{

////////////////////////////////
// Workhorse interface
////////////////////////////////

template<typename T>
void
DistTensor<T>::ReduceUpdateRedistFrom(const RedistType& redistType, const T alpha, const DistTensor<T>& A, const T beta, const ModeArray& rModes)
{
#ifndef RELEASE
    CallStackEntry cse("DistTensor::ReduceUpdateRedistFrom");
#endif
    Unsigned i, j;
    const rote::GridView gv = A.GetGridView();
    const rote::Grid& g = A.Grid();
    TensorDistribution dist = A.TensorDist();

    //Sort the rModes just in case
    ModeArray sortedRModes = rModes;
    SortVector(sortedRModes);

    //Set up tmp for holding alpha*A
    DistTensor<T> tmp(A.TensorDist(), g);
    tmp.AlignWith(A);

    //Set up tmp2 for holding beta*B
    ObjShape tmp2Shape = this->Shape();
    TensorDistribution tmp2Dist = this->TensorDist();
    tmp2Dist.IntroduceUnitModeDists(sortedRModes);
//    std::vector<Unsigned> tmp2Aligns = Alignments();
    std::vector<Unsigned> tmp2PermVals = this->localPerm_.Entries();
    std::vector<Unsigned> tmp2Strides = this->LocalStrides();

    for(i = 0; i < sortedRModes.size(); i++){
        Mode rMode = sortedRModes[i];

        for(j = 0; j < tmp2PermVals.size(); j++)
            if(tmp2PermVals[j] >= rMode)
                tmp2PermVals[j]++;
        tmp2PermVals.insert(tmp2PermVals.begin() + rMode, rMode);

        if(rMode == tmp2Strides.size()){
            if(rMode == 0){
                tmp2Strides.insert(tmp2Strides.begin() + rMode, 1);
            }else{
                tmp2Strides.insert(tmp2Strides.begin() + rMode, tmp2Strides[tmp2Strides.size() - 1] * tmp2Shape[tmp2Shape.size() - 1]);
            }
        }else{
            tmp2Strides.insert(tmp2Strides.begin() + rMode, tmp2Strides[rMode]);
        }
        tmp2Shape.insert(tmp2Shape.begin() + rMode, Min(1, A.Dimension(rMode)));
    }

    DistTensor<T> tmp2(tmp2Shape, tmp2Dist, g);
    Permutation tmp2Perm(tmp2PermVals);
    tmp2.SetLocalPermutation(tmp2Perm);
    std::vector<Unsigned> tmp2Aligns = this->Alignments();
    for(i = 0; i < rModes.size(); i++)
        tmp2Aligns.push_back(0);

    tmp2.Attach(tmp2Shape, tmp2Aligns, this->Buffer(), tmp2Strides, g);

    if(alpha == T(0))
        Zero(tmp);
    else{
        if(ElemwiseLessThanEqualTo(FilterVector(A.Shape(), sortedRModes), FilterVector(A.GridViewShape(), sortedRModes))){
            tmp.LockedAttach(A.Shape(), A.Alignments(), A.LockedBuffer(), A.LocalPermutation(), A.LocalStrides(), g);
        }else{
        	ObjShape tmpShape = A.Shape();

        	for(i = 0; i < sortedRModes.size(); i++)
        		if(A.Dimension(sortedRModes[i]) > gv.Dimension(sortedRModes[i]))
        			tmpShape[sortedRModes[i]] = gv.Dimension(sortedRModes[i]);
        	tmp.ResizeTo(tmpShape);
        	Zero(tmp);
            LocalReduce(A, tmp, sortedRModes);
        }
    }


    ModeArray commModes = A.TensorDist().Filter(sortedRModes).UsedModes().Entries();
    SortVector(commModes);

//    PrintData(tmp, "tmp data");
//    PrintData(tmp2, "tmp2 data");
//    Print(tmp2, "tmp2 before RTO");
//    Print(tmp, "tmp before RTO");

    switch(redistType){
		case RS:  tmp2.ReduceScatterUpdateCommRedist(alpha, tmp, beta, sortedRModes, commModes); break;
		case RTO: tmp2.ReduceToOneUpdateCommRedist(alpha, tmp, beta, commModes); break;
		case AR:  tmp2.AllReduceUpdateCommRedist(alpha, tmp, beta, commModes); break;
		default: break;
    }
}

#define FULL(T) \
    template class DistTensor<T>;

FULL(Int)
#ifndef DISABLE_FLOAT
FULL(float)
#endif
FULL(double)

#ifndef DISABLE_COMPLEX
#ifndef DISABLE_FLOAT
FULL(std::complex<float>)
#endif
FULL(std::complex<double>)
#endif

} //namespace rote
