/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef ROTE_BTAS_NORM_HPP
#define ROTE_BTAS_NORM_HPP

#include "rote/util/vec_util.hpp"
#include "rote/util/btas_util.hpp"
#include "rote/core/view_decl.hpp"

namespace rote{

////////////////////////////////////
// Local interfaces
////////////////////////////////////

template <typename T>
BASE(T) Norm(const Tensor<T>& A){
#ifndef RELEASE
    CallStackEntry("Norm");
#endif
    const std::vector<Unsigned> loopEnd = A.Shape();
    const std::vector<Unsigned> srcBufStrides = A.Strides();
    const T* srcBuf = A.LockedBuffer();
    Unsigned srcBufPtr = 0;
    Unsigned order = loopEnd.size();
    Location curLoc(order, 0);
    Unsigned ptr = 0;
    
    BASE(T) norm = 0;

    if(loopEnd.size() == 0){
		norm += Sqrt(srcBuf[0] * srcBuf[0]);
		return norm;
    }

    bool done = !ElemwiseLessThan(curLoc, loopEnd);

    while(!done){

        norm += srcBuf[srcBufPtr] * srcBuf[srcBufPtr];
        
        //Update
        curLoc[ptr]++;
        srcBufPtr += srcBufStrides[ptr];
        while(ptr < order && curLoc[ptr] >= loopEnd[ptr]){
            curLoc[ptr] = 0;

            srcBufPtr -= srcBufStrides[ptr] * (loopEnd[ptr]);
            ptr++;
            if(ptr >= order){
                done = true;
                break;
            }else{
                curLoc[ptr]++;
                srcBufPtr += srcBufStrides[ptr];
            }
        }
        if(done)
            break;
        ptr = 0;
    }
    return Sqrt(norm);
}

////////////////////////////////////
// Global interfaces
////////////////////////////////////
//TODO: Fix this.  Norm is not calculated correctly :(
template<typename T>
BASE(T) Norm(const DistTensor<T>& A){
#ifndef RELEASE
    CallStackEntry("Norm");
#endif
    BASE(T) local_norm = 0;
    if(A.Participating())
        local_norm = Norm(A.LockedTensor());
    BASE(T) global_norm = mpi::AllReduce(local_norm, mpi::SUM, A.GetParticipatingComm());
    return global_norm;
}

} // namespace rote

#endif // ifndef ROTE_BTAS_NORM_HPP
