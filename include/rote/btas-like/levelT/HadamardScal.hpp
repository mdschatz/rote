/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef ROTE_BTAS_HADAMARDSCAL_HPP
#define ROTE_BTAS_HADAMARDSCAL_HPP

#include "rote/util/vec_util.hpp"
#include "rote/util/btas_util.hpp"
#include "rote/core/view_decl.hpp"
#include "rote/io/Print.hpp"

namespace rote{

////////////////////////////////////
// Workhorse routines
////////////////////////////////////

template<typename T>
inline void
HadamardScalBC_fast(T const * const bufA, T const * const bufB,  T * const bufC, const HadamardScalData& data) {
    const std::vector<Unsigned> loopBCEnd = data.loopShapeBC;
    const std::vector<Unsigned> bufBCBStrides = data.stridesBCB;
    const std::vector<Unsigned> bufBCCStrides = data.stridesBCC;

    ElemScalData elemScalData;
    elemScalData.loopShape = data.loopShapeABC;
    elemScalData.src1Strides = data.stridesABCA;
    elemScalData.src2Strides = data.stridesABCB;
    elemScalData.dstStrides = data.stridesABCC;

    Unsigned bufBPtr = 0;
    Unsigned bufCPtr = 0;
    Unsigned order = loopBCEnd.size();
    Location curLoc(order, 0);
    Unsigned ptr = 0;

    if(loopBCEnd.size() == 0){
      ElemScal_fast(bufA, bufB, bufC, elemScalData);
      return;
    }

    bool done = !ElemwiseLessThan(curLoc, loopBCEnd);


    while(!done){
        ElemScal_fast(bufA, bufB, bufC, elemScalData);
        //Update
        curLoc[ptr]++;
        bufBPtr += bufBCBStrides[ptr];
        bufCPtr += bufBCCStrides[ptr];
        while(ptr < order && curLoc[ptr] >= loopBCEnd[ptr]){
            curLoc[ptr] = 0;

            bufBPtr -= bufBCBStrides[ptr] * (loopBCEnd[ptr]);
            bufCPtr -= bufBCCStrides[ptr] * (loopBCEnd[ptr]);
            ptr++;
            if(ptr >= order){
                done = true;
                break;
            }else{
                curLoc[ptr]++;
                bufBPtr += bufBCBStrides[ptr];
                bufCPtr += bufBCCStrides[ptr];
            }
        }
        if(done)
            break;
        ptr = 0;
    }
}

template<typename T>
inline void
HadamardScalAC_fast(T const * const bufA, T const * const bufB,  T * const bufC, const HadamardScalData& data) {
    const std::vector<Unsigned> loopACEnd = data.loopShapeAC;
    const std::vector<Unsigned> bufACAStrides = data.stridesACA;
    const std::vector<Unsigned> bufACCStrides = data.stridesACC;

    Unsigned bufAPtr = 0;
    Unsigned bufCPtr = 0;
    Unsigned order = loopACEnd.size();
    Location curLoc(order, 0);
    Unsigned ptr = 0;

    if(loopACEnd.size() == 0){
        HadamardScalBC_fast(bufA, bufB, bufC, data);
        return;
    }

    bool done = !ElemwiseLessThan(curLoc, loopACEnd);

    while(!done){
        HadamardScalBC_fast(bufA, bufB, bufC, data);
        //Update
        curLoc[ptr]++;
        bufAPtr += bufACAStrides[ptr];
        bufCPtr += bufACCStrides[ptr];
        while(ptr < order && curLoc[ptr] >= loopACEnd[ptr]){
            curLoc[ptr] = 0;

            bufAPtr -= bufACAStrides[ptr] * (loopACEnd[ptr]);
            bufCPtr -= bufACCStrides[ptr] * (loopACEnd[ptr]);
            ptr++;
            if(ptr >= order){
                done = true;
                break;
            }else{
                curLoc[ptr]++;
                bufAPtr += bufACAStrides[ptr];
                bufCPtr += bufACCStrides[ptr];
            }
        }
        if(done)
            break;
        ptr = 0;
    }
}

////////////////////////////////////
// Local interfaces
////////////////////////////////////

//NOTE: Add checks for conforming shapes
//NOTE: Convert to incorporate blocked tensors.
template <typename T>
void HadamardScal(const Tensor<T>& A, const Tensor<T>& B, Tensor<T>& C, HadamardScalData data){
#ifndef RELEASE
    CallStackEntry("HadamardScal");
#endif
    const T* bufA = A.LockedBuffer();
    const T* bufB = B.LockedBuffer();
    T* bufC = C.Buffer();

    HadamardScalAC_fast(bufA, bufB, bufC, data);
}

} // namespace rote

#endif // ifndef ROTE_BTAS_HADAMARDSCAL_HPP
