/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef ROTE_BTAS_SET_ALL_VAL_HPP
#define ROTE_BTAS_SET_ALL_VAL_HPP

namespace rote {

////////////////////////////////////
// Workhorse routines
////////////////////////////////////

template<typename T>
void SetAllVal_fast(const ObjShape& shape, const std::vector<Unsigned>& strides, T * const buf, T val){
    const std::vector<Unsigned> loopEnd = shape;
    const std::vector<Unsigned> bufStrides = strides;
    Unsigned bufPtr = 0;
    Unsigned order = loopEnd.size();
    Location curLoc(order, 0);
    Unsigned ptr = 0;

    if(loopEnd.size() == 0){
        buf[0] = val;
        return;
    }

    bool done = !ElemwiseLessThan(curLoc, loopEnd);

    while(!done){
        buf[bufPtr] = val;
        //Update
        curLoc[ptr]++;
        bufPtr += bufStrides[ptr];
        while(ptr < order && curLoc[ptr] >= loopEnd[ptr]){
            curLoc[ptr] = 0;

            bufPtr -= bufStrides[ptr] * (loopEnd[ptr]);
            ptr++;
            if(ptr >= order){
                done = true;
                break;
            }else{
                curLoc[ptr]++;
                bufPtr += bufStrides[ptr];
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

template<typename T>
inline void
SetAllVal( Tensor<T>& A, T val )
{
    SetAllVal_fast(A.Shape(), A.Strides(), A.Buffer(), val);
}

////////////////////////////////////
// Global interfaces
////////////////////////////////////

template<typename T>
inline void
SetAllVal( DistTensor<T>& A, T val )
{
    SetAllVal( A.Tensor(), val );
}

} // namespace rote

#endif // ifndef ROTE_BTAS_SET_ALL_VAL_HPP
