/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef TMEN_BTAS_ZERO_HPP
#define TMEN_BTAS_ZERO_HPP

namespace tmen {

////////////////////////////////////
// Workhorse routines
////////////////////////////////////

template<typename T>
void ZeroHelper(Mode mode, const ObjShape& shape, const std::vector<Unsigned>& strides, T * const buf){
    Unsigned i;
    Unsigned bufPtr = 0;
    if(mode == 0){
        if(strides[mode] == 1)
            MemZero(&(buf[bufPtr]), shape[mode]);
        else{
            for(i = 0; i < shape[mode]; i++){
                buf[bufPtr] = 0;
                bufPtr += strides[mode];
            }
        }
    }else{
        for(i = 0; i < shape[mode]; i++){
            ZeroHelper(mode - 1, shape, strides, &(buf[bufPtr]));
            bufPtr += strides[mode];
        }
    }
}

template<typename T>
void Zero_fast(const ObjShape& shape, const std::vector<Unsigned>& strides, T * const buf){
    const std::vector<Unsigned> loopEnd = shape;
    const std::vector<Unsigned> bufStrides = strides;
    Unsigned bufPtr = 0;
    Unsigned order = loopEnd.size();
    Location curLoc(order, 0);
    Unsigned ptr = 0;

//    std::string ident = "";
//    for(i = 0; i < packData.loopShape.size() - packMode; i++)
//        ident += "  ";

    if(loopEnd.size() == 0){
        buf[0] = 0;
        return;
    }

    bool done = !ElemwiseLessThan(curLoc, loopEnd);

    while(!done){
        buf[bufPtr] = 0;
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
Zero( Tensor<T>& A )
{
#ifndef RELEASE
    CallStackEntry entry("Zero");
#endif

    Zero_fast(A.Shape(), A.Strides(), A.Buffer());
}

////////////////////////////////////
// Global interfaces
////////////////////////////////////

template<typename T>
inline void
Zero( DistTensor<T>& A )
{
#ifndef RELEASE
    CallStackEntry entry("Zero");
#endif
    Zero( A.Tensor() );
}

} // namespace tmen

#endif // ifndef TMEN_BTAS_ZERO_HPP
