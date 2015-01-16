/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef TMEN_BTAS_SCAL_HPP
#define TMEN_BTAS_SCAL_HPP

namespace tmen {

////////////////////////////////////
// Workhorse routines
////////////////////////////////////

template<typename T>
inline void
Scal_fast(T alpha, T * srcBuf, const ScalData& data ){
    const std::vector<Unsigned> loopEnd = data.loopShape;
    const std::vector<Unsigned> srcBufStrides = data.srcStrides;
    Unsigned srcBufPtr = 0;
    Unsigned order = loopEnd.size();
    Location curLoc(order, 0);
    Unsigned ptr = 0;

    if(alpha == T(0)){
        Zero_fast(loopEnd, srcBufStrides, srcBuf);
    }else if(alpha == T(1)){

    }else{
        if(order == 0){
            srcBuf[0] *= alpha;
            return;
        }

        bool done = !ElemwiseLessThan(curLoc, loopEnd);

        while(!done){

            srcBuf[srcBufPtr] *= alpha;
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
    }
}

////////////////////////////////////
// Local interfaces
////////////////////////////////////

//NOTE: Place appropriate guards
template<typename T>
void
Scal( T alpha, Tensor<T>& X )
{
#ifndef RELEASE
    CallStackEntry entry("Scal");
#endif
    ScalData data;
    data.loopShape = X.Shape();
    data.srcStrides = X.Strides();

    T* srcBuf = X.Buffer();

    Scal_fast(alpha, srcBuf, data);
}

////////////////////////////////////
// Global interfaces
////////////////////////////////////

template<typename T>
void
Scal( T alpha, DistTensor<T>& X )
{
#ifndef RELEASE
    CallStackEntry entry("Scal");
#endif
    Scal(alpha, X.Tensor());
}

} // namespace tmen

#endif // ifndef TMEN_BTAS_AXPY_HPP
