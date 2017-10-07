/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/

#include "rote.hpp"
namespace rote {
template<typename T>
inline void
YAxpBy_fast(T alpha, T beta, T const * const srcBuf, T * const dstBuf, const YAxpByData& data ){
    const std::vector<Unsigned> loopEnd = data.loopShape;
    const std::vector<Unsigned> srcBufStrides = data.srcStrides;
    const std::vector<Unsigned> dstBufStrides = data.dstStrides;
    Unsigned srcBufPtr = 0;
    Unsigned dstBufPtr = 0;
    Unsigned order = loopEnd.size();
    Location curLoc(order, 0);
    Unsigned ptr = 0;

    bool done = !ElemwiseLessThan(curLoc, loopEnd);

    //Alpha cannot be zero (handled in parent call)
    if(alpha == T(0)){
        if(beta == T(0)){
            Zero_fast(loopEnd, dstBufStrides, dstBuf);
        }else if(beta == T(1)){

        }else{
            ScalData scal_data;
            scal_data.loopShape = loopEnd;
            scal_data.srcStrides = dstBufStrides;
            Scal_fast(alpha, dstBuf, scal_data);
        }
    }else if(alpha == T(1)){
        if(beta == T(0)){
            if(loopEnd.size() == 0){
                dstBuf[0] = srcBuf[0];
                return;
            }

            while(!done){
                dstBuf[dstBufPtr] = srcBuf[srcBufPtr];
                //Update
                curLoc[ptr]++;
                dstBufPtr += dstBufStrides[ptr];
                srcBufPtr += srcBufStrides[ptr];
                while(ptr < order && curLoc[ptr] >= loopEnd[ptr]){
                    curLoc[ptr] = 0;

                    dstBufPtr -= dstBufStrides[ptr] * (loopEnd[ptr]);
                    srcBufPtr -= srcBufStrides[ptr] * (loopEnd[ptr]);
                    ptr++;
                    if(ptr >= order){
                        done = true;
                        break;
                    }else{
                        curLoc[ptr]++;
                        dstBufPtr += dstBufStrides[ptr];
                        srcBufPtr += srcBufStrides[ptr];
                    }
                }
                if(done)
                    break;
                ptr = 0;
            }
        }else if(beta == T(1)){
            if(loopEnd.size() == 0){
                dstBuf[0] = srcBuf[0] + dstBuf[0];
                return;
            }

            while(!done){
                dstBuf[dstBufPtr] = srcBuf[srcBufPtr] + dstBuf[dstBufPtr];
                //Update
                curLoc[ptr]++;
                dstBufPtr += dstBufStrides[ptr];
                srcBufPtr += srcBufStrides[ptr];
                while(ptr < order && curLoc[ptr] >= loopEnd[ptr]){
                    curLoc[ptr] = 0;

                    dstBufPtr -= dstBufStrides[ptr] * (loopEnd[ptr]);
                    srcBufPtr -= srcBufStrides[ptr] * (loopEnd[ptr]);
                    ptr++;
                    if(ptr >= order){
                        done = true;
                        break;
                    }else{
                        curLoc[ptr]++;
                        dstBufPtr += dstBufStrides[ptr];
                        srcBufPtr += srcBufStrides[ptr];
                    }
                }
                if(done)
                    break;
                ptr = 0;
            }
        }else{
            if(loopEnd.size() == 0){
                dstBuf[0] = srcBuf[0] + beta*dstBuf[0];
                return;
            }

            while(!done){
                dstBuf[dstBufPtr] = srcBuf[srcBufPtr] + beta*dstBuf[dstBufPtr];
                //Update
                curLoc[ptr]++;
                dstBufPtr += dstBufStrides[ptr];
                srcBufPtr += srcBufStrides[ptr];
                while(ptr < order && curLoc[ptr] >= loopEnd[ptr]){
                    curLoc[ptr] = 0;

                    dstBufPtr -= dstBufStrides[ptr] * (loopEnd[ptr]);
                    srcBufPtr -= srcBufStrides[ptr] * (loopEnd[ptr]);
                    ptr++;
                    if(ptr >= order){
                        done = true;
                        break;
                    }else{
                        curLoc[ptr]++;
                        dstBufPtr += dstBufStrides[ptr];
                        srcBufPtr += srcBufStrides[ptr];
                    }
                }
                if(done)
                    break;
                ptr = 0;
            }
        }
    }else{
        if(beta == T(0)){
            if(loopEnd.size() == 0){
                dstBuf[0] = alpha*srcBuf[0];
                return;
            }

            while(!done){
                dstBuf[dstBufPtr] = alpha*srcBuf[srcBufPtr];
                //Update
                curLoc[ptr]++;
                dstBufPtr += dstBufStrides[ptr];
                srcBufPtr += srcBufStrides[ptr];
                while(ptr < order && curLoc[ptr] >= loopEnd[ptr]){
                    curLoc[ptr] = 0;

                    dstBufPtr -= dstBufStrides[ptr] * (loopEnd[ptr]);
                    srcBufPtr -= srcBufStrides[ptr] * (loopEnd[ptr]);
                    ptr++;
                    if(ptr >= order){
                        done = true;
                        break;
                    }else{
                        curLoc[ptr]++;
                        dstBufPtr += dstBufStrides[ptr];
                        srcBufPtr += srcBufStrides[ptr];
                    }
                }
                if(done)
                    break;
                ptr = 0;
            }
        }else if(beta == T(1)){
            if(loopEnd.size() == 0){
                dstBuf[0] = alpha*srcBuf[0] + dstBuf[0];
                return;
            }

            while(!done){
                dstBuf[dstBufPtr] = alpha*srcBuf[srcBufPtr] + dstBuf[dstBufPtr];
                //Update
                curLoc[ptr]++;
                dstBufPtr += dstBufStrides[ptr];
                srcBufPtr += srcBufStrides[ptr];
                while(ptr < order && curLoc[ptr] >= loopEnd[ptr]){
                    curLoc[ptr] = 0;

                    dstBufPtr -= dstBufStrides[ptr] * (loopEnd[ptr]);
                    srcBufPtr -= srcBufStrides[ptr] * (loopEnd[ptr]);
                    ptr++;
                    if(ptr >= order){
                        done = true;
                        break;
                    }else{
                        curLoc[ptr]++;
                        dstBufPtr += dstBufStrides[ptr];
                        srcBufPtr += srcBufStrides[ptr];
                    }
                }
                if(done)
                    break;
                ptr = 0;
            }
        }else{
            if(loopEnd.size() == 0){
                dstBuf[0] = alpha*srcBuf[0] + beta*dstBuf[0];
                return;
            }

            while(!done){
                dstBuf[dstBufPtr] = srcBuf[srcBufPtr] + beta*dstBuf[dstBufPtr];
                //Update
                curLoc[ptr]++;
                dstBufPtr += dstBufStrides[ptr];
                srcBufPtr += srcBufStrides[ptr];
                while(ptr < order && curLoc[ptr] >= loopEnd[ptr]){
                    curLoc[ptr] = 0;

                    dstBufPtr -= dstBufStrides[ptr] * (loopEnd[ptr]);
                    srcBufPtr -= srcBufStrides[ptr] * (loopEnd[ptr]);
                    ptr++;
                    if(ptr >= order){
                        done = true;
                        break;
                    }else{
                        curLoc[ptr]++;
                        dstBufPtr += dstBufStrides[ptr];
                        srcBufPtr += srcBufStrides[ptr];
                    }
                }
                if(done)
                    break;
                ptr = 0;
            }
        }
    }
}

////////////////////////////////////
// Local interfaces
////////////////////////////////////

template<typename T>
inline void
YAxpBy( T alpha, const Tensor<T>& X, const Permutation& permXToY, T beta, Tensor<T>& Y){
#ifndef RELEASE
    CallStackEntry entry("YxpBy");
#endif
    YAxpByData data;
    data.loopShape = Y.Shape();
    data.srcStrides = PermuteVector(X.Strides(), permXToY);
    data.dstStrides = Y.Strides();

    const T* srcBuf = X.LockedBuffer();
    T* dstBuf = Y.Buffer();

    YAxpBy_fast(alpha, beta, srcBuf, dstBuf, data);
}


////////////////////////////////////
// Global interfaces
////////////////////////////////////

template<typename T>
inline void
YxpBy( const DistTensor<T>& X, T beta, DistTensor<T>& Y )
{
	YAxpBy(T(1), X, beta, Y);
}

template<typename T>
inline void
YAxpy( T alpha, const DistTensor<T>& X, DistTensor<T>& Y )
{
	YAxpBy(alpha, X, T(1), Y);
}

template<typename T>
inline void
Yxpy( const DistTensor<T>& X, DistTensor<T>& Y )
{
	YAxpBy(T(1), X, T(1), Y);
}

template<typename T>
inline void
YAxpBy( T alpha, const DistTensor<T>& X, T beta, DistTensor<T>& Y )
{
#ifndef RELEASE
    CallStackEntry entry("Yxpy");
    if( X.Grid() != Y.Grid() )
        LogicError
        ("X and Y must be distributed over the same grid");
#endif
    Permutation permXToY = X.LocalPermutation().PermutationTo(Y.LocalPermutation());
    YAxpBy(alpha, X.LockedTensor(), permXToY, beta, Y.Tensor());
}

//Non-template functions
//bool AnyFalseElem(const std::vector<bool>& vec);
#define PROTO(T) \
	template void Yxpy( const DistTensor<T>& X, DistTensor<T>& Y ); \
	template void YAxpy( T alpha, const DistTensor<T>& X, DistTensor<T>& Y ); \
	template void YxpBy( const DistTensor<T>& X, T beta, DistTensor<T>& Y ); \
	template void YAxpBy( T alpha, const DistTensor<T>& X, T beta, DistTensor<T>& Y ); \
	template void YAxpBy_fast(T alpha, T beta, T const * const srcBuf, T * const dstBuf, const YAxpByData& data );


//PROTO(Unsigned)
PROTO(Int)
PROTO(float)
PROTO(double)
//PROTO(char)

#ifndef DISABLE_COMPLEX
#ifndef DISABLE_FLOAT
PROTO(std::complex<float>)
#endif
PROTO(std::complex<double>)
#endif

} // namespace rote
