/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef ROTE_TENSORS_UNIFORM_HPP
#define ROTE_TENSORS_UNIFORM_HPP

namespace rote {

// Draw each entry from a uniform PDF over a closed ball.
template<typename T>
inline void
MakeUniform( Tensor<T>& A, T center=0, BASE(T) radius=1 )
{
    Unsigned order = A.Order();
    Location loc(order, 0);

    Unsigned ptr = 0;
    bool stop = !ElemwiseLessThan(loc, A.Shape());

    while(!stop){
      A.Set(loc, SampleBall(center, radius) );
        if (loc.size() == 0)
          break;

        //Update
        loc[ptr]++;
        while(loc[ptr] == A.Dimension(ptr)){
            loc[ptr] = 0;
            ptr++;
            if(ptr == order){
                stop = true;
                break;
            }else{
                loc[ptr]++;
            }
        }
        ptr = 0;
    }
}

template<typename T>
inline void
Uniform( Tensor<T>& A, const ObjShape& shape, T center=0, BASE(T) radius=1 )
{
    A.ResizeTo( shape );
    MakeUniform( A, center, radius );
}

template<typename T>
inline Tensor<T>
Uniform( const ObjShape& shape, T center=0, BASE(T) radius=1 )
{
    Tensor<T> A( shape );
    MakeUniform( A, center, radius );
    return A;
}

namespace internal {

template<typename T>
struct MakeUniformHelper
{
    static void Func( DistTensor<T>& A, T center, BASE(T) radius ){
        MakeUniform(A.Tensor(), center, radius);

        //Pack
//        const Unsigned bufSize = prod(A.LocalShape());
//        std::vector<T> buffer(bufSize);
//        MemCopy(&(buffer[0]), A.LockedBuffer(), bufSize);

        //BCast data
        mpi::Comm comm = A.GetCommunicatorForModes(A.GetGridView().FreeModes(), A.Grid());
        mpi::Broadcast(A.Buffer(), prod(A.LocalShape()), 0, comm);

        //Unpack
//        T* localBuffer = A.Buffer();
//        MemCopy(&(localBuffer[0]), &(buffer[0]), bufSize);
    };
};

} // namespace internal

template<typename T>
inline void
MakeUniform( DistTensor<T>& A, T center=0, BASE(T) radius=1 )
{
    internal::MakeUniformHelper<T>::Func( A, center, radius );
}

template<typename T>
inline void
Uniform( DistTensor<T>& A, const ObjShape& shape, T center=0, BASE(T) radius=1 )
{
    A.ResizeTo( shape );
    MakeUniform( A, center, radius );
}

template<typename T>
inline DistTensor<T>
Uniform( const Grid& g, const ObjShape& shape, T center=0, BASE(T) radius=1 )
{
    DistTensor<T> A( shape, g );
    MakeUniform( A, center, radius );
    return A;
}

} // namespace rote

#endif // ifndef ROTE_TENSORS_UNIFORM_HPP
