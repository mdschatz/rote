/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef TMEN_TENSORS_UNIFORM_HPP
#define TMEN_TENSORS_UNIFORM_HPP

namespace tmen {

// Draw each entry from a uniform PDF over a closed ball.
template<typename T>
inline void
MakeUniform( Tensor<T>& A, T center=0, BASE(T) radius=1 )
{
#ifndef RELEASE
    CallStackEntry cse("MakeUniform");
#endif
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
//    const Int m = A.Height();
//    const Int n = A.Width();
//    for( Int j=0; j<n; ++j )
//        for( Int i=0; i<m; ++i )
//            A.Set( i, j, SampleBall( center, radius ) );
}

template<typename T>
inline void
Uniform( Tensor<T>& A, const ObjShape& shape, T center=0, BASE(T) radius=1 )
{
#ifndef RELEASE
    CallStackEntry cse("Uniform");
#endif
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
//        Unsigned order = A.Order();
//        Location loc(order, 0);
//
//        Unsigned ptr = 0;
//        bool stop = !ElemwiseLessThan(loc, A.Tensor().Shape());
//
//        while(!stop){
//          A.SetLocal(loc, SampleBall(center , radius) );
//            if (loc.size() == 0)
//              break;
//
//
//            //Update
//            loc[ptr]++;
//            while(loc[ptr] == A.LocalDimension(ptr)){
//                loc[ptr] = 0;
//                ptr++;
//                if(ptr == order){
//                    stop = true;
//                    break;
//                }else{
//                    loc[ptr]++;
//                }
//            }
//            ptr = 0;
//        }

        //Pack
        const Unsigned bufSize = prod(A.LocalShape());
        std::vector<T> buffer(bufSize);
        MemCopy(&(buffer[0]), A.LockedBuffer(), bufSize);

        //BCast result
        mpi::Comm comm = A.GetCommunicatorForModes(A.GetGridView().FreeModes(), A.Grid());
        mpi::Broadcast(&(buffer[0]), bufSize, 0, comm);

//        std::cout << "recv'd random buffer: ";
//        for(Unsigned i = 0; i < bufSize; i++)
//            std::cout << buffer[i] << " ";
//        std::cout << "\n";

        //Unpack
        T* localBuffer = A.Buffer();
        MemCopy(&(localBuffer[0]), &(buffer[0]), bufSize);
    };
};

//template<typename T>
//struct MakeUniformHelper
//{
//    static void Func( DistTensor<T>& A, T center, BASE(T) radius )
//    {
////        Unsigned order = A.Order();
////        Location loc(order, 0);
////
////        Unsigned ptr = 0;
////        bool stop = false;
////
////        while(!stop){
////          A.Set(loc, SampleBall(center , radius) );
////            if (loc.size() == 0)
////              break;
////
////
////            //Update
////            loc[ptr]++;
////            while(loc[ptr] == A.LocalDimension(ptr)){
////                loc[ptr] = 0;
////                ptr++;
////                if(ptr == order){
////                    stop = true;
////                    break;
////                }else{
////                    loc[ptr]++;
////                }
////            }
////            ptr = 0;
////        }
////
////        //Pack
////        const Unsigned bufSize = prod(A.LocalShape());
////        std::vector<T> buffer(bufSize);
////        MemCopy(&(buffer[0]), A.LockedBuffer(), bufSize);
////
////        //BCast result
////        mpi::Comm comm = A.GetCommunicatorForModes(A.GridView().FreeModes());
////        mpi::Broadcast(&(buffer[0]), bufSize, 0, comm);
////
////        //Unpack
////        T* localBuffer = A.Buffer();
////        MemCopy(&(localBuffer[0]), &(buffer[0]), bufSize);
//
////        const Int localHeight = A.LocalHeight();
////        const Int localWidth = A.LocalWidth();
////        for( Int jLoc=0; jLoc<localWidth; ++jLoc )
////            for( Int iLoc=0; iLoc<localHeight; ++iLoc )
////                A.SetLocal( iLoc, jLoc, SampleBall( center, radius ) );
//    }
//};

//
//template<typename T>
//struct MakeUniformHelper<T,MC,STAR>
//{
//    static void Func( DistTensor<T,MC,STAR>& A, T center, BASE(T) radius )
//    {
//        const Grid& grid = A.Grid();
//        if( grid.InGrid() )
//        {
//            const Int n = A.Width();
//            const Int localHeight = A.LocalHeight();
//            const Int bufSize = localHeight*n;
//            std::vector<T> buffer( bufSize );
//
//            // Create random matrix on process column 0, then broadcast
//            if( grid.Col() == 0 )
//            {
//                for( Int j=0; j<n; ++j )
//                    for( Int iLoc=0; iLoc<localHeight; ++iLoc )
//                        buffer[iLoc+j*localHeight] =
//                            SampleBall( center, radius );
//            }
//            mpi::Broadcast( buffer.data(), bufSize, 0, grid.RowComm() );
//
//            // Unpack
//            T* localBuffer = A.Buffer();
//            const Int ldim = A.LDim();
//            PARALLEL_FOR
//            for( Int j=0; j<n; ++j )
//            {
//                const T* bufferCol = &buffer[j*localHeight];
//                T* col = &localBuffer[j*ldim];
//                MemCopy( col, bufferCol, localHeight );
//            }
//        }
//    }
//};
//
//template<typename T>
//struct MakeUniformHelper<T,MD,STAR>
//{
//    static void Func( DistTensor<T,MD,STAR>& A, T center, BASE(T) radius )
//    {
//        const Int n = A.Width();
//        const Int localHeight = A.LocalHeight();
//        for( Int j=0; j<n; ++j )
//            for( Int iLoc=0; iLoc<localHeight; ++iLoc )
//                A.SetLocal( iLoc, j, SampleBall( center, radius ) );
//    }
//};
//
//template<typename T>
//struct MakeUniformHelper<T,MR,MC>
//{
//    static void Func( DistTensor<T,MR,MC>& A, T center, BASE(T) radius )
//    {
//        const Int localHeight = A.LocalHeight();
//        const Int localWidth = A.LocalWidth();
//        for( Int jLoc=0; jLoc<localWidth; ++jLoc )
//            for( Int iLoc=0; iLoc<localHeight; ++iLoc )
//                A.SetLocal( iLoc, jLoc, SampleBall( center, radius ) );
//    }
//};
//
//template<typename T>
//struct MakeUniformHelper<T,MR,STAR>
//{
//    static void Func( DistTensor<T,MR,STAR>& A, T center, BASE(T) radius )
//    {
//        const Grid& grid = A.Grid();
//        const Int n = A.Width();
//        const Int localHeight = A.LocalHeight();
//        const Int bufSize = localHeight*n;
//        std::vector<T> buffer( bufSize );
//
//        // Create random matrix on process row 0, then broadcast
//        if( grid.Row() == 0 )
//        {
//            for( Int j=0; j<n; ++j )
//                for( Int i=0; i<localHeight; ++i )
//                    buffer[i+j*localHeight] = SampleBall( center, radius );
//        }
//        mpi::Broadcast( buffer.data(), bufSize, 0, grid.ColComm() );
//
//        // Unpack
//        T* localBuffer = A.Buffer();
//        const Int ldim = A.LDim();
//        PARALLEL_FOR COLLAPSE(2)
//        for( Int j=0; j<n; ++j )
//            for( Int iLoc=0; iLoc<localHeight; ++iLoc )
//                localBuffer[iLoc+j*ldim] = buffer[iLoc+j*localHeight];
//    }
//};
//
//template<typename T>
//struct MakeUniformHelper<T,STAR,MC>
//{
//    static void Func( DistTensor<T,STAR,MC>& A, T center, BASE(T) radius )
//    {
//        const Grid& grid = A.Grid();
//        const Int m = A.Height();
//        const Int localWidth = A.LocalWidth();
//        const Int bufSize = m*localWidth;
//        std::vector<T> buffer( bufSize );
//
//        // Create a random matrix on process column 0, then broadcast
//        if( grid.Col() == 0 )
//        {
//            for( Int jLoc=0; jLoc<localWidth; ++jLoc )
//                for( Int i=0; i<m; ++i )
//                    buffer[i+jLoc*m] = SampleBall( center, radius );
//        }
//        mpi::Broadcast( buffer.data(), bufSize, 0, grid.RowComm() );
//
//        // Unpack
//        T* localBuffer = A.Buffer();
//        const Int ldim = A.LDim();
//        PARALLEL_FOR
//        for( Int jLoc=0; jLoc<localWidth; ++jLoc )
//        {
//            const T* bufferCol = &buffer[jLoc*m];
//            T* col = &localBuffer[jLoc*ldim];
//            MemCopy( col, bufferCol, m );
//        }
//    }
//};
//
//template<typename T>
//struct MakeUniformHelper<T,STAR,MD>
//{
//    static void Func( DistTensor<T,STAR,MD>& A, T center, BASE(T) radius )
//    {
//        const Int m = A.Height();
//        const Int localWidth = A.LocalWidth();
//        for( Int jLoc=0; jLoc<localWidth; ++jLoc )
//            for( Int i=0; i<m; ++i )
//                A.SetLocal( i, jLoc, SampleBall( center, radius ) );
//    }
//};
//
//template<typename T>
//struct MakeUniformHelper<T,STAR,MR>
//{
//    static void Func( DistTensor<T,STAR,MR>& A, T center, BASE(T) radius )
//    {
//        const Grid& grid = A.Grid();
//        const Int m = A.Height();
//        const Int localWidth = A.LocalWidth();
//        const Int bufSize = m*localWidth;
//        std::vector<T> buffer( bufSize );
//
//        // Create random matrix on process row 0, then broadcast
//        if( grid.Row() == 0 )
//        {
//            for( Int j=0; j<localWidth; ++j )
//                for( Int i=0; i<m; ++i )
//                    buffer[i+j*m] = SampleBall( center, radius );
//        }
//        mpi::Broadcast( buffer.data(), bufSize, 0, grid.ColComm() );
//
//        // Unpack
//        T* localBuffer = A.Buffer();
//        const Int ldim = A.LDim();
//        PARALLEL_FOR
//        for( Int jLoc=0; jLoc<localWidth; ++jLoc )
//        {
//            const T* bufferCol = &buffer[jLoc*m];
//            T* col = &localBuffer[jLoc*ldim];
//            MemCopy( col, bufferCol, m );
//        }
//    }
//};
//
//template<typename T>
//struct MakeUniformHelper<T,STAR,STAR>
//{
//    static void Func( DistTensor<T,STAR,STAR>& A, T center, BASE(T) radius )
//    {
//        const Grid& grid = A.Grid();
//        const Int m = A.Height();
//        const Int n = A.Width();
//        const Int bufSize = m*n;
//
//        if( grid.InGrid() )
//        {
//            std::vector<T> buffer( bufSize );
//
//            if( grid.Rank() == 0 )
//            {
//                for( Int j=0; j<n; ++j )
//                    for( Int i=0; i<m; ++i )
//                        buffer[i+j*m] = SampleBall( center, radius );
//            }
//            mpi::Broadcast( buffer.data(), bufSize, 0, grid.Comm() );
//
//            // Unpack
//            T* localBuffer = A.Buffer();
//            const Int ldim = A.LDim();
//            PARALLEL_FOR
//            for( Int j=0; j<n; ++j )
//            {
//                const T* bufferCol = &buffer[j*m];
//                T* col = &localBuffer[j*ldim];
//                MemCopy( col, bufferCol, m );
//            }
//        }
//    }
//};
//
//template<typename T>
//struct MakeUniformHelper<T,STAR,VC>
//{
//    static void Func( DistTensor<T,STAR,VC>& A, T center, BASE(T) radius )
//    {
//        const Int m = A.Height();
//        const Int localWidth = A.LocalWidth();
//        for( Int jLoc=0; jLoc<localWidth; ++jLoc )
//            for( Int i=0; i<m; ++i )
//                A.SetLocal( i, jLoc, SampleBall( center, radius ) );
//    }
//};
//
//template<typename T>
//struct MakeUniformHelper<T,STAR,VR>
//{
//    static void Func( DistTensor<T,STAR,VR>& A, T center, BASE(T) radius )
//    {
//        const Int m = A.Height();
//        const Int localWidth = A.LocalWidth();
//        for( Int jLoc=0; jLoc<localWidth; ++jLoc )
//            for( Int i=0; i<m; ++i )
//                A.SetLocal( i, jLoc, SampleBall( center, radius ) );
//    }
//};
//
//template<typename T>
//struct MakeUniformHelper<T,VC,STAR>
//{
//    static void Func( DistTensor<T,VC,STAR>& A, T center, BASE(T) radius )
//    {
//        const Int n = A.Width();
//        const Int localHeight = A.LocalHeight();
//        for( Int j=0; j<n; ++j )
//            for( Int iLoc=0; iLoc<localHeight; ++iLoc )
//                A.SetLocal( iLoc, j, SampleBall( center, radius ) );
//    }
//};
//
//template<typename T>
//struct MakeUniformHelper<T,VR,STAR>
//{
//    static void Func( DistTensor<T,VR,STAR>& A, T center, BASE(T) radius )
//    {
//        const Int n = A.Width();
//        const Int localHeight = A.LocalHeight();
//        for( Int j=0; j<n; ++j )
//            for( Int iLoc=0; iLoc<localHeight; ++iLoc )
//                A.SetLocal( iLoc, j, SampleBall( center, radius ) );
//    }
//};

} // namespace internal

template<typename T>
inline void
MakeUniform( DistTensor<T>& A, T center=0, BASE(T) radius=1 )
{
#ifndef RELEASE
    CallStackEntry cse("Uniform");
#endif
    internal::MakeUniformHelper<T>::Func( A, center, radius );
}

template<typename T>
inline void
Uniform( DistTensor<T>& A, const ObjShape& shape, T center=0, BASE(T) radius=1 )
{
#ifndef RELEASE
    CallStackEntry cse("Uniform");
#endif
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

} // namespace tmen

#endif // ifndef TMEN_TENSORS_UNIFORM_HPP
