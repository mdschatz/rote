/*
   Copyright (c) 2009-2015, Jack Poulson
   All rights reserved.

   This file is part of tmenemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef TMEN_READ_BINARYFLAT_HPP
#define TMEN_READ_BINARYFLAT_HPP

#include "tensormental.hpp"

namespace tmen {
namespace read {

template<typename T>
inline void
BinaryFlat( Tensor<T>& A, Int height, Int width, const std::string filename )
{
    std::ifstream file( filename.c_str(), std::ios::binary );
    if( !file.is_open() ){
        std::string msg = "Could not open " + filename;
        RuntimeError(msg);
    }

    const Int numBytes = FileSize( file );
    const Int numBytesExp = height*width*sizeof(T);
    if( numBytes != numBytesExp ){
        std::string msg = "Expected file to be ";
        msg += numBytesExp + " bytes but found " + numBytes;
        RuntimeError(msg);
    }

    A.Resize( height, width );
    if( A.Height() == A.LDim() )
        file.read( (char*)A.Buffer(), height*width*sizeof(T) );
    else
        for( Int j=0; j<width; ++j )
            file.read( (char*)A.Buffer(0,j), height*sizeof(T) );
}

template<typename T>
inline void
BinaryFlat
( DistTensor<T>& A, Int height, Int width, const std::string filename )
{
    std::ifstream file( filename.c_str(), std::ios::binary );
    if( !file.is_open() ){
        std::string msg = "Could not open " + filename;
        RuntimeError(msg);
    }

    const Int numBytes = FileSize( file );
    const Int numBytesExp = height*width*sizeof(T);
    if( numBytes != numBytesExp ){
        std::string msg = "Expected file to be ";
        msg += numBytesExp + " bytes but found " + numBytes;
        RuntimeError(msg.c_str());
    }

    A.Resize( height, width );
    if( A.ColStride() == 1 && A.RowStride() == 1 )
    {
        if( A.CrossRank() == A.Root() )
        {
            if( A.Height() == A.LDim() )
                file.read( (char*)A.Buffer(), height*width*sizeof(T) );
            else
                for( Int j=0; j<width; ++j )
                    file.read( (char*)A.Buffer(0,j), height*sizeof(T) );
        }
    }
    else if( A.ColStride() == 1 )
    {
        const Int localWidth = A.LocalWidth();
        for( Int jLoc=0; jLoc<localWidth; ++jLoc )
        {
            const Int j = A.GlobalCol(jLoc);
            const Int localIndex = j*height;
            const std::streamoff pos = localIndex*sizeof(T);
            file.seekg( pos );
            file.read( (char*)A.Buffer(0,jLoc), height*sizeof(T) );
        }
    }
    else
    {
        const Int localHeight = A.LocalHeight();
        const Int localWidth = A.LocalWidth();
        for( Int jLoc=0; jLoc<localWidth; ++jLoc )
        {
            const Int j = A.GlobalCol(jLoc);
            for( Int iLoc=0; iLoc<localHeight; ++iLoc )
            {
                const Int i = A.GlobalRow(iLoc);
                const Int localIndex = i+j*height;
                const std::streamoff pos = localIndex*sizeof(T);
                file.seekg( pos );
                file.read( (char*)A.Buffer(iLoc,jLoc), sizeof(T) );
            }
        }
    }
}

} // namespace read
} // namespace tmen

#endif // ifndef TMEN_READ_BINARYFLAT_HPP
