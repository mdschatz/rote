/*
   Copyright (c) 2009-2015, Jack Poulson
   All rights reserved.

   This file is part of tmenemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef TMEN_READ_ASCII_HPP
#define TMEN_READ_ASCII_HPP

#include "tensormental.hpp"

namespace tmen {
namespace read {

template<typename T>
inline void
Ascii( Tensor<T>& A, const std::string filename )
{
    LogicError("Not implemented");
//    std::ifstream file( filename.c_str() );
//    if( !file.is_open() ){
//        std::string msg = "Could not open " + filename;
//        RuntimeError(msg);
//    }
//
//    // Walk through the file once to both count the number of rows and
//    // columns and to ensure that the number of columns is consistent
//    Int height=0, width=0;
//    std::string line;
//    while( std::getline( file, line ) )
//    {
//        std::stringstream lineStream( line );
//        Int numCols=0;
//        T value;
//        while( lineStream >> value ) ++numCols;
//        if( numCols != 0 )
//        {
//            if( numCols != width && width != 0 )
//                LogicError("Inconsistent number of columns");
//            else
//                width = numCols;
//            ++height;
//        }
//    }
//    file.clear();
//    file.seekg(0,file.beg);
//
//    // Resize the matrix and then read it
//    A.Resize( height, width );
//    Int i=0;
//    while( std::getline( file, line ) )
//    {
//        std::stringstream lineStream( line );
//        Int j=0;
//        T value;
//        while( lineStream >> value )
//        {
//            A.Set( i, j, value );
//            ++j;
//        }
//        ++i;
//    }
}

} // namespace read
} // namespace tmen

#endif // ifndef TMEN_READ_ASCII_HPP
