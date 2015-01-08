/*
   Copyright (c) 2009-2015, Jack Poulson
   All rights reserved.

   This file is part of tmenemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef TMEN_READ_ASCIIMATLAB_HPP
#define TMEN_READ_ASCIIMATLAB_HPP

#include "tensormental.hpp"

namespace tmen {
namespace read {

template<typename T>
inline void
AsciiMatlab( Tensor<T>& A, const std::string filename )
{
    std::ifstream file( filename.c_str() );
    if( !file.is_open() ){
        std::string msg = "Could not open " + filename;
        RuntimeError(msg);
    }
    LogicError("Not yet written");
}

template<typename T>
inline void
AsciiMatlab( DistTensor<T>& A, const std::string filename )
{
    std::ifstream file( filename.c_str() );
    if( !file.is_open() ){
        std::string msg = "Could not open " + filename;
        RuntimeError(msg);
    }
    LogicError("Not yet written");
}

} // namespace read
} // namespace tmen

#endif // ifndef TMEN_READ_ASCIIMATLAB_HPP
