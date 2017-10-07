/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef ROTE_IO_WRITE_HPP
#define ROTE_IO_WRITE_HPP

#include "rote/core/tensor_forward_decl.hpp"

namespace rote {

template<typename T>
inline void
Write( const Tensor<T>& A, std::string title="", std::string filename="Tensor" )
{
#ifndef RELEASE
    CallStackEntry entry("Write");
#endif
    std::ofstream file( filename.c_str() );
    file.setf( std::ios::scientific );
    Print( A, title, file );
    file.close();
}

// If already in [* ,* ] or [o ,o ] distributions, no copy is needed
template<typename T>
inline void
Write
( const DistTensor<T>& A, std::string title="",
  std::string filename="DistTensor" )
{
#ifndef RELEASE
    CallStackEntry entry("Write"); 
#endif
    if( A.Grid().VCRank() == 0 )
        Write( A.LockedTensor(), title, filename );
}

} // namespace rote

#endif // ifndef ROTE_IO_WRITE_HPP
