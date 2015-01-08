/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef TMEN_IO_DECL_HPP
#define TMEN_IO_DECL_HPP

#define MAX_ELEM_PER_PROC 16

#include "tensormental/core/tensor_forward_decl.hpp"
#include "tensormental/core/environment_decl.hpp"
#include "tensormental/core/types_decl.hpp"

namespace tmen{

std::ifstream::pos_type FileSize( std::ifstream& file );
FileFormat DetectFormat( const std::string filename );

template<typename T>
void Read
( DistTensor<T>& A, const std::string filename, FileFormat format,
  bool sequential );


}
#endif // ifndef TMEN_IO_DECL_HPP
