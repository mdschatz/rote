/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef ROTE_IO_HPP
#define ROTE_IO_HPP

#define MAX_ELEM_PER_PROC 10000000

namespace rote{

// TODO: Move this..
std::ifstream::pos_type FileSize( std::ifstream& file );
FileFormat DetectFormat( const std::string filename );

template<typename T>
void Read
( DistTensor<T>& A, const std::string filename, FileFormat format,
  bool sequential );
}

#include "rote/io/Print.hpp"
#include "rote/io/Write.hpp"

#endif // ifndef ROTE_IO_HPP
