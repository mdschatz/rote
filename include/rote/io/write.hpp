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

namespace rote {

  template<typename T>
  void Write( const Tensor<T>& A, std::string title="", std::string filename="Tensor" );

  template<typename T>
  void
  Write
  ( const DistTensor<T>& A, std::string title="",
    std::string filename="DistTensor" );
} // namespace rote

#endif // ifndef ROTE_IO_WRITE_HPP
