/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#include "rote.hpp"

namespace rote {

  template<typename G>
  Memory<G>::Memory()
  : size_(0), buffer_(0)
  { Require( 1 ); }

  template<typename G>
  Memory<G>::Memory( std::size_t size )
  : size_(0), buffer_(0)
  { Require( size ); }

  template<typename G>
  void
  Memory<G>::Swap( Memory<G>& mem )
  {
      std::swap(size_,mem.size_);
      std::swap(buffer_,mem.buffer_);
  }

  template<typename G>
  Memory<G>::~Memory()
  { delete[] buffer_; }

  template<typename G>
  G*
  Memory<G>::Buffer() const
  { return buffer_; }

  template<typename G>
  std::size_t
  Memory<G>::Size() const
  { return size_; }

  template<typename G>
  G*
  Memory<G>::Require( std::size_t size )
  {
      if( size > size_ )
      {
          delete[] buffer_;
  #ifndef RELEASE
          try {
  #endif
          buffer_ = new G[size];
  #ifndef RELEASE
          }
          catch( std::bad_alloc& e )
          {
              std::ostringstream os;
              os << "Failed to allocate " << size*sizeof(G)
                 << " bytes on process " << mpi::WorldRank() << std::endl;
              std::cerr << os.str();
              throw e;
          }
  #endif
          size_ = size;
      }
      return buffer_;
  }

  template<typename G>
  void
  Memory<G>::Release()
  {
  #ifndef POOL_MEMORY
      Empty();
  #endif
  }

  template<typename G>
  void
  Memory<G>::Empty()
  {
      delete[] buffer_;
      size_ = 0;
      buffer_ = 0;
  }

  #define FULL(G) \
      template class Memory<G>;

  FULL(Int)
  #ifndef DISABLE_FLOAT
  FULL(float)
  #endif
  FULL(double)

  #ifndef DISABLE_COMPLEX
  #ifndef DISABLE_FLOAT
  FULL(std::complex<float>)
  #endif
  FULL(std::complex<double>)
  #endif
}
