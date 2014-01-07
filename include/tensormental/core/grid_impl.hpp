/*
   Copyright (c) 2009-2013, Jack Poulson
                      2013, Jed Brown 
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef TMEN_CORE_GRID_IMPL_HPP
#define TMEN_CORE_GRID_IMPL_HPP

#include "tensormental/core/grid_decl.hpp"
#include "tensormental/util/vec_util.hpp"

namespace tmen {

inline void
Grid::SetMyGridLoc( )
{
  int i;
  std::vector<int> stride(order_);
  stride[0] = 1;
  for(i = 1; i < order_; i++)
    stride[i] = stride[i-1]*dimension_[i-1];

  int gridSliceSize = size_;
  for(i = order_ - 1; i >= 0; i--){
    if(stride[i] > linearRank_)
      gridLoc_[i] = 0;
    else
      break;
    gridSliceSize /= stride[i];
  }

  int remainingRank = linearRank_;
  for(; i >=0 ; i--){
    if(gridLoc_[i] < 0)
        LogicError("Process grid dimensions must be non-negative");
    gridLoc_[i] = remainingRank / stride[i];
    remainingRank -= gridLoc_[i] * stride[i];
  }
}

inline
Grid::Grid( mpi::Comm comm, int order, std::vector<int> dimension )
{
#ifndef RELEASE
    CallStackEntry entry("Grid::Grid");
#endif
    inGrid_ = true; // this is true by assumption for this constructor

    mpi::CommDup(comm, owningComm_);

    order_ = order;
    dimension_ = dimension;
    gridLoc_.resize(order_);
    size_ = mpi::CommSize( comm );
    linearRank_ = mpi::CommRank( comm );

    SetMyGridLoc(); 
    std::cout << "Grid Loc set\n";
    SetUpGrid();
}

inline void 
Grid::SetUpGrid()
{
#ifndef RELEASE
    CallStackEntry entry("Grid::SetUpGrid");
#endif
    int i;
    if( size_ != prod(dimension_))
    {
        std::ostringstream msg;
        msg << "Number of processes must match grid size:\n"
            << "  size=" << size_ << ", dimension=[" << dimension_[0];
        for(i = 1; i < order_; i++)
          msg << ", " << dimension_[i];
        msg << "]";
        LogicError( msg.str() );
    }

    if( inGrid_ )
    {
        // Create a cartesian communicator
        int dimensions[order_];
        int periods[order_];

        for(i = 0; i < order_; i++){
          dimensions[i] = dimension_[i];
          periods[i] = true;
        }
        bool reorder = false;
        mpi::CartCreate
        ( owningComm_, order_, dimensions, periods, reorder, cartComm_ );
    }
}

inline
Grid::~Grid()
{
    if( !mpi::Finalized() )
    {
        if( inGrid_ )
        {
            mpi::CommFree( cartComm_ );
        }

        mpi::CommFree( owningComm_ );
    }
}

inline int 
Grid::Size() const
{ return size_; }

inline
int
Grid::Dimension(int mode) const
{ 
  if (mode > order_ || mode < 0){
    std::ostringstream msg;
    msg << "Dimension must be of valid mode:\n"
        << "  order=" << order_ << ", requested mode=" << mode;
    LogicError( msg.str() );
  }
  return dimension_[mode];
}
//
// Advanced routines
//

inline bool 
Grid::InGrid() const
{ return inGrid_; }

inline mpi::Comm 
Grid::OwningComm() const
{ return owningComm_; }

//
// Comparison functions
//

inline bool 
operator==( const Grid& A, const Grid& B )
{ return &A == &B; }

inline bool 
operator!=( const Grid& A, const Grid& B )
{ return &A != &B; }

} // namespace tmen
#endif
