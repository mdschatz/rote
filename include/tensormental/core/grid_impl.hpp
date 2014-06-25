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
  Unsigned i;
  const Unsigned order = Order();
  std::vector<Unsigned> stride(order);
  stride[0] = 1;
  for(i = 1; i < order; i++)
    stride[i] = stride[i-1]*shape_[i-1];

  Unsigned gridSliceSize = size_;
  for(i = order - 1; i < order; i--){
    if(stride[i] > linearRank_)
      gridLoc_[i] = 0;
    else
      break;
    gridSliceSize /= stride[i];
  }

  Unsigned remainingRank = linearRank_;
  for(; i < order ; i--){
    gridLoc_[i] = remainingRank / stride[i];
    remainingRank -= gridLoc_[i] * stride[i];
  }
}

inline
Grid::Grid( mpi::Comm comm, const ObjShape& shape )
{
#ifndef RELEASE
    CallStackEntry entry("Grid::Grid");
#endif
    inGrid_ = true; // this is true by assumption for this constructor

    mpi::CommDup(comm, owningComm_);

    shape_ = shape;
    gridLoc_.resize(shape_.size());
    size_ = mpi::CommSize( comm );
    linearRank_ = mpi::CommRank( comm );

    SetMyGridLoc();
    SetUpGrid();
}

inline void 
Grid::SetUpGrid()
{
#ifndef RELEASE
    CallStackEntry entry("Grid::SetUpGrid");
#endif
    const Unsigned order = Order();
    int i;
    if( size_ != prod(shape_))
    {
        std::ostringstream msg;
        msg << "Number of processes must match grid size:\n"
            << "  size=" << size_ << ", dimension=[" << shape_[0];
        for(i = 1; i < order; i++)
          msg << ", " << shape_[i];
        msg << "]";
        LogicError( msg.str() );
    }

    if( inGrid_ )
    {
        // Create a cartesian communicator
        int shape[order];
        int periods[order];

        for(i = 0; i < order; i++){
          shape[i] = shape_[i];
          periods[i] = true;
        }
        bool reorder = false;
        mpi::CartCreate
        ( owningComm_, order, shape, periods, reorder, cartComm_ );
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

inline Location
Grid::Loc() const
{
	Location loc = gridLoc_;
	return loc;
}

inline Unsigned
Grid::ModeLoc(Mode mode) const
{
	return gridLoc_[mode];
}

inline Unsigned
Grid::LinearRank() const
{
	return linearRank_;
}



inline Unsigned
Grid::Order() const
{ return shape_.size(); }

inline Unsigned
Grid::Size() const
{ return size_; }

inline ObjShape
Grid::Shape() const
{
	return shape_;
}

inline
Unsigned
Grid::Dimension(Mode mode) const
{ 
  Unsigned order = Order();
  if (mode >= order){
    std::ostringstream msg;
    msg << "Dimension must be of valid mode:\n"
        << "  order=" << order << ", requested mode=" << mode;
    LogicError( msg.str() );
  }
  return shape_[mode];
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

