/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#include "rote.hpp"

namespace rote {

  void
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

  Grid::Grid( mpi::Comm comm, const ObjShape& shape )
  {
      inGrid_ = true; // this is true by assumption for this constructor

      mpi::CommDup(comm, owningComm_);

      shape_ = shape;
      gridLoc_.resize(shape_.size());
      size_ = mpi::CommSize( comm );
      linearRank_ = mpi::CommRank( comm );

      SetMyGridLoc();
      SetUpGrid();
  }

  void
  Grid::SetUpGrid()
  {
      const Unsigned order = Order();
      Unsigned i;
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
          std::vector<int> shape(order);
          std::vector<int> periods(order);

          for(i = 0; i < order; i++){
            shape[i] = shape_[i];
            periods[i] = true;
          }
          bool reorder = false;
          mpi::CartCreate
          ( owningComm_, order, shape.data(), periods.data(), reorder, cartComm_ );
      }
  }

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

  Location
  Grid::Loc() const
  {
  	Location loc = gridLoc_;
  	return loc;
  }

  Unsigned
  Grid::ModeLoc(Mode mode) const
  {
  	return gridLoc_[mode];
  }

  Unsigned
  Grid::LinearRank() const
  {
  	return linearRank_;
  }

  Unsigned
  Grid::Order() const
  { return shape_.size(); }

  Unsigned
  Grid::Size() const
  { return size_; }

  ObjShape
  Grid::Shape() const
  {
  	return shape_;
  }

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

  bool
  Grid::InGrid() const
  { return inGrid_; }

  mpi::Comm
  Grid::OwningComm() const
  { return owningComm_; }

  //
  // Comparison functions
  //

  bool
  operator==( const Grid& A, const Grid& B )
  { return &A == &B; }

  bool
  operator!=( const Grid& A, const Grid& B )
  { return &A != &B; }

  //
  // Util
  //
  Location Grid::ToGridViewLoc(const Location& loc, const GridView& gv) const {
    Unsigned i;

    ObjShape gShape = Shape();
    TensorDistribution tDist = gv.Distribution();
    const Unsigned order = tDist.size();
    Location ret(order);

    for(i = 0; i < order; i++){
        ModeDistribution mDist = tDist[i];
        Location gSliceLoc(mDist.size());
        ObjShape gSliceShape(mDist.size());

        gSliceLoc = FilterVector(loc, mDist.Entries());
        gSliceShape = FilterVector(gShape, mDist.Entries());

        ret[i] = Loc2LinearLoc(gSliceLoc, gSliceShape);
    }
    return ret;
  }

  Location Grid::ToParticipatingGridViewLoc(const Location& loc, const GridView& gv) const {
    Unsigned i;
    ObjShape gShape = Shape();
    TensorDistribution tDist = gv.Distribution();
    const Unsigned order = tDist.size() - 1;
    Location ret(order);

    for(i = 0; i < order; i++){
        ModeDistribution mDist = tDist[i];
        Location gSliceLoc(mDist.size());
        ObjShape gSliceShape(mDist.size());

        gSliceLoc = FilterVector(loc, mDist.Entries());
        gSliceShape = FilterVector(gShape, mDist.Entries());

        ret[i] = Loc2LinearLoc(gSliceLoc, gSliceShape);
    }
    return ret;
  }
}
