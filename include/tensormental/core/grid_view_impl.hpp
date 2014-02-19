/*
   Copyright (c) 2009-2013, Jack Poulson
                      2013, Jed Brown
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef TMEN_CORE_GRID_VIEW_IMPL_HPP
#define TMEN_CORE_GRID_VIEW_IMPL_HPP

#include "tensormental/core/grid_decl.hpp"
#include "tensormental/util/vec_util.hpp"

namespace tmen {

inline void
GridView::SetMyGridViewLoc( )
{
	int i;
	const std::vector<Int> gridShape = grid_->Shape();
	const std::vector<Int> gridLoc = grid_->Loc();

	for(i = 0; i < order_; i++){
		ModeDistribution modeDist = dist_[i];
		std::vector<Int> gridSliceLoc(modeDist.size());
		std::vector<Int> gridSliceShape(modeDist.size());

		gridSliceLoc = FilterVector(gridLoc, modeDist);
		gridSliceShape = FilterVector(gridShape, modeDist);

		loc_[i] = LinearIndex(gridSliceLoc, Dimensions2Strides(gridSliceShape));
	}
}

inline
GridView::GridView( const tmen::Grid* grid, const TensorDistribution& distribution )
: order_(distribution.size()),
  loc_(order_),
  shape_(order_)
{
#ifndef RELEASE
    CallStackEntry entry("GridView::GridView");
#endif
    order_ = distribution.size();
    grid_ = grid;
    dist_ = distribution;

    SetupGridView();

}

inline void
GridView::SetupGridView()
{
#ifndef RELEASE
    CallStackEntry entry("GridView::SetUpGridView");
#endif
    int i, j;

    for(i = 0; i < dist_.size(); i++){
    	ModeDistribution modeDist = dist_[i];
    	int gridViewDim = 1;
    	for(j = 0; j < modeDist.size(); j++){
    		gridViewDim *= grid_->Dimension(modeDist[j]);
    	}
    	shape_[i] = gridViewDim;
    }
    SetMyGridViewLoc();

}

inline
GridView::~GridView()
{
}

inline std::vector<Int>
GridView::Loc() const
{
	return loc_;
}

inline Int
GridView::ModeLoc(int mode) const
{
	return loc_[mode];
}

inline Int
GridView::LinearRank() const
{
	int i;
	int linearRank = 0;
	for(i = 0; i < order_; i++){
		linearRank += this->ModeLoc(i) * this->Dimension(i);
	}
	return linearRank;
}

inline int
GridView::Order() const
{ return order_; }

inline std::vector<Int>
GridView::Shape() const
{
	return shape_;
}

inline
int
GridView::Dimension(int mode) const
{
  if (mode > order_ || mode < 0){
    std::ostringstream msg;
    msg << "Dimension must be of valid mode:\n"
        << "  order=" << order_ << ", requested mode=" << mode;
    LogicError( msg.str() );
  }
  return shape_[mode];
}

//
// Comparison functions
//

inline bool
operator==( const GridView& A, const GridView& B )
{ return &A == &B; }

inline bool
operator!=( const GridView& A, const GridView& B )
{ return &A != &B; }

} // namespace tmen
#endif

