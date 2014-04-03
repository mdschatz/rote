/*
   Copyright (c) 2009-2013, Jack Poulson
                      2013, Jed Brown
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/

//TODO: Consistency among Unsigned/Signed integers (order)
#pragma once
#ifndef TMEN_CORE_GRID_VIEW_IMPL_HPP
#define TMEN_CORE_GRID_VIEW_IMPL_HPP

#include "tensormental/core/grid_decl.hpp"
#include "tensormental/util/vec_util.hpp"

namespace tmen {

inline void
GridView::SetMyGridViewLoc( )
{

//	int i;
//	const int order = this->Order();
	const std::vector<Int> gridShape = grid_->Shape();
	const std::vector<Int> gridLoc = grid_->Loc();

	loc_ = GridLoc2GridViewLoc(gridLoc, gridShape, this->Distribution());
//
//	for(i = 0; i < order; i++){
//		ModeDistribution modeDist = dist_[i];
//		std::vector<Int> gridSliceLoc(modeDist.size());
//		std::vector<Int> gridSliceShape(modeDist.size());
//
//		gridSliceLoc = FilterVector(gridLoc, modeDist);
//		gridSliceShape = FilterVector(gridShape, modeDist);
//
//		loc_[i] = LinearIndex(gridSliceLoc, Dimensions2Strides(gridSliceShape));
//	}
}

inline
GridView::GridView( const tmen::Grid* grid, const TensorDistribution& distribution )
: dist_(distribution),
  shape_(distribution.size()),
  loc_(distribution.size()),
  grid_(grid)
{
#ifndef RELEASE
    CallStackEntry entry("GridView::GridView");
#endif

    SetupGridView();
}

inline void
GridView::SetupGridView()
{
#ifndef RELEASE
    CallStackEntry entry("GridView::SetupGridView");
#endif
    int i;
    Unsigned j;
    const int order = this->Order();

    for(i = 0; i < order; i++){
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
	const int order = this->Order();
	linearRank += this->ModeLoc(0);
	for(i = 1; i < order; i++){
		linearRank += this->ModeLoc(i) * this->Dimension(i-1);
	}
	return linearRank;
}

inline int
GridView::Order() const
{ return shape_.size(); }

inline std::vector<Int>
GridView::Shape() const
{
	return shape_;
}

inline
std::vector<int>
GridView::ModeWrapStrides() const
{
    return shape_;
}

inline
int
GridView::ModeWrapStride(int mode) const
{
    const int order = this->Order();
    if (mode > order || mode < 0){
        std::ostringstream msg;
        msg << "Requested stride must be of valid mode:\n"
            << "  order=" << order << ", requested mode=" << mode;
        LogicError( msg.str() );
    }
    return shape_[mode];
}

inline
TensorDistribution
GridView::Distribution() const
{
	return dist_;
}

inline
const tmen::Grid* GridView::Grid() const
{
	return grid_;
}

inline
int
GridView::Dimension(int mode) const
{
  const int order = this->Order();
  if (mode > order || mode < 0){
    std::ostringstream msg;
    msg << "Dimension must be of valid mode:\n"
        << "  order=" << order << ", requested mode=" << mode;
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

