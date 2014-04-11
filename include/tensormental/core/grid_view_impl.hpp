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

inline
void
GridView::SetMyGridViewLoc( )
{

//	int i;
//	const int order = this->Order();
	const ObjShape gridShape = grid_->Shape();
	const Location gridLoc = grid_->Loc();

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

inline
void
GridView::SetupGridView()
{
#ifndef RELEASE
    CallStackEntry entry("GridView::SetupGridView");
#endif
    Unsigned i;
    Unsigned j;
    const Unsigned order = this->Order();

    for(i = 0; i < order; i++){
    	ModeDistribution modeDist = dist_[i];
    	Unsigned gridViewDim = 1;
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

inline
Location
GridView::Loc() const
{
	return loc_;
}

inline
Unsigned
GridView::ModeLoc(Mode mode) const
{
	return loc_[mode];
}

inline
Unsigned
GridView::LinearRank() const
{
	Unsigned i;
	Unsigned linearRank = 0;
	const Unsigned order = this->Order();
	linearRank += this->ModeLoc(0);
	for(i = 1; i < order; i++){
		linearRank += this->ModeLoc(i) * this->Dimension(i-1);
	}
	return linearRank;
}

inline
Unsigned
GridView::Order() const
{ return shape_.size(); }

inline
ObjShape
GridView::Shape() const
{
	return shape_;
}

inline
std::vector<Unsigned>
GridView::ModeWrapStrides() const
{
    return shape_;
}

inline
Unsigned
GridView::ModeWrapStride(Mode mode) const
{
    const Unsigned order = this->Order();
    if (mode >= order){
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
const tmen::Grid*
GridView::Grid() const
{
	return grid_;
}

inline
Unsigned
GridView::Dimension(Mode mode) const
{
  const Unsigned order = this->Order();
  if (mode >= order){
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

inline
bool
operator==( const GridView& A, const GridView& B )
{ return &A == &B; }

inline
bool
operator!=( const GridView& A, const GridView& B )
{ return &A != &B; }

} // namespace tmen
#endif

