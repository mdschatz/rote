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

	loc_ = GridLoc2GridViewLoc(gridLoc, gridShape, Distribution());
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
void
GridView::SetGridModeTypes(const ModeArray& unusedModes)
{
    const Unsigned gridOrder = grid_->Order();
    const Unsigned order = dist_.size()-1;
    Unsigned i;
    Unsigned j;
    for(i = 0; i < order; i++){
        ModeDistribution modeDist = dist_[i];
        for(j = 0; j < modeDist.size(); j++){
            Mode mode = modeDist[j];
            boundModes_.push_back(mode);
        }
    }
    unusedModes_ = unusedModes;

    for(i = 0; i < gridOrder; i++){
        if(std::find(boundModes_.begin(), boundModes_.end(), i) == boundModes_.end()){
            if(std::find(unusedModes_.begin(), unusedModes_.end(), i) == unusedModes_.end()){
                freeModes_.push_back(i);
            }
        }
    }
}

inline
GridView::GridView( const tmen::Grid* grid, const TensorDistribution& distribution )
: dist_(distribution),
  shape_(distribution.size()),
  loc_(distribution.size()),
  //freeModes_, boundModes_, unusedModes_ are set in SetGridModeTypes
  grid_(grid)
{
#ifndef RELEASE
    CallStackEntry entry("GridView::GridView");
#endif

    SetupGridView(distribution[distribution.size() - 1]);
}

inline
void
GridView::SetupGridView(const ModeArray& unusedModes)
{
#ifndef RELEASE
    CallStackEntry entry("GridView::SetupGridView");
#endif
    Unsigned i;
    Unsigned j;
    const Unsigned order = ParticipatingOrder();

    for(i = 0; i < order; i++){
    	ModeDistribution modeDist = dist_[i];
    	Unsigned gridViewDim = 1;
    	for(j = 0; j < modeDist.size(); j++){
    		gridViewDim *= grid_->Dimension(modeDist[j]);
    	}
    	shape_[i] = gridViewDim;
    }
    SetMyGridViewLoc();
    SetGridModeTypes(unusedModes);
}

inline
void
GridView::AddFreeMode(const Mode& freeMode)
{
#ifndef RELEASE
    CallStackEntry cse("GridView::AddFreeMode");
#endif
    freeModes_.push_back(freeMode);
    boundModes_.erase(std::find(boundModes_.begin(), boundModes_.end(), freeMode));
    unusedModes_.erase(std::find(boundModes_.begin(), boundModes_.end(), freeMode));
}

inline
GridView::~GridView()
{
}

inline
Location
GridView::ParticipatingLoc() const
{
    Location ret(loc_.begin(), loc_.end()-1);
	return ret;
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
	const Unsigned order = ParticipatingOrder();
	linearRank += ModeLoc(0);
	for(i = 1; i < order; i++){
		linearRank += ModeLoc(i) * Dimension(i-1);
	}
	return linearRank;
}

inline
Unsigned
GridView::ParticipatingOrder() const
{ return shape_.size() - 1; }

inline
ObjShape
GridView::ParticipatingShape() const
{
    ObjShape ret(shape_.begin(), shape_.end()-1);
	return ret;
}

inline
std::vector<Unsigned>
GridView::ParticipatingModeWrapStrides() const
{
    return ParticipatingShape();
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
    const Unsigned order = ParticipatingOrder();
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
  const Unsigned order = ParticipatingOrder();
  if (mode >= order){
    std::ostringstream msg;
    msg << "Dimension must be of valid mode:\n"
        << "  order=" << order << ", requested mode=" << mode;
    LogicError( msg.str() );
  }
  return shape_[mode];
}

//
//
//

inline
ModeArray
GridView::BoundModes() const
{ return boundModes_; }

inline
ModeArray
GridView::FreeModes() const
{ return freeModes_; }

inline
ModeArray
GridView::UnusedModes() const
{ return unusedModes_; }

inline
bool
GridView::IsBound(Mode mode) const
{ return std::find(boundModes_.begin(), boundModes_.end(), mode) != boundModes_.end(); }

inline
bool
GridView::IsFree(Mode mode) const
{ return std::find(freeModes_.begin(), freeModes_.end(), mode) != freeModes_.end(); }

inline
bool
GridView::IsUnused(Mode mode) const
{ return std::find(unusedModes_.begin(), unusedModes_.end(), mode) != unusedModes_.end(); }

inline
void
GridView::RemoveUnitModes(const ModeArray& unitModes)
{
    Unsigned i;
    ModeArray sorted = unitModes;
    std::sort(sorted.begin(), sorted.end());
    for(i = sorted.size() - 1; i < sorted.size(); i--){
        shape_.erase(shape_.begin() + sorted[i]);
        loc_.erase(loc_.begin() + sorted[i]);
        dist_.erase(dist_.begin() + sorted[i]);
    }
}

inline
void
GridView::IntroduceUnitModes(const ModeArray& unitModes)
{
    Unsigned i, j;
    ModeArray sorted = unitModes;
    std::sort(sorted.begin(), sorted.end());
    ModeArray blank(0);
    for(i = 0; i < sorted.size(); i++){
        shape_.insert(shape_.begin() + sorted[i], 1);
        loc_.insert(loc_.begin() + sorted[i], 0);
        dist_.insert(dist_.begin() + sorted[i], blank);

        for(j = 0; j < freeModes_.size(); j++)
            if(freeModes_[j] >= sorted[i])
                freeModes_[j] += 1;
        for(j = 0; j < boundModes_.size(); j++)
            if(boundModes_[j] >= sorted[i])
                boundModes_[j] += 1;
        boundModes_.insert(boundModes_.begin(), sorted[i]);
        for(j = 0; j < unusedModes_.size(); j++)
            if(unusedModes_[j] == sorted[i])
                unusedModes_[j] += 1;

    }
}


inline
bool
GridView::Participating() const
{
#ifndef RELEASE
    CallStackEntry cse("GridView::Participating");
#endif
    return loc_[ParticipatingOrder()] == 0;
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

