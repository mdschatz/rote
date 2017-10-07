/*
   Copyright (c) 2009-2013, Jack Poulson
                      2013, Jed Brown
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/

// TODO: Consistency among Unsigned/Signed integers (order)
#pragma once
#ifndef ROTE_CORE_GRID_VIEW_IMPL_HPP
#define ROTE_CORE_GRID_VIEW_IMPL_HPP

#include "rote/core/grid_decl.hpp"
#include "rote/util/vec_util.hpp"

namespace rote {

inline void GridView::SetMyGridViewLoc() {

  //	int i;
  //	const int order = this->Order();
  const ObjShape gridShape = grid_->Shape();
  const Location gridLoc = grid_->Loc();

  loc_ = GridLoc2GridViewLoc(gridLoc, gridShape, Distribution());
}

inline void GridView::SetGridModeTypes(const ModeDistribution &unusedModes) {
  const Unsigned gridOrder = grid_->Order();
  const Unsigned order = dist_.size() - 1;
  Unsigned i;

  for (i = 0; i < order; i++) {
    ModeDistribution modeDist = dist_[i];
    boundModes_ += modeDist;
  }
  unusedModes_ = unusedModes;

  ModeDistribution allModes(OrderedModes(gridOrder));
  freeModes_ = allModes - unusedModes_ - boundModes_;
}

inline GridView::GridView(const rote::Grid *grid,
                          const TensorDistribution &distribution)
    : dist_(distribution), shape_(distribution.size()),
      loc_(distribution.size()),
      // freeModes_, boundModes_, unusedModes_ are set in SetGridModeTypes
      grid_(grid) {
#ifndef RELEASE
  CallStackEntry entry("GridView::GridView");
#endif

  SetupGridView(distribution.UnusedModes());
}

inline void GridView::SetupGridView(const ModeDistribution &unusedModes) {
#ifndef RELEASE
  CallStackEntry entry("GridView::SetupGridView");
#endif
  Unsigned i;
  Unsigned j;
  const Unsigned order = ParticipatingOrder();

  for (i = 0; i < order; i++) {
    ModeDistribution modeDist = dist_[i];
    Unsigned gridViewDim = 1;
    for (j = 0; j < modeDist.size(); j++) {
      gridViewDim *= grid_->Dimension(modeDist[j]);
    }
    shape_[i] = gridViewDim;
  }
  SetMyGridViewLoc();
  SetGridModeTypes(unusedModes);
}

inline GridView::~GridView() {}

inline Location GridView::ParticipatingLoc() const {
  Location ret(loc_.begin(), loc_.end() - 1);
  return ret;
}

inline Location GridView::Loc() const { return loc_; }

inline Unsigned GridView::ModeLoc(Mode mode) const { return loc_[mode]; }

inline Unsigned GridView::LinearRank() const {
  Unsigned i;
  Unsigned linearRank = 0;
  const Unsigned order = ParticipatingOrder();
  linearRank += ModeLoc(0);
  for (i = 1; i < order; i++) {
    linearRank += ModeLoc(i) * Dimension(i - 1);
  }
  return linearRank;
}

inline Unsigned GridView::ParticipatingOrder() const {
  return shape_.size() - 1;
}

inline ObjShape GridView::ParticipatingShape() const {
  ObjShape ret(shape_.begin(), shape_.end() - 1);
  return ret;
}

inline std::vector<Unsigned> GridView::ParticipatingModeWrapStrides() const {
  return ParticipatingShape();
}

inline std::vector<Unsigned> GridView::ModeWrapStrides() const {
  return shape_;
}

inline Unsigned GridView::ModeWrapStride(Mode mode) const {
  const Unsigned order = ParticipatingOrder();
  if (mode >= order) {
    std::ostringstream msg;
    msg << "Requested stride must be of valid mode:\n"
        << "  order=" << order << ", requested mode=" << mode;
    LogicError(msg.str());
  }
  return shape_[mode];
}

inline TensorDistribution GridView::Distribution() const { return dist_; }

inline const rote::Grid *GridView::Grid() const { return grid_; }

inline Unsigned GridView::Dimension(Mode mode) const {
  const Unsigned order = ParticipatingOrder();
  if (mode >= order) {
    std::ostringstream msg;
    msg << "Dimension must be of valid mode:\n"
        << "  order=" << order << ", requested mode=" << mode;
    LogicError(msg.str());
  }
  return shape_[mode];
}

inline ModeArray GridView::UnusedModes() const {
  return unusedModes_.Entries();
}

inline ModeArray GridView::UsedModes() const {
  ModeDistribution ret = freeModes_ + boundModes_;
  return ret.Entries();
}

inline ModeArray GridView::FreeModes() const { return freeModes_.Entries(); }

inline bool GridView::Participating() const {
#ifndef RELEASE
  CallStackEntry cse("GridView::Participating");
#endif
  return loc_[ParticipatingOrder()] == 0;
}

//
// Comparison functions
//

inline bool operator==(const GridView &A, const GridView &B) {
  return &A == &B;
}

inline bool operator!=(const GridView &A, const GridView &B) {
  return &A != &B;
}

} // namespace rote
#endif
