/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#include "rote.hpp"

namespace rote {

  void GridView::SetMyGridViewLoc() {
    const ObjShape gridShape = grid_->Shape();
    const Location gridLoc = grid_->Loc();

    loc_ = GridLoc2GridViewLoc(gridLoc, gridShape, Distribution());
  }

  void GridView::SetGridModeTypes(const ModeDistribution &unusedModes) {
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

  GridView::GridView(
  	const rote::Grid *grid,
    const TensorDistribution &distribution) :
  		dist_(distribution),
  		shape_(distribution.size()),
      loc_(distribution.size()),
      grid_(grid)
  { SetupGridView(distribution.UnusedModes()); }

  void GridView::SetupGridView(const ModeDistribution &unusedModes) {
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

  GridView::~GridView() {}

  Location GridView::ParticipatingLoc() const {
    Location ret(loc_.begin(), loc_.end() - 1);
    return ret;
  }

  Location GridView::Loc() const { return loc_; }

  Unsigned GridView::ModeLoc(Mode mode) const { return loc_[mode]; }

  Unsigned GridView::LinearRank() const {
    Unsigned i;
    Unsigned linearRank = 0;
    const Unsigned order = ParticipatingOrder();
    linearRank += ModeLoc(0);
    for (i = 1; i < order; i++) {
      linearRank += ModeLoc(i) * Dimension(i - 1);
    }
    return linearRank;
  }

  Unsigned GridView::ParticipatingOrder() const {
    return shape_.size() - 1;
  }

  ObjShape GridView::ParticipatingShape() const {
    ObjShape ret(shape_.begin(), shape_.end() - 1);
    return ret;
  }

  std::vector<Unsigned> GridView::ParticipatingModeWrapStrides() const {
    return ParticipatingShape();
  }

  std::vector<Unsigned> GridView::ModeWrapStrides() const {
    return shape_;
  }

  Unsigned GridView::ModeWrapStride(Mode mode) const {
    const Unsigned order = ParticipatingOrder();
    if (mode >= order) {
      std::ostringstream msg;
      msg << "Requested stride must be of valid mode:\n"
          << "  order=" << order << ", requested mode=" << mode;
      LogicError(msg.str());
    }
    return shape_[mode];
  }

  TensorDistribution GridView::Distribution() const { return dist_; }

  const rote::Grid *GridView::Grid() const { return grid_; }

  Unsigned GridView::Dimension(Mode mode) const {
    const Unsigned order = ParticipatingOrder();
    if (mode >= order) {
      std::ostringstream msg;
      msg << "Dimension must be of valid mode:\n"
          << "  order=" << order << ", requested mode=" << mode;
      LogicError(msg.str());
    }
    return shape_[mode];
  }

  ModeArray GridView::UnusedModes() const {
    return unusedModes_.Entries();
  }

  ModeArray GridView::UsedModes() const {
    ModeDistribution ret = freeModes_ + boundModes_;
    return ret.Entries();
  }

  ModeArray GridView::FreeModes() const { return freeModes_.Entries(); }

  bool GridView::Participating() const {
  #ifndef RELEASE
    CallStackEntry cse("GridView::Participating");
  #endif
    return loc_[ParticipatingOrder()] == 0;
  }

  //
  // Util
  //

  Unsigned GridView::ToParticipatingLinearLoc(const Location& loc) const {
    //Get the lin loc of the owner
    Unsigned i, j;
    int ownerLinearLoc = 0;
    const TensorDistribution dist = Distribution();
    const rote::Grid* g = Grid();
    const Unsigned participatingOrder = ParticipatingOrder();
    ModeArray participatingComms = UsedModes();
    SortVector(participatingComms);

    const Location gvParticipatingLoc = ParticipatingLoc();

    ObjShape gridSlice = FilterVector(g->Shape(), participatingComms);
    Location participatingGridLoc(gridSlice.size());

    for(i = 0; i < participatingOrder; i++){
        ModeDistribution modeDist = dist[i];
        ObjShape modeSliceShape = FilterVector(g->Shape(), modeDist.Entries());
        const Location modeSliceLoc = LinearLoc2Loc(loc[i], modeSliceShape);

        for(j = 0; j < modeDist.size(); j++){
            int indexOfMode = std::find(participatingComms.begin(), participatingComms.end(), modeDist[j]) - participatingComms.begin();
            participatingGridLoc[indexOfMode] = modeSliceLoc[j];
        }
    }
    ownerLinearLoc = Loc2LinearLoc(participatingGridLoc, gridSlice);
    return ownerLinearLoc;
  }

  Location GridView::ToGridLoc(const Location& gvLoc) const {
    #ifndef RELEASE
      if(gvLoc.size() != ParticipatingOrder())
          LogicError("Supplied loc must be same order as gridView");
    #endif

    const Unsigned gvOrder = ParticipatingOrder();
    const TensorDistribution tDist = Distribution();

    const rote::Grid* g = Grid();
    const Unsigned gOrder = g->Order();
    const ObjShape gShape = g->Shape();
    Unsigned i, j;

    Location gLoc(gOrder);
    for(i = 0; i < gvOrder; i++){

        const ModeDistribution mDist = tDist[i];
        const ObjShape gSliceShape = FilterVector(gShape, mDist.Entries());
        Location gSliceLoc = LinearLoc2Loc(gvLoc[i], gSliceShape);

        for(j = 0; j < gSliceLoc.size(); j++){
            gLoc[mDist[j]] = gSliceLoc[j];
        }
    }

    return gLoc;
  }
  //
  // Comparison functions
  //

  bool operator==(const GridView &A, const GridView &B) {
    return &A == &B;
  }

  bool operator!=(const GridView &A, const GridView &B) {
    return &A != &B;
  }

}
