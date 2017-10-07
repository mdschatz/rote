/*
   Copyright (c) 2009-2013, Jack Poulson
                      2013, Jed Brown
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef ROTE_CORE_GRID_VIEW_DECL_HPP
#define ROTE_CORE_GRID_VIEW_DECL_HPP

#include "rote.hpp"

namespace rote {

class GridView
{
public:
    explicit GridView( const rote::Grid* g, const TensorDistribution& dist );
    ~GridView();

    // Simple interface (simpler version of distributed-based interface)
    Unsigned ParticipatingOrder() const;
    ObjShape ParticipatingShape() const;
    Unsigned Dimension(Mode mode) const;
    Location ParticipatingLoc() const;
    Location Loc() const;
    Unsigned ModeLoc(Mode mode) const;
    Unsigned ModeWrapStride(Mode mode) const;
    std::vector<Unsigned> ParticipatingModeWrapStrides() const;
    std::vector<Unsigned> ModeWrapStrides() const;
    TensorDistribution Distribution() const;
    const rote::Grid* Grid() const;

    ModeArray UnusedModes() const;
    ModeArray UsedModes() const;
    ModeArray FreeModes() const;

    Unsigned LinearRank() const;
    void SetMyGridViewLoc();

    void IntroduceUnitModes(const ModeArray& unitModes);

    bool Participating() const;

private:
    TensorDistribution dist_;
    ModeDistribution boundModes_;
    ModeDistribution freeModes_;
    ModeDistribution unusedModes_;
    ObjShape shape_;
    Location loc_;

    const rote::Grid* grid_;

    void SetupGridView(const ModeDistribution& unusedModes = ModeDistribution());
    void SetGridModeTypes(const ModeDistribution& unusedModes = ModeDistribution());
};

bool operator== ( const GridView& A, const GridView& B );
bool operator!= ( const GridView& A, const GridView& B );

} // namespace rote

#endif // ifndef ROTE_CORE_GRID_VIEW_DECL_HPP
