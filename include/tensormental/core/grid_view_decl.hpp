/*
   Copyright (c) 2009-2013, Jack Poulson
                      2013, Jed Brown
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef TMEN_CORE_GRID_VIEW_DECL_HPP
#define TMEN_CORE_GRID_VIEW_DECL_HPP

#include <iostream>
#include <vector>
#include "tensormental/core/imports/mpi.hpp"
#include "tensormental/core/environment_decl.hpp"
#include "tensormental/core/grid_decl.hpp"
#include "tensormental/core/types_decl.hpp"

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

    ModeArray BoundModes() const;
    ModeArray FreeModes() const;
    ModeArray UnusedModes() const;

    void AddFreeMode(const Mode& mode);
    bool IsBound(Mode mode) const;
    bool IsFree(Mode mode) const;
    bool IsUnused(Mode mode) const;

    Unsigned LinearRank() const;
    void SetMyGridViewLoc();

    void RemoveUnitModes(const ModeArray& unitModes);
    void IntroduceUnitModes(const ModeArray& unitModes);

    bool Participating() const;

private:
    TensorDistribution dist_;
    ModeArray boundModes_;
    ModeArray freeModes_;
    ModeArray unusedModes_;
    ObjShape shape_;
    Location loc_;

    const rote::Grid* grid_;

    void SetupGridView(const ModeArray& unusedModes = ModeArray());
    void SetGridModeTypes(const ModeArray& unusedModes = ModeArray());
};

bool operator== ( const GridView& A, const GridView& B );
bool operator!= ( const GridView& A, const GridView& B );

} // namespace tmen

#endif // ifndef TMEN_CORE_GRID_VIEW_DECL_HPP
