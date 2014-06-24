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

namespace tmen {

class GridView
{
public:
    explicit GridView( const tmen::Grid* g, const TensorDistribution& dist, const ModeArray& unusedModes=ModeArray() );
    ~GridView();

    // Simple interface (simpler version of distributed-based interface)
    Unsigned Order() const;
    ObjShape Shape() const;
    Unsigned Dimension(Mode mode) const;
    Location Loc() const;
    Location GridLoc() const;
    Unsigned ModeLoc(Mode mode) const;
    Unsigned ModeWrapStride(Mode mode) const;
    std::vector<Unsigned> ModeWrapStrides() const;
    TensorDistribution Distribution() const;
    const tmen::Grid* Grid() const;

    ModeArray BoundModes() const;
    ModeArray FreeModes() const;
    ModeArray UnusedModes() const;

    void AddFreeMode(const Mode& mode);
    bool IsBound(Mode mode) const;
    bool IsFree(Mode mode) const;
    bool IsUnused(Mode mode) const;

    Unsigned LinearRank() const;
    void SetMyGridViewLoc();

private:
    TensorDistribution dist_;
    ModeArray boundModes_;
    ModeArray freeModes_;
    ModeArray unusedModes_;
    ObjShape shape_;
    Location loc_;

    const tmen::Grid* grid_;

    void SetupGridView(const ModeArray& unusedModes = ModeArray());
    void SetGridModeTypes(const ModeArray& unusedModes = ModeArray());
};

bool operator== ( const GridView& A, const GridView& B );
bool operator!= ( const GridView& A, const GridView& B );

} // namespace tmen

#endif // ifndef TMEN_CORE_GRID_VIEW_DECL_HPP
