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
    explicit GridView( const tmen::Grid* g, const TensorDistribution& dist );
    ~GridView();

    // Simple interface (simpler version of distributed-based interface)
    int Order() const;
    std::vector<Int> Shape() const;
    int Dimension(int mode) const;
    std::vector<Int> Loc() const;
    int ModeLoc(int mode) const;
    int ModeWrapStride(int mode) const;
    std::vector<Int> ModeWrapStrides() const;
    TensorDistribution Distribution() const;
    const tmen::Grid* Grid() const;

    int LinearRank() const;
    void SetMyGridViewLoc();

private:
    TensorDistribution dist_;
    std::vector<int> shape_;
    std::vector<int> loc_;

    const tmen::Grid* grid_;

    void SetupGridView();
};

bool operator== ( const GridView& A, const GridView& B );
bool operator!= ( const GridView& A, const GridView& B );

} // namespace tmen

#endif // ifndef TMEN_CORE_GRID_VIEW_DECL_HPP
