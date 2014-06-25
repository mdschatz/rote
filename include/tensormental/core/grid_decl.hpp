/*
   Copyright (c) 2009-2013, Jack Poulson
                      2013, Jed Brown 
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef TMEN_CORE_GRID_DECL_HPP
#define TMEN_CORE_GRID_DECL_HPP

#include <iostream>
#include <vector>
#include "tensormental/core/imports/mpi.hpp"
#include "tensormental/core/environment_decl.hpp"

namespace tmen {

class Grid
{
public:
    explicit Grid( mpi::Comm comm, const ObjShape& shape );
    ~Grid();

    // Simple interface (simpler version of distributed-based interface)
    Unsigned Order() const;
    Unsigned Size() const;
    ObjShape Shape() const;
    Unsigned Dimension(Mode mode) const;
    Location Loc() const;
    Unsigned ModeLoc(Mode mode) const;

    Unsigned LinearRank() const;
    void SetMyGridLoc();

//These should be pushed to a separate GridView class
//Grid is meant for the bottom-most layer
//    int LocUnderView(GridView view) const;
//    int DimensionUnderView(GridView view) const;
//    mpi::Comm CommUnderView(GridView view) const;

    // Advanced routines
    explicit Grid( mpi::Comm viewers, mpi::Group owners, Unsigned height );
    bool InGrid() const;
    mpi::Comm OwningComm() const;

    static int FindFactor( int p );

private:
    ObjShape shape_;
    Unsigned size_;
    Unsigned linearRank_;
    Location gridLoc_;

    //TODO: Look at how these are used
    // The processes that do and do not own data
    //mpi::Group owningGroup_, notOwningGroup_;

    // Keep track of whether or not our process is in the grid. This is 
    // necessary to avoid calls like MPI_Comm_size when we're not in the
    // communicator's group. Note that we _can_ call MPI_Group_rank when not 
    // in the group and that the result is MPI_UNDEFINED.
    bool inGrid_;

    // These will only be valid if we are in the grid
    mpi::Comm cartComm_;  // the processes that are in the grid
    mpi::Comm owningComm_;

    void SetUpGrid();

    // Disable copying this class due to MPI_Comm/MPI_Group ownership issues
    // and potential performance loss from duplicating MPI communicators, e.g.,
    // on Blue Gene/P there is supposedly a performance loss
    const Grid& operator=( Grid& );
    Grid( const Grid& );
};

bool operator== ( const Grid& A, const Grid& B );
bool operator!= ( const Grid& A, const Grid& B );

// Return a grid constructed using mpi::COMM_WORLD.
const Grid& DefaultGrid();

} // namespace tmen

#endif // ifndef TMEN_CORE_GRID_DECL_HPP
