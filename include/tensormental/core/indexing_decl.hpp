/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef TMEN_CORE_INDEXING_DECL_HPP
#define TMEN_CORE_INDEXING_DECL_HPP

namespace tmen {

Unsigned GCD( Unsigned a, Unsigned b );
Unsigned GCD_( Unsigned a, Unsigned b );

Unsigned LCM( Unsigned a, Unsigned b );
Unsigned LCM_( Unsigned a, Unsigned b );

Unsigned Length( Unsigned n, Unsigned shift, Unsigned wrap );
Unsigned Length_( Unsigned n, Unsigned shift, Unsigned wrap );

Unsigned Length( Unsigned n, Int rank, Unsigned alignment, Unsigned wrap );
Unsigned Length_( Unsigned n, Int rank, Unsigned alignment, Unsigned wrap );

std::vector<Unsigned> Lengths(const ObjShape& objShape, const std::vector<Unsigned>& shifts, const ObjShape& wrapShape);
std::vector<Unsigned> Lengths_(const ObjShape& objShape, const std::vector<Unsigned>& shifts, const ObjShape& wrapShape);

Unsigned MaxLength( Unsigned n, Unsigned wrap );
Unsigned MaxLength_( Unsigned n, Unsigned wrap );

std::vector<Unsigned> MaxLengths(const ObjShape& objShape, const ObjShape& wrapShape);
std::vector<Unsigned> MaxLengths_(const ObjShape& objShape, const ObjShape& wrapShape);

Unsigned Shift( Int rank, Unsigned alignment, Unsigned wrap );
Unsigned Shift_( Int rank, Unsigned alignment, Unsigned wrap );

std::vector<Unsigned> Dimensions2Strides(const ObjShape& objShape);

Unsigned LinearIndex(const Location& loc, const std::vector<Unsigned>& strides);
Unsigned LinearIndex_(const Location& loc, const std::vector<Unsigned>& strides);

Location LinearLoc2Loc(Unsigned linearLoc, const ObjShape& objShape, const Permutation& permutation = Permutation());
Location LinearLoc2Loc_(Unsigned linearLoc, const ObjShape& objShape, const Permutation& permutation = Permutation());

Unsigned GridViewLoc2GridLinearLoc(const Location& gridViewLoc, const GridView& gridView);
Unsigned GridViewLoc2GridLinearLoc_(const Location& gridViewLoc, const GridView& gridView);

Location GridViewLoc2GridLoc(const Location& gridViewLoc, const GridView& gridView);
Location GridViewLoc2GridLoc_(const Location& gridViewLoc, const GridView& gridView);

Location GridLoc2GridViewLoc(const Location& gridLoc, const ObjShape& gridShape, const TensorDistribution& tensorDist);

} // namespace tmen

#endif // ifndef TMEN_CORE_INDEXING_DECL_HPP
