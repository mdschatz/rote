/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef ROTE_CORE_INDEXING_HPP
#define ROTE_CORE_INDEXING_HPP

namespace rote {

//
// Scalar
//
Unsigned GCD( Unsigned a, Unsigned b );
Unsigned LCM( Unsigned a, Unsigned b );
Unsigned Length( Unsigned n, Unsigned shift, Unsigned wrap );
Unsigned Length( Unsigned n, Int rank, Unsigned alignment, Unsigned wrap );
Unsigned MaxLength( Unsigned n, Unsigned wrap );
Unsigned Shift( Int rank, Unsigned alignment, Unsigned wrap );

//
// Vector
//
std::vector<Unsigned> LCMs( const std::vector<Unsigned>& a, const std::vector<Unsigned>& b );
std::vector<Unsigned> Lengths(const ObjShape& objShape, const std::vector<Unsigned>& shifts, const ObjShape& wrapShape);
std::vector<Unsigned> MaxLengths(const ObjShape& objShape, const ObjShape& wrapShape);
std::vector<Unsigned> Shifts( const std::vector<Unsigned>& modeRanks, const std::vector<Unsigned> alignments, const std::vector<Unsigned>& wrapShape);
std::vector<Unsigned> IntCeils(const std::vector<Unsigned>& ms, const std::vector<Unsigned>& ns);

// Other
std::vector<Unsigned> Dimensions2Strides(const ObjShape& objShape);
Unsigned Loc2LinearLoc(const Location& loc, const ObjShape& shape);
Location LinearLoc2Loc(Unsigned linearLoc, const ObjShape& objShape);
Unsigned LinearLocFromStrides(const Location& loc, const std::vector<Unsigned>& strides);

} // namespace rote

#endif // ifndef ROTE_CORE_INDEXING_HPP
