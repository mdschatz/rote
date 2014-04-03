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

//Int DiagonalLength( const std::vector<Int>& dims, Int offset=0 );

Int GCD( Int a, Int b ); 
Int GCD_( Int a, Int b ); 

Int LCM( Int a, Int b );
Int LCM_( Int a, Int b );

Int Length( Int n, Int shift, Int numProcs );
Int Length_( Int n, Int shift, Int numProcs );

Int Length( Int n, Int rank, Int firstRank, Int numProcs );
Int Length_( Int n, Int rank, Int firstRank, Int numProcs );

Int MaxLength( Int n, Int numProcs );
Int MaxLength_( Int n, Int numProcs );

std::vector<Int> MaxLengths(const std::vector<Int>& shape, const std::vector<Int>& wrapShape);
std::vector<Int> MaxLengths_(const std::vector<Int>& shape, const std::vector<Int>& wrapShape);

Int Shift( Int rank, Int firstRank, Int numProcs );
Int Shift_( Int rank, Int firstRank, Int numProcs );

std::vector<Int> Lengths(const std::vector<Int>& dimensions, const std::vector<Int>& shifts, const std::vector<Int>& numProcs);
std::vector<Int> Lengths_(const std::vector<Int>& dimensions, const std::vector<Int>& shifts, const std::vector<Int>& numProcs);

std::vector<Int> Dimensions2Strides(const std::vector<Int>& dimensions);
std::vector<Int> Dimensions2Strides_(const std::vector<Int>& dimensions);

Int LinearIndex(const std::vector<Int>& index, const std::vector<Int>& strides);
Int LinearIndex_(const std::vector<Int>& index, const std::vector<Int>& strides);

std::vector<Int> LinearLoc2Loc(const int linearLoc, const std::vector<Int>& shape, const std::vector<int>& permutation = std::vector<int>());
std::vector<Int> LinearLoc2Loc_(const int linearLoc, const std::vector<Int>& shape, const std::vector<int>& permutation = std::vector<int>());

int GridViewLoc2GridLinearLoc(const std::vector<int>& gridViewLoc, const GridView& gridView);
int GridViewLoc2GridLinearLoc_(const std::vector<int>& gridViewLoc, const GridView& gridView);

std::vector<int> GridLoc2GridViewLoc(const std::vector<int>& gridLoc, const std::vector<int>& gridShape, const TensorDistribution& tensorDist);

} // namespace tmen

#endif // ifndef TMEN_CORE_INDEXING_DECL_HPP
