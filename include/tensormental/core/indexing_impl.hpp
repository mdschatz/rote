/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef TMEN_CORE_INDEXING_IMPL_HPP
#define TMEN_CORE_INDEXING_IMPL_HPP

namespace tmen {

inline
Unsigned
GCD( Unsigned a, Unsigned b )
{
    return GCD_( a, b );
}

inline
Unsigned
GCD_( Unsigned a, Unsigned b )
{
    if( b == 0 )
        return a;
    else
        return GCD_( b, a-b*(a/b) );
}

inline
Unsigned
LCM(Unsigned a, Unsigned b)
{
    return LCM_(a, b);
}

inline
Unsigned
LCM_( Unsigned a, Unsigned b )
{
    if(a == 0 || b == 0)
        return 0;
    return a*b/(GCD(a, b));
}

inline
Unsigned
Length( Unsigned n, Unsigned shift, Unsigned wrap )
{
#ifndef RELEASE
    CallStackEntry entry("Length");
    if( shift >= wrap )
    {
        std::ostringstream msg;
        msg << "Invalid shift: "
            << "shift=" << shift << ", wrap=" << wrap;
        LogicError( msg.str() );
    }
    if( wrap == 0 )
        LogicError("Modulus must be positive");
#endif
    return Length_( n, shift, wrap );
}

inline
Unsigned
Length_( Unsigned n, Unsigned shift, Unsigned wrap )
{
    return ( n > shift ? (n - shift - 1)/wrap + 1 : 0 );
}

inline
Unsigned
Length( Unsigned n, Int rank, Unsigned alignment, Unsigned wrap )
{
#ifndef RELEASE
    CallStackEntry entry("Length");
#endif
    Unsigned shift = Shift( rank, alignment, wrap );
    return Length( n, shift, wrap );
}

inline
Unsigned
Length_( Unsigned n, Int rank, Unsigned alignment, Unsigned wrap )
{
    Unsigned shift = Shift_( rank, alignment, wrap );
    return Length_( n, shift, wrap );
}

inline
std::vector<Unsigned>
Lengths( const ObjShape& objShape, const std::vector<Unsigned>& shifts, const ObjShape& wrapShape )
{
#ifndef RELEASE
	Unsigned i;
	Unsigned order;
    CallStackEntry entry("Length");
    if(!(objShape.size() == shifts.size() && objShape.size() == wrapShape.size())){
    	LogicError("dimensions, shifts, modulos must contain same number of elements.");
    }
    order = objShape.size();
    for(i = 0; i < order; i++){
		if( shifts[i] >= wrapShape[i] )
		{
			std::ostringstream msg;
			msg << "Invalid shift: "
				<< "shift[" << i << "]=" << shifts[i] << ", stride[" << i << "]=" << wrapShape[i];
			LogicError( msg.str() );
		}
    }
#endif
    return Lengths_( objShape, shifts, wrapShape );
}

inline
std::vector<Unsigned>
Lengths_( const ObjShape& objShape, const std::vector<Unsigned>& shifts, const ObjShape& wrapShape )
{
	Unsigned i;
	std::vector<Unsigned> lengths(objShape.size());
	for(i = 0; i < objShape.size(); i++)
		lengths[i] = Length(objShape[i], shifts[i], wrapShape[i]);
	return lengths;
}

inline
Unsigned
MaxLength( Unsigned n, Unsigned wrap )
{
#ifndef RELEASE
    CallStackEntry entry("MaxLength");
    if( wrap == 0 )
        LogicError("Modulus must be positive");
#endif
    return MaxLength_( n, wrap );
}

inline
Unsigned
MaxLength_( Unsigned n, Unsigned wrap )
{
    return ( n > 0 ? (n - 1)/wrap + 1 : 0 );
}

inline
std::vector<Unsigned>
MaxLengths( const ObjShape& shape, const ObjShape& wrapShape)
{
#ifndef RELEASE
    CallStackEntry entry("MaxLengths");
    if(wrapShape.size() != shape.size())
        LogicError("shape order and wrapShape order must be the same");
    if( AnyZeroElem(wrapShape) )
        LogicError("wrapShape entries must be positive");
#endif
    return MaxLengths_( shape, wrapShape );
}

inline
std::vector<Unsigned>
MaxLengths_( const ObjShape& shape, const ObjShape& wrapShape)
{
    std::vector<Unsigned> ret(shape.size());
    for(Unsigned i = 0; i < ret.size(); i++)
        ret[i] = MaxLength(shape[i], wrapShape[i]);
    return ret;
}

// For determining the first index assigned to a given rank
inline
Unsigned
Shift( Int rank, Unsigned alignment, Unsigned wrap )
{
#ifndef RELEASE
    CallStackEntry entry("Shift");
    if( rank < 0 || rank >= wrap )
    {
        std::ostringstream msg;
        msg << "Invalid rank: "
            << "rank=" << rank << ", stride=" << wrap;
        LogicError( msg.str() );
    }
    if( alignment >= wrap )
    {
        std::ostringstream msg;
        msg << "Invalid alignment: "
            << "alignment=" << alignment << ", wrap=" << wrap;
        LogicError( msg.str() );
    }
#endif
    return Shift_( rank, alignment, wrap );
}

inline
Unsigned
Shift_( Int rank, Unsigned alignment, Unsigned wrap )
{ return (rank + wrap - alignment) % wrap; }

inline
std::vector<Unsigned>
Dimensions2Strides(const ObjShape& objShape)
{
  std::vector<Unsigned> strides(objShape.size());
  if(strides.size() > 0){
	  strides[0] = 1;
	  for(Unsigned i = 1; i < strides.size(); i++){
		  strides[i] = strides[i-1]*objShape[i-1];
	  }
  }
  return strides;
}

inline
Unsigned
LinearIndex(const Location& loc, const std::vector<Unsigned>& strides)
{
    if(loc.size() != strides.size())
        LogicError( "Invalid index+stride combination");
    return LinearIndex_(loc, strides);
}

inline
Unsigned
LinearIndex_(const Location& loc, const std::vector<Unsigned>& strides)
{
    Unsigned i;
	Unsigned linearInd = 0;
	for(i = 0; i < loc.size(); i++)
		linearInd += loc[i] * strides[i];
	return linearInd;
}

inline
Location
LinearLoc2Loc(const Unsigned linearLoc, const ObjShape& shape, const Permutation& permutation)
{
    if(permutation.size() > 0 && shape.size() != permutation.size())
        LogicError("Shape and Permutation orders differ.");
    if(shape.size() == 0 && linearLoc != 0)
        LogicError("Combination of linearLoc=0 and strides incompatible");
    return LinearLoc2Loc_(linearLoc, shape, permutation);
}

//TODO: Clean up FilterVector with () as filter
inline
Location
LinearLoc2Loc_(const Unsigned linearLoc, const ObjShape& shape, const Permutation& permutation)
{
    Unsigned i;
    const Unsigned order = shape.size();
    Location ret(order);
    Unsigned remainder = linearLoc;
    if(permutation.size() == 0){
        const std::vector<Unsigned> strides = Dimensions2Strides(shape);

        for(i = order - 1; i < order; i--){
                const Unsigned modeLoc = remainder / strides[i];
                ret[i] = modeLoc;
                remainder -= modeLoc * strides[i];
        }
    }else{
        const ObjShape permutedShape = FilterVector(shape, permutation);
        const std::vector<Unsigned> strides = Dimensions2Strides(permutedShape);

        for(i = order - 1; i < order; i--){
            const Unsigned modeLoc = remainder / strides[i];
            ret[permutation[i]] = modeLoc;
            remainder -= modeLoc * strides[i];
        }
    }
    return ret;
}

inline
Unsigned
GridViewLoc2GridLinearLoc(const Location& gridViewLoc, const GridView& gridView)
{
	if(gridViewLoc.size() != gridView.Order())
		LogicError("Supplied loc must be same order as gridView");
	return GridViewLoc2GridLinearLoc_(gridViewLoc, gridView);
}


inline
Unsigned
GridViewLoc2GridLinearLoc_(const Location& gridViewLoc, const GridView& gridView)
{

	const Unsigned gvOrder = gridView.Order();
	const TensorDistribution tensorDist = gridView.Distribution();

	const tmen::Grid* grid = gridView.Grid();
	const Unsigned gridOrder = grid->Order();
	const ObjShape gridShape = grid->Shape();
	Unsigned i, j;

	Location gridLoc(gridOrder);
	for(i = 0; i < gvOrder; i++){
		const ModeDistribution modeDist = tensorDist[i];
		const ObjShape gridSliceShape = FilterVector(gridShape, modeDist);
		Location gridSliceLoc = LinearLoc2Loc(gridViewLoc[i], gridSliceShape);

		for(j = 0; j < gridSliceLoc.size(); j++){
			gridLoc[modeDist[j]] = gridSliceLoc[j];
		}
	}
	return LinearIndex(gridLoc, Dimensions2Strides(gridShape));
}

inline
Location
GridViewLoc2GridLoc(const Location& gridViewLoc, const GridView& gridView)
{
    if(gridViewLoc.size() != gridView.Order())
        LogicError("Supplied loc must be same order as gridView");
    return GridViewLoc2GridLoc_(gridViewLoc, gridView);
}


inline
Location
GridViewLoc2GridLoc_(const Location& gridViewLoc, const GridView& gridView)
{

    const Unsigned gvOrder = gridView.Order();
    const TensorDistribution tensorDist = gridView.Distribution();

    const tmen::Grid* grid = gridView.Grid();
    const Unsigned gridOrder = grid->Order();
    const ObjShape gridShape = grid->Shape();
    Unsigned i, j;

    Location gridLoc(gridOrder);
    for(i = 0; i < gvOrder; i++){

        const ModeDistribution modeDist = tensorDist[i];
        const ObjShape gridSliceShape = FilterVector(gridShape, modeDist);
        Location gridSliceLoc = LinearLoc2Loc(gridViewLoc[i], gridSliceShape);

        for(j = 0; j < gridSliceLoc.size(); j++){
            gridLoc[modeDist[j]] = gridSliceLoc[j];
        }
    }
    return gridLoc;
}

inline
Location
GridLoc2GridViewLoc(const Location& gridLoc, const ObjShape& gridShape, const TensorDistribution& tensorDist)
{
    Unsigned i;
    const Unsigned order = tensorDist.size();
    Location ret(order);

    for(i = 0; i < order; i++){
        ModeDistribution modeDist = tensorDist[i];
        Location gridSliceLoc(modeDist.size());
        ObjShape gridSliceShape(modeDist.size());

        gridSliceLoc = FilterVector(gridLoc, modeDist);
        gridSliceShape = FilterVector(gridShape, modeDist);

        ret[i] = LinearIndex(gridSliceLoc, Dimensions2Strides(gridSliceShape));
    }
    return ret;
}

} // namespace tmen

#endif // ifndef TMEN_CORE_INDEXING_IMPL_HPP
