/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef ROTE_CORE_INDEXING_IMPL_HPP
#define ROTE_CORE_INDEXING_IMPL_HPP

namespace rote {

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
std::vector<Unsigned>
LCMs_(const std::vector<Unsigned>& a, const std::vector<Unsigned>& b)
{
    Unsigned i;
    std::vector<Unsigned> ret(a.size());
    if(a.size() == 0 || b.size() == 0)
        return ret;
    for(i = 0; i < ret.size(); i++)
        ret[i] = LCM(a[i], b[i]);
    return ret;
}

inline
std::vector<Unsigned>
LCMs(const std::vector<Unsigned>& a, const std::vector<Unsigned>& b)
{
    return LCMs_(a, b);
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
    Unsigned i;
    std::vector<Unsigned> ret(shape.size());
    for(i = 0; i < ret.size(); i++)
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
    if( rank < 0 || rank >= Int(wrap) )
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

// For determining the first index assigned to a given rank
inline
std::vector<Unsigned>
Shifts( const std::vector<Unsigned>& modeRanks, const std::vector<Unsigned> alignments, const std::vector<Unsigned>& wrapShape )
{
#ifndef RELEASE
    CallStackEntry entry("Shift");
    if(modeRanks.size() != alignments.size() || alignments.size() != wrapShape.size() || modeRanks.size() != wrapShape.size())
        LogicError("modeRanks, alignments, and wrapShape must be of same order");
    if( !ElemwiseLessThan(modeRanks, wrapShape) )
    {
        Unsigned i;
        std::ostringstream msg;
        msg << "Invalid rank: "
            << "rank= (";
        if(modeRanks.size() > 0)
            msg << modeRanks[0];
        for(i = 1; i < modeRanks.size(); i++)
            msg << ", " << modeRanks[i];
        msg << "), stride=";
        if(wrapShape.size() > 0)
            msg << wrapShape[0];
        for(i = 1; i < wrapShape.size(); i++)
            msg << ", " << wrapShape[i];
        msg <<")";

        LogicError( msg.str() );
    }
    if( !ElemwiseLessThan(alignments, wrapShape) )
    {
        Unsigned i;
        std::ostringstream msg;
        msg << "Invalid alignment: "
            << "alignments= (";
        if(alignments.size() > 0)
            msg << alignments[0];
        for(i = 1; i < alignments.size(); i++)
            msg << ", " << alignments[i];
        msg << "), stride=";
        if(wrapShape.size() > 0)
            msg << wrapShape[0];
        for(i = 1; i < wrapShape.size(); i++)
            msg << ", " << wrapShape[i];
        msg <<")";

        LogicError( msg.str() );
        LogicError( msg.str() );
    }
#endif
    return Shifts_( modeRanks, alignments, wrapShape );
}

inline
std::vector<Unsigned>
Shifts_( const std::vector<Unsigned>& modeRanks, const std::vector<Unsigned> alignments, const std::vector<Unsigned>& wrapShape )
{
    std::vector<Unsigned> ret(modeRanks.size());
    for(Unsigned i = 0; i < ret.size(); i++)
        ret[i] = Shift(modeRanks[i], alignments[i], wrapShape[i]);
    return ret;
}

inline
std::vector<Unsigned>
Dimensions2Strides(const ObjShape& objShape)
{
  std::vector<Unsigned> strides(objShape.size());
  if(strides.size() > 0){
	  strides[0] = 1;
	  for(Unsigned i = 1; i < strides.size(); i++){
	      //NOTE: strides set to Max(1,c) to ensure we don't end up with 0 value stride
		  strides[i] = Max(1, strides[i-1]*objShape[i-1]);
	  }
  }
  return strides;
}

inline
Unsigned
Loc2LinearLoc(const Location& loc, const ObjShape& shape, const std::vector<Unsigned>& strides)
{
    if(strides.size() != 0 && (loc.size() != strides.size() || shape.size() != loc.size())){
//        Unsigned i;
//        printf("loc:");
//        for(i = 0; i < loc.size(); i++)
//            printf(" %d", loc[i]);
//        printf("\n");
//        printf("strides:");
//        for(i = 0; i < strides.size(); i++)
//            printf(" %d", strides[i]);
//        printf("\n");
        LogicError( "Invalid index+stride combination");
    }

    return Loc2LinearLoc_(loc, shape, strides);
}

inline
Unsigned
Loc2LinearLoc_(const Location& loc, const ObjShape& shape, const std::vector<Unsigned>& strides)
{
    Unsigned i;

	Unsigned linearInd = 0;
	if(strides.size() == 0){
	    if(loc.size() > 0){
            const std::vector<Unsigned> shapeStrides = Dimensions2Strides(shape);
            linearInd += loc[0];
            for(i = 1; i < loc.size(); i++)
                linearInd += loc[i] * shape[i-1] * shapeStrides[i-1];
	    }
	}else{
	    for(i = 0; i < loc.size(); i++)
	        linearInd += loc[i] * strides[i];
	}

	return linearInd;
}

inline
Unsigned
LinearLocFromStrides(const Location& loc, const std::vector<Unsigned>& strides)
{
    if(strides.size() != 0 && (loc.size() != strides.size())){
        LogicError( "Invalid index+stride combination");
    }
    return LinearLocFromStrides_(loc, strides);
}

inline
Unsigned
LinearLocFromStrides_(const Location& loc, const std::vector<Unsigned>& strides)
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
        const ObjShape permutedShape = PermuteVector(shape, permutation);
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
	if(gridViewLoc.size() != gridView.ParticipatingOrder())
		LogicError("Supplied loc must be same order as gridView");
	return GridViewLoc2GridLinearLoc_(gridViewLoc, gridView);
}


inline
Unsigned
GridViewLoc2GridLinearLoc_(const Location& gridViewLoc, const GridView& gridView)
{
#ifndef RELEASE
    CallStackEntry entry("GridViewloc2GridLinearLoc");
#endif

	const Unsigned gvOrder = gridView.ParticipatingOrder();
	const TensorDistribution tensorDist = gridView.Distribution();

	const rote::Grid* grid = gridView.Grid();
	const Unsigned gridOrder = grid->Order();
	const ObjShape gridShape = grid->Shape();
	Unsigned i, j;

	Location gridLoc(gridOrder);
	for(i = 0; i < gvOrder; i++){
		const ModeDistribution modeDist = tensorDist[i];
		const ObjShape gridSliceShape = PermuteVector(gridShape, modeDist);
		Location gridSliceLoc = LinearLoc2Loc(gridViewLoc[i], gridSliceShape);

		for(j = 0; j < gridSliceLoc.size(); j++){
			gridLoc[modeDist[j]] = gridSliceLoc[j];
		}
	}
	return Loc2LinearLoc(gridLoc, gridShape);
}

inline
Location
GridViewLoc2GridLoc(const Location& gridViewLoc, const GridView& gridView)
{
    if(gridViewLoc.size() != gridView.ParticipatingOrder())
        LogicError("Supplied loc must be same order as gridView");
    return GridViewLoc2GridLoc_(gridViewLoc, gridView);
}


inline
Location
GridViewLoc2GridLoc_(const Location& gridViewLoc, const GridView& gridView)
{

    const Unsigned gvOrder = gridView.ParticipatingOrder();
    const TensorDistribution tensorDist = gridView.Distribution();

    const rote::Grid* grid = gridView.Grid();
    const Unsigned gridOrder = grid->Order();
    const ObjShape gridShape = grid->Shape();
    Unsigned i, j;

    Location gridLoc(gridOrder);
    for(i = 0; i < gvOrder; i++){

        const ModeDistribution modeDist = tensorDist[i];
        const ObjShape gridSliceShape = PermuteVector(gridShape, modeDist);
        Location gridSliceLoc = LinearLoc2Loc(gridViewLoc[i], gridSliceShape);

        for(j = 0; j < gridSliceLoc.size(); j++){
            gridLoc[modeDist[j]] = gridSliceLoc[j];
        }
    }

    return gridLoc;
}

inline
Location
GridViewLoc2ParticipatingLoc(const Location& gridViewLoc, const GridView& gridView)
{
    return GridViewLoc2ParticipatingLoc_(gridViewLoc, gridView);
}

inline
Location
GridViewLoc2ParticipatingLoc_(const Location& gridViewLoc, const GridView& gridView)
{
    Unsigned i, j;
    const TensorDistribution dist = gridView.Distribution();
    const rote::Grid* g = gridView.Grid();
    const Unsigned participatingOrder = gridView.ParticipatingOrder();
    ModeArray participatingComms = ConcatenateVectors(gridView.FreeModes(), gridView.BoundModes());
    SortVector(participatingComms);

    const Location gvParticipatingLoc = gridView.ParticipatingLoc();

    ObjShape gridSlice = PermuteVector(g->Shape(), participatingComms);
    Location participatingGridLoc(gridSlice.size());

    for(i = 0; i < participatingOrder; i++){
        ModeDistribution modeDist = dist[i];
        ObjShape modeSliceShape = PermuteVector(g->Shape(), modeDist);
        const Location modeSliceLoc = LinearLoc2Loc(gridViewLoc[i], modeSliceShape);

        for(j = 0; j < modeDist.size(); j++){
            int indexOfMode = IndexOf(participatingComms, modeDist[j]);
            participatingGridLoc[indexOfMode] = modeSliceLoc[j];
        }
    }
    return participatingGridLoc;
}
inline
Unsigned
GridViewLoc2ParticipatingLinearLoc(const Location& gridViewLoc, const GridView& gridView)
{
    return GridViewLoc2ParticipatingLinearLoc_(gridViewLoc, gridView);
}

inline
Unsigned
GridViewLoc2ParticipatingLinearLoc_(const Location& gridViewLoc, const GridView& gridView)
{
    //Get the lin loc of the owner
    Unsigned i, j;
    int ownerLinearLoc = 0;
    const TensorDistribution dist = gridView.Distribution();
    const rote::Grid* g = gridView.Grid();
    const Unsigned participatingOrder = gridView.ParticipatingOrder();
    ModeArray participatingComms = ConcatenateVectors(gridView.FreeModes(), gridView.BoundModes());
    SortVector(participatingComms);

    const Location gvParticipatingLoc = gridView.ParticipatingLoc();

    ObjShape gridSlice = PermuteVector(g->Shape(), participatingComms);
    Location participatingGridLoc(gridSlice.size());

    for(i = 0; i < participatingOrder; i++){
        ModeDistribution modeDist = dist[i];
        ObjShape modeSliceShape = PermuteVector(g->Shape(), modeDist);
        const Location modeSliceLoc = LinearLoc2Loc(gridViewLoc[i], modeSliceShape);

        for(j = 0; j < modeDist.size(); j++){
            int indexOfMode = std::find(participatingComms.begin(), participatingComms.end(), modeDist[j]) - participatingComms.begin();
            participatingGridLoc[indexOfMode] = modeSliceLoc[j];
        }
    }
    ownerLinearLoc = Loc2LinearLoc(participatingGridLoc, gridSlice);
    return ownerLinearLoc;
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

        gridSliceLoc = PermuteVector(gridLoc, modeDist);
        gridSliceShape = PermuteVector(gridShape, modeDist);

        ret[i] = Loc2LinearLoc(gridSliceLoc, gridSliceShape);
    }
    return ret;
}

inline
Location
GridLoc2ParticipatingGridViewLoc(const Location& gridLoc, const ObjShape& gridShape, const TensorDistribution& tensorDist)
{
    Unsigned i;
    const Unsigned order = tensorDist.size() - 1;
    Location ret(order);

    for(i = 0; i < order; i++){
        ModeDistribution modeDist = tensorDist[i];
        Location gridSliceLoc(modeDist.size());
        ObjShape gridSliceShape(modeDist.size());

        gridSliceLoc = PermuteVector(gridLoc, modeDist);
        gridSliceShape = PermuteVector(gridShape, modeDist);

        ret[i] = Loc2LinearLoc(gridSliceLoc, gridSliceShape);
    }
    return ret;
}

inline
std::vector<Unsigned>
IntCeils( const std::vector<Unsigned>& ms, const std::vector<Unsigned>& ns)
{
    Unsigned i;
    std::vector<Unsigned> ret(ms.size());
    for(i = 0; i < ret.size(); i++)
        ret[i] = IntCeil(ms[i], ns[i]);
    return ret;
}

} // namespace rote

#endif // ifndef ROTE_CORE_INDEXING_IMPL_HPP
