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
    return b == 0 ? a : GCD( b, a-b*(a/b) );
}

inline
Unsigned
LCM(Unsigned a, Unsigned b)
{
  return (a == 0 || b == 0) ? 0 : a*b/(GCD(a, b));
}

inline
std::vector<Unsigned>
LCMs(const std::vector<Unsigned>& a, const std::vector<Unsigned>& b)
{
  std::vector<Unsigned> ret(a.size());
  if(a.size() == 0 || b.size() == 0)
      return ret;
  for(Unsigned i = 0; i < ret.size(); i++)
      ret[i] = LCM(a[i], b[i]);
  return ret;
}

inline
Unsigned
Length( Unsigned n, Unsigned shift, Unsigned wrap )
{
#ifndef RELEASE
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
    return ( n > shift ? (n - shift - 1)/wrap + 1 : 0 );
}

inline
Unsigned
Length( Unsigned n, Int rank, Unsigned alignment, Unsigned wrap )
{
    Unsigned shift = Shift( rank, alignment, wrap );
    return Length( n, shift, wrap );
}

inline
std::vector<Unsigned>
Lengths( const ObjShape& objShape, const std::vector<Unsigned>& shifts, const ObjShape& wrapShape )
{
  Unsigned i;
#ifndef RELEASE
	Unsigned order;
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
    if( wrap == 0 )
        LogicError("Modulus must be positive");
#endif
    return ( n > 0 ? (n - 1)/wrap + 1 : 0 );
}

inline
std::vector<Unsigned>
MaxLengths( const ObjShape& shape, const ObjShape& wrapShape)
{
#ifndef RELEASE
    if(wrapShape.size() != shape.size())
        LogicError("shape order and wrapShape order must be the same");
    if( AnyZeroElem(wrapShape) )
        LogicError("wrapShape entries must be positive");
#endif
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
    return (rank + wrap - alignment) % wrap;
}

// For determining the first index assigned to a given rank
inline
std::vector<Unsigned>
Shifts( const std::vector<Unsigned>& modeRanks, const std::vector<Unsigned> alignments, const std::vector<Unsigned>& wrapShape )
{
#ifndef RELEASE
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
      LogicError( "Invalid index+stride combination");
  }

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
    Unsigned i;

    Unsigned linearInd = 0;
    for(i = 0; i < loc.size(); i++)
        linearInd += loc[i] * strides[i];

    return linearInd;
}

inline
Location
LinearLoc2Loc(const Unsigned linearLoc, const ObjShape& shape)
{
    if(shape.size() == 0 && linearLoc != 0)
        LogicError("Combination of linearLoc=0 and strides incompatible");
    const Unsigned order = shape.size();
    Unsigned remainder = linearLoc;
    const std::vector<Unsigned> strides = Dimensions2Strides(shape);

    Location ret(order);
    for(Unsigned i = order - 1; i < order; i--){
      const Unsigned modeLoc = remainder / strides[i];
      ret[i] = modeLoc;
      remainder -= modeLoc * strides[i];
    }
    return ret;
}

inline
std::vector<Unsigned>
IntCeils( const std::vector<Unsigned>& ms, const std::vector<Unsigned>& ns)
{
    std::vector<Unsigned> ret(ms.size());
    for(Unsigned i = 0; i < ret.size(); i++)
        ret[i] = IntCeil(ms[i], ns[i]);
    return ret;
}

} // namespace rote

#endif // ifndef ROTE_CORE_INDEXING_IMPL_HPP
