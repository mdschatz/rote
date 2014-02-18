/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#ifndef TMEN_CORE_TYPES_IMPL_HPP
#define TMEN_CORE_TYPES_IMPL_HPP

#include "tensormental/core/types_decl.hpp"
#include "tensormental/core/error_decl.hpp"
#include <stdlib.h>

namespace tmen {

template<typename F>
inline
SafeProduct<F>::SafeProduct( Int numEntries )
: rho(1), kappa(0), n(numEntries)
{ }

namespace distribution_wrapper {

inline std::string 
TensorDistToString( TensorDistribution distribution )
{
    std::stringstream ss;
    ss << "[";// << distribution[0];
    //for(int i = 1; i < distribution.size(); i++)
    //  ss << ", " << distribution[i];
    ss <<  "]";
    return ss.str();
}

inline TensorDistribution
StringToTensorDist( std::string s )
{
    TensorDistribution distribution;
//    std::string delims = "[], ";
//    size_t cur;
//    size_t next = -1;
//    do
//    {
//      cur = next + 1;
//      next = s.find_first_of(delims, cur );
//      if(cur != next)
//        distribution.push_back(atoi(s.substr(cur, next - cur).c_str()));
//    }while(next != std::string::npos);
    return distribution;
}

} // namespace distribution_wrapper

namespace left_or_right_wrapper {

inline char 
LeftOrRightToChar( LeftOrRight side )
{
    char sideChar;
    switch( side )
    {
        case LEFT:  sideChar = 'L'; break;
        default:    sideChar = 'R'; break;
    }
    return sideChar;
}
    
inline LeftOrRight 
CharToLeftOrRight( char c )
{
    LeftOrRight side;
    switch( c )
    {
        case 'L': side = LEFT;  break;
        case 'R': side = RIGHT; break;
        default:
            LogicError("CharToLeftOrRight expects char in {L,R}");
    }
    return side;
}

} // namespace left_or_right_wrapper

namespace orientation_wrapper {

inline char 
OrientationToChar( Orientation orientation )
{
    char orientationChar;
    switch( orientation )
    {
        case NORMAL:    orientationChar = 'N'; break;
        case TRANSPOSE: orientationChar = 'T'; break;
        default:        orientationChar = 'C'; break;
    }
    return orientationChar;
}

inline Orientation 
CharToOrientation( char c )
{
    Orientation orientation;
    switch( c )
    {
        case 'N': orientation = NORMAL;    break;
        case 'T': orientation = TRANSPOSE; break;
        case 'C': orientation = ADJOINT;   break;
        default:
            LogicError
            ("CharToOrientation expects char in {N,T,C}");
    }
    return orientation;
}

} // namespace orientation_wrapper

namespace unit_or_non_unit_wrapper {

inline char 
UnitOrNonUnitToChar( UnitOrNonUnit diag )
{
    char diagChar;
    switch( diag )
    {
        case NON_UNIT: diagChar = 'N'; break;
        default:       diagChar = 'U'; break;
    }
    return diagChar;
}

inline UnitOrNonUnit 
CharToUnitOrNonUnit( char c )
{
    UnitOrNonUnit diag;
    switch( c )
    {
        case 'N': diag = NON_UNIT; break;
        case 'U': diag = UNIT;     break;
        default:
            LogicError("CharToUnitOrNonUnit expects char in {N,U}");
    }
    return diag;
}

} // namespace unit_or_non_unit_wrapper

namespace upper_or_lower_wrapper {

inline char 
UpperOrLowerToChar( UpperOrLower uplo )
{
    char uploChar;
    switch( uplo )
    {
        case LOWER: uploChar = 'L'; break;
        default:    uploChar = 'U'; break;
    }
    return uploChar;
}

inline UpperOrLower 
CharToUpperOrLower( char c )
{
    UpperOrLower uplo;
    switch( c )
    {
        case 'L': uplo = LOWER; break;
        case 'U': uplo = UPPER; break;
        default:
            LogicError("CharToUpperOrLower expects char in {L,U}");
    }
    return uplo;
}

} // namespace upper_or_lower_wrapper

} // namespace tmen
#endif
