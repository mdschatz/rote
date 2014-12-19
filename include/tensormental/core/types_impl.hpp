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
#include <stdio.h>
//#include <regex>

namespace tmen {

template<typename F>
inline
SafeProduct<F>::SafeProduct( Int numEntries )
: rho(1), kappa(0), n(numEntries)
{ }

namespace distribution_wrapper {

inline std::string 
TensorDistToString( const TensorDistribution& distribution, bool endLine )
{
    std::stringstream ss;
    ss << "[";
    if(distribution.size() > 1){
    	ss << ModeDistToString_(distribution[0]);
		for(size_t i = 1; i < distribution.size()-1; i++)
		  ss << ", " << ModeDistToString_(distribution[i]);
    }
    ss <<  "]|";
    ss << ModeDistToString_(distribution[distribution.size()-1]);
    if(endLine)
        ss << std::endl;
    return ss.str();
}

inline std::string
ModeDistToString_( const ModeDistribution& distribution, bool endLine )
{
    std::stringstream ss;
    ss << "(";
    if(distribution.size() >= 1){
    	ss << distribution[0];
		for(size_t i = 1; i < distribution.size(); i++)
		  ss << ", " << distribution[i];
    }
    ss <<  ")";
    if(endLine)
        ss << std::endl;
    return ss.str();
}

inline std::string
ModeDistToString( const ModeDistribution& distribution, bool endLine )
{
    return ModeDistToString_(distribution, endLine);
}

//TODO: Figure out how to error check these without C++11
inline TensorDistribution
StringToTensorDist( const std::string& s )
{
    TensorDistribution distribution;
    ModeArray ignoreModes;

    size_t pos, lastPos, breakPos;
    breakPos = s.find_first_of("|");
    pos = s.find_first_of("[");
    lastPos = s.find_first_of("]");
    pos = s.find_first_of("(", pos);
    while(pos < breakPos){
        lastPos = s.find_first_of(")", pos);
        distribution.push_back(StringToModeDist(s.substr(pos, lastPos - pos + 1)));
        pos = s.find_first_of("(", lastPos + 1);
    }

    if(breakPos != std::string::npos){
        //Break found, ignore modes
        ignoreModes = StringToModeDist(s.substr(breakPos+1, s.length() - breakPos + 1));
    }
    distribution.push_back(ignoreModes);

    return distribution;
}

inline ModeDistribution
StringToModeDist( const std::string& s)
{
	ModeDistribution distribution(0);
	size_t pos, lastPos;
	pos = s.find_first_of("(");
	lastPos = s.find_first_of(")");
	if(pos != 0 || lastPos != s.size() - 1)
		LogicError("Malformed mode distribution string");
	pos = s.find_first_not_of("(,)", pos);
	while(pos != std::string::npos){
		lastPos = s.find_first_of(",)", pos);
		distribution.push_back(atoi(s.substr(pos, lastPos - pos).c_str()));
		pos = s.find_first_not_of("(,)", lastPos+1);
	}
	return distribution;

}

} // namespace distribution_wrapper

namespace left_or_right_wrapper {

inline char 
LeftOrRightToChar( LeftOrRight side )
{
    char sideChar = 'L';
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
    LeftOrRight side = LEFT;
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
    char orientationChar = 'N';
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
    Orientation orientation = NORMAL;
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
    char diagChar = 'N';
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
    UnitOrNonUnit diag = NON_UNIT;
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
    char uploChar = 'L';
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
    UpperOrLower uplo = LOWER;
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
