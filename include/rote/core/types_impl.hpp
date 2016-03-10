/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#ifndef ROTE_CORE_TYPES_IMPL_HPP
#define ROTE_CORE_TYPES_IMPL_HPP

#include "rote/core/types_decl.hpp"
#include "rote/core/error_decl.hpp"
#include <stdlib.h>
#include <stdio.h>
//#include <regex>

namespace rote {

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

} // namespace rote
#endif
