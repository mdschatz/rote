/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef TMEN_CORE_ERROR_DECL_HPP
#define TMEN_CORE_ERROR_DECL_HPP

namespace elem{

inline void LogicError( std::string msg="LogicError" )
{ throw std::logic_error( msg.c_str() ); }

inline void RuntimeError( std::string msg="RuntimeError" )
{ throw std::runtime_error( msg.c_str() ); }

} //end namespace elem
#endif
