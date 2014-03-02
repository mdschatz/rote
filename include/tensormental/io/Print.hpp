/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef TMEN_IO_PRINT_HPP
#define TMEN_IO_PRINT_HPP

#include <string>
#include <iostream>
#include <ostream>
#include "tensormental/core/tensor_forward_decl.hpp"
#include "tensormental/core/environment_decl.hpp"

namespace tmen {

template<typename T>
inline void
Print( const Tensor<T>& A, std::string title="", std::ostream& os=std::cout )
{
#ifndef RELEASE
    CallStackEntry entry("Print");
#endif
    if( title != "" )
        os << title << std::endl;
    
    const Int order = A.Order();
    std::vector<Int> curIndex(order);
    std::fill(curIndex.begin(), curIndex.end(), 0);
    int ptr = 0;
    bool done = false;
    while(true){

    	//Update
    	curIndex[ptr]++;
    	while(ptr < order && curIndex[ptr] == A.Dimension(ptr)){
    		curIndex[ptr] = 0;
    		ptr++;
    		if(ptr >= order){
    			done = true;
    			break;
    		}else{
    			curIndex[ptr]++;
    		}
    	}
    	if(done)
    		break;
    	ptr = 0;
    }
    os << std::endl;
}

// If already in [* ,* ] or [o ,o ] distributions, no copy is needed
template<typename T>
inline void
Print
( const DistTensor<T>& A, std::string title="",
  std::ostream& os=std::cout )
{
#ifndef RELEASE
    CallStackEntry entry("Print");
#endif
    if( A.Grid().LinearRank() == 0 && title != "" )
        os << title << std::endl;

    const Int order = A.Order();
    std::vector<Int> curIndex(order);
    std::fill(curIndex.begin(), curIndex.end(), 0);
    int ptr = 0;
    bool done = false;
    T u;
    while(true){
    	u = A.Get(curIndex);

    	if(A.Grid().LinearRank() == 0){
    		os << u << " ";
    	}

    	//Update
    	curIndex[ptr]++;
    	while(ptr < order && curIndex[ptr] == A.Dimension(ptr)){
    		curIndex[ptr] = 0;
    		ptr++;
    		if(ptr >= order){
    			done = true;
    			break;
    		}else{
    			curIndex[ptr]++;
    		}
    	}
    	if(done)
    		break;
    	ptr = 0;
    }
    if(A.Grid().LinearRank() == 0){
    	os << std::endl;
    }
}

} // namespace tmen

#endif // ifndef TMEN_IO_PRINT_HPP
