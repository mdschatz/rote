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
        os << title << " ";
    
    const Unsigned order = A.Order();
    Location curLoc(order, 0);

    int ptr = 0;
    bool done = order > 0 && !ElemwiseLessThan(curLoc, A.Shape());
    while(!done){
        os.precision(16);
        os << A.Get(curLoc) << " ";
        if(order == 0)
            break;
    	//Update
    	curLoc[ptr]++;
    	while(ptr < order && curLoc[ptr] == A.Dimension(ptr)){
    		curLoc[ptr] = 0;
    		ptr++;
    		if(ptr >= order){
    			done = true;
    			break;
    		}else{
    			curLoc[ptr]++;
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

    const Unsigned order = A.Order();
    Location curLoc(order, 0);

    int ptr = 0;
    bool done = order > 0 && !ElemwiseLessThan(curLoc, A.Shape());
    T u = T(0);
    while(!done){
    	u = A.Get(curLoc);

    	if(A.Grid().LinearRank() == 0){
    	    os.precision(16);
    		os << u << " ";
    	}

    	if(order == 0)
    	    break;
    	//Update
    	curLoc[ptr]++;
    	while(ptr < order && curLoc[ptr] == A.Dimension(ptr)){
    		curLoc[ptr] = 0;
    		ptr++;
    		if(ptr >= order){
    			done = true;
    			break;
    		}else{
    			curLoc[ptr]++;
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

template<typename T>
inline void
PrintVector
( const std::vector<T>& vec, std::string title="", std::ostream& os = std::cout){
    os << title << ":";

    Unsigned i;
    for(i = 0; i < vec.size(); i++)
        os << " " << vec[i];
    os << std::endl;
}

template<typename T>
inline void
PrintData
( const Tensor<T>& A, std::string title="", std::ostream& os = std::cout){
    os << title << std::endl;
    PrintVector(A.Shape(), "    shape", os);
    PrintVector(A.Strides(), "    strides");
//    os << "    local data:";
//    const T* buffer = A.LockedBuffer();
//    for(Unsigned i = 0; i < prod(A.Shape()); i++)
//        os << " " << buffer[i];
//    os << std::endl;
}

template<typename T>
inline void
PrintData
( const DistTensor<T>& A, std::string title="", std::ostream& os = std::cout){
    if( A.Grid().LinearRank() == 0 && title != "" ){
        os << title << std::endl;

        PrintVector(A.Shape(), "shape", os);
        os << "Distribution: " << tmen::TensorDistToString(A.TensorDist()) << std::endl;
        PrintVector(A.Alignments(), "alignments", os);
        PrintVector(A.ModeShifts(), "shifts", os);
        PrintVector(A.LocalPermutation(), "local permutation", os);
        PrintData(A.LockedTensor(), "tensor data", os);
    }
}

inline void
PrintPackData
( const PackData& packData, std::string title="", std::ostream& os = std::cout){
    os << title << std::endl;
    PrintVector(packData.loopShape, "  loopShape", os);
    PrintVector(packData.loopStarts, "  loopStarts", os);
    PrintVector(packData.loopIncs, "  loopIncs", os);
    PrintVector(packData.srcBufStrides, "  srcBufStrides", os);
    PrintVector(packData.dstBufStrides, "  dstBufStrides", os);
}

inline void
PrintElemSelectData
( const ElemSelectData& elemData, std::string title="", std::ostream& os = std::cout){
    os << title << std::endl;
    PrintVector(elemData.loopShape, "  loopShape", os);
    os << "  nElemsPerProc: "<< elemData.nElemsPerProc << std::endl;
    PrintVector(elemData.packElem, "  packElem", os);
    PrintVector(elemData.srcElem, "  srcElem", os);
    PrintVector(elemData.srcStrides, "  srcBufStrides", os);
    PrintVector(elemData.permutation, "  permutation", os);
    PrintVector(elemData.commModes, "  commModes", os);

}
} // namespace tmen

#endif // ifndef TMEN_IO_PRINT_HPP
