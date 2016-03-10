/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef ROTE_IO_PRINT_HPP
#define ROTE_IO_PRINT_HPP

#include <string>
#include <iostream>
#include <ostream>
#include "rote/core/tensor_forward_decl.hpp"
#include "rote/core/environment_decl.hpp"
#include "rote/core/indexing_decl.hpp"

namespace rote {

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
    if(order == 0){
        os.precision(16);
        os << A.Get(curLoc) << " " << std::endl;
        return;
    }
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

    if(A.Grid().LinearRank() == 0 && order == 0){
        os.precision(16);
        os << A.Get(curLoc) << " " << std::endl;
        return;
    }

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
PrintData
( const Tensor<T>& A, std::string title="", std::ostream& os = std::cout){
    os << title << std::endl;
    PrintVector(A.Shape(), "    shape", os);
    PrintVector(A.Strides(), "    strides");
}

template<typename T>
inline void
PrintData
( const DistTensor<T>& A, std::string title="", std::ostream& os = std::cout){
//    if( A.Grid().LinearRank() == 0 && title != "" ){
        os << title << std::endl;

        PrintVector(A.Shape(), "shape", os);
        os << "Distribution: " << rote::TensorDistToString(A.TensorDist()) << std::endl;
        PrintVector(A.Alignments(), "alignments", os);
        PrintVector(A.ModeShifts(), "shifts", os);
        PrintVector(A.LocalPermutation(), "local permutation", os);
        PrintData(A.LockedTensor(), "tensor data", os);
//    }
}

template<typename T>
inline void
PrintArray
( const T* dataBuf, const ObjShape& shape, const ObjShape strides, std::string title="", std::ostream& os = std::cout){

    Unsigned order = shape.size();
    Location curLoc(order, 0);
    Unsigned linLoc = 0;
    Unsigned ptr = 0;

    os << title << ":";

    if(order == 0){
        return;
    }

    bool done = !ElemwiseLessThan(curLoc, shape);

    while(!done){
        os.precision(16);

        os << " " << dataBuf[linLoc];

        //Update
        curLoc[ptr]++;
        linLoc += strides[ptr];
        while(ptr < order && curLoc[ptr] >= shape[ptr]){
            curLoc[ptr] = 0;

            linLoc -= strides[ptr] * (shape[ptr]);
            ptr++;
            if(ptr >= order){
                done = true;
                break;
            }else{
                curLoc[ptr]++;
                linLoc += strides[ptr];
            }
        }
        if(done)
            break;
        ptr = 0;
    }
    os << std::endl;
}

template<typename T>
inline void
PrintArray
( const T* dataBuf, const ObjShape& loopShape, std::string title="", std::ostream& os = std::cout){
    PrintArray(dataBuf, loopShape, Dimensions2Strides(loopShape), title, os);
}


inline void
PrintPackData
( const PackData& packData, std::string title="", std::ostream& os = std::cout){
    os << title << std::endl;
    PrintVector(packData.loopShape, "  loopShape", os);
    PrintVector(packData.srcBufStrides, "  srcBufStrides", os);
    PrintVector(packData.dstBufStrides, "  dstBufStrides", os);
}
} // namespace rote

#endif // ifndef ROTE_IO_PRINT_HPP
