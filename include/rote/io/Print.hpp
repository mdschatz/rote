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
Print( const Tensor<T>& A, std::string title="", bool all = false )
{
#ifndef RELEASE
    CallStackEntry entry("Print");
#endif
    std::ostream& os = std::cout;
    if( title != "" )
      os << title << " ";

    const Unsigned order = A.Order();
    Location curLoc(order, 0);

    int ptr = 0;
    if(order == 0) {
      os.precision(16);
      if (all || mpi::CommRank(MPI_COMM_WORLD) == 0) {
        os << A.Get(curLoc) << " " << std::endl;
      }
      return;
    }
    bool done = order > 0 && !ElemwiseLessThan(curLoc, A.Shape());

    while(!done){
      os.precision(16);
      T val = A.Get(curLoc);
      if (all || mpi::CommRank(MPI_COMM_WORLD) == 0) {
        os << val << " ";
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
    os << std::endl;
}


template<typename T>
inline void
PrintVector
( const std::vector<T>& vec, std::string title="", bool all = false){
  if (all || mpi::CommRank(MPI_COMM_WORLD) == 0) {
    std::ostream& os = std::cout;
    os << title << ":";

    Unsigned i;
    for(i = 0; i < vec.size(); i++)
      os << " " << vec[i];
    os << std::endl;
  }
}

template<typename T>
inline void
Print
( const DistTensor<T>& A, std::string title="")
{
#ifndef RELEASE
    CallStackEntry entry("Print");
#endif
    std::ostream& os = std::cout;
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
( const Tensor<T>& A, std::string title="", bool all=false){
    std::ostream& os = std::cout;
    os << title << std::endl;
    PrintVector(A.Shape(), "    shape", all);
    PrintVector(A.Strides(), "    strides", all);
}

template<typename T>
inline void
PrintData
( const DistTensor<T>& A, std::string title="", bool all=false){
      std::ostream& os = std::cout;
//    if( A.Grid().LinearRank() == 0 && title != "" ){
        os << title << std::endl;

        PrintVector(A.Shape(), "shape", all);
        os << "Distribution: " << A.TensorDist() << std::endl;
        PrintVector(A.Alignments(), "alignments", all);
        PrintVector(A.ModeShifts(), "shifts", all);
        PrintVector(A.LocalPermutation().Entries(), "permutation", all);
        PrintData(A.LockedTensor(), "tensor data", all);
//    }
}

template<typename T>
inline void
PrintArray
( const T* dataBuf, const ObjShape& shape, const ObjShape strides, std::string title=""){
    std::ostream& os = std::cout;
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
( const T* dataBuf, const ObjShape& loopShape, std::string title=""){
    PrintArray(dataBuf, loopShape, Dimensions2Strides(loopShape), title);
}

inline void
PrintPackData
( const PackData& packData, std::string title=""){
    std::ostream& os = std::cout;
    os << title << std::endl;
    PrintVector(packData.loopShape, "  loopShape");
    PrintVector(packData.srcBufStrides, "  srcBufStrides");
    PrintVector(packData.dstBufStrides, "  dstBufStrides");
}

inline void
PrintElemScalData
( const ElemScalData& elemScalData, std::string title="", bool all=false){
    std::ostream& os = std::cout;
    os << title << std::endl;
    PrintVector(elemScalData.loopShape, "  loopShape", all);
    PrintVector(elemScalData.src1Strides, "  src1Strides", all);
    PrintVector(elemScalData.src2Strides, "  src2Strides", all);
    PrintVector(elemScalData.dstStrides, "  dstStrides", all);
}

inline void
PrintHadamardStatCData
( const BlkHadamardStatCInfo& hadamardInfo, std::string title="", bool all=false){
    std::ostream& os = std::cout;
    os << title << std::endl;

    PrintVector(hadamardInfo.partModesACA, "  partModesACA", all);
    PrintVector(hadamardInfo.partModesACC, "  partModesACC", all);
    PrintVector(hadamardInfo.partModesBCB, "  partModesBCB", all);
    PrintVector(hadamardInfo.partModesBCC, "  partModesBCC", all);
    PrintVector(hadamardInfo.permA.Entries(), "  permA", all);
    PrintVector(hadamardInfo.permB.Entries(), "  permB", all);
}

inline void
PrintHadamardScalData
( const HadamardScalData& hadamardInfo, std::string title="", bool all=false){
    std::ostream& os = std::cout;
    os << title << std::endl;

    PrintVector(hadamardInfo.loopShapeAC, "  loopShapeAC", all);
    PrintVector(hadamardInfo.stridesACA, "  stridesACA", all);
    PrintVector(hadamardInfo.stridesACC, "  stridesACC", all);

    PrintVector(hadamardInfo.loopShapeBC, "  loopShapeBC", all);
    PrintVector(hadamardInfo.stridesBCB, "  stridesBCB", all);
    PrintVector(hadamardInfo.stridesBCC, "  stridesBCC", all);

    PrintVector(hadamardInfo.loopShapeABC, "  loopShapeABC", all);
    PrintVector(hadamardInfo.stridesABCA, "  stridesABCA", all);
    PrintVector(hadamardInfo.stridesABCB, "  stridesABCB", all);
    PrintVector(hadamardInfo.stridesABCC, "  stridesABCC", all);
}

} // namespace rote

#endif // ifndef ROTE_IO_PRINT_HPP
