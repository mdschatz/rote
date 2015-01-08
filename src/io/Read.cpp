/*
   Copyright (c) 2009-2015, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#include "tensormental.hpp"

#include "./Read/Ascii.hpp"
#include "./Read/AsciiMatlab.hpp"
#include "./Read/Binary.hpp"
#include "./Read/BinaryFlat.hpp"

namespace tmen {

template<typename T>
void Read( Tensor<T>& A, const std::string filename, FileFormat format )
{
    if( format == AUTO )
        format = DetectFormat( filename );

    switch( format )
    {
    case ASCII:
        read::Ascii( A, filename );
        break;
//    case ASCII_MATLAB:
//        read::AsciiMatlab( A, filename );
//        break;
//    case BINARY:
//        read::Binary( A, filename );
//        break;
//    case BINARY_FLAT:
//        read::BinaryFlat( A, A.Shape, filename );
//        break;
    default:
        LogicError("Format unsupported for reading");
    }
}

template<typename T>
void Read
( DistTensor<T>& A, const std::string filename, FileFormat format,
  bool sequential )
{
    if( format == AUTO )
        format = DetectFormat( filename ); 

    //Everyone accesses data
    if(!sequential)
    {
        switch( format )
        {
        case ASCII:
            read::Ascii( A, filename );
            break;
//        case ASCII_MATLAB:
//            read::AsciiMatlab( A, filename );
//            break;
//        case BINARY:
//            read::Binary( A, filename );
//            break;
//        case BINARY_FLAT:
//            read::BinaryFlat( A, A.Height(), A.Width(), filename );
            break;
        default:
            LogicError("Unsupported distributed read format");
        }
    }
    //Only root process accesses data
    else
    {
        switch( format )
        {
        case ASCII:
            read::AsciiSeq( A, filename );
            break;
//        case ASCII_MATLAB:
//            read::AsciiMatlabSeq( A, filename );
//            break;
//        case BINARY:
//            read::BinarySeq( A, filename );
//            break;
//        case BINARY_FLAT:
//            read::BinaryFlatSeq( A, A.Height(), A.Width(), filename );
//            break;
        default:
            LogicError("Unsupported distributed read format");
        }
    }

}

#define PROTO(T) \
  template void Read \
  ( Tensor<T>& A, const std::string filename, FileFormat format ); \
  template void Read \
  ( DistTensor<T>& A, const std::string filename, \
    FileFormat format, bool sequential );

#define FULL(T) \
  PROTO(T);

FULL(Int);
#ifndef DISABLE_FLOAT
FULL(float);
#endif
FULL(double);


} // namespace tmen
