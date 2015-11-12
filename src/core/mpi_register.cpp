/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#include "tensormental.hpp"

namespace {
rote::mpi::Datatype typeIntInt;
rote::mpi::Datatype typeFloatInt;
rote::mpi::Datatype typeDoubleInt;

rote::mpi::Op maxLocIntOp;
rote::mpi::Op maxLocFloatOp;
rote::mpi::Op maxLocDoubleOp;

rote::mpi::Datatype typeIntIntPair;
rote::mpi::Datatype typeFloatIntPair;
rote::mpi::Datatype typeDoubleIntPair;

rote::mpi::Op maxLocPairIntOp;
rote::mpi::Op maxLocPairFloatOp;
rote::mpi::Op maxLocPairDoubleOp;
} // anonymouse namespace   

namespace rote {
namespace mpi {

template<typename T>
void
MaxLocFunc( void* inVoid, void* outVoid, int* length, Datatype* datatype )
{           
    const ValueInt<T>* inData = static_cast<ValueInt<T>*>(inVoid);
    ValueInt<T>* outData = static_cast<ValueInt<T>*>(outVoid);
    for( int j=0; j<*length; ++j )
    {
        const T inVal = inData[j].value;
        const T outVal = outData[j].value;
        const Int inInd = inData[j].index;
        const Int outInd = outData[j].index; 
        if( inVal > outVal || (inVal == outVal && inInd < outInd) )
            outData[j] = inData[j];
    }
}

template<typename T>
void
MaxLocPairFunc( void* inVoid, void* outVoid, int* length, Datatype* datatype )
{           
    const ValueIntPair<T>* inData = static_cast<ValueIntPair<T>*>(inVoid);
    ValueIntPair<T>* outData = static_cast<ValueIntPair<T>*>(outVoid);
    for( int j=0; j<*length; ++j )
    {
        const T inVal = inData[j].value;
        const T outVal = outData[j].value;
        const Int inInd0 = inData[j].indices[0];
        const Int inInd1 = inData[j].indices[1];
        const Int outInd0 = outData[j].indices[0];
        const Int outInd1 = outData[j].indices[1];
        const bool inIndLess = 
            ( inInd0 < outInd0 || (inInd0 == outInd0 && inInd1 < outInd1) );
        if( inVal > outVal || (inVal == outVal && inIndLess) )
            outData[j] = inData[j];
    }
}

template<>
Datatype& ValueIntType<Int>()
{ return ::typeIntInt; }

template<>
Datatype& ValueIntType<float>()
{ return ::typeFloatInt; }

template<>
Datatype& ValueIntType<double>()
{ return ::typeDoubleInt; }

template<>
Datatype& ValueIntPairType<Int>()
{ return ::typeIntIntPair; }

template<>
Datatype& ValueIntPairType<float>()
{ return ::typeFloatIntPair; }

template<>
Datatype& ValueIntPairType<double>()
{ return ::typeDoubleIntPair; }

template<typename T>
void CreateValueIntType()
{
#ifndef RELEASE
    CallStackEntry cse("CreateValueIntType");
#endif
    Datatype typeList[2];
    typeList[0] = TypeMap<T>();
    typeList[1] = TypeMap<Int>();
    
    int blockLengths[2];
    blockLengths[0] = 1;
    blockLengths[1] = 1; 

    ValueInt<T> v;
    MPI_Aint startAddr, valueAddr, indexAddr;
    MPI_Get_address( &v,       &startAddr );
    MPI_Get_address( &v.value, &valueAddr );
    MPI_Get_address( &v.index, &indexAddr );

    MPI_Aint displs[2];
    displs[0] = valueAddr - startAddr;
    displs[1] = indexAddr - startAddr;

    Datatype& type = ValueIntType<T>();
    MPI_Type_create_struct( 2, blockLengths, displs, typeList, &type );
    MPI_Type_commit( &type );
}
template void CreateValueIntType<Int>();
template void CreateValueIntType<float>();
template void CreateValueIntType<double>();

template<typename T>
void DestroyValueIntType()
{
#ifndef RELEASE
    CallStackEntry cse("DestroyValueIntType");
#endif
    Datatype& type = ValueIntType<T>();
    MPI_Type_free( &type );
}
template void DestroyValueIntType<Int>();
template void DestroyValueIntType<float>();
template void DestroyValueIntType<double>();

template<typename T>
void CreateValueIntPairType()
{
#ifndef RELEASE
    CallStackEntry cse("CreateValueIntPairType");
#endif
    Datatype typeList[2];
    typeList[0] = TypeMap<T>();
    typeList[1] = TypeMap<Int>();
    
    int blockLengths[2];
    blockLengths[0] = 1;
    blockLengths[1] = 2; 

    ValueIntPair<T> v;
    MPI_Aint startAddr, valueAddr, indexAddr;
    MPI_Get_address( &v,        &startAddr );
    MPI_Get_address( &v.value,  &valueAddr );
    MPI_Get_address( v.indices, &indexAddr );

    MPI_Aint displs[2];
    displs[0] = valueAddr - startAddr;
    displs[1] = indexAddr - startAddr;

    Datatype& type = ValueIntPairType<T>();
    MPI_Type_create_struct( 2, blockLengths, displs, typeList, &type );
    MPI_Type_commit( &type );
}
template void CreateValueIntPairType<Int>();
template void CreateValueIntPairType<float>();
template void CreateValueIntPairType<double>();

template<typename T>
void DestroyValueIntPairType()
{
#ifndef RELEASE
    CallStackEntry cse("DestroyValueIntPairType");
#endif
    Datatype& type = ValueIntPairType<T>();
    MPI_Type_free( &type );
}
template void DestroyValueIntPairType<Int>();
template void DestroyValueIntPairType<float>();
template void DestroyValueIntPairType<double>();

template<>
void CreateMaxLocOp<Int>()
{
#ifndef RELEASE
    CallStackEntry cse("CreateMaxLocOp<Int>");
#endif
    OpCreate( (UserFunction*)MaxLocFunc<Int>, true, ::maxLocIntOp );
}

template<>
void CreateMaxLocOp<float>()
{
#ifndef RELEASE
    CallStackEntry cse("CreateMaxLocOp<float>");
#endif
    OpCreate( (UserFunction*)MaxLocFunc<float>, true, ::maxLocFloatOp );
}

template<>
void CreateMaxLocOp<double>()
{
#ifndef RELEASE
    CallStackEntry cse("CreateMaxLocOp<double>");
#endif
    OpCreate( (UserFunction*)MaxLocFunc<double>, true, ::maxLocDoubleOp );
}

template<>
void CreateMaxLocPairOp<Int>()
{
#ifndef RELEASE
    CallStackEntry cse("CreateMaxLocPairOp<Int>");
#endif
    OpCreate( (UserFunction*)MaxLocPairFunc<Int>, true, ::maxLocPairIntOp );
}

template<>
void CreateMaxLocPairOp<float>()
{
#ifndef RELEASE
    CallStackEntry cse("CreateMaxLocPairOp<float>");
#endif
    OpCreate( (UserFunction*)MaxLocPairFunc<float>, true, ::maxLocPairFloatOp );
}

template<>
void CreateMaxLocPairOp<double>()
{
#ifndef RELEASE
    CallStackEntry cse("CreateMaxLocPairOp<double>");
#endif
    OpCreate
    ( (UserFunction*)MaxLocPairFunc<double>, true, ::maxLocPairDoubleOp );
}

template<>
void DestroyMaxLocOp<Int>()
{
#ifndef RELEASE
    CallStackEntry cse("DestroyMaxLocOp<Int>");
#endif
    OpFree( ::maxLocIntOp );
}

template<>
void DestroyMaxLocOp<float>()
{
#ifndef RELEASE
    CallStackEntry cse("DestroyMaxLocOp<float>");
#endif
    OpFree( ::maxLocFloatOp );
}

template<>
void DestroyMaxLocOp<double>()
{
#ifndef RELEASE
    CallStackEntry cse("DestroyMaxLocOp<double>");
#endif
    OpFree( ::maxLocDoubleOp );
}

template<>
void DestroyMaxLocPairOp<Int>()
{
#ifndef RELEASE
    CallStackEntry cse("DestroyMaxLocPairOp<Int>");
#endif
    OpFree( ::maxLocPairIntOp );
}

template<>
void DestroyMaxLocPairOp<float>()
{
#ifndef RELEASE
    CallStackEntry cse("DestroyMaxLocPairOp<float>");
#endif
    OpFree( ::maxLocPairFloatOp );
}

template<>
void DestroyMaxLocPairOp<double>()
{
#ifndef RELEASE
    CallStackEntry cse("DestroyMaxLocPairOp<double>");
#endif
    OpFree( ::maxLocPairDoubleOp );
}

template<>
Op MaxLocOp<Int>()
{
#ifndef RELEASE
    CallStackEntry cse("MaxLocOp<Int>");
#endif
    return ::maxLocIntOp;
}

template<>
Op MaxLocOp<float>()
{
#ifndef RELEASE
    CallStackEntry cse("MaxLocOp<float>");
#endif
    return ::maxLocFloatOp;
}

template<>
Op MaxLocOp<double>()
{
#ifndef RELEASE
    CallStackEntry cse("MaxLocOp<double>");
#endif
    return ::maxLocDoubleOp;
}

template<>
Op MaxLocPairOp<Int>()
{
#ifndef RELEASE
    CallStackEntry cse("MaxLocPairOp<Int>");
#endif
    return ::maxLocPairIntOp;
}

template<>
Op MaxLocPairOp<float>()
{
#ifndef RELEASE
    CallStackEntry cse("MaxLocPairOp<float>");
#endif
    return ::maxLocPairFloatOp;
}

template<>
Op MaxLocPairOp<double>()
{
#ifndef RELEASE
    CallStackEntry cse("MaxLocPairOp<double>");
#endif
    return ::maxLocPairDoubleOp;
}

template void
MaxLocFunc<Int>( void* in, void* out, int* length, Datatype* datatype );
template void
MaxLocFunc<float>( void* in, void* out, int* length, Datatype* datatype );
template void
MaxLocFunc<double>( void* in, void* out, int* length, Datatype* datatype );

template void
MaxLocPairFunc<Int>( void* in, void* out, int* length, Datatype* datatype );
template void
MaxLocPairFunc<float>( void* in, void* out, int* length, Datatype* datatype );
template void
MaxLocPairFunc<double>( void* in, void* out, int* length, Datatype* datatype );

} // namespace mpi
} // namespace tmen
