/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
// NOTE: It is possible to simply include "tensormental.hpp" instead
#include "tensormental.hpp"
using namespace tmen;

void Usage(){
    std::cout << "./Level1 <gridOrder> <gridDim0> <gridDim1> ... <ten1Order> <ten1Dim0> <ten1Dim1> ... \"<tensorDist>\"\n";
    std::cout << "<gridOrder>  : order of the grid ( >0 )\n";
    std::cout << "<gridDimK>   : dimension of mode-K of grid\n";
    std::cout << "<tenIOrder>  : order of the tensor I ( >0 )\n";
    std::cout << "<tenIDimK>   : dimension of mode-K of tensor I\n";
}

typedef struct Arguments{
  Unsigned gridOrder;
  Unsigned nProcs;
  ObjShape gridShape;
  Unsigned tensorOrder;
  ObjShape tensorShape;
  TensorDistribution tensorDist;
} Params;

void ProcessInput(Unsigned argc,  char** const argv, Params& args){
    Unsigned i;
    Unsigned argCount = 0;
    if(argCount + 1 >= argc){
        std::cerr << "Missing required gridOrder argument\n";
        Usage();
        throw ArgException();
    }

    Unsigned gridOrder = atoi(argv[++argCount]);
    args.gridOrder = gridOrder;
    if(gridOrder <= 0){
        std::cerr << "Grid order must be greater than 0\n";
        Usage();
        throw ArgException();
    }

    if(argCount + gridOrder >= argc){
        std::cerr << "Missing required grid dimensions\n";
        Usage();
        throw ArgException();
    }

    args.gridShape.resize(gridOrder);
    for(i = 0; i < gridOrder; i++){
        int gridDim = atoi(argv[++argCount]);
        if(gridDim <= 0){
            std::cerr << "Grid dim must be greater than 0\n";
            Usage();
            throw ArgException();
        }
        args.gridShape[i] = gridDim;
    }
    args.nProcs = tmen::prod(args.gridShape);

    if(argCount + 1 >= argc){
        std::cerr << "Missing required ten1Order argument\n";
        Usage();
        throw ArgException();
    }
    Unsigned ten1Order = atoi(argv[++argCount]);
    args.tensorOrder = ten1Order;

    if(argCount + ten1Order >= argc){
        std::cerr << "Missing required tensor dimensions\n";
        Usage();
        throw ArgException();
    }

    args.tensorShape.resize(ten1Order);
    for(i = 0; i < ten1Order; i++){
        int tensorDim = atoi(argv[++argCount]);
        if(tensorDim <= 0){
            std::cerr << "Tensor dimension must be greater than 0\n";
            Usage();
            throw ArgException();
        }
        args.tensorShape[i] = tensorDim;
    }

    if(argCount + 1 >= argc){
        std::cerr << "Missing tensor distribution argument\n";
        Usage();
        throw ArgException();
    }

    std::string tensorDist(argv[++argCount]);
    args.tensorDist = tmen::StringToTensorDist(tensorDist);


    if(args.tensorDist.size() != args.tensorShape.size() + 1){
        std::cerr << "Tensor distribution must be of same order as tensor\n";
        Usage();
        throw ArgException();
    }
}

template<typename T>
void
Set(Tensor<T>& A)
{
    MakeUniform(A);
}

template<typename T>
void
TestElemScal(const DistTensor<T>& A)
{
#ifndef RELEASE
    CallStackEntry entry("TestElemScal");
#endif
    DistTensor<T> B(A.Shape(), A.TensorDist(), A.Grid());
    MakeUniform(B);

    DistTensor<T> C(A.Shape(), A.TensorDist(), A.Grid());
    MakeZeros(C);

    ElemScal(A, B, C);
    Print(A, "\n\nScaling:");
    Print(B, "Scaled by: ");
    Print(C, "Scale result");
}

template<typename T>
void
TestYAxpPx(const DistTensor<T>& A)
{
#ifndef RELEASE
    CallStackEntry entry("TestyAxpPx");
#endif
    Unsigned i;
    Permutation perm(A.Order());

    DistTensor<T> B(A.Shape(), A.TensorDist(), A.Grid());

    for(i = 0; i < B.Order(); i++)
        perm[i] = i;

    std::sort(perm.begin(), perm.end());
    do{

        MakeZeros(B);
        YAxpPx(T(2), A, T(1), A, perm, B);
        Print(A, "\n\nAxpPxing: ");
        PrintVector(perm, "under permutation");
        Print(B, "Result:");
    } while(std::next_permutation(perm.begin(), perm.end()));
}

template<typename T>
void
TestZAxpBy(const DistTensor<T>& A)
{
#ifndef RELEASE
    CallStackEntry entry("TestZAxpBy");
#endif
    Unsigned i;
    Permutation perm(A.Order());

    DistTensor<T> B(A.Shape(), A.TensorDist(), A.Grid());
    MakeUniform(B);

    DistTensor<T> Z(A.Shape(), A.TensorDist(), A.Grid());

    ZAxpBy(T(2), A, T(3), B, Z);
}

template<typename T>
void
TestAxpy(const DistTensor<T>& A)
{
#ifndef RELEASE
    CallStackEntry entry("TestAxpy");
#endif
    Permutation perm(A.Order());
    DistTensor<T> B(A.Shape(), A.TensorDist(), A.Grid());
    MakeUniform(B);

    Print(A, "\n\nAxpying:");
    Print(B, "to:");
    Axpy(T(2), A, B);
    Print(B, "result:");
}

int 
main( int argc, char* argv[] )
{
    Initialize( argc, argv );
    mpi::Comm comm = mpi::COMM_WORLD;
    const Int commRank = mpi::CommRank( comm );
    const Int commSize = mpi::CommSize( comm );
    printf("My Rank: %d\n", commRank);
    try
    {
        Params args;

        ProcessInput(argc, argv, args);

        if(commRank == 0 && args.nProcs != ((Unsigned)commSize)){
            std::cerr << "program not started with correct number of processes\n";
            Usage();
            throw ArgException();
        }

        const Grid g( comm, args.gridShape );

        if( commRank == 0 )
        {
            std::cout << "------------------" << std::endl
                      << "Testing with ints:" << std::endl
                      << "------------------" << std::endl;
        }
        DistTensor<double> A(args.tensorShape, args.tensorDist, g);
        MakeUniform(A);
//        TestElemScal(A);
        TestYAxpPx(A);
        TestZAxpBy(A);
//        TestAxpy(A);

//        if( commRank == 0 )
//        {
//            std::cout << "--------------------" << std::endl
//                      << "Testing with floats:" << std::endl
//                      << "--------------------" << std::endl;
//        }
//        DistTensorTest<float>( args, g );
//
//        if( commRank == 0 )
//        {
//            std::cout << "---------------------" << std::endl
//                      << "Testing with doubles:" << std::endl
//                      << "---------------------" << std::endl;
//        }
//        DistTensorTest<double>( args, g );

    }
    catch( std::exception& e ) { ReportException(e); }

    Finalize();
    //printf("Completed\n");
    return 0;
}
