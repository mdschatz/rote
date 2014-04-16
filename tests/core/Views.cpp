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
    std::cout << "./DistTensor <gridOrder> <gridDim0> <gridDim1> ... <tenOrder> <tenDim0> <tenDim1> ... \"<tensorDist>\"\n";
    std::cout << "<gridOrder>  : order of the grid ( >0 )\n";
    std::cout << "<gridDimK>   : dimension of mode-K of grid\n";
    std::cout << "<tenOrder>   : order of the tensor ( >0 )\n";
    std::cout << "<tenDimK>    : dimension of mode-K of tensor\n";
    std::cout << "<tensorDist> : distribution of tensor(Must be in quotes)\n";
}

typedef struct Arguments{
  Unsigned gridOrder;
  Unsigned tenOrder;
  Unsigned nProcs;
  ObjShape gridShape;
  ObjShape tensorShape;
  TensorDistribution tensorDist;
} Params;

void ProcessInput(int argc,  char** const argv, Params& args){
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
    for(int i = 0; i < gridOrder; i++){
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
        std::cerr << "Missing required tenOrder argument\n";
        Usage();
        throw ArgException();
    }
    Unsigned tenOrder = atoi(argv[++argCount]);
    args.tenOrder = tenOrder;

    if(argCount + tenOrder >= argc){
        std::cerr << "Missing required tensor dimensions\n";
        Usage();
        throw ArgException();
    }

    args.tensorShape.resize(tenOrder);
    for(i = 0; i < tenOrder; i++){
        Unsigned tensorDim = atoi(argv[++argCount]);
        if(tensorDim == 0){
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


    if(args.tensorDist.size() != args.tensorShape.size()){
        std::cerr << "Tensor distribution must be of same order as tensor\n";
        Usage();
        throw ArgException();
    }
}

template<typename T>
void
PrintLocalView(const Tensor<T>& A){
    Unsigned i;
    const Unsigned order = A.Order();
    printf("      Local tensor info:\n");
    printf("        shape:");
    if(order > 0)
        printf(" %d", A.Dimension(0));
    for(i = 1; i < order; i++)
        printf(" x %d", A.Dimension(i));
    printf("\n");

    Print(A, "        data: ");

}

template<typename T>
void
PrintView(const char* msg, const DistTensor<T>& A){
    Unsigned i;
    const Unsigned order = A.Order();
    printf("Info for: %s\n", msg);
    printf("    shape:");
    if(order > 0)
        printf(" %d", A.Dimension(0));
    for(i = 1; i < order; i++)
        printf(" x %d", A.Dimension(i));
    printf("\n");

    printf("    alignments:");
    if(order > 0)
        printf(" %d", A.ModeAlignment(0));
    for(i = 1; i < order; i++)
        printf(" x %d", A.ModeAlignment(i));
    printf("\n");

    PrintLocalView(A.LockedTensor());
}

template<typename T>
void
TestConstViews(DistTensor<T>& A){
#ifndef RELEASE
    CallStackEntry entry("TestConstViews");
#endif

}

template<typename T>
void
TestNonConstViews(DistTensor<T>& A){
#ifndef RELEASE
    CallStackEntry entry("TestNonConstViews");
#endif
    const tmen::Grid& g = A.Grid();
    Unsigned i;
    const Unsigned order = A.Order();
    Location start(order);
    std::fill(start.begin(), start.end(), 0);
    const IndexArray indices = A.Indices();
    const ObjShape shape = A.Shape();

    DistTensor<T> AT(order, g), AB(order, g), A0(order, g), A1(order, g), A2(order, g);

    for(i = 0; i < order; i++){
        Index index = indices[i];
        Mode mode = A.ModeOfIndex(index);
        printf("Iterating over index: %d\n", index);

        PartitionDown(A, AT, AB, index, 0);

        Unsigned count = 0;
        while(AT.Dimension(mode) < A.Dimension(mode)){
            printf("  iteration: %d\n", count);
            RepartitionDown(AT,   A0,
                                  A1,
                           /**/  /**/
                            AB,   A2, index, 1);
            /////////////////////////////////
            PrintView("A0", A0);
            PrintView("A1", A1);
            PrintView("A2", A2);
            /////////////////////////////////
            SlidePartitionDown(AT,  A0,
                                    A1,
                              /**/ /**/
                               AB,  A2, index);
        }

    }
}

template<typename T>
void
Set(DistTensor<T>& A)
{
    Unsigned order = A.Order();
    Location loc(order);
    std::fill(loc.begin(), loc.end(), 0);
    Unsigned ptr = 0;
    Unsigned counter = 0;
    bool stop = false;

    while(!stop){
        A.Set(loc, counter);

        //Update
        counter++;
        loc[ptr]++;
        while(loc[ptr] == A.Dimension(ptr)){
            loc[ptr] = 0;
            ptr++;
            if(ptr == order){
                stop = true;
                break;
            }else{
                loc[ptr]++;
            }
        }
        ptr = 0;
    }
}

template<typename T>
void
DistTensorTest( const Params& args, const Grid& g )
{
#ifndef RELEASE
    CallStackEntry entry("DistTensorTest");
#endif
    Unsigned i;
    const Int commRank = mpi::CommRank( mpi::COMM_WORLD );
    const Unsigned order = args.tensorShape.size();
    IndexArray indices(order);
    for(i = 0 ; i < indices.size(); i++)
        indices[i] = i;

    DistTensor<T> A(args.tensorShape, args.tensorDist, indices, g);

    Set(A);

    if(commRank == 0){
        printf("Performing Const tests\n");
    }

    TestConstViews(A);

    if(commRank == 0){
        printf("Performing Non-Const tests\n");
    }

    TestNonConstViews(A);
}

int 
main( int argc, char* argv[] )
{
    Initialize( argc, argv );
    Unsigned i;
    mpi::Comm comm = mpi::COMM_WORLD;
    const Int commRank = mpi::CommRank( comm );
    const Int commSize = mpi::CommSize( comm );
    printf("My Rank: %d\n", commRank);
    try
    {
        Params args;

        ProcessInput(argc, argv, args);

        if(commRank == 0 && args.nProcs != commSize){
            std::cerr << "program not started with correct number of processes\n";
            Usage();
            throw ArgException();
        }

        if(commRank == 0){
            printf("Creating %d", args.gridShape[0]);
            for(i = 1; i < args.gridOrder; i++)
                printf(" x %d", args.gridShape[i]);
            printf(" grid\n");

            printf("Creating [%d", args.tensorShape[0]);
            for(i = 1; i < args.tenOrder; i++)
                printf(", %d", args.tensorShape[i]);
            printf("] tensor\n");
        }

        const Grid g( comm, args.gridOrder, args.gridShape );

        DistTensorTest<int>( args, g );

    }
    catch( std::exception& e ) { ReportException(e); }

    Finalize();
    //printf("Completed\n");
    return 0;
}
