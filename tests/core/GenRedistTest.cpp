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

typedef struct ArgumentsGenRedist{
  Unsigned nProcs;
  ObjShape gridShape;
  Unsigned mdim;
  Unsigned kdim;
  Unsigned ndim;
} ParamsGenRedist;

void Usage(){
    std::cout << "./GenRedistTest <gridDim0> <gridDim1> <gridDim2> <gridDim3> <\"m\"-dim> <\"k\"-dim> <\"n\"-dim>\n";
    std::cout << "<gridDimK>   : dimension of mode-K of grid\n";
    std::cout << "<\"m\"-dim>  : dimension of \"m\" modes of tensor\n";
    std::cout << "<\"k\"-dim>  : dimension of \"k\" modes of tensor\n";
    std::cout << "<\"n\"-dim>  : dimension of \"n\" modes of tensor\n";
}

void ProcessInput(Unsigned argc,  char** const argv, Params& args){
    Unsigned i;
    Unsigned argCount = 0;

    Unsigned gridOrder = 4;

    if(argCount + gridOrder >= argc){
        std::cerr << "Missing required grid dimensions\n";
        Usage();
        throw ArgException();
    }

    args.gridShape.resize(gridOrder);
    for(Unsigned i = 0; i < gridOrder; i++){
        int gridDim = atoi(argv[++argCount]);
        if(gridDim <= 0){
            std::cerr << "Grid dim must be greater than 0\n";
            Usage();
            throw ArgException();
        }
        args.gridShape[i] = gridDim;
    }
    args.nProcs = tmen::prod(args.gridShape);

    if(argCount + 3 >= argc){
        std::cerr << "Missing required tensor dimensions\n";
        Usage();
        throw ArgException();
    }

    args.mdim = atoi(argv[++argCount]);
    args.kdim = atoi(argv[++argCount]);
    args.ndim = atoi(argv[++argCount]);
}

template<typename T>
void
PerformTest( DistTensor<T>& A, const Params& args, const Grid& g ){

    Unsigned i;
    Unsigned order = A.Order();

    const ObjShape shape = A.Shape();

    Permutation defaultPerm = DefaultPermutation(order);

    DistTensor<T> AT(order, g), AB(order, g), A0(order, g), A1(order, g), A2(order, g);

    for(i = 0; i < order; i++){
        Mode mode = i;
        printf("Iterating over mode: %d\n", mode);

        A1.AlignWith(A);
        PartitionDown(A, AT, AB, mode, 0);

        while(AT.Dimension(mode) < A.Dimension(mode)){
            RepartitionDown(AT,   A0,
                                  A1,
                           /**/  /**/
                            AB,   A2, mode, 1);
            /////////////////////////////////
            A1.SetLocalPermutation(defaultPerm);
            Print(A1, "A1before");
            DistTensorTest<T>(A1, args, g);
            /////////////////////////////////
            SlidePartitionDown(AT,  A0,
                                    A1,
                              /**/ /**/
                               AB,  A2, mode);
        }
    }
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

        if(commRank == 0 && args.nProcs != ((Unsigned)commSize)){
            std::cerr << "program not started with correct number of processes\n";
            Usage();
            throw ArgException();
        }

        if(commRank == 0){
        	PrintVector(args.gridShape, "Creating grid");
        	PrintVector(args.tensorShape, "Creating tensor");
        }

        const Grid g( comm, args.gridShape );

        PrintVector(g.Loc(), "gridLoc");
        if( commRank == 0 )
        {
            std::cout << "------------------" << std::endl
                      << "Testing with ints:" << std::endl
                      << "------------------" << std::endl;
        }

        std::vector<RedistType> redistsToTest = {AG, A2A, Local, RS, RTO, AR, GTO, BCast, Scatter, Perm};
//        std::vector<RedistType> redistsToTest = {Local};
        DistTensorTest<int>(redistsToTest, args, g);
    }
    catch( std::exception& e ) { ReportException(e); }

    Finalize();
    //printf("Completed\n");
    return 0;
}
