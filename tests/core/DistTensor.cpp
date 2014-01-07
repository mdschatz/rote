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
	std::cout << "./DistTensor <order> <gridDim0> <gridDim1> ... <tenDim0> <tenDim1> ...\n";
	std::cout << "<order>     : order of the grid ( >0 )\n";
	std::cout << "<gridDimK>  : dimension of mode-K of grid\n";
	std::cout << "<tenDimK>   : dimension of mode-K of tensor\n";
}

typedef struct Arguments{
  int order;
  int size;
  std::vector<int> gridDims;
  std::vector<int> tensorDims;
} Params;

void ProcessInput(const int argc,  char** const argv, Params& args){
	if(argc < 2){
		std::cerr << "Missing required order argument\n";
		Usage();
		throw ArgException();
	}

	int order = atoi(argv[1]);
	args.order = order;
	if(order <= 0){
		std::cerr << "grid order must be greater than 0\n";
		Usage();
		throw ArgException();
	}

	if(argc < order + 2){
		std::cerr << "Missing required grid dimensions\n";
		Usage();
		throw ArgException();
	}

	args.size = 1;
	args.gridDims.resize(order);
	for(int i = 0; i < order; i++){
		int gridDim = atoi(argv[i+2]);
		if(gridDim <= 0){
			std::cerr << "grid dim must be greater than 0\n";
			Usage();
			throw ArgException();
		}
		args.size *= gridDim;
		args.gridDims[i] = gridDim;
	}

	if(argc != order + order + 2){
		std::cerr << "Missing required tensor dimensions\n";
		Usage();
		throw ArgException();
	}

	args.tensorDims.resize(order);
	for(int i = 0; i < order; i++){
		int tensorDim = atoi(argv[i + order + 2]);
		if(tensorDim <= 0){
			std::cerr << "tensor dim must be greater than 0\n";
			Usage();
			throw ArgException();
		}
		args.tensorDims[i] = tensorDim;
	}
}

template<typename T>
void
Check( DistTensor<T>& A )
{
/*
#ifndef RELEASE
    CallStackEntry entry("Check");
#endif
    const Grid& g = A.Grid();

    const Int commRank = g.Rank();
    const Int height = B.Height();
    const Int width = B.Width();
    DistTensor<T> A_STAR_STAR(g);
    DistTensor<T> B_STAR_STAR(g);

    if( commRank == 0 )
    {
        std::cout << "Testing [" << (AColDist) << ","
                                 << (ARowDist) << "]"
                  << " <- ["     << (BColDist) << ","
                                 << (BRowDist) << "]...";
        std::cout.flush();
    }
    A = B;

    A_STAR_STAR = A;
    B_STAR_STAR = B;

    Int myErrorFlag = 0;
    for( Int j=0; j<width; ++j )
    {
        for( Int i=0; i<height; ++i )
        {
            if( A_STAR_STAR.GetLocal(i,j) != B_STAR_STAR.GetLocal(i,j) )
            {
                myErrorFlag = 1;
                break;
            }
        }
        if( myErrorFlag != 0 )
            break;
    }

    Int summedErrorFlag;
    mpi::AllReduce( &myErrorFlag, &summedErrorFlag, 1, mpi::SUM, g.Comm() );

    if( summedErrorFlag == 0 )
    {
        if( commRank == 0 )
            std::cout << "PASSED" << std::endl;
    }
    else
        LogicError("Redistribution failed");
*/
}

template<typename T>
void
DistTensorTest( const std::vector<Int>& dims, const Grid& g )
{
#ifndef RELEASE
    CallStackEntry entry("DistTensorTest");
#endif
    DistTensor<T> A(dims, g);

    // Communicate from A[MC,MR] 
    //Uniform( A_MC_MR, m, n );
    //Check( A_MC_STAR,   A_MC_MR );
    //Check( A_STAR_MR,   A_MC_MR );
    //Check( A_MR_MC,     A_MC_MR );
    //Check( A_MR_STAR,   A_MC_MR );
    //Check( A_STAR_MC,   A_MC_MR );
    //Check( A_VC_STAR,   A_MC_MR );
    //Check( A_STAR_VC,   A_MC_MR );
    //Check( A_VR_STAR,   A_MC_MR );
    //Check( A_STAR_VR,   A_MC_MR );
    //Check( A_STAR_STAR, A_MC_MR );
}

int 
main( int argc, char* argv[] )
{
    Initialize( argc, argv );
    mpi::Comm comm = mpi::COMM_WORLD;
    const Int commRank = mpi::CommRank( comm );
    const Int commSize = mpi::CommSize( comm );

    try
    {
	Params args;

	ProcessInput(argc, argv, args);

	if(commRank == 0 && args.size != commSize){
		std::cerr << "program not started with correct number of processes\n";
		Usage();
		throw ArgException();
	}

	if(commRank == 0){
		printf("Creating %d", args.gridDims[0]);
		for(int i = 1; i < args.order; i++)
			printf(" x %d", args.gridDims[i]);
		printf(" grid\n");
	}

        const Grid g( comm, args.order, args.gridDims );

        if( commRank == 0 )
        {
            std::cout << "--------------------\n"
                      << "Testing with floats:\n"
                      << "--------------------" << std::endl;
        }
        DistTensorTest<float>( args.tensorDims, g );

        if( commRank == 0 )
        {
            std::cout << "---------------------\n"
                      << "Testing with doubles:\n"
                      << "---------------------" << std::endl;
        }
        DistTensorTest<double>( args.tensorDims, g );

        if( commRank == 0 )
        {
            std::cout << "--------------------------------------\n"
                      << "Testing with single-precision complex:\n"
                      << "--------------------------------------" << std::endl;
        }
        DistTensorTest<Complex<float> >( args.tensorDims, g );

        if( commRank == 0 )
        {
            std::cout << "--------------------------------------\n"
                      << "Testing with double-precision complex:\n"
                      << "--------------------------------------" << std::endl;
        }
        DistTensorTest<Complex<double> >( args.tensorDims, g );
    }
    catch( std::exception& e ) { ReportException(e); }

    Finalize();
    return 0;
}
