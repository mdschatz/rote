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
	mstd::cout << "./DistTensor <gridOrder> <gridDim0> <gridDim1> ... <tenOrder> <tenDim0> <tenDim1> ...\n";
	std::cout << "<gridOrder>     : order of the grid ( >0 )\n";
	std::cout << "<gridDimK>  : dimension of mode-K of grid\n";
	std::cout << "<tenOrder>     : order of the tensor ( >0 )\n";
	std::cout << "<tenDimK>   : dimension of mode-K of tensor\n";
}

typedef struct Arguments{
  int gridOrder;
  int tenOrder;
  int nProcs;
  std::vector<int> gridShape;
  std::vector<int> tensorShape;
} Params;

void ProcessInput(const int argc,  char** const argv, Params& args){
	int argCount = 0;
	if(argCount + 1 >= argc){
		std::cerr << "Missing required gridOrder argument\n";
		Usage();
		throw ArgException();
	}

	int gridOrder = atoi(argv[++argCount]);
	args.gridOrder = gridOrder;
	if(gridOrder <= 0){
		std::cerr << "grid order must be greater than 0\n";
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
			std::cerr << "grid dim must be greater than 0\n";
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
	int tenOrder = atoi(argv[++argCount]);
	args.tenOrder = tenOrder;

	if(argCount + tenOrder >= argc){
		std::cerr << "Missing required tensor dimensions\n";
		Usage();
		throw ArgException();
	}

	args.tensorShape.resize(tenOrder);
	for(int i = 0; i < tenOrder; i++){
		int tensorDim = atoi(argv[++argCount]);
		if(tensorDim <= 0){
			std::cerr << "tensor dim must be greater than 0\n";
			Usage();
			throw ArgException();
		}
		args.tensorShape[i] = tensorDim;
	}
}

template<typename T>
void
TestRedist( DistTensor<T>& A )
{
#ifndef RELEASE
    CallStackEntry entry("TestRedist");
#endif
    const int order = A.Order();
    const Grid& g = A.Grid();

    TensorDistribution tdist = A.TensorDist();
    tdist[0].clear();

    DistTensor<T> B(A.Shape(), tdist, A.Indices(), g);
    AllGatherRedist(B, A, 0);
}

template<typename T>
void
TestSet(DistTensor<T>& A)
{
	Int order = A.Order();
	std::vector<Int> index(order);
	Int ptr = 0;
	Int counter = 0;
	bool stop = false;

	while(!stop){
		A.Set(index, 2*counter);

		//Update
		counter++;
		index[ptr]++;
		while(index[ptr] == A.Dimension(ptr)){
			index[ptr] = 0;
			ptr++;
			if(ptr == order){
				stop = true;
				break;
			}else{
				index[ptr]++;
			}
		}
		ptr = 0;
	}
}

template<typename T>
void
DistTensorTest( const std::vector<Int>& shape, const Grid& g )
{
#ifndef RELEASE
    CallStackEntry entry("DistTensorTest");
#endif
    const Int commRank = mpi::CommRank( mpi::COMM_WORLD );
    int order = shape.size();
    TensorDistribution tdist(order);
    std::vector<Int> indices(order);
    for(int i = 0; i < order; i++){
    	ModeDistribution mdist;
    	if(i < order - 1){
    		mdist.resize(1);
    		mdist[0] = i;
    	}else{
    		mdist.resize(2);
    		mdist[0] = i;
    		mdist[1] = i+1;
    	}
    	tdist[i] = mdist;
    	indices[i] = i;
    }

    DistTensor<T> A(shape, tdist, indices, g);
    std::vector<Int> index(shape.size());
    std::fill(index.begin(), index.end(), 0);

    TestSet(A);

    if(commRank == 0){
      printf("Created order-%d Distributed tensor of size ", A.Order());
    
      if(A.Order() > 0){
        printf("%d", A.Dimension(0));
      
        for( int i = 1; i < A.Order(); i++){
          printf(" x %d", A.Dimension(i));
        }
        printf(" and local size %d", A.LocalDimension(0));
        for( int i = 1; i < A.Order(); i++){
          printf(" x %d", A.LocalDimension(i));
        }
      }
      printf("\n");
    }


    TestRedist(A);
    Print(A,"A after redistribute");

    // Communicate from A[MC,MR] 
    //Uniform( A_MC_MR, m, n );
    //Check( A_MC_STAR,   A_MC_MR );
    //TestRedist( A_STAR_MR,   A_MC_MR );
    //TestRedist( A_MR_MC,     A_MC_MR );
    //TestRedist( A_MR_STAR,   A_MC_MR );
    //TestRedist( A_STAR_MC,   A_MC_MR );
    //TestRedist( A_VC_STAR,   A_MC_MR );
    //TestRedist( A_STAR_VC,   A_MC_MR );
    //TestRedist( A_VR_STAR,   A_MC_MR );
    //TestRedist( A_STAR_VR,   A_MC_MR );
    //TestRedist( A_STAR_STAR, A_MC_MR );
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

		if(commRank == 0 && args.nProcs != commSize){
			std::cerr << "program not started with correct number of processes\n";
			Usage();
			throw ArgException();
		}

		if(commRank == 0){
			printf("Creating %d", args.gridShape[0]);
			for(int i = 1; i < args.gridOrder; i++)
				printf(" x %d", args.gridShape[i]);
			printf(" grid\n");
		}

        const Grid g( comm, args.gridOrder, args.gridShape );

        if( commRank == 0 )
        {
            std::cout << "--------------------" << std::endl
                      << "Testing with floats:" << std::endl
                      << "--------------------" << std::endl;
        }
        DistTensorTest<float>( args.tensorShape, g );

        if( commRank == 0 )
        {
            std::cout << "---------------------" << std::endl
                      << "Testing with doubles:" << std::endl
                      << "---------------------" << std::endl;
        }
        DistTensorTest<double>( args.tensorShape, g );

        /*
        if( commRank == 0 )
        {
            std::cout << "--------------------------------------" << std::endl
                      << "Testing with single-precision complex:" << std::endl
                      << "--------------------------------------" << std::endl;
        }
        DistTensorTest<Complex<float> >( args.tensorShape, g );

        if( commRank == 0 )
        {
            std::cout << "--------------------------------------" << std::endl
                      << "Testing with double-precision complex:" << std::endl
                      << "--------------------------------------" << std::endl;
        }
        DistTensorTest<Complex<double> >( args.tensorShape, g );
        */
    }
    catch( std::exception& e ) { ReportException(e); }

    Finalize();
    printf("Completed\n");
    return 0;
}
