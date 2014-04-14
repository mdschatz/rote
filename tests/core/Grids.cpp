/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
// NOTE: It is possible to simply include "tensormental.hpp" instead
#include "tensormental.hpp"
#include "unistd.h"
using namespace tmen;

void Usage(){
	std::cout << "./Grids <order> <gridDim0> <gridDim1> ...\n";
	std::cout << "<order>     : order of the grid ( >0 )\n";
	std::cout << "<gridDimK>  : dimension of mode-K of grid\n";
}

typedef struct Arguments{
  Unsigned order;
  Unsigned size;
  ObjShape gridShape;
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

	if(argc != order + 2){
		std::cerr << "Missing required grid dimensions\n";
		Usage();
		throw ArgException();
	}

	args.size = 1;
	args.gridShape.resize(order);
	for(int i = 0; i < order; i++){
		int gridDim = atoi(argv[i+2]);
		if(gridDim <= 0){
			std::cerr << "grid dim must be greater than 0\n";
			Usage();
			throw ArgException();
		}
		args.size *= gridDim;
		args.gridShape[i] = gridDim;
	}
}

int 
main( int argc, char* argv[] )
{

    Initialize( argc, argv );
    mpi::Comm comm = mpi::COMM_WORLD;
    const Int commSize = mpi::CommSize( comm );
    const Int commRank = mpi::CommRank( comm );    
    try
    {
	Params args;

	ProcessInput(argc, argv, args);

	printf("Comm size: %d rank: %d", commSize, commRank);
	std::cout << "Input processed\n";

	if(args.size != commSize){
		std::cerr << "program not started with correct number of processes\n";
		Usage();
		throw ArgException();
	}

	std::cout << "creating grid\n";
        printf("Args order: %d\n", args.order);
	printf("Args dims: [%d", args.gridShape[0]);
	for(int i = 1; i < args.order; i++)
		printf(", %d", args.gridShape[i]);
	printf("]\n");
	const Grid grid( comm, args.order, args.gridShape );
    }
    catch( std::exception& e ) { ReportException(e); }

    Finalize();
    return 0;
}
