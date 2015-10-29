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
    std::cout << "./RedistCheckTest\n";
}

void RunTest(const Grid& g, const char* outDist, const char* inDist){
	mpi::Comm comm = mpi::COMM_WORLD;
	const Int commRank = mpi::CommRank( comm );
	if(commRank == 0){
		std::cout << outDist << " <-- " << inDist
				  << std::endl;
	}
	Unsigned i;
	Unsigned tenOrder = 4;
	ObjShape tenShape(tenOrder);

	for(i = 0; i < tenOrder; i++)
		tenShape[i] = 2;

	DistTensor<double> A(tenShape, inDist, g);
	MakeUniform(A);
	DistTensor<double> B(tenShape, outDist, g);

	B.RedistFrom(A);

}

int
main( int argc, char* argv[] )
{
    Initialize( argc, argv );
    Unsigned i;
    mpi::Comm comm = mpi::COMM_WORLD;
    const Int commRank = mpi::CommRank( comm );
    printf("My Rank: %d\n", commRank);
    try
    {
    	Unsigned gridOrder = 10;
    	ObjShape gridShape(gridOrder);
    	gridShape[0] = 3;
    	gridShape[1] = 2;
    	gridShape[2] = 2;
    	gridShape[3] = 3;
    	for(i = 4; i < gridOrder; i++)
    		gridShape[i] = 1;

    	const Grid g(comm, gridShape);
        RunTest(g, "[(3),(2,0),(4),(1)]", "[(0),(1,3),(4),(2)]");
    }
    catch( std::exception& e ) { ReportException(e); }

    Finalize();
    //printf("Completed\n");
    return 0;
}
