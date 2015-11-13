/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
// NOTE: It is possible to simply include "rote.hpp" instead
#include "rote.hpp"

using namespace rote;

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

void RunReduceTest(const Grid& g, const char* outDist, const char* inDist, const ModeArray& reduceModes){
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

	Unsigned orderB = tenOrder - reduceModes.size();
	ObjShape tenShapeB(orderB);
	for(i = 0; i < orderB; i++)
		tenShapeB[i] = 2;

	printf("what\n");
	DistTensor<double> A(tenShape, inDist, g);
	MakeUniform(A);
	DistTensor<double> B(tenShapeB, outDist, g);

	Print(A, "A");
	B.RedistFrom(A, reduceModes, 1.0, 0.0);
	Print(B, "B");
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
    	gridShape[0] = 1;
    	gridShape[1] = 1;
    	gridShape[2] = 1;
    	gridShape[3] = 1;
    	for(i = 4; i < gridOrder; i++)
    		gridShape[i] = 1;

    	const Grid g(comm, gridShape);
        //RunTest(g, "[(3),(2,0),(4),(1)]", "[(0),(1,3),(4),(2)]");
        ModeArray reduceModes(2);
        reduceModes[0] = 2;
        reduceModes[1] = 3;
//        RunReduceTest(g, "[(0,4),(1,2,3)]", "[(0),(1,3),(4),(2)]", reduceModes);
//        RunReduceTest(g, "[(0,4),(2,3)]", "[(0),(1,3),(4),(2)]", reduceModes);
        RunReduceTest(g, "[(1,4,2),(0,3)]", "[(0),(1,3),(4),(2)]", reduceModes);
    }
    catch( std::exception& e ) { ReportException(e); }

    Finalize();
    //printf("Completed\n");
    return 0;
}
