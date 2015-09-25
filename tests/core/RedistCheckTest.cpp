/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
// NOTE: It is possible to simply include "tensormental.hpp" instead
#include "tensormental.hpp"
#include "tensormental/tests/LGRedist.hpp"
#include "tensormental/tests/GTOGRedist.hpp"
#include "tensormental/tests/AGGRedist.hpp"
#include "tensormental/tests/RTOGRedist.hpp"
#include "tensormental/tests/RSGRedist.hpp"
#include "tensormental/tests/PRedist.hpp"
#include "tensormental/tests/A2ARedist.hpp"

using namespace tmen;

void Usage(){
    std::cout << "./RedistCheckTest\n";
}

void RunTest(const Grid& g, const char* outDist, const char* inDist, const RedistType& redist, ModeArray& modes, bool shouldFail){
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
		tenShape[i] = 4;

	try{
		DistTensor<double> A(tenShape, inDist, g);
		DistTensor<double> B(tenShape, outDist, g);

		switch(redist){
			case AG:      B.AllGatherRedistFrom(A, modes); break;
			case A2A:     B.AllToAllRedistFrom(A, modes); break;
			case Local:   B.LocalRedistFrom(A); break;
			case GTO:     B.GatherToOneRedistFrom(A, modes); break;
//			case BCast:   B.BroadcastRedistFrom(A, modes); break;
//			case Scatter: B.ScatterRedistFrom(A, modes); break;
			case Perm:    B.PermutationRedistFrom(A, modes); break;
			//Following are ignored since are equivalent to non-reduce cases
//			case RS:      B.ReduceScatterRedistFrom(A, modes); break;
//			case RTO:     B.ReduceToOneRedistFrom(A, modes); break;
//			case AR:      B.AllReduceRedistFrom(A, modes); break;
		}
		if(commRank == 0)
			std::cout << (shouldFail ? "Failure" : "Success") << std::endl;
	}catch(std::exception& e) {
								if(commRank == 0)
									std::cout << (shouldFail ? "Success" : "Failure") << std::endl;
								if(!shouldFail)
									ReportException(e);
							  }
}

void CheckAGRedist(const Grid& g){
	mpi::Comm comm = mpi::COMM_WORLD;
	const Int commRank = mpi::CommRank( comm );
	if(commRank == 0)
		printf("Performing AllGather tests:\n");
	RedistType redist = AG;
	ModeArray modes(2);
	modes[0] = 6;
	modes[1] = 8;

	//Check that output is prefix of input and non-distributed modes match
	RunTest(g, "[(0,1),(2,3,9),(4),()]|(5,7)", "[(0,1),(2,3,9),(4),(6,8)]|(7,5)", redist, modes, false);

	//Check that failure occurs when non-dist modes are different
	RunTest(g, "[(0,1),(2,3,9),(4),()]|(5,7,8)", "[(0,1),(2,3,9),(4),(6,8)]|(7,5)", redist, modes, true);

	//Check that failure occurs when add suffix to mode dist
	RunTest(g, "[(0,1),(2,3,9),(4),(6,8)]|(5,7)", "[(0,1),(2,3,9),(4),()]|(7,5)", redist, modes, true);

	//Check that AG fails correctly (not suffix)
	RunTest(g, "[(0,1),(2,3,9),(4),()]|(5,7)", "[(0,1),(2,3,9),(8,4),(6)]|(7,5)", redist, modes, true);

	if(commRank == 0)
		printf("\n");
}

void CheckA2ARedist(const Grid& g){
	mpi::Comm comm = mpi::COMM_WORLD;
	const Int commRank = mpi::CommRank( comm );
	if(commRank == 0)
		printf("Performing AllToAll tests:\n");
	RedistType redist = A2A;
	ModeArray modes(2);
	modes[0] = 6;
	modes[1] = 8;

	//Check that A2A partitioned modes work correctly (from one mode dists to multiple)
	RunTest(g, "[(0,1),(2,3,9,8),(4,6),()]|(5,7)", "[(0,1),(2,3,9),(4),(6,8)]|(7,5)", redist, modes, false);

	//Check that A2A partitioned modes work correctly (from multiple mode dists to one)
	RunTest(g, "[(0,1),(2,3,8,6),(4),(9)]|(5,7)", "[(0,1),(2,3),(4,6),(9,8)]|(7,5)", redist, modes, false);

	//Check "can't form partition" case fails
	RunTest(g, "[(0,1),(2,3),(4,8),(9,6)]|(5,7)", "[(0,1),(2,3),(4,6),(9,8)]|(7,5)", redist, modes, true);

	//Check non-dist mode changes case fails
	RunTest(g, "[(0,1),(2,3),(4),(9)]|(5,7,6,8)", "[(0,1),(2,3),(4,6),(9,8)]|(7,5)", redist, modes, true);

	//Check that A2A fails correctly (not suffix)
	RunTest(g, "[(0,1),(2,3,8,6),(4),(9)]|(5,7)", "[(0,1),(2,3),(4,6),(8,9)]|(7,5)", redist, modes, true);

	if(commRank == 0)
		printf("\n");
}

void CheckLocalRedist(const Grid& g){
	mpi::Comm comm = mpi::COMM_WORLD;
	const Int commRank = mpi::CommRank( comm );
	if(commRank == 0)
		printf("Performing Local tests:\n");
	RedistType redist = Local;
	ModeArray modes(2);
	modes[0] = 6;
	modes[1] = 8;

	//Check that Local Redist works correctly (from one mode dists to multiple)
	RunTest(g, "[(0,1),(2,3,9),(4),(6,8)]|(5,7)", "[(0,1),(2,3,9),(4),()]|(7,5)", redist, modes, false);

	//Check that Local works correctly (to multiple)
	RunTest(g, "[(0,1),(2,3,9,6),(4),(8)]|(5,7)", "[(0,1),(2,3,9),(4),()]|(7,5)", redist, modes, false);

	//Check that Local works correctly (to single)
	RunTest(g, "[(0,1),(2,3,9),(4),(6,8)]|(5,7)", "[(0,1),(2,3,9),(4),()]|(7,5)", redist, modes, false);

	//Check that Local fails correctly (allgather)
	RunTest(g, "[(0,1),(2,3,9),(4),()]|(5,7)", "[(0,1),(2,3,9),(4),(6,8)]|(7,5)", redist, modes, true);

	//Check that Local fails correctly (alltoall)
	RunTest(g, "[(0,1),(2,3),(4,8),(9,6)]|(5,7)", "[(0,1),(2,3),(4,6),(9,8)]|(7,5)", redist, modes, true);

	//Check that Local fails correctly (not suffix)
	RunTest(g, "[(0,1),(2,3,9),(8,4),(6)]|(5,7)", "[(0,1),(2,3,9),(4),()]|(7,5)", redist, modes, true);

	if(commRank == 0)
		printf("\n");
}

void CheckPermuteRedist(const Grid& g){
	mpi::Comm comm = mpi::COMM_WORLD;
	const Int commRank = mpi::CommRank( comm );
	if(commRank == 0)
		printf("Performing Permutation tests:\n");
	RedistType redist = Perm;
	ModeArray modes(3);
	modes[0] = 0;
	modes[1] = 1;
	modes[2] = 2;

	//Check that Permute Redist works correctly (within modes)
	RunTest(g, "[(9),(8),(7),(6,0,1,2)]|(5,4)", "[(9),(8),(7),(6,1,0,2)]|(5,4)", redist, modes, false);

	//Check that Permute works correctly (exchange to same dim modes)
	RunTest(g, "[(9,3),(8),(7,2),(6,0,1)]|(5,4)", "[(9,3),(8),(7,1),(6,0,2)]|(5,4)", redist, modes, false);

	//Check that Permute works correctly (exchange same dim modes)
	RunTest(g, "[(9,3),(8),(7,0),(6,2,1)]|(5,4)", "[(9,3),(8),(7,1,2),(6,0)]|(5,4)", redist, modes, false);

	//Check that Permute fails correctly (different dim exchange)
	RunTest(g, "[(9,3),(8),(7,1,0),(6,2)]|(5,4)", "[(9,3),(8),(7,1,2),(6,0)]|(5,4)", redist, modes, true);

	//Check that Permute works correctly (not suffix)
	RunTest(g, "[(9,3),(8),(7,0),(6,2,1)]|(5,4)", "[(9,3),(8),(7,1,2),(0,6)]|(5,4)", redist, modes, false);

	if(commRank == 0)
		printf("\n");
}

void CheckGatherToOneRedist(const Grid& g){
	mpi::Comm comm = mpi::COMM_WORLD;
	const Int commRank = mpi::CommRank( comm );
	if(commRank == 0)
		printf("Performing Gather-to-one tests:\n");
	RedistType redist = GTO;
	ModeArray modes(3);
	modes[0] = 0;
	modes[1] = 1;
	modes[2] = 2;

	//Check that GTO Redist works correctly (move to non-dist modes)
	RunTest(g, "[(9),(8),(7),(6)]|(5,4,2,0,1)", "[(9),(8),(7,0,2),(6,1)]|(5,4)", redist, modes, false);

	//Check that GTO fails correctly (not suffix)
	RunTest(g, "[(9,3),(8),(7),(6)]|(5,4,1,0,2)", "[(9,3),(8),(7,1),(0,2,6)]|(5,4)", redist, modes, true);

	//Check that GTO fails correctly (not added to non-dist)
	RunTest(g, "[(9,3),(8),(7),(6)]|(5,4,1)", "[(9,3),(8),(7,1),(6,0,2)]|(5,4)", redist, modes, true);

	if(commRank == 0)
		printf("\n");
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
    	Unsigned gridOrder = 10;
    	ObjShape gridShape(gridOrder);
    	gridShape[0] = 4;
    	gridShape[1] = 2;
    	gridShape[2] = 2;
    	gridShape[3] = 1;
    	for(i = 4; i < gridOrder; i++)
    		gridShape[i] = 1;

    	const Grid g(comm, gridShape);
        CheckAGRedist(g);
        CheckA2ARedist(g);
        CheckLocalRedist(g);
        CheckPermuteRedist(g);
        CheckGatherToOneRedist(g);
    }
    catch( std::exception& e ) { ReportException(e); }

    Finalize();
    //printf("Completed\n");
    return 0;
}
