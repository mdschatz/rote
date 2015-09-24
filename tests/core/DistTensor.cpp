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
#include "tensormental/tests/BCastRedist.hpp"
#include "tensormental/tests/ScatterRedist.hpp"

using namespace tmen;

enum CommTypes {AG, A2A, Local, RS, RTO, AR, GTO, BCast, Scatter, Perm};

void Usage(){
    std::cout << "./DistTensor <gridOrder> <gridDim0> <gridDim1> ... <tenOrder> <tenDim0> <tenDim1> ... \"<tensorDist>\"\n";
    std::cout << "<gridOrder>  : order of the grid ( >0 )\n";
    std::cout << "<gridDimK>   : dimension of mode-K of grid\n";
    std::cout << "<tenOrder>   : order of the tensor ( >0 )\n";
    std::cout << "<tenDimK>    : dimension of mode-K of tensor\n";
    std::cout << "<tensorDist> : distribution of tensor(Must be in quotes)\n";
}

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

//TODO: Fix for reduce tests
template<typename T>
void
PerformSingleTest( const CommTypes& commType, const TensorDistribution& resDist, const DistTensor<T>& A, const Permutation& outputPerm, const ModeArray& commModes){
#ifndef RELEASE
    CallStackEntry entry("PerformSingleTest");
#endif
	Unsigned i;
    Unsigned order = A.Order();
    const Int commRank = mpi::CommRank( mpi::COMM_WORLD );
    const Grid& g = A.Grid();

    if(commRank == 0){
        std::cout << (tmen::TensorDistToString(resDist)).c_str() << " <-- "
        		  << (tmen::TensorDistToString(A.TensorDist())).c_str()
				  << std::endl;
        PrintVector(A.LocalPermutation(), "input perm");
        PrintVector(outputPerm, "output perm");

    }

    DistTensor<T> B(resDist, g);
    B.AlignWith(A);
    B.SetLocalPermutation(outputPerm);

    switch(commType){
//		case AG:      B.AllGatherRedistFrom(A, commModes); break;
//		case A2A:     B.AllToAllRedistFrom(A, commModes); break;
//		case RS:      B.ReduceScatterRedistFrom(A, commModes); break;
//		case AR:      B.AllReduceRedistFrom(A, commModes); break;
//		case RTO:     B.ReduceToOneRedistFrom(A, commModes); break;
//		case GTO:     B.GatherToOneRedistFrom(A, commModes); break;
		case BCast:   B.BroadcastRedistFrom(A, commModes); break;
		case Scatter: B.ScatterRedistFrom(A, commModes); break;
//		case Local:   B.LocalRedistFrom(A, commModes); break;
//		case Perm:    B.PermutationRedistFrom(A, commModes); break;
    }

//    Print(A, "input");
//    Print(B, "output");
	DistTensor<T> check(A.Shape(), resDist, g);
	check.SetLocalPermutation(outputPerm);
	Set(check);
	CheckResult(B, check);
}

template<typename T>
void
PerformRedistTests( const CommTypes& commType, const std::vector<Permutation>& perms, const ObjShape& tenShape, const TensorDistribution& inputDist, const Grid& g){
#ifndef RELEASE
    CallStackEntry entry("PerformRedistTests");
#endif
    Unsigned i, j, k;
    const Int commRank = mpi::CommRank( mpi::COMM_WORLD );

    std::vector<RedistTest> tests;
    std::string redistName;
    switch(commType){
//		case AG:      tests = CreateAGGTests(A, args); redistName = "AllGather"; break;
//		case A2A:     tests = CreateA2ATests(A, args); redistName = "AllToAll"; break;
		case BCast:   tests = CreateBCastTests(inputDist); redistName = "Broadcast"; break;
		case Scatter: tests = CreateScatterTests(inputDist); redistName = "Scatter"; break;
//		case Local:   tests = CreateLGTests(A, args); redistName = "Local"; break;
//		case Perm:    tests = CreatePTests(A, args); redistName = "Permutation"; break;
//		case RTO:     tests = CreateRTOGTests(A, args); redistName = "ReduceToOne"; break;
//		case AR:      tests = CreateARTests(A, args); redistName = "AllReduce"; break;
//		case RS:      tests = CreateRSGTests(A, args); redistName = "ReduceScatter"; break;
//		case GTO:     tests = CreateGTOGTests(A, args); redistName = "GatherToOne"; break;
    }

    if(commRank == 0){
    	std::cout << "Performing " << redistName << " tests" << std::endl;
    }

	for(j = 0; j < perms.size(); j++){
		Permutation inputPerm = perms[j];
		DistTensor<T> A(tenShape, inputDist, g);
		A.SetLocalPermutation(inputPerm);
		Set(A);
		for(k = 0; k < perms.size(); k++){
			Permutation outputPerm = perms[k];
			for(i = 0; i < tests.size(); i++){
				RedistTest thisTest = tests[i];
				TensorDistribution resDist = thisTest.first;
				ModeArray commModes = thisTest.second;

				PerformSingleTest(commType, resDist, A, outputPerm, commModes);
    		}
    	}
    }
//
//    std::vector<AGGTest> aggTests = CreateAGGTests(A, args);
//    std::vector<BCastTest> bcastTests = CreateBCastTests(A, args);
//    std::vector<ScatterTest> scatterTests = CreateScatterTests(A, args);
//    std::vector<GTOGTest> gtogTests = CreateGTOGTests(A, args);
//    std::vector<LGTest> lgTests = CreateLGTests(A, args);
//    std::vector<RSGTest> rsgTests = CreateRSGTests(A, args);
//    std::vector<PTest> pTests = CreatePTests(A, args);
//    std::vector<A2ATest> a2aTests = CreateA2ATests(A, args);
//    std::vector<RTOGTest> rtogTests = CreateRTOGTests(A, args);

//
//    if(commRank == 0){
//        printf("Performing All-to-all tests\n");
//    }
//    for(i = 0; i < a2aTests.size(); i++){
//        A2ATest thisTest = a2aTests[i];
//        ModeArray a2aModesFrom = thisTest.first.first.first;
//        ModeArray a2aModesTo = thisTest.first.first.second;
//        std::vector<ModeArray> commModes = thisTest.first.second;
//        TensorDistribution resDist = thisTest.second;
//
//        TestA2ARedist(A, a2aModesFrom, a2aModesTo, commModes, resDist);
//    }
//
//    if(commRank == 0){
//        printf("Performing AllGatherG tests\n");
//    }
//    for(i = 0; i < aggTests.size(); i++){
//        AGGTest thisTest = aggTests[i];
//        ModeArray agModes = thisTest.first.first;
//        std::vector<ModeArray> redistGroups = thisTest.first.second;
//        TensorDistribution resDist = thisTest.second;
//
//        TestAGGRedist(A, agModes, redistGroups, resDist);
//    }
//
//    if(commRank == 0){
//        printf("Performing Gather-to-one-G tests\n");
//    }
//    for(i = 0; i < gtogTests.size(); i++){
//        GTOGTest thisTest = gtogTests[i];
//        ModeArray gModes = thisTest.first.first;
//        std::vector<ModeArray> redistGroups = thisTest.first.second;
//        TensorDistribution resDist = thisTest.second;
//
//        TestGTOGRedist(A, gModes, redistGroups, resDist);
//    }
//
//    if(commRank == 0){
//        printf("Performing LocalG redist tests\n");
//    }
//    for(i = 0; i < lgTests.size(); i++){
//        LGTest thisTest = lgTests[i];
//        ModeArray lModes = thisTest.first.first;
//        std::vector<ModeArray> commGroups = thisTest.first.second;
//        TensorDistribution resDist = thisTest.second;
//
//        TestLGRedist(A, lModes, commGroups, resDist);
//    }
//
//    if(commRank == 0){
//        printf("Performing Permutation tests\n");
//    }
//    for(i = 0; i < pTests.size(); i++){
//        PTest thisTest = pTests[i];
//        Mode pMode = thisTest.first;
//        ModeDistribution resDist = thisTest.second;
//
//        TestPRedist(A, pMode, resDist);
//    }
//
//    if(commRank == 0){
//        printf("Performing ReduceScatterG tests\n");
//    }
//    for(i = 0; i < rsgTests.size(); i++){
//        RSGTest thisTest = rsgTests[i];
//        ModeArray rModes = thisTest.first.first;
//        ModeArray sModes = thisTest.first.second;
//        TensorDistribution resDist = thisTest.second;
//
//        TestRSGRedist(A, rModes, sModes, resDist);
//    }
//
//    if(commRank == 0){
//            printf("Performing ReduceToOneG tests\n");
//    }
//    for(i = 0; i < rtogTests.size(); i++){
//        RTOGTest thisTest = rtogTests[i];
//        ModeArray rModes = thisTest.first;
//        TensorDistribution resDist = thisTest.second;
//
//        TestRTOGRedist(A, rModes, resDist);
//    }
//
//    if(commRank == 0){
//        printf("Performing Broadcast tests\n");
//    }
//    printf("bcast Size: %d\n", bcastTests.size());
//    for(i = 0; i < bcastTests.size(); i++){
//        BCastTest thisTest = bcastTests[i];
//        TensorDistribution resDist = thisTest.first;
//        const ModeArray bcastModes = thisTest.second;
//
//        TestBCastRedist(resDist, A, bcastModes);
//    }

//    if(commRank == 0){
//        printf("Performing Scatter tests\n");
//    }
//    printf("scatter Size: %d\n", scatterTests.size());
//    for(i = 0; i < scatterTests.size(); i++){
//        ScatterTest thisTest = scatterTests[i];
//        TensorDistribution resDist = thisTest.first;
//        const ModeArray bcastModes = thisTest.second;
//
//        TestScatterRedist(resDist, A, bcastModes);
//    }
}

std::vector<Permutation>
CreatePerms(const Unsigned& order){
	Unsigned i, j;
	std::vector<Permutation> ret;


	Permutation perm = DefaultPermutation(order);
	do{
		ret.push_back(perm);
	}while(next_permutation(perm.begin(), perm.end()));

	return ret;
}

template<typename T>
void
DistTensorTest( const Params& args, const Grid& g )
{
#ifndef RELEASE
    CallStackEntry entry("DistTensorTest");
#endif

    std::vector<Permutation> perms = CreatePerms(args.tenOrder);
//    Unsigned count = 0;
//    do{
//        if(count < 0)
//            count++;
//        else{
//            if(commRank == 0){
//                PrintVector(permA, "Testing Input Perm");
//            }
//
//            A.SetLocalPermutation(permA);
//            A.ResizeTo(A.Shape());
//            Set(A);
////                DistTensorTest<int>(A, args, g);
////                PerformTest<int>(A, args, g);
//        }
//    }while(next_permutation(permA.begin(), permA.end()));

    PerformRedistTests<T>(AG,      perms, args.tensorShape, args.tensorDist, g);
    PerformRedistTests<T>(A2A,     perms, args.tensorShape, args.tensorDist, g);
    PerformRedistTests<T>(BCast,   perms, args.tensorShape, args.tensorDist, g);
    PerformRedistTests<T>(Scatter, perms, args.tensorShape, args.tensorDist, g);
    PerformRedistTests<T>(Local,   perms, args.tensorShape, args.tensorDist, g);
    PerformRedistTests<T>(Perm,    perms, args.tensorShape, args.tensorDist, g);
    PerformRedistTests<T>(RTO,     perms, args.tensorShape, args.tensorDist, g);
    PerformRedistTests<T>(AR,      perms, args.tensorShape, args.tensorDist, g);
    PerformRedistTests<T>(RS,      perms, args.tensorShape, args.tensorDist, g);
    PerformRedistTests<T>(GTO,     perms, args.tensorShape, args.tensorDist, g);
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
            printf("Creating ");
            if(args.gridShape.size() > 0)
                printf("%d", args.gridShape[0]);
            for(i = 1; i < args.gridOrder; i++)
                printf(" x %d", args.gridShape[i]);
            printf(" grid\n");

            printf("Creating [");
            if(args.tensorShape.size() > 0)
                printf("%d", args.tensorShape[0]);
            for(i = 1; i < args.tenOrder; i++)
                printf(", %d", args.tensorShape[i]);
            printf("] tensor\n");
        }

        const Grid g( comm, args.gridShape );

        PrintVector(g.Loc(), "gridLoc");
        if( commRank == 0 )
        {
            std::cout << "------------------" << std::endl
                      << "Testing with ints:" << std::endl
                      << "------------------" << std::endl;
        }

        DistTensorTest<int>(args, g);
    }
    catch( std::exception& e ) { ReportException(e); }

    Finalize();
    //printf("Completed\n");
    return 0;
}
