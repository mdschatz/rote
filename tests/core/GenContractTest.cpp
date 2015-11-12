/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
// NOTE: It is possible to simply include "tensormental.hpp" instead
#include "tensormental.hpp"

using namespace rote;

void Usage(){
    std::cout << "./GenContractTest <distA> <indicesA> <distB> <indicesB> <distC> <indicesC> <m-dim> <k-dim> <n-dim>\n";
}

template<typename T>
void
Set(Tensor<T>& A)
{
    Unsigned order = A.Order();
    Location loc(order, 0);

    Unsigned ptr = 0;
    Unsigned counter = 0;
    bool stop = !ElemwiseLessThan(loc, A.Shape());

    while(!stop){
        A.Set(loc, counter);

        //Update
        counter++;
        if(order == 0)
            break;
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
SetWith(DistTensor<T>& A, const T* buf)
{
    Unsigned order = A.Order();
    Location loc(order, 0);

    Unsigned ptr = 0;
    Unsigned counter = 0;
    bool stop = !ElemwiseLessThan(loc, A.Shape());

    while(!stop){
        A.Set(loc, buf[counter]);

        //Update
        counter++;
        if(order == 0)
            break;
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
Set(DistTensor<T>& A)
{
    Unsigned order = A.Order();
    Location loc(order, 0);

    Unsigned ptr = 0;
    Unsigned counter = 0;
    bool stop = !ElemwiseLessThan(loc, A.Shape());

    while(!stop){
        A.Set(loc, counter);

        //Update
        counter++;
        if(order == 0)
            break;
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

typedef struct Arguments{
  TensorDistribution distA;
  TensorDistribution distB;
  TensorDistribution distC;
  IndexArray indicesA;
  IndexArray indicesB;
  IndexArray indicesC;
  Unsigned m_dim;
  Unsigned k_dim;
  Unsigned n_dim;
} Params;

void ProcessInput(Unsigned argc,  char** const argv, Params& args){
    Unsigned i;
    Unsigned argCount = 0;

    if(argCount + 1 >= argc){
        std::cerr << "Missing required distA argument\n";
        Usage();
        throw ArgException();
    }
    args.distA = rote::StringToTensorDist(argv[++argCount]);

    if(argCount + 1 >= argc){
        std::cerr << "Missing required indicesA argument\n";
        Usage();
        throw ArgException();
    }
    std::string indicesA = argv[++argCount];
    IndexArray indAArr(indicesA.size());
    for(i = 0; i < indicesA.size(); i++)
    	indAArr[i] = indicesA[i];
    args.indicesA = indAArr;

    if(argCount + 1 >= argc){
        std::cerr << "Missing required distB argument\n";
        Usage();
        throw ArgException();
    }
    args.distB = rote::StringToTensorDist(argv[++argCount]);

    if(argCount + 1 >= argc){
        std::cerr << "Missing required indicesB argument\n";
        Usage();
        throw ArgException();
    }
    std::string indicesB = argv[++argCount];
    IndexArray indBArr(indicesB.size());
    for(i = 0; i < indicesB.size(); i++)
    	indBArr[i] = indicesB[i];
    args.indicesB = indBArr;

    if(argCount + 1 >= argc){
        std::cerr << "Missing required distC argument\n";
        Usage();
        throw ArgException();
    }
    args.distC = rote::StringToTensorDist(argv[++argCount]);

    if(argCount + 1 >= argc){
        std::cerr << "Missing required indicesC argument\n";
        Usage();
        throw ArgException();
    }
    std::string indicesC = argv[++argCount];
    IndexArray indCArr(indicesC.size());
    for(i = 0; i < indicesC.size(); i++)
    	indCArr[i] = indicesC[i];
    args.indicesC = indCArr;

    if(argCount + 1 >= argc){
        std::cerr << "Missing required m-dim argument\n";
        Usage();
        throw ArgException();
    }
    args.m_dim = atoi(argv[++argCount]);

    if(argCount + 1 >= argc){
        std::cerr << "Missing required k-dim argument\n";
        Usage();
        throw ArgException();
    }
    args.k_dim = atoi(argv[++argCount]);

    if(argCount + 1 >= argc){
        std::cerr << "Missing required n-dim argument\n";
        Usage();
        throw ArgException();
    }
    args.n_dim = atoi(argv[++argCount]);
}

void RunTest(const Grid& g, const Params& args){
	Unsigned i;
	mpi::Comm comm = mpi::COMM_WORLD;
	const Int commRank = mpi::CommRank( comm );
	if(commRank == 0){
		std::cout << "A" << rote::TensorDistToString(args.distA) << " ";
		PrintVector(args.indicesA);
		std::cout << "B" << rote::TensorDistToString(args.distB) << " ";
		PrintVector(args.indicesB);
		std::cout << "C" << rote::TensorDistToString(args.distC) << " ";
		PrintVector(args.indicesC);
	}

	ObjShape shapeA(args.indicesA.size());
	ObjShape shapeB(args.indicesB.size());
	ObjShape shapeC(args.indicesC.size());
	for(i = 0; i < args.indicesA.size(); i++){
		Index index = args.indicesA[i];
		if(Contains(args.indicesB, index))
			shapeA[i] = args.k_dim;
		else
			shapeA[i] = args.m_dim;
	}
	for(i = 0; i < args.indicesB.size(); i++){
		Index index = args.indicesB[i];
		if(Contains(args.indicesA, index))
			shapeB[i] = args.k_dim;
		else
			shapeB[i] = args.n_dim;
	}
	for(i = 0; i < args.indicesC.size(); i++){
		Index index = args.indicesC[i];
		if(Contains(args.indicesA, index))
			shapeC[i] = args.m_dim;
		else
			shapeC[i] = args.n_dim;
	}

	DistTensor<double> A(shapeA, args.distA, g);
	DistTensor<double> B(shapeB, args.distB, g);
	DistTensor<double> C(shapeC, args.distC, g);

	Set(A);
	Set(B);
	Set(C);

	double alpha = 1.0;
	double beta = 0.0;
	GenContract(alpha, A, args.indicesA, B, args.indicesB, beta, C, args.indicesC);

	TensorDistribution distFinalC = args.distC;
	for(i = 0; i < C.Order(); i++){
		ModeDistribution blank(0);
		distFinalC[i] = blank;
	}
	DistTensor<double> finalC(shapeC, distFinalC, g);
	finalC.RedistFrom(C);
	Print(finalC, "final");

	if(commRank == 0){
		Tensor<double> checkA(shapeA);
		Tensor<double> checkB(shapeB);
		Tensor<double> checkC(shapeC);
		Set(checkA);
		Set(checkB);
		Set(checkC);

		LocalContractAndLocalEliminate(alpha, checkA, args.indicesA, checkB, args.indicesB, beta, checkC, args.indicesC);
		double norm;
		Tensor<double> diff(finalC.Tensor().Shape());
		Diff(finalC.Tensor(), checkC, diff);
		norm = Norm(diff);
		std::cout << "Norm: " << norm << std::endl;
	}
}

int
main( int argc, char* argv[] )
{
    Initialize( argc, argv );
    Params args;
    ProcessInput(argc, argv, args);
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
    	gridShape[2] = 3;
    	gridShape[3] = 2;
    	for(i = 4; i < gridOrder; i++)
    		gridShape[i] = 1;

    	const Grid g(comm, gridShape);
        RunTest(g, args);
    }
    catch( std::exception& e ) { ReportException(e); }

    Finalize();
    //printf("Completed\n");
    return 0;
}
