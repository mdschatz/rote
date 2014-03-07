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
  int gridOrder;
  int tenOrder;
  int nProcs;
  std::vector<int> gridShape;
  std::vector<int> tensorShape;
  TensorDistribution tensorDist;
} Params;

typedef std::pair< int, TensorDistribution> AGTest;
typedef std::pair< int, TensorDistribution> PRSTest;
typedef std::pair< std::pair<int, int>, TensorDistribution> RSTest;

void ProcessInput(int argc,  char** const argv, Params& args){
    int argCount = 0;
    if(argCount + 1 >= argc){
        std::cerr << "Missing required gridOrder argument\n";
        Usage();
        throw ArgException();
    }

    int gridOrder = atoi(argv[++argCount]);
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
TestAGRedist( DistTensor<T>& A, int agIndex, const TensorDistribution& resDist )
{
#ifndef RELEASE
    CallStackEntry entry("TestAGRedist");
#endif
    //const int order = A.Order();
    const Grid& g = A.Grid();

    DistTensor<T> B(A.Shape(), resDist, A.Indices(), g);
    AllGatherRedist(B, A, agIndex);
    Print(B, "B after ag redist");
}

template<typename T>
void
TestRSRedist(DistTensor<T>& A, int reduceIndex, int scatterIndex, const TensorDistribution& resDist)
{
#ifndef RELEASE
    CallStackEntry entry("TestRSRedist");
#endif
    const int redistMode = 0;

    const Grid& g = A.Grid();
    const GridView gv = A.GridView();

    std::vector<Int> BIndices = A.Indices();
    const int reduceMode = A.ModeOfIndex(reduceIndex);
    BIndices.erase(BIndices.begin() + reduceMode);
    std::vector<Int> BShape = A.Shape();
    BShape.erase(BShape.begin() + reduceMode);
    DistTensor<T> B(BShape, resDist, BIndices, g);

    ReduceScatterRedist(B, A, reduceIndex, scatterIndex);

    Print(B, "B after rs redist");
}

template<typename T>
void
TestPRSRedist(DistTensor<T>& A, int reduceScatterIndex, const TensorDistribution& resDist)
{
#ifndef RELEASE
    CallStackEntry entry("TestRSRedist");
#endif
    const int redistMode = 0;

    const Grid& g = A.Grid();
    const GridView gv = A.GridView();

    const int modeOfRSIndex = A.ModeOfIndex(reduceScatterIndex);
    std::vector<int> BShape = A.Shape();
    BShape[modeOfRSIndex] = A.Dimension(modeOfRSIndex) / gv.Dimension(modeOfRSIndex);

    DistTensor<T> B(BShape, resDist, A.Indices(), g);

    PartialReduceScatterRedist(B, A, reduceScatterIndex);

    Print(B, "B after prs redist");
}

template<typename T>
void
Set(DistTensor<T>& A)
{
    Int order = A.Order();
    std::vector<Int> index(order);
    Int ptr = 0;
    Int counter = 0;
    bool stop = false;

    while(!stop){
        A.Set(index, counter);

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
TensorDistribution
DetermineResultingDistributionAG(const DistTensor<T>& A, int index){
    TensorDistribution ret;
    const TensorDistribution ADist = A.TensorDist();
    ret = ADist;
    ret[A.ModeOfIndex(index)].clear();
    return ret;
}

template<typename T>
TensorDistribution
DetermineResultingDistributionRS(const DistTensor<T>& A, int reduceIndex, int scatterIndex){
    TensorDistribution ret;
    const TensorDistribution ADist = A.TensorDist();
    ret = ADist;
    ModeDistribution& scatterIndexDist = ret[A.ModeOfIndex(scatterIndex)];
    ModeDistribution& reduceIndexDist = ret[A.ModeOfIndex(reduceIndex)];
    scatterIndexDist.insert(scatterIndexDist.end(), reduceIndexDist.begin(), reduceIndexDist.end());
    ret.erase(ret.begin() + A.ModeOfIndex(reduceIndex));
    return ret;
}

template<typename T>
TensorDistribution
DetermineResultingDistributionPRS(const DistTensor<T>& A, int reduceScatterIndex){
    TensorDistribution ret;
    const TensorDistribution ADist = A.TensorDist();
    ret = ADist;
    return ret;
}

template<typename T>
std::vector<AGTest >
CreateAGTests(const DistTensor<T>& A, const Params& args){
    std::vector<AGTest > ret;

    const int order = A.Order();
    const std::vector<int> indices = A.Indices();

    for(int i = 0; i < order; i++){
        const int indexToRedist = indices[i];
        AGTest test(indexToRedist, DetermineResultingDistributionAG(A, indexToRedist));
        ret.push_back(test);
    }

    return ret;
}

template<typename T>
std::vector<PRSTest >
CreatePRSTests(const DistTensor<T>& A, const Params& args){
    std::vector<PRSTest> ret;

    const int order = A.Order();
    const std::vector<int> indices = A.Indices();

    for(int i = 0; i < order; i++){
        const int indexToReduceScatter = indices[i];
        PRSTest test(indexToReduceScatter, DetermineResultingDistributionPRS(A, indexToReduceScatter));
        ret.push_back(test);
    }

    return ret;
}

template<typename T>
std::vector<RSTest >
CreateRSTests(const DistTensor<T>& A, const Params& args){
    std::vector<RSTest> ret;

    const int order = A.Order();
    const std::vector<int> indices = A.Indices();

    const GridView gv = A.GridView();

    for(int i = 0; i < order; i++){
        for(int j = 0; j < order; j++){
            if(i == j)
                continue;
            const int indexToReduce = indices[i];
            const int reduceMode = A.ModeOfIndex(indexToReduce);
            const int indexToScatter = indices[j];

            if(A.Dimension(reduceMode) > gv.Dimension(A.ModeOfIndex(reduceMode)))
                continue;
            std::pair<int, int> redistIndices(i, j);
            RSTest test(redistIndices, DetermineResultingDistributionRS(A, indexToReduce, indexToScatter));
            ret.push_back(test);
        }
    }

    return ret;
}

template<typename T>
void
DistTensorTest( const Params& args, const Grid& g )
{
#ifndef RELEASE
    CallStackEntry entry("DistTensorTest");
#endif
    const Int commRank = mpi::CommRank( mpi::COMM_WORLD );
    const int order = args.tensorShape.size();
    std::vector<int> indices(order);
    for(int i = 0 ; i < indices.size(); i++)
        indices[i] = i;

    DistTensor<T> A(args.tensorShape, args.tensorDist, indices, g);

    Set(A);

    std::vector<AGTest> agTests = CreateAGTests(A, args);
    std::vector<RSTest> rsTests = CreateRSTests(A, args);
    std::vector<PRSTest> prsTests = CreatePRSTests(A, args);

    if(commRank == 0){
        printf("Performing AllGather tests\n");
    }
    for(int i = 0; i < agTests.size(); i++){
        AGTest thisTest = agTests[i];
        int agIndex = thisTest.first;
        TensorDistribution resDist = thisTest.second;
        if(commRank == 0){
            printf("Allgathering index %d with resulting distribution %s\n", agIndex, (tmen::TensorDistToString(resDist)).c_str());
        }
        TestAGRedist(A, agIndex, resDist);
    }

    if(commRank == 0){
        printf("Performing PartialReduceScatter tests\n");
    }
    for(int i = 0; i < prsTests.size(); i++){
        PRSTest thisTest = prsTests[i];
        int rsIndex = thisTest.first;
        TensorDistribution resDist = thisTest.second;
        if(commRank == 0){
            printf("Partial reduce-scattering index %d with resulting distribution %s\n", rsIndex, (tmen::TensorDistToString(resDist)).c_str());
        }
        TestPRSRedist(A, rsIndex, resDist);
    }

    if(commRank == 0){
        printf("Performing ReduceScatter tests\n");
    }
    for(int i = 0; i < rsTests.size(); i++){
        RSTest thisTest = rsTests[i];
        int reduceIndex = thisTest.first.first;
        int scatterIndex = thisTest.first.second;
        TensorDistribution resDist = thisTest.second;

        if(commRank == 0){
        printf(
                "Reducing index %d, scattering index %d, with resulting distribution %s\n",
                reduceIndex, scatterIndex,
                (tmen::TensorDistToString(resDist)).c_str());
        }
        TestRSRedist(A, reduceIndex, scatterIndex, resDist);
    }
}

int 
main( int argc, char* argv[] )
{
    Initialize( argc, argv );
    mpi::Comm comm = mpi::COMM_WORLD;
    const Int commRank = mpi::CommRank( comm );
    const Int commSize = mpi::CommSize( comm );
    //printf("My Rank: %d\n", commRank);
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
            std::cout << "------------------" << std::endl
                      << "Testing with ints:" << std::endl
                      << "------------------" << std::endl;
        }
        DistTensorTest<int>( args, g );

        if( commRank == 0 )
        {
            std::cout << "--------------------" << std::endl
                      << "Testing with floats:" << std::endl
                      << "--------------------" << std::endl;
        }
        DistTensorTest<float>( args, g );

        if( commRank == 0 )
        {
            std::cout << "---------------------" << std::endl
                      << "Testing with doubles:" << std::endl
                      << "---------------------" << std::endl;
        }
        DistTensorTest<double>( args, g );

    }
    catch( std::exception& e ) { ReportException(e); }

    Finalize();
    //printf("Completed\n");
    return 0;
}
