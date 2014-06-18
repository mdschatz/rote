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
    std::cout << "./Contract <gridOrder> <gridDim0> <gridDim1> ... <ten1Order> <ten1Dim0> <ten1Dim1> ... <ten1Ind1> <ten1Ind2> ... <ten2Order> <ten2Dim0> <ten2Dim1> ... <ten2Ind1> <ten2Ind2> ... \"<tensorDist>\"\n";
    std::cout << "<gridOrder>  : order of the grid ( >0 )\n";
    std::cout << "<gridDimK>   : dimension of mode-K of grid\n";
    std::cout << "<tenIOrder>  : order of the tensor I ( >0 )\n";
    std::cout << "<tenIDimK>   : dimension of mode-K of tensor I\n";
    std::cout << "<tenIIndK>   : index of mode-K of tensor I\n";
    //std::cout << "<tensorDist> : distribution of tensor(Must be in quotes)\n";
}

typedef struct Arguments{
  Unsigned gridOrder;
  Unsigned nProcs;
  ObjShape gridShape;
  Unsigned ten1Order;
  ObjShape ten1Shape;
  IndexArray ten1Indices;
  Unsigned ten2Order;
  ObjShape ten2Shape;
  IndexArray ten2Indices;
  //TensorDistribution tensorDist;
} Params;

typedef std::pair< ModeArray, ObjShape> LocalTest;

void ProcessInput(int argc,  char** const argv, Params& args){
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
        std::cerr << "Missing required ten1Order argument\n";
        Usage();
        throw ArgException();
    }
    Unsigned ten1Order = atoi(argv[++argCount]);
    args.ten1Order = ten1Order;

    if(argCount + ten1Order >= argc){
        std::cerr << "Missing required tensor dimensions\n";
        Usage();
        throw ArgException();
    }

    args.ten1Shape.resize(ten1Order);
    for(i = 0; i < ten1Order; i++){
        Unsigned tensorDim = atoi(argv[++argCount]);
        if(tensorDim == 0){
            std::cerr << "Tensor dimension must be greater than 0\n";
            Usage();
            throw ArgException();
        }
        args.ten1Shape[i] = tensorDim;
    }

    if(argCount + ten1Order >= argc){
        std::cerr << "Missing required tensor indices\n";
        Usage();
        throw ArgException();
    }

    args.ten1Indices.resize(ten1Order);
    for(i = 0; i < ten1Order; i++){
        Unsigned tensorIndex = atoi(argv[++argCount]);
        if(tensorIndex < 0){
            std::cerr << "Tensor index must be >= 0\n";
            Usage();
            throw ArgException();
        }
        args.ten1Indices[i] = tensorIndex;
    }

    if(argCount + 1 >= argc){
        std::cerr << "Missing required ten2Order argument\n";
        Usage();
        throw ArgException();
    }
    Unsigned ten2Order = atoi(argv[++argCount]);
    args.ten2Order = ten2Order;

    if(argCount + ten2Order >= argc){
        std::cerr << "Missing required tensor dimensions\n";
        Usage();
        throw ArgException();
    }

    args.ten2Shape.resize(ten2Order);
    for(i = 0; i < ten2Order; i++){
        Unsigned tensorDim = atoi(argv[++argCount]);
        if(tensorDim == 0){
            std::cerr << "Tensor dimension must be greater than 0\n";
            Usage();
            throw ArgException();
        }
        args.ten2Shape[i] = tensorDim;
    }

    if(argCount + ten2Order >= argc){
        std::cerr << "Missing required tensor indices\n";
        Usage();
        throw ArgException();
    }

    args.ten2Indices.resize(ten2Order);
    for(i = 0; i < ten2Order; i++){
        Unsigned tensorIndex = atoi(argv[++argCount]);
        if(tensorIndex < 0){
            std::cerr << "Tensor index must be >= 0\n";
            Usage();
            throw ArgException();
        }
        args.ten2Indices[i] = tensorIndex;
    }

//    if(argCount + 1 >= argc){
//        std::cerr << "Missing tensor distribution argument\n";
//        Usage();
//        throw ArgException();
//    }
//
//    std::string tensorDist(argv[++argCount]);
//    args.tensorDist = tmen::StringToTensorDist(tensorDist);
//
//
//    if(args.tensorDist.size() != args.tensorShape.size()){
//        std::cerr << "Tensor distribution must be of same order as tensor\n";
//        Usage();
//        throw ArgException();
//    }
}

template<typename T>
void
Set(Tensor<T>& A)
{
    Unsigned order = A.Order();
    Location loc(order);
    std::fill(loc.begin(), loc.end(), 0);
    Unsigned ptr = 0;
    Unsigned counter = 0;
    bool stop = false;

    while(!stop){
        A.Set(loc, counter);

        //Update
        counter++;
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
std::vector<LocalTest>
CreateLocalTests(const Tensor<T>& A, const Tensor<T>& B, const IndexArray& AIndices, const IndexArray& BIndices, const IndexArray& CIndices)
{
#ifndef RELEASE
    CallStackEntry entry("CreateLocalTests");
#endif
    Unsigned i, j;
    std::vector<LocalTest> ret;
    ModeArray testModes(CIndices.size());
    for(i = 0; i < CIndices.size(); i++)
        testModes[i] = i;

    do{
        ObjShape shapeC(testModes.size());
        for(i = 0; i < testModes.size(); i++){
            for(j = 0; j < AIndices.size(); j++){
                if(AIndices[j] == CIndices[i])
                    shapeC[i] = A.Dimension(j);
            }
            for(j = 0; j < BIndices.size(); j++){
                if(BIndices[j] == CIndices[i])
                    shapeC[i] = B.Dimension(j);
            }
        }
        LocalTest test(testModes, shapeC);
        ret.push_back(test);

    }while(std::next_permutation(testModes.begin(), testModes.end()));
    return ret;
}

template<typename T>
void
LocalContractTest( const Params& args )
{
#ifndef RELEASE
    CallStackEntry entry("LocalContractTest");
#endif
    Unsigned i, j;
    const Int commRank = mpi::CommRank( mpi::COMM_WORLD );

    Tensor<T> A(args.ten1Shape);
    Tensor<T> B(args.ten2Shape);
    Set(A);
    Set(B);

    IndexArray AIndices = args.ten1Indices;
    IndexArray BIndices = args.ten2Indices;
    IndexArray CIndices;

    for(i = 0; i < AIndices.size(); i++){
        Index index = AIndices[i];
        if(std::find(BIndices.begin(), BIndices.end(), index) == BIndices.end())
            CIndices.push_back(index);
    }
    for(i = 0; i < BIndices.size(); i++){
        Index index = BIndices[i];
        if(std::find(AIndices.begin(), AIndices.end(), index) == AIndices.end())
            CIndices.push_back(index);
    }

    std::vector<LocalTest> localTests = CreateLocalTests(A, B, AIndices, BIndices, CIndices);

    for(i = 0; i < localTests.size(); i++){
        LocalTest localTest = localTests[i];
        std::vector<IndexArray> indices(3);
        indices[0] = AIndices;
        indices[1] = BIndices;
        indices[2] = CIndices;
        ObjShape testShape = localTest.second;
        Tensor<T> C(testShape);
        MemZero(C.Buffer(), prod(C.Shape()));
        if(commRank == 0){

            ObjShape shapeA = A.Shape();
            ObjShape shapeB = B.Shape();
            ObjShape shapeC = C.Shape();
            printf("Performing LocalTest:\n");
            printf("A[%d", AIndices[0]);
            for(j = 1; j < AIndices.size(); j++)
                printf(" %d", AIndices[j]);
            printf("] of size: [%d", shapeA[0]);
            for(j = 1; j < shapeA.size(); j++)
                printf(" %d", shapeA[j]);
            printf("]\n");
            printf("B[%d", BIndices[0]);
            for(j = 1; j < BIndices.size(); j++)
                printf(" %d", BIndices[j]);
            printf("] of size: [%d", shapeB[0]);
            for(j = 1; j < shapeB.size(); j++)
                printf(" %d", shapeB[j]);
            printf("]\n");
            printf("C[%d", CIndices[0]);
            for(j = 1; j < CIndices.size(); j++)
                printf(" %d", CIndices[j]);
            printf("] of size: [%d", shapeC[0]);
            for(j = 1; j < shapeC.size(); j++)
                printf(" %d", shapeC[j]);
            printf("]\n");
        }
        Print(A, "A");
        Print(B, "B");
        Print(C, "PreC");
        LocalContract(T(1), A, B, T(1), C, indices);
        Print(C, "PostC");
    }
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

        const Grid g( comm, args.gridOrder, args.gridShape );

        if( commRank == 0 )
        {
            std::cout << "------------------" << std::endl
                      << "Testing with ints:" << std::endl
                      << "------------------" << std::endl;
        }
        LocalContractTest<int>( args );

//        if( commRank == 0 )
//        {
//            std::cout << "--------------------" << std::endl
//                      << "Testing with floats:" << std::endl
//                      << "--------------------" << std::endl;
//        }
//        DistTensorTest<float>( args, g );
//
//        if( commRank == 0 )
//        {
//            std::cout << "---------------------" << std::endl
//                      << "Testing with doubles:" << std::endl
//                      << "---------------------" << std::endl;
//        }
//        DistTensorTest<double>( args, g );

    }
    catch( std::exception& e ) { ReportException(e); }

    Finalize();
    //printf("Completed\n");
    return 0;
}
