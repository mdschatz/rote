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
  Unsigned gridOrder;
  Unsigned tenOrder;
  Unsigned nProcs;
  ObjShape gridShape;
  ObjShape tensorShape;
  TensorDistribution tensorDist;
} Params;

typedef std::pair< Index, ModeDistribution> PTest;
typedef std::pair< Index, TensorDistribution> AGTest;
typedef std::pair< Index, ModeDistribution> LTest;
typedef std::pair< Index, TensorDistribution> PRSTest;
typedef std::pair< std::pair<Index, Index>, TensorDistribution> RSTest;
typedef std::pair< std::pair<std::pair<Index, Index>, std::pair<ModeArray, ModeArray > >, TensorDistribution> A2ADITest;

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
        Unsigned tensorDim = atoi(argv[++argCount]);
        if(tensorDim == 0){
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
TestPRedist( DistTensor<T>& A, Index pIndex, const ModeDistribution& resDist )
{
#ifndef RELEASE
    CallStackEntry entry("TestAGRedist");
#endif
    //const int order = A.Order();
    const Grid& g = A.Grid();

    TensorDistribution BDist = A.TensorDist();
    BDist[A.ModeOfIndex(pIndex)] = resDist;

    DistTensor<T> B(A.Shape(), BDist, A.Indices(), g);
    //Print(B, "B before permute redist");
    PermutationRedist(B, A, pIndex);
    Print(B, "B after permute redist");
}

template<typename T>
void
TestAGRedist( DistTensor<T>& A, Index agIndex, const TensorDistribution& resDist )
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
TestLRedist( DistTensor<T>& A, Index lIndex, const ModeDistribution& resDist )
{
#ifndef RELEASE
    CallStackEntry entry("TestAGRedist");
#endif
    const Int commRank = mpi::CommRank( mpi::COMM_WORLD );
    const Grid& g = A.Grid();

    const Mode lModeA = A.ModeOfIndex(lIndex);
    TensorDistribution distB = A.TensorDist();
    distB[lModeA] = resDist;

    DistTensor<T> B(A.Shape(), distB, A.Indices(), g);

    ModeDistribution lIndexDist = A.ModeDist(lModeA);

    if(commRank == 0){
        printf("Locally redistributing index %d: %s <-- %s\n", lIndex, (tmen::TensorDistToString(B.TensorDist())).c_str(), (tmen::TensorDistToString(A.TensorDist())).c_str());
    }

    Print(A, "A before local redist");
    ModeArray gridRedistModes(resDist.begin() + lIndexDist.size(), resDist.end());
    LocalRedist(B, A, lIndex, gridRedistModes);
    Print(B, "B after local redist");
}

template<typename T>
void
TestRSRedist(DistTensor<T>& A, Index rIndex, Index sIndex, const TensorDistribution& resDist)
{
#ifndef RELEASE
    CallStackEntry entry("TestRSRedist");
#endif

    const Grid& g = A.Grid();
    const GridView gv = A.GridView();

    IndexArray BIndices = A.Indices();
    const Mode reduceMode = A.ModeOfIndex(rIndex);
    BIndices.erase(BIndices.begin() + reduceMode);
    ObjShape BShape = A.Shape();
    BShape.erase(BShape.begin() + reduceMode);
    DistTensor<T> B(BShape, resDist, BIndices, g);

    ReduceScatterRedist(B, A, rIndex, sIndex);

    Print(B, "B after rs redist");
}

template<typename T>
void
TestPRSRedist(DistTensor<T>& A, Index rsIndex, const TensorDistribution& resDist)
{
#ifndef RELEASE
    CallStackEntry entry("TestRSRedist");
#endif

    const Grid& g = A.Grid();
    const GridView gv = A.GridView();

    const Mode rsMode = A.ModeOfIndex(rsIndex);
    ObjShape BShape = A.Shape();
    BShape[rsMode] = Max(1, tmen::MaxLength(A.Dimension(rsMode), gv.Dimension(rsMode)));

    DistTensor<T> B(BShape, resDist, A.Indices(), g);

    PartialReduceScatterRedist(B, A, rsIndex);

    Print(B, "B after prs redist");
}

template<typename T>
void
TestA2ADIRedist(DistTensor<T>& A, const std::pair<Index, Index>& a2aIndices, const std::pair<ModeArray, ModeArray >& commGroups, const TensorDistribution& resDist){
#ifndef RELEASE
    CallStackEntry entry("TestA2ADIRedist");
#endif

    const Grid& g = A.Grid();

    DistTensor<T> B(A.Shape(), resDist, A.Indices(), g);

    AllToAllDoubleIndexRedist(B, A, a2aIndices, commGroups);

    Print(B, "B after a2a redist");
}

template<typename T>
void
Set(DistTensor<T>& A)
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
TensorDistribution
DetermineResultingDistributionAG(const DistTensor<T>& A, Index agIndex){
    TensorDistribution ret;
    const TensorDistribution ADist = A.TensorDist();
    ret = ADist;
    ret[A.ModeOfIndex(agIndex)].clear();
    return ret;
}

template<typename T>
TensorDistribution
DetermineResultingDistributionLocal(const DistTensor<T>& A, Unsigned lIndex, const ModeArray& gridRedistModes){
    TensorDistribution ret = A.TensorDist();
    Mode lMode = A.ModeOfIndex(lIndex);
    ret[lMode].insert(ret[lMode].end(), gridRedistModes.begin(), gridRedistModes.end());
    return ret;
}

template<typename T>
TensorDistribution
DetermineResultingDistributionRS(const DistTensor<T>& A, Unsigned rIndex, Unsigned sIndex){
    TensorDistribution ret;
    const TensorDistribution ADist = A.TensorDist();
    ret = ADist;
    ModeDistribution& scatterIndexDist = ret[A.ModeOfIndex(sIndex)];
    ModeDistribution& reduceIndexDist = ret[A.ModeOfIndex(rIndex)];
    scatterIndexDist.insert(scatterIndexDist.end(), reduceIndexDist.begin(), reduceIndexDist.end());
    ret.erase(ret.begin() + A.ModeOfIndex(rIndex));
    return ret;
}

template<typename T>
TensorDistribution
DetermineResultingDistributionPRS(const DistTensor<T>& A, Unsigned rsIndex){
    TensorDistribution ret;
    const TensorDistribution ADist = A.TensorDist();
    ret = ADist;
    return ret;
}

template<typename T>
TensorDistribution
DetermineResultingDistributionA2ADI(const DistTensor<T>& A, const std::pair<Index, Index>& a2aIndices, const std::pair<ModeArray, ModeArray >& commGroups){

    const Index a2aIndex1 = a2aIndices.first;
    const Index a2aIndex2 = a2aIndices.second;

    const Mode a2aIndex1Mode = A.ModeOfIndex(a2aIndex1);
    const Mode a2aIndex2Mode = A.ModeOfIndex(a2aIndex2);

    const ModeArray a2aIndex1CommGroup = commGroups.first;
    const ModeArray a2aIndex2CommGroup = commGroups.second;

    TensorDistribution newDist = A.TensorDist();

    newDist[a2aIndex1Mode].erase(newDist[a2aIndex1Mode].end() - a2aIndex1CommGroup.size(), newDist[a2aIndex1Mode].end());
    newDist[a2aIndex2Mode].erase(newDist[a2aIndex2Mode].end() - a2aIndex2CommGroup.size(), newDist[a2aIndex2Mode].end());

    newDist[a2aIndex1Mode].insert(newDist[a2aIndex1Mode].end(), a2aIndex2CommGroup.begin(), a2aIndex2CommGroup.end());
    newDist[a2aIndex2Mode].insert(newDist[a2aIndex2Mode].end(), a2aIndex1CommGroup.begin(), a2aIndex1CommGroup.end());

    return newDist;
}

template<typename T>
std::vector<PTest>
CreatePTests(const DistTensor<T>& A, const Params& args){
    Unsigned i;
    std::vector<PTest> ret;

    const Unsigned order = A.Order();
    const IndexArray indices = A.Indices();

    for(i = 0; i < order; i++){
        const Index indexToRedist = indices[i];
        ModeDistribution indexDist = A.IndexDist(indexToRedist);
        std::sort(indexDist.begin(), indexDist.end());
        do{
            PTest test(indexToRedist, indexDist);
            ret.push_back(test);
        } while(std::next_permutation(indexDist.begin(), indexDist.end()));
    }
    return ret;
}

template<typename T>
std::vector<AGTest >
CreateAGTests(const DistTensor<T>& A, const Params& args){
    Unsigned i;
    std::vector<AGTest > ret;

    const Unsigned order = A.Order();
    const IndexArray indices = A.Indices();
    const TensorDistribution distA = A.TensorDist();

    for(i = 0; i < order; i++){
        if(distA[i].size() == 0)
            continue;
        const Index indexToRedist = indices[i];
        AGTest test(indexToRedist, DetermineResultingDistributionAG(A, indexToRedist));
        ret.push_back(test);
    }

//    AGTest test(1, DetermineResultingDistributionAG(A, 1));
//    ret.push_back(test);
    return ret;
}

template<typename T>
std::vector<LTest>
CreateLTests(const DistTensor<T>& A, const Params& args){
    Unsigned i, j, k, l;
    std::vector<LTest> ret;
    const tmen::GridView& gv = A.GridView();

    const Unsigned order = A.Order();
    const IndexArray indices = A.Indices();
    const TensorDistribution tDist = A.TensorDist();
    ModeArray freeModes = gv.FreeModes();

    //NOTE: Just picking up to 2 modes to redistribute along
    for(i = 0; i < tDist.size(); i++){
        for(j = 0; j < freeModes.size(); j++){
            for(k = 0; k < freeModes.size(); k++){
                TensorDistribution resDist(tDist.begin(), tDist.end());
                ModeDistribution newDist = resDist[i];
                newDist.insert(newDist.end(), freeModes[j]);
                if(j != k){
                    newDist.insert(newDist.end(), freeModes[k]);
                }
                LTest thisTest(indices[i], newDist);
                ret.push_back(thisTest);
            }
        }
    }

//    ModeDistribution resDist(1);
//    resDist[0] = 2;
//    LTest test(1, resDist);
//    ret.push_back(test);

    return ret;
}

template<typename T>
std::vector<PRSTest >
CreatePRSTests(const DistTensor<T>& A, const Params& args){
    Unsigned i;
    std::vector<PRSTest> ret;

    const Unsigned order = A.Order();
    const IndexArray indices = A.Indices();

    for(i = 0; i < order; i++){
        const Index indexToReduceScatter = indices[i];
        PRSTest test(indexToReduceScatter, DetermineResultingDistributionPRS(A, indexToReduceScatter));
        ret.push_back(test);
    }
    return ret;
}

template<typename T>
std::vector<RSTest >
CreateRSTests(const DistTensor<T>& A, const Params& args){
    Unsigned i, j;
    std::vector<RSTest> ret;

    const Unsigned order = A.Order();
    const IndexArray indices = A.Indices();

    const GridView gv = A.GridView();

    for(i = 0; i < order; i++){
        for(j = 0; j < order; j++){
            if(i == j)
                continue;
            const Index indexToReduce = indices[i];
            const Mode reduceMode = A.ModeOfIndex(indexToReduce);
            const Index indexToScatter = indices[j];

            if(A.Dimension(reduceMode) > gv.Dimension(A.ModeOfIndex(reduceMode)))
                continue;
            std::pair<Index, Index> redistIndices(i, j);
            RSTest test(redistIndices, DetermineResultingDistributionRS(A, indexToReduce, indexToScatter));
            ret.push_back(test);
        }
    }

//    std::pair<int, int> redistIndices(1,0);
//    RSTest test(redistIndices, DetermineResultingDistributionRS(A, 1, 0));
//    ret.push_back(test);
    return ret;
}

template<typename T>
std::vector<A2ADITest>
CreateA2ADITests(const DistTensor<T>& A, const Params& args){
    std::vector<A2ADITest> ret;

    Unsigned i, j, k, l;
    const Unsigned order = A.Order();
    const IndexArray indices = A.Indices();

    for(i = 0; i < order; i++){
        for(j = i+1; j < order; j++){
            //We can make a test
            std::pair<Index, Index> testIndices(indices[i], indices[j]);

            ModeDistribution mode1Dist = A.ModeDist(i);
            ModeDistribution mode2Dist = A.ModeDist(j);

            //Pick the groups of modes to communicate over with the all to all

            for(k = 0; k <= mode1Dist.size(); k++){
                for(l = 0; l <= mode2Dist.size(); l++){
                    ModeArray commGroup1(mode1Dist.end() - k, mode1Dist.end());
                    ModeArray commGroup2(mode2Dist.end() - l, mode2Dist.end());

                    //No communication happens for this "redistribution"
                    if(commGroup1.size() == 0 && commGroup2.size() == 0)
                        continue;
                    std::pair<ModeArray, ModeArray > commGroups(commGroup1, commGroup2);
                    TensorDistribution resDist = DetermineResultingDistributionA2ADI(A, testIndices, commGroups);

                    std::pair<std::pair<Index, Index>, std::pair<ModeArray, ModeArray > > testParams(testIndices, commGroups);
                    A2ADITest test(testParams, resDist);

                    ret.push_back(test);
                }
            }
        }
    }

//    std::pair<Index, Index> a2aModes(0,1);
//    ModeArray mode1CommGroup(1);
//    mode1CommGroup[0] = 0;
//    ModeArray mode2CommGroup;
//    std::pair<ModeArray, ModeArray > modeCommGroups(mode1CommGroup, mode2CommGroup);
//    std::pair<std::pair<Index, Index>, std::pair<ModeArray, ModeArray > > params(a2aModes, modeCommGroups);
//    TensorDistribution  tdist = StringToTensorDist("[(1),(0),()]");
//    A2ADITest test(params, tdist);
//    ret.push_back(test);

    return ret;
}

template<typename T>
void
DistTensorTest( const Params& args, const Grid& g )
{
#ifndef RELEASE
    CallStackEntry entry("DistTensorTest");
#endif
    Unsigned i;
    const Int commRank = mpi::CommRank( mpi::COMM_WORLD );
    const Unsigned order = args.tensorShape.size();
    IndexArray indices(order);
    for(i = 0 ; i < indices.size(); i++)
        indices[i] = i;

    DistTensor<T> A(args.tensorShape, args.tensorDist, indices, g);
    IndexArray testSet(order);
    for(i = 0; i < testSet.size(); i++)
        testSet[i] = indices[order - 1 -i];
    A.SetIndices(testSet);
    Set(A);

    std::vector<AGTest> agTests = CreateAGTests(A, args);
    std::vector<LTest> lTests = CreateLTests(A, args);
    std::vector<RSTest> rsTests = CreateRSTests(A, args);
    std::vector<PRSTest> prsTests = CreatePRSTests(A, args);
    std::vector<PTest> pTests = CreatePTests(A, args);
    std::vector<A2ADITest> a2aTests = CreateA2ADITests(A, args);

//    if(commRank == 0){
//        printf("Performing AllGather tests\n");
//    }
//    for(i = 0; i < agTests.size(); i++){
//        AGTest thisTest = agTests[i];
//        Index agIndex = thisTest.first;
//        TensorDistribution resDist = thisTest.second;
//        if(commRank == 0){
//            printf("Allgathering index %d with resulting distribution %s\n", agIndex, (tmen::TensorDistToString(resDist)).c_str());
//        }
//        TestAGRedist(A, agIndex, resDist);
//    }

    if(commRank == 0){
        printf("Performing Local redist tests\n");
    }
    for(i = 0; i < lTests.size(); i++){
        LTest thisTest = lTests[i];
        Index lIndex = thisTest.first;
        ModeDistribution resDist = thisTest.second;

        TestLRedist(A, lIndex, resDist);
    }

//    if(commRank == 0){
//        printf("Performing PartialReduceScatter tests\n");
//    }
//    for(i = 0; i < prsTests.size(); i++){
//        PRSTest thisTest = prsTests[i];
//        Index rsIndex = thisTest.first;
//        TensorDistribution resDist = thisTest.second;
//        if(commRank == 0){
//            printf("Partial reduce-scattering index %d with resulting distribution %s\n", rsIndex, (tmen::TensorDistToString(resDist)).c_str());
//        }
//        TestPRSRedist(A, rsIndex, resDist);
//    }


//    if(commRank == 0){
//        printf("Performing ReduceScatter tests\n");
//    }
//    for(i = 0; i < rsTests.size(); i++){
//        RSTest thisTest = rsTests[i];
//        Index reduceIndex = thisTest.first.first;
//        Index scatterIndex = thisTest.first.second;
//        TensorDistribution resDist = thisTest.second;
//
//        if(commRank == 0){
//        printf(
//                "Reducing index %d, scattering index %d, with resulting distribution %s\n",
//                reduceIndex, scatterIndex,
//                (tmen::TensorDistToString(resDist)).c_str());
//        }
//        TestRSRedist(A, reduceIndex, scatterIndex, resDist);
//    }

//    if(commRank == 0){
//        printf("Performing Permutation tests\n");
//    }
//    for(i = 0; i <= pTests.size(); i++){
//        PTest thisTest = pTests[1];
//        Index permuteIndex = thisTest.first;
//        ModeDistribution resDist = thisTest.second;
//
//        if(commRank == 0){
//            printf("Permuting index %d with resulting index distribution %s\n", permuteIndex, (tmen::ModeDistToString(resDist)).c_str());
//        }
//        TestPRedist(A, permuteIndex, resDist);
//    }

//    if(commRank == 0){
//        printf("Performing All-to-all (double index) tests\n");
//    }
//    for(i = 0; i < a2aTests.size(); i++){
//        A2ADITest thisTest = a2aTests[i];
//        std::pair<Mode, Mode> indices = thisTest.first.first;
//        std::pair<ModeArray, ModeArray > commGroups = thisTest.first.second;
//        TensorDistribution resDist = thisTest.second;
//
//        if(commRank == 0){
//            printf("Performing all-to-all involving indices (%d, %d) from distribution %s to distribution %s\n", indices.first, indices.second, (tmen::TensorDistToString(A.TensorDist())).c_str(), (tmen::TensorDistToString(resDist)).c_str());
//        }
//        TestA2ADIRedist(A, indices, commGroups, resDist);
//    }
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

        if(commRank == 0 && args.nProcs != commSize){
            std::cerr << "program not started with correct number of processes\n";
            Usage();
            throw ArgException();
        }

        if(commRank == 0){
            printf("Creating %d", args.gridShape[0]);
            for(i = 1; i < args.gridOrder; i++)
                printf(" x %d", args.gridShape[i]);
            printf(" grid\n");

            printf("Creating [%d", args.tensorShape[0]);
            for(i = 1; i < args.tenOrder; i++)
                printf(", %d", args.tensorShape[i]);
            printf("] tensor\n");
        }

        const Grid g( comm, args.gridOrder, args.gridShape );

        if( commRank == 0 )
        {
            std::cout << "------------------" << std::endl
                      << "Testing with ints:" << std::endl
                      << "------------------" << std::endl;
        }
        DistTensorTest<int>( args, g );

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
