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

typedef std::pair< Mode, ModeDistribution> PTest;
typedef std::pair< std::pair<Mode, ModeArray>, TensorDistribution> AGTest;
typedef std::pair< std::pair<Mode, ModeArray>, TensorDistribution> GTOTest;
typedef std::pair< Mode, ModeDistribution> LTest;
typedef std::pair< Mode, TensorDistribution> PRSTest;
typedef std::pair< std::pair<Mode, Mode>, TensorDistribution> RSTest;
typedef std::pair< Mode, TensorDistribution > RTOTest;
typedef std::pair< std::pair<std::pair<Mode, Mode>, std::pair<ModeArray, ModeArray > >, TensorDistribution> A2ADMTest;

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


    if(args.tensorDist.size() != args.tensorShape.size() + 1){
        std::cerr << "Tensor distribution must be of same order as tensor\n";
        Usage();
        throw ArgException();
    }
}

template<typename T>
void
TestGTORedist( DistTensor<T>& A, const Mode& gMode, const ModeArray& gridModes, const TensorDistribution& resDist)
{
#ifndef RELEASE
    CallStackEntry entry("TestGTORedist");
#endif
    const Int commRank = mpi::CommRank( mpi::COMM_WORLD );
    const Grid& g = A.Grid();

    DistTensor<T> B(A.Shape(), resDist, g);

    if(commRank == 0){
        printf("Gathering to one mode %d: %s <-- %s\n", gMode, (tmen::TensorDistToString(B.TensorDist())).c_str(), (tmen::TensorDistToString(A.TensorDist())).c_str());
    }
    B.GatherToOneRedistFrom(A, gMode, gridModes);
    Print(B, "B after gather-to-one redist");
}

template<typename T>
void
TestRTORedist( DistTensor<T>& A, const Mode& rMode, const TensorDistribution& resDist)
{
#ifndef RELEASE
    CallStackEntry entry("TestRTORedist");
#endif
    const Int commRank = mpi::CommRank( mpi::COMM_WORLD );
    const Grid& g = A.Grid();

    ObjShape shapeB = A.Shape();
    TensorDistribution distB = A.TensorDist();
    shapeB.erase(shapeB.begin() + rMode);
    distB.erase(distB.begin() + rMode);

    DistTensor<T> B(shapeB, distB, g);

    if(commRank == 0){
        printf("Reducing to one mode %d: %s <-- %s\n", rMode, (tmen::TensorDistToString(B.TensorDist())).c_str(), (tmen::TensorDistToString(A.TensorDist())).c_str());
    }
    B.ReduceToOneRedistFrom(A, rMode);
    Print(B, "B after reduce-to-one redist");
}

template<typename T>
void
TestPRedist( DistTensor<T>& A, Mode pMode, const ModeDistribution& resDist )
{
#ifndef RELEASE
    CallStackEntry entry("TestPRedist");
#endif
    const Int commRank = mpi::CommRank( mpi::COMM_WORLD );
    //const int order = A.Order();
    const Grid& g = A.Grid();

    TensorDistribution BDist = A.TensorDist();
    BDist[pMode] = resDist;

    DistTensor<T> B(A.Shape(), BDist, g);
    //Print(B, "B before permute redist");

    if(commRank == 0){
        printf("Permuting mode %d: %s <-- %s\n", pMode, (tmen::TensorDistToString(B.TensorDist())).c_str(), (tmen::TensorDistToString(A.TensorDist())).c_str());
    }
    B.PermutationRedistFrom(A, pMode, resDist);
    Print(B, "B after permute redist");
}

template<typename T>
void
TestAGRedist( DistTensor<T>& A, Mode agMode, const ModeArray& redistModes, const TensorDistribution& resDist )
{
#ifndef RELEASE
    CallStackEntry entry("TestAGRedist");
#endif
    const Int commRank = mpi::CommRank( mpi::COMM_WORLD );
    //const int order = A.Order();
    const Grid& g = A.Grid();

    DistTensor<T> B(A.Shape(), resDist, g);

    if(commRank == 0){
        printf("Allgathering mode %d : %s <-- %s\n", agMode, (tmen::TensorDistToString(B.TensorDist())).c_str(), (tmen::TensorDistToString(A.TensorDist())).c_str());
    }

    B.AllGatherRedistFrom(A, agMode, redistModes);
    Print(B, "B after ag redist");
}

template<typename T>
void
TestLRedist( DistTensor<T>& A, Mode lMode, const ModeDistribution& resDist )
{
#ifndef RELEASE
    CallStackEntry entry("TestAGRedist");
#endif
    const Int commRank = mpi::CommRank( mpi::COMM_WORLD );
    const Grid& g = A.Grid();

    TensorDistribution distB = A.TensorDist();
    distB[lMode] = resDist;

    DistTensor<T> B(A.Shape(), distB, g);

    ModeDistribution lModeDist = A.ModeDist(lMode);

    if(commRank == 0){
        printf("Locally redistributing mode %d: %s <-- %s\n", lMode, (tmen::TensorDistToString(B.TensorDist())).c_str(), (tmen::TensorDistToString(A.TensorDist())).c_str());
    }

    Print(A, "A before local redist");
    ModeArray gridRedistModes(resDist.begin() + lModeDist.size(), resDist.end());
    B.LocalRedistFrom(A, lMode, gridRedistModes);
    Print(B, "B after local redist");
}

template<typename T>
void
TestRSRedist(DistTensor<T>& A, Mode rMode, Mode sMode, const TensorDistribution& resDist)
{
#ifndef RELEASE
    CallStackEntry entry("TestRSRedist");
#endif
    const Int commRank = mpi::CommRank( mpi::COMM_WORLD );
    const Grid& g = A.Grid();
    const GridView gv = A.GetGridView();

    ObjShape BShape = A.Shape();
    BShape[rMode] = 1;
    DistTensor<T> B(BShape, resDist, g);

    Print(A, "A before rs redist");
    if(commRank == 0){
        printf("Reducing mode %d and scattering mode %d: %s <-- %s\n", rMode, sMode, (tmen::TensorDistToString(B.TensorDist())).c_str(), (tmen::TensorDistToString(A.TensorDist())).c_str());
    }

    B.ReduceScatterRedistFrom(A, rMode, sMode);

    Print(B, "B after rs redist");
}

template<typename T>
void
TestPRSRedist(DistTensor<T>& A, Mode rsMode, const TensorDistribution& resDist)
{
#ifndef RELEASE
    CallStackEntry entry("TestPRSRedist");
#endif
    const Int commRank = mpi::CommRank( mpi::COMM_WORLD );
    const Grid& g = A.Grid();
    const GridView gv = A.GetGridView();

    ObjShape BShape = A.Shape();
    BShape[rsMode] = Max(1, tmen::MaxLength(A.Dimension(rsMode), gv.Dimension(rsMode)));

    DistTensor<T> B(BShape, resDist, g);

    if(commRank == 0){
        printf("Partially reducing mode %d: %s <-- %s\n", rsMode, (tmen::TensorDistToString(B.TensorDist())).c_str(), (tmen::TensorDistToString(A.TensorDist())).c_str());
    }

    B.PartialReduceScatterRedistFrom(A, rsMode);

    Print(B, "B after prs redist");
}

template<typename T>
void
TestA2ADMRedist(DistTensor<T>& A, const std::pair<Mode, Mode>& a2aModes, const std::pair<ModeArray, ModeArray >& commGroups, const TensorDistribution& resDist){
#ifndef RELEASE
    CallStackEntry entry("TestA2ADMRedist");
#endif
    const Int commRank = mpi::CommRank( mpi::COMM_WORLD );
    const Grid& g = A.Grid();

    DistTensor<T> B(A.Shape(), resDist, g);

    if(commRank == 0){
        printf("All-to-alling modes %d and %d: %s <-- %s\n", a2aModes.first, a2aModes.second, (tmen::TensorDistToString(B.TensorDist())).c_str(), (tmen::TensorDistToString(A.TensorDist())).c_str());
    }

    B.AllToAllDoubleModeRedistFrom(A, a2aModes, commGroups);

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
DetermineResultingDistributionRTO(const DistTensor<T>& A, Mode rMode){
    const TensorDistribution ADist = A.TensorDist();
    TensorDistribution ret(ADist);
    ret.erase(ret.begin() + rMode);
    return ret;
}

template<typename T>
TensorDistribution
DetermineResultingDistributionAG(const DistTensor<T>& A, Mode agMode, const ModeArray& redistModes){
    const TensorDistribution ADist = A.TensorDist();
    TensorDistribution ret(ADist);
    ret[agMode].erase(ret[agMode].begin() + ret[agMode].size() - redistModes.size(), ret[agMode].end());
    return ret;
}

template<typename T>
TensorDistribution
DetermineResultingDistributionGTO(const DistTensor<T>& A, Mode gMode, const ModeArray& redistModes){
    const TensorDistribution ADist = A.TensorDist();
    TensorDistribution ret(ADist);
    ret[gMode].erase(ret[gMode].begin() + ret[gMode].size() - redistModes.size(), ret[gMode].end());
    ret[A.Order()].insert(ret[A.Order()].end(), redistModes.begin(), redistModes.end());
    return ret;
}

template<typename T>
TensorDistribution
DetermineResultingDistributionLocal(const DistTensor<T>& A, Mode lMode, const ModeArray& gridRedistModes){
    TensorDistribution ret = A.TensorDist();
    ret[lMode].insert(ret[lMode].end(), gridRedistModes.begin(), gridRedistModes.end());
    return ret;
}

template<typename T>
TensorDistribution
DetermineResultingDistributionRS(const DistTensor<T>& A, Mode rMode, Mode sMode){
    TensorDistribution ret;
    const TensorDistribution ADist = A.TensorDist();
    ret = ADist;
    ModeDistribution& sModeDist = ret[sMode];
    ModeDistribution& rModeDist = ret[rMode];
    sModeDist.insert(sModeDist.end(), rModeDist.begin(), rModeDist.end());
    ret[rMode].clear();
    return ret;
}

template<typename T>
TensorDistribution
DetermineResultingDistributionPRS(const DistTensor<T>& A, Mode rsMode){
    TensorDistribution ret;
    const TensorDistribution ADist = A.TensorDist();
    ret = ADist;
    return ret;
}

template<typename T>
TensorDistribution
DetermineResultingDistributionA2ADM(const DistTensor<T>& A, const std::pair<Mode, Mode>& a2aModes, const std::pair<ModeArray, ModeArray >& commGroups){
    const Mode a2aMode1 = a2aModes.first;
    const Mode a2aMode2 = a2aModes.second;

    const ModeArray a2aMode1CommGroup = commGroups.first;
    const ModeArray a2aMode2CommGroup = commGroups.second;

    TensorDistribution newDist = A.TensorDist();

    newDist[a2aMode1].erase(newDist[a2aMode1].end() - a2aMode1CommGroup.size(), newDist[a2aMode1].end());
    newDist[a2aMode2].erase(newDist[a2aMode2].end() - a2aMode2CommGroup.size(), newDist[a2aMode2].end());

    newDist[a2aMode1].insert(newDist[a2aMode1].end(), a2aMode2CommGroup.begin(), a2aMode2CommGroup.end());
    newDist[a2aMode2].insert(newDist[a2aMode2].end(), a2aMode1CommGroup.begin(), a2aMode1CommGroup.end());

    return newDist;
}

template<typename T>
std::vector<PTest>
CreatePTests(const DistTensor<T>& A, const Params& args){
    Unsigned i;
    std::vector<PTest> ret;

    const Unsigned order = A.Order();

    for(i = 0; i < order; i++){
        const Mode modeToRedist = i;
        ModeDistribution modeDist = A.ModeDist(modeToRedist);
        std::sort(modeDist.begin(), modeDist.end());
        do{
            PTest test(modeToRedist, modeDist);
            ret.push_back(test);
        } while(std::next_permutation(modeDist.begin(), modeDist.end()));
    }
    return ret;
}

template<typename T>
std::vector<GTOTest >
CreateGTOTests(const DistTensor<T>& A, const Params& args){
    Unsigned i, j;
    std::vector<GTOTest > ret;

    const Unsigned order = A.Order();
    const TensorDistribution distA = A.TensorDist();

    for(i = 0; i < order; i++){
        if(distA[i].size() == 0)
            continue;
        const Mode modeToRedist = i;
        const ModeDistribution modeDist = A.ModeDist(modeToRedist);
        for(j = 0; j <= modeDist.size(); j++){
            const ModeArray suffix(modeDist.begin() + j, modeDist.end());
            std::pair<Mode, ModeArray> testPair(modeToRedist, suffix);
            TensorDistribution resDist = DetermineResultingDistributionGTO(A, modeToRedist, suffix);
            GTOTest test(testPair, resDist);
            ret.push_back(test);
        }
    }

//    AGTest test(1, DetermineResultingDistributionAG(A, 1));
//    ret.push_back(test);
    return ret;
}

template<typename T>
std::vector<AGTest >
CreateAGTests(const DistTensor<T>& A, const Params& args){
    Unsigned i, j;
    std::vector<AGTest > ret;

    const Unsigned order = A.Order();
    const TensorDistribution distA = A.TensorDist();

    for(i = 0; i < order; i++){
        if(distA[i].size() == 0)
            continue;
        const Mode modeToRedist = i;
        const ModeDistribution modeDist = A.ModeDist(modeToRedist);
        for(j = 0; j <= modeDist.size(); j++){
            const ModeArray suffix(modeDist.begin() + j, modeDist.end());
            std::pair<Mode, ModeArray> testPair(modeToRedist, suffix);
            TensorDistribution resDist = DetermineResultingDistributionAG(A, modeToRedist, suffix);
            AGTest test(testPair, resDist);
            ret.push_back(test);
        }
    }

//    AGTest test(1, DetermineResultingDistributionAG(A, 1));
//    ret.push_back(test);
    return ret;
}

template<typename T>
std::vector<LTest>
CreateLTests(const DistTensor<T>& A, const Params& args){
    Unsigned i, j, k;
    std::vector<LTest> ret;
    const tmen::GridView& gv = A.GetGridView();

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
                LTest thisTest(i, newDist);
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

    for(i = 0; i < order; i++){
        const Mode rsMode = i;
        PRSTest test(rsMode, DetermineResultingDistributionPRS(A, rsMode));
        ret.push_back(test);
    }
    return ret;
}

template<typename T>
std::vector<RTOTest>
CreateRTOTests(const DistTensor<T>& A, const Params& args){
    Unsigned i;
    std::vector<RTOTest> ret;

    const Unsigned order = A.Order();
    const GridView gv = A.GetGridView();

    for(i = 0; i < order; i++){
        const Mode rMode = i;
        TensorDistribution retDist = A.TensorDist();
        retDist.erase(retDist.begin() + rMode);
        RTOTest test(rMode, retDist);
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

    const GridView gv = A.GetGridView();

    for(i = 0; i < order; i++){
        for(j = 0; j < order; j++){
            if(i == j)
                continue;
            const Mode rMode = i;
            const Mode sMode = j;

            std::pair<Mode, Mode> redistModes(i, j);
            RSTest test(redistModes, DetermineResultingDistributionRS(A, rMode, sMode));
            ret.push_back(test);
        }
    }

//    std::pair<int, int> redistIndices(1,0);
//    RSTest test(redistIndices, DetermineResultingDistributionRS(A, 1, 0));
//    ret.push_back(test);
    return ret;
}

template<typename T>
std::vector<A2ADMTest>
CreateA2ADMTests(const DistTensor<T>& A, const Params& args){
    std::vector<A2ADMTest> ret;

    Unsigned i, j, k, l;
    const Unsigned order = A.Order();

    for(i = 0; i < order; i++){
        for(j = i+1; j < order; j++){
            //We can make a test
            std::pair<Mode, Mode> testModes(i, j);

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
                    TensorDistribution resDist = DetermineResultingDistributionA2ADM(A, testModes, commGroups);

                    std::pair<std::pair<Mode, Mode>, std::pair<ModeArray, ModeArray > > testParams(testModes, commGroups);
                    A2ADMTest test(testParams, resDist);

                    ret.push_back(test);
                }
            }
        }
    }

//    std::pair<Mode, Mode> a2aModes(0,1);
//    ModeArray mode1CommGroup(1);
//    mode1CommGroup[0] = 0;
//    ModeArray mode2CommGroup;
//    std::pair<ModeArray, ModeArray > modeCommGroups(mode1CommGroup, mode2CommGroup);
//    std::pair<std::pair<Mode, Mode>, std::pair<ModeArray, ModeArray > > params(a2aModes, modeCommGroups);
//    TensorDistribution  tdist = StringToTensorDist("[(1),(0),()]");
//    A2ADMTest test(params, tdist);
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

    DistTensor<T> A(args.tensorShape, args.tensorDist, g);
    Set(A);

    std::vector<AGTest> agTests = CreateAGTests(A, args);
    std::vector<GTOTest> gtoTests = CreateGTOTests(A, args);
    std::vector<LTest> lTests = CreateLTests(A, args);
    std::vector<RSTest> rsTests = CreateRSTests(A, args);
    std::vector<PRSTest> prsTests = CreatePRSTests(A, args);
    std::vector<PTest> pTests = CreatePTests(A, args);
    std::vector<A2ADMTest> a2aTests = CreateA2ADMTests(A, args);
    std::vector<RTOTest> rtoTests = CreateRTOTests(A, args);

//    if(commRank == 0){
//        printf("Performing AllGather tests\n");
//    }
//    for(i = 0; i < agTests.size(); i++){
//        AGTest thisTest = agTests[i];
//        Mode agMode = thisTest.first.first;
//        ModeArray redistModes = thisTest.first.second;
//        TensorDistribution resDist = thisTest.second;
//
//        TestAGRedist(A, agMode, redistModes, resDist);
//    }
//
    if(commRank == 0){
        printf("Performing Gather-to-one tests\n");
    }
    for(i = 0; i < gtoTests.size(); i++){
        GTOTest thisTest = gtoTests[i];
        Mode gMode = thisTest.first.first;
        ModeArray redistModes = thisTest.first.second;
        TensorDistribution resDist = thisTest.second;

        TestGTORedist(A, gMode, redistModes, resDist);
    }
//
//    if(commRank == 0){
//        printf("Performing Local redist tests\n");
//    }
//    for(i = 0; i < lTests.size(); i++){
//        LTest thisTest = lTests[i];
//        Mode lMode = thisTest.first;
//        ModeDistribution resDist = thisTest.second;
//
//        TestLRedist(A, lMode, resDist);
//    }

//    if(commRank == 0){
//            printf("Performing ReduceToOne tests\n");
//    }
//    for(i = 0; i < rtoTests.size(); i++){
//        RTOTest thisTest = rtoTests[i];
//        Mode rsMode = thisTest.first;
//        TensorDistribution resDist = thisTest.second;
//
//        TestRTORedist(A, rsMode, resDist);
//    }

//    if(commRank == 0){
//        printf("Performing PartialReduceScatter tests\n");
//    }
//    for(i = 0; i < prsTests.size(); i++){
//        PRSTest thisTest = prsTests[i];
//        Mode rsMode = thisTest.first;
//        TensorDistribution resDist = thisTest.second;
//
//        TestPRSRedist(A, rsMode, resDist);
//    }
//
//
//    if(commRank == 0){
//        printf("Performing ReduceScatter tests\n");
//    }
//    for(i = 0; i < rsTests.size(); i++){
//        RSTest thisTest = rsTests[i];
//        Mode rMode = thisTest.first.first;
//        Mode sMode = thisTest.first.second;
//        TensorDistribution resDist = thisTest.second;
//
//        TestRSRedist(A, rMode, sMode, resDist);
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
//        printf("Performing All-to-all (double index) tests\n");
//    }
//    for(i = 0; i < a2aTests.size(); i++){
//        A2ADMTest thisTest = a2aTests[i];
//        std::pair<Mode, Mode> a2aModes = thisTest.first.first;
//        std::pair<ModeArray, ModeArray > commGroups = thisTest.first.second;
//        TensorDistribution resDist = thisTest.second;
//
//        TestA2ADMRedist(A, a2aModes, commGroups, resDist);
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

        const Grid g( comm, args.gridShape );

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
