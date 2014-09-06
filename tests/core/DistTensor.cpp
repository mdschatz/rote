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
typedef std::pair< std::pair<ModeArray, std::vector<ModeArray> >, TensorDistribution> AGGTest;
typedef std::pair< std::pair<Mode, ModeArray>, TensorDistribution> GTOTest;
typedef std::pair< std::pair<ModeArray, std::vector<ModeArray>>, TensorDistribution> GTOGTest;
typedef std::pair< Mode, ModeDistribution> LTest;
typedef std::pair< std::pair<ModeArray, std::vector<ModeArray>>, TensorDistribution> LGTest;
typedef std::pair< Mode, TensorDistribution> PRSTest;
typedef std::pair< std::pair<ModeArray, ModeArray>, TensorDistribution> RSGTest;
typedef std::pair< std::pair<Mode, Mode>, TensorDistribution> RSTest;
typedef std::pair< Mode, TensorDistribution > RTOTest;
typedef std::pair< ModeArray, TensorDistribution > RTOGTest;
typedef std::pair< std::pair<std::pair<Mode, Mode>, std::pair<ModeArray, ModeArray > >, TensorDistribution> A2ADMTest;
typedef std::pair< std::pair<std::pair<ModeArray, ModeArray>, std::vector<ModeArray > >, TensorDistribution> A2ATest;



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

void AllCombinationsHelper(const ModeArray& input, Unsigned arrPos, Unsigned k, Unsigned prevLoc, ModeArray& piece, std::vector<ModeArray>& combinations){
    Unsigned i;

    if(arrPos == k){
        combinations.push_back(piece);
    }
    else{
        for(i = prevLoc+1; i < input.size(); i++){
            ModeArray newPiece = piece;
            newPiece[arrPos] = input[i];
            AllCombinationsHelper(input, arrPos+1, k, i, newPiece, combinations);
        }
    }
}

std::vector<ModeArray> AllCombinations(const ModeArray& input, Unsigned k){
    Unsigned i;
    ModeArray start(k);
    std::vector<ModeArray> ret;
    for(i = 0; i < input.size(); i++){
        ModeArray newPiece(k);
        newPiece[0] = input[i];
        AllCombinationsHelper(input, 1, k, i, newPiece, ret);
    }

    return ret;
}

template<typename T>
void
Set(Tensor<T>& A)
{
    Unsigned order = A.Order();
    Location loc(order, 0);

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
void
Set(DistTensor<T>& A)
{
    Unsigned order = A.Order();
    Location loc(order, 0);

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
bool CheckResult(const DistTensor<T>& A){
#ifndef RELEASE
    CallStackEntry entry("CheckResult");
#endif
//    printf("In CheckResult\n");
    mpi::Barrier(mpi::COMM_WORLD);
    const Int commRank = mpi::CommRank( mpi::COMM_WORLD );
    const ObjShape globalShape = A.Shape();
    Tensor<T> check(globalShape);
    Set(check);

    const Unsigned order = A.Order();
    const T* checkBuf = check.LockedBuffer();
    const T* thisBuf = A.LockedBuffer();
    const std::vector<Unsigned> myShifts = A.ModeShifts();
    const tmen::Grid& g = A.Grid();
    const Location myGridLoc = g.Loc();
    const Location myGridViewLoc = A.GridViewLoc();
    const ObjShape localShape = A.LocalShape();
    Tensor<T> localTensor(localShape);
    localTensor = A.LockedTensor();
    const TensorDistribution dist = A.TensorDist();

    //Check that all entries are what they should be
    const std::vector<Unsigned> strides = A.ModeStrides();
    Unsigned res = 1;

    bool participating = !AnyPositiveElem(FilterVector(myGridLoc, dist[order]));
    if(participating && !A.Participating())
        res = 1;

    if(!participating){
        if(localTensor.MemorySize() > 1)
            res = 1;
    }else{
        Location checkLoc = myGridViewLoc;
        Unsigned checkLinLoc = Loc2LinearLoc(checkLoc, globalShape);
        Location distLoc(order, 0);
        Unsigned distLinLoc = 0;

        Unsigned ptr = 0;
        bool stop = !ElemwiseLessThan(checkLoc, globalShape);


        if(stop && localTensor.MemorySize() > 1){
            res = 0;
        }

        while(!stop){

            checkLinLoc = Loc2LinearLoc(checkLoc, globalShape);
            distLinLoc = Loc2LinearLoc(distLoc, localShape);

//            PrintVector(checkLoc, "checkLoc");
//            PrintVector(distLoc, "distLoc");
//            std::cout << "tensor should have: " << checkBuf[checkLinLoc] << ", has: " << thisBuf[distLinLoc] << std::endl;
            if(checkBuf[checkLinLoc] != thisBuf[distLinLoc]){
                res = 0;
                break;
            }

            //Update
            distLoc[ptr]++;
            checkLoc[ptr] += strides[ptr];
            while(checkLoc[ptr] >= globalShape[ptr]){
                checkLoc[ptr] = myGridViewLoc[ptr];
                distLoc[ptr] = 0;
                ptr++;
                if(ptr == order){
                    stop = true;
                    break;
                }else{
                    distLoc[ptr]++;
                    checkLoc[ptr] += strides[ptr];
                }
            }
            ptr = 0;
        }
    }
    Unsigned recv;

//    std::cout << "comm size" << mpi::CommSize(mpi::COMM_WORLD) << std::endl;
//    printf("allreducing\n");
    mpi::AllReduce(&res, &recv, 1, mpi::LOGICAL_AND, mpi::COMM_WORLD);
//    printf("allreducing finished\n");
    if(recv == 0)
        LogicError("redist bug: some process not assigned correct data");

    if(commRank == 0){
        std::cout << "PASS" << std::endl;
        return true;
    }
    return false;
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
    CheckResult(B);
//    Print(B, "B after gather-to-one redist");
}

template<typename T>
void
TestGTORedist( DistTensor<T>& A, const ModeArray& gModes, const std::vector<ModeArray>& gridGroups, const TensorDistribution& resDist)
{
#ifndef RELEASE
    CallStackEntry entry("TestGTOGRedist");
#endif
    Unsigned i;
    const Int commRank = mpi::CommRank( mpi::COMM_WORLD );
    const Grid& g = A.Grid();

    DistTensor<T> B(A.Shape(), resDist, g);

    if(commRank == 0){
        printf("Gathering to one modes (%d", gModes[0]);
        for(i = 1; i < gModes.size(); i++)
            printf(", %d", gModes[i]);
        printf("): %s <-- %s\n", (tmen::TensorDistToString(B.TensorDist())).c_str(), (tmen::TensorDistToString(A.TensorDist())).c_str());
    }
    B.GatherToOneRedistFrom(A, gModes, gridGroups);
    CheckResult(B);
//    Print(B, "B after gather-to-one redist");
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
TestRTOGRedist( DistTensor<T>& A, const ModeArray& rModes, const TensorDistribution& resDist)
{
#ifndef RELEASE
    CallStackEntry entry("TestRTOGRedist");
#endif
    Unsigned i;
    const Int commRank = mpi::CommRank( mpi::COMM_WORLD );
    const Grid& g = A.Grid();

    ObjShape shapeB = A.Shape();
    ModeArray sortedRModes = rModes;
    std::sort(sortedRModes.begin(), sortedRModes.end());
    for(i = sortedRModes.size() - 1; i < sortedRModes.size(); i--){
        shapeB.erase(shapeB.begin() + sortedRModes[i]);
    }

    DistTensor<T> B(shapeB, resDist, g);

    if(commRank == 0){
        printf("Reducing to one modes (%d", rModes[0]);
        for(i = 1; i < rModes.size(); i++)
            printf(", %d", rModes[i]);
        printf("): %s <-- %s\n", (tmen::TensorDistToString(B.TensorDist())).c_str(), (tmen::TensorDistToString(A.TensorDist())).c_str());
    }
    B.ReduceToOneRedistFrom(A, rModes);
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
    CheckResult(B);
//    Print(B, "B after permute redist");
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
    CheckResult(B);
//    Print(B, "B after ag redist");
}

template<typename T>
void
TestAGGRedist( DistTensor<T>& A, const ModeArray& agModes, const std::vector<ModeArray>& redistGroups, const TensorDistribution& resDist )
{
#ifndef RELEASE
    CallStackEntry entry("TestAGRedist");
#endif
    Unsigned i;
    const Int commRank = mpi::CommRank( mpi::COMM_WORLD );
    //const int order = A.Order();
    const Grid& g = A.Grid();

    DistTensor<T> B(A.Shape(), resDist, g);

    if(commRank == 0){
        printf("Allgathering modes (%d", agModes[0]);
        for(i = 1; i < agModes.size(); i++)
            printf(", %d", agModes[i]);
        printf("): %s <-- %s\n", (tmen::TensorDistToString(B.TensorDist())).c_str(), (tmen::TensorDistToString(A.TensorDist())).c_str());
    }

    B.AllGatherRedistFrom(A, agModes, redistGroups);
    CheckResult(B);
//    Print(B, "B after ag redist");
}

template<typename T>
void
TestLRedist( DistTensor<T>& A, Mode lMode, const ModeDistribution& resDist )
{
#ifndef RELEASE
    CallStackEntry entry("TestLRedist");
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

//    Print(A, "A before local redist");
    ModeArray gridRedistModes(resDist.begin() + lModeDist.size(), resDist.end());
    B.LocalRedistFrom(A, lMode, gridRedistModes);
    CheckResult(B);
//    Print(B, "B after local redist");
}

template<typename T>
void
TestLGRedist( DistTensor<T>& A, const ModeArray& lModes, const std::vector<ModeArray>& gridRedistModes, const TensorDistribution& resDist )
{
#ifndef RELEASE
    CallStackEntry entry("TestLGRedist");
#endif
    Unsigned i;
    const Int commRank = mpi::CommRank( mpi::COMM_WORLD );
    const Grid& g = A.Grid();

    TensorDistribution distB = resDist;

    DistTensor<T> B(A.Shape(), distB, g);

    TensorDistribution lModeDist = A.TensorDist();

    if(commRank == 0){
        printf("Locally redistributing modes (%d", lModes[0]);
        for(i = 1; i < lModes.size(); i++)
            printf(", %d", lModes[i]);
        printf("): %s <-- %s\n", (tmen::TensorDistToString(B.TensorDist())).c_str(), (tmen::TensorDistToString(A.TensorDist())).c_str());
    }

//    Print(A, "A before local redist");
    B.LocalRedistFrom(A, lModes, gridRedistModes);
    CheckResult(B);
//    Print(B, "B after local redist");
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
    BShape.erase(BShape.begin() + rMode);
    DistTensor<T> B(BShape, resDist, g);

//    Print(A, "A before rs redist");
    if(commRank == 0){
        printf("Reducing mode %d and scattering mode %d: %s <-- %s\n", rMode, sMode, (tmen::TensorDistToString(B.TensorDist())).c_str(), (tmen::TensorDistToString(A.TensorDist())).c_str());
    }

    B.ReduceScatterRedistFrom(A, rMode, sMode);

    Print(B, "B after rs redist");
}

template<typename T>
void
TestRSGRedist(DistTensor<T>& A, const ModeArray& rModes, const ModeArray& sModes, const TensorDistribution& resDist)
{
#ifndef RELEASE
    CallStackEntry entry("TestRSRedist");
#endif
    Unsigned i;
    const Int commRank = mpi::CommRank( mpi::COMM_WORLD );
    const Grid& g = A.Grid();
    const GridView gv = A.GetGridView();

    ObjShape BShape = A.Shape();
    ModeArray redistModes = rModes;
    std::sort(redistModes.begin(), redistModes.end());
    for(i = redistModes.size() - 1; i < redistModes.size(); i--)
        BShape.erase(BShape.begin() + redistModes[i]);
    DistTensor<T> B(BShape, resDist, g);

//    Print(A, "A before rs redist");
    if(commRank == 0){
        printf("Reducing modes (%d", rModes[0]);
        for(i = 1; i < rModes.size(); i++)
            printf(", %d", rModes[i]);
        printf(") and scattering modes (%d", sModes[0]);
        for(i = 1; i < rModes.size(); i++)
            printf(", %d", sModes[i]);
        printf("): %s <-- %s\n", (tmen::TensorDistToString(B.TensorDist())).c_str(), (tmen::TensorDistToString(A.TensorDist())).c_str());
    }

    B.ReduceScatterRedistFrom(A, rModes, sModes);

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
    CheckResult(B);
//    Print(B, "B after a2a redist");
}

template<typename T>
void
TestA2ARedist(DistTensor<T>& A, const ModeArray& a2aModesFrom, const ModeArray& a2aModesTo, const std::vector<ModeArray >& commGroups, const TensorDistribution& resDist){
#ifndef RELEASE
    CallStackEntry entry("TestA2ARedist");
#endif
    Unsigned i;
    const Int commRank = mpi::CommRank( mpi::COMM_WORLD );
    const Grid& g = A.Grid();

    DistTensor<T> B(A.Shape(), resDist, g);

    if(commRank == 0){
        printf("All-to-alling modes ( %d", a2aModesFrom[0]);
        for(i = 1; i < a2aModesFrom.size(); i++)
            printf(", %d", a2aModesFrom[i]);
        printf("),( %d", a2aModesTo[0]);
        for(i = 1; i < a2aModesTo.size(); i++)
            printf(", %d", a2aModesTo[i]);
        printf("): %s <-- %s\n", (tmen::TensorDistToString(B.TensorDist())).c_str(), (tmen::TensorDistToString(A.TensorDist())).c_str());
    }

    B.AllToAllRedistFrom(A, a2aModesFrom, a2aModesTo, commGroups);
    CheckResult(B);
//    Print(B, "B after a2a redist");
}

template<typename T>
TensorDistribution
DetermineResultingDistributionRTO(const DistTensor<T>& A, Mode rMode){
    Unsigned order = A.Order();
    const TensorDistribution ADist = A.TensorDist();
    const ModeDistribution rModeDist = A.ModeDist(rMode);
    TensorDistribution ret(ADist);
    ret.erase(ret.begin() + rMode);
    ret[order].insert(ret[order].end(), ADist[rMode].begin(), ADist[rMode].end());
    return ret;
}

template<typename T>
TensorDistribution
DetermineResultingDistributionRTOG(const DistTensor<T>& A, const ModeArray& rModes){
    Unsigned i;
    const Unsigned order = A.Order();
    const TensorDistribution ADist = A.TensorDist();
    TensorDistribution ret(ADist);
    ModeArray sortedRModes = rModes;
    std::sort(sortedRModes.begin(), sortedRModes.end());
    for(i = sortedRModes.size() - 1; i < sortedRModes.size(); i--){
        ret.erase(ret.begin() + sortedRModes[i]);
    }

    for(i = 0; i < sortedRModes.size(); i++){
        ret[ret.size()-1].insert(ret[ret.size()-1].end(), ADist[sortedRModes[i]].begin(), ADist[sortedRModes[i]].end());
    }

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
DetermineResultingDistributionAGG(const DistTensor<T>& A, const ModeArray& agModes, const std::vector<ModeArray>& redistModes){
    Unsigned i;
    const TensorDistribution ADist = A.TensorDist();
    TensorDistribution ret(ADist);
    for(i = 0; i < redistModes.size(); i++)
        ret[agModes[i]].erase(ret[agModes[i]].end() - redistModes[i].size(), ret[agModes[i]].end());

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
DetermineResultingDistributionGTOG(const DistTensor<T>& A, const ModeArray& gModes, const std::vector<ModeArray>& redistGroups){
    Unsigned i;
    const TensorDistribution ADist = A.TensorDist();
    TensorDistribution ret(ADist);
    for(i = 0; i < redistGroups.size(); i++){
        ret[gModes[i]].erase(ret[gModes[i]].end() - redistGroups[i].size(), ret[gModes[i]].end());
        ret[A.Order()].insert(ret[A.Order()].end(), redistGroups[i].begin(), redistGroups[i].end());
    }
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
DetermineResultingDistributionLocalG(const DistTensor<T>& A, const ModeArray& lModes, const std::vector<ModeArray>& gridRedistModes){
    Unsigned i;
    TensorDistribution ret = A.TensorDist();
    for(i = 0; i < lModes.size(); i++){
        ret[lModes[i]].insert(ret[lModes[i]].end(), gridRedistModes[i].begin(), gridRedistModes[i].end());
    }
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
    ret.erase(ret.begin() + rMode);
    return ret;
}

//NOTE: Stupidly stacks scatters on top of one another
template<typename T>
TensorDistribution
DetermineResultingDistributionRSG(const DistTensor<T>& A, const ModeArray& rModes, const ModeArray& sModes){
    Unsigned i;
    const TensorDistribution ADist = A.TensorDist();
    TensorDistribution ret = ADist;

    for(i = 0; i < rModes.size(); i++){
        ModeDistribution rModeDist = ADist[rModes[i]];
        ret[sModes[i]].insert(ret[sModes[i]].end(), rModeDist.begin(), rModeDist.end());
    }

    ModeArray reduceModes = rModes;
    std::sort(reduceModes.begin(), reduceModes.end());
    for(i = reduceModes.size() - 1; i < reduceModes.size(); i--)
        ret.erase(ret.begin() + reduceModes[i]);
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
TensorDistribution
DetermineResultingDistributionA2A(const DistTensor<T>& A, const ModeArray& a2aModesFrom, const ModeArray& a2aModesTo, const std::vector<ModeArray >& commGroups){
    Unsigned i;

    TensorDistribution newDist = A.TensorDist();

    for(i = 0; i < a2aModesFrom.size(); i++)
        newDist[a2aModesFrom[i]].erase(newDist[a2aModesFrom[i]].end() - commGroups[i].size(), newDist[a2aModesFrom[i]].end());

    for(i = 0; i < a2aModesTo.size(); i++)
        newDist[a2aModesTo[i]].insert(newDist[a2aModesTo[i]].end(), commGroups[i].begin(), commGroups[i].end());

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
void
CreateGTOGTestsHelper(const DistTensor<T>& A, const ModeArray& gModes, Unsigned pos, const std::vector<std::vector<ModeArray>>& commGroups, const std::vector<ModeArray>& pieceComms, std::vector<GTOGTest>& tests){

//    printf("n: %d, p: %d\n", modesFrom.size(), pos);
    if(pos == gModes.size()){
//        printf("pushing\n");
        ModeArray testGTOModes = gModes;
        std::pair<ModeArray, std::vector<ModeArray> > t1(testGTOModes, pieceComms);
        TensorDistribution resDist = DetermineResultingDistributionGTOG(A, gModes, pieceComms);
        GTOGTest test(t1, resDist);
        tests.push_back(test);
//        printf("done\n");
    }else{
//        printf("recurring\n");
        Unsigned i;
        std::vector<ModeArray> modeCommGroups = commGroups[pos];
//        printf("modeCommGroups size: %d\n", modeCommGroups.size());

        for(i = 0; i < modeCommGroups.size(); i++){
//            printf("ping\n");
            std::vector<ModeArray> newPieceComm = pieceComms;
            newPieceComm[pos] = modeCommGroups[i];
            CreateGTOGTestsHelper(A, gModes, pos + 1, commGroups, newPieceComm, tests);
        }
//        printf("done recurring\n");
    }
}

template<typename T>
std::vector<GTOGTest >
CreateGTOGTests(const DistTensor<T>& A, const Params& args){
    Unsigned i, j, k, l;
    std::vector<GTOGTest > ret;

    const Unsigned order = A.Order();
    const TensorDistribution distA = A.TensorDist();
    ModeArray gridModes(order);
    for(i = 0; i < gridModes.size(); i++)
        gridModes[i] = i;

    for(i = 1; i <= order; i++){
        std::vector<ModeArray> gModeCombos = AllCombinations(gridModes, i);
        for(j = 0; j < gModeCombos.size(); j++){
            ModeArray gModes = gModeCombos[j];

            std::vector<std::vector<ModeArray> > commGroups(gModes.size());

            for(k = 0; k < gModes.size(); k++){
                Mode agMode = gModes[k];
                ModeDistribution gDist = A.ModeDist(agMode);

                for(l = 0; l < gDist.size(); l++){
                    ModeArray commGroup(gDist.end() - l - 1, gDist.end());
                    commGroups[k].push_back(commGroup);
                }
            }

            std::vector<ModeArray> pieceComms(gModes.size());

            CreateGTOGTestsHelper(A, gModes, 0, commGroups, pieceComms, ret);
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
void
CreateAGGTestsHelper(const DistTensor<T>& A, const ModeArray& agModes, Unsigned pos, const std::vector<std::vector<ModeArray>>& commGroups, const std::vector<ModeArray>& pieceComms, std::vector<AGGTest>& tests){

//    printf("n: %d, p: %d\n", modesFrom.size(), pos);
    if(pos == agModes.size()){
//        printf("pushing\n");
        ModeArray testAGModes = agModes;
        std::pair<ModeArray, std::vector<ModeArray> > t1(testAGModes, pieceComms);
        TensorDistribution resDist = DetermineResultingDistributionAGG(A, agModes, pieceComms);
        AGGTest test(t1, resDist);
        tests.push_back(test);
//        printf("done\n");
    }else{
//        printf("recurring\n");
        Unsigned i;
        std::vector<ModeArray> modeCommGroups = commGroups[pos];
//        printf("modeCommGroups size: %d\n", modeCommGroups.size());

        for(i = 0; i < modeCommGroups.size(); i++){
//            printf("ping\n");
            std::vector<ModeArray> newPieceComm = pieceComms;
            newPieceComm[pos] = modeCommGroups[i];
            CreateAGGTestsHelper(A, agModes, pos + 1, commGroups, newPieceComm, tests);
        }
//        printf("done recurring\n");
    }
}

template<typename T>
std::vector<AGGTest >
CreateAGGTests(const DistTensor<T>& A, const Params& args){
    Unsigned i, j, k, l;
    std::vector<AGGTest > ret;

    const Unsigned order = A.Order();
    const TensorDistribution distA = A.TensorDist();
    ModeArray gridModes(order);
    for(i = 0; i < gridModes.size(); i++)
        gridModes[i] = i;

    for(i = 1; i <= order; i++){
        std::vector<ModeArray> agModeCombos = AllCombinations(gridModes, i);
        for(j = 0; j < agModeCombos.size(); j++){
            ModeArray agModes = agModeCombos[j];

            std::vector<std::vector<ModeArray> > commGroups(agModes.size());

            for(k = 0; k < agModes.size(); k++){
                Mode agMode = agModes[k];
                ModeDistribution agDist = A.ModeDist(agMode);

                for(l = 0; l < agDist.size(); l++){
                    ModeArray commGroup(agDist.end() - l - 1, agDist.end());
                    commGroups[k].push_back(commGroup);
                }
            }

            std::vector<ModeArray> pieceComms(agModes.size());

            CreateAGGTestsHelper(A, agModes, 0, commGroups, pieceComms, ret);
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

    const Unsigned order = A.Order();
    const TensorDistribution tDist = A.TensorDist();
    ModeArray freeModes = gv.FreeModes();

    //NOTE: Just picking up to 2 modes to redistribute along
    for(i = 0; i < order; i++){
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
void
CreateLGTestsHelper(const DistTensor<T>& A, const ModeArray& lModes, Unsigned pos, const std::vector<std::vector<ModeArray>>& commGroups, const std::vector<ModeArray>& pieceComms, std::vector<LGTest>& tests){

//    printf("n: %d, p: %d\n", modesFrom.size(), pos);
    if(pos == lModes.size()){
//        printf("pushing\n");
        ModeArray testLModes = lModes;
        std::pair<ModeArray, std::vector<ModeArray> > t1(testLModes, pieceComms);
        TensorDistribution resDist = DetermineResultingDistributionLocalG(A, lModes, pieceComms);
        LGTest test(t1, resDist);
        tests.push_back(test);
//        printf("done\n");
    }else{
//        printf("recurring\n");
        Unsigned i;
        std::vector<ModeArray> modeCommGroups = commGroups[pos];
//        printf("modeCommGroups size: %d\n", modeCommGroups.size());

        for(i = 0; i < modeCommGroups.size(); i++){
//            printf("ping\n");
            std::vector<ModeArray> newPieceComm = pieceComms;
            newPieceComm[pos] = modeCommGroups[i];
            CreateLGTestsHelper(A, lModes, pos + 1, commGroups, newPieceComm, tests);
        }
//        printf("done recurring\n");
    }
}

template<typename T>
void
CreateLGCommGroupsHelper(const DistTensor<T>& A, const Unsigned& nlModes, Unsigned pos, const ModeArray& freeModes, std::vector<std::vector<ModeArray>>& commGroups){

//    printf("n: %d, p: %d\n", modesFrom.size(), pos);
    if(pos == nlModes){

    }else{
//        printf("recurring\n");
        Unsigned i, j, k;
//        printf("modeCommGroups size: %d\n", modeCommGroups.size());

        for(i = 1; i <= freeModes.size(); i++){
            std::vector<ModeArray> lModeCombos = AllCombinations(freeModes, i);
            for(j = 0; j < lModeCombos.size(); j++){
                ModeArray lModeCombo = lModeCombos[j];
                ModeArray newFreeModes = freeModes;
                for(k = 0; k < lModeCombo.size(); k++){
                    newFreeModes.erase(std::find(newFreeModes.begin(), newFreeModes.end(), lModeCombo[k]));
                }
                commGroups[pos].push_back(lModeCombo);
                CreateLGCommGroupsHelper(A, nlModes, pos + 1, newFreeModes, commGroups);
            }
        }
    }
}

template<typename T>
std::vector<LGTest>
CreateLGTests(const DistTensor<T>& A, const Params& args){
    Unsigned i, j, k, l;
    std::vector<LGTest > ret;

    const Unsigned order = A.Order();
    const TensorDistribution distA = A.TensorDist();
    ModeArray tensorModes(order);
    for(i = 0; i < tensorModes.size(); i++)
        tensorModes[i] = i;

    for(i = 1; i <= order; i++){
        std::vector<ModeArray> lModeCombos = AllCombinations(tensorModes, i);
        for(j = 0; j < lModeCombos.size(); j++){
            ModeArray lModes = lModeCombos[j];

            std::vector<std::vector<ModeArray> > commGroups(lModes.size());

            ModeArray freeModes = A.GetGridView().FreeModes();

            CreateLGCommGroupsHelper(A, lModes.size(), 0, freeModes, commGroups);

//
//            for(k = 0; k < lModes.size(); k++){
//                Mode agMode = lModes[k];
//                ModeDistribution lDist = A.ModeDist(agMode);
//
//                for(l = 0; l < lDist.size(); l++){
//                    ModeArray commGroup(lDist.end() - l - 1, lDist.end());
//                    commGroups[k].push_back(commGroup);
//                }
//            }

            std::vector<ModeArray> pieceComms(lModes.size());

            CreateLGTestsHelper(A, lModes, 0, commGroups, pieceComms, ret);
        }
        printf("Up to %d tests for i: %d\n", ret.size(), i);
    }

//    AGTest test(1, DetermineResultingDistributionAG(A, 1));
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
std::vector<RTOGTest>
CreateRTOGTests(const DistTensor<T>& A, const Params& args){
    Unsigned i, j;
    std::vector<RTOGTest> ret;

    const Unsigned order = A.Order();
    const GridView gv = A.GetGridView();
    ModeArray gridModes(order);
    for(i = 0; i < gridModes.size(); i++)
        gridModes[i] = i;

    for(i = 1; i <= order; i++){
        std::vector<ModeArray> combinations = AllCombinations(gridModes, i);
        for(j = 0; j < combinations.size(); j++){
            ModeArray rModes = combinations[j];

            TensorDistribution resDist = DetermineResultingDistributionRTOG(A, rModes);
            RTOGTest test(rModes, resDist);
            ret.push_back(test);
        }
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
std::vector<RSGTest >
CreateRSGTests(const DistTensor<T>& A, const Params& args){
    Unsigned i, j, k;
    std::vector<RSGTest> ret;

    const Unsigned order = A.Order();
    const GridView gv = A.GetGridView();
    ModeArray gridModes(order);
    for(i = 0; i < gridModes.size(); i++)
        gridModes[i] = i;

    for(i = 1; i <= order; i++){
        std::vector<ModeArray> rModeCombos = AllCombinations(gridModes, i);
        for(j = 0; j < rModeCombos.size(); j++){
            ModeArray rModes = rModeCombos[j];
            ModeArray potSModes = NegFilterVector(gridModes, rModes);
            std::vector<ModeArray> sModeCombos = AllCombinations(potSModes, i);
            for(k = 0; k < sModeCombos.size(); k++){
                ModeArray sModes = sModeCombos[k];
                TensorDistribution resDist = DetermineResultingDistributionRSG(A, rModes, sModes);
                std::pair<ModeArray, ModeArray> redistModes(rModes, sModes);
                RSGTest test(redistModes, resDist);
                ret.push_back(test);
            }
        }
    }

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
CreateA2ATestsHelper(const DistTensor<T>& A, const ModeArray& modesFrom, const ModeArray& modesTo, Unsigned pos, const std::vector<std::vector<ModeArray>>& commGroups, const std::vector<ModeArray>& pieceComms, std::vector<A2ATest>& tests){

//    printf("n: %d, p: %d\n", modesFrom.size(), pos);
    if(pos == modesFrom.size()){
//        printf("pushing\n");
        std::pair<ModeArray, ModeArray> modesFromTo(modesFrom, modesTo);
        std::pair<std::pair<ModeArray, ModeArray>, std::vector<ModeArray> > t1(modesFromTo, pieceComms);
        TensorDistribution resDist = DetermineResultingDistributionA2A(A, modesFrom, modesTo, pieceComms);
        A2ATest test(t1, resDist);
        tests.push_back(test);
//        printf("done\n");
    }else{
//        printf("recurring\n");
        Unsigned i;
        std::vector<ModeArray> modeCommGroups = commGroups[pos];
//        printf("modeCommGroups size: %d\n", modeCommGroups.size());

        for(i = 0; i < modeCommGroups.size(); i++){
//            printf("ping\n");
            std::vector<ModeArray> newPieceComm = pieceComms;
            newPieceComm[pos] = modeCommGroups[i];
            CreateA2ATestsHelper(A, modesFrom, modesTo, pos + 1, commGroups, newPieceComm, tests);
        }
//        printf("done recurring\n");
    }
}

template<typename T>
std::vector<A2ATest>
CreateA2ATests(const DistTensor<T>& A, const Params& args){
    std::vector<A2ATest> ret;

    Unsigned i, j, k, l, m;
    const Unsigned order = A.Order();
    const GridView gv = A.GetGridView();
    ModeArray gridModes(order);
    for(i = 0; i < gridModes.size(); i++)
        gridModes[i] = i;

    for(i = 1; i <= order; i++){
        std::vector<ModeArray> a2aModeFromCombos = AllCombinations(gridModes, i);
        std::vector<ModeArray> a2aModeToCombos = AllCombinations(gridModes, i);
        for(j = 0; j < a2aModeFromCombos.size(); j++){
            ModeArray a2aModesFrom = a2aModeFromCombos[j];
            for(k = 0; k < a2aModeToCombos.size(); k++){
                ModeArray a2aModesTo = a2aModeToCombos[k];

                std::vector<std::vector<ModeArray> > commGroups(a2aModesFrom.size());

                for(l = 0; l < a2aModesFrom.size(); l++){
                    Mode a2aModeFrom = a2aModesFrom[l];
                    ModeDistribution a2aFromDist = A.ModeDist(a2aModeFrom);

                    for(m = 0; m < a2aFromDist.size(); m++){
                        ModeArray commGroup(a2aFromDist.end() - m-1, a2aFromDist.end());
                        commGroups[l].push_back(commGroup);
//                        printf("commGroups[%d].size(): %d\n", l, commGroups[l].size());
                    }
                }
//                printf("commGroups.size(): %d\n", commGroups.size());
                std::vector<ModeArray> pieceComms(a2aModesFrom.size());

                CreateA2ATestsHelper(A, a2aModesFrom, a2aModesTo, 0, commGroups, pieceComms, ret);
//                printf("ret size: %d\n", ret.size());
            }
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
    Unsigned i;
    const Int commRank = mpi::CommRank( mpi::COMM_WORLD );

    DistTensor<T> A(args.tensorShape, args.tensorDist, g);
    Set(A);
    Print(A, "A");

    std::vector<AGTest> agTests = CreateAGTests(A, args);
    std::vector<AGGTest> aggTests = CreateAGGTests(A, args);
    std::vector<GTOTest> gtoTests = CreateGTOTests(A, args);
    std::vector<GTOGTest> gtogTests = CreateGTOGTests(A, args);
    std::vector<LTest> lTests = CreateLTests(A, args);
    std::vector<LGTest> lgTests = CreateLGTests(A, args);
    std::vector<RSTest> rsTests = CreateRSTests(A, args);
    std::vector<RSGTest> rsgTests = CreateRSGTests(A, args);
    std::vector<PRSTest> prsTests = CreatePRSTests(A, args);
    std::vector<PTest> pTests = CreatePTests(A, args);
    std::vector<A2ADMTest> a2admTests = CreateA2ADMTests(A, args);
    std::vector<A2ATest> a2aTests = CreateA2ATests(A, args);
    std::vector<RTOTest> rtoTests = CreateRTOTests(A, args);
    std::vector<RTOGTest> rtogTests = CreateRTOGTests(A, args);

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
//
//    if(commRank == 0){
//        printf("Performing Gather-to-one tests\n");
//    }
//    for(i = 0; i < gtoTests.size(); i++){
//        GTOTest thisTest = gtoTests[i];
//        Mode gMode = thisTest.first.first;
//        ModeArray redistModes = thisTest.first.second;
//        TensorDistribution resDist = thisTest.second;
//
//        TestGTORedist(A, gMode, redistModes, resDist);
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
//        TestGTORedist(A, gModes, redistGroups, resDist);
//    }
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

    if(commRank == 0){
        printf("Performing LocalG redist tests\n");
    }
    for(i = 0; i < lgTests.size(); i++){
        LGTest thisTest = lgTests[i];
        ModeArray lModes = thisTest.first.first;
        std::vector<ModeArray> commGroups = thisTest.first.second;
        TensorDistribution resDist = thisTest.second;

        TestLGRedist(A, lModes, commGroups, resDist);
    }
//
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
////
//    if(commRank == 0){
//        printf("Performing ReduceScatterGeneral tests\n");
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
//    for(i = 0; i < a2admTests.size(); i++){
//        A2ADMTest thisTest = a2admTests[i];
//        std::pair<Mode, Mode> a2aModes = thisTest.first.first;
//        std::pair<ModeArray, ModeArray > commGroups = thisTest.first.second;
//        TensorDistribution resDist = thisTest.second;
//
//        TestA2ADMRedist(A, a2aModes, commGroups, resDist);
//    }
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
