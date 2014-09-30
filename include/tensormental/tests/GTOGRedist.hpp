#pragma once
#ifndef TMEN_TESTS_GTOGREDIST_HPP
#define TMEN_TESTS_GTOGREDIST_HPP

#include "tensormental/tests/AllRedists.hpp"
using namespace tmen;

typedef std::pair< std::pair<ModeArray, std::vector<ModeArray>>, TensorDistribution> GTOGTest;

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
void
TestGTOGRedist( const DistTensor<T>& A, const ModeArray& gModes, const std::vector<ModeArray>& gridGroups, const TensorDistribution& resDist)
{
#ifndef RELEASE
    CallStackEntry entry("TestGTOGRedist");
#endif
    Unsigned i;
    const Int commRank = mpi::CommRank( mpi::COMM_WORLD );
    const Grid& g = A.Grid();

    DistTensor<T> B(A.Shape(), resDist, g);
    B.AlignWith(A);
    B.ResizeTo(A);
    B.SetDistribution(resDist);

    if(commRank == 0){
        printf("Gathering to one modes (");
        if(gModes.size() > 0)
            printf("%d", gModes[0]);
        for(i = 1; i < gModes.size(); i++)
            printf(", %d", gModes[i]);
        printf("): %s <-- %s\n", (tmen::TensorDistToString(B.TensorDist())).c_str(), (tmen::TensorDistToString(A.TensorDist())).c_str());
    }
    B.GatherToOneRedistFrom(A, gModes, gridGroups);
//    CheckResult(B);
    Print(B, "B after gather-to-one redist");
}

#endif // ifndef TMEN_TESTS_GTOGREDIST_HPP
