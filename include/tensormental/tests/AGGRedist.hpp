#pragma once
#ifndef TMEN_TESTS_AGGREDIST_HPP
#define TMEN_TESTS_AGGREDIST_HPP

#include "tensormental/tests/AllRedists.hpp"
using namespace tmen;

typedef std::pair< std::pair<ModeArray, std::vector<ModeArray> >, TensorDistribution> AGGTest;


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
void
CreateAGGTestsHelper(const DistTensor<T>& A, const ModeArray& agModes, Unsigned pos, const std::vector<std::vector<ModeArray>>& commGroups, const std::vector<ModeArray>& pieceComms, std::vector<AGGTest>& tests){

//    printf("n: %d, p: %d\n", modesFrom.size(), pos);
    if(pos == agModes.size()){
        ModeArray testAGModes = agModes;
        std::pair<ModeArray, std::vector<ModeArray> > t1(testAGModes, pieceComms);
        TensorDistribution resDist = DetermineResultingDistributionAGG(A, agModes, pieceComms);
        AGGTest test(t1, resDist);
        tests.push_back(test);
    }else{
//        printf("recurring\n");
        Unsigned i;
        std::vector<ModeArray> modeCommGroups = commGroups[pos];

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
void
TestAGGRedist( const DistTensor<T>& A, const ModeArray& agModes, const std::vector<ModeArray>& redistGroups, const TensorDistribution& resDist )
{
#ifndef RELEASE
    CallStackEntry entry("TestAGGRedist");
#endif
    Unsigned i;
    const Int commRank = mpi::CommRank( mpi::COMM_WORLD );
    //const int order = A.Order();
    const Grid& g = A.Grid();

    DistTensor<T> B(A.Shape(), resDist, g);
    B.AlignWith(A);
    B.ResizeTo(A.Shape());
    B.SetDistribution(resDist);

    if(commRank == 0){
        printf("Allgathering modes (");
        if(agModes.size() > 0)
            printf("%d", agModes[0]);
        for(i = 1; i < agModes.size(); i++)
            printf(", %d", agModes[i]);
        printf("): %s <-- %s\n", (tmen::TensorDistToString(B.TensorDist())).c_str(), (tmen::TensorDistToString(A.TensorDist())).c_str());
    }

    B.AllGatherRedistFrom(A, agModes, redistGroups);
//    CheckResult(B);
    Print(B, "B after agg redist");
}

#endif // ifndef TMEN_TESTS_AGGREDIST_HPP
