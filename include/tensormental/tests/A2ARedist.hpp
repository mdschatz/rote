#pragma once
#ifndef TMEN_TESTS_A2AREDIST_HPP
#define TMEN_TESTS_A2AREDIST_HPP

#include "tensormental/tests/AllRedists.hpp"
using namespace tmen;

typedef std::pair< std::pair<std::pair<ModeArray, ModeArray>, std::vector<ModeArray > >, TensorDistribution> A2ATest;

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
TestA2ARedist( const DistTensor<T>& A, const ModeArray& a2aModesFrom, const ModeArray& a2aModesTo, const std::vector<ModeArray >& commGroups, const TensorDistribution& resDist){
#ifndef RELEASE
    CallStackEntry entry("TestA2ARedist");
#endif
    Unsigned i;
    Unsigned order = A.Order();
    const Int commRank = mpi::CommRank( mpi::COMM_WORLD );
    const Grid& g = A.Grid();

    DistTensor<T> B(resDist, g);
    B.AlignWith(A);
//    B.ResizeTo(A);
    B.SetDistribution(resDist);

    if(commRank == 0){
        printf("All-to-alling modes (");
        if(a2aModesFrom.size() > 0)
            printf("%d", a2aModesFrom[0]);
        for(i = 1; i < a2aModesFrom.size(); i++)
            printf(", %d", a2aModesFrom[i]);
        printf("),(");
        if(a2aModesTo.size() > 0)
            printf("%d", a2aModesTo[0]);
        for(i = 1; i < a2aModesTo.size(); i++)
            printf(", %d", a2aModesTo[i]);
        printf("): %s <-- %s\n", (tmen::TensorDistToString(B.TensorDist())).c_str(), (tmen::TensorDistToString(A.TensorDist())).c_str());
    }

//    B.AllToAllRedistFrom(A, a2aModesFrom, a2aModesTo, commGroups);
//    CheckResult(B);

    Tensor<T> check(A.Shape());
    Set(check);

    Permutation perm = DefaultPermutation(order);

    do{
        if(commRank == 0){
            printf("Testing ");
            PrintVector(perm, "permB");
        }
        B.SetLocalPermutation(perm);
//        B.ResizeTo(B.Shape());
        B.AllToAllRedistFrom(A, a2aModesFrom, a2aModesTo, commGroups);
        CheckResult(B, check);
//            Print(B, "after a2a");
    }while(next_permutation(perm.begin(), perm.end()));
}

#endif // ifndef TMEN_TESTS_A2AREDIST_HPP
