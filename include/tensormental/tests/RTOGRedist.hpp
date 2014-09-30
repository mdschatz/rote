#pragma once
#ifndef TMEN_TESTS_RTOGREDIST_HPP
#define TMEN_TESTS_RTOGREDIST_HPP

#include "tensormental/tests/AllRedists.hpp"
using namespace tmen;

typedef std::pair< ModeArray, TensorDistribution > RTOGTest;

template<typename T>
TensorDistribution
DetermineResultingDistributionRTOG(const DistTensor<T>& A, const ModeArray& rModes){
    Unsigned i;
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
void
TestRTOGRedist( const DistTensor<T>& A, const ModeArray& rModes, const TensorDistribution& resDist)
{
#ifndef RELEASE
    CallStackEntry entry("TestRTOGRedist");
#endif
    Unsigned i;
    const Int commRank = mpi::CommRank( mpi::COMM_WORLD );
    const Grid& g = A.Grid();

//    if(commRank == 0){
        printf("Reducing to one modes (");
        if(rModes.size() > 0)
            printf("%d", rModes[0]);
        for(i = 1; i < rModes.size(); i++)
            printf(", %d", rModes[i]);
        printf("): %s <-- %s\n", (tmen::TensorDistToString(resDist)).c_str(), (tmen::TensorDistToString(A.TensorDist())).c_str());
//    }

    ObjShape shapeB = A.Shape();
    ModeArray sortedRModes = rModes;
    std::sort(sortedRModes.begin(), sortedRModes.end());
    for(i = sortedRModes.size() - 1; i < sortedRModes.size(); i--){
        shapeB.erase(shapeB.begin() + sortedRModes[i]);
    }

    DistTensor<T> B(shapeB, resDist, NegFilterVector(A.Alignments(), rModes), g);

    Unsigned order = B.Order();
    Permutation perm(order);
    for(i = 0; i < order; i++)
        perm[i] = i;

    do{
        B.SetLocalPermutation(perm);
        B.ResizeLocalUnderPerm(perm);
        B.ReduceToOneRedistFromWithPermutation(A, rModes);
        Print(B, "B after reduce-to-one redist");
//        CheckResult(B, check);
    }while(next_permutation(perm.begin(), perm.end()));

//    B.ReduceToOneRedistFrom(A, rModes);
//    Print(B, "B after reduce-to-one redist");
}

#endif // ifndef TMEN_TESTS_RTOGREDIST_HPP
