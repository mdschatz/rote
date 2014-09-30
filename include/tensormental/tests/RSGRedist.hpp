#pragma once
#ifndef TMEN_TESTS_RSGREDIST_HPP
#define TMEN_TESTS_RSGREDIST_HPP

#include "tensormental/tests/AllRedists.hpp"
using namespace tmen;

typedef std::pair< std::pair<ModeArray, ModeArray>, TensorDistribution> RSGTest;

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
void
TestRSGRedist( const DistTensor<T>& A, const ModeArray& rModes, const ModeArray& sModes, const TensorDistribution& resDist)
{
#ifndef RELEASE
    CallStackEntry entry("TestRSGRedist");
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
    DistTensor<T> B(BShape, resDist, NegFilterVector(A.Alignments(), rModes), g);

//    Print(A, "A before rs redist");
    if(commRank == 0){
        printf("Reducing modes (");
        if(rModes.size() > 0)
            printf("%d", rModes[0]);
        for(i = 1; i < rModes.size(); i++)
            printf(", %d", rModes[i]);
        printf(") and scattering modes (");
        if(sModes.size() > 0)
            printf("%d", sModes[0]);
        for(i = 1; i < rModes.size(); i++)
            printf(", %d", sModes[i]);
        printf("): %s <-- %s\n", (tmen::TensorDistToString(B.TensorDist())).c_str(), (tmen::TensorDistToString(A.TensorDist())).c_str());
    }

    Permutation perm(B.Order());
    for(i = 0; i < perm.size(); i++)
        perm[i] = (i + 1) % B.Order();
//    PrintVector(perm, "permutation");
//    PrintVector(B.LocalShape(), "LocalShape");
    B.SetLocalPermutation(perm);
    B.ResizeToUnderPerm(B.Shape());
//    PrintVector(B.LocalShape(), "PLocalShape");
    B.ReduceScatterRedistFromWithPermutation(A, rModes, sModes);

    Print(B, "B after rs redist");
}

#endif // ifndef TMEN_TESTS_RSGREDIST_HPP
