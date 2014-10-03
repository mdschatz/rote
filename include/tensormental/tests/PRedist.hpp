#pragma once
#ifndef TMEN_TESTS_PREDIST_HPP
#define TMEN_TESTS_PREDIST_HPP

#include "tensormental/tests/AllRedists.hpp"
using namespace tmen;

typedef std::pair< Mode, ModeDistribution> PTest;

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
void
TestPRedist( const DistTensor<T>& A, Mode pMode, const ModeDistribution& resDist )
{
#ifndef RELEASE
    CallStackEntry entry("TestPRedist");
#endif
    const Int commRank = mpi::CommRank( mpi::COMM_WORLD );
    Unsigned i;
    Unsigned order = A.Order();
    //const int order = A.Order();
    const Grid& g = A.Grid();

    TensorDistribution BDist = A.TensorDist();
    BDist[pMode] = resDist;

    DistTensor<T> B(A.Shape(), BDist, g);
    B.AlignWith(A);
    B.ResizeTo(A);
    B.SetDistribution(BDist);
    //Print(B, "B before permute redist");

    if(commRank == 0){
        printf("Permuting mode %d: %s <-- %s\n", pMode, (tmen::TensorDistToString(B.TensorDist())).c_str(), (tmen::TensorDistToString(A.TensorDist())).c_str());
    }

    Tensor<T> check(A.Shape());
    Set(check);

    Permutation perm = DefaultPermutation(order);

    do{
        B.SetLocalPermutation(perm);
        B.ResizeLocalUnderPerm(perm);
        B.PermutationRedistFrom(A, pMode, resDist);
        CheckResult(B, check);
    }while(next_permutation(perm.begin(), perm.end()));

//    B.PermutationRedistFrom(A, pMode, resDist);
//    CheckResult(B);
//    Print(B, "B after permute redist");
}

#endif // ifndef TMEN_TESTS_PREDIST_HPP
