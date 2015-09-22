#pragma once
#ifndef TMEN_TESTS_BCASTREDIST_HPP
#define TMEN_TESTS_BCASTREDIST_HPP

#include "tensormental/tests/AllRedists.hpp"
using namespace tmen;

typedef std::pair< TensorDistribution, ModeArray> BCastTest;

template<typename T>
std::vector<BCastTest >
CreateBCastTests(const DistTensor<T>& A, const Params& args){
    Unsigned i;
    std::vector<BCastTest > ret;

    const Unsigned order = A.Order();
    const TensorDistribution distA = A.TensorDist();

    for(i = 1; i <= distA[order].size(); i++){
    	BCastTest bcastTest;
    	TensorDistribution resDist = distA;
    	ModeArray commModes;
    	commModes.insert(commModes.end(), resDist[order].end() - i, resDist[order].end());
    	resDist[order].erase(resDist[order].end() - i, resDist[order].end());
    	bcastTest.first = resDist;
    	bcastTest.second = commModes;
    	ret.push_back(bcastTest);
    }
    return ret;
}

template<typename T>
void
TestBCastRedist( const TensorDistribution& resDist, const DistTensor<T>& A, const ModeArray& bcastModes )
{
#ifndef RELEASE
    CallStackEntry entry("TestBCastRedist");
#endif
    Unsigned i;
    Unsigned order = A.Order();
    const Int commRank = mpi::CommRank( mpi::COMM_WORLD );
    const Grid& g = A.Grid();

    DistTensor<T> B(resDist, g);
    B.AlignWith(A);
    B.SetDistribution(resDist);

    if(commRank == 0){
        printf("Broadcasting modes (");
        if(bcastModes.size() > 0)
            printf("%d", bcastModes[0]);
        for(i = 1; i < bcastModes.size(); i++)
            printf(", %d", bcastModes[i]);
        printf("): %s <-- %s\n", (tmen::TensorDistToString(B.TensorDist())).c_str(), (tmen::TensorDistToString(A.TensorDist())).c_str());
    }

    Tensor<T> check(A.Shape());
    Set(check);

    Permutation perm = DefaultPermutation(order);

    do{
    	if(commRank == 0){
            printf("Testing ");
            PrintVector(perm, "Output Perm");
    	}
        B.SetLocalPermutation(perm);
        B.BroadcastRedistFrom(A, bcastModes);
        CheckResult(B, check);
    }while(next_permutation(perm.begin(), perm.end()));
}

#endif // ifndef TMEN_TESTS_BCASTREDIST_HPP
