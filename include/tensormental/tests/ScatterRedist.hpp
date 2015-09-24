#pragma once
#ifndef TMEN_TESTS_SCATTERREDIST_HPP
#define TMEN_TESTS_SCATTERREDIST_HPP

#include "tensormental/tests/AllRedists.hpp"
using namespace tmen;

void
CreateScatterTestsHelper(const ModeArray& modesToMove, const TensorDistribution& distA, const std::vector<RedistTest>& partialTests, std::vector<RedistTest>& fullTests){
	Unsigned order = distA.size() - 1;

	if(modesToMove.size() == 1){
		Unsigned i, j;
		for(i = 0; i < partialTests.size(); i++){
			const TensorDistribution partialDist = partialTests[i].first;
			const ModeArray partialModes = partialTests[i].second;

			for(j = 0; j < order; j++){
				ModeArray resModes = partialModes;
				TensorDistribution resDist = partialDist;
				resDist[j].push_back(modesToMove[modesToMove.size()-1]);
				resModes.push_back(modesToMove[modesToMove.size()-1]);
				RedistTest fullTest;
				fullTest.first = resDist;
				fullTest.second = resModes;
				fullTests.push_back(fullTest);
			}
		}
	}else{
		Unsigned i, j;
		std::vector<RedistTest > newPartialTests;
		ModeArray newModesToMove = modesToMove;
		Mode modeToMove = newModesToMove[newModesToMove.size() - 1];
		newModesToMove.erase(newModesToMove.end() - 1);

		for(i = 0; i < partialTests.size(); i++){
			const TensorDistribution partialDist = partialTests[i].first;
			const ModeArray partialModes = partialTests[i].second;

			for(j = 0; j < order; j++){
				ModeArray resModes = partialModes;
				TensorDistribution resDist = partialDist;
				resDist[j].push_back(modeToMove);
				resModes.push_back(modeToMove);
				std::pair<TensorDistribution, ModeArray> newPartialTest;
				newPartialTest.first = resDist;
				newPartialTest.second = resModes;

				newPartialTests.push_back(newPartialTest);
			}
		}
		CreateScatterTestsHelper(newModesToMove, distA, newPartialTests, fullTests);
	}
}

std::vector<RedistTest>
CreateScatterTests(const TensorDistribution& distA){
    Unsigned i;
    std::vector<RedistTest > ret;

    const Unsigned order = distA.size() - 1;

    for(i = 1; i <= distA[order].size(); i++){
    	RedistTest scatterTest;
    	TensorDistribution resDist = distA;
    	ModeArray commModes;
    	commModes.insert(commModes.end(), resDist[order].end() - i, resDist[order].end());
    	resDist[order].erase(resDist[order].end() - i, resDist[order].end());
    	std::vector<RedistTest> partialTests;
    	RedistTest partialTest;
    	ModeArray blankModes;
    	partialTest.first = resDist;
    	partialTest.second = blankModes;
    	partialTests.push_back(partialTest);

//    	std::cout << "making tests with: " << tmen::TensorDistToString(resDist) << std::endl;
    	CreateScatterTestsHelper(commModes, distA, partialTests, ret);
    }
    return ret;
}

template<typename T>
void
TestScatterRedist( const TensorDistribution& resDist, const DistTensor<T>& A, const Permutation& inputPerm, const Permutation& outputPerm, const ModeArray& scatterModes )
{
#ifndef RELEASE
    CallStackEntry entry("TestScatterRedist");
#endif
    Unsigned i;
    Unsigned order = A.Order();
    const Int commRank = mpi::CommRank( mpi::COMM_WORLD );
    const Grid& g = A.Grid();

    DistTensor<T> B(resDist, g);
    B.AlignWith(A);
    B.SetDistribution(resDist);

    if(commRank == 0){
        printf("Scattering modes (");
        if(scatterModes.size() > 0)
            printf("%d", scatterModes[0]);
        for(i = 1; i < scatterModes.size(); i++)
            printf(", %d", scatterModes[i]);
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
			B.ScatterRedistFrom(A, scatterModes);
			Print(B, "check");
//	        CheckResult(B, check);
    }while(next_permutation(perm.begin(), perm.end()));
}

#endif // ifndef TMEN_TESTS_SCATTERREDIST_HPP
