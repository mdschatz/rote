#pragma once
#ifndef TMEN_TESTS_LGREDIST_HPP
#define TMEN_TESTS_LGREDIST_HPP

#include "tensormental/tests/AllRedists.hpp"

using namespace tmen;

void
CreateLGTestsHelper(const ModeArray& modesToMove, const TensorDistribution& distA, const std::vector<RedistTest>& partialTests, std::vector<RedistTest>& fullTests){
	Unsigned order = distA.size() - 1;
	Unsigned i, j;

	if(modesToMove.size() == 0){
		fullTests.insert(fullTests.end(), partialTests.begin(), partialTests.end());
		return;
	}

	std::vector<RedistTest > newPartialTests;
	ModeArray newModesToMove = modesToMove;
	newModesToMove.erase(newModesToMove.end() - 1);

	for(i = 0; i < partialTests.size(); i++){
		const TensorDistribution partialDist = partialTests[i].first;
		const ModeArray partialModes = partialTests[i].second;

		for(j = 0; j < order; j++){
			ModeArray resModes = partialModes;
			TensorDistribution resDist = partialDist;
			resDist[j].push_back(modesToMove[modesToMove.size()-1]);
			resModes.push_back(modesToMove[modesToMove.size()-1]);
			RedistTest newTest;
			newTest.first = resDist;
			newTest.second = resModes;

			newPartialTests.push_back(newTest);
		}
	}
	CreateLGTestsHelper(newModesToMove, distA, newPartialTests, fullTests);
}

std::vector<RedistTest>
CreateLGTests(const Unsigned& gridOrder, const TensorDistribution& distA){
    Unsigned i;
    std::vector<RedistTest > ret;

    const Unsigned order = distA.size() - 1;

    ModeArray gridModes = DefaultPermutation(gridOrder);
    ModeArray usedGridModes;

    for(i = 0; i < order; i++)
    	usedGridModes.insert(usedGridModes.end(), distA[i].begin(), distA[i].end());
    std::sort(usedGridModes.begin(), usedGridModes.end());

    ModeArray unusedGridModes;
    std::set_difference(gridModes.begin(), gridModes.end(), usedGridModes.begin(), usedGridModes.end(), back_inserter(unusedGridModes));

    PrintVector(unusedGridModes, "unusedModes");
    PrintVector(usedGridModes, "usedModes");

    std::vector<ModeArray> redistModesGroups;
    for(i = 1; i < order; i++){
    	std::vector<ModeArray> newRedistModesGroups = AllCombinations(unusedGridModes, i);
    	redistModesGroups.insert(redistModesGroups.end(), newRedistModesGroups.begin(), newRedistModesGroups.end());
    }

    printf("redistModesGroups: %d\n", redistModesGroups.size());
    for(i = 0; i < redistModesGroups.size(); i++){
    	ModeArray redistModesGroup = redistModesGroups[i];
    	TensorDistribution resDist = distA;
    	std::vector<RedistTest> partialTests;
    	RedistTest partialTest;
    	ModeArray blankModes;
    	partialTest.first = resDist;
    	partialTest.second = blankModes;
    	partialTests.push_back(partialTest);

    	CreateLGTestsHelper(redistModesGroup, distA, partialTests, ret);
    }
    return ret;
}

#endif // ifndef TMEN_TESTS_LGREDIST_HPP
