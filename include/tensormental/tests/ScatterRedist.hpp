#pragma once
#ifndef TMEN_TESTS_SCATTERREDIST_HPP
#define TMEN_TESTS_SCATTERREDIST_HPP

#include "tensormental/tests/AllRedists.hpp"
using namespace tmen;

void
CreateScatterTestsHelper(const ModeArray& modesToMove, const TensorDistribution& distA, const std::vector<RedistTest>& partialTests, std::vector<RedistTest>& fullTests){
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
	CreateScatterTestsHelper(newModesToMove, distA, newPartialTests, fullTests);
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

    	CreateScatterTestsHelper(commModes, distA, partialTests, ret);
    }
    return ret;
}

#endif // ifndef TMEN_TESTS_SCATTERREDIST_HPP
