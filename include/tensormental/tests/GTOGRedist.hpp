#pragma once
#ifndef TMEN_TESTS_GTOGREDIST_HPP
#define TMEN_TESTS_GTOGREDIST_HPP

#include "tensormental/tests/AllRedists.hpp"
using namespace tmen;

void
CreateGTOGTestsHelper(const ModeArray& tenModesToRedist, const TensorDistribution& distA, const std::vector<RedistTest>& partialTests, std::vector<RedistTest>& fullTests){
	Unsigned order = distA.size() - 1;
	Unsigned i, j;

	if(tenModesToRedist.size() == 0){
		fullTests.insert(fullTests.end(), partialTests.begin(), partialTests.end());
		return;
	}

	std::vector<RedistTest > newPartialTests;
	Mode tenModeToRedist = tenModesToRedist[tenModesToRedist.size() - 1];
	ModeArray newTenModesToRedist(tenModesToRedist.begin(), tenModesToRedist.end() - 1);

	for(i = 0; i < partialTests.size(); i++){
		const TensorDistribution partialDist = partialTests[i].first;
		const ModeArray partialCommModes = partialTests[i].second;
		const ModeDistribution modeDistToRedist = partialDist[tenModeToRedist];

		for(j = 1; j <= modeDistToRedist.size(); j++){
			ModeArray thisRedistModes(modeDistToRedist.end() - j, modeDistToRedist.end());
			ModeArray newCommModes = partialCommModes;
			newCommModes.insert(newCommModes.end(), thisRedistModes.begin(), thisRedistModes.end());
			ModeDistribution newModeDist = modeDistToRedist;
			newModeDist.erase(newModeDist.end() - j, newModeDist.end());
			TensorDistribution newTenDist = partialDist;
			newTenDist[tenModeToRedist] = newModeDist;
			newTenDist[order].insert(newTenDist[order].end(), thisRedistModes.begin(), thisRedistModes.end());
			RedistTest newTest;
			newTest.first = newTenDist;
			newTest.second = newCommModes;

			newPartialTests.push_back(newTest);
		}
	}
	CreateGTOGTestsHelper(newTenModesToRedist, distA, newPartialTests, fullTests);
}

std::vector<RedistTest >
CreateGTOGTests(const TensorDistribution& distA){
    Unsigned i;
    std::vector<RedistTest > ret;
    const Unsigned order = distA.size() - 1;
    ModeArray tensorModes = DefaultPermutation(order);

    std::vector<ModeArray> redistModesGroups;
    for(i = 1; i <= order; i++){
    	std::vector<ModeArray> newRedistModesGroups = AllCombinations(tensorModes, i);
    	redistModesGroups.insert(redistModesGroups.end(), newRedistModesGroups.begin(), newRedistModesGroups.end());
    }

    for(i = 0; i < redistModesGroups.size(); i++){
    	ModeArray redistModesGroup = redistModesGroups[i];
    	TensorDistribution resDist = distA;
    	std::vector<RedistTest> partialTests;
    	RedistTest partialTest;
    	ModeArray blankModes;
    	partialTest.first = resDist;
    	partialTest.second = blankModes;
    	partialTests.push_back(partialTest);

    	CreateGTOGTestsHelper(redistModesGroup, distA, partialTests, ret);
    }
    return ret;
}
#endif // ifndef TMEN_TESTS_GTOGREDIST_HPP
