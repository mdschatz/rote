#pragma once
#ifndef ROTE_TESTS_SCATTERREDIST_HPP
#define ROTE_TESTS_SCATTERREDIST_HPP

#include "rote/tests/AllRedists.hpp"
using namespace rote;

void
CreateScatterTestsHelper(const ModeArray& modesToMove, const TensorDistribution& distA, const std::vector<RedistTest>& partialTests, std::vector<RedistTest>& fullTests){
	Unsigned order = distA.size() - 1;
	Unsigned i, j, k;

	if(modesToMove.size() == 0){
		for(i = 0; i < partialTests.size(); i++){
			RedistTest partialTest = partialTests[i];
			bool exists = false;
			for(j = 0; j < fullTests.size(); j++){
				RedistTest check = fullTests[j];
				if(partialTest.first == check.first){
					exists = true;
					break;
				}
			}
			if(!exists)
				fullTests.push_back(partialTest);
		}
		return;
	}

	for(k = 0; k < modesToMove.size(); k++){
		std::vector<RedistTest > newPartialTests;
		Mode modeToMove = modesToMove[k];
		ModeArray newModesToMove = modesToMove;
		newModesToMove.erase(newModesToMove.begin() + k);

		for(i = 0; i < partialTests.size(); i++){
			const TensorDistribution partialDist = partialTests[i].first;
			const ModeArray partialModes = partialTests[i].second;

			for(j = 0; j < order; j++){
				ModeArray resModes = partialModes;
				TensorDistribution resDist = partialDist;
				resDist[j].push_back(modeToMove);
				resModes.push_back(modeToMove);
				RedistTest newTest;
				newTest.first = resDist;
				newTest.second = resModes;

				newPartialTests.push_back(newTest);
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

    for(i = 0; i <= distA[order].size(); i++){
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

#endif // ifndef ROTE_TESTS_SCATTERREDIST_HPP
