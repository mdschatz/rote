#pragma once
#ifndef ROTE_TESTS_PREDIST_HPP
#define ROTE_TESTS_PREDIST_HPP

#include "rote/tests/AllRedists.hpp"
using namespace rote;

void
CreatePTestsSinkHelper(const ModeArray& modesToMove, const ModeArray& sinkModesGroup, const Grid& g, const TensorDistribution& distA, const std::vector<RedistTest>& partialTests, std::vector<RedistTest>& fullTests){
	Unsigned order = distA.size() - 1;
	Unsigned i, j, k;

	if(modesToMove.size() == 0){
		for(i = 0; i < partialTests.size(); i++){
			RedistTest partialTest = partialTests[i];
			TensorDistribution partialDist = partialTest.first;

			bool isValid = true;
			for(j = 0; j < order; j++){
				if(prod(FilterVector(g.Shape(), partialDist[j])) != prod(FilterVector(g.Shape(), distA[j]))){
					isValid = false;
					break;
				}
			}
			if(isValid){
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

			for(j = 0; j < sinkModesGroup.size(); j++){
				Mode modeDistToChange = sinkModesGroup[j];
				TensorDistribution resDist = partialDist;
				resDist[modeDistToChange].push_back(modeToMove);
				RedistTest newTest;
				newTest.first = resDist;
				newTest.second = partialModes;

				newPartialTests.push_back(newTest);
			}
		}
		CreatePTestsSinkHelper(newModesToMove, sinkModesGroup, g, distA, newPartialTests, fullTests);
	}
}

void
CreatePTestsSrcHelper(const ModeArray& srcModesGroup, const ModeArray& sinkModesGroup, const Grid& g, const TensorDistribution& distA, const std::vector<RedistTest>& partialTests, std::vector<RedistTest>& fullTests){
	Unsigned i, j;

	if(srcModesGroup.size() == 0){
		for(i = 0; i < partialTests.size(); i++){
			const TensorDistribution partialDist = partialTests[i].first;
			const ModeArray partialCommModes = partialTests[i].second;

			std::vector<RedistTest> thisPartialTests;
			RedistTest partialTest;
			partialTest.first = partialDist;
			partialTest.second = partialCommModes;
			thisPartialTests.push_back(partialTest);

			CreatePTestsSinkHelper(partialCommModes, sinkModesGroup, g, distA, thisPartialTests, fullTests);
		}
		return;
	}

	std::vector<RedistTest > newPartialTests;
	Mode tenModeToRedist = srcModesGroup[srcModesGroup.size() - 1];
	ModeArray newTenModesToRedist(srcModesGroup.begin(), srcModesGroup.end() - 1);

	for(i = 0; i < partialTests.size(); i++){
		const TensorDistribution partialDist = partialTests[i].first;
		const ModeArray partialCommModes = partialTests[i].second;
		const ModeDistribution modeDistToRedist = partialDist[tenModeToRedist];

		for(j = 0; j <= modeDistToRedist.size(); j++){
			ModeArray newCommModes = partialCommModes;
			newCommModes.insert(newCommModes.end(), modeDistToRedist.end() - j, modeDistToRedist.end());
			ModeDistribution newModeDist = modeDistToRedist;
			newModeDist.erase(newModeDist.end() - j, newModeDist.end());
			TensorDistribution newTenDist = partialDist;
			newTenDist[tenModeToRedist] = newModeDist;
			RedistTest newTest;
			newTest.first = newTenDist;
			newTest.second = newCommModes;

			newPartialTests.push_back(newTest);
		}
	}
	CreatePTestsSrcHelper(newTenModesToRedist, sinkModesGroup, g, distA, newPartialTests, fullTests);
}

std::vector<RedistTest>
CreatePTests(const Grid& g, const TensorDistribution& distA){
    Unsigned i;
    std::vector<RedistTest > ret;
    const Unsigned order = distA.size() - 1;
    ModeArray tensorModes = DefaultPermutation(order);

    std::vector<ModeArray> srcTenModesGroups;
    for(i = 1; i < order; i++){
    	std::vector<ModeArray> newRedistModesGroups = AllCombinations(tensorModes, i);
    	srcTenModesGroups.insert(srcTenModesGroups.end(), newRedistModesGroups.begin(), newRedistModesGroups.end());
    }

    for(i = 0; i < srcTenModesGroups.size(); i++){
    	ModeArray srcModesGroup = srcTenModesGroups[i];

    	TensorDistribution resDist = distA;
    	std::vector<RedistTest> partialTests;
    	RedistTest partialTest;
    	ModeArray blankModes;
    	partialTest.first = resDist;
    	partialTest.second = blankModes;
    	partialTests.push_back(partialTest);

    	CreatePTestsSrcHelper(srcModesGroup, tensorModes, g, distA, partialTests, ret);
    }
    return ret;
}

#endif // ifndef ROTE_TESTS_PREDIST_HPP
