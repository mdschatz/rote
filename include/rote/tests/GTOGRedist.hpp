#pragma once
#ifndef ROTE_TESTS_GTOGREDIST_HPP
#define ROTE_TESTS_GTOGREDIST_HPP

using namespace rote;

void
CreateGTOGTestsHelper(const ModeArray& tenModesToRedist, const TensorDistribution& distA, const std::vector<RedistTest>& partialTests, std::vector<RedistTest>& fullTests){
	Unsigned order = distA.size() - 1;
	Unsigned i, j;

	if(tenModesToRedist.size() == 0){
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

	std::vector<RedistTest > newPartialTests;
	Mode tenModeToRedist = tenModesToRedist[tenModesToRedist.size() - 1];
	ModeArray newTenModesToRedist(tenModesToRedist.begin(), tenModesToRedist.end() - 1);

	for(i = 0; i < partialTests.size(); i++){
		const TensorDistribution partialDist = partialTests[i].first;
		const ModeArray partialCommModes = partialTests[i].second;
		const ModeDistribution modeDistToRedist = partialDist[tenModeToRedist];

		for(j = 0; j <= modeDistToRedist.size(); j++){
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
    ModeArray tensorModes = OrderedModes(order);

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
#endif // ifndef ROTE_TESTS_GTOGREDIST_HPP
