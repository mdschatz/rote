#pragma once
#ifndef ROTE_TESTS_LGREDIST_HPP
#define ROTE_TESTS_LGREDIST_HPP

using namespace rote;

void
CreateLGTestsHelper(const ModeArray& modesToMove, const TensorDistribution& distA, const std::vector<RedistTest>& partialTests, std::vector<RedistTest>& fullTests){
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
		ModeArray newModesToMove = modesToMove;
		Mode modeToMove = newModesToMove[k];
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
		CreateLGTestsHelper(newModesToMove, distA, newPartialTests, fullTests);
	}
}

std::vector<RedistTest>
CreateLGTests(const Unsigned& gridOrder, const TensorDistribution& distA){
    Unsigned i;
    std::vector<RedistTest > ret;

    const Unsigned order = distA.size() - 1;

    ModeArray gridModes = OrderedModes(gridOrder);
    ModeArray usedGridModes;

    for(i = 0; i <= order; i++)
    	usedGridModes.insert(usedGridModes.end(), distA[i].begin(), distA[i].end());
    SortVector(usedGridModes);

    ModeArray unusedGridModes;
    std::set_difference(gridModes.begin(), gridModes.end(), usedGridModes.begin(), usedGridModes.end(), back_inserter(unusedGridModes));

    std::vector<ModeArray> redistModesGroups;
    for(i = 1; i < order; i++){
    	std::vector<ModeArray> newRedistModesGroups = AllCombinations(unusedGridModes, i);
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

    	CreateLGTestsHelper(redistModesGroup, distA, partialTests, ret);
    }
    return ret;
}

#endif // ifndef ROTE_TESTS_LGREDIST_HPP
