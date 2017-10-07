#pragma once
#ifndef ROTE_TESTS_ARREDIST_HPP
#define ROTE_TESTS_ARREDIST_HPP

#include "rote/tests/AllRedists.hpp"
using namespace rote;

std::vector<RedistTest >
CreateARTests(const TensorDistribution& distA){
    Unsigned i, j;
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

    	SortVector(redistModesGroup);
    	for(j = redistModesGroup.size() - 1; j < redistModesGroup.size(); j--){
    		resDist.erase(resDist.begin() + redistModesGroup[j]);
    	}

    	RedistTest test;
    	test.first = resDist;
    	test.second = redistModesGroup;
    	ret.push_back(test);
    }
    return ret;
}

#endif // ifndef ROTE_TESTS_ARREDIST_HPP
