#pragma once
#ifndef TMEN_TESTS_BCASTREDIST_HPP
#define TMEN_TESTS_BCASTREDIST_HPP

#include "tensormental/tests/AllRedists.hpp"
using namespace tmen;

std::vector<RedistTest >
CreateBCastTests(const TensorDistribution& distA){
    Unsigned i;
    std::vector<RedistTest > ret;
    const Unsigned order = distA.size() - 1;

    for(i = 1; i <= distA[order].size(); i++){
    	RedistTest bcastTest;
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

#endif // ifndef TMEN_TESTS_BCASTREDIST_HPP
