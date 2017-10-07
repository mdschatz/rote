#pragma once
#ifndef ROTE_TESTS_BCASTREDIST_HPP
#define ROTE_TESTS_BCASTREDIST_HPP

#include "rote/tests/AllRedists.hpp"
using namespace rote;

std::vector<RedistTest> CreateBCastTests(const TensorDistribution &distA) {
  Unsigned i;
  std::vector<RedistTest> ret;
  const Unsigned order = distA.size() - 1;

  for (i = 0; i <= distA[order].size(); i++) {
    RedistTest bcastTest;
    TensorDistribution resDist = distA;
    ModeArray commModes;
    commModes.insert(commModes.end(), resDist[order].end() - i,
                     resDist[order].end());
    resDist[order].erase(resDist[order].end() - i, resDist[order].end());
    bcastTest.first = resDist;
    bcastTest.second = commModes;
    ret.push_back(bcastTest);
  }
  return ret;
}

#endif // ifndef ROTE_TESTS_BCASTREDIST_HPP
