#pragma once
#ifndef ROTE_TESTS_RSGREDIST_HPP
#define ROTE_TESTS_RSGREDIST_HPP

#include "rote/tests/AllRedists.hpp"
using namespace rote;

void CreateRSGTestsSinkHelper(const ModeArray &modesToMove,
                              const ModeArray &sinkModesGroup,
                              const ModeArray &reduceModes,
                              const TensorDistribution &distA,
                              const std::vector<RedistTest> &partialTests,
                              std::vector<RedistTest> &fullTests) {
  Unsigned i, j, k;

  if (modesToMove.size() == 0) {
    for (i = 0; i < partialTests.size(); i++) {
      RedistTest partialTest = partialTests[i];
      TensorDistribution resDist = partialTest.first;

      ModeArray sortedReduceModes = reduceModes;
      SortVector(sortedReduceModes);

      for (j = sortedReduceModes.size() - 1; j < sortedReduceModes.size();
           j--) {
        resDist.erase(resDist.begin() + sortedReduceModes[j]);
      }
      RedistTest newPartialTest;
      newPartialTest.first = resDist;
      newPartialTest.second = partialTest.second;

      bool exists = false;
      for (j = 0; j < fullTests.size(); j++) {
        RedistTest check = fullTests[j];
        if (newPartialTest.first == check.first) {
          exists = true;
          break;
        }
      }
      if (!exists)
        fullTests.push_back(newPartialTest);
    }

    return;
  }

  for (k = 0; k < modesToMove.size(); k++) {
    std::vector<RedistTest> newPartialTests;
    Mode modeToMove = modesToMove[k];
    ModeArray newModesToMove = modesToMove;
    newModesToMove.erase(newModesToMove.begin() + k);

    for (i = 0; i < partialTests.size(); i++) {
      const TensorDistribution partialDist = partialTests[i].first;
      const ModeArray partialModes = partialTests[i].second;

      for (j = 0; j < sinkModesGroup.size(); j++) {
        Mode modeDistToChange = sinkModesGroup[j];
        TensorDistribution resDist = partialDist;
        resDist[modeDistToChange].push_back(modeToMove);
        RedistTest newTest;
        newTest.first = resDist;
        newTest.second = partialModes;

        newPartialTests.push_back(newTest);
      }
    }
    CreateRSGTestsSinkHelper(newModesToMove, sinkModesGroup, reduceModes, distA,
                             newPartialTests, fullTests);
  }
}

std::vector<RedistTest> CreateRSGTests(const TensorDistribution &distA) {
  Unsigned i, j;
  std::vector<RedistTest> ret;
  const Unsigned order = distA.size() - 1;
  ModeArray tensorModes = OrderedModes(order);

  std::vector<ModeArray> redistModesGroups;
  for (i = 1; i < order; i++) {
    std::vector<ModeArray> newRedistModesGroups =
        AllCombinations(tensorModes, i);
    redistModesGroups.insert(redistModesGroups.end(),
                             newRedistModesGroups.begin(),
                             newRedistModesGroups.end());
  }

  for (i = 0; i < redistModesGroups.size(); i++) {
    ModeArray redistModesGroup = redistModesGroups[i];
    TensorDistribution resDist = distA;
    ModeArray sinkModesGroup;
    std::set_difference(tensorModes.begin(), tensorModes.end(),
                        redistModesGroup.begin(), redistModesGroup.end(),
                        back_inserter(sinkModesGroup));

    SortVector(redistModesGroup);

    ModeArray commModes;
    for (j = redistModesGroup.size() - 1; j < redistModesGroup.size(); j--) {
      Mode redistTenMode = redistModesGroup[j];
      ModeDistribution redistModeDist = resDist[redistTenMode];
      commModes.insert(commModes.end(), redistModeDist.begin(),
                       redistModeDist.end());
      ModeDistribution blank;
      resDist[redistTenMode] = blank;
    }

    std::vector<RedistTest> partialTests;
    RedistTest partialTest;
    partialTest.first = resDist;
    partialTest.second = redistModesGroup;
    partialTests.push_back(partialTest);

    CreateRSGTestsSinkHelper(commModes, sinkModesGroup, redistModesGroup, distA,
                             partialTests, ret);
  }
  return ret;
}

#endif // ifndef ROTE_TESTS_RSGREDIST_HPP
