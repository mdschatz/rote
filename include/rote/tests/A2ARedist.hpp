#pragma once
#ifndef ROTE_TESTS_A2AREDIST_HPP
#define ROTE_TESTS_A2AREDIST_HPP

#include "rote/tests/AllRedists.hpp"
using namespace rote;

void CreateA2ATestsSinkHelper(const ModeArray &modesToMove,
                              const ModeArray &sinkModesGroup,
                              const TensorDistribution &distA,
                              const std::vector<RedistTest> &partialTests,
                              std::vector<RedistTest> &fullTests) {
  Unsigned i, j, k;

  if (modesToMove.size() == 0) {
    for (i = 0; i < partialTests.size(); i++) {
      RedistTest partialTest = partialTests[i];
      bool exists = false;
      for (j = 0; j < fullTests.size(); j++) {
        RedistTest check = fullTests[j];
        if (partialTest.first == check.first) {
          exists = true;
          break;
        }
      }
      if (!exists)
        fullTests.push_back(partialTest);
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
    CreateA2ATestsSinkHelper(newModesToMove, sinkModesGroup, distA,
                             newPartialTests, fullTests);
  }
}

void CreateA2ATestsSrcHelper(const ModeArray &srcModesGroup,
                             const ModeArray &sinkModesGroup,
                             const TensorDistribution &distA,
                             const std::vector<RedistTest> &partialTests,
                             std::vector<RedistTest> &fullTests) {
  Unsigned order = distA.size() - 1;
  Unsigned i, j, k;

  if (srcModesGroup.size() == 0) {
    std::vector<ModeArray> sinkTenModesGroups;
    for (i = 1; i < order; i++) {
      std::vector<ModeArray> newTenModesGroups =
          AllCombinations(sinkModesGroup, i);
      sinkTenModesGroups.insert(sinkTenModesGroups.end(),
                                newTenModesGroups.begin(),
                                newTenModesGroups.end());
    }

    for (i = 0; i < partialTests.size(); i++) {
      const TensorDistribution partialDist = partialTests[i].first;
      const ModeArray partialCommModes = partialTests[i].second;

      std::vector<RedistTest> thisPartialTests;
      RedistTest partialTest;
      partialTest.first = partialDist;
      partialTest.second = partialCommModes;
      thisPartialTests.push_back(partialTest);

      CreateA2ATestsSinkHelper(partialCommModes, sinkModesGroup, distA,
                               thisPartialTests, fullTests);
    }
    return;
  }

  for (k = 0; k < srcModesGroup.size(); k++) {
    std::vector<RedistTest> newPartialTests;
    Mode tenModeToRedist = srcModesGroup[k];
    ModeArray newTenModesToRedist = srcModesGroup;
    newTenModesToRedist.erase(newTenModesToRedist.begin() + k);

    for (i = 0; i < partialTests.size(); i++) {
      const TensorDistribution partialDist = partialTests[i].first;
      const ModeArray partialCommModes = partialTests[i].second;
      const ModeDistribution modeDistToRedist = partialDist[tenModeToRedist];

      for (j = 0; j <= modeDistToRedist.size(); j++) {
        ModeArray newCommModes = partialCommModes;
        newCommModes.insert(newCommModes.end(), modeDistToRedist.end() - j,
                            modeDistToRedist.end());
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
    CreateA2ATestsSrcHelper(newTenModesToRedist, sinkModesGroup, distA,
                            newPartialTests, fullTests);
  }
}

std::vector<RedistTest> CreateA2ATests(const TensorDistribution &distA) {
  Unsigned i;
  std::vector<RedistTest> ret;
  const Unsigned order = distA.size() - 1;
  ModeArray tensorModes = OrderedModes(order);

  std::vector<ModeArray> srcTenModesGroups;
  for (i = 1; i < order; i++) {
    std::vector<ModeArray> newRedistModesGroups =
        AllCombinations(tensorModes, i);
    srcTenModesGroups.insert(srcTenModesGroups.end(),
                             newRedistModesGroups.begin(),
                             newRedistModesGroups.end());
  }

  for (i = 0; i < srcTenModesGroups.size(); i++) {
    ModeArray srcModesGroup = srcTenModesGroups[i];
    ModeArray sinkModesGroup;
    std::set_difference(tensorModes.begin(), tensorModes.end(),
                        srcModesGroup.begin(), srcModesGroup.end(),
                        back_inserter(sinkModesGroup));

    TensorDistribution resDist = distA;
    std::vector<RedistTest> partialTests;
    RedistTest partialTest;
    ModeArray blankModes;
    partialTest.first = resDist;
    partialTest.second = blankModes;
    partialTests.push_back(partialTest);

    CreateA2ATestsSrcHelper(srcModesGroup, sinkModesGroup, distA, partialTests,
                            ret);
  }
  return ret;
}

#endif // ifndef ROTE_TESTS_A2AREDIST_HPP
