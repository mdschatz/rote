/*
   Copyright (c) 2009-2013, Jack Poulson
                      2013, Jed Brown
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/

#include "rote.hpp"

namespace rote {
////
// Redist
////

Redist::Redist(
  const TensorDistribution& dB,
  const TensorDistribution& dA,
  const RedistType& type,
  const ModeArray& modes
): dB_(dB), dA_(dA), type_(type), modes_(modes) {}

Redist::~Redist() {}

////
// RedistPlanInfo
////

RedistPlanInfo::RedistPlanInfo(
  const TensorDistribution& dB,
  const TensorDistribution& dA,
  const ModeArray& reduceModes
) {
  reduceModes_ = reduceModes;
  SortVector(reduceModes_);

  for(Mode rMode: reduceModes) {
    for (Mode gMode: dA[rMode].Entries()) {
      reduceGModes_.push_back(gMode);
    }
  }

  // Determine grid modes that are removed/moved
  ModeArray gModesA = dA.UsedModes().Entries();
  for (Mode m: gModesA) {
    Mode tModeA = dA.TensorModeForGridMode(m);
    Mode tModeB = dB.TensorModeForGridMode(m);

    // Check remove
    if (tModeB == dB.size()) {
      removed_[m] = dA.TensorModeForGridMode(m);
    }
    // Check move
    else if (tModeB != tModeA || (reduceModes_.size() != 0)) {
      // TODO: proper guard for mode elements < m
      std::pair<Mode, Mode> entry(tModeB, tModeA);
      moved_[m] = entry;
    }
  }

  // Determine grid modes added
  ModeArray gModesAdded = DiffVector(dB.UsedModes().Entries(), gModesA);
  for (Mode m: gModesAdded) {
    added_[m] = dB.TensorModeForGridMode(m);
  }
}

////
// RedistPlan
////

void
RedistPlan::AllReduceOpt() {
  if (info_.reduceGModes().size() == 0 || !info_.moved().empty()) {
    return;
  }

  TensorDistribution dB = dCur_;
  dB.RemoveUnitModeDists(info_.reduceModes());

  for (const auto& kv: info_.removed()) {
    for (int i = info_.reduceModes().size() - 1; i < info_.reduceModes().size(); i--) {
      Mode m = info_.reduceModes()[i];
      if (m > kv.second) {
        continue;
      } else if (m == kv.second) {
        info_.removed()[kv.first] = 0;
        break;
      } else if (m < kv.second) {
        info_.removed()[kv.first]--;
      }
    }
  }
  for (const auto& kv: info_.moved()) {
    for (int i = info_.reduceModes().size() - 1; i < info_.reduceModes().size(); i--) {
      Mode m = info_.reduceModes()[i];
      if (m > kv.second.second) {
        continue;
      } else if (m == kv.second.second) {
        info_.moved()[kv.first].second = 0;
        break;
      } else if (m < kv.second.second) {
        info_.moved()[kv.first].second--;
      }
    }
  }
  ModeArray erased;
  for (const auto& kv: info_.moved()) {
    std::pair<Mode, Mode> moveInfo = info_.moved()[kv.first];
    if (moveInfo.first == moveInfo.second) {
      erased.push_back(kv.first);
    }
  }
  for (Mode m: erased) {
    info_.moved().erase(m);
  }

  Redist redist(dB, dCur_, AR, info_.reduceGModes());
  plan_.push_back(redist);
  dCur_ = dB;

  for(Mode m: info_.reduceGModes()) {
    info_.removed().erase(m);
  }
  info_.reduceModes().clear();
  info_.reduceGModes().clear();
}

void RedistPlan::ReduceScatter() {
  if (info_.reduceModes().size() == 0) {
    return;
  }

  // Remove reduced mode distributions
  TensorDistribution dB = dCur_;
  dB.RemoveUnitModeDists(info_.reduceModes());

  for (const auto& kv: info_.removed()) {
    for (int i = info_.reduceModes().size() - 1; i < info_.reduceModes().size(); i--) {
      Mode m = info_.reduceModes()[i];
      if (m > kv.second) {
        continue;
      } else if (m == kv.second) {
        info_.removed()[kv.first] = 0;
        break;
      } else if (m < kv.second) {
        info_.removed()[kv.first]--;
      }
    }
  }
  for (const auto& kv: info_.moved()) {
    for (int i = info_.reduceModes().size() - 1; i < info_.reduceModes().size(); i--) {
      Mode m = info_.reduceModes()[i];
      if (m > kv.second.second) {
        continue;
      } else if (m == kv.second.second) {
        info_.moved()[kv.first].second = 0;
        break;
      } else if (m < kv.second.second) {
        info_.moved()[kv.first].second--;
      }
    }
  }
  ModeArray erased;
  for (const auto& kv: info_.moved()) {
    std::pair<Mode, Mode> moveInfo = info_.moved()[kv.first];
    if (moveInfo.first == moveInfo.second) {
      erased.push_back(kv.first);
    }
  }
  for (Mode m: erased) {
    info_.moved().erase(m);
  }

  // Scatter grid modes
  for (Mode gMode: info_.reduceGModes()) {
    if (info_.moved().find(gMode) != info_.moved().end()) {
      dB[info_.moved()[gMode].first] += gMode;
    } else if (dB.size() >= 1) {
      // Removed grid modes get added to tensor mode 0
      dB[0] += gMode;
    }
    info_.moved().erase(gMode);
  }
  info_.reduceModes().clear();
  info_.reduceGModes().clear();

  Redist redist(dB, dCur_, RS, info_.reduceGModes());
  plan_.push_back(redist);
  dCur_ = dB;
}

void
RedistPlan::Reduce() {
  AllReduceOpt();
  ReduceScatter();
}

void
RedistPlan::Add() {
  if (info_.added().size() == 0) {
    return;
  }

  TensorDistribution dB = dCur_;
  ModeArray commModes;
  for(auto const& kv: info_.added()) {
    dB[kv.second] += kv.first;
    commModes.push_back(kv.first);
  }

  info_.added().clear();

  Redist redist(dB, dCur_, Local, commModes);
  plan_.push_back(redist);
  dCur_ = dB;
}

void
RedistPlan::Remove() {
  if (info_.removed().size() == 0) {
    return;
  }

  TensorDistribution dInt = dCur_;
  TensorDistribution dB = dCur_;
  ModeArray commModes;

  // Create intermediate distribution (removed modes to end)
  for(auto const& kv: info_.removed()) {
    dInt[kv.second] -= kv.first;
    dInt[kv.second] += kv.first;

    dB[kv.second] -= kv.first;
    commModes.push_back(kv.first);
  }
  info_.removed().clear();
  ShuffleTo(dInt);

  if (dB == dInt) {
    return;
  }

  Redist redist(dB, dInt, AG, commModes);
  plan_.push_back(redist);
  dCur_ = dB;
}

void
RedistPlan::ShuffleTo(const TensorDistribution& dB) {
  if (dB == dCur_) {
    return;
  }

  ModeArray commModes;
  TensorDistribution diff = dB - (dB.GetCommonPrefix(dCur_));
  for(int i = 0; i < dB.size(); i++) {
    commModes.insert(commModes.end(), diff[i].Entries().begin(), diff[i].Entries().end());
  }

  Redist redist(dB, dCur_, Perm, commModes);
  plan_.push_back(redist);
  dCur_ = dB;
}

void
RedistPlan::MoveOpt() {
  if (info_.moved().size() == 0) {
    return;
  }

  int maxP = 0;
  std::vector<unsigned> bestComm(info_.moved().size(), 0);

  // Optimization must retain grid view shape.  Initialize
  ObjShape shapeGV(dCur_.size(), 1);
  for(int i = 0; i < shapeGV.size(); i++) {
    for(int j = 0; j < dCur_[i].size(); j++) {
      shapeGV[i] *= g_.Dimension(dCur_[i][j]);
    }
  }

  // Initialize extra map to index linearly into grid mode map
  std::map<int, Mode> gModeMap;
  int count = 0;
  for(auto const& kv: info_.removed()) {
    gModeMap[count++] = kv.first;
  }

  // Iterate through all combinations to find the best optimization
  std::vector<unsigned> testComm(bestComm.size(), 0);
  std::vector<unsigned> end(bestComm.size(), 2);
  unsigned p = 0, o = end.size();
  while(p != 0 && ElemwiseLessThan(testComm, end)) {
    int testCommProc = 1;
    ObjShape testShape(shapeGV);
    for(int i = 0; i < testComm.size(); i++) {
      if (testComm[i]) {
        Mode gMode = gModeMap[i];
        std::pair<Mode, Mode> moveInfo = info_.moved()[gMode];
        int gDim = g_.Dimension(gMode);

        testShape[moveInfo.first] *= gDim;
        testShape[moveInfo.second] /= gDim;
        testCommProc *= gDim;
      }
    }

    // Update best optimization
    if(AnyElemwiseNotEqual(testShape, shapeGV) && testCommProc > maxP) {
      maxP = testCommProc;
      bestComm = testComm;
    }

    while(p < o && testComm[p] >= end[p]) {
      testComm[p] = 0;
      p++;
      if (p == o) {
        break;
      }
      testComm[p]++;
    }
    if (p != o) {
      p = 0;
    }
  }

  // Apply the optimization
  TensorDistribution dB = dCur_;
  for(int i = 0; i < bestComm.size(); i++) {
    if (bestComm[i]) {
      Mode gMode = gModeMap[bestComm[i]];
      std::pair<Mode, Mode> moveInfo = info_.moved()[gMode];

      dB[moveInfo.second] -= gMode;
      dB[moveInfo.first] += gMode;
      info_.moved().erase(gMode);
    }
  }

  ShuffleTo(dB);
}

void
RedistPlan::MoveComplex() {
  if (info_.moved().size() == 0) {
    return;
  }

  std::vector<Mode> modes;
  std::map<Mode, int> modeMap;
  for(auto const& kv: info_.moved()) {
    std::pair<Mode, Mode> moveInfo = kv.second;
    if (!Contains(modes, moveInfo.first)) {
      modes.push_back(moveInfo.first);
    }
    if (!Contains(modes, moveInfo.second)) {
      modes.push_back(moveInfo.second);
    }
  }
  for(int i = 0; i < modes.size(); i++) {
    modeMap[modes[i]] = i;
  }

  int minDiff = info_.moved().size();
  std::vector<Mode> bestComm;

  std::vector<unsigned> testComm(modeMap.size(), 0);
  std::vector<unsigned> end(modeMap.size(), 2);
  unsigned p = 0, o = end.size();
  while(p != o && ElemwiseLessThan(testComm, end)) {

    std::vector<Mode> commModes;
    for(auto const& kv: info_.moved()) {
      std::pair<Mode, Mode> moveInfo = kv.second;
      if (
          testComm[modeMap[moveInfo.first]] == 1 &&
          testComm[modeMap[moveInfo.second]] == 0
      ) {
        commModes.push_back(kv.first);
      }
    }

    int diff = abs(int(2 * sum(testComm) - testComm.size()));
    if (diff < minDiff) {
      minDiff = diff;
      if (commModes.size() > bestComm.size()) {
        bestComm = commModes;
      }
    }

    // Update
    testComm[p]++;
    while(p < o && testComm[p] >= end[p]) {
      testComm[p] = 0;
      p++;
      if (p == o) {
        break;
      }
      testComm[p]++;
    }
    if (p != o) {
      p = 0;
    }
  }

  // Best communication pattern discovered
  TensorDistribution dInt = dCur_;
  TensorDistribution dB = dCur_;
  for(int i = 0; i < bestComm.size(); i++) {
    std::pair<Mode, Mode> moveInfo = info_.moved()[bestComm[i]];
    dInt[moveInfo.second] -= bestComm[i];
    dInt[moveInfo.second] += bestComm[i];

    dB[moveInfo.second] -= bestComm[i];
    dB[moveInfo.first] += bestComm[i];
    info_.moved().erase(bestComm[i]);
  }
  ShuffleTo(dInt);

  if (dB == dInt) {
    return;
  }

  Redist redist(dB, dInt, A2A, bestComm);
  plan_.push_back(redist);
  dCur_ = dB;
}

void
RedistPlan::MoveSimple() {
  if (info_.moved().size() == 0) {
    return;
  }

  std::vector<Mode> srcModes;
  std::vector<Mode> dstModes;
  for(auto const& kv: info_.moved()) {
    std::pair<Mode, Mode> moveInfo = kv.second;
    dstModes.push_back(moveInfo.first);
    srcModes.push_back(moveInfo.second);
  }

  std::vector<Mode> pureDstModes = DiffVector(dstModes, srcModes);
  std::vector<Mode> commModes;
  for(auto const& kv: info_.moved()) {
    std::pair<Mode, Mode> moveInfo = kv.second;
    if (Contains(pureDstModes, moveInfo.first)) {
      commModes.push_back(kv.first);
    }
  }

  // Set up intermediate
  TensorDistribution dInt = dCur_;
  TensorDistribution dB = dCur_;
  for(int i = 0; i < commModes.size(); i++) {
    Mode commMode = commModes[i];
    std::pair<Mode, Mode> moveInfo = info_.moved()[commMode];
    dInt[moveInfo.second] -= commMode;
    dInt[moveInfo.second] += commMode;

    dB[moveInfo.second] -= commMode;
    dB[moveInfo.first] += commMode;
    info_.moved().erase(commMode);
  }
  ShuffleTo(dInt);

  if (dB == dInt) {
    return;
  }

  Redist redist(dB, dInt, A2A, commModes);
  plan_.push_back(redist);
  dCur_ = dB;
}

void
RedistPlan::Move() {
  if (info_.moved().size() == 0) {
    return;
  }

  // Try to convert as many communications to point-to-point
  MoveOpt();
  while (info_.moved().size() > 0) {
    MoveSimple();
    MoveComplex();
  }
}

RedistPlan::RedistPlan(
  const TensorDistribution& dB,
  const TensorDistribution& dA,
  const ModeArray& reduceModes,
  const Grid& g
): info_(dB, dA, reduceModes), plan_(), dCur_(dA), dB_(dB), g_(g) {
  if (dB_ == dCur_) {
    return;
  }

  Add();
  RedistPlanInfo postAddInfo(dB, dCur_, reduceModes);
  // PrintRedistPlanInfo(info_, "RedistPlanInfo");

  info_ = postAddInfo;
  Reduce();
  Move();
  Remove();
  ShuffleTo(dB_);
}

}
