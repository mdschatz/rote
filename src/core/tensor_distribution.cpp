/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#include "rote.hpp"

namespace rote {

TensorDistribution::TensorDistribution() : entries_() {}

TensorDistribution::TensorDistribution(Unsigned order) : entries_(order + 1) {}

TensorDistribution::TensorDistribution(
    const std::vector<ModeDistribution> &dist)
    : entries_(dist) {
  CheckIsValid();
}

TensorDistribution::TensorDistribution(const TensorDistribution &dist)
    : entries_(dist.entries_) {}

TensorDistribution::TensorDistribution(const std::string &dist)
    : entries_(StringToTensorDist(dist).entries_) {}

TensorDistribution::~TensorDistribution() {}

Unsigned TensorDistribution::size() const { return entries_.size(); }

std::vector<ModeDistribution> TensorDistribution::Entries() const {
  return entries_;
}

const ModeDistribution &TensorDistribution::operator[](size_t index) const {
  return entries_[index];
}

TensorDistribution &TensorDistribution::
operator=(const TensorDistribution &rhs) {
  entries_ = rhs.entries_;
  return *this;
}

ModeDistribution TensorDistribution::UnusedModes() const {
  return entries_[entries_.size() - 1];
}

void TensorDistribution::SetToMatch(const TensorDistribution &other,
                                    const IndexArray &otherIndices,
                                    const IndexArray &myIndices) {
  Unsigned i;
  for (i = 0; i < myIndices.size(); i++) {
    int index = IndexOf(otherIndices, myIndices[i]);
    if (index >= 0)
      entries_[i] = other[index];
  }
  entries_[entries_.size() - 1] = other[other.size() - 1];
}

void TensorDistribution::AppendToMatchForGridModes(
    const ModeArray &gridModes, const TensorDistribution &other,
    const IndexArray &otherIndices, const IndexArray &myIndices) {
  Unsigned i;
  TensorDistribution gridModeDistFinal =
      GetTensorDistForGridModes(other, gridModes);
  for (i = 0; i < myIndices.size(); i++) {
    int index = IndexOf(otherIndices, myIndices[i]);
    if (index >= 0)
      entries_[i] += gridModeDistFinal.entries_[index];
  }
}

void TensorDistribution::RemoveUnitModeDists(
    const std::vector<Unsigned> &unitModes) {
  Unsigned i;
  std::vector<Unsigned> sorted = unitModes;
  SortVector(sorted);

  for (i = sorted.size() - 1; i < sorted.size(); i--)
    entries_.erase(entries_.begin() + sorted[i]);
}

void TensorDistribution::IntroduceUnitModeDists(
    const std::vector<Unsigned> &unitModes) {
  Unsigned i;
  std::vector<Unsigned> sorted = unitModes;
  SortVector(sorted);

  ModeDistribution blank;
  for (i = 0; i < sorted.size(); i++)
    entries_.insert(entries_.begin() + sorted[i], blank);
}

TensorDistribution
TensorDistribution::Filter(const std::vector<Unsigned> &filterIndices) const {
  TensorDistribution ret(filterIndices.size() + 1);
  for (int i = 0; i < filterIndices.size(); i++) {
    Unsigned filterIndex = filterIndices[i];
    if (filterIndex < entries_.size())
      ret.entries_[i] = entries_[filterIndex];
  }
  ret.entries_[ret.entries_.size() - 1] = entries_[entries_.size() - 1];

  return ret;
}

TensorDistribution &TensorDistribution::
operator+=(const TensorDistribution &rhs) {
  if (entries_.size() != rhs.entries_.size())
    LogicError("Tensor distributions of unequal size");

  Unsigned i;
  for (i = 0; i < entries_.size(); i++)
    entries_[i] += rhs.entries_[i];
  return *this;
}

TensorDistribution &TensorDistribution::
operator-=(const TensorDistribution &rhs) {
  if (entries_.size() != rhs.entries_.size())
    LogicError("Tensor distributions of unequal size");

  Unsigned i;
  for (i = 0; i < entries_.size(); i++)
    entries_[i] -= rhs.entries_[i];
  return *this;
}

TensorDistribution operator-(const TensorDistribution &lhs,
                             const TensorDistribution &rhs) {
  TensorDistribution ret(lhs);
  for (int i = 0; i < rhs.entries_.size(); i++) {
    ret.entries_[i] = lhs.entries_[i] - rhs.entries_[i];
  }
  return ret;
}

bool operator<=(const TensorDistribution &lhs, const TensorDistribution &rhs) {
  Unsigned i;
  if (lhs.size() != rhs.size())
    LogicError("Tensor distributions must be of equal size to compare");

  for (i = 0; i < lhs.entries_.size(); i++)
    if (!(lhs.entries_[i] <= rhs.entries_[i]))
      return false;
  return true;
}

bool operator<(const TensorDistribution &lhs, const TensorDistribution &rhs) {
  Unsigned i;
  if (lhs.size() != rhs.size())
    LogicError("Tensor distributions must be of equal size to compare");

  for (i = 0; i < lhs.entries_.size(); i++)
    if (!(lhs.entries_[i] < rhs.entries_[i]))
      return false;
  return true;
}

bool operator>=(const TensorDistribution &lhs, const TensorDistribution &rhs) {
  return rhs <= lhs;
}

bool operator>(const TensorDistribution &lhs, const TensorDistribution &rhs) {
  return rhs < lhs;
}

TensorDistribution operator+(const TensorDistribution &lhs,
                             const TensorDistribution &rhs) {
  TensorDistribution ret(lhs);
  for (int i = 0; i < rhs.entries_.size(); i++) {
    ret.entries_[i] = lhs.entries_[i] + rhs.entries_[i];
  }
  return ret;
}

TensorDistribution GetCommonSuffix(const TensorDistribution &lhs,
                                   const TensorDistribution &rhs) {
  if (lhs.entries_.size() != rhs.entries_.size())
    LogicError(
        "Tensor distributions must be of equal size to extract common suffix");

  Unsigned i;
  TensorDistribution ret;
  ret.entries_.resize(lhs.entries_.size());
  for (i = 0; i < lhs.entries_.size(); i++)
    ret.entries_[i] = GetCommonSuffix(lhs.entries_[i], rhs.entries_[i]);
  return ret;
}

TensorDistribution GetCommonPrefix(const TensorDistribution &lhs,
                                   const TensorDistribution &rhs) {
  if (lhs.entries_.size() != rhs.entries_.size())
    LogicError(
        "Tensor distributions must be of equal size to extract common prefix");

  Unsigned i;
  TensorDistribution ret;
  ret.entries_.resize(lhs.entries_.size());
  for (i = 0; i < lhs.entries_.size(); i++)
    ret.entries_[i] = GetCommonPrefix(lhs.entries_[i], rhs.entries_[i]);
  return ret;
}

TensorDistribution
GetTensorDistForGridModes(const TensorDistribution &tenDist,
                          const ModeDistribution &gridModes) {
  Unsigned i;
  ModeDistribution findModes = gridModes;
  TensorDistribution ret(tenDist.size() - 1);
  for (i = 0; i < tenDist.entries_.size(); i++) {
    ModeDistribution overlap = tenDist.entries_[i].overlapWith(findModes);
    ret.entries_[i] = overlap;
    findModes -= overlap;
  }

  if (findModes.size() != 0)
    LogicError("Could not find all grid modes in tensor distribution");

  return ret;
}

ModeDistribution TensorDistribution::UsedModes() const {
  Unsigned i;
  ModeDistribution ret;
  for (i = 0; i < entries_.size() - 1; i++)
    ret += entries_[i];
  return ret;
}

void TensorDistribution::CheckIsValid() {
  std::vector<Unsigned> allEntries;
  for (int i = 0; i < entries_.size(); i++) {
    ModeArray modeEntries = entries_[i].Entries();
    allEntries.insert(allEntries.end(), modeEntries.begin(), modeEntries.end());
  }

  std::vector<Unsigned> unique = Unique(allEntries);
  if (unique.size() != allEntries.size())
    LogicError("Invalid Tensor Distribution");
}

bool operator==(const TensorDistribution &A, const TensorDistribution &B) {
  if (A.entries_.size() != B.entries_.size())
    return false;

  for (int i = 0; i < A.entries_.size(); i++)
    if (A.entries_[i] != B.entries_[i])
      return false;
  return true;
}

bool operator!=(const TensorDistribution &A, const TensorDistribution &B) {
  return !(A == B);
}

std::ostream &operator<<(std::ostream &os, const TensorDistribution &dist) {
  os << TensorDistToString(dist);
  return os;
}

inline std::string TensorDistToString(const TensorDistribution &distribution,
                                      bool endLine) {
  std::stringstream ss;
  ss << "[";
  if (distribution.size() > 1) {
    ss << ModeDistToString_(distribution[0]);
    for (size_t i = 1; i < distribution.size() -
1; i++)
      ss << ", " << ModeDistToString_(distribution[i]);
  }
  ss << "]|";
  ss << ModeDistToString_(distribution[distribution.size() - 1]);
  if (endLine)
    ss << std::endl;
  return ss.str();
}

Mode TensorDistribution::TensorModeForGridMode(const Mode &mode) const {
  Unsigned i;
  for (i = 0; i < entries_.size(); i++)
    if (entries_[i].Contains(mode))
      return i;
  return entries_.size();
}

// TODO: Figure out how to error check these without C++11
inline TensorDistribution StringToTensorDist(const std::string &s) {
  std::vector<ModeDistribution> distVals;
  ModeDistribution ignoreModes;

  size_t pos, lastPos, breakPos;
  breakPos = s.find_first_of("|");
  pos = s.find_first_of("[");
  lastPos = s.find_first_of("]");
  pos = s.find_first_of("(", pos);
  while (pos < breakPos) {
    lastPos = s.find_first_of(")", pos);
    distVals.push_back(StringToModeDist(s.substr(pos, lastPos - pos + 1)));
    pos = s.find_first_of("(", lastPos + 1);
  }

  if (breakPos != std::string::npos) {
    // Break found, ignore modes
    ignoreModes =
        StringToModeDist(s.substr(breakPos + 1, s.length() - breakPos + 1));
  }
  distVals.push_back(ignoreModes);

  TensorDistribution ret(distVals);
  return ret;
}
} // namespace rote
