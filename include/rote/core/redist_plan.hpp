/*
   Copyright (c) 2009-2013, Jack Poulson
                      2013, Jed Brown
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once

namespace rote {

class Redist
{
public:
  Redist(
    const TensorDistribution& dB,
    const TensorDistribution& dA,
    const RedistType& type,
    const ModeArray& modes
  );

  ~Redist();

  const TensorDistribution& dB() const {return dB_;}
  const TensorDistribution& dA() const {return dA_;}
  const RedistType& type() const {return type_;}
  const ModeArray& modes() const {return modes_;}

private:
	TensorDistribution dB_;
	TensorDistribution dA_;
	RedistType type_;
	ModeArray modes_;
};

class RedistPlanInfo
{
public:
  RedistPlanInfo(
    const TensorDistribution& dB,
    const TensorDistribution& dA,
    const ModeArray& reduceModes
  );

  ~RedistPlanInfo() {};

  ModeArray& reduced() {return reduced_;}
  const ModeArray& reduced() const {return reduced_;}
  std::map<Mode, Mode>& added() {return added_;}
  const std::map<Mode, Mode>& added() const {return added_;}
  std::map<Mode, Mode>& removed() {return removed_;}
  const std::map<Mode, Mode>& removed() const {return removed_;}
  std::map<Mode, std::pair<Mode, Mode>>& moved() {return moved_;}
  const std::map<Mode, std::pair<Mode, Mode>>& moved() const {return moved_;}

private:
  ModeArray reduced_;
	std::map<Mode, Mode> added_;  // Val is tMode adding gMode
	std::map<Mode, Mode> removed_;  // Val is tMode rmving gMode
	std::map<Mode, std::pair<Mode, Mode>> moved_; // Val is (tMode add, tMode rmv)
};

class RedistPlan
{
public:
  RedistPlan(
    const TensorDistribution& dB,
    const TensorDistribution& dA,
    const ModeArray& reduceModes,
    const Grid& g
  );

  const Redist& operator[](const int index) const {
    return index < 0 ? plan_[plan_.size() + index] : plan_[index];
  }
  int size() const {return plan_.size();}

  ~RedistPlan() {};

  void MoveOpt();
  void MoveSimple();
  void MoveComplex();
  void Add();
  void Move();
  void Remove();
  void ShuffleTo(const TensorDistribution& dB);

private:
  RedistPlanInfo info_;
  std::vector<Redist> plan_;
  TensorDistribution dCur_;
  TensorDistribution dB_;
  const Grid& g_;
};

} // namespace rote
