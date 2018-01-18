/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#include "rote.hpp"

namespace rote {

ModeDistribution::ModeDistribution()
: entries_()
{ }

ModeDistribution::ModeDistribution(std::initializer_list<Unsigned> list)
: entries_(list)
{ CheckIsValid(); }

ModeDistribution::ModeDistribution(const std::vector<Unsigned>& dist)
: entries_(dist)
{ CheckIsValid(); }

ModeDistribution::ModeDistribution(const ModeDistribution& dist)
: entries_(dist.entries_)
{ }

ModeDistribution::ModeDistribution(const std::string& dist)
: entries_(StringToModeDist(dist).entries_)
{ }

ModeDistribution::~ModeDistribution()
{ }

Unsigned
ModeDistribution::size() const
{ return entries_.size(); }

std::vector<Unsigned>
ModeDistribution::Entries() const
{ return entries_; }

ModeDistribution
operator+(const ModeDistribution& lhs, const ModeDistribution& rhs){
	ModeDistribution ret(lhs);
	ret.entries_.insert(ret.entries_.end(), rhs.entries_.begin(), rhs.entries_.end());
	return ret;
}

ModeDistribution&
ModeDistribution::operator=(const ModeDistribution& rhs){
	entries_ = rhs.entries_;
	return *this;
}

ModeDistribution&
ModeDistribution::operator+=(const ModeDistribution& rhs){
	entries_.insert(entries_.end(), rhs.entries_.begin(), rhs.entries_.end());
	return *this;
}

ModeDistribution&
ModeDistribution::operator+=(const Mode& rhs){
	entries_.push_back(rhs);
	return *this;
}

ModeDistribution&
ModeDistribution::operator-=(const ModeDistribution& rhs){
	for(Unsigned i = 0; i < rhs.entries_.size(); i++)
		(*this) -= rhs.entries_[i];
	return *this;
}

ModeDistribution&
ModeDistribution::operator-=(const Mode& rhs){
	auto loc = std::find(entries_.begin(), entries_.end(), rhs);
	if(loc != entries_.end())
		entries_.erase(loc);
	return *this;
}

ModeDistribution
operator-(const ModeDistribution& lhs, const ModeDistribution& rhs){
	ModeDistribution ret(lhs);
	for(Unsigned i = 0; i < rhs.entries_.size(); i++){
		Unsigned modeToRemove = rhs.entries_[i];
		auto loc = std::find(ret.entries_.begin(), ret.entries_.end(), modeToRemove);
		if(loc != ret.entries_.end())
			ret.entries_.erase(loc);
	}
	return ret;
}

bool
IsPrefix(const ModeDistribution& lhs, const ModeDistribution& rhs){
	Unsigned i;
	for(i = 0; i < lhs.entries_.size(); i++)
		if(!(lhs.entries_[i] == rhs.entries_[i]))
			return false;
	return true;
}

bool
operator<=(const ModeDistribution& lhs, const ModeDistribution& rhs){
	if(lhs.size() > rhs.size())
		return false;

	return IsPrefix(lhs, rhs);
}

bool
operator<(const ModeDistribution& lhs, const ModeDistribution& rhs){
	if(lhs.size() >= rhs.size())
		return false;

	return IsPrefix(lhs, rhs);
}

bool
operator>=(const ModeDistribution& lhs, const ModeDistribution& rhs){
	return rhs <= lhs;
}

bool
operator>(const ModeDistribution& lhs, const ModeDistribution& rhs){
	return rhs < lhs;
}

std::ostream& operator<<( std::ostream& os, const ModeDistribution& dist )
{
    os << ModeDistToString(dist);
    return os;
}

ModeDistribution
ModeDistribution::overlapWith(const ModeDistribution& rhs) const{
	Unsigned i;
	ModeDistribution ret;
	for(i = 0; i < entries_.size(); i++)
		if(std::find(rhs.entries_.begin(), rhs.entries_.end(), entries_[i]) != rhs.entries_.end())
			ret.entries_.push_back(entries_[i]);
	return ret;
}

bool
ModeDistribution::SameModesAs(const ModeDistribution& rhs) const{
	Unsigned i;
	if(entries_.size() != rhs.entries_.size())
		return false;

	std::vector<Mode> sorted = entries_;
	SortVector(sorted);

	std::vector<Mode> sortedRhs = rhs.entries_;
	SortVector(sortedRhs);

	for(i = 0; i < sorted.size(); i++)
		if(sorted[i] != sortedRhs[i])
			return false;
	return true;
}

bool
ModeDistribution::Contains(const Mode& mode) const{
	return std::find(entries_.begin(), entries_.end(), mode) != entries_.end();
}

const Unsigned&
ModeDistribution::operator[](size_t index) const{
	return entries_[index];
}

ModeDistribution
ModeDistribution::Filter(const std::vector<Unsigned>& filterIndices){
	std::vector<Unsigned> newVals;
	for(Unsigned i = 0; i < filterIndices.size(); i++){
		Unsigned filterIndex = filterIndices[i];
		if(filterIndex < entries_.size())
			newVals.push_back(entries_[filterIndex]);
	}

	ModeDistribution ret(newVals);
	return ret;
}

ModeDistribution
ModeDistribution::GetCommonSuffix(const ModeDistribution& other) const {
	return GetSuffix(entries_, other.entries_);
}

ModeDistribution
ModeDistribution::GetCommonPrefix(const ModeDistribution& other) const {
	return GetPrefix(entries_, other.entries_);
}

void
ModeDistribution::CheckIsValid()
{
	std::vector<Unsigned> unique = Unique(entries_);
	if(unique.size() != entries_.size())
		LogicError("Invalid Mode Distribution");
}

bool operator==( const ModeDistribution& A, const ModeDistribution& B ){
	return A.entries_ == B.entries_;
}

bool operator!=( const ModeDistribution& A, const ModeDistribution& B ){
	return !(A.entries_ == B.entries_);
}

inline std::string
ModeDistToString_( const ModeDistribution& distribution, bool endLine )
{
    std::stringstream ss;
    ss << "(";
    if(distribution.size() >= 1){
    	ss << distribution[0];
		for(size_t i = 1; i < distribution.size(); i++)
		  ss << ", " << distribution[i];
    }
    ss <<  ")";
    if(endLine)
        ss << std::endl;
    return ss.str();
}

inline std::string
ModeDistToString( const ModeDistribution& distribution, bool endLine )
{
    return ModeDistToString_(distribution, endLine);
}


ModeDistribution
StringToModeDist( const std::string& s)
{
	std::vector<Unsigned> distVals;
	size_t pos, lastPos;
	pos = s.find_first_of("(");
	lastPos = s.find_first_of(")");
	if(pos != 0 || lastPos != s.size() - 1)
		LogicError("Malformed mode distribution string");
	pos = s.find_first_not_of("(,)", pos);
	while(pos != std::string::npos){
		lastPos = s.find_first_of(",)", pos);
		distVals.push_back(atoi(s.substr(pos, lastPos - pos).c_str()));
		pos = s.find_first_not_of("(,)", lastPos+1);
	}

	ModeDistribution ret(distVals);
	return ret;
}

}
