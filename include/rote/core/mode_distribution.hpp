/*
   Copyright (c) 2009-2013, Jack Poulson
                      2013, Jed Brown
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef ROTE_CORE_MODE_DISTRIBUTION_HPP
#define ROTE_CORE_MODE_DISTRIBUTION_HPP

#include "rote.hpp"

namespace rote {

class ModeDistribution
{
public:
    ModeDistribution( std::initializer_list<Unsigned> list );
    ModeDistribution( const ModeDistribution& dist);
    ModeDistribution( const std::vector<Unsigned>& dist);
    ModeDistribution( const std::string& dist);
    ModeDistribution();

    ~ModeDistribution();

    // Simple interface (simpler version of distributed-based interface)
    Unsigned size() const;
    std::vector<Unsigned> Entries() const;

    ModeDistribution& operator=(const ModeDistribution& rhs);
    ModeDistribution& operator+=(const ModeDistribution& rhs);
    ModeDistribution& operator+=(const Mode& rhs);
    ModeDistribution& operator-=(const ModeDistribution& rhs);
    ModeDistribution& operator-=(const Mode& rhs);
    ModeDistribution  overlapWith(const ModeDistribution& rhs) const;
    bool SameModesAs(const ModeDistribution& rhs) const;

    const Unsigned& operator[](size_t index) const;
    ModeDistribution Filter(const std::vector<Unsigned>& filterIndices);

    bool Contains(const Mode& mode) const;
    friend ModeDistribution GetCommonPrefix(const ModeDistribution& lhs, const ModeDistribution& rhs);
    friend ModeDistribution GetCommonSuffix(const ModeDistribution& lhs, const ModeDistribution& rhs);

    friend bool IsPrefix(const ModeDistribution& lhs, const ModeDistribution& rhs);
    friend bool operator<(const ModeDistribution& lhs, const ModeDistribution& rhs);
    friend bool operator>(const ModeDistribution& lhs, const ModeDistribution& rhs);
    friend bool operator<=(const ModeDistribution& lhs, const ModeDistribution& rhs);
    friend bool operator>=(const ModeDistribution& lhs, const ModeDistribution& rhs);
    friend ModeDistribution operator+(const ModeDistribution& lhs, const ModeDistribution& rhs);
    friend ModeDistribution operator-(const ModeDistribution& lhs, const ModeDistribution& rhs);
    friend bool operator== ( const ModeDistribution& A, const ModeDistribution& B );
    friend bool operator!= ( const ModeDistribution& A, const ModeDistribution& B );
private:
    std::vector<Unsigned> entries_;
    void CheckIsValid();
};

std::string ModeDistToString_( const ModeDistribution&  distribution, bool endLine=false );
std::string ModeDistToString( const ModeDistribution&  distribution, bool endLine=false );
ModeDistribution StringToModeDist( const std::string& s );
std::ostream& operator<<( std::ostream& os, const ModeDistribution& dist );
} // namespace rote

#endif // ifndef ROTE_CORE_MODE_DISTRIBUTION_HPP
