/*
   Copyright (c) 2009-2013, Jack Poulson
                      2013, Jed Brown
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef ROTE_CORE_TENSOR_DISTRIBUTION_HPP
#define ROTE_CORE_TENSOR_DISTRIBUTION_HPP

namespace rote {

class TensorDistribution
{
public:
	TensorDistribution( );
	TensorDistribution( Unsigned order );
//    TensorDistribution( std::initializer_list<std::initializer_list<Unsigned> > list );
    TensorDistribution( const TensorDistribution& dist);
    TensorDistribution( const std::vector<ModeDistribution>& dist);
    TensorDistribution( const std::string& dist);

    ~TensorDistribution();

    // Simple interface (simpler version of distributed-based interface)
    Unsigned size() const;
    std::vector<ModeDistribution> Entries() const;
    ModeDistribution UnusedModes() const;
    ModeDistribution UsedModes() const;

    TensorDistribution& operator=(const TensorDistribution& rhs);
    TensorDistribution& operator+=(const TensorDistribution& rhs);
    TensorDistribution& operator-=(const TensorDistribution& rhs);

    const ModeDistribution& operator[](size_t index) const;
    void RemoveUnitModeDists(const std::vector<Unsigned>& unitModes);
    void IntroduceUnitModeDists(const std::vector<Unsigned>& unitModes);

    void SetToMatch(const TensorDistribution& other, const IndexArray& otherIndices, const IndexArray& myIndices);
    void AppendToMatchForGridModes(const ModeArray& gridModes, const TensorDistribution& other, const IndexArray& otherIndices, const IndexArray& myIndices);
    TensorDistribution Filter(const std::vector<Unsigned>& filterIndices) const;

    Mode TensorModeForGridMode(const Mode& mode) const;
		ModeArray TensorModesForGridModes(const ModeArray& modes) const;
		TensorDistribution TensorDistForGridModes(const ModeDistribution& modes) const;
		TensorDistribution GetCommonSuffix(const TensorDistribution& other) const;
		TensorDistribution GetCommonPrefix(const TensorDistribution& other) const;

    friend bool operator<(const TensorDistribution& lhs, const TensorDistribution& rhs);
    friend bool operator>(const TensorDistribution& lhs, const TensorDistribution& rhs);
    friend bool operator<=(const TensorDistribution& lhs, const TensorDistribution& rhs);
    friend bool operator>=(const TensorDistribution& lhs, const TensorDistribution& rhs);
    friend TensorDistribution operator+(const TensorDistribution& lhs, const TensorDistribution& rhs);
    friend TensorDistribution operator-(const TensorDistribution& lhs, const TensorDistribution& rhs);
    friend bool operator== ( const TensorDistribution& A, const TensorDistribution& B );
    friend bool operator!= ( const TensorDistribution& A, const TensorDistribution& B );

private:
    std::vector<ModeDistribution> entries_;
    void CheckIsValid();
};

std::string TensorDistToString( const TensorDistribution&  distribution, bool endLine=false );
TensorDistribution StringToTensorDist( const std::string& s );
std::ostream& operator<<( std::ostream& os, const TensorDistribution& dist );
} // namespace rote

#endif // ifndef ROTE_CORE_TENSOR_DISTRIBUTION_HPP
