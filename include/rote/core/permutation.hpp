/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef ROTE_CORE_PERMUTATION_HPP
#define ROTE_CORE_PERMUTATION_HPP

#include "error.hpp"
#include "util.hpp"

namespace rote {

class Permutation {
public:
	Permutation();
	Permutation(Unsigned order);
	Permutation(std::initializer_list<Unsigned> list);
	Permutation(const std::vector<Unsigned>& perm); //Needed for methods that dynamically determine permutations
	Permutation(const Permutation& perm);

	template<typename T>
	Permutation(const std::vector<T>& vFrom, const std::vector<T>& vTo): perm_(vFrom.size()) {
		if(vFrom.size() != vTo.size())
				LogicError("reference vector and permuted vector are of different sizes");

		for(Unsigned i = 0; i < vTo.size(); i++){
				perm_[i] = IndexOf(vFrom, vTo[i]);
		}
	};

	~Permutation();

	Permutation& operator=(const Permutation& perm);
	const Unsigned& operator[](std::size_t idx) const;
	Unsigned size() const;

  friend std::ostream& operator<<(std::ostream& o, const Permutation& perm);
  friend bool operator!=(const Permutation& lhs, const Permutation& rhs);

  Permutation InversePermutation() const;
  Permutation PermutationTo(const Permutation& perm) const;
  std::vector<Unsigned> Entries() const;

	template<typename T>
	std::vector<T> applyTo(const std::vector<T>& v) const {
		return FilterVector(v, Entries());
	};

private:
	std::vector<Unsigned> perm_;

	void CheckIsValid() const;
};

} // namespace rote

#endif // ifndef ROTE_CORE_PERMUTAITON_HPP
