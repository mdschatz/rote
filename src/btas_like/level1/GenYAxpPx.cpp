/*
   Copyright (c) 2009-2013, Jack Poulson
                      2013, Jeff Hammond
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#include "rote.hpp"

namespace rote{

template<typename T>
bool CheckYAxpPxArgs(const DistTensor<T>& X, const Permutation& perm, const DistTensor<T>& Y){
	const TensorDistribution yDist = Y.TensorDist();
	const TensorDistribution xDist = X.TensorDist();

	bool ret = true;
	ret &= CheckOrder(X.Order(), Y.Order());
	ret &= (yDist == xDist);
	ret &= CheckIsValidPermutation(X.Order(), perm);
	return ret;
}

//TODO: Handle updates
//Note: StatA equivalent to StatB with rearranging operands
template <typename T>
void GenYAxpPx( T alpha, const DistTensor<T>& X, T beta, const Permutation& perm, DistTensor<T>& Y ){
	if(!CheckYAxpPxArgs(X, perm, Y))
		LogicError("AllToAllDoubleModeRedist: Invalid redistribution request");
	TensorDistribution copy = X.TensorDist();
	std::vector<ModeDistribution> newDistEntries(copy.size());

	for(int i = 0; i < X.Order(); i++)
		newDistEntries[i] = copy[perm[i]];
	newDistEntries[newDistEntries.size() - 1] = copy[copy.size() - 1];

	TensorDistribution newDist(newDistEntries);
	DistTensor<T> tmp(newDist, X.Grid());
	tmp.RedistFrom(X);

	Y.ResizeTo(X);
	YAxpPx(alpha, X, beta, tmp, perm, Y);
}

//Non-template functions
//bool AnyFalseElem(const std::vector<bool>& vec);
#define PROTO(T) \
	template bool CheckYAxpPxArgs(const DistTensor<T>& X, const Permutation& perm, const DistTensor<T>& Y); \
    template void GenYAxpPx( T alpha, const DistTensor<T>& X, T beta, const Permutation& perm, DistTensor<T>& Y );

//PROTO(Unsigned)
//PROTO(Int)
PROTO(float)
PROTO(double)
//PROTO(char)

} // namespace rote
