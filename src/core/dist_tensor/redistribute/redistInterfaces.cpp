/*
   Copyright (c) 2009-2013, Jack Poulson
                      2013, Jeff Hammond
                      2013, Jed Brown
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#include "rote.hpp"

namespace rote{

template <typename T>
void DistTensor<T>::RedistFrom(const DistTensor<T>& A, const ModeArray& reduceModes, const T alpha, const T beta){
  PROFILE_SECTION("RedistFrom");

	const Grid& g = this->Grid();
	RedistPlan redistPlan(this->TensorDist(), A.TensorDist(), reduceModes, g);
	// PrintRedistPlan(redistPlan, "Plan");

	if (redistPlan.size() == 0) {
		// HACK
		ModeArray blank;
		this->PermutationRedistFrom(A, blank);
		return;
	}

  DistTensor<T> tmp(A.TensorDist(), g);
  tmp.LockedAttach(A.Shape(), A.Alignments(), A.LockedBuffer(), A.LocalPermutation(), A.LocalStrides(), this->Grid());

  for(int i = 0; i < redistPlan.size() - 1; i++){
  	Redist redist = redistPlan[i];
  	DistTensor<T> tmp2(redist.dB(), g);

  	switch(redist.type()){
    	case AG: tmp2.AllGatherRedistFrom(tmp, redist.modes()); break;
    	case A2A: tmp2.AllToAllRedistFrom(tmp, redist.modes()); break;
    	case Perm: tmp2.PermutationRedistFrom(tmp, redist.modes()); break;
    	case Local: tmp2.LocalRedistFrom(tmp); break;
    	case RS: tmp2.ReduceScatterRedistFrom(alpha, tmp, reduceModes); break;
    	default: break;
  	}
  	tmp.Empty();
  	tmp = tmp2;
  }

	Redist redist = redistPlan[-1];
	switch(redist.type()){
		case AG: AllGatherRedistFrom(tmp, redist.modes(), beta); break;
		case A2A: AllToAllRedistFrom(tmp, redist.modes(), beta); break;
		case Local: LocalRedistFrom(tmp); break;
		case Perm: PermutationRedistFrom(tmp, redist.modes(), beta); break;
		case RS: ReduceScatterUpdateRedistFrom(tmp, beta, reduceModes); break;
		default: break;
	}

  PROFILE_STOP;
}

template <typename T>
void DistTensor<T>::RedistFrom(const DistTensor<T>& A){
	ModeArray reduceModes;
	RedistFrom(A, reduceModes, T(1), T(0));
}

#define FULL(T) \
    template class DistTensor<T>;

FULL(Int)
#ifndef DISABLE_FLOAT
FULL(float)
#endif
FULL(double)

#ifndef DISABLE_COMPLEX
#ifndef DISABLE_FLOAT
FULL(std::complex<float>)
#endif
FULL(std::complex<double>)
#endif

} //namespace rote
