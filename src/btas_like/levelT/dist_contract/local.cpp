/*
   Copyright (c) 2009-2013, Jack Poulson
                      2013, Jeff Hammond
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#include "rote.hpp"

namespace rote {

// Local interface
template <typename T>
void Contract<T>::run(
  T alpha,
  const Tensor<T>& A, const IndexArray& indicesA,
  const Tensor<T>& B, const IndexArray& indicesB,
  T beta,
        Tensor<T>& C, const IndexArray& indicesC,
  bool doEliminate, bool doPermute
) {
  if (doEliminate) {
    Unsigned i;
    Unsigned order = C.Order();
    IndexArray contractIndices = DetermineContractIndices(indicesA, indicesB);

    ModeArray uModes(contractIndices.size());
    for(i = 0; i < uModes.size(); i++)
        uModes[i] = i + order;

    IndexArray CIndices = ConcatenateVectors(indicesC, contractIndices);

    //LocalContract leaves unit modes in result, so introduce them here
    C.IntroduceUnitModes(uModes);

    LocalContract(
      alpha,
      A, indicesA, doPermute,
      B, indicesB, doPermute,
      beta,
      C, CIndices, doPermute
    );

    //Remove the unit modes
    C.RemoveUnitModes(uModes);
  } else {
    LocalContract(
      alpha,
      A, indicesA, doPermute,
      B, indicesB, doPermute,
      beta,
      C, indicesC, doPermute
    );
  }
}

#define PROTO(T) \
	template class Contract<T>;

//PROTO(Unsigned)
//PROTO(Int)
PROTO(float)
PROTO(double)
//PROTO(char)

#ifndef DISABLE_COMPLEX
#ifndef DISABLE_FLOAT
PROTO(std::complex<float>)
#endif
PROTO(std::complex<double>)
#endif

} // namespace rote
