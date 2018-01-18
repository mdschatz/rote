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

// Main interface
template <typename T>
void Contract<T>::run(
  T alpha,
  const DistTensor<T>& A, const std::string& indicesA,
  const DistTensor<T>& B, const std::string& indicesB,
  T beta,
        DistTensor<T>& C, const std::string& indicesC,
  const std::vector<Unsigned>& blkSizes
) {
  // Convert to index array
  IndexArray indA(indicesA.size());
  for(Unsigned i = 0; i < indicesA.size(); i++)
    indA[i] = indicesA[i];

  IndexArray indB(indicesB.size());
  for(Unsigned i = 0; i < indicesB.size(); i++)
    indB[i] = indicesB[i];

  IndexArray indC(indicesC.size());
  for(Unsigned i = 0; i < indicesC.size(); i++)
    indC[i] = indicesC[i];

  //Determine Stationary variant.
  const Unsigned numElemA = prod(A.Shape());
  const Unsigned numElemB = prod(B.Shape());
  const Unsigned numElemC = prod(C.Shape());

  bool isBiggerAB = numElemA > numElemB;
  bool isBiggerAC = numElemA > numElemC;
  bool isBiggerBC = numElemB > numElemC;

  bool isEqualAB = numElemA == numElemB;

  bool isSmallerAB = numElemA < numElemB;
  bool isSmallerAC = numElemA < numElemC;

  bool isSmallerEqualAC = numElemA <= numElemC;
  bool isSmallerEqualBC = numElemB <= numElemC;

  bool isBiggerEqualAB = numElemA >= numElemB;
  bool isBiggerEqualAC = numElemA >= numElemC;

  if(isBiggerEqualAB && isBiggerAC){
    Contract::run(
      alpha,
      A, indA,
      B, indB,
      beta,
      C, indC,
      blkSizes,
      false
    );
  }else if((isSmallerAB && isBiggerEqualAC) ||
       (isSmallerAB && isSmallerAC && isBiggerBC)){
    Contract::run(
      alpha,
      B, indB,
      A, indA,
      beta,
      C, indC,
      blkSizes,
      false
    );
  }else if((isBiggerAB && isSmallerEqualAC) ||
       (isEqualAB && isSmallerEqualAC) ||
     (isSmallerAB && isSmallerAC && isSmallerEqualBC)){
    Contract::run(
      alpha,
      B, indB,
      A, indA,
      beta,
      C, indC,
      blkSizes,
      false
    );
  }else{
    LogicError("Should never occur");
  }
}

// Internal interface
template <typename T>
void Contract<T>::run(
  T alpha,
	const DistTensor<T>& A, const IndexArray& indicesA,
	const DistTensor<T>& B, const IndexArray& indicesB,
  T beta,
	      DistTensor<T>& C, const IndexArray& indicesC,
	const std::vector<Unsigned>& blkSizes, bool isStatC
) {
  //Determine how to partition
	BlkContractStatCInfo contractInfo;
	Contract<T>::setContractInfo(
		A, indicesA,
		B, indicesB,
		C, indicesC,
		blkSizes, isStatC,
		contractInfo
	);

	if (isStatC) {
		if(contractInfo.permC != C.LocalPermutation()){
			DistTensor<T> tmpC(C.TensorDist(), C.Grid());
			tmpC.SetLocalPermutation(contractInfo.permC);
			Permute(C, tmpC);
			Scal(beta, tmpC);
			Contract<T>::runHelperPartitionAB(0, contractInfo, alpha, A, indicesA, B, indicesB, beta, tmpC, indicesC);
			Permute(tmpC, C);
		}else{
			Scal(beta, C);
			Contract<T>::runHelperPartitionAB(0, contractInfo, alpha, A, indicesA, B, indicesB, beta, C, indicesC);
		}
	} else {
		DistTensor<T> tmpA(A.TensorDist(), A.Grid());
		tmpA.SetLocalPermutation(contractInfo.permA);
		Permute(A, tmpA);

		Contract<T>::runHelperPartitionBC(0, contractInfo, alpha, tmpA, indicesA, B, indicesB, beta, C, indicesC);
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
