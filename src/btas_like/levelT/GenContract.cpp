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

//TODO: Handle updates
//Note: StatA equivalent to StatB with rearranging operands
template <typename T>
void GenContract(T alpha, const DistTensor<T>& A, const IndexArray& indicesA, const DistTensor<T>& B, const IndexArray& indicesB, T beta, DistTensor<T>& C, const IndexArray& indicesC){
	//Determine Stationary variant.
    const Unsigned numElemA = prod(A.Shape());
    const Unsigned numElemB = prod(B.Shape());
    const Unsigned numElemC = prod(C.Shape());

    if(numElemA > numElemB && numElemA > numElemC){
    	//Stationary A variant
    	ContractStatA(alpha, A, indicesA, B, indicesB, beta, C, indicesC);
    }else if(numElemB > numElemA && numElemB > numElemC){
    	//Stationary B variant
    	ContractStatA(alpha, B, indicesB, A, indicesA, beta, C, indicesC);
    }else{
    	//Stationary C variant
    	ContractStatC(alpha, A, indicesA, B, indicesB, beta, C, indicesC);
    }
}

template <typename T>
void GenContract(T alpha, const DistTensor<T>& A, const std::string& indicesA, const DistTensor<T>& B, const std::string& indicesB, T beta, DistTensor<T>& C, const std::string& indicesC){
	IndexArray indA(indicesA.size());
	for(int i = 0; i < indicesA.size(); i++)
		indA[i] = indicesA[i];

	IndexArray indB(indicesB.size());
	for(int i = 0; i < indicesB.size(); i++)
		indB[i] = indicesB[i];

	IndexArray indC(indicesC.size());
	for(int i = 0; i < indicesC.size(); i++)
		indC[i] = indicesC[i];

	GenContract(alpha, A, indA, B, indB, beta, C, indC);
}

//Non-template functions
//bool AnyFalseElem(const std::vector<bool>& vec);
#define PROTO(T) \
	template void GenContract(T alpha, const DistTensor<T>& A, const IndexArray& indicesA, const DistTensor<T>& B, const IndexArray& indicesB, T beta, DistTensor<T>& C, const IndexArray& indicesC); \
	template void GenContract(T alpha, const DistTensor<T>& A, const std::string& indicesA, const DistTensor<T>& B, const std::string& indicesB, T beta, DistTensor<T>& C, const std::string& indicesC);

//PROTO(Unsigned)
//PROTO(Int)
PROTO(float)
PROTO(double)
//PROTO(char)

} // namespace rote
