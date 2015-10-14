/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef TMEN_BTAS_CONTRACT_HPP
#define TMEN_BTAS_CONTRACT_HPP

#include "../level1/Permute.hpp"
#include "tensormental/util/vec_util.hpp"
#include "tensormental/util/btas_util.hpp"
#include "tensormental/core/view_decl.hpp"
#include "tensormental/io/Print.hpp"

namespace tmen{

////////////////////////////////////
// DistContract Workhorse
////////////////////////////////////

template <typename T>
void ContractStatA(T alpha, const Tensor<T>& A, const IndexArray& indicesA, const Tensor<T>& B, const IndexArray& indicesB, T beta, Tensor<T>& C, const IndexArray& indicesC){
	Unsigned i, j, k;
	IndexArray contractIndices = DetermineContractIndices(indicesA, indicesB);
	IndexArray indicesT = ConcatenateVectors(indicesC, contractIndices);
	ModeArray reduceModes;
	for(i = 0; i < contractIndices.size(); i++){
		reduceModes.push_back(C.Order() + i);
	}

	TensorDistribution distA = A.TensorDist();
	TensorDistribution distB = B.TensorDist();
	TensorDistribution distC = C.TensorDist();

	TensorDistribution distIntC;
	TensorDistribution distT(indicesC.size() + contractIndices.size());
	TensorDistribution distIntB(indicesB.size());

	//Setup temp distB
	ModeArray blank;
	for(i = 0; i < indicesB.size(); i++){
		Index indexToFind = indicesB[i];
		IndexArray::iterator it = std::find(indicesA.begin(), indicesA.end(), indexToFind);
		if( it != indicesA.end()){
			distIntB[i] = distA[it - indicesA.begin()];
		}else{
			distIntB[i] = blank;
		}
	}

	//Setup temp distIntC
	distIntC = distC;
	for(i = 0; i < contractIndices.size(); i++){
		Index indexToFind = contractIndices[i];
		IndexArray::iterator it = std::find(indicesA.begin(), indicesA.end(), indexToFind);
		Unsigned pos = it - indicesA.begin();
		ModeDistribution modeDist = distA[pos];
		//For every grid mode we reduce over, place it as suffix of temp dist
		for(j = 0; j < modeDist.size(); j++){
			Mode modeToElim = modeDist[j];
			//Find where this mode being reduced over occurs in distC
			for(k = 0; k < distC.size(); k++){
				ModeDistribution distCModeDist = distC[k];
				Unsigned modePos = std::find(distCModeDist.begin(), distCModeDist.end(), modeToElim);
				//Place grid mode at suffix
				if(modePos != distCModeDist.end()){
					distIntC[k].erase(distIntC[k].begin() + modePos);
					distIntC[k].push_back(modeToElim);
				}
			}
		}
	}

	//Setup temp distT
	for(i = 0; i < indicesT; i++){
		Index indexToFind = indicesT[i];
		IndexArray::iterator it = std::find(indicesA.begin(), indicesA.end(), indexToFind);
		if( it != indicesA.end()){
			distT[i] = distA[it - indicesA.begin()];
		}else{
			distT[i] = blank;
		}
	}

	//Perform the computation
	DistTensor<T> intB(B.Order(), distIntB, B.Grid());
	DistTensor<T> intT(A.Order(), distT, C.Grid());
	DistTensor<T> intC(C.Order(), distIntC, C.Grid());

	intB.RedistFrom(B);
	LocalContract(alpha, A, indicesA, intB, indicesB, beta, intT, indicesC);

	intC.ReduceScatterUpdateRedistFrom(alpha, intT, beta, reduceModes);
	C.RedistFrom(intC);
}

template <typename T>
void ContractStatC(T alpha, const Tensor<T>& A, const IndexArray& indicesA, const Tensor<T>& B, const IndexArray& indicesB, T beta, Tensor<T>& C, const IndexArray& indicesC){
	Unsigned i;

	TensorDistribution distA = A.TensorDist();
	TensorDistribution distB = B.TensorDist();
	TensorDistribution distC = C.TensorDist();

	TensorDistribution distIntA(distA.size());
	TensorDistribution distIntB(distB.size());

	ModeArray blank;

	//Setup temp dist A
	for(i = 0; i < indicesA.size(); i++){
		Index indexToFind = indicesA[i];
		IndexArray::iterator it = std::find(indicesC.begin(), indicesC.end(), indexToFind);
		if( it != indicesC.end()){
			distIntA[i] = distC[it - indicesC.begin()];
		}else{
			distIntA[i] = blank;
		}
	}

	//Setup temp dist B
	for(i = 0; i < indicesB.size(); i++){
		Index indexToFind = indicesB[i];
		IndexArray::iterator it = std::find(indicesC.begin(), indicesC.end(), indexToFind);
		if( it != indicesC.end()){
			distIntB[i] = distC[it - indicesC.begin()];
		}else{
			distIntB[i] = blank;
		}
	}

	//Perform the computation
	DistTensor<T> intA(A.Order(), distIntA, A.Grid());
	DistTensor<T> intB(B.Order(), distIntB, B.Grid());

	intA.RedistFrom(A);
	intB.RedistFrom(B);
	LocalContractAndEliminate(alpha, intA, indicesA, intB, indicesB, beta, C, indicesC);
}

template <typename T>
void Contract(T alpha, const Tensor<T>& A, const IndexArray& indicesA, const Tensor<T>& B, const IndexArray& indicesB, T beta, Tensor<T>& C, const IndexArray& indicesC){
    //Determine Stationary variant.
    const Unsigned numElemA = prod(A.Shape());
    const Unsigned numElemB = prod(B.Shape());
    const Unsigned numElemC = prod(C.Shape());

    if(numElemA > numElemB && numElemA > numElemC){
    	//Stationary A variant
    	ContractStatA(alpha, A, indicesA, B, indicesB, beta, C, indicesC);
    }else if(numElemB > numElemA && numElemB > numElemC){
    	//Stationary B variant
    	//ContractStatB(alpha, A, indicesA, B, indicesB, beta, C, indicesC);
    }else{
    	//Stationary C variant
    	ContractStatC(alpha, A, indicesA, B, indicesB, beta, C, indicesC);
    }
}

} // namespace tmen

#endif // ifndef TMEN_BTAS_CONTRACT_HPP
