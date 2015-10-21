/*
   Copyright (c) 2009-2013, Jack Poulson
                      2013, Jeff Hammond
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#include "tensormental.hpp"

namespace tmen{

template <typename T>
void ContractStatA(T alpha, const DistTensor<T>& A, const IndexArray& indicesA, const DistTensor<T>& B, const IndexArray& indicesB, T beta, DistTensor<T>& C, const IndexArray& indicesC){
	int commRank = mpi::CommRank(MPI_COMM_WORLD);
	Unsigned i, j, k, l;
	IndexArray contractIndices = DetermineContractIndices(indicesA, indicesB);
	IndexArray indicesT = ConcatenateVectors(indicesC, contractIndices);
	PrintVector(indicesT, "indicesT");
	ModeArray reduceModes;
	for(i = 0; i < contractIndices.size(); i++){
		reduceModes.push_back(C.Order() + i);
	}

	TensorDistribution distA = A.TensorDist();
	TensorDistribution distB = B.TensorDist();
	TensorDistribution distC = C.TensorDist();

	TensorDistribution distT(indicesT.size() + 1);
	TensorDistribution distIntB(indicesB.size() + 1);

	//Setup temp distB
	ModeArray blank;
	for(i = 0; i < indicesB.size(); i++){
		Index indexToFind = indicesB[i];
		for(j = 0; j < indicesA.size(); j++){
			if(indicesA[j] == indexToFind){
				distIntB[i] = distA[j];
				break;
			}
		}
		if(j == indicesA.size()){
			distIntB[i] = blank;
		}
	}
	distIntB[B.Order()] = distA[A.Order()];

	//Setup temp distT
	for(i = 0; i < indicesT.size(); i++){
		Index indexToFind = indicesT[i];
		for(j = 0; j < indicesA.size(); j++){
			if(indicesA[j] == indexToFind){
				distT[i] = distA[j];
				break;
			}
		}
		if(j == indicesA.size()){
			distT[i] = blank;
		}
	}
	distT[indicesC.size() + contractIndices.size()] = distA[A.Order()];

	//Setup temp distIntC
	TensorDistribution distIntC(distT.begin(), distT.end() - contractIndices.size());
	for(i = 0; i < contractIndices.size(); i++){
		ModeDistribution reduceModeDist = distT[distT.size() - 1 - i];

		for(j = 0; j < reduceModeDist.size(); j++){
			Mode reduceMode = reduceModeDist[j];

			for(k = 0; k < distC.size(); k++){
				ModeDistribution distCModeDist = distC[k];
				for(l = 0; l < distCModeDist.size(); l++){
					if(distCModeDist[l] == reduceMode){
						distIntC[k].push_back(reduceMode);
					}
				}
			}
		}
	}

	distIntC[C.Order()] = distA[A.Order()];

	if(commRank == 0){
		std::cout << "intB: " << TensorDistToString(distIntB) << std::endl;
		std::cout << "intT: " << TensorDistToString(distT) << std::endl;
		std::cout << "intC: " << TensorDistToString(distIntC) << std::endl;
	}

	//Perform the computation
	DistTensor<T> intB(distIntB, B.Grid());
	DistTensor<T> intT(distT, C.Grid());
	DistTensor<T> intC(distIntC, C.Grid());

	intB.RedistFrom(B);
	Print(intT, "intTbefore");
	LocalContract(alpha, A.LockedTensor(), indicesA, intB.LockedTensor(), indicesB, beta, intT.Tensor(), indicesC);
	Print(intT, "intTafter");

	intC.ReduceScatterUpdateRedistFrom(alpha, intT, beta, reduceModes);
	C.RedistFrom(intC);
}

template <typename T>
void ContractStatC(T alpha, const DistTensor<T>& A, const IndexArray& indicesA, const DistTensor<T>& B, const IndexArray& indicesB, T beta, DistTensor<T>& C, const IndexArray& indicesC){
	int commRank = mpi::CommRank(MPI_COMM_WORLD);
	Unsigned i, j;

	TensorDistribution distA = A.TensorDist();
	TensorDistribution distB = B.TensorDist();
	TensorDistribution distC = C.TensorDist();

	TensorDistribution distIntA(distA.size());
	TensorDistribution distIntB(distB.size());

	ModeArray blank;

	//Setup temp dist A
	for(i = 0; i < indicesA.size(); i++){
		Index indexToFind = indicesA[i];
		for(j = 0; j < indicesC.size(); j++){
			if(indicesC[j] == indexToFind){
				distIntA[i] = distC[j];
				break;
			}
		}
		if(j == indicesC.size()){
			distIntA[i] = blank;
		}
	}
	distIntA[A.Order()] = distC[C.Order()];

	//Setup temp dist B
	for(i = 0; i < indicesB.size(); i++){
		Index indexToFind = indicesB[i];
		for(j = 0; j < indicesC.size(); j++){
			if(indicesC[j] == indexToFind){
				distIntB[i] = distC[j];
				break;
			}
		}
		if(j == indicesC.size()){
			distIntB[i] = blank;
		}
	}
	distIntB[B.Order()] = distC[C.Order()];

	if(commRank == 0){
		std::cout << "intA: " << TensorDistToString(distIntA) << std::endl;
		std::cout << "intB: " << TensorDistToString(distIntB) << std::endl;
	}
	//Perform the computation
	DistTensor<T> intA(distIntA, A.Grid());
	DistTensor<T> intB(distIntB, B.Grid());

	intA.RedistFrom(A);
	intB.RedistFrom(B);
	Print(C, "Cbefore");
	LocalContractAndLocalEliminate(alpha, intA.LockedTensor(), indicesA, intB.LockedTensor(), indicesB, beta, C.Tensor(), indicesC);
	Print(C, "After");
}

//TODO: Handle updates
template <typename T>
void GenContract(T alpha, const DistTensor<T>& A, const IndexArray& indicesA, const DistTensor<T>& B, const IndexArray& indicesB, T beta, DistTensor<T>& C, const IndexArray& indicesC){
    //Determine Stationary variant.
    const Unsigned numElemA = prod(A.Shape());
    const Unsigned numElemB = prod(B.Shape());
    const Unsigned numElemC = prod(C.Shape());

    if(numElemA > numElemB && numElemA > numElemC){
    	//Stationary A variant
    	printf("StatA\n");
    	ContractStatA(alpha, A, indicesA, B, indicesB, beta, C, indicesC);
    }else if(numElemB > numElemA && numElemB > numElemC){
    	//Stationary B variant
    	//ContractStatB(alpha, A, indicesA, B, indicesB, beta, C, indicesC);
    }else{
    	printf("StatC\n");
    	//Stationary C variant
    	ContractStatC(alpha, A, indicesA, B, indicesB, beta, C, indicesC);
    }
}

//Non-template functions
//bool AnyFalseElem(const std::vector<bool>& vec);
#define PROTO(T) \
	template void ContractStatA(T alpha, const DistTensor<T>& A, const IndexArray& indicesA, const DistTensor<T>& B, const IndexArray& indicesB, T beta, DistTensor<T>& C, const IndexArray& indicesC); \
	template void ContractStatC(T alpha, const DistTensor<T>& A, const IndexArray& indicesA, const DistTensor<T>& B, const IndexArray& indicesB, T beta, DistTensor<T>& C, const IndexArray& indicesC); \
	template void GenContract(T alpha, const DistTensor<T>& A, const IndexArray& indicesA, const DistTensor<T>& B, const IndexArray& indicesB, T beta, DistTensor<T>& C, const IndexArray& indicesC);

//PROTO(Unsigned)
//PROTO(Int)
PROTO(float)
PROTO(double)
//PROTO(char)

} // namespace tmen
