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

template <typename T>
void RecurContractStatC(Unsigned depth, BlkContractStatCInfo& contractInfo, T alpha, const DistTensor<T>& A, const IndexArray& indicesA, const DistTensor<T>& B, const IndexArray& indicesB, T beta, DistTensor<T>& C, const IndexArray& indicesC){
	if(depth == contractInfo.partModesA.size()){
		DistTensor<T> intA(contractInfo.distIntA, A.Grid());
		intA.SetLocalPermutation(contractInfo.permA);
		intA.AlignModesWith(contractInfo.alignModesA, C, contractInfo.alignModesATo);
		intA.RedistFrom(A);

		DistTensor<T> intB(contractInfo.distIntB, B.Grid());
		intB.AlignModesWith(contractInfo.alignModesB, C, contractInfo.alignModesBTo);
		intB.SetLocalPermutation(contractInfo.permB);
		intB.RedistFrom(B);

		LocalContractAndLocalEliminate(alpha, intA.LockedTensor(), indicesA, false, intB.LockedTensor(), indicesB, false, T(1), C.Tensor(), indicesC, false);
		return;
	}
	//Must partition and recur
	//Note: pull logic out
	Unsigned blkSize = contractInfo.blkSizes[depth];
	Mode partModeA = contractInfo.partModesA[depth];
	Mode partModeB = contractInfo.partModesB[depth];
	DistTensor<T> A_T(A.TensorDist(), A.Grid());
	DistTensor<T> A_B(A.TensorDist(), A.Grid());
	DistTensor<T> A_0(A.TensorDist(), A.Grid());
	DistTensor<T> A_1(A.TensorDist(), A.Grid());
	DistTensor<T> A_2(A.TensorDist(), A.Grid());

	DistTensor<T> B_T(B.TensorDist(), B.Grid());
	DistTensor<T> B_B(B.TensorDist(), B.Grid());
	DistTensor<T> B_0(B.TensorDist(), B.Grid());
	DistTensor<T> B_1(B.TensorDist(), B.Grid());
	DistTensor<T> B_2(B.TensorDist(), B.Grid());

	//Do the partitioning and looping
	int count = 0;
	LockedPartitionDown(A, A_T, A_B, partModeA, 0);
	LockedPartitionDown(B, B_T, B_B, partModeB, 0);
	while(A_T.Dimension(partModeA) < A.Dimension(partModeA)){
		LockedRepartitionDown(A_T, A_0,
				        /**/ /**/
				             A_1,
				        A_B, A_2, partModeA, blkSize);
		LockedRepartitionDown(B_T, B_0,
						/**/ /**/
							 B_1,
						B_B, B_2, partModeB, blkSize);

		/*----------------------------------------------------------------*/
		RecurContractStatC(depth+1, contractInfo, alpha, A_1, indicesA, B_1, indicesB, beta, C, indicesC);
		count++;
		/*----------------------------------------------------------------*/
		SlideLockedPartitionDown(A_T, A_0,
				                A_1,
						   /**/ /**/
						   A_B, A_2, partModeA);
		SlideLockedPartitionDown(B_T, B_0,
				                B_1,
						   /**/ /**/
						   B_B, B_2, partModeB);
	}
}

template<typename T>
void SetBlkContractStatCInfo(
	const DistTensor<T>& A, const IndexArray& indicesA,
  const DistTensor<T>& B, const IndexArray& indicesB,
  const DistTensor<T>& C, const IndexArray& indicesC,
  const std::vector<Unsigned>& blkSizes, BlkContractStatCInfo& contractInfo
) {
	Unsigned i;
	IndexArray indicesAC = DiffVector(indicesC, indicesB);
	IndexArray indicesBC = DiffVector(indicesC, indicesA);
	IndexArray indicesAB = DiffVector(indicesA, indicesC);

	TensorDistribution distA = A.TensorDist();
	TensorDistribution distB = B.TensorDist();
	TensorDistribution distC = C.TensorDist();

	TensorDistribution distIntA(distA.size() - 1);
	TensorDistribution distIntB(distB.size() - 1);

	//Setup temp dist A
	distIntA.SetToMatch(distC, indicesC, indicesA);

	//Setup temp dist B
	distIntB.SetToMatch(distC, indicesC, indicesB);

	//Set the intermediate dists
	contractInfo.distIntA = distIntA;
	contractInfo.distIntB = distIntB;

	//Determine the modes to partition
	contractInfo.partModesA.resize(indicesAB.size());
	contractInfo.partModesB.resize(indicesAB.size());
	for(i = 0; i < indicesAB.size(); i++){
		contractInfo.partModesA[i] = IndexOf(indicesA, indicesAB[i]);
		contractInfo.partModesB[i] = IndexOf(indicesB, indicesAB[i]);
	}

	//Determine the final alignments needed
	contractInfo.alignModesA.resize(indicesAC.size());
	contractInfo.alignModesATo.resize(indicesAC.size());
	for(i = 0; i < indicesAC.size(); i++){
		contractInfo.alignModesA[i] = IndexOf(indicesA, indicesAC[i]);
		contractInfo.alignModesATo[i] = IndexOf(indicesC, indicesAC[i]);
	}
	contractInfo.alignModesB.resize(indicesBC.size());
	contractInfo.alignModesBTo.resize(indicesBC.size());
	for(i = 0; i < indicesBC.size(); i++){
		contractInfo.alignModesB[i] = IndexOf(indicesB, indicesBC[i]);
		contractInfo.alignModesBTo[i] = IndexOf(indicesC, indicesBC[i]);
	}

	//Set the Block-size info
	//NOTE: There are better ways to do this
	if(blkSizes.size() == 0){
		contractInfo.blkSizes.resize(indicesAB.size());
		for(i = 0; i < indicesAB.size(); i++)
			contractInfo.blkSizes[i] = 32;
	}else{
		contractInfo.blkSizes = blkSizes;
	}

	//Set the local permutation info
	std::cout << "permA\n";
	PrintVector(indicesA, "indicesA");
	PrintVector(indicesAC, "indicesAC");
	PrintVector(indicesAB, "indicesAB");
	Permutation permA(indicesA, ConcatenateVectors(indicesAC, indicesAB));
	Permutation permB(indicesB, ConcatenateVectors(indicesAB, indicesBC));
	Permutation permC(indicesC, ConcatenateVectors(indicesAC, indicesBC));

	contractInfo.permA = permA;
	contractInfo.permB = permB;
	contractInfo.permC = permC;
}

template <typename T>
void ContractStatC(T alpha, const DistTensor<T>& A, const IndexArray& indicesA, const DistTensor<T>& B, const IndexArray& indicesB, T beta, DistTensor<T>& C, const IndexArray& indicesC, const std::vector<Unsigned>& blkSizes){
	//Determine how to partition
	BlkContractStatCInfo contractInfo;
	SetBlkContractStatCInfo(
		A, indicesA,
		B, indicesB,
		C, indicesC,
		blkSizes, contractInfo
	);

	if(contractInfo.permC != C.LocalPermutation()){
		DistTensor<T> tmpC(C.TensorDist(), C.Grid());
		tmpC.SetLocalPermutation(contractInfo.permC);
		Permute(C, tmpC);
		Scal(beta, tmpC);
		RecurContractStatC(0, contractInfo, alpha, A, indicesA, B, indicesB, beta, tmpC, indicesC);
		Permute(tmpC, C);
	}else{
		Scal(beta, C);
		RecurContractStatC(0, contractInfo, alpha, A, indicesA, B, indicesB, beta, C, indicesC);
	}

}

//Non-template functions
//bool AnyFalseElem(const std::vector<bool>& vec);
#define PROTO(T) \
	template void ContractStatC(T alpha, const DistTensor<T>& A, const IndexArray& indicesA, const DistTensor<T>& B, const IndexArray& indicesB, T beta, DistTensor<T>& C, const IndexArray& indicesC, const std::vector<Unsigned>& blkSizes); \
	template void SetBlkContractStatCInfo( \
		const DistTensor<T>& A, const IndexArray& indicesA, \
		const DistTensor<T>& B, const IndexArray& indicesB, \
		const DistTensor<T>& C, const IndexArray& indicesC, \
		const std::vector<Unsigned>& blkSizes, BlkContractStatCInfo& contractInfo);

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
