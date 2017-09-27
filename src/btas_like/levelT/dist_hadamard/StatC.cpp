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

// NOTE: Util functions named "Contract" are actually usable for this operation
//       They match indices, or make redistribution plans which can be shared
//       across operations.
template <typename T>
void RecurHadamardStatCPartBC(Unsigned depth, BlkHadamardStatCInfo& hadamardInfo, const DistTensor<T>& A, const IndexArray& indicesA, const DistTensor<T>& B, const IndexArray& indicesB, DistTensor<T>& C, const IndexArray& indicesC){
	if(depth == hadamardInfo.partModesBCB.size()){
		DistTensor<T> intA(hadamardInfo.distIntA, A.Grid());
		// intA.SetLocalPermutation(hadamardInfo.permA);
		intA.AlignModesWith(hadamardInfo.alignModesA, C, hadamardInfo.alignModesATo);
		intA.RedistFrom(A);

		DistTensor<T> intB(hadamardInfo.distIntB, B.Grid());
		intB.AlignModesWith(hadamardInfo.alignModesB, C, hadamardInfo.alignModesBTo);
		// intB.SetLocalPermutation(hadamardInfo.permB);
		intB.RedistFrom(B);

		PrintHadamardStatCData(hadamardInfo, "HadamardInfo", true);
		PrintData(A, "A", true);
		PrintData(intA, "intA", true);
		PrintData(B, "B", true);
		PrintData(intB, "intB", true);
		PrintData(C, "C", true);
		LocalHadamard(intA.LockedTensor(), indicesA, intB.LockedTensor(), indicesB, C.Tensor(), indicesC);
		return;
	}

	//Must partition and recur
	//Note: pull logic out
	Unsigned blkSize = hadamardInfo.blkSizes[depth];
	Mode partModeB = hadamardInfo.partModesBCB[depth];
	Mode partModeC = hadamardInfo.partModesBCC[depth];
	DistTensor<T> B_T(B.TensorDist(), B.Grid());
	DistTensor<T> B_B(B.TensorDist(), B.Grid());
	DistTensor<T> B_0(B.TensorDist(), B.Grid());
	DistTensor<T> B_1(B.TensorDist(), B.Grid());
	DistTensor<T> B_2(B.TensorDist(), B.Grid());

	DistTensor<T> C_T(C.TensorDist(), C.Grid());
	DistTensor<T> C_B(C.TensorDist(), C.Grid());
	DistTensor<T> C_0(C.TensorDist(), C.Grid());
	DistTensor<T> C_1(C.TensorDist(), C.Grid());
	DistTensor<T> C_2(C.TensorDist(), C.Grid());

	//Do the partitioning and looping
	int count = 0;
	LockedPartitionDown(B, B_T, B_B, partModeB, 0);
	PartitionDown(C, C_T, C_B, partModeC, 0);
	while(B_T.Dimension(partModeB) < B.Dimension(partModeB)){
		LockedRepartitionDown(B_T, B_0,
						/**/ /**/
							 B_1,
						B_B, B_2, partModeB, blkSize);
		RepartitionDown(C_T, C_0,
						/**/ /**/
							 C_1,
						C_B, C_2, partModeC, blkSize);

		/*----------------------------------------------------------------*/
		RecurHadamardStatCPartBC(depth+1, hadamardInfo, A, indicesA, B_1, indicesB, C_1, indicesC);
		count++;
		std::cout << "COUNT: " << count << std::endl;
		/*----------------------------------------------------------------*/
		SlideLockedPartitionDown(B_T, B_0,
				                B_1,
						   /**/ /**/
						   B_B, B_2, partModeB);
	 SlidePartitionDown(C_T, C_0,
				                C_1,
						   /**/ /**/
						   C_B, C_2, partModeC);
	}
}

template <typename T>
void RecurHadamardStatCPartAC(Unsigned depth, BlkHadamardStatCInfo& hadamardInfo, const DistTensor<T>& A, const IndexArray& indicesA, const DistTensor<T>& B, const IndexArray& indicesB, DistTensor<T>& C, const IndexArray& indicesC){
	if(depth == hadamardInfo.partModesACA.size()){
		std::cout << "recur BC\n";
		RecurHadamardStatCPartBC(0, hadamardInfo, A, indicesA, B, indicesB, C, indicesC);
		return;
	}

	//Must partition and recur
	//Note: pull logic out
	Unsigned blkSize = hadamardInfo.blkSizes[depth];
	Mode partModeA = hadamardInfo.partModesACA[depth];
	Mode partModeC = hadamardInfo.partModesACC[depth];
	DistTensor<T> A_T(A.TensorDist(), A.Grid());
	DistTensor<T> A_B(A.TensorDist(), A.Grid());
	DistTensor<T> A_0(A.TensorDist(), A.Grid());
	DistTensor<T> A_1(A.TensorDist(), A.Grid());
	DistTensor<T> A_2(A.TensorDist(), A.Grid());

	DistTensor<T> C_T(C.TensorDist(), C.Grid());
	DistTensor<T> C_B(C.TensorDist(), C.Grid());
	DistTensor<T> C_0(C.TensorDist(), C.Grid());
	DistTensor<T> C_1(C.TensorDist(), C.Grid());
	DistTensor<T> C_2(C.TensorDist(), C.Grid());

	//Do the partitioning and looping
	int count = 0;
	LockedPartitionDown(A, A_T, A_B, partModeA, 0);
	PartitionDown(C, C_T, C_B, partModeC, 0);
	while(A_T.Dimension(partModeA) < A.Dimension(partModeA)){
		LockedRepartitionDown(A_T, A_0,
				        /**/ /**/
				             A_1,
				        A_B, A_2, partModeA, blkSize);
		RepartitionDown(C_T, C_0,
						/**/ /**/
							 C_1,
						C_B, C_2, partModeC, blkSize);

		/*----------------------------------------------------------------*/
		RecurHadamardStatCPartAC(depth+1, hadamardInfo, A_1, indicesA, B, indicesB, C_1, indicesC);
		count++;
		/*----------------------------------------------------------------*/
		SlideLockedPartitionDown(A_T, A_0,
				                A_1,
						   /**/ /**/
						   A_B, A_2, partModeA);
	 SlidePartitionDown(C_T, C_0,
				                C_1,
						   /**/ /**/
						   C_B, C_2, partModeC);
	}
}

void SetBlkHadamardStatCInfo(
	const TensorDistribution& distIntA, const IndexArray& indicesA,
  const TensorDistribution& distIntB, const IndexArray& indicesB,
  const TensorDistribution& distIntC, const IndexArray& indicesC,
  const std::vector<Unsigned>& blkSizes,
  BlkHadamardStatCInfo& hadamardInfo,
  bool isStatC
) {
	Unsigned i;
	IndexArray indicesCA = isStatC
		? DiffVector(IsectVector(indicesC, indicesA), indicesB)
		: DiffVector(IsectVector(indicesA, indicesC), indicesB);
	PrintVector(IsectVector(indicesA, indicesC), "indicesAC");
	IndexArray indicesCB = DiffVector(IsectVector(indicesC, indicesB), indicesA);
	IndexArray indicesAB = DiffVector(IsectVector(indicesA, indicesB), indicesC);
	IndexArray indicesCBA = isStatC
		? IsectVector(IsectVector(indicesC, indicesB), indicesA)
		: IsectVector(IsectVector(indicesA, indicesB), indicesC);

	//Set the intermediate dists
	hadamardInfo.distIntB = distIntB;
	if (isStatC) {
		hadamardInfo.distIntA = distIntA;
	} else {
		hadamardInfo.distIntC = distIntC;
	}

	//Determine the modes to partition
	hadamardInfo.partModesACA.resize(indicesCA.size());
	hadamardInfo.partModesACC.resize(indicesCA.size());
	hadamardInfo.partModesBCB.resize(indicesCB.size());
	hadamardInfo.partModesBCC.resize(indicesCB.size());
	for(i = 0; i < indicesCA.size(); i++){
		hadamardInfo.partModesACA[i] = IndexOf(indicesA, indicesCA[i]);
		hadamardInfo.partModesACC[i] = IndexOf(indicesC, indicesCA[i]);
	}
	for(i = 0; i < indicesCB.size(); i++) {
		hadamardInfo.partModesBCB[i] = IndexOf(indicesB, indicesCB[i]);
		hadamardInfo.partModesBCC[i] = IndexOf(indicesC, indicesCB[i]);
	}

	//Determine the final alignments needed
	hadamardInfo.alignModesB.resize(indicesCB.size());
	hadamardInfo.alignModesBTo.resize(indicesCB.size());
	for(i = 0; i < indicesCB.size(); i++){
		hadamardInfo.alignModesB[i] = IndexOf(indicesB, indicesCB[i]);
		hadamardInfo.alignModesBTo[i] = IndexOf(indicesC, indicesCB[i]);
	}

	if (isStatC) {
		hadamardInfo.alignModesA.resize(indicesCA.size());
		hadamardInfo.alignModesATo.resize(indicesCA.size());
		for(i = 0; i < indicesCA.size(); i++){
			hadamardInfo.alignModesA[i] = IndexOf(indicesA, indicesCA[i]);
			hadamardInfo.alignModesATo[i] = IndexOf(indicesC, indicesCA[i]);
		}
	}
	else {
		hadamardInfo.alignModesC.resize(indicesCA.size());
		hadamardInfo.alignModesCTo.resize(indicesCA.size());
		for(i = 0; i < indicesCA.size(); i++){
			hadamardInfo.alignModesC[i] = IndexOf(indicesC, indicesCA[i]);
			hadamardInfo.alignModesCTo[i] = IndexOf(indicesA, indicesCA[i]);
		}
	}

	//Set the Block-size info
	//NOTE: There are better ways to do this
	if(blkSizes.size() == 0){
		hadamardInfo.blkSizes.resize(indicesCA.size() + indicesCB.size());
		for(i = 0; i < indicesCA.size() + indicesCB.size(); i++)
			hadamardInfo.blkSizes[i] = 1;
	}else{
		hadamardInfo.blkSizes = blkSizes;
	}

	//Set the local permutation info
	PrintVector(indicesA, "indicesA", true);
	PrintVector(indicesB, "indicesB", true);
	PrintVector(indicesC, "indicesC", true);
	PrintVector(indicesCA, "indicesCA", true);
	PrintVector(indicesCB, "indicesCB", true);
	PrintVector(indicesAB, "indicesAB", true);
	PrintVector(indicesCBA, "indicesCBA", true);
	hadamardInfo.permA = DeterminePermutation(indicesA, ConcatenateVectors(indicesCA, indicesCBA));
	PrintVector(indicesB, "indicesB", true);
	PrintVector(indicesCB, "indicesCB", true);
	PrintVector(indicesCBA, "indicesCBA", true);
	hadamardInfo.permB = DeterminePermutation(indicesB, ConcatenateVectors(indicesCBA, indicesCB));
	PrintVector(indicesC, "indicesC", true);
	PrintVector(indicesCBA, "indicesCBA", true);
	PrintVector(indicesCB, "indicesCB", true);
	PrintVector(indicesCA, "indicesCA", true);
	hadamardInfo.permC = DeterminePermutation(indicesC, ConcatenateVectors(ConcatenateVectors(indicesCBA, indicesCB), indicesCA));
	std::cout << "done with mer\n";
}

template <typename T>
void HadamardStatC(const DistTensor<T>& A, const IndexArray& indicesA, const DistTensor<T>& B, const IndexArray& indicesB, DistTensor<T>& C, const IndexArray& indicesC, const std::vector<Unsigned>& blkSizes){
	std::cout << "Stat C\n";
	TensorDistribution distA = A.TensorDist();
	TensorDistribution distB = B.TensorDist();
	TensorDistribution distC = C.TensorDist();

	TensorDistribution distIntA(distA.size() - 1);
	TensorDistribution distIntB(distB.size() - 1);

	//Setup temp dist A
	distIntA.SetToMatch(distC, indicesC, indicesA);

	//Setup temp dist B
	distIntB.SetToMatch(distC, indicesC, indicesB);

	//Determine how to partition
	BlkHadamardStatCInfo hadamardInfo;
	SetBlkHadamardStatCInfo(
		distIntA, indicesA,
		distIntB, indicesB,
		distC, indicesC,
		blkSizes,
		hadamardInfo, true);

	DistTensor<T> tmpC(distC, C.Grid());
	tmpC.SetLocalPermutation(hadamardInfo.permC);
	Permute(C, tmpC);
	RecurHadamardStatCPartAC(0, hadamardInfo, A, indicesA, B, indicesB, tmpC, indicesC);
	Permute(tmpC, C);
}

//Non-template functions
//bool AnyFalseElem(const std::vector<bool>& vec);
#define PROTO(T) \
	template void HadamardStatC(const DistTensor<T>& A, const IndexArray& indicesA, const DistTensor<T>& B, const IndexArray& indicesB, DistTensor<T>& C, const IndexArray& indicesC, const std::vector<Unsigned>& blkSizes);

//PROTO(Unsigned)
//PROTO(Int)
PROTO(float)
PROTO(double)
//PROTO(char)

} // namespace rote
