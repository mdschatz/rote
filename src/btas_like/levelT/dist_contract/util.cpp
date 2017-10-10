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

// Partition helpers
template <typename T>
void Contract<T>::runHelperPartitionAB(
	Unsigned depth, BlkContractStatCInfo& contractInfo,
	T alpha,
  const DistTensor<T>& A, const IndexArray& indicesA,
	const DistTensor<T>& B, const IndexArray& indicesB,
	T beta,
        DistTensor<T>& C, const IndexArray& indicesC
) {
  if(depth == contractInfo.partModesA.size()){
		DistTensor<T> intA(contractInfo.distIntA, A.Grid());
		intA.SetLocalPermutation(contractInfo.permA);
		intA.AlignModesWith(contractInfo.alignModesA, C, contractInfo.alignModesATo);
		intA.RedistFrom(A);

		DistTensor<T> intB(contractInfo.distIntB, B.Grid());
		intB.AlignModesWith(contractInfo.alignModesB, C, contractInfo.alignModesBTo);
		intB.SetLocalPermutation(contractInfo.permB);
		intB.RedistFrom(B);

		Contract<T>::run(
			alpha,
			intA.LockedTensor(), indicesA,
			intB.LockedTensor(), indicesB,
			T(1),
			C.Tensor(), indicesC,
			true, false
		);
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
		Contract<T>::runHelperPartitionAB(depth+1, contractInfo, alpha, A_1, indicesA, B_1, indicesB, beta, C, indicesC);
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

template <typename T>
void Contract<T>::runHelperPartitionBC(
	Unsigned depth, BlkContractStatCInfo& contractInfo,
	T alpha,
  const DistTensor<T>& A, const IndexArray& indicesA,
	const DistTensor<T>& B, const IndexArray& indicesB,
	T beta,
        DistTensor<T>& C, const IndexArray& indicesC
) {
  if(depth == contractInfo.partModesB.size()){
		//Perform the distributed computation
		DistTensor<T> intB(contractInfo.distIntB, B.Grid());

		const rote::GridView gvA = A.GetGridView();
		IndexArray contractIndices = DetermineContractIndices(indicesA, indicesB);
		IndexArray indicesT = ConcatenateVectors(indicesC, contractIndices);
		ObjShape shapeT(indicesT.size());
		//NOTE: Overwrites values, but this is correct (initially sets to match gvA but then overwrites with C)
		SetTensorShapeToMatch(gvA.ParticipatingShape(), indicesA, shapeT, indicesT);
		SetTensorShapeToMatch(C.Shape(), indicesC, shapeT, indicesT);

		DistTensor<T> intT(shapeT, contractInfo.distT, C.Grid());

		intB.AlignModesWith(contractInfo.alignModesB, A, contractInfo.alignModesBTo);
		intB.SetLocalPermutation(contractInfo.permB);
		intB.RedistFrom(B);

		intT.AlignModesWith(contractInfo.alignModesT, A, contractInfo.alignModesTTo);
		intT.SetLocalPermutation(contractInfo.permT);
		intT.ResizeTo(shapeT);

		Contract<T>::run(
			alpha,
			A.LockedTensor(), indicesA,
			intB.LockedTensor(), indicesB,
			T(0),
			intT.Tensor(), indicesT,
			false, false
		);
		C.RedistFrom(intT, contractInfo.reduceTensorModes, T(1), beta);
		return;
	}
	//Must partition and recur
	//Note: pull logic out
	Unsigned blkSize = contractInfo.blkSizes[depth];
	Mode partModeB = contractInfo.partModesB[depth];
	Mode partModeC = contractInfo.partModesC[depth];
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
		Contract<T>::runHelperPartitionBC(depth+1, contractInfo, alpha, A, indicesA, B_1, indicesB, beta, C_1, indicesC);
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

// Struct interface
template<typename T>
void Contract<T>::setContractInfo(
	const DistTensor<T>& A, const IndexArray& indicesA,
  const DistTensor<T>& B, const IndexArray& indicesB,
  const DistTensor<T>& C, const IndexArray& indicesC,
  const std::vector<Unsigned>& blkSizes, bool isStatC,
        BlkContractStatCInfo& contractInfo
) {
  Unsigned i;
	IndexArray indicesAC = DiffVector(indicesC, indicesB);
	IndexArray indicesBC = DiffVector(indicesC, indicesA);
	IndexArray indicesAB = DiffVector(indicesA, indicesC);
	IndexArray indicesT = ConcatenateVectors(indicesC, indicesAB);

	TensorDistribution distA = A.TensorDist();
	TensorDistribution distB = B.TensorDist();
	TensorDistribution distC = C.TensorDist();

	TensorDistribution distT(indicesT.size());
	TensorDistribution distIntA(indicesA.size());
	TensorDistribution distIntB(indicesB.size());

	ModeArray reduceGridModes;
	for(Unsigned i = 0; i < indicesAB.size(); i++){
		int index = IndexOf(indicesA, indicesAB[i]);
		if(index >= 0)
			reduceGridModes = ConcatenateVectors(reduceGridModes, distA[index].Entries());
	}

	// Set up temp dists
	if (isStatC) {
		distIntA.SetToMatch(distC, indicesC, indicesA);
		distIntB.SetToMatch(distC, indicesC, indicesB);
	} else {
		distIntB.SetToMatch(distA, indicesA, indicesB);
		distT.SetToMatch(distA, indicesA, indicesT);
	}


	//Set the intermediate dists
	contractInfo.distIntA = distIntA;
	contractInfo.distIntB = distIntB;
	contractInfo.distT = distT;

	//Determine the reduce tensor modes
	contractInfo.reduceTensorModes.resize(indicesAB.size());
	for(i = 0; i < indicesAB.size(); i++){
		contractInfo.reduceTensorModes[i] = indicesC.size() + i;
	}

	//Determine the modes to partition
	if (isStatC) {
		contractInfo.partModesA.resize(indicesAB.size());
		contractInfo.partModesB.resize(indicesAB.size());
		for(i = 0; i < indicesAB.size(); i++){
			contractInfo.partModesA[i] = IndexOf(indicesA, indicesAB[i]);
			contractInfo.partModesB[i] = IndexOf(indicesB, indicesAB[i]);
		}
	} else {
		contractInfo.partModesB.resize(indicesBC.size());
		contractInfo.partModesC.resize(indicesBC.size());
		for(i = 0; i < indicesBC.size(); i++){
			contractInfo.partModesB[i] = IndexOf(indicesB, indicesBC[i]);
			contractInfo.partModesC[i] = IndexOf(indicesC, indicesBC[i]);
		}
	}

	//Determine the final alignments needed
	if (isStatC) {
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
	} else {
		contractInfo.alignModesB.resize(indicesAB.size());
		contractInfo.alignModesBTo.resize(indicesAB.size());
		for(i = 0; i < indicesAB.size(); i++){
			contractInfo.alignModesB[i] = IndexOf(indicesB, indicesAB[i]);
			contractInfo.alignModesBTo[i] = IndexOf(indicesA, indicesAB[i]);
		}
		contractInfo.alignModesT.resize(indicesAC.size());
		contractInfo.alignModesTTo.resize(indicesAC.size());
		for(i = 0; i < indicesAC.size(); i++){
			contractInfo.alignModesT[i] = IndexOf(indicesT, indicesAC[i]);
			contractInfo.alignModesTTo[i] = IndexOf(indicesA, indicesAC[i]);
		}
	}

	//Set the Block-size info
	//NOTE: There are better ways to do this
	if(blkSizes.size() == 0){
		contractInfo.blkSizes.resize(isStatC ? indicesAB.size() : indicesBC.size());
		for(i = 0; i < contractInfo.blkSizes.size(); i++)
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
	Permutation permT(indicesT, ConcatenateVectors(ConcatenateVectors(indicesAC, indicesBC), indicesAB));

	contractInfo.permA = permA;
	contractInfo.permB = permB;
	contractInfo.permC = permC;
	contractInfo.permT = permT;
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
