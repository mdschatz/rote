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
void RecurHadamardStatA(Unsigned depth, const BlkHadamardStatAInfo& hadamardInfo, T alpha, const DistTensor<T>& A, const IndexArray& indicesA, const DistTensor<T>& B, const IndexArray& indicesB, T beta, DistTensor<T>& C, const IndexArray& indicesC){
	if(depth == hadamardInfo.partModesB.size()){
		//Perform the distributed computation
		DistTensor<T> intB(hadamardInfo.distIntB, B.Grid());

		const rote::GridView gvA = A.GetGridView();
		IndexArray hadamardIndices = DetermineHadamardIndices(indicesA, indicesB);
		IndexArray indicesT = ConcatenateVectors(indicesC, hadamardIndices);
		ObjShape shapeT(indicesT.size());
		//NOTE: Overwrites values, but this is correct (initially sets to match gvA but then overwrites with C)
		SetTensorShapeToMatch(gvA.ParticipatingShape(), indicesA, shapeT, indicesT);
		SetTensorShapeToMatch(C.Shape(), indicesC, shapeT, indicesT);

		DistTensor<T> intT(shapeT, hadamardInfo.distT, C.Grid());

		intB.AlignModesWith(hadamardInfo.alignModesB, A, hadamardInfo.alignModesBTo);
		intB.SetLocalPermutation(hadamardInfo.permB);
		intB.RedistFrom(B);

		intT.AlignModesWith(hadamardInfo.alignModesT, A, hadamardInfo.alignModesTTo);
		intT.SetLocalPermutation(hadamardInfo.permT);
		intT.ResizeTo(shapeT);

		LocalHadamard(alpha, A.LockedTensor(), indicesA, false, intB.LockedTensor(), indicesB, false, T(0), intT.Tensor(), indicesT, false);
		C.RedistFrom(intT, hadamardInfo.reduceTensorModes, T(1), beta);
		return;
	}
	//Must partition and recur
	//Note: pull logic out
	Unsigned blkSize = hadamardInfo.blkSizes[depth];
	Mode partModeB = hadamardInfo.partModesB[depth];
	Mode partModeC = hadamardInfo.partModesC[depth];
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
		RecurHadamardStatA(depth+1, hadamardInfo, alpha, A, indicesA, B_1, indicesB, beta, C_1, indicesC);
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
void HadamardStatA(T alpha, const DistTensor<T>& A, const IndexArray& indicesA, const DistTensor<T>& B, const IndexArray& indicesB, T beta, DistTensor<T>& C, const IndexArray& indicesC, const std::vector<Unsigned>& blkSizes){
	std::cout << "Stat A\n";
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
	SetBlkHadamardStatCInfo(distIntA, indicesA, distIntB, indicesB, indicesC, blkSizes, hadamardInfo, false);

	if (hadamardInfo.permA != A.LocalPermutation()) {
		DistTensor<T> tmpA(distA, A.Grid());
		tmpA.SetLocalPermutation(hadamardInfo.permA);

		Permute(A, tmpA);
		RecurHadamardStatAPartAC(0, hadamardInfo, A, indicesA, B, indicesB, C, indicesC);
	} else {
		RecurHadamardStatAPartAC(0, hadamardInfo, A, indicesA, B, indicesB, C, indicesC);
	}
}

//Non-template functions
//bool AnyFalseElem(const std::vector<bool>& vec);
#define PROTO(T) \
	template void HadamardStatA(T alpha, const DistTensor<T>& A, const IndexArray& indicesA, const DistTensor<T>& B, const IndexArray& indicesB, T beta, DistTensor<T>& C, const IndexArray& indicesC, const std::vector<Unsigned>& blkSizes);

//PROTO(Unsigned)
//PROTO(Int)
PROTO(float)
PROTO(double)
//PROTO(char)

} // namespace rote
