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
void RecurContractStatA(Unsigned depth, const BlkContractStatAInfo& contractInfo, T alpha, const DistTensor<T>& A, const IndexArray& indicesA, const DistTensor<T>& B, const IndexArray& indicesB, T beta, DistTensor<T>& C, const IndexArray& indicesC){
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

		LocalContract(alpha, A.LockedTensor(), indicesA, false, intB.LockedTensor(), indicesB, false, T(0), intT.Tensor(), indicesT, false);
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
		RecurContractStatA(depth+1, contractInfo, alpha, A, indicesA, B_1, indicesB, beta, C_1, indicesC);
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

void SetBlkContractStatAInfo(const TensorDistribution& distT, const IndexArray& indicesT,
							 const IndexArray& indicesA,
							 const TensorDistribution& distIntB, const IndexArray& indicesB,
							 const IndexArray& indicesC,
							 const std::vector<Unsigned>& blkSizes,
							 BlkContractStatAInfo& contractInfo){
	Unsigned i;
	IndexArray indicesAB = DiffVector(indicesA, indicesC);
	IndexArray indicesAC = DiffVector(indicesA, indicesB);
	IndexArray indicesBC = DiffVector(indicesB, indicesA);

	//Set the intermediate dists
	contractInfo.distT = distT;
	contractInfo.distIntB = distIntB;

	//Determine the reduce tensor modes
	contractInfo.reduceTensorModes.resize(indicesAB.size());
	for(i = 0; i < indicesAB.size(); i++){
		contractInfo.reduceTensorModes[i] = indicesC.size() + i;
	}

	//Determine the modes to partition
	contractInfo.partModesB.resize(indicesBC.size());
	contractInfo.partModesC.resize(indicesBC.size());
	for(i = 0; i < indicesBC.size(); i++){
		contractInfo.partModesB[i] = IndexOf(indicesB, indicesBC[i]);
		contractInfo.partModesC[i] = IndexOf(indicesC, indicesBC[i]);
	}

	//Determine the final alignments needed
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

	//Set the Block-size info
	if(blkSizes.size() == 0){
		contractInfo.blkSizes.resize(indicesBC.size());
		for(i = 0; i < indicesBC.size(); i++)
			contractInfo.blkSizes[i] = 32;
	}else{
		contractInfo.blkSizes = blkSizes;
	}

	contractInfo.permA = DeterminePermutation(indicesA, ConcatenateVectors(indicesAC, indicesAB));
	contractInfo.permB = DeterminePermutation(indicesB, ConcatenateVectors(indicesAB, indicesBC));
	contractInfo.permT = DeterminePermutation(indicesT, ConcatenateVectors(ConcatenateVectors(indicesAC, indicesBC), indicesAB));
}

template <typename T>
void ContractStatA(T alpha, const DistTensor<T>& A, const IndexArray& indicesA, const DistTensor<T>& B, const IndexArray& indicesB, T beta, DistTensor<T>& C, const IndexArray& indicesC, const std::vector<Unsigned>& blkSizes){
	Unsigned i;
	const rote::GridView gvA = A.GetGridView();
	IndexArray contractIndices = DetermineContractIndices(indicesA, indicesB);
	IndexArray indicesT = ConcatenateVectors(indicesC, contractIndices);

	TensorDistribution distA = A.TensorDist();
	TensorDistribution distC = C.TensorDist();

	ObjShape shapeC = C.Shape();

	TensorDistribution distT(indicesT.size());
	TensorDistribution distIntB(indicesB.size());
	TensorDistribution distIntC(indicesC.size());

	ModeArray reduceGridModes;
	for(i = 0; i < contractIndices.size(); i++){
		int index = IndexOf(indicesA, contractIndices[i]);
		if(index >= 0)
			reduceGridModes = ConcatenateVectors(reduceGridModes, distA[index].Entries());
	}

	//Setup temp distB
	distIntB.SetToMatch(distA, indicesA, indicesB);

	//Setup temp distT
	distT.SetToMatch(distA, indicesA, indicesT);

	//Setup temp distIntC
	const rote::GridView gvC = C.GetGridView();
	distIntC.SetToMatch(distT, indicesT, indicesC);
	distIntC.AppendToMatchForGridModes(reduceGridModes, distC, indicesC, indicesC);

	//Create the Contract Info
	BlkContractStatAInfo contractInfo;
	SetBlkContractStatAInfo(distT, indicesT, indicesA, distIntB, indicesB, indicesC, blkSizes, contractInfo);

	TensorDistribution tmpDistA = distA;
	DistTensor<T> tmpA(tmpDistA, A.Grid());
	tmpA.SetLocalPermutation(contractInfo.permA);
	Permute(A, tmpA);

	RecurContractStatA(0, contractInfo, alpha, tmpA, indicesA, B, indicesB, beta, C, indicesC);
}

//Non-template functions
//bool AnyFalseElem(const std::vector<bool>& vec);
#define PROTO(T) \
	template void ContractStatA(T alpha, const DistTensor<T>& A, const IndexArray& indicesA, const DistTensor<T>& B, const IndexArray& indicesB, T beta, DistTensor<T>& C, const IndexArray& indicesC, const std::vector<Unsigned>& blkSizes);

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
