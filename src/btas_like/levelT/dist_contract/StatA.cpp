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
//		DistTensor<T> intC(C.Shape(), contractInfo.distIntC, C.Grid());

		const rote::GridView gvA = A.GetGridView();
		IndexArray contractIndices = DetermineContractIndices(indicesA, indicesB);
		IndexArray indicesT = ConcatenateVectors(indicesC, contractIndices);
		ObjShape shapeT(indicesT.size());
		SetTensorShapeToMatch(C.Shape(), indicesC, shapeT, indicesT);
		SetTempShapeToMatch(gvA, indicesA, shapeT, indicesT, contractIndices);

		DistTensor<T> intT(shapeT, contractInfo.distT, C.Grid());
//		Scal(T(0.0), C);
//		Scal(T(0.0), intT);

//		printf("distB: %s\n", TensorDistToString(B.TensorDist()).c_str());
		intB.AlignModesWith(contractInfo.alignModesB, A, contractInfo.alignModesBTo);
		intB.SetLocalPermutation(contractInfo.permB);
		intB.RedistFrom(B);

//		printf("distC: %s\n", TensorDistToString(C.TensorDist()).c_str());
//		Print(A, "compute A");
//		Print(B, "compute B");
		intT.AlignModesWith(contractInfo.alignModesT, A, contractInfo.alignModesTTo);
		intT.SetLocalPermutation(contractInfo.permT);
		intT.ResizeTo(shapeT);

		LocalContract(alpha, A.LockedTensor(), indicesA, false, intB.LockedTensor(), indicesB, false, T(0), intT.Tensor(), indicesT, false);
//		Print(intT, "result T");
//		Print(C, "C before");
		C.RedistFrom(intT, contractInfo.reduceTensorModes, T(1), beta);
//		intC.ReduceScatterUpdateRedistFrom(alpha, intT, beta, contractInfo.reduceTensorModes);
//		Print(intC, "intC");
//		C.RedistFrom(intC);
//		Print(C, "C after update");
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
//		if(commRank == 0){
//			printf("depth: %d, iter: %d\n", depth, count);
//			PrintData(A_1, "A_1");
//			PrintData(B_1, "B_1");
//		}
		RecurContractStatA(depth+1, contractInfo, alpha, A, indicesA, B_1, indicesB, beta, C_1, indicesC);
		count++;
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
	//TODO: Needs to be updated correctly!!!!!!
	contractInfo.blkSizes.resize(indicesAB.size());
	for(i = 0; i < indicesAB.size(); i++)
		contractInfo.blkSizes[i] = 4;

	contractInfo.permA = DeterminePermutation(indicesA, ConcatenateVectors(indicesAC, indicesAB));
	contractInfo.permB = DeterminePermutation(indicesB, ConcatenateVectors(indicesAB, indicesBC));
	contractInfo.permT = DeterminePermutation(indicesT, ConcatenateVectors(ConcatenateVectors(indicesAC, indicesBC), indicesAB));
}

template <typename T>
void ContractStatA(T alpha, const DistTensor<T>& A, const IndexArray& indicesA, const DistTensor<T>& B, const IndexArray& indicesB, T beta, DistTensor<T>& C, const IndexArray& indicesC){
	Unsigned i;
	const rote::GridView gvA = A.GetGridView();
	IndexArray contractIndices = DetermineContractIndices(indicesA, indicesB);
	IndexArray indicesT = ConcatenateVectors(indicesC, contractIndices);

	TensorDistribution distA = A.TensorDist();
	TensorDistribution distB = B.TensorDist();
	TensorDistribution distC = C.TensorDist();

	ObjShape shapeC = C.Shape();

	ModeDistribution blank;
	TensorDistribution distT(indicesT.size() + 1, blank);
	TensorDistribution distIntB(indicesB.size() + 1, blank);
	TensorDistribution distIntC(indicesC.size() + 1, blank);

	ModeArray reduceTenModes;
	for(i = 0; i < contractIndices.size(); i++){
		reduceTenModes.push_back(C.Order() + i);
	}
	ModeArray reduceGridModes;
	for(i = 0; i < contractIndices.size(); i++){
		int index = IndexOf(indicesA, contractIndices[i]);
		if(index >= 0)
			reduceGridModes = ConcatenateVectors(reduceGridModes, distA[index]);
	}

	//Setup temp distB
	SetTensorDistToMatch(distA, indicesA, distIntB, indicesB);

	//Setup temp distT
	ObjShape shapeT(indicesT.size());
	SetTensorDistToMatch(distA, indicesA, distT, indicesT);
	SetTensorShapeToMatch(shapeC, indicesC, shapeT, indicesT);
	SetTempShapeToMatch(gvA, indicesA, shapeT, indicesT, contractIndices);

	//Setup temp distIntC
	const rote::GridView gvC = C.GetGridView();
	SetTensorDistToMatch(distT, indicesT, distIntC, indicesC);
	AppendTensorDistToMatch(reduceGridModes, distC, indicesC, distIntC, indicesC);

	//Create the Contract Info
	BlkContractStatAInfo contractInfo;
	SetBlkContractStatAInfo(distT, indicesT, indicesA, distIntB, indicesB, indicesC, contractInfo);

	TensorDistribution tmpDistA = distA;
	DistTensor<T> tmpA(tmpDistA, A.Grid());
	tmpA.SetLocalPermutation(contractInfo.permA);
	Permute(A, tmpA);

	printf("distIntB: %s\n", TensorDistToString(contractInfo.distIntB).c_str());
	printf("distT: %s\n", TensorDistToString(contractInfo.distT).c_str());
//	printf("alpha: %.3f, beta: %.3f\n", alpha, beta);
//	Print(tmpA, "orig tmpA");
//	Print(B, "orig tmpB");
//	Print(C, "orig C");
	RecurContractStatA(0, contractInfo, alpha, tmpA, indicesA, B, indicesB, beta, C, indicesC);
//	Print(C, "final C");
}

//Non-template functions
//bool AnyFalseElem(const std::vector<bool>& vec);
#define PROTO(T) \
	template void ContractStatA(T alpha, const DistTensor<T>& A, const IndexArray& indicesA, const DistTensor<T>& B, const IndexArray& indicesB, T beta, DistTensor<T>& C, const IndexArray& indicesC);

//PROTO(Unsigned)
//PROTO(Int)
PROTO(float)
PROTO(double)
//PROTO(char)

} // namespace rote
