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

void SetTensorDistToMatch(const TensorDistribution& matchAgainst, const IndexArray& indicesMatchAgainst, TensorDistribution& toMatch, const IndexArray& indicesToMatch){
	Unsigned i;
	ModeArray blank;
	for(i = 0; i < indicesToMatch.size(); i++){
		int index = IndexOf(indicesMatchAgainst, indicesToMatch[i]);
		if(index >= 0)
			toMatch[i] = matchAgainst[index];
	}
	toMatch[toMatch.size() - 1] = matchAgainst[matchAgainst.size() - 1];
}


void SetTensorShapeToMatch(const ObjShape& matchAgainst, const IndexArray& indicesMatchAgainst, ObjShape& toMatch, const IndexArray& indicesToMatch){
	Unsigned i;
	for(i = 0; i < indicesToMatch.size(); i++){
		int index = IndexOf(indicesMatchAgainst, indicesToMatch[i]);
		if(index >= 0)
			toMatch[i] = matchAgainst[index];
	}
}

void SetTempShapeToMatch(const rote::GridView& gvMatchAgainst, const IndexArray& indicesMatchAgainst, ObjShape& toMatch, const IndexArray& indicesToMatch, const IndexArray& sharedIndices){
	Unsigned i;
	for(i = 0; i < indicesToMatch.size(); i++){
		if(IndexOf(sharedIndices, indicesToMatch[i]) == -1)
			continue;
		int index = IndexOf(indicesMatchAgainst, indicesToMatch[i]);
		if(index >= 0)
			toMatch[i] = gvMatchAgainst.Dimension(index);
	}
}

void AppendTensorDistToMatch(const ModeArray& modes, const TensorDistribution& matchAgainst, const IndexArray& indicesMatchAgainst, TensorDistribution& toMatch, const IndexArray& indicesToMatch){
	Unsigned i, j;
	ModeArray modesFound;

	for(i = 0; i < modes.size(); i++){
		Mode modeToMove = modes[i];
		for(j = 0; j < indicesMatchAgainst.size(); j++){
			Index indexMatchAgainst = indicesMatchAgainst[j];
			ModeDistribution modeDistToCheck = matchAgainst[j];
			if(Contains(modeDistToCheck, modeToMove)){
				int index = IndexOf(indicesToMatch, indexMatchAgainst);
				if(index >= 0){
					toMatch[index].push_back(modeToMove);
					modesFound.push_back(modeToMove);
					break;
				}
			}
		}
	}

	//Deal with the modes that disappeared
	ModeArray modesNotFound = DiffVector(modes, modesFound);
	toMatch[indicesToMatch.size() - 1] = ConcatenateVectors(toMatch[indicesToMatch.size() - 1], modesNotFound);
}

template <typename T>
void RecurContractStatA(Unsigned depth, const BlkContractStatAInfo& contractInfo, T alpha, const DistTensor<T>& A, const IndexArray& indicesA, const DistTensor<T>& B, const IndexArray& indicesB, T beta, DistTensor<T>& C, const IndexArray& indicesC){
	if(depth == contractInfo.partModesB.size()){
		//Perform the distributed computation
		DistTensor<T> intB(contractInfo.distIntB, B.Grid());
		DistTensor<T> intC(C.Shape(), contractInfo.distIntC, C.Grid());

		const rote::GridView gvA = A.GetGridView();
		IndexArray contractIndices = DetermineContractIndices(indicesA, indicesB);
		IndexArray indicesT = ConcatenateVectors(indicesC, contractIndices);
		ObjShape shapeT(indicesT.size());
		SetTensorShapeToMatch(intC.Shape(), indicesC, shapeT, indicesT);
		SetTempShapeToMatch(gvA, indicesA, shapeT, indicesT, contractIndices);

		DistTensor<T> intT(shapeT, contractInfo.distT, C.Grid());
//		Scal(T(0.0), C);
//		Scal(T(0.0), intT);

		intB.AlignModesWith(contractInfo.alignModesB, A, contractInfo.alignModesBTo);
		intB.SetLocalPermutation(contractInfo.permB);
		intB.RedistFrom(B);

//		Print(A, "compute A");
//		Print(B, "compute B");
		intT.AlignModesWith(contractInfo.alignModesT, A, contractInfo.alignModesTTo);
		intT.SetLocalPermutation(contractInfo.permT);
		intT.ResizeTo(shapeT);

		LocalContract(alpha, A.LockedTensor(), indicesA, false, intB.LockedTensor(), indicesB, false, T(0), intT.Tensor(), contractInfo.indicesT, false);
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

void SetBlkContractStatAInfo(const ObjShape& shapeT, const TensorDistribution& distT, const IndexArray& indicesT,
							 const ObjShape& shapeA, const IndexArray& indicesA,
							 const ObjShape& shapeB, const TensorDistribution& distIntB, const IndexArray& indicesB,
							 const ObjShape& shapeC, const TensorDistribution& distIntC, const IndexArray& indicesC,
							 BlkContractStatAInfo& contractInfo){
	Unsigned i;
	IndexArray indicesAB = DiffVector(indicesB, indicesC);
	IndexArray indicesAC = DiffVector(indicesA, indicesB);
	IndexArray indicesBC = DiffVector(indicesB, indicesA);

	//Set the intermediate dists
	contractInfo.distT = distT;
	contractInfo.distIntB = distIntB;
	contractInfo.distIntC = distIntC;

	//Set the shape and indices of T
	contractInfo.shapeT = shapeT;
	contractInfo.indicesT = indicesT;

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

//	printf("determining StatA\n");
//	PrintVector(indicesA, "indicesA");
//	PrintVector(indicesB, "indicesB");
//	PrintVector(indicesC, "indicesC");
//	PrintVector(indicesC, "indicesT");
//	PrintVector(ConcatenateVectors(indicesAC, indicesAB), "ConcatedindicesA");
//	PrintVector(ConcatenateVectors(indicesAB, indicesBC), "ConcatedindicesB");
//	PrintVector(ConcatenateVectors(indicesAC, indicesBC), "ConcatedindicesC");
//	PrintVector(ConcatenateVectors(ConcatenateVectors(indicesAC, indicesBC), indicesAB), "CocatedindicesT");
	contractInfo.permA = DeterminePermutation(indicesA, ConcatenateVectors(indicesAC, indicesAB));
	contractInfo.permB = DeterminePermutation(indicesB, ConcatenateVectors(indicesAB, indicesBC));
	contractInfo.permT = DeterminePermutation(indicesT, ConcatenateVectors(ConcatenateVectors(indicesAC, indicesBC), indicesAB));
//	PrintVector(contractInfo.permA, "permA");
//	PrintVector(contractInfo.permT, "permT");
//	PrintVector(contractInfo.permB, "permB");
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
	SetBlkContractStatAInfo(shapeT, distT, indicesT, A.Shape(), indicesA, B.Shape(), distIntB, indicesB, C.Shape(), distIntC, indicesC, contractInfo);

	TensorDistribution tmpDistA = distA;
	DistTensor<T> tmpA(tmpDistA, A.Grid());
	tmpA.SetLocalPermutation(contractInfo.permA);
	Permute(A, tmpA);

//	printf("alpha: %.3f, beta: %.3f\n", alpha, beta);
//	Print(tmpA, "orig tmpA");
//	Print(B, "orig tmpB");
//	Print(C, "orig C");
	RecurContractStatA(0, contractInfo, alpha, tmpA, indicesA, B, indicesB, beta, C, indicesC);
//	Print(C, "final C");
}

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

//		if(commRank == 0)
//			PrintData(A, "compute A");
//		Print(A, "compute A");
//		if(commRank == 0)
//			PrintData(intA, "intA");
//		Print(intA, "intA");
//		if(commRank == 0)
//			PrintData(B, "compute B");
//		Print(B, "compute B");
//		if(commRank == 0)
//			PrintData(intB, "intB");
//		Print(intB, "intB");
//		Print(C, "before update");
//		if(commRank == 0){
//			PrintVector(indicesA, "indicesA");
//			PrintVector(indicesB, "indicesB");
//			PrintVector(indicesC, "indicesC");
//		}
		LocalContractAndLocalEliminate(alpha, intA.LockedTensor(), indicesA, false, intB.LockedTensor(), indicesB, false, T(1), C.Tensor(), indicesC, false);

		contractInfo.firstIter = false;
//		Print(C, "after update");
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
//		if(commRank == 0){
//			printf("depth: %d, iter: %d\n", depth, count);
//			PrintData(A_1, "A_1");
//			PrintData(B_1, "B_1");
//		}
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

void SetBlkContractStatCInfo(const ObjShape& shapeA, const TensorDistribution& distIntA, const IndexArray& indicesA,
		                     const ObjShape& shapeB, const TensorDistribution& distIntB, const IndexArray& indicesB,
							 const ObjShape& shapeC, const IndexArray& indicesC,
							 BlkContractStatCInfo& contractInfo){
	Unsigned i;
	IndexArray indicesAB = DiffVector(indicesB, indicesC);
	IndexArray indicesAC = DiffVector(indicesA, indicesB);
	IndexArray indicesBC = DiffVector(indicesB, indicesA);

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
	//TODO: Needs to be updated correctly!!!!!!
	contractInfo.blkSizes.resize(indicesAB.size());
	for(i = 0; i < indicesAB.size(); i++)
		contractInfo.blkSizes[i] = 4;
	contractInfo.firstIter = true;

	//Set the local permutation info
	contractInfo.permA = DeterminePermutation(indicesA, ConcatenateVectors(indicesAC, indicesAB));
	contractInfo.permB = DeterminePermutation(indicesB, ConcatenateVectors(indicesAB, indicesBC));
	contractInfo.permC = DeterminePermutation(indicesC, ConcatenateVectors(indicesAC, indicesBC));
}

template <typename T>
void ContractStatC(T alpha, const DistTensor<T>& A, const IndexArray& indicesA, const DistTensor<T>& B, const IndexArray& indicesB, T beta, DistTensor<T>& C, const IndexArray& indicesC){
	TensorDistribution distA = A.TensorDist();
	TensorDistribution distB = B.TensorDist();
	TensorDistribution distC = C.TensorDist();

	ModeArray blank;
	TensorDistribution distIntA(distA.size(), blank);
	TensorDistribution distIntB(distB.size(), blank);

	//Setup temp dist A
	SetTensorDistToMatch(distC, indicesC, distIntA, indicesA);

	//Setup temp dist B
	SetTensorDistToMatch(distC, indicesC, distIntB, indicesB);

	//Determine how to partition
	BlkContractStatCInfo contractInfo;
	SetBlkContractStatCInfo(A.Shape(), distIntA, indicesA, B.Shape(), distIntB, indicesB, C.Shape(), indicesC, contractInfo);

//	Print(C, "before scal");
	TensorDistribution tmpDistC = distC;
	DistTensor<T> tmpC(tmpDistC, C.Grid());
	tmpC.SetLocalPermutation(contractInfo.permC);
	Permute(C, tmpC);
	Scal(beta, tmpC);
	RecurContractStatC(0, contractInfo, alpha, A, indicesA, B, indicesB, beta, tmpC, indicesC);

	Permute(tmpC, C);
}

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
    	printf("StatA\n");
    	ContractStatA(alpha, A, indicesA, B, indicesB, beta, C, indicesC);
    }else if(numElemB > numElemA && numElemB > numElemC){
    	//Stationary B variant
    	printf("StatB\n");
    	ContractStatA(alpha, B, indicesB, A, indicesA, beta, C, indicesC);
    }else{
    	//Stationary C variant
    	printf("StatC\n");
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

} // namespace rote
