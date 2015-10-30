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

void SetTempShapeToMatch(const tmen::GridView& gvMatchAgainst, const IndexArray& indicesMatchAgainst, ObjShape& toMatch, const IndexArray& indicesToMatch){
	Unsigned i;
	for(i = 0; i < indicesToMatch.size(); i++){
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
void ContractStatA(T alpha, const DistTensor<T>& A, const IndexArray& indicesA, const DistTensor<T>& B, const IndexArray& indicesB, T beta, DistTensor<T>& C, const IndexArray& indicesC){
	Unsigned i;
	const tmen::GridView gvA = A.GetGridView();
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
	SetTempShapeToMatch(gvA, indicesA, shapeT, indicesT);

	//Setup temp distIntC
	const tmen::GridView gvC = C.GetGridView();
	SetTensorDistToMatch(distT, indicesT, distIntC, indicesC);
	AppendTensorDistToMatch(reduceGridModes, distC, indicesC, distIntC, indicesC);

	//Perform the distributed computation
	DistTensor<T> intB(distIntB, B.Grid());
	DistTensor<T> intT(shapeT, distT, C.Grid());
	DistTensor<T> intC(C.Shape(), distIntC, C.Grid());

	intB.RedistFrom(B);
	LocalContract(alpha, A.LockedTensor(), indicesA, intB.LockedTensor(), indicesB, T(0), intT.Tensor(), indicesT);
	intC.ReduceScatterUpdateRedistFrom(alpha, intT, beta, reduceTenModes);
	C.RedistFrom(intC);
}

template <typename T>
void ContractStatB(T alpha, const DistTensor<T>& A, const IndexArray& indicesA, const DistTensor<T>& B, const IndexArray& indicesB, T beta, DistTensor<T>& C, const IndexArray& indicesC){
	Unsigned i;
	const tmen::GridView gvB = B.GetGridView();
	IndexArray contractIndices = DetermineContractIndices(indicesA, indicesB);
	IndexArray indicesT = ConcatenateVectors(indicesC, contractIndices);

	TensorDistribution distA = A.TensorDist();
	TensorDistribution distB = B.TensorDist();
	TensorDistribution distC = C.TensorDist();

	ObjShape shapeC = C.Shape();

	ModeDistribution blank;
	TensorDistribution distT(indicesT.size() + 1, blank);
	TensorDistribution distIntA(indicesA.size() + 1, blank);
	TensorDistribution distIntC(indicesC.size() + 1, blank);

	ModeArray reduceTenModes;
	for(i = 0; i < contractIndices.size(); i++){
		reduceTenModes.push_back(C.Order() + i);
	}
	ModeArray reduceGridModes;
	for(i = 0; i < contractIndices.size(); i++){
		int index = IndexOf(indicesB, contractIndices[i]);
		if(index >= 0)
			reduceGridModes = ConcatenateVectors(reduceGridModes, distB[index]);
	}

	//Setup temp distA
	SetTensorDistToMatch(distB, indicesB, distIntA, indicesA);

	//Setup temp distT
	ObjShape shapeT(indicesT.size());
	SetTensorDistToMatch(distB, indicesB, distT, indicesT);
	SetTensorShapeToMatch(shapeC, indicesC, shapeT, indicesT);
	SetTempShapeToMatch(gvB, indicesB, shapeT, indicesT);

	//Setup temp distIntC
	const tmen::GridView gvC = C.GetGridView();
	SetTensorDistToMatch(distT, indicesT, distIntC, indicesC);
	AppendTensorDistToMatch(reduceGridModes, distC, indicesC, distIntC, indicesC);

	//Perform the distributed computation
	DistTensor<T> intA(distIntA, A.Grid());
	DistTensor<T> intT(shapeT, distT, C.Grid());
	DistTensor<T> intC(C.Shape(), distIntC, C.Grid());

	intA.RedistFrom(A);
	LocalContract(alpha, intA.LockedTensor(), indicesA, B.LockedTensor(), indicesB, T(0), intT.Tensor(), indicesT);
	intC.ReduceScatterUpdateRedistFrom(alpha, intT, beta, reduceTenModes);
	C.RedistFrom(intC);
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

	//Perform the distributed computation
	DistTensor<T> intA(distIntA, A.Grid());
	DistTensor<T> intB(distIntB, B.Grid());

	intA.RedistFrom(A);
	intB.RedistFrom(B);
	LocalContractAndLocalEliminate(alpha, intA.LockedTensor(), indicesA, intB.LockedTensor(), indicesB, beta, C.Tensor(), indicesC);

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
//    	printf("StatA\n");
    	ContractStatA(alpha, A, indicesA, B, indicesB, beta, C, indicesC);
    }else if(numElemB > numElemA && numElemB > numElemC){
    	//Stationary B variant
//    	printf("StatB\n");
    	ContractStatB(alpha, A, indicesA, B, indicesB, beta, C, indicesC);
    }else{
    	//Stationary C variant
//    	printf("StatC\n");
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
