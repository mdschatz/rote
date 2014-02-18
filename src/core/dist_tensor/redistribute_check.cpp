/*
   Copyright (c) 2009-2013, Jack Poulson
                      2013, Jeff Hammond
                      2013, Jed Brown
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/

#include "tensormental.hpp"
#include <algorithm>

namespace tmen{

template <typename T>
int CheckReduceScatterRedist(const DistTensor<T>& A, const DistTensor<T>& B, const int reduceIndex, const int scatterIndex){
	if(A.Order() != B.Order() - 1){
		LogicError("CheckReduceScatterRedist: Invalid redistribution request");
	}

	int AOrder = A.Order();
	std::vector<Int> AIndices = A.Indices();
	std::vector<Int> BIndices = B.Indices();

	std::vector<bool> foundIndices(AOrder);
	for(int i = 0; i < AOrder; i++){
		if(std::find(BIndices.begin(), BIndices.end(), AIndices[i]) != BIndices.end())
			foundIndices[i] = true;
	}

	if(AnyZeroElem(foundIndices)){
		LogicError("CheckReduceScatterRedist: Invalid redistribution request");
	}

	//Check that redist modes are assigned properly on input and output
	bool checkPrefix = true;
	bool checkSuffix = false;

	ModeDistribution::iterator redistIndexLocA = std::find(AIndices.begin(), AIndices.end(), scatterIndex);
	ModeDistribution::iterator indexLocB = std::find(BIndices.begin(), BIndices.end(), reduceIndex);
	ModeDistribution::iterator redistIndexLocB = std::find(BIndices.begin(), BIndices.end(), scatterIndex);

	ModeDistribution AScatterIndexDist = A.ModeDist(*redistIndexLocA);
	ModeDistribution BReduceIndexDist = B.ModeDist(*indexLocB);
	ModeDistribution BScatterIndexDist = B.ModeDist(*redistIndexLocB);
	ModeDistribution checkA, checkB;
	checkA = AScatterIndexDist;
	checkB = BScatterIndexDist;
	checkB.insert(checkB.end(), BReduceIndexDist.begin(), BReduceIndexDist.end());

	if(AnyElemwiseNotEqual(checkA, checkB))
		LogicError("CheckReduceScatterRedist: Invalid redistribution request");

	return 1;
}

template<typename T>
int CheckAllGatherRedist(const DistTensor<T>& A, const DistTensor<T>& B, const int allGatherIndex){
	if(A.Order() != B.Order() - 1){
		LogicError("CheckAllGatherRedist: Invalid redistribution request");
	}

	int AOrder = A.Order();
	std::vector<Int> AIndices = A.Indices();
	std::vector<Int> BIndices = B.Indices();

	std::vector<bool> foundIndices(AOrder);
	for(int i = 0; i < AOrder; i++){
		if(std::find(BIndices.begin(), BIndices.end(), AIndices[i]) != BIndices.end())
			foundIndices[i] = true;
	}

	if(AnyZeroElem(foundIndices)){
		LogicError("CheckAllGatherRedist: Invalid redistribution request");
	}

	//Check that redist modes are assigned properly on input and output
	ModeDistribution::iterator allGatherIndexLocA = std::find(AIndices.begin(), AIndices.end(), allGatherIndex);

	ModeDistribution AAllGatherIndexDist = A.ModeDist(*allGatherIndexLocA);
	if(AAllGatherIndexDist.size() != 0)
		LogicError("CheckAllGatherRedist: Invalid redistribution request");

	return 1;
}

template <typename T>
int CheckPartialReduceScatterRedist(const DistTensor<T>& A, const DistTensor<T>& B, const int index, const std::vector<int> redistModes){
	LogicError("CheckPartialReduceScatterRedist: Not implemented");
	//if(AnyElemwiseNotEqual(A.Indices(), B.Indices()))
	//	LogicError("CheckPartialReduceScatterRedist: Invalid redistribution request");
	return 1;
}

#define PROTO(T) \
		template int CheckReduceScatterRedist(const DistTensor<T>& A, const DistTensor<T>& B, const int reduceIndex, const int scatterIndex); \
		template int CheckAllGatherRedist(const DistTensor<T>& A, const DistTensor<T>& B, const int allGatherIndex); \
		template int CheckPartialReduceScatterRedist(const DistTensor<T>& A, const DistTensor<T>& B, const int index, const std::vector<int> redistModes);

PROTO(int)
PROTO(float)
PROTO(double)

} // namespace tmen
