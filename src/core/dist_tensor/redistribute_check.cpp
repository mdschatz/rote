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
#include "tensormental/util/vec_util.hpp"

namespace tmen{

//TODO: Check all unaffected indices are distributed similarly (Only done for CheckPermutationRedist currently)
template <typename T>
Int CheckPermutationRedist(const DistTensor<T>& B, const DistTensor<T>& A, const Index permuteIndex){
    Unsigned i;
    const tmen::GridView gvA = A.GridView();

    const Unsigned AOrder = A.Order();
    const Unsigned BOrder = B.Order();

    //Test indices are correct
    IndexArray BIndices = B.Indices();
    IndexArray AIndices = A.Indices();
    std::vector<bool> foundIndices(BOrder,0);

    //Test index being reduced has been reduced to correct dimension
    const Mode permuteModeA = A.ModeOfIndex(permuteIndex);
    const Mode permuteModeB = B.ModeOfIndex(permuteIndex);

    //Test order retained
    if(BOrder != AOrder){
        LogicError("CheckPermutationRedist: Permutation retains the same order of objects");
    }

    //Test dimension has been resized correctly
    //NOTE: Uses fancy way of performing Ceil() on integer division
    if(B.Dimension(permuteModeB) != A.Dimension(permuteModeA))
        LogicError("CheckPartialReduceScatterRedist: Permutation retains the same dimension of indices");

    //Ensure indices of input and output are similar
    for(i = 0; i < BOrder; i++){
        if(std::find(AIndices.begin(), AIndices.end(), BIndices[i]) != AIndices.end())
            foundIndices[i] = true;
    }
    if(AnyFalseElem(foundIndices)){
        LogicError("CheckPermutationRedist: Input and Output objects represent different indices");
    }

    //Make sure all indices are distributed similarly
    for(i = 0; i < BOrder; i++){
        Index index = B.IndexOfMode(i);
        if(index == permuteIndex){
            if(!EqualUnderPermutation(B.IndexDist(index), A.IndexDist(index)))
                LogicError("CheckPermutationRedist: Distribution of permuted index does not involve same modes of grid as input");
        }else{
            if(AnyElemwiseNotEqual(B.IndexDist(index), A.IndexDist(index)))
                LogicError("CheckPartialReduceScatterRedist: All indices must be distributed similarly");
        }
    }
    return 1;

}

//TODO: Properly Check indices and distributions match between input and output
//TODO: Make sure outgoing reduce Index differs from incoming (partial reduction forms a new Index)
template <typename T>
Int CheckPartialReduceScatterRedist(const DistTensor<T>& B, const DistTensor<T>& A, const Index reduceScatterIndex){
    Unsigned i;
    const tmen::GridView gvA = A.GridView();

    const Unsigned AOrder = A.Order();
    const Unsigned BOrder = B.Order();

    //Test indices are correct
    IndexArray BIndices = B.Indices();
    IndexArray AIndices = A.Indices();
    std::vector<bool> foundIndices(BOrder,0);

    //Test index being reduced has been reduced to correct dimension
    const Mode reduceScatterModeA = A.ModeOfIndex(reduceScatterIndex);
    const Mode reduceScatterModeB = B.ModeOfIndex(reduceScatterIndex);


    //Test order retained
    if(BOrder != AOrder){
        LogicError("CheckPartialReduceScatterRedist: Partial Reduction must retain index being reduced");
    }

    //Test dimension has been resized correctly
    //NOTE: Uses fancy way of performing Ceil() on integer division
    if(B.Dimension(reduceScatterModeB) != Max(1,MaxLength(A.Dimension(reduceScatterModeA), gvA.Dimension(reduceScatterModeA))))
        LogicError("CheckPartialReduceScatterRedist: Partial Reduction must reduce index dimension by factor Dimension/Grid Dimension");

    //Ensure indices of input and output are similar
    for(i = 0; i < BOrder; i++){
        if(std::find(AIndices.begin(), AIndices.end(), BIndices[i]) != AIndices.end())
            foundIndices[i] = true;
    }
    if(AnyFalseElem(foundIndices)){
        LogicError("CheckPartialReduceScatterRedist: Input and Output objects represent different indices");
    }

    //Make sure all indices are distributed similarly between input and output (excluding reduce+scatter indices)
    for(i = 0; i < BOrder; i++){
        Index index = B.IndexOfMode(i);
        if(AnyElemwiseNotEqual(B.IndexDist(index), A.IndexDist(index)))
            LogicError("CheckPartialReduceScatterRedist: All indices must be distributed similarly");
    }
    return 1;
}

//TODO: Properly Check indices and distributions match between input and output
template <typename T>
Int CheckReduceScatterRedist(const DistTensor<T>& B, const DistTensor<T>& A, const Index reduceIndex, const Index scatterIndex){
    Unsigned i;
    const tmen::GridView gvA = A.GridView();

	//Test elimination of index
	const Unsigned AOrder = A.Order();
	const Unsigned BOrder = B.Order();

	//Test indices are correct
	IndexArray BIndices = B.Indices();
	IndexArray AIndices = A.Indices();
	std::vector<bool> foundIndices(BOrder,0);


	//Check that redist modes are assigned properly on input and output
	ModeDistribution BScatterIndexDist = B.IndexDist(scatterIndex);
	ModeDistribution AReduceIndexDist = A.IndexDist(reduceIndex);
	ModeDistribution AScatterIndexDist = A.IndexDist(scatterIndex);

	//Test elimination of index
	if(BOrder != AOrder - 1){
		LogicError("CheckReduceScatterRedist: Full Reduction requires elimination of index being reduced");
	}

	//Test no wrapping of index to reduce
	if(A.Dimension(reduceIndex) > gvA.Dimension(reduceIndex))
		LogicError("CheckReduceScatterRedist: Full Reduction requires global mode dimension <= gridView dimension");

	//Ensure indices of input and output are similar
	for(i = 0; i < BOrder; i++){
		if(std::find(AIndices.begin(), AIndices.end(), BIndices[i]) != AIndices.end())
			foundIndices[i] = true;
	}
	if(AnyFalseElem(foundIndices)){
		LogicError("CheckReduceScatterRedist: Input and Output objects represent different indices");
	}

	//Make sure all indices are distributed similarly between input and output (excluding reduce+scatter indices)
	for(i = 0; i < BOrder; i++){
	    Index index = B.IndexOfMode(i);
	    if(index == scatterIndex){
	        ModeDistribution check(BScatterIndexDist.end() - AReduceIndexDist.size(), BScatterIndexDist.end());
            if(AnyElemwiseNotEqual(check, AReduceIndexDist))
                LogicError("CheckReduceScatterRedist: Reduce index distribution of A must be a suffix of Scatter index distribution of B");
	    }
	    else{
	        if(AnyElemwiseNotEqual(B.IndexDist(index), A.IndexDist(index)))
	            LogicError("CheckReduceScatterRedist: All indices not involved in reduce-scatter must be distributed similarly");
	    }
	}

	return 1;
}

template<typename T>
Int CheckAllGatherRedist(const DistTensor<T>& A, const DistTensor<T>& B, const Index allGatherIndex){
	if(A.Order() != B.Order()){
		LogicError("CheckAllGatherRedist: Objects being redistributed must be of same order");
	}

	Unsigned AOrder = A.Order();
	IndexArray AIndices = A.Indices();
	IndexArray BIndices = B.Indices();

	std::vector<bool> foundIndices(AOrder, 0);
	for(int i = 0; i < AOrder; i++){
		if(std::find(BIndices.begin(), BIndices.end(), AIndices[i]) != BIndices.end())
			foundIndices[i] = true;
	}

	if(AnyFalseElem(foundIndices)){
		LogicError("CheckAllGatherRedist: Objects being redistributed must represent same indices");
	}

	//Check that redist modes are assigned properly on input and output
	ModeDistribution::iterator allGatherIndexLocA = std::find(AIndices.begin(), AIndices.end(), allGatherIndex);

	ModeDistribution AAllGatherIndexDist = A.ModeDist(*allGatherIndexLocA);
	if(AAllGatherIndexDist.size() != 0)
		LogicError("CheckAllGatherRedist: Allgather only redistributes to * (for now)");

	return true;
}

template <typename T>
Int CheckPartialReduceScatterRedist(const DistTensor<T>& A, const DistTensor<T>& B, const Index rsIndex, const ModeArray& rsGridModes){
	LogicError("CheckPartialReduceScatterRedist: Not implemented");
	//if(AnyElemwiseNotEqual(A.Indices(), B.Indices()))
	//	LogicError("CheckPartialReduceScatterRedist: Invalid redistribution request");
	return 1;
}


//TODO: Check that allToAllIndices and commGroups are valid
template <typename T>
Int CheckAllToAllDoubleIndexRedist(const DistTensor<T>& A, const DistTensor<T>& B, const std::pair<Index, Index>& allToAllIndices, const std::pair<ModeArray, ModeArray >& a2aCommGroups){
    if(A.Order() != B.Order())
        LogicError("CheckAllToAllDoubleIndexRedist: Objects being redistributed must be of same order");

    Unsigned order = A.Order();
    IndexArray AIndices = A.Indices();
    IndexArray BIndices = B.Indices();

    std::vector<bool> foundIndices(order, 0);
    for(Unsigned i = 0; i < order; i++){
        if(std::find(BIndices.begin(), BIndices.end(), AIndices[i]) != BIndices.end())
            foundIndices[i] = true;
    }

    if(AnyFalseElem(foundIndices)){
        LogicError("CheckAllToAllDoubleIndexRedist: Objects being redistributed must represent same indices");
    }
    return 1;
}

template<typename T>
Int CheckLocalRedist(DistTensor<T>& B, const DistTensor<T>& A, const Index localIndex, const ModeArray& gridRedistModes){
    if(A.Order() != B.Order())
        LogicError("CheckLocalRedist: Objects being redistributed must be of same order");

    Unsigned i, j;
    Unsigned order = A.Order();
    IndexArray AIndices = A.Indices();
    IndexArray BIndices = B.Indices();
    TensorDistribution distB = B.TensorDist();
    ModeDistribution localIndexDistA = A.ModeDist(A.ModeOfIndex(localIndex));
    ModeDistribution localIndexDistB = B.ModeDist(B.ModeOfIndex(localIndex));

    if(localIndexDistB.size() != localIndexDistA.size() + gridRedistModes.size())
        LogicError("CheckLocalReist: Input object cannot be redistributed to output object");

    std::vector<bool> foundIndices(order, 0);
    for(i = 0; i < order; i++){
        if(std::find(BIndices.begin(), BIndices.end(), AIndices[i]) != BIndices.end())
            foundIndices[i] = true;
    }

    if(AnyFalseElem(foundIndices)){
        LogicError("CheckLocalRedist: Objects being redistributed must represent same indices");
    }

    if(std::find(AIndices.begin(), AIndices.end(), localIndex) == AIndices.end())
        LogicError("CheckLocalRedist: Input object does not represent index to be redistributed");

    ModeArray check(localIndexDistB);
    for(i = 0; i < localIndexDistA.size(); i++)
        check[i] = localIndexDistA[i];
    for(i = 0; i < gridRedistModes.size(); i++)
        check[localIndexDistA.size() + i] = gridRedistModes[i];

    for(i = 0; i < check.size(); i++){
        if(check[i] != localIndexDistB[i])
            LogicError("CheckLocalRedist: Output distribution cannot be formed from supplied parameters");
    }

    ModeArray boundModes;
    for(i = 0; i < distB.size(); i++){
        for(j = 0; j < distB[i].size(); j++){
            boundModes.push_back(distB[i][j]);
        }
    }

    for(i = 0; i < gridRedistModes.size(); i++)
        if(std::find(boundModes.begin(), boundModes.end(), gridRedistModes[i]) != boundModes.end())
            LogicError("CheckLocalRedist: Attempting to redistribute with already bound mode of the grid");

    return 1;
}

//TODO: Make sure these checks are correct (look at LDim, strides, distributions, etc).
template <typename T>
Int CheckRemoveUnitIndicesRedist(const DistTensor<T>& B, const DistTensor<T>& A, const IndexArray& unitIndices){
    const Unsigned orderA = A.Order();
    const Unsigned orderB = B.Order();
    const Unsigned nIndicesRemove = unitIndices.size();

    if(orderB != orderA - nIndicesRemove)
        LogicError("CheckRemoveUnitIndicesRedist: Object being redistributed must be of correct order");

    Unsigned order = A.Order();
    IndexArray AIndices = A.Indices();
    IndexArray BIndices = B.Indices();

    std::vector<bool> foundIndices(order, 0);
    for(Unsigned i = 0; i < order; i++){
        if(std::find(AIndices.begin(), AIndices.end(), BIndices[i]) != AIndices.end())
            foundIndices[i] = true;
    }

    if(AnyFalseElem(foundIndices)){
        LogicError("CheckAllToAllDoubleIndexRedist: Input object must represent same indices as output object");
    }
    return 1;
}

template <typename T>
Int CheckIntroduceUnitIndicesRedist(const DistTensor<T>& B, const DistTensor<T>& A, const std::vector<Unsigned>& newIndexPositions){
    const Unsigned orderA = A.Order();
    const Unsigned orderB = B.Order();
    const Unsigned nIndicesIntroduce = newIndexPositions.size();

    if(orderB != orderA + nIndicesIntroduce)
        LogicError("CheckIntroduceUnitIndicesRedist: Object being redistributed must be of correct order");

    Unsigned order = A.Order();
    IndexArray AIndices = A.Indices();
    IndexArray BIndices = B.Indices();

    std::vector<bool> foundIndices(order, 0);
    for(Unsigned i = 0; i < order; i++){
        if(std::find(BIndices.begin(), BIndices.end(), AIndices[i]) != BIndices.end())
            foundIndices[i] = true;
    }

    if(AnyFalseElem(foundIndices)){
        LogicError("CheckAllToAllDoubleIndexRedist: Output object must represent same indices as input object");
    }
    return 1;
}

#define PROTO(T) \
        template Int CheckPermutationRedist(const DistTensor<T>& B, const DistTensor<T>& A, const Index permuteIndex); \
        template Int CheckPartialReduceScatterRedist(const DistTensor<T>& B, const DistTensor<T>& A, const Index reduceScatterIndex); \
		template Int CheckReduceScatterRedist(const DistTensor<T>& A, const DistTensor<T>& B, const Index reduceIndex, const Index scatterIndex); \
		template Int CheckAllGatherRedist(const DistTensor<T>& A, const DistTensor<T>& B, const Index allGatherIndex); \
		template Int CheckAllToAllDoubleIndexRedist(const DistTensor<T>& A, const DistTensor<T>& B, const std::pair<Index, Index>& a2aIndices, const std::pair<ModeArray, ModeArray >& a2aCommGroups); \
		template Int CheckLocalRedist(DistTensor<T>& B, const DistTensor<T>& A, const Index localIndex, const ModeArray& gridRedistModes); \
		template Int CheckRemoveUnitIndicesRedist(const DistTensor<T>& B, const DistTensor<T>& A, const IndexArray& unitIndices); \
		template Int CheckIntroduceUnitIndicesRedist(const DistTensor<T>& B, const DistTensor<T>& A, const std::vector<Unsigned>& newIndexPositions); \



PROTO(int)
PROTO(float)
PROTO(double)

} // namespace tmen
