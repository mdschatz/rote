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
Int CheckPermutationRedist(const DistTensor<T>& B, const DistTensor<T>& A, const Mode permuteMode){
    Unsigned i;
    const tmen::GridView gvA = A.GridView();

    const Unsigned AOrder = A.Order();
    const Unsigned BOrder = B.Order();

    //Test order retained
    if(BOrder != AOrder){
        LogicError("CheckPermutationRedist: Permutation retains the same order of objects");
    }

    //Test dimension has been resized correctly
    //NOTE: Uses fancy way of performing Ceil() on integer division
    if(B.Dimension(permuteMode) != A.Dimension(permuteMode))
        LogicError("CheckPartialReduceScatterRedist: Permutation retains the same dimension of indices");

    //Make sure all indices are distributed similarly
    for(i = 0; i < BOrder; i++){
        Mode mode = i;
        if(mode == permuteMode){
            if(!EqualUnderPermutation(B.ModeDist(mode), A.ModeDist(mode)))
                LogicError("CheckPermutationRedist: Distribution of permuted mode does not involve same modes of grid as input");
        }else{
            if(AnyElemwiseNotEqual(B.ModeDist(mode), A.ModeDist(mode)))
                LogicError("CheckPartialReduceScatterRedist: All modes must be distributed similarly");
        }
    }
    return 1;

}

//TODO: Properly Check indices and distributions match between input and output
//TODO: Make sure outgoing reduce Mode differs from incoming (partial reduction forms a new Mode)
template <typename T>
Int CheckPartialReduceScatterRedist(const DistTensor<T>& B, const DistTensor<T>& A, const Mode reduceScatterMode){
    Unsigned i;
    const tmen::GridView gvA = A.GridView();

    const Unsigned AOrder = A.Order();
    const Unsigned BOrder = B.Order();

    //Test order retained
    if(BOrder != AOrder){
        LogicError("CheckPartialReduceScatterRedist: Partial Reduction must retain mode being reduced");
    }

    //Test dimension has been resized correctly
    //NOTE: Uses fancy way of performing Ceil() on integer division
    if(B.Dimension(reduceScatterMode) != Max(1,MaxLength(A.Dimension(reduceScatterMode), gvA.Dimension(reduceScatterMode))))
        LogicError("CheckPartialReduceScatterRedist: Partial Reduction must reduce mode dimension by factor Dimension/Grid Dimension");

    //Make sure all indices are distributed similarly between input and output (excluding reduce+scatter indices)
    for(i = 0; i < BOrder; i++){
        Mode mode = i;
        if(AnyElemwiseNotEqual(B.ModeDist(mode), A.ModeDist(mode)))
            LogicError("CheckPartialReduceScatterRedist: All modes must be distributed similarly");
    }
    return 1;
}

//TODO: Properly Check indices and distributions match between input and output
template <typename T>
Int CheckReduceScatterRedist(const DistTensor<T>& B, const DistTensor<T>& A, const Mode reduceMode, const Mode scatterMode){
    Unsigned i;
    const tmen::GridView gvA = A.GridView();

	//Test elimination of mode
	const Unsigned AOrder = A.Order();
	const Unsigned BOrder = B.Order();

	//Check that redist modes are assigned properly on input and output
	ModeDistribution BScatterModeDist = B.ModeDist(scatterMode);
	ModeDistribution AReduceModeDist = A.ModeDist(reduceMode);
	ModeDistribution AScatterModeDist = A.ModeDist(scatterMode);

	//Test elimination of mode
	if(BOrder != AOrder - 1){
		LogicError("CheckReduceScatterRedist: Full Reduction requires elimination of mode being reduced");
	}

	//Test no wrapping of mode to reduce
	if(A.Dimension(reduceMode) > gvA.Dimension(reduceMode))
		LogicError("CheckReduceScatterRedist: Full Reduction requires global mode dimension <= gridView dimension");

	//Make sure all indices are distributed similarly between input and output (excluding reduce+scatter indices)
	for(i = 0; i < BOrder; i++){
	    Mode mode = i;
	    if(mode == scatterMode){
	        ModeDistribution check(BScatterModeDist.end() - AReduceModeDist.size(), BScatterModeDist.end());
            if(AnyElemwiseNotEqual(check, AReduceModeDist))
                LogicError("CheckReduceScatterRedist: Reduce mode distribution of A must be a suffix of Scatter mode distribution of B");
	    }
	    else{
	        if(AnyElemwiseNotEqual(B.ModeDist(mode), A.ModeDist(mode)))
	            LogicError("CheckReduceScatterRedist: All modes not involved in reduce-scatter must be distributed similarly");
	    }
	}

	return 1;
}

template<typename T>
Int CheckAllGatherRedist(const DistTensor<T>& B, const DistTensor<T>& A, const Mode allGatherMode, const ModeArray& redistModes){
    if(A.Order() != B.Order()){
        LogicError("CheckAllGatherRedist: Objects being redistributed must be of same order");
    }

    ModeDistribution allGatherDistA = A.ModeDist(allGatherMode);
    ModeDistribution allGatherDistB = B.ModeDist(allGatherMode);

    const ModeDistribution check = ConcatenateVectors(allGatherDistB, redistModes);
    if(AnyElemwiseNotEqual(check, allGatherDistA)){
        LogicError("CheckAllGatherRedist: [Output distribution ++ redistModes] does not match Input distribution");
    }

    return true;
}

template<typename T>
Int CheckAllGatherRedist(const DistTensor<T>& B, const DistTensor<T>& A, const Mode allGatherMode){
	if(A.Order() != B.Order()){
		LogicError("CheckAllGatherRedist: Objects being redistributed must be of same order");
	}

	ModeDistribution AAllGatherModeDist = A.ModeDist(allGatherMode);
	if(AAllGatherModeDist.size() != 0)
		LogicError("CheckAllGatherRedist: Allgather only redistributes to * (for now)");

	return true;
}

template <typename T>
Int CheckPartialReduceScatterRedist(const DistTensor<T>& B, const DistTensor<T>& A, const Mode rsMode, const ModeArray& rsGridModes){
	LogicError("CheckPartialReduceScatterRedist: Not implemented");
	//if(AnyElemwiseNotEqual(A.Indices(), B.Indices()))
	//	LogicError("CheckPartialReduceScatterRedist: Invalid redistribution request");
	return 1;
}


//TODO: Check that allToAllIndices and commGroups are valid
template <typename T>
Int CheckAllToAllDoubleModeRedist(const DistTensor<T>& B, const DistTensor<T>& A, const std::pair<Mode, Mode>& allToAllModes, const std::pair<ModeArray, ModeArray >& a2aCommGroups){
    if(A.Order() != B.Order())
        LogicError("CheckAllToAllDoubleModeRedist: Objects being redistributed must be of same order");
    Unsigned i;
    for(i = 0; i < A.Order(); i++){
        if(i != allToAllModes.first && i != allToAllModes.second){
            if(B.ModeDist(i) != A.ModeDist(i))
                LogicError("CheckAlLToAllDoubleModeRedist: Non-redist modes must have same distribution");
        }
    }
    return 1;
}

template<typename T>
Int CheckLocalRedist(DistTensor<T>& B, const DistTensor<T>& A, const Mode localMode, const ModeArray& gridRedistModes){
    if(A.Order() != B.Order())
        LogicError("CheckLocalRedist: Objects being redistributed must be of same order");

    Unsigned i, j;
    TensorDistribution distA = A.TensorDist();
    TensorDistribution distB = B.TensorDist();
    ModeDistribution localModeDistA = A.ModeDist(localMode);
    ModeDistribution localModeDistB = B.ModeDist(localMode);

    if(localModeDistB.size() != localModeDistA.size() + gridRedistModes.size())
        LogicError("CheckLocalReist: Input object cannot be redistributed to output object");

    ModeArray check(localModeDistB);
    for(i = 0; i < localModeDistA.size(); i++)
        check[i] = localModeDistA[i];
    for(i = 0; i < gridRedistModes.size(); i++)
        check[localModeDistA.size() + i] = gridRedistModes[i];

    for(i = 0; i < check.size(); i++){
        if(check[i] != localModeDistB[i])
            LogicError("CheckLocalRedist: Output distribution cannot be formed from supplied parameters");
    }

    ModeArray boundModes;
    for(i = 0; i < distA.size(); i++){
        for(j = 0; j < distA[i].size(); j++){
            boundModes.push_back(distA[i][j]);
        }
    }

    for(i = 0; i < gridRedistModes.size(); i++)
        if(std::find(boundModes.begin(), boundModes.end(), gridRedistModes[i]) != boundModes.end())
            LogicError("CheckLocalRedist: Attempting to redistribute with already bound mode of the grid");

    return 1;
}

//TODO: Make sure these checks are correct (look at LDim, strides, distributions, etc).
template <typename T>
Int CheckRemoveUnitModesRedist(const DistTensor<T>& B, const DistTensor<T>& A, const ModeArray& unitModes){
    const Unsigned orderA = A.Order();
    const Unsigned orderB = B.Order();
    const Unsigned nModesRemove = unitModes.size();

    if(orderB != orderA - nModesRemove)
        LogicError("CheckRemoveUnitIndicesRedist: Object being redistributed must be of correct order");

    return 1;
}

template <typename T>
Int CheckIntroduceUnitModesRedist(const DistTensor<T>& B, const DistTensor<T>& A, const std::vector<Unsigned>& newModePositions){
    const Unsigned orderA = A.Order();
    const Unsigned orderB = B.Order();
    const Unsigned nModesIntroduce = newModePositions.size();

    if(orderB != orderA + nModesIntroduce)
        LogicError("CheckIntroduceUnitIndicesRedist: Object being redistributed must be of correct order");
    return 1;
}

#define PROTO(T) \
        template Int CheckPermutationRedist(const DistTensor<T>& B, const DistTensor<T>& A, const Mode permuteMode); \
        template Int CheckPartialReduceScatterRedist(const DistTensor<T>& B, const DistTensor<T>& A, const Mode reduceScatterMode); \
		template Int CheckReduceScatterRedist(const DistTensor<T>& B, const DistTensor<T>& A, const Mode reduceMode, const Mode scatterMode); \
		template Int CheckAllGatherRedist(const DistTensor<T>& B, const DistTensor<T>& A, const Mode allGatherMode, const ModeArray& redistModes); \
		template Int CheckAllGatherRedist(const DistTensor<T>& B, const DistTensor<T>& A, const Mode allGatherMode); \
		template Int CheckAllToAllDoubleModeRedist(const DistTensor<T>& B, const DistTensor<T>& A, const std::pair<Mode, Mode>& a2aModes, const std::pair<ModeArray, ModeArray >& a2aCommGroups); \
		template Int CheckLocalRedist(DistTensor<T>& B, const DistTensor<T>& A, const Mode localMode, const ModeArray& gridRedistModes); \
		template Int CheckRemoveUnitModesRedist(const DistTensor<T>& B, const DistTensor<T>& A, const ModeArray& unitIndices); \
		template Int CheckIntroduceUnitModesRedist(const DistTensor<T>& B, const DistTensor<T>& A, const std::vector<Unsigned>& newModePositions); \



PROTO(int)
PROTO(float)
PROTO(double)

} // namespace tmen
