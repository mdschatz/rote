/*
   Copyright (c) 2009-2013, Jack Poulson
                      2013, Jeff Hammond
                      2013, Jed Brown
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#include "rote.hpp"
#include <functional>
#include <numeric>
#include <algorithm>

namespace rote{

template<typename T>
void DistTensor<T>::CommRedistReduce(const TensorDistribution& finalDist, const TensorDistribution& startDist, RedistPlanInfo& redistData, std::vector<Redist>& intDists){
	Unsigned i;
	TensorDistribution intDist = startDist;

	ModeArray reduceModes = redistData.tenModesReduced;
	SortVector(reduceModes);
	ModeArray gridModesReduced;
	for(i = 0; i < reduceModes.size(); i++){
		Mode reduceMode = reduceModes[reduceModes.size() - 1 - i];
		gridModesReduced = ConcatenateVectors(gridModesReduced, intDist[reduceMode]);
		intDist.erase(intDist.begin() + reduceMode);
	}

//	std::cout << TensorDistToString(intDist) << "before adding reductions\n";
//	printf("ping1\n");
	ModeArray sinkTenModes = GetModeDistOfGridMode(gridModesReduced, finalDist);
//	printf("ping1\n");
//	PrintVector(sinkTenModes, "sinkTenModes");
	for(i = 0; i < sinkTenModes.size(); i++){
		Mode sinkTenMode = sinkTenModes[i];
		//Check that this mode didn't disappear
		//and place it in the final mode distribution
		//Otherwise, by default append to mode dist 0
		if(sinkTenMode != finalDist.size()){
			intDist[sinkTenMode].push_back(gridModesReduced[i]);
		}else{
			intDist[0].push_back(gridModesReduced[i]);
		}
	}
//	std::cout << TensorDistToString(intDist) << "after adding reductions\n";
//	printf("ping1\n");
	Redist newRedistInfo;
	newRedistInfo.redistType = RS;
	newRedistInfo.dist = intDist;
	newRedistInfo.modes = reduceModes;
	intDists.push_back(newRedistInfo);

	redistData.tenModesReduced.clear();

	CommRedist(finalDist, intDist, redistData, intDists);
}

template <typename T>
void DistTensor<T>::CommRedistAdd(const TensorDistribution& finalDist, const TensorDistribution& startDist, RedistPlanInfo& redistData, std::vector<Redist>& intDists){
	Unsigned i;

	//Get the prefix tensor distribution
	TensorDistribution intDist = startDist;
	for(i = 0; i < redistData.gridModesAppeared.size(); i++){
		Mode appearedMode = redistData.gridModesAppeared[i];
		intDist[redistData.gridModesAppearedSinks[i]].push_back(appearedMode);
	}

	//Add the intermediate distribution to the list
	Redist newRedistInfo;
	newRedistInfo.redistType = Local;
	newRedistInfo.dist = intDist;
	newRedistInfo.modes = redistData.gridModesMoved;
	intDists.push_back(newRedistInfo);

	//Update redistData
	redistData.gridModesAppeared.clear();
	redistData.gridModesAppearedSinks.clear();

	//Recur to get remaining intermediates
	CommRedist(finalDist, intDist, redistData, intDists);
}

template <typename T>
void DistTensor<T>::CommRedistRemove(const TensorDistribution& finalDist, const TensorDistribution& startDist, RedistPlanInfo& redistData, std::vector<Redist>& intDists){
	Unsigned i, j;
	Unsigned order = startDist.size() - 1;
	//Get the prefix tensor distribution
	TensorDistribution intDist = startDist;
	for(i = 0; i < redistData.gridModesRemoved.size(); i++){
		Mode removedMode = redistData.gridModesRemoved[i];
		for(j = 0; j < order; j++){
			ModeDistribution modeDist = intDist[j];
			modeDist.erase(std::remove(modeDist.begin(), modeDist.end(), removedMode), modeDist.end());
			intDist[j] = modeDist;
		}
	}

	//Now place all modes to be removed at the end
	TensorDistribution prepDist = intDist;
	for(i = 0; i < redistData.gridModesRemoved.size(); i++){
		Mode removedMode = redistData.gridModesRemoved[i];
		Mode srcMode = redistData.gridModesRemovedSrcs[i];
		prepDist[srcMode].push_back(removedMode);
	}
	//Add the intermediate distribution to the list (if needed)
	if(prepDist != intDist && prepDist != startDist){
		Redist prepRedistInfo;
		prepRedistInfo.redistType = Perm;
		prepRedistInfo.dist = prepDist;
		prepRedistInfo.modes = DefaultPermutation(this->Grid().Order());
		intDists.push_back(prepRedistInfo);
	}

	//Now remove the modes
	Redist newRedistInfo;
	newRedistInfo.redistType = AG;
	newRedistInfo.dist = intDist;
	newRedistInfo.modes = redistData.gridModesRemoved;
	intDists.push_back(newRedistInfo);

	//Update redistData
	redistData.gridModesRemoved.clear();
	redistData.gridModesRemovedSrcs.clear();

	//Recur to get remaining intermediates
	CommRedist(finalDist, intDist, redistData, intDists);
}

template <typename T>
void DistTensor<T>::CommRedistMove(const TensorDistribution& finalDist, const TensorDistribution& startDist, RedistPlanInfo& redistData, std::vector<Redist>& intDists){
	Unsigned i, j;

	//Pick the grid modes to move in this pass
	//Greedy approach
	ModeArray gridModesToMove;
	ModeArray gridModesToMoveSrcs;
	ModeArray gridModesToMoveSinks;
	ModeArray excludeModes;

	redistData.gridModesMovedSrcs = GetModeDistOfGridMode(redistData.gridModesMoved, startDist);
	redistData.gridModesMovedSinks = GetModeDistOfGridMode(redistData.gridModesMoved, finalDist);

	for(i = 0; i < redistData.gridModesMoved.size(); i++){
		if(!Contains(excludeModes, redistData.gridModesMovedSrcs[i]) &&
		   !Contains(excludeModes, redistData.gridModesMovedSinks[i])){
			gridModesToMove.push_back(redistData.gridModesMoved[i]);
			gridModesToMoveSrcs.push_back(redistData.gridModesMovedSrcs[i]);
			gridModesToMoveSinks.push_back(redistData.gridModesMovedSinks[i]);
			excludeModes.push_back(redistData.gridModesMovedSrcs[i]);
			excludeModes.push_back(redistData.gridModesMovedSinks[i]);
		}
	}

	//Before we move modes among mode dists, we have to "prep" the tensor dist
	//Create the "prep" distribution
	TensorDistribution prepDist = startDist;
	for(i = 0; i < gridModesToMove.size(); i++){
		Mode removedMode = gridModesToMove[i];
		ModeDistribution removeModeDist = prepDist[gridModesToMoveSrcs[i]];
		removeModeDist.erase(std::remove(removeModeDist.begin(), removeModeDist.end(), removedMode));
		removeModeDist.push_back(removedMode);
		prepDist[gridModesToMoveSrcs[i]] = removeModeDist;
	}

	//Create the "prep" redistribution
	if(prepDist != startDist){
		Redist prepRedistInfo;
		prepRedistInfo.redistType = Perm;
		prepRedistInfo.dist = prepDist;
		prepRedistInfo.modes = DefaultPermutation(this->Grid().Order());
		intDists.push_back(prepRedistInfo);
	}

	//"Move" the modes
	//1. Erase them from the distribution
	TensorDistribution intDist = prepDist;
	for(i = 0; i < gridModesToMove.size(); i++){
		intDist[gridModesToMoveSrcs[i]].pop_back();
	}

	//2. Move the modes where they need to go
	for(i = 0; i < gridModesToMove.size(); i++){
		intDist[gridModesToMoveSinks[i]].push_back(gridModesToMove[i]);
	}

	Redist moveRedistInfo;
	moveRedistInfo.redistType = A2A;
	moveRedistInfo.dist = intDist;
	moveRedistInfo.modes = gridModesToMove;
	intDists.push_back(moveRedistInfo);

	//Update redistData
	for(i = 0; i < gridModesToMove.size(); i++){
		Mode modeToFind = gridModesToMove[i];
		for(j = 0; j < redistData.gridModesMoved.size(); j++){
			if(redistData.gridModesMoved[j] == modeToFind){
				redistData.gridModesMoved.erase(redistData.gridModesMoved.begin() + j);
//				redistData.gridModesMovedSrcs.erase(redistData.gridModesMovedSrcs.begin() + j);
//				redistData.gridModesMovedSinks.erase(redistData.gridModesMovedSinks.begin() + j);
				break;
			}
		}
	}

	//Recur to get remaining intermediates
	CommRedist(finalDist, intDist, redistData, intDists);
}

template <typename T>
void DistTensor<T>::CommRedistP2P(const TensorDistribution& finalDist, const TensorDistribution& startDist, RedistPlanInfo& redistData, std::vector<Redist>& intDists){
	Unsigned i;
	Unsigned order = startDist.size() - 1;
	const rote::Grid& g = this->Grid();

	ModeArray movedModes = redistData.gridModesMoved;
//	ModeArray movedModesSrcs = redistData.gridModesMovedSrcs;
//	ModeArray movedModesSinks = redistData.gridModesMovedSinks;
	ModeArray movedModesSrcs = GetModeDistOfGridMode(redistData.gridModesMoved, startDist);
	ModeArray movedModesSinks = GetModeDistOfGridMode(redistData.gridModesMoved, finalDist);

//	PrintVector(movedModesSrcs, "srcs");
//	PrintVector(movedModesSinks, "sinks");
	ObjShape startShape(order, 1);
	Unsigned numMoved = redistData.gridModesMoved.size();

	for(i = 0; i < numMoved; i++){
		Mode movedMode = movedModes[i];
		Mode movedModeSrc = movedModesSrcs[i];
		startShape[movedModeSrc] *= g.Dimension(movedMode);
	}

	//Loop generating possible candidates
	Location startLoc(numMoved, 0);
    Location curLoc = startLoc;
    Location end(numMoved, 2);
    Unsigned ptr = 0;

    ObjShape curShape = startShape;
    Unsigned maxFound = 0;
    Location maxLoc = curLoc;
    bool done = numMoved == 0 || !ElemwiseLessThan(curLoc, end);

//    printf("ping1\n");
    while(!done){
    	curShape = startShape;
    	for(i = 0; i < numMoved; i++){
    		Mode movedMode = movedModes[i];
			curShape[movedModesSinks[i]] *= (curLoc[i] ? g.Dimension(movedMode) : 1);
			curShape[movedModesSrcs[i]] /= (curLoc[i] ? g.Dimension(movedMode) : 1);
    	}

        Unsigned curFound = sum(curLoc);
    	if(curFound > maxFound && !AnyElemwiseNotEqual(curShape, startShape)){
    		maxFound = curFound;
    		maxLoc = curLoc;
    	}
        //Update
        curLoc[ptr]++;

        while(ptr < numMoved && curLoc[ptr] >= end[ptr]){
            curLoc[ptr] = 0;

            ptr++;
            if(ptr >= numMoved){
                done = true;
                break;
            }else{
                curLoc[ptr]++;
            }
        }
        if(done)
            break;
        ptr = 0;
    }

//    printf("ping2\n");
    if(AnyElemwiseNotEqual(maxLoc, startLoc)){
		//Form the base distribution
		TensorDistribution intDist = startDist;
		for(i = 0; i < numMoved; i++){
			Unsigned index = numMoved - 1 - i;
			if(maxLoc[index]){
				//Erase from the source mode dist
				int foundPos = IndexOf(intDist[movedModesSrcs[index]], movedModes[index]);
				intDist[movedModesSrcs[index]].erase(intDist[movedModesSrcs[index]].begin() + foundPos);
				intDist[movedModesSinks[index]].push_back(movedModes[index]);

				//Update GenRedistData object
				redistData.gridModesMoved.erase(redistData.gridModesMoved.begin() + index);
//				redistData.gridModesMovedSrcs.erase(redistData.gridModesMovedSrcs.begin() + index);
//				redistData.gridModesMovedSinks.erase(redistData.gridModesMovedSinks.begin() + index);
			}
		}

		Redist newRedistInfo;
		newRedistInfo.dist = intDist;
		newRedistInfo.redistType = Perm;
		newRedistInfo.modes = DefaultPermutation(this->Grid().Order());

		intDists.push_back(newRedistInfo);

		//Recur
		CommRedist(finalDist, intDist, redistData, intDists);
    }
    else{
		CommRedistMove(finalDist, startDist, redistData, intDists);
    }
}

template <typename T>
void DistTensor<T>::CommRedist(const TensorDistribution& finalDist, const TensorDistribution& startDist, RedistPlanInfo& redistData, std::vector<Redist>& intDists){
	//Add any grid modes to reduce communicated data
	if(finalDist == startDist){
		return;
	}
	else if(redistData.gridModesAppeared.size() > 0){
//		if(commRank == 0)
//			std::cout << "Add: " << TensorDistToString(finalDist) << " <-- " << TensorDistToString(startDist) << std::endl;
		CommRedistAdd(finalDist, startDist, redistData, intDists);
	}
	else if(redistData.tenModesReduced.size() > 0){
//		if(commRank == 0)
//			std::cout << "Reduce: " << TensorDistToString(finalDist) << " <-- " << TensorDistToString(startDist) << std::endl;
		CommRedistReduce(finalDist, startDist, redistData, intDists);
	}
	//Move any grid modes with Perm + A2A
	//Try the Perm first
	else if(redistData.gridModesMoved.size() > 0){
//		if(commRank == 0)
//			std::cout << "P2P: " << TensorDistToString(finalDist) << " <-- " << TensorDistToString(startDist) << std::endl;
		CommRedistP2P(finalDist, startDist, redistData, intDists);
	}
	//Finally, perform the necessary replication
	else if(redistData.gridModesRemoved.size() > 0){
//		if(commRank == 0)
//			std::cout << "Rmv: " << TensorDistToString(finalDist) << " <-- " << TensorDistToString(startDist) << std::endl;
		CommRedistRemove(finalDist, startDist, redistData, intDists);
	}
	//All thats left is a shuffle
	else{
//		if(commRank == 0)
//			std::cout << "Final: " << TensorDistToString(finalDist) << " <-- " << TensorDistToString(startDist) << std::endl;
		//Add the final redistribution
		//NOTE: modes is clearly a hack until interface changes
		Redist redistInfo;
		redistInfo.redistType = Perm;
		redistInfo.dist = this->TensorDist();
		redistInfo.modes = DefaultPermutation(this->Grid().Order());
		intDists.push_back(redistInfo);
	}
}

#define FULL(T) \
    template class DistTensor<T>;

FULL(Int)
#ifndef DISABLE_FLOAT
FULL(float)
#endif
FULL(double)

#ifndef DISABLE_COMPLEX
#ifndef DISABLE_FLOAT
FULL(std::complex<float>)
#endif
FULL(std::complex<double>)
#endif

} //namespace rote
