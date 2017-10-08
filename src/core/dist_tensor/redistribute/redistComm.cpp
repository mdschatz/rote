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
	TensorDistribution intDist = startDist;

	ModeArray reduceModes = redistData.tenModesReduced;
	SortVector(reduceModes);

	ModeDistribution gridModesReduced = intDist.Filter(reduceModes).UsedModes();
	intDist.RemoveUnitModeDists(reduceModes);

	ModeDistribution gridModesReducedAndDisappeared = gridModesReduced - finalDist.UsedModes();
	ModeDistribution gridModesReducedAndKept = gridModesReduced - gridModesReducedAndDisappeared;
	TensorDistribution finalReduceModesDist = finalDist.TensorDistForGridModes(gridModesReducedAndKept);
	std::vector<ModeDistribution> tenDistReducedAndDisappearedVals(intDist.size());
	TensorDistribution tenDistReducedAndDisappeared(tenDistReducedAndDisappearedVals);

	intDist += finalReduceModesDist;
	intDist += tenDistReducedAndDisappeared;

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
	//Get the prefix tensor distribution
	TensorDistribution intDist = startDist;
	TensorDistribution appearedTenDist = finalDist.TensorDistForGridModes(redistData.gridModesAppeared);
	intDist += appearedTenDist;

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
	//Get the prefix tensor distribution
	TensorDistribution intDist = startDist;
	TensorDistribution removedModesTenDist = startDist.TensorDistForGridModes(redistData.gridModesRemoved);
	intDist -= removedModesTenDist;

	TensorDistribution prepDist = intDist + removedModesTenDist;

	//Add the intermediate distribution to the list (if needed)
	if(prepDist != intDist && prepDist != startDist){
		Redist prepRedistInfo;
		prepRedistInfo.redistType = Perm;
		prepRedistInfo.dist = prepDist;
		prepRedistInfo.modes = OrderedModes(this->Grid().Order());
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
	Unsigned i;
	//Pick the grid modes to move in this pass
	//Greedy approach

	ModeDistribution modesToMove;
	ModeDistribution modesToMoveSrcs;
	ModeDistribution modesToMoveSinks;
	ModeArray possModes = redistData.gridModesMoved;
	ModeArray srcModes = redistData.gridModesMovedSrcs;
	ModeArray sinkModes = redistData.gridModesMovedSinks;

	while(possModes.size() > 0){
		Unsigned index = possModes.size() - 1;
		Mode possMode     = possModes[index];
		Mode possModeSrc  = srcModes[index];
		Mode possModeSink = sinkModes[index];
		if(!modesToMoveSrcs.Contains(possModeSink) && !modesToMoveSinks.Contains(possModeSrc)){
			modesToMove += possMode;
			modesToMoveSrcs += possModeSrc;
			modesToMoveSinks += possModeSink;
		}
		possModes.pop_back();
		srcModes.pop_back();
		sinkModes.pop_back();
	}

	for(i = 0; i < possModes.size(); i++){
		// std::cout << "attempting to remove gMode: " << modesToMove[i] << std::endl;
		// std::cout << "attempting to remove gModeSrc: " << modesToMoveSrcs[i] << std::endl;
		// std::cout << "attempting to remove gModeSink: " << modesToMoveSinks[i] << std::endl;
		redistData.gridModesMoved.erase(std::find(redistData.gridModesMoved.begin(), redistData.gridModesMoved.end(), modesToMove[i]));
		redistData.gridModesMovedSrcs.erase(std::find(redistData.gridModesMovedSrcs.begin(), redistData.gridModesMovedSrcs.end(), modesToMoveSrcs[i]));
		redistData.gridModesMovedSinks.erase(std::find(redistData.gridModesMovedSinks.begin(), redistData.gridModesMovedSinks.end(), modesToMoveSinks[i]));
	}

	TensorDistribution gridModesMovedStartDist = startDist.TensorDistForGridModes(modesToMove);
	TensorDistribution gridModesMovedFinalDist = finalDist.TensorDistForGridModes(modesToMove);

	TensorDistribution prepDist = startDist - gridModesMovedStartDist;
	prepDist += gridModesMovedStartDist;

	//Create the "prep" redistribution
	if(prepDist != startDist){
		Redist prepRedistInfo;
		prepRedistInfo.redistType = Perm;
		prepRedistInfo.dist = prepDist;
		prepRedistInfo.modes = OrderedModes(this->Grid().Order());
		intDists.push_back(prepRedistInfo);
	}

	TensorDistribution intDist = prepDist - gridModesMovedStartDist;
	intDist += gridModesMovedFinalDist;

	Redist moveRedistInfo;
	moveRedistInfo.redistType = A2A;
	moveRedistInfo.dist = intDist;
	moveRedistInfo.modes = OrderedModes(this->Grid().Order());
	intDists.push_back(moveRedistInfo);

	//Recur to get remaining intermediates
	CommRedist(finalDist, intDist, redistData, intDists);
}

template <typename T>
void DistTensor<T>::CommRedistP2P(const TensorDistribution& finalDist, const TensorDistribution& startDist, RedistPlanInfo& redistData, std::vector<Redist>& intDists){
	Unsigned i;
	Unsigned order = startDist.size() - 1;
	const rote::Grid& g = this->Grid();

	ModeArray movedModes = redistData.gridModesMoved;
	ModeArray movedModesSrcs = startDist.TensorModesForGridModes(redistData.gridModesMoved);
	ModeArray movedModesSinks = finalDist.TensorModesForGridModes(redistData.gridModesMoved);

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

    if(AnyElemwiseNotEqual(maxLoc, startLoc)){
		//Form the base distribution
		TensorDistribution intDist = startDist;
		ModeDistribution movedModesDist;
		for(i = 0; i < numMoved; i++){
			Unsigned index = numMoved - 1 - i;
			if(maxLoc[index]){
				movedModesDist += movedModes[index];
			}
		}
		TensorDistribution movedModesStartDist = startDist.TensorDistForGridModes(movedModesDist);
		TensorDistribution movedModesFinalDist = finalDist.TensorDistForGridModes(movedModesDist);
		intDist -= movedModesStartDist;
		intDist += movedModesFinalDist;
		//NOTE: REMOVE THIS TWO-LINE HACK (STILL NEED THE REDISTDATA UPDATE)
		ModeDistribution gridModesMoved(redistData.gridModesMoved);
		for(i = movedModesDist.Entries().size() - 1; i < movedModesDist.Entries().size(); i--) {
			Unsigned index = IndexOf(redistData.gridModesMoved, movedModesDist[i]);
			redistData.gridModesMoved.erase(redistData.gridModesMoved.begin() + index);
			redistData.gridModesMovedSrcs.erase(redistData.gridModesMovedSrcs.begin() + index);
			redistData.gridModesMovedSinks.erase(redistData.gridModesMovedSinks.begin() + index);
		}
		// gridModesMoved -= movedModesDist;
		// redistData.gridModesMoved = gridModesMoved.Entries();


		Redist newRedistInfo;
		newRedistInfo.dist = intDist;
		newRedistInfo.redistType = Perm;
		newRedistInfo.modes = OrderedModes(this->Grid().Order());

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
		// if(commRank == 0)
			// std::cout << "Add: " << TensorDistToString(finalDist) << " <-- " << TensorDistToString(startDist) << std::endl;
		CommRedistAdd(finalDist, startDist, redistData, intDists);
	}
	else if(redistData.tenModesReduced.size() > 0){
		// if(commRank == 0)
			// std::cout << "Reduce: " << TensorDistToString(finalDist) << " <-- " << TensorDistToString(startDist) << std::endl;
		CommRedistReduce(finalDist, startDist, redistData, intDists);
	}
	//Move any grid modes with Perm + A2A
	//Try the Perm first
	else if(redistData.gridModesMoved.size() > 0){
		// if(commRank == 0)
			// std::cout << "P2P: " << TensorDistToString(finalDist) << " <-- " << TensorDistToString(startDist) << std::endl;
		CommRedistP2P(finalDist, startDist, redistData, intDists);
	}
	//Finally, perform the necessary replication
	else if(redistData.gridModesRemoved.size() > 0){
		// if(commRank == 0)
			// std::cout << "Rmv: " << TensorDistToString(finalDist) << " <-- " << TensorDistToString(startDist) << std::endl;
		CommRedistRemove(finalDist, startDist, redistData, intDists);
	}
	//All thats left is a shuffle
	else{
		// printf("startDist: %s\n", TensorDistToString(startDist).c_str());
		// printf("endDist: %s\n", TensorDistToString(finalDist).c_str());
		// PrintVector(redistData.gridModesMoved);
//		printf("failure\n");
//		if(commRank == 0)
//			std::cout << "Final: " << TensorDistToString(finalDist) << " <-- " << TensorDistToString(startDist) << std::endl;
		//Add the final redistribution
		//NOTE: modes is clearly a hack until interface changes
		Redist redistInfo;
		redistInfo.redistType = Perm;
		redistInfo.dist = this->TensorDist();
		redistInfo.modes = OrderedModes(this->Grid().Order());
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
