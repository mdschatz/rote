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
#include <functional>
#include <numeric>
#include <algorithm>

namespace tmen{

template <typename T>
void DistTensor<T>::CommRedistAdd(const TensorDistribution& finalDist, const TensorDistribution& startDist, GenRedistData& redistData, std::vector<RedistInfo>& intDists){
	Unsigned i, j;
	int commRank = mpi::CommRank(MPI_COMM_WORLD);
	Unsigned order = startDist.size() - 1;
	//Get the prefix tensor distribution
	TensorDistribution intDist = startDist;
	for(i = 0; i < redistData.gridModesAppeared.size(); i++){
		Mode appearedMode = redistData.gridModesAppeared[i];
		intDist[redistData.gridModesAppearedSinks[i]].push_back(appearedMode);
	}

	//Add the intermediate distribution to the list
	RedistInfo newRedistInfo;
	newRedistInfo.redistType = Local;
	newRedistInfo.dist = intDist;
	newRedistInfo.modes = redistData.gridModesMoved;
	intDists.push_back(newRedistInfo);

	//Update redistData
	redistData.gridModesAppeared.clear();
	redistData.gridModesAppearedSinks.clear();

	//Recur to get remaining intermediates
	if(commRank == 0)
		std::cout << "  Rec: " << TensorDistToString(finalDist) << " <-- " << TensorDistToString(intDist) << std::endl;
	CommRedist(finalDist, intDist, redistData, intDists);
}

template <typename T>
void DistTensor<T>::CommRedistRemove(const TensorDistribution& finalDist, const TensorDistribution& startDist, GenRedistData& redistData, std::vector<RedistInfo>& intDists){
	Unsigned i, j;
	Unsigned order = startDist.size() - 1;
	int commRank = mpi::CommRank(MPI_COMM_WORLD);
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
	for(i = 0; i < redistData.gridModesRemoved.size(); i++){
		Mode removedMode = redistData.gridModesRemoved[i];
		Mode srcMode = redistData.gridModesRemovedSrcs[i];
		intDist[srcMode].push_back(removedMode);
	}
	//Add the intermediate distribution to the list
	RedistInfo newRedistInfo;
	newRedistInfo.redistType = Perm;
	newRedistInfo.dist = intDist;
	newRedistInfo.modes = redistData.gridModesMoved;
	intDists.push_back(newRedistInfo);

	//Update redistData
	redistData.gridModesRemoved.clear();
	redistData.gridModesRemovedSrcs.clear();

	//Recur to get remaining intermediates
	if(commRank == 0)
		std::cout << "  Rec: " << TensorDistToString(finalDist) << " <-- " << TensorDistToString(intDist) << std::endl;
	CommRedist(finalDist, intDist, redistData, intDists);
}

template <typename T>
void DistTensor<T>::CommRedistMove(const TensorDistribution& finalDist, const TensorDistribution& startDist, GenRedistData& redistData, std::vector<RedistInfo>& intDists){
	Unsigned i, j;
	int commRank = mpi::CommRank(MPI_COMM_WORLD);
	Unsigned order = startDist.size() - 1;

	//Pick the grid modes to move in this pass
	//Greedy approach
	ModeArray gridModesToMove;
	ModeArray gridModesToMoveSrcs;
	ModeArray gridModesToMoveSinks;
	ModeArray excludeModes;

//	PrintVector(redistData.gridModesMoved, "gridModesMoved");
//	PrintVector(redistData.gridModesMovedSrcs, "gridModesMovedSrcs");
//	PrintVector(redistData.gridModesMovedSinks, "gridModesMovedSinks");

	for(i = 0; i < redistData.gridModesMoved.size(); i++){
		if(std::find(excludeModes.begin(), excludeModes.end(), redistData.gridModesMovedSrcs[i]) == excludeModes.end() &&
		   std::find(excludeModes.begin(), excludeModes.end(), redistData.gridModesMovedSinks[i]) == excludeModes.end()){
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

//	std::cout << "prep dist: " << TensorDistToString(prepDist) << std::endl;
	//Create the "prep" redistribution
	if(prepDist != startDist){
		RedistInfo prepRedistInfo;
		prepRedistInfo.redistType = Perm;
		prepRedistInfo.dist = prepDist;
		prepRedistInfo.modes = redistData.gridModesMoved;
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

	RedistInfo moveRedistInfo;
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
				redistData.gridModesMovedSrcs.erase(redistData.gridModesMovedSrcs.begin() + j);
				redistData.gridModesMovedSinks.erase(redistData.gridModesMovedSinks.begin() + j);
				break;
			}
		}
	}

	//Recur to get remaining intermediates
	if(commRank == 0)
		std::cout << "  Rec: " << TensorDistToString(finalDist) << " <-- " << TensorDistToString(intDist) << std::endl;
	CommRedist(finalDist, intDist, redistData, intDists);
}

template <typename T>
void DistTensor<T>::CommRedistP2P(const TensorDistribution& finalDist, const TensorDistribution& startDist, GenRedistData& redistData, std::vector<RedistInfo>& intDists){
	Unsigned i, j;
	Unsigned order = startDist.size() - 1;
	int commRank = mpi::CommRank(MPI_COMM_WORLD);
	const tmen::Grid& g = Grid();

//	printf("starting\n");
	ModeArray movedModes = redistData.gridModesMoved;
	ModeArray movedModesSrcs = redistData.gridModesMovedSrcs;
	ModeArray movedModesSinks = redistData.gridModesMovedSinks;

	ObjShape startShape(order, 1);
	Unsigned numMoved = redistData.gridModesMoved.size();

	for(i = 0; i < numMoved; i++){
		Mode movedMode = movedModes[i];
		Mode movedModeSrc = movedModesSrcs[i];
		startShape[movedModeSrc] *= g.Dimension(movedMode);
	}

//	std::cout << "P2P start dist: " << TensorDistToString(startDist) << std::endl;
//	std::cout << "P2P final dist: " << TensorDistToString(finalDist) << std::endl;

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
//    	if(commRank == 0){
//			printf("next iter:\n");
//			PrintVector(movedModes, "movedModes");
//			PrintVector(movedModesSrcs, "movedModesSrcs");
//			PrintVector(movedModesSinks, "movedModesSinks");
//			PrintVector(curLoc, "curLoc");
//			PrintVector(curShape, "curShape");
//			PrintVector(startShape, "startShape");
//    	}

    	curShape = startShape;
    	for(i = 0; i < numMoved; i++){
    		Mode movedMode = movedModes[i];
			curShape[movedModesSinks[i]] *= (curLoc[i] ? g.Dimension(movedMode) : 1);
			curShape[movedModesSrcs[i]] /= (curLoc[i] ? g.Dimension(movedMode) : 1);
    	}

        Unsigned curFound = sum(curLoc);
    	if(curFound > maxFound && !AnyElemwiseNotEqual(curShape, startShape)){
//    		if(commRank == 0)
//    			printf("gotcha\n");
    		maxFound = curFound;
    		maxLoc = curLoc;
    	}
//    	if(commRank == 0)
//    		printf("\n");
        //Update
        curLoc[ptr]++;
//        curShape[movedModesSinks[ptr]] *= g.Dimension(movedModes[ptr]);
//        curShape[movedModesSrcs[ptr]]  /= g.Dimension(movedModes[ptr]);


        while(ptr < numMoved && curLoc[ptr] >= end[ptr]){
            curLoc[ptr] = 0;
//            curShape[movedModesSinks[ptr]] /= g.Dimension(movedModes[ptr]);
//			curShape[movedModesSrcs[ptr]]  *= g.Dimension(movedModes[ptr]);

            ptr++;
            if(ptr >= numMoved){
                done = true;
                break;
            }else{
                curLoc[ptr]++;
//                curShape[movedModesSinks[ptr]] *= g.Dimension(movedModes[ptr]);
//    			curShape[movedModesSrcs[ptr]]  /= g.Dimension(movedModes[ptr]);
            }
        }
        if(done)
            break;
        ptr = 0;
    }
//    PrintVector(movedModes, "modes moved");
//    PrintVector(maxLoc, "max matching");

    if(AnyElemwiseNotEqual(maxLoc, startLoc)){
		//Form the base distribution
		TensorDistribution intDist = startDist;
		for(i = 0; i < numMoved; i++){
			Unsigned index = numMoved - 1 - i;
			if(maxLoc[index]){
				//Erase from the source mode dist
				ModeDistribution::iterator foundPos = std::find(intDist[movedModesSrcs[index]].begin(), intDist[movedModesSrcs[index]].end(), movedModes[index]);
				intDist[movedModesSrcs[index]].erase(foundPos);
				intDist[movedModesSinks[index]].push_back(movedModes[index]);

				//Update GenRedistData object
				redistData.gridModesMoved.erase(redistData.gridModesMoved.begin() + index);
				redistData.gridModesMovedSrcs.erase(redistData.gridModesMovedSrcs.begin() + index);
				redistData.gridModesMovedSinks.erase(redistData.gridModesMovedSinks.begin() + index);
			}
		}
//		std::cout << "best dist: " << TensorDistToString(intDist) << std::endl;

		RedistInfo newRedistInfo;
		newRedistInfo.dist = intDist;
		newRedistInfo.redistType = Perm;
		newRedistInfo.modes = DefaultPermutation(order);

		intDists.push_back(newRedistInfo);

		//Recur
		if(commRank == 0)
		std::cout << "  Rec: " << TensorDistToString(finalDist) << " <-- " << TensorDistToString(intDist) << std::endl;
		CommRedist(finalDist, intDist, redistData, intDists);
    }
    else{
		if(commRank == 0)
		std::cout << "  P2M: " << TensorDistToString(finalDist) << " <-- " << TensorDistToString(startDist) << std::endl;
		CommRedistMove(finalDist, startDist, redistData, intDists);
    }
}

template <typename T>
void DistTensor<T>::CommRedist(const TensorDistribution& finalDist, const TensorDistribution& startDist, GenRedistData& redistData, std::vector<RedistInfo>& intDists){
	int commRank = mpi::CommRank(MPI_COMM_WORLD);
	//Add any grid modes to reduce communicated data
	if(finalDist == startDist){
		return;
	}
	else if(redistData.gridModesAppeared.size() > 0){
		if(commRank == 0)
		std::cout << "Add: " << TensorDistToString(finalDist) << " <-- " << TensorDistToString(startDist) << std::endl;
		CommRedistAdd(finalDist, startDist, redistData, intDists);
	}
	//Move any grid modes with Perm + A2A
	//Try the Perm first
	else if(redistData.gridModesMoved.size() > 0){
		if(commRank == 0)
		std::cout << "P2P: " << TensorDistToString(finalDist) << " <-- " << TensorDistToString(startDist) << std::endl;
		CommRedistP2P(finalDist, startDist, redistData, intDists);
	}
	//Finally, perform the necessary replication
	else if(redistData.gridModesRemoved.size() > 0){
		if(commRank == 0)
		std::cout << "Rmv: " << TensorDistToString(finalDist) << " <-- " << TensorDistToString(startDist) << std::endl;
		CommRedistRemove(finalDist, startDist, redistData, intDists);
	}
	//All thats left is a shuffle
	else{
		if(commRank == 0)
		std::cout << "Final: " << TensorDistToString(finalDist) << " <-- " << TensorDistToString(startDist) << std::endl;
		//Add the final redistribution
		//NOTE: modes is clearly a hack until interface changes
		RedistInfo redistInfo;
		redistInfo.redistType = Perm;
		redistInfo.dist = TensorDist();
		redistInfo.modes = DefaultPermutation(Grid().Order());
		intDists.push_back(redistInfo);
	}
}

#define PROTO(T) template class DistTensor<T>
#define COPY(T) \
  template DistTensor<T>::DistTensor( const DistTensor<T>& A )
#define FULL(T) \
  PROTO(T);


FULL(Int);
#ifndef DISABLE_FLOAT
FULL(float);
#endif
FULL(double);

#ifndef DISABLE_COMPLEX
#ifndef DISABLE_FLOAT
FULL(std::complex<float>);
#endif
FULL(std::complex<double>);
#endif

} //namespace tmen
