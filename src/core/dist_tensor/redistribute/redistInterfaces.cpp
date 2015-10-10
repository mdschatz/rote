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



////////////////////////////////
// Workhorse interface
////////////////////////////////

template<typename T>
GenRedistData DistTensor<T>::CreateGenRedistData(const TensorDistribution& tenDistA, const TensorDistribution& tenDistB){
	 Unsigned i, j, k;
	 Unsigned commRank = mpi::CommRank(MPI_COMM_WORLD);
	 Unsigned order = tenDistA.size() - 1;
	 GenRedistData redistData;

	 //Determine the set of grid modes in A;
	 ModeArray gridModesA;
	 for(i = 0; i < order; i++){
		ModeDistribution modeDist = tenDistA[i];
		for(j = 0; j < modeDist.size(); j++){
			gridModesA.push_back(modeDist[j]);
		}
	 }
	 std::sort(gridModesA.begin(), gridModesA.end());
//	 PrintVector(gridModesA, "gridModesA");
//	 printf("gotcah1\n");

	 //Determine the set of grid modes in B;
	 ModeArray gridModesB;
	 for(i = 0; i < order; i++){
		ModeDistribution modeDist = tenDistB[i];
		for(j = 0; j < modeDist.size(); j++){
			gridModesB.push_back(modeDist[j]);
		}
	 }
	 std::sort(gridModesB.begin(), gridModesB.end());

//	 PrintVector(gridModesB, "gridModesB");

	 //Determine grid modes that are removed
	 std::set_difference(gridModesA.begin(), gridModesA.end(), gridModesB.begin(), gridModesB.end(), std::back_inserter(redistData.gridModesRemoved));
	 //Determine the associated Srcs
	 for(i = 0; i < redistData.gridModesRemoved.size(); i++){
		Mode removedMode = redistData.gridModesRemoved[i];
		for(j = 0; j < order; j++){
			ModeDistribution modeDistA = tenDistA[j];
			if(std::find(modeDistA.begin(), modeDistA.end(), removedMode) != modeDistA.end())
				redistData.gridModesRemovedSrcs.push_back(j);
		}
	 }

//	 PrintVector(redistData.gridModesRemoved, "removed");
//	 PrintVector(redistData.gridModesRemovedSrcs, "removedsrcs");

//	 printf("gotcah2\n");
	 //Determine grid modes that are added
	 std::set_difference(gridModesB.begin(), gridModesB.end(), gridModesA.begin(), gridModesA.end(), std::back_inserter(redistData.gridModesAppeared));
	 //Determine the associated Sinks
	 for(i = 0; i < redistData.gridModesAppeared.size(); i++){
		Mode removedMode = redistData.gridModesAppeared[i];
		for(j = 0; j < order; j++){
			ModeDistribution modeDistB = tenDistB[j];
			if(std::find(modeDistB.begin(), modeDistB.end(), removedMode) != modeDistB.end())
				redistData.gridModesAppearedSinks.push_back(j);
		}
	 }

//	 PrintVector(redistData.gridModesAppeared, "appeared");
//	 PrintVector(redistData.gridModesAppearedSinks, "appearedsinks");

	 //Determine grid modes that moved mode distributions
	 //Determine their Src tensor mode dist along the way
	 for(i = 0; i < order; i++){
		ModeDistribution modeDistA = tenDistA[i];
		for(j = 0; j < modeDistA.size(); j++){
			Mode mode = modeDistA[j];

			ModeDistribution modeDistB = tenDistB[i];
			if(std::find(modeDistB.begin(), modeDistB.end(), mode) == modeDistB.end()){
				redistData.gridModesMoved.push_back(mode);
				redistData.gridModesMovedSrcs.push_back(i);
				continue;
			}
		}
	 }
//	 PrintVector(redistData.gridModesMoved, "moved");
//	 PrintVector(redistData.gridModesMovedSrcs, "movedSrcs");
//	 printf("gotcah3\n");
	 //Determine the sinks for each grid mode that is moved
	 for(i = 0; i < redistData.gridModesMoved.size(); i++){
		Mode mode = redistData.gridModesMoved[i];
		for(j = 0; j < order; j++){
			ModeDistribution modeDistB = tenDistB[j];
			if(std::find(modeDistB.begin(), modeDistB.end(), mode) != modeDistB.end()){
				redistData.gridModesMovedSinks.push_back(j);
				break;
			}
		}
	 }
//	 PrintVector(redistData.gridModesMovedSinks, "movedSinks");

	 return redistData;
}

template <typename T>
void DistTensor<T>::RedistFrom(const DistTensor<T>& A){
	int commRank = mpi::CommRank(MPI_COMM_WORLD);
    PROFILE_SECTION("Redist");
    ResizeTo(A);

    Unsigned i;
    std::vector<RedistInfo> intermediateDists;
    GenRedistData redistData = CreateGenRedistData(A.TensorDist(), TensorDist());

    CommRedist(TensorDist(), A.TensorDist(), redistData, intermediateDists);

    if(commRank == 0){
		printf("gotcah\n");
		for(i = 0; i < intermediateDists.size(); i++){
			RedistInfo redistInfo = intermediateDists[i];
			std::cout << "int dist: (";
			switch(redistInfo.redistType){
			case AG: std::cout << "AG"; break;
			case A2A: std::cout << "A2A"; break;
			case Perm: std::cout << "P2P"; break;
			case Local: std::cout << "Local"; break;
			default: break;
			}
			std::cout << ") " << TensorDistToString(redistInfo.dist) << " ";
			PrintVector(redistInfo.modes, "");
		}
    }
    PROFILE_STOP;
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
