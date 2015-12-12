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
#include <algorithm>

namespace rote{



////////////////////////////////
// Workhorse interface
////////////////////////////////

template<typename T>
RedistPlanInfo DistTensor<T>::CreateGenRedistData(const TensorDistribution& tenDistA, const TensorDistribution& tenDistB, const ModeArray& reduceModes){
//	 Unsigned commRank = mpi::CommRank(MPI_COMM_WORLD);
	 Unsigned i, j;
	 RedistPlanInfo redistData;

	 //Determine tensor modes that are reduced
//	 redistData.gridModesReduced = GetBoundGridModes(tenDistA, reduceModes);
	 redistData.tenModesReduced = reduceModes;
	 //gridModesA = DiffVector(gridModesA, redistData.gridModesReduced);

	 //Determine grid modes that are removed/added
	 ModeArray gridModesA = GetBoundGridModes(tenDistA);
	 ModeArray gridModesB = GetBoundGridModes(tenDistB);

	 redistData.gridModesRemoved = DiffVector(gridModesA, gridModesB);
	 redistData.gridModesRemovedSrcs = GetModeDistOfGridMode(redistData.gridModesRemoved, tenDistA);

	 redistData.gridModesAppeared = DiffVector(gridModesB, gridModesA);
	 redistData.gridModesAppearedSinks = GetModeDistOfGridMode(redistData.gridModesAppeared, tenDistB);

	 //Determine grid modes that moved mode distributions
	 //Determine their Src tensor mode dist along the way
	 //Note: Adjusting for any reductions that may occur (we assume reductions are performed before this step
	 ModeArray reducedGridModes;
	 for(i = 0; i < reduceModes.size(); i++){
		 Mode reduceMode = reduceModes[i];
		 reducedGridModes = ConcatenateVectors(reducedGridModes, tenDistA[reduceMode]);
	 }

	 ModeArray nonReducedGridModes = DiffVector(gridModesA, reducedGridModes);
	 nonReducedGridModes = DiffVector(nonReducedGridModes, redistData.gridModesRemoved);
	 ModeArray finalModeMap(tenDistA.size());
	 for(i = 0; i < tenDistA.size(); i++){
		 if(Contains(reduceModes, i)){
			 finalModeMap[i] = tenDistA.size();
		 }else{
			 for(j = 0; j < reduceModes.size(); j++){
				 if(reduceModes[j] > i){
					 finalModeMap[i] = i - j;
					 break;
				 }
			 }
		 }
	 }
	 ModeArray srcs = GetModeDistOfGridMode(nonReducedGridModes, tenDistA);
	 ModeArray sinks = GetModeDistOfGridMode(nonReducedGridModes, tenDistB);

	 for(i = 0; i < nonReducedGridModes.size(); i++){
		 Mode possMovedMode = nonReducedGridModes[i];
		 if(finalModeMap[srcs[i]] != sinks[i])
			 redistData.gridModesMoved.push_back(possMovedMode);
	 }

//	 redistData.gridModesMoved = DiffVector(gridModesA, redistData.gridModesAppeared);
//	 redistData.gridModesMoved = DiffVector(redistData.gridModesMoved, redistData.gridModesRemoved);

//	 redistData.gridModesMovedSrcs = GetModeDistOfGridMode(redistData.gridModesMoved, tenDistA);
//	 redistData.gridModesMovedSinks = GetModeDistOfGridMode(redistData.gridModesMoved, tenDistB);

//	 PrintVector(redistData.tenModesReduced, "reducedTenModes");
//	 PrintVector(redistData.gridModesRemoved, "removed");
//	 PrintVector(redistData.gridModesRemovedSrcs, "removedSrcs");
//	 PrintVector(redistData.gridModesAppeared, "appeared");
//	 PrintVector(redistData.gridModesAppearedSinks, "appearedSinks");
//	 PrintVector(redistData.gridModesRemoved, "removed");
//	 PrintVector(redistData.gridModesMoved, "moved");
//	 PrintVector(redistData.gridModesMovedSrcs, "movedSrcs");
//	 PrintVector(redistData.gridModesMovedSinks, "movedSinks");

	 return redistData;
}

template <typename T>
void DistTensor<T>::RedistFrom(const DistTensor<T>& A, const ModeArray& reduceModes, const T alpha, const T beta){
	int commRank = mpi::CommRank(MPI_COMM_WORLD);
    PROFILE_SECTION("Redist");
//    ResizeTo(A);

    const rote::Grid& g = this->Grid();
    Unsigned i;
    std::vector<Redist> intermediateDists;
    RedistPlanInfo redistData = CreateGenRedistData(A.TensorDist(), this->TensorDist(), reduceModes);

//    printf("making plan\n");
    CommRedist(this->TensorDist(), A.TensorDist(), redistData, intermediateDists);

//    if(commRank == 0){
//		printf("plan created\n");
//		std::cout << "start dist: " << TensorDistToString(A.TensorDist()) << std::endl;
//		for(i = 0; i < intermediateDists.size(); i++){
//			Redist redistInfo = intermediateDists[i];
//			std::cout << "int dist: (";
//			switch(redistInfo.redistType){
//			case AG: std::cout << "AG"; break;
//			case A2A: std::cout << "A2A"; break;
//			case Perm: std::cout << "P2P"; break;
//			case Local: std::cout << "Local"; break;
//			default: break;
//			}
//			std::cout << ") " << TensorDistToString(redistInfo.dist) << " ";
//			PrintVector(redistInfo.modes, "");
//		}
//		std::cout << "final dist: " << TensorDistToString(this->TensorDist()) << std::endl;
//    }

    DistTensor<T> tmp(A.TensorDist(), g);
    tmp.LockedAttach(A.Shape(), A.Alignments(), A.LockedBuffer(), A.LocalPermutation(), A.LocalStrides(), g);

    for(i = 0; i < intermediateDists.size() - 1; i++){
    	Redist intRedist = intermediateDists[i];
    	DistTensor<T> tmp2(intRedist.dist, g);

//    	if(commRank == 0){
//    		std::cout << "from: " << TensorDistToString(tmp.TensorDist()) << std::endl;
//    		std::cout << "to: " << TensorDistToString(tmp2.TensorDist()) << std::endl;
//    	}
    	switch(intRedist.redistType){
    	case AG: tmp2.AllGatherRedistFrom(tmp, intRedist.modes); break;
    	case A2A: tmp2.AllToAllRedistFrom(tmp, intRedist.modes); break;
    	case Perm: tmp2.PermutationRedistFrom(tmp, intRedist.modes); break;
    	case Local: tmp2.LocalRedistFrom(tmp); break;
    	case RS: tmp2.ReduceScatterRedistFrom(alpha, tmp, reduceModes); break;
    	default: break;
    	}
    	tmp.Empty();
    	tmp = tmp2;
    }

//    PrintData(tmp, "tmp last data");
//    Print(tmp, "tmp last");
//    PrintData(*this, "this last data");
//    Print(*this, "this last");
//    printf("beta is: %.3f\n", beta);
	Redist lastRedist = intermediateDists[intermediateDists.size() - 1];
	switch(lastRedist.redistType){
	case AG: AllGatherRedistFrom(tmp, lastRedist.modes, beta); break;
	case A2A: AllToAllRedistFrom(tmp, lastRedist.modes, beta); break;
	case Local: LocalRedistFrom(tmp); break;
	case Perm: PermutationRedistFrom(tmp, lastRedist.modes, beta); break;
	case RS: ReduceScatterUpdateRedistFrom(tmp, beta, reduceModes); break;
	default: break;
	}

//	Print(*this, "this after");
    PROFILE_STOP;
}

template <typename T>
void DistTensor<T>::RedistFrom(const DistTensor<T>& A){
	ModeArray reduceModes;
	RedistFrom(A, reduceModes, T(1), T(0));
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
