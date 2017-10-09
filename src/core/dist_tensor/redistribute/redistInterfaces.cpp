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
	 ModeDistribution gridModesReduced = tenDistA.Filter(reduceModes).UsedModes();
	 //gridModesA = DiffVector(gridModesA, redistData.gridModesReduced);

	 //Determine grid modes that are removed/added
	 ModeArray gridModesA = tenDistA.UsedModes().Entries();
	 ModeArray gridModesB = tenDistB.UsedModes().Entries();
	 ModeDistribution gridModesADist(gridModesA);
	 ModeDistribution gridModesBDist(gridModesB);

	 redistData.gridModesRemoved = DiffVector(gridModesA, gridModesB);
	 ModeDistribution gridModesRemovedDist(redistData.gridModesRemoved);
	 redistData.gridModesRemovedSrcs = tenDistA.TensorModesForGridModes(redistData.gridModesRemoved);

	 //By default, we put AR modes to ten mode 0
	 for(i = 0; i < redistData.gridModesRemoved.size(); i++){
		 Mode gridMode = redistData.gridModesRemoved[i];
		 if(gridModesReduced.Contains(gridMode)){
			 redistData.gridModesRemovedSrcs[i] = 0;
		 }
	 }

	 redistData.gridModesAppeared = DiffVector(gridModesB, gridModesA);
	 redistData.gridModesAppearedSinks = tenDistB.TensorModesForGridModes(redistData.gridModesAppeared);

	 //Determine grid modes that moved mode distributions
	 //Determine their Src tensor mode dist along the way
	 //Note: Adjusting for any reductions that may occur (we assume reductions are performed before this step
	 ModeDistribution reducedGridModes = gridModesReduced;

	 ModeDistribution nonReducedGridModes = gridModesADist - reducedGridModes;
	 nonReducedGridModes -= gridModesRemovedDist;
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

	 TensorDistribution srcTenDist = tenDistA.TensorDistForGridModes(nonReducedGridModes);
	 srcTenDist.RemoveUnitModeDists(reduceModes);
	 TensorDistribution sinkTenDist = tenDistB.TensorDistForGridModes(nonReducedGridModes);

	//  std::cout << "Planning srcTenDist: " << srcTenDist << std::endl;
	//  std::cout << "Planning sinkTenDist: " << sinkTenDist << std::endl;
	 ModeDistribution finalModesMoved;
	 for(i = 0; i < srcTenDist.size(); i++){
		 ModeDistribution diff = srcTenDist[i] - sinkTenDist[i];
		 finalModesMoved += diff;
	 }
	//  PrintVector(finalModesMoved.Entries(), "final modes moved");
	 redistData.gridModesMoved = finalModesMoved.Entries();
	 redistData.gridModesMovedSrcs.resize(redistData.gridModesMoved.size());
	 redistData.gridModesMovedSinks.resize(redistData.gridModesMoved.size());

	 redistData.gridModesMovedSrcs = srcTenDist.TensorModesForGridModes(redistData.gridModesMoved);
	 redistData.gridModesMovedSinks = sinkTenDist.TensorModesForGridModes(redistData.gridModesMoved);

	//  PrintRedistPlanInfo(redistData, "RedistInfo pre return", true);
	 return redistData;
}

template <typename T>
void DistTensor<T>::RedistFrom(const DistTensor<T>& A, const ModeArray& reduceModes, const T alpha, const T beta){
    PROFILE_SECTION("Redist");
//    ResizeTo(A);

    const rote::Grid& g = this->Grid();
    Unsigned i;
    std::vector<Redist> intermediateDists;
    RedistPlanInfo redistData = CreateGenRedistData(A.TensorDist(), this->TensorDist(), reduceModes);

    CommRedist(this->TensorDist(), A.TensorDist(), redistData, intermediateDists);

		// PrintRedistPlanInfo(redistData, "RedistInfo", true);
		// PrintRedistPlan(A.TensorDist(), intermediateDists, "RedistPlan: ");

    DistTensor<T> tmp(A.TensorDist(), g);
    tmp.LockedAttach(A.Shape(), A.Alignments(), A.LockedBuffer(), A.LocalPermutation(), A.LocalStrides(), g);

		if (intermediateDists.size() == 0) {
			ModeArray blank;
			this->PermutationRedistFrom(tmp, blank);
			return;
		}

    for(i = 0; i < intermediateDists.size() - 1; i++){
    	Redist intRedist = intermediateDists[i];
    	DistTensor<T> tmp2(intRedist.dist, g);

    	switch(intRedist.redistType){
    	case AG: tmp2.AllGatherRedistFrom(tmp, intRedist.modes); break;
    	case A2A: tmp2.AllToAllRedistFrom(tmp, intRedist.modes); break;
    	case Perm: tmp2.PermutationRedistFrom(tmp, intRedist.modes); break;
    	case Local: tmp2.LocalRedistFrom(tmp); break;
//    	case AR: tmp2.AllReduceRedistFrom(tmp, reduceModes); break;
    	case RS: tmp2.ReduceScatterRedistFrom(alpha, tmp, reduceModes); break;
    	default: break;
    	}
    	tmp.Empty();
    	tmp = tmp2;
    }

	Redist lastRedist = intermediateDists[intermediateDists.size() - 1];
	switch(lastRedist.redistType){
	case AG: AllGatherRedistFrom(tmp, lastRedist.modes, beta); break;
	case A2A: AllToAllRedistFrom(tmp, lastRedist.modes, beta); break;
	case Local: LocalRedistFrom(tmp); break;
	case Perm: PermutationRedistFrom(tmp, lastRedist.modes, beta); break;
//	case AR: tmp2.AllReduceUpdateRedistFrom(tmp, beta, reduceModes); break;
	case RS: ReduceScatterUpdateRedistFrom(tmp, beta, reduceModes); break;
	default: break;
	}

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
