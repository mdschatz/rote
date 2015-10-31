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
GenRedistData DistTensor<T>::CreateGenRedistData(const TensorDistribution& tenDistA, const TensorDistribution& tenDistB, const ModeArray& reduceModes){
//	 Unsigned commRank = mpi::CommRank(MPI_COMM_WORLD);
	 GenRedistData redistData;

	 //Determine the set of grid modes in A;
	 ModeArray gridModesA = GetBoundGridModes(tenDistA);
//	 PrintVector(gridModesA, "gridModesA");
//	 printf("gotcah1\n");

	 //Determine the set of grid modes in B;
	 ModeArray gridModesB = GetBoundGridModes(tenDistB);

//	 PrintVector(gridModesB, "gridModesB");

	 //Determine grid modes that are reduced
//	 redistData.gridModesReduced = GetBoundGridModes(tenDistA, reduceModes);
	 redistData.tenModesReduced = reduceModes;
	 //gridModesA = DiffVector(gridModesA, redistData.gridModesReduced);

	 //Determine grid modes that are removed
	 redistData.gridModesRemoved = DiffVector(gridModesA, gridModesB);
	 redistData.gridModesRemovedSrcs = GetModeDistOfGridMode(redistData.gridModesRemoved, tenDistA);

//	 gridModesA = DiffVector(gridModesA, redistData.gridModesRemoved);

//	 PrintVector(redistData.gridModesRemoved, "removed");
//	 PrintVector(redistData.gridModesRemovedSrcs, "removedsrcs");

//	 printf("gotcah2\n");
	 //Determine grid modes that are added
	 redistData.gridModesAppeared = DiffVector(gridModesB, gridModesA);
	 redistData.gridModesAppearedSinks = GetModeDistOfGridMode(redistData.gridModesAppeared, tenDistB);

//	 PrintVector(redistData.gridModesAppeared, "appeared");
//	 PrintVector(redistData.gridModesAppearedSinks, "appearedsinks");

	 //Determine grid modes that moved mode distributions
	 //Determine their Src tensor mode dist along the way
	 redistData.gridModesMoved = DiffVector(gridModesA, redistData.gridModesAppeared);
	 redistData.gridModesMoved = DiffVector(redistData.gridModesMoved, redistData.gridModesRemoved);
//	 redistData.gridModesMoved = DiffVector(redistData.gridModesMoved, redistData.gridModesAppeared);

	 redistData.gridModesMovedSrcs = GetModeDistOfGridMode(redistData.gridModesMoved, tenDistA);
	 redistData.gridModesMovedSinks = GetModeDistOfGridMode(redistData.gridModesMoved, tenDistB);

	 return redistData;
}

template <typename T>
void DistTensor<T>::RedistFrom(const DistTensor<T>& A, const ModeArray& reduceModes){
//	int commRank = mpi::CommRank(MPI_COMM_WORLD);
    PROFILE_SECTION("Redist");
    ResizeTo(A);

    const tmen::Grid& g = Grid();
    Unsigned i;
    std::vector<RedistInfo> intermediateDists;
    GenRedistData redistData = CreateGenRedistData(A.TensorDist(), TensorDist(), reduceModes);

    CommRedist(TensorDist(), A.TensorDist(), redistData, intermediateDists);

//    if(commRank == 0){
//		printf("plan created\n");
//		std::cout << "start dist: " << TensorDistToString(A.TensorDist()) << std::endl;
//		for(i = 0; i < intermediateDists.size(); i++){
//			RedistInfo redistInfo = intermediateDists[i];
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
//		std::cout << "final dist: " << TensorDistToString(TensorDist()) << std::endl;
//    }

    DistTensor<T> tmp(A.TensorDist(), g);
    tmp.LockedAttach(A.Shape(), A.Alignments(), A.LockedBuffer(), A.LocalPermutation(), A.LocalStrides(), g);

    for(i = 0; i < intermediateDists.size() - 1; i++){
    	RedistInfo intRedist = intermediateDists[i];
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
//    	case RS: tmp2.ReduceScatterUpdateRedistFrom(tmp, lastRedist.reduceModes); break;
    	default: break;
    	}
//    	if(commRank == 0){
//			printf("done redist\n");
//		}
    	tmp.Empty();
    	tmp.SetDistribution(tmp2.TensorDist());
    	tmp = tmp2;
    }

	RedistInfo lastRedist = intermediateDists[intermediateDists.size() - 1];
	switch(lastRedist.redistType){
	case AG: AllGatherRedistFrom(tmp, lastRedist.modes); break;
	case A2A: AllToAllRedistFrom(tmp, lastRedist.modes); break;
	case Local: LocalRedistFrom(tmp); break;
	case Perm: PermutationRedistFrom(tmp, lastRedist.modes); break;
//	case RS: ReduceScatterUpdateRedistFrom(tmp, lastRedist.reduceModes); break;
	default: break;
	}

    PROFILE_STOP;
}

template <typename T>
void DistTensor<T>::RedistFrom(const DistTensor<T>& A){
	ModeArray reduceModes;
	RedistFrom(A, reduceModes);
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

} //namespace tmen
