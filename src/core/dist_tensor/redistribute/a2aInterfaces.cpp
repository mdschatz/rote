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

//TODO: MAKE SURE ALL REDISTS WORK WITH blank commModes (size=0)
template <typename T>
void DistTensor<T>::AllToAllRedistFrom(const DistTensor<T>& A, const ModeArray& commModes){
    PROFILE_SECTION("A2ARedist");
    ResizeTo(A);
    ModeArray sortedCommModes = commModes;
    std::sort(sortedCommModes.begin(), sortedCommModes.end());

    //BEGIN A2A code
    A2AP2PData optData = DetermineA2AP2POptData(A, commModes);

    int commRank = mpi::CommRank(MPI_COMM_WORLD);
//    if(commRank == 0){
//        PrintA2AOptData(optData, "optData");
//    }

    if(optData.doOpt == true){
        const tmen::Grid& g = A.Grid();
        ModeArray p2pModes = optData.p2pModes;
        ModeArray a2aModes = optData.a2aModes;
        std::sort(a2aModes.begin(), a2aModes.end());
        ModeArray a2ap2pModes = ConcatenateVectors(a2aModes, p2pModes);

        DistTensor<T> temp1(optData.opt1Dist, g);
        temp1.PermutationRedistFrom(A, a2ap2pModes);
//        Print(temp1, "temp1");
        DistTensor<T> temp2(optData.opt2Dist, g);
        temp2.PermutationRedistFrom(temp1, p2pModes);
//        Print(temp2, "temp2");
        temp1.EmptyData();
        DistTensor<T> temp3(optData.opt3Dist, g);
        temp3.PermutationRedistFrom(temp2, a2ap2pModes);
//        Print(temp3, "temp3");
        temp2.EmptyData();
        DistTensor<T> temp4(optData.opt4Dist, g);
        temp4.ResizeTo(temp3);
        //NOTE: Fix this bug
        if(a2aModes.size() == 0)
            temp4.PermutationRedistFrom(temp3, a2aModes);
        else
            temp4.AllToAllCommRedist(temp3, a2aModes);
//        Print(temp4, "temp4");
        temp3.EmptyData();
        PermutationRedistFrom(temp4, a2ap2pModes);
//        Print(*this, "final");
        temp4.EmptyData();
    }else{
        AllToAllCommRedist(A, sortedCommModes);
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
