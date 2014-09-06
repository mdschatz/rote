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

//TODO: Properly Check indices and distributions match between input and output
//TODO: FLESH OUT THIS CHECK
template <typename T>
Int DistTensor<T>::CheckGatherToOneCommRedist(const DistTensor<T>& A, const Mode rMode, const ModeArray& gridModes){
//    Unsigned i;
//    const tmen::GridView gvA = A.GetGridView();
//
//  //Test elimination of mode
//  const Unsigned AOrder = A.Order();
//  const Unsigned BOrder = B.Order();
//
//  //Check that redist modes are assigned properly on input and output
//  ModeDistribution BScatterModeDist = B.ModeDist(scatterMode);
//  ModeDistribution AGatherModeDist = A.ModeDist(reduceMode);
//  ModeDistribution AScatterModeDist = A.ModeDist(scatterMode);
//
//  //Test elimination of mode
//  if(BOrder != AOrder - 1){
//      LogicError("CheckGatherScatterRedist: Full Reduction requires elimination of mode being reduced");
//  }
//
//  //Test no wrapping of mode to reduce
//  if(A.Dimension(reduceMode) > gvA.Dimension(reduceMode))
//      LogicError("CheckGatherScatterRedist: Full Reduction requires global mode dimension <= gridView dimension");

    //Make sure all indices are distributed similarly between input and output (excluding reduce+scatter indices)

//  for(i = 0; i < BOrder; i++){
//      Mode mode = i;
//      if(mode == scatterMode){
//          ModeDistribution check(BScatterModeDist.end() - AGatherModeDist.size(), BScatterModeDist.end());
//            if(AnyElemwiseNotEqual(check, AGatherModeDist))
//                LogicError("CheckGatherScatterRedist: Gather mode distribution of A must be a suffix of Scatter mode distribution of B");
//      }
//  }

    return 1;
}

template <typename T>
void DistTensor<T>::GatherToOneCommRedist(const DistTensor<T>& A, const ModeArray& gatherModes, const std::vector<ModeArray>& commGroups){
//    if(!CheckGatherToOneCommRedist(A, gatherMode, commGroups))
//      LogicError("GatherToOneRedist: Invalid redistribution request");

    Unsigned i;
    const tmen::Grid& g = A.Grid();

    ModeArray commModes;
    for(i = 0; i < commGroups.size(); i++)
        commModes.insert(commModes.end(), commGroups[i].begin(), commGroups[i].end());
    std::sort(commModes.begin(), commModes.end());

    //NOTE: Hack for testing.  We actually need to let the user specify the commModes
    //NOTE: THIS NEEDS TO BE BEFORE Participating() OTHERWISE PROCESSES GET OUT OF SYNC
    const mpi::Comm comm = GetCommunicatorForModes(commModes, A.Grid());

    if(!A.Participating())
        return;
    Unsigned sendSize, recvSize;

    //Determine buffer sizes for communication
    const Unsigned nRedistProcs = Max(1, prod(FilterVector(A.Grid().Shape(), commModes)));
    const ObjShape maxLocalShapeA = A.MaxLocalShape();
    sendSize = prod(maxLocalShapeA);
    recvSize = sendSize;

    Memory<T> auxMemory;
    T* auxBuf = auxMemory.Require(sendSize + nRedistProcs*recvSize);
    MemZero(&(auxBuf[0]), sendSize + recvSize);
    T* sendBuf = &(auxBuf[0]);
    T* recvBuf = &(auxBuf[sendSize]);

    //NOTE: AG and GTO pack routines are the exact same
    PackAGCommSendBuf(A, sendBuf);

    mpi::Gather(sendBuf, sendSize, recvBuf, recvSize, 0, comm);

    if(!(Participating()))
        return;

    //NOTE: AG and GTO unpack routines are the exact same
    UnpackAGCommRecvBuf(recvBuf, gatherModes, commModes, maxLocalShapeA, A);
}

#define PROTO(T) \
        template Int  DistTensor<T>::CheckGatherToOneCommRedist(const DistTensor<T>& A, const Mode gMode, const ModeArray& gridModes); \
        template void DistTensor<T>::GatherToOneCommRedist(const DistTensor<T>& A, const ModeArray& gatherModes, const std::vector<ModeArray>& commGroups);


PROTO(int)
PROTO(float)
PROTO(double)
PROTO(Complex<float>)
PROTO(Complex<double>)

} //namespace tmen
