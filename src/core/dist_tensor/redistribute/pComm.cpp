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

namespace rote {

// TODO: Check all unaffected indices are distributed similarly (Only done for
// CheckPermutationRedist currently)
template <typename T>
bool DistTensor<T>::CheckPermutationCommRedist(const DistTensor<T> &A) {
  const TensorDistribution outDist = this->TensorDist();
  const TensorDistribution inDist = A.TensorDist();

  bool ret = true;
  ret &= CheckOrder(this->Order(), A.Order());
  ret &= CheckSameGridViewShape(this->GridViewShape(), A.GridViewShape());
  ret &= CheckSameNonDist(outDist, inDist);

  return ret;
}

template <typename T>
void DistTensor<T>::PermutationCommRedist(const DistTensor<T> &A,
                                          const ModeArray &commModes,
                                          const T alpha) {
  if (!CheckPermutationCommRedist(A))
    LogicError("PermutationRedist: Invalid redistribution request");

  const rote::Grid &g = A.Grid();
  const rote::GridView gvA = A.GetGridView();
  const rote::GridView gvB = this->GetGridView();

  // Ripped from AlignCommBufRedist

  ModeArray misalignedModes = this->GetMisalignedModes(A);
  ModeArray actualCommModes = misalignedModes;
  for (Unsigned i = 0; i < commModes.size(); i++) {
    if (!Contains(actualCommModes, commModes[i])) {
      actualCommModes.insert(actualCommModes.end(), commModes[i]);
    }
  }
  SortVector(actualCommModes);

  mpi::Comm sendRecvComm = this->GetCommunicatorForModes(actualCommModes, g);

  // Skip if we aren't participating
  if (!A.Participating()) {
    std::cout << "NOT PARTICIPATING\n";
    return;
  }

  // Determine buffer sizes for communication
  const ObjShape commDataShape = this->MaxLocalShape();
  const Unsigned recvSize = prod(commDataShape);
  const Unsigned sendSize = recvSize;

  T *auxBuf = this->auxMemory_.Require(sendSize + recvSize);
  T *sendBuf = &(auxBuf[0]);
  T *recvBuf = &(auxBuf[sendSize]);

  //    const T* dataBuf = A.LockedBuffer();
  //    PrintArray(dataBuf, A.LocalShape(), A.LocalStrides(), "srcBuf");

  // Pack the data
  PROFILE_SECTION("PermutationPack");
  this->PackAGCommSendBuf(A, sendBuf);
  PROFILE_STOP;

  //    ObjShape sendShape = commDataShape;
  //    PrintArray(sendBuf, sendShape, "sendBuf");

  // Determine who I send+recv data from
  PROFILE_SECTION("PermutationComm");

  ModeArray sortedCommModes = commModes;
  SortVector(sortedCommModes);

  const Location myFirstElemA = A.DetermineFirstElem(gvA.ParticipatingLoc());
  const Location ownergvB = this->DetermineOwner(myFirstElemA);

  const Location myFirstElemB =
      this->DetermineFirstElem(gvB.ParticipatingLoc());
  const Location ownergvA = A.DetermineOwner(myFirstElemB);

  const Location sendLoc = GridViewLoc2GridLoc(ownergvB, gvB);
  const Unsigned sendLinLoc =
      Loc2LinearLoc(FilterVector(sendLoc, actualCommModes),
                    FilterVector(g.Shape(), actualCommModes));

  const Location recvLoc = GridViewLoc2GridLoc(ownergvA, gvA);
  const Unsigned recvLinLoc =
      Loc2LinearLoc(FilterVector(recvLoc, actualCommModes),
                    FilterVector(g.Shape(), actualCommModes));

  // PrintVector(actualCommModes, "actualCommModes", true);
  // std::cout << "sendLoc: " << sendLinLoc << " recvLoc: " << recvLinLoc <<
  // std::endl;
  mpi::SendRecv(sendBuf, sendSize, sendLinLoc, recvBuf, recvSize, recvLinLoc,
                sendRecvComm);
  PROFILE_STOP;
  // std::cout << "unpacking " << std::endl;
  //    ObjShape recvShape = commDataShape;
  //    PrintArray(recvBuf, recvShape, "recvBuf");

  // Unpack the data (if participating)
  PROFILE_SECTION("PermutationUnpack");
  this->UnpackPCommRecvBuf(recvBuf, alpha);
  PROFILE_STOP;

  //    const T* myBuf = LockedBuffer();
  //    PrintArray(myBuf, LocalShape(), LocalStrides(), "myBuf");

  this->auxMemory_.Release();
}

template <typename T>
void DistTensor<T>::UnpackPCommRecvBuf(const T *const recvBuf, const T alpha) {
  T *dataBuf = this->Buffer();

  PackData unpackData;
  unpackData.loopShape = this->LocalShape();
  unpackData.dstBufStrides = this->LocalStrides();
  unpackData.srcBufStrides = Dimensions2Strides(
      PermuteVector(this->MaxLocalShape(), this->localPerm_));

  if (alpha == T(0))
    PackCommHelper(unpackData, &(recvBuf[0]), &(dataBuf[0]));
  else {
    YAxpByData data;
    data.loopShape = unpackData.loopShape;
    data.dstStrides = unpackData.dstBufStrides;
    data.srcStrides = unpackData.srcBufStrides;
    YAxpBy_fast(T(1), alpha, &(recvBuf[0]), &(dataBuf[0]), data);
  }
}

#define FULL(T) template class DistTensor<T>;

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

} // namespace rote
