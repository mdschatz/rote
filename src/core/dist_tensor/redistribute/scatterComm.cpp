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
#include "rote/core/tensor.hpp"
#include <algorithm>
namespace rote {

template <typename T>
bool DistTensor<T>::CheckScatterCommRedist(const DistTensor<T> &A) {
  const TensorDistribution outDist = this->TensorDist();
  const TensorDistribution inDist = A.TensorDist();

  bool ret = true;
  ret &= CheckOrder(this->Order(), A.Order());
  ret &= CheckSameCommModes(outDist, inDist);
  ret &= CheckInIsPrefix(outDist, inDist);
  ret &= CheckNonDistOutIsPrefix(outDist, inDist);

  return ret;
}

template <typename T>
void DistTensor<T>::ScatterCommRedist(const DistTensor<T> &A,
                                      const ModeArray &commModes,
                                      const T alpha) {
  if (!CheckScatterCommRedist(A))
    LogicError("ScatterRedist: Invalid redistribution request");

  const rote::Grid &g = A.Grid();
  const mpi::Comm comm = this->GetCommunicatorForModes(commModes, g);

  if (!this->Participating())
    return;

  // Determine buffer sizes for communication
  const Unsigned nRedistProcs =
      Max(1, prod(FilterVector(g.Shape(), commModes)));
  const ObjShape commDataShape = this->MaxLocalShape();

  const Unsigned sendSize = prod(commDataShape);
  const Unsigned recvSize = sendSize;

  T *auxBuf = this->auxMemory_.Require((sendSize + recvSize) * nRedistProcs);

  T *sendBuf = &(auxBuf[0]);
  T *recvBuf = &(auxBuf[sendSize * nRedistProcs]);

  //	const T* dataBuf = A.LockedBuffer();
  //	PrintArray(dataBuf, A.LocalShape(), A.LocalStrides(), "srcBuf");

  // Pack the data
  PROFILE_SECTION("ScatterPack");
  if (A.Participating())
    this->PackA2ACommSendBuf(A, commModes, commDataShape, sendBuf);
  PROFILE_STOP;

  //	ObjShape sendShape = commDataShape;
  //	sendShape.insert(sendShape.end(), nRedistProcs);
  //	PrintArray(sendBuf, sendShape, "sendBuf");

  // Communicate the data
  PROFILE_SECTION("ScatterComm");
  // Realignment
  T *alignSendBuf = &(sendBuf[0]);
  T *alignRecvBuf = &(sendBuf[sendSize * nRedistProcs]);

  bool didAlign =
      this->AlignCommBufRedist(A, alignSendBuf, sendSize * nRedistProcs,
                               alignRecvBuf, sendSize * nRedistProcs);
  if (didAlign) {
    sendBuf = &(alignRecvBuf[0]);
    recvBuf = &(alignSendBuf[0]);
  }

  mpi::Scatter(sendBuf, sendSize, recvBuf, recvSize, 0, comm);
  PROFILE_STOP;

  //	ObjShape recvShape = commDataShape;
  //	PrintArray(recvBuf, recvShape, "recvBuf");

  // Unpack the data (if participating)
  PROFILE_SECTION("ScatterUnpack");
  PROFILE_MEMOPS(prod(this->MaxLocalShape()));
  this->UnpackPCommRecvBuf(recvBuf, alpha);
  PROFILE_STOP;

  //	const T* myBuf = LockedBuffer();
  //	PrintArray(myBuf, LocalShape(), LocalStrides(), "myBuf");

  this->auxMemory_.Release();
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
