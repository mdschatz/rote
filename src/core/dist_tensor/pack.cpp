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

template <typename T>
void PackRSSendBuf(const DistTensor<T>& A, const int reduceIndex, const int scatterIndex, T * const sendBuf)
{

}

template <typename T>
void UnpackRSRecvBuf(const T * const recvBuf, const Int reduceIndex, const Int scatterIndex, DistTensor<T>& A)
{

}

#define PROTO(T) \
		template void PackRSSendBuf(const DistTensor<T>& A, const int reduceIndex, const int scatterIndex, T * const sendBuf); \
		template void UnpackRSRecvBuf(const T * const recvBuf, const Int reduceIndex, const Int scatterIndex, DistTensor<T>& A);

PROTO(int)
PROTO(float)
PROTO(double)

} // namespace tmen
