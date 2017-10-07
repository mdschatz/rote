/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef ROTE_BTAS_DIFF_HPP
#define ROTE_BTAS_DIFF_HPP

#include "rote/core/tensor_forward_decl.hpp"
#include "rote/core/view_decl.hpp"
#include "rote/util/btas_util.hpp"
#include "rote/util/vec_util.hpp"

namespace rote {

////////////////////////////////////
// Workhorse routines
////////////////////////////////////

template <typename T>
inline void Diff_fast(T const *const src1Buf, T const *const src2Buf,
                      T *const dstBuf, const DiffData &data) {
  const std::vector<Unsigned> loopEnd = data.loopShape;
  const std::vector<Unsigned> src1BufStrides = data.src1Strides;
  const std::vector<Unsigned> src2BufStrides = data.src2Strides;
  const std::vector<Unsigned> dstBufStrides = data.dstStrides;
  Unsigned src1BufPtr = 0;
  Unsigned src2BufPtr = 0;
  Unsigned dstBufPtr = 0;
  Unsigned order = loopEnd.size();
  Location curLoc(order, 0);
  Unsigned ptr = 0;

  if (loopEnd.size() == 0) {
    dstBuf[0] = src1Buf[0] - src2Buf[0];
    return;
  }

  bool done = !ElemwiseLessThan(curLoc, loopEnd);

  while (!done) {

    dstBuf[dstBufPtr] = src1Buf[src1BufPtr] - src2Buf[src2BufPtr];
    // Update
    curLoc[ptr]++;
    dstBufPtr += dstBufStrides[ptr];
    src1BufPtr += src1BufStrides[ptr];
    src2BufPtr += src2BufStrides[ptr];
    while (ptr < order && curLoc[ptr] >= loopEnd[ptr]) {
      curLoc[ptr] = 0;

      dstBufPtr -= dstBufStrides[ptr] * (loopEnd[ptr]);
      src1BufPtr -= src1BufStrides[ptr] * (loopEnd[ptr]);
      src2BufPtr -= src2BufStrides[ptr] * (loopEnd[ptr]);
      ptr++;
      if (ptr >= order) {
        done = true;
        break;
      } else {
        curLoc[ptr]++;
        dstBufPtr += dstBufStrides[ptr];
        src1BufPtr += src1BufStrides[ptr];
        src2BufPtr += src2BufStrides[ptr];
      }
    }
    if (done)
      break;
    ptr = 0;
  }
}

////////////////////////////////////
// Local interface
////////////////////////////////////
template <typename T>
void Diff(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C) {
  DiffData data;
  data.loopShape = C.Shape();
  data.src1Strides = A.Strides();
  data.src2Strides = B.Strides();
  data.dstStrides = C.Strides();

  const T *src1Buf = A.LockedBuffer();
  const T *src2Buf = B.LockedBuffer();
  T *dstBuf = C.Buffer();

  Diff_fast(src1Buf, src2Buf, dstBuf, data);
}

////////////////////////////////////
// Global interface
////////////////////////////////////
template <typename T>
void Diff(const DistTensor<T> &A, const DistTensor<T> &B, DistTensor<T> &C) {
  Diff(A.LockedTensor(), B.LockedTensor(), C.Tensor());
}

} // namespace rote

#endif // ifndef ROTE_BTAS_DIFF_HPP
