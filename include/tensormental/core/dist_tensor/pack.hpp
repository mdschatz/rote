/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/

#pragma once
#ifndef TMEN_CORE_DISTTENSOR_PACK_DECL_HPP
#define TMEN_CORE_DISTTENSOR_PACK_DECL_HPP

#include "tensormental/core/imports/mpi.hpp"
#include "tensormental/core/imports/choice.hpp"
#include "tensormental/core/imports/mpi_choice.hpp"

#include <vector>
#include <iostream>
#include "tensormental/core/dist_tensor/mc_mr.hpp"

namespace tmen{

template <typename T>
void PackRSSendBuf(const DistTensor<T>& A, const Int reduceScatterMode, T * const sendBuf);

template <typename T>
void PackRSSendBuf(const DistTensor<T>& A, const Int reduceMode, const Int scatterMode, T * const sendBuf);

template <typename T>
void UnpackRSRecvBuf(const T * const recvBuf, const Int reduceScatterMode, DistTensor<T>& A);

template <typename T>
void UnpackRSRecvBuf(const T* const recvBuf, const Int reduceMode,
        const Int scatterMode, const DistTensor<T>& A, DistTensor<T>& B) {
    printf("B can unpack %d elems\n", prod(B.LocalShape()));
    const std::vector<Int> start(B.Order(), 0);
    T* dataBuf = B.Buffer(start);
    const tmen::GridView gv = A.GridView();
    const int nModeProcs = gv.Dimension(reduceMode); //Number of procs per wrap
    const int sModeGlobalDim = B.Dimension(scatterMode); //Number of indices in the mode we are redistributing
    const std::vector<Int> maxRecvLocalShape = MaxLengths(A.Shape(),
            gv.Shape());
    const int sModeMaxLocalDim = maxRecvLocalShape[scatterMode];
    const std::vector<Int> localShape = B.LocalShape(); //Shape of the local tensor we are packing
    const int sModeLocalDim = B.LocalDimension(scatterMode); //Local version
    const int sModeLocalStride = B.LocalModeStride(scatterMode);
    //Loop packing bounds variables
    const int nWraps = MaxLength(sModeGlobalDim, nModeProcs);
    //Number of local slices and slice size we must pack per proc per wrap
    const int nLocalSlices = Max(1, prod(localShape, scatterMode + 1));
    const int nMaxSlices = Max(1, prod(maxRecvLocalShape, scatterMode + 1));
    //Variables for calculating elements to copy
    const int nMaxElemsPerProc = prod(maxRecvLocalShape);
    const int copySliceSize = sModeLocalStride;
    //Loop iteration vars
    int procRecvNum, wrapNum, sliceNum; //Pack data for slice "sliceNum" (<nSlices) of wrap "wrapNum" (<nWraps) for proc "procSendNum" int offSliceRecvBuf, offWrapRecvBuf;  //Offsets used to index into sendBuf array
    int offSliceRecvBuf, offWrapRecvBuf; //Offsets used to index into recvBuf array
    int offSliceDataBuf, offWrapDataBuf; //Offsets used to index into dataBuf array
    int startRecvBuf, startDataBuf;
    printf("alloced %d local elems for output\n", prod(B.LocalShape()));
    printf("data: [%.0f", (double) ((recvBuf[0])));
    for (int i = 1; i < nMaxElemsPerProc * nModeProcs; i++)
        printf(", %.0f", (double) ((recvBuf[i])));
    printf("]\n");
    for (sliceNum = 0; sliceNum < nMaxSlices; sliceNum++) {
        offSliceRecvBuf = copySliceSize * sModeMaxLocalDim * sliceNum;
        offSliceDataBuf = copySliceSize * sModeGlobalDim * sliceNum;
        if (sliceNum >= nLocalSlices) {
            break;
        }
        for (wrapNum = 0; wrapNum < nWraps; wrapNum++) {
            offWrapRecvBuf = copySliceSize * wrapNum;
            offWrapDataBuf = copySliceSize * nModeProcs * wrapNum;
            for (procRecvNum = 0; procRecvNum < nModeProcs; procRecvNum++) {
                startRecvBuf = offSliceRecvBuf + offWrapRecvBuf
                        + (nMaxElemsPerProc * procRecvNum);
                startDataBuf = offSliceDataBuf + offWrapDataBuf
                        + (copySliceSize * procRecvNum);
                if (wrapNum * nModeProcs + procRecvNum >= sModeLocalDim) {
                    break;
                }
                printf("startRecvBuf: %d startDataBuf: %d copySliceSize: %d\n",
                        startRecvBuf, startDataBuf, copySliceSize);
                MemCopy(&(dataBuf[startDataBuf]), &(recvBuf[startRecvBuf]),
                        copySliceSize);
            }
        }
    }
}

template <typename T>
void PackAGSendBuf(const DistTensor<T>& A, const Int allGatherMode, T * const sendBuf);

template <typename T>
void UnpackAGRecvBuf(const T * const recvBuf, const Int allGatherMode, const DistTensor<T>& A, DistTensor<T>& B);

}
#endif // ifndef TMEN_CORE_DISTTENSOR_REDISTRIBUTE_UTIL_DECL_HPP
