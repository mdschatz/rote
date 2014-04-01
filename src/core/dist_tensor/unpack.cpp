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
void UnpackPermutationRecvBuf(const T * const recvBuf, const Int permuteIndex, const DistTensor<T>& A, DistTensor<T>& B)
{
    //  printf("B can unpack %d elems\n", prod(B.LocalShape()));
        const std::vector<Int> start(B.Order(), 0);
        T* dataBuf = B.Buffer(start);
        const tmen::GridView gv = A.GridView();

        const int permuteModeA = A.ModeOfIndex(permuteIndex);

        const std::vector<Int> maxRecvLocalShape = MaxLengths(A.Shape(), gv.Shape());

        const std::vector<Int> localShape = B.LocalShape();         //Shape of the local tensor we are packing
        const int pModeLocalDim = B.LocalDimension(permuteModeA); //Local version
        const int pModeLocalStride = B.LocalModeStride(permuteModeA);

        //Number of local slices and slice size we must pack per proc per wrap
        const int nLocalSlices = Max(1, prod(localShape, permuteModeA + 1));
        const int nMaxSlices = Max(1, prod(maxRecvLocalShape, permuteModeA + 1));

        //Variables for calculating elements to copy
        const int maxCopySliceSize = Max(1, prod(maxRecvLocalShape, 0, permuteModeA + 1));
        const int copySliceSize = pModeLocalStride * pModeLocalDim;

        //Loop iteration vars
        int sliceNum;  //Pack data for slice "sliceNum" (<nSlices) of wrap "wrapNum" (<nWraps) for proc "procSendNum" int offSliceRecvBuf, offWrapRecvBuf;  //Offsets used to index into sendBuf array
        int offSliceRecvBuf;  //Offsets used to index into recvBuf array
        int offSliceDataBuf;  //Offsets used to index into dataBuf array
        int startRecvBuf, startDataBuf;

        //printf("alloced %d local elems for output\n", prod(B.LocalShape()));

        for(sliceNum = 0; sliceNum < nMaxSlices; sliceNum++){
            offSliceRecvBuf = maxCopySliceSize * sliceNum;
            offSliceDataBuf = copySliceSize * sliceNum;
            if(sliceNum >= nLocalSlices){
                break;
            }
            startRecvBuf = offSliceRecvBuf;
            startDataBuf = offSliceDataBuf;
            //printf("startRecvBuf: %d startDataBuf: %d copySliceSize: %d\n", startRecvBuf, startDataBuf, copySliceSize);
            MemCopy(&(dataBuf[startDataBuf]), &(recvBuf[startRecvBuf]), copySliceSize);
        }
//        printf("unpacking %d elems\n", nModeProcs * nMaxElemsPerProc);
//        std::ostringstream msg;
//        msg << "recv'd data: [" << recvBuf[0];
//        for (int i = 1; i < nMaxElemsPerProc * nModeProcs; i++)
//            msg << ", " << recvBuf[i];
//        msg << "]" << std::endl;
//        std::cout << msg.str();

}

template <typename T>
void UnpackPartialRSRecvBuf(const T * const recvBuf, const Int reduceScatterIndex, const DistTensor<T>& A, DistTensor<T>& B)
{
    UnpackRSRecvBuf(recvBuf, reduceScatterIndex, reduceScatterIndex, A, B);
}

//Only called when fully reducing an index
//NOTE: Looks an awful lot like PackAGSendBuf...
//TODO: Merge with PackAGSendBuf?
//TODO: Make this work with blocks (more general is commented out code
template <typename T>
void UnpackRSRecvBuf(const T * const recvBuf, const Int reduceIndex, const Int scatterIndex, const DistTensor<T>& A, DistTensor<T>& B)
{
//    printf("B can unpack %d elems\n", prod(B.LocalShape()));
    const std::vector<Int> start(B.Order(), 0);
    T* dataBuf = B.Buffer(start);

    const int scatterModeB = B.ModeOfIndex(scatterIndex);

    const tmen::GridView gvA = A.GridView();
    const tmen::GridView gvB = B.GridView();
    const int nModeProcs = gvB.Dimension(scatterModeB);   //Number of procs per wrap
    const int sModeGlobalDim = B.Dimension(scatterModeB);           //Number of indices in the mode we are redistributing

    const std::vector<Int> maxRecvLocalShape = MaxLengths(B.Shape(), gvB.Shape());

    const std::vector<Int> localShapeB = B.LocalShape();         //Shape of the local tensor we are packing
    const int sModeLocalDimB = B.LocalDimension(scatterModeB); //Local version
    const int sModeLocalStrideB = B.LocalModeStride(scatterModeB);

    //Loop packing bounds variables
    const int nMaxWraps = MaxLength(sModeGlobalDim, nModeProcs);
    //Number of local slices and slice size we must pack per proc per wrap
    const int nLocalSlices = Max(1, prod(localShapeB, scatterModeB + 1));
    const int nMaxSlices = Max(1, prod(maxRecvLocalShape, scatterModeB + 1));

    //Variables for calculating elements to copy
    const int copySliceSize = sModeLocalStrideB;
    const int maxCopySliceSize = Max(1, prod(maxRecvLocalShape, 0, scatterModeB));

    //Loop iteration vars
    int wrapNum, sliceNum;  //Pack data for slice "sliceNum" (<nSlices) of wrap "wrapNum" (<nWraps) for proc "procSendNum" int offSliceRecvBuf, offWrapRecvBuf;  //Offsets used to index into sendBuf array
    int offSliceRecvBuf;  //Offsets used to index into recvBuf array
    int offSliceDataBuf;  //Offsets used to index into dataBuf array
    int startRecvBuf, startDataBuf;

//    printf("alloced %d local elems for output\n", prod(B.LocalShape()));
//    std::ostringstream msg;
//    msg << "recv'd data: [" << recvBuf[0];
//    const int nMaxElemsPerProc = prod(maxRecvLocalShape);
//    for(int i = 1; i < nMaxElemsPerProc; i++)
//        msg << ", " << recvBuf[i];
//    msg << "]" << std::endl;
//    std::cout << msg.str();

    for(sliceNum = 0; sliceNum < nMaxSlices; sliceNum++){
        if(sliceNum >= nLocalSlices)
            break;
        offSliceRecvBuf = maxCopySliceSize * nMaxWraps * sliceNum;
        offSliceDataBuf = copySliceSize * sModeLocalDimB * sliceNum;
        for(wrapNum = 0; wrapNum < nMaxWraps; wrapNum++){
            if(wrapNum >= sModeLocalDimB)
                break;
            startRecvBuf = offSliceRecvBuf + (copySliceSize * wrapNum);
            startDataBuf = offSliceDataBuf + (maxCopySliceSize * wrapNum);

            //printf("startRecvBuf: %d startDataBuf: %d copySliceSize: %d\n", startRecvBuf, startDataBuf, copySliceSize);
            MemCopy(&(dataBuf[startDataBuf]), &(recvBuf[startRecvBuf]), copySliceSize);
        }
    }
}

//Given following set of strides in a tensor (s1, s2, ..., sm) and mode x we wish to redistribute
//Pack according to following scheme:
//ptr = 0  <-- Ptr into the sendBuf
//foreach proc in nProcs:
//  foreach wrapNum in ceil((mx-procNum)/nProcs)  <-- Number of wraps we have to send to each proc
//    startIndex = sx * (wrapNum* nProcs + proc)  <-- Sets the first index we should copy from, i.e, (0,...,0,wrapNum*nProcs + proc, 0, ..., 0)
//    foreach prod(m(x+1), ..., m(m)):  <-- Defines the number of slices we have to copy over per wrap
//      memcpy(dataBuf[startIndex], sendBuf[ptr], max(1, prod(s1,..., s(x-1)))) <-- Copy the slice (0,..., 0, proc, 0, ..., 0) to (m1, m2, ..., m(x-1), proc, 0, ..., 0)
//      startIndex += sx*mx  <-- Put us in the next slice , i.e., (0,..., 0, proc, 1, ..., 0) to (m1, m2, ..., m(x-1), proc, 1, ..., 0)
//      ptr += max(1, prod(s1,..., s(x-1))) <-- Increment ptr in the sendBuf
//TODO: Make this work with blocks
//TODO: Swap nWrap test to be more like sliceNum test (Refer to local information)
template <typename T>
void UnpackAGRecvBuf(const T * const recvBuf, const Int allGatherIndex, const DistTensor<T>& A, DistTensor<T>& B)
{
//  printf("B can unpack %d elems\n", prod(B.LocalShape()));
    const std::vector<Int> start(B.Order(), 0);
    T* dataBuf = B.Buffer(start);
    const tmen::GridView gv = A.GridView();

    const int allGatherMode = A.ModeOfIndex(allGatherIndex);
    const int nModeProcs = gv.Dimension(allGatherMode);   //Number of procs per wrap
    const int agModeGlobalDim = B.Dimension(allGatherMode);           //Number of indices in the mode we are redistributing

    const std::vector<Int> maxRecvLocalShape = MaxLengths(A.Shape(), gv.Shape());
    const int agModeMaxLocalDim = maxRecvLocalShape[allGatherMode];

    const std::vector<Int> localShape = B.LocalShape();         //Shape of the local tensor we are packing
    const int agModeLocalDim = B.LocalDimension(allGatherMode); //Local version
    const int agModeLocalStride = B.LocalModeStride(allGatherMode);

    //Loop packing bounds variables
    const int nWraps = MaxLength(agModeGlobalDim, nModeProcs);
    //Number of local slices and slice size we must pack per proc per wrap
    const int nLocalSlices = Max(1, prod(localShape, allGatherMode + 1));
    const int nMaxSlices = Max(1, prod(maxRecvLocalShape, allGatherMode + 1));

    //Variables for calculating elements to copy
    const int nMaxElemsPerProc = prod(maxRecvLocalShape);
    const int copySliceSize = agModeLocalStride;

    //Loop iteration vars
    int procRecvNum, wrapNum, sliceNum;  //Pack data for slice "sliceNum" (<nSlices) of wrap "wrapNum" (<nWraps) for proc "procSendNum" int offSliceRecvBuf, offWrapRecvBuf;  //Offsets used to index into sendBuf array
    int offSliceRecvBuf, offWrapRecvBuf;  //Offsets used to index into recvBuf array
    int offSliceDataBuf, offWrapDataBuf;  //Offsets used to index into dataBuf array
    int startRecvBuf, startDataBuf;

    //printf("alloced %d local elems for output\n", prod(B.LocalShape()));
    //std::ostringstream msg;
    //msg << "recv'd data: [" << recvBuf[0];
    //for(int i = 1; i < nMaxElemsPerProc * nModeProcs; i++)
    //    msg << ", " << recvBuf[i];
    //msg << "]" << std::endl;
    //std::cout << msg.str();

    for(sliceNum = 0; sliceNum < nMaxSlices; sliceNum++){
        offSliceRecvBuf = copySliceSize * agModeMaxLocalDim * sliceNum;
        offSliceDataBuf = copySliceSize * agModeGlobalDim * sliceNum;
        if(sliceNum >= nLocalSlices){
            break;
        }
        for(wrapNum = 0; wrapNum < nWraps; wrapNum++){
            offWrapRecvBuf = copySliceSize * wrapNum;
            offWrapDataBuf = copySliceSize * nModeProcs * wrapNum;
            for(procRecvNum = 0; procRecvNum < nModeProcs; procRecvNum++){
                startRecvBuf = offSliceRecvBuf + offWrapRecvBuf + (nMaxElemsPerProc * procRecvNum);
                startDataBuf = offSliceDataBuf + offWrapDataBuf + (copySliceSize * procRecvNum);
                if(wrapNum * nModeProcs + procRecvNum >= agModeLocalDim){
                    break;
                }
                //printf("startRecvBuf: %d startDataBuf: %d copySliceSize: %d\n", startRecvBuf, startDataBuf, copySliceSize);
                MemCopy(&(dataBuf[startDataBuf]), &(recvBuf[startRecvBuf]), copySliceSize);
            }
        }
    }
//    printf("unpacking %d elems\n", nModeProcs * nMaxElemsPerProc);
//    std::ostringstream msg;
//    msg << "recv'd data: [" << recvBuf[0];
//    for (int i = 1; i < nMaxElemsPerProc * nModeProcs; i++)
//        msg << ", " << recvBuf[i];
//    msg << "]" << std::endl;
//    std::cout << msg.str();
}

template<typename T>
void UnpackA2ADoubleIndexRecvBuf(const T * const recvBuf, const std::pair<int, int>& a2aIndices, const std::vector<int>& commModes, const DistTensor<T>& A, DistTensor<T>& B){

}

#define PROTO(T) \
        template void UnpackPermutationRecvBuf(const T * const recvBuf, const Int permuteIndex, const DistTensor<T>& A, DistTensor<T>& B); \
        template void UnpackPartialRSRecvBuf(const T * const recvBuf, const Int reduceScatterIndex, const DistTensor<T>& A, DistTensor<T>& B); \
        template void UnpackRSRecvBuf(const T * const recvBuf, const Int reduceIndex, const Int scatterIndex, const DistTensor<T>& A, DistTensor<T>& B); \
        template void UnpackAGRecvBuf(const T * const recvBuf, const Int allGatherIndex, const DistTensor<T>& A, DistTensor<T>& B); \
        template void UnpackA2ADoubleIndexRecvBuf(const T * const recvBuf, const std::pair<int, int>& a2aIndices, const std::vector<int>& commModes, const DistTensor<T>& A, DistTensor<T>& B);

PROTO(int)
PROTO(float)
PROTO(double)

} // namespace tmen
