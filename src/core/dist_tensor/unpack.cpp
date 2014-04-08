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
void UnpackA2ADoubleIndexRecvBuf(const T * const recvBuf, const std::pair<int, int>& a2aIndices, const std::pair<std::vector<int>, std::vector<int> >& commGroups, const std::vector<std::vector<int> >& recvFirstLocs, const DistTensor<T>& A, DistTensor<T>& B){
    const std::vector<int> start(A.Order(), 0);
    T* dataBuf = B.Buffer(start);

    const tmen::GridView gvA = A.GridView();
    const tmen::GridView gvB = B.GridView();

    const tmen::Grid& g = A.Grid();

    int a2aMode1 = A.ModeOfIndex(a2aIndices.first);
    int a2aMode2 = A.ModeOfIndex(a2aIndices.second);

    std::vector<int> commGroup1 = commGroups.first;
    std::vector<int> commGroup2 = commGroups.second;

    std::vector<int> distAa2aMode1 = A.ModeDist(a2aMode1);
    std::vector<int> distAa2aMode2 = A.ModeDist(a2aMode2);

    std::vector<int> distBa2aMode1 = B.ModeDist(a2aMode1);
    std::vector<int> distBa2aMode2 = B.ModeDist(a2aMode2);

    //For convenience make sure that a2aMode1 is earlier in the packing
    if(a2aMode1 > a2aMode2){
        std::swap(a2aMode1, a2aMode2);
        std::swap(commGroup1, commGroup2);
    }

    std::vector<int> commModes  = commGroup1;
    commModes.insert(commModes.end(), commGroup2.begin(), commGroup2.end());

    std::vector<int> nonCommModes;
    for(int i = 0; i < g.Order(); i++){
        if(std::find(commModes.begin(), commModes.end(), i) == commModes.end()){
            nonCommModes.push_back(i);
        }
    }

    std::vector<int> myGridLoc = g.Loc();
    std::vector<int> gridShape = g.Shape();

    std::vector<int> modeLCMs(B.Order());
    for(int i = 0; i < B.Order(); i++)
    	modeLCMs[i] = tmen::LCM(gvA.ModeWrapStride(i), gvB.ModeWrapStride(i));

    std::vector<int> modeUnpackWrapStrides(B.Order());
    for(int i = 0; i < B.Order(); i++)
    	modeUnpackWrapStrides[i] = modeLCMs[i] / gvB.ModeWrapStride(i);

    const int a2aMode2LCM = tmen::LCM(gvA.ModeWrapStride(a2aMode2), gvB.ModeWrapStride(a2aMode2));
    const int a2aMode1LCM = tmen::LCM(gvA.ModeWrapStride(a2aMode1), gvB.ModeWrapStride(a2aMode1));

    const int a2aMode2UnpackWrapStride = a2aMode2LCM / gvB.ModeWrapStride(a2aMode2);
    const int a2aMode1UnpackWrapStride = a2aMode1LCM / gvB.ModeWrapStride(a2aMode1);

    const int nRedistProcs = prod(FilterVector(g.Shape(), commModes));
    const std::vector<Int> gridCommModeSliceShape = FilterVector(g.Shape(), commModes);

    const std::vector<Int> localShapeB = B.LocalShape();
    std::vector<Int> maxLocalShapeB = MaxLengths(B.Shape(), gvB.Shape());

    const int a2aMode1LocalDim = B.LocalDimension(a2aMode1);
    const int a2aMode1MaxLocalDim = maxLocalShapeB[a2aMode1];
    const int a2aMode1LocalStride = B.LocalModeStride(a2aMode1);

    const int a2aMode2LocalDim = B.LocalDimension(a2aMode2);
    const int a2aMode2MaxLocalDim = maxLocalShapeB[a2aMode2];
    const int a2aMode2LocalStride = B.LocalModeStride(a2aMode2);

    const int nLocalSlices2 = Max(1, prod(localShapeB, a2aMode2)) / a2aMode2UnpackWrapStride;
    const int nMaxSlices2 = Max(1, prod(maxLocalShapeB, a2aMode2)) / a2aMode2UnpackWrapStride;

    //Slices1 only counts up to next a2aIndex
    const int nLocalSlices1 = Max(1, prod(localShapeB, a2aMode1)) / prod(localShapeB, a2aMode2) / a2aMode1UnpackWrapStride;
    const int nMaxSlices1 = Max(1, prod(maxLocalShapeB, a2aMode1)) / prod(maxLocalShapeB, a2aMode2) / a2aMode1UnpackWrapStride;

    const int copySliceSize = a2aMode1LocalStride;

    int nUnpackSlices = a2aMode2UnpackWrapStride * a2aMode1UnpackWrapStride;

    int sliceNum1, sliceNum2;  //Which slice we are packing for indexK
    int offRecvBufSlice2, offDataBufSlice2;  //Offsets used to index into data arrays
    int offRecvBufSlice1, offDataBufSlice1;  //Offsets used to index into data arrays
    int offRecvBufUnpackSlice, offDataBufUnpackSlice;
    int startRecvBuf, startDataBuf;

    int unpackSlice;

    std::vector<int> commGroupShape(B.Order());
    for(int i = 0; i < B.Order(); i++)
    	commGroupShape[i] = 1;
    commGroupShape[a2aMode1] = a2aMode1UnpackWrapStride;
    commGroupShape[a2aMode2] = a2aMode2UnpackWrapStride;

    printf("recvBuf:");
    for(int i = 0; i < prod(localShapeB); i++){
        printf(" %d", recvBuf[i]);
    }
    printf("\n");

    std::vector<int> startLoc(B.Order());
    for(int i = 0; i < B.Order(); i++)
        startLoc[i] = B.ModeShift(i);

    for(unpackSlice = 0; unpackSlice < nUnpackSlices; unpackSlice++){
        std::vector<int> unpackSliceMultiLoc = LinearLoc2Loc(unpackSlice, modeUnpackWrapStrides);

        std::vector<int> startSliceLoc = startLoc;
        for(int i = 0; i < B.Order(); i++){
            startSliceLoc[i] += unpackSliceMultiLoc[i] * gvB.ModeWrapStride(i);
        }

        //HACK TO DETERMINE WHICH PROC SEND THIS ELEMENT
        int owningProc = 0;
        for(int i = 0; i < nRedistProcs; i++){
            bool found = true;
            for(int j = 0; j < recvFirstLocs[i].size(); j++){
                if(recvFirstLocs[i][j] != startSliceLoc[j]){
                    found = false;
                    break;
                }
            }
            if(found){
                owningProc = i;
                break;
            }
        }

        offRecvBufUnpackSlice = owningProc * prod(maxLocalShapeB) / nRedistProcs;
        offDataBufUnpackSlice = LinearIndex(B.Global2LocalIndex(startSliceLoc), Dimensions2Strides(localShapeB));

        for(sliceNum2 = 0; sliceNum2 < nMaxSlices2; sliceNum2++){
            if(sliceNum2 >= nLocalSlices2)
                break;
            offRecvBufSlice2 = copySliceSize * nMaxSlices1 * sliceNum2;
            offDataBufSlice2 = copySliceSize * nLocalSlices1 * a2aMode1UnpackWrapStride * sliceNum2 * a2aMode2UnpackWrapStride;

            for(sliceNum1 = 0; sliceNum1 < nMaxSlices1; sliceNum1++){
                if(sliceNum1 >= nLocalSlices1)
                    break;
                offRecvBufSlice1 = copySliceSize * sliceNum1;
                offDataBufSlice1 = copySliceSize * sliceNum1 * a2aMode1UnpackWrapStride;

                startRecvBuf = offRecvBufUnpackSlice + offRecvBufSlice2 + offRecvBufSlice1;
                startDataBuf = offDataBufUnpackSlice + offDataBufSlice2 + offDataBufSlice1;
                MemCopy(&(dataBuf[startDataBuf]), &(recvBuf[startRecvBuf]), copySliceSize);
            }
        }
    }


    printf("Unpacked dataBuf:");
    for(int i = 0; i < prod(maxLocalShapeB); i++){
        printf(" %d", dataBuf[i]);
    }
    printf("\n");
}

#define PROTO(T) \
        template void UnpackPermutationRecvBuf(const T * const recvBuf, const Int permuteIndex, const DistTensor<T>& A, DistTensor<T>& B); \
        template void UnpackPartialRSRecvBuf(const T * const recvBuf, const Int reduceScatterIndex, const DistTensor<T>& A, DistTensor<T>& B); \
        template void UnpackRSRecvBuf(const T * const recvBuf, const Int reduceIndex, const Int scatterIndex, const DistTensor<T>& A, DistTensor<T>& B); \
        template void UnpackAGRecvBuf(const T * const recvBuf, const Int allGatherIndex, const DistTensor<T>& A, DistTensor<T>& B); \
        template void UnpackA2ADoubleIndexRecvBuf(const T * const recvBuf, const std::pair<int, int>& a2aIndices, const std::pair<std::vector<int>, std::vector<int> >& commGroups, const std::vector<std::vector<int> >& recvFirstLocs, const DistTensor<T>& A, DistTensor<T>& B);

PROTO(int)
PROTO(float)
PROTO(double)

} // namespace tmen
