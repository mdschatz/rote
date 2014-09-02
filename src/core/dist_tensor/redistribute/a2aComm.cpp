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

//TODO: Check that allToAllIndices and commGroups are valid
template <typename T>
Int DistTensor<T>::CheckAllToAllDoubleModeCommRedist(const DistTensor<T>& A, const std::pair<Mode, Mode>& allToAllModes, const std::pair<ModeArray, ModeArray >& a2aCommGroups){
    if(A.Order() != this->Order())
        LogicError("CheckAllToAllDoubleModeRedist: Objects being redistributed must be of same order");
    Unsigned i;
    for(i = 0; i < A.Order(); i++){
        if(i != allToAllModes.first && i != allToAllModes.second){
            if(this->ModeDist(i) != A.ModeDist(i))
                LogicError("CheckAllToAllDoubleModeRedist: Non-redist modes must have same distribution");
        }
    }
    return 1;
}

template <typename T>
void DistTensor<T>::AllToAllDoubleModeCommRedist(const DistTensor<T>& A, const std::pair<Mode, Mode>& a2aModes, const std::pair<ModeArray, ModeArray >& a2aCommGroups){
    if(!this->CheckAllToAllDoubleModeCommRedist(A, a2aModes, a2aCommGroups))
        LogicError("AllToAllDoubleModeRedist: Invalid redistribution request");

    //Determine buffer sizes for communication
    //NOTE: Swap to concatenate vectors
    ModeArray commModes = a2aCommGroups.first;
    commModes.insert(commModes.end(), a2aCommGroups.second.begin(), a2aCommGroups.second.end());
    std::sort(commModes.begin(), commModes.end());

    const mpi::Comm comm = this->GetCommunicatorForModes(commModes, A.Grid());

    if(!A.Participating())
        return;
    Unsigned sendSize, recvSize;

    const Unsigned nRedistProcs = Max(1, prod(FilterVector(A.Grid().Shape(), commModes)));
    const ObjShape maxLocalShapeA = A.MaxLocalShape();
    const ObjShape maxLocalShapeB = MaxLocalShape();
    const ObjShape commDataShape = prod(maxLocalShapeA) < prod(maxLocalShapeB) ? maxLocalShapeA : maxLocalShapeB;

    sendSize = prod(commDataShape);
    recvSize = sendSize;

    Memory<T> auxMemory;
    T* auxBuf = auxMemory.Require((sendSize + recvSize) * nRedistProcs);
    MemZero(&(auxBuf[0]), (sendSize + recvSize) * nRedistProcs);

    T* sendBuf = &(auxBuf[0]);
    T* recvBuf = &(auxBuf[sendSize*nRedistProcs]);

    PackA2ADoubleModeCommSendBuf(A, a2aModes, a2aCommGroups, sendBuf);

    mpi::AllToAll(sendBuf, sendSize, recvBuf, recvSize, comm);

    if(!(this->Participating()))
        return;
    UnpackA2ADoubleModeCommRecvBuf(recvBuf, a2aModes, a2aCommGroups, A);
}

template <typename T>
void DistTensor<T>::PackA2ADoubleModeCommSendBuf(const DistTensor<T>& A, const std::pair<Mode, Mode>& a2aModes, const std::pair<ModeArray, ModeArray >& commGroups, T * const sendBuf){
    Unsigned i, j;
    const Unsigned order = A.Order();
    const T* dataBuf = A.LockedBuffer();

//    const tmen::GridView gvA = A.GetGridView();
//    const tmen::GridView gvB = GetGridView();

//    std::cout << "dataBuf:";
//    for(Unsigned i = 0; i < prod(A.LocalShape()); i++){
//        std::cout << " " <<  dataBuf[i];
//    }
//    std::cout << std::endl;

    const tmen::Grid& g = A.Grid();

    const Location zeros(order, 0);
    const Location ones(order, 1);

    Mode a2aMode1 = a2aModes.first;
    Mode a2aMode2 = a2aModes.second;

    ModeArray commGroup1 = commGroups.first;
    ModeArray commGroup2 = commGroups.second;

    //For convenience make sure that a2aMode1 is earlier in the packing
    if(a2aMode1 > a2aMode2){
        std::swap(a2aMode1, a2aMode2);
        std::swap(commGroup1, commGroup2);
    }

    //----------------------------------------
    //----------------------------------------
    //----------------------------------------

    const tmen::GridView gvA = A.GetGridView();
    const tmen::GridView gvB = GetGridView();

    ModeArray commModesAll = commGroup1;
    commModesAll.insert(commModesAll.end(), commGroup2.begin(), commGroup2.end());
    std::sort(commModesAll.begin(), commModesAll.end());
    const ObjShape commShapeAll = FilterVector(Grid().Shape(), commModesAll);

    const Unsigned comm1LCM = tmen::LCM(gvA.Dimension(a2aMode1), gvB.Dimension(a2aMode1));
    const Unsigned comm2LCM = tmen::LCM(gvA.Dimension(a2aMode2), gvB.Dimension(a2aMode2));
    PackData packData;
    const ObjShape maxLocalShapeA = A.MaxLocalShape();
    const ObjShape maxLocalShapeB = MaxLocalShape();
    const ObjShape sendShape = prod(maxLocalShapeA) < prod(maxLocalShapeB) ? maxLocalShapeA : maxLocalShapeB;

    const Unsigned a2aMode2StrideFactor = comm2LCM/gvA.Dimension(a2aMode2);
    const Unsigned a2aMode1StrideFactor = comm1LCM/gvA.Dimension(a2aMode1);

    packData.loopShape = A.LocalShape();
    packData.srcBufStrides = A.LocalStrides();
    packData.srcBufStrides[a2aMode2] *= a2aMode2StrideFactor;
    packData.srcBufStrides[a2aMode1] *= a2aMode1StrideFactor;

    packData.dstBufStrides = Dimensions2Strides(sendShape);

    packData.loopStarts = zeros;
    packData.loopIncs = ones;
    packData.loopIncs[a2aMode2] = a2aMode2StrideFactor;
    packData.loopIncs[a2aMode1] = a2aMode1StrideFactor;

    Unsigned a2aMode2Stride = A.LocalModeStride(a2aMode2);
    Unsigned a2aMode1Stride = A.LocalModeStride(a2aMode1);

    const Unsigned nCommElemsPerProc = prod(sendShape);
    const Location myFirstElemLoc = A.ModeShifts();
    Location packElem = myFirstElemLoc;
    //Pack only if we can
    if(ElemwiseLessThan(packElem, A.Shape())){
        for(i = 0; i < a2aMode2StrideFactor; i++){
            packElem[a2aMode2] = myFirstElemLoc[a2aMode2] + i * gvA.ModeWrapStride(a2aMode2);

            if(packElem[a2aMode2] >= Dimension(a2aMode2))
                continue;
            packData.loopStarts[a2aMode2] = i;

            for(j = 0; j < a2aMode1StrideFactor; j++){
                packElem[a2aMode1] = myFirstElemLoc[a2aMode1] + j * gvA.ModeWrapStride(a2aMode1);

                if(packElem[a2aMode1] >= Dimension(a2aMode1))
                    continue;
                packData.loopStarts[a2aMode1] = j;
//                PrintVector(packElem, "packElem");
                Location ownerB = DetermineOwner(packElem);
                Location ownerGridLoc = GridViewLoc2GridLoc(ownerB, gvB);
                Unsigned commLinLoc = Loc2LinearLoc(FilterVector(ownerGridLoc, commModesAll), FilterVector(g.Shape(), commModesAll));


//                std::cout << "startDataBufLoc: " << (i * a2aMode2Stride) + (j * a2aMode1Stride) << std::endl;
//                std::cout << "startSendBufLoc: " << commLinLoc * nCommElemsPerProc << std::endl;
                PackCommHelper(packData, order - 1, &(dataBuf[(i * a2aMode2Stride) + (j * a2aMode1Stride)]), &(sendBuf[commLinLoc * nCommElemsPerProc]));
//                std::cout << "packed sendBuf:";
//                for(Unsigned i = 0; i < nCommElemsPerProc * nRedistProcsAll; i++)
//                    std::cout << " " << sendBuf[i];
//                std::cout << std::endl;
            }
        }
    }

//    std::cout << "packed sendBuf:";
//    for(Unsigned i = 0; i < nCommElemsPerProc * nRedistProcsAll; i++)
//        std::cout << " " << sendBuf[i];
//    std::cout << std::endl;
}

template<typename T>
void DistTensor<T>::UnpackA2ADoubleModeCommRecvBuf(const T * const recvBuf, const std::pair<Mode, Mode>& a2aModes, const std::pair<ModeArray, ModeArray >& commGroups, const DistTensor<T>& A){
    Unsigned i, j;
    const Unsigned order = A.Order();
    T* dataBuf = this->Buffer();

    const tmen::GridView gvA = A.GetGridView();
    const tmen::GridView gvB = GetGridView();

    const Location zeros(order, 0);
    const Location ones(order, 1);

    const tmen::Grid& g = A.Grid();

    Mode a2aMode1 = a2aModes.first;
    Mode a2aMode2 = a2aModes.second;

    ModeArray commGroup1 = commGroups.first;
    ModeArray commGroup2 = commGroups.second;

    //For convenience make sure that a2aMode1 is earlier in the packing
    if(a2aMode1 > a2aMode2){
        std::swap(a2aMode1, a2aMode2);
        std::swap(commGroup1, commGroup2);
    }

    //------------------------------------
    //------------------------------------
    //------------------------------------

    ModeArray commModesAll = commGroup1;
    commModesAll.insert(commModesAll.end(), commGroup2.begin(), commGroup2.end());
    std::sort(commModesAll.begin(), commModesAll.end());
    const ObjShape commShapeAll = FilterVector(Grid().Shape(), commModesAll);

    const Unsigned comm1LCM = tmen::LCM(gvA.Dimension(a2aMode1), gvB.Dimension(a2aMode1));
    const Unsigned comm2LCM = tmen::LCM(gvA.Dimension(a2aMode2), gvB.Dimension(a2aMode2));

    const ObjShape maxLocalShapeA = A.MaxLocalShape();
    const ObjShape maxLocalShapeB = MaxLocalShape();
    const ObjShape recvShape = prod(maxLocalShapeA) < prod(maxLocalShapeB) ? maxLocalShapeA : maxLocalShapeB;

    const Unsigned a2aMode2StrideFactor = comm2LCM/gvB.Dimension(a2aMode2);
    const Unsigned a2aMode1StrideFactor = comm1LCM/gvB.Dimension(a2aMode1);

    PackData unpackData;
    unpackData.loopShape = LocalShape();
    unpackData.dstBufStrides = LocalStrides();
    unpackData.dstBufStrides[a2aMode2] *= a2aMode2StrideFactor;
    unpackData.dstBufStrides[a2aMode1] *= a2aMode1StrideFactor;

    unpackData.srcBufStrides = Dimensions2Strides(recvShape);

    unpackData.loopStarts = zeros;
    unpackData.loopIncs = ones;
    unpackData.loopIncs[a2aMode2] = a2aMode2StrideFactor;
    unpackData.loopIncs[a2aMode1] = a2aMode1StrideFactor;

    Unsigned a2aMode2Stride = LocalModeStride(a2aMode2);
    Unsigned a2aMode1Stride = LocalModeStride(a2aMode1);

    const Unsigned nCommElemsPerProc = prod(recvShape);
    const Location myFirstElemLoc = ModeShifts();

//    std::cout << "recvBuf:";
//    for(Unsigned i = 0; i < nRedistProcsAll * nCommElemsPerProc; i++){
//        std::cout << " " << recvBuf[i];
//    }
//    std::cout << std::endl;

    Location unpackElem = myFirstElemLoc;

    //Unpack only if we can
    if(ElemwiseLessThan(unpackElem, Shape())){
        for(i = 0; i < a2aMode2StrideFactor; i++){
            unpackElem[a2aMode2] = myFirstElemLoc[a2aMode2] + i*gvB.ModeWrapStride(a2aMode2);
            if(unpackElem[a2aMode2] >= Dimension(a2aMode2))
                continue;
            unpackData.loopStarts[a2aMode2] = i;

            for(j = 0; j < a2aMode1StrideFactor; j++){
                unpackElem[a2aMode1] = myFirstElemLoc[a2aMode1] + j*gvB.ModeWrapStride(a2aMode1);
                if(unpackElem[a2aMode1] >= Dimension(a2aMode1))
                    continue;
                unpackData.loopStarts[a2aMode1] = j;

                Location ownerA = A.DetermineOwner(unpackElem);
                Location ownerGridLoc = GridViewLoc2GridLoc(ownerA, gvA);
                Unsigned commLinLoc = Loc2LinearLoc(FilterVector(ownerGridLoc, commModesAll), FilterVector(g.Shape(), commModesAll));

//                PrintVector(unpackElem, "unpackElem");
//                std::cout << "startDataBufLoc: " << (i * a2aMode2Stride) + (j * a2aMode1Stride) << std::endl;
//                std::cout << "startRecvBufLoc: " << commLinLoc * nCommElemsPerProc << std::endl;
                PackCommHelper(unpackData, order - 1, &(recvBuf[commLinLoc * nCommElemsPerProc]), &(dataBuf[(i * a2aMode2Stride) + (j * a2aMode1Stride)]));
//                std::cout << "data:";
//                for(Unsigned k = 0; k < prod(unpackData.localShape); k++)
//                    std::cout << " " << dataBuf[k];
//                std::cout << std::endl;
            }
        }
    }
}


#define PROTO(T) \
        template Int  DistTensor<T>::CheckAllToAllDoubleModeCommRedist(const DistTensor<T>& A, const std::pair<Mode, Mode>& a2aModes, const std::pair<ModeArray, ModeArray >& a2aCommGroups); \
        template void DistTensor<T>::AllToAllDoubleModeCommRedist(const DistTensor<T>& A, const std::pair<Mode, Mode>& a2aIndices, const std::pair<ModeArray, ModeArray >& commGroups); \
        template void DistTensor<T>::PackA2ADoubleModeCommSendBuf(const DistTensor<T>& A, const std::pair<Mode, Mode>& a2aModes, const std::pair<ModeArray, ModeArray >& commGroups, T * const sendBuf); \
        template void DistTensor<T>::UnpackA2ADoubleModeCommRecvBuf(const T * const recvBuf, const std::pair<Mode, Mode>& a2aModes, const std::pair<ModeArray, ModeArray >& commGroups, const DistTensor<T>& A);

PROTO(int)
PROTO(float)
PROTO(double)
PROTO(Complex<double>)
PROTO(Complex<float>)

} //namespace tmen
