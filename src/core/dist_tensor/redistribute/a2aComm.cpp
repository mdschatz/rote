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
    const ObjShape maxLocalShapeA = A.MaxShape();
    const ObjShape maxLocalShapeB = MaxShape();
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
void DistTensor<T>::PackA2ACommSendBufHelper(const A2APackData& packData, const Mode packMode, T const * const dataBuf, T * const sendBuf){
    Unsigned packSlice = packMode;
    Unsigned packSliceLocalDim = packData.dataShape[packSlice];
    Unsigned packSliceSendBufStride = packData.sendBufModeStrides[packSlice];
    Unsigned packSliceDataBufStride = packData.dataBufModeStrides[packSlice];
    Unsigned elemSlice2 = packData.elemSlice2;
    Unsigned maxElemSlice2 = packData.elemSlice2Stride;
    Unsigned elemSlice1 = packData.elemSlice1;
    Unsigned maxElemSlice1 = packData.elemSlice1Stride;
    Unsigned commMode1 = packData.commMode1;
    Unsigned commMode2 = packData.commMode2;

    Unsigned sendBufPtr = 0;
    Unsigned dataBufPtr = 0;

    //    Unsigned order = Order();
    //    Unsigned i;
    //    std::string ident = "";
    //    for(i = 0; i < order - unpackMode; i++)
    //        ident += "  ";
    //    std::cout << ident << "Unpacking mode " << unpackMode << std::endl;
    if(packMode == 0){
//        std::cout << ident << "recvBuf loc " << &(recvBuf[pRecvBufPtr]) << std::endl;
//        std::cout << ident << "dataBuf loc " << &(dataBuf[pDataBufPtr]) << std::endl;
//        std::cout << ident << "unpacking recv data:";
//        for(unpackSlice = 0; unpackSlice < unpackSliceMaxDim && unpackSlice < unpackSliceLocalDim; unpackSlice++){
//            std::cout << " " << recvBuf[pRecvBufPtr];
//            pRecvBufPtr += unpackSliceRecvBufStride;
//            pDataBufPtr += unpackSliceDataBufStride;
//        }
//        std::cout << std::endl;

        if(packMode == commMode1 || packMode == commMode2){
            const Unsigned maxElemSlice = (packMode == commMode1) ? maxElemSlice1 : maxElemSlice2;
            const Unsigned elemSlice = (packMode == commMode1) ? elemSlice1 : elemSlice2;
            //NOTE: Each tensor data we unpack is strided by nRedistProcs (maxElemSlice) entries away
            packSliceDataBufStride *= maxElemSlice;

            for(packSlice = 0; (packSlice + elemSlice) < packSliceLocalDim; packSlice += maxElemSlice){
                sendBuf[sendBufPtr] = dataBuf[dataBufPtr];

//                    std::cout << "recvBuf inc by " << unpackSliceRecvBufStride << std::endl;
//                    std::cout << "dataBuf inc by " << unpackSliceDataBufStride << std::endl;
                sendBufPtr += packSliceSendBufStride;
                dataBufPtr += packSliceDataBufStride;
            }
        }else{
            if(packSliceSendBufStride == 1 && packSliceDataBufStride == 1){
                MemCopy(&(sendBuf[0]), &(dataBuf[0]), packSliceLocalDim);
            }else{
                for(packSlice = 0; packSlice < packSliceLocalDim; packSlice++){
                    sendBuf[sendBufPtr] = dataBuf[dataBufPtr];

    //                std::cout << "recvBuf inc by " << unpackSliceRecvBufStride << std::endl;
    //                std::cout << "dataBuf inc by " << unpackSliceDataBufStride << std::endl;
                    sendBufPtr += packSliceSendBufStride;
                    dataBufPtr += packSliceDataBufStride;
                }
            }
        }
    }else {
        if(packMode == commMode1 || packMode == commMode2){
            const Unsigned maxElemSlice = (packMode == commMode1) ? maxElemSlice1 : maxElemSlice2;
            const Unsigned elemSlice = (packMode == commMode1) ? elemSlice1 : elemSlice2;
            //NOTE: Each tensor data we unpack is strided by nRedistProcs (maxElemSlice) entries away
            packSliceDataBufStride *= maxElemSlice;

            for(packSlice = 0; (packSlice + elemSlice) < packSliceLocalDim; packSlice += maxElemSlice){
                PackA2ACommSendBufHelper(packData, packMode-1, &(dataBuf[dataBufPtr]), &(sendBuf[sendBufPtr]));

//                    std::cout << "recvBuf inc by " << unpackSliceRecvBufStride << std::endl;
//                    std::cout << "dataBuf inc by " << unpackSliceDataBufStride << std::endl;
                sendBufPtr += packSliceSendBufStride;
                dataBufPtr += packSliceDataBufStride;
            }
        }else{
            for(packSlice = 0; packSlice < packSliceLocalDim; packSlice++){
                PackA2ACommSendBufHelper(packData, packMode-1, &(dataBuf[dataBufPtr]), &(sendBuf[sendBufPtr]));
    //            std::cout << "recvBuf inc by " << unpackSliceRecvBufStride << std::endl;
    //            std::cout << "dataBuf inc by " << unpackSliceDataBufStride << std::endl;
                sendBufPtr += packSliceSendBufStride;
                dataBufPtr += packSliceDataBufStride;
            }
        }
    }
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
    A2APackData packData;
    const ObjShape maxLocalShapeA = A.MaxShape();
    const ObjShape maxLocalShapeB = MaxShape();
    const ObjShape sendShape = prod(maxLocalShapeA) < prod(maxLocalShapeB) ? maxLocalShapeA : maxLocalShapeB;

    packData.dataShape = A.LocalShape();
    packData.dataBufModeStrides = A.LocalStrides();

    packData.sendBufModeStrides = Dimensions2Strides(sendShape);
    packData.elemSlice1Stride = comm1LCM/gvA.Dimension(a2aMode1);
    packData.commMode1 = a2aMode1;

    packData.elemSlice2Stride = comm2LCM/gvA.Dimension(a2aMode2);
    packData.commMode2 = a2aMode2;

    Unsigned a2aMode2Stride = A.LocalModeStride(a2aMode2);
    Unsigned a2aMode1Stride = A.LocalModeStride(a2aMode1);

    const Unsigned nCommElemsPerProc = prod(sendShape);
    const Location myFirstElemLoc = A.ModeShifts();
    Location packElem = myFirstElemLoc;
    //Pack only if we can
    if(ElemwiseLessThan(packElem, A.Shape())){
        for(i = 0; i < packData.elemSlice2Stride; i++){
            packElem[a2aMode2] = myFirstElemLoc[a2aMode2] + i * gvA.ModeWrapStride(a2aMode2);

            if(packElem[a2aMode2] >= Dimension(a2aMode2))
                continue;
            packData.elemSlice2 = i;

            for(j = 0; j < packData.elemSlice1Stride; j++){
                packElem[a2aMode1] = myFirstElemLoc[a2aMode1] + j * gvA.ModeWrapStride(a2aMode1);

                if(packElem[a2aMode1] >= Dimension(a2aMode1))
                    continue;
                packData.elemSlice1 = j;
//                PrintVector(packElem, "packElem");
                Location ownerB = DetermineOwner(packElem);
                Location ownerGridLoc = GridViewLoc2GridLoc(ownerB, gvB);
                Unsigned commLinLoc = Loc2LinearLoc(FilterVector(ownerGridLoc, commModesAll), FilterVector(g.Shape(), commModesAll));


//                std::cout << "startDataBufLoc: " << (i * a2aMode2Stride) + (j * a2aMode1Stride) << std::endl;
//                std::cout << "startSendBufLoc: " << commLinLoc * nCommElemsPerProc << std::endl;
                PackA2ACommSendBufHelper(packData, order - 1, &(dataBuf[(i * a2aMode2Stride) + (j * a2aMode1Stride)]), &(sendBuf[commLinLoc * nCommElemsPerProc]));
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
void DistTensor<T>::UnpackA2ACommRecvBufHelper(const A2AUnpackData& unpackData, const Mode unpackMode, T const * const recvBuf, T * const dataBuf){
    Unsigned unpackSlice = unpackMode;
    Unsigned unpackSliceLocalDim = unpackData.dataShape[unpackSlice];
    Unsigned unpackSliceRecvBufStride = unpackData.recvBufModeStrides[unpackSlice];
    Unsigned unpackSliceDataBufStride = unpackData.dataBufModeStrides[unpackSlice];
    Unsigned commMode1 = unpackData.commMode1;
    Unsigned commMode2 = unpackData.commMode2;
    Unsigned recvBufPtr = 0;
    Unsigned dataBufPtr = 0;

//    Unsigned pRecvBufPtr = 0;
//    Unsigned pDataBufPtr = 0;
//    Unsigned order = Order();
//    Unsigned i;
//    std::string ident = "";
//    for(i = 0; i < order - unpackMode; i++)
//        ident += "  ";
//    std::cout << ident << "Unpacking mode " << unpackMode << std::endl;

    if(unpackMode == 0){
        if(unpackMode == commMode1 || unpackMode == commMode2){
            Unsigned elemSlice = (unpackMode == commMode1) ? unpackData.elemSlice1 : unpackData.elemSlice2;
            Unsigned elemSliceStride = (unpackMode == commMode1) ? unpackData.elemSlice1Stride : unpackData.elemSlice2Stride;

            //NOTE: Each tensor data we unpack is strided by nRedistProcs (maxElemSlice) entries away
            unpackSliceDataBufStride *= elemSliceStride;
//            std::cout << ident << "recvBuf loc " << &(recvBuf[pRecvBufPtr]) << std::endl;
//            std::cout << ident << "dataBuf loc " << &(dataBuf[pDataBufPtr]) << std::endl;
//            std::cout << ident << "unpacking recv data:";
//            for(unpackSlice = 0; unpackSlice < unpackSliceMaxDim && (unpackSlice + elemSlice) < unpackSliceLocalDim; unpackSlice += maxElemSlice){
//                std::cout << " " << recvBuf[pRecvBufPtr];
//
//                pRecvBufPtr += unpackSliceRecvBufStride;
//                pDataBufPtr += unpackSliceDataBufStride;
//            }
//            std::cout << std::endl;

            for(unpackSlice = 0; (unpackSlice + elemSlice) < unpackSliceLocalDim; unpackSlice += elemSliceStride){
                dataBuf[dataBufPtr] = recvBuf[recvBufPtr];

//                std::cout << ident << "recvBuf inc by " << unpackSliceRecvBufStride << std::endl;
//                std::cout << ident << "dataBuf inc by " << unpackSliceDataBufStride << std::endl;
                recvBufPtr += unpackSliceRecvBufStride;
                dataBufPtr += unpackSliceDataBufStride;
            }
        }else{
            if(unpackSliceRecvBufStride == 1 && unpackSliceDataBufStride == 1){
                MemCopy(&(dataBuf[0]), &(recvBuf[0]), unpackSliceLocalDim);
            }else{
    //            std::cout << ident << "unpacking recv data:";
    //            for(unpackSlice = 0; unpackSlice < unpackSliceLocalDim; unpackSlice++){
    //                std::cout << " " << recvBuf[pRecvBufPtr];
    //                pRecvBufPtr += unpackSliceRecvBufStride;
    //                pDataBufPtr += unpackSliceDataBufStride;
    //            }
    //            std::cout << std::endl;
                for(unpackSlice = 0; unpackSlice < unpackSliceLocalDim; unpackSlice++){
                    dataBuf[dataBufPtr] = recvBuf[recvBufPtr];

    //                std::cout << ident << "recvBuf inc by " << unpackSliceRecvBufStride << std::endl;
    //                std::cout << ident << "dataBuf inc by " << unpackSliceDataBufStride << std::endl;
                    recvBufPtr += unpackSliceRecvBufStride;
                    dataBufPtr += unpackSliceDataBufStride;
                }
            }
        }
    }else {
        if(unpackMode == commMode1 || unpackMode == commMode2){
            Unsigned elemSlice = (unpackMode == commMode1) ? unpackData.elemSlice1 : unpackData.elemSlice2;
            Unsigned elemSliceStride = (unpackMode == commMode1) ? unpackData.elemSlice1Stride : unpackData.elemSlice2Stride;

            //NOTE: Each tensor data we unpack is strided by nRedistProcs (maxElemSlice) entries away
            unpackSliceDataBufStride *= elemSliceStride;

            for(unpackSlice = 0; (unpackSlice + elemSlice) < unpackSliceLocalDim; unpackSlice += elemSliceStride){
                UnpackA2ACommRecvBufHelper(unpackData, unpackMode-1, &(recvBuf[recvBufPtr]), &(dataBuf[dataBufPtr]));

//                std::cout << ident << "recvBuf inc by " << unpackSliceRecvBufStride << std::endl;
//                std::cout << ident << "dataBuf inc by " << unpackSliceDataBufStride << std::endl;
                recvBufPtr += unpackSliceRecvBufStride;
                dataBufPtr += unpackSliceDataBufStride;
            }
        }else{
            for(unpackSlice = 0; unpackSlice < unpackSliceLocalDim; unpackSlice++){
                UnpackA2ACommRecvBufHelper(unpackData, unpackMode-1, &(recvBuf[recvBufPtr]), &(dataBuf[dataBufPtr]));
//                std::cout << ident << "recvBuf inc by " << unpackSliceRecvBufStride << std::endl;
//                std::cout << ident << "dataBuf inc by " << unpackSliceDataBufStride << std::endl;
                recvBufPtr += unpackSliceRecvBufStride;
                dataBufPtr += unpackSliceDataBufStride;
            }
        }
    }
}

template<typename T>
void DistTensor<T>::UnpackA2ADoubleModeCommRecvBuf(const T * const recvBuf, const std::pair<Mode, Mode>& a2aModes, const std::pair<ModeArray, ModeArray >& commGroups, const DistTensor<T>& A){
    Unsigned i, j;
    const Unsigned order = A.Order();
    T* dataBuf = this->Buffer();

    const tmen::GridView gvA = A.GetGridView();
    const tmen::GridView gvB = GetGridView();

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

    const ObjShape maxLocalShapeA = A.MaxShape();
    const ObjShape maxLocalShapeB = MaxShape();
    const ObjShape recvShape = prod(maxLocalShapeA) < prod(maxLocalShapeB) ? maxLocalShapeA : maxLocalShapeB;

    A2AUnpackData unpackData;
    unpackData.dataShape = LocalShape();
    unpackData.dataBufModeStrides = LocalStrides();

    unpackData.recvBufModeStrides = Dimensions2Strides(recvShape);
    unpackData.elemSlice1Stride = comm1LCM/gvB.Dimension(a2aMode1);
    unpackData.commMode1 = a2aMode1;

    unpackData.elemSlice2Stride = comm2LCM/gvB.Dimension(a2aMode2);
    unpackData.commMode2 = a2aMode2;

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
        for(i = 0; i < unpackData.elemSlice2Stride; i++){
            unpackElem[a2aMode2] = myFirstElemLoc[a2aMode2] + i*gvB.ModeWrapStride(a2aMode2);
            if(unpackElem[a2aMode2] >= Dimension(a2aMode2))
                continue;
            unpackData.elemSlice2 = i;

            for(j = 0; j < unpackData.elemSlice1Stride; j++){
                unpackElem[a2aMode1] = myFirstElemLoc[a2aMode1] + j*gvB.ModeWrapStride(a2aMode1);
                if(unpackElem[a2aMode1] >= Dimension(a2aMode1))
                    continue;
                unpackData.elemSlice1 = j;

                Location ownerA = A.DetermineOwner(unpackElem);
                Location ownerGridLoc = GridViewLoc2GridLoc(ownerA, gvA);
                Unsigned commLinLoc = Loc2LinearLoc(FilterVector(ownerGridLoc, commModesAll), FilterVector(g.Shape(), commModesAll));

//                PrintVector(unpackElem, "unpackElem");
//                std::cout << "startDataBufLoc: " << (i * a2aMode2Stride) + (j * a2aMode1Stride) << std::endl;
//                std::cout << "startRecvBufLoc: " << commLinLoc * nCommElemsPerProc << std::endl;
                UnpackA2ACommRecvBufHelper(unpackData, order - 1, &(recvBuf[commLinLoc * nCommElemsPerProc]), &(dataBuf[(i * a2aMode2Stride) + (j * a2aMode1Stride)]));
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
