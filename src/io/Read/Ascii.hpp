/*
   Copyright (c) 2009-2015, Jack Poulson
   All rights reserved.

   This file is part of tmenemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef TMEN_READ_ASCII_HPP
#define TMEN_READ_ASCII_HPP

#include "tensormental.hpp"

namespace tmen {
namespace read {

template<typename T>
inline void
Ascii( Tensor<T>& A, const std::string filename )
{
    LogicError("Not implemented");
//    std::ifstream file( filename.c_str() );
//    if( !file.is_open() ){
//        std::string msg = "Could not open " + filename;
//        RuntimeError(msg);
//    }
//
//    // Walk through the file once to both count the number of rows and
//    // columns and to ensure that the number of columns is consistent
//    Int height=0, width=0;
//    std::string line;
//    while( std::getline( file, line ) )
//    {
//        std::stringstream lineStream( line );
//        Int numCols=0;
//        T value;
//        while( lineStream >> value ) ++numCols;
//        if( numCols != 0 )
//        {
//            if( numCols != width && width != 0 )
//                LogicError("Inconsistent number of columns");
//            else
//                width = numCols;
//            ++height;
//        }
//    }
//    file.clear();
//    file.seekg(0,file.beg);
//
//    // Resize the matrix and then read it
//    A.Resize( height, width );
//    Int i=0;
//    while( std::getline( file, line ) )
//    {
//        std::stringstream lineStream( line );
//        Int j=0;
//        T value;
//        while( lineStream >> value )
//        {
//            A.Set( i, j, value );
//            ++j;
//        }
//        ++i;
//    }
}

template<typename T>
inline void
Ascii( DistTensor<T>& A, const std::string filename )
{
    LogicError("Not implemented");
//    std::ifstream file( filename.c_str() );
//    if( !file.is_open() ){
//        std::string msg = "Could not open " + filename;
//        RuntimeError(msg);
//    }
//
//    // Walk through the file once to both count the number of rows and
//    // columns and to ensure that the number of columns is consistent
//    Int height=0, width=0;
//    std::string line;
//    while( std::getline( file, line ) )
//    {
//        std::stringstream lineStream( line );
//        Int numCols=0;
//        T value;
//        while( lineStream >> value ) ++numCols;
//        if( numCols != 0 )
//        {
//            if( numCols != width && width != 0 )
//                LogicError("Inconsistent number of columns");
//            else
//                width = numCols;
//            ++height;
//        }
//    }
//    file.clear();
//    file.seekg(0,file.beg);
//
//    // Resize the matrix and then read in our local portion
//    A.Resize( height, width );
//    Int i=0;
//    while( std::getline( file, line ) )
//    {
//        std::stringstream lineStream( line );
//        Int j=0;
//        T value;
//        while( lineStream >> value )
//        {
//            A.Set( i, j, value );
//            ++j;
//        }
//        ++i;
//    }
}

template<typename T>
inline void
AsciiSeqPack(const DistTensor<T>& A, const ObjShape& packShape, const ObjShape& commGridViewShape, const Unsigned& nElemsPerProc, const Location& fileLoc, const ObjShape& gblDataShape, std::stringstream& dataStream, T* sendBuf)
{
    Unsigned order = packShape.size();

    //Info about the data tensor we're reading
    Unsigned gblDataPtr = 0;
    Location gblDataLoc = fileLoc;

    //Info about how much data we have packed that we possibly can pack (memory constraint)
    Unsigned packedPtr = 0;
    Location packedShape(order, 0);

    //Info about which process we should be packing.
    Unsigned procPtr = 0;
    Unsigned whichProc = 0;

    Location procLoc(order, 0);
    std::vector<Unsigned> nElemsPackedPerProc(prod(commGridViewShape), 0);

    bool done = !ElemwiseLessThan(gblDataLoc, gblDataShape);

//    PrintVector(packShape, "packShape");
    while(!done){
//        PrintVector(gblDataLoc, "gblDataLoc");
//        PrintVector(packedShape, "packedShape");
        T value;
        dataStream >> value;
//        printf("new val: %.3f\n", value);
        Location procGVLoc = A.DetermineOwner(gblDataLoc);
//        PrintVector(procGVLoc, "procGVLoc");
        Unsigned whichProc = GridViewLoc2ParticipatingLinearLoc(procGVLoc, A.GetGridView());
//        PrintVector(procGVLoc, "procLoc");
//        printf("copying to proc: %d with offset: %d at location: %d\n", whichProc, whichProc*nElemsPerProc, nElemsPackedPerProc[whichProc]);
        sendBuf[whichProc*nElemsPerProc + nElemsPackedPerProc[whichProc]] = value;
//        PrintArray(sendBuf, packShape, "current sendBuf");

        //Update
        nElemsPackedPerProc[whichProc]++;
        gblDataLoc[gblDataPtr]++;
        packedShape[packedPtr]++;

        //Update the pack counter
        while (packedPtr < packShape.size() && packedShape[packedPtr] >= packShape[packedPtr]) {
            packedShape[gblDataPtr] = 0;

            packedPtr++;
            //packShape[i] can only be either gblDataShape[i] or < gblDataShape[i]
            //if packShape[i] !=0 when gblDataShape[i] == 0
            //we've run off the edge
            if (packedPtr >= packShape.size()) {
                done = true;
                break;
            } else {
                packedShape[packedPtr]++;
            }
        }
        if (done)
            break;

        //Update the counters
        while (gblDataPtr < order && gblDataLoc[gblDataPtr] >= gblDataShape[gblDataPtr]) {
            //packShape[i] can only be either gblDataShape[i] or < gblDataShape[i]
            //if packShape[i] !=0 when resetting gblDataShape[i]
            //we've run off the edge
            if(gblDataPtr < packShape.size() && packedShape[gblDataPtr] != 0){
                done = true;
                break;
            }
            gblDataLoc[gblDataPtr] = 0;

            gblDataPtr++;
            if (gblDataPtr >= order) {
                done = true;
                break;
            } else {
                gblDataLoc[gblDataPtr]++;
            }
        }
        if (done)
            break;
        gblDataPtr = 0;
        packedPtr = 0;
    }
}

template<typename T>
inline void
AsciiSeqUnpack(DistTensor<T>& A, const Location& firstGblLoc, const ObjShape& packetShape, const T* recvBuf){
    //packetShape.size() always less than A.Order()
    Unsigned order = packetShape.size();
    Location firstLocalLocUnpack = A.Global2LocalIndex(firstGblLoc);
    //TODO:  Respect permutations...
    T* dstBuf = A.Buffer(firstLocalLocUnpack);
    ObjShape localShape = A.LocalShape();

    Unsigned dstBufPtr = 0;
    std::vector<Unsigned> dstBufStrides = A.LocalStrides();
    Unsigned recvBufPtr = 0;
    std::vector<Unsigned> recvBufStrides = Dimensions2Strides(packetShape);

    Unsigned localUnpackPtr = 0;
    Unsigned recvPtr = 0;
    Location localUnpackLoc = firstLocalLocUnpack;
    Location recvLoc(packetShape.size(), 0);

    bool done = !(ElemwiseLessThan(localUnpackLoc, localShape) && ElemwiseLessThan(recvLoc, packetShape));

    //TODO: Detect "nice" strides and use MemCopy
    //NOTE: This is basically a copy of PackCommHelper routine
    //      but modified as the termination condition depends
    //      on both the packShape and the local shape
//    PrintVector(packetShape, "packetShape");
//    PrintVector(localShape, "localShape");
    while (!done) {
        PrintVector(localUnpackLoc, "localLoc");
        PrintVector(recvLoc, "recvLoc");
        dstBuf[dstBufPtr] = recvBuf[recvBufPtr];
        printf("recvBuf val: %.3f\n", recvBuf[recvBufPtr]);
        printf("copying to location: %d\n", dstBufPtr);
        //Update
        localUnpackLoc[localUnpackPtr]++;
        recvLoc[recvPtr]++;

        dstBufPtr += dstBufStrides[0];
        recvBufPtr += recvBufStrides[0];

        while (recvPtr < order && recvLoc[recvPtr] >= packetShape[recvPtr]) {
            recvLoc[recvPtr] = 0;

            recvBufPtr -= recvBufStrides[recvPtr] * packetShape[recvPtr];
            recvPtr++;
            if (recvPtr >= order) {
                done = true;
                break;
            } else {
                recvLoc[recvPtr]++;
                recvBufPtr += recvBufStrides[localUnpackPtr];
            }
        }
        if (done)
            break;
        recvPtr = 0;

        while (localUnpackPtr < order && localUnpackLoc[localUnpackPtr] >= localShape[localUnpackPtr]) {
            localUnpackLoc[localUnpackPtr] = firstLocalLocUnpack[localUnpackPtr];

            dstBufPtr -= dstBufStrides[localUnpackPtr] * localShape[localUnpackPtr];
            localUnpackPtr++;
            if (localUnpackPtr >= order) {
                done = true;
                break;
            } else {
                localUnpackLoc[localUnpackPtr]++;
                dstBufPtr += dstBufStrides[localUnpackPtr];
            }
        }
        if (done)
            break;
        localUnpackPtr = 0;
    }
}

template<typename T>
inline void
AsciiSeq(DistTensor<T>& A, const std::string filename)
{
    std::ifstream file( filename.c_str() );
    if( !file.is_open() ){
        std::string msg = "Could not open " + filename;
        RuntimeError(msg);
    }
    //Get the shape of the object we are trying to fill
    std::string line;
    std::getline(file, line);
    std::stringstream dataShapeStream(line);

    ObjShape dataShape;
    Unsigned value;
    while( dataShapeStream >> value ) dataShape.push_back(value);

    A.ResizeTo(dataShape);
    //End shape getting

    Unsigned i;
    Unsigned order = A.Order();
    const tmen::GridView& gvA = A.GetGridView();

    //Determine the max shape we can pack with available memory
    //Figure out the first mode that a packet does not fully pack
    ObjShape packetShape(A.Shape());
    PrintVector(A.Shape(), "shapeA");
    Unsigned firstPartialPackMode = order - 1;

    //MAX_ELEM_PER_PROC must account for entire send buffer size
    Unsigned remainder = MaxLength(MAX_ELEM_PER_PROC, prod(gvA.ParticipatingShape()));
    Unsigned readStride = 1;
    for(i = 0; i < order; i++){
        printf("remainder: %d\n", remainder);
        printf("read stride: %d\n", readStride);
        PrintVector(packetShape, "gblSendShape");

        if(remainder < (A.Dimension(i) * readStride)){
            packetShape[i] = Max(1, remainder);
        }

        if(remainder == 0 && firstPartialPackMode == order - 1){
            firstPartialPackMode = i - 1;
        }

        Unsigned testRemainder = remainder - packetShape[i] * readStride;
        if(testRemainder > remainder)
            remainder = 0;
        else
            remainder = testRemainder;
        readStride *= packetShape[i];
    }

    //Get the subset of processes involved in the communication
    ObjShape gvAShape = gvA.ParticipatingShape();
    Unsigned nCommProcs = prod(gvAShape);

    //Determine the tensor shape we will send to each process
    PrintVector(packetShape, "gblSendShape");
    PrintVector(gvAShape, "commGridViewShape");
    PrintVector(packetShape, "packetShape");
    ObjShape sendShape = MaxLengths(packetShape, gvAShape);
    PrintVector(sendShape, "sendShape");
    Unsigned nElemsPerProc = prod(sendShape);

    T* auxBuf = new T[prod(sendShape) * (nCommProcs + 1)];
//    T* auxBuf = A.auxMemory_.Require(prod(sendShape) * (nCommProcs + 1));
    T* sendBuf = &(auxBuf[prod(sendShape)]);
    T* recvBuf = &(auxBuf[0]);

    //Set up the communicator information, including
    //the permutation from TensorDist proc order -> comm proc order
    ModeArray commModes;
    for(i = 0; i < A.Order(); i++){
        ModeDistribution modeDist = A.ModeDist(i);
        commModes.insert(commModes.end(), modeDist.begin(), modeDist.end());
    }
    ModeArray sortedCommModes = commModes;
    std::sort(sortedCommModes.begin(), sortedCommModes.end());
    mpi::Comm comm = A.GetCommunicatorForModes(sortedCommModes, A.Grid());

    //Perform the read
    Location dataLoc(order, 0);
    Unsigned ptr = firstPartialPackMode;

    Unsigned sendBufPtr = 0;
    std::vector<Unsigned> sendBufStrides = Dimensions2Strides(dataShape);

    std::getline(file, line);
    std::stringstream dataStream(line);

    bool done = !ElemwiseLessThan(dataLoc, dataShape);

    Location myFirstGblElemLocUnpack;
    Location firstLocalElemLocUnpack;
    Location myPackGridViewLoc = gvA.ParticipatingLoc();
    PrintVector(myPackGridViewLoc, "myPackGridViewLoc");
//    for(i = firstPartialPackMode + 1; i < A.Order(); i++)
//        myPackGridViewLoc[i] = 0;

    printf("ping2\n");
    while(!done){
        printf("ping2.1\n");
        PrintVector(dataLoc, "dataLoc");
        if(A.Grid().LinearRank() == 0){
            MemZero(&(sendBuf[0]), prod(sendShape) * nCommProcs);
            AsciiSeqPack(A, packetShape, gvAShape, prod(sendShape), dataLoc, dataShape, dataStream, &(sendBuf[0]));
            ObjShape commShape = sendShape;
            commShape.insert(commShape.end(), nCommProcs);
            PrintArray(sendBuf, commShape, "comm sendBuf");
        }
        printf("ping2.2\n");

        //Communicate the data
        mpi::Scatter(sendBuf, prod(sendShape), recvBuf, prod(sendShape), 0, comm);

        printf("UNPACKING\n");
        PrintArray(recvBuf, sendShape, "recvBuf");

        //Unpack it
        Location packLastLoc = dataLoc;
        for(i = 0; i < firstPartialPackMode + 1; i++)
            packLastLoc[i] += packetShape[i] - 1;

        //Determine the first element I own
        Location owner = A.DetermineOwner(dataLoc);
        myFirstGblElemLocUnpack = ElemwiseSum(dataLoc, ElemwiseSubtract(myPackGridViewLoc, owner));

        PrintVector(packLastLoc, "packLastLoc");
        PrintVector(myFirstGblElemLocUnpack, "firstGblElemLocUnpack");

        if(ElemwiseLessThan(myFirstGblElemLocUnpack, dataShape) && ElemwiseLessThanEqualTo(dataLoc, myFirstGblElemLocUnpack) && !AnyElemwiseGreaterThan(myFirstGblElemLocUnpack, packLastLoc)){
            AsciiSeqUnpack(A, myFirstGblElemLocUnpack, sendShape, recvBuf);
            PrintArray(A.Buffer(), A.LocalShape(), "dataBuf");
        }

        //Update
        dataLoc[ptr] += packetShape[ptr];

        while (ptr < order && dataLoc[ptr] >= dataShape[ptr]) {
            dataLoc[ptr] = 0;

            ptr++;
            if (ptr >= order) {
                done = true;
                break;
            } else {
                dataLoc[ptr]++;
            }
        }
        if (done)
            break;
        ptr = firstPartialPackMode;
    }
    delete [] auxBuf;
//    A.auxMemory_.Release();

}

} // namespace read
} // namespace tmen

#endif // ifndef TMEN_READ_ASCII_HPP
