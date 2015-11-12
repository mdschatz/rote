/*
   Copyright (c) 2009-2015, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#include "rote.hpp"

//#include "./Read/Ascii.hpp"
//#include "./Read/AsciiMatlab.hpp"
//#include "./Read/Binary.hpp"
//#include "./Read/BinaryFlat.hpp"

namespace rote {

template<typename T>
void Read( Tensor<T>& A, const std::string filename, FileFormat format )
{
    if( format == AUTO )
        format = DetectFormat( filename );

    switch( format )
    {
//    case ASCII:
//        read::Ascii( A, filename );
//        break;
    default:
        LogicError("Format unsupported for reading");
    }
}

template<typename T>
inline void
ReadBinarySeqPack(const DistTensor<T>& A, const ObjShape& packShape, const ObjShape& commGridViewShape, const Unsigned& nElemsPerProc, const Location& fileLoc, const ObjShape& gblDataShape, std::ifstream& fileStream, T* sendBuf)
{
    Unsigned order = packShape.size();

    //Info about the data tensor we're reading
    Unsigned gblDataPtr = 0;
    Location gblDataLoc = fileLoc;

    //Info about how much data we have packed that we possibly can pack (memory constraint)
    Unsigned packedPtr = 0;
    Location packedShape(order, 0);

    Location procLoc(order, 0);
    std::vector<Unsigned> nElemsPackedPerProc(prod(commGridViewShape), 0);

    bool done = !ElemwiseLessThan(gblDataLoc, gblDataShape);

    while(!done){
        T value;
        fileStream.read((char*)&value, sizeof(T));
        Location procGVLoc = A.DetermineOwner(gblDataLoc);

        Unsigned whichProc = GridViewLoc2ParticipatingLinearLoc(procGVLoc, A.GetGridView());
        sendBuf[whichProc*nElemsPerProc + nElemsPackedPerProc[whichProc]] = value;
//        PrintVector(gblDataLoc, "packing gblDataLoc");
//        printf("packing to loc: %d\n", whichProc*nElemsPerProc + nElemsPackedPerProc[whichProc]);


        //Update
        nElemsPackedPerProc[whichProc]++;
        gblDataLoc[gblDataPtr]++;
        packedShape[packedPtr]++;

        //Update the pack counter
        while (packedPtr < packShape.size() && packedShape[packedPtr] >= packShape[packedPtr]) {
            packedShape[packedPtr] = 0;

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
//        PrintVector(packedShape, "packedShape after update");
//        PrintVector(packShape, "Pack packShape after update");
        if (done)
            break;

        //Update the counters
        while (gblDataPtr < order && gblDataLoc[gblDataPtr] >= gblDataShape[gblDataPtr]) {
            gblDataLoc[gblDataPtr] = 0;

            gblDataPtr++;
            if (gblDataPtr >= order) {
                done = true;
                break;
            } else {
                gblDataLoc[gblDataPtr]++;
            }
        }
//        PrintVector(gblDataLoc, "gblDataLoc after update");
//        PrintVector(gblDataShape, "gblDataShape");

        if (done)
            break;
        gblDataPtr = 0;
        packedPtr = 0;
    }
}

template<typename T>
inline void
ReadAsciiSeqPack(const DistTensor<T>& A, const ObjShape& packShape, const ObjShape& commGridViewShape, const Unsigned& nElemsPerProc, const Location& fileLoc, const ObjShape& gblDataShape, std::stringstream& dataStream, T* sendBuf)
{
    Unsigned order = packShape.size();

    //Info about the data tensor we're reading
    Unsigned gblDataPtr = 0;
    Location gblDataLoc = fileLoc;

    //Info about how much data we have packed that we possibly can pack (memory constraint)
    Unsigned packedPtr = 0;
    Location packedShape(order, 0);

    Location procLoc(order, 0);
    std::vector<Unsigned> nElemsPackedPerProc(prod(commGridViewShape), 0);

    bool done = !ElemwiseLessThan(gblDataLoc, gblDataShape);

    while(!done){
        T value;
        dataStream >> value;
        Location procGVLoc = A.DetermineOwner(gblDataLoc);
        Unsigned whichProc = GridViewLoc2ParticipatingLinearLoc(procGVLoc, A.GetGridView());
        sendBuf[whichProc*nElemsPerProc + nElemsPackedPerProc[whichProc]] = value;

        //Update
        nElemsPackedPerProc[whichProc]++;
        gblDataLoc[gblDataPtr]++;
        packedShape[packedPtr]++;

        //Update the pack counter
        while (packedPtr < packShape.size() && packedShape[packedPtr] >= packShape[packedPtr]) {
            packedShape[packedPtr] = 0;

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
ReadSeqUnpack(DistTensor<T>& A, const Location& firstGblLoc, const ObjShape& recvShape, const T* recvBuf){
    //packetShape.size() always less than A.Order()
    Unsigned order = recvShape.size();
    Location firstLocalLocUnpack = A.Global2LocalIndex(firstGblLoc);
    //TODO:  Respect permutations...
    T* dstBuf = A.Buffer(firstLocalLocUnpack);
    ObjShape localShape = A.LocalShape();

    Unsigned dstBufPtr = 0;
    std::vector<Unsigned> dstBufStrides = A.LocalStrides();
    Unsigned recvBufPtr = 0;
    std::vector<Unsigned> recvBufStrides = Dimensions2Strides(recvShape);

    Unsigned localUnpackPtr = 0;
    Unsigned recvPtr = 0;
    Location localUnpackLoc = firstLocalLocUnpack;
    Location recvLoc(recvShape.size(), 0);

    bool done = !(ElemwiseLessThan(localUnpackLoc, localShape) && ElemwiseLessThan(recvLoc, recvShape));

    //TODO: Detect "nice" strides and use MemCopy
    //NOTE: This is basically a copy of PackCommHelper routine
    //      but modified as the termination condition depends
    //      on both the packShape and the local shape

    while (!done) {
        dstBuf[dstBufPtr] = recvBuf[recvBufPtr];
//        PrintVector(localUnpackLoc, "localUnpackLoc");
//        PrintVector(recvLoc, "recvLoc");
        //Update
        localUnpackLoc[localUnpackPtr]++;
        recvLoc[recvPtr]++;

        dstBufPtr += dstBufStrides[0];
        recvBufPtr += recvBufStrides[0];

        while (recvPtr < order && recvLoc[recvPtr] >= recvShape[recvPtr]) {
            recvLoc[recvPtr] = 0;

            recvBufPtr -= recvBufStrides[recvPtr] * recvShape[recvPtr];
            recvPtr++;
            if (recvPtr >= order) {
                done = true;
                break;
            } else {
                recvLoc[recvPtr]++;
                recvBufPtr += recvBufStrides[recvPtr];
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
void
ReadSeq(DistTensor<T>& A, const std::string filename, FileFormat format)
{
    std::ifstream file;
    switch(format){
        case ASCII_MATLAB:
        case ASCII: file.open(filename.c_str()); break;
        case BINARY_FLAT:
        case BINARY: file.open(filename.c_str(), std::ios::binary); break;
        default: LogicError("Unsupported distributed read format");
    }
    if( !file.is_open() ){
        std::string msg = "Could not open " + filename;
        RuntimeError(msg);
    }
    //Get the shape of the object we are trying to fill
    ObjShape dataShape;
    if(format == ASCII_MATLAB || format == ASCII){
        std::string line;
        std::getline(file, line);
        std::stringstream dataShapeStream(line);
        Unsigned value;
        while( dataShapeStream >> value ) dataShape.push_back(value);
        A.ResizeTo(dataShape);
    }else if(format == BINARY){
        //Ignore tensor order?
        Unsigned i;
        Unsigned order;
        file.read( (char*)&order, sizeof(Unsigned) );
        dataShape.resize(order);

        for(i = 0; i < order; i++)
            file.read( (char*)&(dataShape[i]), sizeof(Unsigned));
        A.ResizeTo(dataShape);
    }else if(format == BINARY_FLAT){
        dataShape = A.Shape();
    }

    Unsigned i;
    Unsigned order = A.Order();
    const rote::GridView& gvA = A.GetGridView();

    //Determine the max shape we can pack with available memory
    //Figure out the first mode that a packet does not fully pack
    ObjShape packetShape(A.Shape());
    Unsigned firstPartialPackMode = order - 1;

    //MAX_ELEM_PER_PROC must account for entire send buffer size
    Unsigned remainder = MaxLength(MAX_ELEM_PER_PROC, prod(gvA.ParticipatingShape()));
    Unsigned readStride = 1;
    for(i = 0; i < order; i++){
        if(remainder < (A.Dimension(i) * readStride)){
            packetShape[i] = Max(1, remainder / readStride);
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

//    PrintVector(gvA.ParticipatingShape(), "gvA.ParticipatingShape()");
//    PrintVector(A.Shape(), "A.Shape()");
//    PrintVector(packetShape, "packetShape");
    //Get the subset of processes involved in the communication
    ObjShape gvAShape = gvA.ParticipatingShape();
    Unsigned nCommProcs = prod(gvAShape);

    //Determine the tensor shape we will send to each process
    ObjShape sendShape = MaxLengths(packetShape, gvAShape);
//    PrintVector(sendShape, "sendShape");

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
    SortVector(sortedCommModes);
    mpi::Comm comm = A.GetCommunicatorForModes(sortedCommModes, A.Grid());

    //Perform the read
    Location dataLoc(order, 0);
    Unsigned ptr = firstPartialPackMode;

    std::vector<Unsigned> sendBufStrides = Dimensions2Strides(dataShape);

    //For the case that we're dealing with ASCII, read into a stringstream
    std::stringstream dataStream;
    std::string line;

    if(format == ASCII || format == ASCII_MATLAB){
        std::getline(file, line);
        dataStream.str(line);
    }

    bool done = !ElemwiseLessThan(dataLoc, dataShape);

    Location myFirstGblElemLocUnpack;
    Location firstLocalElemLocUnpack;
    Location myPackGridViewLoc = gvA.ParticipatingLoc();

    while(!done){
        if(A.Grid().LinearRank() == 0){
            MemZero(&(sendBuf[0]), prod(sendShape) * nCommProcs);
            if(format == ASCII || format == ASCII_MATLAB){
                ReadAsciiSeqPack(A, packetShape, gvAShape, prod(sendShape), dataLoc, dataShape, dataStream, &(sendBuf[0]));
            }else{
                ReadBinarySeqPack(A, packetShape, gvAShape, prod(sendShape), dataLoc, dataShape, file, &(sendBuf[0]));
            }
            ObjShape commShape = sendShape;
            commShape.insert(commShape.end(), nCommProcs);
        }

        //Communicate the data
//        ObjShape sendBufShape = sendShape;
//        sendBufShape.push_back(nCommProcs);
//        PrintArray(sendBuf, sendBufShape, "sendBuf");

//        if(gvA.LinearRank() == 0)
//            printf("scattering on read\n");
        mpi::Scatter(sendBuf, prod(sendShape), recvBuf, prod(sendShape), 0, comm);

//        PrintArray(recvBuf, sendShape, "recvBuf");

        //Unpack it
        Location packLastLoc = dataLoc;
        for(i = 0; i < firstPartialPackMode + 1; i++)
            packLastLoc[i] += packetShape[i] - 1;

        //Determine the first element I own
        Location owner = A.DetermineOwner(dataLoc);
        myFirstGblElemLocUnpack = ElemwiseSum(dataLoc, ElemwiseSubtract(myPackGridViewLoc, owner));

        if(ElemwiseLessThan(myFirstGblElemLocUnpack, dataShape) && ElemwiseLessThanEqualTo(dataLoc, myFirstGblElemLocUnpack) && !AnyElemwiseGreaterThan(myFirstGblElemLocUnpack, packLastLoc)){
            ReadSeqUnpack(A, myFirstGblElemLocUnpack, sendShape, recvBuf);
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

template<typename T>
void
ReadNonSeq(DistTensor<T>& A, const std::string filename, FileFormat format){
    std::ifstream file;
    switch(format){
        case ASCII_MATLAB:
        case ASCII: file.open(filename.c_str()); break;
        case BINARY_FLAT:
        case BINARY: file.open(filename.c_str(), std::ios::binary); break;
        default: LogicError("Unsupported distributed read format");
    }
    if( !file.is_open() ){
        std::string msg = "Could not open " + filename;
        RuntimeError(msg);
    }
    //Get the shape of the object we are trying to fill
    ObjShape dataShape;
    if(format == ASCII_MATLAB || format == ASCII){
        std::string line;
        std::getline(file, line);
        std::stringstream dataShapeStream(line);
        Unsigned value;
        while( dataShapeStream >> value ) dataShape.push_back(value);
        A.ResizeTo(dataShape);
    }else if(format == BINARY){
        //Ignore tensor order?
        Unsigned i;
        Unsigned order;
        file.read( (char*)&order, sizeof(Unsigned) );

        dataShape.resize(order);

        for(i = 0; i < order; i++)
            file.read( (char*)&(dataShape[i]), sizeof(Unsigned));
        A.ResizeTo(dataShape);
    }
    Unsigned i;
    Unsigned order = A.Order();
    const rote::GridView& gvA = A.GetGridView();

    ObjShape shapeA = A.Shape();
    Location myLoc = gvA.ParticipatingLoc();
    ObjShape gvAShape = gvA.ParticipatingShape();
    std::vector<Unsigned> readInc = gvAShape;
    std::vector<Unsigned> loopSpace = ElemwiseSubtract(shapeA, myLoc);
    std::vector<Unsigned> loopIters = MaxLengths(loopSpace, gvAShape);

    Unsigned startLinLoc = Loc2LinearLoc(myLoc, shapeA);
    std::vector<Unsigned> srcBufStrides = Dimensions2Strides(A.Shape());
    std::vector<Unsigned> dstBufStrides = A.LocalStrides();
    T* dstBuf = A.Buffer();

    std::string line;
    std::stringstream dataStream;
    if(format == ASCII_MATLAB || format == ASCII){
        std::getline(file, line);
        dataStream.str(line);
        T value;
        for(i = 0; i < startLinLoc; i++)
            dataStream >> value;
    }else if(format == BINARY){
        file.seekg( ((1 + order) * sizeof(Unsigned)) + (startLinLoc * sizeof(T)));
    }else if(format == BINARY_FLAT){
        file.seekg( startLinLoc * sizeof(T));
    }
    Unsigned ptr = 0;
    Unsigned dstBufPtr = 0;

    Location curLoc = myLoc;
    Unsigned srcBufPtr = startLinLoc;
    Unsigned newSrcBufPtr = 0;

    bool done = !ElemwiseLessThan(curLoc, A.Shape());

//    printf("order: %d\n", order);
//    PrintVector(myLoc, "myLoc");
//    PrintVector(readInc, "readInc");
//    PrintVector(loopIters, "loopIters");
//    PrintVector(srcBufStrides, "srcBufStrides"); 
//    PrintVector(dstBufStrides, "dstBufStrides");
    while(!done){
        T value;
        if(format == ASCII_MATLAB || format == ASCII){
            dataStream >> value;
        }else{
            file.read((char*)&value, sizeof(T));
        }
//        PrintVector(curLoc, "curLoc");
//        printf("read val: %.3f\n", value);
        dstBuf[dstBufPtr] = value;

        //Update
        curLoc[ptr] += readInc[ptr];
        dstBufPtr += dstBufStrides[ptr];
        newSrcBufPtr = srcBufPtr;
        newSrcBufPtr += srcBufStrides[ptr] * readInc[ptr];

        while (ptr < order && curLoc[ptr] >= shapeA[ptr]) {
//            PrintVector(curLoc, "curLoc before reset");
            curLoc[ptr] = myLoc[ptr];
//            PrintVector(curLoc, "curLoc after reset");
//            printf("newSrcBufPtr before reset: %d\n", newSrcBufPtr);
            newSrcBufPtr -= srcBufStrides[ptr] * readInc[ptr] * loopIters[ptr];
//            printf("newSrcBufPtr after reset: %d\n", newSrcBufPtr);
            ptr++;

            if (ptr >= order) {
                done = true;
                break;
            } else {
                curLoc[ptr] += readInc[ptr];
//                PrintVector(curLoc, "curLoc after inc");
                newSrcBufPtr += srcBufStrides[ptr] * readInc[ptr];
//                printf("newSrcBufPtr after inc: %d\n", newSrcBufPtr);
            }
        }
        if (done)
            break;
        ptr = 0;
        //Adjust streams (we read in 1 value, so adjust by -1 )
        Unsigned adjustAmount = newSrcBufPtr - srcBufPtr - 1;
//        printf("newSrcBufPtr: %d, oldSrcBufPtr: %d\n", newSrcBufPtr, srcBufPtr);
        if(format == ASCII_MATLAB || format == ASCII){
            T val;
            for(i = 0; i < adjustAmount; i++)
                dataStream >> val;
        }else if(format == BINARY){
            file.seekg(((1 + order) * sizeof(Unsigned)) + (newSrcBufPtr * sizeof(T)));
        }else if(format == BINARY_FLAT){
            file.seekg((newSrcBufPtr * sizeof(T)));
        }
        srcBufPtr = newSrcBufPtr;
    }

}

template<typename T>
void Read
( DistTensor<T>& A, const std::string filename, FileFormat format,
  bool sequential )
{
    if( format == AUTO )
        format = DetectFormat( filename ); 

    //Everyone accesses data
    if(!sequential)
    {
        ReadNonSeq(A, filename, format);
    }
    //Only root process accesses data
    else
    {
        ReadSeq( A, filename, format );
    }

}

#define FULL(T) \
  template void Read \
  ( Tensor<T>& A, const std::string filename, FileFormat format ); \
  template void Read \
  ( DistTensor<T>& A, const std::string filename, \
    FileFormat format, bool sequential );

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
