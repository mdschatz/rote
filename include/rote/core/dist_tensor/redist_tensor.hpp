/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef ROTE_CORE_DISTTENSOR_DISTTENSOR_HPP
#define ROTE_CORE_DISTTENSOR_DISTTENSOR_HPP

namespace rote {

#ifndef RELEASE
    template<typename T>
    void AssertConforming2x1( const DistTensor<T>& AT, const DistTensor<T>& AB, Mode mode);
#endif

template<typename T>
class DistTensor: public DistTensorBase<T>
{
public:
	//Copying constructors for inheritence
    // Create a 0 distributed tensor
    DistTensor( const rote::Grid& g=DefaultGrid() );

    // Create a 0 distributed tensor
    DistTensor( const Unsigned order, const rote::Grid& g=DefaultGrid() );

    // Create a distributed tensor based on a supplied distribution
    DistTensor( const TensorDistribution& dist, const rote::Grid& g=DefaultGrid() );

    // Create a "shape" distributed tensor
    DistTensor
    ( const ObjShape& shape, const TensorDistribution& dist, const rote::Grid& g=DefaultGrid() );

    // Create a "shape" distributed tensor with specified alignments
    DistTensor
    ( const ObjShape& shape, const TensorDistribution& dist, const std::vector<Unsigned>& modeAligns, const rote::Grid& g );

    // Create a "shape" distributed tensor with specified alignments
    // and leading dimension
    DistTensor
    ( const ObjShape& shape, const TensorDistribution& dist, const std::vector<Unsigned>& modeAligns, const std::vector<Unsigned>& strides, const rote::Grid& g );

    // View a constant distributed tensor's buffer
    DistTensor
    ( const ObjShape& shape, const TensorDistribution& dist, const std::vector<Unsigned>& modeAligns,
      const T* buffer, const std::vector<Unsigned>& strides, const rote::Grid& g );

    // View a mutable distributed tensor's buffer
    DistTensor
    ( const ObjShape& shape, const TensorDistribution& dist, const std::vector<Unsigned>& modeAligns,
      T* buffer, const std::vector<Unsigned>& strides, const rote::Grid& g );

    //////////////////////////////////
    /// String distribution versions
    //////////////////////////////////

    // Create a distributed tensor based on a supplied distribution
    DistTensor( const std::string& dist, const rote::Grid& g=DefaultGrid() );

    // Create a "shape" distributed tensor
    DistTensor
    ( const ObjShape& shape, const std::string& dist, const rote::Grid& g=DefaultGrid() );

    // Create a "shape" distributed tensor with specified alignments
    DistTensor
    ( const ObjShape& shape, const std::string& dist, const std::vector<Unsigned>& modeAligns, const rote::Grid& g );

    // Create a "shape" distributed tensor with specified alignments
    // and leading dimension
    DistTensor
    ( const ObjShape& shape, const std::string& dist, const std::vector<Unsigned>& modeAligns, const std::vector<Unsigned>& strides, const rote::Grid& g );

    // View a constant distributed tensor's buffer
    DistTensor
    ( const ObjShape& shape, const std::string& dist, const std::vector<Unsigned>& modeAligns,
      const T* buffer, const std::vector<Unsigned>& strides, const rote::Grid& g );

    // View a mutable distributed tensor's buffer
    DistTensor
    ( const ObjShape& shape, const std::string& dist, const std::vector<Unsigned>& modeAligns,
      T* buffer, const std::vector<Unsigned>& strides, const rote::Grid& g );

    // Create a copy of distributed matrix A
//    DistTensor( const DistTensor<T>& A );

    ~DistTensor();


    ///////////////////////////////////////
    //
    // Redist interface routines
    //
    void RedistFrom(const DistTensor<T>& A, const ModeArray& reduceModes, const T alpha=T(1), const T beta=T(0));
    void RedistFrom(const DistTensor<T>& A);

    //
    // All-to-all interface routines
    //
    void AllToAllRedistFrom(const DistTensor<T>& A, const ModeArray& commModes, const T alpha=T(1), const T beta=T(0));

    //
    // Allgather interface routines
    //
    void AllGatherRedistFrom(const DistTensor<T>& A, const ModeArray& commModes, const T alpha=T(1), const T beta=T(0));

    //
    // Broadcast interface routines
    //
    void BroadcastRedistFrom(const DistTensor<T>& A, const ModeArray& commModes, const T alpha=T(1), const T beta=T(0));

    //
    // Gather-to-one interface routines
    //
    void GatherToOneRedistFrom(const DistTensor<T>& A, const ModeArray& commModes, const T alpha=T(1), const T beta=T(0));

    //
    // Local redist workhorse routines
    //
    bool CheckLocalCommRedist(const DistTensor<T>& A);
    void LocalCommRedist(const DistTensor<T>& A, const T alpha=T(1), const T beta=T(0));
    void UnpackLocalCommRecvBuf(const DistTensor<T>& A, const T* unpackBuf, const T alpha=T(1), const T beta=T(0));

    //
    // Local redist interface routines
    //
    void LocalRedistFrom(const DistTensor<T>& A, const T alpha=T(1), const T beta=T(0));

    //
    // Point-to-point interface routines
    //
    void PermutationRedistFrom(const DistTensor<T>& A, const ModeArray& redistModes, const T alpha=T(1), const T beta=T(0));

    //
    // Reduce Redist routine
    //
    void ReduceUpdateRedistFrom(const RedistType& redistType, const T alpha, const DistTensor<T>& A, const T beta, const ModeArray& reduceModes);

    //
    // AllReduce interface routines
    //
    void AllReduceUpdateRedistFrom(const T alpha, const DistTensor<T>& A, const T beta, const ModeArray& reduceModes);
    void AllReduceRedistFrom(const DistTensor<T>& A, const Mode reduceMode);
    void AllReduceUpdateRedistFrom(const T alpha, const DistTensor<T>& A, const T beta, const Mode reduceMode);
    void AllReduceRedistFrom(const DistTensor<T>& A, const ModeArray& reduceModes);

    //
    // Reduce-scatter interface routines
    //
    void ReduceScatterUpdateRedistFrom(const T alpha, const DistTensor<T>& A, const T beta, const ModeArray& reduceModes);
    void ReduceScatterRedistFrom(const DistTensor<T>& A, const Mode reduceMode);
    void ReduceScatterRedistFrom(const T alpha, const DistTensor<T>& A, const Mode reduceMode);
    void ReduceScatterUpdateRedistFrom(const T alpha, const DistTensor<T>& A, const T beta, const Mode reduceMode);
    void ReduceScatterRedistFrom(const T alpha, const DistTensor<T>& A, const ModeArray& reduceModes);
    void ReduceScatterRedistFrom(const DistTensor<T>& A, const ModeArray& reduceModes);
    void ReduceScatterUpdateRedistFrom(const DistTensor<T>& A, const ModeArray& reduceModes);

    //
    // Reduce-to-one interface routines
    //
    void ReduceToOneUpdateRedistFrom(T alpha, const DistTensor<T>& A, const T beta, const ModeArray& rModes);
    void ReduceToOneRedistFrom(const DistTensor<T>& A, const ModeArray& rModes);
    void ReduceToOneRedistFrom(const DistTensor<T>& A, const Mode rMode);

    //
    // Scatter interface routines
    //
    void ScatterRedistFrom(const DistTensor<T>& A, const ModeArray& commModes, const T alpha=T(1), const T beta=T(0));

private:
    //
    // All-to-all workhorse routines
    //
    bool CheckAllToAllCommRedist(const DistTensor<T>& A);
    void AllToAllCommRedist(const DistTensor<T>& A, const ModeArray& commModes, const T alpha=T(1), const T beta=T(0));
    void PackA2ACommSendBuf(const DistTensor<T>& A, const ModeArray& commModes, const ObjShape& sendShape, T * const sendBuf);
    void UnpackA2ACommRecvBuf(const T * const recvBuf, const ModeArray& commModes, const ObjShape& sendShape, const DistTensor<T>& A, const T alpha=T(0), const T beta=T(0));

    //
    // Allgather workhorse routines
    //
    bool CheckAllGatherCommRedist(const DistTensor<T>& A);
    void AllGatherCommRedist(const DistTensor<T>& A, const ModeArray& commModes, const T alpha=T(1), const T beta=T(0));
    void PackAGCommSendBuf(const DistTensor<T>& A, T * const sendBuf);

    //
    // Broadcast workhorse routines
    //
    bool CheckBroadcastCommRedist(const DistTensor<T>& A);
    void BroadcastCommRedist(const DistTensor<T>& A, const ModeArray& commModes, const T alpha=T(1), const T beta=T(0));

    //
    // Gather-to-one workhorse routines
    //
    bool CheckGatherToOneCommRedist(const DistTensor<T>& A);
    void GatherToOneCommRedist(const DistTensor<T>& A, const ModeArray& commModes, const T alpha=T(1), const T beta=T(0));

    //
    // Point-to-point workhorse routines
    //
    bool CheckPermutationCommRedist(const DistTensor<T>& A);
    void PermutationCommRedist(const DistTensor<T>& A, const ModeArray& redistModes, const T alpha=T(1), const T beta=T(0));
    void UnpackPCommRecvBuf(const T* const recvBuf, const T alpha=T(1), const T beta=T(0));

    //
    // AllReduce workhorse routines
    //
    bool CheckAllReduceCommRedist(const DistTensor<T>& A);
    void AllReduceUpdateCommRedist(const T alpha, const DistTensor<T>& A, const T beta, const ModeArray& commModes);
    void PackARCommSendBuf(const DistTensor<T>& A, const ModeArray& reduceModes, const ModeArray& commModes, T * const sendBuf);
    void UnpackARUCommRecvBuf(const T* const recvBuf, const T alpha, const DistTensor<T>& A, const T beta);

    //
    // Reduce-scatter workhorse routines
    //
    bool CheckReduceScatterCommRedist(const DistTensor<T>& A);
    void ReduceScatterUpdateCommRedist(const T alpha, const DistTensor<T>& A, const T beta, const ModeArray& reduceModes, const ModeArray& commModes);
    void PackRSCommSendBuf(const DistTensor<T>& A, const ModeArray& reduceModes, const ModeArray& commModes, T * const sendBuf);
    void UnpackRSUCommRecvBuf(const T* const recvBuf, const T alpha, const T beta);

    //
    // Reduce-to-one workhorse routines
    //
    bool CheckReduceToOneCommRedist(const DistTensor<T>& A);
    void ReduceToOneUpdateCommRedist(const T alpha, const DistTensor<T>& A, const T beta, const ModeArray& commModes);

    //
    // Scatter workhorse routines
    //
    bool CheckScatterCommRedist(const DistTensor<T>& A);
    void ScatterCommRedist(const DistTensor<T>& A, const ModeArray& commModes, const T alpha=T(1), const T beta=T(0));

    bool AlignCommBufRedist(const DistTensor<T>& A, const T* unalignedSendBuf, const Unsigned sendSize, T* alignedSendBuf, const Unsigned recvSize);

};

} // namespace rote

#endif // ifndef ROTE_CORE_DISTTENSOR_DISTTENSOR_HPP
