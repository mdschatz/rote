/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef TMEN_CORE_DISTTENSOR_DECL_HPP
#define TMEN_CORE_DISTTENSOR_DECL_HPP

#include<vector>
namespace tmen {

#ifndef RELEASE
    template<typename T>
    void AssertConforming2x1( const DistTensor<T>& AT, const DistTensor<T>& AB, Mode mode);
#endif

template<typename T>
class DistTensor
{
public:

#ifndef RELEASE
    void AssertNotLocked() const;
    void AssertNotStoringData() const;
    void AssertValidEntry( const Location& loc ) const;
    void AssertValidSubtensor
    ( const Location& loc, const ObjShape& shape ) const;
    void AssertSameGrid( const tmen::Grid& grid ) const;
    void AssertSameSize( const ObjShape& shape ) const;
    //TODO: REMOVE THIS
    void AssertMergeableModes(const std::vector<ModeArray>& oldModes) const;
#endif // ifndef RELEASE

    //Constructors

    void ClearCommMap();
    Unsigned CommMapSize();
    // Create a 0 distributed tensor
    DistTensor( const tmen::Grid& g=DefaultGrid() );

    // Create a 0 distributed tensor
    DistTensor( const Unsigned order, const tmen::Grid& g=DefaultGrid() );

    // Create a distributed tensor based on a supplied distribution
    DistTensor( const TensorDistribution& dist, const tmen::Grid& g=DefaultGrid() );

    // Create a "shape" distributed tensor
    DistTensor
    ( const ObjShape& shape, const TensorDistribution& dist, const tmen::Grid& g=DefaultGrid() );

    // Create a "shape" distributed tensor with specified alignments
    DistTensor
    ( const ObjShape& shape, const TensorDistribution& dist, const std::vector<Unsigned>& modeAligns, const tmen::Grid& g );

    // Create a "shape" distributed tensor with specified alignments
    // and leading dimension
    DistTensor
    ( const ObjShape& shape, const TensorDistribution& dist, const std::vector<Unsigned>& modeAligns, const std::vector<Unsigned>& strides, const tmen::Grid& g );

    // View a constant distributed tensor's buffer
    DistTensor
    ( const ObjShape& shape, const TensorDistribution& dist, const std::vector<Unsigned>& modeAligns,
      const T* buffer, const std::vector<Unsigned>& strides, const tmen::Grid& g );

    // View a mutable distributed tensor's buffer
    DistTensor
    ( const ObjShape& shape, const TensorDistribution& dist, const std::vector<Unsigned>& modeAligns,
      T* buffer, const std::vector<Unsigned>& strides, const tmen::Grid& g );

    //////////////////////////////////
    /// String distribution versions
    //////////////////////////////////

    // Create a distributed tensor based on a supplied distribution
    DistTensor( const std::string& dist, const tmen::Grid& g=DefaultGrid() );

    // Create a "shape" distributed tensor
    DistTensor
    ( const ObjShape& shape, const std::string& dist, const tmen::Grid& g=DefaultGrid() );

    // Create a "shape" distributed tensor with specified alignments
    DistTensor
    ( const ObjShape& shape, const std::string& dist, const std::vector<Unsigned>& modeAligns, const tmen::Grid& g );

    // Create a "shape" distributed tensor with specified alignments
    // and leading dimension
    DistTensor
    ( const ObjShape& shape, const std::string& dist, const std::vector<Unsigned>& modeAligns, const std::vector<Unsigned>& strides, const tmen::Grid& g );

    // View a constant distributed tensor's buffer
    DistTensor
    ( const ObjShape& shape, const std::string& dist, const std::vector<Unsigned>& modeAligns,
      const T* buffer, const std::vector<Unsigned>& strides, const tmen::Grid& g );

    // View a mutable distributed tensor's buffer
    DistTensor
    ( const ObjShape& shape, const std::string& dist, const std::vector<Unsigned>& modeAligns,
      T* buffer, const std::vector<Unsigned>& strides, const tmen::Grid& g );

    // Create a copy of distributed matrix A
    DistTensor( const DistTensor<T>& A );

    ~DistTensor();

#ifndef SWIG
    // Move constructor
    //DistTensor( DistTensor<T>&& A );
    // Move assignment
    //DistTensor<T>& operator=( DistTensor<T>&& A );
#endif

    const DistTensor<T>& operator=( const DistTensor<T>& A );
    

    Unsigned Order() const;
    Unsigned Dimension(Mode mode) const;
    ObjShape Shape() const;
    ObjShape MaxLocalShape() const;
    ObjShape LocalShape() const;
    Unsigned LocalDimension(Mode mode) const;
    std::vector<Unsigned> LocalStrides() const;
    Unsigned LocalModeStride(Mode mode) const;

    TensorDistribution TensorDist() const;
    ModeDistribution ModeDist(Mode mode) const;

    void SetDistribution(const TensorDistribution& tenDist);

    std::vector<Unsigned> Strides() const;
    Unsigned Stride(Mode mode) const;

    size_t AllocatedMemory() const;

    const tmen::Grid& Grid() const;
    const tmen::GridView GetGridView() const;

          T* Buffer();
          T* Buffer( const Location& loc );
    const T* LockedBuffer() const;
    const T* LockedBuffer( const Location& loc ) const;


          tmen::Tensor<T>& Tensor();
    const tmen::Tensor<T>& LockedTensor() const;

    //
    // Entry manipulation
    //
    Location DetermineOwner(const Location& loc) const;
    Location DetermineOwnerNewAlignment(const Location& loc, std::vector<Unsigned>& newAlignment) const;
    Location DetermineFirstElem(const Location& gridLoc) const;
    Location DetermineFirstUnalignedElem(const Location& gridViewLoc, const std::vector<Unsigned>& alignmentDiff) const;
    Location Global2LocalIndex(const Location& loc) const;

    //
    // Alignments
    //

    void FreeAlignments();
    bool ConstrainedModeAlignment(Mode mode) const;
    std::vector<Unsigned> Alignments() const;
    Unsigned ModeAlignment(Mode mode) const;
    Unsigned ModeShift(Mode mode) const;
    std::vector<Unsigned> ModeShifts() const;

    void Align( const std::vector<Unsigned>& modeAlign );
    void AlignMode( Mode mode, Unsigned align );

    //
    // Local entry manipulation
    //

    T GetLocal( const Location& loc ) const;
    void SetLocal( const Location& loc, T alpha );
    void UpdateLocal( const Location& loc, T alpha );
    void SetLocalPermutation(const Permutation& perm);
    Permutation LocalPermutation() const;
    //
    // Though the following routines are meant for complex data, all but two
    // logically apply to real data.
    //

    BASE(T) GetRealPart( const Location& loc ) const;
    BASE(T) GetImagPart( const Location& loc ) const;
    BASE(T) GetLocalRealPart( const Location& loc ) const;
    BASE(T) GetLocalImagPart( const Location& loc ) const;
    void SetLocalRealPart( const Location& loc, BASE(T) alpha );
    void UpdateLocalRealPart( const Location& loc, BASE(T) alpha );
    // Only valid for complex data
    void SetLocalImagPart( const Location& loc, BASE(T) alpha );
    void UpdateLocalImagPart( const Location& loc, BASE(T) alpha );

    //
    // Viewing
    //

    bool Viewing() const;
    bool Locked()  const;

    //
    // Utilities
    //
    Location GridViewLoc() const;
    ObjShape GridViewShape() const;
    mpi::Comm GetCommunicatorForModes(const ModeArray& modes, const tmen::Grid& grid);
    void SetParticipatingComm();
    mpi::Comm GetParticipatingComm() const;
    void Empty();
    void EmptyData();
    void SetGrid( const tmen::Grid& grid );

    void Swap( DistTensor<T>& A );

    bool Participating() const;

    void MakeConsistent();

    void CopyLocalBuffer(const DistTensor<T>& A);
    //------------------------------------------------------------------------//
    // Overrides of AbstractDistTensor                                        //
    //------------------------------------------------------------------------//

    //
    // Non-collective routines
    //

    Unsigned ModeStride(Mode mode) const;
    std::vector<Unsigned> ModeStrides() const;
    Unsigned ModeRank(Mode mode) const;
    tmen::DistData DistData() const;

    //
    // All-to-all workhorse routines
    //
    bool CheckAllToAllCommRedist(const DistTensor<T>& A);
    void AllToAllCommRedist(const DistTensor<T>& A, const ModeArray& commModes);
    A2AP2PData DetermineA2AP2POptData(const DistTensor<T>& A, const ModeArray& commModes);
    void PackA2ACommSendBuf(const DistTensor<T>& A, const ModeArray& commModes, const ObjShape& sendShape, T * const sendBuf);
    void UnpackA2ACommRecvBuf(const T * const recvBuf, const ModeArray& commModes, const ObjShape& sendShape, const DistTensor<T>& A);

    //
    // All-to-all interface routines
    //
    void AllToAllRedistFrom(const DistTensor<T>& A, const ModeArray& commModes);

    //
    // Allgather workhorse routines
    //
    bool CheckAllGatherCommRedist(const DistTensor<T>& A);
    void AllGatherCommRedist(const DistTensor<T>& A, const ModeArray& commModes);
    void PackAGCommSendBuf(const DistTensor<T>& A, T * const sendBuf);

    //
    // Allgather interface routines
    //
    void AllGatherRedistFrom(const DistTensor<T>& A, const ModeArray& commModes);

    //
    // Broadcast workhorse routines
    //
    bool CheckBroadcastCommRedist(const DistTensor<T>& A);
    void BroadcastCommRedist(const DistTensor<T>& A, const ModeArray& commModes);

    //
    // Broadcast interface routines
    //
    void BroadcastRedistFrom(const DistTensor<T>& A, const ModeArray& commModes);

    //
    // Gather-to-one workhorse routines
    //
    bool CheckGatherToOneCommRedist(const DistTensor<T>& A);
    void GatherToOneCommRedist(const DistTensor<T>& A, const ModeArray& commModes);

    //
    // Gather-to-one interface routines
    //
    void GatherToOneRedistFrom(const DistTensor<T>& A, const ModeArray& commModes);

    //
    // Local redist workhorse routines
    //
    bool CheckLocalCommRedist(const DistTensor<T>& A);
    void LocalCommRedist(const DistTensor<T>& A);
    void UnpackLocalCommRedist(const DistTensor<T>& A, const T* unpackBuf);

    //
    // Local redist interface routines
    //
    void LocalRedistFrom(const DistTensor<T>& A);

    //
    // Point-to-point workhorse routines
    //
    bool CheckPermutationCommRedist(const DistTensor<T>& A);
    void PermutationCommRedist(const DistTensor<T>& A, const ModeArray& redistModes);
    void UnpackPCommRecvBuf(const T* const recvBuf, const DistTensor<T>& A);

    //
    // Point-to-point interface routines
    //
    void PermutationRedistFrom(const DistTensor<T>& A, const ModeArray& redistModes);

    //
    // Reduce Redist routine
    //
    void ReduceUpdateRedistFrom(const RedistType& redistType, const T alpha, const DistTensor<T>& A, const T beta, const ModeArray& reduceModes);

    //
    // AllReduce workhorse routines
    //
    bool CheckAllReduceCommRedist(const DistTensor<T>& A);
    void AllReduceUpdateCommRedist(const T alpha, const DistTensor<T>& A, const T beta, const ModeArray& reduceModes, const ModeArray& commModes);
    void PackARCommSendBuf(const DistTensor<T>& A, const ModeArray& reduceModes, const ModeArray& commModes, T * const sendBuf);
    void UnpackARUCommRecvBuf(const T* const recvBuf, const T alpha, const DistTensor<T>& A, const T beta);

    //
    // AllReduce interface routines
    //
    void AllReduceUpdateRedistFrom(const T alpha, const DistTensor<T>& A, const T beta, const ModeArray& reduceModes);
    void AllReduceRedistFrom(const DistTensor<T>& A, const Mode reduceMode);
    void AllReduceUpdateRedistFrom(const DistTensor<T>& A, const T beta, const Mode reduceMode);
    void AllReduceRedistFrom(const DistTensor<T>& A, const ModeArray& reduceModes);
    void AllReduceUpdateRedistFrom(const DistTensor<T>& A, const T beta, const ModeArray& reduceModes);

    //
    // Reduce-scatter workhorse routines
    //
    bool CheckReduceScatterCommRedist(const DistTensor<T>& A);
    void ReduceScatterUpdateCommRedist(const T alpha, const DistTensor<T>& A, const T beta, const ModeArray& reduceModes, const ModeArray& commModes);
    void PackRSCommSendBuf(const DistTensor<T>& A, const ModeArray& reduceModes, const ModeArray& commModes, T * const sendBuf);
    void UnpackRSUCommRecvBuf(const T* const recvBuf, const T alpha, const DistTensor<T>& A, const T beta);

    //
    // Reduce-scatter interface routines
    //
    void ReduceScatterUpdateRedistFrom(const T alpha, const DistTensor<T>& A, const T beta, const ModeArray& reduceModes);
    void ReduceScatterRedistFrom(const DistTensor<T>& A, const Mode reduceMode);
    void ReduceScatterUpdateRedistFrom(const DistTensor<T>& A, const T beta, const Mode reduceMode);
    void ReduceScatterRedistFrom(const DistTensor<T>& A, const ModeArray& reduceModes);
    void ReduceScatterUpdateRedistFrom(const DistTensor<T>& A, const T beta, const ModeArray& reduceModes);

    //
    // Reduce-to-one workhorse routines
    //
    bool CheckReduceToOneCommRedist(const DistTensor<T>& A);
    void ReduceToOneUpdateCommRedist(const T alpha, const DistTensor<T>& A, const T beta, const ModeArray& rModes, const ModeArray& commModes);

    //
    // Reduce-to-one interface routines
    //
    void ReduceToOneUpdateRedistFrom(T alpha, const DistTensor<T>& A, const T beta, const ModeArray& rModes);
    void ReduceToOneRedistFrom(const DistTensor<T>& A, const ModeArray& rModes);
    void ReduceToOneRedistFrom(const DistTensor<T>& A, const Mode rMode);
    void ReduceToOneUpdateRedistFrom(const DistTensor<T>& A, const T beta, const ModeArray& rModes);
    void ReduceToOneUpdateRedistFrom(const DistTensor<T>& A, const T beta, const Mode rMode);

    //
    // Scatter workhorse routines
    //
    bool CheckScatterCommRedist(const DistTensor<T>& A);
    void ScatterCommRedist(const DistTensor<T>& A, const ModeArray& commModes);

    //
    // Scatter interface routines
    //
    void ScatterRedistFrom(const DistTensor<T>& A, const ModeArray& commModes);

    bool AlignCommBufRedist(const DistTensor<T>& A, const T* unalignedSendBuf, const Unsigned sendSize, T* alignedSendBuf, const Unsigned recvSize);

    //
    //Unit mode intro/remove routines
    //
    void RemoveUnitModesRedist(const ModeArray& unitModes);
    void RemoveUnitModeRedist(const Mode& unitMode);
    void IntroduceUnitModesRedist(const std::vector<Unsigned>& newModePositions);
    void IntroduceUnitModeRedist(const Unsigned& newModePosition);


    //
    // Routines needed for indexing
    //
    T Get( const Location& loc ) const;
    void Set( const Location& loc, T alpha );
    void Update( const Location& loc, T alpha );

    void ResizeTo( const DistTensor<T>& A);
    void ResizeTo( const ObjShape& shape );
    void ResizeTo( const ObjShape& shape, const std::vector<Unsigned>& strides );

    // Distribution alignment
    void AlignWith( const tmen::DistData& data );
    void AlignWith( const DistTensor<T>& A );
    void AlignModeWith( Mode mode, const tmen::DistData& data );
    void AlignModeWith( Mode mode, const DistTensor<T>& A );
    void AlignModeWith( Mode mode, const DistTensor<T>& A, Mode modeA );
    void AlignModesWith( const ModeArray& modes, const DistTensor<T>& A, const ModeArray& modesA );

    //
    // Though the following routines are meant for complex data, all but two
    // logically apply to real data.
    //

    void SetRealPart( const Location& loc, BASE(T) u );
    // Only valid for complex data
    void SetImagPart( const Location& loc, BASE(T) u );
    void UpdateRealPart( const Location& loc, BASE(T) u );
    // Only valid for complex data
    void UpdateImagPart( const Location& loc, BASE(T) u );

    //-----------------------------------------------------------------------//
    // Routines specific to [MC,MR] distribution                             //
    //-----------------------------------------------------------------------//

    //
    // Collective routines
    //
   
    void GetDiagonal( DistTensor<T>& d, Int offset=0 ) const;
    DistTensor<T> GetDiagonal( Int offset=0 ) const;

    void SetDiagonal( const DistTensor<T>& d, Int offset=0 );

    // (Immutable) view of a distributed matrix's buffer
    void Attach
    ( const ObjShape& shape, const std::vector<Unsigned>& modeAligns,
      T* buffer, const std::vector<Unsigned>& strides, const tmen::Grid& grid );
    void LockedAttach
    ( const ObjShape& shape, const std::vector<Unsigned>& modeAligns,
      const T* buffer, const std::vector<Unsigned>& strides, const tmen::Grid& grid );
    void LockedAttach
        ( const ObjShape& shape, const std::vector<Unsigned>& modeAligns,
          const T* buffer, const Permutation& perm, const std::vector<Unsigned>& strides, const tmen::Grid& grid );

    //
    // Though the following routines are meant for complex data, all
    // logically apply to real data.
    //

    void GetRealPartOfDiagonal
    ( DistTensor<BASE(T)>& d, Int offset=0 ) const;
    void GetImagPartOfDiagonal
    ( DistTensor<BASE(T)>& d, Int offset=0 ) const;
    DistTensor<BASE(T)> GetRealPartOfDiagonal( Int offset=0 ) const;
    DistTensor<BASE(T)> GetImagPartOfDiagonal( Int offset=0 ) const;

    void SetRealPartOfDiagonal
    ( const DistTensor<BASE(T)>& d, Int offset=0 );
    // Only valid for complex datatypes
    void SetImagPartOfDiagonal
    ( const DistTensor<BASE(T)>& d, Int offset=0 );

protected:
    //Distributed information
    TensorDistribution dist_;
    ObjShape shape_;

    //Wrapping information
    std::vector<bool> constrainedModeAlignments_;
    std::vector<Unsigned> modeAlignments_;
    std::vector<Unsigned> modeShifts_;

    //Local information
    tmen::Tensor<T> tensor_;
    Permutation localPerm_;

    //Grid information
    const tmen::Grid* grid_;
    tmen::mpi::CommMap* commMap_;
    tmen::GridView gridView_;
    mpi::Comm participatingComm_;

    ViewType viewType_;
    Memory<T> auxMemory_;

private:
    void CopyFromDifferentGrid( const DistTensor<T>& A );

    void SetShifts();
    void SetDefaultPermutation();
    void SetModeShift(Mode mode);
    void SetGrid();

    void ComplainIfReal() const;

    void SetAlignmentsAndResize
    ( const std::vector<Unsigned>& aligns, const ObjShape& shape );
    void ForceAlignmentsAndResize
    ( const std::vector<Unsigned>& aligns, const ObjShape& shape );

    void SetModeAlignmentAndResize
    ( Mode mode, Unsigned align, const ObjShape& shape );
    void ForceModeAlignmentAndResize
    ( Mode mode, Unsigned align, const ObjShape& shape );

#ifndef SWIG
    template<typename S>
    friend class DistTensor;

    template<typename S>
    friend void ViewHelper( DistTensor<S>& A, const DistTensor<S>& B, bool isLocked );
    template<typename S>
    friend void ViewHelper
    ( DistTensor<S>& A, const DistTensor<S>& B,
      const Location& loc, const ObjShape& shape, bool isLocked );
    template<typename S>
    friend void View2x1Helper
    ( DistTensor<S>& A, const DistTensor<S>& BT, const DistTensor<S>& BB, Mode mode, bool isLocked );

    template<typename S>
    friend class DistTensor;
//    template<typename S,Distribution U,Distribution V>
//    friend class DistTensor;
#endif // ifndef SWIG
};

Permutation DefaultPermutation(Unsigned order);

} // namespace tmen

#endif // ifndef TMEN_CORE_DISTTENSOR_DECL_HPP
