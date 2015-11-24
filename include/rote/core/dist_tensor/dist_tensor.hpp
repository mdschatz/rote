/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef ROTE_CORE_DISTTENSORBASE_DECL_HPP
#define ROTE_CORE_DISTTENSORBASE_DECL_HPP

#include<vector>
namespace rote {

template<typename T>
class DistTensorBase
{
public:

#ifndef RELEASE
    void AssertNotLocked() const;
    void AssertNotStoringData() const;
    void AssertValidEntry( const Location& loc ) const;
    void AssertValidSubtensor
    ( const Location& loc, const ObjShape& shape ) const;
    void AssertSameGrid( const rote::Grid& grid ) const;
    void AssertSameSize( const ObjShape& shape ) const;
    //TODO: REMOVE THIS
    void AssertMergeableModes(const std::vector<ModeArray>& oldModes) const;
#endif // ifndef RELEASE

    //Constructors

    void ClearCommMap();
    Unsigned CommMapSize();

    // Create a 0 distributed tensor
    DistTensorBase( const rote::Grid& g=DefaultGrid() );

    // Create a 0 distributed tensor
    DistTensorBase( const Unsigned order, const rote::Grid& g=DefaultGrid() );

    // Create a distributed tensor based on a supplied distribution
    DistTensorBase( const TensorDistribution& dist, const rote::Grid& g=DefaultGrid() );

    // Create a "shape" distributed tensor
    DistTensorBase
    ( const ObjShape& shape, const TensorDistribution& dist, const rote::Grid& g=DefaultGrid() );

    // Create a "shape" distributed tensor with specified alignments
    DistTensorBase
    ( const ObjShape& shape, const TensorDistribution& dist, const std::vector<Unsigned>& modeAligns, const rote::Grid& g );

    // Create a "shape" distributed tensor with specified alignments
    // and leading dimension
    DistTensorBase
    ( const ObjShape& shape, const TensorDistribution& dist, const std::vector<Unsigned>& modeAligns, const std::vector<Unsigned>& strides, const rote::Grid& g );

    // View a constant distributed tensor's buffer
    DistTensorBase
    ( const ObjShape& shape, const TensorDistribution& dist, const std::vector<Unsigned>& modeAligns,
      const T* buffer, const std::vector<Unsigned>& strides, const rote::Grid& g );

    // View a mutable distributed tensor's buffer
    DistTensorBase
    ( const ObjShape& shape, const TensorDistribution& dist, const std::vector<Unsigned>& modeAligns,
      T* buffer, const std::vector<Unsigned>& strides, const rote::Grid& g );

    //////////////////////////////////
    /// String distribution versions
    //////////////////////////////////

    // Create a distributed tensor based on a supplied distribution
    DistTensorBase( const std::string& dist, const rote::Grid& g=DefaultGrid() );

    // Create a "shape" distributed tensor
    DistTensorBase
    ( const ObjShape& shape, const std::string& dist, const rote::Grid& g=DefaultGrid() );

    // Create a "shape" distributed tensor with specified alignments
    DistTensorBase
    ( const ObjShape& shape, const std::string& dist, const std::vector<Unsigned>& modeAligns, const rote::Grid& g );

    // Create a "shape" distributed tensor with specified alignments
    // and leading dimension
    DistTensorBase
    ( const ObjShape& shape, const std::string& dist, const std::vector<Unsigned>& modeAligns, const std::vector<Unsigned>& strides, const rote::Grid& g );

    // View a constant distributed tensor's buffer
    DistTensorBase
    ( const ObjShape& shape, const std::string& dist, const std::vector<Unsigned>& modeAligns,
      const T* buffer, const std::vector<Unsigned>& strides, const rote::Grid& g );

    // View a mutable distributed tensor's buffer
    DistTensorBase
    ( const ObjShape& shape, const std::string& dist, const std::vector<Unsigned>& modeAligns,
      T* buffer, const std::vector<Unsigned>& strides, const rote::Grid& g );

    // Create a copy of distributed matrix A
    DistTensorBase( const DistTensorBase<T>& A );

    virtual ~DistTensorBase();

#ifndef SWIG
    // Move constructor
    //DistTensor( DistTensor<T>&& A );
    // Move assignment
    //DistTensor<T>& operator=( DistTensor<T>&& A );
#endif

    const DistTensorBase<T>& operator=( const DistTensorBase<T>& A );
    

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

    const rote::Grid& Grid() const;
    const rote::GridView GetGridView() const;

          T* Buffer();
          T* Buffer( const Location& loc );
    const T* LockedBuffer() const;
    const T* LockedBuffer( const Location& loc ) const;


          rote::Tensor<T>& Tensor();
    const rote::Tensor<T>& LockedTensor() const;

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
    mpi::Comm GetCommunicatorForModes(const ModeArray& modes, const rote::Grid& grid);
    void SetParticipatingComm();
    mpi::Comm GetParticipatingComm() const;
    void Empty();
    void EmptyData();
    void SetGrid( const rote::Grid& grid );

    void Swap( DistTensorBase<T>& A );

    bool Participating() const;

    void MakeConsistent();

    void CopyLocalBuffer(const DistTensorBase<T>& A);
    //------------------------------------------------------------------------//
    // Overrides of AbstractDistTensor                                        //
    //------------------------------------------------------------------------//

    //
    // Non-collective routines
    //

    Unsigned ModeStride(Mode mode) const;
    std::vector<Unsigned> ModeStrides() const;
    Unsigned ModeRank(Mode mode) const;
    rote::DistData DistData() const;

    //
    // Routines needed for indexing
    //
    T Get( const Location& loc ) const;
    void Set( const Location& loc, T alpha );
    void Update( const Location& loc, T alpha );

    void ResizeTo( const DistTensorBase<T>& A);
    void ResizeTo( const ObjShape& shape );
    void ResizeTo( const ObjShape& shape, const std::vector<Unsigned>& strides );

    // Distribution alignment
    void AlignWith( const rote::DistData& data );
    void AlignWith( const DistTensorBase<T>& A );
    void AlignModeWith( Mode mode, const rote::DistData& data );
    void AlignModeWith( Mode mode, const DistTensorBase<T>& A );
    void AlignModeWith( Mode mode, const DistTensorBase<T>& A, Mode modeA );
    void AlignModesWith( const ModeArray& modes, const DistTensorBase<T>& A, const ModeArray& modesA );

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
   
    void GetDiagonal( DistTensorBase<T>& d, Int offset=0 ) const;
    DistTensorBase<T> GetDiagonal( Int offset=0 ) const;

    void SetDiagonal( const DistTensorBase<T>& d, Int offset=0 );

    // (Immutable) view of a distributed matrix's buffer
    void Attach
    ( const ObjShape& shape, const std::vector<Unsigned>& modeAligns,
      T* buffer, const std::vector<Unsigned>& strides, const rote::Grid& grid );
    void LockedAttach
    ( const ObjShape& shape, const std::vector<Unsigned>& modeAligns,
      const T* buffer, const std::vector<Unsigned>& strides, const rote::Grid& grid );
    void LockedAttach
        ( const ObjShape& shape, const std::vector<Unsigned>& modeAligns,
          const T* buffer, const Permutation& perm, const std::vector<Unsigned>& strides, const rote::Grid& grid );

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
    rote::Tensor<T> tensor_;
    Permutation localPerm_;

    //Grid information
    const rote::Grid* grid_;
    rote::mpi::CommMap* commMap_;
    rote::GridView gridView_;
    mpi::Comm participatingComm_;

    ViewType viewType_;
    Memory<T> auxMemory_;

private:
    void CopyFromDifferentGrid( const DistTensorBase<T>& A );

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

} // namespace rote

#endif // ifndef ROTE_CORE_DISTTENSORBASE_DECL_HPP
