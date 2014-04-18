/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef TMEN_CORE_DISTTENSOR_ABSTRACT_DECL_HPP
#define TMEN_CORE_DISTTENSOR_ABSTRACT_DECL_HPP

#include <vector>
namespace tmen {

#ifndef RELEASE
    template<typename T>
    void AssertConforming2x1( const AbstractDistTensor<T>& AT, const AbstractDistTensor<T>& AB, Index index);
#endif

template<typename T> 
class AbstractDistTensor
{
public:
    virtual ~AbstractDistTensor();

#ifndef RELEASE
    void AssertNotLocked() const;
    void AssertNotStoringData() const;
    void AssertValidEntry( const Location& loc ) const;
    void AssertValidSubtensor
    ( const Location& loc, const ObjShape& shape ) const;
    void AssertSameGrid( const tmen::Grid& grid ) const;
    void AssertSameSize( const ObjShape& shape ) const;
    void AssertMergeableIndices(const IndexArray& newIndices, const std::vector<IndexArray>& oldIndices) const;
#endif // ifndef RELEASE

    //
    // Basic information
    //

    Unsigned Order() const;
    Unsigned Dimension(Mode mode) const;
    ObjShape Shape() const;
    ObjShape LocalShape() const;
    Unsigned LocalDimension(Mode mode) const;
    Unsigned LocalModeStride(Mode mode) const;
    IndexArray Indices() const;
    Index IndexOfMode(Mode mode) const;
    Mode ModeOfIndex(Index index) const;

    TensorDistribution TensorDist() const;
    ModeDistribution ModeDist(Mode mode) const;
    ModeDistribution IndexDist(Index index) const;


    Unsigned LDim(Mode mode) const;
    size_t AllocatedMemory() const;

    const tmen::Grid& Grid() const;
    const tmen::GridView GridView() const;

          T* Buffer( const Location& loc );
    const T* LockedBuffer( const Location& loc ) const;

          tmen::Tensor<T>& Tensor();
    const tmen::Tensor<T>& LockedTensor() const;

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
    mpi::Comm GetCommunicator(Index index) const;
    mpi::Comm GetCommunicatorForModes(const ModeArray& modes) const;
    void Empty();
    void EmptyData();
    void SetGrid( const tmen::Grid& grid );

    //------------------------------------------------------------------------//
    // Routines that can be overridden in derived classes                     //
    //------------------------------------------------------------------------//

    virtual void Swap( AbstractDistTensor<T>& A );

    virtual bool Participating() const;
    virtual void AlignWith( const tmen::DistData& data );
    virtual void AlignWith( const AbstractDistTensor<T>& A );
    virtual void AlignModeWith( Mode mode, const tmen::DistData& data );
    virtual void AlignModeWith( Mode mode, const AbstractDistTensor<T>& A );

    virtual void MakeConsistent();

    //------------------------------------------------------------------------//
    // Routines that MUST be implemented in non-abstract derived classes      //
    //------------------------------------------------------------------------//

    //
    // Basic information
    //

    virtual tmen::DistData DistData() const = 0;
    virtual Unsigned ModeStride(Mode mode) const = 0;
    virtual Unsigned ModeRank(Mode mode) const = 0;

    //
    // Entry manipulation
    //
    virtual Location DetermineOwner(const Location& loc) const;
    virtual Location Global2LocalIndex(const Location& loc) const;
    virtual T Get( const Location& loc ) const = 0;
    virtual void Set( const Location& loc, T alpha ) = 0;
    virtual void Update( const Location& loc, T alpha ) = 0;

    //
    // Though the following routines are meant for complex data, all but two
    // logically applies to real data.
    //

    virtual void SetRealPart( const Location& loc, BASE(T) alpha ) = 0;
    // Only valid for complex data
    virtual void SetImagPart( const Location& loc, BASE(T) alpha ) = 0;
    virtual void UpdateRealPart( const Location& loc, BASE(T) alpha ) = 0;
    // Only valid for complex data
    virtual void UpdateImagPart( const Location& loc, BASE(T) alpha ) = 0;

    //
    // Utilities
    //
    
    virtual void ResizeTo( const ObjShape& shape ) = 0;
    virtual void ResizeTo( const ObjShape& shape, const std::vector<Unsigned>& ldims ) = 0;

protected:

    //Distributed information
    ObjShape shape_;
    TensorDistribution dist_;
    
    //Wrapping information
    std::vector<bool> constrainedModeAlignments_;
    std::vector<Unsigned> modeAlignments_;
    std::vector<Unsigned> modeShifts_;

    //Local information
    tmen::Tensor<T> tensor_;

    //Grid information
    const tmen::Grid* grid_;
    tmen::GridView gridView_;

    ViewType viewType_;
    Memory<T> auxMemory_;

    // Build around a particular grid
    AbstractDistTensor( const tmen::Grid& g );
    AbstractDistTensor( const Unsigned order, const tmen::Grid& g );
    //NOTE: Decide whether to remove the following constructor (should we allow creating a tensor without supplying the indices?)
    AbstractDistTensor( const ObjShape& shape, const TensorDistribution& dist, const tmen::Grid& g );
    AbstractDistTensor( const ObjShape& shape, const TensorDistribution& dist, const IndexArray& indices, const tmen::Grid& g );

    void SetShifts();
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
    friend void ViewHelper( DistTensor<S>& A, const DistTensor<S>& B, bool isLocked );
    template<typename S>
    friend void ViewHelper
    ( DistTensor<S>& A, const DistTensor<S>& B,
      const Location& loc, const ObjShape& shape, bool isLocked );
    template<typename S>
    friend void View2x1Helper
    ( DistTensor<S>& A, const DistTensor<S>& BT, const DistTensor<S>& BB, Index index, bool isLocked );

    template<typename S>
    friend class DistTensor;
//    template<typename S,Distribution U,Distribution V>
//    friend class DistTensor;
#endif // ifndef SWIG
};

} // namespace tmen

#endif // ifndef TMEN_CORE_DISTTENSOR_ABSTRACT_DECL_HPP

