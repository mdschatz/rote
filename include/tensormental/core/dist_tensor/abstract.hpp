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

namespace tmen {
#ifndef RELEASE
//template<typename T>
//void AssertConforming1x2
//( const AbstractDistTensor<T>& AL, const AbstractDistTensor<T>& AR );
//
//template<typename T>
//void AssertConforming2x1
//( const AbstractDistTensor<T>& AT, const AbstractDistTensor<T>& AB );
//
//template<typename T>
//void AssertConforming2x2
//( const AbstractDistTensor<T>& ATL, const AbstractDistTensor<T>& ATR,
//  const AbstractDistTensor<T>& ABL, const AbstractDistTensor<T>& ABR );
#endif // ifndef RELEASE

template<typename T> 
class AbstractDistTensor
{
public:
    virtual ~AbstractDistTensor();

    //-----------------------------------------------------------------------//
    // Routines that do NOT need to be implemented in derived classes        //
    //-----------------------------------------------------------------------//

#ifndef SWIG
    // Move constructor
//    AbstractDistTensor( AbstractDistTensor<T>&& A );
    // Move assignment
//    AbstractDistTensor<T>& operator=( AbstractDistTensor<T>&& A );
#endif

#ifndef RELEASE
    void AssertNotLocked() const;
    void AssertNotStoringData() const;
    void AssertValidEntry( const std::vector<Int>& index ) const;
    void AssertValidSubtensor
    ( const std::vector<Int>& index, const std::vector<Int>& dims ) const;
    void AssertSameGrid( const tmen::Grid& grid ) const;
    void AssertSameSize( const std::vector<Int>& dims ) const;
#endif // ifndef RELEASE

    //
    // Basic information
    //

    Int Dimension(Int mode) const;
    Int LocalDimension(Int mode) const;
//    Int Height() const;
//    Int Width() const;
//    Int DiagonalLength( Int offset=0 ) const;
//    Int LocalHeight() const;
//    Int LocalWidth() const;
    Int LDim(Int mode) const;
    size_t AllocatedMemory() const;

    const tmen::Grid& Grid() const;

          T* Buffer( const std::vector<Int>& loc );
    const T* LockedBuffer( const std::vector<Int>& loc ) const;

          tmen::Tensor<T>& Tensor();
    const tmen::Tensor<T>& LockedTensor() const;

    //
    // Alignments
    //

    void FreeAlignments();
    bool ConstrainedModeAlignment(Int mode) const;
    //bool ConstrainedRowAlignment() const;
    Int ModeAlignment(Int mode) const;
    //Int RowAlignment() const;
    Int ModeShift(Int mode) const;
    //Int RowShift() const;

    void Align( const std::vector<Int>& modeAlign );
    void AlignMode( Int mode, Int align );
    //void AlignRows( Int rowAlign );

    //
    // Local entry manipulation
    //

    T GetLocal( const std::vector<Int>& loc ) const;
    void SetLocal( const std::vector<Int>& loc, T alpha );
    void UpdateLocal( const std::vector<Int>& loc, T alpha );

    //
    // Though the following routines are meant for complex data, all but two
    // logically apply to real data.
    //

    BASE(T) GetRealPart( const std::vector<Int>& index ) const;
    BASE(T) GetImagPart( const std::vector<Int>& index ) const;
    BASE(T) GetLocalRealPart( const std::vector<Int>& index ) const;
    BASE(T) GetLocalImagPart( const std::vector<Int>& index ) const;
    void SetLocalRealPart( const std::vector<Int>& index, BASE(T) alpha );
    void UpdateLocalRealPart( const std::vector<Int>& index, BASE(T) alpha );
    // Only valid for complex data
    void SetLocalImagPart( const std::vector<Int>& index, BASE(T) alpha );
    void UpdateLocalImagPart( const std::vector<Int>& index, BASE(T) alpha );

    //
    // Viewing 
    //

    bool Viewing() const;
    bool Locked()  const;

    //
    // Utilities
    //

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
    virtual void AlignModeWith( Int mode, const tmen::DistData& data );
    virtual void AlignModeWith( Int mode, const AbstractDistTensor<T>& A );
//    virtual void AlignColsWith( const AbstractDistTensor<T>& A );
//    virtual void AlignRowsWith( const tmen::DistData& data );
//    virtual void AlignRowsWith( const AbstractDistTensor<T>& A );

    virtual void MakeConsistent();

    //------------------------------------------------------------------------//
    // Routines that MUST be implemented in non-abstract derived classes      //
    //------------------------------------------------------------------------//

    //
    // Basic information
    //

    virtual tmen::DistData DistData() const = 0;

    // So that the local row indices are given by
    //   A.ColShift():A.ColStride():A.Height()
//    virtual Int ColStride() const = 0; 
    virtual Int ModeStride(Int mode) const = 0; 
    // So that the local column indices are given by
    //   A.RowShift():A.RowStride():A.Width()
//    virtual Int RowStride() const = 0;
    virtual Int ModeRank(Int mode) const = 0;
//    virtual Int RowRank() const = 0;

    //
    // Entry manipulation
    //

    virtual T Get( const std::vector<Int>& index ) const = 0;
    virtual void Set( const std::vector<Int>& index, T alpha ) = 0;
    virtual void Update( const std::vector<Int>& index, T alpha ) = 0;

    //
    // Though the following routines are meant for complex data, all but two
    // logically applies to real data.
    //

    virtual void SetRealPart( Int i, Int j, BASE(T) alpha ) = 0;
    // Only valid for complex data
    virtual void SetImagPart( Int i, Int j, BASE(T) alpha ) = 0;
    virtual void UpdateRealPart( Int i, Int j, BASE(T) alpha ) = 0;
    // Only valid for complex data
    virtual void UpdateImagPart( Int i, Int j, BASE(T) alpha ) = 0;

    //
    // Utilities
    //
    
    virtual void ResizeTo( const std::vector<Int>& dims ) = 0;
    virtual void ResizeTo( const std::vector<Int>& dims, const std::vector<Int>& ldims ) = 0;

protected:
    Int order_;
    ViewType viewType_;
    std::vector<Int> dims_;
    Memory<T> auxMemory_;
    tmen::Tensor<T> tensor_;
    
    std::vector<bool> constrainedModeAlignments_;
    std::vector<Int> modeAlignments_;
    std::vector<Int> modeShifts_;
    const tmen::Grid* grid_;

    // Build around a particular grid
    AbstractDistTensor( const tmen::Grid& g );

    void SetShifts();
    void SetModeShift(Int Mode);
    void SetGrid();

    void ComplainIfReal() const;

    void SetAlignmentsAndResize
    ( const std::vector<Int>& aligns, const std::vector<Int>& dims );
    void ForceAlignmentsAndResize
    ( const std::vector<Int>& aligns, const std::vector<Int>& dims );

    void SetModeAlignmentAndResize
    ( Int mode, Int align, const std::vector<Int>& dims );
    void ForceModeAlignmentAndResize
    ( Int mode, Int align, const std::vector<Int>& dims );

//    void SetRowAlignmentAndResize
//    ( Int rowAlign, Int height, Int width );
//    void ForceRowAlignmentAndResize
//    ( Int rowAlign, Int height, Int width );

#ifndef SWIG
 //   template<typename S,Distribution U,Distribution V> 
 //   friend void View( DistTensor<S,U,V>& A, DistTensor<S,U,V>& B );
 //   template<typename S,Distribution U,Distribution V> 
 //   friend void LockedView( DistTensor<S,U,V>& A, const DistTensor<S,U,V>& B );
 //   template<typename S,Distribution U,Distribution V> 
 //   friend void View
 //   ( DistTensor<S,U,V>& A, DistTensor<S,U,V>& B,
 //     Int i, Int j, Int height, Int width );
 //   template<typename S,Distribution U,Distribution V> 
 //   friend void LockedView
 //   ( DistTensor<S,U,V>& A, const DistTensor<S,U,V>& B,
 //     Int i, Int j, Int height, Int width );
 //   template<typename S,Distribution U,Distribution V> 
 //   friend void View1x2
 //   ( DistTensor<S,U,V>& A, DistTensor<S,U,V>& BL, DistTensor<S,U,V>& BR );
 //   template<typename S,Distribution U,Distribution V> 
 //   friend void LockedView1x2
 //   (       DistTensor<S,U,V>& A,
 //     const DistTensor<S,U,V>& BL, const DistTensor<S,U,V>& BR );
 //   template<typename S,Distribution U,Distribution V> 
 //   friend void View2x1
 //   ( DistTensor<S,U,V>& A, DistTensor<S,U,V>& BT, DistTensor<S,U,V>& BB );
 //   template<typename S,Distribution U,Distribution V> 
 //   friend void LockedView2x1
 //   (       DistTensor<S,U,V>& A,
 //     const DistTensor<S,U,V>& BT, const DistTensor<S,U,V>& BB );
 //   template<typename S,Distribution U,Distribution V> 
 //   friend void View2x2
 //   ( DistTensor<S,U,V>& A,
 //     DistTensor<S,U,V>& BTL, DistTensor<S,U,V>& BTR,
 //     DistTensor<S,U,V>& BBL, DistTensor<S,U,V>& BBR );
 //   template<typename S,Distribution U,Distribution V> 
 //   friend void LockedView2x2
 //   (       DistTensor<S,U,V>& A,
 //     const DistTensor<S,U,V>& BTL, const DistTensor<S,U,V>& BTR,
 //     const DistTensor<S,U,V>& BBL, const DistTensor<S,U,V>& BBR );

    template<typename S>
    friend class DistTensor;
//    template<typename S,Distribution U,Distribution V>
//    friend class DistTensor;
#endif // ifndef SWIG
};

} // namespace tmen

#endif // ifndef TMEN_CORE_DISTTENSOR_ABSTRACT_DECL_HPP
