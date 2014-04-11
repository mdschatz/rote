/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef TMEN_CORE_DISTTENSOR_MC_MR_DECL_HPP
#define TMEN_CORE_DISTTENSOR_MC_MR_DECL_HPP

namespace tmen {

// Partial specialization to A[MC,MR].
//
// The columns of these matrices will be distributed among columns of the
// process grid, and the rows will be distributed among rows of the process
// grid.

template<typename T>
class DistTensor : public AbstractDistTensor<T>
{
public:
    // Create a 0 distributed tensor
    DistTensor( const tmen::Grid& g=DefaultGrid() );

    // Create a "shape" distributed tensor
    DistTensor
    ( const ObjShape& shape, const TensorDistribution& dist, const tmen::Grid& g=DefaultGrid() );

    // Create a "shape" distributed tensor specifying the associated indices
    DistTensor
    ( const ObjShape& shape, const TensorDistribution& dist, const std::vector<Index>& indices, const tmen::Grid& g=DefaultGrid() );

    // Create a "shape" distributed tensor with specified alignments
    DistTensor
    ( const ObjShape& shape, const TensorDistribution& dist, const std::vector<Index>& indices, const std::vector<Unsigned>& modeAligns,
      const tmen::Grid& g );

    // Create a "shape" distributed tensor with specified alignments
    // and leading dimension
    DistTensor
    ( const ObjShape& shape, const TensorDistribution& dist, const std::vector<Index>& indices, const std::vector<Unsigned>& modeAligns, const std::vector<Unsigned>& ldims, const tmen::Grid& g );

    // View a constant distributed tensor's buffer
    DistTensor
    ( const ObjShape& shape, const TensorDistribution& dist, const std::vector<Index>& indices, const std::vector<Unsigned>& modeAligns,
      const T* buffer, const std::vector<Unsigned>& ldims, const tmen::Grid& g );

    // View a mutable distributed tensor's buffer
    DistTensor
    ( const ObjShape& shape, const TensorDistribution& dist, const std::vector<Index>& indices, const std::vector<Unsigned>& modeAligns,
      T* buffer, const std::vector<Unsigned>& ldims, const tmen::Grid& g );

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
    
    //------------------------------------------------------------------------//
    // Overrides of AbstractDistTensor                                        //
    //------------------------------------------------------------------------//

    //
    // Non-collective routines
    //

    virtual Unsigned ModeStride(Mode mode) const;
    virtual Int ModeRank(Mode mode) const;
    virtual tmen::DistData DistData() const;

    //
    // Collective routines
    //

    //
    // Routines needed for indexing
    //
    virtual T Get( const Location& loc ) const;
    virtual void Set( const Location& loc, T alpha );
    virtual void Update( const Location& loc, T alpha );

    virtual void ResizeTo( const ObjShape& shape );
    virtual void ResizeTo( const ObjShape& shape, const std::vector<Unsigned>& ldims );

    // Distribution alignment
    virtual void AlignWith( const tmen::DistData& data );
    virtual void AlignWith( const AbstractDistTensor<T>& A );
    virtual void AlignModeWith( Mode mode, const tmen::DistData& data );
    virtual void AlignModeWith( Mode mode, const AbstractDistTensor<T>& A );

    //
    // Though the following routines are meant for complex data, all but two
    // logically apply to real data.
    //

    virtual void SetRealPart( const Location& loc, BASE(T) u );
    // Only valid for complex data
    virtual void SetImagPart( const Location& loc, BASE(T) u );
    virtual void UpdateRealPart( const Location& loc, BASE(T) u );
    // Only valid for complex data
    virtual void UpdateImagPart( const Location& loc, BASE(T) u );

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
      T* buffer, const std::vector<Unsigned>& ldims, const tmen::Grid& grid );
    void LockedAttach
    ( const ObjShape& shape, const std::vector<Unsigned>& modeAligns,
      const T* buffer, const std::vector<Unsigned>& ldims, const tmen::Grid& grid );

    // Equate/Update with the scattered summation of A[MC,* ] across process
    // rows
    void SumScatterFrom( const DistTensor<T>& A );
    void SumScatterUpdate( T alpha, const DistTensor<T>& A );

    // Auxiliary routines needed to implement algorithms that avoid 
    // inefficient unpackings of partial matrix distributions
    //void AdjointFrom( const DistTensor<T>& A );
    //void AdjointSumScatterFrom( const DistTensor<T>& A );
    //void AdjointSumScatterUpdate( T alpha, const DistTensor<T>& A );

    //void TransposeFrom
    //( const DistTensor<T>& A, bool conjugate=false );
    //void TransposeSumScatterFrom
    //( const DistTensor<T>& A, bool conjugate=false );
    //void TransposeSumScatterUpdate
    //( T alpha, const DistTensor<T>& A, bool conjugate=false );

    //
    // Though the following routines are meant for complex data, all but two
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

private:
    void CopyFromDifferentGrid( const DistTensor<T>& A );
#ifndef SWIG
    template<typename S>
    friend class DistTensor;
#endif // ifndef SWIG
};

} // namespace tmen

#endif // ifndef TMEN_CORE_DISTTENSOR_MC_MR_DECL_HPP
