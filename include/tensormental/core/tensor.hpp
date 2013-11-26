/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef TMEN_CORE_TENSOR_HPP
#define TMEN_CORE_TENSOR_HPP

#include <iostream>
#include "tensormental/core/view_decl.hpp"

namespace elem {

// Tensor base for arbitrary rings
template<typename T>
class Tensor
{
public:    
    //
    // Assertions
    //
    
    void AssertValidDimensions( Int height, Int width ) const;
    void AssertValidDimensions( Int height, Int width, Int ldim ) const;
    void AssertValidEntry( Int i, Int j ) const;
    
    //
    // Constructors
    // 

    Tensor( bool fixed=false );
    Tensor( Int height, Int width, bool fixed=false );
    Tensor( Int height, Int width, Int ldim, bool fixed=false );
    Tensor
    ( Int height, Int width, const T* buffer, Int ldim, bool fixed=false );
    Tensor( Int height, Int width, T* buffer, Int ldim, bool fixed=false );
    Tensor( const Tensor<T>& A );

    // Move constructor
    Tensor( Tensor<T>&& A );

    // Move assignment
    Tensor<T>& operator=( Tensor<T>&& A );

    // Swap
    void Swap( Tensor<T>& A );

    //
    // Destructor
    //

    virtual ~Tensor();

    //
    // Basic information
    //

    Int Height() const;
    Int Width() const;
    Int DiagonalLength( Int offset=0 ) const;
    Int LDim() const;
    Int MemorySize() const;

    T* Buffer();
    T* Buffer( Int i, Int j );

    const T* LockedBuffer() const;
    const T* LockedBuffer( Int i, Int j ) const;

    //
    // Entry manipulation
    //

    T Get( Int i, Int j ) const;
    void Set( Int i, Int j, T alpha );
    void Update( Int i, Int j, T alpha );

    void GetDiagonal( Tensor<T>& d, Int offset=0 ) const;
    Tensor<T> GetDiagonal( Int offset=0 ) const;

    void SetDiagonal( const Tensor<T>& d, Int offset=0 );
    void UpdateDiagonal( const Tensor<T>& d, Int offset=0 );

    //
    // Though the following routines are meant for complex data, all but four
    // logically apply to real data.
    //

    BASE(T) GetRealPart( Int i, Int j ) const;
    BASE(T) GetImagPart( Int i, Int j ) const;
    void SetRealPart( Int i, Int j, BASE(T) alpha );
    // Only valid for complex data
    void SetImagPart( Int i, Int j, BASE(T) alpha );
    void UpdateRealPart( Int i, Int j, BASE(T) alpha );
    // Only valid for complex data
    void UpdateImagPart( Int i, Int j, BASE(T) alpha );

    void GetRealPartOfDiagonal( Tensor<BASE(T)>& d, Int offset=0 ) const;
    void GetImagPartOfDiagonal( Tensor<BASE(T)>& d, Int offset=0 ) const;
    Tensor<BASE(T)> GetRealPartOfDiagonal( Int offset=0 ) const;
    Tensor<BASE(T)> GetImagPartOfDiagonal( Int offset=0 ) const;

    void SetRealPartOfDiagonal( const Tensor<BASE(T)>& d, Int offset=0 );
    // Only valid for complex data
    void SetImagPartOfDiagonal( const Tensor<BASE(T)>& d, Int offset=0 );
    void UpdateRealPartOfDiagonal( const Tensor<BASE(T)>& d, Int offset=0 );
    // Only valid for complex data
    void UpdateImagPartOfDiagonal( const Tensor<BASE(T)>& d, Int offset=0 );

    //
    // Viewing other matrix instances (or buffers)
    //

    bool Owner()       const;
    bool Shrinkable()  const;
    bool FixedSize()   const;
    bool Viewing()     const;
    bool Locked()      const;

    void Attach( Int height, Int width, T* buffer, Int ldim );
    void LockedAttach
    ( Int height, Int width, const T* buffer, Int ldim );

    // Use this memory *as if it were not a view*, but do not take control of 
    // its deallocation. If Resize() forces reallocation, this buffer is 
    // released from control but not deleted.
    void Control( Int height, Int width, T* buffer, Int ldim );

    //
    // Utilities
    //

    const Tensor<T>& operator=( const Tensor<T>& A );

    void Empty();
    void ResizeTo( Int height, Int width );
    void ResizeTo( Int height, Int width, Int ldim );

private:
    ViewType viewType_;
    Int height_, width_, ldim_;
    const T* data_;
    Memory<T> memory_;

    void ComplainIfReal() const;

    const T& Get_( Int i, Int j ) const;
    T& Set_( Int i, Int j );

    // These bypass fixed-size checking and are used by DistTensor
    void Empty_();
    void ResizeTo_( Int height, Int width );
    void ResizeTo_( Int height, Int width, Int ldim );
    void Control_( Int height, Int width, T* buffer, Int ldim );
    void Attach_( Int height, Int width, T* buffer, Int ldim );
    void LockedAttach_( Int height, Int width, const T* buffer, Int ldim );
    
    template <typename F> 
    friend class Tensor;
//    template <typename F,Distribution U,Distribution V> 
//    friend class DistTensor;
//    friend class AbstractDistTensor<T>;

    friend void View<T>( Tensor<T>& A, Tensor<T>& B );
//    friend void View<T>
//    ( Tensor<T>& A, Tensor<T>& B, Int i, Int j, Int height, Int width );
//    friend void View1x2<T>( Tensor<T>& A, Tensor<T>& BL, Tensor<T>& BR );
//    friend void View2x1<T>( Tensor<T>& A, Tensor<T>& BT, Tensor<T>& BB );
//    friend void View2x2<T>
//    ( Tensor<T>& A, Tensor<T>& BTL, Tensor<T>& BTR,
//                    Tensor<T>& BBL, Tensor<T>& BBR );

//    friend void LockedView<T>( Tensor<T>& A, const Tensor<T>& B );
//    friend void LockedView<T>
//    ( Tensor<T>& A, const Tensor<T>& B, Int i, Int j, Int height, Int width );
//    friend void LockedView1x2<T>
//    ( Tensor<T>& A, const Tensor<T>& BL, const Tensor<T>& BR );
//    friend void LockedView2x1<T>
//    ( Tensor<T>& A, const Tensor<T>& BT, const Tensor<T>& BB );
//    friend void LockedView2x2<T>
//    ( Tensor<T>& A, const Tensor<T>& BTL, const Tensor<T>& BTR,
//                    const Tensor<T>& BBL, const Tensor<T>& BBR );
};

} // namespace elem

#endif // ifndef TMEN_CORE_TENSOR_HPP
