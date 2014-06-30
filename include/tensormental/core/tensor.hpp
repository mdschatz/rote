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
#include <map>
#include <set>
#include "tensormental/core/view_decl.hpp"

namespace tmen {

// Tensor base for arbitrary rings
template<typename T>
class Tensor
{
public:    
    //
    // Assertions
    //
    
    void AssertValidDimensions( const ObjShape& shape ) const;
    void AssertValidDimensions( const ObjShape& shape, const std::vector<Unsigned>& ldims ) const;
    void AssertValidEntry( const Location& loc ) const;
    void AssertMergeableModes( const std::vector<ModeArray>& oldModes ) const;
    void AssertSplittableModes( const ModeArray& oldModes, const std::vector<ObjShape>& newShape) const;
    
    //
    // Constructors
    // 

    Tensor( bool fixed=false );
    Tensor( const Unsigned order, bool fixed = false);
    Tensor( const ObjShape& shape, bool fixed=false );
    Tensor( const ObjShape& shape, const std::vector<Unsigned>& ldims, bool fixed=false );
    Tensor
    ( const ObjShape& shape, const T* buffer, const std::vector<Unsigned>& ldims, bool fixed=false );
    Tensor( const ObjShape& shape, T* buffer, const std::vector<Unsigned>& ldims, bool fixed=false );
    Tensor( const Tensor<T>& A );

    // Move constructor
    //Tensor( Tensor<T>&& A );

    // Move assignment
    //Tensor<T>& operator=( Tensor<T>&& A );

    // Swap
    void Swap( Tensor<T>& A );

    //
    // Destructor
    //

    ~Tensor();

    //
    // Basic information
    //

    Unsigned Order() const;
    ObjShape Shape() const;
    Unsigned Dimension(Mode mode) const;
    Unsigned ModeStride(Mode mode) const;

    std::vector<Unsigned> LDims() const;
    Unsigned LDim(Mode mode) const;
    Unsigned MemorySize() const;

    Unsigned LinearOffset(const Location& loc) const;
    T* Buffer();
    T* Buffer( const Location& loc );

    const T* LockedBuffer() const;
    const T* LockedBuffer( const Location& loc ) const;

    //
    // Unit mode info
    //

    void RemoveUnitModes(const ModeArray& modes);
    void RemoveUnitMode(const Mode& mode);
    void IntroduceUnitMode(const Unsigned& modePosition);

    //
    // Entry manipulation
    //

    T Get( const Location& loc ) const;
    void Set( const Location& loc, T alpha );
    void Update( const Location& loc, T alpha );

    //void GetDiagonal( Tensor<T>& d, Int offset=0 ) const;
    //Tensor<T> GetDiagonal( Int offset=0 ) const;

    //void SetDiagonal( const Tensor<T>& d, Int offset=0 );
    //void UpdateDiagonal( const Tensor<T>& d, Int offset=0 );

    //
    // Though the following routines are meant for complex data, all but four
    // logically apply to real data.
    //

    BASE(T) GetRealPart( const Location& loc ) const;
    BASE(T) GetImagPart( const Location& loc ) const;
    void SetRealPart( const Location& loc, BASE(T) alpha );
    // Only valid for complex data
    void SetImagPart( const Location& loc, BASE(T) alpha );
    void UpdateRealPart( const Location& loc, BASE(T) alpha );
    // Only valid for complex data
    void UpdateImagPart( const Location& loc, BASE(T) alpha );

    //void GetRealPartOfDiagonal( Tensor<BASE(T)>& d, Int offset=0 ) const;
    //void GetImagPartOfDiagonal( Tensor<BASE(T)>& d, Int offset=0 ) const;
    //Tensor<BASE(T)> GetRealPartOfDiagonal( Int offset=0 ) const;
    //Tensor<BASE(T)> GetImagPartOfDiagonal( Int offset=0 ) const;

    //void SetRealPartOfDiagonal( const Tensor<BASE(T)>& d, Int offset=0 );
    // Only valid for complex data
    //void SetImagPartOfDiagonal( const Tensor<BASE(T)>& d, Int offset=0 );
    //void UpdateRealPartOfDiagonal( const Tensor<BASE(T)>& d, Int offset=0 );
    // Only valid for complex data
    //void UpdateImagPartOfDiagonal( const Tensor<BASE(T)>& d, Int offset=0 );

    //
    // Viewing other matrix instances (or buffers)
    //

    bool Owner()       const;
    bool Shrinkable()  const;
    bool FixedSize()   const;
    bool Viewing()     const;
    bool Locked()      const;

    void Attach( const ObjShape& shape, T* buffer, const std::vector<Unsigned>& ldims );
    void LockedAttach
    ( const ObjShape& shape, const T* buffer, const std::vector<Unsigned>& ldims );

    // Use this memory *as if it were not a view*, but do not take control of 
    // its deallocation. If Resize() forces reallocation, this buffer is 
    // released from control but not deleted.
    //void Control( Int height, Int width, T* buffer, Int ldim );

    //
    // Utilities
    //

    const Tensor<T>& operator=( const Tensor<T>& A );

    void Empty();
    void ResizeTo( const ObjShape& shape );
    void ResizeTo( const ObjShape& shape, const std::vector<Unsigned>& ldims );

private:
    ObjShape shape_;
    std::vector<Unsigned> strides_;
    std::vector<Unsigned> ldims_;

    ViewType viewType_;

    const T* data_;
    Memory<T> memory_;

    void ComplainIfReal() const;

    const T& Get_( const Location& loc ) const;
    T& Set_( const Location& loc );

    void SetLDims(const ObjShape& shape);

    // These bypass fixed-size checking and are used by DistTensor
    void Empty_();
    void ResizeTo( const Tensor<T>& A);
    void ResizeTo_( const ObjShape& shape );
    void ResizeTo_( const ObjShape& shape, const std::vector<Unsigned>& ldims );
    void Control_( const ObjShape& shape, T* buffer, const std::vector<Unsigned>& ldims );
    void Attach_( const ObjShape& shape, T* buffer, const std::vector<Unsigned>& ldims );
    void LockedAttach_( const ObjShape& shape, const T* buffer, const std::vector<Unsigned>& ldims );
    
    template <typename F> 
    friend class Tensor;
    template <typename F> 
    friend class DistTensor;
    friend class AbstractDistTensor<T>;

    friend void ViewHelper<T>( Tensor<T>& A, const Tensor<T>& B, bool isLocked );
    friend void ViewHelper<T>
    ( Tensor<T>& A, const Tensor<T>& B, const Location& loc, const ObjShape& shape, bool isLocked );
    friend void View2x1Helper<T>( Tensor<T>& A, const Tensor<T>& BT, const Tensor<T>& BB, Mode mode, bool isLocked );
    friend void ViewAsLowerOrderHelper<T>( Tensor<T>& A, const Tensor<T>& B,
                                   const std::vector<ModeArray>& oldModes, bool isLocked );
    friend void ViewAsHigherOrderHelper<T>( Tensor<T>& A, const Tensor<T>& B,
                                   const ModeArray& oldModes, const std::vector<ObjShape>& newShape, bool isLocked );

    friend void View<T>( Tensor<T>& A, Tensor<T>& B);
    friend void LockedView<T>( Tensor<T>& A, const Tensor<T>& B);
    friend void View<T>( Tensor<T>& A, Tensor<T>& B, const Location& loc, const ObjShape& shape );
    friend void LockedView<T>( Tensor<T>& A, const Tensor<T>& B, const Location& loc, const ObjShape& shape );

    friend void View2x1<T>
    ( Tensor<T>& A,
      Tensor<T>& BT,
      Tensor<T>& BB, Mode mode );
    friend void LockedView2x1<T>
    (       Tensor<T>& A,
      const Tensor<T>& BT,
      const Tensor<T>& BB, Mode mode );

    friend
    void ViewAsLowerOrder<T>
    ( Tensor<T>& A,
      Tensor<T>& B,
      const std::vector<ModeArray>& oldModes );
    friend
    void LockedViewAsLowerOrder<T>
    ( Tensor<T>& A,
      const Tensor<T>& B,
      const std::vector<ModeArray>& oldModes );

    friend
    void ViewAsHigherOrder<T>
    ( Tensor<T>& A,
      Tensor<T>& B,
      const ModeArray& oldModes,
      const std::vector<ObjShape>& newShape );
    friend
    void LockedViewAsHigherOrder<T>
    ( Tensor<T>& A,
      const Tensor<T>& B,
      const ModeArray& oldModes,
      const std::vector<ObjShape>& newShape );
};

} // namespace tmen

#endif // ifndef TMEN_CORE_TENSOR_HPP
