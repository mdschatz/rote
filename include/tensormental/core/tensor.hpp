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
    
    void AssertValidDimensions( const std::vector<Int>& shape ) const;
    void AssertValidDimensions( const std::vector<Int>& shape, const std::vector<Int>& ldims ) const;
    void AssertValidEntry( const std::vector<Int>& indices ) const;
    void AssertValidIndices() const;
    
    //
    // Constructors
    // 

    Tensor( bool fixed=false );
    Tensor( const std::vector<Int>& indices, bool fixed=false );
    Tensor( const std::vector<Int>& indices, const std::vector<Int>& shape, bool fixed=false );
    Tensor( const std::vector<Int>& indices, const std::vector<Int>& shape, const std::vector<Int>& ldims, bool fixed=false );
    Tensor
    ( const std::vector<Int>& indices, const std::vector<Int>& shape, const T* buffer, const std::vector<Int>& ldims, bool fixed=false );
    Tensor( const std::vector<Int>& indices, const std::vector<Int>& shape, T* buffer, const std::vector<Int>& ldims, bool fixed=false );
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

    virtual ~Tensor();

    //
    // Basic information
    //

    Int Order() const;
    std::vector<Int> Shape() const;
    Int Dimension(Int mode) const;
    std::vector<Int> Indices() const;
    Int ModeStride(Int mode) const;

    Int ModeOfIndex(Int index) const;
    Int IndexOfMode(Int mode) const;

    Int LDim(Int mode) const;
    Int MemorySize() const;

    Int LinearOffset(const std::vector<int>& index) const;
    T* Buffer();
    T* Buffer( const std::vector<Int>& index );

    const T* LockedBuffer() const;
    const T* LockedBuffer( const std::vector<Int>& index ) const;

    //
    // Entry manipulation
    //

    T Get( const std::vector<Int>& index ) const;
    void Set( const std::vector<Int>& index, T alpha );
    void Update( const std::vector<Int>& index, T alpha );

    //void GetDiagonal( Tensor<T>& d, Int offset=0 ) const;
    //Tensor<T> GetDiagonal( Int offset=0 ) const;

    //void SetDiagonal( const Tensor<T>& d, Int offset=0 );
    //void UpdateDiagonal( const Tensor<T>& d, Int offset=0 );

    //
    // Though the following routines are meant for complex data, all but four
    // logically apply to real data.
    //

    BASE(T) GetRealPart( const std::vector<Int>& index ) const;
    BASE(T) GetImagPart( const std::vector<Int>& index ) const;
    void SetRealPart( const std::vector<Int>& index, BASE(T) alpha );
    // Only valid for complex data
    void SetImagPart( const std::vector<Int>& index, BASE(T) alpha );
    void UpdateRealPart( const std::vector<Int>& index, BASE(T) alpha );
    // Only valid for complex data
    void UpdateImagPart( const std::vector<Int>& index, BASE(T) alpha );

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

    void Attach( const std::vector<Int>& shape, T* buffer, const std::vector<Int>& ldims );
    void LockedAttach
    ( const std::vector<Int>& shape, const T* buffer, const std::vector<Int>& ldims );

    // Use this memory *as if it were not a view*, but do not take control of 
    // its deallocation. If Resize() forces reallocation, this buffer is 
    // released from control but not deleted.
    //void Control( Int height, Int width, T* buffer, Int ldim );

    //
    // Utilities
    //

    const Tensor<T>& operator=( const Tensor<T>& A );

    void Empty();
    void ResizeTo( const std::vector<Int>& shape );
    void ResizeTo( const std::vector<Int>& shape, const std::vector<Int>& ldims );

private:
    std::vector<Int> indices_;
    std::vector<Int> shape_;
    std::vector<Int> strides_;
    std::vector<Int> ldims_;

    //Index<->Mode maps
    //NOTE: Move this information to separate class
    std::map<Int, Int> index2modeMap_;
    std::map<Int, Int> mode2indexMap_;

    ViewType viewType_;

    const T* data_;
    Memory<T> memory_;

    void ComplainIfReal() const;

    const T& Get_( const std::vector<Int>& index ) const;
    T& Set_( const std::vector<Int>& index );

    void SetLDims(const std::vector<Int>& shape);
    void SetIndexMaps();

    // These bypass fixed-size checking and are used by DistTensor
    void Empty_();
    void ResizeTo_( const std::vector<Int>& shape );
    void ResizeTo_( const std::vector<Int>& shape, const std::vector<Int>& ldims );
    void Control_( const std::vector<Int>& shape, T* buffer, const std::vector<Int>& ldims );
    void Attach_( const std::vector<Int>& shape, T* buffer, const std::vector<Int>& ldims );
    void LockedAttach_( const std::vector<Int>& shape, const T* buffer, const std::vector<Int>& ldims );
    
    template <typename F> 
    friend class Tensor;
    template <typename F> 
    friend class DistTensor;
    friend class AbstractDistTensor<T>;

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

} // namespace tmen

#endif // ifndef TMEN_CORE_TENSOR_HPP
