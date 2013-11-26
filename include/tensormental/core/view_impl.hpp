/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef TMEN_CORE_VIEW_IMPL_HPP
#define TMEN_CORE_VIEW_IMPL_HPP

namespace elem {

//template<typename T,Distribution U,Distribution V>
//inline void HandleDiagPath
//( DistTensor<T,U,V>& A, const DistTensor<T,U,V>& B )
//{ }
//
//template<typename T>
//inline void HandleDiagPath
//( DistTensor<T,MD,STAR>& A, const DistTensor<T,MD,STAR>& B )
//{ A.diagPath_ = B.diagPath_; } 
//
//template<typename T>
//inline void HandleDiagPath
//( DistTensor<T,STAR,MD>& A, const DistTensor<T,STAR,MD>& B )
//{ A.diagPath_ = B.diagPath_; } 

template<typename T>
inline void View( Tensor<T>& A, Tensor<T>& B )
{
#ifndef RELEASE
    CallStackEntry entry("View");
#endif
    A.memory_.Empty();
    A.height_   = B.height_;
    A.width_    = B.width_;
    A.ldim_     = B.ldim_;
    A.data_     = B.data_;
    A.viewType_ = VIEW;
}

template<typename T>
inline Tensor<T> View( Tensor<T>& B )
{
    Tensor<T> A;
    View( A, B );
    return A;
}

//template<typename T,Distribution U,Distribution V>
//inline void View( DistTensor<T,U,V>& A, DistTensor<T,U,V>& B )
//{
//#ifndef RELEASE
//    CallStackEntry entry("View");
//#endif
//    A.Empty();
//    A.grid_ = B.grid_;
//    A.height_ = B.Height();
//    A.width_ = B.Width();
//    A.colAlignment_ = B.ColAlignment();
//    A.rowAlignment_ = B.RowAlignment();
//    HandleDiagPath( A, B );
//    A.viewType_ = VIEW;
//    if( A.Participating() )
//    {
//        A.colShift_ = B.ColShift();
//        A.rowShift_ = B.RowShift();
//        View( A.Tensor(), B.Tensor() );
//    }
//    else
//    {
//        A.colShift_ = 0;
//        A.rowShift_ = 0;
//    }
//}
//
//template<typename T,Distribution U,Distribution V>
//inline DistTensor<T,U,V> View( DistTensor<T,U,V>& B )
//{
//    DistTensor<T,U,V> A(B.Grid());
//    View( A, B );
//    return A;
//}

template<typename T>
inline void LockedView( Tensor<T>& A, const Tensor<T>& B )
{
#ifndef RELEASE
    CallStackEntry entry("LockedView");
#endif
    A.memory_.Empty();
    A.height_   = B.height_;
    A.width_    = B.width_;
    A.ldim_     = B.ldim_;
    A.data_     = B.data_;
    A.viewType_ = LOCKED_VIEW;
}

template<typename T>
inline Tensor<T> LockedView( const Tensor<T>& B )
{
    Tensor<T> A;
    LockedView( A, B );
    return A;
}

//template<typename T,Distribution U,Distribution V>
//inline void LockedView( DistTensor<T,U,V>& A, const DistTensor<T,U,V>& B )
//{
//#ifndef RELEASE
//    CallStackEntry entry("LockedView");
//#endif
//    A.Empty();
//    A.grid_ = B.grid_;
//    A.height_ = B.Height();
//    A.width_ = B.Width();
//    A.colAlignment_ = B.ColAlignment();
//    A.rowAlignment_ = B.RowAlignment();
//    HandleDiagPath( A, B );
//    A.viewType_ = LOCKED_VIEW;
//    if( A.Participating() )
//    {
//        A.colShift_ = B.ColShift();
//        A.rowShift_ = B.RowShift();
//        LockedView( A.Tensor(), B.LockedTensor() );
//    }
//    else 
//    {
//        A.colShift_ = 0;
//        A.rowShift_ = 0;
//    }
//}
//
//template<typename T,Distribution U,Distribution V>
//inline DistTensor<T,U,V> LockedView( const DistTensor<T,U,V>& B )
//{
//    DistTensor<T,U,V> A(B.Grid());
//    LockedView( A, B );
//    return A;
//}

template<typename T>
inline void View
( Tensor<T>& A, Tensor<T>& B,
  Int i, Int j, Int height, Int width )
{
#ifndef RELEASE
    CallStackEntry entry("View");
    if( i < 0 || j < 0 )
        LogicError("Indices must be non-negative");
    if( height < 0 || width < 0 )
        LogicError("Height and width must be non-negative");
    if( (i+height) > B.Height() || (j+width) > B.Width() )
    {
        std::ostringstream msg;
        msg << "Trying to view outside of a Tensor: "
            << "(" << i << "," << j << ") up to (" 
            << i+height-1 << "," << j+width-1 << ") "
            << "of " << B.Height() << " x " << B.Width() << " Tensor.";
        LogicError( msg.str() );
    }
#endif
    A.memory_.Empty();
    A.height_   = height;
    A.width_    = width;
    A.ldim_     = B.ldim_;
    A.data_     = &B.data_[i+j*B.ldim_];
    A.viewType_ = VIEW;
}

template<typename T>
inline Tensor<T> View( Tensor<T>& B, Int i, Int j, Int height, Int width )
{
    Tensor<T> A;
    View( A, B, i, j, height, width );
    return A;
}

//template<typename T,Distribution U,Distribution V>
//inline void View
//( DistTensor<T,U,V>& A, DistTensor<T,U,V>& B,
//  Int i, Int j, Int height, Int width )
//{
//#ifndef RELEASE
//    CallStackEntry entry("View");
//    B.AssertValidSubmatrix( i, j, height, width );
//#endif
//    A.Empty();
//
//    const elem::Grid& g = B.Grid();
//    const Int colStride = B.ColStride();
//    const Int rowStride = B.RowStride();
//
//    A.grid_ = &g;
//    A.height_ = height;
//    A.width_  = width;
//
//    A.colAlignment_ = (B.ColAlignment()+i) % colStride;
//    A.rowAlignment_ = (B.RowAlignment()+j) % rowStride;
//    HandleDiagPath( A, B );
//    A.viewType_ = VIEW;
//
//    if( A.Participating() )
//    {
//        const Int colRank = B.ColRank();
//        const Int rowRank = B.RowRank();
//        A.colShift_ = Shift( colRank, A.ColAlignment(), colStride );
//        A.rowShift_ = Shift( rowRank, A.RowAlignment(), rowStride );
//
//        const Int localHeightBehind = Length(i,B.ColShift(),colStride);
//        const Int localWidthBehind  = Length(j,B.RowShift(),rowStride);
//
//        const Int localHeight = Length( height, A.ColShift(), colStride );
//        const Int localWidth  = Length( width,  A.RowShift(), rowStride );
//
//        View
//        ( A.Tensor(), B.Tensor(), 
//          localHeightBehind, localWidthBehind, localHeight, localWidth );
//    }
//    else
//    {
//        A.colShift_ = 0;
//        A.rowShift_ = 0;
//    }
//}
//
//template<typename T,Distribution U,Distribution V>
//inline DistTensor<T,U,V> View
//( DistTensor<T,U,V>& B, Int i, Int j, Int height, Int width )
//{
//    DistTensor<T,U,V> A(B.Grid());
//    View( A, B, i, j, height, width );
//    return A;
//}

template<typename T>
inline void LockedView
( Tensor<T>& A, const Tensor<T>& B,
  Int i, Int j, Int height, Int width )
{
#ifndef RELEASE
    CallStackEntry entry("LockedView");
    if( i < 0 || j < 0 )
        LogicError("Indices must be non-negative");
    if( height < 0 || width < 0 )
        LogicError("Height and width must be non-negative");
    if( (i+height) > B.Height() || (j+width) > B.Width() )
    {
        std::ostringstream msg;
        msg << "Trying to view outside of a Tensor: "
            << "(" << i << "," << j << ") up to (" 
            << i+height-1 << "," << j+width-1 << ") "
            << "of " << B.Height() << " x " << B.Width() << " Tensor.";
        LogicError( msg.str() );
    }
#endif
    A.memory_.Empty();
    A.height_   = height;
    A.width_    = width;
    A.ldim_     = B.ldim_;
    A.data_     = &B.data_[i+j*B.ldim_];
    A.viewType_ = LOCKED_VIEW;
}

template<typename T>
inline Tensor<T> LockedView
( const Tensor<T>& B, Int i, Int j, Int height, Int width )
{
    Tensor<T> A;
    LockedView( A, B, i, j, height, width );
    return A;
}

//template<typename T,Distribution U,Distribution V>
//inline void LockedView
//( DistTensor<T,U,V>& A, const DistTensor<T,U,V>& B,
//  Int i, Int j, Int height, Int width )
//{
//#ifndef RELEASE
//    CallStackEntry entry("LockedView");
//    B.AssertValidSubmatrix( i, j, height, width );
//#endif
//    A.Empty();
//
//    const elem::Grid& g = B.Grid();
//    const Int colStride = B.ColStride();
//    const Int rowStride = B.RowStride();
//    const Int colRank = B.ColRank();
//    const Int rowRank = B.RowRank();
//
//    A.grid_ = &g;
//    A.height_ = height;
//    A.width_  = width;
//
//    A.colAlignment_ = (B.ColAlignment()+i) % colStride;
//    A.rowAlignment_ = (B.RowAlignment()+j) % rowStride;
//    HandleDiagPath( A, B );
//    A.viewType_ = LOCKED_VIEW;
//
//    if( A.Participating() )
//    {
//        A.colShift_ = Shift( colRank, A.ColAlignment(), colStride );
//        A.rowShift_ = Shift( rowRank, A.RowAlignment(), rowStride );
//
//        const Int localHeightBehind = Length(i,B.ColShift(),colStride);
//        const Int localWidthBehind  = Length(j,B.RowShift(),rowStride);
//
//        const Int localHeight = Length( height, A.ColShift(), colStride );
//        const Int localWidth  = Length( width,  A.RowShift(), rowStride );
//
//        LockedView
//        ( A.Tensor(), B.LockedTensor(), 
//          localHeightBehind, localWidthBehind, localHeight, localWidth );
//    }
//    else
//    {
//        A.colShift_ = 0;
//        A.rowShift_ = 0;
//    }
//}
//
//template<typename T,Distribution U,Distribution V>
//inline DistTensor<T,U,V> LockedView
//( const DistTensor<T,U,V>& B, Int i, Int j, Int height, Int width )
//{
//    DistTensor<T,U,V> A(B.Grid());
//    LockedView( A, B, i, j, height, width );
//    return A;
//}

template<typename T>
inline void View1x2
( Tensor<T>& A,
  Tensor<T>& BL, Tensor<T>& BR )
{
#ifndef RELEASE
    CallStackEntry entry("View1x2");
    if( BL.Height() != BR.Height() )
        LogicError("1x2 must have consistent height to combine");
    if( BL.LDim() != BR.LDim() )
        LogicError("1x2 must have consistent ldims to combine");
    if( BR.Buffer() != (BL.Buffer()+BL.LDim()*BL.Width()) )
        LogicError("1x2 must have contiguous memory");
#endif
    A.memory_.Empty();
    A.height_   = BL.height_;
    A.width_    = BL.width_ + BR.width_;
    A.ldim_     = BL.ldim_;
    A.data_     = BL.data_;
    A.viewType_ = VIEW;
}

template<typename T>
inline Tensor<T> View1x2( Tensor<T>& BL, Tensor<T>& BR )
{
    Tensor<T> A;
    View1x2( A, BL, BR );
    return A;
}

//template<typename T,Distribution U,Distribution V>
//inline void View1x2
//( DistTensor<T,U,V>& A, DistTensor<T,U,V>& BL, DistTensor<T,U,V>& BR )
//{
//#ifndef RELEASE
//    CallStackEntry entry("View1x2");
//    AssertConforming1x2( BL, BR );
//    BL.AssertSameGrid( BR.Grid() );
//#endif
//    A.Empty();
//    A.grid_ = BL.grid_;
//    A.height_ = BL.Height();
//    A.width_ = BL.Width() + BR.Width();
//    A.colAlignment_ = BL.ColAlignment();
//    A.rowAlignment_ = BL.RowAlignment();
//    HandleDiagPath( A, BL );
//    A.viewType_ = VIEW;
//    if( A.Participating() )
//    {
//        A.colShift_ = BL.ColShift();
//        A.rowShift_ = BL.RowShift();
//        View1x2( A.Tensor(), BL.Tensor(), BR.Tensor() );
//    }
//    else
//    {
//        A.colShift_ = 0;
//        A.rowShift_ = 0;
//    }
//}
//
//template<typename T,Distribution U,Distribution V>
//inline DistTensor<T,U,V> View1x2( DistTensor<T,U,V>& BL, DistTensor<T,U,V>& BR )
//{
//    DistTensor<T,U,V> A(BL.Grid());
//    View1x2( A, BL, BR );
//    return A;
//}

template<typename T>
inline void LockedView1x2
( Tensor<T>& A, const Tensor<T>& BL, const Tensor<T>& BR )
{
#ifndef RELEASE
    CallStackEntry entry("LockedView1x2");
    if( BL.Height() != BR.Height() )
        LogicError("1x2 must have consistent height to combine");
    if( BL.LDim() != BR.LDim() )
        LogicError("1x2 must have consistent ldims to combine");
    if( BR.LockedBuffer() != (BL.LockedBuffer()+BL.LDim()*BL.Width()) )
        LogicError("1x2 must have contiguous memory");
#endif
    A.memory_.Empty();
    A.height_   = BL.height_;
    A.width_    = BL.width_ + BR.width_;
    A.ldim_     = BL.ldim_;
    A.data_     = BL.data_;
    A.viewType_ = LOCKED_VIEW;
}

template<typename T>
inline Tensor<T> LockedView1x2( const Tensor<T>& BL, const Tensor<T>& BR )
{
    Tensor<T> A;
    LockedView1x2( A, BL, BR );
    return A;
}

//template<typename T,Distribution U,Distribution V>
//inline void LockedView1x2
//(       DistTensor<T,U,V>& A,
//  const DistTensor<T,U,V>& BL,
//  const DistTensor<T,U,V>& BR )
//{
//#ifndef RELEASE
//    CallStackEntry entry("LockedView1x2");
//    AssertConforming1x2( BL, BR );
//    BL.AssertSameGrid( BR.Grid() );
//#endif
//    A.Empty();
//    A.grid_ = BL.grid_;
//    A.height_ = BL.Height();
//    A.width_ = BL.Width() + BR.Width();
//    A.colAlignment_ = BL.ColAlignment();
//    A.rowAlignment_ = BL.RowAlignment();
//    HandleDiagPath( A, BL );
//    A.viewType_ = LOCKED_VIEW;
//    if( A.Participating() )
//    {
//        A.colShift_ = BL.ColShift();
//        A.rowShift_ = BL.RowShift();
//        LockedView1x2( A.Tensor(), BL.LockedTensor(), BR.LockedTensor() );
//    }
//    else
//    {
//        A.colShift_ = 0;
//        A.rowShift_ = 0;
//    }
//}
//
//template<typename T,Distribution U,Distribution V>
//inline DistTensor<T,U,V> LockedView1x2
//( const DistTensor<T,U,V>& BL, const DistTensor<T,U,V>& BR )
//{
//    DistTensor<T,U,V> A(BL.Grid());
//    LockedView1x2( A, BL, BR );
//    return A;
//}

template<typename T>
inline void View2x1( Tensor<T>& A, Tensor<T>& BT, Tensor<T>& BB )
{
#ifndef RELEASE
    CallStackEntry entry("View2x1");
    if( BT.Width() != BB.Width() )
        LogicError("2x1 must have consistent width to combine");
    if( BT.LDim() != BB.LDim() )
        LogicError("2x1 must have consistent ldim to combine");
    if( BB.Buffer() != (BT.Buffer() + BT.Height()) )
        LogicError("2x1 must have contiguous memory");
#endif
    A.memory_.Empty();
    A.height_   = BT.height_ + BB.height_;
    A.width_    = BT.width_;
    A.ldim_     = BT.ldim_;
    A.data_     = BT.data_;
    A.viewType_ = VIEW;
}

template<typename T>
inline Tensor<T> View2x1( Tensor<T>& BT, Tensor<T>& BB )
{
    Tensor<T> A;
    View2x1( A, BT, BB );
    return A;
}

//template<typename T,Distribution U,Distribution V>
//inline void View2x1
//( DistTensor<T,U,V>& A, DistTensor<T,U,V>& BT, DistTensor<T,U,V>& BB )
//{
//#ifndef RELEASE
//    CallStackEntry entry("View2x1");
//    AssertConforming2x1( BT, BB );
//    BT.AssertSameGrid( BB.Grid() );
//#endif
//    A.Empty();
//    A.grid_ = BT.grid_;
//    A.height_ = BT.Height() + BB.Height();
//    A.width_ = BT.Width();
//    A.colAlignment_ = BT.ColAlignment();
//    A.rowAlignment_ = BT.RowAlignment();
//    HandleDiagPath( A, BT );
//    A.viewType_ = LOCKED_VIEW;
//    if( A.Participating() )
//    {
//        A.colShift_ = BT.ColShift();
//        A.rowShift_ = BT.RowShift();
//        View2x1( A.Tensor(), BT.Tensor(), BB.Tensor() );
//    }
//    else
//    {
//        A.colShift_ = 0;
//        A.rowShift_ = 0;
//    }
//}
//
//template<typename T,Distribution U,Distribution V>
//inline DistTensor<T,U,V> View2x1( DistTensor<T,U,V>& BT, DistTensor<T,U,V>& BB )
//{
//    DistTensor<T,U,V> A(BT.Grid());
//    View2x1( A, BT, BB );
//    return A;
//}

template<typename T>
inline void LockedView2x1
( Tensor<T>& A, const Tensor<T>& BT, const Tensor<T>& BB )
{
#ifndef RELEASE
    CallStackEntry entry("LockedView2x1");
    if( BT.Width() != BB.Width() )
        LogicError("2x1 must have consistent width to combine");
    if( BT.LDim() != BB.LDim() )
        LogicError("2x1 must have consistent ldim to combine");
    if( BB.LockedBuffer() != (BT.LockedBuffer() + BT.Height()) )
        LogicError("2x1 must have contiguous memory");
#endif
    A.memory_.Empty();
    A.height_   = BT.height_ + BB.height_;
    A.width_    = BT.width_;
    A.ldim_     = BT.ldim_;
    A.data_     = BT.data_;
    A.viewType_ = LOCKED_VIEW;
}

template<typename T>
inline Tensor<T> LockedView2x1
( const Tensor<T>& BT, const Tensor<T>& BB )
{
    Tensor<T> A;
    LockedView2x1( A, BT, BB );
    return A;
}

//template<typename T,Distribution U,Distribution V>
//inline void LockedView2x1
//(       DistTensor<T,U,V>& A,
//  const DistTensor<T,U,V>& BT,
//  const DistTensor<T,U,V>& BB )
//{
//#ifndef RELEASE
//    CallStackEntry entry("LockedView2x1");
//    AssertConforming2x1( BT, BB );
//    BT.AssertSameGrid( BB.Grid() );
//#endif
//    A.Empty();
//    A.grid_ = BT.grid_;
//    A.height_ = BT.Height() + BB.Height();
//    A.width_ = BT.Width();
//    A.colAlignment_ = BT.ColAlignment();
//    A.rowAlignment_ = BT.RowAlignment();
//    HandleDiagPath( A, BT );
//    A.viewType_ = LOCKED_VIEW;
//    if( A.Participating() )
//    {
//        A.colShift_ = BT.ColShift();
//        A.rowShift_ = BT.RowShift();
//        LockedView2x1( A.Tensor(), BT.LockedTensor(), BB.LockedTensor() );
//    }
//    else
//    {
//        A.colShift_ = 0;
//        A.rowShift_ = 0;
//    }
//}
//
//template<typename T,Distribution U,Distribution V>
//inline DistTensor<T,U,V> LockedView2x1
//( const DistTensor<T,U,V>& BT, const DistTensor<T,U,V>& BB )
//{
//    DistTensor<T,U,V> A(BT.Grid());
//    LockedView2x1( A, BT, BB );
//    return A;
//}

template<typename T>
inline void View2x2
( Tensor<T>& A,
  Tensor<T>& BTL, Tensor<T>& BTR,
  Tensor<T>& BBL, Tensor<T>& BBR )
{
#ifndef RELEASE
    CallStackEntry entry("View2x2");
    if( BTL.Width() != BBL.Width()   ||
        BTR.Width() != BBR.Width()   ||
        BTL.Height() != BTR.Height() ||
        BBL.Height() != BBR.Height()   )
        LogicError("2x2 must conform to combine");
    if( BTL.LDim() != BTR.LDim() ||
        BTR.LDim() != BBL.LDim() ||
        BBL.LDim() != BBR.LDim()   )
        LogicError("2x2 must have consistent ldims to combine");
    if( BBL.Buffer() != (BTL.Buffer() + BTL.Height()) ||
        BBR.Buffer() != (BTR.Buffer() + BTR.Height()) ||
        BTR.Buffer() != (BTL.Buffer() + BTL.LDim()*BTL.Width()) )
        LogicError("2x2 must have contiguous memory");
#endif
    A.memory_.Empty();
    A.height_   = BTL.height_ + BBL.height_;
    A.width_    = BTL.width_ + BTR.width_;
    A.ldim_     = BTL.ldim_;
    A.data_     = BTL.data_;
    A.viewType_ = VIEW;
}

template<typename T>
inline Tensor<T> View2x2
( Tensor<T>& BTL, Tensor<T>& BTR,
  Tensor<T>& BBL, Tensor<T>& BBR )
{
    Tensor<T> A;
    View2x2( A, BTL, BTR, BBL, BBR );
    return A;
}

//template<typename T,Distribution U,Distribution V>
//inline void View2x2
//( DistTensor<T,U,V>& A,
//  DistTensor<T,U,V>& BTL, DistTensor<T,U,V>& BTR,
//  DistTensor<T,U,V>& BBL, DistTensor<T,U,V>& BBR )
//{
//#ifndef RELEASE
//    CallStackEntry entry("View2x2");
//    AssertConforming2x2( BTL, BTR, BBL, BBR );
//    BTL.AssertSameGrid( BTR.Grid() );
//    BTL.AssertSameGrid( BBL.Grid() );
//    BTL.AssertSameGrid( BBR.Grid() );
//#endif
//    A.Empty();
//    A.grid_ = BTL.grid_;
//    A.height_ = BTL.Height() + BBL.Height();
//    A.width_ = BTL.Width() + BTR.Width();
//    A.colAlignment_ = BTL.ColAlignment();
//    A.rowAlignment_ = BTL.RowAlignment();
//    HandleDiagPath( A, BTL );
//    A.viewType_ = VIEW;
//    if( A.Participating() )
//    {
//        A.colShift_ = BTL.ColShift();
//        A.rowShift_ = BTL.RowShift();
//        View2x2
//        ( A.Tensor(), BTL.Tensor(), BTR.Tensor(),
//                      BBL.Tensor(), BBR.Tensor() ); 
//    }
//    else
//    {
//        A.colShift_ = 0;
//        A.rowShift_ = 0;
//    }
//}
//
//template<typename T,Distribution U,Distribution V>
//inline DistTensor<T,U,V> View2x2
//( DistTensor<T,U,V>& BTL, DistTensor<T,U,V>& BTR,
//  DistTensor<T,U,V>& BBL, DistTensor<T,U,V>& BBR )
//{
//    DistTensor<T,U,V> A(BTL.Grid());
//    View2x2( A, BTL, BTR, BBL, BBR );
//    return A;
//}

template<typename T>
inline void LockedView2x2
(       Tensor<T>& A,
  const Tensor<T>& BTL,
  const Tensor<T>& BTR,
  const Tensor<T>& BBL,
  const Tensor<T>& BBR )
{
#ifndef RELEASE
    CallStackEntry entry("LockedView2x2");
    if( BTL.Width() != BBL.Width()   ||
        BTR.Width() != BBR.Width()   ||
        BTL.Height() != BTR.Height() ||
        BBL.Height() != BBR.Height()   )
        LogicError("2x2 must conform to combine");
    if( BTL.LDim() != BTR.LDim() ||
        BTR.LDim() != BBL.LDim() ||
        BBL.LDim() != BBR.LDim()   )
        LogicError("2x2 must have consistent ldims to combine");
    if( BBL.LockedBuffer() != (BTL.LockedBuffer() + BTL.Height()) ||
        BBR.LockedBuffer() != (BTR.LockedBuffer() + BTR.Height()) ||
        BTR.LockedBuffer() != (BTL.LockedBuffer() + BTL.LDim()*BTL.Width()) )
        LogicError("2x2 must have contiguous memory");
#endif
    A.memory_.Empty();
    A.height_   = BTL.height_ + BBL.height_;
    A.width_    = BTL.width_ + BTR.width_;
    A.ldim_     = BTL.ldim_;
    A.data_     = BTL.data_;
    A.viewType_ = LOCKED_VIEW;
}

template<typename T>
inline Tensor<T> LockedView2x2
( const Tensor<T>& BTL, const Tensor<T>& BTR,
  const Tensor<T>& BBL, const Tensor<T>& BBR )
{
    Tensor<T> A;
    LockedView2x2( A, BTL, BTR, BBL, BBR );
    return A;
}

//template<typename T,Distribution U,Distribution V>
//inline void LockedView2x2
//(       DistTensor<T,U,V>& A,
//  const DistTensor<T,U,V>& BTL, const DistTensor<T,U,V>& BTR,
//  const DistTensor<T,U,V>& BBL, const DistTensor<T,U,V>& BBR )
//{
//#ifndef RELEASE
//    CallStackEntry entry("LockedView2x2");
//    AssertConforming2x2( BTL, BTR, BBL, BBR );
//    BTL.AssertSameGrid( BTR.Grid() );
//    BTL.AssertSameGrid( BBL.Grid() );
//    BTL.AssertSameGrid( BBR.Grid() );
//#endif
//    A.Empty();
//    A.grid_ = BTL.grid_;
//    A.height_ = BTL.Height() + BBL.Height();
//    A.width_ = BTL.Width() + BTR.Width();
//    A.colAlignment_ = BTL.ColAlignment();
//    A.rowAlignment_ = BTL.RowAlignment();
//    HandleDiagPath( A, BTL );
//    A.viewType_ = LOCKED_VIEW;
//    if( A.Participating() )
//    {
//        A.colShift_ = BTL.ColShift();
//        A.rowShift_ = BTL.RowShift();
//        LockedView2x2
//        ( A.Tensor(), BTL.LockedTensor(), BTR.LockedTensor(),
//                      BBL.LockedTensor(), BBR.LockedTensor() );
//    }
//    else
//    {
//        A.colShift_ = 0;
//        A.rowShift_ = 0;
//    }
//}
//
//template<typename T,Distribution U,Distribution V>
//inline DistTensor<T,U,V> LockedView2x2
//( const DistTensor<T,U,V>& BTL, const DistTensor<T,U,V>& BTR,
//  const DistTensor<T,U,V>& BBL, const DistTensor<T,U,V>& BBR )
//{
//    DistTensor<T,U,V> A(BTL.Grid());
//    LockedView2x2( A, BTL, BTR, BBL, BBR );
//    return A;
//}

} // namespace elem

#endif // ifndef TMEN_CORE_VIEW_IMPL_HPP
