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

namespace tmen {

template<typename T>
inline void View( Tensor<T>& A, Tensor<T>& B )
{
#ifndef RELEASE
    CallStackEntry entry("View");
#endif
    A.memory_.Empty();
    A.shape_ = B.shape_;
    A.ldims_     = B.ldims_;
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

template<typename T>
inline void View( DistTensor<T>& A, DistTensor<T>& B )
{
#ifndef RELEASE
    CallStackEntry entry("View");
#endif
    A.Empty();
    A.grid_ = B.grid_;
    A.shape_ = B.Shape();
    A.modeAlignments_ = B.Alignments();
    A.viewType_ = VIEW;
    if( A.Participating() )
    {
        A.modeShifts_ = B.ModeShifts();
        View( A.Tensor(), B.Tensor() );
    }
    else
    {
        std::fill(A.modeShifts_.begin(), A.modeShifts.end(), 0);
    }
}

template<typename T>
inline DistTensor<T> View( DistTensor<T>& B )
{
    DistTensor<T> A(B.Grid());
    View( A, B );
    return A;
}

template<typename T>
inline void LockedView( Tensor<T>& A, const Tensor<T>& B )
{
#ifndef RELEASE
    CallStackEntry entry("LockedView");
#endif
    A.memory_.Empty();
    A.shape_    = B.shape_;
    A.ldims_     = B.ldims_;
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

template<typename T>
inline void LockedView( DistTensor<T>& A, const DistTensor<T>& B )
{
#ifndef RELEASE
    CallStackEntry entry("LockedView");
#endif
    A.Empty();
    A.grid_ = B.grid_;
    A.shape_ = B.Shape();
    A.modeAlignments_ = B.Alignments();
    HandleDiagPath( A, B );
    A.viewType_ = LOCKED_VIEW;
    if( A.Participating() )
    {
        A.modeShifts_ = B.ModeShifts();
        LockedView( A.Tensor(), B.LockedTensor() );
    }
    else
    {
        std::fill(A.modeShifts_.begin(), A.modeShifts.end(), 0);
    }
}

template<typename T>
inline DistTensor<T> LockedView( const DistTensor<T>& B )
{
    DistTensor<T> A(B.Grid());
    LockedView( A, B );
    return A;
}

template<typename T>
inline void View
( Tensor<T>& A, Tensor<T>& B,
  const Location& loc, const ObjShape& shape )
{
#ifndef RELEASE
    CallStackEntry entry("View");
    const ObjShape shapeB = B.Shape();
    Location maxLoc(shape.size());
    ElemwiseSum(loc, shape, maxLoc);

    if( !ElemwiseLessThan(maxLoc, shapeB) )
    {
        Unsigned i;
        std::ostringstream msg;

        msg << "Trying to view outside of a Tensor: "
            << "(";
        if(loc.size() > 0)
            msg << loc[0];
        for(i = 1; i < loc.size(); i++)
            msg << ", " << loc[i];
        msg << ") up to (";
        if(loc.size() > 0)
            msg << maxLoc[0];
        for(i = 1; i < maxLoc.size(); i++)
            msg << ", " << maxLoc[i];
        msg << "of ";
        if(shapeB.size() > 0)
            msg << shapeB[0];
        for(i = 1; i < shapeB.size(); i++)
            msg << "x " << shapeB[i];
        msg << " Tensor.";
        LogicError( msg.str() );
    }
#endif
    A.memory_.Empty();
    A.shape_ = B.Shape();
    A.ldims_ = B.ldims_;
    A.data_     = B.Buffer(loc);
    A.viewType_ = VIEW;
}

template<typename T>
inline Tensor<T> View( Tensor<T>& B, const Location& loc, const ObjShape& shape )
{
    Tensor<T> A;
    View( A, B, loc, shape );
    return A;
}

template<typename T>
inline void View
( DistTensor<T>& A, DistTensor<T>& B,
  const Location& loc, const ObjShape& shape )
{
#ifndef RELEASE
    CallStackEntry entry("View");
    B.AssertValidSubtensor( loc, shape );
#endif
    A.Empty();

    Unsigned i;
    const Unsigned order = B.Order();
    const tmen::Grid& g = B.Grid();
    const std::vector<Unsigned> modeShifts = B.ModeShifts();
    const std::vector<Unsigned> modeWrapStrides = B.GridViewShape();


    A.grid_ = &g;
    A.shape_ = shape;

    for(i = 0; i < order; i++)
        A.modeAlignments_[i] = (B.ModeAlignment(i) + loc[i]) % modeWrapStrides[i];

    A.viewType_ = VIEW;

    if( A.Participating() )
    {
        const std::vector<Unsigned> modeRanks = B.GridViewLoc();
        A.modeShifts_ = Shifts(modeRanks, A.Alignments(), modeWrapStrides);

        const std::vector<Unsigned> localShapeBehind = Lengths(loc, B.ModeShifts(), modeWrapStrides);

        const std::vector<Unsigned> localShape = Lengths(shape, A.ModeShifts(), modeWrapStrides);

        View
        ( A.Tensor(), B.Tensor(),
          localShapeBehind, localShape );
    }
    else
    {
        std::fill(A.modeShifts_.begin(), A.modeShifts_.end(), 0);
    }
}

template<typename T>
inline DistTensor<T> View
( DistTensor<T>& B, const Location& loc, const ObjShape& shape )
{
    DistTensor<T> A(B.Grid());
    View( A, B, loc, shape );
    return A;
}

template<typename T>
inline void LockedView
( Tensor<T>& A, const Tensor<T>& B,
  const Location& loc, const ObjShape& shape )
{
#ifndef RELEASE
    CallStackEntry entry("LockedView");

    const ObjShape shapeB = B.Shape();
    Location maxLoc(shape.size());
    ElemwiseSum(loc, shape, maxLoc);

    if( !ElemwiseLessThan(maxLoc, shapeB) )
    {
        Unsigned i;
        std::ostringstream msg;

        msg << "Trying to view outside of a Tensor: "
            << "(";
        if(loc.size() > 0)
            msg << loc[0];
        for(i = 1; i < loc.size(); i++)
            msg << ", " << loc[i];
        msg << ") up to (";
        if(loc.size() > 0)
            msg << maxLoc[0];
        for(i = 1; i < maxLoc.size(); i++)
            msg << ", " << maxLoc[i];
        msg << "of ";
        if(shapeB.size() > 0)
            msg << shapeB[0];
        for(i = 1; i < shapeB.size(); i++)
            msg << "x " << shapeB[i];
        msg << " Tensor.";
        LogicError( msg.str() );
    }
#endif
    A.memory_.Empty();
    A.shape_ = shape;
    A.ldims_     = B.ldims_;
    A.data_     = &B.Buffer(loc);
    A.viewType_ = LOCKED_VIEW;
}

template<typename T>
inline Tensor<T> LockedView
( const Tensor<T>& B, const Location& loc, const ObjShape& shape )
{
    Tensor<T> A;
    LockedView( A, B, loc, shape );
    return A;
}

template<typename T>
inline void LockedView
( DistTensor<T>& A, const DistTensor<T>& B,
  const Location& loc, const ObjShape& shape )
{
#ifndef RELEASE
    CallStackEntry entry("LockedView");
    B.AssertValidSubtensor( loc, shape );
#endif
    A.Empty();

    Unsigned i;
    const Unsigned order = B.order();
    const tmen::Grid& g = B.Grid();
    const std::vector<Unsigned> modeShifts = B.ModeShifts();
    const std::vector<Unsigned> modeWrapStrides = B.GridViewShape();

    A.grid_ = &g;
    A.shape_ = shape;

    for(i = 0; i < order; i++)
        A.modeAlignments_[i] = (B.ModeAlignment(i) + loc[i]) % modeWrapStrides;

    HandleDiagPath( A, B );
    A.viewType_ = LOCKED_VIEW;

    if( A.Participating() )
    {
        const std::vector<Unsigned> modeRanks = B.GridViewLoc();
        A.modeShifts_ = Shifts(modeRanks, A.Alignments(), modeShifts);

        const std::vector<Unsigned> localShapeBehind = Lengths(loc, B.ModeShifts(), modeShifts);

        const std::vector<Unsigned> localShape = Lengths(shape, A.ModeShifts(), modeWrapStrides);

        LockedView
        ( A.Tensor(), B.LockedTensor(),
          localShapeBehind, localShape );
    }
    else
    {
        std::fill(A.modeShifts_.begin(), A.modeShifts_.end(), 0);
    }
}

template<typename T>
inline DistTensor<T> LockedView
( const DistTensor<T>& B, const Location& loc, const ObjShape& shape )
{
    DistTensor<T> A(B.Grid());
    LockedView( A, B, loc, shape );
    return A;
}

template<typename T>
inline void View2x1( Tensor<T>& A, Tensor<T>& BT, Tensor<T>& BB, Index index )
{
    Mode indexModeA, indexModeBT, indexModeBB;
#ifndef RELEASE
    indexModeA = A.ModeOfIndex(index);
    indexModeBT = BT.ModeOfIndex(index);
    indexModeBB = BB.ModeOfIndex(index);
    CallStackEntry entry("View2x1");
    if( BT.Dimension(indexModeBT) != BB.Dimension(indexModeBB) )
        LogicError("2x1 must have consistent width to combine");
    if( BT.LDim(indexModeBT) != BB.LDim(indexModeBB) )
        LogicError("2x1 must have consistent ldim to combine");
    if( BB.Buffer() != (BT.Buffer() + BT.Dimension(indexModeBT) * BT.LDim(indexModeBT)) )
        LogicError("2x1 must have contiguous memory");
#endif
    A.memory_.Empty();
    A.shape_    = BT.shape_;
    A.shape_[indexModeA] += BB.shape_[indexModeBB];
    A.ldims_    = BT.ldims_;
    A.data_     = BT.data_;
    A.viewType_ = VIEW;
}

template<typename T>
inline Tensor<T> View2x1( Tensor<T>& BT, Tensor<T>& BB, Index index )
{
    Tensor<T> A;
    View2x1( A, BT, BB, index );
    return A;
}

template<typename T>
inline void View2x1
( DistTensor<T>& A, DistTensor<T>& BT, DistTensor<T>& BB, Index index )
{
#ifndef RELEASE
    CallStackEntry entry("View2x1");
    AssertConforming2x1( BT, BB, index );
    BT.AssertSameGrid( BB.Grid() );
#endif
    const Mode indexModeA = A.ModeOfIndex(index);
    const Mode indexModeBB = BB.ModeOfIndex(index);
    A.Empty();
    A.grid_ = BT.grid_;
    A.shape_ = BT.shape_;
    A.shape_[indexModeA] += BB.shape_[indexModeBB];
    A.modeAlignments_ = BT.modeAlignments_;
    A.viewType_ = LOCKED_VIEW;
    if( A.Participating() )
    {
        A.modeShifts_ = BT.modeShifts_;
        View2x1( A.Tensor(), BT.Tensor(), BB.Tensor(), index );
    }
    else
    {
        std::fill(A.modeShifts_.begin(), A.modeShifts_.end(), 0);
    }
}

template<typename T>
inline DistTensor<T> View2x1( DistTensor<T>& BT, DistTensor<T>& BB, Index index )
{
    DistTensor<T> A(BT.Grid());
    View2x1( A, BT, BB, index );
    return A;
}

template<typename T>
inline void LockedView2x1
( Tensor<T>& A, const Tensor<T>& BT, const Tensor<T>& BB, Index index )
{
    Mode indexModeA, indexModeBT, indexModeBB;
#ifndef RELEASE
    indexModeBT = BT.ModeOfIndex(index);
    indexModeBB = BB.ModeOfIndex(index);
    CallStackEntry entry("LockedView2x1");
    if( BT.Dimension(indexModeBT) != BB.Dimension(indexModeBB) )
        LogicError("2x1 must have consistent width to combine");
    if( BT.LDim(indexModeBT) != BB.LDim(indexModeBB) )
        LogicError("2x1 must have consistent ldim to combine");
    if( BB.LockedBuffer() != (BT.LockedBuffer() + BT.Dimension(indexModeBT)*BT.LDim(indexModeBT)) )
        LogicError("2x1 must have contiguous memory");
#endif
    indexModeA = A.ModeOfIndex(index);
    indexModeBT = BT.ModeOfIndex(index);
    indexModeBB = BB.ModeOfIndex(index);
    A.memory_.Empty();
    A.shape_    = BT.shape_;
    A.shape_[indexModeA] += BB.shape_[indexModeBB];
    A.ldims_     = BT.ldims_;
    A.data_     = BT.data_;
    A.viewType_ = LOCKED_VIEW;
}

template<typename T>
inline Tensor<T> LockedView2x1
( const Tensor<T>& BT, const Tensor<T>& BB, Index index )
{
    Tensor<T> A;
    LockedView2x1( A, BT, BB, index );
    return A;
}

template<typename T>
inline void LockedView2x1
(       DistTensor<T>& A,
  const DistTensor<T>& BT,
  const DistTensor<T>& BB, Index index )
{
#ifndef RELEASE
    CallStackEntry entry("LockedView2x1");
    AssertConforming2x1( BT, BB, index );
    BT.AssertSameGrid( BB.Grid() );
#endif
    const Mode indexModeA = A.ModeOfIndex(index);
    const Mode indexModeBB = BB.ModeOfIndex(index);
    A.Empty();
    A.grid_ = BT.grid_;
    A.shape_ = BT.shape_;
    A.shape_[indexModeA] += BB.shape_[indexModeBB];
    A.modeAlignments_ = BT.modeAlignments_;
    A.viewType_ = LOCKED_VIEW;
    if( A.Participating() )
    {
        A.modeShifts_ = BT.modeShifts_;
        LockedView2x1( A.Tensor(), BT.LockedTensor(), BB.LockedTensor(), index );
    }
    else
    {
        std::fill(A.modeShifts_.begin(), A.modeShifts_.end(), 0);
    }
}

template<typename T>
inline DistTensor<T> LockedView2x1
( const DistTensor<T>& BT, const DistTensor<T>& BB, Index index )
{
    DistTensor<T> A(BT.Grid());
    LockedView2x1( A, BT, BB, index );
    return A;
}

} // namespace tmen

#endif // ifndef TMEN_CORE_VIEW_IMPL_HPP
