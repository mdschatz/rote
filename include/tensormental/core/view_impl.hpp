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

//NOTE: FIX PROBLEM WITH HIGHER/LOWER ORDER VIEWS ERASING DATA (A.memory_.Empty())
//////////////////////////////////////
// Helper routines for updating views
//////////////////////////////////////
template<typename T>
inline void ViewHelper( Tensor<T>& A, const Tensor<T>& B, bool isLocked){
#ifndef RELEASE
    CallStackEntry entry("ViewHelper");
#endif
    A.memory_.Empty();
    A.shape_ = B.shape_;
    A.ldims_     = B.ldims_;
    A.strides_     = B.strides_;
    //A.data_     = B.data_;
    if(isLocked)
        A.viewType_ = LOCKED_VIEW;
    else
        A.viewType_ = VIEW;
}

template<typename T>
inline void ViewHelper( DistTensor<T>& A, const DistTensor<T>& B, bool isLocked){
#ifndef RELEASE
    CallStackEntry entry("ViewHelper");
#endif
    A.Empty();
    A.grid_ = B.grid_;
    A.gridView_ = B.gridView_;
    A.shape_ = B.shape_;
    A.modeAlignments_ = B.modeAlignments_;
    if(isLocked)
        A.viewType_ = LOCKED_VIEW;
    else
        A.viewType_ = VIEW;
    if( A.Participating() )
    {
        A.modeShifts_ = B.ModeShifts();
//        ViewHelper(A.Tensor(), B.LockedTensor(), isLocked);
//        if(isLocked)
//            LockedView( A.Tensor(), B.LockedTensor() );
//        else
//            View( A.Tensor(), B.Tensor() );
    }
    else
    {
        std::fill(A.modeShifts_.begin(), A.modeShifts_.end(), 0);
    }
}

template<typename T>
inline void ViewHelper
( Tensor<T>& A, const Tensor<T>& B,
  const Location& loc, const ObjShape& shape, bool isLocked )
{
#ifndef RELEASE
    CallStackEntry entry("ViewHelper");
    const ObjShape shapeB = B.Shape();
    Location maxLoc(shape.size());
    ElemwiseSum(loc, shape, maxLoc);

    if( AnyElemwiseGreaterThan(maxLoc, shapeB) )
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
    A.ldims_ = B.ldims_;
    A.strides_     = B.strides_;
    if(isLocked){
        A.viewType_ = LOCKED_VIEW;
    }else{
        A.viewType_ = VIEW;
    }
}

template<typename T>
inline void ViewHelper
( DistTensor<T>& A, const DistTensor<T>& B,
  const Location& loc, const ObjShape& shape, bool isLocked )
{
#ifndef RELEASE
    CallStackEntry entry("ViewHelper");
    B.AssertValidSubtensor( loc, shape );
#endif
    A.Empty();

    Unsigned i;
    const Unsigned order = B.Order();
    const tmen::Grid& g = B.Grid();
    const std::vector<Unsigned> modeShifts = B.ModeShifts();
    const std::vector<Unsigned> modeWrapStrides = B.GridViewShape();

    A.grid_ = &g;
    A.gridView_ = B.gridView_;
    A.shape_ = shape;

    for(i = 0; i < order; i++)
        A.modeAlignments_[i] = (B.ModeAlignment(i) + loc[i]) % modeWrapStrides[i];

    if(isLocked)
        A.viewType_ = LOCKED_VIEW;
    else
        A.viewType_ = VIEW;

    if( A.Participating() )
    {
        const std::vector<Unsigned> modeRanks = B.GridViewLoc();
        A.modeShifts_ = Shifts(modeRanks, A.Alignments(), modeWrapStrides);

        const std::vector<Unsigned> localShapeBehind = Lengths(loc, B.ModeShifts(), modeWrapStrides);

        const std::vector<Unsigned> localShape = Lengths(shape, A.ModeShifts(), modeWrapStrides);

//        ViewHelper(A.Tensor(), B.LockedTensor(), localShapeBehind, localShape, isLocked);
//        if(isLocked){
//            LockedView( A.Tensor(), B.LockedTensor(), localShapeBehind, localShape );
//        }else{
//            View( A.Tensor(), B.Tensor(), localShapeBehind, localShape );
//        }
    }
    else
    {
        std::fill(A.modeShifts_.begin(), A.modeShifts_.end(), 0);
    }
}

template<typename T>
inline void View2x1Helper
( Tensor<T>& A, const Tensor<T>& BT, const Tensor<T>& BB, Mode mode, bool isLocked )
{
#ifndef RELEASE
    CallStackEntry entry("View2x1Helper");

    std::vector<Mode> negFilter(1);
    negFilter[0] = mode;

    if( AnyElemwiseNotEqual(NegFilterVector(BT.Shape(), negFilter), NegFilterVector(BB.Shape(), negFilter)) )
        LogicError("2x1 must have consistent width to combine");
    if( BT.LDim(mode) != BB.LDim(mode) )
        LogicError("2x1 must have consistent ldim to combine");
    if( BB.LockedBuffer() != (BT.LockedBuffer() + BT.Dimension(mode)*BT.LDim(mode)) )
        LogicError("2x1 must have contiguous memory");
#endif
    A.memory_.Empty();
    A.shape_    = BT.shape_;
    A.shape_[mode] += BB.shape_[mode];
    A.ldims_    = BT.ldims_;
    A.strides_     = BT.strides_;
//    A.data_     = BT.data_;
    if(isLocked)
        A.viewType_ = LOCKED_VIEW;
    else
        A.viewType_ = VIEW;
}

template<typename T>
inline void View2x1Helper
(       DistTensor<T>& A,
        const DistTensor<T>& BT,
        const DistTensor<T>& BB, Mode mode, bool isLocked )
{
#ifndef RELEASE
    CallStackEntry entry("View2x1Helper");
    AssertConforming2x1( BT, BB, mode );
    BT.AssertSameGrid( BB.Grid() );
#endif
    A.Empty();
    A.grid_ = BT.grid_;
    A.gridView_ = BT.gridView_;
    A.shape_ = BT.shape_;
    A.shape_[mode] += BB.shape_[mode];
    A.modeAlignments_ = BT.modeAlignments_;
    if(isLocked)
        A.viewType_ = LOCKED_VIEW;
    else
        A.viewType_ = VIEW;
    if( A.Participating() )
    {
        A.modeShifts_ = BT.modeShifts_;
//        View2x1Helper(A.Tensor(), BT.LockedTensor(), BB.LockedTensor(), mode, isLocked);
//        if(isLocked)
//            LockedView2x1( A.Tensor(), BT.LockedTensor(), BB.LockedTensor(), mode );
//        else
//            View2x1( A.Tensor(), BT.Tensor(), BB.Tensor(), mode );
    }
    else
    {
        std::fill(A.modeShifts_.begin(), A.modeShifts_.end(), 0);
    }
}

template<typename T>
inline void ViewAsLowerOrderHelper
( Tensor<T>& A,
  const Tensor<T>& B,
  const std::vector<ModeArray>& oldModes, bool isLocked )
{
#ifndef RELEASE
    CallStackEntry entry("ViewAsLowerOrderHelper");
    B.AssertMergeableModes(oldModes);
#endif
    Unsigned i;
    const Unsigned newOrder = oldModes.size();
    A.memory_.Empty();

    //Update the shape, ldims_, strides_, maps_
    A.shape_.resize(newOrder);
    A.ldims_.resize(newOrder);
    A.strides_.resize(newOrder);
    A.shape_[0] = prod(FilterVector(B.Shape(), oldModes[0]));
    A.strides_[0] = B.LDim(oldModes[0][0]);
    A.ldims_[0] = B.LDim(oldModes[0][0]);
    for(i = 1; i < newOrder; i++){
        ModeArray modesToMerge = oldModes[i];
        A.shape_[i] = prod(FilterVector(B.Shape(), modesToMerge));

        //NOTE: strides are set to Max(c, 1) to ensure we don't end up with 0 value strides
        A.ldims_[i] = Max(1, A.shape_[i-1] * B.LDim(oldModes[i-1][0]));
        A.strides_[i] = Max(1, A.shape_[i-1] * B.LDim(oldModes[i-1][0]));
    }

//    A.data_     = B.data_;
    if(isLocked)
        A.viewType_ = LOCKED_VIEW;
    else
        A.viewType_ = VIEW;
}

template<typename T>
inline void ViewAsHigherOrderHelper
( Tensor<T>& A,
  const Tensor<T>& B,
  const ModeArray& oldModes,
  const std::vector<ObjShape>& newShape,
  bool isLocked)
{
#ifndef RELEASE
    CallStackEntry entry("ViewAsHigherOrderHelper");
    B.AssertSplittableModes(oldModes, newShape);
#endif
    Unsigned i, j;
    Unsigned newOrder = 0;
    for(i = 0; i < newShape.size(); i++)
        newOrder += newShape[i].size();
    //A.memory_.Empty();

    //Update the shape, ldims_, strides_, maps_
    A.shape_.resize(newOrder);
    A.ldims_.resize(newOrder);
    A.strides_.resize(newOrder);
    Unsigned modeCount = 0;
    for(i = 0; i < newShape.size(); i++){
        ObjShape newModeGroupShape = newShape[i];
        for(j = 0; j < newModeGroupShape.size(); j++){
            A.shape_[modeCount] = newModeGroupShape[j];
            if(modeCount == 0){
                A.ldims_[modeCount] = 1;
                A.strides_[modeCount] = 1;
            }else{
                A.ldims_[modeCount] = A.shape_[modeCount - 1] * A.ldims_[modeCount - 1];
                A.strides_[modeCount] = A.shape_[modeCount - 1] * A.strides_[modeCount - 1];
            }
            modeCount++;
        }
    }

//    A.data_     = B.data_;
    if(isLocked)
        A.viewType_ = LOCKED_VIEW;
    else
        A.viewType_ = VIEW;
}

template<typename T>
inline void ViewAsMatrixHelper
( Tensor<T>& A,
  const Tensor<T>& B,
  const std::vector<ModeArray>& oldModes, bool isLocked )
{
#ifndef RELEASE
    CallStackEntry entry("ViewAsMartixHelper");
    B.AssertMergeableModes(oldModes);
#endif
    if(oldModes[0].size() > 0 && oldModes[1].size() > 0)
        ViewAsLowerOrderHelper(A, B, oldModes, isLocked);
    else{
        Unsigned i;
        const Unsigned newOrder = 2;
        A.memory_.Empty();

        //Update the shape, ldims_, strides_, maps_
        A.shape_.resize(newOrder);
        A.ldims_.resize(newOrder);
        A.strides_.resize(newOrder);
        A.shape_[0] = Max(1,prod(FilterVector(B.Shape(), oldModes[0])));
        A.strides_[0] = oldModes[0].size() == 0 ? 1 : B.LDim(oldModes[0][0]);
        A.ldims_[0] = oldModes[0].size() == 0 ? 1 : B.LDim(oldModes[0][0]);

        A.shape_[1] = Max(1,prod(FilterVector(B.Shape(), oldModes[1])));
        A.strides_[1] = A.shape_[0] * A.strides_[0];
        A.ldims_[1] = A.shape_[0] * A.ldims_[0];

    //    A.data_     = B.data_;
        if(isLocked)
            A.viewType_ = LOCKED_VIEW;
        else
            A.viewType_ = VIEW;
    }
}

////////////////////////////
// Interface to user
////////////////////////////

template<typename T>
inline void View( Tensor<T>& A, Tensor<T>& B )
{
#ifndef RELEASE
    CallStackEntry entry("View");
#endif
    ViewHelper(A, B, false);
    //Set the data we can't set in helper
    A.data_ = B.data_;
}

template<typename T>
inline Tensor<T> View( Tensor<T>& B )
{
    Tensor<T> A(B.Order());
    View( A, B );
    return A;
}

template<typename T>
inline void View( DistTensor<T>& A, DistTensor<T>& B )
{
#ifndef RELEASE
    CallStackEntry entry("View");
#endif
    ViewHelper(A, B, false);
    //Set the data we can't set in helper
    if(A.Participating() )
    {
        View( A.Tensor(), B.Tensor() );
    }
}

template<typename T>
inline DistTensor<T> View( DistTensor<T>& B )
{
    DistTensor<T> A(B.Order(), B.Grid());
    View( A, B );
    return A;
}

template<typename T>
inline void LockedView( Tensor<T>& A, const Tensor<T>& B )
{
#ifndef RELEASE
    CallStackEntry entry("LockedView");
#endif
    ViewHelper(A, B, true);
    //Set the data we can't set in helper
    A.data_ = B.data_;
}

template<typename T>
inline Tensor<T> LockedView( const Tensor<T>& B )
{
    Tensor<T> A(B.Order());
    LockedView( A, B );
    return A;
}

template<typename T>
inline void LockedView( DistTensor<T>& A, const DistTensor<T>& B )
{
#ifndef RELEASE
    CallStackEntry entry("LockedView");
#endif
    ViewHelper(A, B, true);
    //Set the data we can't set in helper
    if(A.Participating()){
        LockedView(A.Tensor(), B.LockedTensor());
    }

}

template<typename T>
inline DistTensor<T> LockedView( const DistTensor<T>& B )
{
    DistTensor<T> A(B.Order(), B.Grid());
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
#endif
    ViewHelper(A, B, loc, shape, false);
    //Set the data we can't set in helper
    A.data_ = B.Buffer(loc);
}

template<typename T>
inline Tensor<T> View( Tensor<T>& B, const Location& loc, const ObjShape& shape )
{
    Tensor<T> A(B.Order());
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
#endif
    ViewHelper(A, B, loc, shape, false);
    //Set the data we can't set in helper
    if(A.Participating()){
        const std::vector<Unsigned> modeWrapStrides = B.GridViewShape();
        const std::vector<Unsigned> localShapeBehind = Lengths(loc, B.ModeShifts(), modeWrapStrides);
        const std::vector<Unsigned> localShape = Lengths(shape, A.ModeShifts(), modeWrapStrides);

        View( A.Tensor(), B.Tensor(), localShapeBehind, localShape );
    }
}

template<typename T>
inline DistTensor<T> View
( DistTensor<T>& B, const Location& loc, const ObjShape& shape )
{
    DistTensor<T> A(B.Order(), B.Grid());
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
#endif
    ViewHelper(A, B, loc, shape, true);
    //Set the data we can't set in helper
    A.data_ = B.LockedBuffer(loc);
}

template<typename T>
inline Tensor<T> LockedView
( const Tensor<T>& B, const Location& loc, const ObjShape& shape )
{
    Tensor<T> A(B.Order());
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
#endif
    ViewHelper(A, B, loc, shape, true);
    //Set the data we can't set in helper
    if(A.Participating()){
        const std::vector<Unsigned> modeWrapStrides = B.GridViewShape();
        const std::vector<Unsigned> localShapeBehind = Lengths(loc, B.ModeShifts(), modeWrapStrides);
        const std::vector<Unsigned> localShape = Lengths(shape, A.ModeShifts(), modeWrapStrides);

        LockedView( A.Tensor(), B.LockedTensor(), localShapeBehind, localShape );
    }
}

template<typename T>
inline DistTensor<T> LockedView
( const DistTensor<T>& B, const Location& loc, const ObjShape& shape )
{
    DistTensor<T> A(B.Order(), B.Grid());
    LockedView( A, B, loc, shape );
    return A;
}

template<typename T>
inline void View2x1( Tensor<T>& A, Tensor<T>& BT, Tensor<T>& BB, Mode mode )
{
#ifndef RELEASE
    CallStackEntry entry("View2x1");
#endif
    View2x1Helper(A, BT, BB, mode, false);
    //Set the data we can't set in helper
    A.data_ = BT.data_;
}

template<typename T>
inline Tensor<T> View2x1( Tensor<T>& BT, Tensor<T>& BB, Mode mode )
{
    Tensor<T> A(BT.Order());
    View2x1( A, BT, BB, mode );
    return A;
}

template<typename T>
inline void View2x1
( DistTensor<T>& A, DistTensor<T>& BT, DistTensor<T>& BB, Mode mode )
{
#ifndef RELEASE
    CallStackEntry entry("View2x1");
#endif
    View2x1Helper(A, BT, BB, mode, false);
    //Set the data we can't set in helper
    if(A.Participating()){
        View2x1( A.Tensor(), BT.Tensor(), BB.Tensor(), mode );
    }
}

template<typename T>
inline DistTensor<T> View2x1( DistTensor<T>& BT, DistTensor<T>& BB, Mode mode )
{
    DistTensor<T> A(BT.Order(), BT.Grid());
    View2x1( A, BT, BB, mode );
    return A;
}

template<typename T>
inline void LockedView2x1
( Tensor<T>& A, const Tensor<T>& BT, const Tensor<T>& BB, Mode mode )
{
#ifndef RELEASE
    CallStackEntry entry("LockedView2x1");
#endif
    View2x1Helper(A, BT, BB, mode, true);
    //Set the data we can't set in helper
    A.data_ = BT.data_;
}

template<typename T>
inline Tensor<T> LockedView2x1
( const Tensor<T>& BT, const Tensor<T>& BB, Mode mode )
{
    Tensor<T> A(BT.Order());
    LockedView2x1( A, BT, BB, mode );
    return A;
}

template<typename T>
inline void LockedView2x1
(       DistTensor<T>& A,
  const DistTensor<T>& BT,
  const DistTensor<T>& BB, Mode mode )
{
#ifndef RELEASE
    CallStackEntry entry("LockedView2x1");
#endif
    View2x1Helper(A, BT, BB, mode, true);
    //Set the data we can't set in helper
    if(A.Participating()){
        LockedView2x1( A.Tensor(), BT.LockedTensor(), BB.LockedTensor(), mode );
    }
}

template<typename T>
inline DistTensor<T> LockedView2x1
( const DistTensor<T>& BT, const DistTensor<T>& BB, Mode mode )
{
    DistTensor<T> A(BT.Order(), BT.Grid());
    LockedView2x1( A, BT, BB, mode );
    return A;
}

template<typename T>
inline void ViewAsLowerOrder
( Tensor<T>& A,
  Tensor<T>& B,
  const std::vector<ModeArray>& oldModes )
{
#ifndef RELEASE
    CallStackEntry entry("ViewAsLowerOrder");
#endif
    ViewAsLowerOrderHelper(A, B, oldModes, false);
    //Set the data we can't set in helper
    A.data_ = B.data_;
}

template<typename T>
inline Tensor<T> ViewAsLowerOrder
( Tensor<T>& B,
  const std::vector<ModeArray>& oldModes )
{
   Tensor<T> A(B.Order());
   ViewAsLowerOrder(A, B, oldModes);
   return A;
}

template<typename T>
inline void LockedViewAsLowerOrder
( Tensor<T>& A,
  const Tensor<T>& B,
  const std::vector<ModeArray>& oldModes )
{
#ifndef RELEASE
    CallStackEntry entry("LockedViewAsLowerOrder");
#endif
    ViewAsLowerOrderHelper(A, B, oldModes, true);
    //Set the data we can't set in helper
    A.data_ = B.data_;
}

template<typename T>
inline Tensor<T> LockedViewAsLowerOrder
( const Tensor<T>& B,
  const std::vector<ModeArray>& oldModes )
{
   Tensor<T> A(B.Order());
   LockedViewAsLowerOrder(A, B, oldModes);
   return A;
}

template<typename T>
inline void ViewAsHigherOrder
( Tensor<T>& A,
  Tensor<T>& B,
  const ModeArray& oldModes,
  const std::vector<ObjShape>& newShape)
{
#ifndef RELEASE
    CallStackEntry entry("ViewAsHigherOrder");
#endif
    ViewAsHigherOrderHelper(A, B, oldModes, newShape, false);
    //Set the data we can't set in helper
    A.data_ = B.data_;
}

template<typename T>
inline Tensor<T> ViewAsHigherOrder
( Tensor<T>& B,
  const std::vector<ModeArray>& oldModes )
{
   Tensor<T> A(B.Order());
   ViewAsHigherOrder(A, B, oldModes);
   return A;
}

template<typename T>
inline void LockedViewAsHigherOrder
( Tensor<T>& A,
  const Tensor<T>& B,
  const ModeArray& oldModes,
  const std::vector<ObjShape>& newShape)
{
#ifndef RELEASE
    CallStackEntry entry("ViewAsHigherOrder");
#endif
    ViewAsHigherOrderHelper(A, B, oldModes, newShape, true);
    //Set the data we can't set in helper
    A.data_ = B.data_;
}

template<typename T>
inline Tensor<T> LockedViewAsHigherOrder
( const Tensor<T>& B,
  const std::vector<ModeArray>& oldModes )
{
   Tensor<T> A(B.Order());
   LockedViewAsHigherOrder(A, B, oldModes);
   return A;
}

template<typename T>
inline void ViewAsMatrix
( Tensor<T>& A,
  Tensor<T>& B,
  const std::vector<ModeArray>& oldModes )
{
#ifndef RELEASE
    CallStackEntry entry("ViewAsLowerOrder");
#endif
    ViewAsMatrixHelper(A, B, oldModes, false);
    //Set the data we can't set in helper
    A.data_ = B.data_;
}

template<typename T>
inline Tensor<T> ViewAsMatrix
( Tensor<T>& B,
  const std::vector<ModeArray>& oldModes )
{
   Tensor<T> A(2);
   ViewAsMatrix(A, B, oldModes);
   return A;
}

template<typename T>
inline void LockedViewAsMatrix
( Tensor<T>& A,
  const Tensor<T>& B,
  const std::vector<ModeArray>& oldModes )
{
#ifndef RELEASE
    CallStackEntry entry("LockedViewAsLowerOrder");
#endif
    ViewAsMatrixHelper(A, B, oldModes, true);
    //Set the data we can't set in helper
    A.data_ = B.data_;
}

template<typename T>
inline Tensor<T> LockedViewAsMatrix
( const Tensor<T>& B,
  const std::vector<ModeArray>& oldModes )
{
   Tensor<T> A(2);
   LockedViewAsMatrix(A, B, oldModes);
   return A;
}

} // namespace tmen

#endif // ifndef TMEN_CORE_VIEW_IMPL_HPP
