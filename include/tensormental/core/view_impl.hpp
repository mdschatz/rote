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
    A.indices_ = B.indices_;
    A.index2modeMap_ = B.index2modeMap_;
    A.mode2indexMap_ = B.mode2indexMap_;
    A.ldims_     = B.ldims_;
    A.strides_     = B.strides_;
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
    A.gridView_ = B.gridView_;
    A.shape_ = B.shape_;
    A.modeAlignments_ = B.modeAlignments_;
    A.viewType_ = VIEW;
    if( A.Participating() )
    {
        A.modeShifts_ = B.ModeShifts();
        View( A.Tensor(), B.Tensor() );
    }
    else
    {
        std::fill(A.modeShifts_.begin(), A.modeShifts_.end(), 0);
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
    A.indices_ = B.indices_;
    A.index2modeMap_ = B.index2modeMap_;
    A.mode2indexMap_ = B.mode2indexMap_;
    A.ldims_     = B.ldims_;
    A.strides_     = B.strides_;
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
    A.gridView_ = B.gridView_;
    A.shape_ = B.shape_;
    A.modeAlignments_ = B.modeAlignments_;
    A.viewType_ = LOCKED_VIEW;
    if( A.Participating() )
    {
        A.modeShifts_ = B.ModeShifts();
        LockedView( A.Tensor(), B.LockedTensor() );
    }
    else
    {
        std::fill(A.modeShifts_.begin(), A.modeShifts_.end(), 0);
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
    A.indices_ = B.indices_;
    A.index2modeMap_ = B.index2modeMap_;
    A.mode2indexMap_ = B.mode2indexMap_;
    A.ldims_ = B.ldims_;
    A.strides_     = B.strides_;
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
    A.gridView_ = B.gridView_;
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
    A.indices_ = B.indices_;
    A.index2modeMap_ = B.index2modeMap_;
    A.mode2indexMap_ = B.mode2indexMap_;
    A.ldims_     = B.ldims_;
    A.strides_     = B.strides_;
    A.data_     = B.LockedBuffer(loc);
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
    const Unsigned order = B.Order();
    const tmen::Grid& g = B.Grid();
    const std::vector<Unsigned> modeShifts = B.ModeShifts();
    const std::vector<Unsigned> modeWrapStrides = B.GridViewShape();

    A.grid_ = &g;
    A.gridView_ = B.gridView_;
    A.shape_ = shape;

    for(i = 0; i < order; i++)
        A.modeAlignments_[i] = (B.ModeAlignment(i) + loc[i]) % modeWrapStrides[i];

    A.viewType_ = LOCKED_VIEW;

    if( A.Participating() )
    {
        const std::vector<Unsigned> modeRanks = B.GridViewLoc();
        A.modeShifts_ = Shifts(modeRanks, A.Alignments(), modeWrapStrides);

        const std::vector<Unsigned> localShapeBehind = Lengths(loc, B.ModeShifts(), modeWrapStrides);

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
    indexModeA = A.ModeOfIndex(index);
    indexModeBT = BT.ModeOfIndex(index);
    indexModeBB = BB.ModeOfIndex(index);
#ifndef RELEASE
    CallStackEntry entry("View2x1");

    std::vector<Mode> negFilterBT(1);
    std::vector<Mode> negFilterBB(1);
    negFilterBT[0] = indexModeBT;
    negFilterBB[0] = indexModeBB;

    if( AnyElemwiseNotEqual(NegFilterVector(BT.Shape(), negFilterBT), NegFilterVector(BB.Shape(), negFilterBB)) )
        LogicError("2x1 must have consistent width to combine");
    if( BT.LDim(indexModeBT) != BB.LDim(indexModeBB) )
        LogicError("2x1 must have consistent ldim to combine");
    if( BB.Buffer() != (BT.Buffer() + BT.Dimension(indexModeBT) * BT.LDim(indexModeBT)) )
        LogicError("2x1 must have contiguous memory");
#endif
    A.memory_.Empty();
    A.shape_    = BT.shape_;
    A.shape_[indexModeA] += BB.shape_[indexModeBB];
    A.indices_  = BT.indices_;
    A.index2modeMap_ = BT.index2modeMap_;
    A.mode2indexMap_ = BT.mode2indexMap_;
    A.ldims_    = BT.ldims_;
    A.strides_     = BT.strides_;
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
    A.gridView_ = BT.gridView_;
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
    indexModeA = A.ModeOfIndex(index);
    indexModeBT = BT.ModeOfIndex(index);
    indexModeBB = BB.ModeOfIndex(index);
#ifndef RELEASE
    CallStackEntry entry("LockedView2x1");

    std::vector<Mode> negFilterBT(1);
    std::vector<Mode> negFilterBB(1);
    negFilterBT[0] = indexModeBT;
    negFilterBB[0] = indexModeBB;

    if( AnyElemwiseNotEqual(NegFilterVector(BT.Shape(), negFilterBT), NegFilterVector(BB.Shape(), negFilterBB)) )
        LogicError("2x1 must have consistent width to combine");
    if( BT.LDim(indexModeBT) != BB.LDim(indexModeBB) )
        LogicError("2x1 must have consistent ldim to combine");
    if( BB.LockedBuffer() != (BT.LockedBuffer() + BT.Dimension(indexModeBT)*BT.LDim(indexModeBT)) )
        LogicError("2x1 must have contiguous memory");
#endif
    A.memory_.Empty();
    A.shape_    = BT.shape_;
    A.shape_[indexModeA] += BB.shape_[indexModeBB];
    A.indices_  = BT.indices_;
    A.index2modeMap_ = BT.index2modeMap_;
    A.mode2indexMap_ = BT.mode2indexMap_;
    A.ldims_    = BT.ldims_;
    A.strides_     = BT.strides_;
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
    A.gridView_ = BT.gridView_;
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

template<typename T>
inline void ViewAsLowerOrder
( Tensor<T>& A,
  Tensor<T>& B,
  const IndexArray& newIndices,
  const std::vector<IndexArray>& oldIndices )
{
#ifndef RELEASE
    CallStackEntry entry("ViewAsLowerOrder");
    B.AssertMergeableIndices(newIndices, oldIndices);
#endif
    Unsigned i;
    const Unsigned newOrder = newIndices.size();
    A.memory_.Empty();
    A.indices_ = newIndices;

    //Update the shape, ldims_, strides_, maps_
    A.shape_.resize(newOrder);
    A.ldims_.resize(newOrder);
    A.strides_.resize(newOrder);
    A.index2modeMap_.clear();
    A.mode2indexMap_.clear();
    for(i = 0; i < newOrder; i++){
        Index newIndex = newIndices[i];
        IndexArray mergedIndices = oldIndices[i];
        A.shape[i] = prod(FilterVector(B.Shape(), mergedIndices));
        A.ldims_[i] = B.LDim(B.ModeOfIndex(mergedIndices[0]));
        A.strides_[i] = B.LDim(B.ModeOfIndex(mergedIndices[0]));
        A.index2modeMap_[newIndex] = i;
        A.mode2indexMap_[i] = newIndex;
    }

    A.data_     = B.data_;
    A.viewType_ = VIEW;
}

template<typename T>
inline Tensor<T> ViewAsLowerOrder
( Tensor<T>& B,
  const IndexArray& newIndices,
  const std::vector<IndexArray>& oldIndices )
{
   Tensor<T> A;
   ViewAsLowerOrder(A, B, newIndices, oldIndices);
   return A;
}

template<typename T>
inline void ViewAsLowerOrder
( DistTensor<T>& A,
  DistTensor<T>& B,
  const IndexArray& newIndices,
  const std::vector<IndexArray>& oldIndices )
{
#ifndef RELEASE
    CallStackEntry entry("ViewAsLowerOrder");
    B.AssertMergeableIndices(newIndices, oldIndices);
#endif
    Unsigned i, j;
    const Unsigned newOrder = newIndices.size();
    const tmen::Grid& g = B.Grid();

    A.Empty();
    A.grid_ = &g;

    //Adjust shape_ alignments_, shifts_, distributions_
    //Adjust gridView_
    A.shape_.resize(newOrder);
    A.modeAlignments_.resize(newOrder);
    A.constrainedModeAlignments_.resize(newOrder);
    A.modeShifts_.resize(newOrder);
    A.dist_.resize(newOrder);
    A.gridView_.grid_ = &g;
    A.gridView_.shape_.resize(newOrder);
    A.gridView_.loc_.resize(newOrder);

    for(i = 0; i < newOrder; i++){
        IndexArray mergedIndices = oldIndices[i];
        A.shape_[i] = prod(FilterVector(B.Shape(), mergedIndices));
        A.modeAlignments_[i] = Loc2LinearLoc(FilterVector(B.modeAlignments_, mergedIndices), A.shape_[i]);
        A.constrainedModeAlignments_[i] = Loc2LinearLoc(FilterVector(B.constrainedModeAlignments_, mergedIndices), A.shape_[i]);
        A.modeShifts_[i] = Loc2LinearLoc(FilterVector(B.modeShifts_, mergedIndices), A.shape_[i]);
        //NOTE: Figure out how to change the distribution more intelligently
        A.dist_[i].resize(0);
        for(j = 0; j < mergedIndices.size(); j++){
            ModeDistribution distToAppend = B.ModeDist(B.ModeOfIndex(mergedIndices[j]));
            A.dist_[i].insert(A.dist_[i].end(), distToAppend.begin(), distToAppend.end());
        }
        A.gridView_.shape_[i] = prod(FilterVector(g.Shape(), A.dist_[i]));
        A.gridView_.loc_[i] = Loc2LinearLoc(FilterVector(B.gridView_.loc_, mergedIndices), FilterVector(B.gridView_.shape_, mergedIndices));
    }

    //Adjust gridView_ distribution
    A.gridView_.dist_ = A.dist_;

    ViewAsLowerOrder(A.tensor_, B.tensor_, newIndices, oldIndices);

    A.viewType_ = VIEW;
}

template<typename T>
inline DistTensor<T> ViewAsLowerOrder
( DistTensor<T>& B,
  const IndexArray& newIndices,
  const std::vector<IndexArray>& oldIndices )
{
   DistTensor<T> A;
   ViewAsLowerOrder(A, B, newIndices, oldIndices);
   return A;
}

template<typename T>
inline void LockedViewAsLowerOrder
( Tensor<T>& A,
  Tensor<T>& B,
  const IndexArray& newIndices,
  const std::vector<IndexArray>& oldIndices )
{
#ifndef RELEASE
    CallStackEntry entry("LockedViewAsLowerOrder");
    B.AssertMergeableIndices(newIndices, oldIndices);
#endif
    Unsigned i;
    const Unsigned newOrder = newIndices.size();
    A.memory_.Empty();
    A.indices_ = newIndices;

    //Update the shape, ldims_, strides_, maps_
    A.shape_.resize(newOrder);
    A.ldims_.resize(newOrder);
    A.strides_.resize(newOrder);
    A.index2modeMap_.clear();
    A.mode2indexMap_.clear();
    for(i = 0; i < newOrder; i++){
        Index newIndex = newIndices[i];
        IndexArray mergedIndices = oldIndices[i];
        A.shape[i] = prod(FilterVector(B.Shape(), mergedIndices));
        A.ldims_[i] = B.LDim(B.ModeOfIndex(mergedIndices[0]));
        A.strides_[i] = B.LDim(B.ModeOfIndex(mergedIndices[0]));
        A.index2modeMap_[newIndex] = i;
        A.mode2indexMap_[i] = newIndex;
    }

    A.data_     = B.data_;
    A.viewType_ = LOCKED_VIEW;
}

template<typename T>
inline Tensor<T> LockedViewAsLowerOrder
( Tensor<T>& B,
  const IndexArray& newIndices,
  const std::vector<IndexArray>& oldIndices )
{
   Tensor<T> A;
   LockedViewAsLowerOrder(A, B, newIndices, oldIndices);
   return A;
}

template<typename T>
inline void LockedViewAsLowerOrder
( DistTensor<T>& A,
  DistTensor<T>& B,
  const IndexArray& newIndices,
  const std::vector<IndexArray>& oldIndices )
{
#ifndef RELEASE
    CallStackEntry entry("ViewAs");
    B.AssertMergeableIndices(newIndices, oldIndices);
#endif
    Unsigned i, j;
    const Unsigned newOrder = newIndices.size();
    const tmen::Grid& g = B.Grid();

    A.Empty();
    A.grid_ = &g;

    //Adjust shape_ alignments_, shifts_, distributions_
    //Adjust gridView_
    A.shape_.resize(newOrder);
    A.modeAlignments_.resize(newOrder);
    A.constrainedModeAlignments_.resize(newOrder);
    A.modeShifts_.resize(newOrder);
    A.dist_.resize(newOrder);
    A.gridView_.grid_ = &g;
    A.gridView_.shape_.resize(newOrder);
    A.gridView_.loc_.resize(newOrder);

    for(i = 0; i < newOrder; i++){
        IndexArray mergedIndices = oldIndices[i];
        A.shape_[i] = prod(FilterVector(B.Shape(), mergedIndices));
        A.modeAlignments_[i] = Loc2LinearLoc(FilterVector(B.modeAlignments_, mergedIndices), A.shape_[i]);
        A.constrainedModeAlignments_[i] = Loc2LinearLoc(FilterVector(B.constrainedModeAlignments_, mergedIndices), A.shape_[i]);
        A.modeShifts_[i] = Loc2LinearLoc(FilterVector(B.modeShifts_, mergedIndices), A.shape_[i]);
        //NOTE: Figure out how to change the distribution more intelligently
        A.dist_[i].resize(0);
        for(j = 0; j < mergedIndices.size(); j++){
            ModeDistribution distToAppend = B.ModeDist(B.ModeOfIndex(mergedIndices[j]));
            A.dist_[i].insert(A.dist_[i].end(), distToAppend.begin(), distToAppend.end());
        }
        A.gridView_.shape_[i] = prod(FilterVector(g.Shape(), A.dist_[i]));
        A.gridView_.loc_[i] = Loc2LinearLoc(FilterVector(B.gridView_.loc_, mergedIndices), FilterVector(B.gridView_.shape_, mergedIndices));
    }

    //Adjust gridView_ distribution
    A.gridView_.dist_ = A.dist_;

    LockedViewAsLowerOrder(A.tensor_, B.tensor_, newIndices, oldIndices);

    A.viewType_ = VIEW;
}

template<typename T>
inline DistTensor<T> LockedViewAsLowerOrder
( DistTensor<T>& B,
  const IndexArray& newIndices,
  const std::vector<IndexArray>& oldIndices )
{
   DistTensor<T> A;
   LockedViewAsLowerOrder(A, B, newIndices, oldIndices);
   return A;
}

} // namespace tmen

#endif // ifndef TMEN_CORE_VIEW_IMPL_HPP
