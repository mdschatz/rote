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
    A.indices_ = B.indices_;
    A.index2modeMap_ = B.index2modeMap_;
    A.mode2indexMap_ = B.mode2indexMap_;
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
    A.indices_ = B.indices_;
    A.index2modeMap_ = B.index2modeMap_;
    A.mode2indexMap_ = B.mode2indexMap_;
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
( Tensor<T>& A, const Tensor<T>& BT, const Tensor<T>& BB, Index index, bool isLocked )
{
    Mode indexModeA, indexModeBT, indexModeBB;
    indexModeA = A.ModeOfIndex(index);
    indexModeBT = BT.ModeOfIndex(index);
    indexModeBB = BB.ModeOfIndex(index);
#ifndef RELEASE
    CallStackEntry entry("View2x1Helper");

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
        const DistTensor<T>& BB, Index index, bool isLocked )
{
#ifndef RELEASE
    CallStackEntry entry("View2x1Helper");
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
    if(isLocked)
        A.viewType_ = LOCKED_VIEW;
    else
        A.viewType_ = VIEW;
    if( A.Participating() )
    {
        A.modeShifts_ = BT.modeShifts_;
//        View2x1Helper(A.Tensor(), BT.LockedTensor(), BB.LockedTensor(), index, isLocked);
//        if(isLocked)
//            LockedView2x1( A.Tensor(), BT.LockedTensor(), BB.LockedTensor(), index );
//        else
//            View2x1( A.Tensor(), BT.Tensor(), BB.Tensor(), index );
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
  const IndexArray& newIndices,
  const std::vector<IndexArray>& oldIndices, bool isLocked )
{
#ifndef RELEASE
    CallStackEntry entry("ViewAsLowerOrderHelper");
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

//    A.data_     = B.data_;
    if(isLocked)
        A.viewType_ = LOCKED_VIEW;
    else
        A.viewType_ = VIEW;
}

template<typename T>
inline Tensor<T> ViewAsHigherOrderHelper
( Tensor<T>& A,
  const Tensor<T>& B,
  const std::vector<IndexArray>& newIndices,
  const IndexArray& oldIndices,
  const std::vector<ObjShape>& newIndicesShape,
  bool isLocked)
{
#ifndef RELEASE
    CallStackEntry entry("ViewAsHigherOrderHelper");
    B.AssertSplittableIndices(newIndices, oldIndices, newIndicesShape);
#endif
    Unsigned i, j;
    const Unsigned oldOrder = B.Order();
    Unsigned newOrder = oldOrder - oldIndices.size();
    for(i = 0; i < newIndices.size(); i++){
        newOrder += newIndices[i].size();
    }
    A.memory_.Empty();

    //Update the shape, ldims_, strides_, maps_
    A.indices_.resize(newOrder);
    A.shape_.resize(newOrder);
    A.ldims_.resize(newOrder);
    A.strides_.resize(newOrder);
    A.index2modeMap_.clear();
    A.mode2indexMap_.clear();

    Unsigned splitIndexCounter = 0;
    Unsigned newMode = 0;
    for(i = 0; i < oldOrder; i++){
        Index oldIndex = B.IndexOfMode(i);
        Index indexToSplit = oldIndices[splitIndexCounter];
        //We are splitting this index
        if(oldIndex == indexToSplit){
            IndexArray splitIndices = newIndices[splitIndexCounter];
            ObjShape splitIndicesShape = newIndicesShape[splitIndexCounter];
            Unsigned newLDim = B.LDim(B.ModeOfIndex(indexToSplit));
            for(j = 0; j < splitIndices.size(); j++){
                Index newIndex = splitIndices[j];
                Unsigned newIndexDimension = splitIndicesShape[j];
                A.indices_[newMode] = newIndex;
                A.shape_[newMode] = newIndexDimension;
                A.ldims_[newMode] = newLDim;
                A.strides_[newMode] = newLDim;
                A.index2modeMap[newIndex] = newMode;
                A.mode2indexMap[newMode] = newIndex;

                //Update counters
                newLDim *= newIndexDimension;
            }
        }
        //Not splitting, so copy over info
        else{

            A.indices[newMode] = oldIndex;
            A.shape_[newMode] = B.Dimension(i);
            A.ldims_[newMode] = B.LDim(i);
            A.strides_[newMode] = B.LDim(i);
            A.index2modeMap[B.IndexOfMode(i)] = newMode;
            A.mode2indexMap[newMode] = B.IndexOfMode(i);
        }
        newMode++;
    }

//    A.data_     = B.data_;
    if(isLocked)
        A.viewType_ = LOCKED_VIEW;
    else
        A.viewType_ = VIEW;
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
    A.data_ = B.data_;
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
    ViewHelper(A, B, false);
    if(A.Participating() )
    {
        View( A.Tensor(), B.Tensor() );
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
    ViewHelper(A, B, true);
    A.data_ = B.data_;
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
    ViewHelper(A, B, true);
    if(A.Participating()){
        LockedView(A.Tensor(), B.LockedTensor());
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
#endif
    ViewHelper(A, B, loc, shape, false);
    A.data_ = B.Buffer(loc);
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
#endif
    ViewHelper(A, B, loc, shape, false);
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
#endif
    ViewHelper(A, B, loc, shape, true);
    A.data_ = B.LockedBuffer(loc);
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
#endif
    ViewHelper(A, B, loc, shape, true);
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
    DistTensor<T> A(B.Grid());
    LockedView( A, B, loc, shape );
    return A;
}

template<typename T>
inline void View2x1( Tensor<T>& A, Tensor<T>& BT, Tensor<T>& BB, Index index )
{
#ifndef RELEASE
    CallStackEntry entry("View2x1");
#endif
    View2x1Helper(A, BT, BB, index, false);
    A.data_ = BT.data_;
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
#endif
    View2x1Helper(A, BT, BB, index, false);
    if(A.Participating()){
        View2x1( A.Tensor(), BT.Tensor(), BB.Tensor(), index );
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
#ifndef RELEASE
    CallStackEntry entry("LockedView2x1");
#endif
    View2x1Helper(A, BT, BB, index, true);
    A.data_ = BT.data_;
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
#endif
    View2x1Helper(A, BT, BB, index, true);
    if(A.Participating()){
        LockedView2x1( A.Tensor(), BT.LockedTensor(), BB.LockedTensor(), index );
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
#endif
    ViewAsLowerOrderHelper(A, B, newIndices, oldIndices, false);
    A.data_ = B.data_;
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
inline void LockedViewAsLowerOrder
( Tensor<T>& A,
  Tensor<T>& B,
  const IndexArray& newIndices,
  const std::vector<IndexArray>& oldIndices )
{
#ifndef RELEASE
    CallStackEntry entry("LockedViewAsLowerOrder");
#endif
    ViewAsLowerOrderHelper(A, B, newIndices, oldIndices, true);
    A.data_ = B.data_;
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
inline Tensor<T> ViewAsHigherOrder
( Tensor<T>& A,
  Tensor<T>& B,
  const std::vector<IndexArray>& newIndices,
  const IndexArray& oldIndices,
  const std::vector<ObjShape>& newIndicesShape)
{
#ifndef RELEASE
    CallStackEntry entry("ViewAsHigherOrder");
#endif
    ViewAsHigherOrderHelper(A, B, newIndices, oldIndices, newIndicesShape, false);
    A.data_ = B.data_;
}

template<typename T>
inline Tensor<T> ViewAsHigherOrder
( Tensor<T>& B,
  const IndexArray& newIndices,
  const std::vector<IndexArray>& oldIndices )
{
   Tensor<T> A;
   ViewAsHigherOrder(A, B, newIndices, oldIndices);
   return A;
}

template<typename T>
inline Tensor<T> LockedViewAsHigherOrder
( Tensor<T>& A,
  Tensor<T>& B,
  const std::vector<IndexArray>& newIndices,
  const IndexArray& oldIndices,
  const std::vector<ObjShape>& newIndicesShape)
{
#ifndef RELEASE
    CallStackEntry entry("ViewAsHigherOrder");
#endif
    ViewAsHigherOrderHelper(A, B, newIndices, oldIndices, newIndicesShape, true);
    A.data_ = B.data_;
}

template<typename T>
inline Tensor<T> LockedViewAsHigherOrder
( Tensor<T>& B,
  const IndexArray& newIndices,
  const std::vector<IndexArray>& oldIndices )
{
   Tensor<T> A;
   LockedViewAsHigherOrder(A, B, newIndices, oldIndices);
   return A;
}

} // namespace tmen

#endif // ifndef TMEN_CORE_VIEW_IMPL_HPP
