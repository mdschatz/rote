/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef ROTE_CORE_VIEW_IMPL_HPP
#define ROTE_CORE_VIEW_IMPL_HPP

namespace rote {

//////////////////////////////////////
// Helper routines for updating views
//////////////////////////////////////
template<typename T>
inline void ViewHelper( Tensor<T>& A, const Tensor<T>& B, bool isLocked){
    A.memory_.Empty();
    A.shape_ = B.shape_;
    A.strides_     = B.strides_;
    //A.data_     = B.data_;
    if(isLocked)
        A.viewType_ = LOCKED_VIEW;
    else
        A.viewType_ = VIEW;
}

template<typename T>
inline void ViewHelper( DistTensor<T>& A, const DistTensor<T>& B, bool isLocked){
    A.Empty();
    A.grid_ = B.grid_;
    A.gridView_ = B.gridView_;
    A.shape_ = B.shape_;
    A.modeAlignments_ = B.modeAlignments_;
    A.dist_ = B.dist_;
    A.localPerm_ = B.localPerm_;
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
    const ObjShape shapeB = B.Shape();
    Location maxLoc = ElemwiseSum(loc, shape);

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

    A.memory_.Empty();
    A.shape_ = shape;
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
    B.AssertValidSubtensor( loc, shape );
#endif
    A.Empty();

    Unsigned i;
    const Unsigned order = B.Order();
    const rote::Grid& g = B.Grid();
    const std::vector<Unsigned> modeShifts = B.ModeShifts();
    const std::vector<Unsigned> modeWrapStrides = B.GridViewShape();

    A.grid_ = &g;
    A.gridView_ = B.gridView_;
    A.shape_ = shape;
    A.dist_ = B.dist_;
    A.localPerm_ = B.localPerm_;
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
    std::vector<Mode> negFilter(1);
    negFilter[0] = mode;

    if( AnyElemwiseNotEqual(NegFilterVector(BT.Shape(), negFilter), NegFilterVector(BB.Shape(), negFilter)) )
        LogicError("2x1 must have consistent width to combine");
    if( BT.Stride(mode) != BB.Stride(mode) )
        LogicError("2x1 must have consistent stride to combine");
    if( BB.LockedBuffer() != (BT.LockedBuffer() + BT.Dimension(mode)*BT.Stride(mode)) )
        LogicError("2x1 must have contiguous memory");
#endif
    A.memory_.Empty();
    A.shape_    = BT.shape_;
    A.shape_[mode] += BB.shape_[mode];
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
    AssertConforming2x1( BT, BB, mode );
    BT.AssertSameGrid( BB.Grid() );
#endif
    A.Empty();
    A.grid_ = BT.grid_;
    A.gridView_ = BT.gridView_;
    A.shape_ = BT.shape_;
    A.dist_ = BT.dist_;
    A.shape_[mode] += BB.shape_[mode];
    A.modeAlignments_ = BT.modeAlignments_;
    A.localPerm_ = BT.localPerm_;
    if(isLocked)
        A.viewType_ = LOCKED_VIEW;
    else
        A.viewType_ = VIEW;
    if( A.Participating() )
    {
        A.modeShifts_ = BT.modeShifts_;
        View2x1Helper(A.Tensor(), BT.LockedTensor(), BB.LockedTensor(), mode, isLocked);
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
    B.AssertMergeableModes(oldModes);
#endif
    Unsigned i;
    const Unsigned newOrder = oldModes.size();
    A.memory_.Empty();

    //Update the shape, strides_, maps_
    A.shape_.resize(newOrder);
    A.strides_.resize(newOrder);
    A.shape_[0] = prod(FilterVector(B.Shape(), oldModes[0]));
    A.strides_[0] = B.Stride(oldModes[0][0]);
    for(i = 1; i < newOrder; i++){
        ModeArray modesToMerge = oldModes[i];
        A.shape_[i] = prod(FilterVector(B.Shape(), modesToMerge));

        //NOTE: strides are set to Max(c, 1) to ensure we don't end up with 0 value strides
        A.strides_[i] = Max(1, A.shape_[i-1] * B.Stride(oldModes[i-1][0]));
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
  const std::vector<ObjShape>& splitShape,
  bool isLocked)
{
    Unsigned i, j;
    Unsigned newOrder = 0;
    for(i = 0; i < splitShape.size(); i++)
        newOrder += splitShape[i].size();
    A.memory_.Empty();

    //Update the shape, strides_, maps_
    A.shape_.resize(newOrder);
    A.strides_.resize(newOrder);
    Unsigned modeCount = 0;
    for(i = 0; i < splitShape.size(); i++){
        ObjShape newModeGroupShape = splitShape[i];
        for(j = 0; j < newModeGroupShape.size(); j++){
            A.shape_[modeCount] = newModeGroupShape[j];
            if(modeCount == 0){
                A.strides_[modeCount] = 1;
            }else{
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
  const Unsigned& nModesMergeCol, bool isLocked )
{
    Unsigned order = B.Order();
    Unsigned i;
    std::vector<ModeArray> mergeModes(2);
    mergeModes[0].resize(nModesMergeCol);
    mergeModes[1].resize(order - nModesMergeCol);
    for(i = 0; i < nModesMergeCol; i++)
        mergeModes[0][i] = i;
    for(i = nModesMergeCol; i < order; i++)
        mergeModes[1][i-nModesMergeCol] = i;

    if(nModesMergeCol > 0 && order - nModesMergeCol > 0){

        ViewAsLowerOrderHelper(A, B, mergeModes, isLocked);
    }else{
        A.memory_.Empty();

        ObjShape shapeB = B.Shape();
        ObjShape stridesB = B.Strides();

        //Update the shape, strides_, maps_
        A.shape_.resize(2);
        A.strides_.resize(2);

        if(order == 0){
            A.shape_[0] = 1;
            A.strides_[0] = 1;

            A.shape_[1] = 1;
            A.strides_[1] = 1;
        }else if(order == 1){
            if(mergeModes[0].size() == 0){
                if(shapeB[0] == 0){
                    A.shape_[0] = 0;
                }else{
                    A.shape_[0] = 1;
                }

                A.strides_[0] = 1;

                A.shape_[1] = shapeB[0];
                A.strides_[1] = Max(1, stridesB[0]);
            }else{
                A.shape_[0] = shapeB[0];
                A.strides_[0] = Max(1, stridesB[0]);

                if(shapeB[0] == 0){
                    A.shape_[1] = 0;
                }else{
                    A.shape_[1] = 1;
                }
                A.strides_[1] = Max(1, stridesB[0] * shapeB[0]);
            }
        }else{
            if(mergeModes[0].size() == 0){
                A.shape_[1] = prod(FilterVector(shapeB, mergeModes[1]));
                A.strides_[1] = Max(1, stridesB[0]);

                if(A.shape_[1] == 0){
                    A.shape_[0] = 0;
                }else{
                    A.shape_[0] = 1;
                }
                A.strides_[0] = 1;
            }else if(mergeModes[1].size() == 0){
                A.shape_[0] = prod(FilterVector(shapeB, mergeModes[0]));
                A.strides_[0] = Max(1, stridesB[0]);

                if(A.shape_[0] == 0){
                    A.shape_[1] = 0;
                }else{
                    A.shape_[1] = 1;
                }
                A.strides_[1] = Max(1, stridesB[order-1] * shapeB[order - 1]);
            }else{
                A.shape_[0] = prod(FilterVector(shapeB, mergeModes[0]));
                A.strides_[0] = Max(1, stridesB[0]);

                A.shape_[1] = prod(FilterVector(shapeB, mergeModes[1]));
                A.strides_[1] = Max(1, stridesB[nModesMergeCol-1] * shapeB[nModesMergeCol - 1]);
            }
        }
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
  const std::vector<ObjShape>& splitShape)
{
    ViewAsHigherOrderHelper(A, splitShape, false);
    //Set the data we can't set in helper
    A.data_ = B.data_;
}

template<typename T>
inline Tensor<T> ViewAsHigherOrder
( Tensor<T>& B,
  const std::vector<ObjShape>& splitShape )
{
   Tensor<T> A(B.Order());
   ViewAsHigherOrder(A, B, splitShape);
   return A;
}

template<typename T>
inline void LockedViewAsHigherOrder
( Tensor<T>& A,
  const Tensor<T>& B,
  const std::vector<ObjShape>& splitShape)
{
    ViewAsHigherOrderHelper(A, B, splitShape, true);
    //Set the data we can't set in helper
    A.data_ = B.data_;
}

template<typename T>
inline Tensor<T> LockedViewAsHigherOrder
( const Tensor<T>& B,
  const std::vector<ObjShape>& splitShape )
{
   Tensor<T> A(B.Order());
   LockedViewAsHigherOrder(A, B, splitShape);
   return A;
}

template<typename T>
inline void ViewAsMatrix
( Tensor<T>& A,
  const Tensor<T>& B,
  const Unsigned& nModesMergeCol )
{
    ViewAsMatrixHelper(A, B, nModesMergeCol, false);
    //Set the data we can't set in helper
    A.data_ = B.data_;
}

template<typename T>
inline Tensor<T> ViewAsMatrix
( const Tensor<T>& B,
  const Unsigned& nModesMergeCol )
{
   Tensor<T> A(2);
   ViewAsMatrix(A, B, nModesMergeCol);
   return A;
}

template<typename T>
inline void LockedViewAsMatrix
( Tensor<T>& A,
  const Tensor<T>& B,
  const Unsigned& nModesMergeCol )
{
    ViewAsMatrixHelper(A, B, nModesMergeCol, true);
    //Set the data we can't set in helper
    A.data_ = B.data_;
}

template<typename T>
inline Tensor<T> LockedViewAsMatrix
( const Tensor<T>& B,
  const Unsigned& nModesMergeCol )
{
   Tensor<T> A(2);
   LockedViewAsMatrix(A, B, nModesMergeCol);
   return A;
}

} // namespace rote

#endif // ifndef ROTE_CORE_VIEW_IMPL_HPP
