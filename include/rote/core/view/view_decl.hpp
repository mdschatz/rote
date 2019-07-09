/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef ROTE_CORE_VIEW_DECL_HPP
#define ROTE_CORE_VIEW_DECL_HPP

namespace rote {

//
// Viewing a full tensor
//

template<typename T>
void ViewHelper( Tensor<T>& A, const Tensor<T>& B, bool isLocked );
template<typename T>
void ViewHelper( DistTensor<T>& A, const DistTensor<T>& B, bool isLocked );

template<typename T>
void View( Tensor<T>& A, Tensor<T>& B );
template<typename T>
Tensor<T> View( Tensor<T>& B );

template<typename T>
void View( DistTensor<T>& A, DistTensor<T>& B );
template<typename T>
DistTensor<T> View( DistTensor<T>& B );

template<typename T>
void LockedView( Tensor<T>& A, const Tensor<T>& B );
template<typename T>
Tensor<T> LockedView( const Tensor<T>& B );

template<typename T>
void LockedView( DistTensor<T>& A, const DistTensor<T>& B );
template<typename T>
DistTensor<T> LockedView( const DistTensor<T>& B );

//
// Viewing a subtensor
//

template<typename T>
void ViewHelper( Tensor<T>& A, const Tensor<T>& B, const Location& loc, const ObjShape& shape, bool isLocked );

template<typename T>
void ViewHelper( DistTensor<T>& A, const DistTensor<T>& B, const Location& loc, const ObjShape& shape, bool isLocked );

template<typename T>
void View( Tensor<T>& A, Tensor<T>& B, const Location& loc, const ObjShape& shape );
template<typename T>
Tensor<T> View( Tensor<T>& B, const Location& loc, const ObjShape& shape );

template<typename T>
void View( DistTensor<T>& A, DistTensor<T>& B, const Location& loc, const ObjShape& shape );
template<typename T>
DistTensor<T> View( DistTensor<T>& B, const Location& loc, const ObjShape& shape );

template<typename T>
void LockedView( Tensor<T>& A, const Tensor<T>& B, const Location& loc, const ObjShape& shape );
template<typename T>
Tensor<T> LockedView( const Tensor<T>& B, const Location& loc, const ObjShape& shape );

template<typename T>
void LockedView( DistTensor<T>& A, const DistTensor<T>& B, const Location& loc, const ObjShape& shape );
template<typename T>
DistTensor<T> LockedView( const DistTensor<T>& B, const Location& loc, const ObjShape& shape );

//
// View two mode-ly connected tensors
//

template<typename T>
void View2x1Helper
(       Tensor<T>& A,
  const Tensor<T>& BT,
  const Tensor<T>& BB, Mode mode, bool isLocked );

template<typename T>
void View2x1Helper
(       DistTensor<T>& A,
  const DistTensor<T>& BT,
  const DistTensor<T>& BB, Mode mode, bool isLocked );

template<typename T>
void View2x1
( Tensor<T>& A,
  Tensor<T>& BT,
  Tensor<T>& BB, Mode mode );
template<typename T>
Tensor<T> View2x1( Tensor<T>& BT, Tensor<T>& BB, Mode mode );
template<typename T>
void View2x1
( DistTensor<T>& A,
  DistTensor<T>& BT,
  DistTensor<T>& BB, Mode mode );
template<typename T>
DistTensor<T> View2x1( DistTensor<T>& BT, DistTensor<T>& BB, Mode mode );

template<typename T>
void LockedView2x1
(       Tensor<T>& A,
  const Tensor<T>& BT,
  const Tensor<T>& BB, Mode mode );
template<typename T>
Tensor<T> LockedView2x1( const Tensor<T>& BT, const Tensor<T>& BB, Mode mode );
template<typename T>
void LockedView2x1
(       DistTensor<T>& A,
  const DistTensor<T>& BT,
  const DistTensor<T>& BB, Mode mode );
template<typename T>
DistTensor<T> LockedView2x1
( const DistTensor<T>& BT, const DistTensor<T>& BB, Mode mode );

//
// View object as lower order object
//

template<typename T>
void ViewAsLowerOrderHelper
( Tensor<T>& A,
  const Tensor<T>& B,
  const std::vector<ModeArray>& oldModes, bool isLocked );

template<typename T>
void ViewAsLowerOrder
( Tensor<T>& A,
  Tensor<T>& B,
  const std::vector<ModeArray>& oldModes );

template<typename T>
Tensor<T> ViewAsLowerOrder
( Tensor<T>& B,
  const std::vector<ModeArray>& oldModes );

template<typename T>
void LockedViewAsLowerOrder
( Tensor<T>& A,
  const Tensor<T>& B,
  const std::vector<ModeArray>& oldModes );

template<typename T>
Tensor<T> LockedViewAsLowerOrder
( const Tensor<T>& B,
  const std::vector<ModeArray>& oldModes );

//
// View as higher order object
//

template<typename T>
void ViewAsHigherOrderHelper
( Tensor<T>& A,
  const std::vector<ObjShape>& splitShape, bool isLocked );

template<typename T>
void ViewAsHigherOrder
( Tensor<T>& A,
  Tensor<T>& B,
  const std::vector<ObjShape>& splitShape );

template<typename T>
Tensor<T> ViewAsHigherOrder
( Tensor<T>& B,
  const std::vector<ObjShape>& splitShape );

template<typename T>
void LockedViewAsHigherOrder
( Tensor<T>& A,
  const Tensor<T>& B,
  const std::vector<ObjShape>& splitShape );

template<typename T>
Tensor<T> LockedViewAsHigherOrder
( const Tensor<T>& B,
  const std::vector<ObjShape>& splitShape );

//
// View as matrix
//

template<typename T>
void ViewAsMatrixHelper
( Tensor<T>& A,
  const Tensor<T>& B,
  const Unsigned& nModesMergeCol, bool isLocked );

template<typename T>
void ViewAsMatrix
( Tensor<T>& A,
  const Tensor<T>& B,
  const Unsigned& nModesMergeCol );

template<typename T>
Tensor<T> ViewAsMatrix
( const Tensor<T>& B,
  const Unsigned& nModesMergeCol );

template<typename T>
void LockedViewAsMatrix
( Tensor<T>& A,
  const Tensor<T>& B,
  const Unsigned& nModesMergeCol );

} // namespace rote

#endif // ifndef ROTE_CORE_VIEW_DECL_HPP
