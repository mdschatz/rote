/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef TMEN_CORE_VIEW_DECL_HPP
#define TMEN_CORE_VIEW_DECL_HPP

namespace tmen {

//
// Viewing a full matrix
//

template<typename T>
void View( Tensor<T>& A, Tensor<T>& B );
template<typename T>
Tensor<T> View( Tensor<T>& B );
//template<typename T,Distribution U,Distribution V>
//void View( DistTensor<T,U,V>& A, DistTensor<T,U,V>& B );
//template<typename T,Distribution U,Distribution V>
//DistTensor<T,U,V> View( DistTensor<T,U,V>& B );

template<typename T>
void LockedView( Tensor<T>& A, const Tensor<T>& B );
template<typename T>
Tensor<T> LockedView( const Tensor<T>& B );
//template<typename T,Distribution U,Distribution V>
//void LockedView( DistTensor<T,U,V>& A, const DistTensor<T,U,V>& B );
//template<typename T,Distribution U,Distribution V>
//DistTensor<T,U,V> LockedView( const DistTensor<T,U,V>& B );

//
// Viewing a submatrix
//

template<typename T>
void View( Tensor<T>& A, Tensor<T>& B, Int i, Int j, Int height, Int width );
template<typename T>
Tensor<T> View( Tensor<T>& B, Int i, Int j, Int height, Int width );
//template<typename T,Distribution U,Distribution V>
//void View
//( DistTensor<T,U,V>& A, DistTensor<T,U,V>& B,
//  Int i, Int j, Int height, Int width );
//template<typename T,Distribution U,Distribution V>
//DistTensor<T,U,V> View
//( DistTensor<T,U,V>& B, Int i, Int j, Int height, Int width );

template<typename T>
void LockedView
( Tensor<T>& A, const Tensor<T>& B,
  Int i, Int j, Int height, Int width );
template<typename T>
Tensor<T> LockedView( const Tensor<T>& B, Int i, Int j, Int height, Int width );
//template<typename T,Distribution U,Distribution V>
//void LockedView
//( DistTensor<T,U,V>& A, const DistTensor<T,U,V>& B,
//  Int i, Int j, Int height, Int width );
//template<typename T,Distribution U,Distribution V>
//DistTensor<T,U,V> LockedView
//( const DistTensor<T,U,V>& B, Int i, Int j, Int height, Int width );

//
// View two horizontally connected matrices
//

template<typename T>
void View1x2
( Tensor<T>& A,
  Tensor<T>& BL, Tensor<T>& BR );
template<typename T>
Tensor<T> View1x2( Tensor<T>& BL, Tensor<T>& BR );
//template<typename T,Distribution U,Distribution V>
//void View1x2
//( DistTensor<T,U,V>& A,
//  DistTensor<T,U,V>& BL, DistTensor<T,U,V>& BR );
//template<typename T,Distribution U,Distribution V>
//DistTensor<T,U,V> View1x2( DistTensor<T,U,V>& BL, DistTensor<T,U,V>& BR );

template<typename T>
void LockedView1x2
(       Tensor<T>& A,
  const Tensor<T>& BL,
  const Tensor<T>& BR );
template<typename T>
Tensor<T> LockedView1x2( const Tensor<T>& BL, const Tensor<T>& BR );
//template<typename T,Distribution U,Distribution V>
//void LockedView1x2
//(       DistTensor<T,U,V>& A,
//  const DistTensor<T,U,V>& BL,
//  const DistTensor<T,U,V>& BR );
//template<typename T,Distribution U,Distribution V>
//DistTensor<T,U,V> LockedView1x2
//( const DistTensor<T,U,V>& BL, const DistTensor<T,U,V>& BR );

//
// View two vertically connected matrices
//

template<typename T>
void View2x1
( Tensor<T>& A,
  Tensor<T>& BT,
  Tensor<T>& BB );
template<typename T>
Tensor<T> View2x1( Tensor<T>& BT, Tensor<T>& BB );
//template<typename T,Distribution U,Distribution V>
//void View2x1
//( DistTensor<T,U,V>& A,
//  DistTensor<T,U,V>& BT,
//  DistTensor<T,U,V>& BB );
//template<typename T,Distribution U,Distribution V>
//DistTensor<T,U,V> View2x1( DistTensor<T,U,V>& BT, DistTensor<T,U,V>& BB );

template<typename T>
void LockedView2x1
(       Tensor<T>& A,
  const Tensor<T>& BT,
  const Tensor<T>& BB );
template<typename T>
Tensor<T> LockedView2x1( const Tensor<T>& BT, const Tensor<T>& BB );
//template<typename T,Distribution U,Distribution V>
//void LockedView2x1
//(       DistTensor<T,U,V>& A,
//  const DistTensor<T,U,V>& BT,
//  const DistTensor<T,U,V>& BB );
//template<typename T,Distribution U,Distribution V>
//DistTensor<T,U,V> LockedView2x1
//( const DistTensor<T,U,V>& BT, const DistTensor<T,U,V>& BB );

//
// View a two-by-two set of connected matrices
//

template<typename T>
void View2x2
( Tensor<T>& A,
  Tensor<T>& BTL, Tensor<T>& BTR,
  Tensor<T>& BBL, Tensor<T>& BBR );
template<typename T>
Tensor<T> View2x2
( Tensor<T>& BTL, Tensor<T>& BTR,
  Tensor<T>& BBL, Tensor<T>& BBR );
//template<typename T,Distribution U,Distribution V>
//void View2x2
//( DistTensor<T,U,V>& A,
//  DistTensor<T,U,V>& BTL, DistTensor<T,U,V>& BTR,
//  DistTensor<T,U,V>& BBL, DistTensor<T,U,V>& BBR );
//template<typename T,Distribution U,Distribution V>
//DistTensor<T,U,V> View2x2
//( DistTensor<T,U,V>& BTL, DistTensor<T,U,V>& BTR,
//  DistTensor<T,U,V>& BBL, DistTensor<T,U,V>& BBR );

template<typename T>
void LockedView2x2
(       Tensor<T>& A,
  const Tensor<T>& BTL, const Tensor<T>& BTR,
  const Tensor<T>& BBL, const Tensor<T>& BBR );
template<typename T>
Tensor<T> LockedView2x2
( const Tensor<T>& BTL, const Tensor<T>& BTR,
  const Tensor<T>& BBL, const Tensor<T>& BBR );
//template<typename T,Distribution U,Distribution V>
//void LockedView2x2
//(       DistTensor<T,U,V>& A,
//  const DistTensor<T,U,V>& BTL, const DistTensor<T,U,V>& BTR,
//  const DistTensor<T,U,V>& BBL, const DistTensor<T,U,V>& BBR );
//template<typename T,Distribution U,Distribution V>
//DistTensor<T,U,V> LockedView2x2
//( const DistTensor<T,U,V>& BTL, const DistTensor<T,U,V>& BTR,
//  const DistTensor<T,U,V>& BBL, const DistTensor<T,U,V>& BBR );

// Utilities for handling the extra information needed for [MD,* ] and [* ,MD]
//template<typename T,Distribution U,Distribution V>
//void HandleDiagPath
//( DistTensor<T,U,V>& A, const DistTensor<T,U,V>& B );
//template<typename T>
//void HandleDiagPath
//( DistTensor<T,MD,STAR>& A, const DistTensor<T,MD,STAR>& B );
//template<typename T>
//void HandleDiagPath
//( DistTensor<T,STAR,MD>& A, const DistTensor<T,STAR,MD>& B );

} // namespace tmen

#endif // ifndef TMEN_CORE_VIEW_DECL_HPP
