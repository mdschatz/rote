/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef TMEN_CORE_SLIDEPARTITION_IMPL_HPP
#define TMEN_CORE_SLIDEPARTITION_IMPL_HPP

namespace tmen {

// To make our life easier. Undef'd at the bottom of the header
#define TEN  Tensor<T>
#define DTEN DistTensor<T>

//
// SlidePartitionUp
//

template<typename T>
inline void
SlidePartitionUp
( TEN& AT, TEN& A0,
         TEN& A1,
  TEN& AB, TEN& A2, Index index )
{
#ifndef RELEASE
    CallStackEntry entry("SlidePartitionUp [Tensor]");
#endif
    View( AT, A0, index );
    View2x1( AB, A1, A2, index );
}

template<typename T>
inline void
SlidePartitionUp
( DTEN& AT, DTEN& A0,
          DTEN& A1,
  DTEN& AB, DTEN& A2, Index index )
{
#ifndef RELEASE
    CallStackEntry entry("SlidePartitionUp [DistTensor]");
#endif
    View( AT, A0, index );
    View2x1( AB, A1, A2, index );
}

template<typename T>
inline void
SlideLockedPartitionUp
( TEN& AT, const TEN& A0,
         const TEN& A1,
  TEN& AB, const TEN& A2, Index index )
{
#ifndef RELEASE
    CallStackEntry entry("SlideLockedPartitionUp [Tensor]");
#endif
    LockedView( AT, A0, index );
    LockedView2x1( AB, A1, A2, index );
}

template<typename T>
inline void
SlideLockedPartitionUp
( DTEN& AT, const DTEN& A0,
          const DTEN& A1,
  DTEN& AB, const DTEN& A2, Index index )
{
#ifndef RELEASE
    CallStackEntry entry("SlideLockedPartitionUp [DistTensor]");
#endif
    LockedView( AT, A0, index );
    LockedView2x1( AB, A1, A2, index );
}

//
// SlidePartitionDown
//

template<typename T>
inline void
SlidePartitionDown
( TEN& AT, TEN& A0,
         TEN& A1,
  TEN& AB, TEN& A2, Index index )
{
#ifndef RELEASE
    CallStackEntry entry("SlidePartitionDown [Tensor]");
#endif
    View2x1( AT, A0, A1, index );
    View( AB, A2, index );
}

template<typename T>
inline void
SlidePartitionDown
( DTEN& AT, DTEN& A0,
          DTEN& A1,
  DTEN& AB, DTEN& A2, Index index )
{
#ifndef RELEASE
    CallStackEntry entry("SlidePartitionDown [DistTensor]");
#endif
    View2x1( AT, A0, A1, index );
    View( AB, A2, index );
}

template<typename T>
inline void
SlideLockedPartitionDown
( TEN& AT, const TEN& A0,
         const TEN& A1,
  TEN& AB, const TEN& A2, Index index )
{
#ifndef RELEASE
    CallStackEntry entry("SlideLockedPartitionDown [Tensor]");
#endif
    LockedView2x1( AT, A0, A1, index );
    LockedView( AB, A2, index );
}

template<typename T>
inline void
SlideLockedPartitionDown
( DTEN& AT, const DTEN& A0,
          const DTEN& A1,
  DTEN& AB, const DTEN& A2, Index index )
{
#ifndef RELEASE
    CallStackEntry entry("SlideLockedPartitionDown [DistTensor]");
#endif
    LockedView2x1( AT, A0, A1, index );
    LockedView( AB, A2, index );
}

#undef DTEN
#undef TEN

} // namespace tmen

#endif // ifndef TMEN_CORE_SLIDEPARTITION_IMPL_HPP
