/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef TMEN_CORE_REPARTITION_IMPL_HPP
#define TMEN_CORE_REPARTITION_IMPL_HPP

namespace rote {

// To make our life easier. Undef'd at the bottom of the header
#define TEN  Tensor<T>
#define DTEN DistTensor<T>

//
// RepartitionUp
//

template<typename T>
inline void
RepartitionUp
( TEN& AT, TEN& A0,
         TEN& A1,
  TEN& AB, TEN& A2, Mode mode, Unsigned A1Dimension )
{
#ifndef RELEASE
    CallStackEntry cse("RepartitionUp [Tensor]");
#endif
    PartitionUp( AT, A0, A1, mode, A1Dimension );
    View( A2, AB );
}

template<typename T>
inline void
RepartitionUp
( DTEN& AT, DTEN& A0,
          DTEN& A1,
  DTEN& AB, DTEN& A2, Mode mode, Unsigned A1Dimension )
{
#ifndef RELEASE
    CallStackEntry cse("RepartitionUp [DistTensor]");
#endif
    PartitionUp( AT, A0, A1, mode, A1Dimension );
    View( A2, AB );
}

template<typename T>
inline void
LockedRepartitionUp
( const TEN& AT, TEN& A0,
               TEN& A1,
  const TEN& AB, TEN& A2, Mode mode, Unsigned A1Dimension )
{
#ifndef RELEASE
    CallStackEntry cse("LockedRepartitionUp [Tensor]");
#endif
    LockedPartitionUp( AT, A0, A1, mode, A1Dimension );
    LockedView( A2, AB );
}

template<typename T>
inline void
LockedRepartitionUp
( const DTEN& AT, DTEN& A0,
                DTEN& A1,
  const DTEN& AB, DTEN& A2, Mode mode, Unsigned A1Dimension )
{
#ifndef RELEASE
    CallStackEntry cse("LockedRepartitionUp [DistTensor]");
#endif
    LockedPartitionUp( AT, A0, A1, mode, A1Dimension );
    LockedView( A2, AB );
}

//
// RepartitionDown
//

template<typename T>
inline void
RepartitionDown
( TEN& AT, TEN& A0,
         TEN& A1,
  TEN& AB, TEN& A2, Mode mode, Unsigned A1Dimension )
{
#ifndef RELEASE
    CallStackEntry cse("RepartitionDown [Tensor]");
#endif
    View( A0, AT );
    PartitionDown( AB, A1, A2, mode, A1Dimension );
}

template<typename T>
inline void
RepartitionDown
( DTEN& AT, DTEN& A0,
          DTEN& A1,
  DTEN& AB, DTEN& A2, Mode mode, Unsigned A1Dimension )
{
#ifndef RELEASE
    CallStackEntry cse("RepartitionDown [DistTensor]");
#endif
    View( A0, AT );
    PartitionDown( AB, A1, A2, mode, A1Dimension );
}

template<typename T>
inline void
LockedRepartitionDown
( const TEN& AT, TEN& A0,
               TEN& A1,
  const TEN& AB, TEN& A2, Mode mode, Unsigned A1Dimension )
{
#ifndef RELEASE
    CallStackEntry cse("LockedRepartitionDown [Tensor]");
#endif
    LockedView( A0, AT );
    LockedPartitionDown( AB, A1, A2, mode, A1Dimension );
}

template<typename T>
inline void
LockedRepartitionDown
( const DTEN& AT, DTEN& A0,
                DTEN& A1,
  const DTEN& AB, DTEN& A2, Mode mode, Unsigned A1Dimension )
{
#ifndef RELEASE
    CallStackEntry cse("LockedRepartitionDown [DistTensor]");
#endif
    LockedView( A0, AT );
    LockedPartitionDown( AB, A1, A2, mode, A1Dimension );
}

#undef DTEN
#undef TEN

} // namespace tmen

#endif // ifndef TMEN_CORE_REPARTITION_IMPL_HPP
