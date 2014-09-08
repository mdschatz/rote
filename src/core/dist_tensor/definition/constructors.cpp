/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#include "tensormental.hpp"

namespace tmen {

template<typename T>
DistTensor<T>::DistTensor( const tmen::Grid& grid )
: dist_(),
  shape_(),

  constrainedModeAlignments_(),
  modeAlignments_(),
  modeShifts_(),

  tensor_(true),

  grid_(&grid),
  commMap_(&(DefaultCommMap())),
  gridView_(grid_, dist_),
  participatingComm_(),

  viewType_(OWNER),
  auxMemory_()
{ SetShifts(); SetParticipatingComm();}

template<typename T>
DistTensor<T>::DistTensor( const Unsigned order, const tmen::Grid& grid )
: dist_(order+1),
  shape_(order, 0),

  constrainedModeAlignments_(order, 0),
  modeAlignments_(order, 0),
  modeShifts_(order, 0),

  tensor_(order, false),

  grid_(&grid),
  commMap_(&(DefaultCommMap())),
  gridView_(grid_, dist_),
  participatingComm_(),

  viewType_(OWNER),
  auxMemory_()
{ SetShifts(); SetParticipatingComm();}

template<typename T>
DistTensor<T>::DistTensor( const TensorDistribution& dist, const tmen::Grid& grid )
: dist_(dist),
  shape_(dist_.size()-1, 0),

  constrainedModeAlignments_(shape_.size(), 0),
  modeAlignments_(shape_.size(), 0),
  modeShifts_(shape_.size(), 0),

  tensor_(shape_.size(), false),

  grid_(&grid),
  commMap_(&(DefaultCommMap())),
  gridView_(grid_, dist_),
  participatingComm_(),

  viewType_(OWNER),
  auxMemory_()
{ SetShifts(); SetParticipatingComm();}

template<typename T>
DistTensor<T>::DistTensor
( const ObjShape& shape, const TensorDistribution& dist, const tmen::Grid& grid )
: dist_(dist),
  shape_(shape),

  constrainedModeAlignments_(shape_.size(), 0),
  modeAlignments_(shape_.size(), 0),
  modeShifts_(shape_.size(), 0),

  tensor_(shape_.size(), false),

  grid_(&grid),
  commMap_(&(DefaultCommMap())),
  gridView_(grid_, dist_),
  participatingComm_(),

  viewType_(OWNER),
  auxMemory_()
{
    if(shape_.size() + 1 != dist_.size())
        LogicError("Error: Distribution must be of same order as object");

    SetShifts();
    ResizeTo( shape );
    SetParticipatingComm();
}

template<typename T>
DistTensor<T>::DistTensor
( const ObjShape& shape, const TensorDistribution& dist, const std::vector<Unsigned>& modeAlignments,
  const tmen::Grid& g )
: dist_(dist),
  shape_(shape),

  constrainedModeAlignments_(shape_.size(), 0),
  modeAlignments_(shape_.size(), 0),
  modeShifts_(shape_.size(), 0),

  tensor_(shape_.size(), false),

  grid_(&g),
  commMap_(&(DefaultCommMap())),
  gridView_(grid_, dist_),
  participatingComm_(),

  viewType_(OWNER),
  auxMemory_()
{
    if(shape_.size() + 1 != dist_.size())
        LogicError("Error: Distribution must be of same order as object");
    Align( modeAlignments );
    dist_ = dist;
    ResizeTo( shape );
    SetParticipatingComm();
}

template<typename T>
DistTensor<T>::DistTensor
( const ObjShape& shape, const TensorDistribution& dist, const std::vector<Unsigned>& modeAlignments,
  const std::vector<Unsigned>& ldims, const tmen::Grid& g )
: dist_(dist),
  shape_(shape),

  constrainedModeAlignments_(shape_.size(), 0),
  modeAlignments_(shape_.size(), 0),
  modeShifts_(shape_.size(), 0),

  tensor_(shape_.size(), false),

  grid_(&g),
  commMap_(&(DefaultCommMap())),
  gridView_(grid_, dist_),
  participatingComm_(),

  viewType_(OWNER),
  auxMemory_()
{
    if(shape_.size() + 1 != dist_.size())
        LogicError("Error: Distribution must be of same order as object");
    Align( modeAlignments );
    ResizeTo( shape, ldims );
    SetParticipatingComm();
}

template<typename T>
DistTensor<T>::DistTensor
( const ObjShape& shape, const TensorDistribution& dist, const std::vector<Unsigned>& modeAlignments,
  const T* buffer, const std::vector<Unsigned>& ldims, const tmen::Grid& g )
: dist_(dist),
  shape_(shape),

  constrainedModeAlignments_(shape_.size(), 0),
  modeAlignments_(shape_.size(), 0),
  modeShifts_(shape_.size(), 0),

  tensor_(shape_.size(), false),

  grid_(&g),
  commMap_(&(DefaultCommMap())),
  gridView_(grid_, dist_),
  participatingComm_(),

  viewType_(OWNER),
  auxMemory_()
{
    if(shape_.size() + 1 != dist_.size())
        LogicError("Error: Distribution must be of same order as object");

    LockedAttach
    ( shape, modeAlignments, buffer, ldims, g );
    SetParticipatingComm();
}

template<typename T>
DistTensor<T>::DistTensor
( const ObjShape& shape, const TensorDistribution& dist, const std::vector<Unsigned>& modeAlignments,
  T* buffer, const std::vector<Unsigned>& ldims, const tmen::Grid& g )
: dist_(dist),
  shape_(shape),

  constrainedModeAlignments_(shape_.size(), 0),
  modeAlignments_(shape_.size(), 0),
  modeShifts_(shape_.size(), 0),

  tensor_(shape_.size(), false),

  grid_(&g),
  commMap_(&(DefaultCommMap())),
  gridView_(grid_, dist_),
  participatingComm_(),

  viewType_(OWNER),
  auxMemory_()
{
    if(shape_.size() + 1 != dist_.size())
        LogicError("Error: Distribution must be of same order as object");

    Attach
    ( shape, modeAlignments, buffer, ldims, g );
    SetParticipatingComm();
}

//////////////////////////////////
/// String distribution versions
//////////////////////////////////

template<typename T>
DistTensor<T>::DistTensor( const std::string& dist, const tmen::Grid& grid )
: dist_(StringToTensorDist(dist)),
  shape_(dist_.size()-1, 0),

  constrainedModeAlignments_(shape_.size(), 0),
  modeAlignments_(shape_.size(), 0),
  modeShifts_(shape_.size(), 0),

  tensor_(shape_.size(), false),

  grid_(&grid),
  commMap_(&(DefaultCommMap())),
  gridView_(grid_, dist_),
  participatingComm_(),

  viewType_(OWNER),
  auxMemory_()
{ SetShifts(); SetParticipatingComm();}

template<typename T>
DistTensor<T>::DistTensor
( const ObjShape& shape, const std::string& dist, const tmen::Grid& grid )
: dist_(StringToTensorDist(dist)),
  shape_(shape),

  constrainedModeAlignments_(shape_.size(), 0),
  modeAlignments_(shape_.size(), 0),
  modeShifts_(shape_.size(), 0),

  tensor_(shape_.size(), false),

  grid_(&grid),
  commMap_(&(DefaultCommMap())),
  gridView_(grid_, dist_),
  participatingComm_(),

  viewType_(OWNER),
  auxMemory_()
{
    if(shape_.size() + 1 != dist_.size())
        LogicError("Error: Distribution must be of same order as object");

    SetShifts();
    ResizeTo( shape );
    SetParticipatingComm();
}

template<typename T>
DistTensor<T>::DistTensor
( const ObjShape& shape, const std::string& dist, const std::vector<Unsigned>& modeAlignments,
  const tmen::Grid& g )
: dist_(StringToTensorDist(dist)),
  shape_(shape),

  constrainedModeAlignments_(shape_.size(), 0),
  modeAlignments_(shape_.size(), 0),
  modeShifts_(shape_.size(), 0),

  tensor_(shape_.size(), false),

  grid_(&g),
  commMap_(&(DefaultCommMap())),
  gridView_(grid_, dist_),
  participatingComm_(),

  viewType_(OWNER),
  auxMemory_()
{
    if(shape_.size() + 1 != dist_.size())
        LogicError("Error: Distribution must be of same order as object");
    Align( modeAlignments );
    ResizeTo( shape );
    SetParticipatingComm();
}

template<typename T>
DistTensor<T>::DistTensor
( const ObjShape& shape, const std::string& dist, const std::vector<Unsigned>& modeAlignments,
  const std::vector<Unsigned>& ldims, const tmen::Grid& g )
: dist_(StringToTensorDist(dist)),
  shape_(shape),

  constrainedModeAlignments_(shape_.size(), 0),
  modeAlignments_(shape_.size(), 0),
  modeShifts_(shape_.size(), 0),

  tensor_(shape_.size(), false),

  grid_(&g),
  commMap_(&(DefaultCommMap())),
  gridView_(grid_, dist_),
  participatingComm_(),

  viewType_(OWNER),
  auxMemory_()
{
    if(shape_.size() + 1 != dist_.size())
        LogicError("Error: Distribution must be of same order as object");
    Align( modeAlignments );
    ResizeTo( shape, ldims );
    SetParticipatingComm();
}

template<typename T>
DistTensor<T>::DistTensor
( const ObjShape& shape, const std::string& dist, const std::vector<Unsigned>& modeAlignments,
  const T* buffer, const std::vector<Unsigned>& ldims, const tmen::Grid& g )
: dist_(StringToTensorDist(dist)),
  shape_(shape),

  constrainedModeAlignments_(shape_.size(), 0),
  modeAlignments_(shape_.size(), 0),
  modeShifts_(shape_.size(), 0),

  tensor_(shape_.size(), false),

  grid_(&g),
  commMap_(&(DefaultCommMap())),
  gridView_(grid_, dist_),
  participatingComm_(),

  viewType_(OWNER),
  auxMemory_()
{
    if(shape_.size() + 1 != dist_.size())
        LogicError("Error: Distribution must be of same order as object");

    LockedAttach
    ( shape, modeAlignments, buffer, ldims, g );
    SetParticipatingComm();
}

template<typename T>
DistTensor<T>::DistTensor
( const ObjShape& shape, const std::string& dist, const std::vector<Unsigned>& modeAlignments,
  T* buffer, const std::vector<Unsigned>& ldims, const tmen::Grid& g )
: dist_(StringToTensorDist(dist)),
  shape_(shape),

  constrainedModeAlignments_(shape_.size(), 0),
  modeAlignments_(shape_.size(), 0),
  modeShifts_(shape_.size(), 0),

  tensor_(shape_.size(), false),

  grid_(&g),
  commMap_(&(DefaultCommMap())),
  gridView_(grid_, dist_),
  participatingComm_(),

  viewType_(OWNER),
  auxMemory_()
{
    if(shape_.size() + 1 != dist_.size())
        LogicError("Error: Distribution must be of same order as object");

    Attach
    ( shape, modeAlignments, buffer, ldims, g );
    SetParticipatingComm();
}

template<typename T>
DistTensor<T>::DistTensor( const DistTensor<T>& A )
: dist_(),
  shape_(),

  constrainedModeAlignments_(),
  modeAlignments_(),
  modeShifts_(),

  tensor_(true),

  grid_(&(A.Grid())),
  commMap_(&(DefaultCommMap())),
  gridView_(grid_, dist_),
  participatingComm_(),

  viewType_(OWNER),
  auxMemory_()
{
#ifndef RELEASE
    CallStackEntry entry("DistTensor[MC,MR]::DistTensor");
#endif
    SetShifts();
    if( &A != this )
        *this = A;
    else
        LogicError("Tried to construct [MC,MR] with itself");
    SetParticipatingComm();
}

template<typename T>
DistTensor<T>::~DistTensor()
{ }

template<typename T>
void
DistTensor<T>::Swap( DistTensor<T>& A )
{
    std::swap( shape_ , A.shape_ );
    std::swap( dist_, A.dist_ );

    std::swap( constrainedModeAlignments_, A.constrainedModeAlignments_ );
    std::swap( modeAlignments_, A.modeAlignments_ );
    std::swap( modeShifts_, A.modeShifts_ );

    tensor_.Swap( A.tensor_ );

    std::swap( grid_, A.grid_ );
    std::swap( gridView_, A.gridView_ );

    std::swap( viewType_, A.viewType_ );
    auxMemory_.Swap( A.auxMemory_ );
}

template<typename T>
tmen::DistData
DistTensor<T>::DistData() const
{
    tmen::DistData data;
    //data.colDist = MC;
    //data.rowDist = MR;
    data.modeAlignments = modeAlignments_;
    data.distribution = dist_;
    data.grid = grid_;
    return data;
}


//
// Utility functions, e.g., TransposeFrom
//

//NOTE: No check for equal distributions
//NOTE: Generalize alignments mismatched case
//NOTE: Generalize CopyFromDifferentGrid case
template<typename T>
const DistTensor<T>&
DistTensor<T>::operator=( const DistTensor<T>& A )
{
#ifndef RELEASE
    CallStackEntry entry("DistTensor = DistTensor");
    AssertNotLocked();
#endif
    if( Grid() == A.Grid() )
    {
        ResizeTo(A);
        if( !Participating() && !A.Participating() )
            return *this;
        if( !AnyElemwiseNotEqual(Alignments(), A.Alignments()) )
        {
            //dist_ = A.TensorDist();
            tensor_ = A.LockedTensor();
            gridView_ = A.gridView_;
            participatingComm_ = A.participatingComm_;
            grid_ = A.grid_;
        }

//        else
//        {
//            const tmen::Grid& g = Grid();
//#ifdef UNALIGNED_WARNINGS
//            if( g.Rank() == 0 )
//                std::cerr << "Unaligned [MC,MR] <- [MC,MR]." << std::endl;
//#endif
//            const Int colRank = ColRank();
//            const Int rowRank = RowRank();
//            const Int colStride = ColStride();
//            const Int rowStride = RowStride();
//            const Int colAlignment = ColAlignment();
//            const Int rowAlignment = RowAlignment();
//            const Int colAlignmentA = A.ColAlignment();
//            const Int rowAlignmentA = A.RowAlignment();
//            const Int colDiff = colAlignment - colAlignmentA;
//            const Int rowDiff = rowAlignment - rowAlignmentA;
//            const Int sendRow = (colRank + colStride + colDiff) % colStride;
//            const Int recvRow = (colRank + colStride - colDiff) % colStride;
//            const Int sendCol = (rowRank + rowStride + rowDiff) % rowStride;
//            const Int recvCol = (rowRank + rowStride - rowDiff) % rowStride;
//            const Int sendRank = sendRow + sendCol*colStride;
//            const Int recvRank = recvRow + recvCol*colStride;
//
//            const Int localHeight = LocalHeight();
//            const Int localWidth = LocalWidth();
//            const Int localHeightA = A.LocalHeight();
//            const Int localWidthA = A.LocalWidth();
//            const Int sendSize = localHeightA*localWidthA;
//            const Int recvSize = localHeight*localWidth;
//            T* auxBuf = auxMemory_.Require( sendSize + recvSize );
//            T* sendBuf = &auxBuf[0];
//            T* recvBuf = &auxBuf[sendSize];
//
//            // Pack
//            const Int ALDim = A.LDim();
//            const T* ABuffer = A.LockedBuffer();
//            PARALLEL_FOR
//            for( Int jLoc=0; jLoc<localWidthA; ++jLoc )
//                MemCopy
//                ( &sendBuf[jLoc*localHeightA],
//                  &ABuffer[jLoc*ALDim], localHeightA );
//
//            // Communicate
//            mpi::SendRecv
//            ( sendBuf, sendSize, sendRank,
//              recvBuf, recvSize, recvRank, g.VCComm() );
//
//            // Unpack
//            T* buffer = Buffer();
//            const Int ldim = LDim();
//            PARALLEL_FOR
//            for( Int jLoc=0; jLoc<localWidth; ++jLoc )
//                MemCopy
//                ( &buffer[jLoc*ldim],
//                  &recvBuf[jLoc*localHeight], localHeight );
//            auxMemory_.Release();
//        }
    }
//    else // the grids don't match
//    {
//        CopyFromDifferentGrid( A );
//    }
    return *this;

}


/*
template<typename T>
void DistTensor<T>::CopyFromDifferentGrid( const DistTensor<T>& A )
{
#ifndef RELEASE
    CallStackEntry cse("[MC,MR]::CopyFromDifferentGrid");
#endif
    ResizeTo( A.Height(), A.Width() );
    // Just need to ensure that each viewing comm contains the other team's
    // owning comm. Congruence is too strong.

    // Compute the number of process rows and columns that each process
    // needs to send to.
    const Int colStride = ColStride();
    const Int rowStride = RowStride();
    const Int colRank = ColRank();
    const Int rowRank = RowRank();
    const Int colStrideA = A.ColStride();
    const Int rowStrideA = A.RowStride();
    const Int colRankA = A.ColRank();
    const Int rowRankA = A.RowRank();
    const Int colGCD = GCD( colStride, colStrideA );
    const Int rowGCD = GCD( rowStride, rowStrideA );
    const Int colLCM = colStride*colStrideA / colGCD;
    const Int rowLCM = rowStride*rowStrideA / rowGCD;
    const Int numColSends = colStride / colGCD;
    const Int numRowSends = rowStride / rowGCD;
    const Int localColStride = colLCM / colStride;
    const Int localRowStride = rowLCM / rowStride;
    const Int localColStrideA = numColSends;
    const Int localRowStrideA = numRowSends;

    const Int colAlign = ColAlignment();
    const Int rowAlign = RowAlignment();
    const Int colAlignA = A.ColAlignment();
    const Int rowAlignA = A.RowAlignment();

    const bool inThisGrid = Participating();
    const bool inAGrid = A.Participating();
    if( !inThisGrid && !inAGrid )
        return;

    const Int maxSendSize =
        (A.Height()/(colStrideA*localColStrideA)+1) *
        (A.Width()/(rowStrideA*localRowStrideA)+1);

    // Translate the ranks from A's VC communicator to this's viewing so that
    // we can match send/recv communicators
    const int sizeA = A.Grid().Size();
    std::vector<int> rankMap(sizeA), ranks(sizeA);
    for( int j=0; j<sizeA; ++j )
        ranks[j] = j;
    mpi::Group viewingGroup;
    mpi::CommGroup( Grid().ViewingComm(), viewingGroup );
    mpi::GroupTranslateRanks
    ( A.Grid().OwningGroup(), sizeA, &ranks[0], viewingGroup, &rankMap[0] );

    // Have each member of A's grid individually send to all numRow x numCol
    // processes in order, while the members of this grid receive from all
    // necessary processes at each step.
    Int requiredMemory = 0;
    if( inAGrid )
        requiredMemory += maxSendSize;
    if( inThisGrid )
        requiredMemory += maxSendSize;
    T* auxBuf = auxMemory_.Require( requiredMemory );
    Int offset = 0;
    T* sendBuf = &auxBuf[offset];
    if( inAGrid )
        offset += maxSendSize;
    T* recvBuf = &auxBuf[offset];

    Int recvRow = 0; // avoid compiler warnings...
    if( inAGrid )
        recvRow = (((colRankA+colStrideA-colAlignA)%colStrideA)+colAlign) %
                  colStride;
    for( Int colSend=0; colSend<numColSends; ++colSend )
    {
        Int recvCol = 0; // avoid compiler warnings...
        if( inAGrid )
            recvCol = (((rowRankA+rowStrideA-rowAlignA)%rowStrideA)+rowAlign) %
                      rowStride;
        for( Int rowSend=0; rowSend<numRowSends; ++rowSend )
        {
            mpi::Request sendRequest;
            // Fire off this round of non-blocking sends
            if( inAGrid )
            {
                // Pack the data
                Int sendHeight = Length(A.LocalHeight(),colSend,numColSends);
                Int sendWidth = Length(A.LocalWidth(),rowSend,numRowSends);
                const T* ABuffer = A.LockedBuffer();
                const Int ALDim = A.LDim();
                PARALLEL_FOR
                for( Int jLoc=0; jLoc<sendWidth; ++jLoc )
                {
                    const Int j = rowSend+jLoc*localRowStrideA;
                    for( Int iLoc=0; iLoc<sendHeight; ++iLoc )
                    {
                        const Int i = colSend+iLoc*localColStrideA;
                        sendBuf[iLoc+jLoc*sendHeight] = ABuffer[i+j*ALDim];
                    }
                }
                // Send data
                const Int recvVCRank = recvRow + recvCol*colStride;
                const Int recvViewingRank =
                    Grid().VCToViewingMap( recvVCRank );
                mpi::ISend
                ( sendBuf, sendHeight*sendWidth, recvViewingRank,
                  Grid().ViewingComm(), sendRequest );
            }
            // Perform this round of recv's
            if( inThisGrid )
            {
                const Int sendColOffset = (colSend*colStrideA+colAlignA) % colStrideA;
                const Int recvColOffset = (colSend*colStrideA+colAlign) % colStride;
                const Int sendRowOffset = (rowSend*rowStrideA+rowAlignA) % rowStrideA;
                const Int recvRowOffset = (rowSend*rowStrideA+rowAlign) % rowStride;

                const Int firstSendRow = (((colRank+colStride-recvColOffset)%colStride)+sendColOffset)%colStrideA;
                const Int firstSendCol = (((rowRank+rowStride-recvRowOffset)%rowStride)+sendRowOffset)%rowStrideA;

                const Int colShift = (colRank+colStride-recvColOffset)%colStride;
                const Int rowShift = (rowRank+rowStride-recvRowOffset)%rowStride;
                const Int numColRecvs = Length( colStrideA, colShift, colStride );
                const Int numRowRecvs = Length( rowStrideA, rowShift, rowStride );

                // Recv data
                // For now, simply receive sequentially. Until we switch to
                // nonblocking recv's, we won't be using much of the
                // recvBuf
                Int sendRow = firstSendRow;
                for( Int colRecv=0; colRecv<numColRecvs; ++colRecv )
                {
                    const Int sendColShift = Shift( sendRow, colAlignA, colStrideA ) + colSend*colStrideA;
                    const Int sendHeight = Length( A.Height(), sendColShift, colLCM );
                    const Int localColOffset = (sendColShift-ColShift()) / colStride;

                    Int sendCol = firstSendCol;
                    for( Int rowRecv=0; rowRecv<numRowRecvs; ++rowRecv )
                    {
                        const Int sendRowShift = Shift( sendCol, rowAlignA, rowStrideA ) + rowSend*rowStrideA;
                        const Int sendWidth = Length( A.Width(), sendRowShift, rowLCM );
                        const Int localRowOffset = (sendRowShift-RowShift()) / rowStride;

                        const Int sendVCRank = sendRow+sendCol*colStrideA;
                        mpi::Recv
                        ( recvBuf, sendHeight*sendWidth, rankMap[sendVCRank],
                          Grid().ViewingComm() );

                        // Unpack the data
                        T* buffer = Buffer();
                        const Int ldim = LDim();
                        PARALLEL_FOR
                        for( Int jLoc=0; jLoc<sendWidth; ++jLoc )
                        {
                            const Int j = localRowOffset+jLoc*localRowStride;
                            for( Int iLoc=0; iLoc<sendHeight; ++iLoc )
                            {
                                const Int i = localColOffset+iLoc*localColStride;
                                buffer[i+j*ldim] = recvBuf[iLoc+jLoc*sendHeight];
                            }
                        }
                        // Set up the next send col
                        sendCol = (sendCol + rowStride) % rowStrideA;
                    }
                    // Set up the next send row
                    sendRow = (sendRow + colStride) % colStrideA;
                }
            }
            // Ensure that this round of non-blocking sends completes
            if( inAGrid )
            {
                mpi::Wait( sendRequest );
                recvCol = (recvCol + rowStrideA) % rowStride;
            }
        }
        if( inAGrid )
            recvRow = (recvRow + colStrideA) % colStride;
    }
    auxMemory_.Release();
}
*/
// PAUSED PASS HERE

#define FULL(T) \
    template class DistTensor<T>;

FULL(Int);
#ifndef DISABLE_FLOAT
FULL(float);
#endif
FULL(double);

#ifndef DISABLE_COMPLEX
#ifndef DISABLE_FLOAT
FULL(Complex<float>);
#endif
FULL(Complex<double>);
#endif

}
