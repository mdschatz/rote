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
: AbstractDistTensor<T>(grid)
{ this->SetShifts(); }

template<typename T>
DistTensor<T>::DistTensor
( const std::vector<Int>& shape, const TensorDistribution& dist, const tmen::Grid& grid )
: AbstractDistTensor<T>(shape, dist, grid)
{
	if(shape.size() != dist.size())
		LogicError("Error: Distribution must be of same order as object");

	this->SetShifts();
	this->ResizeTo( shape );
}

template<typename T>
DistTensor<T>::DistTensor
( const std::vector<Int>& shape, const TensorDistribution& dist, const std::vector<Int>& indices, const tmen::Grid& grid )
: AbstractDistTensor<T>(shape, dist, indices, grid)
{
	if(shape.size() != dist.size())
		LogicError("Error: Distribution must be of same order as object");

	this->SetShifts();
	this->ResizeTo( shape );
}

template<typename T>
DistTensor<T>::DistTensor
( const std::vector<Int>& shape, const TensorDistribution& dist, const std::vector<Int>& indices, const std::vector<Int>& modeAlignments,
  const tmen::Grid& g )
: AbstractDistTensor<T>(shape, dist, indices, g)
{
	if(shape.size() != dist.size())
		LogicError("Error: Distribution must be of same order as object");
    this->Align( modeAlignments );
    this->ResizeTo( shape );
}

template<typename T>
DistTensor<T>::DistTensor
( const std::vector<Int>& shape, const TensorDistribution& dist, const std::vector<Int>& indices, const std::vector<Int>& modeAlignments,
  const std::vector<Int>& ldims, const tmen::Grid& g )
: AbstractDistTensor<T>(shape, dist, indices, g)
{ 
	if(shape.size() != dist.size())
		LogicError("Error: Distribution must be of same order as object");
    this->Align( modeAlignments );
    this->ResizeTo( shape, ldims );
}

template<typename T>
DistTensor<T>::DistTensor
( const std::vector<Int>& shape, const TensorDistribution& dist, const std::vector<Int>& indices, const std::vector<Int>& modeAlignments,
  const T* buffer, const std::vector<Int>& ldims, const tmen::Grid& g )
: AbstractDistTensor<T>(shape, dist, indices, g)
{
	if(shape.size() != dist.size())
		LogicError("Error: Distribution must be of same order as object");

    this->LockedAttach
    ( shape, modeAlignments, buffer, ldims, g );
}

template<typename T>
DistTensor<T>::DistTensor
( const std::vector<Int>& shape, const TensorDistribution& dist, const std::vector<Int>& indices, const std::vector<Int>& modeAlignments,
  T* buffer, const std::vector<Int>& ldims, const tmen::Grid& g )
: AbstractDistTensor<T>(shape, dist, indices, g)
{
	if(shape.size() != dist.size())
		LogicError("Error: Distribution must be of same order as object");

    this->Attach
    ( shape, modeAlignments, buffer, ldims, g );
}

template<typename T>
DistTensor<T>::DistTensor( const DistTensor<T>& A )
: AbstractDistTensor<T>(A.Grid())
{
#ifndef RELEASE
    CallStackEntry entry("DistTensor[MC,MR]::DistTensor");
#endif
    this->SetShifts();
    if( &A != this )
        *this = A;
    else
        LogicError("Tried to construct [MC,MR] with itself");
}

/*
template<typename T>
DistTensor<T>::DistTensor( DistTensor<T>&& A )
: AbstractDistTensor<T>(std::move(A))
{ }

template<typename T>
DistTensor<T>&
DistTensor<T>::operator=( DistTensor<T>&& A )
{
    AbstractDistTensor<T>::operator=( std::move(A) );
    return *this;
}
*/

template<typename T>
DistTensor<T>::~DistTensor()
{ }

template<typename T>
tmen::DistData
DistTensor<T>::DistData() const
{
    tmen::DistData data;
    //data.colDist = MC;
    //data.rowDist = MR;
    data.modeAlignments = this->modeAlignments_;
    data.distribution = this->dist_;
    data.grid = this->grid_;
    return data;
}


//NOTE: This refers to the stride within grid mode.  NOT the stride through index of elements of tensor
template<typename T>
Int
DistTensor<T>::ModeStride(Int mode) const
{
	return this->gridView_.ModeWrapStride(mode);
}

template<typename T>
Int
DistTensor<T>::ModeRank(Int mode) const
{ return this->gridView_.ModeLoc(mode); }

template<typename T>
void
DistTensor<T>::AlignWith( const tmen::DistData& data )
{
/*
#ifndef RELEASE
    CallStackEntry entry("[MC,MR]::AlignWith");
#endif
    const Grid& grid = *data.grid;
    this->SetGrid( grid );
    if( data.colDist == MC && data.rowDist == MR )
    {
        this->colAlignment_ = data.colAlignment;
        this->rowAlignment_ = data.rowAlignment;
        this->constrainedColAlignment_ = true;
        this->constrainedRowAlignment_ = true;
    }
    else if( data.colDist == MC && data.rowDist == STAR )
    {
        this->colAlignment_ = data.colAlignment;
        this->constrainedColAlignment_ = true; 
    }
    else if( data.colDist == MR && data.rowDist == MC )
    {
        this->colAlignment_ = data.rowAlignment;
        this->rowAlignment_ = data.colAlignment;
        this->constrainedColAlignment_ = true;
        this->constrainedRowAlignment_ = true;
    }
    else if( data.colDist == MR && data.rowDist == STAR )
    {
        this->rowAlignment_ = data.colAlignment;
        this->constrainedRowAlignment_ = true;
    }
    else if( data.colDist == STAR && data.rowDist == MC )
    {
        this->colAlignment_ = data.rowAlignment;
        this->constrainedColAlignment_ = true;
    }
    else if( data.colDist == STAR && data.rowDist == MR )
    {
        this->rowAlignment_ = data.rowAlignment;
        this->constrainedRowAlignment_ = true;
    }
    else if( data.colDist == STAR && data.rowDist == VC )
    {
        this->colAlignment_ = data.rowAlignment % this->ColStride();
        this->constrainedColAlignment_ = true;
    }
    else if( data.colDist == STAR && data.rowDist == VR )
    {
        this->rowAlignment_ = data.rowAlignment % this->RowStride();
        this->constrainedRowAlignment_ = true;
    }
    else if( data.colDist == VC && data.rowDist == STAR )
    {
        this->colAlignment_ = data.colAlignment % this->ColStride();
        this->constrainedColAlignment_ = true;
    }
    else if( data.colDist == VR && data.rowDist == STAR )
    {
        this->rowAlignment_ = data.colAlignment % this->RowStride();
        this->constrainedRowAlignment_ = true;
    }
#ifndef RELEASE
    else LogicError("Nonsensical alignment");
#endif
    this->SetShifts();
*/
}

template<typename T>
void
DistTensor<T>::AlignWith( const AbstractDistTensor<T>& A )
{ this->AlignWith( A.DistData() ); }

template<typename T>
void
DistTensor<T>::AlignModeWith( Int mode, const tmen::DistData& data )
{
/*
#ifndef RELEASE
    CallStackEntry entry("[MC,MR]::AlignColsWith");
    if( *this->grid_ != *data.grid )
        LogicError("Grids do not match");
#endif
    if( data.colDist == MC )
        this->colAlignment_ = data.colAlignment;
    else if( data.rowDist == MC )
        this->colAlignment_ = data.rowAlignment;
    else if( data.colDist == VC )
        this->colAlignment_ = data.colAlignment % this->ColStride();
    else if( data.rowDist == VC )
        this->colAlignment_ = data.rowAlignment % this->ColStride();
#ifndef RELEASE
    else LogicError("Nonsensical alignment");
#endif
    this->constrainedColAlignment_ = true;
    this->SetShifts();
*/
}

template<typename T>
void
DistTensor<T>::AlignModeWith( Int mode, const AbstractDistTensor<T>& A )
{ this->AlignModeWith( mode, A.DistData() ); }

template<typename T>
void
DistTensor<T>::Attach
( const std::vector<Int>& dims, const std::vector<Int>& modeAlignments, 
  T* buffer, const std::vector<Int>& ldims, const tmen::Grid& g )
{
/*
#ifndef RELEASE
    CallStackEntry entry("[MC,MR]::Attach");
#endif
    this->Empty();

    this->grid_ = &g;
    this->shape_ = dims;
    this->modeAlignments_ = modeAligns;
    this->viewType_ = VIEW;
    this->SetShifts();
    if( this->Participating() )
    {
        Int localHeight = Length(height,this->colShift_,this->ColStride());
        Int localWidth = Length(width,this->rowShift_,this->RowStride());
        this->matrix_.Attach_( localHeight, localWidth, buffer, ldim );
    }
*/
}

template<typename T>
void
DistTensor<T>::LockedAttach
( const std::vector<Int>& dims, const std::vector<Int>& modeAlignments, 
  const T* buffer, const std::vector<Int>& ldims, const tmen::Grid& g )
{
#ifndef RELEASE
    CallStackEntry entry("[MC,MR]::LockedAttach");
#endif
    this->grid_ = &g;
    this->shape_ = dims;
    this->modeAlignments_ = modeAlignments;
    this->SetShifts();
/*
    this->Empty();

    this->grid_ = &g;
    this->height_ = height;
    this->width_ = width;
    this->colAlignment_ = colAlignment;
    this->rowAlignment_ = rowAlignment;
    this->viewType_ = LOCKED_VIEW;
    this->SetShifts();
    if( this->Participating() )
    {
        Int localHeight = Length(height,this->colShift_,this->ColStride());
        Int localWidth = Length(width,this->rowShift_,this->RowStride());
        this->matrix_.LockedAttach_( localHeight, localWidth, buffer, ldim );
    }
*/
}

//TODO: FIX Participating
template<typename T>
void
DistTensor<T>::ResizeTo( const std::vector<Int>& dims )
{
#ifndef RELEASE
    CallStackEntry entry("[MC,MR]::ResizeTo");
    this->AssertNotLocked();
#endif
    this->shape_ = dims;
    this->tensor_.ResizeTo(Lengths(dims, this->modeShifts_, this->gridView_.Shape()));
/*
    this->height_ = height;
    this->width_ = width;
    if( this->Participating() )
        this->matrix_.ResizeTo_
        ( Length(height,this->ColShift(),this->ColStride()),
          Length(width, this->RowShift(),this->RowStride()) );
*/
}

template<typename T>
void
DistTensor<T>::ResizeTo( const std::vector<Int>& dims, const std::vector<Int>& ldims )
{
#ifndef RELEASE
    CallStackEntry entry("[MC,MR]::ResizeTo");
    this->AssertNotLocked();
#endif
    this->shape_ = dims;
/*
    this->height_ = height;
    this->width_ = width;
    if( this->Participating() )
        this->matrix_.ResizeTo_
        ( Length(height,this->ColShift(),this->ColStride()),
          Length(width, this->RowShift(),this->RowStride()), ldim );
*/
}


//NOTE: INCREDIBLY INEFFICIENT (RECREATING A COMMUNICATOR ON EVERY REQUEST!!!
//TODO: FIX THIS
template<typename T>
T
DistTensor<T>::Get( const std::vector<Int>& index ) const
{
#ifndef RELEASE
    CallStackEntry entry("[MC,MR]::Get");
    this->AssertValidEntry( index );
#endif
    const Int owningProc = this->DetermineLinearIndexOwner(index);
    mpi::Comm comm;

    const tmen::GridView& gv = this->GridView();
    int myLinearRank = gv.LinearRank();
    int commColor = 0;
    int commKey = myLinearRank;
    mpi::CommSplit(mpi::COMM_WORLD, commColor, commKey, comm);

    T u;
    if(gv.LinearRank() == owningProc){
    	const std::vector<Int> localLoc = this->Global2LocalIndex(index);
    	u = this->GetLocal(localLoc);
    }

    mpi::Broadcast( u, owningProc, comm);
    return u;
}

template<typename T>
void
DistTensor<T>::Set( const std::vector<Int>& index, T u )
{
#ifndef RELEASE
    CallStackEntry entry("[MC,MR]::Set");
    this->AssertValidEntry( index );
#endif
    const Int owningProc = this->DetermineLinearIndexOwner(index);


    if(this->Grid().LinearRank() == owningProc){
    	const std::vector<Int> localLoc = this->Global2LocalIndex(index);
        //std::ostringstream msg;
        /*
        msg << "GlobalIndex: [" << index[0];
        for(int i = 1; i < index.size(); i++)
            msg << ", " << index[i];
        msg << "] is owned by proc" << owningProc;
    	msg << " at local index [" << localLoc[0];
    	for(int i = 1; i < localLoc.size(); i++)
    	    msg << ", " << localLoc[i];
    	msg << "]\n";
    	printf("%s", msg.str().c_str());
    	*/
    	this->SetLocal(localLoc, u);
    }

}

template<typename T>
void
DistTensor<T>::Update( const std::vector<Int>& index, T u )
{
#ifndef RELEASE
    CallStackEntry entry("[MC,MR]::Update");
    this->AssertValidEntry( index );
#endif
    const Int owningProc = this->DetermineLinearIndexOwner(index);
    if(this->Grid().LinearRank() == owningProc){
    	const std::vector<Int> localLoc = this->Global2LocalIndex(index);
    	this->UpdateLocal(localLoc, u);
    }
}

template<typename T>
void
DistTensor<T>::GetDiagonal
( DistTensor<T>& d, Int offset ) const
{
/*
#ifndef RELEASE
    CallStackEntry entry("[MC,MR]::GetDiagonal");
    if( d.Viewing() )
        this->AssertSameGrid( d.Grid() );
    if( ( d.Viewing() || d.ConstrainedColAlignment() ) &&
        !d.AlignedWithDiagonal( *this, offset ) )
    {
        std::ostringstream os;
        os << mpi::WorldRank() << "\n"
           << "offset:         " << offset << "\n"
           << "colAlignment:   " << this->colAlignment_ << "\n"
           << "rowAlignment:   " << this->rowAlignment_ << "\n"
           << "d.diagPath:     " << d.diagPath_ << "\n"
           << "d.colAlignment: " << d.colAlignment_ << std::endl;
        std::cerr << os.str();
        LogicError("d must be aligned with the 'offset' diagonal");
    }
#endif

    const tmen::Grid& g = this->Grid();
    if( !d.Viewing() )
    {
        d.SetGrid( g );
        if( !d.ConstrainedColAlignment() )
            d.AlignWithDiagonal( *this, offset );
    }
    const Int diagLength = this->DiagonalLength(offset);
    d.ResizeTo( diagLength, 1 );
    if( !d.Participating() )
        return;

    Int iStart, jStart;
    const Int diagShift = d.ColShift();
    if( offset >= 0 )
    {
        iStart = diagShift;
        jStart = diagShift+offset;
    }
    else
    {
        iStart = diagShift-offset;
        jStart = diagShift;
    }

    const Int colStride = this->ColStride();
    const Int rowStride = this->RowStride();
    const Int colShift = this->ColShift();
    const Int rowShift = this->RowShift();
    const Int iLocStart = (iStart-colShift) / colStride;
    const Int jLocStart = (jStart-rowShift) / rowStride;

    const Int lcm = g.LCM();
    const Int localDiagLength = d.LocalHeight();
    T* dBuf = d.Buffer();
    const T* buffer = this->LockedBuffer();
    const Int ldim = this->LDim();

    PARALLEL_FOR
    for( Int k=0; k<localDiagLength; ++k )
    {
        const Int iLoc = iLocStart + k*(lcm/colStride);
        const Int jLoc = jLocStart + k*(lcm/rowStride);
        dBuf[k] = buffer[iLoc+jLoc*ldim];
    }
*/
}

template<typename T>
DistTensor<T>
DistTensor<T>::GetDiagonal( Int offset ) const
{
    DistTensor<T> d( this->Grid() );
    GetDiagonal( d, offset );
    return d;
}

template<typename T>
void
DistTensor<T>::SetDiagonal
( const DistTensor<T>& d, Int offset )
{
/*
#ifndef RELEASE
    CallStackEntry entry("[MC,MR]::SetDiagonal");
    this->AssertSameGrid( d.Grid() );
    if( d.Width() != 1 )
        LogicError("d must be a column vector");
    const Int diagLength = this->DiagonalLength(offset);
    if( diagLength != d.Height() )
    {
        std::ostringstream msg;
        msg << "d is not of the same length as the diagonal:\n"
            << "  A ~ " << this->Height() << " x " << this->Width() << "\n"
            << "  d ~ " << d.Height() << " x " << d.Width() << "\n"
            << "  A diag length: " << diagLength << "\n";
        LogicError( msg.str() );
    }
    if( !d.AlignedWithDiagonal( *this, offset ) )
        LogicError("d must be aligned with the 'offset' diagonal");
#endif

    if( !d.Participating() )
        return;

    Int iStart,jStart;
    const Int diagShift = d.ColShift();
    if( offset >= 0 )
    {
        iStart = diagShift;
        jStart = diagShift+offset;
    }
    else
    {
        iStart = diagShift-offset;
        jStart = diagShift;
    }

    const Int colStride = this->ColStride();
    const Int rowStride = this->RowStride();
    const Int colShift = this->ColShift();
    const Int rowShift = this->RowShift();
    const Int iLocStart = (iStart-colShift) / colStride;
    const Int jLocStart = (jStart-rowShift) / rowStride;

    const Int localDiagLength = d.LocalHeight();
    const T* dBuf = d.LockedBuffer();
    T* buffer = this->Buffer();
    const Int ldim = this->LDim();
    const Int lcm = this->Grid().LCM();
    PARALLEL_FOR
    for( Int k=0; k<localDiagLength; ++k )
    {
        const Int iLoc = iLocStart + k*(lcm/colStride);
        const Int jLoc = jLocStart + k*(lcm/rowStride);
        buffer[iLoc+jLoc*ldim] = dBuf[k];
    }
*/
}

//
// Utility functions, e.g., TransposeFrom
//

template<typename T>
const DistTensor<T>&
DistTensor<T>::operator=( const DistTensor<T>& A )
{
/*
#ifndef RELEASE
    CallStackEntry entry("[MC,MR] = [MC,MR]");
    this->AssertNotLocked();
#endif
    if( this->Grid() == A.Grid() )
    {
        this->SetAlignmentsAndResize
        ( A.ColAlignment(), A.RowAlignment(), A.Height(), A.Width() );
        if( !this->Participating() && !A.Participating() )
            return *this;
        if( this->ColAlignment() == A.ColAlignment() &&
            this->RowAlignment() == A.RowAlignment() )
        {
            this->matrix_ = A.LockedTensor();
        }
        else
        {
            const tmen::Grid& g = this->Grid();
#ifdef UNALIGNED_WARNINGS
            if( g.Rank() == 0 )
                std::cerr << "Unaligned [MC,MR] <- [MC,MR]." << std::endl;
#endif
            const Int colRank = this->ColRank();
            const Int rowRank = this->RowRank();
            const Int colStride = this->ColStride();
            const Int rowStride = this->RowStride();
            const Int colAlignment = this->ColAlignment();
            const Int rowAlignment = this->RowAlignment();
            const Int colAlignmentA = A.ColAlignment();
            const Int rowAlignmentA = A.RowAlignment();
            const Int colDiff = colAlignment - colAlignmentA;
            const Int rowDiff = rowAlignment - rowAlignmentA;
            const Int sendRow = (colRank + colStride + colDiff) % colStride;
            const Int recvRow = (colRank + colStride - colDiff) % colStride;
            const Int sendCol = (rowRank + rowStride + rowDiff) % rowStride;
            const Int recvCol = (rowRank + rowStride - rowDiff) % rowStride;
            const Int sendRank = sendRow + sendCol*colStride;
            const Int recvRank = recvRow + recvCol*colStride;

            const Int localHeight = this->LocalHeight();
            const Int localWidth = this->LocalWidth();
            const Int localHeightA = A.LocalHeight();
            const Int localWidthA = A.LocalWidth();
            const Int sendSize = localHeightA*localWidthA;
            const Int recvSize = localHeight*localWidth;
            T* auxBuf = this->auxMemory_.Require( sendSize + recvSize );
            T* sendBuf = &auxBuf[0];
            T* recvBuf = &auxBuf[sendSize];

            // Pack
            const Int ALDim = A.LDim();
            const T* ABuffer = A.LockedBuffer();
            PARALLEL_FOR
            for( Int jLoc=0; jLoc<localWidthA; ++jLoc )
                MemCopy
                ( &sendBuf[jLoc*localHeightA], 
                  &ABuffer[jLoc*ALDim], localHeightA );

            // Communicate
            mpi::SendRecv
            ( sendBuf, sendSize, sendRank, 
              recvBuf, recvSize, recvRank, g.VCComm() );

            // Unpack
            T* buffer = this->Buffer();
            const Int ldim = this->LDim();
            PARALLEL_FOR
            for( Int jLoc=0; jLoc<localWidth; ++jLoc )
                MemCopy
                ( &buffer[jLoc*ldim], 
                  &recvBuf[jLoc*localHeight], localHeight );
            this->auxMemory_.Release();
        }
    }
    else // the grids don't match
    {
        CopyFromDifferentGrid( A );
    }
*/
    return *this;

}


/*
template<typename T>
void DistTensor<T>::CopyFromDifferentGrid( const DistTensor<T>& A )
{
#ifndef RELEASE
    CallStackEntry cse("[MC,MR]::CopyFromDifferentGrid");
#endif
    this->ResizeTo( A.Height(), A.Width() ); 
    // Just need to ensure that each viewing comm contains the other team's
    // owning comm. Congruence is too strong.

    // Compute the number of process rows and columns that each process 
    // needs to send to.
    const Int colStride = this->ColStride();
    const Int rowStride = this->RowStride();
    const Int colRank = this->ColRank();
    const Int rowRank = this->RowRank();
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

    const Int colAlign = this->ColAlignment();
    const Int rowAlign = this->RowAlignment();
    const Int colAlignA = A.ColAlignment();
    const Int rowAlignA = A.RowAlignment();

    const bool inThisGrid = this->Participating();
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
    mpi::CommGroup( this->Grid().ViewingComm(), viewingGroup );
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
    T* auxBuf = this->auxMemory_.Require( requiredMemory );
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
                    this->Grid().VCToViewingMap( recvVCRank );
                mpi::ISend
                ( sendBuf, sendHeight*sendWidth, recvViewingRank,
                  this->Grid().ViewingComm(), sendRequest );
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
                    const Int localColOffset = (sendColShift-this->ColShift()) / colStride;

                    Int sendCol = firstSendCol;
                    for( Int rowRecv=0; rowRecv<numRowRecvs; ++rowRecv )
                    {
                        const Int sendRowShift = Shift( sendCol, rowAlignA, rowStrideA ) + rowSend*rowStrideA;
                        const Int sendWidth = Length( A.Width(), sendRowShift, rowLCM );
                        const Int localRowOffset = (sendRowShift-this->RowShift()) / rowStride;

                        const Int sendVCRank = sendRow+sendCol*colStrideA;
                        mpi::Recv
                        ( recvBuf, sendHeight*sendWidth, rankMap[sendVCRank],
                          this->Grid().ViewingComm() );
                        
                        // Unpack the data
                        T* buffer = this->Buffer();
                        const Int ldim = this->LDim();
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
    this->auxMemory_.Release();
}
*/
// PAUSED PASS HERE

//
// Functions which explicitly work in the complex plane
//

template<typename T>
void
DistTensor<T>::SetRealPart( Int i, Int j, BASE(T) u )
{
/*
#ifndef RELEASE
    CallStackEntry entry("[MC,MR]::SetRealPart");
    this->AssertValidEntry( i, j );
#endif
    const tmen::Grid& g = this->Grid(); 
    const Int ownerRow = (i + this->ColAlignment()) % g.Height();
    const Int ownerCol = (j + this->RowAlignment()) % g.Width();
    const Int ownerRank = ownerRow + ownerCol*g.Height();
    if( g.VCRank() == ownerRank )
    {
        const Int iLoc = (i-this->ColShift()) / g.Height();
        const Int jLoc = (j-this->RowShift()) / g.Width();
        this->SetLocalRealPart( iLoc, jLoc, u );
    }
*/
}

template<typename T>
void
DistTensor<T>::SetImagPart( Int i, Int j, BASE(T) u )
{
/*
#ifndef RELEASE
    CallStackEntry entry("[MC,MR]::SetImagPart");
    this->AssertValidEntry( i, j );
#endif
    this->ComplainIfReal();
    const tmen::Grid& g = this->Grid(); 
    const Int ownerRow = (i + this->ColAlignment()) % g.Height();
    const Int ownerCol = (j + this->RowAlignment()) % g.Width();
    const Int ownerRank = ownerRow + ownerCol*g.Height();
    if( g.VCRank() == ownerRank )
    {
        const Int iLoc = (i-this->ColShift()) / g.Height();
        const Int jLoc = (j-this->RowShift()) / g.Width();
        this->SetLocalImagPart( iLoc, jLoc, u );
    }
*/
}

template<typename T>
void
DistTensor<T>::UpdateRealPart( Int i, Int j, BASE(T) u )
{
/*
#ifndef RELEASE
    CallStackEntry entry("[MC,MR]::UpdateRealPart");
    this->AssertValidEntry( i, j );
#endif
    const tmen::Grid& g = this->Grid(); 
    const Int ownerRow = (i + this->ColAlignment()) % g.Height();
    const Int ownerCol = (j + this->RowAlignment()) % g.Width();
    const Int ownerRank = ownerRow + ownerCol*g.Height();
    if( g.VCRank() == ownerRank )
    {
        const Int iLoc = (i-this->ColShift()) / g.Height();
        const Int jLoc = (j-this->RowShift()) / g.Width();
        this->UpdateLocalRealPart( iLoc, jLoc, u );
    }
*/
}

template<typename T>
void
DistTensor<T>::UpdateImagPart( Int i, Int j, BASE(T) u )
{
/*
#ifndef RELEASE
    CallStackEntry entry("[MC,MR]::UpdateImagPart");
    this->AssertValidEntry( i, j );
#endif
    this->ComplainIfReal();
    const tmen::Grid& g = this->Grid(); 
    const Int ownerRow = (i + this->ColAlignment()) % g.Height();
    const Int ownerCol = (j + this->RowAlignment()) % g.Width();
    const Int ownerRank = ownerRow + ownerCol*g.Height();
    if( g.VCRank() == ownerRank )
    {
        const Int iLoc = (i-this->ColShift()) / g.Height();
        const Int jLoc = (j-this->RowShift()) / g.Width();
        this->UpdateLocalImagPart( iLoc, jLoc, u );
    }
*/
}

template<typename T>
void
DistTensor<T>::GetRealPartOfDiagonal
( DistTensor<BASE(T)>& d, Int offset ) const
{
/*
#ifndef RELEASE
    CallStackEntry entry("[MC,MR]::GetRealPartOfDiagonal");
    if( d.Viewing() )
        this->AssertSameGrid( d.Grid() );
#endif
    typedef BASE(T) R;
    const tmen::Grid& g = this->Grid();
    if( !d.Viewing() )
    {
        d.SetGrid( g );
        if( !d.ConstrainedColAlignment() )
            d.AlignWithDiagonal( this->DistData(), offset );
    }
    const Int length = this->DiagonalLength( offset );
    d.ResizeTo( length, 1 );
    if( !d.Participating() )
        return;

    const Int r = g.Height();
    const Int c = g.Width();
    const Int lcm = g.LCM();
    const Int colShift = this->ColShift();
    const Int rowShift = this->RowShift();
    const Int diagShift = d.ColShift();

    Int iStart, jStart;
    if( offset >= 0 )
    {
        iStart = diagShift;
        jStart = diagShift+offset;
    }
    else
    {
        iStart = diagShift-offset;
        jStart = diagShift;
    }

    const Int iLocStart = (iStart-colShift) / r;
    const Int jLocStart = (jStart-rowShift) / c;

    const Int localDiagLength = d.LocalHeight();

    const T* thisBuffer = this->LockedBuffer();
    const Int thisLDim = this->LDim();
    R* dBuf = d.Buffer();
    PARALLEL_FOR
    for( Int k=0; k<localDiagLength; ++k )
    {
        const Int iLoc = iLocStart + k*(lcm/r);
        const Int jLoc = jLocStart + k*(lcm/c);
        dBuf[k] = RealPart(thisBuffer[iLoc+jLoc*thisLDim]);
    }
*/
}

template<typename T>
void
DistTensor<T>::GetImagPartOfDiagonal
( DistTensor<BASE(T)>& d, Int offset ) const
{
/*
#ifndef RELEASE
    CallStackEntry entry("[MC,MR]::GetImagPartOfDiagonal");
    if( d.Viewing() )
        this->AssertSameGrid( d.Grid() );
#endif
    typedef BASE(T) R;
    const tmen::Grid& g = this->Grid();
    if( !d.Viewing() )
    {
        d.SetGrid( g );
        if( !d.ConstrainedColAlignment() )
            d.AlignWithDiagonal( this->DistData(), offset );
    }
    const Int length = this->DiagonalLength( offset );
    d.ResizeTo( length, 1 );
    if( !d.Participating() )
        return;

    const Int r = g.Height();
    const Int c = g.Width();
    const Int lcm = g.LCM();
    const Int colShift = this->ColShift();
    const Int rowShift = this->RowShift();
    const Int diagShift = d.ColShift();

    Int iStart, jStart;
    if( offset >= 0 )
    {
        iStart = diagShift;
        jStart = diagShift+offset;
    }
    else
    {
        iStart = diagShift-offset;
        jStart = diagShift;
    }

    const Int iLocStart = (iStart-colShift) / r;
    const Int jLocStart = (jStart-rowShift) / c;

    const Int localDiagLength = d.LocalHeight();

    const T* thisBuffer = this->LockedBuffer();
    const Int thisLDim = this->LDim();
    R* dBuf = d.Buffer();
    PARALLEL_FOR
    for( Int k=0; k<localDiagLength; ++k )
    {
        const Int iLoc = iLocStart + k*(lcm/r);
        const Int jLoc = jLocStart + k*(lcm/c);
        dBuf[k] = ImagPart(thisBuffer[iLoc+jLoc*thisLDim]);
    }
*/
}

/*
template<typename T>
DistTensor<BASE(T)>
DistTensor<T>::GetRealPartOfDiagonal( Int offset ) const
{
    DistTensor<BASE(T)> d( this->Grid() );
    GetRealPartOfDiagonal( d, offset );
    return d;
}

template<typename T>
DistTensor<BASE(T)>
DistTensor<T>::GetImagPartOfDiagonal( Int offset ) const
{
    DistTensor<BASE(T)> d( this->Grid() );
    GetImagPartOfDiagonal( d, offset );
    return d;
}
*/

template<typename T>
void
DistTensor<T>::SetRealPartOfDiagonal
( const DistTensor<BASE(T)>& d, Int offset )
{
/*
#ifndef RELEASE
    CallStackEntry entry("[MC,MR]::SetRealPartOfDiagonal");
    this->AssertSameGrid( d.Grid() );
    if( d.Width() != 1 )
        LogicError("d must be a column vector");
    const Int length = this->DiagonalLength( offset );
    if( length != d.Height() )
    {
        std::ostringstream msg;
        msg << "d is not of the same length as the diagonal:\n"
            << "  A ~ " << this->Height() << " x " << this->Width() << "\n"
            << "  d ~ " << d.Height() << " x " << d.Width() << "\n"
            << "  A diag length: " << length << "\n";
        LogicError( msg.str() );
    }
#endif
    typedef BASE(T) R;
    if( !d.Participating() )
        return;

    const tmen::Grid& g = this->Grid();
    const Int r = g.Height();
    const Int c = g.Width();
    const Int lcm = g.LCM();
    const Int colShift = this->ColShift();
    const Int rowShift = this->RowShift();
    const Int diagShift = d.ColShift();

    Int iStart,jStart;
    if( offset >= 0 )
    {
        iStart = diagShift;
        jStart = diagShift+offset;
    }
    else
    {
        iStart = diagShift-offset;
        jStart = diagShift;
    }

    const Int iLocStart = (iStart-colShift) / r;
    const Int jLocStart = (jStart-rowShift) / c;

    const Int localDiagLength = d.LocalHeight();
    const R* dBuf = d.LockedBuffer();
    PARALLEL_FOR
    for( Int k=0; k<localDiagLength; ++k )
    {
        const Int iLoc = iLocStart + k*(lcm/r);
        const Int jLoc = jLocStart + k*(lcm/c);
        this->SetLocalRealPart( iLoc, jLoc, dBuf[k] );
    }
*/
}

template<typename T>
void
DistTensor<T>::SetImagPartOfDiagonal
( const DistTensor<BASE(T)>& d, Int offset )
{
/*
#ifndef RELEASE
    CallStackEntry entry("[MC,MR]::SetImagPartOfDiagonal");
    this->AssertSameGrid( d.Grid() );
    if( d.Width() != 1 )
        LogicError("d must be a column vector");
    const Int length = this->DiagonalLength( offset );
    if( length != d.Height() )
    {
        std::ostringstream msg;
        msg << "d is not of the same length as the diagonal:\n"
            << "  A ~ " << this->Height() << " x " << this->Width() << "\n"
            << "  d ~ " << d.Height() << " x " << d.Width() << "\n"
            << "  A diag length: " << length << "\n";
        LogicError( msg.str() );
    }
#endif
    this->ComplainIfReal();
    typedef BASE(T) R;
    if( !d.Participating() )
        return;

    const tmen::Grid& g = this->Grid();
    const Int r = g.Height();
    const Int c = g.Width();
    const Int lcm = g.LCM();
    const Int colShift = this->ColShift();
    const Int rowShift = this->RowShift();
    const Int diagShift = d.ColShift();

    Int iStart,jStart;
    if( offset >= 0 )
    {
        iStart = diagShift;
        jStart = diagShift+offset;
    }
    else
    {
        iStart = diagShift-offset;
        jStart = diagShift;
    }

    const Int iLocStart = (iStart-colShift) / r;
    const Int jLocStart = (jStart-rowShift) / c;

    const Int localDiagLength = d.LocalHeight();
    const R* dBuf = d.LockedBuffer();
    PARALLEL_FOR
    for( Int k=0; k<localDiagLength; ++k )
    {
        const Int iLoc = iLocStart + k*(lcm/r);
        const Int jLoc = jLocStart + k*(lcm/c);
        this->SetLocalImagPart( iLoc, jLoc, dBuf[k] );
    }
*/
}

#define PROTO(T) template class DistTensor<T>
#define COPY(T) \
  template DistTensor<T>::DistTensor( const DistTensor<T>& A )
#define FULL(T) \
  PROTO(T);


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

} // namespace tmen
