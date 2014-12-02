/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#include "tensormental.hpp"

namespace tmen {

#ifndef RELEASE
template<typename T>
void
DistTensor<T>::AssertNotLocked() const
{
    if( Locked() )
        LogicError("Assertion that tensor not be a locked view failed");
}

template<typename T>
void
DistTensor<T>::AssertNotStoringData() const
{
    if( tensor_.MemorySize() > 0 )
        LogicError("Assertion that tensor not be storing data failed");
}

template<typename T>
void
DistTensor<T>::AssertValidEntry( const Location& loc ) const
{
#ifndef RELEASE
    CallStackEntry entry("[MC,MR]::AssertValidEntry");
#endif
    const Unsigned order = Order();
    if(loc.size() != order )
    {
        LogicError("Index must be of same order as object");
    }
    if(!ElemwiseLessThan(loc, shape_))
    {
        Unsigned i;
        std::ostringstream msg;
        msg << "Entry (";
        for(i = 0; i < order - 1; i++)
          msg << loc[i] << ", ";
        msg << loc[order - 1] << ") is out of bounds of ";

        for(i = 0; i < order - 1; i++)
              msg << shape_[i] << " x ";
        msg << shape_[order - 1] << " tensor.";
        LogicError( msg.str() );
    }
}

//TODO: FIX ASSERTIONS
template<typename T>
void
DistTensor<T>::AssertValidSubtensor
( const Location& loc, const ObjShape& shape ) const
{
    const Unsigned order = Order();
    if(shape.size() != order)
        LogicError("Shape must be of same order as object");
    if(loc.size() != order)
        LogicError("Indices must be of same order as object");
    if( AnyNegativeElem(loc) )
        LogicError("Indices of subtensor must not be negative");
    if( AnyNegativeElem(shape) )
        LogicError("Dimensions of subtensor must not be negative");

    Location maxLoc(order);
    ElemwiseSum(loc, shape, maxLoc);

    if( AnyElemwiseGreaterThan(maxLoc, shape_) )
    {
        Unsigned i;
        std::ostringstream msg;
        msg << "Subtensor is out of bounds: accessing up to (";
        for(i = 0; i < order - 1; i++)
          msg << maxLoc[i] << ",";
        msg << maxLoc[order - 1] << ") of ";

        for(i = 0; i < order - 1; i++)
          msg << Dimension(i) << " x ";
        msg << Dimension(order - 1) << " tensor.";
        LogicError( msg.str() );
    }
}

template<typename T>
void
DistTensor<T>::AssertSameGrid( const tmen::Grid& grid ) const
{
    if( Grid() != grid )
        LogicError("Assertion that grids match failed");
}

template<typename T>
void
DistTensor<T>::AssertSameSize( const ObjShape& shape ) const
{
    const Unsigned order = Order();
    if( shape.size() != order)
      LogicError("Argument must be of same order as object");
    if( AnyElemwiseNotEqual(shape, shape_) )
        LogicError("Argument must match shape of this object");
}

template<typename T>
void
DistTensor<T>::AssertMergeableModes(const std::vector<ModeArray>& oldModes) const
{
    tensor_.AssertMergeableModes(oldModes);
}

template<typename T>
void
AssertConforming2x1
( const DistTensor<T>& AT, const DistTensor<T>& AB, Mode mode )
{
    std::vector<Mode> negFilterAT(1);
    std::vector<Mode> negFilterAB(1);
    negFilterAT[0] = mode;
    negFilterAB[0] = mode;

    if( AnyElemwiseNotEqual(NegFilterVector(AT.Shape(), negFilterAT), NegFilterVector(AB.Shape(), negFilterAB)) )
    {
        Unsigned i;
        std::ostringstream msg;
        msg << "2x1 is not conformant. Top is ";
        if(AT.Order() > 0)
            msg << AT.Dimension(0);
        for(i = 1; i < AT.Order(); i++)
            msg << " x " << AT.Dimension(i);
        msg << ", bottom is ";
        if(AB.Order() > 0)
            msg << AB.Dimension(0);
        for(i = 1; i < AB.Order(); i++)
            msg << " x " << AB.Dimension(i);
        LogicError( msg.str() );
    }
    if( AnyElemwiseNotEqual(NegFilterVector(AT.Alignments(), negFilterAT), NegFilterVector(AB.Alignments(), negFilterAB)) )
        LogicError("2x1 is not aligned");
}

#endif // RELEASE

#define FULL(T) \
    template class DistTensor<T>;

FULL(Int);
#ifndef DISABLE_FLOAT
FULL(float);
#endif
FULL(double);

#ifndef DISABLE_COMPLEX
#ifndef DISABLE_FLOAT
FULL(std::complex<float>);
#endif
FULL(std::complex<double>);
#endif


#ifndef RELEASE

#define CONFORMING(T) \
  template void AssertConforming2x1( const DistTensor<T>& AT, const DistTensor<T>& AB, Mode mode ); \

CONFORMING(Int);
#ifndef DISABLE_FLOAT
CONFORMING(float);
#endif // ifndef DISABLE_FLOAT
CONFORMING(double);
#ifndef DISABLE_COMPLEX
#ifndef DISABLE_FLOAT
CONFORMING(std::complex<float>);
#endif // ifndef DISABLE_FLOAT
CONFORMING(std::complex<double>);
#endif // ifndef DISABLE_COMPLEX

#endif

}
