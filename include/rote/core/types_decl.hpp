/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#ifndef ROTE_CORE_TYPES_DECL_HPP
#define ROTE_CORE_TYPES_DECL_HPP

#include <complex>
#include <vector>

namespace rote {

typedef unsigned char byte;

// If these are changes, you must make sure that they have 
// existing MPI datatypes. This is only sometimes true for 'long long'
typedef int Int;
typedef unsigned Unsigned;

typedef char Index;
typedef Unsigned Mode;
typedef std::vector<Index> IndexArray;
typedef std::vector<Mode> ModeArray;

//NOTE: I'm really not sure why, but I cannot have Shape
typedef std::vector<Unsigned> ObjShape;
typedef std::vector<Unsigned> Location;
//typedef std::vector<Unsigned> Permutation;

//Redistribution enum
enum RedistType {AG, A2A, Local, RS, RTO, AR, GTO, BCast, Scatter, Perm};

//template<typename Real>
//using Complex = std::complex<Real>;

typedef std::complex<float>  scomplex;
typedef std::complex<double> dcomplex;

// For extracting the underlying real datatype, 
// e.g., typename Base<Scalar>::type a = 3.0;
template<typename Real>
struct Base { typedef Real type; };
template<typename Real>
struct Base<std::complex<Real> > { typedef Real type; };
#define BASE(F) typename Base<F>::type 
#define COMPLEX(F) std::complex<BASE(F)>

template<typename Real>
struct ValueInt
{
    Real value;
    Int index;

    static bool Lesser( const ValueInt<Real>& a, const ValueInt<Real>& b )
    { return a.value < b.value; }
    static bool Greater( const ValueInt<Real>& a, const ValueInt<Real>& b )
    { return a.value > b.value; }
};

template<typename Real>
struct ValueInt<std::complex<Real> >
{
    std::complex<Real> value;
    Int index;

    static bool Lesser( const ValueInt<Real>& a, const ValueInt<Real>& b )
    { return Abs(a.value) < Abs(b.value); }
    static bool Greater( const ValueInt<Real>& a, const ValueInt<Real>& b )
    { return Abs(a.value) > Abs(b.value); }
};

template<typename Real>
struct ValueIntPair
{
    Real value;
    Int indices[2];
    
    static bool Lesser( const ValueInt<Real>& a, const ValueInt<Real>& b )
    { return a.value < b.value; }
    static bool Greater( const ValueInt<Real>& a, const ValueInt<Real>& b )
    { return a.value > b.value; }
};

template<typename Real>
struct ValueIntPair<std::complex<Real> >
{
    std::complex<Real> value;
    Int indices[2];
    
    static bool Lesser
    ( const ValueIntPair<Real>& a, const ValueIntPair<Real>& b )
    { return Abs(a.value) < Abs(b.value); }
    static bool Greater
    ( const ValueIntPair<Real>& a, const ValueIntPair<Real>& b )
    { return Abs(a.value) > Abs(b.value); }
};

namespace viewtype_wrapper {
enum ViewType
{
    OWNER = 0x0,
    VIEW = 0x1,
    OWNER_FIXED = 0x2,
    VIEW_FIXED = 0x3,
    LOCKED_OWNER = 0x4, // unused
    LOCKED_VIEW = 0x5,
    LOCKED_OWNER_FIXED = 0x6, // unused
    LOCKED_VIEW_FIXED = 0x7
};
static inline bool IsOwner( ViewType v ) 
{ return ( v & VIEW  ) == 0; }
static inline bool IsViewing( ViewType v )
{ return ( v & VIEW  ) != 0; }
static inline bool IsShrinkable( ViewType v )
{ return ( v & OWNER_FIXED ) == 0; }
static inline bool IsFixedSize( ViewType v )
{ return ( v & OWNER_FIXED ) != 0; }
static inline bool IsUnlocked( ViewType v )
{ return ( v & LOCKED_OWNER     ) == 0; }
static inline bool IsLocked( ViewType v )
{ return ( v & LOCKED_OWNER     ) != 0; }
}
using namespace viewtype_wrapper;

namespace norm_type_wrapper {
enum NormType
{
    ONE_NORM,           // Operator one norm
    INFINITY_NORM,      // Operator infinity norm
    ENTRYWISE_ONE_NORM, // One-norm of vectorized matrix
    MAX_NORM,           // Maximum entry-wise magnitude
    NUCLEAR_NORM,       // One-norm of the singular values
    FROBENIUS_NORM,     // Two-norm of the singular values
    TWO_NORM            // Infinity-norm of the singular values
};
}
using namespace norm_type_wrapper;

namespace FileFormatNS {
enum FileFormat
{
    AUTO, // Automatically detect from file extension
    ASCII,
    ASCII_MATLAB,
    BINARY,
    BINARY_FLAT,
    BMP,
    JPG,
    JPEG,
    TENSOR_MARKET,
    PNG,
    PPM,
    XBM,
    XPM,
    FileFormat_MAX // For detecting number of entries in enum
};
}
using namespace FileFormatNS;

} // namespace rote

#endif // ifndef ROTE_CORE_TYPES_DECL_HPP
