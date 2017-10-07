/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#ifndef ROTE_CORE_COMPLEX_DECL_HPP
#define ROTE_CORE_COMPLEX_DECL_HPP

namespace rote {

template<typename Real>
std::ostream& operator<<( std::ostream& os, std::complex<Real> alpha );

// For querying whether or not a scalar is complex,
// e.g., Isstd::complex<Scalar>::val
template<typename Real>
struct IsComplex { enum { val=0 }; };
template<typename Real>
struct IsComplex<std::complex<Real> > { enum { val=1 }; };

// Return the real/imaginary part of a real or complex number
template<typename Real>
Real RealPart( const Real& alpha );
template<typename Real>
Real RealPart( const std::complex<Real>& alpha );
template<typename Real>
Real ImagPart( const Real& alpha );
template<typename Real>
Real ImagPart( const std::complex<Real>& alpha );

// Set the real/imaginary part of a real or complex number
template<typename Real>
void SetRealPart( Real& alpha, const Real& beta );
template<typename Real>
void SetRealPart( std::complex<Real>& alpha, const Real& beta );
template<typename Real>
void SetImagPart( Real& alpha, const Real& beta );
template<typename Real>
void SetImagPart( std::complex<Real>& alpha, const Real& beta );

// Update the real/imaginary part of a real or complex number
template<typename Real>
void UpdateRealPart( Real& alpha, const Real& beta );
template<typename Real>
void UpdateRealPart( std::complex<Real>& alpha, const Real& beta );
template<typename Real>
void UpdateImagPart( Real& alpha, const Real& beta );
template<typename Real>
void UpdateImagPart( std::complex<Real>& alpha, const Real& beta );

// Euclidean (l_2) magnitudes
template<typename F>
BASE(F) Abs( const F& alpha );

// Square-root free (l_1) magnitudes
template<typename F>
BASE(F) FastAbs( const F& alpha );

// Conjugation
template<typename Real>
Real Conj( const Real& alpha );
template<typename Real>
std::complex<Real> Conj( const std::complex<Real>& alpha );

// Square root
template<typename F>
F Sqrt( const F& alpha );

// Cosine
template<typename F>
F Cos( const F& alpha );

// Sine
template<typename F>
F Sin( const F& alpha );

// Tangent
template<typename F>
F Tan( const F& alpha );

// Hyperbolic cosine
template<typename F>
F Cosh( const F& alpha );

// Hyperbolic sine
template<typename F>
F Sinh( const F& alpha );

// Inverse cosine
template<typename F>
F Acos( const F& alpha );

// Inverse sine
template<typename F>
F Asin( const F& alpha );

// Inverse tangent
template<typename F>
F Atan( const F& alpha );

// Coordinate-based inverse tangent
template<typename Real>
Real Atan2( const Real& y, const Real& x );

// Inverse hyperbolic cosine
template<typename F>
F Acosh( const F& alpha );

// Inverse hyperbolic sine
template<typename F>
F Asinh( const F& alpha );

// Inverse hyperbolic tangent
template<typename F>
F Atanh( const F& alpha );

// Complex argument
template<typename F>
F Arg( const F& alpha );

#ifndef SWIG
// Convert polar coordinates to the complex number
template<typename Real>
std::complex<Real> Polar( const Real& r, const Real& theta=0 );
#endif

// Exponential
template<typename F>
F Exp( const F& alpha );

// Power, return alpha^beta
// (every combination supported by std::pow)
template<typename F,typename T>
F Pow( const F& alpha, const T& beta );

// Logarithm
template<typename F>
F Log( const F& alpha );

} // namespace rote

#endif // ifndef ROTE_CORE_COMPLEX_DECL_HPP
