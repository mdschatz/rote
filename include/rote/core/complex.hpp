/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef ROTE_CORE_COMPLEX_HPP
#define ROTE_CORE_COMPLEX_HPP

#include "environment.hpp"

namespace rote {

template<typename Real>
std::ostream& operator<<( std::ostream& os, std::complex<Real> alpha ) {
    os << alpha.real() << "+" << alpha.imag() << "i";
    return os;
};

// For querying whether or not a scalar is complex,
// e.g., Isstd::complex<Scalar>::val
template<typename Real>
struct IsComplex {
  enum { val=0 };
};

template<typename Real>
struct IsComplex<std::complex<Real> > {
  enum { val=1 };
};

// Return the real/imaginary part of a real or complex number
template<typename Real>
inline
Real RealPart( const Real& alpha ){
  return alpha;
};

template<typename Real>
inline
Real RealPart( const std::complex<Real>& alpha ){
  return alpha.real();
};

template<typename Real>
inline
Real ImagPart( const Real& alpha ){
  NOT_USED(alpha); return 0;
};

template<typename Real>
inline
Real ImagPart( const std::complex<Real>& alpha ){
  return alpha.imag();
};

// Set the real/imaginary part of a real or complex number
template<typename Real>
inline
void SetRealPart( Real& alpha, const Real& beta ){
  alpha = beta;
};

template<typename Real>
inline
void SetRealPart( std::complex<Real>& alpha, const Real& beta ){
  #if __cplusplus > 199711L
  alpha.real(beta);
  #else
  std::complex<Real> tmp(beta, 0);
  alpha = tmp;
  #endif
};

template<typename Real>
inline
void SetImagPart( Real& alpha, const Real& beta ){
      NOT_USED(alpha); NOT_USED(beta);
      LogicError("Nonsensical assignment");
};

template<typename Real>
inline
void SetImagPart( std::complex<Real>& alpha, const Real& beta ){
  #if __cplusplus > 199711L
  alpha.imag(beta);
  #else
  std::complex<Real> tmp(0, beta);
  alpha = tmp;
  #endif
};

// Update the real/imaginary part of a real or complex number
template<typename Real>
inline
void UpdateRealPart( Real& alpha, const Real& beta ){
  alpha += beta;
};

template<typename Real>
inline
void UpdateRealPart( std::complex<Real>& alpha, const Real& beta ) {
  #if __cplusplus > 199711L
  alpha.real( alpha.real()+beta );
  #else
  std::complex<Real> tmp(beta, 0);
  alpha += tmp;
  #endif
};

template<typename Real>
inline
void UpdateImagPart( Real& alpha, const Real& beta ) {
    (void) alpha; (void) beta;
    LogicError("Nonsensical update");
};

template<typename Real>
inline
void UpdateImagPart( std::complex<Real>& alpha, const Real& beta ) {
  #if __cplusplus > 199711L
  alpha.imag( alpha.imag()+beta );
  #else
  std::complex<Real> tmp(0, beta);
  alpha += beta;
  #endif
};

// Euclidean (l_2) magnitudes
template<typename F>
inline
BASE(F) Abs( const F& alpha ){
  return std::abs(alpha);
};

// Square-root free (l_1) magnitudes
template<typename F>
inline
BASE(F) FastAbs( const F& alpha ){
  return Abs(RealPart(alpha)) + Abs(ImagPart(alpha));
};

// Conjugation
template<typename Real>
inline
Real Conj( const Real& alpha ){
  return alpha;
};
template<typename Real>
inline
std::complex<Real> Conj( const std::complex<Real>& alpha ){
  return std::complex<Real>(alpha.real(),-alpha.imag());
};

// Square root
template<typename F>
inline
F Sqrt( const F& alpha ){
  return std::sqrt(alpha);
};

// Cosine
template<typename F>
inline
F Cos( const F& alpha ){
  return std::cos(alpha);
};

// Sine
template<typename F>
inline
F Sin( const F& alpha ){
  return std::sin(alpha);
};

// Tangent
template<typename F>
inline
F Tan( const F& alpha ){
  return std::tan(alpha);
};

// Hyperbolic cosine
template<typename F>
inline
F Cosh( const F& alpha ){
  return std::cosh(alpha);
};

// Hyperbolic sine
template<typename F>
inline
F Sinh( const F& alpha ){
  return std::sinh(alpha);
};

// Hyperbolic tan
template<typename F>
inline
F Tanh( const F& alpha ){
  return std::tanh(alpha);
};

// Inverse cosine
template<typename F>
inline
F Acos( const F& alpha ){
  return std::acos(alpha);
};

// Inverse sine
template<typename F>
inline
F Asin( const F& alpha ){
  return std::asin(alpha);
};

// Inverse tangent
template<typename F>
inline
F Atan( const F& alpha ){
  return std::atan(alpha);
};

// Coordinate-based inverse tangent
template<typename Real>
inline
Real Atan2( const Real& y, const Real& x ){
  return std::atan2( y, x );
};

// Inverse hyperbolic cosine
template<typename F>
inline
F Acosh( const F& alpha ){
  return alpha;
};

// Inverse hyperbolic sine
template<typename F>
inline
F Asinh( const F& alpha ){
  return alpha; };

// Inverse hyperbolic tangent
template<typename F>
inline
F Atanh( const F& alpha ){
  return alpha; };

// Complex argument
template<typename F>
inline
F Arg( const F& alpha ){
  return Atan2( ImagPart(alpha), RealPart(alpha) );
};

#ifndef SWIG
// Convert polar coordinates to the complex number
template<typename Real>
inline
std::complex<Real> Polar( const Real& r, const Real& theta=0 ){
  return std::polar(r,theta);
};
#endif

// Exponential
template<typename F>
inline
F Exp( const F& alpha ){ return std::exp(alpha); };

// Power, return alpha^beta
// (every combination supported by std::pow)
template<typename F,typename T>
inline
F Pow( const F& alpha, const T& beta ){ return std::pow(alpha,beta); };

// Logarithm
template<typename F>
inline
F Log( const F& alpha ){ return std::log(alpha); };

} // namespace rote

#endif // ifndef ROTE_CORE_COMPLEX_HPP
