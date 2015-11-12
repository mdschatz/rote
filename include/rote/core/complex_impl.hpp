/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#ifndef ROTE_CORE_COMPLEX_IMPL_HPP
#define ROTE_CORE_COMPLEX_IMPL_HPP

#include "rote/core/complex_decl.hpp"

namespace rote {

template<typename Real>
std::ostream& operator<<( std::ostream& os, std::complex<Real> alpha )
{
    os << alpha.real() << "+" << alpha.imag() << "i";
    return os;
}

template<typename Real>
inline Real
RealPart( const Real& alpha )
{ return alpha; }

template<typename Real>
inline Real
RealPart( const std::complex<Real>& alpha )
{ return alpha.real(); }

template<typename Real>
inline Real
ImagPart( const Real& alpha )
{ return 0; }

template<typename Real>
inline Real
ImagPart( const std::complex<Real>& alpha )
{ return alpha.imag(); }

template<typename Real>
inline void
SetRealPart( Real& alpha, const Real& beta )
{ alpha = beta; }

template<typename Real>
inline void
SetRealPart( std::complex<Real>& alpha, const Real& beta )
{ 
#if __cplusplus > 199711L
alpha.real(beta); 
#else
std::complex<Real> tmp(beta, 0);
alpha = tmp;
#endif
}

template<typename Real>
inline void
SetImagPart( Real& alpha, const Real& beta )
{
#ifndef RELEASE
    CallStackEntry cse("SetImagPart");
#endif
    LogicError("Nonsensical assignment");
}

template<typename Real>
inline void
SetImagPart( std::complex<Real>& alpha, const Real& beta )
{ 
#if __cplusplus > 199711L
alpha.imag(beta); 
#else
std::complex<Real> tmp(0, beta);
alpha = tmp;
#endif
}

template<typename Real>
inline void
UpdateRealPart( Real& alpha, const Real& beta )
{ alpha += beta; }

template<typename Real>
inline void
UpdateRealPart( std::complex<Real>& alpha, const Real& beta )
{ 
#if __cplusplus > 199711L
alpha.real( alpha.real()+beta ); 
#else
std::complex<Real> tmp(beta, 0);
alpha += tmp;
#endif
}

template<typename Real>
inline void
UpdateImagPart( Real& alpha, const Real& beta )
{
#ifndef RELEASE
    CallStackEntry cse("UpdateImagPart");
#endif
    LogicError("Nonsensical update");
}

template<typename Real>
inline void
UpdateImagPart( std::complex<Real>& alpha, const Real& beta )
{ 
#if __cplusplus > 199711L
alpha.imag( alpha.imag()+beta ); 
#else
std::complex<Real> tmp(0, beta);
alpha += beta;
#endif
}

template<typename F>
inline BASE(F)
Abs( const F& alpha )
{ return std::abs(alpha); }

template<typename F>
inline BASE(F)
FastAbs( const F& alpha )
{ return Abs(RealPart(alpha)) + Abs(ImagPart(alpha)); }

template<typename Real>
inline Real
Conj( const Real& alpha )
{ return alpha; }

template<typename Real>
inline std::complex<Real>
Conj( const std::complex<Real>& alpha )
{ return std::complex<Real>(alpha.real(),-alpha.imag()); }

template<typename F>
inline F
Sqrt( const F& alpha )
{ return std::sqrt(alpha); }

template<typename F>
inline F
Cos( const F& alpha )
{ return std::cos(alpha); }

template<typename F>
inline F
Sin( const F& alpha )
{ return std::sin(alpha); }

template<typename F>
inline F
Tan( const F& alpha )
{ return std::tan(alpha); }

template<typename F>
inline F
Cosh( const F& alpha )
{ return std::cosh(alpha); }

template<typename F>
inline F
Sinh( const F& alpha )
{ return std::sinh(alpha); }

template<typename F>
inline F
Tanh( const F& alpha )
{ return std::tanh(alpha); }

template<typename F>
inline F
Acos( const F& alpha )
{ return std::acos(alpha); }

template<typename F>
inline F
Asin( const F& alpha )
{ return std::asin(alpha); }

template<typename F>
inline F
Atan( const F& alpha )
{ return std::atan(alpha); }

template<typename Real>
inline Real
Atan2( const Real& y, const Real& x )
{ return std::atan2( y, x ); }

template<typename F>
inline F
Acosh( const F& alpha )
{ return alpha; }

template<typename F>
inline F
Asinh( const F& alpha )
{ return alpha; }

template<typename F>
inline F
Atanh( const F& alpha )
{ return alpha; }

template<typename F>
inline BASE(F)
Arg( const F& alpha )
{ return Atan2( ImagPart(alpha), RealPart(alpha) ); }

template<typename Real>
inline std::complex<Real>
Polar( const Real& r, const Real& theta )
{ return std::polar(r,theta); }

template<typename F>
inline F
Exp( const F& alpha )
{ return std::exp(alpha); }

template<typename F,typename T>
inline F
Pow( const F& alpha, const T& beta )
{ return std::pow(alpha,beta); }

template<typename F>
inline F
Log( const F& alpha )
{ return std::log(alpha); }

} // namespace rote
#endif
