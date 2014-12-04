/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef TMEN_CORE_FLAME_HPP
#define TMEN_CORE_FLAME_HPP

extern "C" {

typedef int FLA_Error;
FLA_Error FLA_Bsvd_v_opd_var1
( int       k,
  int       mU,
  int       mV,
  int       nGH,
  int       nIterMax,
  double*   d, int dInc,
  double*   e, int eInc,
  tmen::dcomplex* G, int rsG, int csG,
  tmen::dcomplex* H, int rsH, int csH,
  double*   U, int rsU, int csU,
  double*   V, int rsV, int csV,
  int       nb );

FLA_Error FLA_Bsvd_v_opz_var1
( int       k,
  int       mU,
  int       mV,
  int       nGH,
  int       nIterMax,
  double*   d, int dInc,
  double*   e, int eInc,
  tmen::dcomplex* G, int rsG, int csG,
  tmen::dcomplex* H, int rsH, int csH,
  tmen::dcomplex* U, int rsU, int csU,
  tmen::dcomplex* V, int rsV, int csV,
  int       nb );

} // extern "C"

#ifdef HAVE_FLA_BSVD
namespace tmen {

inline void FlaSVD
( int k, int mU, int mV, double* d, double* e, 
  double* U, int ldu, double* V, int ldv, 
  int numAccum=32, int maxNumIts=30, int bAlg=512 )
{
    std::vector<std::complex<double>> G( (k-1)*numAccum ), H( (k-1)*numAccum );
    FLA_Bsvd_v_opd_var1
    ( k, mU, mV, numAccum, maxNumIts, d, 1, e, 1, 
      G.data(), 1, k-1, H.data(), 1, k-1, U, 1, ldu, V, 1, ldv, bAlg );
}

inline void FlaSVD
( int k, int mU, int mV, double* d, double* e, 
  std::complex<double>* U, int ldu, std::complex<double>* V, int ldv,
  int numAccum=32, int maxNumIts=30, int bAlg=512 )
{
    std::vector<std::complex<double>> G( (k-1)*numAccum ), H( (k-1)*numAccum );
    FLA_Bsvd_v_opz_var1
    ( k, mU, mV, numAccum, maxNumIts, d, 1, e, 1, 
      G.data(), 1, k-1, H.data(), 1, k-1, U, 1, ldu, V, 1, ldv, bAlg );
}

} // namespace tmen
#endif // ifdef HAVE_FLA_BSVD

#endif // ifndef TMEN_CORE_FLAME_HPP
