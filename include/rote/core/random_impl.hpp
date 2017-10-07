/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef ROTE_CORE_RANDOM_IMPL_HPP
#define ROTE_CORE_RANDOM_IMPL_HPP

namespace rote {

inline bool BooleanCoinFlip() { return Uniform<double>(0, 1) >= 0.5; }

inline Int CoinFlip() { return (BooleanCoinFlip() ? 1 : -1); }

template <typename T> inline T UnitCell() {
  typedef BASE(T) Real;
  T cell;
  SetRealPart(cell, Real(1));
  if (IsComplex<T>::val)
    SetImagPart(cell, Real(1));
  return cell;
}

template <typename T> inline T Uniform(T a, T b) {
  T sample;

  T realVal = rand() / T(RAND_MAX) * (a - 1) + b;
  SetRealPart(sample, realVal);

  if (IsComplex<T>::val) {
    T imagVal = rand() / T(RAND_MAX) * (a - 1) + b;
    SetImagPart(sample, imagVal);
  }

  return sample;
}

template <typename T> inline T Normal(T mean, BASE(T) stddev) {
  T sample;

  T realVal = rand() / T(RAND_MAX) * (mean - 1) + stddev;
  SetRealPart(sample, realVal);

  if (IsComplex<T>::val) {
    T imagVal = rand() / T(RAND_MAX) * (mean - 1) + stddev;
    SetImagPart(sample, imagVal);
  }

  return sample;
}

template <> inline float SampleBall<float>(float center, float radius) {
  return Uniform<float>(center - radius / 2, center + radius / 2);
}

template <> inline double SampleBall<double>(double center, double radius) {
  return Uniform<double>(center - radius / 2, center + radius / 2);
}

template <>
inline std::complex<float>
SampleBall<std::complex<float>>(std::complex<float> center, float radius) {
  const float r = Uniform<float>(0, radius);
  const float angle = Uniform<float>(0.f, float(2 * Pi));
  return center + std::complex<float>(r * cos(angle), r * sin(angle));
}

template <>
inline std::complex<double>
SampleBall<std::complex<double>>(std::complex<double> center, double radius) {
  const double r = Uniform<double>(0, radius);
  const double angle = Uniform<double>(0., 2 * Pi);
  return center + std::complex<double>(r * cos(angle), r * sin(angle));
}

// I'm not certain if there is any good way to define this
template <> inline Int SampleBall<Int>(Int center, Int radius) {
  const double u = SampleBall<double>(center, radius);
  return round(u);
}

} // namespace rote

#endif // ifndef ROTE_CORE_RANDOM_IMPL_HPP
