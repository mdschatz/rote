/*
   Copyright (c) 2009-2013, Jack Poulson
                      2013, Jeff Hammond
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#include "rote.hpp"

namespace rote {

// Local interface
template <typename T>
void Conv2D<T>::run(
  const Tensor<T>& weights,
  const Tensor<T>& input_activations,
        Tensor<T>& output_activations
) {
  run(T(1), weights, input_activations, T(0), output_activations);
}

template <typename T>
void Conv2D<T>::run(
        T alpha,
  const Tensor<T>& weights,
  const Tensor<T>& input_activations,
        T beta,
        Tensor<T>& output_activations
) {
  for(Unsigned n = 0; n < output_activations.Dimension(0); n++) {
    for(Unsigned k = 0; k < output_activations.Dimension(1); k++) {
      for(Unsigned h = 0; h < output_activations.Dimension(2); h++) {
        for(Unsigned w = 0; w < output_activations.Dimension(3); w++) {
          Location outputLoc = {n, k, h, w};
          T val = beta * output_activations.Get(outputLoc);

          for(Unsigned fh = 0; fh < weights.Dimension(1); fh++) {
            for(Unsigned fw = 0; fw < weights.Dimension(2); fw++) {
              for(Unsigned c = 0; c < weights.Dimension(3); c++) {
                Location weightsLoc = {k, fh, fw, c};
                Location inputActivationsLoc = {n, c, h + fh, w + fw};

                val += alpha * weights.Get(weightsLoc) * input_activations.Get(inputActivationsLoc);
              }
            }
          }

          output_activations.Set(outputLoc, val);
        }
      }
    }
  }
}

#define PROTO(T) \
	template class Conv2D<T>;

PROTO(float)
PROTO(double)

} // namespace rote
