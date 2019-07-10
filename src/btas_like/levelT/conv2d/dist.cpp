/*
   Copyright (c) 2009-2013, Jack Poulson
                      2013, Jeff Hammond
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#include "rote.hpp"
#include "macros.hpp"

namespace rote {

// Main interface
template <typename T>
void Conv2D<T>::run(
  const DistTensor<T>& weights,
  const DistTensor<T>& input_activations,
        DistTensor<T>& output_activations
) {
  Conv2D<T>::runStatOutputActivations(
    weights,
    input_activations,
    output_activations
  );
}

// weights: [(1);();();()] (K, Fh, Fw, C)
// input_activations: [(1);();();()] (N, C, H, W)
// output_activations: [(0);(1);();()] (N, K, H, W)
template <typename T>
void Conv2D<T>::runStatOutputActivations(
  const DistTensor<T>& weights,
  const DistTensor<T>& input_activations,
        DistTensor<T>& output_activations
) {
  const int bs_C = 32;
  const int bs_Fh = 32;
  const int bs_Fw = 32;;

  PrintData(weights, "weights", false);
  PrintData(input_activations, "input_activations", false);
  PrintData(output_activations, "output_activations", false);

  const Grid& g = weights.Grid();
  DistTensor<T> a(weights.TensorDist(), g);

  T beta = T(0);

  // C
  LOCKEDPART2(
    input_activations, 1,
    weights, 3,
    bs_C,

    // Fh
    LOCKEDPART2(
      input_activations_1, 2,
      weights_1, 1,
      bs_Fh,

      // Fw
      LOCKEDPART2(
        input_activations_1_1, 3,
        weights_1_1, 2,
        bs_Fw,

        /*************************************/
        DistTensor<T> weights_update("[(1);();();()]", g);
        DistTensor<T> in_act_update("[(0);();();()]", g);

        weights_update.RedistributeFrom(weights_1_1_1);
        in_act_update.RedistributeFrom(input_activations_1_1_1);

        Conv2D<T>::run(
          T(1),
          weights_update.LockedTensor(),
          in_act_update.LockedTensor(),
          beta,
          output_activations.Tensor()
        );
        /*************************************/
        beta = T(1.0);
      ); // Fw
    ); // Fh
  ); // C
}

// Internal interface
// weights: [();();();(1)] (K, Fh, Fw, C)
// input_activations: [(0);(1);();()] (N, C, H, W)
// output_activations: [(0);();();()] (N, K, H, W)
template <typename T>
void Conv2D<T>::runStatInputActivations(
  const DistTensor<T>& weights,
  const DistTensor<T>& input_activations,
        DistTensor<T>& output_activations
) {
  const int bs_K = 32;
  const int bs_Fh = 32;
  const int bs_Fw = 32;

  PrintData(weights, "weights", false);
  PrintData(input_activations, "input_activations", false);
  PrintData(output_activations, "output_activations", false);

  const Grid& g = weights.Grid();
  DistTensor<T> a(weights.TensorDist(), g);

  // K
  PARTLOCKEDPART(
    output_activations, 1,
    weights, 0,
    bs_K,

    // Fh
    LOCKEDPART(
      weights_1, 1,
      bs_Fh,

      // Fw
      LOCKEDPART(
        weights_1_1, 2,
        bs_Fw,

        /*************************************/
        DistTensor<T> weights_update("[();();();(1)]", g);
        weights_update.RedistributeFrom(weights_1_1_1);

        ObjShape temp_shape = {
          output_activations_1.Dimension(0),
          output_activations_1.Dimension(1),
          output_activations_1.Dimension(2),
          output_activations_1.Dimension(3),
          weights.GetGridView().Dimension(3)
        };
        DistTensor<T> out_act_temp(temp_shape, "[(0);();();();(1)]", g);

        out_act_temp.ResizeTo(temp_shape);

        out_act_temp.Tensor().PopUnitMode();
        Conv2D<T>::run(
          T(1),
          weights_update.LockedTensor(),
          input_activations.LockedTensor(),
          T(0),
          out_act_temp.Tensor()
        );
        out_act_temp.Tensor().PushUnitMode();

        output_activations_1.ReduceFrom(out_act_temp, 4);
        /*************************************/

      ); // Fw
    ); // Fh
  ); // K
}

// weights: [(0);();();(1)] (K, Fh, Fw, C)
// input_activations: [();(1);();()] (N, C, H, W)
// output_activations: [();(0);();()] (N, K, H, W)
template <typename T>
void Conv2D<T>::runStatWeights(
  const DistTensor<T>& weights,
  const DistTensor<T>& input_activations,
        DistTensor<T>& output_activations
) {
  const int bs_N = 32;
  const int bs_H = 32;
  const int bs_W = 32;

  PrintData(weights, "weights", false);
  PrintData(input_activations, "input_activations", false);
  PrintData(output_activations, "output_activations", false);

  const Grid& g = weights.Grid();
  DistTensor<T> a(weights.TensorDist(), g);

  // N
  PARTLOCKEDPART(
    output_activations, 0,
    input_activations, 0,
    bs_N,

    // H
    PARTLOCKEDHALOPART(
      output_activations_1, 2,
      input_activations_1, 2, fh,
      bs_H,

      // W
      PARTLOCKEDHALOPART(
        output_activations_1_1, 3,
        input_activations_1_1, 3, fh,
        bs_W,

        /*************************************/
        DistTensor<T> in_act_update("[();(1);();()]", g);
        in_act_update.RedistributeFrom(input_activations_1_1_1);

        ObjShape temp_shape = {
          output_activations_1.Dimension(0),
          output_activations_1.Dimension(1),
          output_activations_1.Dimension(2),
          output_activations_1.Dimension(3),
          weights.GetGridView().Dimension(3)
        };
        DistTensor<T> out_act_temp(temp_shape, "[();(0);();();(1)]", g);

        out_act_temp.Tensor().PopUnitMode();
        Conv2D<T>::run(
          T(1),
          weights.LockedTensor(),
          in_act_update.LockedTensor(),
          T(0),
          out_act_temp.Tensor()
        );
        out_act_temp.Tensor().PushUnitMode();

        output_activations_1_1_1.ReduceFrom(out_act_temp, 4);
        /*************************************/
      ); // W
    ); // H
  ); // N
}

#define PROTO(T) \
  template class Conv2D<T>;

PROTO(float)
PROTO(double)

} // namespace rote
