#pragma once
#ifndef ROTE_BTAS_CONV2D_HPP
#define ROTE_BTAS_CONV2D_HPP

namespace rote{

// Assumed order is:
// weights: K, Fh, Fw, C
// input_activations: N, C, H, W
// output_activations: N, K, H, W
template<typename T>
class Conv2D {
public:
	// Main interface
	static void run(
    const DistTensor<T>& weights,
    const DistTensor<T>& input_activations,
          DistTensor<T>& output_activations
	);

	static void runStatOutputActivations(
    const DistTensor<T>& weights,
    const DistTensor<T>& input_activations,
          DistTensor<T>& output_activations
  );

	static void runStatWeights(
    const DistTensor<T>& weights,
    const DistTensor<T>& input_activations,
          DistTensor<T>& output_activations
  );

	static void runStatInputActivations(
		const DistTensor<T>& weights,
		const DistTensor<T>& input_activations,
					DistTensor<T>& output_activations
	);

private:
	//Struct interface
	static void setConv2DInfo(
    const DistTensor<T>& weights,
    const DistTensor<T>& input_activations,
          DistTensor<T>& output_activations,
          Conv2DInfo& conv2DInfo
  );

  // static void runStatWeights(
  //   const DistTensor<T>& weights,
  //   const DistTensor<T>& input_activations,
  //         DistTensor<T>& output_activations
  // );

	// Internal interface
	static void run(
    const DistTensor<T>& weights,
    const DistTensor<T>& input_activations,
          DistTensor<T>& output_activations,
		const std::vector<Unsigned>& blkSizes,
		bool isStatC
	);

	// Local interface
	static void run(
					T alpha,
    const Tensor<T>& weights,
    const Tensor<T>& input_activations,
					T beta,
          Tensor<T>& output_activations
	);
};

} // namespace rote

#endif // ifndef ROTE_BTAS_CONV2D_HPP
