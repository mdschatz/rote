/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef ROTE_BTAS_HADAMARD_HPP
#define ROTE_BTAS_HADAMARD_HPP

#include "../level1/Permute.hpp"
#include "rote/util/btas_util.hpp"
#include "rote/util/vec_util.hpp"
#include "rote/util/btas_util.hpp"
#include "rote/core/view_decl.hpp"
#include "rote/io/Print.hpp"
#include "rote/core/dist_tensor_forward_decl.hpp"

namespace rote {

template<typename T>
class Hadamard {
public:
  static void run(
    const DistTensor<T>& A, const std::string& indicesA,
    const DistTensor<T>& B, const std::string& indicesB,
          DistTensor<T>& C, const std::string& indicesC,
    const std::vector<Unsigned>& blkSizes
  );

private:
  class StatA {
  public:
    static void run(
      const DistTensor<T>& A, const IndexArray& indicesA,
      const DistTensor<T>& B, const IndexArray& indicesB,
            DistTensor<T>& C, const IndexArray& indicesC,
      const std::vector<Unsigned>& blkSizes
    );
  };

  class StatC {
  public:
    static void run(
      const DistTensor<T>& A, const IndexArray& indicesA,
      const DistTensor<T>& B, const IndexArray& indicesB,
            DistTensor<T>& C, const IndexArray& indicesC,
      const std::vector<Unsigned>& blkSizes
    );
  };

  static void setHadamardInfo(
    const DistTensor<T>& A, const IndexArray& indicesA,
    const DistTensor<T>& B, const IndexArray& indicesB,
    const DistTensor<T>& C, const IndexArray& indicesC,
    const std::vector<Unsigned>& blkSizes, bool isStatC,
          BlkHadamardStatCInfo& hadamardInfo
  );

  static void runHelperPartitionBC(
    Unsigned depth, BlkHadamardStatCInfo& hadamardInfo,
    const DistTensor<T>& A, const IndexArray& indicesA,
    const DistTensor<T>& B, const IndexArray& indicesB,
          DistTensor<T>& C, const IndexArray& indicesC
  );

  static void runHelperPartitionAC(
    Unsigned depth, BlkHadamardStatCInfo& hadamardInfo,
    const DistTensor<T>& A, const IndexArray& indicesA,
    const DistTensor<T>& B, const IndexArray& indicesB,
          DistTensor<T>& C, const IndexArray& indicesC
  );

  //TODO: Merge these two with above partition methods
  static void runHelperStatAPartitionBC(
    Unsigned depth, BlkHadamardStatCInfo& hadamardInfo,
    const DistTensor<T>& A, const IndexArray& indicesA,
    const DistTensor<T>& B, const IndexArray& indicesB,
          DistTensor<T>& C, const IndexArray& indicesC
  );

  static void runHelperStatAPartitionAC(
    Unsigned depth, BlkHadamardStatCInfo& hadamardInfo,
    const DistTensor<T>& A, const IndexArray& indicesA,
    const DistTensor<T>& B, const IndexArray& indicesB,
          DistTensor<T>& C, const IndexArray& indicesC
  );

  static void run(
    const DistTensor<T>& A, const IndexArray& indicesA,
    const DistTensor<T>& B, const IndexArray& indicesB,
          DistTensor<T>& C, const IndexArray& indicesC,
    const std::vector<Unsigned>& blkSizes
  );

  static void run(
    const Tensor<T>& A, const IndexArray& indicesA,
    const Tensor<T>& B, const IndexArray& indicesB,
          Tensor<T>& C, const IndexArray& indicesC
  );
};

} // namespace rote

#endif // ifndef ROTE_BTAS_HADAMARD_HPP
