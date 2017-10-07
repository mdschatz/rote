/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
// NOTE: It is possible to simply include "rote.hpp" instead
#include "rote.hpp"
using namespace rote;

template <typename T> void TestTensor(Int m, Int n, Int ldim) {
  if (m > ldim || ldim == 0)
    LogicError("Leading dimension must be >= m and nonzero");
  std::vector<T> buffer(ldim * n);
  for (Int j = 0; j < n; ++j)
    for (Int i = 0; i < m; ++i)
      buffer[i + j * ldim] = i + j * m;

  Tensor<T> A(m, n, buffer.data(), ldim);
  for (Int j = 0; j < n; ++j)
    for (Int i = 0; i < m; ++i)
      if (A.Get(i, j) != buffer[i + j * ldim])
        LogicError("Tensor class was not properly filled with buffer");

  const Tensor<T> B(m, n, (const T *)buffer.data(), ldim);
  for (Int j = 0; j < n; ++j)
    for (Int i = 0; i < m; ++i)
      if (B.Get(i, j) != buffer[i + j * ldim])
        LogicError("Tensor class was not properly filled with const buffer");

  const Int commRank = mpi::CommRank(mpi::COMM_WORLD);
  if (commRank == 0)
    std::cout << "passed" << std::endl;
}

int main(int argc, char *argv[]) {
  Initialize(argc, argv);
  try {
    const Int m = Input("--height", "height of matrix", 100);
    const Int n = Input("--width", "width of matrix", 100);
    const Int ldim = Input("--ldim", "leading dimension", 100);
    ProcessInput();
    PrintInputReport();

    if (mpi::WorldRank() == 0) {
      std::cout << "Testing with doubles...";
      std::cout.flush();
    }
    TestTensor<double>(m, n, ldim);

    if (mpi::WorldRank() == 0) {
      std::cout << "Testing with double-precision complex...";
      std::cout.flush();
    }
    TestTensor<Complex<double>>(m, n, ldim);
  } catch (std::exception &e) {
    ReportException(e);
  }

  Finalize();
  return 0;
}
