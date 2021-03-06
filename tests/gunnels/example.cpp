#include "rote.hpp"
#include "preprocess.hpp"

using namespace rote;
using namespace std;

// TODO: Set BlkSizes variable to be max order of tensor objects.
//       Error checking to guard against malformed objects isn't in place.
void Example1(const mpi::Comm& comm, ObjShape gridShape, Unsigned tensorDim, Unsigned blkSize) {
  std::cout << "Running Example 1\n";
  // TensorA["abcdefghijklmnop"] = Tensor["cpgo"] * TensorB["abcdefghijklmnop"]; // Mult without contract

  // Init
  std::cout << "init\n";
  ObjShape shapeA(16, tensorDim);
  ObjShape shapeC(4, tensorDim);
  ObjShape shapeB(16, tensorDim);

  const Grid g(comm, gridShape);
  DistTensor<double> A(shapeA, "[(0),(1),(2),(3),(),(),(),(),(),(),(),(),(),(),(),()]", g);
  DistTensor<double> C(shapeC, "[(0),(1),(2),(3)]", g);
  DistTensor<double> B(shapeB, "[(0),(1),(2),(3),(),(),(),(),(),(),(),(),(),(),(),()]", g);


  // Compute
  const std::vector<Unsigned> blkSizes(16, blkSize);
  std::cout << "Computing\n";
  mpi::Barrier(g.OwningComm());

  SetAllVal(A, 1.0);
  SetAllVal(C, 2.0);
  SetAllVal(B, 3.0);
  Hadamard<double>::run(
    C, "cpgo",
    B, "abcdefghijklmnop",
    A, "abcdefghijklmnop",
    blkSizes
  );
  Print(A, "A");
}

void Example2(const mpi::Comm& comm, ObjShape gridShape, Unsigned tensorDim, Unsigned blkSize) {
  std::cout << "Running Example 2\n";
  // TensorC["abcdefghijklmnop"] = TensorD["abcdefgh123456"] * TensorE["ijklmnop615234"]; // 6 dim contraction

  // Init
  std::cout << "Init\n";
  ObjShape shapeC(16, tensorDim);
  ObjShape shapeD(14, tensorDim);
  ObjShape shapeE(14, tensorDim);

  const Grid g(comm, gridShape);

  DistTensor<double> C(shapeC, "[(0),(1),(2),(3),(),(),(),(),(),(),(),(),(),(),(),()]", g);
  DistTensor<double> D(shapeD, "[(0),(1),(2),(3),(),(),(),(),(),(),(),(),(),()]", g);
  DistTensor<double> E(shapeE, "[(0),(1),(2),(3),(),(),(),(),(),(),(),(),(),()]", g);

  // Compute
  const std::vector<Unsigned> blkSizes(16, blkSize);
  std::cout << "Computing\n";
  mpi::Barrier(g.OwningComm());

  Contract<double>::run(
    1.0,
    D, "abcdefgh123456",
    E, "ijklmnop615234",
    1.0,
    C, "abcdefghijklmnop",
    blkSizes
  );
}

void Example3(const mpi::Comm& comm, ObjShape gridShape, Unsigned tensorDim, Unsigned blkSize) {
  std::cout << "Running Example 3\n";
  // TensorF["1bc5e3ghi2kl4no6"] = TensorG["p6a1f3d5j2m4"] * TensorH["abcdefghijklmnop"]; // Probably redundant with previous example

  // Init
  std::cout << "Init\n";
  ObjShape shapeF(16, tensorDim);
  ObjShape shapeG(12, tensorDim);
  ObjShape shapeH(16, tensorDim);

  const Grid g(comm, gridShape);
  DistTensor<double> F(shapeF, "[(0),(1),(2),(3),(),(),(),(),(),(),(),(),(),(),(),()]", g);
  DistTensor<double> G(shapeG, "[(0),(1),(2),(3),(),(),(),(),(),(),(),()]", g);
  DistTensor<double> H(shapeH, "[(0),(1),(2),(3),(),(),(),(),(),(),(),(),(),(),(),()]", g);

  // Compute
  const std::vector<Unsigned> blkSizes(16, blkSize);
  std::cout << "Computing\n";
  mpi::Barrier(g.OwningComm());

  Contract<double>::run(
    1.0,
    G, "p6a1f3d5j2m4",
    H, "abcdefghijklmnop",
    1.0,
    F, "1bc5e3ghi2kl4no6",
    blkSizes
  );
}

void Example4(const mpi::Comm& comm, ObjShape gridShape, Unsigned tensorDim) {
  // Oh, and just initializing a ROTE tensor to all (1.0,0.0) ... or any specified complex value.
  // Init
  std::cout << "Init\n";
  ObjShape shapeA(16, tensorDim);

  const Grid g(comm, gridShape);
  DistTensor<double> A(shapeA, "[(0),(1),(2),(3),(),(),(),(),(),(),(),(),(),(),(),()]", g);

  // Compute
  std::cout << "Computing\n";
  SetAllVal(A, 1.0);
}

void Example5(const mpi::Comm& comm, ObjShape gridShape, Unsigned tensorDim, Unsigned blkSize) {
  std::cout << "Running Example 5\n";
  // TensorA["abcdefghijklmnop"] = Tensor["cpgo"] * TensorB["abcdefghijklmnop"]; // Mult without contract

  // Init
  std::cout << "init\n";
  ObjShape shapeA(16, tensorDim);
  ObjShape shapeC(4, tensorDim);
  ObjShape shapeB(16, tensorDim);

  const Grid g(comm, gridShape);
  DistTensor<double> A(shapeA, "[(0),(1),(2),(3),(),(),(),(),(),(),(),(),(),(),(),()]", g);
  DistTensor<double> C(shapeC, "[(0),(1),(2),(3)]", g);
  DistTensor<double> B(shapeB, "[(0),(1),(2),(3),(),(),(),(),(),(),(),(),(),(),(),()]", g);
  SetAllVal(A, 1.0);
  SetAllVal(C, 2.0);
  SetAllVal(B, 3.0);

  std::string indicesA = "abcdefghijklmnop";
  std::string indicesB = "abcdefghijklmnop";
  std::string indicesC = "cpgo";

  // Compute
  const std::vector<Unsigned> blkSizes(16, blkSize);
  std::cout << "Computing\n";
  mpi::Barrier(g.OwningComm());

  Print(A, "A");
  Print(B, "B");
  Print(C, "C");
  Hadamard<double>::run(
    C, indicesC,
    B, indicesB,
    A, indicesA,
    blkSizes
  );
  Print(A, "A");
}

void Example6(const mpi::Comm& comm, ObjShape gridShape, Unsigned tensorDim) {
  // Oh, and just initializing a ROTE tensor to all (1.0,0.0) ... or any specified complex value.
  // Init
  std::cout << "Init\n";
  ObjShape shapeA(16, tensorDim);

  const Grid g(comm, gridShape);
  DistTensor<std::complex<double>> A(shapeA, "[(0),(1),(2),(3),(),(),(),(),(),(),(),(),(),(),(),()]", g);

  // Compute
  std::cout << "Computing\n";
  std::complex<double> val(2.0,1.0);
  SetAllVal(A, val);
  Print(A, "A");
}

void Example7(const mpi::Comm& comm, ObjShape gridShape, Unsigned tensorDim, Unsigned blkSize) {
  std::cout << "Running Example 3\n";
  // TensorC["dbc"] = TensorA["ab"] * TensorB["acd"];

  // Init
  std::cout << "Init\n";
  ObjShape shapeA(2, tensorDim);
  ObjShape shapeB(3, tensorDim);
  ObjShape shapeC(3, tensorDim);

  const Grid g(comm, gridShape);
  DistTensor<double> A(shapeA, "[(0,1),(2,3)]", g);
  DistTensor<double> B(shapeB, "[(0),(1),(2,3)]", g);
  DistTensor<double> C(shapeC, "[(0),(1),(3,2)]", g);
  SetAllVal(A, 2.0);
  SetAllVal(B, 3.0);
  SetAllVal(C, 2.0);

  // Compute
  const std::vector<Unsigned> blkSizes(16, blkSize);
  std::cout << "Computing\n";
  mpi::Barrier(g.OwningComm());

  Contract<double>::run(
    1.0,
    A, "ab",
    B, "acd",
    1.0,
    C, "dbc",
    blkSizes
  );
  Print(C, "C");
}


int main(int argc, char* argv[]) {
  Initialize(argc, argv);
  mpi::Comm comm = mpi::COMM_WORLD;
  const Int commRank = mpi::CommRank(comm);
  const Int commSize = mpi::CommSize(comm);
  try {
    Params args;
    ProcessInput(argc, argv, args);

    if (commRank == 0 && commSize != args.nProcs) {
        std::cerr << "program not started with correct number of processes\n";
        std::cerr << commSize << " vs " << args.nProcs << std::endl;
        Usage();
        throw ArgException();
    }

    switch(args.exNum) {
      case 1: Example1(comm, args.gridShape, args.tensorDim, args.blkSize); break;
      case 2: Example2(comm, args.gridShape, args.tensorDim, args.blkSize); break;
      case 3: Example3(comm, args.gridShape, args.tensorDim, args.blkSize); break;
      case 4: Example4(comm, args.gridShape, args.tensorDim); break;
      case 5: Example5(comm, args.gridShape, args.tensorDim, args.blkSize); break;
      case 6: Example6(comm, args.gridShape, args.tensorDim); break;
      case 7: Example7(comm, args.gridShape, args.tensorDim, args.blkSize); break;
    }
    std::cout << "OKAY!\n";
  } catch (std::exception& e) {
      ReportException(e);
  }

  Finalize();
  return 0;
}
