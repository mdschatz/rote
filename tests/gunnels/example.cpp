#include "rote.hpp"
#include "preprocess.hpp"

using namespace rote;
using namespace std;

void Example1(const mpi::Comm& comm, ObjShape gridShape, Unsigned tensorDim, Unsigned blkSize) {
  std::cout << "Running Example 1\n";
  // TensorA["abcdefghijklmnop"] = Tensor["cpgo"] * TensorB["abcdefghijklmnop"]; // Mult without contract

  // Init
  std::cout << "init\n";
  ObjShape shapeA(16, tensorDim);
  ObjShape shapeC(4, tensorDim);
  ObjShape shapeB(16, tensorDim);

  const Grid g(comm, gridShape);
  DistTensor<double> A(shapeA, "[(0),(1),(2),(3),(4),(5),(6),(7),(8),(9),(10),(11),(12),(13),(14),(15)]", g);
  DistTensor<double> C(shapeC, "[(0,1,2,3),(4,5,6,7),(8,9,10,11),(12,13,14,15)]", g);
  DistTensor<double> B(shapeB, "[(0),(1),(2),(3),(4),(5),(6),(7),(8),(9),(10),(11),(12),(13),(14),(15)]", g);


  // Compute
  const std::vector<Unsigned> blkSizes(4, blkSize);
  std::cout << "Contracting\n";
  mpi::Barrier(g.OwningComm());

  std::cout << "Not yet implemented!";
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
  const std::vector<Unsigned> blkSizes(6, blkSize);
  std::cout << "Contracting\n";
  mpi::Barrier(g.OwningComm());

  GenContract(1.0, D, "abcdefgh123456", E, "ijklmnop615234", 1.0, C, "abcdefghijklmnop", blkSizes);
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
  const std::vector<Unsigned> blkSizes(6, blkSize);
  std::cout << "Contracting\n";
  mpi::Barrier(g.OwningComm());

  GenContract(1.0, G, "p6a1f3d5j2m4", H, "abcdefghijklmnop", 1.0, F, "1bc5e3ghi2kl4no6", blkSizes);
}

void ExampleInit(const mpi::Comm& comm, ObjShape gridShape, Unsigned tensorDim) {
  // Oh, and just initializing a ROTE tensor to all (1.0,0.0) ... or any specified complex value.
  // Init
  std::cout << "Init\n";
  ObjShape shapeA(16, tensorDim);

  const Grid g(comm, gridShape);
  DistTensor<double> A(shapeA, "[(0),(1),(2),(3),(),(),(),(),(),(),(),(),(),(),(),()]", g);
  SetAllVal(A, 1.0);
  Print(A, "A");
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
    std::cout << "siwtching\n";
    switch(args.exNum) {
      case 1: Example1(comm, args.gridShape, args.tensorDim, args.blkSize); break;
      case 2: Example2(comm, args.gridShape, args.tensorDim, args.blkSize); break;
      case 3: Example3(comm, args.gridShape, args.tensorDim, args.blkSize); break;
      case 4: ExampleInit(comm, args.gridShape, args.tensorDim);
    }
  } catch (std::exception& e) {
      ReportException(e);
  }

  Finalize();
  return 0;
}
