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

void Usage(){
    std::cout << "./testRedist <orderG> <shapeG> <oorderT> <shapeT> <distB> <distA\n";
}

typedef struct Arguments {
    ObjShape sT;
    ObjShape sG;
    TensorDistribution dB;
    TensorDistribution dA;
} Params;

void ProcessShape(int& iArg, int nArg, char* arg[], ObjShape& s) {
  int oS = atoi(arg[++iArg]);
  if (iArg + oS >= nArg) {
    std::cout << "Malformed Shape";
    Usage();
    throw ArgException();
  }

  s.resize(oS);
  for(int i = 0; i < oS; i++) {
    s[i] = atoi(arg[++iArg]);
  }
}

void ProcessTensorDistribution(int& iArg, int nArg, char* arg[], TensorDistribution& dist) {
  if (iArg + 1 >= nArg) {
    std::cout << "Malformed TensorDistribution";
    Usage();
    throw ArgException();
  }

  dist = StringToTensorDist(arg[++iArg]);
}

void ProcessInput(int nArg, char* arg[], Params& args) {
    int iArg = 0;
    if (iArg + 1 >= nArg) {
        Usage();
        throw ArgException();
    }
    ProcessShape(iArg, nArg, arg, args.sG);
    ProcessShape(iArg, nArg, arg, args.sT);
    ProcessTensorDistribution(iArg, nArg, arg, args.dB);
    ProcessTensorDistribution(iArg, nArg, arg, args.dA);
}

void PrintLocation(const char* msg, const Location& l) {
  if (mpi::CommRank(MPI_COMM_WORLD) != 0) {
    return;
  }

  std::cout << msg << " [";
  for(int i = 0; i < l.size(); i++) {
    std::cout << l[i] << " ";
  }
  std::cout << "]\n";
}

void PrintObjShape(const char* msg, const ObjShape& o) {
  if (mpi::CommRank(MPI_COMM_WORLD) != 0) {
    return;
  }

  std::cout << msg << " [";
  for(int i = 0; i < o.size(); i++) {
    std::cout << o[i] << " ";
  }
  std::cout << "]\n";
}

template<typename T>
bool Test(const DistTensor<T>& B, const DistTensor<T>& A) {
  bool test = true;

  Unsigned oA = A.Order();
  const ObjShape sA = A.Shape();
  int mA = 0;
  Location lA(oA, 0);

  while(mA != oA && ElemwiseLessThan(lA, sA)) {
    Permutation pBFromA = A.LocalPermutation().PermutationTo(B.LocalPermutation());
    Location lB = pBFromA.applyTo(lA);

    T vB = B.Get(lB);
    T vA = A.Get(lA);
    if (vB != vA) {
      // std::cout << " vB: " << vB << " <-- vA: " << vA << std::endl;
      test = false;
      break;
    }

    // Update
    lA[mA]++;
    while(lA[mA] >= sA[mA]) {
      lA[mA] = 0;
      mA++;
      if (mA == oA) {
        break;
      }
      lA[mA]++;
    }

    if (mA != oA) {
      mA = 0;
    }
  }

  // Communicate tests
  Unsigned rL = test ? 1 : 0;
  Unsigned rG;
  mpi::AllReduce(&rL, &rG, 1, mpi::LOGICAL_AND, mpi::COMM_WORLD);
  return rG == 1;
}

int nElem(const ObjShape& s) {
  int n = 1;
  for(int i = 0; i < s.size(); i++) {
    n *= s[i];
  }
  return n;
}

template<typename T>
bool TestRedist(const Grid& g, const ObjShape& sT, const TensorDistribution& dB, const TensorDistribution& dA) {
  DistTensor<T> B(sT, dB, g), A(sT, dA, g);
  MakeUniform(A);

  B.RedistFrom(A);
  return Test<T>(B, A);
}

int
main( int argc, char* argv[] )
{
  bool test = true;
  Initialize(argc, argv);
  mpi::Comm c = mpi::COMM_WORLD;

  try
  {
    Params args;
    ProcessInput(argc, argv, args);

    const Int nC = mpi::CommSize(c);
    const Int nG = nElem(args.sG);

    if (nC != nG) {
      std::cerr << "Started with incorrect number of processes\n";
      std::cerr << "nGrid: " << nG << " vs  nComm: " << nC << std::endl;
      Usage();
      throw ArgException();
    }

    const Grid g(mpi::COMM_WORLD, args.sG);
    test &= TestRedist<double>(g, args.sT, args.dB, args.dA);
    test &= TestRedist<float>(g, args.sT, args.dB, args.dA);
    test &= TestRedist<int>(g, args.sT, args.dB, args.dA);

    if (mpi::CommRank(mpi::COMM_WORLD) == 0) {
      std::cerr << "Redist "
        << TensorDistToString(args.dB)
        << " <-- "
        << TensorDistToString(args.dA)
        << " "
        << (test ? "SUCCESS" : "FAILURE")
        << "\n";
    }
  } catch( std::exception& e ) {
    test = false;
    ReportException(e);
  }
  Finalize();
  return test ? 0 : 1;
}
