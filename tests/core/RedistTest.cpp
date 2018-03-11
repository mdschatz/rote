/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
// NOTE: It is possible to simply include "rote.hpp" instead
#include "rote.hpp"
#include <iostream>
#include <sstream>
#include <string>
#include <fstream>

using namespace rote;

void Usage(){
    std::cout << "./testRedist <cfg>\n"
      << "<cfg> format:\n"
      << "<orderG> <shapeG> <orderT> <shapeT> <distB> <distA>\n";
}

typedef struct TestParameters {
    ObjShape sT;
    ObjShape sG;
    TensorDistribution dB;
    TensorDistribution dA;
} Params;

void ProcessShape(const std::vector<std::string>& args, int& i, ObjShape& s) {
  int o = atoi(args[i++].c_str());
  // std::cout << "i: " << i << " o: " << o << " args.size(): " << args.size() << " " << args[0];
  if (i + o >= args.size()) {
    std::cout << "Malformed Shape\n";
    Usage();
    throw ArgException();
  }

  s.resize(o);
  for(int j = 0; j < o; j++) {
    s[j] = atoi(args[i++].c_str());
  }
}

void ProcessTensorDistribution(const std::vector<std::string>& args, int& i, TensorDistribution& dist) {
  if (i >= args.size()) {
    std::cout << "Malformed TensorDistribution\n";
    Usage();
    throw ArgException();
  }

  dist = StringToTensorDist(args[i++]);
}

void ProcessInput(const std::vector<std::string>& args, Params& params) {
    int i = 0;
    if (i >= args.size()) {
        Usage();
        throw ArgException();
    }

    ProcessShape(args, i, params.sG);
    ProcessShape(args, i, params.sT);
    ProcessTensorDistribution(args, i, params.dB);
    ProcessTensorDistribution(args, i, params.dA);
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

std::vector<std::string> SplitLine(const std::string& s, char delim='\t') {
  std::vector<std::string> items;

  std::stringstream ss(s);
  std::string item;
  while(std::getline(ss, item, delim)) {
    items.push_back(item);
  }
  return items;
}

int
main( int argc, char* argv[] )
{
  bool test = true;
  Initialize(argc, argv);
  mpi::Comm comm = mpi::COMM_WORLD;

  try
  {
    if (argc < 2) {
      Usage();
      throw ArgException();
    }

    std::ifstream cfg(argv[1]);
    std::string line;

    int testNum = 0;
    while(std::getline(cfg, line)) {
      testNum += 1;
      Params params;
      ProcessInput(SplitLine(line), params);

      const Int nG = nElem(params.sG);

      if (nG != mpi::CommSize(comm)) {
        std::cerr << "Started with incorrect number of processes\n";
        std::cerr << "nGrid: " << nG << " vs  nComm: " << mpi::CommSize(comm) << std::endl;
        Usage();
        throw ArgException();
      }

      const Grid g(comm, params.sG);
      test &= TestRedist<double>(g, params.sT, params.dB, params.dA);
      test &= TestRedist<float>(g, params.sT, params.dB, params.dA);
      test &= TestRedist<int>(g, params.sT, params.dB, params.dA);

      if (testNum % 100 == 0 && mpi::CommRank(comm) == 0) {
        std::cout << "Finished " << testNum << " tests\n";
      }
      if (!test && mpi::CommRank(comm) == 0) {
        std::cerr << "Redist "
          << TensorDistToString(params.dB)
          << " <-- "
          << TensorDistToString(params.dA)
          << " "
          << (test ? "SUCCESS" : "FAILURE")
          << "\n";
      }
    }
  } catch( std::exception& e ) {
    test = false;
    ReportException(e);
  }
  if(mpi::CommRank(comm) == 0) {
    std::cout << "RedistTest: " << (test ? "SUCCESS" : "FAILURE") << "\n";
  }
  Finalize();
  return 0;
}
