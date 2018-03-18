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
  ModeArray reduceModes;
  ObjShape sT;
  ObjShape sG;
  TensorDistribution dB;
  TensorDistribution dA;
} Params;

void ProcessModeArray(const std::vector<std::string>& args, int& i, ModeArray& ma) {
  // std::cout << "i: " << i;
  int o = atoi(args[i++].c_str());
  // std::cout << " o: " << o << " args: " << args.size() << "\n";
  if (i + o - 1 >= args.size()) {
    std::cout << "Malformed ModeArray\n";
    Usage();
    throw ArgException();
  }

  ma.resize(o);
  for(int j = 0; j < o; j++) {
    ma[j] = atoi(args[i++].c_str());
  }
}

void ProcessShape(const std::vector<std::string>& args, int& i, ObjShape& s) {
  int o = atoi(args[i++].c_str());
  if (i + o - 1 >= args.size()) {
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
    if (i != args.size()) {
      ProcessModeArray(args, i, params.reduceModes);
    }
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

ModeArray InsertUnitModes(const ModeArray& ma, const ModeArray& modes) {
  ModeArray ret = ma;
  ModeArray sorted = modes;
  SortVector(sorted);

  for(int i = 0; i < sorted.size(); i++) {
    ret.insert(ret.begin() + sorted[i], 0);
  }

  return ret;
}

template<typename T>
T Sum(const DistTensor<T>& A, const Location& lA, const ModeArray& modes) {
  T sum = 0;

  Unsigned o = modes.size();
  Location l = lA;
  ObjShape sA = A.Shape();

  int p = 0;
  while(p != o) {
    sum += A.Get(l);

    // Update
    l[modes[p]]++;
    while(l[modes[p]] >= sA[modes[p]]) {
      l[modes[p]] = 0;
      p++;
      if (p == o) {
        break;
      }
      l[modes[p]]++;
    }

    if (p == o) {
      break;
    }
    p = 0;
  }

  // std::cout << "sum: " << sum << std::endl;
  return sum;
}

template<typename T>
bool Test(const DistTensor<T>& B, const DistTensor<T>& A, const ModeArray& reduceModes, const T alpha, const T beta) {
  bool test = true;

  Unsigned oB = B.Order();
  const ObjShape sB = B.Shape();


  Location lB(oB, 0);
  int mB = 0;

  while(mB != oB) {
    Location lA = InsertUnitModes(lB, reduceModes);

    T check = alpha * (reduceModes.empty() ? A.Get(lA) : Sum(A, lA, reduceModes));
    T vB = B.Get(lB);
    // std::cout << "check: " << check << " vB: " << vB << std::endl;
    double epsilon = 1E-6;
    if (vB - check > epsilon) {
      test = false;
      break;
    }

    // Update
    lB[mB]++;
    while(lB[mB] >= sB[mB]) {
      lB[mB] = 0;
      mB++;
      if (mB == oB) {
        break;
      }
      lB[mB]++;
    }

    if (mB != oB) {
      mB = 0;
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
bool TestRedist(const Grid& g, const Params& params) {
  ObjShape shapeB(params.sT.size() - params.reduceModes.size(), params.sT[0]);
  DistTensor<T> B(shapeB, params.dB, g), A(params.sT, params.dA, g);
  MakeUniform(A);

  T alpha = T(2);
  T beta = T(0);
  B.RedistFrom(A, params.reduceModes, alpha, beta);
  return Test<T>(B, A, params.reduceModes, alpha, beta);
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

      if (mpi::CommRank(comm) == 0) {
        std::cout << "Starting Redist "
          << TensorDistToString(params.dB)
          << " <-- "
          << TensorDistToString(params.dA) << " ";
          if (params.reduceModes.size() > 0) {
            PrintVector(params.reduceModes, "ReduceModes");
          } else {
            std::cout << "\n";
          }
      }
      const Int nG = nElem(params.sG);

      if (nG != mpi::CommSize(comm)) {
        std::cout << "Started with incorrect number of processes\n";
        std::cout << "nGrid: " << nG << " vs  nComm: " << mpi::CommSize(comm) << std::endl;
        Usage();
        throw ArgException();
      }

      const Grid g(comm, params.sG);
      test &= TestRedist<double>(g, params);
      test &= TestRedist<float>(g, params);
      test &= TestRedist<int>(g, params);

      if (testNum % 100 == 0 && mpi::CommRank(comm) == 0) {
        std::cout << "Finished " << testNum << " tests\n";
      }
      if (!test && mpi::CommRank(comm) == 0) {
        std::cout << "Redist "
          << TensorDistToString(params.dB)
          << " <-- "
          << TensorDistToString(params.dA)
          << " "
          << (test ? "SUCCESS" : "FAILURE")
          << "\n";
      }
      if (!test) {
        throw ArgException();
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
