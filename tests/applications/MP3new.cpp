/*
  This file is part of DxTer.
  DxTer is a prototype using the Design by Transformation (DxT)
  approach to program generation.

  Copyright (C) 2014, The University of Texas and Bryan Marker

  DxTer is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  DxTer is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with DxTer.  If not, see <http://www.gnu.org/licenses/>.
*/
// NOTE: It is possible to simply include "tensormental.hpp" instead
#include "tensormental.hpp"

#ifdef PROFILE
#ifdef BGQ
#include <spi/include/kernel/memory.h>
#endif
#endif

using namespace tmen;
using namespace std;

#define GRIDORDER 4

template <typename T>
void PrintLocalSizes(const DistTensor<T>& A)
{
  const Int commRank = mpi::CommRank( mpi::COMM_WORLD );
  if (commRank == 0) {
    for (Unsigned i = 0; i < A.Order(); ++i) {
      cout << i << " is " << A.LocalDimension(i) << endl;
    }
  }
}

void Usage(){
  std::cout << "./DistTensor <gridDim0> <gridDim1> ... \n";
  std::cout << "<gridDimK>   : dimension of mode-K of grid\n";
}

typedef struct Arguments{
  ObjShape gridShape;
  Unsigned nProcs;
  Unsigned n_o;
  Unsigned n_v;
  Unsigned blkSize;
} Params;

void ProcessInput(int argc,  char** const argv, Params& args){
  Unsigned i;
  Unsigned argCount = 0;
  if(argCount + 1 >= argc){
    std::cerr << "Missing required gridOrder argument\n";
    Usage();
    throw ArgException();
  }

  if(argCount + GRIDORDER >= argc){
    std::cerr << "Missing required grid dimensions\n";
    Usage();
    throw ArgException();
  }

  args.gridShape.resize(GRIDORDER);
  args.nProcs = 1;
  for(int i = 0; i < GRIDORDER; i++){
    int gridDim = atoi(argv[++argCount]);
    if(gridDim <= 0){
      std::cerr << "Grid dim must be greater than 0\n";
      Usage();
      throw ArgException();
    }
    args.nProcs *= gridDim;
    args.gridShape[i] = gridDim;
  }

  args.n_o = atoi(argv[++argCount]);
  args.n_v = atoi(argv[++argCount]);
  args.blkSize = atoi(argv[++argCount]);
}

template<typename T>
void Load_Tensor_Helper(ifstream& fid, Mode mode, const Location& curLoc, DistTensor<T>& A){
    Unsigned i;
    Unsigned dim = A.Dimension(mode);
    Location newCurLoc = curLoc;
    for(i = 0; i < dim; i++){
        newCurLoc[mode] = i;
        if(mode == 0){
            char* valS = new char[8];
            fid.read(valS, 8);
            double val = *reinterpret_cast<double*>(valS);
//          std::cout << "val: " << val << std::endl;
//          std::memcpy(&val, &(valS[0]), sizeof(double));
//          printf("newVal %.03f\n", val);
            A.Set(newCurLoc, val);
        }else{
            if(mode == 3)
                printf("loading mode 3 index: %d\n", i);
            Load_Tensor_Helper(fid, mode - 1, newCurLoc, A);
        }
    }
}

template<typename T>
void Load_Tensor(DistTensor<T>& A, const std::string& filename){
    printf("Loading tensor\n");
    PrintVector(A.Shape(), "of size");
    Unsigned order = A.Order();
    ifstream fid;
    fid.open(filename, std::ios::in | std::ios::binary);
    //Skip 4 bytes of Fortran
    fid.seekg(4);
    Location zeros(order, 0);
    Load_Tensor_Helper(fid, order - 1, zeros, A);
    fid.close();
}

template<typename T>
void Load_Tensor_efgh_Helper(ifstream& fid, Mode mode, const Location& curLoc, DistTensor<T>& A){
    Unsigned i;
    Unsigned dim = A.Dimension(mode);
    Location newCurLoc = curLoc;
    for(i = 0; i < dim; i++){
        if(mode == 3)
            newCurLoc[2] = i;
        else if(mode == 2)
            newCurLoc[3] = i;
        else
            newCurLoc[mode] = i;
        if(mode == 0){
            char* valS = new char[8];
            fid.read(valS, 8);
            double val = *reinterpret_cast<double*>(valS);
//          PrintVector(newCurLoc, "Setting loc");
//          std::cout << "to val: " << val << std::endl;
//          std::cout << "val: " << val << std::endl;
//          std::memcpy(&val, &(valS[0]), sizeof(double));
//          printf("newVal %.03f\n", val);
            A.Set(newCurLoc, -val);
        }else{
            Load_Tensor_efgh_Helper(fid, mode - 1, newCurLoc, A);
        }
    }
}

template<typename T>
void Load_Tensor_efgh(DistTensor<T>& A, const std::string& filename){
    printf("Loading tensor\n");
    PrintVector(A.Shape(), "of size");
    Unsigned order = A.Order();
    ifstream fid;
    fid.open(filename, std::ios::in | std::ios::binary);
    //Skip 4 bytes of Fortran
    fid.seekg(4);
    Location zeros(order, 0);
    Load_Tensor_efgh_Helper(fid, order - 1, zeros, A);
    fid.close();
}

template<typename T>
void Load_Tensor_aijb_Helper(ifstream& fid, Mode mode, const Location& curLoc, DistTensor<T>& A){
    Unsigned i;
    Unsigned dim;
    if(mode == 3)
        dim = A.Dimension(0);
    else if(mode == 2)
        dim = A.Dimension(2);
    else if(mode == 1)
        dim = A.Dimension(3);
    else if (mode == 0)
        dim = A.Dimension(1);
    Location newCurLoc = curLoc;
    for(i = 0; i < dim; i++){
        if(mode == 3)
            newCurLoc[0] = i;
        else if(mode == 2)
            newCurLoc[2] = i;
        else if(mode == 1)
            newCurLoc[3] = i;
        else if(mode == 0)
            newCurLoc[1] = i;
        if(mode == 0){
            char* valS = new char[8];
            fid.read(valS, 8);
            double val = *reinterpret_cast<double*>(valS);
//          PrintVector(newCurLoc, "Setting loc");
//          std::cout << "to val: " << val << std::endl;
//          std::cout << "val: " << val << std::endl;
//          std::memcpy(&val, &(valS[0]), sizeof(double));
//          printf("newVal %.03f\n", val);
            A.Set(newCurLoc, val);
        }else{
            Load_Tensor_aijb_Helper(fid, mode - 1, newCurLoc, A);
        }
    }
}

template<typename T>
void Load_Tensor_aijb(DistTensor<T>& A, const std::string& filename){
    printf("Loading tensor\n");
    PrintVector(A.Shape(), "of size");
    Unsigned order = A.Order();
    ifstream fid;
    fid.open(filename, std::ios::in | std::ios::binary);
    //Skip 4 bytes of Fortran
    fid.seekg(4);
    Location zeros(order, 0);
    Load_Tensor_aijb_Helper(fid, order - 1, zeros, A);
    fid.close();
}

template<typename T>
void Form_D_abij_Helper(const DistTensor<T>& epsilonA, const DistTensor<T>& epsilonB, Mode mode, const Location& loc, DistTensor<T>& D_abij){
    Unsigned i;
    Unsigned dim = D_abij.Dimension(mode);
    Location newCurLoc = loc;
    for(i = 0; i < dim; i++){
        newCurLoc[mode] = i;
        if(mode == 0){
            Location epsLoc(1);
            epsLoc[0] = newCurLoc[0];
            double e_a = epsilonA.Get(epsLoc);

            epsLoc[0] = newCurLoc[1];
            double e_b = epsilonA.Get(epsLoc);

            epsLoc[0] = newCurLoc[2];
            double e_i = epsilonB.Get(epsLoc);

            epsLoc[0] = newCurLoc[3];
            double e_j = epsilonB.Get(epsLoc);
            double val = -1.0 / (e_a + e_b - e_i - e_j);
            D_abij.Set(newCurLoc, val);
        }else{
            Form_D_abij_Helper(epsilonA, epsilonB, mode - 1, newCurLoc, D_abij);
        }
    }
}

template<typename T>
void Form_D_abij(const DistTensor<T>& epsilonA, const DistTensor<T>& epsilonB, DistTensor<T>& D_abij){
    Unsigned order = D_abij.Order();

    Location zeros(order, 0);
    Form_D_abij_Helper(epsilonA, epsilonB, order - 1, zeros, D_abij);
}

template<typename T>
void
DistTensorTest( const Grid& g, Unsigned n_o, Unsigned n_v, Unsigned blkSize )
{
#ifndef RELEASE
  CallStackEntry entry("DistTensorTest");
#endif
  Unsigned i;
  const Int commRank = mpi::CommRank( mpi::COMM_WORLD );
  ObjShape tempShape;
  TensorDistribution dist____N_D_0_1_2_3 = tmen::StringToTensorDist("[]|(0,1,2,3)");
  TensorDistribution dist__S__S__D_2__D_3 = tmen::StringToTensorDist("[(),(),(2),(3)]");
  TensorDistribution dist__S__D_1__S__D_2__D_0__D_3 = tmen::StringToTensorDist("[(),(1),(),(2),(0),(3)]");
  TensorDistribution dist__S__D_1__S__D_3__D_0__D_2 = tmen::StringToTensorDist("[(),(1),(),(3),(0),(2)]");
  TensorDistribution dist__S__D_1__D_2_0__S = tmen::StringToTensorDist("[(),(1),(2,0),()]");
  TensorDistribution dist__S__D_1__D_2_0__D_3 = tmen::StringToTensorDist("[(),(1),(2,0),(3)]");
  TensorDistribution dist__S__D_1__D_2__D_3 = tmen::StringToTensorDist("[(),(1),(2),(3)]");
  TensorDistribution dist__D_0__D_1__S__S = tmen::StringToTensorDist("[(0),(1),(),()]");
  TensorDistribution dist__D_0__D_1__S__D_2_3 = tmen::StringToTensorDist("[(0),(1),(),(2,3)]");
  TensorDistribution dist__D_0__D_1__S__D_3_2 = tmen::StringToTensorDist("[(0),(1),(),(3,2)]");
  TensorDistribution dist__D_0__D_1__S__D_3 = tmen::StringToTensorDist("[(0),(1),(),(3)]");
  TensorDistribution dist__D_0__D_1__D_2__D_3 = tmen::StringToTensorDist("[(0),(1),(2),(3)]");
  TensorDistribution dist__D_0__D_1__D_3__D_2 = tmen::StringToTensorDist("[(0),(1),(3),(2)]");
  TensorDistribution dist__D_2__S__D_0__S = tmen::StringToTensorDist("[(2),(),(0),()]");
  TensorDistribution dist__D_2__D_1__D_0__D_3 = tmen::StringToTensorDist("[(2),(1),(0),(3)]");
  TensorDistribution dist__D_3__S__D_0__S = tmen::StringToTensorDist("[(3),(),(0),()]");
  TensorDistribution dist__D_3__S__D_0_2__S = tmen::StringToTensorDist("[(3),(),(0,2),()]");
  TensorDistribution dist__D_3__D_1__D_2_0__S = tmen::StringToTensorDist("[(3),(1),(2,0),()]");
  TensorDistribution dist__D_3__D_1__D_0_2__S = tmen::StringToTensorDist("[(3),(1),(0,2),()]");
  Permutation perm;
  Permutation perm_0_1_2_3;
  perm_0_1_2_3.push_back(0);
  perm_0_1_2_3.push_back(1);
  perm_0_1_2_3.push_back(2);
  perm_0_1_2_3.push_back(3);
  Permutation perm_0_1_2_3_4_5;
  perm_0_1_2_3_4_5.push_back(0);
  perm_0_1_2_3_4_5.push_back(1);
  perm_0_1_2_3_4_5.push_back(2);
  perm_0_1_2_3_4_5.push_back(3);
  perm_0_1_2_3_4_5.push_back(4);
  perm_0_1_2_3_4_5.push_back(5);
  Permutation perm_0_1_3_2;
  perm_0_1_3_2.push_back(0);
  perm_0_1_3_2.push_back(1);
  perm_0_1_3_2.push_back(3);
  perm_0_1_3_2.push_back(2);
  Permutation perm_0_2_1_3_5_4;
  perm_0_2_1_3_5_4.push_back(0);
  perm_0_2_1_3_5_4.push_back(2);
  perm_0_2_1_3_5_4.push_back(1);
  perm_0_2_1_3_5_4.push_back(3);
  perm_0_2_1_3_5_4.push_back(5);
  perm_0_2_1_3_5_4.push_back(4);
  Permutation perm_1_3_0_2;
  perm_1_3_0_2.push_back(1);
  perm_1_3_0_2.push_back(3);
  perm_1_3_0_2.push_back(0);
  perm_1_3_0_2.push_back(2);
  Permutation perm_2_0_1_3;
  perm_2_0_1_3.push_back(2);
  perm_2_0_1_3.push_back(0);
  perm_2_0_1_3.push_back(1);
  perm_2_0_1_3.push_back(3);
  Permutation perm_2_3_0_1;
  perm_2_3_0_1.push_back(2);
  perm_2_3_0_1.push_back(3);
  perm_2_3_0_1.push_back(0);
  perm_2_3_0_1.push_back(1);
  Permutation perm_3_0_1_2;
  perm_3_0_1_2.push_back(3);
  perm_3_0_1_2.push_back(0);
  perm_3_0_1_2.push_back(1);
  perm_3_0_1_2.push_back(2);
  ModeArray modes_0;
  modes_0.push_back(0);
  ModeArray modes_1_3;
  modes_1_3.push_back(1);
  modes_1_3.push_back(3);
  ModeArray modes_0_1;
  modes_0_1.push_back(0);
  modes_0_1.push_back(1);
  ModeArray modes_0_1_2_3;
  modes_0_1_2_3.push_back(0);
  modes_0_1_2_3.push_back(1);
  modes_0_1_2_3.push_back(2);
  modes_0_1_2_3.push_back(3);
  ModeArray modes_0_1_3_2;
  modes_0_1_3_2.push_back(0);
  modes_0_1_3_2.push_back(1);
  modes_0_1_3_2.push_back(3);
  modes_0_1_3_2.push_back(2);
  ModeArray modes_0_1_2;
  modes_0_1_2.push_back(0);
  modes_0_1_2.push_back(1);
  modes_0_1_2.push_back(2);
  ModeArray modes_0_2;
  modes_0_2.push_back(0);
  modes_0_2.push_back(2);
  ModeArray modes_1;
  modes_1.push_back(1);
  ModeArray modes_2;
  modes_2.push_back(2);
  ModeArray modes_2_0;
  modes_2_0.push_back(2);
  modes_2_0.push_back(0);
  ModeArray modes_2_3;
  modes_2_3.push_back(2);
  modes_2_3.push_back(3);
  ModeArray modes_3;
  modes_3.push_back(3);
  ModeArray modes_3_0;
  modes_3_0.push_back(3);
  modes_3_0.push_back(0);
  ModeArray modes_5_4;
  modes_5_4.push_back(5);
  modes_5_4.push_back(4);
  IndexArray indices_efgh( 4 );
  indices_efgh[0] = 'e';
  indices_efgh[1] = 'f';
  indices_efgh[2] = 'g';
  indices_efgh[3] = 'h';
  IndexArray indices_efmn( 4 );
  indices_efmn[0] = 'e';
  indices_efmn[1] = 'f';
  indices_efmn[2] = 'm';
  indices_efmn[3] = 'n';
  IndexArray indices_emfnog( 6 );
  indices_emfnog[0] = 'e';
  indices_emfnog[1] = 'm';
  indices_emfnog[2] = 'f';
  indices_emfnog[3] = 'n';
  indices_emfnog[4] = 'o';
  indices_emfnog[5] = 'g';
  IndexArray indices_emog( 4 );
  indices_emog[0] = 'e';
  indices_emog[1] = 'm';
  indices_emog[2] = 'o';
  indices_emog[3] = 'g';
  IndexArray indices_ghmn( 4 );
  indices_ghmn[0] = 'g';
  indices_ghmn[1] = 'h';
  indices_ghmn[2] = 'm';
  indices_ghmn[3] = 'n';
  IndexArray indices_mnef( 4 );
  indices_mnef[0] = 'm';
  indices_mnef[1] = 'n';
  indices_mnef[2] = 'e';
  indices_mnef[3] = 'f';
  IndexArray indices_mnop( 4 );
  indices_mnop[0] = 'm';
  indices_mnop[1] = 'n';
  indices_mnop[2] = 'o';
  indices_mnop[3] = 'p';
  IndexArray indices_ogfn( 4 );
  indices_ogfn[0] = 'o';
  indices_ogfn[1] = 'g';
  indices_ogfn[2] = 'f';
  indices_ogfn[3] = 'n';
  IndexArray indices_opef( 4 );
  indices_opef[0] = 'o';
  indices_opef[1] = 'p';
  indices_opef[2] = 'e';
  indices_opef[3] = 'f';
  std::vector<ModeArray> modeArrayArray;
  std::vector<ModeArray> modeArrayArray___0;
  modeArrayArray___0.push_back(modes_0);
  std::vector<ModeArray> modeArrayArray___0___1;
  modeArrayArray___0___1.push_back(modes_0);
  modeArrayArray___0___1.push_back(modes_1);
  std::vector<ModeArray> modeArrayArray___1;
  modeArrayArray___1.push_back(modes_1);
  std::vector<ModeArray> modeArrayArray___1___3;
  modeArrayArray___1___3.push_back(modes_1);
  modeArrayArray___1___3.push_back(modes_3);
  std::vector<ModeArray> modeArrayArray___2;
  modeArrayArray___2.push_back(modes_2);
  std::vector<ModeArray> modeArrayArray___2___3;
  modeArrayArray___2___3.push_back(modes_2);
  modeArrayArray___2___3.push_back(modes_3);
  std::vector<ModeArray> modeArrayArray___3;
  modeArrayArray___3.push_back(modes_3);
      //E_MP3[D0,D1,D2,D3]
  DistTensor<double> E_MP3__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //E_MP3[] | {0,1,2,3}
  DistTensor<double> E_MP3____N_D_0_1_2_3( dist____N_D_0_1_2_3, g );
      //accum_temp[D0,D1,D2,D3]
  DistTensor<double> accum_temp__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //accum_temp_part0B[D0,D1,D2,D3]
  DistTensor<double> accum_temp_part0B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //accum_temp_part0T[D0,D1,D2,D3]
  DistTensor<double> accum_temp_part0T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //accum_temp_part0_0[D0,D1,D2,D3]
  DistTensor<double> accum_temp_part0_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //accum_temp_part0_1[D0,D1,D2,D3]
  DistTensor<double> accum_temp_part0_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //accum_temp_part0_1_part1B[D0,D1,D2,D3]
  DistTensor<double> accum_temp_part0_1_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //accum_temp_part0_1_part1T[D0,D1,D2,D3]
  DistTensor<double> accum_temp_part0_1_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //accum_temp_part0_1_part1_0[D0,D1,D2,D3]
  DistTensor<double> accum_temp_part0_1_part1_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //accum_temp_part0_1_part1_1[D0,D1,D2,D3]
  DistTensor<double> accum_temp_part0_1_part1_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //accum_temp_part0_1_part1_2[D0,D1,D2,D3]
  DistTensor<double> accum_temp_part0_1_part1_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //accum_temp_part0_1_part2B[D0,D1,D2,D3]
  DistTensor<double> accum_temp_part0_1_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //accum_temp_part0_1_part2T[D0,D1,D2,D3]
  DistTensor<double> accum_temp_part0_1_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //accum_temp_part0_1_part2_0[D0,D1,D2,D3]
  DistTensor<double> accum_temp_part0_1_part2_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //accum_temp_part0_1_part2_1[D0,D1,D2,D3]
  DistTensor<double> accum_temp_part0_1_part2_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //accum_temp_part0_1_part2_1[*,D1,*,D3,D0,D2]
  DistTensor<double> accum_temp_part0_1_part2_1__S__D_1__S__D_3__D_0__D_2( dist__S__D_1__S__D_3__D_0__D_2, g );
      //accum_temp_part0_1_part2_1[*,D1,*,D3,D0,D2]
  DistTensor<double> accum_temp_part0_1_part2_1_perm021354__S__S__D_1__D_3__D_2__D_0( dist__S__D_1__S__D_3__D_0__D_2, g );
  accum_temp_part0_1_part2_1_perm021354__S__S__D_1__D_3__D_2__D_0.SetLocalPermutation( perm_0_2_1_3_5_4 );
      //accum_temp_part0_1_part2_2[D0,D1,D2,D3]
  DistTensor<double> accum_temp_part0_1_part2_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //accum_temp_part0_2[D0,D1,D2,D3]
  DistTensor<double> accum_temp_part0_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //accum_temp[D0,D1,D2,D3]
  DistTensor<double> accum_temp_perm2301__D_2__D_3__D_0__D_1( dist__D_0__D_1__D_2__D_3, g );
  accum_temp_perm2301__D_2__D_3__D_0__D_1.SetLocalPermutation( perm_2_3_0_1 );
      //axppx2_temp[D0,D1,D2,D3]
  DistTensor<double> axppx2_temp__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //axppx2_temp_part0B[D0,D1,D2,D3]
  DistTensor<double> axppx2_temp_part0B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //axppx2_temp_part0T[D0,D1,D2,D3]
  DistTensor<double> axppx2_temp_part0T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //axppx2_temp_part0_0[D0,D1,D2,D3]
  DistTensor<double> axppx2_temp_part0_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //axppx2_temp_part0_1[D0,D1,D2,D3]
  DistTensor<double> axppx2_temp_part0_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //axppx2_temp_part0_1_part1B[D0,D1,D2,D3]
  DistTensor<double> axppx2_temp_part0_1_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //axppx2_temp_part0_1_part1T[D0,D1,D2,D3]
  DistTensor<double> axppx2_temp_part0_1_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //axppx2_temp_part0_1_part1_0[D0,D1,D2,D3]
  DistTensor<double> axppx2_temp_part0_1_part1_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //axppx2_temp_part0_1_part1_1[D0,D1,D2,D3]
  DistTensor<double> axppx2_temp_part0_1_part1_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //axppx2_temp_part0_1_part1_2[D0,D1,D2,D3]
  DistTensor<double> axppx2_temp_part0_1_part1_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //axppx2_temp_part0_2[D0,D1,D2,D3]
  DistTensor<double> axppx2_temp_part0_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //axppx2_temp_part1B[D0,D1,D2,D3]
  DistTensor<double> axppx2_temp_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //axppx2_temp_part1T[D0,D1,D2,D3]
  DistTensor<double> axppx2_temp_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //axppx2_temp_part1_0[D0,D1,D2,D3]
  DistTensor<double> axppx2_temp_part1_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //axppx2_temp_part1_1[D0,D1,D2,D3]
  DistTensor<double> axppx2_temp_part1_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //axppx2_temp_part1_1_part0B[D0,D1,D2,D3]
  DistTensor<double> axppx2_temp_part1_1_part0B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //axppx2_temp_part1_1_part0T[D0,D1,D2,D3]
  DistTensor<double> axppx2_temp_part1_1_part0T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //axppx2_temp_part1_1_part0_0[D0,D1,D2,D3]
  DistTensor<double> axppx2_temp_part1_1_part0_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //axppx2_temp_part1_1_part0_1[D0,D1,D2,D3]
  DistTensor<double> axppx2_temp_part1_1_part0_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //axppx2_temp_part1_1_part0_2[D0,D1,D2,D3]
  DistTensor<double> axppx2_temp_part1_1_part0_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //axppx2_temp_part1_2[D0,D1,D2,D3]
  DistTensor<double> axppx2_temp_part1_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //axppx2_temp[D0,D1,D2,D3]
  DistTensor<double> axppx2_temp_perm2013__D_2__D_0__D_1__D_3( dist__D_0__D_1__D_2__D_3, g );
  axppx2_temp_perm2013__D_2__D_0__D_1__D_3.SetLocalPermutation( perm_2_0_1_3 );
      //axppx3_temp[D0,D1,D2,D3]
  DistTensor<double> axppx3_temp__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //axppx3_temp_part1B[D0,D1,D2,D3]
  DistTensor<double> axppx3_temp_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //axppx3_temp_part1T[D0,D1,D2,D3]
  DistTensor<double> axppx3_temp_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //axppx3_temp_part1_0[D0,D1,D2,D3]
  DistTensor<double> axppx3_temp_part1_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //axppx3_temp_part1_1[D0,D1,D2,D3]
  DistTensor<double> axppx3_temp_part1_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //axppx3_temp_part1_1_part3B[D0,D1,D2,D3]
  DistTensor<double> axppx3_temp_part1_1_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //axppx3_temp_part1_1_part3T[D0,D1,D2,D3]
  DistTensor<double> axppx3_temp_part1_1_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //axppx3_temp_part1_1_part3_0[D0,D1,D2,D3]
  DistTensor<double> axppx3_temp_part1_1_part3_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //axppx3_temp_part1_1_part3_1[D0,D1,D2,D3]
  DistTensor<double> axppx3_temp_part1_1_part3_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //axppx3_temp_part1_1_part3_1[D2,D1,D0,D3]
  DistTensor<double> axppx3_temp_part1_1_part3_1__D_2__D_1__D_0__D_3( dist__D_2__D_1__D_0__D_3, g );
      //axppx3_temp_part1_1_part3_1[D2,*,D0,*]
  DistTensor<double> axppx3_temp_part1_1_part3_1_perm1302__S__S__D_2__D_0( dist__D_2__S__D_0__S, g );
  axppx3_temp_part1_1_part3_1_perm1302__S__S__D_2__D_0.SetLocalPermutation( perm_1_3_0_2 );
      //axppx3_temp_part1_1_part3_2[D0,D1,D2,D3]
  DistTensor<double> axppx3_temp_part1_1_part3_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //axppx3_temp_part1_2[D0,D1,D2,D3]
  DistTensor<double> axppx3_temp_part1_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //cont1_temp[D0,D1,D2,D3]
  DistTensor<double> cont1_temp__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //cont1_temp_part0B[D0,D1,D2,D3]
  DistTensor<double> cont1_temp_part0B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //cont1_temp_part0T[D0,D1,D2,D3]
  DistTensor<double> cont1_temp_part0T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //cont1_temp_part0_0[D0,D1,D2,D3]
  DistTensor<double> cont1_temp_part0_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //cont1_temp_part0_1[D0,D1,D2,D3]
  DistTensor<double> cont1_temp_part0_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //cont1_temp_part0_1_part1B[D0,D1,D2,D3]
  DistTensor<double> cont1_temp_part0_1_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //cont1_temp_part0_1_part1T[D0,D1,D2,D3]
  DistTensor<double> cont1_temp_part0_1_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //cont1_temp_part0_1_part1_0[D0,D1,D2,D3]
  DistTensor<double> cont1_temp_part0_1_part1_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //cont1_temp_part0_1_part1_1[D0,D1,D2,D3]
  DistTensor<double> cont1_temp_part0_1_part1_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //cont1_temp_part0_1_part1_1[D0,D1,D3,D2]
  DistTensor<double> cont1_temp_part0_1_part1_1__D_0__D_1__D_3__D_2( dist__D_0__D_1__D_3__D_2, g );
      //cont1_temp_part0_1_part1_2[D0,D1,D2,D3]
  DistTensor<double> cont1_temp_part0_1_part1_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //cont1_temp_part0_1_part2B[D0,D1,D2,D3]
  DistTensor<double> cont1_temp_part0_1_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //cont1_temp_part0_1_part2T[D0,D1,D2,D3]
  DistTensor<double> cont1_temp_part0_1_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //cont1_temp_part0_1_part2_0[D0,D1,D2,D3]
  DistTensor<double> cont1_temp_part0_1_part2_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //cont1_temp_part0_1_part2_1[D0,D1,D2,D3]
  DistTensor<double> cont1_temp_part0_1_part2_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //cont1_temp_part0_1_part2_1[D0,D1,*,D23]
  DistTensor<double> cont1_temp_part0_1_part2_1__D_0__D_1__S__D_2_3( dist__D_0__D_1__S__D_2_3, g );
      //cont1_temp_part0_1_part2_1[D0,D1,*,D3]
  DistTensor<double> cont1_temp_part0_1_part2_1__D_0__D_1__S__D_3( dist__D_0__D_1__S__D_3, g );
      //cont1_temp_part0_1_part2_1[D0,D1,*,D32]
  DistTensor<double> cont1_temp_part0_1_part2_1__D_0__D_1__S__D_3_2( dist__D_0__D_1__S__D_3_2, g );
      //cont1_temp_part0_1_part2_1[*,D1,*,D2,D0,D3]
  DistTensor<double> cont1_temp_part0_1_part2_1__S__D_1__S__D_2__D_0__D_3( dist__S__D_1__S__D_2__D_0__D_3, g );
      //cont1_temp_part0_1_part2_1[*,D1,*,D2,D0,D3]
  DistTensor<double> cont1_temp_part0_1_part2_1_perm021354__S__S__D_1__D_2__D_3__D_0( dist__S__D_1__S__D_2__D_0__D_3, g );
  cont1_temp_part0_1_part2_1_perm021354__S__S__D_1__D_2__D_3__D_0.SetLocalPermutation( perm_0_2_1_3_5_4 );
      //cont1_temp_part0_1_part2_2[D0,D1,D2,D3]
  DistTensor<double> cont1_temp_part0_1_part2_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //cont1_temp_part0_2[D0,D1,D2,D3]
  DistTensor<double> cont1_temp_part0_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //t_efmn[D0,D1,D2,D3]
  DistTensor<double> t_efmn__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //t_efmn_part0B[D0,D1,D2,D3]
  DistTensor<double> t_efmn_part0B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //t_efmn_part0T[D0,D1,D2,D3]
  DistTensor<double> t_efmn_part0T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //t_efmn_part0_0[D0,D1,D2,D3]
  DistTensor<double> t_efmn_part0_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //t_efmn_part0_1[D0,D1,D2,D3]
  DistTensor<double> t_efmn_part0_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //t_efmn_part0_1_part1B[D0,D1,D2,D3]
  DistTensor<double> t_efmn_part0_1_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //t_efmn_part0_1_part1T[D0,D1,D2,D3]
  DistTensor<double> t_efmn_part0_1_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //t_efmn_part0_1_part1_0[D0,D1,D2,D3]
  DistTensor<double> t_efmn_part0_1_part1_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //t_efmn_part0_1_part1_1[D0,D1,D2,D3]
  DistTensor<double> t_efmn_part0_1_part1_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //t_efmn_part0_1_part1_1[*,*,D2,D3]
  DistTensor<double> t_efmn_part0_1_part1_1__S__S__D_2__D_3( dist__S__S__D_2__D_3, g );
      //t_efmn_part0_1_part1_2[D0,D1,D2,D3]
  DistTensor<double> t_efmn_part0_1_part1_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //t_efmn_part0_2[D0,D1,D2,D3]
  DistTensor<double> t_efmn_part0_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //t_efmn_part1B[D0,D1,D2,D3]
  DistTensor<double> t_efmn_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //t_efmn_part1T[D0,D1,D2,D3]
  DistTensor<double> t_efmn_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //t_efmn_part1_0[D0,D1,D2,D3]
  DistTensor<double> t_efmn_part1_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //t_efmn_part1_1[D0,D1,D2,D3]
  DistTensor<double> t_efmn_part1_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //t_efmn_part1_1_part0B[D0,D1,D2,D3]
  DistTensor<double> t_efmn_part1_1_part0B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //t_efmn_part1_1_part0T[D0,D1,D2,D3]
  DistTensor<double> t_efmn_part1_1_part0T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //t_efmn_part1_1_part0_0[D0,D1,D2,D3]
  DistTensor<double> t_efmn_part1_1_part0_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //t_efmn_part1_1_part0_1[D0,D1,D2,D3]
  DistTensor<double> t_efmn_part1_1_part0_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //t_efmn_part1_1_part0_1[D0,D1,D3,D2]
  DistTensor<double> t_efmn_part1_1_part0_1__D_0__D_1__D_3__D_2( dist__D_0__D_1__D_3__D_2, g );
      //t_efmn_part1_1_part0_2[D0,D1,D2,D3]
  DistTensor<double> t_efmn_part1_1_part0_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //t_efmn_part1_2[D0,D1,D2,D3]
  DistTensor<double> t_efmn_part1_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //t_efmn_part2B[D0,D1,D2,D3]
  DistTensor<double> t_efmn_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //t_efmn_part2T[D0,D1,D2,D3]
  DistTensor<double> t_efmn_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //t_efmn_part2_0[D0,D1,D2,D3]
  DistTensor<double> t_efmn_part2_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //t_efmn_part2_1[D0,D1,D2,D3]
  DistTensor<double> t_efmn_part2_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //t_efmn_part2_1_part3B[D0,D1,D2,D3]
  DistTensor<double> t_efmn_part2_1_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //t_efmn_part2_1_part3T[D0,D1,D2,D3]
  DistTensor<double> t_efmn_part2_1_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //t_efmn_part2_1_part3_0[D0,D1,D2,D3]
  DistTensor<double> t_efmn_part2_1_part3_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //t_efmn_part2_1_part3_1[D0,D1,D2,D3]
  DistTensor<double> t_efmn_part2_1_part3_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //t_efmn_part2_1_part3_1[D0,D1,*,*]
  DistTensor<double> t_efmn_part2_1_part3_1_perm2301__S__S__D_0__D_1( dist__D_0__D_1__S__S, g );
  t_efmn_part2_1_part3_1_perm2301__S__S__D_0__D_1.SetLocalPermutation( perm_2_3_0_1 );
      //t_efmn_part2_1_part3_2[D0,D1,D2,D3]
  DistTensor<double> t_efmn_part2_1_part3_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //t_efmn_part2_2[D0,D1,D2,D3]
  DistTensor<double> t_efmn_part2_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //t_efmn[D0,D1,D2,D3]
  DistTensor<double> t_efmn_perm3012__D_3__D_0__D_1__D_2( dist__D_0__D_1__D_2__D_3, g );
  t_efmn_perm3012__D_3__D_0__D_1__D_2.SetLocalPermutation( perm_3_0_1_2 );
      //v2_oegm[D0,D1,D2,D3]
  DistTensor<double> v2_oegm__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //v2_oegm_part1B[D0,D1,D2,D3]
  DistTensor<double> v2_oegm_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //v2_oegm_part1T[D0,D1,D2,D3]
  DistTensor<double> v2_oegm_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //v2_oegm_part1_0[D0,D1,D2,D3]
  DistTensor<double> v2_oegm_part1_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //v2_oegm_part1_1[D0,D1,D2,D3]
  DistTensor<double> v2_oegm_part1_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //v2_oegm_part1_1_part3B[D0,D1,D2,D3]
  DistTensor<double> v2_oegm_part1_1_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //v2_oegm_part1_1_part3T[D0,D1,D2,D3]
  DistTensor<double> v2_oegm_part1_1_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //v2_oegm_part1_1_part3_0[D0,D1,D2,D3]
  DistTensor<double> v2_oegm_part1_1_part3_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //v2_oegm_part1_1_part3_1[D0,D1,D2,D3]
  DistTensor<double> v2_oegm_part1_1_part3_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //v2_oegm_part1_1_part3_1[D3,D1,D02,*]
  DistTensor<double> v2_oegm_part1_1_part3_1__D_3__D_1__D_0_2__S( dist__D_3__D_1__D_0_2__S, g );
      //v2_oegm_part1_1_part3_1[D3,D1,D20,*]
  DistTensor<double> v2_oegm_part1_1_part3_1__D_3__D_1__D_2_0__S( dist__D_3__D_1__D_2_0__S, g );
      //v2_oegm_part1_1_part3_1[D3,*,D02,*]
  DistTensor<double> v2_oegm_part1_1_part3_1__D_3__S__D_0_2__S( dist__D_3__S__D_0_2__S, g );
      //v2_oegm_part1_1_part3_1[*,D1,D20,D3]
  DistTensor<double> v2_oegm_part1_1_part3_1__S__D_1__D_2_0__D_3( dist__S__D_1__D_2_0__D_3, g );
      //v2_oegm_part1_1_part3_1[*,D1,D20,*]
  DistTensor<double> v2_oegm_part1_1_part3_1__S__D_1__D_2_0__S( dist__S__D_1__D_2_0__S, g );
      //v2_oegm_part1_1_part3_1[*,D1,D2,D3]
  DistTensor<double> v2_oegm_part1_1_part3_1__S__D_1__D_2__D_3( dist__S__D_1__D_2__D_3, g );
      //v2_oegm_part1_1_part3_1[D3,*,D0,*]
  DistTensor<double> v2_oegm_part1_1_part3_1_perm1302__S__S__D_3__D_0( dist__D_3__S__D_0__S, g );
  v2_oegm_part1_1_part3_1_perm1302__S__S__D_3__D_0.SetLocalPermutation( perm_1_3_0_2 );
      //v2_oegm_part1_1_part3_2[D0,D1,D2,D3]
  DistTensor<double> v2_oegm_part1_1_part3_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //v2_oegm_part1_2[D0,D1,D2,D3]
  DistTensor<double> v2_oegm_part1_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //v_efgh[D0,D1,D2,D3]
  DistTensor<double> v_efgh__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //v_efgh_part2B[D0,D1,D2,D3]
  DistTensor<double> v_efgh_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //v_efgh_part2T[D0,D1,D2,D3]
  DistTensor<double> v_efgh_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //v_efgh_part2_0[D0,D1,D2,D3]
  DistTensor<double> v_efgh_part2_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //v_efgh_part2_1[D0,D1,D2,D3]
  DistTensor<double> v_efgh_part2_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //v_efgh_part2_1_part3B[D0,D1,D2,D3]
  DistTensor<double> v_efgh_part2_1_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //v_efgh_part2_1_part3T[D0,D1,D2,D3]
  DistTensor<double> v_efgh_part2_1_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //v_efgh_part2_1_part3_0[D0,D1,D2,D3]
  DistTensor<double> v_efgh_part2_1_part3_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //v_efgh_part2_1_part3_1[D0,D1,D2,D3]
  DistTensor<double> v_efgh_part2_1_part3_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //v_efgh_part2_1_part3_1[D0,D1,*,*]
  DistTensor<double> v_efgh_part2_1_part3_1__D_0__D_1__S__S( dist__D_0__D_1__S__S, g );
      //v_efgh_part2_1_part3_2[D0,D1,D2,D3]
  DistTensor<double> v_efgh_part2_1_part3_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //v_efgh_part2_2[D0,D1,D2,D3]
  DistTensor<double> v_efgh_part2_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //v_oegm[D0,D1,D2,D3]
  DistTensor<double> v_oegm__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //v_opmn[D0,D1,D2,D3]
  DistTensor<double> v_opmn__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //v_opmn_part0B[D0,D1,D2,D3]
  DistTensor<double> v_opmn_part0B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //v_opmn_part0T[D0,D1,D2,D3]
  DistTensor<double> v_opmn_part0T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //v_opmn_part0_0[D0,D1,D2,D3]
  DistTensor<double> v_opmn_part0_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //v_opmn_part0_1[D0,D1,D2,D3]
  DistTensor<double> v_opmn_part0_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //v_opmn_part0_1_part1B[D0,D1,D2,D3]
  DistTensor<double> v_opmn_part0_1_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //v_opmn_part0_1_part1T[D0,D1,D2,D3]
  DistTensor<double> v_opmn_part0_1_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //v_opmn_part0_1_part1_0[D0,D1,D2,D3]
  DistTensor<double> v_opmn_part0_1_part1_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //v_opmn_part0_1_part1_1[D0,D1,D2,D3]
  DistTensor<double> v_opmn_part0_1_part1_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //v_opmn_part0_1_part1_1[*,*,D2,D3]
  DistTensor<double> v_opmn_part0_1_part1_1_perm2301__D_2__D_3__S__S( dist__S__S__D_2__D_3, g );
  v_opmn_part0_1_part1_1_perm2301__D_2__D_3__S__S.SetLocalPermutation( perm_2_3_0_1 );
      //v_opmn_part0_1_part1_2[D0,D1,D2,D3]
  DistTensor<double> v_opmn_part0_1_part1_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
      //v_opmn_part0_2[D0,D1,D2,D3]
  DistTensor<double> v_opmn_part0_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
  // t_efmn has 4 dims
  //  Starting distribution: [D0,D1,D2,D3] or _D_0__D_1__D_2__D_3
  ObjShape t_efmn__D_0__D_1__D_2__D_3_tempShape;
  t_efmn__D_0__D_1__D_2__D_3_tempShape.push_back( n_v );
  t_efmn__D_0__D_1__D_2__D_3_tempShape.push_back( n_v );
  t_efmn__D_0__D_1__D_2__D_3_tempShape.push_back( n_o );
  t_efmn__D_0__D_1__D_2__D_3_tempShape.push_back( n_o );
  t_efmn__D_0__D_1__D_2__D_3.ResizeTo( t_efmn__D_0__D_1__D_2__D_3_tempShape );
  MakeUniform( t_efmn__D_0__D_1__D_2__D_3 );
  // axppx2_temp has 4 dims
  //  Starting distribution: [D0,D1,D2,D3] or _D_0__D_1__D_2__D_3
  ObjShape axppx2_temp__D_0__D_1__D_2__D_3_tempShape;
  axppx2_temp__D_0__D_1__D_2__D_3_tempShape.push_back( n_v );
  axppx2_temp__D_0__D_1__D_2__D_3_tempShape.push_back( n_v );
  axppx2_temp__D_0__D_1__D_2__D_3_tempShape.push_back( n_o );
  axppx2_temp__D_0__D_1__D_2__D_3_tempShape.push_back( n_o );
  axppx2_temp__D_0__D_1__D_2__D_3.ResizeTo( axppx2_temp__D_0__D_1__D_2__D_3_tempShape );
  MakeUniform( axppx2_temp__D_0__D_1__D_2__D_3 );
  // v_opmn has 4 dims
  //  Starting distribution: [D0,D1,D2,D3] or _D_0__D_1__D_2__D_3
  ObjShape v_opmn__D_0__D_1__D_2__D_3_tempShape;
  v_opmn__D_0__D_1__D_2__D_3_tempShape.push_back( n_o );
  v_opmn__D_0__D_1__D_2__D_3_tempShape.push_back( n_o );
  v_opmn__D_0__D_1__D_2__D_3_tempShape.push_back( n_o );
  v_opmn__D_0__D_1__D_2__D_3_tempShape.push_back( n_o );
  v_opmn__D_0__D_1__D_2__D_3.ResizeTo( v_opmn__D_0__D_1__D_2__D_3_tempShape );
  MakeUniform( v_opmn__D_0__D_1__D_2__D_3 );
  // v_efgh has 4 dims
  //  Starting distribution: [D0,D1,D2,D3] or _D_0__D_1__D_2__D_3
  ObjShape v_efgh__D_0__D_1__D_2__D_3_tempShape;
  v_efgh__D_0__D_1__D_2__D_3_tempShape.push_back( n_v );
  v_efgh__D_0__D_1__D_2__D_3_tempShape.push_back( n_v );
  v_efgh__D_0__D_1__D_2__D_3_tempShape.push_back( n_v );
  v_efgh__D_0__D_1__D_2__D_3_tempShape.push_back( n_v );
  v_efgh__D_0__D_1__D_2__D_3.ResizeTo( v_efgh__D_0__D_1__D_2__D_3_tempShape );
  MakeUniform( v_efgh__D_0__D_1__D_2__D_3 );
  // v_oegm has 4 dims
  //  Starting distribution: [D0,D1,D2,D3] or _D_0__D_1__D_2__D_3
  ObjShape v_oegm__D_0__D_1__D_2__D_3_tempShape;
  v_oegm__D_0__D_1__D_2__D_3_tempShape.push_back( n_o );
  v_oegm__D_0__D_1__D_2__D_3_tempShape.push_back( n_v );
  v_oegm__D_0__D_1__D_2__D_3_tempShape.push_back( n_v );
  v_oegm__D_0__D_1__D_2__D_3_tempShape.push_back( n_o );
  v_oegm__D_0__D_1__D_2__D_3.ResizeTo( v_oegm__D_0__D_1__D_2__D_3_tempShape );
  MakeUniform( v_oegm__D_0__D_1__D_2__D_3 );
  // v2_oegm has 4 dims
  //  Starting distribution: [D0,D1,D2,D3] or _D_0__D_1__D_2__D_3
  ObjShape v2_oegm__D_0__D_1__D_2__D_3_tempShape;
  v2_oegm__D_0__D_1__D_2__D_3_tempShape.push_back( n_o );
  v2_oegm__D_0__D_1__D_2__D_3_tempShape.push_back( n_v );
  v2_oegm__D_0__D_1__D_2__D_3_tempShape.push_back( n_v );
  v2_oegm__D_0__D_1__D_2__D_3_tempShape.push_back( n_o );
  v2_oegm__D_0__D_1__D_2__D_3.ResizeTo( v2_oegm__D_0__D_1__D_2__D_3_tempShape );
  MakeUniform( v2_oegm__D_0__D_1__D_2__D_3 );
  // axppx3_temp has 4 dims
  //  Starting distribution: [D0,D1,D2,D3] or _D_0__D_1__D_2__D_3
  ObjShape axppx3_temp__D_0__D_1__D_2__D_3_tempShape;
  axppx3_temp__D_0__D_1__D_2__D_3_tempShape.push_back( n_o );
  axppx3_temp__D_0__D_1__D_2__D_3_tempShape.push_back( n_v );
  axppx3_temp__D_0__D_1__D_2__D_3_tempShape.push_back( n_v );
  axppx3_temp__D_0__D_1__D_2__D_3_tempShape.push_back( n_o );
  axppx3_temp__D_0__D_1__D_2__D_3.ResizeTo( axppx3_temp__D_0__D_1__D_2__D_3_tempShape );
  MakeUniform( axppx3_temp__D_0__D_1__D_2__D_3 );
  // cont1_temp has 4 dims
  //  Starting distribution: [D0,D1,D2,D3] or _D_0__D_1__D_2__D_3
  ObjShape cont1_temp__D_0__D_1__D_2__D_3_tempShape;
  cont1_temp__D_0__D_1__D_2__D_3_tempShape.push_back( n_v );
  cont1_temp__D_0__D_1__D_2__D_3_tempShape.push_back( n_v );
  cont1_temp__D_0__D_1__D_2__D_3_tempShape.push_back( n_o );
  cont1_temp__D_0__D_1__D_2__D_3_tempShape.push_back( n_o );
  cont1_temp__D_0__D_1__D_2__D_3.ResizeTo( cont1_temp__D_0__D_1__D_2__D_3_tempShape );
  MakeUniform( cont1_temp__D_0__D_1__D_2__D_3 );
  // accum_temp has 4 dims
  //  Starting distribution: [D0,D1,D2,D3] or _D_0__D_1__D_2__D_3
  ObjShape accum_temp__D_0__D_1__D_2__D_3_tempShape;
  accum_temp__D_0__D_1__D_2__D_3_tempShape.push_back( n_v );
  accum_temp__D_0__D_1__D_2__D_3_tempShape.push_back( n_v );
  accum_temp__D_0__D_1__D_2__D_3_tempShape.push_back( n_o );
  accum_temp__D_0__D_1__D_2__D_3_tempShape.push_back( n_o );
  accum_temp__D_0__D_1__D_2__D_3.ResizeTo( accum_temp__D_0__D_1__D_2__D_3_tempShape );
  MakeUniform( accum_temp__D_0__D_1__D_2__D_3 );
  // scalar input has 0 dims
  //  Starting distribution: [] | {0,1,2,3} or ___N_D_0_1_2_3
  ObjShape E_MP3____N_D_0_1_2_3_tempShape;
  E_MP3____N_D_0_1_2_3.ResizeTo( E_MP3____N_D_0_1_2_3_tempShape );
  MakeUniform( E_MP3____N_D_0_1_2_3 );

/////////////////////////////////////////////
/////////////////////////////////////////////
//**** (out of 1)
    //------------------------------------//
//******************************
//* Load tensors
//******************************
////////////////////////////////
//Performance testing
////////////////////////////////
#ifdef CORRECTNESS
  DistTensor<T> epsilonA( tmen::StringToTensorDist("[(0)]|()"), g);
  ObjShape epsilonAShape;
  epsilonAShape.push_back(n_v);
  epsilonA.ResizeTo(epsilonAShape);
  std::string epsilonAFilename = "data/ea";
  printf("loading epsilonA\n");
  Load_Tensor(epsilonA, epsilonAFilename);
  //Print(epsilonA, "eps_a");

  DistTensor<T> epsilonB( tmen::StringToTensorDist("[(0)]|()"), g);
  ObjShape epsilonBShape;
  epsilonBShape.push_back(n_o);
  epsilonB.ResizeTo(epsilonBShape);
  std::string epsilonBFilename = "data/ei";
  printf("loading epsilonB\n");
  Load_Tensor(epsilonB, epsilonBFilename);
  //Print(epsilonB, "eps_b");

  DistTensor<T> D_abij( tmen::StringToTensorDist("[(0),(1),(2),(3)]|()"), g);
  ObjShape D_abijShape;
  D_abijShape.push_back(n_v);
  D_abijShape.push_back(n_v);
  D_abijShape.push_back(n_o);
  D_abijShape.push_back(n_o);
  D_abij.ResizeTo(D_abijShape);

  DistTensor<T> V_abij( tmen::StringToTensorDist("[(0),(1),(2),(3)]|()"), g);
  V_abij.ResizeTo(D_abijShape);
  std::string v_abijFilename = "data/abij";
  printf("loading V_abij\n");
  Load_Tensor(V_abij, v_abijFilename);
  //Print(V_abij, "v_abij");

  std::string v_opmnFilename = "data/ijkl";
  printf("loading v_opmn\n");
  Load_Tensor(v_opmn__D_0__D_1__D_2__D_3, v_opmnFilename);
  //Print(v_opmn__D_0__D_1__D_2__D_3, "v_opmn");

  printf("loading 4\n");
  std::string v_oegmFilename = "data/aijb";
  printf("loading v_oegm\n");
  Load_Tensor_aijb(v_oegm__D_0__D_1__D_2__D_3, v_oegmFilename);
  //Print(v_oegm__D_0__D_1__D_2__D_3, "v_oegm");

  printf("loading 5\n");
  std::string v2_oegmFilename = "data/aibj";
  printf("loading v2_oegm\n");
  Load_Tensor_aijb(v2_oegm__D_0__D_1__D_2__D_3, v2_oegmFilename);
  //Print(v2_oegm__D_0__D_1__D_2__D_3, "v2_oegm");

  printf("loading 3\n");
  std::string v_efghFilename = "data/abcd";
  printf("loading v_efgh\n");
  Load_Tensor(v_efgh__D_0__D_1__D_2__D_3, v_efghFilename);
  //Print(v_efgh__D_0__D_1__D_2__D_3, "v_efgh");

  printf("elemScaling\n");
  Form_D_abij(epsilonA, epsilonB, D_abij);
  tmen::ElemScal(V_abij, D_abij, t_efmn__D_0__D_1__D_2__D_3);
//  Print(t_efmn__D_0__D_1__D_2__D_3, "t_efmn");
#endif
//******************************
//* Load tensors
//******************************

    double gflops;
    double startTime;
    double runTime;
    if(commRank == 0){
        std::cout << "starting\n";
#ifdef PROFILE
#ifdef BGQ
        uint64_t heap, heapavail;
        Kernel_GetMemorySize(KERNEL_MEMSIZE_HEAP, &heap);
        Kernel_GetMemorySize(KERNEL_MEMSIZE_HEAPAVAIL, &heapavail);
        printf("Allocated heap: %.2f MB, avail. heap: %.2f MB\n", (double)heap/(1024*1024),(double)heapavail/(1024*1024));
#endif
#endif
    }
    mpi::Barrier(g.OwningComm());
    startTime = mpi::Time();

    //**** (out of 1)
        //------------------------------------//

        ZAxpBy( 2.0, v_oegm__D_0__D_1__D_2__D_3, -1.0, v2_oegm__D_0__D_1__D_2__D_3, axppx3_temp__D_0__D_1__D_2__D_3 );
//        v2_oegm__D_0__D_1__D_2__D_3.EmptyData();
//        v_oegm__D_0__D_1__D_2__D_3.EmptyData();

        //------------------------------------//

//        PrintData(v2_oegm__D_0__D_1__D_2__D_3, "v2_oegm");
//        PrintData(axppx2_temp__D_0__D_1__D_2__D_3, "axppx2");
//        PrintData(axppx3_temp__D_0__D_1__D_2__D_3, "axppx3");
//        PrintData(t_efmn__D_0__D_1__D_2__D_3, "t_efmn");
//        PrintData(cont1_temp__D_0__D_1__D_2__D_3, "cont1");

    //****
    //**** (out of 1)
        //------------------------------------//

        Scal( 0.0, E_MP3____N_D_0_1_2_3 );
        Permute( t_efmn__D_0__D_1__D_2__D_3, t_efmn_perm3012__D_3__D_0__D_1__D_2 );

        //**** (out of 1)
        //**** Is real  0 shadows
            //Outputs:
            //  axppx2_temp__D_0__D_1__D_2__D_3
            //  cont1_temp__D_0__D_1__D_2__D_3
        PartitionDown(t_efmn__D_0__D_1__D_2__D_3, t_efmn_part1T__D_0__D_1__D_2__D_3, t_efmn_part1B__D_0__D_1__D_2__D_3, 1, 0);
        PartitionDown(axppx2_temp__D_0__D_1__D_2__D_3, axppx2_temp_part1T__D_0__D_1__D_2__D_3, axppx2_temp_part1B__D_0__D_1__D_2__D_3, 1, 0);
        PartitionDown(v2_oegm__D_0__D_1__D_2__D_3, v2_oegm_part1T__D_0__D_1__D_2__D_3, v2_oegm_part1B__D_0__D_1__D_2__D_3, 1, 0);
        PartitionDown(cont1_temp__D_0__D_1__D_2__D_3, cont1_temp_part0T__D_0__D_1__D_2__D_3, cont1_temp_part0B__D_0__D_1__D_2__D_3, 0, 0);
        while(axppx2_temp_part1T__D_0__D_1__D_2__D_3.Dimension(1) < axppx2_temp__D_0__D_1__D_2__D_3.Dimension(1))
        {
            RepartitionDown
            ( t_efmn_part1T__D_0__D_1__D_2__D_3,  t_efmn_part1_0__D_0__D_1__D_2__D_3,
              /**/ /**/
                   t_efmn_part1_1__D_0__D_1__D_2__D_3,
              t_efmn_part1B__D_0__D_1__D_2__D_3, t_efmn_part1_2__D_0__D_1__D_2__D_3, 1, blkSize );
            RepartitionDown
            ( axppx2_temp_part1T__D_0__D_1__D_2__D_3,  axppx2_temp_part1_0__D_0__D_1__D_2__D_3,
              /**/ /**/
                   axppx2_temp_part1_1__D_0__D_1__D_2__D_3,
              axppx2_temp_part1B__D_0__D_1__D_2__D_3, axppx2_temp_part1_2__D_0__D_1__D_2__D_3, 1, blkSize );
            RepartitionDown
            ( v2_oegm_part1T__D_0__D_1__D_2__D_3,  v2_oegm_part1_0__D_0__D_1__D_2__D_3,
              /**/ /**/
                   v2_oegm_part1_1__D_0__D_1__D_2__D_3,
              v2_oegm_part1B__D_0__D_1__D_2__D_3, v2_oegm_part1_2__D_0__D_1__D_2__D_3, 1, blkSize );
            RepartitionDown
            ( cont1_temp_part0T__D_0__D_1__D_2__D_3,  cont1_temp_part0_0__D_0__D_1__D_2__D_3,
              /**/ /**/
                   cont1_temp_part0_1__D_0__D_1__D_2__D_3,
              cont1_temp_part0B__D_0__D_1__D_2__D_3, cont1_temp_part0_2__D_0__D_1__D_2__D_3, 0, blkSize );
            //------------------------------------//

            //**** (out of 1)
            //**** Is real  0 shadows
                //Outputs:
                //  cont1_temp_part0_1__D_0__D_1__D_2__D_3
            PartitionDown(v2_oegm_part1_1__D_0__D_1__D_2__D_3, v2_oegm_part1_1_part3T__D_0__D_1__D_2__D_3, v2_oegm_part1_1_part3B__D_0__D_1__D_2__D_3, 3, 0);
            PartitionDown(cont1_temp_part0_1__D_0__D_1__D_2__D_3, cont1_temp_part0_1_part2T__D_0__D_1__D_2__D_3, cont1_temp_part0_1_part2B__D_0__D_1__D_2__D_3, 2, 0);
            while(cont1_temp_part0_1_part2T__D_0__D_1__D_2__D_3.Dimension(2) < cont1_temp_part0_1__D_0__D_1__D_2__D_3.Dimension(2))
            {
                RepartitionDown
                ( v2_oegm_part1_1_part3T__D_0__D_1__D_2__D_3,  v2_oegm_part1_1_part3_0__D_0__D_1__D_2__D_3,
                  /**/ /**/
                       v2_oegm_part1_1_part3_1__D_0__D_1__D_2__D_3,
                  v2_oegm_part1_1_part3B__D_0__D_1__D_2__D_3, v2_oegm_part1_1_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );
                RepartitionDown
                ( cont1_temp_part0_1_part2T__D_0__D_1__D_2__D_3,  cont1_temp_part0_1_part2_0__D_0__D_1__D_2__D_3,
                  /**/ /**/
                       cont1_temp_part0_1_part2_1__D_0__D_1__D_2__D_3,
                  cont1_temp_part0_1_part2B__D_0__D_1__D_2__D_3, cont1_temp_part0_1_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );
                //------------------------------------//

//                printf("ping1\n");
                tempShape = cont1_temp_part0_1_part2_1__D_0__D_1__D_2__D_3.Shape();
                tempShape.push_back( g.Shape()[0] );
                tempShape.push_back( g.Shape()[3] );
                cont1_temp_part0_1_part2_1__S__D_1__S__D_2__D_0__D_3.ResizeTo( tempShape );
                tempShape = cont1_temp_part0_1_part2_1__D_0__D_1__D_2__D_3.Shape();
                cont1_temp_part0_1_part2_1__D_0__D_1__S__D_2_3.ResizeTo( tempShape );
                   // v2_oegm_part1_1_part3_1[*,D1,D2,D3] <- v2_oegm_part1_1_part3_1[D0,D1,D2,D3]
                v2_oegm_part1_1_part3_1__S__D_1__D_2__D_3.AlignModesWith( modes_0_2, t_efmn__D_0__D_1__D_2__D_3, modes_3_0 );
                v2_oegm_part1_1_part3_1__S__D_1__D_2__D_3.AllGatherRedistFrom( v2_oegm_part1_1_part3_1__D_0__D_1__D_2__D_3, modes_0 );
//                printf("ping2\n");
                   // v2_oegm_part1_1_part3_1[*,D1,D20,D3] <- v2_oegm_part1_1_part3_1[*,D1,D2,D3]
                v2_oegm_part1_1_part3_1__S__D_1__D_2_0__D_3.AlignModesWith( modes_0_2, t_efmn__D_0__D_1__D_2__D_3, modes_3_0 );
                v2_oegm_part1_1_part3_1__S__D_1__D_2_0__D_3.LocalRedistFrom( v2_oegm_part1_1_part3_1__S__D_1__D_2__D_3 );
                v2_oegm_part1_1_part3_1__S__D_1__D_2__D_3.EmptyData();
//                printf("ping3\n");
                   // v2_oegm_part1_1_part3_1[*,D1,D20,*] <- v2_oegm_part1_1_part3_1[*,D1,D20,D3]
                v2_oegm_part1_1_part3_1__S__D_1__D_2_0__S.AlignModesWith( modes_0_2, t_efmn__D_0__D_1__D_2__D_3, modes_3_0 );
                v2_oegm_part1_1_part3_1__S__D_1__D_2_0__S.AllGatherRedistFrom( v2_oegm_part1_1_part3_1__S__D_1__D_2_0__D_3, modes_3 );
                v2_oegm_part1_1_part3_1__S__D_1__D_2_0__D_3.EmptyData();
//                printf("ping4\n");
                   // v2_oegm_part1_1_part3_1[D3,D1,D20,*] <- v2_oegm_part1_1_part3_1[*,D1,D20,*]
                v2_oegm_part1_1_part3_1__D_3__D_1__D_2_0__S.AlignModesWith( modes_0_2, t_efmn__D_0__D_1__D_2__D_3, modes_3_0 );
                v2_oegm_part1_1_part3_1__D_3__D_1__D_2_0__S.LocalRedistFrom( v2_oegm_part1_1_part3_1__S__D_1__D_2_0__S );
                v2_oegm_part1_1_part3_1__S__D_1__D_2_0__S.EmptyData();
//                printf("ping5\n");
                   // v2_oegm_part1_1_part3_1[D3,D1,D02,*] <- v2_oegm_part1_1_part3_1[D3,D1,D20,*]
                v2_oegm_part1_1_part3_1__D_3__D_1__D_0_2__S.AlignModesWith( modes_0_2, t_efmn__D_0__D_1__D_2__D_3, modes_3_0 );
                v2_oegm_part1_1_part3_1__D_3__D_1__D_0_2__S.PermutationRedistFrom( v2_oegm_part1_1_part3_1__D_3__D_1__D_2_0__S, modes_2_0 );
                v2_oegm_part1_1_part3_1__D_3__D_1__D_2_0__S.EmptyData();
//                printf("ping6\n");
                   // v2_oegm_part1_1_part3_1[D3,*,D02,*] <- v2_oegm_part1_1_part3_1[D3,D1,D02,*]
                v2_oegm_part1_1_part3_1__D_3__S__D_0_2__S.AlignModesWith( modes_0_2, t_efmn__D_0__D_1__D_2__D_3, modes_3_0 );
                v2_oegm_part1_1_part3_1__D_3__S__D_0_2__S.AllGatherRedistFrom( v2_oegm_part1_1_part3_1__D_3__D_1__D_0_2__S, modes_1 );
                v2_oegm_part1_1_part3_1__D_3__D_1__D_0_2__S.EmptyData();
//                printf("ping7\n");
                   // v2_oegm_part1_1_part3_1[D3,*,D0,*] <- v2_oegm_part1_1_part3_1[D3,*,D02,*]
                v2_oegm_part1_1_part3_1_perm1302__S__S__D_3__D_0.AlignModesWith( modes_0_2, t_efmn__D_0__D_1__D_2__D_3, modes_3_0 );
                v2_oegm_part1_1_part3_1_perm1302__S__S__D_3__D_0.AllGatherRedistFrom( v2_oegm_part1_1_part3_1__D_3__S__D_0_2__S, modes_2 );
                v2_oegm_part1_1_part3_1__D_3__S__D_0_2__S.EmptyData();

//                printf("ping7.1\n");
                Permute( cont1_temp_part0_1_part2_1__S__D_1__S__D_2__D_0__D_3, cont1_temp_part0_1_part2_1_perm021354__S__S__D_1__D_2__D_3__D_0 );
                cont1_temp_part0_1_part2_1__S__D_1__S__D_2__D_0__D_3.EmptyData();

//                PrintData(v2_oegm_part1_1_part3_1_perm1302__S__S__D_3__D_0, "v2_in");
//                PrintData(t_efmn_perm3012__D_3__D_0__D_1__D_2, "t_in");
//                PrintData(cont1_temp_part0_1_part2_1_perm021354__S__S__D_1__D_2__D_3__D_0, "cont_in");
//                printf("ping7.2\n");
                   // 1.0 * v2_oegm_part1_1_part3_1[D3,*,D0,*]_emog * t_efmn[D0,D1,D2,D3]_ogfn + 0.0 * cont1_temp_part0_1_part2_1[*,D1,*,D2,D0,D3]_emfnog
                LocalContract(1.0, v2_oegm_part1_1_part3_1_perm1302__S__S__D_3__D_0.LockedTensor(), indices_emog, false,
                    t_efmn_perm3012__D_3__D_0__D_1__D_2.LockedTensor(), indices_ogfn, false,
                    0.0, cont1_temp_part0_1_part2_1_perm021354__S__S__D_1__D_2__D_3__D_0.Tensor(), indices_emfnog, false);
                   // cont1_temp_part0_1_part2_1[D0,D1,*,D23] <- cont1_temp_part0_1_part2_1[*,D1,*,D2,D0,D3] (with SumScatter on (D0)(D3))
//                MakeUniform(cont1_temp_part0_1_part2_1__D_0__D_1__S__D_2_3);

//                printf("ping7.3\n");
                cont1_temp_part0_1_part2_1__D_0__D_1__S__D_2_3.AlignModesWith( modes_0_1_2_3, cont1_temp_part0_1_part2_1_perm021354__S__S__D_1__D_2__D_3__D_0, modes_0_1_2_3 );
                cont1_temp_part0_1_part2_1__D_0__D_1__S__D_2_3.ReduceScatterRedistFrom( cont1_temp_part0_1_part2_1_perm021354__S__S__D_1__D_2__D_3__D_0, modes_5_4 );
                cont1_temp_part0_1_part2_1_perm021354__S__S__D_1__D_2__D_3__D_0.EmptyData();
//                Print(cont1_temp_part0_1_part2_1__D_0__D_1__S__D_2_3, "cont_out0");
                v2_oegm_part1_1_part3_1_perm1302__S__S__D_3__D_0.EmptyData();
//                printf("ping7.4\n");
                   // cont1_temp_part0_1_part2_1[D0,D1,*,D32] <- cont1_temp_part0_1_part2_1[D0,D1,*,D23]
//                PrintData(cont1_temp_part0_1_part2_1__D_0__D_1__S__D_3_2, "out_data_before");
                cont1_temp_part0_1_part2_1__D_0__D_1__S__D_3_2.AlignModesWith( modes_0_1_2_3, cont1_temp_part0_1_part2_1__D_0__D_1__D_2__D_3, modes_0_1_2_3 );
//                PrintData(cont1_temp_part0_1_part2_1__D_0__D_1__S__D_3_2, "out_data");
//                PrintData(cont1_temp_part0_1_part2_1__D_0__D_1__S__D_2_3, "in_data");
//                Print(cont1_temp_part0_1_part2_1__D_0__D_1__S__D_3_2, "out");
                cont1_temp_part0_1_part2_1__D_0__D_1__S__D_3_2.PermutationRedistFrom( cont1_temp_part0_1_part2_1__D_0__D_1__S__D_2_3, modes_2_3 );
//                Print(cont1_temp_part0_1_part2_1__D_0__D_1__S__D_2_3, "in");
                cont1_temp_part0_1_part2_1__D_0__D_1__S__D_2_3.EmptyData();
//                printf("ping8\n");
//                Print(cont1_temp_part0_1_part2_1__D_0__D_1__S__D_3_2, "cont_out1");
                   // cont1_temp_part0_1_part2_1[D0,D1,*,D3] <- cont1_temp_part0_1_part2_1[D0,D1,*,D32]
                cont1_temp_part0_1_part2_1__D_0__D_1__S__D_3.AlignModesWith( modes_0_1_2_3, cont1_temp_part0_1_part2_1__D_0__D_1__D_2__D_3, modes_0_1_2_3 );
                cont1_temp_part0_1_part2_1__D_0__D_1__S__D_3.AllGatherRedistFrom( cont1_temp_part0_1_part2_1__D_0__D_1__S__D_3_2, modes_2 );
                cont1_temp_part0_1_part2_1__D_0__D_1__S__D_3_2.EmptyData();
//                printf("ping9\n");
//                Print(cont1_temp_part0_1_part2_1__D_0__D_1__S__D_3, "cont_out2");
                   // cont1_temp_part0_1_part2_1[D0,D1,D2,D3] <- cont1_temp_part0_1_part2_1[D0,D1,*,D3]
//                cont1_temp_part0_1_part2_1__D_0__D_1__D_2__D_3.AlignModesWith( modes_0_1_2_3, cont1_temp_part0_1_part2_1__D_0__D_1__D_2__D_3, modes_0_1_2_3 );

                cont1_temp_part0_1_part2_1__D_0__D_1__D_2__D_3.LocalRedistFrom( cont1_temp_part0_1_part2_1__D_0__D_1__S__D_3 );
                cont1_temp_part0_1_part2_1__D_0__D_1__S__D_3.EmptyData();
//                printf("ping10\n");
//                Print(cont1_temp_part0_1_part2_1__D_0__D_1__D_2__D_3, "cont_out3");

                //------------------------------------//
                SlidePartitionDown
                ( v2_oegm_part1_1_part3T__D_0__D_1__D_2__D_3,  v2_oegm_part1_1_part3_0__D_0__D_1__D_2__D_3,
                       v2_oegm_part1_1_part3_1__D_0__D_1__D_2__D_3,
                  /**/ /**/
                  v2_oegm_part1_1_part3B__D_0__D_1__D_2__D_3, v2_oegm_part1_1_part3_2__D_0__D_1__D_2__D_3, 3 );

                SlidePartitionDown
                ( cont1_temp_part0_1_part2T__D_0__D_1__D_2__D_3,  cont1_temp_part0_1_part2_0__D_0__D_1__D_2__D_3,
                       cont1_temp_part0_1_part2_1__D_0__D_1__D_2__D_3,
                  /**/ /**/
                  cont1_temp_part0_1_part2B__D_0__D_1__D_2__D_3, cont1_temp_part0_1_part2_2__D_0__D_1__D_2__D_3, 2 );

            }
//            printf("cont1\n");
            //****
            //**** (out of 1)
            //**** Is real  0 shadows
                //Outputs:
                //  axppx2_temp_part1_1__D_0__D_1__D_2__D_3
            PartitionDown(t_efmn_part1_1__D_0__D_1__D_2__D_3, t_efmn_part1_1_part0T__D_0__D_1__D_2__D_3, t_efmn_part1_1_part0B__D_0__D_1__D_2__D_3, 0, 0);
            PartitionDown(t_efmn_part1_1__D_0__D_1__D_2__D_3, t_efmn_part1_1_part0T__D_0__D_1__D_2__D_3, t_efmn_part1_1_part0B__D_0__D_1__D_2__D_3, 0, 0);
            PartitionDown(axppx2_temp_part1_1__D_0__D_1__D_2__D_3, axppx2_temp_part1_1_part0T__D_0__D_1__D_2__D_3, axppx2_temp_part1_1_part0B__D_0__D_1__D_2__D_3, 0, 0);
            while(axppx2_temp_part1_1_part0T__D_0__D_1__D_2__D_3.Dimension(0) < axppx2_temp_part1_1__D_0__D_1__D_2__D_3.Dimension(0))
            {
                RepartitionDown
                ( t_efmn_part1_1_part0T__D_0__D_1__D_2__D_3,  t_efmn_part1_1_part0_0__D_0__D_1__D_2__D_3,
                  /**/ /**/
                       t_efmn_part1_1_part0_1__D_0__D_1__D_2__D_3,
                  t_efmn_part1_1_part0B__D_0__D_1__D_2__D_3, t_efmn_part1_1_part0_2__D_0__D_1__D_2__D_3, 0, blkSize );
                RepartitionDown
                ( t_efmn_part1_1_part0T__D_0__D_1__D_2__D_3,  t_efmn_part1_1_part0_0__D_0__D_1__D_2__D_3,
                  /**/ /**/
                       t_efmn_part1_1_part0_1__D_0__D_1__D_2__D_3,
                  t_efmn_part1_1_part0B__D_0__D_1__D_2__D_3, t_efmn_part1_1_part0_2__D_0__D_1__D_2__D_3, 0, blkSize );
                RepartitionDown
                ( axppx2_temp_part1_1_part0T__D_0__D_1__D_2__D_3,  axppx2_temp_part1_1_part0_0__D_0__D_1__D_2__D_3,
                  /**/ /**/
                       axppx2_temp_part1_1_part0_1__D_0__D_1__D_2__D_3,
                  axppx2_temp_part1_1_part0B__D_0__D_1__D_2__D_3, axppx2_temp_part1_1_part0_2__D_0__D_1__D_2__D_3, 0, blkSize );
                //------------------------------------//

//                printf("ping3\n");
                   // t_efmn_part1_1_part0_1[D0,D1,D3,D2] <- t_efmn_part1_1_part0_1[D0,D1,D2,D3]
//                printf("here too\n");
//                PrintData(t_efmn_part1_1_part0_1__D_0__D_1__D_2__D_3, "t_efmn_part1_1_part0_1__D_0__D_1__D_2__D_3");
//                PrintData(t_efmn_part1_1_part0_1__D_0__D_1__D_3__D_2, "t_efmn_part1_1_part0_1__D_0__D_1__D_3__D_2");
                t_efmn_part1_1_part0_1__D_0__D_1__D_3__D_2.AlignModesWith( modes_0_1_2_3, t_efmn_part1_1_part0_1__D_0__D_1__D_2__D_3, modes_0_1_3_2 );
//                PrintData(t_efmn_part1_1_part0_1__D_0__D_1__D_3__D_2, "t_efmn_part1_1_part0_1__D_0__D_1__D_3__D_2_after");
                t_efmn_part1_1_part0_1__D_0__D_1__D_3__D_2.AllToAllRedistFrom( t_efmn_part1_1_part0_1__D_0__D_1__D_2__D_3, modes_2_3 );
//                printf("mhm\n");
//                printf("ping3.1\n");
//                PrintData(t_efmn_part1_1_part0_1__D_0__D_1__D_2__D_3, "t_efmn_part1_1_part0_1__D_0__D_1__D_2__D_3_1");
//                PrintData(t_efmn_part1_1_part0_1__D_0__D_1__D_3__D_2, "t_efmn_part1_1_part0_1__D_0__D_1__D_3__D_2_2");
//                PrintData(axppx2_temp_part1_1_part0_1__D_0__D_1__D_2__D_3, "axppx2_temp_part1_1_part0_1__D_0__D_1__D_2__D_3");
                YAxpPx( 2.0, t_efmn_part1_1_part0_1__D_0__D_1__D_2__D_3, -1.0, t_efmn_part1_1_part0_1__D_0__D_1__D_3__D_2, perm_0_1_3_2, axppx2_temp_part1_1_part0_1__D_0__D_1__D_2__D_3 );
                t_efmn_part1_1_part0_1__D_0__D_1__D_3__D_2.EmptyData();

//                printf("ping3.2\n");
                //------------------------------------//
                SlidePartitionDown
                ( t_efmn_part1_1_part0T__D_0__D_1__D_2__D_3,  t_efmn_part1_1_part0_0__D_0__D_1__D_2__D_3,
                       t_efmn_part1_1_part0_1__D_0__D_1__D_2__D_3,
                  /**/ /**/
                  t_efmn_part1_1_part0B__D_0__D_1__D_2__D_3, t_efmn_part1_1_part0_2__D_0__D_1__D_2__D_3, 0 );
                SlidePartitionDown
                ( t_efmn_part1_1_part0T__D_0__D_1__D_2__D_3,  t_efmn_part1_1_part0_0__D_0__D_1__D_2__D_3,
                       t_efmn_part1_1_part0_1__D_0__D_1__D_2__D_3,
                  /**/ /**/
                  t_efmn_part1_1_part0B__D_0__D_1__D_2__D_3, t_efmn_part1_1_part0_2__D_0__D_1__D_2__D_3, 0 );
                SlidePartitionDown
                ( axppx2_temp_part1_1_part0T__D_0__D_1__D_2__D_3,  axppx2_temp_part1_1_part0_0__D_0__D_1__D_2__D_3,
                       axppx2_temp_part1_1_part0_1__D_0__D_1__D_2__D_3,
                  /**/ /**/
                  axppx2_temp_part1_1_part0B__D_0__D_1__D_2__D_3, axppx2_temp_part1_1_part0_2__D_0__D_1__D_2__D_3, 0 );
//                printf("3.1.0\n");

            }
            //****

            //------------------------------------//
            SlidePartitionDown
            ( t_efmn_part1T__D_0__D_1__D_2__D_3,  t_efmn_part1_0__D_0__D_1__D_2__D_3,
                   t_efmn_part1_1__D_0__D_1__D_2__D_3,
              /**/ /**/
              t_efmn_part1B__D_0__D_1__D_2__D_3, t_efmn_part1_2__D_0__D_1__D_2__D_3, 1 );
            SlidePartitionDown
            ( axppx2_temp_part1T__D_0__D_1__D_2__D_3,  axppx2_temp_part1_0__D_0__D_1__D_2__D_3,
                   axppx2_temp_part1_1__D_0__D_1__D_2__D_3,
              /**/ /**/
              axppx2_temp_part1B__D_0__D_1__D_2__D_3, axppx2_temp_part1_2__D_0__D_1__D_2__D_3, 1 );
            SlidePartitionDown
            ( v2_oegm_part1T__D_0__D_1__D_2__D_3,  v2_oegm_part1_0__D_0__D_1__D_2__D_3,
                   v2_oegm_part1_1__D_0__D_1__D_2__D_3,
              /**/ /**/
              v2_oegm_part1B__D_0__D_1__D_2__D_3, v2_oegm_part1_2__D_0__D_1__D_2__D_3, 1 );
            SlidePartitionDown
            ( cont1_temp_part0T__D_0__D_1__D_2__D_3,  cont1_temp_part0_0__D_0__D_1__D_2__D_3,
                   cont1_temp_part0_1__D_0__D_1__D_2__D_3,
              /**/ /**/
              cont1_temp_part0B__D_0__D_1__D_2__D_3, cont1_temp_part0_2__D_0__D_1__D_2__D_3, 0 );

        }
//        printf("axppx2\n");
//        printf("3.1.1\n");
        //****
        Permute( axppx2_temp__D_0__D_1__D_2__D_3, axppx2_temp_perm2013__D_2__D_0__D_1__D_3 );
//        Print(cont1_temp__D_0__D_1__D_2__D_3, "cont1");

        //**** (out of 1)
        //**** Is real  0 shadows
            //Outputs:
            //  accum_temp__D_0__D_1__D_2__D_3
        PartitionDown(axppx3_temp__D_0__D_1__D_2__D_3, axppx3_temp_part1T__D_0__D_1__D_2__D_3, axppx3_temp_part1B__D_0__D_1__D_2__D_3, 1, 0);
        PartitionDown(cont1_temp__D_0__D_1__D_2__D_3, cont1_temp_part0T__D_0__D_1__D_2__D_3, cont1_temp_part0B__D_0__D_1__D_2__D_3, 0, 0);
        PartitionDown(accum_temp__D_0__D_1__D_2__D_3, accum_temp_part0T__D_0__D_1__D_2__D_3, accum_temp_part0B__D_0__D_1__D_2__D_3, 0, 0);
        while(accum_temp_part0T__D_0__D_1__D_2__D_3.Dimension(0) < accum_temp__D_0__D_1__D_2__D_3.Dimension(0))
        {
            RepartitionDown
            ( axppx3_temp_part1T__D_0__D_1__D_2__D_3,  axppx3_temp_part1_0__D_0__D_1__D_2__D_3,
              /**/ /**/
                   axppx3_temp_part1_1__D_0__D_1__D_2__D_3,
              axppx3_temp_part1B__D_0__D_1__D_2__D_3, axppx3_temp_part1_2__D_0__D_1__D_2__D_3, 1, blkSize );
            RepartitionDown
            ( cont1_temp_part0T__D_0__D_1__D_2__D_3,  cont1_temp_part0_0__D_0__D_1__D_2__D_3,
              /**/ /**/
                   cont1_temp_part0_1__D_0__D_1__D_2__D_3,
              cont1_temp_part0B__D_0__D_1__D_2__D_3, cont1_temp_part0_2__D_0__D_1__D_2__D_3, 0, blkSize );
            RepartitionDown
            ( accum_temp_part0T__D_0__D_1__D_2__D_3,  accum_temp_part0_0__D_0__D_1__D_2__D_3,
              /**/ /**/
                   accum_temp_part0_1__D_0__D_1__D_2__D_3,
              accum_temp_part0B__D_0__D_1__D_2__D_3, accum_temp_part0_2__D_0__D_1__D_2__D_3, 0, blkSize );
            //------------------------------------//

            //**** (out of 1)
            //**** Is real  0 shadows
                //Outputs:
                //  accum_temp_part0_1__D_0__D_1__D_2__D_3
            PartitionDown(cont1_temp_part0_1__D_0__D_1__D_2__D_3, cont1_temp_part0_1_part1T__D_0__D_1__D_2__D_3, cont1_temp_part0_1_part1B__D_0__D_1__D_2__D_3, 1, 0);
            PartitionDown(cont1_temp_part0_1__D_0__D_1__D_2__D_3, cont1_temp_part0_1_part1T__D_0__D_1__D_2__D_3, cont1_temp_part0_1_part1B__D_0__D_1__D_2__D_3, 1, 0);
            PartitionDown(accum_temp_part0_1__D_0__D_1__D_2__D_3, accum_temp_part0_1_part1T__D_0__D_1__D_2__D_3, accum_temp_part0_1_part1B__D_0__D_1__D_2__D_3, 1, 0);
            while(accum_temp_part0_1_part1T__D_0__D_1__D_2__D_3.Dimension(1) < accum_temp_part0_1__D_0__D_1__D_2__D_3.Dimension(1))
            {
                RepartitionDown
                ( cont1_temp_part0_1_part1T__D_0__D_1__D_2__D_3,  cont1_temp_part0_1_part1_0__D_0__D_1__D_2__D_3,
                  /**/ /**/
                       cont1_temp_part0_1_part1_1__D_0__D_1__D_2__D_3,
                  cont1_temp_part0_1_part1B__D_0__D_1__D_2__D_3, cont1_temp_part0_1_part1_2__D_0__D_1__D_2__D_3, 1, blkSize );
                RepartitionDown
                ( cont1_temp_part0_1_part1T__D_0__D_1__D_2__D_3,  cont1_temp_part0_1_part1_0__D_0__D_1__D_2__D_3,
                  /**/ /**/
                       cont1_temp_part0_1_part1_1__D_0__D_1__D_2__D_3,
                  cont1_temp_part0_1_part1B__D_0__D_1__D_2__D_3, cont1_temp_part0_1_part1_2__D_0__D_1__D_2__D_3, 1, blkSize );
                RepartitionDown
                ( accum_temp_part0_1_part1T__D_0__D_1__D_2__D_3,  accum_temp_part0_1_part1_0__D_0__D_1__D_2__D_3,
                  /**/ /**/
                       accum_temp_part0_1_part1_1__D_0__D_1__D_2__D_3,
                  accum_temp_part0_1_part1B__D_0__D_1__D_2__D_3, accum_temp_part0_1_part1_2__D_0__D_1__D_2__D_3, 1, blkSize );
                //------------------------------------//

//                printf("ping4\n");
                   // cont1_temp_part0_1_part1_1[D0,D1,D3,D2] <- cont1_temp_part0_1_part1_1[D0,D1,D2,D3]
                cont1_temp_part0_1_part1_1__D_0__D_1__D_3__D_2.AlignModesWith( modes_0_1_2_3, cont1_temp_part0_1_part1_1__D_0__D_1__D_2__D_3, modes_0_1_3_2 );
                cont1_temp_part0_1_part1_1__D_0__D_1__D_3__D_2.AllToAllRedistFrom( cont1_temp_part0_1_part1_1__D_0__D_1__D_2__D_3, modes_2_3 );
                YAxpPx( 0.5, cont1_temp_part0_1_part1_1__D_0__D_1__D_2__D_3, 1.0, cont1_temp_part0_1_part1_1__D_0__D_1__D_3__D_2, perm_0_1_3_2, accum_temp_part0_1_part1_1__D_0__D_1__D_2__D_3 );
                cont1_temp_part0_1_part1_1__D_0__D_1__D_3__D_2.EmptyData();

                //------------------------------------//
                SlidePartitionDown
                ( cont1_temp_part0_1_part1T__D_0__D_1__D_2__D_3,  cont1_temp_part0_1_part1_0__D_0__D_1__D_2__D_3,
                       cont1_temp_part0_1_part1_1__D_0__D_1__D_2__D_3,
                  /**/ /**/
                  cont1_temp_part0_1_part1B__D_0__D_1__D_2__D_3, cont1_temp_part0_1_part1_2__D_0__D_1__D_2__D_3, 1 );
                SlidePartitionDown
                ( cont1_temp_part0_1_part1T__D_0__D_1__D_2__D_3,  cont1_temp_part0_1_part1_0__D_0__D_1__D_2__D_3,
                       cont1_temp_part0_1_part1_1__D_0__D_1__D_2__D_3,
                  /**/ /**/
                  cont1_temp_part0_1_part1B__D_0__D_1__D_2__D_3, cont1_temp_part0_1_part1_2__D_0__D_1__D_2__D_3, 1 );
                SlidePartitionDown
                ( accum_temp_part0_1_part1T__D_0__D_1__D_2__D_3,  accum_temp_part0_1_part1_0__D_0__D_1__D_2__D_3,
                       accum_temp_part0_1_part1_1__D_0__D_1__D_2__D_3,
                  /**/ /**/
                  accum_temp_part0_1_part1B__D_0__D_1__D_2__D_3, accum_temp_part0_1_part1_2__D_0__D_1__D_2__D_3, 1 );

            }
            //****
            //**** (out of 1)
            //**** Is real  0 shadows
                //Outputs:
                //  accum_temp_part0_1__D_0__D_1__D_2__D_3
            PartitionDown(axppx3_temp_part1_1__D_0__D_1__D_2__D_3, axppx3_temp_part1_1_part3T__D_0__D_1__D_2__D_3, axppx3_temp_part1_1_part3B__D_0__D_1__D_2__D_3, 3, 0);
            PartitionDown(accum_temp_part0_1__D_0__D_1__D_2__D_3, accum_temp_part0_1_part2T__D_0__D_1__D_2__D_3, accum_temp_part0_1_part2B__D_0__D_1__D_2__D_3, 2, 0);
            while(accum_temp_part0_1_part2T__D_0__D_1__D_2__D_3.Dimension(2) < accum_temp_part0_1__D_0__D_1__D_2__D_3.Dimension(2))
            {
                RepartitionDown
                ( axppx3_temp_part1_1_part3T__D_0__D_1__D_2__D_3,  axppx3_temp_part1_1_part3_0__D_0__D_1__D_2__D_3,
                  /**/ /**/
                       axppx3_temp_part1_1_part3_1__D_0__D_1__D_2__D_3,
                  axppx3_temp_part1_1_part3B__D_0__D_1__D_2__D_3, axppx3_temp_part1_1_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );
                RepartitionDown
                ( accum_temp_part0_1_part2T__D_0__D_1__D_2__D_3,  accum_temp_part0_1_part2_0__D_0__D_1__D_2__D_3,
                  /**/ /**/
                       accum_temp_part0_1_part2_1__D_0__D_1__D_2__D_3,
                  accum_temp_part0_1_part2B__D_0__D_1__D_2__D_3, accum_temp_part0_1_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );
                //------------------------------------//

                tempShape = accum_temp_part0_1_part2_1__D_0__D_1__D_2__D_3.Shape();
                tempShape.push_back( g.Shape()[0] );
                tempShape.push_back( g.Shape()[2] );
                accum_temp_part0_1_part2_1__S__D_1__S__D_3__D_0__D_2.ResizeTo( tempShape );
                   // axppx3_temp_part1_1_part3_1[D2,D1,D0,D3] <- axppx3_temp_part1_1_part3_1[D0,D1,D2,D3]
//                printf("here?\n");
                axppx3_temp_part1_1_part3_1__D_2__D_1__D_0__D_3.AlignModesWith( modes_0_2, axppx2_temp__D_0__D_1__D_2__D_3, modes_2_0 );
//                PrintData(axppx3_temp_part1_1_part3_1__D_0__D_1__D_2__D_3, "axppx3_temp_part1_1_part3_1__D_0__D_1__D_2__D_3");
//                PrintData(axppx3_temp_part1_1_part3_1__D_2__D_1__D_0__D_3, "axppx3_temp_part1_1_part3_1__D_2__D_1__D_0__D_3");
                axppx3_temp_part1_1_part3_1__D_2__D_1__D_0__D_3.AllToAllRedistFrom( axppx3_temp_part1_1_part3_1__D_0__D_1__D_2__D_3, modes_0_1_2 );
//                printf("yup\n");
                   // axppx3_temp_part1_1_part3_1[D2,*,D0,*] <- axppx3_temp_part1_1_part3_1[D2,D1,D0,D3]
                axppx3_temp_part1_1_part3_1_perm1302__S__S__D_2__D_0.AlignModesWith( modes_0_2, axppx2_temp__D_0__D_1__D_2__D_3, modes_2_0 );
                axppx3_temp_part1_1_part3_1_perm1302__S__S__D_2__D_0.AllGatherRedistFrom( axppx3_temp_part1_1_part3_1__D_2__D_1__D_0__D_3, modes_1_3 );
                axppx3_temp_part1_1_part3_1__D_2__D_1__D_0__D_3.EmptyData();
                Permute( accum_temp_part0_1_part2_1__S__D_1__S__D_3__D_0__D_2, accum_temp_part0_1_part2_1_perm021354__S__S__D_1__D_3__D_2__D_0 );
                accum_temp_part0_1_part2_1__S__D_1__S__D_3__D_0__D_2.EmptyData();
//                printf("ping5\n");
                   // 0.5 * axppx3_temp_part1_1_part3_1[D2,*,D0,*]_emog * axppx2_temp[D0,D1,D2,D3]_ogfn + 0.0 * accum_temp_part0_1_part2_1[*,D1,*,D3,D0,D2]_emfnog
                LocalContract(0.5, axppx3_temp_part1_1_part3_1_perm1302__S__S__D_2__D_0.LockedTensor(), indices_emog, false,
                    axppx2_temp_perm2013__D_2__D_0__D_1__D_3.LockedTensor(), indices_ogfn, false,
                    0.0, accum_temp_part0_1_part2_1_perm021354__S__S__D_1__D_3__D_2__D_0.Tensor(), indices_emfnog, false);
                   // accum_temp_part0_1_part2_1[D0,D1,D2,D3] <- accum_temp_part0_1_part2_1[*,D1,*,D3,D0,D2] (with SumScatter on (D0)(D2))
//                PrintData(accum_temp_part0_1_part2_1_perm021354__S__S__D_1__D_3__D_2__D_0, "accum_in");
//                PrintData(accum_temp_part0_1_part2_1__D_0__D_1__D_2__D_3, "accum_out");

                accum_temp_part0_1_part2_1__D_0__D_1__D_2__D_3.ReduceScatterUpdateRedistFrom( accum_temp_part0_1_part2_1_perm021354__S__S__D_1__D_3__D_2__D_0, -1.0, modes_5_4 );
                accum_temp_part0_1_part2_1_perm021354__S__S__D_1__D_3__D_2__D_0.EmptyData();
                axppx3_temp_part1_1_part3_1_perm1302__S__S__D_2__D_0.EmptyData();

                //------------------------------------//
                SlidePartitionDown
                ( axppx3_temp_part1_1_part3T__D_0__D_1__D_2__D_3,  axppx3_temp_part1_1_part3_0__D_0__D_1__D_2__D_3,
                       axppx3_temp_part1_1_part3_1__D_0__D_1__D_2__D_3,
                  /**/ /**/
                  axppx3_temp_part1_1_part3B__D_0__D_1__D_2__D_3, axppx3_temp_part1_1_part3_2__D_0__D_1__D_2__D_3, 3 );
                SlidePartitionDown
                ( accum_temp_part0_1_part2T__D_0__D_1__D_2__D_3,  accum_temp_part0_1_part2_0__D_0__D_1__D_2__D_3,
                       accum_temp_part0_1_part2_1__D_0__D_1__D_2__D_3,
                  /**/ /**/
                  accum_temp_part0_1_part2B__D_0__D_1__D_2__D_3, accum_temp_part0_1_part2_2__D_0__D_1__D_2__D_3, 2 );
            }

            //****

            //------------------------------------//
            SlidePartitionDown
            ( axppx3_temp_part1T__D_0__D_1__D_2__D_3,  axppx3_temp_part1_0__D_0__D_1__D_2__D_3,
                   axppx3_temp_part1_1__D_0__D_1__D_2__D_3,
              /**/ /**/
              axppx3_temp_part1B__D_0__D_1__D_2__D_3, axppx3_temp_part1_2__D_0__D_1__D_2__D_3, 1 );
            SlidePartitionDown
            ( accum_temp_part0T__D_0__D_1__D_2__D_3,  accum_temp_part0_0__D_0__D_1__D_2__D_3,
                   accum_temp_part0_1__D_0__D_1__D_2__D_3,
              /**/ /**/
              accum_temp_part0B__D_0__D_1__D_2__D_3, accum_temp_part0_2__D_0__D_1__D_2__D_3, 0 );
            SlidePartitionDown
            ( cont1_temp_part0T__D_0__D_1__D_2__D_3,  cont1_temp_part0_0__D_0__D_1__D_2__D_3,
                   cont1_temp_part0_1__D_0__D_1__D_2__D_3,
              /**/ /**/
              cont1_temp_part0B__D_0__D_1__D_2__D_3, cont1_temp_part0_2__D_0__D_1__D_2__D_3, 0 );

        }

//        printf("accuming\n");
//        Print(accum_temp_part0_1_part2_1__D_0__D_1__D_2__D_3, "accum_after LC1");
        //****
        //**** (out of 1)
        //**** Is real  0 shadows
            //Outputs:
            //  accum_temp__D_0__D_1__D_2__D_3
        PartitionDown(v_efgh__D_0__D_1__D_2__D_3, v_efgh_part2T__D_0__D_1__D_2__D_3, v_efgh_part2B__D_0__D_1__D_2__D_3, 2, 0);
        PartitionDown(t_efmn__D_0__D_1__D_2__D_3, t_efmn_part0T__D_0__D_1__D_2__D_3, t_efmn_part0B__D_0__D_1__D_2__D_3, 0, 0);
        while(v_efgh_part2T__D_0__D_1__D_2__D_3.Dimension(2) < v_efgh__D_0__D_1__D_2__D_3.Dimension(2))
        {
            RepartitionDown
            ( v_efgh_part2T__D_0__D_1__D_2__D_3,  v_efgh_part2_0__D_0__D_1__D_2__D_3,
              /**/ /**/
                   v_efgh_part2_1__D_0__D_1__D_2__D_3,
              v_efgh_part2B__D_0__D_1__D_2__D_3, v_efgh_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );
            RepartitionDown
            ( t_efmn_part0T__D_0__D_1__D_2__D_3,  t_efmn_part0_0__D_0__D_1__D_2__D_3,
              /**/ /**/
                   t_efmn_part0_1__D_0__D_1__D_2__D_3,
              t_efmn_part0B__D_0__D_1__D_2__D_3, t_efmn_part0_2__D_0__D_1__D_2__D_3, 0, blkSize );
            //------------------------------------//

            //**** (out of 1)
            //**** Is real  0 shadows
                //Outputs:
                //  accum_temp__D_0__D_1__D_2__D_3
            PartitionDown(v_efgh_part2_1__D_0__D_1__D_2__D_3, v_efgh_part2_1_part3T__D_0__D_1__D_2__D_3, v_efgh_part2_1_part3B__D_0__D_1__D_2__D_3, 3, 0);
            PartitionDown(t_efmn_part0_1__D_0__D_1__D_2__D_3, t_efmn_part0_1_part1T__D_0__D_1__D_2__D_3, t_efmn_part0_1_part1B__D_0__D_1__D_2__D_3, 1, 0);
            while(v_efgh_part2_1_part3T__D_0__D_1__D_2__D_3.Dimension(3) < v_efgh_part2_1__D_0__D_1__D_2__D_3.Dimension(3))
            {
                RepartitionDown
                ( v_efgh_part2_1_part3T__D_0__D_1__D_2__D_3,  v_efgh_part2_1_part3_0__D_0__D_1__D_2__D_3,
                  /**/ /**/
                       v_efgh_part2_1_part3_1__D_0__D_1__D_2__D_3,
                  v_efgh_part2_1_part3B__D_0__D_1__D_2__D_3, v_efgh_part2_1_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );
                RepartitionDown
                ( t_efmn_part0_1_part1T__D_0__D_1__D_2__D_3,  t_efmn_part0_1_part1_0__D_0__D_1__D_2__D_3,
                  /**/ /**/
                       t_efmn_part0_1_part1_1__D_0__D_1__D_2__D_3,
                  t_efmn_part0_1_part1B__D_0__D_1__D_2__D_3, t_efmn_part0_1_part1_2__D_0__D_1__D_2__D_3, 1, blkSize );
                //------------------------------------//

                   // v_efgh_part2_1_part3_1[D0,D1,*,*] <- v_efgh_part2_1_part3_1[D0,D1,D2,D3]
                v_efgh_part2_1_part3_1__D_0__D_1__S__S.AlignModesWith( modes_0_1, accum_temp__D_0__D_1__D_2__D_3, modes_0_1 );
//                v_efgh_part2_1_part3_1__D_0__D_1__S__S.AlignModesWith( modes_2_3, v_efgh_part2_1_part3_1__D_0__D_1__D_2__D_3, modes_2_3 );
                v_efgh_part2_1_part3_1__D_0__D_1__S__S.AllGatherRedistFrom( v_efgh_part2_1_part3_1__D_0__D_1__D_2__D_3, modes_2_3 );
                   // t_efmn_part0_1_part1_1[*,*,D2,D3] <- t_efmn_part0_1_part1_1[D0,D1,D2,D3]
//                t_efmn_part0_1_part1_1__S__S__D_2__D_3.AlignModesWith( modes_0_1, t_efmn_part0_1_part1_1__D_0__D_1__D_2__D_3, modes_0_1 );
                t_efmn_part0_1_part1_1__S__S__D_2__D_3.AlignModesWith( modes_2_3, accum_temp__D_0__D_1__D_2__D_3, modes_2_3 );
                t_efmn_part0_1_part1_1__S__S__D_2__D_3.AllGatherRedistFrom( t_efmn_part0_1_part1_1__D_0__D_1__D_2__D_3, modes_0_1 );

//                printf("ping7\n");
                   // 0.5 * v_efgh_part2_1_part3_1[D0,D1,*,*]_efgh * t_efmn_part0_1_part1_1[*,*,D2,D3]_ghmn + 1.0 * accum_temp[D0,D1,D2,D3]_efmn
                LocalContractAndLocalEliminate(0.5, v_efgh_part2_1_part3_1__D_0__D_1__S__S.LockedTensor(), indices_efgh, false,
                    t_efmn_part0_1_part1_1__S__S__D_2__D_3.LockedTensor(), indices_ghmn, false,
                    1.0, accum_temp__D_0__D_1__D_2__D_3.Tensor(), indices_efmn, false);

                t_efmn_part0_1_part1_1__S__S__D_2__D_3.EmptyData();
                v_efgh_part2_1_part3_1__D_0__D_1__S__S.EmptyData();

                //------------------------------------//
                SlidePartitionDown
                ( v_efgh_part2_1_part3T__D_0__D_1__D_2__D_3,  v_efgh_part2_1_part3_0__D_0__D_1__D_2__D_3,
                       v_efgh_part2_1_part3_1__D_0__D_1__D_2__D_3,
                  /**/ /**/
                  v_efgh_part2_1_part3B__D_0__D_1__D_2__D_3, v_efgh_part2_1_part3_2__D_0__D_1__D_2__D_3, 3 );
                SlidePartitionDown
                ( t_efmn_part0_1_part1T__D_0__D_1__D_2__D_3,  t_efmn_part0_1_part1_0__D_0__D_1__D_2__D_3,
                       t_efmn_part0_1_part1_1__D_0__D_1__D_2__D_3,
                  /**/ /**/
                  t_efmn_part0_1_part1B__D_0__D_1__D_2__D_3, t_efmn_part0_1_part1_2__D_0__D_1__D_2__D_3, 1 );

            }
            //****

            //------------------------------------//
            SlidePartitionDown
            ( v_efgh_part2T__D_0__D_1__D_2__D_3,  v_efgh_part2_0__D_0__D_1__D_2__D_3,
                   v_efgh_part2_1__D_0__D_1__D_2__D_3,
              /**/ /**/
              v_efgh_part2B__D_0__D_1__D_2__D_3, v_efgh_part2_2__D_0__D_1__D_2__D_3, 2 );
            SlidePartitionDown
            ( t_efmn_part0T__D_0__D_1__D_2__D_3,  t_efmn_part0_0__D_0__D_1__D_2__D_3,
                   t_efmn_part0_1__D_0__D_1__D_2__D_3,
              /**/ /**/
              t_efmn_part0B__D_0__D_1__D_2__D_3, t_efmn_part0_2__D_0__D_1__D_2__D_3, 0 );

        }
//        printf("accum_t\n");
//        Print(accum_temp__D_0__D_1__D_2__D_3, "accum LC2");
        //****
        Permute( accum_temp__D_0__D_1__D_2__D_3, accum_temp_perm2301__D_2__D_3__D_0__D_1 );
        //**** (out of 1)
        //**** Is real  0 shadows
            //Outputs:
            //  accum_temp_perm2301__D_2__D_3__D_0__D_1
        PartitionDown(v_opmn__D_0__D_1__D_2__D_3, v_opmn_part0T__D_0__D_1__D_2__D_3, v_opmn_part0B__D_0__D_1__D_2__D_3, 0, 0);
        PartitionDown(t_efmn__D_0__D_1__D_2__D_3, t_efmn_part2T__D_0__D_1__D_2__D_3, t_efmn_part2B__D_0__D_1__D_2__D_3, 2, 0);
        while(v_opmn_part0T__D_0__D_1__D_2__D_3.Dimension(0) < v_opmn__D_0__D_1__D_2__D_3.Dimension(0))
        {
            RepartitionDown
            ( v_opmn_part0T__D_0__D_1__D_2__D_3,  v_opmn_part0_0__D_0__D_1__D_2__D_3,
              /**/ /**/
                   v_opmn_part0_1__D_0__D_1__D_2__D_3,
              v_opmn_part0B__D_0__D_1__D_2__D_3, v_opmn_part0_2__D_0__D_1__D_2__D_3, 0, blkSize );
            RepartitionDown
            ( t_efmn_part2T__D_0__D_1__D_2__D_3,  t_efmn_part2_0__D_0__D_1__D_2__D_3,
              /**/ /**/
                   t_efmn_part2_1__D_0__D_1__D_2__D_3,
              t_efmn_part2B__D_0__D_1__D_2__D_3, t_efmn_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );
            //------------------------------------//

            //**** (out of 1)
            //**** Is real  0 shadows
                //Outputs:
                //  accum_temp_perm2301__D_2__D_3__D_0__D_1
            PartitionDown(v_opmn_part0_1__D_0__D_1__D_2__D_3, v_opmn_part0_1_part1T__D_0__D_1__D_2__D_3, v_opmn_part0_1_part1B__D_0__D_1__D_2__D_3, 1, 0);
            PartitionDown(t_efmn_part2_1__D_0__D_1__D_2__D_3, t_efmn_part2_1_part3T__D_0__D_1__D_2__D_3, t_efmn_part2_1_part3B__D_0__D_1__D_2__D_3, 3, 0);
            while(v_opmn_part0_1_part1T__D_0__D_1__D_2__D_3.Dimension(1) < v_opmn_part0_1__D_0__D_1__D_2__D_3.Dimension(1))
            {
                RepartitionDown
                ( v_opmn_part0_1_part1T__D_0__D_1__D_2__D_3,  v_opmn_part0_1_part1_0__D_0__D_1__D_2__D_3,
                  /**/ /**/
                       v_opmn_part0_1_part1_1__D_0__D_1__D_2__D_3,
                  v_opmn_part0_1_part1B__D_0__D_1__D_2__D_3, v_opmn_part0_1_part1_2__D_0__D_1__D_2__D_3, 1, blkSize );
                RepartitionDown
                ( t_efmn_part2_1_part3T__D_0__D_1__D_2__D_3,  t_efmn_part2_1_part3_0__D_0__D_1__D_2__D_3,
                  /**/ /**/
                       t_efmn_part2_1_part3_1__D_0__D_1__D_2__D_3,
                  t_efmn_part2_1_part3B__D_0__D_1__D_2__D_3, t_efmn_part2_1_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );
                //------------------------------------//

                   // v_opmn_part0_1_part1_1[*,*,D2,D3] <- v_opmn_part0_1_part1_1[D0,D1,D2,D3]
                v_opmn_part0_1_part1_1_perm2301__D_2__D_3__S__S.AlignModesWith( modes_2_3, accum_temp__D_0__D_1__D_2__D_3, modes_2_3 );
                v_opmn_part0_1_part1_1_perm2301__D_2__D_3__S__S.AllGatherRedistFrom( v_opmn_part0_1_part1_1__D_0__D_1__D_2__D_3, modes_0_1 );
                   // t_efmn_part2_1_part3_1[D0,D1,*,*] <- t_efmn_part2_1_part3_1[D0,D1,D2,D3]
                t_efmn_part2_1_part3_1_perm2301__S__S__D_0__D_1.AlignModesWith( modes_0_1, accum_temp__D_0__D_1__D_2__D_3, modes_0_1 );
                t_efmn_part2_1_part3_1_perm2301__S__S__D_0__D_1.AllGatherRedistFrom( t_efmn_part2_1_part3_1__D_0__D_1__D_2__D_3, modes_2_3 );
                   // 0.5 * v_opmn_part0_1_part1_1[*,*,D2,D3]_mnop * t_efmn_part2_1_part3_1[D0,D1,*,*]_opef + 1.0 * accum_temp[D0,D1,D2,D3]_mnef
                LocalContractAndLocalEliminate(0.5, v_opmn_part0_1_part1_1_perm2301__D_2__D_3__S__S.LockedTensor(), indices_mnop, false,
                    t_efmn_part2_1_part3_1_perm2301__S__S__D_0__D_1.LockedTensor(), indices_opef, false,
                    1.0, accum_temp_perm2301__D_2__D_3__D_0__D_1.Tensor(), indices_mnef, false);
                t_efmn_part2_1_part3_1_perm2301__S__S__D_0__D_1.EmptyData();
                v_opmn_part0_1_part1_1_perm2301__D_2__D_3__S__S.EmptyData();

                //------------------------------------//
                SlidePartitionDown
                ( v_opmn_part0_1_part1T__D_0__D_1__D_2__D_3,  v_opmn_part0_1_part1_0__D_0__D_1__D_2__D_3,
                       v_opmn_part0_1_part1_1__D_0__D_1__D_2__D_3,
                  /**/ /**/
                  v_opmn_part0_1_part1B__D_0__D_1__D_2__D_3, v_opmn_part0_1_part1_2__D_0__D_1__D_2__D_3, 1 );
                SlidePartitionDown
                ( t_efmn_part2_1_part3T__D_0__D_1__D_2__D_3,  t_efmn_part2_1_part3_0__D_0__D_1__D_2__D_3,
                       t_efmn_part2_1_part3_1__D_0__D_1__D_2__D_3,
                  /**/ /**/
                  t_efmn_part2_1_part3B__D_0__D_1__D_2__D_3, t_efmn_part2_1_part3_2__D_0__D_1__D_2__D_3, 3 );

            }
            //****

            //------------------------------------//
            SlidePartitionDown
            ( v_opmn_part0T__D_0__D_1__D_2__D_3,  v_opmn_part0_0__D_0__D_1__D_2__D_3,
                   v_opmn_part0_1__D_0__D_1__D_2__D_3,
              /**/ /**/
              v_opmn_part0B__D_0__D_1__D_2__D_3, v_opmn_part0_2__D_0__D_1__D_2__D_3, 0 );
            SlidePartitionDown
            ( t_efmn_part2T__D_0__D_1__D_2__D_3,  t_efmn_part2_0__D_0__D_1__D_2__D_3,
                   t_efmn_part2_1__D_0__D_1__D_2__D_3,
              /**/ /**/
              t_efmn_part2B__D_0__D_1__D_2__D_3, t_efmn_part2_2__D_0__D_1__D_2__D_3, 2 );

        }
        //****
//        printf("accum after\n");
        Permute( accum_temp_perm2301__D_2__D_3__D_0__D_1, accum_temp__D_0__D_1__D_2__D_3 );
//        Print(accum_temp__D_0__D_1__D_2__D_3, "accum LC3");
        accum_temp_perm2301__D_2__D_3__D_0__D_1.EmptyData();
        //**** (out of 1)
        //**** Is real  0 shadows
            //Outputs:
            //  E_MP3____N_D_0_1_2_3
        PartitionDown(axppx2_temp__D_0__D_1__D_2__D_3, axppx2_temp_part0T__D_0__D_1__D_2__D_3, axppx2_temp_part0B__D_0__D_1__D_2__D_3, 0, 0);
        PartitionDown(accum_temp__D_0__D_1__D_2__D_3, accum_temp_part0T__D_0__D_1__D_2__D_3, accum_temp_part0B__D_0__D_1__D_2__D_3, 0, 0);
        while(axppx2_temp_part0T__D_0__D_1__D_2__D_3.Dimension(0) < axppx2_temp__D_0__D_1__D_2__D_3.Dimension(0))
        {
            RepartitionDown
            ( axppx2_temp_part0T__D_0__D_1__D_2__D_3,  axppx2_temp_part0_0__D_0__D_1__D_2__D_3,
              /**/ /**/
                   axppx2_temp_part0_1__D_0__D_1__D_2__D_3,
              axppx2_temp_part0B__D_0__D_1__D_2__D_3, axppx2_temp_part0_2__D_0__D_1__D_2__D_3, 0, blkSize );
            RepartitionDown
            ( accum_temp_part0T__D_0__D_1__D_2__D_3,  accum_temp_part0_0__D_0__D_1__D_2__D_3,
              /**/ /**/
                   accum_temp_part0_1__D_0__D_1__D_2__D_3,
              accum_temp_part0B__D_0__D_1__D_2__D_3, accum_temp_part0_2__D_0__D_1__D_2__D_3, 0, blkSize );
            //------------------------------------//

            //**** (out of 1)
            //**** Is real  0 shadows
                //Outputs:
                //  E_MP3____N_D_0_1_2_3
            PartitionDown(axppx2_temp_part0_1__D_0__D_1__D_2__D_3, axppx2_temp_part0_1_part1T__D_0__D_1__D_2__D_3, axppx2_temp_part0_1_part1B__D_0__D_1__D_2__D_3, 1, 0);
            PartitionDown(accum_temp_part0_1__D_0__D_1__D_2__D_3, accum_temp_part0_1_part1T__D_0__D_1__D_2__D_3, accum_temp_part0_1_part1B__D_0__D_1__D_2__D_3, 1, 0);
            while(axppx2_temp_part0_1_part1T__D_0__D_1__D_2__D_3.Dimension(1) < axppx2_temp_part0_1__D_0__D_1__D_2__D_3.Dimension(1))
            {
                RepartitionDown
                ( axppx2_temp_part0_1_part1T__D_0__D_1__D_2__D_3,  axppx2_temp_part0_1_part1_0__D_0__D_1__D_2__D_3,
                  /**/ /**/
                       axppx2_temp_part0_1_part1_1__D_0__D_1__D_2__D_3,
                  axppx2_temp_part0_1_part1B__D_0__D_1__D_2__D_3, axppx2_temp_part0_1_part1_2__D_0__D_1__D_2__D_3, 1, blkSize );
                RepartitionDown
                ( accum_temp_part0_1_part1T__D_0__D_1__D_2__D_3,  accum_temp_part0_1_part1_0__D_0__D_1__D_2__D_3,
                  /**/ /**/
                       accum_temp_part0_1_part1_1__D_0__D_1__D_2__D_3,
                  accum_temp_part0_1_part1B__D_0__D_1__D_2__D_3, accum_temp_part0_1_part1_2__D_0__D_1__D_2__D_3, 1, blkSize );
                //------------------------------------//

                tempShape = E_MP3____N_D_0_1_2_3.Shape();
                tempShape.push_back( g.Shape()[0] );
                tempShape.push_back( g.Shape()[1] );
                tempShape.push_back( g.Shape()[2] );
                tempShape.push_back( g.Shape()[3] );
                E_MP3__D_0__D_1__D_2__D_3.ResizeTo( tempShape );
                Zero(E_MP3__D_0__D_1__D_2__D_3);

                   // 2.0 * axppx2_temp_part0_1_part1_1[D0,D1,D2,D3]_efmn * accum_temp_part0_1_part1_1[D0,D1,D2,D3]_efmn + 0.0 * E_MP3[D0,D1,D2,D3]_efmn
                LocalContract(2.0, axppx2_temp_part0_1_part1_1__D_0__D_1__D_2__D_3.LockedTensor(), indices_efmn, true,
                    accum_temp_part0_1_part1_1__D_0__D_1__D_2__D_3.LockedTensor(), indices_efmn, true,
                    0.0, E_MP3__D_0__D_1__D_2__D_3.Tensor(), indices_efmn, true);

                   // E_MP3[] | {0,1,2,3} <- E_MP3[D0,D1,D2,D3] (with SumScatter on (D0)(D1)(D2)(D3))
                E_MP3____N_D_0_1_2_3.ReduceToOneUpdateRedistFrom( E_MP3__D_0__D_1__D_2__D_3, 1.0, modes_0_1_2_3 );
                E_MP3__D_0__D_1__D_2__D_3.EmptyData();

                //------------------------------------//
                SlidePartitionDown
                ( axppx2_temp_part0_1_part1T__D_0__D_1__D_2__D_3,  axppx2_temp_part0_1_part1_0__D_0__D_1__D_2__D_3,
                       axppx2_temp_part0_1_part1_1__D_0__D_1__D_2__D_3,
                  /**/ /**/
                  axppx2_temp_part0_1_part1B__D_0__D_1__D_2__D_3, axppx2_temp_part0_1_part1_2__D_0__D_1__D_2__D_3, 1 );
                SlidePartitionDown
                ( accum_temp_part0_1_part1T__D_0__D_1__D_2__D_3,  accum_temp_part0_1_part1_0__D_0__D_1__D_2__D_3,
                       accum_temp_part0_1_part1_1__D_0__D_1__D_2__D_3,
                  /**/ /**/
                  accum_temp_part0_1_part1B__D_0__D_1__D_2__D_3, accum_temp_part0_1_part1_2__D_0__D_1__D_2__D_3, 1 );

            }
            //****

            //------------------------------------//
            SlidePartitionDown
            ( axppx2_temp_part0T__D_0__D_1__D_2__D_3,  axppx2_temp_part0_0__D_0__D_1__D_2__D_3,
                   axppx2_temp_part0_1__D_0__D_1__D_2__D_3,
              /**/ /**/
              axppx2_temp_part0B__D_0__D_1__D_2__D_3, axppx2_temp_part0_2__D_0__D_1__D_2__D_3, 0 );
            SlidePartitionDown
            ( accum_temp_part0T__D_0__D_1__D_2__D_3,  accum_temp_part0_0__D_0__D_1__D_2__D_3,
                   accum_temp_part0_1__D_0__D_1__D_2__D_3,
              /**/ /**/
              accum_temp_part0B__D_0__D_1__D_2__D_3, accum_temp_part0_2__D_0__D_1__D_2__D_3, 0 );

        }

//****

/*****************************************/
    mpi::Barrier(g.OwningComm());
    runTime = mpi::Time() - startTime;
    double flops = pow(n_o, 2) * pow(n_v, 2) * ( 11 + 2 * pow(n_o + n_v, 2));
    gflops = flops / (1e9 * runTime);

    //****

    //------------------------------------//

//****

    if(commRank == 0)
        Timer::printTimers();


Print(E_MP3____N_D_0_1_2_3, "E_MP3");


  //****
    if(commRank == 0){
        std::cout << "RUNTIME " << runTime << std::endl;
        std::cout << "FLOPS " << flops << std::endl;
        std::cout << "GFLOPS " << gflops << std::endl;
    }
    E_MP3____N_D_0_1_2_3.ClearCommMap();
}

int
main( int argc, char* argv[] )
{
    Initialize( argc, argv );
    Unsigned i;
    mpi::Comm comm = mpi::COMM_WORLD;
    const Int commRank = mpi::CommRank( comm );
    const Int commSize = mpi::CommSize( comm );
//    printf("My Rank: %d\n", commRank);
    try
    {
        Params args;

        ProcessInput(argc, argv, args);

        if(commRank == 0 && commSize != args.nProcs){
            std::cerr << "program not started with correct number of processes\n";
        std::cerr << commSize << " vs " << args.nProcs << std::endl;
            Usage();
            throw ArgException();
        }

//        if(commRank == 0){
//            printf("Creating %d", args.gridShape[0]);
//            for(i = 1; i < GRIDORDER; i++)
//                printf(" x %d", args.gridShape[i]);
//            printf(" grid\n");
//        }

        const Grid g( comm, args.gridShape );

//        if( commRank == 0 )
//        {
//            std::cout << "------------------" << std::endl
//                      << "Testing with doubles:" << std::endl
//                      << "------------------" << std::endl;
//        }



        DistTensorTest<double>( g, args.n_o, args.n_v, args.blkSize );

    }
    catch( std::exception& e ) { ReportException(e); }

    Finalize();
    //printf("Completed\n");
    return 0;
}

