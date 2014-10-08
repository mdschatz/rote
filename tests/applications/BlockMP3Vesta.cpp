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
  Unsigned tenDimFive;
  Unsigned tenDimFiftyThree;
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

  args.tenDimFive = atoi(argv[++argCount]);
  args.tenDimFiftyThree = atoi(argv[++argCount]);
}

template<typename T>
void
DistTensorTest( const Grid& g, Unsigned tenDimFive, Unsigned tenDimFiftyThree )
{
#ifndef RELEASE
  CallStackEntry entry("DistTensorTest");
#endif
  Unsigned i;
  const Int commRank = mpi::CommRank( mpi::COMM_WORLD );
  const Unsigned gridOrder = 4;


ObjShape tempShape;
TensorDistribution dist____N_D_0_1_2_3 = tmen::StringToTensorDist("[]|(0,1,2,3)");
TensorDistribution dist__S__S__D_2__D_3 = tmen::StringToTensorDist("[(),(),(2),(3)]");
TensorDistribution dist__S__D_0__S__D_2 = tmen::StringToTensorDist("[(),(0),(),(2)]");
TensorDistribution dist__S__D_1__S__D_3__D_0__D_2 = tmen::StringToTensorDist("[(),(1),(),(3),(0),(2)]");
TensorDistribution dist__S__D_1__D_3__S = tmen::StringToTensorDist("[(),(1),(3),()]");
TensorDistribution dist__D_0__D_1__S__S = tmen::StringToTensorDist("[(0),(1),(),()]");
TensorDistribution dist__D_0__D_1__D_2__D_3 = tmen::StringToTensorDist("[(0),(1),(2),(3)]");
TensorDistribution dist__D_0__D_1__D_3__D_2 = tmen::StringToTensorDist("[(0),(1),(3),(2)]");
TensorDistribution dist__D_1__D_0__D_2__D_3 = tmen::StringToTensorDist("[(1),(0),(2),(3)]");
TensorDistribution dist__D_1__D_0__D_3__D_2 = tmen::StringToTensorDist("[(1),(0),(3),(2)]");
TensorDistribution dist__D_2__S__D_0__S = tmen::StringToTensorDist("[(2),(),(0),()]");
TensorDistribution dist__D_2__D_1__D_0__D_3 = tmen::StringToTensorDist("[(2),(1),(0),(3)]");
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
ModeArray modes_0;
modes_0.push_back(0);
ModeArray modes_0_1;
modes_0_1.push_back(0);
modes_0_1.push_back(1);
ModeArray modes_0_1_2_3;
modes_0_1_2_3.push_back(0);
modes_0_1_2_3.push_back(1);
modes_0_1_2_3.push_back(2);
modes_0_1_2_3.push_back(3);
ModeArray modes_0_2;
modes_0_2.push_back(0);
modes_0_2.push_back(2);
ModeArray modes_0_3;
modes_0_3.push_back(0);
modes_0_3.push_back(3);
ModeArray modes_1;
modes_1.push_back(1);
ModeArray modes_1_0;
modes_1_0.push_back(1);
modes_1_0.push_back(0);
ModeArray modes_1_3;
modes_1_3.push_back(1);
modes_1_3.push_back(3);
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
ModeArray modes_3_2;
modes_3_2.push_back(3);
modes_3_2.push_back(2);
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
IndexArray indices_efmngo( 6 );
indices_efmngo[0] = 'e';
indices_efmngo[1] = 'f';
indices_efmngo[2] = 'm';
indices_efmngo[3] = 'n';
indices_efmngo[4] = 'g';
indices_efmngo[5] = 'o';
IndexArray indices_efop( 4 );
indices_efop[0] = 'e';
indices_efop[1] = 'f';
indices_efop[2] = 'o';
indices_efop[3] = 'p';
IndexArray indices_gfno( 4 );
indices_gfno[0] = 'g';
indices_gfno[1] = 'f';
indices_gfno[2] = 'n';
indices_gfno[3] = 'o';
IndexArray indices_gfon( 4 );
indices_gfon[0] = 'g';
indices_gfon[1] = 'f';
indices_gfon[2] = 'o';
indices_gfon[3] = 'n';
IndexArray indices_ghmn( 4 );
indices_ghmn[0] = 'g';
indices_ghmn[1] = 'h';
indices_ghmn[2] = 'm';
indices_ghmn[3] = 'n';
IndexArray indices_oegm( 4 );
indices_oegm[0] = 'o';
indices_oegm[1] = 'e';
indices_oegm[2] = 'g';
indices_oegm[3] = 'm';
IndexArray indices_opmn( 4 );
indices_opmn[0] = 'o';
indices_opmn[1] = 'p';
indices_opmn[2] = 'm';
indices_opmn[3] = 'n';
std::vector<ModeArray> modeArrayArray___0___1;
modeArrayArray___0___1.push_back(modes_0);
modeArrayArray___0___1.push_back(modes_1);
std::vector<ModeArray> modeArrayArray___0___2;
modeArrayArray___0___2.push_back(modes_0);
modeArrayArray___0___2.push_back(modes_2);
std::vector<ModeArray> modeArrayArray___1___3;
modeArrayArray___1___3.push_back(modes_1);
modeArrayArray___1___3.push_back(modes_3);
std::vector<ModeArray> modeArrayArray___2___3;
modeArrayArray___2___3.push_back(modes_2);
modeArrayArray___2___3.push_back(modes_3);
	//E_MP3[D0,D1,D2,D3]
DistTensor<double> E_MP3__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
E_MP3__D_0__D_1__D_2__D_3.SetLocalPermutation( perm_0_1_2_3 );
	//E_MP3[] | {0,1,2,3}
DistTensor<double> E_MP3____N_D_0_1_2_3( dist____N_D_0_1_2_3, g );
E_MP3____N_D_0_1_2_3.SetLocalPermutation( perm );
	//accum_temp[D0,D1,D2,D3]
DistTensor<double> accum_temp__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
accum_temp__D_0__D_1__D_2__D_3.SetLocalPermutation( perm_0_1_2_3 );
	//accum_temp_part0B[D0,D1,D2,D3]
DistTensor<double> accum_temp_part0B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
accum_temp_part0B__D_0__D_1__D_2__D_3.SetLocalPermutation( perm_0_1_2_3 );
	//accum_temp_part0T[D0,D1,D2,D3]
DistTensor<double> accum_temp_part0T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
accum_temp_part0T__D_0__D_1__D_2__D_3.SetLocalPermutation( perm_0_1_2_3 );
	//accum_temp_part0_0[D0,D1,D2,D3]
DistTensor<double> accum_temp_part0_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
accum_temp_part0_0__D_0__D_1__D_2__D_3.SetLocalPermutation( perm_0_1_2_3 );
	//accum_temp_part0_1[D0,D1,D2,D3]
DistTensor<double> accum_temp_part0_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
accum_temp_part0_1__D_0__D_1__D_2__D_3.SetLocalPermutation( perm_0_1_2_3 );
	//accum_temp_part0_1[*,D1,*,D3,D0,D2]
DistTensor<double> accum_temp_part0_1__S__D_1__S__D_3__D_0__D_2( dist__S__D_1__S__D_3__D_0__D_2, g );
accum_temp_part0_1__S__D_1__S__D_3__D_0__D_2.SetLocalPermutation( perm_0_1_2_3_4_5 );
	//accum_temp_part0_2[D0,D1,D2,D3]
DistTensor<double> accum_temp_part0_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
accum_temp_part0_2__D_0__D_1__D_2__D_3.SetLocalPermutation( perm_0_1_2_3 );
	//axppx2_temp[D0,D1,D2,D3]
DistTensor<double> axppx2_temp__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
axppx2_temp__D_0__D_1__D_2__D_3.SetLocalPermutation( perm_0_1_2_3 );
	//axppx2_temp_part0B[D0,D1,D2,D3]
DistTensor<double> axppx2_temp_part0B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
axppx2_temp_part0B__D_0__D_1__D_2__D_3.SetLocalPermutation( perm_0_1_2_3 );
	//axppx2_temp_part0T[D0,D1,D2,D3]
DistTensor<double> axppx2_temp_part0T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
axppx2_temp_part0T__D_0__D_1__D_2__D_3.SetLocalPermutation( perm_0_1_2_3 );
	//axppx2_temp_part0_0[D0,D1,D2,D3]
DistTensor<double> axppx2_temp_part0_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
axppx2_temp_part0_0__D_0__D_1__D_2__D_3.SetLocalPermutation( perm_0_1_2_3 );
	//axppx2_temp_part0_1[D0,D1,D2,D3]
DistTensor<double> axppx2_temp_part0_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
axppx2_temp_part0_1__D_0__D_1__D_2__D_3.SetLocalPermutation( perm_0_1_2_3 );
	//axppx2_temp_part0_2[D0,D1,D2,D3]
DistTensor<double> axppx2_temp_part0_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
axppx2_temp_part0_2__D_0__D_1__D_2__D_3.SetLocalPermutation( perm_0_1_2_3 );
	//axppx3_temp[D0,D1,D2,D3]
DistTensor<double> axppx3_temp__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
axppx3_temp__D_0__D_1__D_2__D_3.SetLocalPermutation( perm_0_1_2_3 );
	//axppx3_temp_part1B[D0,D1,D2,D3]
DistTensor<double> axppx3_temp_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
axppx3_temp_part1B__D_0__D_1__D_2__D_3.SetLocalPermutation( perm_0_1_2_3 );
	//axppx3_temp_part1T[D0,D1,D2,D3]
DistTensor<double> axppx3_temp_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
axppx3_temp_part1T__D_0__D_1__D_2__D_3.SetLocalPermutation( perm_0_1_2_3 );
	//axppx3_temp_part1_0[D0,D1,D2,D3]
DistTensor<double> axppx3_temp_part1_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
axppx3_temp_part1_0__D_0__D_1__D_2__D_3.SetLocalPermutation( perm_0_1_2_3 );
	//axppx3_temp_part1_1[D0,D1,D2,D3]
DistTensor<double> axppx3_temp_part1_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
axppx3_temp_part1_1__D_0__D_1__D_2__D_3.SetLocalPermutation( perm_0_1_2_3 );
	//axppx3_temp_part1_1[D2,D1,D0,D3]
DistTensor<double> axppx3_temp_part1_1__D_2__D_1__D_0__D_3( dist__D_2__D_1__D_0__D_3, g );
axppx3_temp_part1_1__D_2__D_1__D_0__D_3.SetLocalPermutation( perm_0_1_2_3 );
	//axppx3_temp_part1_1[D2,*,D0,*]
DistTensor<double> axppx3_temp_part1_1__D_2__S__D_0__S( dist__D_2__S__D_0__S, g );
axppx3_temp_part1_1__D_2__S__D_0__S.SetLocalPermutation( perm_0_1_2_3 );
	//axppx3_temp_part1_2[D0,D1,D2,D3]
DistTensor<double> axppx3_temp_part1_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
axppx3_temp_part1_2__D_0__D_1__D_2__D_3.SetLocalPermutation( perm_0_1_2_3 );
	//cont1_temp[D0,D1,D2,D3]
DistTensor<double> cont1_temp__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
cont1_temp__D_0__D_1__D_2__D_3.SetLocalPermutation( perm_0_1_2_3 );
	//cont1_temp[D0,D1,D3,D2]
DistTensor<double> cont1_temp__D_0__D_1__D_3__D_2( dist__D_0__D_1__D_3__D_2, g );
cont1_temp__D_0__D_1__D_3__D_2.SetLocalPermutation( perm_0_1_2_3 );
	//t_efmn[D0,D1,D2,D3]
DistTensor<double> t_efmn__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
t_efmn__D_0__D_1__D_2__D_3.SetLocalPermutation( perm_0_1_2_3 );
	//t_efmn[D0,D1,D3,D2]
DistTensor<double> t_efmn__D_0__D_1__D_3__D_2( dist__D_0__D_1__D_3__D_2, g );
t_efmn__D_0__D_1__D_3__D_2.SetLocalPermutation( perm_0_1_2_3 );
	//t_efmn_part0B[D0,D1,D2,D3]
DistTensor<double> t_efmn_part0B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
t_efmn_part0B__D_0__D_1__D_2__D_3.SetLocalPermutation( perm_0_1_2_3 );
	//t_efmn_part0T[D0,D1,D2,D3]
DistTensor<double> t_efmn_part0T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
t_efmn_part0T__D_0__D_1__D_2__D_3.SetLocalPermutation( perm_0_1_2_3 );
	//t_efmn_part0_0[D0,D1,D2,D3]
DistTensor<double> t_efmn_part0_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
t_efmn_part0_0__D_0__D_1__D_2__D_3.SetLocalPermutation( perm_0_1_2_3 );
	//t_efmn_part0_1[D0,D1,D2,D3]
DistTensor<double> t_efmn_part0_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
t_efmn_part0_1__D_0__D_1__D_2__D_3.SetLocalPermutation( perm_0_1_2_3 );
	//t_efmn_part0_1[D0,D1,D3,D2]
DistTensor<double> t_efmn_part0_1__D_0__D_1__D_3__D_2( dist__D_0__D_1__D_3__D_2, g );
t_efmn_part0_1__D_0__D_1__D_3__D_2.SetLocalPermutation( perm_0_1_2_3 );
	//t_efmn_part0_1[D0,D1,*,*]
DistTensor<double> t_efmn_part0_1__D_0__D_1__S__S( dist__D_0__D_1__S__S, g );
t_efmn_part0_1__D_0__D_1__S__S.SetLocalPermutation( perm_0_1_2_3 );
	//t_efmn_part0_1[*,D1,D3,*]
DistTensor<double> t_efmn_part0_1__S__D_1__D_3__S( dist__S__D_1__D_3__S, g );
t_efmn_part0_1__S__D_1__D_3__S.SetLocalPermutation( perm_0_1_2_3 );
	//t_efmn_part0_1[*,*,D2,D3]
DistTensor<double> t_efmn_part0_1__S__S__D_2__D_3( dist__S__S__D_2__D_3, g );
t_efmn_part0_1__S__S__D_2__D_3.SetLocalPermutation( perm_0_1_2_3 );
	//t_efmn_part0_2[D0,D1,D2,D3]
DistTensor<double> t_efmn_part0_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
t_efmn_part0_2__D_0__D_1__D_2__D_3.SetLocalPermutation( perm_0_1_2_3 );
	//v2_oegm[D0,D1,D2,D3]
DistTensor<double> v2_oegm__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
v2_oegm__D_0__D_1__D_2__D_3.SetLocalPermutation( perm_0_1_2_3 );
	//v2_oegm_part2B[D0,D1,D2,D3]
DistTensor<double> v2_oegm_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
v2_oegm_part2B__D_0__D_1__D_2__D_3.SetLocalPermutation( perm_0_1_2_3 );
	//v2_oegm_part2T[D0,D1,D2,D3]
DistTensor<double> v2_oegm_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
v2_oegm_part2T__D_0__D_1__D_2__D_3.SetLocalPermutation( perm_0_1_2_3 );
	//v2_oegm_part2_0[D0,D1,D2,D3]
DistTensor<double> v2_oegm_part2_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
v2_oegm_part2_0__D_0__D_1__D_2__D_3.SetLocalPermutation( perm_0_1_2_3 );
	//v2_oegm_part2_1[D0,D1,D2,D3]
DistTensor<double> v2_oegm_part2_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
v2_oegm_part2_1__D_0__D_1__D_2__D_3.SetLocalPermutation( perm_0_1_2_3 );
	//v2_oegm_part2_1[D1,D0,D2,D3]
DistTensor<double> v2_oegm_part2_1__D_1__D_0__D_2__D_3( dist__D_1__D_0__D_2__D_3, g );
v2_oegm_part2_1__D_1__D_0__D_2__D_3.SetLocalPermutation( perm_0_1_2_3 );
	//v2_oegm_part2_1[D1,D0,D3,D2]
DistTensor<double> v2_oegm_part2_1__D_1__D_0__D_3__D_2( dist__D_1__D_0__D_3__D_2, g );
v2_oegm_part2_1__D_1__D_0__D_3__D_2.SetLocalPermutation( perm_0_1_2_3 );
	//v2_oegm_part2_1[*,D0,*,D2]
DistTensor<double> v2_oegm_part2_1__S__D_0__S__D_2( dist__S__D_0__S__D_2, g );
v2_oegm_part2_1__S__D_0__S__D_2.SetLocalPermutation( perm_0_1_2_3 );
	//v2_oegm_part2_2[D0,D1,D2,D3]
DistTensor<double> v2_oegm_part2_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
v2_oegm_part2_2__D_0__D_1__D_2__D_3.SetLocalPermutation( perm_0_1_2_3 );
	//v_efgh[D0,D1,D2,D3]
DistTensor<double> v_efgh__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
v_efgh__D_0__D_1__D_2__D_3.SetLocalPermutation( perm_0_1_2_3 );
	//v_efgh_part2B[D0,D1,D2,D3]
DistTensor<double> v_efgh_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
v_efgh_part2B__D_0__D_1__D_2__D_3.SetLocalPermutation( perm_0_1_2_3 );
	//v_efgh_part2T[D0,D1,D2,D3]
DistTensor<double> v_efgh_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
v_efgh_part2T__D_0__D_1__D_2__D_3.SetLocalPermutation( perm_0_1_2_3 );
	//v_efgh_part2_0[D0,D1,D2,D3]
DistTensor<double> v_efgh_part2_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
v_efgh_part2_0__D_0__D_1__D_2__D_3.SetLocalPermutation( perm_0_1_2_3 );
	//v_efgh_part2_1[D0,D1,D2,D3]
DistTensor<double> v_efgh_part2_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
v_efgh_part2_1__D_0__D_1__D_2__D_3.SetLocalPermutation( perm_0_1_2_3 );
	//v_efgh_part2_1[D0,D1,*,*]
DistTensor<double> v_efgh_part2_1__D_0__D_1__S__S( dist__D_0__D_1__S__S, g );
v_efgh_part2_1__D_0__D_1__S__S.SetLocalPermutation( perm_0_1_2_3 );
	//v_efgh_part2_2[D0,D1,D2,D3]
DistTensor<double> v_efgh_part2_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
v_efgh_part2_2__D_0__D_1__D_2__D_3.SetLocalPermutation( perm_0_1_2_3 );
	//v_oegm[D0,D1,D2,D3]
DistTensor<double> v_oegm__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
v_oegm__D_0__D_1__D_2__D_3.SetLocalPermutation( perm_0_1_2_3 );
	//v_opmn[D0,D1,D2,D3]
DistTensor<double> v_opmn__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
v_opmn__D_0__D_1__D_2__D_3.SetLocalPermutation( perm_0_1_2_3 );
	//v_opmn[*,*,D2,D3]
DistTensor<double> v_opmn__S__S__D_2__D_3( dist__S__S__D_2__D_3, g );
v_opmn__S__S__D_2__D_3.SetLocalPermutation( perm_0_1_2_3 );
// t_efmn has 4 dims
//	Starting distribution: [D0,D1,D2,D3] or _D_0__D_1__D_2__D_3
ObjShape t_efmn__D_0__D_1__D_2__D_3_tempShape;
t_efmn__D_0__D_1__D_2__D_3_tempShape.push_back(tenDimFiftyThree);
t_efmn__D_0__D_1__D_2__D_3_tempShape.push_back(tenDimFiftyThree);
t_efmn__D_0__D_1__D_2__D_3_tempShape.push_back(tenDimFive);
t_efmn__D_0__D_1__D_2__D_3_tempShape.push_back(tenDimFive);
t_efmn__D_0__D_1__D_2__D_3.ResizeTo( t_efmn__D_0__D_1__D_2__D_3_tempShape );
MakeUniform( t_efmn__D_0__D_1__D_2__D_3 );
DistTensor<T> t_efmn_local( tmen::StringToTensorDist("[(),(),(),()]|(0,1,2,3)"), g );
//GatherAllModes( t_efmn__D_0__D_1__D_2__D_3, t_efmn_local );
// axppx2_temp has 4 dims
//	Starting distribution: [D0,D1,D2,D3] or _D_0__D_1__D_2__D_3
ObjShape axppx2_temp__D_0__D_1__D_2__D_3_tempShape;
axppx2_temp__D_0__D_1__D_2__D_3_tempShape.push_back(tenDimFiftyThree);
axppx2_temp__D_0__D_1__D_2__D_3_tempShape.push_back(tenDimFiftyThree);
axppx2_temp__D_0__D_1__D_2__D_3_tempShape.push_back(tenDimFive);
axppx2_temp__D_0__D_1__D_2__D_3_tempShape.push_back(tenDimFive);
axppx2_temp__D_0__D_1__D_2__D_3.ResizeTo( axppx2_temp__D_0__D_1__D_2__D_3_tempShape );
MakeUniform( axppx2_temp__D_0__D_1__D_2__D_3 );
DistTensor<T> axppx2_temp_local( tmen::StringToTensorDist("[(),(),(),()]|(0,1,2,3)"), g );
//GatherAllModes( axppx2_temp__D_0__D_1__D_2__D_3, axppx2_temp_local );
// v_opmn has 4 dims
//	Starting distribution: [D0,D1,D2,D3] or _D_0__D_1__D_2__D_3
ObjShape v_opmn__D_0__D_1__D_2__D_3_tempShape;
v_opmn__D_0__D_1__D_2__D_3_tempShape.push_back(tenDimFive);
v_opmn__D_0__D_1__D_2__D_3_tempShape.push_back(tenDimFive);
v_opmn__D_0__D_1__D_2__D_3_tempShape.push_back(tenDimFive);
v_opmn__D_0__D_1__D_2__D_3_tempShape.push_back(tenDimFive);
v_opmn__D_0__D_1__D_2__D_3.ResizeTo( v_opmn__D_0__D_1__D_2__D_3_tempShape );
MakeUniform( v_opmn__D_0__D_1__D_2__D_3 );
DistTensor<T> v_opmn_local( tmen::StringToTensorDist("[(),(),(),()]|(0,1,2,3)"), g );
//GatherAllModes( v_opmn__D_0__D_1__D_2__D_3, v_opmn_local );
// v_efgh has 4 dims
//	Starting distribution: [D0,D1,D2,D3] or _D_0__D_1__D_2__D_3
ObjShape v_efgh__D_0__D_1__D_2__D_3_tempShape;
v_efgh__D_0__D_1__D_2__D_3_tempShape.push_back(tenDimFiftyThree);
v_efgh__D_0__D_1__D_2__D_3_tempShape.push_back(tenDimFiftyThree);
v_efgh__D_0__D_1__D_2__D_3_tempShape.push_back(tenDimFiftyThree);
v_efgh__D_0__D_1__D_2__D_3_tempShape.push_back(tenDimFiftyThree);
v_efgh__D_0__D_1__D_2__D_3.ResizeTo( v_efgh__D_0__D_1__D_2__D_3_tempShape );
MakeUniform( v_efgh__D_0__D_1__D_2__D_3 );
DistTensor<T> v_efgh_local( tmen::StringToTensorDist("[(),(),(),()]|(0,1,2,3)"), g );
//GatherAllModes( v_efgh__D_0__D_1__D_2__D_3, v_efgh_local );
// v_oegm has 4 dims
//	Starting distribution: [D0,D1,D2,D3] or _D_0__D_1__D_2__D_3
ObjShape v_oegm__D_0__D_1__D_2__D_3_tempShape;
v_oegm__D_0__D_1__D_2__D_3_tempShape.push_back(tenDimFive);
v_oegm__D_0__D_1__D_2__D_3_tempShape.push_back(tenDimFiftyThree);
v_oegm__D_0__D_1__D_2__D_3_tempShape.push_back(tenDimFiftyThree);
v_oegm__D_0__D_1__D_2__D_3_tempShape.push_back(tenDimFive);
v_oegm__D_0__D_1__D_2__D_3.ResizeTo( v_oegm__D_0__D_1__D_2__D_3_tempShape );
MakeUniform( v_oegm__D_0__D_1__D_2__D_3 );
DistTensor<T> v_oegm_local( tmen::StringToTensorDist("[(),(),(),()]|(0,1,2,3)"), g );
//GatherAllModes( v_oegm__D_0__D_1__D_2__D_3, v_oegm_local );
// v2_oegm has 4 dims
//	Starting distribution: [D0,D1,D2,D3] or _D_0__D_1__D_2__D_3
ObjShape v2_oegm__D_0__D_1__D_2__D_3_tempShape;
v2_oegm__D_0__D_1__D_2__D_3_tempShape.push_back(tenDimFive);
v2_oegm__D_0__D_1__D_2__D_3_tempShape.push_back(tenDimFiftyThree);
v2_oegm__D_0__D_1__D_2__D_3_tempShape.push_back(tenDimFiftyThree);
v2_oegm__D_0__D_1__D_2__D_3_tempShape.push_back(tenDimFive);
v2_oegm__D_0__D_1__D_2__D_3.ResizeTo( v2_oegm__D_0__D_1__D_2__D_3_tempShape );
MakeUniform( v2_oegm__D_0__D_1__D_2__D_3 );
DistTensor<T> v2_oegm_local( tmen::StringToTensorDist("[(),(),(),()]|(0,1,2,3)"), g );
//GatherAllModes( v2_oegm__D_0__D_1__D_2__D_3, v2_oegm_local );
// axppx3_temp has 4 dims
//	Starting distribution: [D0,D1,D2,D3] or _D_0__D_1__D_2__D_3
ObjShape axppx3_temp__D_0__D_1__D_2__D_3_tempShape;
axppx3_temp__D_0__D_1__D_2__D_3_tempShape.push_back(tenDimFive);
axppx3_temp__D_0__D_1__D_2__D_3_tempShape.push_back(tenDimFiftyThree);
axppx3_temp__D_0__D_1__D_2__D_3_tempShape.push_back(tenDimFiftyThree);
axppx3_temp__D_0__D_1__D_2__D_3_tempShape.push_back(tenDimFive);
axppx3_temp__D_0__D_1__D_2__D_3.ResizeTo( axppx3_temp__D_0__D_1__D_2__D_3_tempShape );
MakeUniform( axppx3_temp__D_0__D_1__D_2__D_3 );
DistTensor<T> axppx3_temp_local( tmen::StringToTensorDist("[(),(),(),()]|(0,1,2,3)"), g );
//GatherAllModes( axppx3_temp__D_0__D_1__D_2__D_3, axppx3_temp_local );
// cont1_temp has 4 dims
//	Starting distribution: [D0,D1,D2,D3] or _D_0__D_1__D_2__D_3
ObjShape cont1_temp__D_0__D_1__D_2__D_3_tempShape;
cont1_temp__D_0__D_1__D_2__D_3_tempShape.push_back(tenDimFiftyThree);
cont1_temp__D_0__D_1__D_2__D_3_tempShape.push_back(tenDimFiftyThree);
cont1_temp__D_0__D_1__D_2__D_3_tempShape.push_back(tenDimFive);
cont1_temp__D_0__D_1__D_2__D_3_tempShape.push_back(tenDimFive);
cont1_temp__D_0__D_1__D_2__D_3.ResizeTo( cont1_temp__D_0__D_1__D_2__D_3_tempShape );
MakeUniform( cont1_temp__D_0__D_1__D_2__D_3 );
DistTensor<T> cont1_temp_local( tmen::StringToTensorDist("[(),(),(),()]|(0,1,2,3)"), g );
//GatherAllModes( cont1_temp__D_0__D_1__D_2__D_3, cont1_temp_local );
// accum_temp has 4 dims
//	Starting distribution: [D0,D1,D2,D3] or _D_0__D_1__D_2__D_3
ObjShape accum_temp__D_0__D_1__D_2__D_3_tempShape;
accum_temp__D_0__D_1__D_2__D_3_tempShape.push_back(tenDimFiftyThree);
accum_temp__D_0__D_1__D_2__D_3_tempShape.push_back(tenDimFiftyThree);
accum_temp__D_0__D_1__D_2__D_3_tempShape.push_back(tenDimFive);
accum_temp__D_0__D_1__D_2__D_3_tempShape.push_back(tenDimFive);
accum_temp__D_0__D_1__D_2__D_3.ResizeTo( accum_temp__D_0__D_1__D_2__D_3_tempShape );
MakeUniform( accum_temp__D_0__D_1__D_2__D_3 );
DistTensor<T> accum_temp_local( tmen::StringToTensorDist("[(),(),(),()]|(0,1,2,3)"), g );
//GatherAllModes( accum_temp__D_0__D_1__D_2__D_3, accum_temp_local );
// scalar input has 0 dims
//	Starting distribution: [] | {0,1,2,3} or ___N_D_0_1_2_3
ObjShape E_MP3____N_D_0_1_2_3_tempShape;
E_MP3____N_D_0_1_2_3.ResizeTo( E_MP3____N_D_0_1_2_3_tempShape );
MakeUniform( E_MP3____N_D_0_1_2_3 );
DistTensor<T> E_MP3_local( tmen::StringToTensorDist("[]|(0,1,2,3)"), g );
//GatherAllModes( E_MP3____N_D_0_1_2_3, E_MP3_local );

//**** (out of 1)
	//------------------------------------//
//******************************
//* Load tensors
//******************************

//  double startLoadTime = mpi::Time();
//  DistTensor<T> epsilonA( tmen::StringToTensorDist("[(0)]|()"), g);
//  ObjShape epsilonAShape;
//  epsilonAShape.push_back(tenDimFiftyThree);
//  epsilonA.ResizeTo(epsilonAShape);
//  std::string epsilonAFilename = "data/ea";
//  printf("loading epsilonA\n");
//  Load_Tensor(epsilonA, epsilonAFilename);
//  //Print(epsilonA, "eps_a");
//
//  DistTensor<T> epsilonB( tmen::StringToTensorDist("[(0)]|()"), g);
//  ObjShape epsilonBShape;
//  epsilonBShape.push_back(tenDimFive);
//  epsilonB.ResizeTo(epsilonBShape);
//  std::string epsilonBFilename = "data/ei";
//  printf("loading epsilonB\n");
//  Load_Tensor(epsilonB, epsilonBFilename);
//  //Print(epsilonB, "eps_b");
//
//  DistTensor<T> D_abij( tmen::StringToTensorDist("[(0),(1),(2),(3)]|()"), g);
//  ObjShape D_abijShape;
//  D_abijShape.push_back(tenDimFiftyThree);
//  D_abijShape.push_back(tenDimFiftyThree);
//  D_abijShape.push_back(tenDimFive);
//  D_abijShape.push_back(tenDimFive);
//  D_abij.ResizeTo(D_abijShape);
//
//  DistTensor<T> V_abij( tmen::StringToTensorDist("[(0),(1),(2),(3)]|()"), g);
//  V_abij.ResizeTo(D_abijShape);
//  std::string v_abijFilename = "data/abij";
//  printf("loading V_abij\n");
//  Load_Tensor(V_abij, v_abijFilename);
//  //Print(V_abij, "v_abij");
//
//  std::string v_opmnFilename = "data/ijkl";
//  printf("loading v_opmn\n");
//  Load_Tensor(v_opmn__D_0__D_1__D_2__D_3, v_opmnFilename);
//  //Print(v_opmn__D_0__D_1__D_2__D_3, "v_opmn");
//
//  printf("loading 4\n");
//  std::string v_oegmFilename = "data/aijb";
//  printf("loading v_oegm\n");
//  Load_Tensor_aijb(v_oegm__D_0__D_1__D_2__D_3, v_oegmFilename);
//  //Print(v_oegm__D_0__D_1__D_2__D_3, "v_oegm");
//
//  printf("loading 5\n");
//  std::string v2_oegmFilename = "data/aibj";
//  printf("loading v2_oegm\n");
//  Load_Tensor_aijb(v2_oegm__D_0__D_1__D_2__D_3, v2_oegmFilename);
//  //Print(v2_oegm__D_0__D_1__D_2__D_3, "v2_oegm");
//
//  printf("loading 3\n");
//  std::string v_efghFilename = "data/abcd";
//  printf("loading v_efgh\n");
//  Load_Tensor(v_efgh__D_0__D_1__D_2__D_3, v_efghFilename);
//  //Print(v_efgh__D_0__D_1__D_2__D_3, "v_efgh");
//  double runLoadTime = mpi::Time() - startLoadTime;
//
//  printf("load time: %d\n", runLoadTime);
//  printf("elemScaling\n");
//  Form_D_abij(epsilonA, epsilonB, D_abij);
//  tmen::ElemScal(V_abij, D_abij, t_efmn__D_0__D_1__D_2__D_3);
//  //Print(t_efmn__D_0__D_1__D_2__D_3, "t_efmn");
//
//  printf("zeroing\n");
//  //Zero out the temporaries
//  Zero(axppx2_temp__D_0__D_1__D_2__D_3);
//  Zero(axppx3_temp__D_0__D_1__D_2__D_3);
//  Zero(cont1_temp__D_0__D_1__D_2__D_3);
//  Zero(accum_temp__D_0__D_1__D_2__D_3);
////  Zero(accum_temp_temp__D_0__D_1__D_2__D_3);
//  Zero(E_MP3____N_D_0_1_2_3);
//
//  printf("gathering\n");
//  //Form local versions of tensors
//  GatherAllModes( t_efmn__D_0__D_1__D_2__D_3, t_efmn_local );
//  GatherAllModes( v_opmn__D_0__D_1__D_2__D_3, v_opmn_local );
//  GatherAllModes( v_efgh__D_0__D_1__D_2__D_3, v_efgh_local );
//  GatherAllModes( v_oegm__D_0__D_1__D_2__D_3, v_oegm_local );
//  GatherAllModes( v2_oegm__D_0__D_1__D_2__D_3, v2_oegm_local );
//  GatherAllModes( axppx2_temp__D_0__D_1__D_2__D_3, axppx2_temp_local );
//  GatherAllModes( axppx3_temp__D_0__D_1__D_2__D_3, axppx3_temp_local );
//  GatherAllModes( cont1_temp__D_0__D_1__D_2__D_3, cont1_temp_local );
//  GatherAllModes( accum_temp__D_0__D_1__D_2__D_3, accum_temp_local );
//  GatherAllModes( E_MP3____N_D_0_1_2_3, E_MP3_local );
  
//******************************
//* Load tensors
//******************************
    double gflops;
    double startTime;
    double runTime;
//    if(commRank == 0)
//        std::cout << "starting\n";
//    printf("starting\n");
    mpi::Barrier(g.OwningComm());
    startTime = mpi::Time();
//	printf("started\n");

	   // t_efmn[D0,D1,D3,D2] <- t_efmn[D0,D1,D2,D3]
	   // t_efmn[D0,D1,D3,D2] <- t_efmn[D0,D1,D2,D3]
	   // t_efmn[D0,D1,D3,D2] <- t_efmn[D0,D1,D2,D3]
	t_efmn__D_0__D_1__D_3__D_2.AllToAllRedistFrom( t_efmn__D_0__D_1__D_2__D_3, modes_2_3, modes_3_2, modeArrayArray___2___3 );
	YAxpPx( 2.0, t_efmn__D_0__D_1__D_2__D_3, -1.0, t_efmn__D_0__D_1__D_3__D_2, perm_0_1_3_2, axppx2_temp__D_0__D_1__D_2__D_3 );
	t_efmn__D_0__D_1__D_3__D_2.Empty();
//	Print(axppx2_temp__D_0__D_1__D_2__D_3, "After YAxpPx: axppx2_temp__D_0__D_1__D_2__D_3");

	//------------------------------------//

//****
//**** (out of 1)
	//------------------------------------//

	ZAxpBy( 2.0, v_oegm__D_0__D_1__D_2__D_3, -1.0, v2_oegm__D_0__D_1__D_2__D_3, axppx3_temp__D_0__D_1__D_2__D_3 );
//	Print(axppx3_temp__D_0__D_1__D_2__D_3, "axppx3_temp__D_0__D_1__D_2__D_3");

	//------------------------------------//

//****
//**** (out of 3)
	//------------------------------------//
//	Print(axppx2_temp__D_0__D_1__D_2__D_3, "After YAxpPx: axppx2_temp__D_0__D_1__D_2__D_3");

	Scal( 0.0, cont1_temp__D_0__D_1__D_2__D_3 );
	//**** (out of 2)
	//**** Is real	0 shadows
		//Outputs:
		//  cont1_temp__D_0__D_1__D_2__D_3
//	v2_oegm_part2_1__D_1__D_0__D_2__D_3.AlignWith(v2_oegm__D_0__D_1__D_2__D_3);
//	v2_oegm_part2_1__D_1__D_0__D_3__D_2.AlignWith(v2_oegm__D_0__D_1__D_2__D_3);
//	t_efmn_part0_1__D_0__D_1__D_3__D_2.AlignWith(t_efmn__D_0__D_1__D_2__D_3);
//	v2_oegm_part2_1__S__D_0__S__D_2.AlignWith(v2_oegm__D_0__D_1__D_2__D_3);
//	t_efmn_part0_1__S__D_1__D_3__S.AlignWith(t_efmn__D_0__D_1__D_2__D_3);
	v2_oegm_part2_1__D_1__D_0__D_2__D_3.AlignWith(cont1_temp__D_0__D_1__D_2__D_3);
	v2_oegm_part2_1__D_1__D_0__D_3__D_2.AlignWith(cont1_temp__D_0__D_1__D_2__D_3);
	t_efmn_part0_1__D_0__D_1__D_3__D_2.AlignWith(cont1_temp__D_0__D_1__D_2__D_3);
	v2_oegm_part2_1__S__D_0__S__D_2.AlignWith(cont1_temp__D_0__D_1__D_2__D_3);
	t_efmn_part0_1__S__D_1__D_3__S.AlignWith(cont1_temp__D_0__D_1__D_2__D_3);

//	Print(v2_oegm__D_0__D_1__D_2__D_3, "Incoming v2_oegm");
//	Print(t_efmn__D_0__D_1__D_2__D_3, "Incoming t_efmn");
//	Print(cont1_temp__D_0__D_1__D_2__D_3, "Incoming cont1");

	PartitionDown(v2_oegm__D_0__D_1__D_2__D_3, v2_oegm_part2T__D_0__D_1__D_2__D_3, v2_oegm_part2B__D_0__D_1__D_2__D_3, 2, 0);
	PartitionDown(t_efmn__D_0__D_1__D_2__D_3, t_efmn_part0T__D_0__D_1__D_2__D_3, t_efmn_part0B__D_0__D_1__D_2__D_3, 0, 0);
	while(v2_oegm_part2T__D_0__D_1__D_2__D_3.Dimension(2) < v2_oegm__D_0__D_1__D_2__D_3.Dimension(2))
	{
		RepartitionDown
		( v2_oegm_part2T__D_0__D_1__D_2__D_3,  v2_oegm_part2_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       v2_oegm_part2_1__D_0__D_1__D_2__D_3,
		  v2_oegm_part2B__D_0__D_1__D_2__D_3, v2_oegm_part2_2__D_0__D_1__D_2__D_3, 2, 16 );
		RepartitionDown
		( t_efmn_part0T__D_0__D_1__D_2__D_3,  t_efmn_part0_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       t_efmn_part0_1__D_0__D_1__D_2__D_3,
		  t_efmn_part0B__D_0__D_1__D_2__D_3, t_efmn_part0_2__D_0__D_1__D_2__D_3, 0, 16 );
		//------------------------------------//

		   // v2_oegm_part2_1[D1,D0,D2,D3] <- v2_oegm_part2_1[D0,D1,D2,D3]
//		Print(v2_oegm_part2_1__D_0__D_1__D_2__D_3, "v2_oegm[D_0,D_1,D_2,D_3]");
		v2_oegm_part2_1__D_1__D_0__D_2__D_3.AlignWith(v2_oegm_part2_1__D_0__D_1__D_2__D_3);
		v2_oegm_part2_1__D_1__D_0__D_2__D_3.AllToAllRedistFrom( v2_oegm_part2_1__D_0__D_1__D_2__D_3, modes_0_1, modes_1_0, modeArrayArray___0___1 );
//		Print(v2_oegm_part2_1__D_1__D_0__D_2__D_3, "v2_oegm[D_1,D_0,D_2,D_3]");
		   // v2_oegm_part2_1[D1,D0,D3,D2] <- v2_oegm_part2_1[D1,D0,D2,D3]
		v2_oegm_part2_1__D_1__D_0__D_3__D_2.AllToAllRedistFrom( v2_oegm_part2_1__D_1__D_0__D_2__D_3, modes_2_3, modes_3_2, modeArrayArray___2___3 );
//		Print(v2_oegm_part2_1__D_1__D_0__D_3__D_2, "v2_oegm[D_1,D_0,D_3,D_2]");
		   // t_efmn_part0_1[D0,D1,D3,D2] <- t_efmn_part0_1[D0,D1,D2,D3]
		t_efmn_part0_1__D_0__D_1__D_3__D_2.AllToAllRedistFrom( t_efmn_part0_1__D_0__D_1__D_2__D_3, modes_2_3, modes_3_2, modeArrayArray___2___3 );
		   // v2_oegm_part2_1[*,D0,*,D2] <- v2_oegm_part2_1[D1,D0,D3,D2]
		v2_oegm_part2_1__S__D_0__S__D_2.AllGatherRedistFrom( v2_oegm_part2_1__D_1__D_0__D_3__D_2, modes_0_2, modeArrayArray___1___3 );
		   // t_efmn_part0_1[*,D1,D3,*] <- t_efmn_part0_1[D0,D1,D3,D2]
		t_efmn_part0_1__S__D_1__D_3__S.AllGatherRedistFrom( t_efmn_part0_1__D_0__D_1__D_3__D_2, modes_0_3, modeArrayArray___0___2 );
		   // 1.0 * v2_oegm_part2_1[*,D0,*,D2]_oegm * t_efmn_part0_1[*,D1,D3,*]_gfno + 0.0 * cont1_temp[D0,D1,D2,D3]_efmn
//		PrintData(v2_oegm_part2T__D_0__D_1__D_2__D_3, "v2_oegm_part2T");
//		PrintData(v2_oegm_part2B__D_0__D_1__D_2__D_3, "v2_oegm_part2B");
//		PrintData(v2_oegm_part2_0__D_0__D_1__D_2__D_3, "v2_oegm_part_0");
//		PrintData(v2_oegm_part2_1__D_0__D_1__D_2__D_3, "v2_oegm_part_1");
//		PrintData(v2_oegm_part2_2__D_0__D_1__D_2__D_3, "v2_oegm_part_2");
//		PrintData(v2_oegm_part2_1__S__D_0__S__D_2, "v2_oegm_part");
//		PrintData(t_efmn_part0_1__S__D_1__D_3__S, "t_efmn_part");
//		PrintData(cont1_temp__D_0__D_1__D_2__D_3, "cont1_temp_part");

//		PrintData(v2_oegm_part2_1__S__D_0__S__D_2, "contrib v2_oegm data");
//		Print(v2_oegm_part2_1__S__D_0__S__D_2, "contrib v2_oegm");
//		Print(t_efmn_part0_1__S__D_1__D_3__S, "contrib t_efmn");
		LocalContractAndLocalEliminate(1.0, v2_oegm_part2_1__S__D_0__S__D_2.LockedTensor(), indices_oegm, true,
			t_efmn_part0_1__S__D_1__D_3__S.LockedTensor(), indices_gfno, true,
			1.0, cont1_temp__D_0__D_1__D_2__D_3.Tensor(), indices_efmn, true);

		//------------------------------------//
		SlidePartitionDown
		( v2_oegm_part2T__D_0__D_1__D_2__D_3,  v2_oegm_part2_0__D_0__D_1__D_2__D_3,
		       v2_oegm_part2_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  v2_oegm_part2B__D_0__D_1__D_2__D_3, v2_oegm_part2_2__D_0__D_1__D_2__D_3, 2 );
		SlidePartitionDown
		( t_efmn_part0T__D_0__D_1__D_2__D_3,  t_efmn_part0_0__D_0__D_1__D_2__D_3,
		       t_efmn_part0_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  t_efmn_part0B__D_0__D_1__D_2__D_3, t_efmn_part0_2__D_0__D_1__D_2__D_3, 0 );

	}
	v2_oegm_part2_1__D_1__D_0__D_2__D_3.Empty();
	v2_oegm_part2_1__D_1__D_0__D_3__D_2.Empty();
	v2_oegm_part2_1__S__D_0__S__D_2.Empty();
	t_efmn_part0_1__D_0__D_1__D_3__D_2.Empty();
	t_efmn_part0_1__S__D_1__D_3__S.Empty();
	//****

//	Print(cont1_temp__D_0__D_1__D_2__D_3, "After Contract: cont1_temp__D_0__D_1__D_2__D_3");
	//------------------------------------//

//****
//**** (out of 1)
	//------------------------------------//

	   // cont1_temp[D0,D1,D3,D2] <- cont1_temp[D0,D1,D2,D3]
	cont1_temp__D_0__D_1__D_3__D_2.AllToAllRedistFrom( cont1_temp__D_0__D_1__D_2__D_3, modes_2_3, modes_3_2, modeArrayArray___2___3 );
	YAxpPx( 0.5, cont1_temp__D_0__D_1__D_2__D_3, 1.0, cont1_temp__D_0__D_1__D_3__D_2, perm_0_1_3_2, accum_temp__D_0__D_1__D_2__D_3 );
	cont1_temp__D_0__D_1__D_3__D_2.Empty();
//	Print(accum_temp__D_0__D_1__D_2__D_3, "After YAxpPx: accum_temp__D_0__D_1__D_2__D_3");

	//------------------------------------//

//****
//**** (out of 1)
	//------------------------------------//

	Scal( 0.0, E_MP3____N_D_0_1_2_3 );
//	Scal( 0.0, accum_temp__D_0__D_1__D_2__D_3 );
	//**** (out of 4)
	//**** Is real	0 shadows
		//Outputs:
		//  accum_temp__D_0__D_1__D_2__D_3
	axppx3_temp_part1_1__D_2__D_1__D_0__D_3.AlignWith(axppx3_temp__D_0__D_1__D_2__D_3);
	axppx3_temp_part1_1__D_2__S__D_0__S.AlignWith(axppx3_temp__D_0__D_1__D_2__D_3);
	accum_temp_part0_1__S__D_1__S__D_3__D_0__D_2.AlignModeWith(5, axppx3_temp__D_0__D_1__D_2__D_3, 0);
	accum_temp_part0_1__S__D_1__S__D_3__D_0__D_2.AlignModeWith(0, axppx3_temp__D_0__D_1__D_2__D_3, 1);
	accum_temp_part0_1__S__D_1__S__D_3__D_0__D_2.AlignModeWith(4, axppx3_temp__D_0__D_1__D_2__D_3, 2);
	accum_temp_part0_1__S__D_1__S__D_3__D_0__D_2.AlignModeWith(2, axppx3_temp__D_0__D_1__D_2__D_3, 3);
	accum_temp_part0_1__S__D_1__S__D_3__D_0__D_2.AlignModeWith(1, axppx2_temp__D_0__D_1__D_2__D_3, 1);
	accum_temp_part0_1__S__D_1__S__D_3__D_0__D_2.AlignModeWith(3, axppx2_temp__D_0__D_1__D_2__D_3, 3);

//	accum_temp_part0_1__D_0__D_1__D_2__D_3.AlignModeWith(0, accum_temp__D_0__D_1__D_2__D_3, 0);
//	accum_temp_part0_1__D_0__D_1__D_2__D_3.AlignModeWith(1, accum_temp__D_0__D_1__D_2__D_3, 1);
//	accum_temp_part0_1__D_0__D_1__D_2__D_3.AlignModeWith(2, accum_temp__D_0__D_1__D_2__D_3, 2);
//	accum_temp_part0_1__D_0__D_1__D_2__D_3.AlignModeWith(3, accum_temp__D_0__D_1__D_2__D_3, 3);

	PartitionDown(axppx3_temp__D_0__D_1__D_2__D_3, axppx3_temp_part1T__D_0__D_1__D_2__D_3, axppx3_temp_part1B__D_0__D_1__D_2__D_3, 1, 0);
	PartitionDown(accum_temp__D_0__D_1__D_2__D_3, accum_temp_part0T__D_0__D_1__D_2__D_3, accum_temp_part0B__D_0__D_1__D_2__D_3, 0, 0);
	while(accum_temp_part0T__D_0__D_1__D_2__D_3.Dimension(0) < accum_temp__D_0__D_1__D_2__D_3.Dimension(0))
	{
		RepartitionDown
		( axppx3_temp_part1T__D_0__D_1__D_2__D_3,  axppx3_temp_part1_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       axppx3_temp_part1_1__D_0__D_1__D_2__D_3,
		  axppx3_temp_part1B__D_0__D_1__D_2__D_3, axppx3_temp_part1_2__D_0__D_1__D_2__D_3, 1, 16 );
		RepartitionDown
		( accum_temp_part0T__D_0__D_1__D_2__D_3,  accum_temp_part0_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       accum_temp_part0_1__D_0__D_1__D_2__D_3,
		  accum_temp_part0B__D_0__D_1__D_2__D_3, accum_temp_part0_2__D_0__D_1__D_2__D_3, 0, 16 );
		//------------------------------------//

		tempShape = accum_temp_part0_1__D_0__D_1__D_2__D_3.Shape();
		tempShape.push_back( g.Shape()[0] );
		tempShape.push_back( g.Shape()[2] );
		accum_temp_part0_1__S__D_1__S__D_3__D_0__D_2.ResizeTo( tempShape );
		   // axppx3_temp_part1_1[D2,D1,D0,D3] <- axppx3_temp_part1_1[D0,D1,D2,D3]
		axppx3_temp_part1_1__D_2__D_1__D_0__D_3.AlignWith(axppx3_temp_part1_1__D_0__D_1__D_2__D_3);
		axppx3_temp_part1_1__D_2__D_1__D_0__D_3.AllToAllRedistFrom( axppx3_temp_part1_1__D_0__D_1__D_2__D_3, modes_0_2, modes_2_0, modeArrayArray___0___2 );
		   // axppx3_temp_part1_1[D2,*,D0,*] <- axppx3_temp_part1_1[D2,D1,D0,D3]
		axppx3_temp_part1_1__D_2__S__D_0__S.AllGatherRedistFrom( axppx3_temp_part1_1__D_2__D_1__D_0__D_3, modes_1_3, modeArrayArray___1___3 );
		   // 0.5 * axppx3_temp_part1_1[D2,*,D0,*]_oegm * axppx2_temp[D0,D1,D2,D3]_gfon + 0.0 * accum_temp_part0_1[*,D1,*,D3,D0,D2]_efmngo
		LocalContract(0.5, axppx3_temp_part1_1__D_2__S__D_0__S.LockedTensor(), indices_oegm, true,
			axppx2_temp__D_0__D_1__D_2__D_3.LockedTensor(), indices_gfon, true,
			0.0, accum_temp_part0_1__S__D_1__S__D_3__D_0__D_2.Tensor(), indices_efmngo, true);
		   // accum_temp_part0_1[D0,D1,D2,D3] <- accum_temp_part0_1[*,D1,*,D3,D0,D2] (with SumScatter on (D0)(D2))
		accum_temp_part0_1__D_0__D_1__D_2__D_3.ReduceScatterUpdateRedistFrom( accum_temp_part0_1__S__D_1__S__D_3__D_0__D_2, -1.0, modes_5_4, modes_2_0 );

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

	}
	axppx3_temp_part1_1__D_2__D_1__D_0__D_3.Empty();
	axppx3_temp_part1_1__D_2__S__D_0__S.Empty();
	accum_temp_part0_1__S__D_1__S__D_3__D_0__D_2.Empty();
//	Print(accum_temp__D_0__D_1__D_2__D_3, "After YxpBy: accum_temp__D_0__D_1__D_2__D_3");
	//****
	//**** (out of 10)
	//**** Is real	0 shadows
		//Outputs:
		//  accum_temp__D_0__D_1__D_2__D_3
	v_efgh_part2_1__D_0__D_1__S__S.AlignWith(v_efgh__D_0__D_1__D_2__D_3);
	t_efmn_part0_1__S__S__D_2__D_3.AlignWith(t_efmn__D_0__D_1__D_2__D_3);
	PartitionDown(v_efgh__D_0__D_1__D_2__D_3, v_efgh_part2T__D_0__D_1__D_2__D_3, v_efgh_part2B__D_0__D_1__D_2__D_3, 2, 0);
	PartitionDown(t_efmn__D_0__D_1__D_2__D_3, t_efmn_part0T__D_0__D_1__D_2__D_3, t_efmn_part0B__D_0__D_1__D_2__D_3, 0, 0);
	while(v_efgh_part2T__D_0__D_1__D_2__D_3.Dimension(2) < v_efgh__D_0__D_1__D_2__D_3.Dimension(2))
	{
		RepartitionDown
		( v_efgh_part2T__D_0__D_1__D_2__D_3,  v_efgh_part2_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       v_efgh_part2_1__D_0__D_1__D_2__D_3,
		  v_efgh_part2B__D_0__D_1__D_2__D_3, v_efgh_part2_2__D_0__D_1__D_2__D_3, 2, 16 );
		RepartitionDown
		( t_efmn_part0T__D_0__D_1__D_2__D_3,  t_efmn_part0_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       t_efmn_part0_1__D_0__D_1__D_2__D_3,
		  t_efmn_part0B__D_0__D_1__D_2__D_3, t_efmn_part0_2__D_0__D_1__D_2__D_3, 0, 16 );
		//------------------------------------//

		   // v_efgh_part2_1[D0,D1,*,*] <- v_efgh_part2_1[D0,D1,D2,D3]
		v_efgh_part2_1__D_0__D_1__S__S.AlignWith(v_efgh_part2_1__D_0__D_1__D_2__D_3);
		v_efgh_part2_1__D_0__D_1__S__S.AllGatherRedistFrom( v_efgh_part2_1__D_0__D_1__D_2__D_3, modes_2_3, modeArrayArray___2___3 );
		   // t_efmn_part0_1[*,*,D2,D3] <- t_efmn_part0_1[D0,D1,D2,D3]
		t_efmn_part0_1__S__S__D_2__D_3.AlignWith(t_efmn_part0_1__D_0__D_1__D_2__D_3);
		t_efmn_part0_1__S__S__D_2__D_3.AllGatherRedistFrom( t_efmn_part0_1__D_0__D_1__D_2__D_3, modes_0_1, modeArrayArray___0___1 );
		   // 0.5 * v_efgh_part2_1[D0,D1,*,*]_efgh * t_efmn_part0_1[*,*,D2,D3]_ghmn + 0.0 * accum_temp[D0,D1,D2,D3]_efmn
		LocalContractAndLocalEliminate(0.5, v_efgh_part2_1__D_0__D_1__S__S.LockedTensor(), indices_efgh, true,
			t_efmn_part0_1__S__S__D_2__D_3.LockedTensor(), indices_ghmn, true,
			1.0, accum_temp__D_0__D_1__D_2__D_3.Tensor(), indices_efmn, true);

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
	v_efgh_part2_1__D_0__D_1__S__S.Empty();
	t_efmn_part0_1__S__S__D_2__D_3.Empty();
//	Print(accum_temp__D_0__D_1__D_2__D_3, "After One Contract: accum_temp__D_0__D_1__D_2__D_3");
	//****
	//**** (out of 10)
	//**** Is real	0 shadows
		//Outputs:
		//  accum_temp__D_0__D_1__D_2__D_3
	v_opmn__S__S__D_2__D_3.AlignWith(t_efmn__D_0__D_1__D_2__D_3);
	t_efmn_part0_1__D_0__D_1__D_2__D_3.AlignWith(t_efmn__D_0__D_1__D_2__D_3);
	accum_temp_part0_1__D_0__D_1__D_2__D_3.AlignWith(accum_temp__D_0__D_1__D_2__D_3);
	PartitionDown(t_efmn__D_0__D_1__D_2__D_3, t_efmn_part0T__D_0__D_1__D_2__D_3, t_efmn_part0B__D_0__D_1__D_2__D_3, 0, 0);
	PartitionDown(accum_temp__D_0__D_1__D_2__D_3, accum_temp_part0T__D_0__D_1__D_2__D_3, accum_temp_part0B__D_0__D_1__D_2__D_3, 0, 0);
	while(accum_temp_part0T__D_0__D_1__D_2__D_3.Dimension(0) < accum_temp__D_0__D_1__D_2__D_3.Dimension(0))
	{
		RepartitionDown
		( t_efmn_part0T__D_0__D_1__D_2__D_3,  t_efmn_part0_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       t_efmn_part0_1__D_0__D_1__D_2__D_3,
		  t_efmn_part0B__D_0__D_1__D_2__D_3, t_efmn_part0_2__D_0__D_1__D_2__D_3, 0, 16 );
		RepartitionDown
		( accum_temp_part0T__D_0__D_1__D_2__D_3,  accum_temp_part0_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       accum_temp_part0_1__D_0__D_1__D_2__D_3,
		  accum_temp_part0B__D_0__D_1__D_2__D_3, accum_temp_part0_2__D_0__D_1__D_2__D_3, 0, 16 );
		//------------------------------------//

		   // v_opmn[*,*,D2,D3] <- v_opmn[D0,D1,D2,D3]
		v_opmn__S__S__D_2__D_3.AlignWith(v_opmn__D_0__D_1__D_2__D_3);
		v_opmn__S__S__D_2__D_3.AllGatherRedistFrom( v_opmn__D_0__D_1__D_2__D_3, modes_0_1, modeArrayArray___0___1 );
		   // t_efmn_part0_1[D0,D1,*,*] <- t_efmn_part0_1[D0,D1,D2,D3]
		t_efmn_part0_1__D_0__D_1__S__S.AlignWith(t_efmn_part0_1__D_0__D_1__D_2__D_3);
		t_efmn_part0_1__D_0__D_1__S__S.AllGatherRedistFrom( t_efmn_part0_1__D_0__D_1__D_2__D_3, modes_2_3, modeArrayArray___2___3 );
		   // 0.5 * v_opmn[*,*,D2,D3]_opmn * t_efmn_part0_1[D0,D1,*,*]_efop + 1.0 * accum_temp_part0_1[D0,D1,D2,D3]_efmn
		LocalContractAndLocalEliminate(0.5, v_opmn__S__S__D_2__D_3.LockedTensor(), indices_opmn, true,
			t_efmn_part0_1__D_0__D_1__S__S.LockedTensor(), indices_efop, true,
			1.0, accum_temp_part0_1__D_0__D_1__D_2__D_3.Tensor(), indices_efmn, true);

		//------------------------------------//
		SlidePartitionDown
		( t_efmn_part0T__D_0__D_1__D_2__D_3,  t_efmn_part0_0__D_0__D_1__D_2__D_3,
		       t_efmn_part0_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  t_efmn_part0B__D_0__D_1__D_2__D_3, t_efmn_part0_2__D_0__D_1__D_2__D_3, 0 );
		SlidePartitionDown
		( accum_temp_part0T__D_0__D_1__D_2__D_3,  accum_temp_part0_0__D_0__D_1__D_2__D_3,
		       accum_temp_part0_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  accum_temp_part0B__D_0__D_1__D_2__D_3, accum_temp_part0_2__D_0__D_1__D_2__D_3, 0 );

	}
	v_opmn__S__S__D_2__D_3.Empty();
	t_efmn_part0_1__D_0__D_1__S__S.Empty();
//	Print(accum_temp__D_0__D_1__D_2__D_3, "After Two Contract: accum_temp__D_0__D_1__D_2__D_3");
	//****
	//**** (out of 1)
	//**** Is real	0 shadows
		//Outputs:
		//  E_MP3____N_D_0_1_2_3

	//E_MP3__D_0__D_1__D_2__D_3.AlignWith(axppx2_temp__D_0__D_1__D_2__D_3);
	PartitionDown(axppx2_temp__D_0__D_1__D_2__D_3, axppx2_temp_part0T__D_0__D_1__D_2__D_3, axppx2_temp_part0B__D_0__D_1__D_2__D_3, 0, 0);
	PartitionDown(accum_temp__D_0__D_1__D_2__D_3, accum_temp_part0T__D_0__D_1__D_2__D_3, accum_temp_part0B__D_0__D_1__D_2__D_3, 0, 0);
	while(axppx2_temp_part0T__D_0__D_1__D_2__D_3.Dimension(0) < axppx2_temp__D_0__D_1__D_2__D_3.Dimension(0))
	{
		RepartitionDown
		( axppx2_temp_part0T__D_0__D_1__D_2__D_3,  axppx2_temp_part0_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       axppx2_temp_part0_1__D_0__D_1__D_2__D_3,
		  axppx2_temp_part0B__D_0__D_1__D_2__D_3, axppx2_temp_part0_2__D_0__D_1__D_2__D_3, 0, 16 );
		RepartitionDown
		( accum_temp_part0T__D_0__D_1__D_2__D_3,  accum_temp_part0_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       accum_temp_part0_1__D_0__D_1__D_2__D_3,
		  accum_temp_part0B__D_0__D_1__D_2__D_3, accum_temp_part0_2__D_0__D_1__D_2__D_3, 0, 16 );
		//------------------------------------//

		tempShape = E_MP3____N_D_0_1_2_3.Shape();
		tempShape.push_back( g.Shape()[0] );
		tempShape.push_back( g.Shape()[1] );
		tempShape.push_back( g.Shape()[2] );
		tempShape.push_back( g.Shape()[3] );
		E_MP3__D_0__D_1__D_2__D_3.ResizeTo( tempShape );
		   // 2.0 * axppx2_temp_part0_1[D0,D1,D2,D3]_efmn * accum_temp_part0_1[D0,D1,D2,D3]_efmn + 0.0 * E_MP3[D0,D1,D2,D3]_efmn
		LocalContract(2.0, axppx2_temp_part0_1__D_0__D_1__D_2__D_3.LockedTensor(), indices_efmn, true,
			accum_temp_part0_1__D_0__D_1__D_2__D_3.LockedTensor(), indices_efmn, true,
			0.0, E_MP3__D_0__D_1__D_2__D_3.Tensor(), indices_efmn, true);
		   // E_MP3[] | {0,1,2,3} <- E_MP3[D0,D1,D2,D3] (with SumScatter on (D0)(D1)(D2)(D3))
		E_MP3____N_D_0_1_2_3.ReduceToOneUpdateRedistFrom( E_MP3__D_0__D_1__D_2__D_3, 1.0, modes_0_1_2_3 );
//		PrintData(E_MP3____N_D_0_1_2_3, "EMP3 accum data");

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
//Print(E_MP3____N_D_0_1_2_3, "E_MP3");

	//------------------------------------//

//****

/*****************************************/
    mpi::Barrier(g.OwningComm());
    runTime = mpi::Time() - startTime;
    double flops = pow(tenDimFive, 2) * pow(tenDimFiftyThree, 2) * ( 11 + 2 * pow(tenDimFive + tenDimFiftyThree, 2));
    gflops = flops / (1e9 * runTime);

	//****

	//------------------------------------//

//****

//Print(E_MP3____N_D_0_1_2_3, "E_MP3");
  

  //****
    if(commRank == 0){
        std::cout << "RUNTIME " << runTime << std::endl;
        std::cout << "FLOPS " << flops << std::endl;
        std::cout << "GFLOPS " << gflops << std::endl;
    }

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



        DistTensorTest<double>( g, args.tenDimFive, args.tenDimFiftyThree );

    }
    catch( std::exception& e ) { ReportException(e); }

    Finalize();
    //printf("Completed\n");
    return 0;
}
