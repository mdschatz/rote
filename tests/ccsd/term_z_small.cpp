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

template<typename T>
void PrintLocalSizes(const DistTensor<T>& A) {
    const Int commRank = mpi::CommRank(mpi::COMM_WORLD);
    if (commRank == 0) {
        for (Unsigned i = 0; i < A.Order(); ++i) {
            cout << i << " is " << A.LocalDimension(i) << endl;
        }
    }
}

void Usage() {
    std::cout << "./DistTensor <gridDim0> <gridDim1> ... \n";
    std::cout << "<gridDimK>   : dimension of mode-K of grid\n";
}

typedef struct Arguments {
    ObjShape gridShape;
    Unsigned nProcs;
    Unsigned n_o;
    Unsigned n_v;
    Unsigned blkSize;
    Unsigned testIter;
} Params;

void ProcessInput(int argc, char** const argv, Params& args) {
    Unsigned i;
    Unsigned argCount = 0;
    if (argCount + 1 >= argc) {
        std::cerr << "Missing required gridOrder argument\n";
        Usage();
        throw ArgException();
    }

    if (argCount + GRIDORDER >= argc) {
        std::cerr << "Missing required grid dimensions\n";
        Usage();
        throw ArgException();
    }

    args.gridShape.resize(GRIDORDER);
    args.nProcs = 1;
    for (int i = 0; i < GRIDORDER; i++) {
        int gridDim = atoi(argv[++argCount]);
        if (gridDim <= 0) {
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
    args.testIter = atoi(argv[++argCount]);
}

template<typename T>
void DistTensorTest(const Grid& g, Unsigned n_o, Unsigned n_v,
        Unsigned blkSize, Unsigned testIter) {
#ifndef RELEASE
    CallStackEntry entry("DistTensorTest");
#endif
    Unsigned i;
    const Int commRank = mpi::CommRank(mpi::COMM_WORLD);

//START_DECL
ObjShape overwrite_tmpShape_z_small;
TensorDistribution dist__S__S = tmen::StringToTensorDist("[(),()]");
TensorDistribution dist__S__D_2_3 = tmen::StringToTensorDist("[(),(2,3)]");
TensorDistribution dist__D_0__S__D_1__D_2__D_3 = tmen::StringToTensorDist("[(0),(),(1),(2),(3)]");
TensorDistribution dist__D_0__S__D_2__D_3__D_1 = tmen::StringToTensorDist("[(0),(),(2),(3),(1)]");
TensorDistribution dist__D_0__D_1__D_2__D_3 = tmen::StringToTensorDist("[(0),(1),(2),(3)]");
TensorDistribution dist__D_0__D_1__D_3__D_2 = tmen::StringToTensorDist("[(0),(1),(3),(2)]");
TensorDistribution dist__D_0__D_3__D_2__D_1 = tmen::StringToTensorDist("[(0),(3),(2),(1)]");
TensorDistribution dist__D_1__D_0__D_2__D_3 = tmen::StringToTensorDist("[(1),(0),(2),(3)]");
TensorDistribution dist__D_2__D_3__S__D_1 = tmen::StringToTensorDist("[(2),(3),(),(1)]");
TensorDistribution dist__D_0_1__S__D_2_3__S = tmen::StringToTensorDist("[(0,1),(),(2,3),()]");
TensorDistribution dist__D_0_1__S__D_2__D_3 = tmen::StringToTensorDist("[(0,1),(),(2),(3)]");
TensorDistribution dist__D_0_1__S = tmen::StringToTensorDist("[(0,1),()]");
TensorDistribution dist__D_0_1__D_2_3 = tmen::StringToTensorDist("[(0,1),(2,3)]");
Permutation perm_0_1( 2 );
perm_0_1[0] = 0;
perm_0_1[1] = 1;
Permutation perm_0_1_2_3( 4 );
perm_0_1_2_3[0] = 0;
perm_0_1_2_3[1] = 1;
perm_0_1_2_3[2] = 2;
perm_0_1_2_3[3] = 3;
Permutation perm_0_1_2_3_4( 5 );
perm_0_1_2_3_4[0] = 0;
perm_0_1_2_3_4[1] = 1;
perm_0_1_2_3_4[2] = 2;
perm_0_1_2_3_4[3] = 3;
perm_0_1_2_3_4[4] = 4;
Permutation perm_0_1_3_2( 4 );
perm_0_1_3_2[0] = 0;
perm_0_1_3_2[1] = 1;
perm_0_1_3_2[2] = 3;
perm_0_1_3_2[3] = 2;
Permutation perm_0_2_1_3( 4 );
perm_0_2_1_3[0] = 0;
perm_0_2_1_3[1] = 2;
perm_0_2_1_3[2] = 1;
perm_0_2_1_3[3] = 3;
Permutation perm_0_2_3_1( 4 );
perm_0_2_3_1[0] = 0;
perm_0_2_3_1[1] = 2;
perm_0_2_3_1[2] = 3;
perm_0_2_3_1[3] = 1;
Permutation perm_1_0( 2 );
perm_1_0[0] = 1;
perm_1_0[1] = 0;
Permutation perm_1_0_2_3( 4 );
perm_1_0_2_3[0] = 1;
perm_1_0_2_3[1] = 0;
perm_1_0_2_3[2] = 2;
perm_1_0_2_3[3] = 3;
Permutation perm_1_0_3_4_2( 5 );
perm_1_0_3_4_2[0] = 1;
perm_1_0_3_4_2[1] = 0;
perm_1_0_3_4_2[2] = 3;
perm_1_0_3_4_2[3] = 4;
perm_1_0_3_4_2[4] = 2;
Permutation perm_2_0_1_3( 4 );
perm_2_0_1_3[0] = 2;
perm_2_0_1_3[1] = 0;
perm_2_0_1_3[2] = 1;
perm_2_0_1_3[3] = 3;
Permutation perm_2_3_1_0( 4 );
perm_2_3_1_0[0] = 2;
perm_2_3_1_0[1] = 3;
perm_2_3_1_0[2] = 1;
perm_2_3_1_0[3] = 0;
ModeArray modes( 0 );
ModeArray modes_0( 1 );
modes_0[0] = 0;
ModeArray modes_0_1( 2 );
modes_0_1[0] = 0;
modes_0_1[1] = 1;
ModeArray modes_0_1_2_3( 4 );
modes_0_1_2_3[0] = 0;
modes_0_1_2_3[1] = 1;
modes_0_1_2_3[2] = 2;
modes_0_1_2_3[3] = 3;
ModeArray modes_0_1_3( 3 );
modes_0_1_3[0] = 0;
modes_0_1_3[1] = 1;
modes_0_1_3[2] = 3;
ModeArray modes_0_1_3_2( 4 );
modes_0_1_3_2[0] = 0;
modes_0_1_3_2[1] = 1;
modes_0_1_3_2[2] = 3;
modes_0_1_3_2[3] = 2;
ModeArray modes_0_2( 2 );
modes_0_2[0] = 0;
modes_0_2[1] = 2;
ModeArray modes_1( 1 );
modes_1[0] = 1;
ModeArray modes_1_0_2_3( 4 );
modes_1_0_2_3[0] = 1;
modes_1_0_2_3[1] = 0;
modes_1_0_2_3[2] = 2;
modes_1_0_2_3[3] = 3;
ModeArray modes_1_3( 2 );
modes_1_3[0] = 1;
modes_1_3[1] = 3;
ModeArray modes_2_3( 2 );
modes_2_3[0] = 2;
modes_2_3[1] = 3;
ModeArray modes_2_3_1( 3 );
modes_2_3_1[0] = 2;
modes_2_3_1[1] = 3;
modes_2_3_1[2] = 1;
ModeArray modes_3( 1 );
modes_3[0] = 3;
ModeArray modes_4_3_2( 3 );
modes_4_3_2[0] = 4;
modes_4_3_2[1] = 3;
modes_4_3_2[2] = 2;
IndexArray indices_aefm( 4 );
indices_aefm[0] = 'a';
indices_aefm[1] = 'e';
indices_aefm[2] = 'f';
indices_aefm[3] = 'm';
IndexArray indices_ai( 2 );
indices_ai[0] = 'a';
indices_ai[1] = 'i';
IndexArray indices_aiefm( 5 );
indices_aiefm[0] = 'a';
indices_aiefm[1] = 'i';
indices_aiefm[2] = 'e';
indices_aiefm[3] = 'f';
indices_aiefm[4] = 'm';
IndexArray indices_aime( 4 );
indices_aime[0] = 'a';
indices_aime[1] = 'i';
indices_aime[2] = 'm';
indices_aime[3] = 'e';
IndexArray indices_efmi( 4 );
indices_efmi[0] = 'e';
indices_efmi[1] = 'f';
indices_efmi[2] = 'm';
indices_efmi[3] = 'i';
IndexArray indices_ia( 2 );
indices_ia[0] = 'i';
indices_ia[1] = 'a';
IndexArray indices_iamne( 5 );
indices_iamne[0] = 'i';
indices_iamne[1] = 'a';
indices_iamne[2] = 'm';
indices_iamne[3] = 'n';
indices_iamne[4] = 'e';
IndexArray indices_im( 2 );
indices_im[0] = 'i';
indices_im[1] = 'm';
IndexArray indices_imne( 4 );
indices_imne[0] = 'i';
indices_imne[1] = 'm';
indices_imne[2] = 'n';
indices_imne[3] = 'e';
IndexArray indices_ma( 2 );
indices_ma[0] = 'm';
indices_ma[1] = 'a';
IndexArray indices_me( 2 );
indices_me[0] = 'm';
indices_me[1] = 'e';
IndexArray indices_mnea( 4 );
indices_mnea[0] = 'm';
indices_mnea[1] = 'n';
indices_mnea[2] = 'e';
indices_mnea[3] = 'a';
	//G_mi[D01,D23]
DistTensor<double> G_mi__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//G_mi_lvl0_part1B[D01,D23]
DistTensor<double> G_mi_lvl0_part1B__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//G_mi_lvl0_part1T[D01,D23]
DistTensor<double> G_mi_lvl0_part1T__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//G_mi_lvl1_part1B[D01,D23]
DistTensor<double> G_mi_lvl1_part1B__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//G_mi_lvl1_part1T[D01,D23]
DistTensor<double> G_mi_lvl1_part1T__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//G_mi_lvl1_part1_0[D01,D23]
DistTensor<double> G_mi_lvl1_part1_0__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//G_mi_lvl1_part1_1[D01,D23]
DistTensor<double> G_mi_lvl1_part1_1__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//G_mi_lvl1_part1_1_lvl1_part0B[D01,D23]
DistTensor<double> G_mi_lvl1_part1_1_lvl1_part0B__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//G_mi_lvl1_part1_1_lvl1_part0T[D01,D23]
DistTensor<double> G_mi_lvl1_part1_1_lvl1_part0T__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//G_mi_lvl1_part1_1_lvl2_part0B[D01,D23]
DistTensor<double> G_mi_lvl1_part1_1_lvl2_part0B__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//G_mi_lvl1_part1_1_lvl2_part0T[D01,D23]
DistTensor<double> G_mi_lvl1_part1_1_lvl2_part0T__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//G_mi_lvl1_part1_1_lvl2_part0_0[D01,D23]
DistTensor<double> G_mi_lvl1_part1_1_lvl2_part0_0__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//G_mi_lvl1_part1_1_lvl2_part0_1[D01,D23]
DistTensor<double> G_mi_lvl1_part1_1_lvl2_part0_1__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//G_mi_lvl1_part1_1_lvl2_part0_1[*,D23]
DistTensor<double> G_mi_lvl1_part1_1_lvl2_part0_1_perm10__D_2_3__S( dist__S__D_2_3, g );
G_mi_lvl1_part1_1_lvl2_part0_1_perm10__D_2_3__S.SetLocalPermutation( perm_1_0 );
	//G_mi_lvl1_part1_1_lvl2_part0_2[D01,D23]
DistTensor<double> G_mi_lvl1_part1_1_lvl2_part0_2__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//G_mi_lvl1_part1_2[D01,D23]
DistTensor<double> G_mi_lvl1_part1_2__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//H_me[D01,D23]
DistTensor<double> H_me__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//H_me_lvl1_part0B[D01,D23]
DistTensor<double> H_me_lvl1_part0B__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//H_me_lvl1_part0T[D01,D23]
DistTensor<double> H_me_lvl1_part0T__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//H_me_lvl2_part0B[D01,D23]
DistTensor<double> H_me_lvl2_part0B__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//H_me_lvl2_part0T[D01,D23]
DistTensor<double> H_me_lvl2_part0T__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//H_me_lvl2_part0_0[D01,D23]
DistTensor<double> H_me_lvl2_part0_0__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//H_me_lvl2_part0_1[D01,D23]
DistTensor<double> H_me_lvl2_part0_1__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//H_me_lvl2_part0_1[*,*]
DistTensor<double> H_me_lvl2_part0_1__S__S( dist__S__S, g );
	//H_me_lvl2_part0_2[D01,D23]
DistTensor<double> H_me_lvl2_part0_2__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//T_bfnj[D0,D1,D2,D3]
DistTensor<double> T_bfnj__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//T_bfnj_lvl0_part2B[D0,D1,D2,D3]
DistTensor<double> T_bfnj_lvl0_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//T_bfnj_lvl0_part2T[D0,D1,D2,D3]
DistTensor<double> T_bfnj_lvl0_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//T_bfnj_lvl0_part3B[D0,D1,D2,D3]
DistTensor<double> T_bfnj_lvl0_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//T_bfnj_lvl0_part3T[D0,D1,D2,D3]
DistTensor<double> T_bfnj_lvl0_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//T_bfnj_lvl1_part2B[D0,D1,D2,D3]
DistTensor<double> T_bfnj_lvl1_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//T_bfnj_lvl1_part2T[D0,D1,D2,D3]
DistTensor<double> T_bfnj_lvl1_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//T_bfnj_lvl1_part2_0[D0,D1,D2,D3]
DistTensor<double> T_bfnj_lvl1_part2_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//T_bfnj_lvl1_part2_1[D0,D1,D2,D3]
DistTensor<double> T_bfnj_lvl1_part2_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//T_bfnj_lvl1_part2_1_lvl1_part3B[D0,D1,D2,D3]
DistTensor<double> T_bfnj_lvl1_part2_1_lvl1_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//T_bfnj_lvl1_part2_1_lvl1_part3T[D0,D1,D2,D3]
DistTensor<double> T_bfnj_lvl1_part2_1_lvl1_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//T_bfnj_lvl1_part2_1_lvl2_part3B[D0,D1,D2,D3]
DistTensor<double> T_bfnj_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//T_bfnj_lvl1_part2_1_lvl2_part3T[D0,D1,D2,D3]
DistTensor<double> T_bfnj_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//T_bfnj_lvl1_part2_1_lvl2_part3_0[D0,D1,D2,D3]
DistTensor<double> T_bfnj_lvl1_part2_1_lvl2_part3_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//T_bfnj_lvl1_part2_1_lvl2_part3_1[D0,D1,D2,D3]
DistTensor<double> T_bfnj_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//T_bfnj_lvl1_part2_1_lvl2_part3_2[D0,D1,D2,D3]
DistTensor<double> T_bfnj_lvl1_part2_1_lvl2_part3_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//T_bfnj_lvl1_part2_2[D0,D1,D2,D3]
DistTensor<double> T_bfnj_lvl1_part2_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//T_bfnj_lvl1_part3B[D0,D1,D2,D3]
DistTensor<double> T_bfnj_lvl1_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//T_bfnj_lvl1_part3T[D0,D1,D2,D3]
DistTensor<double> T_bfnj_lvl1_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//T_bfnj_lvl1_part3_0[D0,D1,D2,D3]
DistTensor<double> T_bfnj_lvl1_part3_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//T_bfnj_lvl1_part3_1[D0,D1,D2,D3]
DistTensor<double> T_bfnj_lvl1_part3_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//T_bfnj_lvl1_part3_1_lvl1_part2B[D0,D1,D2,D3]
DistTensor<double> T_bfnj_lvl1_part3_1_lvl1_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//T_bfnj_lvl1_part3_1_lvl1_part2T[D0,D1,D2,D3]
DistTensor<double> T_bfnj_lvl1_part3_1_lvl1_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//T_bfnj_lvl1_part3_1_lvl2_part2B[D0,D1,D2,D3]
DistTensor<double> T_bfnj_lvl1_part3_1_lvl2_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//T_bfnj_lvl1_part3_1_lvl2_part2T[D0,D1,D2,D3]
DistTensor<double> T_bfnj_lvl1_part3_1_lvl2_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//T_bfnj_lvl1_part3_1_lvl2_part2_0[D0,D1,D2,D3]
DistTensor<double> T_bfnj_lvl1_part3_1_lvl2_part2_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//T_bfnj_lvl1_part3_1_lvl2_part2_1[D0,D1,D2,D3]
DistTensor<double> T_bfnj_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//T_bfnj_lvl1_part3_1_lvl2_part2_1[D0,D1,D3,D2]
DistTensor<double> T_bfnj_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_3__D_2( dist__D_0__D_1__D_3__D_2, g );
	//T_bfnj_lvl1_part3_1_lvl2_part2_2[D0,D1,D2,D3]
DistTensor<double> T_bfnj_lvl1_part3_1_lvl2_part2_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//T_bfnj_lvl1_part3_2[D0,D1,D2,D3]
DistTensor<double> T_bfnj_lvl1_part3_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//T_bfnj_lvl2_part2B[D0,D1,D2,D3]
DistTensor<double> T_bfnj_lvl2_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//T_bfnj_lvl2_part2T[D0,D1,D2,D3]
DistTensor<double> T_bfnj_lvl2_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//T_bfnj_lvl2_part2_0[D0,D1,D2,D3]
DistTensor<double> T_bfnj_lvl2_part2_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//T_bfnj_lvl2_part2_1[D0,D1,D2,D3]
DistTensor<double> T_bfnj_lvl2_part2_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> T_bfnj_lvl2_part2_1_perm2310__D_2__D_3__D_1__D_0( dist__D_0__D_1__D_2__D_3, g );
T_bfnj_lvl2_part2_1_perm2310__D_2__D_3__D_1__D_0.SetLocalPermutation( perm_2_3_1_0 );
	//T_bfnj_lvl2_part2_2[D0,D1,D2,D3]
DistTensor<double> T_bfnj_lvl2_part2_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Tau_efmn[D0,D1,D2,D3]
DistTensor<double> Tau_efmn__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Tau_efmn_lvl0_part3B[D0,D1,D2,D3]
DistTensor<double> Tau_efmn_lvl0_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Tau_efmn_lvl0_part3T[D0,D1,D2,D3]
DistTensor<double> Tau_efmn_lvl0_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Tau_efmn_lvl1_part3B[D0,D1,D2,D3]
DistTensor<double> Tau_efmn_lvl1_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Tau_efmn_lvl1_part3T[D0,D1,D2,D3]
DistTensor<double> Tau_efmn_lvl1_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Tau_efmn_lvl1_part3_0[D0,D1,D2,D3]
DistTensor<double> Tau_efmn_lvl1_part3_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Tau_efmn_lvl1_part3_1[D0,D1,D2,D3]
DistTensor<double> Tau_efmn_lvl1_part3_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Tau_efmn_lvl1_part3_1_lvl1_part2B[D0,D1,D2,D3]
DistTensor<double> Tau_efmn_lvl1_part3_1_lvl1_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Tau_efmn_lvl1_part3_1_lvl1_part2T[D0,D1,D2,D3]
DistTensor<double> Tau_efmn_lvl1_part3_1_lvl1_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Tau_efmn_lvl1_part3_1_lvl2_part2B[D0,D1,D2,D3]
DistTensor<double> Tau_efmn_lvl1_part3_1_lvl2_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Tau_efmn_lvl1_part3_1_lvl2_part2T[D0,D1,D2,D3]
DistTensor<double> Tau_efmn_lvl1_part3_1_lvl2_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Tau_efmn_lvl1_part3_1_lvl2_part2_0[D0,D1,D2,D3]
DistTensor<double> Tau_efmn_lvl1_part3_1_lvl2_part2_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Tau_efmn_lvl1_part3_1_lvl2_part2_1[D0,D1,D2,D3]
DistTensor<double> Tau_efmn_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Tau_efmn_lvl1_part3_1_lvl2_part2_1[D0,D3,D2,D1]
DistTensor<double> Tau_efmn_lvl1_part3_1_lvl2_part2_1__D_0__D_3__D_2__D_1( dist__D_0__D_3__D_2__D_1, g );
	//Tau_efmn_lvl1_part3_1_lvl2_part2_1[D2,D3,*,D1]
DistTensor<double> Tau_efmn_lvl1_part3_1_lvl2_part2_1_perm0132__D_2__D_3__D_1__S( dist__D_2__D_3__S__D_1, g );
Tau_efmn_lvl1_part3_1_lvl2_part2_1_perm0132__D_2__D_3__D_1__S.SetLocalPermutation( perm_0_1_3_2 );
	//Tau_efmn_lvl1_part3_1_lvl2_part2_2[D0,D1,D2,D3]
DistTensor<double> Tau_efmn_lvl1_part3_1_lvl2_part2_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Tau_efmn_lvl1_part3_2[D0,D1,D2,D3]
DistTensor<double> Tau_efmn_lvl1_part3_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//U_mnie[D0,D1,D2,D3]
DistTensor<double> U_mnie__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//U_mnie_lvl0_part0B[D0,D1,D2,D3]
DistTensor<double> U_mnie_lvl0_part0B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//U_mnie_lvl0_part0T[D0,D1,D2,D3]
DistTensor<double> U_mnie_lvl0_part0T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//U_mnie_lvl0_part1B[D0,D1,D2,D3]
DistTensor<double> U_mnie_lvl0_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//U_mnie_lvl0_part1T[D0,D1,D2,D3]
DistTensor<double> U_mnie_lvl0_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//U_mnie_lvl1_part0B[D0,D1,D2,D3]
DistTensor<double> U_mnie_lvl1_part0B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//U_mnie_lvl1_part0T[D0,D1,D2,D3]
DistTensor<double> U_mnie_lvl1_part0T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//U_mnie_lvl1_part0_0[D0,D1,D2,D3]
DistTensor<double> U_mnie_lvl1_part0_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//U_mnie_lvl1_part0_1[D0,D1,D2,D3]
DistTensor<double> U_mnie_lvl1_part0_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//U_mnie_lvl1_part0_1_lvl1_part1B[D0,D1,D2,D3]
DistTensor<double> U_mnie_lvl1_part0_1_lvl1_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//U_mnie_lvl1_part0_1_lvl1_part1T[D0,D1,D2,D3]
DistTensor<double> U_mnie_lvl1_part0_1_lvl1_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//U_mnie_lvl1_part0_1_lvl2_part1B[D0,D1,D2,D3]
DistTensor<double> U_mnie_lvl1_part0_1_lvl2_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//U_mnie_lvl1_part0_1_lvl2_part1T[D0,D1,D2,D3]
DistTensor<double> U_mnie_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//U_mnie_lvl1_part0_1_lvl2_part1_0[D0,D1,D2,D3]
DistTensor<double> U_mnie_lvl1_part0_1_lvl2_part1_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//U_mnie_lvl1_part0_1_lvl2_part1_1[D0,D1,D2,D3]
DistTensor<double> U_mnie_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//U_mnie_lvl1_part0_1_lvl2_part1_2[D0,D1,D2,D3]
DistTensor<double> U_mnie_lvl1_part0_1_lvl2_part1_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//U_mnie_lvl1_part0_2[D0,D1,D2,D3]
DistTensor<double> U_mnie_lvl1_part0_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//U_mnie_lvl1_part1B[D0,D1,D2,D3]
DistTensor<double> U_mnie_lvl1_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//U_mnie_lvl1_part1T[D0,D1,D2,D3]
DistTensor<double> U_mnie_lvl1_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//U_mnie_lvl1_part1_0[D0,D1,D2,D3]
DistTensor<double> U_mnie_lvl1_part1_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//U_mnie_lvl1_part1_1[D0,D1,D2,D3]
DistTensor<double> U_mnie_lvl1_part1_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//U_mnie_lvl1_part1_1_lvl1_part0B[D0,D1,D2,D3]
DistTensor<double> U_mnie_lvl1_part1_1_lvl1_part0B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//U_mnie_lvl1_part1_1_lvl1_part0T[D0,D1,D2,D3]
DistTensor<double> U_mnie_lvl1_part1_1_lvl1_part0T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//U_mnie_lvl1_part1_1_lvl2_part0B[D0,D1,D2,D3]
DistTensor<double> U_mnie_lvl1_part1_1_lvl2_part0B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//U_mnie_lvl1_part1_1_lvl2_part0T[D0,D1,D2,D3]
DistTensor<double> U_mnie_lvl1_part1_1_lvl2_part0T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//U_mnie_lvl1_part1_1_lvl2_part0_0[D0,D1,D2,D3]
DistTensor<double> U_mnie_lvl1_part1_1_lvl2_part0_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//U_mnie_lvl1_part1_1_lvl2_part0_1[D0,D1,D2,D3]
DistTensor<double> U_mnie_lvl1_part1_1_lvl2_part0_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//U_mnie_lvl1_part1_1_lvl2_part0_1[D1,D0,D2,D3]
DistTensor<double> U_mnie_lvl1_part1_1_lvl2_part0_1__D_1__D_0__D_2__D_3( dist__D_1__D_0__D_2__D_3, g );
	//U_mnie_lvl1_part1_1_lvl2_part0_2[D0,D1,D2,D3]
DistTensor<double> U_mnie_lvl1_part1_1_lvl2_part0_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//U_mnie_lvl1_part1_2[D0,D1,D2,D3]
DistTensor<double> U_mnie_lvl1_part1_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//r_bmfe[D0,D1,D2,D3]
DistTensor<double> r_bmfe__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//r_bmfe_lvl0_part0B[D0,D1,D2,D3]
DistTensor<double> r_bmfe_lvl0_part0B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//r_bmfe_lvl0_part0T[D0,D1,D2,D3]
DistTensor<double> r_bmfe_lvl0_part0T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//r_bmfe_lvl1_part0B[D0,D1,D2,D3]
DistTensor<double> r_bmfe_lvl1_part0B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//r_bmfe_lvl1_part0T[D0,D1,D2,D3]
DistTensor<double> r_bmfe_lvl1_part0T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//r_bmfe_lvl1_part0_0[D0,D1,D2,D3]
DistTensor<double> r_bmfe_lvl1_part0_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//r_bmfe_lvl1_part0_1[D0,D1,D2,D3]
DistTensor<double> r_bmfe_lvl1_part0_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//r_bmfe_lvl1_part0_1_lvl1_part1B[D0,D1,D2,D3]
DistTensor<double> r_bmfe_lvl1_part0_1_lvl1_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//r_bmfe_lvl1_part0_1_lvl1_part1T[D0,D1,D2,D3]
DistTensor<double> r_bmfe_lvl1_part0_1_lvl1_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//r_bmfe_lvl1_part0_1_lvl2_part1B[D0,D1,D2,D3]
DistTensor<double> r_bmfe_lvl1_part0_1_lvl2_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//r_bmfe_lvl1_part0_1_lvl2_part1T[D0,D1,D2,D3]
DistTensor<double> r_bmfe_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//r_bmfe_lvl1_part0_1_lvl2_part1_0[D0,D1,D2,D3]
DistTensor<double> r_bmfe_lvl1_part0_1_lvl2_part1_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//r_bmfe_lvl1_part0_1_lvl2_part1_1[D0,D1,D2,D3]
DistTensor<double> r_bmfe_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//r_bmfe_lvl1_part0_1_lvl2_part1_1[D0,D1,D3,D2]
DistTensor<double> r_bmfe_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_3__D_2( dist__D_0__D_1__D_3__D_2, g );
	//r_bmfe_lvl1_part0_1_lvl2_part1_2[D0,D1,D2,D3]
DistTensor<double> r_bmfe_lvl1_part0_1_lvl2_part1_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//r_bmfe_lvl1_part0_2[D0,D1,D2,D3]
DistTensor<double> r_bmfe_lvl1_part0_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//t_fj[D01,D23]
DistTensor<double> t_fj__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//t_fj_lvl1_part1B[D01,D23]
DistTensor<double> t_fj_lvl1_part1B__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//t_fj_lvl1_part1T[D01,D23]
DistTensor<double> t_fj_lvl1_part1T__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//t_fj_lvl2_part1B[D01,D23]
DistTensor<double> t_fj_lvl2_part1B__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//t_fj_lvl2_part1T[D01,D23]
DistTensor<double> t_fj_lvl2_part1T__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//t_fj_lvl2_part1_0[D01,D23]
DistTensor<double> t_fj_lvl2_part1_0__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//t_fj_lvl2_part1_1[D01,D23]
DistTensor<double> t_fj_lvl2_part1_1__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//t_fj_lvl2_part1_1[D01,*]
DistTensor<double> t_fj_lvl2_part1_1_perm10__S__D_0_1( dist__D_0_1__S, g );
t_fj_lvl2_part1_1_perm10__S__D_0_1.SetLocalPermutation( perm_1_0 );
	//t_fj_lvl2_part1_1[*,*]
DistTensor<double> t_fj_lvl2_part1_1_perm10__S__S( dist__S__S, g );
t_fj_lvl2_part1_1_perm10__S__S.SetLocalPermutation( perm_1_0 );
	//t_fj_lvl2_part1_2[D01,D23]
DistTensor<double> t_fj_lvl2_part1_2__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//z_small_temp2[D0,D1,D2,D3]
DistTensor<double> z_small_temp2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//z_small_temp2_lvl0_part0B[D0,D1,D2,D3]
DistTensor<double> z_small_temp2_lvl0_part0B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//z_small_temp2_lvl0_part0T[D0,D1,D2,D3]
DistTensor<double> z_small_temp2_lvl0_part0T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//z_small_temp2_lvl0_part1B[D0,D1,D2,D3]
DistTensor<double> z_small_temp2_lvl0_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//z_small_temp2_lvl0_part1T[D0,D1,D2,D3]
DistTensor<double> z_small_temp2_lvl0_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//z_small_temp2_lvl1_part0B[D0,D1,D2,D3]
DistTensor<double> z_small_temp2_lvl1_part0B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//z_small_temp2_lvl1_part0T[D0,D1,D2,D3]
DistTensor<double> z_small_temp2_lvl1_part0T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//z_small_temp2_lvl1_part0_0[D0,D1,D2,D3]
DistTensor<double> z_small_temp2_lvl1_part0_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//z_small_temp2_lvl1_part0_1[D0,D1,D2,D3]
DistTensor<double> z_small_temp2_lvl1_part0_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//z_small_temp2_lvl1_part0_1_lvl1_part1B[D0,D1,D2,D3]
DistTensor<double> z_small_temp2_lvl1_part0_1_lvl1_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//z_small_temp2_lvl1_part0_1_lvl1_part1T[D0,D1,D2,D3]
DistTensor<double> z_small_temp2_lvl1_part0_1_lvl1_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//z_small_temp2_lvl1_part0_1_lvl2_part1B[D0,D1,D2,D3]
DistTensor<double> z_small_temp2_lvl1_part0_1_lvl2_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//z_small_temp2_lvl1_part0_1_lvl2_part1T[D0,D1,D2,D3]
DistTensor<double> z_small_temp2_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//z_small_temp2_lvl1_part0_1_lvl2_part1_0[D0,D1,D2,D3]
DistTensor<double> z_small_temp2_lvl1_part0_1_lvl2_part1_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//z_small_temp2_lvl1_part0_1_lvl2_part1_1[D0,D1,D2,D3]
DistTensor<double> z_small_temp2_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//z_small_temp2_lvl1_part0_1_lvl2_part1_2[D0,D1,D2,D3]
DistTensor<double> z_small_temp2_lvl1_part0_1_lvl2_part1_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//z_small_temp2_lvl1_part0_2[D0,D1,D2,D3]
DistTensor<double> z_small_temp2_lvl1_part0_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//z_small_temp2_lvl1_part1B[D0,D1,D2,D3]
DistTensor<double> z_small_temp2_lvl1_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//z_small_temp2_lvl1_part1T[D0,D1,D2,D3]
DistTensor<double> z_small_temp2_lvl1_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//z_small_temp2_lvl1_part1_0[D0,D1,D2,D3]
DistTensor<double> z_small_temp2_lvl1_part1_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//z_small_temp2_lvl1_part1_1[D0,D1,D2,D3]
DistTensor<double> z_small_temp2_lvl1_part1_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> z_small_temp2_lvl1_part1_1_perm0231__D_0__D_2__D_3__D_1( dist__D_0__D_1__D_2__D_3, g );
z_small_temp2_lvl1_part1_1_perm0231__D_0__D_2__D_3__D_1.SetLocalPermutation( perm_0_2_3_1 );
	//z_small_temp2_lvl1_part1_2[D0,D1,D2,D3]
DistTensor<double> z_small_temp2_lvl1_part1_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//z_small_temp3_lvl1_part2_1_lvl2_part3_1[D01,*,D2,D3]
DistTensor<double> z_small_temp3_lvl1_part2_1_lvl2_part3_1__D_0_1__S__D_2__D_3( dist__D_0_1__S__D_2__D_3, g );
	//z_small_temp3_lvl1_part2_1_lvl2_part3_1[D0,D1,D2,D3]
DistTensor<double> z_small_temp3_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//z_small_temp3_lvl1_part2_1_lvl2_part3_1[D01,*,D23,*]
DistTensor<double> z_small_temp3_lvl1_part2_1_lvl2_part3_1_perm0231__D_0_1__D_2_3__S__S( dist__D_0_1__S__D_2_3__S, g );
z_small_temp3_lvl1_part2_1_lvl2_part3_1_perm0231__D_0_1__D_2_3__S__S.SetLocalPermutation( perm_0_2_3_1 );
	//z_small_temp4[D0,D1,D2,D3]
DistTensor<double> z_small_temp4__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//z_small_temp4_lvl0_part1B[D0,D1,D2,D3]
DistTensor<double> z_small_temp4_lvl0_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//z_small_temp4_lvl0_part1T[D0,D1,D2,D3]
DistTensor<double> z_small_temp4_lvl0_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//z_small_temp4_lvl0_part2B[D0,D1,D2,D3]
DistTensor<double> z_small_temp4_lvl0_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//z_small_temp4_lvl0_part2T[D0,D1,D2,D3]
DistTensor<double> z_small_temp4_lvl0_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//z_small_temp4_lvl1_part1B[D0,D1,D2,D3]
DistTensor<double> z_small_temp4_lvl1_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//z_small_temp4_lvl1_part1T[D0,D1,D2,D3]
DistTensor<double> z_small_temp4_lvl1_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//z_small_temp4_lvl1_part1_0[D0,D1,D2,D3]
DistTensor<double> z_small_temp4_lvl1_part1_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//z_small_temp4_lvl1_part1_1[D0,D1,D2,D3]
DistTensor<double> z_small_temp4_lvl1_part1_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//z_small_temp4_lvl1_part1_1_lvl1_part2B[D0,D1,D2,D3]
DistTensor<double> z_small_temp4_lvl1_part1_1_lvl1_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//z_small_temp4_lvl1_part1_1_lvl1_part2T[D0,D1,D2,D3]
DistTensor<double> z_small_temp4_lvl1_part1_1_lvl1_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//z_small_temp4_lvl1_part1_1_lvl2_part2B[D0,D1,D2,D3]
DistTensor<double> z_small_temp4_lvl1_part1_1_lvl2_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//z_small_temp4_lvl1_part1_1_lvl2_part2T[D0,D1,D2,D3]
DistTensor<double> z_small_temp4_lvl1_part1_1_lvl2_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//z_small_temp4_lvl1_part1_1_lvl2_part2_0[D0,D1,D2,D3]
DistTensor<double> z_small_temp4_lvl1_part1_1_lvl2_part2_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//z_small_temp4_lvl1_part1_1_lvl2_part2_1[D0,D1,D2,D3]
DistTensor<double> z_small_temp4_lvl1_part1_1_lvl2_part2_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//z_small_temp4_lvl1_part1_1_lvl2_part2_2[D0,D1,D2,D3]
DistTensor<double> z_small_temp4_lvl1_part1_1_lvl2_part2_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//z_small_temp4_lvl1_part1_2[D0,D1,D2,D3]
DistTensor<double> z_small_temp4_lvl1_part1_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//z_small_temp4_lvl1_part2B[D0,D1,D2,D3]
DistTensor<double> z_small_temp4_lvl1_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//z_small_temp4_lvl1_part2T[D0,D1,D2,D3]
DistTensor<double> z_small_temp4_lvl1_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//z_small_temp4_lvl1_part2_0[D0,D1,D2,D3]
DistTensor<double> z_small_temp4_lvl1_part2_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//z_small_temp4_lvl1_part2_1[D0,D1,D2,D3]
DistTensor<double> z_small_temp4_lvl1_part2_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//z_small_temp4_lvl1_part2_1_lvl1_part1B[D0,D1,D2,D3]
DistTensor<double> z_small_temp4_lvl1_part2_1_lvl1_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//z_small_temp4_lvl1_part2_1_lvl1_part1T[D0,D1,D2,D3]
DistTensor<double> z_small_temp4_lvl1_part2_1_lvl1_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//z_small_temp4_lvl1_part2_1_lvl2_part1B[D0,D1,D2,D3]
DistTensor<double> z_small_temp4_lvl1_part2_1_lvl2_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//z_small_temp4_lvl1_part2_1_lvl2_part1T[D0,D1,D2,D3]
DistTensor<double> z_small_temp4_lvl1_part2_1_lvl2_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//z_small_temp4_lvl1_part2_1_lvl2_part1_0[D0,D1,D2,D3]
DistTensor<double> z_small_temp4_lvl1_part2_1_lvl2_part1_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//z_small_temp4_lvl1_part2_1_lvl2_part1_1[D01,*,D2,D3]
DistTensor<double> z_small_temp4_lvl1_part2_1_lvl2_part1_1__D_0_1__S__D_2__D_3( dist__D_0_1__S__D_2__D_3, g );
	//z_small_temp4_lvl1_part2_1_lvl2_part1_1[D0,D1,D2,D3]
DistTensor<double> z_small_temp4_lvl1_part2_1_lvl2_part1_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//z_small_temp4_lvl1_part2_1_lvl2_part1_1[D01,*,D23,*]
DistTensor<double> z_small_temp4_lvl1_part2_1_lvl2_part1_1_perm0213__D_0_1__D_2_3__S__S( dist__D_0_1__S__D_2_3__S, g );
z_small_temp4_lvl1_part2_1_lvl2_part1_1_perm0213__D_0_1__D_2_3__S__S.SetLocalPermutation( perm_0_2_1_3 );
	//z_small_temp4_lvl1_part2_1_lvl2_part1_2[D0,D1,D2,D3]
DistTensor<double> z_small_temp4_lvl1_part2_1_lvl2_part1_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//z_small_temp4_lvl1_part2_2[D0,D1,D2,D3]
DistTensor<double> z_small_temp4_lvl1_part2_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//z_small_temp5[D0,D1,D2,D3]
DistTensor<double> z_small_temp5__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//z_small_temp5_lvl0_part0B[D0,D1,D2,D3]
DistTensor<double> z_small_temp5_lvl0_part0B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//z_small_temp5_lvl0_part0T[D0,D1,D2,D3]
DistTensor<double> z_small_temp5_lvl0_part0T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//z_small_temp5_lvl0_part2B[D0,D1,D2,D3]
DistTensor<double> z_small_temp5_lvl0_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//z_small_temp5_lvl0_part2T[D0,D1,D2,D3]
DistTensor<double> z_small_temp5_lvl0_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//z_small_temp5_lvl1_part0B[D0,D1,D2,D3]
DistTensor<double> z_small_temp5_lvl1_part0B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//z_small_temp5_lvl1_part0T[D0,D1,D2,D3]
DistTensor<double> z_small_temp5_lvl1_part0T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//z_small_temp5_lvl1_part0_0[D0,D1,D2,D3]
DistTensor<double> z_small_temp5_lvl1_part0_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//z_small_temp5_lvl1_part0_1[D0,D1,D2,D3]
DistTensor<double> z_small_temp5_lvl1_part0_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//z_small_temp5_lvl1_part0_1_lvl1_part1B[D0,D1,D2,D3]
DistTensor<double> z_small_temp5_lvl1_part0_1_lvl1_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//z_small_temp5_lvl1_part0_1_lvl1_part1T[D0,D1,D2,D3]
DistTensor<double> z_small_temp5_lvl1_part0_1_lvl1_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//z_small_temp5_lvl1_part0_1_lvl2_part1B[D0,D1,D2,D3]
DistTensor<double> z_small_temp5_lvl1_part0_1_lvl2_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//z_small_temp5_lvl1_part0_1_lvl2_part1T[D0,D1,D2,D3]
DistTensor<double> z_small_temp5_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//z_small_temp5_lvl1_part0_1_lvl2_part1_0[D0,D1,D2,D3]
DistTensor<double> z_small_temp5_lvl1_part0_1_lvl2_part1_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//z_small_temp5_lvl1_part0_1_lvl2_part1_1[D0,D1,D2,D3]
DistTensor<double> z_small_temp5_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//z_small_temp5_lvl1_part0_1_lvl2_part1_2[D0,D1,D2,D3]
DistTensor<double> z_small_temp5_lvl1_part0_1_lvl2_part1_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//z_small_temp5_lvl1_part0_2[D0,D1,D2,D3]
DistTensor<double> z_small_temp5_lvl1_part0_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//z_small_temp5_lvl1_part2B[D0,D1,D2,D3]
DistTensor<double> z_small_temp5_lvl1_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//z_small_temp5_lvl1_part2T[D0,D1,D2,D3]
DistTensor<double> z_small_temp5_lvl1_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//z_small_temp5_lvl1_part2_0[D0,D1,D2,D3]
DistTensor<double> z_small_temp5_lvl1_part2_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//z_small_temp5_lvl1_part2_1[D0,D1,D2,D3]
DistTensor<double> z_small_temp5_lvl1_part2_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//z_small_temp5_lvl1_part2_1_lvl1_part0B[D0,D1,D2,D3]
DistTensor<double> z_small_temp5_lvl1_part2_1_lvl1_part0B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//z_small_temp5_lvl1_part2_1_lvl1_part0T[D0,D1,D2,D3]
DistTensor<double> z_small_temp5_lvl1_part2_1_lvl1_part0T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//z_small_temp5_lvl1_part2_1_lvl2_part0B[D0,D1,D2,D3]
DistTensor<double> z_small_temp5_lvl1_part2_1_lvl2_part0B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//z_small_temp5_lvl1_part2_1_lvl2_part0T[D0,D1,D2,D3]
DistTensor<double> z_small_temp5_lvl1_part2_1_lvl2_part0T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//z_small_temp5_lvl1_part2_1_lvl2_part0_0[D0,D1,D2,D3]
DistTensor<double> z_small_temp5_lvl1_part2_1_lvl2_part0_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//z_small_temp5_lvl1_part2_1_lvl2_part0_1[D0,D1,D2,D3]
DistTensor<double> z_small_temp5_lvl1_part2_1_lvl2_part0_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//z_small_temp5_lvl1_part2_1_lvl2_part0_1[D0,D3,D2,D1]
DistTensor<double> z_small_temp5_lvl1_part2_1_lvl2_part0_1__D_0__D_3__D_2__D_1( dist__D_0__D_3__D_2__D_1, g );
	//z_small_temp5_lvl1_part2_1_lvl2_part0_1[D2,D3,*,D1]
DistTensor<double> z_small_temp5_lvl1_part2_1_lvl2_part0_1_perm2013__S__D_2__D_3__D_1( dist__D_2__D_3__S__D_1, g );
z_small_temp5_lvl1_part2_1_lvl2_part0_1_perm2013__S__D_2__D_3__D_1.SetLocalPermutation( perm_2_0_1_3 );
	//z_small_temp5_lvl1_part2_1_lvl2_part0_2[D0,D1,D2,D3]
DistTensor<double> z_small_temp5_lvl1_part2_1_lvl2_part0_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//z_small_temp5_lvl1_part2_2[D0,D1,D2,D3]
DistTensor<double> z_small_temp5_lvl1_part2_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//w_bmje[D0,D1,D2,D3]
DistTensor<double> w_bmje__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//w_bmje_lvl0_part1B[D0,D1,D2,D3]
DistTensor<double> w_bmje_lvl0_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//w_bmje_lvl0_part1T[D0,D1,D2,D3]
DistTensor<double> w_bmje_lvl0_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//w_bmje_lvl1_part1B[D0,D1,D2,D3]
DistTensor<double> w_bmje_lvl1_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//w_bmje_lvl1_part1T[D0,D1,D2,D3]
DistTensor<double> w_bmje_lvl1_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//w_bmje_lvl1_part1_0[D0,D1,D2,D3]
DistTensor<double> w_bmje_lvl1_part1_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//w_bmje_lvl1_part1_1[D0,D1,D2,D3]
DistTensor<double> w_bmje_lvl1_part1_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//w_bmje_lvl1_part1_1_lvl1_part2B[D0,D1,D2,D3]
DistTensor<double> w_bmje_lvl1_part1_1_lvl1_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//w_bmje_lvl1_part1_1_lvl1_part2T[D0,D1,D2,D3]
DistTensor<double> w_bmje_lvl1_part1_1_lvl1_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//w_bmje_lvl1_part1_1_lvl2_part2B[D0,D1,D2,D3]
DistTensor<double> w_bmje_lvl1_part1_1_lvl2_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//w_bmje_lvl1_part1_1_lvl2_part2T[D0,D1,D2,D3]
DistTensor<double> w_bmje_lvl1_part1_1_lvl2_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//w_bmje_lvl1_part1_1_lvl2_part2_0[D0,D1,D2,D3]
DistTensor<double> w_bmje_lvl1_part1_1_lvl2_part2_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//w_bmje_lvl1_part1_1_lvl2_part2_1[D0,D1,D2,D3]
DistTensor<double> w_bmje_lvl1_part1_1_lvl2_part2_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//w_bmje_lvl1_part1_1_lvl2_part2_2[D0,D1,D2,D3]
DistTensor<double> w_bmje_lvl1_part1_1_lvl2_part2_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//w_bmje_lvl1_part1_2[D0,D1,D2,D3]
DistTensor<double> w_bmje_lvl1_part1_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//x_bmej[D0,D1,D2,D3]
DistTensor<double> x_bmej__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//x_bmej_lvl0_part1B[D0,D1,D2,D3]
DistTensor<double> x_bmej_lvl0_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//x_bmej_lvl0_part1T[D0,D1,D2,D3]
DistTensor<double> x_bmej_lvl0_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//x_bmej_lvl1_part1B[D0,D1,D2,D3]
DistTensor<double> x_bmej_lvl1_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//x_bmej_lvl1_part1T[D0,D1,D2,D3]
DistTensor<double> x_bmej_lvl1_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//x_bmej_lvl1_part1_0[D0,D1,D2,D3]
DistTensor<double> x_bmej_lvl1_part1_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//x_bmej_lvl1_part1_1[D0,D1,D2,D3]
DistTensor<double> x_bmej_lvl1_part1_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//x_bmej_lvl1_part1_1_lvl1_part3B[D0,D1,D2,D3]
DistTensor<double> x_bmej_lvl1_part1_1_lvl1_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//x_bmej_lvl1_part1_1_lvl1_part3T[D0,D1,D2,D3]
DistTensor<double> x_bmej_lvl1_part1_1_lvl1_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//x_bmej_lvl1_part1_1_lvl2_part3B[D0,D1,D2,D3]
DistTensor<double> x_bmej_lvl1_part1_1_lvl2_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//x_bmej_lvl1_part1_1_lvl2_part3T[D0,D1,D2,D3]
DistTensor<double> x_bmej_lvl1_part1_1_lvl2_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//x_bmej_lvl1_part1_1_lvl2_part3_0[D0,D1,D2,D3]
DistTensor<double> x_bmej_lvl1_part1_1_lvl2_part3_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//x_bmej_lvl1_part1_1_lvl2_part3_1[D0,D1,D2,D3]
DistTensor<double> x_bmej_lvl1_part1_1_lvl2_part3_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//x_bmej_lvl1_part1_1_lvl2_part3_1[D0,D1,D3,D2]
DistTensor<double> x_bmej_lvl1_part1_1_lvl2_part3_1__D_0__D_1__D_3__D_2( dist__D_0__D_1__D_3__D_2, g );
	//x_bmej_lvl1_part1_1_lvl2_part3_2[D0,D1,D2,D3]
DistTensor<double> x_bmej_lvl1_part1_1_lvl2_part3_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//x_bmej_lvl1_part1_2[D0,D1,D2,D3]
DistTensor<double> x_bmej_lvl1_part1_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//z_ai[D01,D23]
DistTensor<double> z_ai__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//z_ai_lvl0_part1B[D01,D23]
DistTensor<double> z_ai_lvl0_part1B__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//z_ai_lvl0_part1T[D01,D23]
DistTensor<double> z_ai_lvl0_part1T__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//z_ai_lvl1_part1B[D01,D23]
DistTensor<double> z_ai_lvl1_part1B__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//z_ai_lvl1_part1T[D01,D23]
DistTensor<double> z_ai_lvl1_part1T__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//z_ai_lvl1_part1_0[D01,D23]
DistTensor<double> z_ai_lvl1_part1_0__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//z_ai_lvl1_part1_1[D01,D23]
DistTensor<double> z_ai_lvl1_part1_1__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//z_ai_lvl1_part1_1[D0,*,D1,D2,D3]
DistTensor<double> z_ai_lvl1_part1_1_perm10342__S__D_0__D_2__D_3__D_1( dist__D_0__S__D_1__D_2__D_3, g );
z_ai_lvl1_part1_1_perm10342__S__D_0__D_2__D_3__D_1.SetLocalPermutation( perm_1_0_3_4_2 );
DistTensor<double> z_ai_lvl1_part1_1_perm10__D_2_3__D_0_1( dist__D_0_1__D_2_3, g );
z_ai_lvl1_part1_1_perm10__D_2_3__D_0_1.SetLocalPermutation( perm_1_0 );
	//z_ai_lvl1_part1_2[D01,D23]
DistTensor<double> z_ai_lvl1_part1_2__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//z_ai_lvl2_part1B[D01,D23]
DistTensor<double> z_ai_lvl2_part1B__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//z_ai_lvl2_part1T[D01,D23]
DistTensor<double> z_ai_lvl2_part1T__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//z_ai_lvl2_part1_0[D01,D23]
DistTensor<double> z_ai_lvl2_part1_0__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//z_ai_lvl2_part1_1[D01,D23]
DistTensor<double> z_ai_lvl2_part1_1__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//z_ai_lvl2_part1_1[D0,*,D2,D3,D1]
DistTensor<double> z_ai_lvl2_part1_1__D_0__S__D_2__D_3__D_1( dist__D_0__S__D_2__D_3__D_1, g );
	//z_ai_lvl2_part1_2[D01,D23]
DistTensor<double> z_ai_lvl2_part1_2__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
// G_mi has 2 dims
//	Starting distribution: [D01,D23] or _D_0_1__D_2_3
ObjShape G_mi__D_0_1__D_2_3_tmpShape_z_small( 2 );
G_mi__D_0_1__D_2_3_tmpShape_z_small[ 0 ] = n_o;
G_mi__D_0_1__D_2_3_tmpShape_z_small[ 1 ] = n_o;
G_mi__D_0_1__D_2_3.ResizeTo( G_mi__D_0_1__D_2_3_tmpShape_z_small );
MakeUniform( G_mi__D_0_1__D_2_3 );
// t_fj has 2 dims
ObjShape t_fj__D_0_1__D_2_3_tmpShape_z_small( 2 );
t_fj__D_0_1__D_2_3_tmpShape_z_small[ 0 ] = n_v;
t_fj__D_0_1__D_2_3_tmpShape_z_small[ 1 ] = n_o;
t_fj__D_0_1__D_2_3.ResizeTo( t_fj__D_0_1__D_2_3_tmpShape_z_small );
MakeUniform( t_fj__D_0_1__D_2_3 );
// T_bfnj has 4 dims
//	Starting distribution: [D0,D1,D2,D3] or _D_0__D_1__D_2__D_3
ObjShape T_bfnj__D_0__D_1__D_2__D_3_tmpShape_z_small( 4 );
T_bfnj__D_0__D_1__D_2__D_3_tmpShape_z_small[ 0 ] = n_v;
T_bfnj__D_0__D_1__D_2__D_3_tmpShape_z_small[ 1 ] = n_v;
T_bfnj__D_0__D_1__D_2__D_3_tmpShape_z_small[ 2 ] = n_o;
T_bfnj__D_0__D_1__D_2__D_3_tmpShape_z_small[ 3 ] = n_o;
T_bfnj__D_0__D_1__D_2__D_3.ResizeTo( T_bfnj__D_0__D_1__D_2__D_3_tmpShape_z_small );
MakeUniform( T_bfnj__D_0__D_1__D_2__D_3 );
// U_mnie has 4 dims
ObjShape U_mnie__D_0__D_1__D_2__D_3_tmpShape_z_small( 4 );
U_mnie__D_0__D_1__D_2__D_3_tmpShape_z_small[ 0 ] = n_o;
U_mnie__D_0__D_1__D_2__D_3_tmpShape_z_small[ 1 ] = n_o;
U_mnie__D_0__D_1__D_2__D_3_tmpShape_z_small[ 2 ] = n_o;
U_mnie__D_0__D_1__D_2__D_3_tmpShape_z_small[ 3 ] = n_v;
U_mnie__D_0__D_1__D_2__D_3.ResizeTo( U_mnie__D_0__D_1__D_2__D_3_tmpShape_z_small );
MakeUniform( U_mnie__D_0__D_1__D_2__D_3 );
// w_bmje has 4 dims
ObjShape w_bmje__D_0__D_1__D_2__D_3_tmpShape_z_small( 4 );
w_bmje__D_0__D_1__D_2__D_3_tmpShape_z_small[ 0 ] = n_v;
w_bmje__D_0__D_1__D_2__D_3_tmpShape_z_small[ 1 ] = n_o;
w_bmje__D_0__D_1__D_2__D_3_tmpShape_z_small[ 2 ] = n_o;
w_bmje__D_0__D_1__D_2__D_3_tmpShape_z_small[ 3 ] = n_v;
w_bmje__D_0__D_1__D_2__D_3.ResizeTo( w_bmje__D_0__D_1__D_2__D_3_tmpShape_z_small );
MakeUniform( w_bmje__D_0__D_1__D_2__D_3 );
// x_bmej has 4 dims
ObjShape x_bmej__D_0__D_1__D_2__D_3_tmpShape_z_small( 4 );
x_bmej__D_0__D_1__D_2__D_3_tmpShape_z_small[ 0 ] = n_v;
x_bmej__D_0__D_1__D_2__D_3_tmpShape_z_small[ 1 ] = n_o;
x_bmej__D_0__D_1__D_2__D_3_tmpShape_z_small[ 2 ] = n_v;
x_bmej__D_0__D_1__D_2__D_3_tmpShape_z_small[ 3 ] = n_o;
x_bmej__D_0__D_1__D_2__D_3.ResizeTo( x_bmej__D_0__D_1__D_2__D_3_tmpShape_z_small );
MakeUniform( x_bmej__D_0__D_1__D_2__D_3 );
// H_me has 2 dims
ObjShape H_me__D_0_1__D_2_3_tmpShape_z_small( 2 );
H_me__D_0_1__D_2_3_tmpShape_z_small[ 0 ] = n_o;
H_me__D_0_1__D_2_3_tmpShape_z_small[ 1 ] = n_v;
H_me__D_0_1__D_2_3.ResizeTo( H_me__D_0_1__D_2_3_tmpShape_z_small );
MakeUniform( H_me__D_0_1__D_2_3 );
// Tau_efmn has 4 dims
ObjShape Tau_efmn__D_0__D_1__D_2__D_3_tmpShape_z_small( 4 );
Tau_efmn__D_0__D_1__D_2__D_3_tmpShape_z_small[ 0 ] = n_v;
Tau_efmn__D_0__D_1__D_2__D_3_tmpShape_z_small[ 1 ] = n_v;
Tau_efmn__D_0__D_1__D_2__D_3_tmpShape_z_small[ 2 ] = n_o;
Tau_efmn__D_0__D_1__D_2__D_3_tmpShape_z_small[ 3 ] = n_o;
Tau_efmn__D_0__D_1__D_2__D_3.ResizeTo( Tau_efmn__D_0__D_1__D_2__D_3_tmpShape_z_small );
MakeUniform( Tau_efmn__D_0__D_1__D_2__D_3 );
// z_ai has 2 dims
ObjShape z_ai__D_0_1__D_2_3_tmpShape_z_small( 2 );
z_ai__D_0_1__D_2_3_tmpShape_z_small[ 0 ] = n_v;
z_ai__D_0_1__D_2_3_tmpShape_z_small[ 1 ] = n_o;
z_ai__D_0_1__D_2_3.ResizeTo( z_ai__D_0_1__D_2_3_tmpShape_z_small );
MakeUniform( z_ai__D_0_1__D_2_3 );
// r_bmfe has 4 dims
ObjShape r_bmfe__D_0__D_1__D_2__D_3_tmpShape_z_small( 4 );
r_bmfe__D_0__D_1__D_2__D_3_tmpShape_z_small[ 0 ] = n_v;
r_bmfe__D_0__D_1__D_2__D_3_tmpShape_z_small[ 1 ] = n_o;
r_bmfe__D_0__D_1__D_2__D_3_tmpShape_z_small[ 2 ] = n_v;
r_bmfe__D_0__D_1__D_2__D_3_tmpShape_z_small[ 3 ] = n_v;
r_bmfe__D_0__D_1__D_2__D_3.ResizeTo( r_bmfe__D_0__D_1__D_2__D_3_tmpShape_z_small );
MakeUniform( r_bmfe__D_0__D_1__D_2__D_3 );
//END_DECL

//******************************
//* Load tensors
//******************************
////////////////////////////////
//Performance testing
////////////////////////////////
std::stringstream fullName;
#ifdef CORRECTNESS
DistTensor<T> check_z_small(dist__D_0_1__D_2_3, g);
check_z_small.ResizeTo(z_ai__D_0_1__D_2_3.Shape());
Read(r_bmfe__D_0__D_1__D_2__D_3, "ccsd_terms/term_r_small", BINARY_FLAT, false);
Read(w_bmje__D_0__D_1__D_2__D_3, "ccsd_terms/term_w_small", BINARY_FLAT, false);
Read(x_bmej__D_0__D_1__D_2__D_3, "ccsd_terms/term_x_small", BINARY_FLAT, false);
fullName.str("");
fullName.clear();
fullName << "ccsd_terms/term_H_iter" << testIter;
Read(H_me__D_0_1__D_2_3, fullName.str(), BINARY_FLAT, false);
fullName.str("");
fullName.clear();
fullName << "ccsd_terms/term_G_iter" << testIter;
Read(G_mi__D_0_1__D_2_3, fullName.str(), BINARY_FLAT, false);
fullName.str("");
fullName.clear();
fullName << "ccsd_terms/term_U_iter" << testIter;
Read(U_mnie__D_0__D_1__D_2__D_3, fullName.str(), BINARY_FLAT, false);
fullName.str("");
fullName.clear();
fullName << "ccsd_terms/term_t_small_iter" << testIter;
Read(t_fj__D_0_1__D_2_3, fullName.str(), BINARY_FLAT, false);
fullName.str("");
fullName.clear();
fullName << "ccsd_terms/term_Tau_iter" << testIter;
Read(Tau_efmn__D_0__D_1__D_2__D_3, fullName.str(), BINARY_FLAT, false);
fullName.str("");
fullName.clear();
fullName << "ccsd_terms/term_T_iter" << testIter;
Read(T_bfnj__D_0__D_1__D_2__D_3, fullName.str(), BINARY_FLAT, false);
fullName.str("");
fullName.clear();
fullName << "ccsd_terms/term_z_small_iter" << testIter;
Read(check_z_small, fullName.str(), BINARY_FLAT, false);
#endif
//******************************
//* Load tensors
//******************************
    long long flops = 0;
    double gflops;
    double startTime;
    double runTime;
    double norm = 1.0;
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

//START_CODE
overwrite_tmpShape_z_small = r_bmfe__D_0__D_1__D_2__D_3.Shape();
z_small_temp2__D_0__D_1__D_2__D_3.ResizeTo( overwrite_tmpShape_z_small );
overwrite_tmpShape_z_small = w_bmje__D_0__D_1__D_2__D_3.Shape();
z_small_temp4__D_0__D_1__D_2__D_3.ResizeTo( overwrite_tmpShape_z_small );
overwrite_tmpShape_z_small = U_mnie__D_0__D_1__D_2__D_3.Shape();
z_small_temp5__D_0__D_1__D_2__D_3.ResizeTo( overwrite_tmpShape_z_small );
//**** (out of 1)
//**** Is real	0 shadows
	//Outputs:
	//  z_ai__D_0_1__D_2_3

	Scal( 0.0, z_ai__D_0_1__D_2_3 );
	//**** (out of 1)
	//**** Is real	0 shadows
		//Outputs:
		//  z_small_temp2__D_0__D_1__D_2__D_3
	PartitionDown(r_bmfe__D_0__D_1__D_2__D_3, r_bmfe_lvl1_part0T__D_0__D_1__D_2__D_3, r_bmfe_lvl1_part0B__D_0__D_1__D_2__D_3, 0, 0);
	PartitionDown(z_small_temp2__D_0__D_1__D_2__D_3, z_small_temp2_lvl1_part0T__D_0__D_1__D_2__D_3, z_small_temp2_lvl1_part0B__D_0__D_1__D_2__D_3, 0, 0);
	while(z_small_temp2_lvl1_part0T__D_0__D_1__D_2__D_3.Dimension(0) < z_small_temp2__D_0__D_1__D_2__D_3.Dimension(0))
	{
		RepartitionDown
		( r_bmfe_lvl1_part0T__D_0__D_1__D_2__D_3,  r_bmfe_lvl1_part0_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       r_bmfe_lvl1_part0_1__D_0__D_1__D_2__D_3,
		  r_bmfe_lvl1_part0B__D_0__D_1__D_2__D_3, r_bmfe_lvl1_part0_2__D_0__D_1__D_2__D_3, 0, blkSize );
		RepartitionDown
		( z_small_temp2_lvl1_part0T__D_0__D_1__D_2__D_3,  z_small_temp2_lvl1_part0_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       z_small_temp2_lvl1_part0_1__D_0__D_1__D_2__D_3,
		  z_small_temp2_lvl1_part0B__D_0__D_1__D_2__D_3, z_small_temp2_lvl1_part0_2__D_0__D_1__D_2__D_3, 0, blkSize );

		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  z_small_temp2_lvl1_part0_1__D_0__D_1__D_2__D_3
		PartitionDown(r_bmfe_lvl1_part0_1__D_0__D_1__D_2__D_3, r_bmfe_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3, r_bmfe_lvl1_part0_1_lvl2_part1B__D_0__D_1__D_2__D_3, 1, 0);
		PartitionDown(z_small_temp2_lvl1_part0_1__D_0__D_1__D_2__D_3, z_small_temp2_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3, z_small_temp2_lvl1_part0_1_lvl2_part1B__D_0__D_1__D_2__D_3, 1, 0);
		while(z_small_temp2_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3.Dimension(1) < z_small_temp2_lvl1_part0_1__D_0__D_1__D_2__D_3.Dimension(1))
		{
			RepartitionDown
			( r_bmfe_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3,  r_bmfe_lvl1_part0_1_lvl2_part1_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       r_bmfe_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3,
			  r_bmfe_lvl1_part0_1_lvl2_part1B__D_0__D_1__D_2__D_3, r_bmfe_lvl1_part0_1_lvl2_part1_2__D_0__D_1__D_2__D_3, 1, blkSize );
			RepartitionDown
			( z_small_temp2_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3,  z_small_temp2_lvl1_part0_1_lvl2_part1_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       z_small_temp2_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3,
			  z_small_temp2_lvl1_part0_1_lvl2_part1B__D_0__D_1__D_2__D_3, z_small_temp2_lvl1_part0_1_lvl2_part1_2__D_0__D_1__D_2__D_3, 1, blkSize );

			   // r_bmfe_lvl1_part0_1_lvl2_part1_1[D0,D1,D3,D2] <- r_bmfe_lvl1_part0_1_lvl2_part1_1[D0,D1,D2,D3]
			r_bmfe_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_3__D_2.AlignModesWith( modes_0_1_2_3, r_bmfe_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3, modes_0_1_3_2 );
			r_bmfe_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_3__D_2.AllToAllRedistFrom( r_bmfe_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3, modes_2_3 );
if(commRank == 0){
flops += 3*prod(r_bmfe_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3.Shape());
}
			YAxpPx( 2.0, r_bmfe_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3, -1.0, r_bmfe_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_3__D_2, perm_0_1_3_2, z_small_temp2_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3 );
			r_bmfe_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_3__D_2.EmptyData();

			SlidePartitionDown
			( r_bmfe_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3,  r_bmfe_lvl1_part0_1_lvl2_part1_0__D_0__D_1__D_2__D_3,
			       r_bmfe_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  r_bmfe_lvl1_part0_1_lvl2_part1B__D_0__D_1__D_2__D_3, r_bmfe_lvl1_part0_1_lvl2_part1_2__D_0__D_1__D_2__D_3, 1 );
			SlidePartitionDown
			( z_small_temp2_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3,  z_small_temp2_lvl1_part0_1_lvl2_part1_0__D_0__D_1__D_2__D_3,
			       z_small_temp2_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  z_small_temp2_lvl1_part0_1_lvl2_part1B__D_0__D_1__D_2__D_3, z_small_temp2_lvl1_part0_1_lvl2_part1_2__D_0__D_1__D_2__D_3, 1 );

		}
		//****

		SlidePartitionDown
		( r_bmfe_lvl1_part0T__D_0__D_1__D_2__D_3,  r_bmfe_lvl1_part0_0__D_0__D_1__D_2__D_3,
		       r_bmfe_lvl1_part0_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  r_bmfe_lvl1_part0B__D_0__D_1__D_2__D_3, r_bmfe_lvl1_part0_2__D_0__D_1__D_2__D_3, 0 );
		SlidePartitionDown
		( z_small_temp2_lvl1_part0T__D_0__D_1__D_2__D_3,  z_small_temp2_lvl1_part0_0__D_0__D_1__D_2__D_3,
		       z_small_temp2_lvl1_part0_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  z_small_temp2_lvl1_part0B__D_0__D_1__D_2__D_3, z_small_temp2_lvl1_part0_2__D_0__D_1__D_2__D_3, 0 );

	}
	//****
	//**** (out of 1)
	//**** Is real	0 shadows
		//Outputs:
		//  z_ai__D_0_1__D_2_3
	PartitionDown(z_small_temp2__D_0__D_1__D_2__D_3, z_small_temp2_lvl1_part1T__D_0__D_1__D_2__D_3, z_small_temp2_lvl1_part1B__D_0__D_1__D_2__D_3, 1, 0);
	PartitionDown(Tau_efmn__D_0__D_1__D_2__D_3, Tau_efmn_lvl1_part3T__D_0__D_1__D_2__D_3, Tau_efmn_lvl1_part3B__D_0__D_1__D_2__D_3, 3, 0);
	while(z_small_temp2_lvl1_part1T__D_0__D_1__D_2__D_3.Dimension(1) < z_small_temp2__D_0__D_1__D_2__D_3.Dimension(1))
	{
		RepartitionDown
		( z_small_temp2_lvl1_part1T__D_0__D_1__D_2__D_3,  z_small_temp2_lvl1_part1_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       z_small_temp2_lvl1_part1_1__D_0__D_1__D_2__D_3,
		  z_small_temp2_lvl1_part1B__D_0__D_1__D_2__D_3, z_small_temp2_lvl1_part1_2__D_0__D_1__D_2__D_3, 1, blkSize );
		RepartitionDown
		( Tau_efmn_lvl1_part3T__D_0__D_1__D_2__D_3,  Tau_efmn_lvl1_part3_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       Tau_efmn_lvl1_part3_1__D_0__D_1__D_2__D_3,
		  Tau_efmn_lvl1_part3B__D_0__D_1__D_2__D_3, Tau_efmn_lvl1_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );

		Permute( z_small_temp2_lvl1_part1_1__D_0__D_1__D_2__D_3, z_small_temp2_lvl1_part1_1_perm0231__D_0__D_2__D_3__D_1 );
		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  z_ai__D_0_1__D_2_3
		PartitionDown(Tau_efmn_lvl1_part3_1__D_0__D_1__D_2__D_3, Tau_efmn_lvl1_part3_1_lvl2_part2T__D_0__D_1__D_2__D_3, Tau_efmn_lvl1_part3_1_lvl2_part2B__D_0__D_1__D_2__D_3, 2, 0);
		PartitionDown(z_ai__D_0_1__D_2_3, z_ai_lvl2_part1T__D_0_1__D_2_3, z_ai_lvl2_part1B__D_0_1__D_2_3, 1, 0);
		while(z_ai_lvl2_part1T__D_0_1__D_2_3.Dimension(1) < z_ai__D_0_1__D_2_3.Dimension(1))
		{
			RepartitionDown
			( Tau_efmn_lvl1_part3_1_lvl2_part2T__D_0__D_1__D_2__D_3,  Tau_efmn_lvl1_part3_1_lvl2_part2_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       Tau_efmn_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_2__D_3,
			  Tau_efmn_lvl1_part3_1_lvl2_part2B__D_0__D_1__D_2__D_3, Tau_efmn_lvl1_part3_1_lvl2_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );
			RepartitionDown
			( z_ai_lvl2_part1T__D_0_1__D_2_3,  z_ai_lvl2_part1_0__D_0_1__D_2_3,
			  /**/ /**/
			       z_ai_lvl2_part1_1__D_0_1__D_2_3,
			  z_ai_lvl2_part1B__D_0_1__D_2_3, z_ai_lvl2_part1_2__D_0_1__D_2_3, 1, blkSize );

			   // Tau_efmn_lvl1_part3_1_lvl2_part2_1[D0,D3,D2,D1] <- Tau_efmn_lvl1_part3_1_lvl2_part2_1[D0,D1,D2,D3]
			Tau_efmn_lvl1_part3_1_lvl2_part2_1__D_0__D_3__D_2__D_1.AlignModesWith( modes_0_1_3, z_small_temp2_lvl1_part1_1__D_0__D_1__D_2__D_3, modes_2_3_1 );
			Tau_efmn_lvl1_part3_1_lvl2_part2_1__D_0__D_3__D_2__D_1.AllToAllRedistFrom( Tau_efmn_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_2__D_3, modes_1_3 );
			   // Tau_efmn_lvl1_part3_1_lvl2_part2_1[D2,D3,*,D1] <- Tau_efmn_lvl1_part3_1_lvl2_part2_1[D0,D3,D2,D1]
			Tau_efmn_lvl1_part3_1_lvl2_part2_1_perm0132__D_2__D_3__D_1__S.AlignModesWith( modes_0_1_3, z_small_temp2_lvl1_part1_1__D_0__D_1__D_2__D_3, modes_2_3_1 );
			Tau_efmn_lvl1_part3_1_lvl2_part2_1_perm0132__D_2__D_3__D_1__S.AllToAllRedistFrom( Tau_efmn_lvl1_part3_1_lvl2_part2_1__D_0__D_3__D_2__D_1, modes_0_2 );
			Tau_efmn_lvl1_part3_1_lvl2_part2_1__D_0__D_3__D_2__D_1.EmptyData();
			z_ai_lvl2_part1_1__D_0__S__D_2__D_3__D_1.AlignModesWith( modes_0, z_small_temp2_lvl1_part1_1__D_0__D_1__D_2__D_3, modes_0 );
			overwrite_tmpShape_z_small = z_ai_lvl2_part1_1__D_0_1__D_2_3.Shape();
			overwrite_tmpShape_z_small.push_back( g.Shape()[2] );
			overwrite_tmpShape_z_small.push_back( g.Shape()[3] );
			overwrite_tmpShape_z_small.push_back( g.Shape()[1] );
			z_ai_lvl2_part1_1__D_0__S__D_2__D_3__D_1.ResizeTo( overwrite_tmpShape_z_small );
			   // 1.0 * z_small_temp2_lvl1_part1_1[D0,D1,D2,D3]_aefm * Tau_efmn_lvl1_part3_1_lvl2_part2_1[D2,D3,*,D1]_efmi + 0.0 * z_ai_lvl2_part1_1[D0,*,D2,D3,D1]_aiefm
if(commRank == 0){
flops += 2*z_small_temp2_lvl1_part1_1_perm0231__D_0__D_2__D_3__D_1.Dimension(2)*z_small_temp2_lvl1_part1_1_perm0231__D_0__D_2__D_3__D_1.Dimension(1)*z_small_temp2_lvl1_part1_1_perm0231__D_0__D_2__D_3__D_1.Dimension(3)*z_small_temp2_lvl1_part1_1_perm0231__D_0__D_2__D_3__D_1.Dimension(0)*Tau_efmn_lvl1_part3_1_lvl2_part2_1_perm0132__D_2__D_3__D_1__S.Dimension(2);
}
			LocalContract(1.0, z_small_temp2_lvl1_part1_1_perm0231__D_0__D_2__D_3__D_1.LockedTensor(), indices_aefm, false,
				Tau_efmn_lvl1_part3_1_lvl2_part2_1_perm0132__D_2__D_3__D_1__S.LockedTensor(), indices_efmi, false,
				0.0, z_ai_lvl2_part1_1__D_0__S__D_2__D_3__D_1.Tensor(), indices_aiefm, false);
			Tau_efmn_lvl1_part3_1_lvl2_part2_1_perm0132__D_2__D_3__D_1__S.EmptyData();
			   // z_ai_lvl2_part1_1[D01,D23] <- z_ai_lvl2_part1_1[D0,*,D2,D3,D1] (with SumScatter on (D2)(D3)(D1))
			z_ai_lvl2_part1_1__D_0_1__D_2_3.ReduceScatterUpdateRedistFrom( z_ai_lvl2_part1_1__D_0__S__D_2__D_3__D_1, 1.0, modes_4_3_2 );
			z_ai_lvl2_part1_1__D_0__S__D_2__D_3__D_1.EmptyData();

			SlidePartitionDown
			( Tau_efmn_lvl1_part3_1_lvl2_part2T__D_0__D_1__D_2__D_3,  Tau_efmn_lvl1_part3_1_lvl2_part2_0__D_0__D_1__D_2__D_3,
			       Tau_efmn_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  Tau_efmn_lvl1_part3_1_lvl2_part2B__D_0__D_1__D_2__D_3, Tau_efmn_lvl1_part3_1_lvl2_part2_2__D_0__D_1__D_2__D_3, 2 );
			SlidePartitionDown
			( z_ai_lvl2_part1T__D_0_1__D_2_3,  z_ai_lvl2_part1_0__D_0_1__D_2_3,
			       z_ai_lvl2_part1_1__D_0_1__D_2_3,
			  /**/ /**/
			  z_ai_lvl2_part1B__D_0_1__D_2_3, z_ai_lvl2_part1_2__D_0_1__D_2_3, 1 );

		}
		//****
		z_small_temp2_lvl1_part1_1_perm0231__D_0__D_2__D_3__D_1.EmptyData();

		SlidePartitionDown
		( z_small_temp2_lvl1_part1T__D_0__D_1__D_2__D_3,  z_small_temp2_lvl1_part1_0__D_0__D_1__D_2__D_3,
		       z_small_temp2_lvl1_part1_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  z_small_temp2_lvl1_part1B__D_0__D_1__D_2__D_3, z_small_temp2_lvl1_part1_2__D_0__D_1__D_2__D_3, 1 );
		SlidePartitionDown
		( Tau_efmn_lvl1_part3T__D_0__D_1__D_2__D_3,  Tau_efmn_lvl1_part3_0__D_0__D_1__D_2__D_3,
		       Tau_efmn_lvl1_part3_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  Tau_efmn_lvl1_part3B__D_0__D_1__D_2__D_3, Tau_efmn_lvl1_part3_2__D_0__D_1__D_2__D_3, 3 );

	}
	//****
	z_small_temp2__D_0__D_1__D_2__D_3.EmptyData();
	//**** (out of 1)
	//**** Is real	0 shadows
		//Outputs:
		//  z_small_temp4__D_0__D_1__D_2__D_3
	PartitionDown(w_bmje__D_0__D_1__D_2__D_3, w_bmje_lvl1_part1T__D_0__D_1__D_2__D_3, w_bmje_lvl1_part1B__D_0__D_1__D_2__D_3, 1, 0);
	PartitionDown(x_bmej__D_0__D_1__D_2__D_3, x_bmej_lvl1_part1T__D_0__D_1__D_2__D_3, x_bmej_lvl1_part1B__D_0__D_1__D_2__D_3, 1, 0);
	PartitionDown(z_small_temp4__D_0__D_1__D_2__D_3, z_small_temp4_lvl1_part1T__D_0__D_1__D_2__D_3, z_small_temp4_lvl1_part1B__D_0__D_1__D_2__D_3, 1, 0);
	while(z_small_temp4_lvl1_part1T__D_0__D_1__D_2__D_3.Dimension(1) < z_small_temp4__D_0__D_1__D_2__D_3.Dimension(1))
	{
		RepartitionDown
		( w_bmje_lvl1_part1T__D_0__D_1__D_2__D_3,  w_bmje_lvl1_part1_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       w_bmje_lvl1_part1_1__D_0__D_1__D_2__D_3,
		  w_bmje_lvl1_part1B__D_0__D_1__D_2__D_3, w_bmje_lvl1_part1_2__D_0__D_1__D_2__D_3, 1, blkSize );
		RepartitionDown
		( x_bmej_lvl1_part1T__D_0__D_1__D_2__D_3,  x_bmej_lvl1_part1_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       x_bmej_lvl1_part1_1__D_0__D_1__D_2__D_3,
		  x_bmej_lvl1_part1B__D_0__D_1__D_2__D_3, x_bmej_lvl1_part1_2__D_0__D_1__D_2__D_3, 1, blkSize );
		RepartitionDown
		( z_small_temp4_lvl1_part1T__D_0__D_1__D_2__D_3,  z_small_temp4_lvl1_part1_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       z_small_temp4_lvl1_part1_1__D_0__D_1__D_2__D_3,
		  z_small_temp4_lvl1_part1B__D_0__D_1__D_2__D_3, z_small_temp4_lvl1_part1_2__D_0__D_1__D_2__D_3, 1, blkSize );

		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  z_small_temp4_lvl1_part1_1__D_0__D_1__D_2__D_3
		PartitionDown(w_bmje_lvl1_part1_1__D_0__D_1__D_2__D_3, w_bmje_lvl1_part1_1_lvl2_part2T__D_0__D_1__D_2__D_3, w_bmje_lvl1_part1_1_lvl2_part2B__D_0__D_1__D_2__D_3, 2, 0);
		PartitionDown(x_bmej_lvl1_part1_1__D_0__D_1__D_2__D_3, x_bmej_lvl1_part1_1_lvl2_part3T__D_0__D_1__D_2__D_3, x_bmej_lvl1_part1_1_lvl2_part3B__D_0__D_1__D_2__D_3, 3, 0);
		PartitionDown(z_small_temp4_lvl1_part1_1__D_0__D_1__D_2__D_3, z_small_temp4_lvl1_part1_1_lvl2_part2T__D_0__D_1__D_2__D_3, z_small_temp4_lvl1_part1_1_lvl2_part2B__D_0__D_1__D_2__D_3, 2, 0);
		while(z_small_temp4_lvl1_part1_1_lvl2_part2T__D_0__D_1__D_2__D_3.Dimension(2) < z_small_temp4_lvl1_part1_1__D_0__D_1__D_2__D_3.Dimension(2))
		{
			RepartitionDown
			( w_bmje_lvl1_part1_1_lvl2_part2T__D_0__D_1__D_2__D_3,  w_bmje_lvl1_part1_1_lvl2_part2_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       w_bmje_lvl1_part1_1_lvl2_part2_1__D_0__D_1__D_2__D_3,
			  w_bmje_lvl1_part1_1_lvl2_part2B__D_0__D_1__D_2__D_3, w_bmje_lvl1_part1_1_lvl2_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );
			RepartitionDown
			( x_bmej_lvl1_part1_1_lvl2_part3T__D_0__D_1__D_2__D_3,  x_bmej_lvl1_part1_1_lvl2_part3_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       x_bmej_lvl1_part1_1_lvl2_part3_1__D_0__D_1__D_2__D_3,
			  x_bmej_lvl1_part1_1_lvl2_part3B__D_0__D_1__D_2__D_3, x_bmej_lvl1_part1_1_lvl2_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );
			RepartitionDown
			( z_small_temp4_lvl1_part1_1_lvl2_part2T__D_0__D_1__D_2__D_3,  z_small_temp4_lvl1_part1_1_lvl2_part2_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       z_small_temp4_lvl1_part1_1_lvl2_part2_1__D_0__D_1__D_2__D_3,
			  z_small_temp4_lvl1_part1_1_lvl2_part2B__D_0__D_1__D_2__D_3, z_small_temp4_lvl1_part1_1_lvl2_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );

			   // x_bmej_lvl1_part1_1_lvl2_part3_1[D0,D1,D3,D2] <- x_bmej_lvl1_part1_1_lvl2_part3_1[D0,D1,D2,D3]
			x_bmej_lvl1_part1_1_lvl2_part3_1__D_0__D_1__D_3__D_2.AlignModesWith( modes_0_1_2_3, w_bmje_lvl1_part1_1_lvl2_part2_1__D_0__D_1__D_2__D_3, modes_0_1_3_2 );
			x_bmej_lvl1_part1_1_lvl2_part3_1__D_0__D_1__D_3__D_2.AllToAllRedistFrom( x_bmej_lvl1_part1_1_lvl2_part3_1__D_0__D_1__D_2__D_3, modes_2_3 );
if(commRank == 0){
flops += 3*prod(w_bmje_lvl1_part1_1_lvl2_part2_1__D_0__D_1__D_2__D_3.Shape());
}
			YAxpPx( 2.0, w_bmje_lvl1_part1_1_lvl2_part2_1__D_0__D_1__D_2__D_3, -1.0, x_bmej_lvl1_part1_1_lvl2_part3_1__D_0__D_1__D_3__D_2, perm_0_1_3_2, z_small_temp4_lvl1_part1_1_lvl2_part2_1__D_0__D_1__D_2__D_3 );
			x_bmej_lvl1_part1_1_lvl2_part3_1__D_0__D_1__D_3__D_2.EmptyData();

			SlidePartitionDown
			( w_bmje_lvl1_part1_1_lvl2_part2T__D_0__D_1__D_2__D_3,  w_bmje_lvl1_part1_1_lvl2_part2_0__D_0__D_1__D_2__D_3,
			       w_bmje_lvl1_part1_1_lvl2_part2_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  w_bmje_lvl1_part1_1_lvl2_part2B__D_0__D_1__D_2__D_3, w_bmje_lvl1_part1_1_lvl2_part2_2__D_0__D_1__D_2__D_3, 2 );
			SlidePartitionDown
			( x_bmej_lvl1_part1_1_lvl2_part3T__D_0__D_1__D_2__D_3,  x_bmej_lvl1_part1_1_lvl2_part3_0__D_0__D_1__D_2__D_3,
			       x_bmej_lvl1_part1_1_lvl2_part3_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  x_bmej_lvl1_part1_1_lvl2_part3B__D_0__D_1__D_2__D_3, x_bmej_lvl1_part1_1_lvl2_part3_2__D_0__D_1__D_2__D_3, 3 );
			SlidePartitionDown
			( z_small_temp4_lvl1_part1_1_lvl2_part2T__D_0__D_1__D_2__D_3,  z_small_temp4_lvl1_part1_1_lvl2_part2_0__D_0__D_1__D_2__D_3,
			       z_small_temp4_lvl1_part1_1_lvl2_part2_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  z_small_temp4_lvl1_part1_1_lvl2_part2B__D_0__D_1__D_2__D_3, z_small_temp4_lvl1_part1_1_lvl2_part2_2__D_0__D_1__D_2__D_3, 2 );

		}
		//****

		SlidePartitionDown
		( w_bmje_lvl1_part1T__D_0__D_1__D_2__D_3,  w_bmje_lvl1_part1_0__D_0__D_1__D_2__D_3,
		       w_bmje_lvl1_part1_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  w_bmje_lvl1_part1B__D_0__D_1__D_2__D_3, w_bmje_lvl1_part1_2__D_0__D_1__D_2__D_3, 1 );
		SlidePartitionDown
		( x_bmej_lvl1_part1T__D_0__D_1__D_2__D_3,  x_bmej_lvl1_part1_0__D_0__D_1__D_2__D_3,
		       x_bmej_lvl1_part1_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  x_bmej_lvl1_part1B__D_0__D_1__D_2__D_3, x_bmej_lvl1_part1_2__D_0__D_1__D_2__D_3, 1 );
		SlidePartitionDown
		( z_small_temp4_lvl1_part1T__D_0__D_1__D_2__D_3,  z_small_temp4_lvl1_part1_0__D_0__D_1__D_2__D_3,
		       z_small_temp4_lvl1_part1_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  z_small_temp4_lvl1_part1B__D_0__D_1__D_2__D_3, z_small_temp4_lvl1_part1_2__D_0__D_1__D_2__D_3, 1 );

	}
	//****
	//**** (out of 1)
	//**** Is real	0 shadows
		//Outputs:
		//  z_small_temp5__D_0__D_1__D_2__D_3
	PartitionDown(U_mnie__D_0__D_1__D_2__D_3, U_mnie_lvl1_part0T__D_0__D_1__D_2__D_3, U_mnie_lvl1_part0B__D_0__D_1__D_2__D_3, 0, 0);
	PartitionDown(U_mnie__D_0__D_1__D_2__D_3, U_mnie_lvl1_part1T__D_0__D_1__D_2__D_3, U_mnie_lvl1_part1B__D_0__D_1__D_2__D_3, 1, 0);
	PartitionDown(z_small_temp5__D_0__D_1__D_2__D_3, z_small_temp5_lvl1_part0T__D_0__D_1__D_2__D_3, z_small_temp5_lvl1_part0B__D_0__D_1__D_2__D_3, 0, 0);
	while(z_small_temp5_lvl1_part0T__D_0__D_1__D_2__D_3.Dimension(0) < z_small_temp5__D_0__D_1__D_2__D_3.Dimension(0))
	{
		RepartitionDown
		( U_mnie_lvl1_part0T__D_0__D_1__D_2__D_3,  U_mnie_lvl1_part0_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       U_mnie_lvl1_part0_1__D_0__D_1__D_2__D_3,
		  U_mnie_lvl1_part0B__D_0__D_1__D_2__D_3, U_mnie_lvl1_part0_2__D_0__D_1__D_2__D_3, 0, blkSize );
		RepartitionDown
		( U_mnie_lvl1_part1T__D_0__D_1__D_2__D_3,  U_mnie_lvl1_part1_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       U_mnie_lvl1_part1_1__D_0__D_1__D_2__D_3,
		  U_mnie_lvl1_part1B__D_0__D_1__D_2__D_3, U_mnie_lvl1_part1_2__D_0__D_1__D_2__D_3, 1, blkSize );
		RepartitionDown
		( z_small_temp5_lvl1_part0T__D_0__D_1__D_2__D_3,  z_small_temp5_lvl1_part0_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       z_small_temp5_lvl1_part0_1__D_0__D_1__D_2__D_3,
		  z_small_temp5_lvl1_part0B__D_0__D_1__D_2__D_3, z_small_temp5_lvl1_part0_2__D_0__D_1__D_2__D_3, 0, blkSize );

		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  z_small_temp5_lvl1_part0_1__D_0__D_1__D_2__D_3
		PartitionDown(U_mnie_lvl1_part0_1__D_0__D_1__D_2__D_3, U_mnie_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3, U_mnie_lvl1_part0_1_lvl2_part1B__D_0__D_1__D_2__D_3, 1, 0);
		PartitionDown(U_mnie_lvl1_part1_1__D_0__D_1__D_2__D_3, U_mnie_lvl1_part1_1_lvl2_part0T__D_0__D_1__D_2__D_3, U_mnie_lvl1_part1_1_lvl2_part0B__D_0__D_1__D_2__D_3, 0, 0);
		PartitionDown(z_small_temp5_lvl1_part0_1__D_0__D_1__D_2__D_3, z_small_temp5_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3, z_small_temp5_lvl1_part0_1_lvl2_part1B__D_0__D_1__D_2__D_3, 1, 0);
		while(z_small_temp5_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3.Dimension(1) < z_small_temp5_lvl1_part0_1__D_0__D_1__D_2__D_3.Dimension(1))
		{
			RepartitionDown
			( U_mnie_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3,  U_mnie_lvl1_part0_1_lvl2_part1_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       U_mnie_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3,
			  U_mnie_lvl1_part0_1_lvl2_part1B__D_0__D_1__D_2__D_3, U_mnie_lvl1_part0_1_lvl2_part1_2__D_0__D_1__D_2__D_3, 1, blkSize );
			RepartitionDown
			( U_mnie_lvl1_part1_1_lvl2_part0T__D_0__D_1__D_2__D_3,  U_mnie_lvl1_part1_1_lvl2_part0_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       U_mnie_lvl1_part1_1_lvl2_part0_1__D_0__D_1__D_2__D_3,
			  U_mnie_lvl1_part1_1_lvl2_part0B__D_0__D_1__D_2__D_3, U_mnie_lvl1_part1_1_lvl2_part0_2__D_0__D_1__D_2__D_3, 0, blkSize );
			RepartitionDown
			( z_small_temp5_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3,  z_small_temp5_lvl1_part0_1_lvl2_part1_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       z_small_temp5_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3,
			  z_small_temp5_lvl1_part0_1_lvl2_part1B__D_0__D_1__D_2__D_3, z_small_temp5_lvl1_part0_1_lvl2_part1_2__D_0__D_1__D_2__D_3, 1, blkSize );

			   // U_mnie_lvl1_part1_1_lvl2_part0_1[D1,D0,D2,D3] <- U_mnie_lvl1_part1_1_lvl2_part0_1[D0,D1,D2,D3]
			U_mnie_lvl1_part1_1_lvl2_part0_1__D_1__D_0__D_2__D_3.AlignModesWith( modes_0_1_2_3, U_mnie_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3, modes_1_0_2_3 );
			U_mnie_lvl1_part1_1_lvl2_part0_1__D_1__D_0__D_2__D_3.AllToAllRedistFrom( U_mnie_lvl1_part1_1_lvl2_part0_1__D_0__D_1__D_2__D_3, modes_0_1 );
if(commRank == 0){
flops += 3*prod(U_mnie_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3.Shape());
}
			YAxpPx( 2.0, U_mnie_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3, -1.0, U_mnie_lvl1_part1_1_lvl2_part0_1__D_1__D_0__D_2__D_3, perm_1_0_2_3, z_small_temp5_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3 );
			U_mnie_lvl1_part1_1_lvl2_part0_1__D_1__D_0__D_2__D_3.EmptyData();

			SlidePartitionDown
			( U_mnie_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3,  U_mnie_lvl1_part0_1_lvl2_part1_0__D_0__D_1__D_2__D_3,
			       U_mnie_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  U_mnie_lvl1_part0_1_lvl2_part1B__D_0__D_1__D_2__D_3, U_mnie_lvl1_part0_1_lvl2_part1_2__D_0__D_1__D_2__D_3, 1 );
			SlidePartitionDown
			( U_mnie_lvl1_part1_1_lvl2_part0T__D_0__D_1__D_2__D_3,  U_mnie_lvl1_part1_1_lvl2_part0_0__D_0__D_1__D_2__D_3,
			       U_mnie_lvl1_part1_1_lvl2_part0_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  U_mnie_lvl1_part1_1_lvl2_part0B__D_0__D_1__D_2__D_3, U_mnie_lvl1_part1_1_lvl2_part0_2__D_0__D_1__D_2__D_3, 0 );
			SlidePartitionDown
			( z_small_temp5_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3,  z_small_temp5_lvl1_part0_1_lvl2_part1_0__D_0__D_1__D_2__D_3,
			       z_small_temp5_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  z_small_temp5_lvl1_part0_1_lvl2_part1B__D_0__D_1__D_2__D_3, z_small_temp5_lvl1_part0_1_lvl2_part1_2__D_0__D_1__D_2__D_3, 1 );

		}
		//****

		SlidePartitionDown
		( U_mnie_lvl1_part0T__D_0__D_1__D_2__D_3,  U_mnie_lvl1_part0_0__D_0__D_1__D_2__D_3,
		       U_mnie_lvl1_part0_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  U_mnie_lvl1_part0B__D_0__D_1__D_2__D_3, U_mnie_lvl1_part0_2__D_0__D_1__D_2__D_3, 0 );
		SlidePartitionDown
		( U_mnie_lvl1_part1T__D_0__D_1__D_2__D_3,  U_mnie_lvl1_part1_0__D_0__D_1__D_2__D_3,
		       U_mnie_lvl1_part1_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  U_mnie_lvl1_part1B__D_0__D_1__D_2__D_3, U_mnie_lvl1_part1_2__D_0__D_1__D_2__D_3, 1 );
		SlidePartitionDown
		( z_small_temp5_lvl1_part0T__D_0__D_1__D_2__D_3,  z_small_temp5_lvl1_part0_0__D_0__D_1__D_2__D_3,
		       z_small_temp5_lvl1_part0_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  z_small_temp5_lvl1_part0B__D_0__D_1__D_2__D_3, z_small_temp5_lvl1_part0_2__D_0__D_1__D_2__D_3, 0 );

	}
	//****
	//**** (out of 1)
	//**** Is real	0 shadows
		//Outputs:
		//  z_ai__D_0_1__D_2_3
	PartitionDown(T_bfnj__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part2T__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part2B__D_0__D_1__D_2__D_3, 2, 0);
	PartitionDown(T_bfnj__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part3T__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part3B__D_0__D_1__D_2__D_3, 3, 0);
	PartitionDown(z_small_temp5__D_0__D_1__D_2__D_3, z_small_temp5_lvl1_part2T__D_0__D_1__D_2__D_3, z_small_temp5_lvl1_part2B__D_0__D_1__D_2__D_3, 2, 0);
	PartitionDown(G_mi__D_0_1__D_2_3, G_mi_lvl1_part1T__D_0_1__D_2_3, G_mi_lvl1_part1B__D_0_1__D_2_3, 1, 0);
	PartitionDown(z_small_temp4__D_0__D_1__D_2__D_3, z_small_temp4_lvl1_part2T__D_0__D_1__D_2__D_3, z_small_temp4_lvl1_part2B__D_0__D_1__D_2__D_3, 2, 0);
	PartitionDown(z_ai__D_0_1__D_2_3, z_ai_lvl1_part1T__D_0_1__D_2_3, z_ai_lvl1_part1B__D_0_1__D_2_3, 1, 0);
	while(T_bfnj_lvl1_part2T__D_0__D_1__D_2__D_3.Dimension(2) < T_bfnj__D_0__D_1__D_2__D_3.Dimension(2))
	{
		RepartitionDown
		( T_bfnj_lvl1_part2T__D_0__D_1__D_2__D_3,  T_bfnj_lvl1_part2_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       T_bfnj_lvl1_part2_1__D_0__D_1__D_2__D_3,
		  T_bfnj_lvl1_part2B__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );
		RepartitionDown
		( T_bfnj_lvl1_part3T__D_0__D_1__D_2__D_3,  T_bfnj_lvl1_part3_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       T_bfnj_lvl1_part3_1__D_0__D_1__D_2__D_3,
		  T_bfnj_lvl1_part3B__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );
		RepartitionDown
		( z_small_temp5_lvl1_part2T__D_0__D_1__D_2__D_3,  z_small_temp5_lvl1_part2_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       z_small_temp5_lvl1_part2_1__D_0__D_1__D_2__D_3,
		  z_small_temp5_lvl1_part2B__D_0__D_1__D_2__D_3, z_small_temp5_lvl1_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );
		RepartitionDown
		( G_mi_lvl1_part1T__D_0_1__D_2_3,  G_mi_lvl1_part1_0__D_0_1__D_2_3,
		  /**/ /**/
		       G_mi_lvl1_part1_1__D_0_1__D_2_3,
		  G_mi_lvl1_part1B__D_0_1__D_2_3, G_mi_lvl1_part1_2__D_0_1__D_2_3, 1, blkSize );
		RepartitionDown
		( z_small_temp4_lvl1_part2T__D_0__D_1__D_2__D_3,  z_small_temp4_lvl1_part2_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       z_small_temp4_lvl1_part2_1__D_0__D_1__D_2__D_3,
		  z_small_temp4_lvl1_part2B__D_0__D_1__D_2__D_3, z_small_temp4_lvl1_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );
		RepartitionDown
		( z_ai_lvl1_part1T__D_0_1__D_2_3,  z_ai_lvl1_part1_0__D_0_1__D_2_3,
		  /**/ /**/
		       z_ai_lvl1_part1_1__D_0_1__D_2_3,
		  z_ai_lvl1_part1B__D_0_1__D_2_3, z_ai_lvl1_part1_2__D_0_1__D_2_3, 1, blkSize );

		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  z_ai_lvl1_part1_1__D_0_1__D_2_3
		PartitionDown(T_bfnj_lvl1_part2_1__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, 3, 0);
		PartitionDown(T_bfnj_lvl1_part3_1__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part3_1_lvl2_part2T__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part3_1_lvl2_part2B__D_0__D_1__D_2__D_3, 2, 0);
		PartitionDown(H_me__D_0_1__D_2_3, H_me_lvl2_part0T__D_0_1__D_2_3, H_me_lvl2_part0B__D_0_1__D_2_3, 0, 0);
		while(T_bfnj_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3.Dimension(3) < T_bfnj_lvl1_part2_1__D_0__D_1__D_2__D_3.Dimension(3))
		{
			RepartitionDown
			( T_bfnj_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3,  T_bfnj_lvl1_part2_1_lvl2_part3_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       T_bfnj_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3,
			  T_bfnj_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part2_1_lvl2_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );
			RepartitionDown
			( T_bfnj_lvl1_part3_1_lvl2_part2T__D_0__D_1__D_2__D_3,  T_bfnj_lvl1_part3_1_lvl2_part2_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       T_bfnj_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_2__D_3,
			  T_bfnj_lvl1_part3_1_lvl2_part2B__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part3_1_lvl2_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );
			RepartitionDown
			( H_me_lvl2_part0T__D_0_1__D_2_3,  H_me_lvl2_part0_0__D_0_1__D_2_3,
			  /**/ /**/
			       H_me_lvl2_part0_1__D_0_1__D_2_3,
			  H_me_lvl2_part0B__D_0_1__D_2_3, H_me_lvl2_part0_2__D_0_1__D_2_3, 0, blkSize );

			   // H_me_lvl2_part0_1[*,*] <- H_me_lvl2_part0_1[D01,D23]
			H_me_lvl2_part0_1__S__S.AllGatherRedistFrom( H_me_lvl2_part0_1__D_0_1__D_2_3, modes_0_1_2_3 );
			z_small_temp3_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3.AlignModesWith( modes_0_1_2_3, T_bfnj_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3, modes_0_1_2_3 );
			overwrite_tmpShape_z_small = T_bfnj_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3.Shape();
			z_small_temp3_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3.ResizeTo( overwrite_tmpShape_z_small );
			   // T_bfnj_lvl1_part3_1_lvl2_part2_1[D0,D1,D3,D2] <- T_bfnj_lvl1_part3_1_lvl2_part2_1[D0,D1,D2,D3]
			T_bfnj_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_3__D_2.AlignModesWith( modes_0_1_2_3, T_bfnj_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3, modes_0_1_3_2 );
			T_bfnj_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_3__D_2.AllToAllRedistFrom( T_bfnj_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_2__D_3, modes_2_3 );
if(commRank == 0){
flops += 3*prod(T_bfnj_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3.Shape());
}
			YAxpPx( 2.0, T_bfnj_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3, -1.0, T_bfnj_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_3__D_2, perm_0_1_3_2, z_small_temp3_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3 );
			T_bfnj_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_3__D_2.EmptyData();
			   // z_small_temp3_lvl1_part2_1_lvl2_part3_1[D01,*,D2,D3] <- z_small_temp3_lvl1_part2_1_lvl2_part3_1[D0,D1,D2,D3]
			z_small_temp3_lvl1_part2_1_lvl2_part3_1__D_0_1__S__D_2__D_3.AlignModesWith( modes_0_2, z_ai_lvl1_part1_1__D_0_1__D_2_3, modes_0_1 );
			z_small_temp3_lvl1_part2_1_lvl2_part3_1__D_0_1__S__D_2__D_3.AllToAllRedistFrom( z_small_temp3_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3, modes_1 );
			z_small_temp3_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3.EmptyData();
			   // z_small_temp3_lvl1_part2_1_lvl2_part3_1[D01,*,D23,*] <- z_small_temp3_lvl1_part2_1_lvl2_part3_1[D01,*,D2,D3]
			z_small_temp3_lvl1_part2_1_lvl2_part3_1_perm0231__D_0_1__D_2_3__S__S.AlignModesWith( modes_0_2, z_ai_lvl1_part1_1__D_0_1__D_2_3, modes_0_1 );
			z_small_temp3_lvl1_part2_1_lvl2_part3_1_perm0231__D_0_1__D_2_3__S__S.AllToAllRedistFrom( z_small_temp3_lvl1_part2_1_lvl2_part3_1__D_0_1__S__D_2__D_3, modes_3 );
			z_small_temp3_lvl1_part2_1_lvl2_part3_1__D_0_1__S__D_2__D_3.EmptyData();
			   // 1.0 * z_small_temp3_lvl1_part2_1_lvl2_part3_1[D01,*,D23,*]_aime * H_me_lvl2_part0_1[*,*]_me + 1.0 * z_ai_lvl1_part1_1[D01,D23]_ai
if(commRank == 0){
flops += 2*z_small_temp3_lvl1_part2_1_lvl2_part3_1_perm0231__D_0_1__D_2_3__S__S.Dimension(3)*z_small_temp3_lvl1_part2_1_lvl2_part3_1_perm0231__D_0_1__D_2_3__S__S.Dimension(1)*z_small_temp3_lvl1_part2_1_lvl2_part3_1_perm0231__D_0_1__D_2_3__S__S.Dimension(2)*z_small_temp3_lvl1_part2_1_lvl2_part3_1_perm0231__D_0_1__D_2_3__S__S.Dimension(0)*1;
}
			LocalContractAndLocalEliminate(1.0, z_small_temp3_lvl1_part2_1_lvl2_part3_1_perm0231__D_0_1__D_2_3__S__S.LockedTensor(), indices_aime, false,
				H_me_lvl2_part0_1__S__S.LockedTensor(), indices_me, false,
				1.0, z_ai_lvl1_part1_1__D_0_1__D_2_3.Tensor(), indices_ai, false);
			z_small_temp3_lvl1_part2_1_lvl2_part3_1_perm0231__D_0_1__D_2_3__S__S.EmptyData();
			H_me_lvl2_part0_1__S__S.EmptyData();

			SlidePartitionDown
			( T_bfnj_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3,  T_bfnj_lvl1_part2_1_lvl2_part3_0__D_0__D_1__D_2__D_3,
			       T_bfnj_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  T_bfnj_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part2_1_lvl2_part3_2__D_0__D_1__D_2__D_3, 3 );
			SlidePartitionDown
			( T_bfnj_lvl1_part3_1_lvl2_part2T__D_0__D_1__D_2__D_3,  T_bfnj_lvl1_part3_1_lvl2_part2_0__D_0__D_1__D_2__D_3,
			       T_bfnj_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  T_bfnj_lvl1_part3_1_lvl2_part2B__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part3_1_lvl2_part2_2__D_0__D_1__D_2__D_3, 2 );
			SlidePartitionDown
			( H_me_lvl2_part0T__D_0_1__D_2_3,  H_me_lvl2_part0_0__D_0_1__D_2_3,
			       H_me_lvl2_part0_1__D_0_1__D_2_3,
			  /**/ /**/
			  H_me_lvl2_part0B__D_0_1__D_2_3, H_me_lvl2_part0_2__D_0_1__D_2_3, 0 );

		}
		//****
		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  z_ai_lvl1_part1_1__D_0_1__D_2_3
		PartitionDown(z_small_temp4_lvl1_part2_1__D_0__D_1__D_2__D_3, z_small_temp4_lvl1_part2_1_lvl2_part1T__D_0__D_1__D_2__D_3, z_small_temp4_lvl1_part2_1_lvl2_part1B__D_0__D_1__D_2__D_3, 1, 0);
		PartitionDown(t_fj__D_0_1__D_2_3, t_fj_lvl2_part1T__D_0_1__D_2_3, t_fj_lvl2_part1B__D_0_1__D_2_3, 1, 0);
		while(z_small_temp4_lvl1_part2_1_lvl2_part1T__D_0__D_1__D_2__D_3.Dimension(1) < z_small_temp4_lvl1_part2_1__D_0__D_1__D_2__D_3.Dimension(1))
		{
			RepartitionDown
			( z_small_temp4_lvl1_part2_1_lvl2_part1T__D_0__D_1__D_2__D_3,  z_small_temp4_lvl1_part2_1_lvl2_part1_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       z_small_temp4_lvl1_part2_1_lvl2_part1_1__D_0__D_1__D_2__D_3,
			  z_small_temp4_lvl1_part2_1_lvl2_part1B__D_0__D_1__D_2__D_3, z_small_temp4_lvl1_part2_1_lvl2_part1_2__D_0__D_1__D_2__D_3, 1, blkSize );
			RepartitionDown
			( t_fj_lvl2_part1T__D_0_1__D_2_3,  t_fj_lvl2_part1_0__D_0_1__D_2_3,
			  /**/ /**/
			       t_fj_lvl2_part1_1__D_0_1__D_2_3,
			  t_fj_lvl2_part1B__D_0_1__D_2_3, t_fj_lvl2_part1_2__D_0_1__D_2_3, 1, blkSize );

			   // t_fj_lvl2_part1_1[*,*] <- t_fj_lvl2_part1_1[D01,D23]
			t_fj_lvl2_part1_1_perm10__S__S.AllGatherRedistFrom( t_fj_lvl2_part1_1__D_0_1__D_2_3, modes_0_1_2_3 );
			   // z_small_temp4_lvl1_part2_1_lvl2_part1_1[D01,*,D2,D3] <- z_small_temp4_lvl1_part2_1_lvl2_part1_1[D0,D1,D2,D3]
			z_small_temp4_lvl1_part2_1_lvl2_part1_1__D_0_1__S__D_2__D_3.AlignModesWith( modes_0_2, z_ai_lvl1_part1_1__D_0_1__D_2_3, modes_0_1 );
			z_small_temp4_lvl1_part2_1_lvl2_part1_1__D_0_1__S__D_2__D_3.AllToAllRedistFrom( z_small_temp4_lvl1_part2_1_lvl2_part1_1__D_0__D_1__D_2__D_3, modes_1 );
			   // z_small_temp4_lvl1_part2_1_lvl2_part1_1[D01,*,D23,*] <- z_small_temp4_lvl1_part2_1_lvl2_part1_1[D01,*,D2,D3]
			z_small_temp4_lvl1_part2_1_lvl2_part1_1_perm0213__D_0_1__D_2_3__S__S.AlignModesWith( modes_0_2, z_ai_lvl1_part1_1__D_0_1__D_2_3, modes_0_1 );
			z_small_temp4_lvl1_part2_1_lvl2_part1_1_perm0213__D_0_1__D_2_3__S__S.AllToAllRedistFrom( z_small_temp4_lvl1_part2_1_lvl2_part1_1__D_0_1__S__D_2__D_3, modes_3 );
			z_small_temp4_lvl1_part2_1_lvl2_part1_1__D_0_1__S__D_2__D_3.EmptyData();
			   // 1.0 * z_small_temp4_lvl1_part2_1_lvl2_part1_1[D01,*,D23,*]_aime * t_fj_lvl2_part1_1[*,*]_me + 1.0 * z_ai_lvl1_part1_1[D01,D23]_ai
if(commRank == 0){
flops += 2*z_small_temp4_lvl1_part2_1_lvl2_part1_1_perm0213__D_0_1__D_2_3__S__S.Dimension(1)*z_small_temp4_lvl1_part2_1_lvl2_part1_1_perm0213__D_0_1__D_2_3__S__S.Dimension(3)*z_small_temp4_lvl1_part2_1_lvl2_part1_1_perm0213__D_0_1__D_2_3__S__S.Dimension(2)*z_small_temp4_lvl1_part2_1_lvl2_part1_1_perm0213__D_0_1__D_2_3__S__S.Dimension(0)*1;
}
			LocalContractAndLocalEliminate(1.0, z_small_temp4_lvl1_part2_1_lvl2_part1_1_perm0213__D_0_1__D_2_3__S__S.LockedTensor(), indices_aime, false,
				t_fj_lvl2_part1_1_perm10__S__S.LockedTensor(), indices_me, false,
				1.0, z_ai_lvl1_part1_1__D_0_1__D_2_3.Tensor(), indices_ai, false);
			z_small_temp4_lvl1_part2_1_lvl2_part1_1_perm0213__D_0_1__D_2_3__S__S.EmptyData();
			t_fj_lvl2_part1_1_perm10__S__S.EmptyData();

			SlidePartitionDown
			( z_small_temp4_lvl1_part2_1_lvl2_part1T__D_0__D_1__D_2__D_3,  z_small_temp4_lvl1_part2_1_lvl2_part1_0__D_0__D_1__D_2__D_3,
			       z_small_temp4_lvl1_part2_1_lvl2_part1_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  z_small_temp4_lvl1_part2_1_lvl2_part1B__D_0__D_1__D_2__D_3, z_small_temp4_lvl1_part2_1_lvl2_part1_2__D_0__D_1__D_2__D_3, 1 );
			SlidePartitionDown
			( t_fj_lvl2_part1T__D_0_1__D_2_3,  t_fj_lvl2_part1_0__D_0_1__D_2_3,
			       t_fj_lvl2_part1_1__D_0_1__D_2_3,
			  /**/ /**/
			  t_fj_lvl2_part1B__D_0_1__D_2_3, t_fj_lvl2_part1_2__D_0_1__D_2_3, 1 );

		}
		//****
		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  z_ai_lvl1_part1_1__D_0_1__D_2_3
		PartitionDown(z_small_temp5_lvl1_part2_1__D_0__D_1__D_2__D_3, z_small_temp5_lvl1_part2_1_lvl2_part0T__D_0__D_1__D_2__D_3, z_small_temp5_lvl1_part2_1_lvl2_part0B__D_0__D_1__D_2__D_3, 0, 0);
		PartitionDown(T_bfnj__D_0__D_1__D_2__D_3, T_bfnj_lvl2_part2T__D_0__D_1__D_2__D_3, T_bfnj_lvl2_part2B__D_0__D_1__D_2__D_3, 2, 0);
		while(z_small_temp5_lvl1_part2_1_lvl2_part0T__D_0__D_1__D_2__D_3.Dimension(0) < z_small_temp5_lvl1_part2_1__D_0__D_1__D_2__D_3.Dimension(0))
		{
			RepartitionDown
			( z_small_temp5_lvl1_part2_1_lvl2_part0T__D_0__D_1__D_2__D_3,  z_small_temp5_lvl1_part2_1_lvl2_part0_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       z_small_temp5_lvl1_part2_1_lvl2_part0_1__D_0__D_1__D_2__D_3,
			  z_small_temp5_lvl1_part2_1_lvl2_part0B__D_0__D_1__D_2__D_3, z_small_temp5_lvl1_part2_1_lvl2_part0_2__D_0__D_1__D_2__D_3, 0, blkSize );
			RepartitionDown
			( T_bfnj_lvl2_part2T__D_0__D_1__D_2__D_3,  T_bfnj_lvl2_part2_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       T_bfnj_lvl2_part2_1__D_0__D_1__D_2__D_3,
			  T_bfnj_lvl2_part2B__D_0__D_1__D_2__D_3, T_bfnj_lvl2_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );

			Permute( T_bfnj_lvl2_part2_1__D_0__D_1__D_2__D_3, T_bfnj_lvl2_part2_1_perm2310__D_2__D_3__D_1__D_0 );
			   // z_small_temp5_lvl1_part2_1_lvl2_part0_1[D0,D3,D2,D1] <- z_small_temp5_lvl1_part2_1_lvl2_part0_1[D0,D1,D2,D3]
			z_small_temp5_lvl1_part2_1_lvl2_part0_1__D_0__D_3__D_2__D_1.AlignModesWith( modes_0_1_3, T_bfnj_lvl2_part2_1__D_0__D_1__D_2__D_3, modes_2_3_1 );
			z_small_temp5_lvl1_part2_1_lvl2_part0_1__D_0__D_3__D_2__D_1.AllToAllRedistFrom( z_small_temp5_lvl1_part2_1_lvl2_part0_1__D_0__D_1__D_2__D_3, modes_1_3 );
			   // z_small_temp5_lvl1_part2_1_lvl2_part0_1[D2,D3,*,D1] <- z_small_temp5_lvl1_part2_1_lvl2_part0_1[D0,D3,D2,D1]
			z_small_temp5_lvl1_part2_1_lvl2_part0_1_perm2013__S__D_2__D_3__D_1.AlignModesWith( modes_0_1_3, T_bfnj_lvl2_part2_1__D_0__D_1__D_2__D_3, modes_2_3_1 );
			z_small_temp5_lvl1_part2_1_lvl2_part0_1_perm2013__S__D_2__D_3__D_1.AllToAllRedistFrom( z_small_temp5_lvl1_part2_1_lvl2_part0_1__D_0__D_3__D_2__D_1, modes_0_2 );
			z_small_temp5_lvl1_part2_1_lvl2_part0_1__D_0__D_3__D_2__D_1.EmptyData();
			z_ai_lvl1_part1_1_perm10342__S__D_0__D_2__D_3__D_1.AlignModesWith( modes_0, T_bfnj_lvl2_part2_1__D_0__D_1__D_2__D_3, modes_0 );
			overwrite_tmpShape_z_small = z_ai_lvl1_part1_1__D_0_1__D_2_3.Shape();
			overwrite_tmpShape_z_small.push_back( g.Shape()[1] );
			overwrite_tmpShape_z_small.push_back( g.Shape()[2] );
			overwrite_tmpShape_z_small.push_back( g.Shape()[3] );
			z_ai_lvl1_part1_1_perm10342__S__D_0__D_2__D_3__D_1.ResizeTo( overwrite_tmpShape_z_small );
			   // -1.0 * z_small_temp5_lvl1_part2_1_lvl2_part0_1[D2,D3,*,D1]_imne * T_bfnj_lvl2_part2_1[D0,D1,D2,D3]_mnea + 0.0 * z_ai_lvl1_part1_1[D0,*,D1,D2,D3]_iamne
if(commRank == 0){
flops += 2*z_small_temp5_lvl1_part2_1_lvl2_part0_1_perm2013__S__D_2__D_3__D_1.Dimension(0)*z_small_temp5_lvl1_part2_1_lvl2_part0_1_perm2013__S__D_2__D_3__D_1.Dimension(3)*z_small_temp5_lvl1_part2_1_lvl2_part0_1_perm2013__S__D_2__D_3__D_1.Dimension(1)*z_small_temp5_lvl1_part2_1_lvl2_part0_1_perm2013__S__D_2__D_3__D_1.Dimension(2)*T_bfnj_lvl2_part2_1_perm2310__D_2__D_3__D_1__D_0.Dimension(0);
}
			LocalContract(-1.0, z_small_temp5_lvl1_part2_1_lvl2_part0_1_perm2013__S__D_2__D_3__D_1.LockedTensor(), indices_imne, false,
				T_bfnj_lvl2_part2_1_perm2310__D_2__D_3__D_1__D_0.LockedTensor(), indices_mnea, false,
				0.0, z_ai_lvl1_part1_1_perm10342__S__D_0__D_2__D_3__D_1.Tensor(), indices_iamne, false);
			z_small_temp5_lvl1_part2_1_lvl2_part0_1_perm2013__S__D_2__D_3__D_1.EmptyData();
			T_bfnj_lvl2_part2_1_perm2310__D_2__D_3__D_1__D_0.EmptyData();
			   // z_ai_lvl1_part1_1[D01,D23] <- z_ai_lvl1_part1_1[D0,*,D1,D2,D3] (with SumScatter on (D1)(D2)(D3))
			z_ai_lvl1_part1_1__D_0_1__D_2_3.ReduceScatterUpdateRedistFrom( z_ai_lvl1_part1_1_perm10342__S__D_0__D_2__D_3__D_1, 1.0, modes_4_3_2 );
			z_ai_lvl1_part1_1_perm10342__S__D_0__D_2__D_3__D_1.EmptyData();

			SlidePartitionDown
			( z_small_temp5_lvl1_part2_1_lvl2_part0T__D_0__D_1__D_2__D_3,  z_small_temp5_lvl1_part2_1_lvl2_part0_0__D_0__D_1__D_2__D_3,
			       z_small_temp5_lvl1_part2_1_lvl2_part0_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  z_small_temp5_lvl1_part2_1_lvl2_part0B__D_0__D_1__D_2__D_3, z_small_temp5_lvl1_part2_1_lvl2_part0_2__D_0__D_1__D_2__D_3, 0 );
			SlidePartitionDown
			( T_bfnj_lvl2_part2T__D_0__D_1__D_2__D_3,  T_bfnj_lvl2_part2_0__D_0__D_1__D_2__D_3,
			       T_bfnj_lvl2_part2_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  T_bfnj_lvl2_part2B__D_0__D_1__D_2__D_3, T_bfnj_lvl2_part2_2__D_0__D_1__D_2__D_3, 2 );

		}
		//****
		Permute( z_ai_lvl1_part1_1__D_0_1__D_2_3, z_ai_lvl1_part1_1_perm10__D_2_3__D_0_1 );
		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  z_ai_lvl1_part1_1_perm10__D_2_3__D_0_1
		PartitionDown(G_mi_lvl1_part1_1__D_0_1__D_2_3, G_mi_lvl1_part1_1_lvl2_part0T__D_0_1__D_2_3, G_mi_lvl1_part1_1_lvl2_part0B__D_0_1__D_2_3, 0, 0);
		PartitionDown(t_fj__D_0_1__D_2_3, t_fj_lvl2_part1T__D_0_1__D_2_3, t_fj_lvl2_part1B__D_0_1__D_2_3, 1, 0);
		while(G_mi_lvl1_part1_1_lvl2_part0T__D_0_1__D_2_3.Dimension(0) < G_mi_lvl1_part1_1__D_0_1__D_2_3.Dimension(0))
		{
			RepartitionDown
			( G_mi_lvl1_part1_1_lvl2_part0T__D_0_1__D_2_3,  G_mi_lvl1_part1_1_lvl2_part0_0__D_0_1__D_2_3,
			  /**/ /**/
			       G_mi_lvl1_part1_1_lvl2_part0_1__D_0_1__D_2_3,
			  G_mi_lvl1_part1_1_lvl2_part0B__D_0_1__D_2_3, G_mi_lvl1_part1_1_lvl2_part0_2__D_0_1__D_2_3, 0, blkSize );
			RepartitionDown
			( t_fj_lvl2_part1T__D_0_1__D_2_3,  t_fj_lvl2_part1_0__D_0_1__D_2_3,
			  /**/ /**/
			       t_fj_lvl2_part1_1__D_0_1__D_2_3,
			  t_fj_lvl2_part1B__D_0_1__D_2_3, t_fj_lvl2_part1_2__D_0_1__D_2_3, 1, blkSize );

			   // t_fj_lvl2_part1_1[D01,*] <- t_fj_lvl2_part1_1[D01,D23]
			t_fj_lvl2_part1_1_perm10__S__D_0_1.AlignModesWith( modes_0, z_ai_lvl1_part1_1__D_0_1__D_2_3, modes_0 );
			t_fj_lvl2_part1_1_perm10__S__D_0_1.AllGatherRedistFrom( t_fj_lvl2_part1_1__D_0_1__D_2_3, modes_2_3 );
			   // G_mi_lvl1_part1_1_lvl2_part0_1[*,D23] <- G_mi_lvl1_part1_1_lvl2_part0_1[D01,D23]
			G_mi_lvl1_part1_1_lvl2_part0_1_perm10__D_2_3__S.AlignModesWith( modes_1, z_ai_lvl1_part1_1__D_0_1__D_2_3, modes_1 );
			G_mi_lvl1_part1_1_lvl2_part0_1_perm10__D_2_3__S.AllGatherRedistFrom( G_mi_lvl1_part1_1_lvl2_part0_1__D_0_1__D_2_3, modes_0_1 );
			   // -1.0 * G_mi_lvl1_part1_1_lvl2_part0_1[*,D23]_im * t_fj_lvl2_part1_1[D01,*]_ma + 1.0 * z_ai_lvl1_part1_1[D01,D23]_ia
if(commRank == 0){
flops += 2*G_mi_lvl1_part1_1_lvl2_part0_1_perm10__D_2_3__S.Dimension(0)*G_mi_lvl1_part1_1_lvl2_part0_1_perm10__D_2_3__S.Dimension(1)*t_fj_lvl2_part1_1_perm10__S__D_0_1.Dimension(0);
}
			LocalContractAndLocalEliminate(-1.0, G_mi_lvl1_part1_1_lvl2_part0_1_perm10__D_2_3__S.LockedTensor(), indices_im, false,
				t_fj_lvl2_part1_1_perm10__S__D_0_1.LockedTensor(), indices_ma, false,
				1.0, z_ai_lvl1_part1_1_perm10__D_2_3__D_0_1.Tensor(), indices_ia, false);
			G_mi_lvl1_part1_1_lvl2_part0_1_perm10__D_2_3__S.EmptyData();
			t_fj_lvl2_part1_1_perm10__S__D_0_1.EmptyData();

			SlidePartitionDown
			( G_mi_lvl1_part1_1_lvl2_part0T__D_0_1__D_2_3,  G_mi_lvl1_part1_1_lvl2_part0_0__D_0_1__D_2_3,
			       G_mi_lvl1_part1_1_lvl2_part0_1__D_0_1__D_2_3,
			  /**/ /**/
			  G_mi_lvl1_part1_1_lvl2_part0B__D_0_1__D_2_3, G_mi_lvl1_part1_1_lvl2_part0_2__D_0_1__D_2_3, 0 );
			SlidePartitionDown
			( t_fj_lvl2_part1T__D_0_1__D_2_3,  t_fj_lvl2_part1_0__D_0_1__D_2_3,
			       t_fj_lvl2_part1_1__D_0_1__D_2_3,
			  /**/ /**/
			  t_fj_lvl2_part1B__D_0_1__D_2_3, t_fj_lvl2_part1_2__D_0_1__D_2_3, 1 );

		}
		//****
		Permute( z_ai_lvl1_part1_1_perm10__D_2_3__D_0_1, z_ai_lvl1_part1_1__D_0_1__D_2_3 );
		z_ai_lvl1_part1_1_perm10__D_2_3__D_0_1.EmptyData();

		SlidePartitionDown
		( T_bfnj_lvl1_part2T__D_0__D_1__D_2__D_3,  T_bfnj_lvl1_part2_0__D_0__D_1__D_2__D_3,
		       T_bfnj_lvl1_part2_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  T_bfnj_lvl1_part2B__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part2_2__D_0__D_1__D_2__D_3, 2 );
		SlidePartitionDown
		( T_bfnj_lvl1_part3T__D_0__D_1__D_2__D_3,  T_bfnj_lvl1_part3_0__D_0__D_1__D_2__D_3,
		       T_bfnj_lvl1_part3_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  T_bfnj_lvl1_part3B__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part3_2__D_0__D_1__D_2__D_3, 3 );
		SlidePartitionDown
		( z_small_temp5_lvl1_part2T__D_0__D_1__D_2__D_3,  z_small_temp5_lvl1_part2_0__D_0__D_1__D_2__D_3,
		       z_small_temp5_lvl1_part2_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  z_small_temp5_lvl1_part2B__D_0__D_1__D_2__D_3, z_small_temp5_lvl1_part2_2__D_0__D_1__D_2__D_3, 2 );
		SlidePartitionDown
		( G_mi_lvl1_part1T__D_0_1__D_2_3,  G_mi_lvl1_part1_0__D_0_1__D_2_3,
		       G_mi_lvl1_part1_1__D_0_1__D_2_3,
		  /**/ /**/
		  G_mi_lvl1_part1B__D_0_1__D_2_3, G_mi_lvl1_part1_2__D_0_1__D_2_3, 1 );
		SlidePartitionDown
		( z_ai_lvl1_part1T__D_0_1__D_2_3,  z_ai_lvl1_part1_0__D_0_1__D_2_3,
		       z_ai_lvl1_part1_1__D_0_1__D_2_3,
		  /**/ /**/
		  z_ai_lvl1_part1B__D_0_1__D_2_3, z_ai_lvl1_part1_2__D_0_1__D_2_3, 1 );
		SlidePartitionDown
		( z_small_temp4_lvl1_part2T__D_0__D_1__D_2__D_3,  z_small_temp4_lvl1_part2_0__D_0__D_1__D_2__D_3,
		       z_small_temp4_lvl1_part2_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  z_small_temp4_lvl1_part2B__D_0__D_1__D_2__D_3, z_small_temp4_lvl1_part2_2__D_0__D_1__D_2__D_3, 2 );

	}
	//****
	z_small_temp5__D_0__D_1__D_2__D_3.EmptyData();
	z_small_temp4__D_0__D_1__D_2__D_3.EmptyData();


//****


//END_CODE

    /*****************************************/
    mpi::Barrier(g.OwningComm());
    runTime = mpi::Time() - startTime;
    gflops = flops / (1e9 * runTime);
#ifdef CORRECTNESS
    DistTensor<double> diff_z_small(dist__D_0_1__D_2_3, g);
    diff_z_small.ResizeTo(check_z_small);
    Diff(check_z_small, z_ai__D_0_1__D_2_3, diff_z_small);
   norm = 1.0;
   norm = Norm(diff_z_small);
   if (commRank == 0){
     std::cout << "NORM_z_small " << norm << std::endl;
   }
#endif

    //****

    //------------------------------------//

    //****
#ifdef PROFILE
    if (commRank == 0)
        Timer::printTimers();
#endif

    //****
    if (commRank == 0) {
        std::cout << "RUNTIME " << runTime << std::endl;
        std::cout << "FLOPS " << flops << std::endl;
        std::cout << "GFLOPS " << gflops << std::endl;
    }
}

int main(int argc, char* argv[]) {
    Initialize(argc, argv);
    Unsigned i;
    mpi::Comm comm = mpi::COMM_WORLD;
    const Int commRank = mpi::CommRank(comm);
    const Int commSize = mpi::CommSize(comm);
    //    printf("My Rank: %d\n", commRank);
    try {
        Params args;

        ProcessInput(argc, argv, args);

        if (commRank == 0 && commSize != args.nProcs) {
            std::cerr
                    << "program not started with correct number of processes\n";
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

        const Grid g(comm, args.gridShape);
        DistTensorTest<double>(g, args.n_o, args.n_v, args.blkSize, args.testIter);

    } catch (std::exception& e) {
        ReportException(e);
    }

    Finalize();
    //printf("Completed\n");
    return 0;
}


