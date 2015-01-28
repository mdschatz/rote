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
ObjShape overwrite_tmpShape_G;
TensorDistribution dist__S__S = tmen::StringToTensorDist("[(),()]");
TensorDistribution dist__S__D_2_3 = tmen::StringToTensorDist("[(),(2,3)]");
TensorDistribution dist__S__D_2__D_0__D_1__D_3 = tmen::StringToTensorDist("[(),(2),(0),(1),(3)]");
TensorDistribution dist__D_0__S__D_2_3__S = tmen::StringToTensorDist("[(0),(),(2,3),()]");
TensorDistribution dist__D_0__D_1__S__D_3 = tmen::StringToTensorDist("[(0),(1),(),(3)]");
TensorDistribution dist__D_0__D_1__D_2_3__S = tmen::StringToTensorDist("[(0),(1),(2,3),()]");
TensorDistribution dist__D_0__D_1__D_2__S = tmen::StringToTensorDist("[(0),(1),(2),()]");
TensorDistribution dist__D_0__D_1__D_2__D_3 = tmen::StringToTensorDist("[(0),(1),(2),(3)]");
TensorDistribution dist__D_0__D_1__D_3__D_2 = tmen::StringToTensorDist("[(0),(1),(3),(2)]");
TensorDistribution dist__D_1__D_0__D_2__D_3 = tmen::StringToTensorDist("[(1),(0),(2),(3)]");
TensorDistribution dist__D_0_1__S__D_2_3__S = tmen::StringToTensorDist("[(0,1),(),(2,3),()]");
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
Permutation perm_1_0( 2 );
perm_1_0[0] = 1;
perm_1_0[1] = 0;
Permutation perm_1_0_2_3( 4 );
perm_1_0_2_3[0] = 1;
perm_1_0_2_3[1] = 0;
perm_1_0_2_3[2] = 2;
perm_1_0_2_3[3] = 3;
Permutation perm_2_0_1_3( 4 );
perm_2_0_1_3[0] = 2;
perm_2_0_1_3[1] = 0;
perm_2_0_1_3[2] = 1;
perm_2_0_1_3[3] = 3;
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
ModeArray modes_2( 1 );
modes_2[0] = 2;
ModeArray modes_2_3( 2 );
modes_2_3[0] = 2;
modes_2_3[1] = 3;
ModeArray modes_3( 1 );
modes_3[0] = 3;
ModeArray modes_4_3_2( 3 );
modes_4_3_2[0] = 4;
modes_4_3_2[1] = 3;
modes_4_3_2[2] = 2;
IndexArray indices_efni( 4 );
indices_efni[0] = 'e';
indices_efni[1] = 'f';
indices_efni[2] = 'n';
indices_efni[3] = 'i';
IndexArray indices_ei( 2 );
indices_ei[0] = 'e';
indices_ei[1] = 'i';
IndexArray indices_me( 2 );
indices_me[0] = 'm';
indices_me[1] = 'e';
IndexArray indices_mefn( 4 );
indices_mefn[0] = 'm';
indices_mefn[1] = 'e';
indices_mefn[2] = 'f';
indices_mefn[3] = 'n';
IndexArray indices_mi( 2 );
indices_mi[0] = 'm';
indices_mi[1] = 'i';
IndexArray indices_miefn( 5 );
indices_miefn[0] = 'm';
indices_miefn[1] = 'i';
indices_miefn[2] = 'e';
indices_miefn[3] = 'f';
indices_miefn[4] = 'n';
IndexArray indices_mine( 4 );
indices_mine[0] = 'm';
indices_mine[1] = 'i';
indices_mine[2] = 'n';
indices_mine[3] = 'e';
IndexArray indices_ne( 2 );
indices_ne[0] = 'n';
indices_ne[1] = 'e';
	//G_mi[D01,D23]
DistTensor<double> G_mi__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//G_mi_lvl0_part0B[D01,D23]
DistTensor<double> G_mi_lvl0_part0B__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//G_mi_lvl0_part0T[D01,D23]
DistTensor<double> G_mi_lvl0_part0T__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//G_mi_lvl1_part0B[D01,D23]
DistTensor<double> G_mi_lvl1_part0B__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//G_mi_lvl1_part0T[D01,D23]
DistTensor<double> G_mi_lvl1_part0T__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//G_mi_lvl1_part0_0[D01,D23]
DistTensor<double> G_mi_lvl1_part0_0__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//G_mi_lvl1_part0_1[D01,D23]
DistTensor<double> G_mi_lvl1_part0_1__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//G_mi_lvl1_part0_2[D01,D23]
DistTensor<double> G_mi_lvl1_part0_2__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//G_mi_lvl2_part0B[D01,D23]
DistTensor<double> G_mi_lvl2_part0B__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//G_mi_lvl2_part0T[D01,D23]
DistTensor<double> G_mi_lvl2_part0T__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//G_mi_lvl2_part0_0[D01,D23]
DistTensor<double> G_mi_lvl2_part0_0__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//G_mi_lvl2_part0_1[D01,D23]
DistTensor<double> G_mi_lvl2_part0_1__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//G_mi_lvl2_part0_1[*,D2,D0,D1,D3]
DistTensor<double> G_mi_lvl2_part0_1__S__D_2__D_0__D_1__D_3( dist__S__D_2__D_0__D_1__D_3, g );
	//G_mi_lvl2_part0_2[D01,D23]
DistTensor<double> G_mi_lvl2_part0_2__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//H_me[D01,D23]
DistTensor<double> H_me__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//H_me_lvl0_part0B[D01,D23]
DistTensor<double> H_me_lvl0_part0B__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//H_me_lvl0_part0T[D01,D23]
DistTensor<double> H_me_lvl0_part0T__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//H_me_lvl1_part0B[D01,D23]
DistTensor<double> H_me_lvl1_part0B__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//H_me_lvl1_part0T[D01,D23]
DistTensor<double> H_me_lvl1_part0T__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//H_me_lvl1_part0_0[D01,D23]
DistTensor<double> H_me_lvl1_part0_0__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//H_me_lvl1_part0_1[D01,D23]
DistTensor<double> H_me_lvl1_part0_1__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//H_me_lvl1_part0_1_lvl1_part1B[D01,D23]
DistTensor<double> H_me_lvl1_part0_1_lvl1_part1B__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//H_me_lvl1_part0_1_lvl1_part1T[D01,D23]
DistTensor<double> H_me_lvl1_part0_1_lvl1_part1T__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//H_me_lvl1_part0_1_lvl2_part1B[D01,D23]
DistTensor<double> H_me_lvl1_part0_1_lvl2_part1B__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//H_me_lvl1_part0_1_lvl2_part1T[D01,D23]
DistTensor<double> H_me_lvl1_part0_1_lvl2_part1T__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//H_me_lvl1_part0_1_lvl2_part1_0[D01,D23]
DistTensor<double> H_me_lvl1_part0_1_lvl2_part1_0__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//H_me_lvl1_part0_1_lvl2_part1_1[D01,D23]
DistTensor<double> H_me_lvl1_part0_1_lvl2_part1_1__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//H_me_lvl1_part0_1_lvl2_part1_1[D01,*]
DistTensor<double> H_me_lvl1_part0_1_lvl2_part1_1__D_0_1__S( dist__D_0_1__S, g );
	//H_me_lvl1_part0_1_lvl2_part1_2[D01,D23]
DistTensor<double> H_me_lvl1_part0_1_lvl2_part1_2__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//H_me_lvl1_part0_2[D01,D23]
DistTensor<double> H_me_lvl1_part0_2__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//T_bfnj[D0,D1,D2,D3]
DistTensor<double> T_bfnj__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//T_bfnj_lvl0_part3B[D0,D1,D2,D3]
DistTensor<double> T_bfnj_lvl0_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//T_bfnj_lvl0_part3T[D0,D1,D2,D3]
DistTensor<double> T_bfnj_lvl0_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//T_bfnj_lvl1_part3B[D0,D1,D2,D3]
DistTensor<double> T_bfnj_lvl1_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//T_bfnj_lvl1_part3T[D0,D1,D2,D3]
DistTensor<double> T_bfnj_lvl1_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//T_bfnj_lvl1_part3_0[D0,D1,D2,D3]
DistTensor<double> T_bfnj_lvl1_part3_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//T_bfnj_lvl1_part3_1[D0,D1,D2,D3]
DistTensor<double> T_bfnj_lvl1_part3_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> T_bfnj_lvl1_part3_1_perm0132__D_0__D_1__D_3__D_2( dist__D_0__D_1__D_2__D_3, g );
T_bfnj_lvl1_part3_1_perm0132__D_0__D_1__D_3__D_2.SetLocalPermutation( perm_0_1_3_2 );
	//T_bfnj_lvl1_part3_2[D0,D1,D2,D3]
DistTensor<double> T_bfnj_lvl1_part3_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//t_fj[D01,D23]
DistTensor<double> t_fj__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//t_fj_lvl0_part0B[D01,D23]
DistTensor<double> t_fj_lvl0_part0B__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//t_fj_lvl0_part0T[D01,D23]
DistTensor<double> t_fj_lvl0_part0T__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//t_fj_lvl1_part0B[D01,D23]
DistTensor<double> t_fj_lvl1_part0B__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//t_fj_lvl1_part0T[D01,D23]
DistTensor<double> t_fj_lvl1_part0T__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//t_fj_lvl1_part0_0[D01,D23]
DistTensor<double> t_fj_lvl1_part0_0__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//t_fj_lvl1_part0_1[D01,D23]
DistTensor<double> t_fj_lvl1_part0_1__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//t_fj_lvl1_part0_1_lvl1_part1B[D01,D23]
DistTensor<double> t_fj_lvl1_part0_1_lvl1_part1B__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//t_fj_lvl1_part0_1_lvl1_part1T[D01,D23]
DistTensor<double> t_fj_lvl1_part0_1_lvl1_part1T__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//t_fj_lvl1_part0_1_lvl2_part1B[D01,D23]
DistTensor<double> t_fj_lvl1_part0_1_lvl2_part1B__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//t_fj_lvl1_part0_1_lvl2_part1T[D01,D23]
DistTensor<double> t_fj_lvl1_part0_1_lvl2_part1T__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//t_fj_lvl1_part0_1_lvl2_part1_0[D01,D23]
DistTensor<double> t_fj_lvl1_part0_1_lvl2_part1_0__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//t_fj_lvl1_part0_1_lvl2_part1_1[D01,D23]
DistTensor<double> t_fj_lvl1_part0_1_lvl2_part1_1__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//t_fj_lvl1_part0_1_lvl2_part1_1[*,*]
DistTensor<double> t_fj_lvl1_part0_1_lvl2_part1_1_perm10__S__S( dist__S__S, g );
t_fj_lvl1_part0_1_lvl2_part1_1_perm10__S__S.SetLocalPermutation( perm_1_0 );
	//t_fj_lvl1_part0_1_lvl2_part1_2[D01,D23]
DistTensor<double> t_fj_lvl1_part0_1_lvl2_part1_2__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//t_fj_lvl1_part0_2[D01,D23]
DistTensor<double> t_fj_lvl1_part0_2__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//t_fj_lvl2_part0B[D01,D23]
DistTensor<double> t_fj_lvl2_part0B__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//t_fj_lvl2_part0T[D01,D23]
DistTensor<double> t_fj_lvl2_part0T__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//t_fj_lvl2_part0_0[D01,D23]
DistTensor<double> t_fj_lvl2_part0_0__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//t_fj_lvl2_part0_1[D01,D23]
DistTensor<double> t_fj_lvl2_part0_1__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//t_fj_lvl2_part0_1[*,D23]
DistTensor<double> t_fj_lvl2_part0_1__S__D_2_3( dist__S__D_2_3, g );
	//t_fj_lvl2_part0_2[D01,D23]
DistTensor<double> t_fj_lvl2_part0_2__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//G_temp1[D0,D1,D2,D3]
DistTensor<double> G_temp1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//G_temp1_lvl0_part2B[D0,D1,D2,D3]
DistTensor<double> G_temp1_lvl0_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//G_temp1_lvl0_part2T[D0,D1,D2,D3]
DistTensor<double> G_temp1_lvl0_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//G_temp1_lvl0_part3B[D0,D1,D2,D3]
DistTensor<double> G_temp1_lvl0_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//G_temp1_lvl0_part3T[D0,D1,D2,D3]
DistTensor<double> G_temp1_lvl0_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//G_temp1_lvl1_part2B[D0,D1,D2,D3]
DistTensor<double> G_temp1_lvl1_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//G_temp1_lvl1_part2T[D0,D1,D2,D3]
DistTensor<double> G_temp1_lvl1_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//G_temp1_lvl1_part2_0[D0,D1,D2,D3]
DistTensor<double> G_temp1_lvl1_part2_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//G_temp1_lvl1_part2_1[D0,D1,D2,D3]
DistTensor<double> G_temp1_lvl1_part2_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//G_temp1_lvl1_part2_1_lvl1_part3B[D0,D1,D2,D3]
DistTensor<double> G_temp1_lvl1_part2_1_lvl1_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//G_temp1_lvl1_part2_1_lvl1_part3T[D0,D1,D2,D3]
DistTensor<double> G_temp1_lvl1_part2_1_lvl1_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//G_temp1_lvl1_part2_1_lvl2_part3B[D0,D1,D2,D3]
DistTensor<double> G_temp1_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//G_temp1_lvl1_part2_1_lvl2_part3T[D0,D1,D2,D3]
DistTensor<double> G_temp1_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//G_temp1_lvl1_part2_1_lvl2_part3_0[D0,D1,D2,D3]
DistTensor<double> G_temp1_lvl1_part2_1_lvl2_part3_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//G_temp1_lvl1_part2_1_lvl2_part3_1[D0,D1,D2,D3]
DistTensor<double> G_temp1_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//G_temp1_lvl1_part2_1_lvl2_part3_2[D0,D1,D2,D3]
DistTensor<double> G_temp1_lvl1_part2_1_lvl2_part3_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//G_temp1_lvl1_part2_2[D0,D1,D2,D3]
DistTensor<double> G_temp1_lvl1_part2_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//G_temp1_lvl1_part3B[D0,D1,D2,D3]
DistTensor<double> G_temp1_lvl1_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//G_temp1_lvl1_part3T[D0,D1,D2,D3]
DistTensor<double> G_temp1_lvl1_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//G_temp1_lvl1_part3_0[D0,D1,D2,D3]
DistTensor<double> G_temp1_lvl1_part3_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//G_temp1_lvl1_part3_1[D0,D1,D2,D3]
DistTensor<double> G_temp1_lvl1_part3_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//G_temp1_lvl1_part3_1_lvl1_part2B[D0,D1,D2,D3]
DistTensor<double> G_temp1_lvl1_part3_1_lvl1_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//G_temp1_lvl1_part3_1_lvl1_part2T[D0,D1,D2,D3]
DistTensor<double> G_temp1_lvl1_part3_1_lvl1_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//G_temp1_lvl1_part3_1_lvl2_part2B[D0,D1,D2,D3]
DistTensor<double> G_temp1_lvl1_part3_1_lvl2_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//G_temp1_lvl1_part3_1_lvl2_part2T[D0,D1,D2,D3]
DistTensor<double> G_temp1_lvl1_part3_1_lvl2_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//G_temp1_lvl1_part3_1_lvl2_part2_0[D0,D1,D2,D3]
DistTensor<double> G_temp1_lvl1_part3_1_lvl2_part2_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//G_temp1_lvl1_part3_1_lvl2_part2_1[D0,D1,D2,D3]
DistTensor<double> G_temp1_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//G_temp1_lvl1_part3_1_lvl2_part2_1[D0,D1,*,D3]
DistTensor<double> G_temp1_lvl1_part3_1_lvl2_part2_1_perm2013__S__D_0__D_1__D_3( dist__D_0__D_1__S__D_3, g );
G_temp1_lvl1_part3_1_lvl2_part2_1_perm2013__S__D_0__D_1__D_3.SetLocalPermutation( perm_2_0_1_3 );
	//G_temp1_lvl1_part3_1_lvl2_part2_2[D0,D1,D2,D3]
DistTensor<double> G_temp1_lvl1_part3_1_lvl2_part2_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//G_temp1_lvl1_part3_2[D0,D1,D2,D3]
DistTensor<double> G_temp1_lvl1_part3_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//G_temp2[D0,D1,D2,D3]
DistTensor<double> G_temp2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//G_temp2_lvl0_part0B[D0,D1,D2,D3]
DistTensor<double> G_temp2_lvl0_part0B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//G_temp2_lvl0_part0T[D0,D1,D2,D3]
DistTensor<double> G_temp2_lvl0_part0T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//G_temp2_lvl0_part3B[D0,D1,D2,D3]
DistTensor<double> G_temp2_lvl0_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//G_temp2_lvl0_part3T[D0,D1,D2,D3]
DistTensor<double> G_temp2_lvl0_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//G_temp2_lvl1_part0B[D0,D1,D2,D3]
DistTensor<double> G_temp2_lvl1_part0B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//G_temp2_lvl1_part0T[D0,D1,D2,D3]
DistTensor<double> G_temp2_lvl1_part0T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//G_temp2_lvl1_part0_0[D0,D1,D2,D3]
DistTensor<double> G_temp2_lvl1_part0_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//G_temp2_lvl1_part0_1[D0,D1,D2,D3]
DistTensor<double> G_temp2_lvl1_part0_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//G_temp2_lvl1_part0_1_lvl1_part1B[D0,D1,D2,D3]
DistTensor<double> G_temp2_lvl1_part0_1_lvl1_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//G_temp2_lvl1_part0_1_lvl1_part1T[D0,D1,D2,D3]
DistTensor<double> G_temp2_lvl1_part0_1_lvl1_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//G_temp2_lvl1_part0_1_lvl2_part1B[D0,D1,D2,D3]
DistTensor<double> G_temp2_lvl1_part0_1_lvl2_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//G_temp2_lvl1_part0_1_lvl2_part1T[D0,D1,D2,D3]
DistTensor<double> G_temp2_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//G_temp2_lvl1_part0_1_lvl2_part1_0[D0,D1,D2,D3]
DistTensor<double> G_temp2_lvl1_part0_1_lvl2_part1_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//G_temp2_lvl1_part0_1_lvl2_part1_1[D0,D1,D2,D3]
DistTensor<double> G_temp2_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//G_temp2_lvl1_part0_1_lvl2_part1_2[D0,D1,D2,D3]
DistTensor<double> G_temp2_lvl1_part0_1_lvl2_part1_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//G_temp2_lvl1_part0_2[D0,D1,D2,D3]
DistTensor<double> G_temp2_lvl1_part0_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//G_temp2_lvl1_part3B[D0,D1,D2,D3]
DistTensor<double> G_temp2_lvl1_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//G_temp2_lvl1_part3T[D0,D1,D2,D3]
DistTensor<double> G_temp2_lvl1_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//G_temp2_lvl1_part3_0[D0,D1,D2,D3]
DistTensor<double> G_temp2_lvl1_part3_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//G_temp2_lvl1_part3_1[D0,D1,D2,D3]
DistTensor<double> G_temp2_lvl1_part3_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//G_temp2_lvl1_part3_1_lvl1_part1B[D0,D1,D2,D3]
DistTensor<double> G_temp2_lvl1_part3_1_lvl1_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//G_temp2_lvl1_part3_1_lvl1_part1T[D0,D1,D2,D3]
DistTensor<double> G_temp2_lvl1_part3_1_lvl1_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//G_temp2_lvl1_part3_1_lvl2_part1B[D0,D1,D2,D3]
DistTensor<double> G_temp2_lvl1_part3_1_lvl2_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//G_temp2_lvl1_part3_1_lvl2_part1T[D0,D1,D2,D3]
DistTensor<double> G_temp2_lvl1_part3_1_lvl2_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//G_temp2_lvl1_part3_1_lvl2_part1_0[D0,D1,D2,D3]
DistTensor<double> G_temp2_lvl1_part3_1_lvl2_part1_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//G_temp2_lvl1_part3_1_lvl2_part1_1[D0,D1,D23,*]
DistTensor<double> G_temp2_lvl1_part3_1_lvl2_part1_1__D_0__D_1__D_2_3__S( dist__D_0__D_1__D_2_3__S, g );
	//G_temp2_lvl1_part3_1_lvl2_part1_1[D0,D1,D2,D3]
DistTensor<double> G_temp2_lvl1_part3_1_lvl2_part1_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//G_temp2_lvl1_part3_1_lvl2_part1_1[D0,D1,D2,*]
DistTensor<double> G_temp2_lvl1_part3_1_lvl2_part1_1__D_0__D_1__D_2__S( dist__D_0__D_1__D_2__S, g );
	//G_temp2_lvl1_part3_1_lvl2_part1_1[D0,*,D23,*]
DistTensor<double> G_temp2_lvl1_part3_1_lvl2_part1_1__D_0__S__D_2_3__S( dist__D_0__S__D_2_3__S, g );
	//G_temp2_lvl1_part3_1_lvl2_part1_1[D01,*,D23,*]
DistTensor<double> G_temp2_lvl1_part3_1_lvl2_part1_1_perm0213__D_0_1__D_2_3__S__S( dist__D_0_1__S__D_2_3__S, g );
G_temp2_lvl1_part3_1_lvl2_part1_1_perm0213__D_0_1__D_2_3__S__S.SetLocalPermutation( perm_0_2_1_3 );
	//G_temp2_lvl1_part3_1_lvl2_part1_2[D0,D1,D2,D3]
DistTensor<double> G_temp2_lvl1_part3_1_lvl2_part1_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//G_temp2_lvl1_part3_2[D0,D1,D2,D3]
DistTensor<double> G_temp2_lvl1_part3_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//u_mnje[D0,D1,D2,D3]
DistTensor<double> u_mnje__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//u_mnje_lvl0_part0B[D0,D1,D2,D3]
DistTensor<double> u_mnje_lvl0_part0B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//u_mnje_lvl0_part0T[D0,D1,D2,D3]
DistTensor<double> u_mnje_lvl0_part0T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//u_mnje_lvl0_part1B[D0,D1,D2,D3]
DistTensor<double> u_mnje_lvl0_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//u_mnje_lvl0_part1T[D0,D1,D2,D3]
DistTensor<double> u_mnje_lvl0_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//u_mnje_lvl1_part0B[D0,D1,D2,D3]
DistTensor<double> u_mnje_lvl1_part0B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//u_mnje_lvl1_part0T[D0,D1,D2,D3]
DistTensor<double> u_mnje_lvl1_part0T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//u_mnje_lvl1_part0_0[D0,D1,D2,D3]
DistTensor<double> u_mnje_lvl1_part0_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//u_mnje_lvl1_part0_1[D0,D1,D2,D3]
DistTensor<double> u_mnje_lvl1_part0_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//u_mnje_lvl1_part0_1_lvl1_part1B[D0,D1,D2,D3]
DistTensor<double> u_mnje_lvl1_part0_1_lvl1_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//u_mnje_lvl1_part0_1_lvl1_part1T[D0,D1,D2,D3]
DistTensor<double> u_mnje_lvl1_part0_1_lvl1_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//u_mnje_lvl1_part0_1_lvl2_part1B[D0,D1,D2,D3]
DistTensor<double> u_mnje_lvl1_part0_1_lvl2_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//u_mnje_lvl1_part0_1_lvl2_part1T[D0,D1,D2,D3]
DistTensor<double> u_mnje_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//u_mnje_lvl1_part0_1_lvl2_part1_0[D0,D1,D2,D3]
DistTensor<double> u_mnje_lvl1_part0_1_lvl2_part1_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//u_mnje_lvl1_part0_1_lvl2_part1_1[D0,D1,D2,D3]
DistTensor<double> u_mnje_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//u_mnje_lvl1_part0_1_lvl2_part1_2[D0,D1,D2,D3]
DistTensor<double> u_mnje_lvl1_part0_1_lvl2_part1_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//u_mnje_lvl1_part0_2[D0,D1,D2,D3]
DistTensor<double> u_mnje_lvl1_part0_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//u_mnje_lvl1_part1B[D0,D1,D2,D3]
DistTensor<double> u_mnje_lvl1_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//u_mnje_lvl1_part1T[D0,D1,D2,D3]
DistTensor<double> u_mnje_lvl1_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//u_mnje_lvl1_part1_0[D0,D1,D2,D3]
DistTensor<double> u_mnje_lvl1_part1_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//u_mnje_lvl1_part1_1[D0,D1,D2,D3]
DistTensor<double> u_mnje_lvl1_part1_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//u_mnje_lvl1_part1_1_lvl1_part0B[D0,D1,D2,D3]
DistTensor<double> u_mnje_lvl1_part1_1_lvl1_part0B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//u_mnje_lvl1_part1_1_lvl1_part0T[D0,D1,D2,D3]
DistTensor<double> u_mnje_lvl1_part1_1_lvl1_part0T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//u_mnje_lvl1_part1_1_lvl2_part0B[D0,D1,D2,D3]
DistTensor<double> u_mnje_lvl1_part1_1_lvl2_part0B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//u_mnje_lvl1_part1_1_lvl2_part0T[D0,D1,D2,D3]
DistTensor<double> u_mnje_lvl1_part1_1_lvl2_part0T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//u_mnje_lvl1_part1_1_lvl2_part0_0[D0,D1,D2,D3]
DistTensor<double> u_mnje_lvl1_part1_1_lvl2_part0_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//u_mnje_lvl1_part1_1_lvl2_part0_1[D0,D1,D2,D3]
DistTensor<double> u_mnje_lvl1_part1_1_lvl2_part0_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//u_mnje_lvl1_part1_1_lvl2_part0_1[D1,D0,D2,D3]
DistTensor<double> u_mnje_lvl1_part1_1_lvl2_part0_1__D_1__D_0__D_2__D_3( dist__D_1__D_0__D_2__D_3, g );
	//u_mnje_lvl1_part1_1_lvl2_part0_2[D0,D1,D2,D3]
DistTensor<double> u_mnje_lvl1_part1_1_lvl2_part0_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//u_mnje_lvl1_part1_2[D0,D1,D2,D3]
DistTensor<double> u_mnje_lvl1_part1_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//v_femn[D0,D1,D2,D3]
DistTensor<double> v_femn__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//v_femn_lvl0_part2B[D0,D1,D2,D3]
DistTensor<double> v_femn_lvl0_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//v_femn_lvl0_part2T[D0,D1,D2,D3]
DistTensor<double> v_femn_lvl0_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//v_femn_lvl0_part3B[D0,D1,D2,D3]
DistTensor<double> v_femn_lvl0_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//v_femn_lvl0_part3T[D0,D1,D2,D3]
DistTensor<double> v_femn_lvl0_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//v_femn_lvl1_part2B[D0,D1,D2,D3]
DistTensor<double> v_femn_lvl1_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//v_femn_lvl1_part2T[D0,D1,D2,D3]
DistTensor<double> v_femn_lvl1_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//v_femn_lvl1_part2_0[D0,D1,D2,D3]
DistTensor<double> v_femn_lvl1_part2_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//v_femn_lvl1_part2_1[D0,D1,D2,D3]
DistTensor<double> v_femn_lvl1_part2_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//v_femn_lvl1_part2_1_lvl1_part3B[D0,D1,D2,D3]
DistTensor<double> v_femn_lvl1_part2_1_lvl1_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//v_femn_lvl1_part2_1_lvl1_part3T[D0,D1,D2,D3]
DistTensor<double> v_femn_lvl1_part2_1_lvl1_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//v_femn_lvl1_part2_1_lvl2_part3B[D0,D1,D2,D3]
DistTensor<double> v_femn_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//v_femn_lvl1_part2_1_lvl2_part3T[D0,D1,D2,D3]
DistTensor<double> v_femn_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//v_femn_lvl1_part2_1_lvl2_part3_0[D0,D1,D2,D3]
DistTensor<double> v_femn_lvl1_part2_1_lvl2_part3_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//v_femn_lvl1_part2_1_lvl2_part3_1[D0,D1,D2,D3]
DistTensor<double> v_femn_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//v_femn_lvl1_part2_1_lvl2_part3_2[D0,D1,D2,D3]
DistTensor<double> v_femn_lvl1_part2_1_lvl2_part3_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//v_femn_lvl1_part2_2[D0,D1,D2,D3]
DistTensor<double> v_femn_lvl1_part2_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//v_femn_lvl1_part3B[D0,D1,D2,D3]
DistTensor<double> v_femn_lvl1_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//v_femn_lvl1_part3T[D0,D1,D2,D3]
DistTensor<double> v_femn_lvl1_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//v_femn_lvl1_part3_0[D0,D1,D2,D3]
DistTensor<double> v_femn_lvl1_part3_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//v_femn_lvl1_part3_1[D0,D1,D2,D3]
DistTensor<double> v_femn_lvl1_part3_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//v_femn_lvl1_part3_1_lvl1_part2B[D0,D1,D2,D3]
DistTensor<double> v_femn_lvl1_part3_1_lvl1_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//v_femn_lvl1_part3_1_lvl1_part2T[D0,D1,D2,D3]
DistTensor<double> v_femn_lvl1_part3_1_lvl1_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//v_femn_lvl1_part3_1_lvl2_part2B[D0,D1,D2,D3]
DistTensor<double> v_femn_lvl1_part3_1_lvl2_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//v_femn_lvl1_part3_1_lvl2_part2T[D0,D1,D2,D3]
DistTensor<double> v_femn_lvl1_part3_1_lvl2_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//v_femn_lvl1_part3_1_lvl2_part2_0[D0,D1,D2,D3]
DistTensor<double> v_femn_lvl1_part3_1_lvl2_part2_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//v_femn_lvl1_part3_1_lvl2_part2_1[D0,D1,D2,D3]
DistTensor<double> v_femn_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//v_femn_lvl1_part3_1_lvl2_part2_1[D0,D1,D3,D2]
DistTensor<double> v_femn_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_3__D_2( dist__D_0__D_1__D_3__D_2, g );
	//v_femn_lvl1_part3_1_lvl2_part2_2[D0,D1,D2,D3]
DistTensor<double> v_femn_lvl1_part3_1_lvl2_part2_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//v_femn_lvl1_part3_2[D0,D1,D2,D3]
DistTensor<double> v_femn_lvl1_part3_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
// H_me has 2 dims
//	Starting distribution: [D01,D23] or _D_0_1__D_2_3
ObjShape H_me__D_0_1__D_2_3_tmpShape_G( 2 );
H_me__D_0_1__D_2_3_tmpShape_G[ 0 ] = n_o;
H_me__D_0_1__D_2_3_tmpShape_G[ 1 ] = n_v;
H_me__D_0_1__D_2_3.ResizeTo( H_me__D_0_1__D_2_3_tmpShape_G );
MakeUniform( H_me__D_0_1__D_2_3 );
// t_fj has 2 dims
ObjShape t_fj__D_0_1__D_2_3_tmpShape_G( 2 );
t_fj__D_0_1__D_2_3_tmpShape_G[ 0 ] = n_v;
t_fj__D_0_1__D_2_3_tmpShape_G[ 1 ] = n_o;
t_fj__D_0_1__D_2_3.ResizeTo( t_fj__D_0_1__D_2_3_tmpShape_G );
MakeUniform( t_fj__D_0_1__D_2_3 );
// u_mnje has 4 dims
//	Starting distribution: [D0,D1,D2,D3] or _D_0__D_1__D_2__D_3
ObjShape u_mnje__D_0__D_1__D_2__D_3_tmpShape_G( 4 );
u_mnje__D_0__D_1__D_2__D_3_tmpShape_G[ 0 ] = n_o;
u_mnje__D_0__D_1__D_2__D_3_tmpShape_G[ 1 ] = n_o;
u_mnje__D_0__D_1__D_2__D_3_tmpShape_G[ 2 ] = n_o;
u_mnje__D_0__D_1__D_2__D_3_tmpShape_G[ 3 ] = n_v;
u_mnje__D_0__D_1__D_2__D_3.ResizeTo( u_mnje__D_0__D_1__D_2__D_3_tmpShape_G );
MakeUniform( u_mnje__D_0__D_1__D_2__D_3 );
// T_bfnj has 4 dims
ObjShape T_bfnj__D_0__D_1__D_2__D_3_tmpShape_G( 4 );
T_bfnj__D_0__D_1__D_2__D_3_tmpShape_G[ 0 ] = n_v;
T_bfnj__D_0__D_1__D_2__D_3_tmpShape_G[ 1 ] = n_v;
T_bfnj__D_0__D_1__D_2__D_3_tmpShape_G[ 2 ] = n_o;
T_bfnj__D_0__D_1__D_2__D_3_tmpShape_G[ 3 ] = n_o;
T_bfnj__D_0__D_1__D_2__D_3.ResizeTo( T_bfnj__D_0__D_1__D_2__D_3_tmpShape_G );
MakeUniform( T_bfnj__D_0__D_1__D_2__D_3 );
// G_mi has 2 dims
ObjShape G_mi__D_0_1__D_2_3_tmpShape_G( 2 );
G_mi__D_0_1__D_2_3_tmpShape_G[ 0 ] = n_o;
G_mi__D_0_1__D_2_3_tmpShape_G[ 1 ] = n_o;
G_mi__D_0_1__D_2_3.ResizeTo( G_mi__D_0_1__D_2_3_tmpShape_G );
MakeUniform( G_mi__D_0_1__D_2_3 );
// v_femn has 4 dims
ObjShape v_femn__D_0__D_1__D_2__D_3_tmpShape_G( 4 );
v_femn__D_0__D_1__D_2__D_3_tmpShape_G[ 0 ] = n_v;
v_femn__D_0__D_1__D_2__D_3_tmpShape_G[ 1 ] = n_v;
v_femn__D_0__D_1__D_2__D_3_tmpShape_G[ 2 ] = n_o;
v_femn__D_0__D_1__D_2__D_3_tmpShape_G[ 3 ] = n_o;
v_femn__D_0__D_1__D_2__D_3.ResizeTo( v_femn__D_0__D_1__D_2__D_3_tmpShape_G );
MakeUniform( v_femn__D_0__D_1__D_2__D_3 );
//END_DECL

//******************************
//* Load tensors
//******************************
////////////////////////////////
//Performance testing
////////////////////////////////
std::stringstream fullName;
#ifdef CORRECTNESS
DistTensor<T> check_G(dist__D_0_1__D_2_3, g);
check_G.ResizeTo(G_mi__D_0_1__D_2_3.Shape());
Read(u_mnje__D_0__D_1__D_2__D_3, "ccsd_terms/term_u_small", BINARY_FLAT, false);
Read(v_femn__D_0__D_1__D_2__D_3, "ccsd_terms/term_v_small", BINARY_FLAT, false);
fullName.str("");
fullName.clear();
fullName << "ccsd_terms/term_H_iter" << testIter;
Read(H_me__D_0_1__D_2_3, fullName.str(), BINARY_FLAT, false);
fullName.str("");
fullName.clear();
fullName << "ccsd_terms/term_t_small_iter" << testIter;
Read(t_fj__D_0_1__D_2_3, fullName.str(), BINARY_FLAT, false);
fullName.str("");
fullName.clear();
fullName << "ccsd_terms/term_T_iter" << testIter;
Read(T_bfnj__D_0__D_1__D_2__D_3, fullName.str(), BINARY_FLAT, false);
fullName.str("");
fullName.clear();
fullName << "ccsd_terms/term_G_iter" << testIter;
Read(check_G, fullName.str(), BINARY_FLAT, false);
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
overwrite_tmpShape_G = v_femn__D_0__D_1__D_2__D_3.Shape();
G_temp1__D_0__D_1__D_2__D_3.ResizeTo( overwrite_tmpShape_G );
overwrite_tmpShape_G = u_mnje__D_0__D_1__D_2__D_3.Shape();
G_temp2__D_0__D_1__D_2__D_3.ResizeTo( overwrite_tmpShape_G );
//**** (out of 1)
//**** Is real	0 shadows
	//Outputs:
	//  G_mi__D_0_1__D_2_3

	Scal( 0.0, G_mi__D_0_1__D_2_3 );
	//**** (out of 1)
	//**** Is real	0 shadows
		//Outputs:
		//  G_temp1__D_0__D_1__D_2__D_3
	PartitionDown(v_femn__D_0__D_1__D_2__D_3, v_femn_lvl1_part2T__D_0__D_1__D_2__D_3, v_femn_lvl1_part2B__D_0__D_1__D_2__D_3, 2, 0);
	PartitionDown(v_femn__D_0__D_1__D_2__D_3, v_femn_lvl1_part3T__D_0__D_1__D_2__D_3, v_femn_lvl1_part3B__D_0__D_1__D_2__D_3, 3, 0);
	PartitionDown(G_temp1__D_0__D_1__D_2__D_3, G_temp1_lvl1_part2T__D_0__D_1__D_2__D_3, G_temp1_lvl1_part2B__D_0__D_1__D_2__D_3, 2, 0);
	while(G_temp1_lvl1_part2T__D_0__D_1__D_2__D_3.Dimension(2) < G_temp1__D_0__D_1__D_2__D_3.Dimension(2))
	{
		RepartitionDown
		( v_femn_lvl1_part2T__D_0__D_1__D_2__D_3,  v_femn_lvl1_part2_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       v_femn_lvl1_part2_1__D_0__D_1__D_2__D_3,
		  v_femn_lvl1_part2B__D_0__D_1__D_2__D_3, v_femn_lvl1_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );
		RepartitionDown
		( v_femn_lvl1_part3T__D_0__D_1__D_2__D_3,  v_femn_lvl1_part3_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       v_femn_lvl1_part3_1__D_0__D_1__D_2__D_3,
		  v_femn_lvl1_part3B__D_0__D_1__D_2__D_3, v_femn_lvl1_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );
		RepartitionDown
		( G_temp1_lvl1_part2T__D_0__D_1__D_2__D_3,  G_temp1_lvl1_part2_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       G_temp1_lvl1_part2_1__D_0__D_1__D_2__D_3,
		  G_temp1_lvl1_part2B__D_0__D_1__D_2__D_3, G_temp1_lvl1_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );

		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  G_temp1_lvl1_part2_1__D_0__D_1__D_2__D_3
		PartitionDown(v_femn_lvl1_part2_1__D_0__D_1__D_2__D_3, v_femn_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3, v_femn_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, 3, 0);
		PartitionDown(v_femn_lvl1_part3_1__D_0__D_1__D_2__D_3, v_femn_lvl1_part3_1_lvl2_part2T__D_0__D_1__D_2__D_3, v_femn_lvl1_part3_1_lvl2_part2B__D_0__D_1__D_2__D_3, 2, 0);
		PartitionDown(G_temp1_lvl1_part2_1__D_0__D_1__D_2__D_3, G_temp1_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3, G_temp1_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, 3, 0);
		while(G_temp1_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3.Dimension(3) < G_temp1_lvl1_part2_1__D_0__D_1__D_2__D_3.Dimension(3))
		{
			RepartitionDown
			( v_femn_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3,  v_femn_lvl1_part2_1_lvl2_part3_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       v_femn_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3,
			  v_femn_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, v_femn_lvl1_part2_1_lvl2_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );
			RepartitionDown
			( v_femn_lvl1_part3_1_lvl2_part2T__D_0__D_1__D_2__D_3,  v_femn_lvl1_part3_1_lvl2_part2_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       v_femn_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_2__D_3,
			  v_femn_lvl1_part3_1_lvl2_part2B__D_0__D_1__D_2__D_3, v_femn_lvl1_part3_1_lvl2_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );
			RepartitionDown
			( G_temp1_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3,  G_temp1_lvl1_part2_1_lvl2_part3_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       G_temp1_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3,
			  G_temp1_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, G_temp1_lvl1_part2_1_lvl2_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );

			   // v_femn_lvl1_part3_1_lvl2_part2_1[D0,D1,D3,D2] <- v_femn_lvl1_part3_1_lvl2_part2_1[D0,D1,D2,D3]
			v_femn_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_3__D_2.AlignModesWith( modes_0_1_2_3, v_femn_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3, modes_0_1_3_2 );
			v_femn_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_3__D_2.AllToAllRedistFrom( v_femn_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_2__D_3, modes_2_3 );
if(commRank == 0){
flops += 3*prod(v_femn_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3.Shape());
}
			YAxpPx( 2.0, v_femn_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3, -1.0, v_femn_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_3__D_2, perm_0_1_3_2, G_temp1_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3 );
			v_femn_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_3__D_2.EmptyData();

			SlidePartitionDown
			( v_femn_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3,  v_femn_lvl1_part2_1_lvl2_part3_0__D_0__D_1__D_2__D_3,
			       v_femn_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  v_femn_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, v_femn_lvl1_part2_1_lvl2_part3_2__D_0__D_1__D_2__D_3, 3 );
			SlidePartitionDown
			( v_femn_lvl1_part3_1_lvl2_part2T__D_0__D_1__D_2__D_3,  v_femn_lvl1_part3_1_lvl2_part2_0__D_0__D_1__D_2__D_3,
			       v_femn_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  v_femn_lvl1_part3_1_lvl2_part2B__D_0__D_1__D_2__D_3, v_femn_lvl1_part3_1_lvl2_part2_2__D_0__D_1__D_2__D_3, 2 );
			SlidePartitionDown
			( G_temp1_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3,  G_temp1_lvl1_part2_1_lvl2_part3_0__D_0__D_1__D_2__D_3,
			       G_temp1_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  G_temp1_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, G_temp1_lvl1_part2_1_lvl2_part3_2__D_0__D_1__D_2__D_3, 3 );

		}
		//****

		SlidePartitionDown
		( v_femn_lvl1_part2T__D_0__D_1__D_2__D_3,  v_femn_lvl1_part2_0__D_0__D_1__D_2__D_3,
		       v_femn_lvl1_part2_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  v_femn_lvl1_part2B__D_0__D_1__D_2__D_3, v_femn_lvl1_part2_2__D_0__D_1__D_2__D_3, 2 );
		SlidePartitionDown
		( v_femn_lvl1_part3T__D_0__D_1__D_2__D_3,  v_femn_lvl1_part3_0__D_0__D_1__D_2__D_3,
		       v_femn_lvl1_part3_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  v_femn_lvl1_part3B__D_0__D_1__D_2__D_3, v_femn_lvl1_part3_2__D_0__D_1__D_2__D_3, 3 );
		SlidePartitionDown
		( G_temp1_lvl1_part2T__D_0__D_1__D_2__D_3,  G_temp1_lvl1_part2_0__D_0__D_1__D_2__D_3,
		       G_temp1_lvl1_part2_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  G_temp1_lvl1_part2B__D_0__D_1__D_2__D_3, G_temp1_lvl1_part2_2__D_0__D_1__D_2__D_3, 2 );

	}
	//****
	//**** (out of 1)
	//**** Is real	0 shadows
		//Outputs:
		//  G_mi__D_0_1__D_2_3
	PartitionDown(G_temp1__D_0__D_1__D_2__D_3, G_temp1_lvl1_part3T__D_0__D_1__D_2__D_3, G_temp1_lvl1_part3B__D_0__D_1__D_2__D_3, 3, 0);
	PartitionDown(T_bfnj__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part3T__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part3B__D_0__D_1__D_2__D_3, 3, 0);
	while(G_temp1_lvl1_part3T__D_0__D_1__D_2__D_3.Dimension(3) < G_temp1__D_0__D_1__D_2__D_3.Dimension(3))
	{
		RepartitionDown
		( G_temp1_lvl1_part3T__D_0__D_1__D_2__D_3,  G_temp1_lvl1_part3_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       G_temp1_lvl1_part3_1__D_0__D_1__D_2__D_3,
		  G_temp1_lvl1_part3B__D_0__D_1__D_2__D_3, G_temp1_lvl1_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );
		RepartitionDown
		( T_bfnj_lvl1_part3T__D_0__D_1__D_2__D_3,  T_bfnj_lvl1_part3_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       T_bfnj_lvl1_part3_1__D_0__D_1__D_2__D_3,
		  T_bfnj_lvl1_part3B__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );

		Permute( T_bfnj_lvl1_part3_1__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part3_1_perm0132__D_0__D_1__D_3__D_2 );
		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  G_mi__D_0_1__D_2_3
		PartitionDown(G_temp1_lvl1_part3_1__D_0__D_1__D_2__D_3, G_temp1_lvl1_part3_1_lvl2_part2T__D_0__D_1__D_2__D_3, G_temp1_lvl1_part3_1_lvl2_part2B__D_0__D_1__D_2__D_3, 2, 0);
		PartitionDown(G_mi__D_0_1__D_2_3, G_mi_lvl2_part0T__D_0_1__D_2_3, G_mi_lvl2_part0B__D_0_1__D_2_3, 0, 0);
		while(G_mi_lvl2_part0T__D_0_1__D_2_3.Dimension(0) < G_mi__D_0_1__D_2_3.Dimension(0))
		{
			RepartitionDown
			( G_temp1_lvl1_part3_1_lvl2_part2T__D_0__D_1__D_2__D_3,  G_temp1_lvl1_part3_1_lvl2_part2_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       G_temp1_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_2__D_3,
			  G_temp1_lvl1_part3_1_lvl2_part2B__D_0__D_1__D_2__D_3, G_temp1_lvl1_part3_1_lvl2_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );
			RepartitionDown
			( G_mi_lvl2_part0T__D_0_1__D_2_3,  G_mi_lvl2_part0_0__D_0_1__D_2_3,
			  /**/ /**/
			       G_mi_lvl2_part0_1__D_0_1__D_2_3,
			  G_mi_lvl2_part0B__D_0_1__D_2_3, G_mi_lvl2_part0_2__D_0_1__D_2_3, 0, blkSize );

			   // G_temp1_lvl1_part3_1_lvl2_part2_1[D0,D1,*,D3] <- G_temp1_lvl1_part3_1_lvl2_part2_1[D0,D1,D2,D3]
			G_temp1_lvl1_part3_1_lvl2_part2_1_perm2013__S__D_0__D_1__D_3.AlignModesWith( modes_0_1_3, T_bfnj_lvl1_part3_1__D_0__D_1__D_2__D_3, modes_0_1_3 );
			G_temp1_lvl1_part3_1_lvl2_part2_1_perm2013__S__D_0__D_1__D_3.AllGatherRedistFrom( G_temp1_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_2__D_3, modes_2 );
			G_mi_lvl2_part0_1__S__D_2__D_0__D_1__D_3.AlignModesWith( modes_1, T_bfnj_lvl1_part3_1__D_0__D_1__D_2__D_3, modes_2 );
			overwrite_tmpShape_G = G_mi_lvl2_part0_1__D_0_1__D_2_3.Shape();
			overwrite_tmpShape_G.push_back( g.Shape()[0] );
			overwrite_tmpShape_G.push_back( g.Shape()[1] );
			overwrite_tmpShape_G.push_back( g.Shape()[3] );
			G_mi_lvl2_part0_1__S__D_2__D_0__D_1__D_3.ResizeTo( overwrite_tmpShape_G );
			   // 1.0 * G_temp1_lvl1_part3_1_lvl2_part2_1[D0,D1,*,D3]_mefn * T_bfnj_lvl1_part3_1[D0,D1,D2,D3]_efni + 0.0 * G_mi_lvl2_part0_1[*,D2,D0,D1,D3]_miefn
if(commRank == 0){
flops += 2*G_temp1_lvl1_part3_1_lvl2_part2_1_perm2013__S__D_0__D_1__D_3.Dimension(1)*G_temp1_lvl1_part3_1_lvl2_part2_1_perm2013__S__D_0__D_1__D_3.Dimension(0)*G_temp1_lvl1_part3_1_lvl2_part2_1_perm2013__S__D_0__D_1__D_3.Dimension(3)*G_temp1_lvl1_part3_1_lvl2_part2_1_perm2013__S__D_0__D_1__D_3.Dimension(2)*T_bfnj_lvl1_part3_1_perm0132__D_0__D_1__D_3__D_2.Dimension(2);
}
			LocalContract(1.0, G_temp1_lvl1_part3_1_lvl2_part2_1_perm2013__S__D_0__D_1__D_3.LockedTensor(), indices_mefn, false,
				T_bfnj_lvl1_part3_1_perm0132__D_0__D_1__D_3__D_2.LockedTensor(), indices_efni, false,
				0.0, G_mi_lvl2_part0_1__S__D_2__D_0__D_1__D_3.Tensor(), indices_miefn, false);
			G_temp1_lvl1_part3_1_lvl2_part2_1_perm2013__S__D_0__D_1__D_3.EmptyData();
			   // G_mi_lvl2_part0_1[D01,D23] <- G_mi_lvl2_part0_1[*,D2,D0,D1,D3] (with SumScatter on (D0)(D1)(D3))
			G_mi_lvl2_part0_1__D_0_1__D_2_3.ReduceScatterUpdateRedistFrom( G_mi_lvl2_part0_1__S__D_2__D_0__D_1__D_3, 1.0, modes_4_3_2 );
			G_mi_lvl2_part0_1__S__D_2__D_0__D_1__D_3.EmptyData();

			SlidePartitionDown
			( G_temp1_lvl1_part3_1_lvl2_part2T__D_0__D_1__D_2__D_3,  G_temp1_lvl1_part3_1_lvl2_part2_0__D_0__D_1__D_2__D_3,
			       G_temp1_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  G_temp1_lvl1_part3_1_lvl2_part2B__D_0__D_1__D_2__D_3, G_temp1_lvl1_part3_1_lvl2_part2_2__D_0__D_1__D_2__D_3, 2 );
			SlidePartitionDown
			( G_mi_lvl2_part0T__D_0_1__D_2_3,  G_mi_lvl2_part0_0__D_0_1__D_2_3,
			       G_mi_lvl2_part0_1__D_0_1__D_2_3,
			  /**/ /**/
			  G_mi_lvl2_part0B__D_0_1__D_2_3, G_mi_lvl2_part0_2__D_0_1__D_2_3, 0 );

		}
		//****
		T_bfnj_lvl1_part3_1_perm0132__D_0__D_1__D_3__D_2.EmptyData();

		SlidePartitionDown
		( G_temp1_lvl1_part3T__D_0__D_1__D_2__D_3,  G_temp1_lvl1_part3_0__D_0__D_1__D_2__D_3,
		       G_temp1_lvl1_part3_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  G_temp1_lvl1_part3B__D_0__D_1__D_2__D_3, G_temp1_lvl1_part3_2__D_0__D_1__D_2__D_3, 3 );
		SlidePartitionDown
		( T_bfnj_lvl1_part3T__D_0__D_1__D_2__D_3,  T_bfnj_lvl1_part3_0__D_0__D_1__D_2__D_3,
		       T_bfnj_lvl1_part3_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  T_bfnj_lvl1_part3B__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part3_2__D_0__D_1__D_2__D_3, 3 );

	}
	//****
	G_temp1__D_0__D_1__D_2__D_3.EmptyData();
	//**** (out of 1)
	//**** Is real	0 shadows
		//Outputs:
		//  G_temp2__D_0__D_1__D_2__D_3
	PartitionDown(u_mnje__D_0__D_1__D_2__D_3, u_mnje_lvl1_part0T__D_0__D_1__D_2__D_3, u_mnje_lvl1_part0B__D_0__D_1__D_2__D_3, 0, 0);
	PartitionDown(u_mnje__D_0__D_1__D_2__D_3, u_mnje_lvl1_part1T__D_0__D_1__D_2__D_3, u_mnje_lvl1_part1B__D_0__D_1__D_2__D_3, 1, 0);
	PartitionDown(G_temp2__D_0__D_1__D_2__D_3, G_temp2_lvl1_part0T__D_0__D_1__D_2__D_3, G_temp2_lvl1_part0B__D_0__D_1__D_2__D_3, 0, 0);
	while(G_temp2_lvl1_part0T__D_0__D_1__D_2__D_3.Dimension(0) < G_temp2__D_0__D_1__D_2__D_3.Dimension(0))
	{
		RepartitionDown
		( u_mnje_lvl1_part0T__D_0__D_1__D_2__D_3,  u_mnje_lvl1_part0_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       u_mnje_lvl1_part0_1__D_0__D_1__D_2__D_3,
		  u_mnje_lvl1_part0B__D_0__D_1__D_2__D_3, u_mnje_lvl1_part0_2__D_0__D_1__D_2__D_3, 0, blkSize );
		RepartitionDown
		( u_mnje_lvl1_part1T__D_0__D_1__D_2__D_3,  u_mnje_lvl1_part1_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       u_mnje_lvl1_part1_1__D_0__D_1__D_2__D_3,
		  u_mnje_lvl1_part1B__D_0__D_1__D_2__D_3, u_mnje_lvl1_part1_2__D_0__D_1__D_2__D_3, 1, blkSize );
		RepartitionDown
		( G_temp2_lvl1_part0T__D_0__D_1__D_2__D_3,  G_temp2_lvl1_part0_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       G_temp2_lvl1_part0_1__D_0__D_1__D_2__D_3,
		  G_temp2_lvl1_part0B__D_0__D_1__D_2__D_3, G_temp2_lvl1_part0_2__D_0__D_1__D_2__D_3, 0, blkSize );

		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  G_temp2_lvl1_part0_1__D_0__D_1__D_2__D_3
		PartitionDown(u_mnje_lvl1_part0_1__D_0__D_1__D_2__D_3, u_mnje_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3, u_mnje_lvl1_part0_1_lvl2_part1B__D_0__D_1__D_2__D_3, 1, 0);
		PartitionDown(u_mnje_lvl1_part1_1__D_0__D_1__D_2__D_3, u_mnje_lvl1_part1_1_lvl2_part0T__D_0__D_1__D_2__D_3, u_mnje_lvl1_part1_1_lvl2_part0B__D_0__D_1__D_2__D_3, 0, 0);
		PartitionDown(G_temp2_lvl1_part0_1__D_0__D_1__D_2__D_3, G_temp2_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3, G_temp2_lvl1_part0_1_lvl2_part1B__D_0__D_1__D_2__D_3, 1, 0);
		while(G_temp2_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3.Dimension(1) < G_temp2_lvl1_part0_1__D_0__D_1__D_2__D_3.Dimension(1))
		{
			RepartitionDown
			( u_mnje_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3,  u_mnje_lvl1_part0_1_lvl2_part1_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       u_mnje_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3,
			  u_mnje_lvl1_part0_1_lvl2_part1B__D_0__D_1__D_2__D_3, u_mnje_lvl1_part0_1_lvl2_part1_2__D_0__D_1__D_2__D_3, 1, blkSize );
			RepartitionDown
			( u_mnje_lvl1_part1_1_lvl2_part0T__D_0__D_1__D_2__D_3,  u_mnje_lvl1_part1_1_lvl2_part0_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       u_mnje_lvl1_part1_1_lvl2_part0_1__D_0__D_1__D_2__D_3,
			  u_mnje_lvl1_part1_1_lvl2_part0B__D_0__D_1__D_2__D_3, u_mnje_lvl1_part1_1_lvl2_part0_2__D_0__D_1__D_2__D_3, 0, blkSize );
			RepartitionDown
			( G_temp2_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3,  G_temp2_lvl1_part0_1_lvl2_part1_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       G_temp2_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3,
			  G_temp2_lvl1_part0_1_lvl2_part1B__D_0__D_1__D_2__D_3, G_temp2_lvl1_part0_1_lvl2_part1_2__D_0__D_1__D_2__D_3, 1, blkSize );

			   // u_mnje_lvl1_part1_1_lvl2_part0_1[D1,D0,D2,D3] <- u_mnje_lvl1_part1_1_lvl2_part0_1[D0,D1,D2,D3]
			u_mnje_lvl1_part1_1_lvl2_part0_1__D_1__D_0__D_2__D_3.AlignModesWith( modes_0_1_2_3, u_mnje_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3, modes_1_0_2_3 );
			u_mnje_lvl1_part1_1_lvl2_part0_1__D_1__D_0__D_2__D_3.AllToAllRedistFrom( u_mnje_lvl1_part1_1_lvl2_part0_1__D_0__D_1__D_2__D_3, modes_0_1 );
if(commRank == 0){
flops += 3*prod(u_mnje_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3.Shape());
}
			YAxpPx( 2.0, u_mnje_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3, -1.0, u_mnje_lvl1_part1_1_lvl2_part0_1__D_1__D_0__D_2__D_3, perm_1_0_2_3, G_temp2_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3 );
			u_mnje_lvl1_part1_1_lvl2_part0_1__D_1__D_0__D_2__D_3.EmptyData();

			SlidePartitionDown
			( u_mnje_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3,  u_mnje_lvl1_part0_1_lvl2_part1_0__D_0__D_1__D_2__D_3,
			       u_mnje_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  u_mnje_lvl1_part0_1_lvl2_part1B__D_0__D_1__D_2__D_3, u_mnje_lvl1_part0_1_lvl2_part1_2__D_0__D_1__D_2__D_3, 1 );
			SlidePartitionDown
			( u_mnje_lvl1_part1_1_lvl2_part0T__D_0__D_1__D_2__D_3,  u_mnje_lvl1_part1_1_lvl2_part0_0__D_0__D_1__D_2__D_3,
			       u_mnje_lvl1_part1_1_lvl2_part0_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  u_mnje_lvl1_part1_1_lvl2_part0B__D_0__D_1__D_2__D_3, u_mnje_lvl1_part1_1_lvl2_part0_2__D_0__D_1__D_2__D_3, 0 );
			SlidePartitionDown
			( G_temp2_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3,  G_temp2_lvl1_part0_1_lvl2_part1_0__D_0__D_1__D_2__D_3,
			       G_temp2_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  G_temp2_lvl1_part0_1_lvl2_part1B__D_0__D_1__D_2__D_3, G_temp2_lvl1_part0_1_lvl2_part1_2__D_0__D_1__D_2__D_3, 1 );

		}
		//****

		SlidePartitionDown
		( u_mnje_lvl1_part0T__D_0__D_1__D_2__D_3,  u_mnje_lvl1_part0_0__D_0__D_1__D_2__D_3,
		       u_mnje_lvl1_part0_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  u_mnje_lvl1_part0B__D_0__D_1__D_2__D_3, u_mnje_lvl1_part0_2__D_0__D_1__D_2__D_3, 0 );
		SlidePartitionDown
		( u_mnje_lvl1_part1T__D_0__D_1__D_2__D_3,  u_mnje_lvl1_part1_0__D_0__D_1__D_2__D_3,
		       u_mnje_lvl1_part1_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  u_mnje_lvl1_part1B__D_0__D_1__D_2__D_3, u_mnje_lvl1_part1_2__D_0__D_1__D_2__D_3, 1 );
		SlidePartitionDown
		( G_temp2_lvl1_part0T__D_0__D_1__D_2__D_3,  G_temp2_lvl1_part0_0__D_0__D_1__D_2__D_3,
		       G_temp2_lvl1_part0_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  G_temp2_lvl1_part0B__D_0__D_1__D_2__D_3, G_temp2_lvl1_part0_2__D_0__D_1__D_2__D_3, 0 );

	}
	//****
	//**** (out of 1)
	//**** Is real	0 shadows
		//Outputs:
		//  G_mi__D_0_1__D_2_3
	PartitionDown(G_temp2__D_0__D_1__D_2__D_3, G_temp2_lvl1_part3T__D_0__D_1__D_2__D_3, G_temp2_lvl1_part3B__D_0__D_1__D_2__D_3, 3, 0);
	PartitionDown(t_fj__D_0_1__D_2_3, t_fj_lvl1_part0T__D_0_1__D_2_3, t_fj_lvl1_part0B__D_0_1__D_2_3, 0, 0);
	while(G_temp2_lvl1_part3T__D_0__D_1__D_2__D_3.Dimension(3) < G_temp2__D_0__D_1__D_2__D_3.Dimension(3))
	{
		RepartitionDown
		( G_temp2_lvl1_part3T__D_0__D_1__D_2__D_3,  G_temp2_lvl1_part3_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       G_temp2_lvl1_part3_1__D_0__D_1__D_2__D_3,
		  G_temp2_lvl1_part3B__D_0__D_1__D_2__D_3, G_temp2_lvl1_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );
		RepartitionDown
		( t_fj_lvl1_part0T__D_0_1__D_2_3,  t_fj_lvl1_part0_0__D_0_1__D_2_3,
		  /**/ /**/
		       t_fj_lvl1_part0_1__D_0_1__D_2_3,
		  t_fj_lvl1_part0B__D_0_1__D_2_3, t_fj_lvl1_part0_2__D_0_1__D_2_3, 0, blkSize );

		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  G_mi__D_0_1__D_2_3
		PartitionDown(G_temp2_lvl1_part3_1__D_0__D_1__D_2__D_3, G_temp2_lvl1_part3_1_lvl2_part1T__D_0__D_1__D_2__D_3, G_temp2_lvl1_part3_1_lvl2_part1B__D_0__D_1__D_2__D_3, 1, 0);
		PartitionDown(t_fj_lvl1_part0_1__D_0_1__D_2_3, t_fj_lvl1_part0_1_lvl2_part1T__D_0_1__D_2_3, t_fj_lvl1_part0_1_lvl2_part1B__D_0_1__D_2_3, 1, 0);
		while(G_temp2_lvl1_part3_1_lvl2_part1T__D_0__D_1__D_2__D_3.Dimension(1) < G_temp2_lvl1_part3_1__D_0__D_1__D_2__D_3.Dimension(1))
		{
			RepartitionDown
			( G_temp2_lvl1_part3_1_lvl2_part1T__D_0__D_1__D_2__D_3,  G_temp2_lvl1_part3_1_lvl2_part1_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       G_temp2_lvl1_part3_1_lvl2_part1_1__D_0__D_1__D_2__D_3,
			  G_temp2_lvl1_part3_1_lvl2_part1B__D_0__D_1__D_2__D_3, G_temp2_lvl1_part3_1_lvl2_part1_2__D_0__D_1__D_2__D_3, 1, blkSize );
			RepartitionDown
			( t_fj_lvl1_part0_1_lvl2_part1T__D_0_1__D_2_3,  t_fj_lvl1_part0_1_lvl2_part1_0__D_0_1__D_2_3,
			  /**/ /**/
			       t_fj_lvl1_part0_1_lvl2_part1_1__D_0_1__D_2_3,
			  t_fj_lvl1_part0_1_lvl2_part1B__D_0_1__D_2_3, t_fj_lvl1_part0_1_lvl2_part1_2__D_0_1__D_2_3, 1, blkSize );

			   // t_fj_lvl1_part0_1_lvl2_part1_1[*,*] <- t_fj_lvl1_part0_1_lvl2_part1_1[D01,D23]
			t_fj_lvl1_part0_1_lvl2_part1_1_perm10__S__S.AllGatherRedistFrom( t_fj_lvl1_part0_1_lvl2_part1_1__D_0_1__D_2_3, modes_0_1_2_3 );
			   // G_temp2_lvl1_part3_1_lvl2_part1_1[D0,D1,D2,*] <- G_temp2_lvl1_part3_1_lvl2_part1_1[D0,D1,D2,D3]
			G_temp2_lvl1_part3_1_lvl2_part1_1__D_0__D_1__D_2__S.AlignModesWith( modes_0_2, G_mi__D_0_1__D_2_3, modes_0_1 );
			G_temp2_lvl1_part3_1_lvl2_part1_1__D_0__D_1__D_2__S.AllGatherRedistFrom( G_temp2_lvl1_part3_1_lvl2_part1_1__D_0__D_1__D_2__D_3, modes_3 );
			   // G_temp2_lvl1_part3_1_lvl2_part1_1[D0,D1,D23,*] <- G_temp2_lvl1_part3_1_lvl2_part1_1[D0,D1,D2,*]
			G_temp2_lvl1_part3_1_lvl2_part1_1__D_0__D_1__D_2_3__S.AlignModesWith( modes_0_2, G_mi__D_0_1__D_2_3, modes_0_1 );
			G_temp2_lvl1_part3_1_lvl2_part1_1__D_0__D_1__D_2_3__S.LocalRedistFrom( G_temp2_lvl1_part3_1_lvl2_part1_1__D_0__D_1__D_2__S );
			G_temp2_lvl1_part3_1_lvl2_part1_1__D_0__D_1__D_2__S.EmptyData();
			   // G_temp2_lvl1_part3_1_lvl2_part1_1[D0,*,D23,*] <- G_temp2_lvl1_part3_1_lvl2_part1_1[D0,D1,D23,*]
			G_temp2_lvl1_part3_1_lvl2_part1_1__D_0__S__D_2_3__S.AlignModesWith( modes_0_2, G_mi__D_0_1__D_2_3, modes_0_1 );
			G_temp2_lvl1_part3_1_lvl2_part1_1__D_0__S__D_2_3__S.AllGatherRedistFrom( G_temp2_lvl1_part3_1_lvl2_part1_1__D_0__D_1__D_2_3__S, modes_1 );
			G_temp2_lvl1_part3_1_lvl2_part1_1__D_0__D_1__D_2_3__S.EmptyData();
			   // G_temp2_lvl1_part3_1_lvl2_part1_1[D01,*,D23,*] <- G_temp2_lvl1_part3_1_lvl2_part1_1[D0,*,D23,*]
			G_temp2_lvl1_part3_1_lvl2_part1_1_perm0213__D_0_1__D_2_3__S__S.AlignModesWith( modes_0_2, G_mi__D_0_1__D_2_3, modes_0_1 );
			G_temp2_lvl1_part3_1_lvl2_part1_1_perm0213__D_0_1__D_2_3__S__S.LocalRedistFrom( G_temp2_lvl1_part3_1_lvl2_part1_1__D_0__S__D_2_3__S );
			G_temp2_lvl1_part3_1_lvl2_part1_1__D_0__S__D_2_3__S.EmptyData();
			   // 1.0 * G_temp2_lvl1_part3_1_lvl2_part1_1[D01,*,D23,*]_mine * t_fj_lvl1_part0_1_lvl2_part1_1[*,*]_ne + 1.0 * G_mi[D01,D23]_mi
if(commRank == 0){
flops += 2*G_temp2_lvl1_part3_1_lvl2_part1_1_perm0213__D_0_1__D_2_3__S__S.Dimension(3)*G_temp2_lvl1_part3_1_lvl2_part1_1_perm0213__D_0_1__D_2_3__S__S.Dimension(1)*G_temp2_lvl1_part3_1_lvl2_part1_1_perm0213__D_0_1__D_2_3__S__S.Dimension(0)*G_temp2_lvl1_part3_1_lvl2_part1_1_perm0213__D_0_1__D_2_3__S__S.Dimension(2)*1;
}
			LocalContractAndLocalEliminate(1.0, G_temp2_lvl1_part3_1_lvl2_part1_1_perm0213__D_0_1__D_2_3__S__S.LockedTensor(), indices_mine, false,
				t_fj_lvl1_part0_1_lvl2_part1_1_perm10__S__S.LockedTensor(), indices_ne, false,
				1.0, G_mi__D_0_1__D_2_3.Tensor(), indices_mi, false);
			G_temp2_lvl1_part3_1_lvl2_part1_1_perm0213__D_0_1__D_2_3__S__S.EmptyData();
			t_fj_lvl1_part0_1_lvl2_part1_1_perm10__S__S.EmptyData();

			SlidePartitionDown
			( G_temp2_lvl1_part3_1_lvl2_part1T__D_0__D_1__D_2__D_3,  G_temp2_lvl1_part3_1_lvl2_part1_0__D_0__D_1__D_2__D_3,
			       G_temp2_lvl1_part3_1_lvl2_part1_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  G_temp2_lvl1_part3_1_lvl2_part1B__D_0__D_1__D_2__D_3, G_temp2_lvl1_part3_1_lvl2_part1_2__D_0__D_1__D_2__D_3, 1 );
			SlidePartitionDown
			( t_fj_lvl1_part0_1_lvl2_part1T__D_0_1__D_2_3,  t_fj_lvl1_part0_1_lvl2_part1_0__D_0_1__D_2_3,
			       t_fj_lvl1_part0_1_lvl2_part1_1__D_0_1__D_2_3,
			  /**/ /**/
			  t_fj_lvl1_part0_1_lvl2_part1B__D_0_1__D_2_3, t_fj_lvl1_part0_1_lvl2_part1_2__D_0_1__D_2_3, 1 );

		}
		//****

		SlidePartitionDown
		( G_temp2_lvl1_part3T__D_0__D_1__D_2__D_3,  G_temp2_lvl1_part3_0__D_0__D_1__D_2__D_3,
		       G_temp2_lvl1_part3_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  G_temp2_lvl1_part3B__D_0__D_1__D_2__D_3, G_temp2_lvl1_part3_2__D_0__D_1__D_2__D_3, 3 );
		SlidePartitionDown
		( t_fj_lvl1_part0T__D_0_1__D_2_3,  t_fj_lvl1_part0_0__D_0_1__D_2_3,
		       t_fj_lvl1_part0_1__D_0_1__D_2_3,
		  /**/ /**/
		  t_fj_lvl1_part0B__D_0_1__D_2_3, t_fj_lvl1_part0_2__D_0_1__D_2_3, 0 );

	}
	//****
	G_temp2__D_0__D_1__D_2__D_3.EmptyData();
	//**** (out of 1)
	//**** Is real	0 shadows
		//Outputs:
		//  G_mi__D_0_1__D_2_3
	PartitionDown(H_me__D_0_1__D_2_3, H_me_lvl1_part0T__D_0_1__D_2_3, H_me_lvl1_part0B__D_0_1__D_2_3, 0, 0);
	PartitionDown(G_mi__D_0_1__D_2_3, G_mi_lvl1_part0T__D_0_1__D_2_3, G_mi_lvl1_part0B__D_0_1__D_2_3, 0, 0);
	while(G_mi_lvl1_part0T__D_0_1__D_2_3.Dimension(0) < G_mi__D_0_1__D_2_3.Dimension(0))
	{
		RepartitionDown
		( H_me_lvl1_part0T__D_0_1__D_2_3,  H_me_lvl1_part0_0__D_0_1__D_2_3,
		  /**/ /**/
		       H_me_lvl1_part0_1__D_0_1__D_2_3,
		  H_me_lvl1_part0B__D_0_1__D_2_3, H_me_lvl1_part0_2__D_0_1__D_2_3, 0, blkSize );
		RepartitionDown
		( G_mi_lvl1_part0T__D_0_1__D_2_3,  G_mi_lvl1_part0_0__D_0_1__D_2_3,
		  /**/ /**/
		       G_mi_lvl1_part0_1__D_0_1__D_2_3,
		  G_mi_lvl1_part0B__D_0_1__D_2_3, G_mi_lvl1_part0_2__D_0_1__D_2_3, 0, blkSize );

		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  G_mi_lvl1_part0_1__D_0_1__D_2_3
		PartitionDown(H_me_lvl1_part0_1__D_0_1__D_2_3, H_me_lvl1_part0_1_lvl2_part1T__D_0_1__D_2_3, H_me_lvl1_part0_1_lvl2_part1B__D_0_1__D_2_3, 1, 0);
		PartitionDown(t_fj__D_0_1__D_2_3, t_fj_lvl2_part0T__D_0_1__D_2_3, t_fj_lvl2_part0B__D_0_1__D_2_3, 0, 0);
		while(H_me_lvl1_part0_1_lvl2_part1T__D_0_1__D_2_3.Dimension(1) < H_me_lvl1_part0_1__D_0_1__D_2_3.Dimension(1))
		{
			RepartitionDown
			( H_me_lvl1_part0_1_lvl2_part1T__D_0_1__D_2_3,  H_me_lvl1_part0_1_lvl2_part1_0__D_0_1__D_2_3,
			  /**/ /**/
			       H_me_lvl1_part0_1_lvl2_part1_1__D_0_1__D_2_3,
			  H_me_lvl1_part0_1_lvl2_part1B__D_0_1__D_2_3, H_me_lvl1_part0_1_lvl2_part1_2__D_0_1__D_2_3, 1, blkSize );
			RepartitionDown
			( t_fj_lvl2_part0T__D_0_1__D_2_3,  t_fj_lvl2_part0_0__D_0_1__D_2_3,
			  /**/ /**/
			       t_fj_lvl2_part0_1__D_0_1__D_2_3,
			  t_fj_lvl2_part0B__D_0_1__D_2_3, t_fj_lvl2_part0_2__D_0_1__D_2_3, 0, blkSize );

			   // t_fj_lvl2_part0_1[*,D23] <- t_fj_lvl2_part0_1[D01,D23]
			t_fj_lvl2_part0_1__S__D_2_3.AlignModesWith( modes_1, G_mi_lvl1_part0_1__D_0_1__D_2_3, modes_1 );
			t_fj_lvl2_part0_1__S__D_2_3.AllGatherRedistFrom( t_fj_lvl2_part0_1__D_0_1__D_2_3, modes_0_1 );
			   // H_me_lvl1_part0_1_lvl2_part1_1[D01,*] <- H_me_lvl1_part0_1_lvl2_part1_1[D01,D23]
			H_me_lvl1_part0_1_lvl2_part1_1__D_0_1__S.AlignModesWith( modes_0, G_mi_lvl1_part0_1__D_0_1__D_2_3, modes_0 );
			H_me_lvl1_part0_1_lvl2_part1_1__D_0_1__S.AllGatherRedistFrom( H_me_lvl1_part0_1_lvl2_part1_1__D_0_1__D_2_3, modes_2_3 );
			   // 1.0 * H_me_lvl1_part0_1_lvl2_part1_1[D01,*]_me * t_fj_lvl2_part0_1[*,D23]_ei + 1.0 * G_mi_lvl1_part0_1[D01,D23]_mi
if(commRank == 0){
flops += 2*H_me_lvl1_part0_1_lvl2_part1_1__D_0_1__S.Dimension(1)*H_me_lvl1_part0_1_lvl2_part1_1__D_0_1__S.Dimension(0)*t_fj_lvl2_part0_1__S__D_2_3.Dimension(1);
}
			LocalContractAndLocalEliminate(1.0, H_me_lvl1_part0_1_lvl2_part1_1__D_0_1__S.LockedTensor(), indices_me, false,
				t_fj_lvl2_part0_1__S__D_2_3.LockedTensor(), indices_ei, false,
				1.0, G_mi_lvl1_part0_1__D_0_1__D_2_3.Tensor(), indices_mi, false);
			H_me_lvl1_part0_1_lvl2_part1_1__D_0_1__S.EmptyData();
			t_fj_lvl2_part0_1__S__D_2_3.EmptyData();

			SlidePartitionDown
			( H_me_lvl1_part0_1_lvl2_part1T__D_0_1__D_2_3,  H_me_lvl1_part0_1_lvl2_part1_0__D_0_1__D_2_3,
			       H_me_lvl1_part0_1_lvl2_part1_1__D_0_1__D_2_3,
			  /**/ /**/
			  H_me_lvl1_part0_1_lvl2_part1B__D_0_1__D_2_3, H_me_lvl1_part0_1_lvl2_part1_2__D_0_1__D_2_3, 1 );
			SlidePartitionDown
			( t_fj_lvl2_part0T__D_0_1__D_2_3,  t_fj_lvl2_part0_0__D_0_1__D_2_3,
			       t_fj_lvl2_part0_1__D_0_1__D_2_3,
			  /**/ /**/
			  t_fj_lvl2_part0B__D_0_1__D_2_3, t_fj_lvl2_part0_2__D_0_1__D_2_3, 0 );

		}
		//****

		SlidePartitionDown
		( H_me_lvl1_part0T__D_0_1__D_2_3,  H_me_lvl1_part0_0__D_0_1__D_2_3,
		       H_me_lvl1_part0_1__D_0_1__D_2_3,
		  /**/ /**/
		  H_me_lvl1_part0B__D_0_1__D_2_3, H_me_lvl1_part0_2__D_0_1__D_2_3, 0 );
		SlidePartitionDown
		( G_mi_lvl1_part0T__D_0_1__D_2_3,  G_mi_lvl1_part0_0__D_0_1__D_2_3,
		       G_mi_lvl1_part0_1__D_0_1__D_2_3,
		  /**/ /**/
		  G_mi_lvl1_part0B__D_0_1__D_2_3, G_mi_lvl1_part0_2__D_0_1__D_2_3, 0 );

	}
	//****


//****


//END_CODE

    /*****************************************/
    mpi::Barrier(g.OwningComm());
    runTime = mpi::Time() - startTime;
    gflops = flops / (1e9 * runTime);
#ifdef CORRECTNESS
    DistTensor<double> diff_G(dist__D_0_1__D_2_3, g);
    diff_G.ResizeTo(check_G);
    Diff(check_G, G_mi__D_0_1__D_2_3, diff_G);
   norm = 1.0;
   norm = Norm(diff_G);
   if (commRank == 0){
     std::cout << "NORM_G " << norm << std::endl;
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


