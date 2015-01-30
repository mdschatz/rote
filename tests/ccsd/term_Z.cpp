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
ObjShape tempShape;
TensorDistribution dist__S__S__D_2__D_3 = tmen::StringToTensorDist("[(),(),(2),(3)]");
TensorDistribution dist__S__D_1__D_2__D_3 = tmen::StringToTensorDist("[(),(1),(2),(3)]");
TensorDistribution dist__S__D_2 = tmen::StringToTensorDist("[(),(2)]");
TensorDistribution dist__D_0__S__S__D_2 = tmen::StringToTensorDist("[(0),(),(),(2)]");
TensorDistribution dist__D_0__S__D_2__S = tmen::StringToTensorDist("[(0),(),(2),()]");
TensorDistribution dist__D_0__S = tmen::StringToTensorDist("[(0),()]");
TensorDistribution dist__D_0__D_1__S__S__D_2__D_3 = tmen::StringToTensorDist("[(0),(1),(),(),(2),(3)]");
TensorDistribution dist__D_0__D_1__S__S = tmen::StringToTensorDist("[(0),(1),(),()]");
TensorDistribution dist__D_0__D_1__S__D_2 = tmen::StringToTensorDist("[(0),(1),(),(2)]");
TensorDistribution dist__D_0__D_1__S__D_3 = tmen::StringToTensorDist("[(0),(1),(),(3)]");
TensorDistribution dist__D_0__D_1__D_2__D_3 = tmen::StringToTensorDist("[(0),(1),(2),(3)]");
TensorDistribution dist__D_0__D_1__D_3__D_2 = tmen::StringToTensorDist("[(0),(1),(3),(2)]");
TensorDistribution dist__D_0__D_3__D_2__D_1 = tmen::StringToTensorDist("[(0),(3),(2),(1)]");
TensorDistribution dist__D_2_0__D_1__S__D_3 = tmen::StringToTensorDist("[(2,0),(1),(),(3)]");
TensorDistribution dist__D_2_0__D_3__S__D_1 = tmen::StringToTensorDist("[(2,0),(3),(),(1)]");
TensorDistribution dist__D_1__S__S__D_3 = tmen::StringToTensorDist("[(1),(),(),(3)]");
TensorDistribution dist__D_1__S__D_2__D_3 = tmen::StringToTensorDist("[(1),(),(2),(3)]");
TensorDistribution dist__D_1__S__D_3__S = tmen::StringToTensorDist("[(1),(),(3),()]");
TensorDistribution dist__D_1__D_0__D_2__D_3 = tmen::StringToTensorDist("[(1),(0),(2),(3)]");
TensorDistribution dist__D_1__D_0__D_3__S = tmen::StringToTensorDist("[(1),(0),(3),()]");
TensorDistribution dist__D_1__D_0__D_3__D_2 = tmen::StringToTensorDist("[(1),(0),(3),(2)]");
TensorDistribution dist__D_2__D_1__D_0__D_3 = tmen::StringToTensorDist("[(2),(1),(0),(3)]");
TensorDistribution dist__D_2__D_3__S__S = tmen::StringToTensorDist("[(2),(3),(),()]");
TensorDistribution dist__D_2__D_3__S__D_1__D_0 = tmen::StringToTensorDist("[(2),(3),(),(1),(0)]");
TensorDistribution dist__D_2__D_3__S__D_1 = tmen::StringToTensorDist("[(2),(3),(),(1)]");
TensorDistribution dist__D_2__D_3__D_0__S = tmen::StringToTensorDist("[(2),(3),(0),()]");
TensorDistribution dist__D_0_1__D_2_3 = tmen::StringToTensorDist("[(0,1),(2,3)]");
Permutation perm_0_1( 2 );
perm_0_1[0] = 0;
perm_0_1[1] = 1;
Permutation perm_0_1_2_3( 4 );
perm_0_1_2_3[0] = 0;
perm_0_1_2_3[1] = 1;
perm_0_1_2_3[2] = 2;
perm_0_1_2_3[3] = 3;
Permutation perm_0_1_2_3_4_5( 6 );
perm_0_1_2_3_4_5[0] = 0;
perm_0_1_2_3_4_5[1] = 1;
perm_0_1_2_3_4_5[2] = 2;
perm_0_1_2_3_4_5[3] = 3;
perm_0_1_2_3_4_5[4] = 4;
perm_0_1_2_3_4_5[5] = 5;
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
Permutation perm_0_3_1_2( 4 );
perm_0_3_1_2[0] = 0;
perm_0_3_1_2[1] = 3;
perm_0_3_1_2[2] = 1;
perm_0_3_1_2[3] = 2;
Permutation perm_1_0( 2 );
perm_1_0[0] = 1;
perm_1_0[1] = 0;
Permutation perm_1_0_3_2( 4 );
perm_1_0_3_2[0] = 1;
perm_1_0_3_2[1] = 0;
perm_1_0_3_2[2] = 3;
perm_1_0_3_2[3] = 2;
Permutation perm_1_2_0_3( 4 );
perm_1_2_0_3[0] = 1;
perm_1_2_0_3[1] = 2;
perm_1_2_0_3[2] = 0;
perm_1_2_0_3[3] = 3;
Permutation perm_1_2_3_0( 4 );
perm_1_2_3_0[0] = 1;
perm_1_2_3_0[1] = 2;
perm_1_2_3_0[2] = 3;
perm_1_2_3_0[3] = 0;
Permutation perm_1_3_0_2( 4 );
perm_1_3_0_2[0] = 1;
perm_1_3_0_2[1] = 3;
perm_1_3_0_2[2] = 0;
perm_1_3_0_2[3] = 2;
Permutation perm_2_0_1_3( 4 );
perm_2_0_1_3[0] = 2;
perm_2_0_1_3[1] = 0;
perm_2_0_1_3[2] = 1;
perm_2_0_1_3[3] = 3;
Permutation perm_2_0_3_1( 4 );
perm_2_0_3_1[0] = 2;
perm_2_0_3_1[1] = 0;
perm_2_0_3_1[2] = 3;
perm_2_0_3_1[3] = 1;
Permutation perm_2_1_0_3( 4 );
perm_2_1_0_3[0] = 2;
perm_2_1_0_3[1] = 1;
perm_2_1_0_3[2] = 0;
perm_2_1_0_3[3] = 3;
Permutation perm_2_3_0_1( 4 );
perm_2_3_0_1[0] = 2;
perm_2_3_0_1[1] = 3;
perm_2_3_0_1[2] = 0;
perm_2_3_0_1[3] = 1;
Permutation perm_2_3_1_0( 4 );
perm_2_3_1_0[0] = 2;
perm_2_3_1_0[1] = 3;
perm_2_3_1_0[2] = 1;
perm_2_3_1_0[3] = 0;
Permutation perm_3_0_1_2_4( 5 );
perm_3_0_1_2_4[0] = 3;
perm_3_0_1_2_4[1] = 0;
perm_3_0_1_2_4[2] = 1;
perm_3_0_1_2_4[3] = 2;
perm_3_0_1_2_4[4] = 4;
Permutation perm_3_1_0_2( 4 );
perm_3_1_0_2[0] = 3;
perm_3_1_0_2[1] = 1;
perm_3_1_0_2[2] = 0;
perm_3_1_0_2[3] = 2;
Permutation perm_3_2_0_1( 4 );
perm_3_2_0_1[0] = 3;
perm_3_2_0_1[1] = 2;
perm_3_2_0_1[2] = 0;
perm_3_2_0_1[3] = 1;
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
ModeArray modes_0_3( 2 );
modes_0_3[0] = 0;
modes_0_3[1] = 3;
ModeArray modes_1( 1 );
modes_1[0] = 1;
ModeArray modes_1_0_3_2( 4 );
modes_1_0_3_2[0] = 1;
modes_1_0_3_2[1] = 0;
modes_1_0_3_2[2] = 3;
modes_1_0_3_2[3] = 2;
ModeArray modes_1_2_3( 3 );
modes_1_2_3[0] = 1;
modes_1_2_3[1] = 2;
modes_1_2_3[2] = 3;
ModeArray modes_1_3( 2 );
modes_1_3[0] = 1;
modes_1_3[1] = 3;
ModeArray modes_2( 1 );
modes_2[0] = 2;
ModeArray modes_2_3( 2 );
modes_2_3[0] = 2;
modes_2_3[1] = 3;
ModeArray modes_2_3_1( 3 );
modes_2_3_1[0] = 2;
modes_2_3_1[1] = 3;
modes_2_3_1[2] = 1;
ModeArray modes_5_4( 2 );
modes_5_4[0] = 5;
modes_5_4[1] = 4;
IndexArray indices_abef( 4 );
indices_abef[0] = 'a';
indices_abef[1] = 'b';
indices_abef[2] = 'e';
indices_abef[3] = 'f';
IndexArray indices_abij( 4 );
indices_abij[0] = 'a';
indices_abij[1] = 'b';
indices_abij[2] = 'i';
indices_abij[3] = 'j';
IndexArray indices_abijef( 6 );
indices_abijef[0] = 'a';
indices_abijef[1] = 'b';
indices_abijef[2] = 'i';
indices_abijef[3] = 'j';
indices_abijef[4] = 'e';
indices_abijef[5] = 'f';
IndexArray indices_ae( 2 );
indices_ae[0] = 'a';
indices_ae[1] = 'e';
IndexArray indices_bjai( 4 );
indices_bjai[0] = 'b';
indices_bjai[1] = 'j';
indices_bjai[2] = 'a';
indices_bjai[3] = 'i';
IndexArray indices_bjme( 4 );
indices_bjme[0] = 'b';
indices_bjme[1] = 'j';
indices_bjme[2] = 'm';
indices_bjme[3] = 'e';
IndexArray indices_ebij( 4 );
indices_ebij[0] = 'e';
indices_ebij[1] = 'b';
indices_ebij[2] = 'i';
indices_ebij[3] = 'j';
IndexArray indices_efij( 4 );
indices_efij[0] = 'e';
indices_efij[1] = 'f';
indices_efij[2] = 'i';
indices_efij[3] = 'j';
IndexArray indices_ei( 2 );
indices_ei[0] = 'e';
indices_ei[1] = 'i';
IndexArray indices_iabj( 4 );
indices_iabj[0] = 'i';
indices_iabj[1] = 'a';
indices_iabj[2] = 'b';
indices_iabj[3] = 'j';
IndexArray indices_ijab( 4 );
indices_ijab[0] = 'i';
indices_ijab[1] = 'j';
indices_ijab[2] = 'a';
indices_ijab[3] = 'b';
IndexArray indices_ijba( 4 );
indices_ijba[0] = 'i';
indices_ijba[1] = 'j';
indices_ijba[2] = 'b';
indices_ijba[3] = 'a';
IndexArray indices_ijbm( 4 );
indices_ijbm[0] = 'i';
indices_ijbm[1] = 'j';
indices_ijbm[2] = 'b';
indices_ijbm[3] = 'm';
IndexArray indices_ijmn( 4 );
indices_ijmn[0] = 'i';
indices_ijmn[1] = 'j';
indices_ijmn[2] = 'm';
indices_ijmn[3] = 'n';
IndexArray indices_im( 2 );
indices_im[0] = 'i';
indices_im[1] = 'm';
IndexArray indices_jabe( 4 );
indices_jabe[0] = 'j';
indices_jabe[1] = 'a';
indices_jabe[2] = 'b';
indices_jabe[3] = 'e';
IndexArray indices_jabie( 5 );
indices_jabie[0] = 'j';
indices_jabie[1] = 'a';
indices_jabie[2] = 'b';
indices_jabie[3] = 'i';
indices_jabie[4] = 'e';
IndexArray indices_ma( 2 );
indices_ma[0] = 'm';
indices_ma[1] = 'a';
IndexArray indices_mabj( 4 );
indices_mabj[0] = 'm';
indices_mabj[1] = 'a';
indices_mabj[2] = 'b';
indices_mabj[3] = 'j';
IndexArray indices_meai( 4 );
indices_meai[0] = 'm';
indices_meai[1] = 'e';
indices_meai[2] = 'a';
indices_meai[3] = 'i';
IndexArray indices_mnab( 4 );
indices_mnab[0] = 'm';
indices_mnab[1] = 'n';
indices_mnab[2] = 'a';
indices_mnab[3] = 'b';
	//F_ae[D01,D23]
DistTensor<double> F_ae__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//F_ae[D0,*]
DistTensor<double> F_ae__D_0__S( dist__D_0__S, g );
	//G_mi[D01,D23]
DistTensor<double> G_mi__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//G_mi_lvl1_part0B[D01,D23]
DistTensor<double> G_mi_lvl1_part0B__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//G_mi_lvl1_part0T[D01,D23]
DistTensor<double> G_mi_lvl1_part0T__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//G_mi_lvl2_part0B[D01,D23]
DistTensor<double> G_mi_lvl2_part0B__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//G_mi_lvl2_part0T[D01,D23]
DistTensor<double> G_mi_lvl2_part0T__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//G_mi_lvl2_part0_0[D01,D23]
DistTensor<double> G_mi_lvl2_part0_0__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//G_mi_lvl2_part0_1[D01,D23]
DistTensor<double> G_mi_lvl2_part0_1__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//G_mi_lvl2_part0_1[*,D2]
DistTensor<double> G_mi_lvl2_part0_1_perm10__D_2__S( dist__S__D_2, g );
G_mi_lvl2_part0_1_perm10__D_2__S.SetLocalPermutation( perm_1_0 );
	//G_mi_lvl2_part0_2[D01,D23]
DistTensor<double> G_mi_lvl2_part0_2__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//P_jimb[D0,D1,D2,D3]
DistTensor<double> P_jimb__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//P_jimb_lvl0_part0B[D0,D1,D2,D3]
DistTensor<double> P_jimb_lvl0_part0B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//P_jimb_lvl0_part0T[D0,D1,D2,D3]
DistTensor<double> P_jimb_lvl0_part0T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//P_jimb_lvl1_part0B[D0,D1,D2,D3]
DistTensor<double> P_jimb_lvl1_part0B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//P_jimb_lvl1_part0T[D0,D1,D2,D3]
DistTensor<double> P_jimb_lvl1_part0T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//P_jimb_lvl1_part0_0[D0,D1,D2,D3]
DistTensor<double> P_jimb_lvl1_part0_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//P_jimb_lvl1_part0_1[D0,D1,D2,D3]
DistTensor<double> P_jimb_lvl1_part0_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//P_jimb_lvl1_part0_1_lvl1_part2B[D0,D1,D2,D3]
DistTensor<double> P_jimb_lvl1_part0_1_lvl1_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//P_jimb_lvl1_part0_1_lvl1_part2T[D0,D1,D2,D3]
DistTensor<double> P_jimb_lvl1_part0_1_lvl1_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//P_jimb_lvl1_part0_1_lvl2_part2B[D0,D1,D2,D3]
DistTensor<double> P_jimb_lvl1_part0_1_lvl2_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//P_jimb_lvl1_part0_1_lvl2_part2T[D0,D1,D2,D3]
DistTensor<double> P_jimb_lvl1_part0_1_lvl2_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//P_jimb_lvl1_part0_1_lvl2_part2_0[D0,D1,D2,D3]
DistTensor<double> P_jimb_lvl1_part0_1_lvl2_part2_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//P_jimb_lvl1_part0_1_lvl2_part2_1[D0,D1,D2,D3]
DistTensor<double> P_jimb_lvl1_part0_1_lvl2_part2_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//P_jimb_lvl1_part0_1_lvl2_part2_1[D0,D3,D2,D1]
DistTensor<double> P_jimb_lvl1_part0_1_lvl2_part2_1__D_0__D_3__D_2__D_1( dist__D_0__D_3__D_2__D_1, g );
	//P_jimb_lvl1_part0_1_lvl2_part2_1[D2,D3,*,D1]
DistTensor<double> P_jimb_lvl1_part0_1_lvl2_part2_1_perm0132__D_2__D_3__D_1__S( dist__D_2__D_3__S__D_1, g );
P_jimb_lvl1_part0_1_lvl2_part2_1_perm0132__D_2__D_3__D_1__S.SetLocalPermutation( perm_0_1_3_2 );
	//P_jimb_lvl1_part0_1_lvl2_part2_2[D0,D1,D2,D3]
DistTensor<double> P_jimb_lvl1_part0_1_lvl2_part2_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//P_jimb_lvl1_part0_2[D0,D1,D2,D3]
DistTensor<double> P_jimb_lvl1_part0_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Q_mnij[D0,D1,D2,D3]
DistTensor<double> Q_mnij__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Q_mnij_lvl0_part0B[D0,D1,D2,D3]
DistTensor<double> Q_mnij_lvl0_part0B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Q_mnij_lvl0_part0T[D0,D1,D2,D3]
DistTensor<double> Q_mnij_lvl0_part0T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Q_mnij_lvl1_part0B[D0,D1,D2,D3]
DistTensor<double> Q_mnij_lvl1_part0B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Q_mnij_lvl1_part0T[D0,D1,D2,D3]
DistTensor<double> Q_mnij_lvl1_part0T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Q_mnij_lvl1_part0_0[D0,D1,D2,D3]
DistTensor<double> Q_mnij_lvl1_part0_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Q_mnij_lvl1_part0_1[D0,D1,D2,D3]
DistTensor<double> Q_mnij_lvl1_part0_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Q_mnij_lvl1_part0_1_lvl1_part1B[D0,D1,D2,D3]
DistTensor<double> Q_mnij_lvl1_part0_1_lvl1_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Q_mnij_lvl1_part0_1_lvl1_part1T[D0,D1,D2,D3]
DistTensor<double> Q_mnij_lvl1_part0_1_lvl1_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Q_mnij_lvl1_part0_1_lvl2_part1B[D0,D1,D2,D3]
DistTensor<double> Q_mnij_lvl1_part0_1_lvl2_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Q_mnij_lvl1_part0_1_lvl2_part1T[D0,D1,D2,D3]
DistTensor<double> Q_mnij_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Q_mnij_lvl1_part0_1_lvl2_part1_0[D0,D1,D2,D3]
DistTensor<double> Q_mnij_lvl1_part0_1_lvl2_part1_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Q_mnij_lvl1_part0_1_lvl2_part1_1[D0,D1,D2,D3]
DistTensor<double> Q_mnij_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Q_mnij_lvl1_part0_1_lvl2_part1_1[*,*,D2,D3]
DistTensor<double> Q_mnij_lvl1_part0_1_lvl2_part1_1_perm2301__D_2__D_3__S__S( dist__S__S__D_2__D_3, g );
Q_mnij_lvl1_part0_1_lvl2_part1_1_perm2301__D_2__D_3__S__S.SetLocalPermutation( perm_2_3_0_1 );
	//Q_mnij_lvl1_part0_1_lvl2_part1_2[D0,D1,D2,D3]
DistTensor<double> Q_mnij_lvl1_part0_1_lvl2_part1_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Q_mnij_lvl1_part0_2[D0,D1,D2,D3]
DistTensor<double> Q_mnij_lvl1_part0_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//T_bfnj[D0,D1,D2,D3]
DistTensor<double> T_bfnj__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//T_bfnj_lvl0_part1B[D0,D1,D2,D3]
DistTensor<double> T_bfnj_lvl0_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//T_bfnj_lvl0_part1T[D0,D1,D2,D3]
DistTensor<double> T_bfnj_lvl0_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//T_bfnj_lvl0_part2B[D0,D1,D2,D3]
DistTensor<double> T_bfnj_lvl0_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//T_bfnj_lvl0_part2T[D0,D1,D2,D3]
DistTensor<double> T_bfnj_lvl0_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//T_bfnj_lvl0_part3B[D0,D1,D2,D3]
DistTensor<double> T_bfnj_lvl0_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//T_bfnj_lvl0_part3T[D0,D1,D2,D3]
DistTensor<double> T_bfnj_lvl0_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//T_bfnj_lvl1_part1B[D0,D1,D2,D3]
DistTensor<double> T_bfnj_lvl1_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//T_bfnj_lvl1_part1T[D0,D1,D2,D3]
DistTensor<double> T_bfnj_lvl1_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//T_bfnj_lvl1_part1_0[D0,D1,D2,D3]
DistTensor<double> T_bfnj_lvl1_part1_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//T_bfnj_lvl1_part1_1[D0,D1,D2,D3]
DistTensor<double> T_bfnj_lvl1_part1_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//T_bfnj_lvl1_part1_1_lvl1_part2B[D0,D1,D2,D3]
DistTensor<double> T_bfnj_lvl1_part1_1_lvl1_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//T_bfnj_lvl1_part1_1_lvl1_part2T[D0,D1,D2,D3]
DistTensor<double> T_bfnj_lvl1_part1_1_lvl1_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//T_bfnj_lvl1_part1_1_lvl2_part2B[D0,D1,D2,D3]
DistTensor<double> T_bfnj_lvl1_part1_1_lvl2_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//T_bfnj_lvl1_part1_1_lvl2_part2T[D0,D1,D2,D3]
DistTensor<double> T_bfnj_lvl1_part1_1_lvl2_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//T_bfnj_lvl1_part1_1_lvl2_part2_0[D0,D1,D2,D3]
DistTensor<double> T_bfnj_lvl1_part1_1_lvl2_part2_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//T_bfnj_lvl1_part1_1_lvl2_part2_1[D0,D1,D2,D3]
DistTensor<double> T_bfnj_lvl1_part1_1_lvl2_part2_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//T_bfnj_lvl1_part1_1_lvl2_part2_1[D0,D1,*,D2]
DistTensor<double> T_bfnj_lvl1_part1_1_lvl2_part2_1__D_0__D_1__S__D_2( dist__D_0__D_1__S__D_2, g );
	//T_bfnj_lvl1_part1_1_lvl2_part2_1[D0,*,*,D2]
DistTensor<double> T_bfnj_lvl1_part1_1_lvl2_part2_1_perm2103__S__S__D_0__D_2( dist__D_0__S__S__D_2, g );
T_bfnj_lvl1_part1_1_lvl2_part2_1_perm2103__S__S__D_0__D_2.SetLocalPermutation( perm_2_1_0_3 );
	//T_bfnj_lvl1_part1_1_lvl2_part2_2[D0,D1,D2,D3]
DistTensor<double> T_bfnj_lvl1_part1_1_lvl2_part2_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//T_bfnj_lvl1_part1_2[D0,D1,D2,D3]
DistTensor<double> T_bfnj_lvl1_part1_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
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
	//T_bfnj_lvl1_part2_1_lvl2_part3_1[*,D1,D2,D3]
DistTensor<double> T_bfnj_lvl1_part2_1_lvl2_part3_1__S__D_1__D_2__D_3( dist__S__D_1__D_2__D_3, g );
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
	//T_bfnj_lvl1_part3_1_lvl2_part2_1[D0,D1,*,D3]
DistTensor<double> T_bfnj_lvl1_part3_1_lvl2_part2_1_perm2013__S__D_0__D_1__D_3( dist__D_0__D_1__S__D_3, g );
T_bfnj_lvl1_part3_1_lvl2_part2_1_perm2013__S__D_0__D_1__D_3.SetLocalPermutation( perm_2_0_1_3 );
	//T_bfnj_lvl1_part3_1_lvl2_part2_2[D0,D1,D2,D3]
DistTensor<double> T_bfnj_lvl1_part3_1_lvl2_part2_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//T_bfnj_lvl1_part3_2[D0,D1,D2,D3]
DistTensor<double> T_bfnj_lvl1_part3_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Tau_efmn[D0,D1,D2,D3]
DistTensor<double> Tau_efmn__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Tau_efmn_lvl0_part2B[D0,D1,D2,D3]
DistTensor<double> Tau_efmn_lvl0_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Tau_efmn_lvl0_part2T[D0,D1,D2,D3]
DistTensor<double> Tau_efmn_lvl0_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Tau_efmn_lvl1_part2B[D0,D1,D2,D3]
DistTensor<double> Tau_efmn_lvl1_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Tau_efmn_lvl1_part2T[D0,D1,D2,D3]
DistTensor<double> Tau_efmn_lvl1_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Tau_efmn_lvl1_part2_0[D0,D1,D2,D3]
DistTensor<double> Tau_efmn_lvl1_part2_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Tau_efmn_lvl1_part2_1[D0,D1,D2,D3]
DistTensor<double> Tau_efmn_lvl1_part2_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Tau_efmn_lvl1_part2_1_lvl1_part3B[D0,D1,D2,D3]
DistTensor<double> Tau_efmn_lvl1_part2_1_lvl1_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Tau_efmn_lvl1_part2_1_lvl1_part3T[D0,D1,D2,D3]
DistTensor<double> Tau_efmn_lvl1_part2_1_lvl1_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Tau_efmn_lvl1_part2_1_lvl2_part3B[D0,D1,D2,D3]
DistTensor<double> Tau_efmn_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Tau_efmn_lvl1_part2_1_lvl2_part3T[D0,D1,D2,D3]
DistTensor<double> Tau_efmn_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Tau_efmn_lvl1_part2_1_lvl2_part3_0[D0,D1,D2,D3]
DistTensor<double> Tau_efmn_lvl1_part2_1_lvl2_part3_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Tau_efmn_lvl1_part2_1_lvl2_part3_1[D0,D1,D2,D3]
DistTensor<double> Tau_efmn_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Tau_efmn_lvl1_part2_1_lvl2_part3_1[D2,D1,D0,D3]
DistTensor<double> Tau_efmn_lvl1_part2_1_lvl2_part3_1__D_2__D_1__D_0__D_3( dist__D_2__D_1__D_0__D_3, g );
	//Tau_efmn_lvl1_part2_1_lvl2_part3_1[D2,D3,D0,*]
DistTensor<double> Tau_efmn_lvl1_part2_1_lvl2_part3_1__D_2__D_3__D_0__S( dist__D_2__D_3__D_0__S, g );
	//Tau_efmn_lvl1_part2_1_lvl2_part3_1[D2,D3,*,*]
DistTensor<double> Tau_efmn_lvl1_part2_1_lvl2_part3_1__D_2__D_3__S__S( dist__D_2__D_3__S__S, g );
	//Tau_efmn_lvl1_part2_1_lvl2_part3_1[D0,D1,*,*]
DistTensor<double> Tau_efmn_lvl1_part2_1_lvl2_part3_1_perm2301__S__S__D_0__D_1( dist__D_0__D_1__S__S, g );
Tau_efmn_lvl1_part2_1_lvl2_part3_1_perm2301__S__S__D_0__D_1.SetLocalPermutation( perm_2_3_0_1 );
	//Tau_efmn_lvl1_part2_1_lvl2_part3_2[D0,D1,D2,D3]
DistTensor<double> Tau_efmn_lvl1_part2_1_lvl2_part3_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Tau_efmn_lvl1_part2_2[D0,D1,D2,D3]
DistTensor<double> Tau_efmn_lvl1_part2_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_bmje[D0,D1,D2,D3]
DistTensor<double> W_bmje__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_bmje_lvl0_part3B[D0,D1,D2,D3]
DistTensor<double> W_bmje_lvl0_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_bmje_lvl0_part3T[D0,D1,D2,D3]
DistTensor<double> W_bmje_lvl0_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_bmje_lvl1_part3B[D0,D1,D2,D3]
DistTensor<double> W_bmje_lvl1_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_bmje_lvl1_part3T[D0,D1,D2,D3]
DistTensor<double> W_bmje_lvl1_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_bmje_lvl1_part3_0[D0,D1,D2,D3]
DistTensor<double> W_bmje_lvl1_part3_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_bmje_lvl1_part3_1[D0,D1,D2,D3]
DistTensor<double> W_bmje_lvl1_part3_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_bmje_lvl1_part3_1_lvl1_part1B[D0,D1,D2,D3]
DistTensor<double> W_bmje_lvl1_part3_1_lvl1_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_bmje_lvl1_part3_1_lvl1_part1T[D0,D1,D2,D3]
DistTensor<double> W_bmje_lvl1_part3_1_lvl1_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_bmje_lvl1_part3_1_lvl2_part1B[D0,D1,D2,D3]
DistTensor<double> W_bmje_lvl1_part3_1_lvl2_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_bmje_lvl1_part3_1_lvl2_part1T[D0,D1,D2,D3]
DistTensor<double> W_bmje_lvl1_part3_1_lvl2_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_bmje_lvl1_part3_1_lvl2_part1_0[D0,D1,D2,D3]
DistTensor<double> W_bmje_lvl1_part3_1_lvl2_part1_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_bmje_lvl1_part3_1_lvl2_part1_1[D0,D1,D2,D3]
DistTensor<double> W_bmje_lvl1_part3_1_lvl2_part1_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_bmje_lvl1_part3_1_lvl2_part1_1[D1,D0,D2,D3]
DistTensor<double> W_bmje_lvl1_part3_1_lvl2_part1_1__D_1__D_0__D_2__D_3( dist__D_1__D_0__D_2__D_3, g );
	//W_bmje_lvl1_part3_1_lvl2_part1_1[D1,D0,D3,*]
DistTensor<double> W_bmje_lvl1_part3_1_lvl2_part1_1__D_1__D_0__D_3__S( dist__D_1__D_0__D_3__S, g );
	//W_bmje_lvl1_part3_1_lvl2_part1_1[D1,*,D3,*]
DistTensor<double> W_bmje_lvl1_part3_1_lvl2_part1_1_perm0213__D_1__D_3__S__S( dist__D_1__S__D_3__S, g );
W_bmje_lvl1_part3_1_lvl2_part1_1_perm0213__D_1__D_3__S__S.SetLocalPermutation( perm_0_2_1_3 );
	//W_bmje_lvl1_part3_1_lvl2_part1_2[D0,D1,D2,D3]
DistTensor<double> W_bmje_lvl1_part3_1_lvl2_part1_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_bmje_lvl1_part3_2[D0,D1,D2,D3]
DistTensor<double> W_bmje_lvl1_part3_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//X_bmej[D0,D1,D2,D3]
DistTensor<double> X_bmej__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//X_bmej_lvl0_part2B[D0,D1,D2,D3]
DistTensor<double> X_bmej_lvl0_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//X_bmej_lvl0_part2T[D0,D1,D2,D3]
DistTensor<double> X_bmej_lvl0_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//X_bmej_lvl1_part2B[D0,D1,D2,D3]
DistTensor<double> X_bmej_lvl1_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//X_bmej_lvl1_part2T[D0,D1,D2,D3]
DistTensor<double> X_bmej_lvl1_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//X_bmej_lvl1_part2_0[D0,D1,D2,D3]
DistTensor<double> X_bmej_lvl1_part2_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//X_bmej_lvl1_part2_1[D0,D1,D2,D3]
DistTensor<double> X_bmej_lvl1_part2_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//X_bmej_lvl1_part2_1_lvl1_part1B[D0,D1,D2,D3]
DistTensor<double> X_bmej_lvl1_part2_1_lvl1_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//X_bmej_lvl1_part2_1_lvl1_part1T[D0,D1,D2,D3]
DistTensor<double> X_bmej_lvl1_part2_1_lvl1_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//X_bmej_lvl1_part2_1_lvl2_part1B[D0,D1,D2,D3]
DistTensor<double> X_bmej_lvl1_part2_1_lvl2_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//X_bmej_lvl1_part2_1_lvl2_part1T[D0,D1,D2,D3]
DistTensor<double> X_bmej_lvl1_part2_1_lvl2_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//X_bmej_lvl1_part2_1_lvl2_part1_0[D0,D1,D2,D3]
DistTensor<double> X_bmej_lvl1_part2_1_lvl2_part1_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//X_bmej_lvl1_part2_1_lvl2_part1_1[D0,D1,D2,D3]
DistTensor<double> X_bmej_lvl1_part2_1_lvl2_part1_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//X_bmej_lvl1_part2_1_lvl2_part1_1[D1,*,D2,D3]
DistTensor<double> X_bmej_lvl1_part2_1_lvl2_part1_1__D_1__S__D_2__D_3( dist__D_1__S__D_2__D_3, g );
	//X_bmej_lvl1_part2_1_lvl2_part1_1[D1,*,*,D3]
DistTensor<double> X_bmej_lvl1_part2_1_lvl2_part1_1_perm0312__D_1__D_3__S__S( dist__D_1__S__S__D_3, g );
X_bmej_lvl1_part2_1_lvl2_part1_1_perm0312__D_1__D_3__S__S.SetLocalPermutation( perm_0_3_1_2 );
	//X_bmej_lvl1_part2_1_lvl2_part1_2[D0,D1,D2,D3]
DistTensor<double> X_bmej_lvl1_part2_1_lvl2_part1_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//X_bmej_lvl1_part2_2[D0,D1,D2,D3]
DistTensor<double> X_bmej_lvl1_part2_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Z_abij[D0,D1,D2,D3]
DistTensor<double> Z_abij__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Z_abij_lvl0_part2B[D0,D1,D2,D3]
DistTensor<double> Z_abij_lvl0_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Z_abij_lvl0_part2T[D0,D1,D2,D3]
DistTensor<double> Z_abij_lvl0_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Z_abij_lvl1_part2B[D0,D1,D2,D3]
DistTensor<double> Z_abij_lvl1_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Z_abij_lvl1_part2T[D0,D1,D2,D3]
DistTensor<double> Z_abij_lvl1_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Z_abij_lvl1_part2_0[D0,D1,D2,D3]
DistTensor<double> Z_abij_lvl1_part2_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Z_abij_lvl1_part2_1[D0,D1,D2,D3]
DistTensor<double> Z_abij_lvl1_part2_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Z_abij_lvl1_part2_1_lvl1_part3B[D0,D1,D2,D3]
DistTensor<double> Z_abij_lvl1_part2_1_lvl1_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Z_abij_lvl1_part2_1_lvl1_part3T[D0,D1,D2,D3]
DistTensor<double> Z_abij_lvl1_part2_1_lvl1_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Z_abij_lvl1_part2_1_lvl2_part3B[D0,D1,D2,D3]
DistTensor<double> Z_abij_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Z_abij_lvl1_part2_1_lvl2_part3T[D0,D1,D2,D3]
DistTensor<double> Z_abij_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Z_abij_lvl1_part2_1_lvl2_part3_0[D0,D1,D2,D3]
DistTensor<double> Z_abij_lvl1_part2_1_lvl2_part3_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Z_abij_lvl1_part2_1_lvl2_part3_1[D0,D1,D2,D3]
DistTensor<double> Z_abij_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Z_abij_lvl1_part2_1_lvl2_part3_1[D0,D1,*,*,D2,D3]
DistTensor<double> Z_abij_lvl1_part2_1_lvl2_part3_1__D_0__D_1__S__S__D_2__D_3( dist__D_0__D_1__S__S__D_2__D_3, g );
	//Z_abij_lvl1_part2_1_lvl2_part3_2[D0,D1,D2,D3]
DistTensor<double> Z_abij_lvl1_part2_1_lvl2_part3_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Z_abij_lvl1_part2_2[D0,D1,D2,D3]
DistTensor<double> Z_abij_lvl1_part2_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> Z_abij_perm2301__D_2__D_3__D_0__D_1( dist__D_0__D_1__D_2__D_3, g );
Z_abij_perm2301__D_2__D_3__D_0__D_1.SetLocalPermutation( perm_2_3_0_1 );
	//Zaccum[D0,D1,D2,D3]
DistTensor<double> Zaccum__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Zaccum_lvl0_part2B[D0,D1,D2,D3]
DistTensor<double> Zaccum_lvl0_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Zaccum_lvl0_part2T[D0,D1,D2,D3]
DistTensor<double> Zaccum_lvl0_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Zaccum_lvl0_part3B[D0,D1,D2,D3]
DistTensor<double> Zaccum_lvl0_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Zaccum_lvl0_part3T[D0,D1,D2,D3]
DistTensor<double> Zaccum_lvl0_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Zaccum_lvl1_part2B[D0,D1,D2,D3]
DistTensor<double> Zaccum_lvl1_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Zaccum_lvl1_part2T[D0,D1,D2,D3]
DistTensor<double> Zaccum_lvl1_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Zaccum_lvl1_part2_0[D0,D1,D2,D3]
DistTensor<double> Zaccum_lvl1_part2_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Zaccum_lvl1_part2_1[D0,D1,D2,D3]
DistTensor<double> Zaccum_lvl1_part2_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Zaccum_lvl1_part2_1_lvl1_part3B[D0,D1,D2,D3]
DistTensor<double> Zaccum_lvl1_part2_1_lvl1_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Zaccum_lvl1_part2_1_lvl1_part3T[D0,D1,D2,D3]
DistTensor<double> Zaccum_lvl1_part2_1_lvl1_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Zaccum_lvl1_part2_1_lvl2_part3B[D0,D1,D2,D3]
DistTensor<double> Zaccum_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Zaccum_lvl1_part2_1_lvl2_part3T[D0,D1,D2,D3]
DistTensor<double> Zaccum_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Zaccum_lvl1_part2_1_lvl2_part3_0[D0,D1,D2,D3]
DistTensor<double> Zaccum_lvl1_part2_1_lvl2_part3_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Zaccum_lvl1_part2_1_lvl2_part3_1[D0,D1,D2,D3]
DistTensor<double> Zaccum_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Zaccum_lvl1_part2_1_lvl2_part3_1[D2,D3,*,D1,D0]
DistTensor<double> Zaccum_lvl1_part2_1_lvl2_part3_1_perm30124__D_1__D_2__D_3__S__D_0( dist__D_2__D_3__S__D_1__D_0, g );
Zaccum_lvl1_part2_1_lvl2_part3_1_perm30124__D_1__D_2__D_3__S__D_0.SetLocalPermutation( perm_3_0_1_2_4 );
	//Zaccum_lvl1_part2_1_lvl2_part3_1_temp[D0,D1,D2,D3]
DistTensor<double> Zaccum_lvl1_part2_1_lvl2_part3_1_temp__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Zaccum_lvl1_part2_1_lvl2_part3_1_temp[D20,D1,*,D3]
DistTensor<double> Zaccum_lvl1_part2_1_lvl2_part3_1_temp__D_2_0__D_1__S__D_3( dist__D_2_0__D_1__S__D_3, g );
	//Zaccum_lvl1_part2_1_lvl2_part3_1_temp[D20,D3,*,D1]
DistTensor<double> Zaccum_lvl1_part2_1_lvl2_part3_1_temp__D_2_0__D_3__S__D_1( dist__D_2_0__D_3__S__D_1, g );
	//Zaccum_lvl1_part2_1_lvl2_part3_2[D0,D1,D2,D3]
DistTensor<double> Zaccum_lvl1_part2_1_lvl2_part3_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> Zaccum_lvl1_part2_1_perm2310__D_2__D_3__D_1__D_0( dist__D_0__D_1__D_2__D_3, g );
Zaccum_lvl1_part2_1_perm2310__D_2__D_3__D_1__D_0.SetLocalPermutation( perm_2_3_1_0 );
	//Zaccum_lvl1_part2_2[D0,D1,D2,D3]
DistTensor<double> Zaccum_lvl1_part2_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Zaccum_lvl1_part3B[D0,D1,D2,D3]
DistTensor<double> Zaccum_lvl1_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Zaccum_lvl1_part3T[D0,D1,D2,D3]
DistTensor<double> Zaccum_lvl1_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Zaccum_lvl1_part3_0[D0,D1,D2,D3]
DistTensor<double> Zaccum_lvl1_part3_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Zaccum_lvl1_part3_1[D0,D1,D2,D3]
DistTensor<double> Zaccum_lvl1_part3_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Zaccum_lvl1_part3_1_lvl1_part2B[D0,D1,D2,D3]
DistTensor<double> Zaccum_lvl1_part3_1_lvl1_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Zaccum_lvl1_part3_1_lvl1_part2T[D0,D1,D2,D3]
DistTensor<double> Zaccum_lvl1_part3_1_lvl1_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Zaccum_lvl1_part3_1_lvl2_part2B[D0,D1,D2,D3]
DistTensor<double> Zaccum_lvl1_part3_1_lvl2_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Zaccum_lvl1_part3_1_lvl2_part2T[D0,D1,D2,D3]
DistTensor<double> Zaccum_lvl1_part3_1_lvl2_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Zaccum_lvl1_part3_1_lvl2_part2_0[D0,D1,D2,D3]
DistTensor<double> Zaccum_lvl1_part3_1_lvl2_part2_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Zaccum_lvl1_part3_1_lvl2_part2_1[D0,D1,D2,D3]
DistTensor<double> Zaccum_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Zaccum_lvl1_part3_1_lvl2_part2_1[D1,D0,D2,D3]
DistTensor<double> Zaccum_lvl1_part3_1_lvl2_part2_1__D_1__D_0__D_2__D_3( dist__D_1__D_0__D_2__D_3, g );
	//Zaccum_lvl1_part3_1_lvl2_part2_1[D1,D0,D3,D2]
DistTensor<double> Zaccum_lvl1_part3_1_lvl2_part2_1__D_1__D_0__D_3__D_2( dist__D_1__D_0__D_3__D_2, g );
	//Zaccum_lvl1_part3_1_lvl2_part2_2[D0,D1,D2,D3]
DistTensor<double> Zaccum_lvl1_part3_1_lvl2_part2_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> Zaccum_lvl1_part3_1_perm2013__D_2__D_0__D_1__D_3( dist__D_0__D_1__D_2__D_3, g );
Zaccum_lvl1_part3_1_perm2013__D_2__D_0__D_1__D_3.SetLocalPermutation( perm_2_0_1_3 );
	//Zaccum_lvl1_part3_2[D0,D1,D2,D3]
DistTensor<double> Zaccum_lvl1_part3_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> Zaccum_perm1302__D_1__D_3__D_0__D_2( dist__D_0__D_1__D_2__D_3, g );
Zaccum_perm1302__D_1__D_3__D_0__D_2.SetLocalPermutation( perm_1_3_0_2 );
	//Ztemp1[D0,D1,D2,D3]
DistTensor<double> Ztemp1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Ztemp1_lvl0_part2B[D0,D1,D2,D3]
DistTensor<double> Ztemp1_lvl0_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Ztemp1_lvl0_part2T[D0,D1,D2,D3]
DistTensor<double> Ztemp1_lvl0_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Ztemp1_lvl0_part3B[D0,D1,D2,D3]
DistTensor<double> Ztemp1_lvl0_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Ztemp1_lvl0_part3T[D0,D1,D2,D3]
DistTensor<double> Ztemp1_lvl0_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Ztemp1_lvl1_part2B[D0,D1,D2,D3]
DistTensor<double> Ztemp1_lvl1_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Ztemp1_lvl1_part2T[D0,D1,D2,D3]
DistTensor<double> Ztemp1_lvl1_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Ztemp1_lvl1_part2_0[D0,D1,D2,D3]
DistTensor<double> Ztemp1_lvl1_part2_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Ztemp1_lvl1_part2_1[D0,D1,D2,D3]
DistTensor<double> Ztemp1_lvl1_part2_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Ztemp1_lvl1_part2_1_lvl1_part3B[D0,D1,D2,D3]
DistTensor<double> Ztemp1_lvl1_part2_1_lvl1_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Ztemp1_lvl1_part2_1_lvl1_part3T[D0,D1,D2,D3]
DistTensor<double> Ztemp1_lvl1_part2_1_lvl1_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Ztemp1_lvl1_part2_1_lvl2_part3B[D0,D1,D2,D3]
DistTensor<double> Ztemp1_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Ztemp1_lvl1_part2_1_lvl2_part3T[D0,D1,D2,D3]
DistTensor<double> Ztemp1_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Ztemp1_lvl1_part2_1_lvl2_part3_0[D0,D1,D2,D3]
DistTensor<double> Ztemp1_lvl1_part2_1_lvl2_part3_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Ztemp1_lvl1_part2_1_lvl2_part3_1[D0,D1,D2,D3]
DistTensor<double> Ztemp1_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Ztemp1_lvl1_part2_1_lvl2_part3_2[D0,D1,D2,D3]
DistTensor<double> Ztemp1_lvl1_part2_1_lvl2_part3_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Ztemp1_lvl1_part2_2[D0,D1,D2,D3]
DistTensor<double> Ztemp1_lvl1_part2_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Ztemp1_lvl1_part3B[D0,D1,D2,D3]
DistTensor<double> Ztemp1_lvl1_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Ztemp1_lvl1_part3T[D0,D1,D2,D3]
DistTensor<double> Ztemp1_lvl1_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Ztemp1_lvl1_part3_0[D0,D1,D2,D3]
DistTensor<double> Ztemp1_lvl1_part3_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Ztemp1_lvl1_part3_1[D0,D1,D2,D3]
DistTensor<double> Ztemp1_lvl1_part3_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Ztemp1_lvl1_part3_1_lvl1_part2B[D0,D1,D2,D3]
DistTensor<double> Ztemp1_lvl1_part3_1_lvl1_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Ztemp1_lvl1_part3_1_lvl1_part2T[D0,D1,D2,D3]
DistTensor<double> Ztemp1_lvl1_part3_1_lvl1_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Ztemp1_lvl1_part3_1_lvl2_part2B[D0,D1,D2,D3]
DistTensor<double> Ztemp1_lvl1_part3_1_lvl2_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Ztemp1_lvl1_part3_1_lvl2_part2T[D0,D1,D2,D3]
DistTensor<double> Ztemp1_lvl1_part3_1_lvl2_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Ztemp1_lvl1_part3_1_lvl2_part2_0[D0,D1,D2,D3]
DistTensor<double> Ztemp1_lvl1_part3_1_lvl2_part2_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Ztemp1_lvl1_part3_1_lvl2_part2_1[D0,D1,D2,D3]
DistTensor<double> Ztemp1_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Ztemp1_lvl1_part3_1_lvl2_part2_1[D0,D1,D3,D2]
DistTensor<double> Ztemp1_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_3__D_2( dist__D_0__D_1__D_3__D_2, g );
	//Ztemp1_lvl1_part3_1_lvl2_part2_2[D0,D1,D2,D3]
DistTensor<double> Ztemp1_lvl1_part3_1_lvl2_part2_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Ztemp1_lvl1_part3_2[D0,D1,D2,D3]
DistTensor<double> Ztemp1_lvl1_part3_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> Ztemp1_perm1302__D_1__D_3__D_0__D_2( dist__D_0__D_1__D_2__D_3, g );
Ztemp1_perm1302__D_1__D_3__D_0__D_2.SetLocalPermutation( perm_1_3_0_2 );
	//Ztemp2[D0,D1,D2,D3]
DistTensor<double> Ztemp2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Ztemp2_lvl0_part1B[D0,D1,D2,D3]
DistTensor<double> Ztemp2_lvl0_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Ztemp2_lvl0_part1T[D0,D1,D2,D3]
DistTensor<double> Ztemp2_lvl0_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Ztemp2_lvl0_part2B[D0,D1,D2,D3]
DistTensor<double> Ztemp2_lvl0_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Ztemp2_lvl0_part2T[D0,D1,D2,D3]
DistTensor<double> Ztemp2_lvl0_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Ztemp2_lvl1_part1B[D0,D1,D2,D3]
DistTensor<double> Ztemp2_lvl1_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Ztemp2_lvl1_part1T[D0,D1,D2,D3]
DistTensor<double> Ztemp2_lvl1_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Ztemp2_lvl1_part1_0[D0,D1,D2,D3]
DistTensor<double> Ztemp2_lvl1_part1_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Ztemp2_lvl1_part1_1[D0,D1,D2,D3]
DistTensor<double> Ztemp2_lvl1_part1_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Ztemp2_lvl1_part1_1_lvl1_part3B[D0,D1,D2,D3]
DistTensor<double> Ztemp2_lvl1_part1_1_lvl1_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Ztemp2_lvl1_part1_1_lvl1_part3T[D0,D1,D2,D3]
DistTensor<double> Ztemp2_lvl1_part1_1_lvl1_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Ztemp2_lvl1_part1_1_lvl2_part3B[D0,D1,D2,D3]
DistTensor<double> Ztemp2_lvl1_part1_1_lvl2_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Ztemp2_lvl1_part1_1_lvl2_part3T[D0,D1,D2,D3]
DistTensor<double> Ztemp2_lvl1_part1_1_lvl2_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Ztemp2_lvl1_part1_1_lvl2_part3_0[D0,D1,D2,D3]
DistTensor<double> Ztemp2_lvl1_part1_1_lvl2_part3_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Ztemp2_lvl1_part1_1_lvl2_part3_1[D0,D1,D2,D3]
DistTensor<double> Ztemp2_lvl1_part1_1_lvl2_part3_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Ztemp2_lvl1_part1_1_lvl2_part3_1[D0,*,D2,*]
DistTensor<double> Ztemp2_lvl1_part1_1_lvl2_part3_1_perm3102__S__S__D_0__D_2( dist__D_0__S__D_2__S, g );
Ztemp2_lvl1_part1_1_lvl2_part3_1_perm3102__S__S__D_0__D_2.SetLocalPermutation( perm_3_1_0_2 );
	//Ztemp2_lvl1_part1_1_lvl2_part3_2[D0,D1,D2,D3]
DistTensor<double> Ztemp2_lvl1_part1_1_lvl2_part3_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Ztemp2_lvl1_part1_2[D0,D1,D2,D3]
DistTensor<double> Ztemp2_lvl1_part1_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Ztemp2_lvl1_part2B[D0,D1,D2,D3]
DistTensor<double> Ztemp2_lvl1_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Ztemp2_lvl1_part2T[D0,D1,D2,D3]
DistTensor<double> Ztemp2_lvl1_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Ztemp2_lvl1_part2_0[D0,D1,D2,D3]
DistTensor<double> Ztemp2_lvl1_part2_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Ztemp2_lvl1_part2_1[D0,D1,D2,D3]
DistTensor<double> Ztemp2_lvl1_part2_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Ztemp2_lvl1_part2_1_lvl1_part3B[D0,D1,D2,D3]
DistTensor<double> Ztemp2_lvl1_part2_1_lvl1_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Ztemp2_lvl1_part2_1_lvl1_part3T[D0,D1,D2,D3]
DistTensor<double> Ztemp2_lvl1_part2_1_lvl1_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Ztemp2_lvl1_part2_1_lvl2_part3B[D0,D1,D2,D3]
DistTensor<double> Ztemp2_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Ztemp2_lvl1_part2_1_lvl2_part3T[D0,D1,D2,D3]
DistTensor<double> Ztemp2_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Ztemp2_lvl1_part2_1_lvl2_part3_0[D0,D1,D2,D3]
DistTensor<double> Ztemp2_lvl1_part2_1_lvl2_part3_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Ztemp2_lvl1_part2_1_lvl2_part3_1[D0,D1,D2,D3]
DistTensor<double> Ztemp2_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Ztemp2_lvl1_part2_1_lvl2_part3_2[D0,D1,D2,D3]
DistTensor<double> Ztemp2_lvl1_part2_1_lvl2_part3_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Ztemp2_lvl1_part2_2[D0,D1,D2,D3]
DistTensor<double> Ztemp2_lvl1_part2_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//r_bmfe[D0,D1,D2,D3]
DistTensor<double> r_bmfe__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//r_bmfe_lvl1_part1B[D0,D1,D2,D3]
DistTensor<double> r_bmfe_lvl1_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//r_bmfe_lvl1_part1T[D0,D1,D2,D3]
DistTensor<double> r_bmfe_lvl1_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//r_bmfe_lvl2_part1B[D0,D1,D2,D3]
DistTensor<double> r_bmfe_lvl2_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//r_bmfe_lvl2_part1T[D0,D1,D2,D3]
DistTensor<double> r_bmfe_lvl2_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//r_bmfe_lvl2_part1_0[D0,D1,D2,D3]
DistTensor<double> r_bmfe_lvl2_part1_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//r_bmfe_lvl2_part1_1[D0,D1,D2,D3]
DistTensor<double> r_bmfe_lvl2_part1_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> r_bmfe_lvl2_part1_1_perm1230__D_1__D_2__D_3__D_0( dist__D_0__D_1__D_2__D_3, g );
r_bmfe_lvl2_part1_1_perm1230__D_1__D_2__D_3__D_0.SetLocalPermutation( perm_1_2_3_0 );
	//r_bmfe_lvl2_part1_2[D0,D1,D2,D3]
DistTensor<double> r_bmfe_lvl2_part1_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//t_fj[D01,D23]
DistTensor<double> t_fj__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//t_fj_lvl0_part1B[D01,D23]
DistTensor<double> t_fj_lvl0_part1B__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//t_fj_lvl0_part1T[D01,D23]
DistTensor<double> t_fj_lvl0_part1T__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//t_fj_lvl1_part1B[D01,D23]
DistTensor<double> t_fj_lvl1_part1B__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//t_fj_lvl1_part1T[D01,D23]
DistTensor<double> t_fj_lvl1_part1T__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//t_fj_lvl1_part1_0[D01,D23]
DistTensor<double> t_fj_lvl1_part1_0__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//t_fj_lvl1_part1_1[D01,D23]
DistTensor<double> t_fj_lvl1_part1_1__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//t_fj_lvl1_part1_1[D0,*]
DistTensor<double> t_fj_lvl1_part1_1__D_0__S( dist__D_0__S, g );
	//t_fj_lvl1_part1_2[D01,D23]
DistTensor<double> t_fj_lvl1_part1_2__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//t_fj_lvl2_part1B[D01,D23]
DistTensor<double> t_fj_lvl2_part1B__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//t_fj_lvl2_part1T[D01,D23]
DistTensor<double> t_fj_lvl2_part1T__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//t_fj_lvl2_part1_0[D01,D23]
DistTensor<double> t_fj_lvl2_part1_0__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//t_fj_lvl2_part1_1[D01,D23]
DistTensor<double> t_fj_lvl2_part1_1__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//t_fj_lvl2_part1_1[D0,*]
DistTensor<double> t_fj_lvl2_part1_1_perm10__S__D_0( dist__D_0__S, g );
t_fj_lvl2_part1_1_perm10__S__D_0.SetLocalPermutation( perm_1_0 );
	//t_fj_lvl2_part1_2[D01,D23]
DistTensor<double> t_fj_lvl2_part1_2__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//v_femn[D0,D1,D2,D3]
DistTensor<double> v_femn__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//y_abef[D0,D1,D2,D3]
DistTensor<double> y_abef__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
// v_femn has 4 dims
//	Starting distribution: [D0,D1,D2,D3] or _D_0__D_1__D_2__D_3
ObjShape v_femn__D_0__D_1__D_2__D_3_tempShape( 4 );
v_femn__D_0__D_1__D_2__D_3_tempShape[ 0 ] = n_v;
v_femn__D_0__D_1__D_2__D_3_tempShape[ 1 ] = n_v;
v_femn__D_0__D_1__D_2__D_3_tempShape[ 2 ] = n_o;
v_femn__D_0__D_1__D_2__D_3_tempShape[ 3 ] = n_o;
v_femn__D_0__D_1__D_2__D_3.ResizeTo( v_femn__D_0__D_1__D_2__D_3_tempShape );
MakeUniform( v_femn__D_0__D_1__D_2__D_3 );
// Q_mnij has 4 dims
ObjShape Q_mnij__D_0__D_1__D_2__D_3_tempShape( 4 );
Q_mnij__D_0__D_1__D_2__D_3_tempShape[ 0 ] = n_o;
Q_mnij__D_0__D_1__D_2__D_3_tempShape[ 1 ] = n_o;
Q_mnij__D_0__D_1__D_2__D_3_tempShape[ 2 ] = n_o;
Q_mnij__D_0__D_1__D_2__D_3_tempShape[ 3 ] = n_o;
Q_mnij__D_0__D_1__D_2__D_3.ResizeTo( Q_mnij__D_0__D_1__D_2__D_3_tempShape );
MakeUniform( Q_mnij__D_0__D_1__D_2__D_3 );
// Tau_efmn has 4 dims
ObjShape Tau_efmn__D_0__D_1__D_2__D_3_tempShape( 4 );
Tau_efmn__D_0__D_1__D_2__D_3_tempShape[ 0 ] = n_v;
Tau_efmn__D_0__D_1__D_2__D_3_tempShape[ 1 ] = n_v;
Tau_efmn__D_0__D_1__D_2__D_3_tempShape[ 2 ] = n_o;
Tau_efmn__D_0__D_1__D_2__D_3_tempShape[ 3 ] = n_o;
Tau_efmn__D_0__D_1__D_2__D_3.ResizeTo( Tau_efmn__D_0__D_1__D_2__D_3_tempShape );
MakeUniform( Tau_efmn__D_0__D_1__D_2__D_3 );
// y_abef has 4 dims
ObjShape y_abef__D_0__D_1__D_2__D_3_tempShape( 4 );
y_abef__D_0__D_1__D_2__D_3_tempShape[ 0 ] = n_v;
y_abef__D_0__D_1__D_2__D_3_tempShape[ 1 ] = n_v;
y_abef__D_0__D_1__D_2__D_3_tempShape[ 2 ] = n_v;
y_abef__D_0__D_1__D_2__D_3_tempShape[ 3 ] = n_v;
y_abef__D_0__D_1__D_2__D_3.ResizeTo( y_abef__D_0__D_1__D_2__D_3_tempShape );
MakeUniform( y_abef__D_0__D_1__D_2__D_3 );
// Z_abij has 4 dims
ObjShape Z_abij__D_0__D_1__D_2__D_3_tempShape( 4 );
Z_abij__D_0__D_1__D_2__D_3_tempShape[ 0 ] = n_v;
Z_abij__D_0__D_1__D_2__D_3_tempShape[ 1 ] = n_v;
Z_abij__D_0__D_1__D_2__D_3_tempShape[ 2 ] = n_o;
Z_abij__D_0__D_1__D_2__D_3_tempShape[ 3 ] = n_o;
Z_abij__D_0__D_1__D_2__D_3.ResizeTo( Z_abij__D_0__D_1__D_2__D_3_tempShape );
MakeUniform( Z_abij__D_0__D_1__D_2__D_3 );
// r_bmfe has 4 dims
ObjShape r_bmfe__D_0__D_1__D_2__D_3_tempShape( 4 );
r_bmfe__D_0__D_1__D_2__D_3_tempShape[ 0 ] = n_v;
r_bmfe__D_0__D_1__D_2__D_3_tempShape[ 1 ] = n_o;
r_bmfe__D_0__D_1__D_2__D_3_tempShape[ 2 ] = n_v;
r_bmfe__D_0__D_1__D_2__D_3_tempShape[ 3 ] = n_v;
r_bmfe__D_0__D_1__D_2__D_3.ResizeTo( r_bmfe__D_0__D_1__D_2__D_3_tempShape );
MakeUniform( r_bmfe__D_0__D_1__D_2__D_3 );
// t_fj has 2 dims
//	Starting distribution: [D01,D23] or _D_0_1__D_2_3
ObjShape t_fj__D_0_1__D_2_3_tempShape( 2 );
t_fj__D_0_1__D_2_3_tempShape[ 0 ] = n_v;
t_fj__D_0_1__D_2_3_tempShape[ 1 ] = n_o;
t_fj__D_0_1__D_2_3.ResizeTo( t_fj__D_0_1__D_2_3_tempShape );
MakeUniform( t_fj__D_0_1__D_2_3 );
// P_jimb has 4 dims
ObjShape P_jimb__D_0__D_1__D_2__D_3_tempShape( 4 );
P_jimb__D_0__D_1__D_2__D_3_tempShape[ 0 ] = n_o;
P_jimb__D_0__D_1__D_2__D_3_tempShape[ 1 ] = n_o;
P_jimb__D_0__D_1__D_2__D_3_tempShape[ 2 ] = n_o;
P_jimb__D_0__D_1__D_2__D_3_tempShape[ 3 ] = n_v;
P_jimb__D_0__D_1__D_2__D_3.ResizeTo( P_jimb__D_0__D_1__D_2__D_3_tempShape );
MakeUniform( P_jimb__D_0__D_1__D_2__D_3 );
// F_ae has 2 dims
ObjShape F_ae__D_0_1__D_2_3_tempShape( 2 );
F_ae__D_0_1__D_2_3_tempShape[ 0 ] = n_v;
F_ae__D_0_1__D_2_3_tempShape[ 1 ] = n_v;
F_ae__D_0_1__D_2_3.ResizeTo( F_ae__D_0_1__D_2_3_tempShape );
MakeUniform( F_ae__D_0_1__D_2_3 );
// T_bfnj has 4 dims
ObjShape T_bfnj__D_0__D_1__D_2__D_3_tempShape( 4 );
T_bfnj__D_0__D_1__D_2__D_3_tempShape[ 0 ] = n_v;
T_bfnj__D_0__D_1__D_2__D_3_tempShape[ 1 ] = n_v;
T_bfnj__D_0__D_1__D_2__D_3_tempShape[ 2 ] = n_o;
T_bfnj__D_0__D_1__D_2__D_3_tempShape[ 3 ] = n_o;
T_bfnj__D_0__D_1__D_2__D_3.ResizeTo( T_bfnj__D_0__D_1__D_2__D_3_tempShape );
MakeUniform( T_bfnj__D_0__D_1__D_2__D_3 );
// G_mi has 2 dims
ObjShape G_mi__D_0_1__D_2_3_tempShape( 2 );
G_mi__D_0_1__D_2_3_tempShape[ 0 ] = n_o;
G_mi__D_0_1__D_2_3_tempShape[ 1 ] = n_o;
G_mi__D_0_1__D_2_3.ResizeTo( G_mi__D_0_1__D_2_3_tempShape );
MakeUniform( G_mi__D_0_1__D_2_3 );
// W_bmje has 4 dims
ObjShape W_bmje__D_0__D_1__D_2__D_3_tempShape( 4 );
W_bmje__D_0__D_1__D_2__D_3_tempShape[ 0 ] = n_v;
W_bmje__D_0__D_1__D_2__D_3_tempShape[ 1 ] = n_o;
W_bmje__D_0__D_1__D_2__D_3_tempShape[ 2 ] = n_o;
W_bmje__D_0__D_1__D_2__D_3_tempShape[ 3 ] = n_v;
W_bmje__D_0__D_1__D_2__D_3.ResizeTo( W_bmje__D_0__D_1__D_2__D_3_tempShape );
MakeUniform( W_bmje__D_0__D_1__D_2__D_3 );
// X_bmej has 4 dims
ObjShape X_bmej__D_0__D_1__D_2__D_3_tempShape( 4 );
X_bmej__D_0__D_1__D_2__D_3_tempShape[ 0 ] = n_v;
X_bmej__D_0__D_1__D_2__D_3_tempShape[ 1 ] = n_o;
X_bmej__D_0__D_1__D_2__D_3_tempShape[ 2 ] = n_v;
X_bmej__D_0__D_1__D_2__D_3_tempShape[ 3 ] = n_o;
X_bmej__D_0__D_1__D_2__D_3.ResizeTo( X_bmej__D_0__D_1__D_2__D_3_tempShape );
MakeUniform( X_bmej__D_0__D_1__D_2__D_3 );
//END_DECL

//******************************
//* Load tensors
//******************************
////////////////////////////////
//Performance testing
////////////////////////////////
std::stringstream fullName;
#ifdef CORRECTNESS
DistTensor<T> check_Z(dist__D_0__D_1__D_2__D_3, g);
check_Z.ResizeTo(Z_abij__D_0__D_1__D_2__D_3.Shape());
Read(v_femn__D_0__D_1__D_2__D_3, "ccsd_terms/term_v_small", BINARY_FLAT, false);
Read(y_abef__D_0__D_1__D_2__D_3, "ccsd_terms/term_y_small", BINARY_FLAT, false);
Read(r_bmfe__D_0__D_1__D_2__D_3, "ccsd_terms/term_r_small", BINARY_FLAT, false);
fullName.str("");
fullName.clear();
fullName << "ccsd_terms/term_Q_iter" << testIter;
Read(Q_mnij__D_0__D_1__D_2__D_3, fullName.str(), BINARY_FLAT, false);
fullName.str("");
fullName.clear();
fullName << "ccsd_terms/term_W_iter" << testIter;
Read(W_bmje__D_0__D_1__D_2__D_3, fullName.str(), BINARY_FLAT, false);
fullName.str("");
fullName.clear();
fullName << "ccsd_terms/term_P_iter" << testIter;
Read(P_jimb__D_0__D_1__D_2__D_3, fullName.str(), BINARY_FLAT, false);
fullName.str("");
fullName.clear();
fullName << "ccsd_terms/term_X_iter" << testIter;
Read(X_bmej__D_0__D_1__D_2__D_3, fullName.str(), BINARY_FLAT, false);
fullName.str("");
fullName.clear();
fullName << "ccsd_terms/term_F_iter" << testIter;
Read(F_ae__D_0_1__D_2_3, fullName.str(), BINARY_FLAT, false);
fullName.str("");
fullName.clear();
fullName << "ccsd_terms/term_G_iter" << testIter;
Read(G_mi__D_0_1__D_2_3, fullName.str(), BINARY_FLAT, false);
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
fullName << "ccsd_terms/term_Z_iter" << testIter;
Read(check_Z, fullName.str(), BINARY_FLAT, false);
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
//**** (out of 1)
//**** Is real	0 shadows
	//Outputs:
	//  Z_abij__D_0__D_1__D_2__D_3

	tempShape = T_bfnj__D_0__D_1__D_2__D_3.Shape();
	Ztemp2__D_0__D_1__D_2__D_3.ResizeTo( tempShape );
	//**** (out of 1)
	//**** Is real	0 shadows
		//Outputs:
		//  Ztemp2__D_0__D_1__D_2__D_3
	PartitionDown(T_bfnj__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part2T__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part2B__D_0__D_1__D_2__D_3, 2, 0);
	PartitionDown(T_bfnj__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part3T__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part3B__D_0__D_1__D_2__D_3, 3, 0);
	PartitionDown(Ztemp2__D_0__D_1__D_2__D_3, Ztemp2_lvl1_part2T__D_0__D_1__D_2__D_3, Ztemp2_lvl1_part2B__D_0__D_1__D_2__D_3, 2, 0);
	while(Ztemp2_lvl1_part2T__D_0__D_1__D_2__D_3.Dimension(2) < Ztemp2__D_0__D_1__D_2__D_3.Dimension(2))
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
		( Ztemp2_lvl1_part2T__D_0__D_1__D_2__D_3,  Ztemp2_lvl1_part2_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       Ztemp2_lvl1_part2_1__D_0__D_1__D_2__D_3,
		  Ztemp2_lvl1_part2B__D_0__D_1__D_2__D_3, Ztemp2_lvl1_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );

		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  Ztemp2_lvl1_part2_1__D_0__D_1__D_2__D_3
		PartitionDown(T_bfnj_lvl1_part2_1__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, 3, 0);
		PartitionDown(T_bfnj_lvl1_part3_1__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part3_1_lvl2_part2T__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part3_1_lvl2_part2B__D_0__D_1__D_2__D_3, 2, 0);
		PartitionDown(Ztemp2_lvl1_part2_1__D_0__D_1__D_2__D_3, Ztemp2_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3, Ztemp2_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, 3, 0);
		while(Ztemp2_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3.Dimension(3) < Ztemp2_lvl1_part2_1__D_0__D_1__D_2__D_3.Dimension(3))
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
			( Ztemp2_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3,  Ztemp2_lvl1_part2_1_lvl2_part3_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       Ztemp2_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3,
			  Ztemp2_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, Ztemp2_lvl1_part2_1_lvl2_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );

			   // T_bfnj_lvl1_part3_1_lvl2_part2_1[D0,D1,D3,D2] <- T_bfnj_lvl1_part3_1_lvl2_part2_1[D0,D1,D2,D3]
			T_bfnj_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_3__D_2.AlignModesWith( modes_0_1_2_3, T_bfnj_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3, modes_0_1_3_2 );
			T_bfnj_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_3__D_2.AllToAllRedistFrom( T_bfnj_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_2__D_3, modes_2_3 );
if(commRank == 0){
flops += 3*prod(T_bfnj_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3.Shape());
}
			YAxpPx( 2.0, T_bfnj_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3, -1.0, T_bfnj_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_3__D_2, perm_0_1_3_2, Ztemp2_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3 );
			T_bfnj_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_3__D_2.EmptyData();

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
			( Ztemp2_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3,  Ztemp2_lvl1_part2_1_lvl2_part3_0__D_0__D_1__D_2__D_3,
			       Ztemp2_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  Ztemp2_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, Ztemp2_lvl1_part2_1_lvl2_part3_2__D_0__D_1__D_2__D_3, 3 );

		}
		//****

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
		( Ztemp2_lvl1_part2T__D_0__D_1__D_2__D_3,  Ztemp2_lvl1_part2_0__D_0__D_1__D_2__D_3,
		       Ztemp2_lvl1_part2_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  Ztemp2_lvl1_part2B__D_0__D_1__D_2__D_3, Ztemp2_lvl1_part2_2__D_0__D_1__D_2__D_3, 2 );

	}
	//****
	tempShape = Z_abij__D_0__D_1__D_2__D_3.Shape();
	Ztemp1__D_0__D_1__D_2__D_3.ResizeTo( tempShape );
	tempShape = Ztemp1__D_0__D_1__D_2__D_3.Shape();
	Ztemp1_perm1302__D_1__D_3__D_0__D_2.ResizeTo( tempShape );
if(commRank == 0){
flops += prod(Ztemp1_perm1302__D_1__D_3__D_0__D_2.Shape());
}
	Scal( 0.0, Ztemp1_perm1302__D_1__D_3__D_0__D_2 );
	//**** (out of 1)
	//**** Is real	0 shadows
		//Outputs:
		//  Ztemp1_perm1302__D_1__D_3__D_0__D_2
	PartitionDown(X_bmej__D_0__D_1__D_2__D_3, X_bmej_lvl1_part2T__D_0__D_1__D_2__D_3, X_bmej_lvl1_part2B__D_0__D_1__D_2__D_3, 2, 0);
	PartitionDown(T_bfnj__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part1T__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part1B__D_0__D_1__D_2__D_3, 1, 0);
	while(X_bmej_lvl1_part2T__D_0__D_1__D_2__D_3.Dimension(2) < X_bmej__D_0__D_1__D_2__D_3.Dimension(2))
	{
		RepartitionDown
		( X_bmej_lvl1_part2T__D_0__D_1__D_2__D_3,  X_bmej_lvl1_part2_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       X_bmej_lvl1_part2_1__D_0__D_1__D_2__D_3,
		  X_bmej_lvl1_part2B__D_0__D_1__D_2__D_3, X_bmej_lvl1_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );
		RepartitionDown
		( T_bfnj_lvl1_part1T__D_0__D_1__D_2__D_3,  T_bfnj_lvl1_part1_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       T_bfnj_lvl1_part1_1__D_0__D_1__D_2__D_3,
		  T_bfnj_lvl1_part1B__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part1_2__D_0__D_1__D_2__D_3, 1, blkSize );

		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  Ztemp1_perm1302__D_1__D_3__D_0__D_2
		PartitionDown(X_bmej_lvl1_part2_1__D_0__D_1__D_2__D_3, X_bmej_lvl1_part2_1_lvl2_part1T__D_0__D_1__D_2__D_3, X_bmej_lvl1_part2_1_lvl2_part1B__D_0__D_1__D_2__D_3, 1, 0);
		PartitionDown(T_bfnj_lvl1_part1_1__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part1_1_lvl2_part2T__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part1_1_lvl2_part2B__D_0__D_1__D_2__D_3, 2, 0);
		while(X_bmej_lvl1_part2_1_lvl2_part1T__D_0__D_1__D_2__D_3.Dimension(1) < X_bmej_lvl1_part2_1__D_0__D_1__D_2__D_3.Dimension(1))
		{
			RepartitionDown
			( X_bmej_lvl1_part2_1_lvl2_part1T__D_0__D_1__D_2__D_3,  X_bmej_lvl1_part2_1_lvl2_part1_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       X_bmej_lvl1_part2_1_lvl2_part1_1__D_0__D_1__D_2__D_3,
			  X_bmej_lvl1_part2_1_lvl2_part1B__D_0__D_1__D_2__D_3, X_bmej_lvl1_part2_1_lvl2_part1_2__D_0__D_1__D_2__D_3, 1, blkSize );
			RepartitionDown
			( T_bfnj_lvl1_part1_1_lvl2_part2T__D_0__D_1__D_2__D_3,  T_bfnj_lvl1_part1_1_lvl2_part2_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       T_bfnj_lvl1_part1_1_lvl2_part2_1__D_0__D_1__D_2__D_3,
			  T_bfnj_lvl1_part1_1_lvl2_part2B__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part1_1_lvl2_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );

			   // X_bmej_lvl1_part2_1_lvl2_part1_1[D1,*,D2,D3] <- X_bmej_lvl1_part2_1_lvl2_part1_1[D0,D1,D2,D3]
			X_bmej_lvl1_part2_1_lvl2_part1_1__D_1__S__D_2__D_3.AlignModesWith( modes_0_3, Ztemp1__D_0__D_1__D_2__D_3, modes_1_3 );
			X_bmej_lvl1_part2_1_lvl2_part1_1__D_1__S__D_2__D_3.AllToAllRedistFrom( X_bmej_lvl1_part2_1_lvl2_part1_1__D_0__D_1__D_2__D_3, modes_0_1 );
			   // X_bmej_lvl1_part2_1_lvl2_part1_1[D1,*,*,D3] <- X_bmej_lvl1_part2_1_lvl2_part1_1[D1,*,D2,D3]
			X_bmej_lvl1_part2_1_lvl2_part1_1_perm0312__D_1__D_3__S__S.AlignModesWith( modes_0_3, Ztemp1__D_0__D_1__D_2__D_3, modes_1_3 );
			X_bmej_lvl1_part2_1_lvl2_part1_1_perm0312__D_1__D_3__S__S.AllGatherRedistFrom( X_bmej_lvl1_part2_1_lvl2_part1_1__D_1__S__D_2__D_3, modes_2 );
			X_bmej_lvl1_part2_1_lvl2_part1_1__D_1__S__D_2__D_3.EmptyData();
			   // T_bfnj_lvl1_part1_1_lvl2_part2_1[D0,D1,*,D2] <- T_bfnj_lvl1_part1_1_lvl2_part2_1[D0,D1,D2,D3]
			T_bfnj_lvl1_part1_1_lvl2_part2_1__D_0__D_1__S__D_2.AlignModesWith( modes_0_3, Ztemp1__D_0__D_1__D_2__D_3, modes_0_2 );
			T_bfnj_lvl1_part1_1_lvl2_part2_1__D_0__D_1__S__D_2.AllToAllRedistFrom( T_bfnj_lvl1_part1_1_lvl2_part2_1__D_0__D_1__D_2__D_3, modes_2_3 );
			   // T_bfnj_lvl1_part1_1_lvl2_part2_1[D0,*,*,D2] <- T_bfnj_lvl1_part1_1_lvl2_part2_1[D0,D1,*,D2]
			T_bfnj_lvl1_part1_1_lvl2_part2_1_perm2103__S__S__D_0__D_2.AlignModesWith( modes_0_3, Ztemp1__D_0__D_1__D_2__D_3, modes_0_2 );
			T_bfnj_lvl1_part1_1_lvl2_part2_1_perm2103__S__S__D_0__D_2.AllGatherRedistFrom( T_bfnj_lvl1_part1_1_lvl2_part2_1__D_0__D_1__S__D_2, modes_1 );
			T_bfnj_lvl1_part1_1_lvl2_part2_1__D_0__D_1__S__D_2.EmptyData();
			   // 1.0 * X_bmej_lvl1_part2_1_lvl2_part1_1[D1,*,*,D3]_bjme * T_bfnj_lvl1_part1_1_lvl2_part2_1[D0,*,*,D2]_meai + 1.0 * Ztemp1[D0,D1,D2,D3]_bjai
if(commRank == 0){
flops += 2*1*X_bmej_lvl1_part2_1_lvl2_part1_1_perm0312__D_1__D_3__S__S.Dimension(1)*X_bmej_lvl1_part2_1_lvl2_part1_1_perm0312__D_1__D_3__S__S.Dimension(2)*1*X_bmej_lvl1_part2_1_lvl2_part1_1_perm0312__D_1__D_3__S__S.Dimension(0)*X_bmej_lvl1_part2_1_lvl2_part1_1_perm0312__D_1__D_3__S__S.Dimension(3)*1*T_bfnj_lvl1_part1_1_lvl2_part2_1_perm2103__S__S__D_0__D_2.Dimension(0)*T_bfnj_lvl1_part1_1_lvl2_part2_1_perm2103__S__S__D_0__D_2.Dimension(3);
}
			LocalContractAndLocalEliminate(1.0, X_bmej_lvl1_part2_1_lvl2_part1_1_perm0312__D_1__D_3__S__S.LockedTensor(), indices_bjme, false,
				T_bfnj_lvl1_part1_1_lvl2_part2_1_perm2103__S__S__D_0__D_2.LockedTensor(), indices_meai, false,
				1.0, Ztemp1_perm1302__D_1__D_3__D_0__D_2.Tensor(), indices_bjai, false);
			X_bmej_lvl1_part2_1_lvl2_part1_1_perm0312__D_1__D_3__S__S.EmptyData();
			T_bfnj_lvl1_part1_1_lvl2_part2_1_perm2103__S__S__D_0__D_2.EmptyData();

			SlidePartitionDown
			( X_bmej_lvl1_part2_1_lvl2_part1T__D_0__D_1__D_2__D_3,  X_bmej_lvl1_part2_1_lvl2_part1_0__D_0__D_1__D_2__D_3,
			       X_bmej_lvl1_part2_1_lvl2_part1_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  X_bmej_lvl1_part2_1_lvl2_part1B__D_0__D_1__D_2__D_3, X_bmej_lvl1_part2_1_lvl2_part1_2__D_0__D_1__D_2__D_3, 1 );
			SlidePartitionDown
			( T_bfnj_lvl1_part1_1_lvl2_part2T__D_0__D_1__D_2__D_3,  T_bfnj_lvl1_part1_1_lvl2_part2_0__D_0__D_1__D_2__D_3,
			       T_bfnj_lvl1_part1_1_lvl2_part2_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  T_bfnj_lvl1_part1_1_lvl2_part2B__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part1_1_lvl2_part2_2__D_0__D_1__D_2__D_3, 2 );

		}
		//****

		SlidePartitionDown
		( X_bmej_lvl1_part2T__D_0__D_1__D_2__D_3,  X_bmej_lvl1_part2_0__D_0__D_1__D_2__D_3,
		       X_bmej_lvl1_part2_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  X_bmej_lvl1_part2B__D_0__D_1__D_2__D_3, X_bmej_lvl1_part2_2__D_0__D_1__D_2__D_3, 2 );
		SlidePartitionDown
		( T_bfnj_lvl1_part1T__D_0__D_1__D_2__D_3,  T_bfnj_lvl1_part1_0__D_0__D_1__D_2__D_3,
		       T_bfnj_lvl1_part1_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  T_bfnj_lvl1_part1B__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part1_2__D_0__D_1__D_2__D_3, 1 );

	}
	//****
	Permute( Ztemp1_perm1302__D_1__D_3__D_0__D_2, Ztemp1__D_0__D_1__D_2__D_3 );
	Ztemp1_perm1302__D_1__D_3__D_0__D_2.EmptyData();
	tempShape = Z_abij__D_0__D_1__D_2__D_3.Shape();
	Zaccum__D_0__D_1__D_2__D_3.ResizeTo( tempShape );
	//**** (out of 1)
	//**** Is real	0 shadows
		//Outputs:
		//  Zaccum__D_0__D_1__D_2__D_3
	PartitionDown(Ztemp1__D_0__D_1__D_2__D_3, Ztemp1_lvl1_part2T__D_0__D_1__D_2__D_3, Ztemp1_lvl1_part2B__D_0__D_1__D_2__D_3, 2, 0);
	PartitionDown(Ztemp1__D_0__D_1__D_2__D_3, Ztemp1_lvl1_part3T__D_0__D_1__D_2__D_3, Ztemp1_lvl1_part3B__D_0__D_1__D_2__D_3, 3, 0);
	PartitionDown(Zaccum__D_0__D_1__D_2__D_3, Zaccum_lvl1_part2T__D_0__D_1__D_2__D_3, Zaccum_lvl1_part2B__D_0__D_1__D_2__D_3, 2, 0);
	while(Zaccum_lvl1_part2T__D_0__D_1__D_2__D_3.Dimension(2) < Zaccum__D_0__D_1__D_2__D_3.Dimension(2))
	{
		RepartitionDown
		( Ztemp1_lvl1_part2T__D_0__D_1__D_2__D_3,  Ztemp1_lvl1_part2_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       Ztemp1_lvl1_part2_1__D_0__D_1__D_2__D_3,
		  Ztemp1_lvl1_part2B__D_0__D_1__D_2__D_3, Ztemp1_lvl1_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );
		RepartitionDown
		( Ztemp1_lvl1_part3T__D_0__D_1__D_2__D_3,  Ztemp1_lvl1_part3_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       Ztemp1_lvl1_part3_1__D_0__D_1__D_2__D_3,
		  Ztemp1_lvl1_part3B__D_0__D_1__D_2__D_3, Ztemp1_lvl1_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );
		RepartitionDown
		( Zaccum_lvl1_part2T__D_0__D_1__D_2__D_3,  Zaccum_lvl1_part2_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       Zaccum_lvl1_part2_1__D_0__D_1__D_2__D_3,
		  Zaccum_lvl1_part2B__D_0__D_1__D_2__D_3, Zaccum_lvl1_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );

		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  Zaccum_lvl1_part2_1__D_0__D_1__D_2__D_3
		PartitionDown(Ztemp1_lvl1_part2_1__D_0__D_1__D_2__D_3, Ztemp1_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3, Ztemp1_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, 3, 0);
		PartitionDown(Ztemp1_lvl1_part3_1__D_0__D_1__D_2__D_3, Ztemp1_lvl1_part3_1_lvl2_part2T__D_0__D_1__D_2__D_3, Ztemp1_lvl1_part3_1_lvl2_part2B__D_0__D_1__D_2__D_3, 2, 0);
		PartitionDown(Zaccum_lvl1_part2_1__D_0__D_1__D_2__D_3, Zaccum_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3, Zaccum_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, 3, 0);
		while(Zaccum_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3.Dimension(3) < Zaccum_lvl1_part2_1__D_0__D_1__D_2__D_3.Dimension(3))
		{
			RepartitionDown
			( Ztemp1_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3,  Ztemp1_lvl1_part2_1_lvl2_part3_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       Ztemp1_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3,
			  Ztemp1_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, Ztemp1_lvl1_part2_1_lvl2_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );
			RepartitionDown
			( Ztemp1_lvl1_part3_1_lvl2_part2T__D_0__D_1__D_2__D_3,  Ztemp1_lvl1_part3_1_lvl2_part2_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       Ztemp1_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_2__D_3,
			  Ztemp1_lvl1_part3_1_lvl2_part2B__D_0__D_1__D_2__D_3, Ztemp1_lvl1_part3_1_lvl2_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );
			RepartitionDown
			( Zaccum_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3,  Zaccum_lvl1_part2_1_lvl2_part3_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       Zaccum_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3,
			  Zaccum_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, Zaccum_lvl1_part2_1_lvl2_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );

			   // Ztemp1_lvl1_part3_1_lvl2_part2_1[D0,D1,D3,D2] <- Ztemp1_lvl1_part3_1_lvl2_part2_1[D0,D1,D2,D3]
			Ztemp1_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_3__D_2.AlignModesWith( modes_0_1_2_3, Ztemp1_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3, modes_0_1_3_2 );
			Ztemp1_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_3__D_2.AllToAllRedistFrom( Ztemp1_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_2__D_3, modes_2_3 );
if(commRank == 0){
flops += 3*prod(Ztemp1_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3.Shape());
}
			YAxpPx( -0.5, Ztemp1_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3, -1.0, Ztemp1_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_3__D_2, perm_0_1_3_2, Zaccum_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3 );
			Ztemp1_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_3__D_2.EmptyData();

			SlidePartitionDown
			( Ztemp1_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3,  Ztemp1_lvl1_part2_1_lvl2_part3_0__D_0__D_1__D_2__D_3,
			       Ztemp1_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  Ztemp1_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, Ztemp1_lvl1_part2_1_lvl2_part3_2__D_0__D_1__D_2__D_3, 3 );
			SlidePartitionDown
			( Ztemp1_lvl1_part3_1_lvl2_part2T__D_0__D_1__D_2__D_3,  Ztemp1_lvl1_part3_1_lvl2_part2_0__D_0__D_1__D_2__D_3,
			       Ztemp1_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  Ztemp1_lvl1_part3_1_lvl2_part2B__D_0__D_1__D_2__D_3, Ztemp1_lvl1_part3_1_lvl2_part2_2__D_0__D_1__D_2__D_3, 2 );
			SlidePartitionDown
			( Zaccum_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3,  Zaccum_lvl1_part2_1_lvl2_part3_0__D_0__D_1__D_2__D_3,
			       Zaccum_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  Zaccum_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, Zaccum_lvl1_part2_1_lvl2_part3_2__D_0__D_1__D_2__D_3, 3 );

		}
		//****

		SlidePartitionDown
		( Ztemp1_lvl1_part2T__D_0__D_1__D_2__D_3,  Ztemp1_lvl1_part2_0__D_0__D_1__D_2__D_3,
		       Ztemp1_lvl1_part2_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  Ztemp1_lvl1_part2B__D_0__D_1__D_2__D_3, Ztemp1_lvl1_part2_2__D_0__D_1__D_2__D_3, 2 );
		SlidePartitionDown
		( Ztemp1_lvl1_part3T__D_0__D_1__D_2__D_3,  Ztemp1_lvl1_part3_0__D_0__D_1__D_2__D_3,
		       Ztemp1_lvl1_part3_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  Ztemp1_lvl1_part3B__D_0__D_1__D_2__D_3, Ztemp1_lvl1_part3_2__D_0__D_1__D_2__D_3, 3 );
		SlidePartitionDown
		( Zaccum_lvl1_part2T__D_0__D_1__D_2__D_3,  Zaccum_lvl1_part2_0__D_0__D_1__D_2__D_3,
		       Zaccum_lvl1_part2_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  Zaccum_lvl1_part2B__D_0__D_1__D_2__D_3, Zaccum_lvl1_part2_2__D_0__D_1__D_2__D_3, 2 );

	}
	//****
	Ztemp1__D_0__D_1__D_2__D_3.EmptyData();
	Ztemp1__D_0__D_1__D_2__D_3.EmptyData();
	Permute( Zaccum__D_0__D_1__D_2__D_3, Zaccum_perm1302__D_1__D_3__D_0__D_2 );
	//**** (out of 1)
	//**** Is real	0 shadows
		//Outputs:
		//  Zaccum_perm1302__D_1__D_3__D_0__D_2
	PartitionDown(W_bmje__D_0__D_1__D_2__D_3, W_bmje_lvl1_part3T__D_0__D_1__D_2__D_3, W_bmje_lvl1_part3B__D_0__D_1__D_2__D_3, 3, 0);
	PartitionDown(Ztemp2__D_0__D_1__D_2__D_3, Ztemp2_lvl1_part1T__D_0__D_1__D_2__D_3, Ztemp2_lvl1_part1B__D_0__D_1__D_2__D_3, 1, 0);
	while(W_bmje_lvl1_part3T__D_0__D_1__D_2__D_3.Dimension(3) < W_bmje__D_0__D_1__D_2__D_3.Dimension(3))
	{
		RepartitionDown
		( W_bmje_lvl1_part3T__D_0__D_1__D_2__D_3,  W_bmje_lvl1_part3_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       W_bmje_lvl1_part3_1__D_0__D_1__D_2__D_3,
		  W_bmje_lvl1_part3B__D_0__D_1__D_2__D_3, W_bmje_lvl1_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );
		RepartitionDown
		( Ztemp2_lvl1_part1T__D_0__D_1__D_2__D_3,  Ztemp2_lvl1_part1_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       Ztemp2_lvl1_part1_1__D_0__D_1__D_2__D_3,
		  Ztemp2_lvl1_part1B__D_0__D_1__D_2__D_3, Ztemp2_lvl1_part1_2__D_0__D_1__D_2__D_3, 1, blkSize );

		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  Zaccum_perm1302__D_1__D_3__D_0__D_2
		PartitionDown(W_bmje_lvl1_part3_1__D_0__D_1__D_2__D_3, W_bmje_lvl1_part3_1_lvl2_part1T__D_0__D_1__D_2__D_3, W_bmje_lvl1_part3_1_lvl2_part1B__D_0__D_1__D_2__D_3, 1, 0);
		PartitionDown(Ztemp2_lvl1_part1_1__D_0__D_1__D_2__D_3, Ztemp2_lvl1_part1_1_lvl2_part3T__D_0__D_1__D_2__D_3, Ztemp2_lvl1_part1_1_lvl2_part3B__D_0__D_1__D_2__D_3, 3, 0);
		while(W_bmje_lvl1_part3_1_lvl2_part1T__D_0__D_1__D_2__D_3.Dimension(1) < W_bmje_lvl1_part3_1__D_0__D_1__D_2__D_3.Dimension(1))
		{
			RepartitionDown
			( W_bmje_lvl1_part3_1_lvl2_part1T__D_0__D_1__D_2__D_3,  W_bmje_lvl1_part3_1_lvl2_part1_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       W_bmje_lvl1_part3_1_lvl2_part1_1__D_0__D_1__D_2__D_3,
			  W_bmje_lvl1_part3_1_lvl2_part1B__D_0__D_1__D_2__D_3, W_bmje_lvl1_part3_1_lvl2_part1_2__D_0__D_1__D_2__D_3, 1, blkSize );
			RepartitionDown
			( Ztemp2_lvl1_part1_1_lvl2_part3T__D_0__D_1__D_2__D_3,  Ztemp2_lvl1_part1_1_lvl2_part3_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       Ztemp2_lvl1_part1_1_lvl2_part3_1__D_0__D_1__D_2__D_3,
			  Ztemp2_lvl1_part1_1_lvl2_part3B__D_0__D_1__D_2__D_3, Ztemp2_lvl1_part1_1_lvl2_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );

			   // W_bmje_lvl1_part3_1_lvl2_part1_1[D1,D0,D2,D3] <- W_bmje_lvl1_part3_1_lvl2_part1_1[D0,D1,D2,D3]
			W_bmje_lvl1_part3_1_lvl2_part1_1__D_1__D_0__D_2__D_3.AlignModesWith( modes_0_2, Zaccum__D_0__D_1__D_2__D_3, modes_1_3 );
			W_bmje_lvl1_part3_1_lvl2_part1_1__D_1__D_0__D_2__D_3.AllToAllRedistFrom( W_bmje_lvl1_part3_1_lvl2_part1_1__D_0__D_1__D_2__D_3, modes_0_1 );
			   // W_bmje_lvl1_part3_1_lvl2_part1_1[D1,D0,D3,*] <- W_bmje_lvl1_part3_1_lvl2_part1_1[D1,D0,D2,D3]
			W_bmje_lvl1_part3_1_lvl2_part1_1__D_1__D_0__D_3__S.AlignModesWith( modes_0_2, Zaccum__D_0__D_1__D_2__D_3, modes_1_3 );
			W_bmje_lvl1_part3_1_lvl2_part1_1__D_1__D_0__D_3__S.AllToAllRedistFrom( W_bmje_lvl1_part3_1_lvl2_part1_1__D_1__D_0__D_2__D_3, modes_2_3 );
			W_bmje_lvl1_part3_1_lvl2_part1_1__D_1__D_0__D_2__D_3.EmptyData();
			   // W_bmje_lvl1_part3_1_lvl2_part1_1[D1,*,D3,*] <- W_bmje_lvl1_part3_1_lvl2_part1_1[D1,D0,D3,*]
			W_bmje_lvl1_part3_1_lvl2_part1_1_perm0213__D_1__D_3__S__S.AlignModesWith( modes_0_2, Zaccum__D_0__D_1__D_2__D_3, modes_1_3 );
			W_bmje_lvl1_part3_1_lvl2_part1_1_perm0213__D_1__D_3__S__S.AllGatherRedistFrom( W_bmje_lvl1_part3_1_lvl2_part1_1__D_1__D_0__D_3__S, modes_0 );
			W_bmje_lvl1_part3_1_lvl2_part1_1__D_1__D_0__D_3__S.EmptyData();
			   // Ztemp2_lvl1_part1_1_lvl2_part3_1[D0,*,D2,*] <- Ztemp2_lvl1_part1_1_lvl2_part3_1[D0,D1,D2,D3]
			Ztemp2_lvl1_part1_1_lvl2_part3_1_perm3102__S__S__D_0__D_2.AlignModesWith( modes_0_2, Zaccum__D_0__D_1__D_2__D_3, modes_0_2 );
			Ztemp2_lvl1_part1_1_lvl2_part3_1_perm3102__S__S__D_0__D_2.AllGatherRedistFrom( Ztemp2_lvl1_part1_1_lvl2_part3_1__D_0__D_1__D_2__D_3, modes_1_3 );
			   // 0.5 * W_bmje_lvl1_part3_1_lvl2_part1_1[D1,*,D3,*]_bjme * Ztemp2_lvl1_part1_1_lvl2_part3_1[D0,*,D2,*]_meai + 1.0 * Zaccum[D0,D1,D2,D3]_bjai
if(commRank == 0){
flops += 2*1*W_bmje_lvl1_part3_1_lvl2_part1_1_perm0213__D_1__D_3__S__S.Dimension(1)*W_bmje_lvl1_part3_1_lvl2_part1_1_perm0213__D_1__D_3__S__S.Dimension(3)*1*W_bmje_lvl1_part3_1_lvl2_part1_1_perm0213__D_1__D_3__S__S.Dimension(0)*W_bmje_lvl1_part3_1_lvl2_part1_1_perm0213__D_1__D_3__S__S.Dimension(2)*1*Ztemp2_lvl1_part1_1_lvl2_part3_1_perm3102__S__S__D_0__D_2.Dimension(0)*Ztemp2_lvl1_part1_1_lvl2_part3_1_perm3102__S__S__D_0__D_2.Dimension(2);
}
			LocalContractAndLocalEliminate(0.5, W_bmje_lvl1_part3_1_lvl2_part1_1_perm0213__D_1__D_3__S__S.LockedTensor(), indices_bjme, false,
				Ztemp2_lvl1_part1_1_lvl2_part3_1_perm3102__S__S__D_0__D_2.LockedTensor(), indices_meai, false,
				1.0, Zaccum_perm1302__D_1__D_3__D_0__D_2.Tensor(), indices_bjai, false);
			W_bmje_lvl1_part3_1_lvl2_part1_1_perm0213__D_1__D_3__S__S.EmptyData();
			Ztemp2_lvl1_part1_1_lvl2_part3_1_perm3102__S__S__D_0__D_2.EmptyData();

			SlidePartitionDown
			( W_bmje_lvl1_part3_1_lvl2_part1T__D_0__D_1__D_2__D_3,  W_bmje_lvl1_part3_1_lvl2_part1_0__D_0__D_1__D_2__D_3,
			       W_bmje_lvl1_part3_1_lvl2_part1_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  W_bmje_lvl1_part3_1_lvl2_part1B__D_0__D_1__D_2__D_3, W_bmje_lvl1_part3_1_lvl2_part1_2__D_0__D_1__D_2__D_3, 1 );
			SlidePartitionDown
			( Ztemp2_lvl1_part1_1_lvl2_part3T__D_0__D_1__D_2__D_3,  Ztemp2_lvl1_part1_1_lvl2_part3_0__D_0__D_1__D_2__D_3,
			       Ztemp2_lvl1_part1_1_lvl2_part3_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  Ztemp2_lvl1_part1_1_lvl2_part3B__D_0__D_1__D_2__D_3, Ztemp2_lvl1_part1_1_lvl2_part3_2__D_0__D_1__D_2__D_3, 3 );

		}
		//****

		SlidePartitionDown
		( W_bmje_lvl1_part3T__D_0__D_1__D_2__D_3,  W_bmje_lvl1_part3_0__D_0__D_1__D_2__D_3,
		       W_bmje_lvl1_part3_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  W_bmje_lvl1_part3B__D_0__D_1__D_2__D_3, W_bmje_lvl1_part3_2__D_0__D_1__D_2__D_3, 3 );
		SlidePartitionDown
		( Ztemp2_lvl1_part1T__D_0__D_1__D_2__D_3,  Ztemp2_lvl1_part1_0__D_0__D_1__D_2__D_3,
		       Ztemp2_lvl1_part1_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  Ztemp2_lvl1_part1B__D_0__D_1__D_2__D_3, Ztemp2_lvl1_part1_2__D_0__D_1__D_2__D_3, 1 );

	}
	//****
	Ztemp2__D_0__D_1__D_2__D_3.EmptyData();
	Permute( Zaccum_perm1302__D_1__D_3__D_0__D_2, Zaccum__D_0__D_1__D_2__D_3 );
	Zaccum_perm1302__D_1__D_3__D_0__D_2.EmptyData();
	//**** (out of 1)
	//**** Is real	0 shadows
		//Outputs:
		//  Zaccum__D_0__D_1__D_2__D_3
	PartitionDown(T_bfnj__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part3T__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part3B__D_0__D_1__D_2__D_3, 3, 0);
	PartitionDown(Zaccum__D_0__D_1__D_2__D_3, Zaccum_lvl1_part3T__D_0__D_1__D_2__D_3, Zaccum_lvl1_part3B__D_0__D_1__D_2__D_3, 3, 0);
	while(Zaccum_lvl1_part3T__D_0__D_1__D_2__D_3.Dimension(3) < Zaccum__D_0__D_1__D_2__D_3.Dimension(3))
	{
		RepartitionDown
		( T_bfnj_lvl1_part3T__D_0__D_1__D_2__D_3,  T_bfnj_lvl1_part3_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       T_bfnj_lvl1_part3_1__D_0__D_1__D_2__D_3,
		  T_bfnj_lvl1_part3B__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );
		RepartitionDown
		( Zaccum_lvl1_part3T__D_0__D_1__D_2__D_3,  Zaccum_lvl1_part3_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       Zaccum_lvl1_part3_1__D_0__D_1__D_2__D_3,
		  Zaccum_lvl1_part3B__D_0__D_1__D_2__D_3, Zaccum_lvl1_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );

		Permute( Zaccum_lvl1_part3_1__D_0__D_1__D_2__D_3, Zaccum_lvl1_part3_1_perm2013__D_2__D_0__D_1__D_3 );
		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  Zaccum_lvl1_part3_1_perm2013__D_2__D_0__D_1__D_3
		PartitionDown(G_mi__D_0_1__D_2_3, G_mi_lvl2_part0T__D_0_1__D_2_3, G_mi_lvl2_part0B__D_0_1__D_2_3, 0, 0);
		PartitionDown(T_bfnj_lvl1_part3_1__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part3_1_lvl2_part2T__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part3_1_lvl2_part2B__D_0__D_1__D_2__D_3, 2, 0);
		while(G_mi_lvl2_part0T__D_0_1__D_2_3.Dimension(0) < G_mi__D_0_1__D_2_3.Dimension(0))
		{
			RepartitionDown
			( G_mi_lvl2_part0T__D_0_1__D_2_3,  G_mi_lvl2_part0_0__D_0_1__D_2_3,
			  /**/ /**/
			       G_mi_lvl2_part0_1__D_0_1__D_2_3,
			  G_mi_lvl2_part0B__D_0_1__D_2_3, G_mi_lvl2_part0_2__D_0_1__D_2_3, 0, blkSize );
			RepartitionDown
			( T_bfnj_lvl1_part3_1_lvl2_part2T__D_0__D_1__D_2__D_3,  T_bfnj_lvl1_part3_1_lvl2_part2_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       T_bfnj_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_2__D_3,
			  T_bfnj_lvl1_part3_1_lvl2_part2B__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part3_1_lvl2_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );

			   // G_mi_lvl2_part0_1[*,D2] <- G_mi_lvl2_part0_1[D01,D23]
			G_mi_lvl2_part0_1_perm10__D_2__S.AlignModesWith( modes_1, Zaccum_lvl1_part3_1__D_0__D_1__D_2__D_3, modes_2 );
			G_mi_lvl2_part0_1_perm10__D_2__S.AllGatherRedistFrom( G_mi_lvl2_part0_1__D_0_1__D_2_3, modes_0_1_3 );
			   // T_bfnj_lvl1_part3_1_lvl2_part2_1[D0,D1,*,D3] <- T_bfnj_lvl1_part3_1_lvl2_part2_1[D0,D1,D2,D3]
			T_bfnj_lvl1_part3_1_lvl2_part2_1_perm2013__S__D_0__D_1__D_3.AlignModesWith( modes_0_1_3, Zaccum_lvl1_part3_1__D_0__D_1__D_2__D_3, modes_0_1_3 );
			T_bfnj_lvl1_part3_1_lvl2_part2_1_perm2013__S__D_0__D_1__D_3.AllGatherRedistFrom( T_bfnj_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_2__D_3, modes_2 );
			   // -1.0 * G_mi_lvl2_part0_1[*,D2]_im * T_bfnj_lvl1_part3_1_lvl2_part2_1[D0,D1,*,D3]_mabj + 1.0 * Zaccum_lvl1_part3_1[D0,D1,D2,D3]_iabj
if(commRank == 0){
flops += 2*1*G_mi_lvl2_part0_1_perm10__D_2__S.Dimension(0)*1*G_mi_lvl2_part0_1_perm10__D_2__S.Dimension(1)*1*T_bfnj_lvl1_part3_1_lvl2_part2_1_perm2013__S__D_0__D_1__D_3.Dimension(0)*T_bfnj_lvl1_part3_1_lvl2_part2_1_perm2013__S__D_0__D_1__D_3.Dimension(1)*T_bfnj_lvl1_part3_1_lvl2_part2_1_perm2013__S__D_0__D_1__D_3.Dimension(3);
}
			LocalContractAndLocalEliminate(-1.0, G_mi_lvl2_part0_1_perm10__D_2__S.LockedTensor(), indices_im, false,
				T_bfnj_lvl1_part3_1_lvl2_part2_1_perm2013__S__D_0__D_1__D_3.LockedTensor(), indices_mabj, false,
				1.0, Zaccum_lvl1_part3_1_perm2013__D_2__D_0__D_1__D_3.Tensor(), indices_iabj, false);
			G_mi_lvl2_part0_1_perm10__D_2__S.EmptyData();
			T_bfnj_lvl1_part3_1_lvl2_part2_1_perm2013__S__D_0__D_1__D_3.EmptyData();

			SlidePartitionDown
			( G_mi_lvl2_part0T__D_0_1__D_2_3,  G_mi_lvl2_part0_0__D_0_1__D_2_3,
			       G_mi_lvl2_part0_1__D_0_1__D_2_3,
			  /**/ /**/
			  G_mi_lvl2_part0B__D_0_1__D_2_3, G_mi_lvl2_part0_2__D_0_1__D_2_3, 0 );
			SlidePartitionDown
			( T_bfnj_lvl1_part3_1_lvl2_part2T__D_0__D_1__D_2__D_3,  T_bfnj_lvl1_part3_1_lvl2_part2_0__D_0__D_1__D_2__D_3,
			       T_bfnj_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  T_bfnj_lvl1_part3_1_lvl2_part2B__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part3_1_lvl2_part2_2__D_0__D_1__D_2__D_3, 2 );

		}
		//****
		Permute( Zaccum_lvl1_part3_1_perm2013__D_2__D_0__D_1__D_3, Zaccum_lvl1_part3_1__D_0__D_1__D_2__D_3 );
		Zaccum_lvl1_part3_1_perm2013__D_2__D_0__D_1__D_3.EmptyData();

		SlidePartitionDown
		( T_bfnj_lvl1_part3T__D_0__D_1__D_2__D_3,  T_bfnj_lvl1_part3_0__D_0__D_1__D_2__D_3,
		       T_bfnj_lvl1_part3_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  T_bfnj_lvl1_part3B__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part3_2__D_0__D_1__D_2__D_3, 3 );
		SlidePartitionDown
		( Zaccum_lvl1_part3T__D_0__D_1__D_2__D_3,  Zaccum_lvl1_part3_0__D_0__D_1__D_2__D_3,
		       Zaccum_lvl1_part3_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  Zaccum_lvl1_part3B__D_0__D_1__D_2__D_3, Zaccum_lvl1_part3_2__D_0__D_1__D_2__D_3, 3 );

	}
	//****
	//**** (out of 1)
	//**** Is real	0 shadows
		//Outputs:
		//  Zaccum__D_0__D_1__D_2__D_3
	PartitionDown(T_bfnj__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part2T__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part2B__D_0__D_1__D_2__D_3, 2, 0);
	PartitionDown(Zaccum__D_0__D_1__D_2__D_3, Zaccum_lvl1_part2T__D_0__D_1__D_2__D_3, Zaccum_lvl1_part2B__D_0__D_1__D_2__D_3, 2, 0);
	while(Zaccum_lvl1_part2T__D_0__D_1__D_2__D_3.Dimension(2) < Zaccum__D_0__D_1__D_2__D_3.Dimension(2))
	{
		RepartitionDown
		( T_bfnj_lvl1_part2T__D_0__D_1__D_2__D_3,  T_bfnj_lvl1_part2_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       T_bfnj_lvl1_part2_1__D_0__D_1__D_2__D_3,
		  T_bfnj_lvl1_part2B__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );
		RepartitionDown
		( Zaccum_lvl1_part2T__D_0__D_1__D_2__D_3,  Zaccum_lvl1_part2_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       Zaccum_lvl1_part2_1__D_0__D_1__D_2__D_3,
		  Zaccum_lvl1_part2B__D_0__D_1__D_2__D_3, Zaccum_lvl1_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );

		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  Zaccum_lvl1_part2_1__D_0__D_1__D_2__D_3
		PartitionDown(T_bfnj_lvl1_part2_1__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, 3, 0);
		PartitionDown(Zaccum_lvl1_part2_1__D_0__D_1__D_2__D_3, Zaccum_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3, Zaccum_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, 3, 0);
		while(Zaccum_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3.Dimension(3) < Zaccum_lvl1_part2_1__D_0__D_1__D_2__D_3.Dimension(3))
		{
			RepartitionDown
			( T_bfnj_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3,  T_bfnj_lvl1_part2_1_lvl2_part3_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       T_bfnj_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3,
			  T_bfnj_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part2_1_lvl2_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );
			RepartitionDown
			( Zaccum_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3,  Zaccum_lvl1_part2_1_lvl2_part3_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       Zaccum_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3,
			  Zaccum_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, Zaccum_lvl1_part2_1_lvl2_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );

			   // F_ae[D0,*] <- F_ae[D01,D23]
			F_ae__D_0__S.AlignModesWith( modes_0, Zaccum_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3, modes_0 );
			F_ae__D_0__S.AllGatherRedistFrom( F_ae__D_0_1__D_2_3, modes_1_2_3 );
			   // T_bfnj_lvl1_part2_1_lvl2_part3_1[*,D1,D2,D3] <- T_bfnj_lvl1_part2_1_lvl2_part3_1[D0,D1,D2,D3]
			T_bfnj_lvl1_part2_1_lvl2_part3_1__S__D_1__D_2__D_3.AlignModesWith( modes_1_2_3, Zaccum_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3, modes_1_2_3 );
			T_bfnj_lvl1_part2_1_lvl2_part3_1__S__D_1__D_2__D_3.AllGatherRedistFrom( T_bfnj_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3, modes_0 );
			   // 1.0 * F_ae[D0,*]_ae * T_bfnj_lvl1_part2_1_lvl2_part3_1[*,D1,D2,D3]_ebij + 1.0 * Zaccum_lvl1_part2_1_lvl2_part3_1[D0,D1,D2,D3]_abij
if(commRank == 0){
flops += 2*1*F_ae__D_0__S.Dimension(1)*1*F_ae__D_0__S.Dimension(0)*1*T_bfnj_lvl1_part2_1_lvl2_part3_1__S__D_1__D_2__D_3.Dimension(2)*T_bfnj_lvl1_part2_1_lvl2_part3_1__S__D_1__D_2__D_3.Dimension(1)*T_bfnj_lvl1_part2_1_lvl2_part3_1__S__D_1__D_2__D_3.Dimension(3);
}
			LocalContractAndLocalEliminate(1.0, F_ae__D_0__S.LockedTensor(), indices_ae, false,
				T_bfnj_lvl1_part2_1_lvl2_part3_1__S__D_1__D_2__D_3.LockedTensor(), indices_ebij, false,
				1.0, Zaccum_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3.Tensor(), indices_abij, false);
			F_ae__D_0__S.EmptyData();
			T_bfnj_lvl1_part2_1_lvl2_part3_1__S__D_1__D_2__D_3.EmptyData();

			SlidePartitionDown
			( T_bfnj_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3,  T_bfnj_lvl1_part2_1_lvl2_part3_0__D_0__D_1__D_2__D_3,
			       T_bfnj_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  T_bfnj_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part2_1_lvl2_part3_2__D_0__D_1__D_2__D_3, 3 );
			SlidePartitionDown
			( Zaccum_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3,  Zaccum_lvl1_part2_1_lvl2_part3_0__D_0__D_1__D_2__D_3,
			       Zaccum_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  Zaccum_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, Zaccum_lvl1_part2_1_lvl2_part3_2__D_0__D_1__D_2__D_3, 3 );

		}
		//****

		SlidePartitionDown
		( T_bfnj_lvl1_part2T__D_0__D_1__D_2__D_3,  T_bfnj_lvl1_part2_0__D_0__D_1__D_2__D_3,
		       T_bfnj_lvl1_part2_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  T_bfnj_lvl1_part2B__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part2_2__D_0__D_1__D_2__D_3, 2 );
		SlidePartitionDown
		( Zaccum_lvl1_part2T__D_0__D_1__D_2__D_3,  Zaccum_lvl1_part2_0__D_0__D_1__D_2__D_3,
		       Zaccum_lvl1_part2_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  Zaccum_lvl1_part2B__D_0__D_1__D_2__D_3, Zaccum_lvl1_part2_2__D_0__D_1__D_2__D_3, 2 );

	}
	//****
	//**** (out of 1)
	//**** Is real	0 shadows
		//Outputs:
		//  Zaccum__D_0__D_1__D_2__D_3
	PartitionDown(P_jimb__D_0__D_1__D_2__D_3, P_jimb_lvl1_part0T__D_0__D_1__D_2__D_3, P_jimb_lvl1_part0B__D_0__D_1__D_2__D_3, 0, 0);
	PartitionDown(Zaccum__D_0__D_1__D_2__D_3, Zaccum_lvl1_part2T__D_0__D_1__D_2__D_3, Zaccum_lvl1_part2B__D_0__D_1__D_2__D_3, 2, 0);
	while(Zaccum_lvl1_part2T__D_0__D_1__D_2__D_3.Dimension(2) < Zaccum__D_0__D_1__D_2__D_3.Dimension(2))
	{
		RepartitionDown
		( P_jimb_lvl1_part0T__D_0__D_1__D_2__D_3,  P_jimb_lvl1_part0_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       P_jimb_lvl1_part0_1__D_0__D_1__D_2__D_3,
		  P_jimb_lvl1_part0B__D_0__D_1__D_2__D_3, P_jimb_lvl1_part0_2__D_0__D_1__D_2__D_3, 0, blkSize );
		RepartitionDown
		( Zaccum_lvl1_part2T__D_0__D_1__D_2__D_3,  Zaccum_lvl1_part2_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       Zaccum_lvl1_part2_1__D_0__D_1__D_2__D_3,
		  Zaccum_lvl1_part2B__D_0__D_1__D_2__D_3, Zaccum_lvl1_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );

		Permute( Zaccum_lvl1_part2_1__D_0__D_1__D_2__D_3, Zaccum_lvl1_part2_1_perm2310__D_2__D_3__D_1__D_0 );
		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  Zaccum_lvl1_part2_1_perm2310__D_2__D_3__D_1__D_0
		PartitionDown(P_jimb_lvl1_part0_1__D_0__D_1__D_2__D_3, P_jimb_lvl1_part0_1_lvl2_part2T__D_0__D_1__D_2__D_3, P_jimb_lvl1_part0_1_lvl2_part2B__D_0__D_1__D_2__D_3, 2, 0);
		PartitionDown(t_fj__D_0_1__D_2_3, t_fj_lvl2_part1T__D_0_1__D_2_3, t_fj_lvl2_part1B__D_0_1__D_2_3, 1, 0);
		while(P_jimb_lvl1_part0_1_lvl2_part2T__D_0__D_1__D_2__D_3.Dimension(2) < P_jimb_lvl1_part0_1__D_0__D_1__D_2__D_3.Dimension(2))
		{
			RepartitionDown
			( P_jimb_lvl1_part0_1_lvl2_part2T__D_0__D_1__D_2__D_3,  P_jimb_lvl1_part0_1_lvl2_part2_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       P_jimb_lvl1_part0_1_lvl2_part2_1__D_0__D_1__D_2__D_3,
			  P_jimb_lvl1_part0_1_lvl2_part2B__D_0__D_1__D_2__D_3, P_jimb_lvl1_part0_1_lvl2_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );
			RepartitionDown
			( t_fj_lvl2_part1T__D_0_1__D_2_3,  t_fj_lvl2_part1_0__D_0_1__D_2_3,
			  /**/ /**/
			       t_fj_lvl2_part1_1__D_0_1__D_2_3,
			  t_fj_lvl2_part1B__D_0_1__D_2_3, t_fj_lvl2_part1_2__D_0_1__D_2_3, 1, blkSize );

			   // P_jimb_lvl1_part0_1_lvl2_part2_1[D0,D3,D2,D1] <- P_jimb_lvl1_part0_1_lvl2_part2_1[D0,D1,D2,D3]
			P_jimb_lvl1_part0_1_lvl2_part2_1__D_0__D_3__D_2__D_1.AlignModesWith( modes_0_1_3, Zaccum_lvl1_part2_1__D_0__D_1__D_2__D_3, modes_2_3_1 );
			P_jimb_lvl1_part0_1_lvl2_part2_1__D_0__D_3__D_2__D_1.AllToAllRedistFrom( P_jimb_lvl1_part0_1_lvl2_part2_1__D_0__D_1__D_2__D_3, modes_1_3 );
			   // P_jimb_lvl1_part0_1_lvl2_part2_1[D2,D3,*,D1] <- P_jimb_lvl1_part0_1_lvl2_part2_1[D0,D3,D2,D1]
			P_jimb_lvl1_part0_1_lvl2_part2_1_perm0132__D_2__D_3__D_1__S.AlignModesWith( modes_0_1_3, Zaccum_lvl1_part2_1__D_0__D_1__D_2__D_3, modes_2_3_1 );
			P_jimb_lvl1_part0_1_lvl2_part2_1_perm0132__D_2__D_3__D_1__S.AllToAllRedistFrom( P_jimb_lvl1_part0_1_lvl2_part2_1__D_0__D_3__D_2__D_1, modes_0_2 );
			P_jimb_lvl1_part0_1_lvl2_part2_1__D_0__D_3__D_2__D_1.EmptyData();
			   // t_fj_lvl2_part1_1[D0,*] <- t_fj_lvl2_part1_1[D01,D23]
			t_fj_lvl2_part1_1_perm10__S__D_0.AlignModesWith( modes_0, Zaccum_lvl1_part2_1__D_0__D_1__D_2__D_3, modes_0 );
			t_fj_lvl2_part1_1_perm10__S__D_0.AllGatherRedistFrom( t_fj_lvl2_part1_1__D_0_1__D_2_3, modes_1_2_3 );
			   // -1.0 * P_jimb_lvl1_part0_1_lvl2_part2_1[D2,D3,*,D1]_ijbm * t_fj_lvl2_part1_1[D0,*]_ma + 1.0 * Zaccum_lvl1_part2_1[D0,D1,D2,D3]_ijba
if(commRank == 0){
flops += 2*1*P_jimb_lvl1_part0_1_lvl2_part2_1_perm0132__D_2__D_3__D_1__S.Dimension(2)*1*P_jimb_lvl1_part0_1_lvl2_part2_1_perm0132__D_2__D_3__D_1__S.Dimension(0)*P_jimb_lvl1_part0_1_lvl2_part2_1_perm0132__D_2__D_3__D_1__S.Dimension(3)*P_jimb_lvl1_part0_1_lvl2_part2_1_perm0132__D_2__D_3__D_1__S.Dimension(1)*1*t_fj_lvl2_part1_1_perm10__S__D_0.Dimension(0);
}
			LocalContractAndLocalEliminate(-1.0, P_jimb_lvl1_part0_1_lvl2_part2_1_perm0132__D_2__D_3__D_1__S.LockedTensor(), indices_ijbm, false,
				t_fj_lvl2_part1_1_perm10__S__D_0.LockedTensor(), indices_ma, false,
				1.0, Zaccum_lvl1_part2_1_perm2310__D_2__D_3__D_1__D_0.Tensor(), indices_ijba, false);
			P_jimb_lvl1_part0_1_lvl2_part2_1_perm0132__D_2__D_3__D_1__S.EmptyData();
			t_fj_lvl2_part1_1_perm10__S__D_0.EmptyData();

			SlidePartitionDown
			( P_jimb_lvl1_part0_1_lvl2_part2T__D_0__D_1__D_2__D_3,  P_jimb_lvl1_part0_1_lvl2_part2_0__D_0__D_1__D_2__D_3,
			       P_jimb_lvl1_part0_1_lvl2_part2_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  P_jimb_lvl1_part0_1_lvl2_part2B__D_0__D_1__D_2__D_3, P_jimb_lvl1_part0_1_lvl2_part2_2__D_0__D_1__D_2__D_3, 2 );
			SlidePartitionDown
			( t_fj_lvl2_part1T__D_0_1__D_2_3,  t_fj_lvl2_part1_0__D_0_1__D_2_3,
			       t_fj_lvl2_part1_1__D_0_1__D_2_3,
			  /**/ /**/
			  t_fj_lvl2_part1B__D_0_1__D_2_3, t_fj_lvl2_part1_2__D_0_1__D_2_3, 1 );

		}
		//****
		Permute( Zaccum_lvl1_part2_1_perm2310__D_2__D_3__D_1__D_0, Zaccum_lvl1_part2_1__D_0__D_1__D_2__D_3 );
		Zaccum_lvl1_part2_1_perm2310__D_2__D_3__D_1__D_0.EmptyData();

		SlidePartitionDown
		( P_jimb_lvl1_part0T__D_0__D_1__D_2__D_3,  P_jimb_lvl1_part0_0__D_0__D_1__D_2__D_3,
		       P_jimb_lvl1_part0_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  P_jimb_lvl1_part0B__D_0__D_1__D_2__D_3, P_jimb_lvl1_part0_2__D_0__D_1__D_2__D_3, 0 );
		SlidePartitionDown
		( Zaccum_lvl1_part2T__D_0__D_1__D_2__D_3,  Zaccum_lvl1_part2_0__D_0__D_1__D_2__D_3,
		       Zaccum_lvl1_part2_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  Zaccum_lvl1_part2B__D_0__D_1__D_2__D_3, Zaccum_lvl1_part2_2__D_0__D_1__D_2__D_3, 2 );

	}
	//****
	//**** (out of 1)
	//**** Is real	0 shadows
		//Outputs:
		//  Zaccum__D_0__D_1__D_2__D_3
	PartitionDown(t_fj__D_0_1__D_2_3, t_fj_lvl1_part1T__D_0_1__D_2_3, t_fj_lvl1_part1B__D_0_1__D_2_3, 1, 0);
	PartitionDown(Zaccum__D_0__D_1__D_2__D_3, Zaccum_lvl1_part2T__D_0__D_1__D_2__D_3, Zaccum_lvl1_part2B__D_0__D_1__D_2__D_3, 2, 0);
	while(Zaccum_lvl1_part2T__D_0__D_1__D_2__D_3.Dimension(2) < Zaccum__D_0__D_1__D_2__D_3.Dimension(2))
	{
		RepartitionDown
		( t_fj_lvl1_part1T__D_0_1__D_2_3,  t_fj_lvl1_part1_0__D_0_1__D_2_3,
		  /**/ /**/
		       t_fj_lvl1_part1_1__D_0_1__D_2_3,
		  t_fj_lvl1_part1B__D_0_1__D_2_3, t_fj_lvl1_part1_2__D_0_1__D_2_3, 1, blkSize );
		RepartitionDown
		( Zaccum_lvl1_part2T__D_0__D_1__D_2__D_3,  Zaccum_lvl1_part2_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       Zaccum_lvl1_part2_1__D_0__D_1__D_2__D_3,
		  Zaccum_lvl1_part2B__D_0__D_1__D_2__D_3, Zaccum_lvl1_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );

		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  Zaccum_lvl1_part2_1__D_0__D_1__D_2__D_3
		PartitionDown(r_bmfe__D_0__D_1__D_2__D_3, r_bmfe_lvl2_part1T__D_0__D_1__D_2__D_3, r_bmfe_lvl2_part1B__D_0__D_1__D_2__D_3, 1, 0);
		PartitionDown(Zaccum_lvl1_part2_1__D_0__D_1__D_2__D_3, Zaccum_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3, Zaccum_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, 3, 0);
		while(Zaccum_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3.Dimension(3) < Zaccum_lvl1_part2_1__D_0__D_1__D_2__D_3.Dimension(3))
		{
			RepartitionDown
			( r_bmfe_lvl2_part1T__D_0__D_1__D_2__D_3,  r_bmfe_lvl2_part1_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       r_bmfe_lvl2_part1_1__D_0__D_1__D_2__D_3,
			  r_bmfe_lvl2_part1B__D_0__D_1__D_2__D_3, r_bmfe_lvl2_part1_2__D_0__D_1__D_2__D_3, 1, blkSize );
			RepartitionDown
			( Zaccum_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3,  Zaccum_lvl1_part2_1_lvl2_part3_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       Zaccum_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3,
			  Zaccum_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, Zaccum_lvl1_part2_1_lvl2_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );

			Permute( r_bmfe_lvl2_part1_1__D_0__D_1__D_2__D_3, r_bmfe_lvl2_part1_1_perm1230__D_1__D_2__D_3__D_0 );
			   // t_fj_lvl1_part1_1[D0,*] <- t_fj_lvl1_part1_1[D01,D23]
			t_fj_lvl1_part1_1__D_0__S.AlignModesWith( modes_0, r_bmfe_lvl2_part1_1__D_0__D_1__D_2__D_3, modes_0 );
			t_fj_lvl1_part1_1__D_0__S.AllGatherRedistFrom( t_fj_lvl1_part1_1__D_0_1__D_2_3, modes_1_2_3 );
			Zaccum_lvl1_part2_1_lvl2_part3_1_perm30124__D_1__D_2__D_3__S__D_0.AlignModesWith( modes_0_1_3, r_bmfe_lvl2_part1_1__D_0__D_1__D_2__D_3, modes_2_3_1 );
			tempShape = Zaccum_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3.Shape();
			tempShape.push_back( g.Shape()[0] );
			Zaccum_lvl1_part2_1_lvl2_part3_1_perm30124__D_1__D_2__D_3__S__D_0.ResizeTo( tempShape );
			   // 1.0 * r_bmfe_lvl2_part1_1[D0,D1,D2,D3]_jabe * t_fj_lvl1_part1_1[D0,*]_ei + 0.0 * Zaccum_lvl1_part2_1_lvl2_part3_1[D2,D3,*,D1,D0]_jabie
if(commRank == 0){
flops += 2*1*r_bmfe_lvl2_part1_1_perm1230__D_1__D_2__D_3__D_0.Dimension(0)*1*r_bmfe_lvl2_part1_1_perm1230__D_1__D_2__D_3__D_0.Dimension(2)*r_bmfe_lvl2_part1_1_perm1230__D_1__D_2__D_3__D_0.Dimension(3)*r_bmfe_lvl2_part1_1_perm1230__D_1__D_2__D_3__D_0.Dimension(1)*1*t_fj_lvl1_part1_1__D_0__S.Dimension(1);
}
			LocalContract(1.0, r_bmfe_lvl2_part1_1_perm1230__D_1__D_2__D_3__D_0.LockedTensor(), indices_jabe, false,
				t_fj_lvl1_part1_1__D_0__S.LockedTensor(), indices_ei, false,
				0.0, Zaccum_lvl1_part2_1_lvl2_part3_1_perm30124__D_1__D_2__D_3__S__D_0.Tensor(), indices_jabie, false);
			r_bmfe_lvl2_part1_1_perm1230__D_1__D_2__D_3__D_0.EmptyData();
			t_fj_lvl1_part1_1__D_0__S.EmptyData();
			Zaccum_lvl1_part2_1_lvl2_part3_1_temp__D_2_0__D_3__S__D_1.AlignModesWith( modes_0_1_2_3, Zaccum_lvl1_part2_1_lvl2_part3_1_temp__D_0__D_1__D_2__D_3, modes_0_1_2_3 );
			tempShape = Zaccum_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3.Shape();
			Zaccum_lvl1_part2_1_lvl2_part3_1_temp__D_2_0__D_3__S__D_1.ResizeTo( tempShape );
			   // Zaccum_lvl1_part2_1_lvl2_part3_1_temp[D20,D3,*,D1] <- Zaccum_lvl1_part2_1_lvl2_part3_1[D2,D3,*,D1,D0] (with SumScatter on D0)
			Zaccum_lvl1_part2_1_lvl2_part3_1_temp__D_2_0__D_3__S__D_1.ReduceScatterRedistFrom( Zaccum_lvl1_part2_1_lvl2_part3_1_perm30124__D_1__D_2__D_3__S__D_0, 4 );
			Zaccum_lvl1_part2_1_lvl2_part3_1_perm30124__D_1__D_2__D_3__S__D_0.EmptyData();
			   // Zaccum_lvl1_part2_1_lvl2_part3_1_temp[D20,D1,*,D3] <- Zaccum_lvl1_part2_1_lvl2_part3_1_temp[D20,D3,*,D1]
			Zaccum_lvl1_part2_1_lvl2_part3_1_temp__D_2_0__D_1__S__D_3.AlignModesWith( modes_0_1_2_3, Zaccum_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3, modes_0_1_2_3 );
			Zaccum_lvl1_part2_1_lvl2_part3_1_temp__D_2_0__D_1__S__D_3.AllToAllRedistFrom( Zaccum_lvl1_part2_1_lvl2_part3_1_temp__D_2_0__D_3__S__D_1, modes_1_3 );
			Zaccum_lvl1_part2_1_lvl2_part3_1_temp__D_2_0__D_3__S__D_1.EmptyData();
			   // Zaccum_lvl1_part2_1_lvl2_part3_1_temp[D0,D1,D2,D3] <- Zaccum_lvl1_part2_1_lvl2_part3_1_temp[D20,D1,*,D3]
			Zaccum_lvl1_part2_1_lvl2_part3_1_temp__D_0__D_1__D_2__D_3.AlignModesWith( modes_0_1_2_3, Zaccum_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3, modes_0_1_2_3 );
			Zaccum_lvl1_part2_1_lvl2_part3_1_temp__D_0__D_1__D_2__D_3.AllToAllRedistFrom( Zaccum_lvl1_part2_1_lvl2_part3_1_temp__D_2_0__D_1__S__D_3, modes_0_2 );
			Zaccum_lvl1_part2_1_lvl2_part3_1_temp__D_2_0__D_1__S__D_3.EmptyData();
if(commRank == 0){
flops += 2*prod(Zaccum_lvl1_part2_1_lvl2_part3_1_temp__D_0__D_1__D_2__D_3.Shape());
}
			YxpBy( Zaccum_lvl1_part2_1_lvl2_part3_1_temp__D_0__D_1__D_2__D_3, 1.0, Zaccum_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3 );
			Zaccum_lvl1_part2_1_lvl2_part3_1_temp__D_0__D_1__D_2__D_3.EmptyData();

			SlidePartitionDown
			( r_bmfe_lvl2_part1T__D_0__D_1__D_2__D_3,  r_bmfe_lvl2_part1_0__D_0__D_1__D_2__D_3,
			       r_bmfe_lvl2_part1_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  r_bmfe_lvl2_part1B__D_0__D_1__D_2__D_3, r_bmfe_lvl2_part1_2__D_0__D_1__D_2__D_3, 1 );
			SlidePartitionDown
			( Zaccum_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3,  Zaccum_lvl1_part2_1_lvl2_part3_0__D_0__D_1__D_2__D_3,
			       Zaccum_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  Zaccum_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, Zaccum_lvl1_part2_1_lvl2_part3_2__D_0__D_1__D_2__D_3, 3 );

		}
		//****

		SlidePartitionDown
		( t_fj_lvl1_part1T__D_0_1__D_2_3,  t_fj_lvl1_part1_0__D_0_1__D_2_3,
		       t_fj_lvl1_part1_1__D_0_1__D_2_3,
		  /**/ /**/
		  t_fj_lvl1_part1B__D_0_1__D_2_3, t_fj_lvl1_part1_2__D_0_1__D_2_3, 1 );
		SlidePartitionDown
		( Zaccum_lvl1_part2T__D_0__D_1__D_2__D_3,  Zaccum_lvl1_part2_0__D_0__D_1__D_2__D_3,
		       Zaccum_lvl1_part2_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  Zaccum_lvl1_part2B__D_0__D_1__D_2__D_3, Zaccum_lvl1_part2_2__D_0__D_1__D_2__D_3, 2 );

	}
	//****
	//**** (out of 1)
	//**** Is real	0 shadows
		//Outputs:
		//  Z_abij__D_0__D_1__D_2__D_3
	PartitionDown(Zaccum__D_0__D_1__D_2__D_3, Zaccum_lvl1_part2T__D_0__D_1__D_2__D_3, Zaccum_lvl1_part2B__D_0__D_1__D_2__D_3, 2, 0);
	PartitionDown(Zaccum__D_0__D_1__D_2__D_3, Zaccum_lvl1_part3T__D_0__D_1__D_2__D_3, Zaccum_lvl1_part3B__D_0__D_1__D_2__D_3, 3, 0);
	PartitionDown(Z_abij__D_0__D_1__D_2__D_3, Z_abij_lvl1_part2T__D_0__D_1__D_2__D_3, Z_abij_lvl1_part2B__D_0__D_1__D_2__D_3, 2, 0);
	while(Z_abij_lvl1_part2T__D_0__D_1__D_2__D_3.Dimension(2) < Z_abij__D_0__D_1__D_2__D_3.Dimension(2))
	{
		RepartitionDown
		( Zaccum_lvl1_part2T__D_0__D_1__D_2__D_3,  Zaccum_lvl1_part2_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       Zaccum_lvl1_part2_1__D_0__D_1__D_2__D_3,
		  Zaccum_lvl1_part2B__D_0__D_1__D_2__D_3, Zaccum_lvl1_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );
		RepartitionDown
		( Zaccum_lvl1_part3T__D_0__D_1__D_2__D_3,  Zaccum_lvl1_part3_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       Zaccum_lvl1_part3_1__D_0__D_1__D_2__D_3,
		  Zaccum_lvl1_part3B__D_0__D_1__D_2__D_3, Zaccum_lvl1_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );
		RepartitionDown
		( Z_abij_lvl1_part2T__D_0__D_1__D_2__D_3,  Z_abij_lvl1_part2_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       Z_abij_lvl1_part2_1__D_0__D_1__D_2__D_3,
		  Z_abij_lvl1_part2B__D_0__D_1__D_2__D_3, Z_abij_lvl1_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );

		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  Z_abij_lvl1_part2_1__D_0__D_1__D_2__D_3
		PartitionDown(Zaccum_lvl1_part2_1__D_0__D_1__D_2__D_3, Zaccum_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3, Zaccum_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, 3, 0);
		PartitionDown(Zaccum_lvl1_part3_1__D_0__D_1__D_2__D_3, Zaccum_lvl1_part3_1_lvl2_part2T__D_0__D_1__D_2__D_3, Zaccum_lvl1_part3_1_lvl2_part2B__D_0__D_1__D_2__D_3, 2, 0);
		PartitionDown(Z_abij_lvl1_part2_1__D_0__D_1__D_2__D_3, Z_abij_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3, Z_abij_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, 3, 0);
		while(Z_abij_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3.Dimension(3) < Z_abij_lvl1_part2_1__D_0__D_1__D_2__D_3.Dimension(3))
		{
			RepartitionDown
			( Zaccum_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3,  Zaccum_lvl1_part2_1_lvl2_part3_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       Zaccum_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3,
			  Zaccum_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, Zaccum_lvl1_part2_1_lvl2_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );
			RepartitionDown
			( Zaccum_lvl1_part3_1_lvl2_part2T__D_0__D_1__D_2__D_3,  Zaccum_lvl1_part3_1_lvl2_part2_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       Zaccum_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_2__D_3,
			  Zaccum_lvl1_part3_1_lvl2_part2B__D_0__D_1__D_2__D_3, Zaccum_lvl1_part3_1_lvl2_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );
			RepartitionDown
			( Z_abij_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3,  Z_abij_lvl1_part2_1_lvl2_part3_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       Z_abij_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3,
			  Z_abij_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, Z_abij_lvl1_part2_1_lvl2_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );

			   // Zaccum_lvl1_part3_1_lvl2_part2_1[D1,D0,D2,D3] <- Zaccum_lvl1_part3_1_lvl2_part2_1[D0,D1,D2,D3]
			Zaccum_lvl1_part3_1_lvl2_part2_1__D_1__D_0__D_2__D_3.AlignModesWith( modes_0_1_2_3, Zaccum_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3, modes_1_0_3_2 );
			Zaccum_lvl1_part3_1_lvl2_part2_1__D_1__D_0__D_2__D_3.AllToAllRedistFrom( Zaccum_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_2__D_3, modes_0_1 );
			   // Zaccum_lvl1_part3_1_lvl2_part2_1[D1,D0,D3,D2] <- Zaccum_lvl1_part3_1_lvl2_part2_1[D1,D0,D2,D3]
			Zaccum_lvl1_part3_1_lvl2_part2_1__D_1__D_0__D_3__D_2.AlignModesWith( modes_0_1_2_3, Zaccum_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3, modes_1_0_3_2 );
			Zaccum_lvl1_part3_1_lvl2_part2_1__D_1__D_0__D_3__D_2.AllToAllRedistFrom( Zaccum_lvl1_part3_1_lvl2_part2_1__D_1__D_0__D_2__D_3, modes_2_3 );
			Zaccum_lvl1_part3_1_lvl2_part2_1__D_1__D_0__D_2__D_3.EmptyData();
if(commRank == 0){
flops += 3*prod(Zaccum_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3.Shape());
}
			YAxpPx( 1.0, Zaccum_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3, 1.0, Zaccum_lvl1_part3_1_lvl2_part2_1__D_1__D_0__D_3__D_2, perm_1_0_3_2, Z_abij_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3 );
			Zaccum_lvl1_part3_1_lvl2_part2_1__D_1__D_0__D_3__D_2.EmptyData();

			SlidePartitionDown
			( Zaccum_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3,  Zaccum_lvl1_part2_1_lvl2_part3_0__D_0__D_1__D_2__D_3,
			       Zaccum_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  Zaccum_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, Zaccum_lvl1_part2_1_lvl2_part3_2__D_0__D_1__D_2__D_3, 3 );
			SlidePartitionDown
			( Zaccum_lvl1_part3_1_lvl2_part2T__D_0__D_1__D_2__D_3,  Zaccum_lvl1_part3_1_lvl2_part2_0__D_0__D_1__D_2__D_3,
			       Zaccum_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  Zaccum_lvl1_part3_1_lvl2_part2B__D_0__D_1__D_2__D_3, Zaccum_lvl1_part3_1_lvl2_part2_2__D_0__D_1__D_2__D_3, 2 );
			SlidePartitionDown
			( Z_abij_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3,  Z_abij_lvl1_part2_1_lvl2_part3_0__D_0__D_1__D_2__D_3,
			       Z_abij_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  Z_abij_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, Z_abij_lvl1_part2_1_lvl2_part3_2__D_0__D_1__D_2__D_3, 3 );

		}
		//****

		SlidePartitionDown
		( Zaccum_lvl1_part2T__D_0__D_1__D_2__D_3,  Zaccum_lvl1_part2_0__D_0__D_1__D_2__D_3,
		       Zaccum_lvl1_part2_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  Zaccum_lvl1_part2B__D_0__D_1__D_2__D_3, Zaccum_lvl1_part2_2__D_0__D_1__D_2__D_3, 2 );
		SlidePartitionDown
		( Zaccum_lvl1_part3T__D_0__D_1__D_2__D_3,  Zaccum_lvl1_part3_0__D_0__D_1__D_2__D_3,
		       Zaccum_lvl1_part3_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  Zaccum_lvl1_part3B__D_0__D_1__D_2__D_3, Zaccum_lvl1_part3_2__D_0__D_1__D_2__D_3, 3 );
		SlidePartitionDown
		( Z_abij_lvl1_part2T__D_0__D_1__D_2__D_3,  Z_abij_lvl1_part2_0__D_0__D_1__D_2__D_3,
		       Z_abij_lvl1_part2_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  Z_abij_lvl1_part2B__D_0__D_1__D_2__D_3, Z_abij_lvl1_part2_2__D_0__D_1__D_2__D_3, 2 );

	}
	//****
	Zaccum__D_0__D_1__D_2__D_3.EmptyData();
	Zaccum__D_0__D_1__D_2__D_3.EmptyData();
	//**** (out of 1)
	//**** Is real	0 shadows
		//Outputs:
		//  Z_abij__D_0__D_1__D_2__D_3
	PartitionDown(Tau_efmn__D_0__D_1__D_2__D_3, Tau_efmn_lvl1_part2T__D_0__D_1__D_2__D_3, Tau_efmn_lvl1_part2B__D_0__D_1__D_2__D_3, 2, 0);
	PartitionDown(Z_abij__D_0__D_1__D_2__D_3, Z_abij_lvl1_part2T__D_0__D_1__D_2__D_3, Z_abij_lvl1_part2B__D_0__D_1__D_2__D_3, 2, 0);
	while(Z_abij_lvl1_part2T__D_0__D_1__D_2__D_3.Dimension(2) < Z_abij__D_0__D_1__D_2__D_3.Dimension(2))
	{
		RepartitionDown
		( Tau_efmn_lvl1_part2T__D_0__D_1__D_2__D_3,  Tau_efmn_lvl1_part2_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       Tau_efmn_lvl1_part2_1__D_0__D_1__D_2__D_3,
		  Tau_efmn_lvl1_part2B__D_0__D_1__D_2__D_3, Tau_efmn_lvl1_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );
		RepartitionDown
		( Z_abij_lvl1_part2T__D_0__D_1__D_2__D_3,  Z_abij_lvl1_part2_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       Z_abij_lvl1_part2_1__D_0__D_1__D_2__D_3,
		  Z_abij_lvl1_part2B__D_0__D_1__D_2__D_3, Z_abij_lvl1_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );

		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  Z_abij_lvl1_part2_1__D_0__D_1__D_2__D_3
		PartitionDown(Tau_efmn_lvl1_part2_1__D_0__D_1__D_2__D_3, Tau_efmn_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3, Tau_efmn_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, 3, 0);
		PartitionDown(Z_abij_lvl1_part2_1__D_0__D_1__D_2__D_3, Z_abij_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3, Z_abij_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, 3, 0);
		while(Z_abij_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3.Dimension(3) < Z_abij_lvl1_part2_1__D_0__D_1__D_2__D_3.Dimension(3))
		{
			RepartitionDown
			( Tau_efmn_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3,  Tau_efmn_lvl1_part2_1_lvl2_part3_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       Tau_efmn_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3,
			  Tau_efmn_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, Tau_efmn_lvl1_part2_1_lvl2_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );
			RepartitionDown
			( Z_abij_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3,  Z_abij_lvl1_part2_1_lvl2_part3_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       Z_abij_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3,
			  Z_abij_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, Z_abij_lvl1_part2_1_lvl2_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );

			   // Tau_efmn_lvl1_part2_1_lvl2_part3_1[D2,D1,D0,D3] <- Tau_efmn_lvl1_part2_1_lvl2_part3_1[D0,D1,D2,D3]
			Tau_efmn_lvl1_part2_1_lvl2_part3_1__D_2__D_1__D_0__D_3.AlignModesWith( modes_0_1, y_abef__D_0__D_1__D_2__D_3, modes_2_3 );
			Tau_efmn_lvl1_part2_1_lvl2_part3_1__D_2__D_1__D_0__D_3.AllToAllRedistFrom( Tau_efmn_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3, modes_0_2 );
			   // Tau_efmn_lvl1_part2_1_lvl2_part3_1[D2,D3,D0,*] <- Tau_efmn_lvl1_part2_1_lvl2_part3_1[D2,D1,D0,D3]
			Tau_efmn_lvl1_part2_1_lvl2_part3_1__D_2__D_3__D_0__S.AlignModesWith( modes_0_1, y_abef__D_0__D_1__D_2__D_3, modes_2_3 );
			Tau_efmn_lvl1_part2_1_lvl2_part3_1__D_2__D_3__D_0__S.AllToAllRedistFrom( Tau_efmn_lvl1_part2_1_lvl2_part3_1__D_2__D_1__D_0__D_3, modes_1_3 );
			Tau_efmn_lvl1_part2_1_lvl2_part3_1__D_2__D_1__D_0__D_3.EmptyData();
			   // Tau_efmn_lvl1_part2_1_lvl2_part3_1[D2,D3,*,*] <- Tau_efmn_lvl1_part2_1_lvl2_part3_1[D2,D3,D0,*]
			Tau_efmn_lvl1_part2_1_lvl2_part3_1__D_2__D_3__S__S.AlignModesWith( modes_0_1, y_abef__D_0__D_1__D_2__D_3, modes_2_3 );
			Tau_efmn_lvl1_part2_1_lvl2_part3_1__D_2__D_3__S__S.AllGatherRedistFrom( Tau_efmn_lvl1_part2_1_lvl2_part3_1__D_2__D_3__D_0__S, modes_0 );
			Tau_efmn_lvl1_part2_1_lvl2_part3_1__D_2__D_3__D_0__S.EmptyData();
			Z_abij_lvl1_part2_1_lvl2_part3_1__D_0__D_1__S__S__D_2__D_3.AlignModesWith( modes_0_1, y_abef__D_0__D_1__D_2__D_3, modes_0_1 );
			tempShape = Z_abij_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3.Shape();
			tempShape.push_back( g.Shape()[2] );
			tempShape.push_back( g.Shape()[3] );
			Z_abij_lvl1_part2_1_lvl2_part3_1__D_0__D_1__S__S__D_2__D_3.ResizeTo( tempShape );
			   // 1.0 * y_abef[D0,D1,D2,D3]_abef * Tau_efmn_lvl1_part2_1_lvl2_part3_1[D2,D3,*,*]_efij + 0.0 * Z_abij_lvl1_part2_1_lvl2_part3_1[D0,D1,*,*,D2,D3]_abijef
if(commRank == 0){
flops += 2*1*y_abef__D_0__D_1__D_2__D_3.Dimension(2)*y_abef__D_0__D_1__D_2__D_3.Dimension(3)*1*y_abef__D_0__D_1__D_2__D_3.Dimension(0)*y_abef__D_0__D_1__D_2__D_3.Dimension(1)*1*Tau_efmn_lvl1_part2_1_lvl2_part3_1__D_2__D_3__S__S.Dimension(2)*Tau_efmn_lvl1_part2_1_lvl2_part3_1__D_2__D_3__S__S.Dimension(3);
}
			LocalContract(1.0, y_abef__D_0__D_1__D_2__D_3.LockedTensor(), indices_abef, false,
				Tau_efmn_lvl1_part2_1_lvl2_part3_1__D_2__D_3__S__S.LockedTensor(), indices_efij, false,
				0.0, Z_abij_lvl1_part2_1_lvl2_part3_1__D_0__D_1__S__S__D_2__D_3.Tensor(), indices_abijef, false);
			Tau_efmn_lvl1_part2_1_lvl2_part3_1__D_2__D_3__S__S.EmptyData();
			   // Z_abij_lvl1_part2_1_lvl2_part3_1[D0,D1,D2,D3] <- Z_abij_lvl1_part2_1_lvl2_part3_1[D0,D1,*,*,D2,D3] (with SumScatter on (D2)(D3))
			Z_abij_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3.ReduceScatterUpdateRedistFrom( Z_abij_lvl1_part2_1_lvl2_part3_1__D_0__D_1__S__S__D_2__D_3, 1.0, modes_5_4 );
			Z_abij_lvl1_part2_1_lvl2_part3_1__D_0__D_1__S__S__D_2__D_3.EmptyData();

			SlidePartitionDown
			( Tau_efmn_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3,  Tau_efmn_lvl1_part2_1_lvl2_part3_0__D_0__D_1__D_2__D_3,
			       Tau_efmn_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  Tau_efmn_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, Tau_efmn_lvl1_part2_1_lvl2_part3_2__D_0__D_1__D_2__D_3, 3 );
			SlidePartitionDown
			( Z_abij_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3,  Z_abij_lvl1_part2_1_lvl2_part3_0__D_0__D_1__D_2__D_3,
			       Z_abij_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  Z_abij_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, Z_abij_lvl1_part2_1_lvl2_part3_2__D_0__D_1__D_2__D_3, 3 );

		}
		//****

		SlidePartitionDown
		( Tau_efmn_lvl1_part2T__D_0__D_1__D_2__D_3,  Tau_efmn_lvl1_part2_0__D_0__D_1__D_2__D_3,
		       Tau_efmn_lvl1_part2_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  Tau_efmn_lvl1_part2B__D_0__D_1__D_2__D_3, Tau_efmn_lvl1_part2_2__D_0__D_1__D_2__D_3, 2 );
		SlidePartitionDown
		( Z_abij_lvl1_part2T__D_0__D_1__D_2__D_3,  Z_abij_lvl1_part2_0__D_0__D_1__D_2__D_3,
		       Z_abij_lvl1_part2_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  Z_abij_lvl1_part2B__D_0__D_1__D_2__D_3, Z_abij_lvl1_part2_2__D_0__D_1__D_2__D_3, 2 );

	}
	//****
	Permute( Z_abij__D_0__D_1__D_2__D_3, Z_abij_perm2301__D_2__D_3__D_0__D_1 );
	//**** (out of 1)
	//**** Is real	0 shadows
		//Outputs:
		//  Z_abij_perm2301__D_2__D_3__D_0__D_1
	PartitionDown(Q_mnij__D_0__D_1__D_2__D_3, Q_mnij_lvl1_part0T__D_0__D_1__D_2__D_3, Q_mnij_lvl1_part0B__D_0__D_1__D_2__D_3, 0, 0);
	PartitionDown(Tau_efmn__D_0__D_1__D_2__D_3, Tau_efmn_lvl1_part2T__D_0__D_1__D_2__D_3, Tau_efmn_lvl1_part2B__D_0__D_1__D_2__D_3, 2, 0);
	while(Q_mnij_lvl1_part0T__D_0__D_1__D_2__D_3.Dimension(0) < Q_mnij__D_0__D_1__D_2__D_3.Dimension(0))
	{
		RepartitionDown
		( Q_mnij_lvl1_part0T__D_0__D_1__D_2__D_3,  Q_mnij_lvl1_part0_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       Q_mnij_lvl1_part0_1__D_0__D_1__D_2__D_3,
		  Q_mnij_lvl1_part0B__D_0__D_1__D_2__D_3, Q_mnij_lvl1_part0_2__D_0__D_1__D_2__D_3, 0, blkSize );
		RepartitionDown
		( Tau_efmn_lvl1_part2T__D_0__D_1__D_2__D_3,  Tau_efmn_lvl1_part2_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       Tau_efmn_lvl1_part2_1__D_0__D_1__D_2__D_3,
		  Tau_efmn_lvl1_part2B__D_0__D_1__D_2__D_3, Tau_efmn_lvl1_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );

		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  Z_abij_perm2301__D_2__D_3__D_0__D_1
		PartitionDown(Q_mnij_lvl1_part0_1__D_0__D_1__D_2__D_3, Q_mnij_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3, Q_mnij_lvl1_part0_1_lvl2_part1B__D_0__D_1__D_2__D_3, 1, 0);
		PartitionDown(Tau_efmn_lvl1_part2_1__D_0__D_1__D_2__D_3, Tau_efmn_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3, Tau_efmn_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, 3, 0);
		while(Q_mnij_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3.Dimension(1) < Q_mnij_lvl1_part0_1__D_0__D_1__D_2__D_3.Dimension(1))
		{
			RepartitionDown
			( Q_mnij_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3,  Q_mnij_lvl1_part0_1_lvl2_part1_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       Q_mnij_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3,
			  Q_mnij_lvl1_part0_1_lvl2_part1B__D_0__D_1__D_2__D_3, Q_mnij_lvl1_part0_1_lvl2_part1_2__D_0__D_1__D_2__D_3, 1, blkSize );
			RepartitionDown
			( Tau_efmn_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3,  Tau_efmn_lvl1_part2_1_lvl2_part3_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       Tau_efmn_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3,
			  Tau_efmn_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, Tau_efmn_lvl1_part2_1_lvl2_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );

			   // Q_mnij_lvl1_part0_1_lvl2_part1_1[*,*,D2,D3] <- Q_mnij_lvl1_part0_1_lvl2_part1_1[D0,D1,D2,D3]
			Q_mnij_lvl1_part0_1_lvl2_part1_1_perm2301__D_2__D_3__S__S.AlignModesWith( modes_2_3, Z_abij__D_0__D_1__D_2__D_3, modes_2_3 );
			Q_mnij_lvl1_part0_1_lvl2_part1_1_perm2301__D_2__D_3__S__S.AllGatherRedistFrom( Q_mnij_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3, modes_0_1 );
			   // Tau_efmn_lvl1_part2_1_lvl2_part3_1[D0,D1,*,*] <- Tau_efmn_lvl1_part2_1_lvl2_part3_1[D0,D1,D2,D3]
			Tau_efmn_lvl1_part2_1_lvl2_part3_1_perm2301__S__S__D_0__D_1.AlignModesWith( modes_0_1, Z_abij__D_0__D_1__D_2__D_3, modes_0_1 );
			Tau_efmn_lvl1_part2_1_lvl2_part3_1_perm2301__S__S__D_0__D_1.AllGatherRedistFrom( Tau_efmn_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3, modes_2_3 );
			   // 1.0 * Q_mnij_lvl1_part0_1_lvl2_part1_1[*,*,D2,D3]_ijmn * Tau_efmn_lvl1_part2_1_lvl2_part3_1[D0,D1,*,*]_mnab + 1.0 * Z_abij[D0,D1,D2,D3]_ijab
if(commRank == 0){
flops += 2*1*Q_mnij_lvl1_part0_1_lvl2_part1_1_perm2301__D_2__D_3__S__S.Dimension(0)*Q_mnij_lvl1_part0_1_lvl2_part1_1_perm2301__D_2__D_3__S__S.Dimension(1)*1*Q_mnij_lvl1_part0_1_lvl2_part1_1_perm2301__D_2__D_3__S__S.Dimension(2)*Q_mnij_lvl1_part0_1_lvl2_part1_1_perm2301__D_2__D_3__S__S.Dimension(3)*1*Tau_efmn_lvl1_part2_1_lvl2_part3_1_perm2301__S__S__D_0__D_1.Dimension(0)*Tau_efmn_lvl1_part2_1_lvl2_part3_1_perm2301__S__S__D_0__D_1.Dimension(1);
}
			LocalContractAndLocalEliminate(1.0, Q_mnij_lvl1_part0_1_lvl2_part1_1_perm2301__D_2__D_3__S__S.LockedTensor(), indices_ijmn, false,
				Tau_efmn_lvl1_part2_1_lvl2_part3_1_perm2301__S__S__D_0__D_1.LockedTensor(), indices_mnab, false,
				1.0, Z_abij_perm2301__D_2__D_3__D_0__D_1.Tensor(), indices_ijab, false);
			Q_mnij_lvl1_part0_1_lvl2_part1_1_perm2301__D_2__D_3__S__S.EmptyData();
			Tau_efmn_lvl1_part2_1_lvl2_part3_1_perm2301__S__S__D_0__D_1.EmptyData();

			SlidePartitionDown
			( Q_mnij_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3,  Q_mnij_lvl1_part0_1_lvl2_part1_0__D_0__D_1__D_2__D_3,
			       Q_mnij_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  Q_mnij_lvl1_part0_1_lvl2_part1B__D_0__D_1__D_2__D_3, Q_mnij_lvl1_part0_1_lvl2_part1_2__D_0__D_1__D_2__D_3, 1 );
			SlidePartitionDown
			( Tau_efmn_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3,  Tau_efmn_lvl1_part2_1_lvl2_part3_0__D_0__D_1__D_2__D_3,
			       Tau_efmn_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  Tau_efmn_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, Tau_efmn_lvl1_part2_1_lvl2_part3_2__D_0__D_1__D_2__D_3, 3 );

		}
		//****

		SlidePartitionDown
		( Q_mnij_lvl1_part0T__D_0__D_1__D_2__D_3,  Q_mnij_lvl1_part0_0__D_0__D_1__D_2__D_3,
		       Q_mnij_lvl1_part0_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  Q_mnij_lvl1_part0B__D_0__D_1__D_2__D_3, Q_mnij_lvl1_part0_2__D_0__D_1__D_2__D_3, 0 );
		SlidePartitionDown
		( Tau_efmn_lvl1_part2T__D_0__D_1__D_2__D_3,  Tau_efmn_lvl1_part2_0__D_0__D_1__D_2__D_3,
		       Tau_efmn_lvl1_part2_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  Tau_efmn_lvl1_part2B__D_0__D_1__D_2__D_3, Tau_efmn_lvl1_part2_2__D_0__D_1__D_2__D_3, 2 );

	}
	//****
	Permute( Z_abij_perm2301__D_2__D_3__D_0__D_1, Z_abij__D_0__D_1__D_2__D_3 );
	Z_abij_perm2301__D_2__D_3__D_0__D_1.EmptyData();


//****
//**** (out of 1)
//**** Is real	0 shadows
	//Outputs:
	//  Z_abij__D_0__D_1__D_2__D_3

if(commRank == 0){
flops += prod(v_femn__D_0__D_1__D_2__D_3.Shape());
}
	Yxpy( v_femn__D_0__D_1__D_2__D_3, Z_abij__D_0__D_1__D_2__D_3 );


//****


//END_CODE

    /*****************************************/
    mpi::Barrier(g.OwningComm());
    runTime = mpi::Time() - startTime;
    gflops = flops / (1e9 * runTime);
#ifdef CORRECTNESS
    DistTensor<double> diff_Z(dist__D_0__D_1__D_2__D_3, g);
    diff_Z.ResizeTo(check_Z);
    Diff(check_Z, Z_abij__D_0__D_1__D_2__D_3, diff_Z);
   norm = 1.0;
   norm = Norm(diff_Z);
   if (commRank == 0){
     std::cout << "NORM_Z " << norm << std::endl;
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


