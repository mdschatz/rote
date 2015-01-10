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
    
ObjShape tempShape;
TensorDistribution dist__S__S__D_2__D_3 = tmen::StringToTensorDist("[(),(),(2),(3)]");
TensorDistribution dist__S__D_1__D_2__D_3 = tmen::StringToTensorDist("[(),(1),(2),(3)]");
TensorDistribution dist__S__D_2 = tmen::StringToTensorDist("[(),(2)]");
TensorDistribution dist__S__D_3__S__D_1 = tmen::StringToTensorDist("[(),(3),(),(1)]");
TensorDistribution dist__D_0__S__S__D_2 = tmen::StringToTensorDist("[(0),(),(),(2)]");
TensorDistribution dist__D_0__S__D_2__S__D_1__D_3 = tmen::StringToTensorDist("[(0),(),(2),(),(1),(3)]");
TensorDistribution dist__D_0__S = tmen::StringToTensorDist("[(0),()]");
TensorDistribution dist__D_0__D_1__S__S__D_2__D_3 = tmen::StringToTensorDist("[(0),(1),(),(),(2),(3)]");
TensorDistribution dist__D_0__D_1__S__S = tmen::StringToTensorDist("[(0),(1),(),()]");
TensorDistribution dist__D_0__D_1__S__D_3 = tmen::StringToTensorDist("[(0),(1),(),(3)]");
TensorDistribution dist__D_0__D_1__D_2__D_3 = tmen::StringToTensorDist("[(0),(1),(2),(3)]");
TensorDistribution dist__D_0__D_1__D_3__D_2 = tmen::StringToTensorDist("[(0),(1),(3),(2)]");
TensorDistribution dist__D_0__D_3__D_2__D_1 = tmen::StringToTensorDist("[(0),(3),(2),(1)]");
TensorDistribution dist__D_2_0__D_1__S__D_3 = tmen::StringToTensorDist("[(2,0),(1),(),(3)]");
TensorDistribution dist__D_2_0__D_3__S__D_1 = tmen::StringToTensorDist("[(2,0),(3),(),(1)]");
TensorDistribution dist__D_1__S__S__D_3 = tmen::StringToTensorDist("[(1),(),(),(3)]");
TensorDistribution dist__D_1__D_0__D_2__D_3 = tmen::StringToTensorDist("[(1),(0),(2),(3)]");
TensorDistribution dist__D_1__D_0__D_3__D_2 = tmen::StringToTensorDist("[(1),(0),(3),(2)]");
TensorDistribution dist__D_2__D_1__D_0__D_3 = tmen::StringToTensorDist("[(2),(1),(0),(3)]");
TensorDistribution dist__D_2__D_3__S__S = tmen::StringToTensorDist("[(2),(3),(),()]");
TensorDistribution dist__D_2__D_3__S__D_1__D_0 = tmen::StringToTensorDist("[(2),(3),(),(1),(0)]");
TensorDistribution dist__D_2__D_3__S__D_1 = tmen::StringToTensorDist("[(2),(3),(),(1)]");
TensorDistribution dist__D_2__D_3__D_0__D_1 = tmen::StringToTensorDist("[(2),(3),(0),(1)]");
TensorDistribution dist__D_0_1__D_2_3 = tmen::StringToTensorDist("[(0,1),(2,3)]");
Permutation perm_0_1;
perm_0_1.push_back(0);
perm_0_1.push_back(1);
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
Permutation perm_0_2_1_3;
perm_0_2_1_3.push_back(0);
perm_0_2_1_3.push_back(2);
perm_0_2_1_3.push_back(1);
perm_0_2_1_3.push_back(3);
Permutation perm_0_3_1_2;
perm_0_3_1_2.push_back(0);
perm_0_3_1_2.push_back(3);
perm_0_3_1_2.push_back(1);
perm_0_3_1_2.push_back(2);
Permutation perm_1_0;
perm_1_0.push_back(1);
perm_1_0.push_back(0);
Permutation perm_1_0_3_2;
perm_1_0_3_2.push_back(1);
perm_1_0_3_2.push_back(0);
perm_1_0_3_2.push_back(3);
perm_1_0_3_2.push_back(2);
Permutation perm_1_2_0_3;
perm_1_2_0_3.push_back(1);
perm_1_2_0_3.push_back(2);
perm_1_2_0_3.push_back(0);
perm_1_2_0_3.push_back(3);
Permutation perm_1_2_3_0;
perm_1_2_3_0.push_back(1);
perm_1_2_3_0.push_back(2);
perm_1_2_3_0.push_back(3);
perm_1_2_3_0.push_back(0);
Permutation perm_1_3_0_2;
perm_1_3_0_2.push_back(1);
perm_1_3_0_2.push_back(3);
perm_1_3_0_2.push_back(0);
perm_1_3_0_2.push_back(2);
Permutation perm_1_3_0_2_5_4;
perm_1_3_0_2_5_4.push_back(1);
perm_1_3_0_2_5_4.push_back(3);
perm_1_3_0_2_5_4.push_back(0);
perm_1_3_0_2_5_4.push_back(2);
perm_1_3_0_2_5_4.push_back(5);
perm_1_3_0_2_5_4.push_back(4);
Permutation perm_2_0_1_3;
perm_2_0_1_3.push_back(2);
perm_2_0_1_3.push_back(0);
perm_2_0_1_3.push_back(1);
perm_2_0_1_3.push_back(3);
Permutation perm_2_0_3_1;
perm_2_0_3_1.push_back(2);
perm_2_0_3_1.push_back(0);
perm_2_0_3_1.push_back(3);
perm_2_0_3_1.push_back(1);
Permutation perm_2_1_0_3;
perm_2_1_0_3.push_back(2);
perm_2_1_0_3.push_back(1);
perm_2_1_0_3.push_back(0);
perm_2_1_0_3.push_back(3);
Permutation perm_2_3_0_1;
perm_2_3_0_1.push_back(2);
perm_2_3_0_1.push_back(3);
perm_2_3_0_1.push_back(0);
perm_2_3_0_1.push_back(1);
Permutation perm_2_3_1_0;
perm_2_3_1_0.push_back(2);
perm_2_3_1_0.push_back(3);
perm_2_3_1_0.push_back(1);
perm_2_3_1_0.push_back(0);
Permutation perm_3_0_1_2_4;
perm_3_0_1_2_4.push_back(3);
perm_3_0_1_2_4.push_back(0);
perm_3_0_1_2_4.push_back(1);
perm_3_0_1_2_4.push_back(2);
perm_3_0_1_2_4.push_back(4);
Permutation perm_3_1_0_2;
perm_3_1_0_2.push_back(3);
perm_3_1_0_2.push_back(1);
perm_3_1_0_2.push_back(0);
perm_3_1_0_2.push_back(2);
Permutation perm_3_2_0_1;
perm_3_2_0_1.push_back(3);
perm_3_2_0_1.push_back(2);
perm_3_2_0_1.push_back(0);
perm_3_2_0_1.push_back(1);
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
ModeArray modes_0_1_3;
modes_0_1_3.push_back(0);
modes_0_1_3.push_back(1);
modes_0_1_3.push_back(3);
ModeArray modes_0_1_3_2;
modes_0_1_3_2.push_back(0);
modes_0_1_3_2.push_back(1);
modes_0_1_3_2.push_back(3);
modes_0_1_3_2.push_back(2);
ModeArray modes_0_2;
modes_0_2.push_back(0);
modes_0_2.push_back(2);
ModeArray modes_0_3;
modes_0_3.push_back(0);
modes_0_3.push_back(3);
ModeArray modes_1;
modes_1.push_back(1);
ModeArray modes_1_0_3_2;
modes_1_0_3_2.push_back(1);
modes_1_0_3_2.push_back(0);
modes_1_0_3_2.push_back(3);
modes_1_0_3_2.push_back(2);
ModeArray modes_1_2_3;
modes_1_2_3.push_back(1);
modes_1_2_3.push_back(2);
modes_1_2_3.push_back(3);
ModeArray modes_1_3;
modes_1_3.push_back(1);
modes_1_3.push_back(3);
ModeArray modes_2;
modes_2.push_back(2);
ModeArray modes_2_3;
modes_2_3.push_back(2);
modes_2_3.push_back(3);
ModeArray modes_2_3_1;
modes_2_3_1.push_back(2);
modes_2_3_1.push_back(3);
modes_2_3_1.push_back(1);
ModeArray modes_3_1;
modes_3_1.push_back(3);
modes_3_1.push_back(1);
ModeArray modes_5_4;
modes_5_4.push_back(5);
modes_5_4.push_back(4);
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
IndexArray indices_bjaime( 6 );
indices_bjaime[0] = 'b';
indices_bjaime[1] = 'j';
indices_bjaime[2] = 'a';
indices_bjaime[3] = 'i';
indices_bjaime[4] = 'm';
indices_bjaime[5] = 'e';
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
	//G_mi_part0B[D01,D23]
DistTensor<double> G_mi_part0B__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//G_mi_part0T[D01,D23]
DistTensor<double> G_mi_part0T__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//G_mi_part0_0[D01,D23]
DistTensor<double> G_mi_part0_0__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//G_mi_part0_1[D01,D23]
DistTensor<double> G_mi_part0_1__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//G_mi_part0_1[*,D2]
DistTensor<double> G_mi_part0_1_perm10__D_2__S( dist__S__D_2, g );
G_mi_part0_1_perm10__D_2__S.SetLocalPermutation( perm_1_0 );
	//G_mi_part0_2[D01,D23]
DistTensor<double> G_mi_part0_2__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//P_ijmb[D0,D1,D2,D3]
DistTensor<double> P_ijmb__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//P_ijmb_part0B[D0,D1,D2,D3]
DistTensor<double> P_ijmb_part0B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//P_ijmb_part0T[D0,D1,D2,D3]
DistTensor<double> P_ijmb_part0T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//P_ijmb_part0_0[D0,D1,D2,D3]
DistTensor<double> P_ijmb_part0_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//P_ijmb_part0_1[D0,D1,D2,D3]
DistTensor<double> P_ijmb_part0_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//P_ijmb_part0_1_part2B[D0,D1,D2,D3]
DistTensor<double> P_ijmb_part0_1_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//P_ijmb_part0_1_part2T[D0,D1,D2,D3]
DistTensor<double> P_ijmb_part0_1_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//P_ijmb_part0_1_part2_0[D0,D1,D2,D3]
DistTensor<double> P_ijmb_part0_1_part2_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//P_ijmb_part0_1_part2_1[D0,D1,D2,D3]
DistTensor<double> P_ijmb_part0_1_part2_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//P_ijmb_part0_1_part2_1[D0,D3,D2,D1]
DistTensor<double> P_ijmb_part0_1_part2_1__D_0__D_3__D_2__D_1( dist__D_0__D_3__D_2__D_1, g );
	//P_ijmb_part0_1_part2_1[D2,D3,*,D1]
DistTensor<double> P_ijmb_part0_1_part2_1_perm0132__D_2__D_3__D_1__S( dist__D_2__D_3__S__D_1, g );
P_ijmb_part0_1_part2_1_perm0132__D_2__D_3__D_1__S.SetLocalPermutation( perm_0_1_3_2 );
	//P_ijmb_part0_1_part2_2[D0,D1,D2,D3]
DistTensor<double> P_ijmb_part0_1_part2_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//P_ijmb_part0_2[D0,D1,D2,D3]
DistTensor<double> P_ijmb_part0_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Q_mnij[D0,D1,D2,D3]
DistTensor<double> Q_mnij__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Q_mnij_part0B[D0,D1,D2,D3]
DistTensor<double> Q_mnij_part0B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Q_mnij_part0T[D0,D1,D2,D3]
DistTensor<double> Q_mnij_part0T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Q_mnij_part0_0[D0,D1,D2,D3]
DistTensor<double> Q_mnij_part0_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Q_mnij_part0_1[D0,D1,D2,D3]
DistTensor<double> Q_mnij_part0_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Q_mnij_part0_1_part1B[D0,D1,D2,D3]
DistTensor<double> Q_mnij_part0_1_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Q_mnij_part0_1_part1T[D0,D1,D2,D3]
DistTensor<double> Q_mnij_part0_1_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Q_mnij_part0_1_part1_0[D0,D1,D2,D3]
DistTensor<double> Q_mnij_part0_1_part1_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Q_mnij_part0_1_part1_1[D0,D1,D2,D3]
DistTensor<double> Q_mnij_part0_1_part1_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Q_mnij_part0_1_part1_1[*,*,D2,D3]
DistTensor<double> Q_mnij_part0_1_part1_1_perm2301__D_2__D_3__S__S( dist__S__S__D_2__D_3, g );
Q_mnij_part0_1_part1_1_perm2301__D_2__D_3__S__S.SetLocalPermutation( perm_2_3_0_1 );
	//Q_mnij_part0_1_part1_2[D0,D1,D2,D3]
DistTensor<double> Q_mnij_part0_1_part1_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Q_mnij_part0_2[D0,D1,D2,D3]
DistTensor<double> Q_mnij_part0_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//T_bfnj[D0,D1,D2,D3]
DistTensor<double> T_bfnj__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//T_bfnj_part1B[D0,D1,D2,D3]
DistTensor<double> T_bfnj_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//T_bfnj_part1T[D0,D1,D2,D3]
DistTensor<double> T_bfnj_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//T_bfnj_part1_0[D0,D1,D2,D3]
DistTensor<double> T_bfnj_part1_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//T_bfnj_part1_1[D0,D1,D2,D3]
DistTensor<double> T_bfnj_part1_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//T_bfnj_part1_1_part2B[D0,D1,D2,D3]
DistTensor<double> T_bfnj_part1_1_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//T_bfnj_part1_1_part2T[D0,D1,D2,D3]
DistTensor<double> T_bfnj_part1_1_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//T_bfnj_part1_1_part2_0[D0,D1,D2,D3]
DistTensor<double> T_bfnj_part1_1_part2_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//T_bfnj_part1_1_part2_1[D0,D1,D2,D3]
DistTensor<double> T_bfnj_part1_1_part2_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//T_bfnj_part1_1_part2_1[D0,D1,D3,D2]
DistTensor<double> T_bfnj_part1_1_part2_1__D_0__D_1__D_3__D_2( dist__D_0__D_1__D_3__D_2, g );
	//T_bfnj_part1_1_part2_1[D0,*,*,D2]
DistTensor<double> T_bfnj_part1_1_part2_1_perm2103__S__S__D_0__D_2( dist__D_0__S__S__D_2, g );
T_bfnj_part1_1_part2_1_perm2103__S__S__D_0__D_2.SetLocalPermutation( perm_2_1_0_3 );
	//T_bfnj_part1_1_part2_2[D0,D1,D2,D3]
DistTensor<double> T_bfnj_part1_1_part2_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//T_bfnj_part1_2[D0,D1,D2,D3]
DistTensor<double> T_bfnj_part1_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//T_bfnj_part2B[D0,D1,D2,D3]
DistTensor<double> T_bfnj_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//T_bfnj_part2T[D0,D1,D2,D3]
DistTensor<double> T_bfnj_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//T_bfnj_part2_0[D0,D1,D2,D3]
DistTensor<double> T_bfnj_part2_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//T_bfnj_part2_1[D0,D1,D2,D3]
DistTensor<double> T_bfnj_part2_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//T_bfnj_part2_1_part3B[D0,D1,D2,D3]
DistTensor<double> T_bfnj_part2_1_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//T_bfnj_part2_1_part3T[D0,D1,D2,D3]
DistTensor<double> T_bfnj_part2_1_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//T_bfnj_part2_1_part3_0[D0,D1,D2,D3]
DistTensor<double> T_bfnj_part2_1_part3_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//T_bfnj_part2_1_part3_1[D0,D1,D2,D3]
DistTensor<double> T_bfnj_part2_1_part3_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//T_bfnj_part2_1_part3_1[*,D1,D2,D3]
DistTensor<double> T_bfnj_part2_1_part3_1__S__D_1__D_2__D_3( dist__S__D_1__D_2__D_3, g );
	//T_bfnj_part2_1_part3_2[D0,D1,D2,D3]
DistTensor<double> T_bfnj_part2_1_part3_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//T_bfnj_part2_2[D0,D1,D2,D3]
DistTensor<double> T_bfnj_part2_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//T_bfnj_part3B[D0,D1,D2,D3]
DistTensor<double> T_bfnj_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//T_bfnj_part3T[D0,D1,D2,D3]
DistTensor<double> T_bfnj_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//T_bfnj_part3_0[D0,D1,D2,D3]
DistTensor<double> T_bfnj_part3_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//T_bfnj_part3_1[D0,D1,D2,D3]
DistTensor<double> T_bfnj_part3_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//T_bfnj_part3_1_part2B[D0,D1,D2,D3]
DistTensor<double> T_bfnj_part3_1_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//T_bfnj_part3_1_part2T[D0,D1,D2,D3]
DistTensor<double> T_bfnj_part3_1_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//T_bfnj_part3_1_part2_0[D0,D1,D2,D3]
DistTensor<double> T_bfnj_part3_1_part2_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//T_bfnj_part3_1_part2_1[D0,D1,D2,D3]
DistTensor<double> T_bfnj_part3_1_part2_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//T_bfnj_part3_1_part2_1[D0,D1,D3,D2]
DistTensor<double> T_bfnj_part3_1_part2_1__D_0__D_1__D_3__D_2( dist__D_0__D_1__D_3__D_2, g );
	//T_bfnj_part3_1_part2_1[D0,D1,*,D3]
DistTensor<double> T_bfnj_part3_1_part2_1_perm2013__S__D_0__D_1__D_3( dist__D_0__D_1__S__D_3, g );
T_bfnj_part3_1_part2_1_perm2013__S__D_0__D_1__D_3.SetLocalPermutation( perm_2_0_1_3 );
	//T_bfnj_part3_1_part2_2[D0,D1,D2,D3]
DistTensor<double> T_bfnj_part3_1_part2_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//T_bfnj_part3_2[D0,D1,D2,D3]
DistTensor<double> T_bfnj_part3_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Tau_efmn[D0,D1,D2,D3]
DistTensor<double> Tau_efmn__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Tau_efmn_part2B[D0,D1,D2,D3]
DistTensor<double> Tau_efmn_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Tau_efmn_part2T[D0,D1,D2,D3]
DistTensor<double> Tau_efmn_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Tau_efmn_part2_0[D0,D1,D2,D3]
DistTensor<double> Tau_efmn_part2_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Tau_efmn_part2_1[D0,D1,D2,D3]
DistTensor<double> Tau_efmn_part2_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Tau_efmn_part2_1_part3B[D0,D1,D2,D3]
DistTensor<double> Tau_efmn_part2_1_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Tau_efmn_part2_1_part3T[D0,D1,D2,D3]
DistTensor<double> Tau_efmn_part2_1_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Tau_efmn_part2_1_part3_0[D0,D1,D2,D3]
DistTensor<double> Tau_efmn_part2_1_part3_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Tau_efmn_part2_1_part3_1[D0,D1,D2,D3]
DistTensor<double> Tau_efmn_part2_1_part3_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Tau_efmn_part2_1_part3_1[D2,D1,D0,D3]
DistTensor<double> Tau_efmn_part2_1_part3_1__D_2__D_1__D_0__D_3( dist__D_2__D_1__D_0__D_3, g );
	//Tau_efmn_part2_1_part3_1[D2,D3,D0,D1]
DistTensor<double> Tau_efmn_part2_1_part3_1__D_2__D_3__D_0__D_1( dist__D_2__D_3__D_0__D_1, g );
	//Tau_efmn_part2_1_part3_1[D2,D3,*,*]
DistTensor<double> Tau_efmn_part2_1_part3_1__D_2__D_3__S__S( dist__D_2__D_3__S__S, g );
	//Tau_efmn_part2_1_part3_1[D0,D1,*,*]
DistTensor<double> Tau_efmn_part2_1_part3_1_perm2301__S__S__D_0__D_1( dist__D_0__D_1__S__S, g );
Tau_efmn_part2_1_part3_1_perm2301__S__S__D_0__D_1.SetLocalPermutation( perm_2_3_0_1 );
	//Tau_efmn_part2_1_part3_2[D0,D1,D2,D3]
DistTensor<double> Tau_efmn_part2_1_part3_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Tau_efmn_part2_2[D0,D1,D2,D3]
DistTensor<double> Tau_efmn_part2_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_bmje[D0,D1,D2,D3]
DistTensor<double> W_bmje__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_bmje_part0B[D0,D1,D2,D3]
DistTensor<double> W_bmje_part0B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_bmje_part0T[D0,D1,D2,D3]
DistTensor<double> W_bmje_part0T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_bmje_part0_0[D0,D1,D2,D3]
DistTensor<double> W_bmje_part0_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_bmje_part0_1[D0,D1,D2,D3]
DistTensor<double> W_bmje_part0_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_bmje_part0_1_part2B[D0,D1,D2,D3]
DistTensor<double> W_bmje_part0_1_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_bmje_part0_1_part2T[D0,D1,D2,D3]
DistTensor<double> W_bmje_part0_1_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_bmje_part0_1_part2_0[D0,D1,D2,D3]
DistTensor<double> W_bmje_part0_1_part2_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_bmje_part0_1_part2_1[D0,D1,D2,D3]
DistTensor<double> W_bmje_part0_1_part2_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_bmje_part0_1_part2_1[D0,D3,D2,D1]
DistTensor<double> W_bmje_part0_1_part2_1__D_0__D_3__D_2__D_1( dist__D_0__D_3__D_2__D_1, g );
	//W_bmje_part0_1_part2_1[*,D3,*,D1]
DistTensor<double> W_bmje_part0_1_part2_1_perm0213__S__S__D_3__D_1( dist__S__D_3__S__D_1, g );
W_bmje_part0_1_part2_1_perm0213__S__S__D_3__D_1.SetLocalPermutation( perm_0_2_1_3 );
	//W_bmje_part0_1_part2_2[D0,D1,D2,D3]
DistTensor<double> W_bmje_part0_1_part2_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_bmje_part0_2[D0,D1,D2,D3]
DistTensor<double> W_bmje_part0_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//X_bmej[D0,D1,D2,D3]
DistTensor<double> X_bmej__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//X_bmej_part2B[D0,D1,D2,D3]
DistTensor<double> X_bmej_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//X_bmej_part2T[D0,D1,D2,D3]
DistTensor<double> X_bmej_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//X_bmej_part2_0[D0,D1,D2,D3]
DistTensor<double> X_bmej_part2_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//X_bmej_part2_1[D0,D1,D2,D3]
DistTensor<double> X_bmej_part2_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//X_bmej_part2_1_part1B[D0,D1,D2,D3]
DistTensor<double> X_bmej_part2_1_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//X_bmej_part2_1_part1T[D0,D1,D2,D3]
DistTensor<double> X_bmej_part2_1_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//X_bmej_part2_1_part1_0[D0,D1,D2,D3]
DistTensor<double> X_bmej_part2_1_part1_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//X_bmej_part2_1_part1_1[D0,D1,D2,D3]
DistTensor<double> X_bmej_part2_1_part1_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//X_bmej_part2_1_part1_1[D1,D0,D2,D3]
DistTensor<double> X_bmej_part2_1_part1_1__D_1__D_0__D_2__D_3( dist__D_1__D_0__D_2__D_3, g );
	//X_bmej_part2_1_part1_1[D1,*,*,D3]
DistTensor<double> X_bmej_part2_1_part1_1_perm0312__D_1__D_3__S__S( dist__D_1__S__S__D_3, g );
X_bmej_part2_1_part1_1_perm0312__D_1__D_3__S__S.SetLocalPermutation( perm_0_3_1_2 );
	//X_bmej_part2_1_part1_2[D0,D1,D2,D3]
DistTensor<double> X_bmej_part2_1_part1_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//X_bmej_part2_2[D0,D1,D2,D3]
DistTensor<double> X_bmej_part2_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Z_abij[D0,D1,D2,D3]
DistTensor<double> Z_abij__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Z_abij_part2B[D0,D1,D2,D3]
DistTensor<double> Z_abij_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Z_abij_part2T[D0,D1,D2,D3]
DistTensor<double> Z_abij_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Z_abij_part2_0[D0,D1,D2,D3]
DistTensor<double> Z_abij_part2_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Z_abij_part2_1[D0,D1,D2,D3]
DistTensor<double> Z_abij_part2_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Z_abij_part2_1_part3B[D0,D1,D2,D3]
DistTensor<double> Z_abij_part2_1_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Z_abij_part2_1_part3T[D0,D1,D2,D3]
DistTensor<double> Z_abij_part2_1_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Z_abij_part2_1_part3_0[D0,D1,D2,D3]
DistTensor<double> Z_abij_part2_1_part3_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Z_abij_part2_1_part3_1[D0,D1,D2,D3]
DistTensor<double> Z_abij_part2_1_part3_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Z_abij_part2_1_part3_1[D0,D1,*,*,D2,D3]
DistTensor<double> Z_abij_part2_1_part3_1__D_0__D_1__S__S__D_2__D_3( dist__D_0__D_1__S__S__D_2__D_3, g );
	//Z_abij_part2_1_part3_2[D0,D1,D2,D3]
DistTensor<double> Z_abij_part2_1_part3_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Z_abij_part2_2[D0,D1,D2,D3]
DistTensor<double> Z_abij_part2_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Z_abij[D0,D1,D2,D3]
DistTensor<double> Z_abij_perm2301__D_2__D_3__D_0__D_1( dist__D_0__D_1__D_2__D_3, g );
Z_abij_perm2301__D_2__D_3__D_0__D_1.SetLocalPermutation( perm_2_3_0_1 );
	//accum[D0,D1,D2,D3]
DistTensor<double> accum__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//accum_part1B[D0,D1,D2,D3]
DistTensor<double> accum_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//accum_part1T[D0,D1,D2,D3]
DistTensor<double> accum_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//accum_part1_0[D0,D1,D2,D3]
DistTensor<double> accum_part1_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//accum_part1_1[D0,D1,D2,D3]
DistTensor<double> accum_part1_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//accum_part1_1_part3B[D0,D1,D2,D3]
DistTensor<double> accum_part1_1_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//accum_part1_1_part3T[D0,D1,D2,D3]
DistTensor<double> accum_part1_1_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//accum_part1_1_part3_0[D0,D1,D2,D3]
DistTensor<double> accum_part1_1_part3_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//accum_part1_1_part3_1[D0,D1,D2,D3]
DistTensor<double> accum_part1_1_part3_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//accum_part1_1_part3_1[D0,*,D2,*,D1,D3]
DistTensor<double> accum_part1_1_part3_1_perm130254__S__S__D_0__D_2__D_3__D_1( dist__D_0__S__D_2__S__D_1__D_3, g );
accum_part1_1_part3_1_perm130254__S__S__D_0__D_2__D_3__D_1.SetLocalPermutation( perm_1_3_0_2_5_4 );
	//accum_part1_1_part3_2[D0,D1,D2,D3]
DistTensor<double> accum_part1_1_part3_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//accum_part1_2[D0,D1,D2,D3]
DistTensor<double> accum_part1_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//accum_part2B[D0,D1,D2,D3]
DistTensor<double> accum_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//accum_part2T[D0,D1,D2,D3]
DistTensor<double> accum_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//accum_part2_0[D0,D1,D2,D3]
DistTensor<double> accum_part2_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//accum_part2_1[D0,D1,D2,D3]
DistTensor<double> accum_part2_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//accum_part2_1_part3B[D0,D1,D2,D3]
DistTensor<double> accum_part2_1_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//accum_part2_1_part3T[D0,D1,D2,D3]
DistTensor<double> accum_part2_1_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//accum_part2_1_part3_0[D0,D1,D2,D3]
DistTensor<double> accum_part2_1_part3_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//accum_part2_1_part3_1[D0,D1,D2,D3]
DistTensor<double> accum_part2_1_part3_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//accum_part2_1_part3_1[D2,D3,*,D1,D0]
DistTensor<double> accum_part2_1_part3_1_perm30124__D_1__D_2__D_3__S__D_0( dist__D_2__D_3__S__D_1__D_0, g );
accum_part2_1_part3_1_perm30124__D_1__D_2__D_3__S__D_0.SetLocalPermutation( perm_3_0_1_2_4 );
	//accum_part2_1_part3_1_temp[D0,D1,D2,D3]
DistTensor<double> accum_part2_1_part3_1_temp__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//accum_part2_1_part3_1_temp[D20,D1,*,D3]
DistTensor<double> accum_part2_1_part3_1_temp__D_2_0__D_1__S__D_3( dist__D_2_0__D_1__S__D_3, g );
	//accum_part2_1_part3_1_temp[D20,D3,*,D1]
DistTensor<double> accum_part2_1_part3_1_temp__D_2_0__D_3__S__D_1( dist__D_2_0__D_3__S__D_1, g );
	//accum_part2_1_part3_2[D0,D1,D2,D3]
DistTensor<double> accum_part2_1_part3_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//accum_part2_1[D0,D1,D2,D3]
DistTensor<double> accum_part2_1_perm2310__D_2__D_3__D_1__D_0( dist__D_0__D_1__D_2__D_3, g );
accum_part2_1_perm2310__D_2__D_3__D_1__D_0.SetLocalPermutation( perm_2_3_1_0 );
	//accum_part2_2[D0,D1,D2,D3]
DistTensor<double> accum_part2_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//accum_part3B[D0,D1,D2,D3]
DistTensor<double> accum_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//accum_part3T[D0,D1,D2,D3]
DistTensor<double> accum_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//accum_part3_0[D0,D1,D2,D3]
DistTensor<double> accum_part3_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//accum_part3_1[D0,D1,D2,D3]
DistTensor<double> accum_part3_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//accum_part3_1_part2B[D0,D1,D2,D3]
DistTensor<double> accum_part3_1_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//accum_part3_1_part2T[D0,D1,D2,D3]
DistTensor<double> accum_part3_1_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//accum_part3_1_part2_0[D0,D1,D2,D3]
DistTensor<double> accum_part3_1_part2_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//accum_part3_1_part2_1[D0,D1,D2,D3]
DistTensor<double> accum_part3_1_part2_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//accum_part3_1_part2_1[D0,D1,D3,D2]
DistTensor<double> accum_part3_1_part2_1__D_0__D_1__D_3__D_2( dist__D_0__D_1__D_3__D_2, g );
	//accum_part3_1_part2_1[D1,D0,D3,D2]
DistTensor<double> accum_part3_1_part2_1__D_1__D_0__D_3__D_2( dist__D_1__D_0__D_3__D_2, g );
	//accum_part3_1_part2_2[D0,D1,D2,D3]
DistTensor<double> accum_part3_1_part2_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//accum_part3_1[D0,D1,D2,D3]
DistTensor<double> accum_part3_1_perm2013__D_2__D_0__D_1__D_3( dist__D_0__D_1__D_2__D_3, g );
accum_part3_1_perm2013__D_2__D_0__D_1__D_3.SetLocalPermutation( perm_2_0_1_3 );
	//accum_part3_2[D0,D1,D2,D3]
DistTensor<double> accum_part3_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//r_bmfe[D0,D1,D2,D3]
DistTensor<double> r_bmfe__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//r_bmfe_part1B[D0,D1,D2,D3]
DistTensor<double> r_bmfe_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//r_bmfe_part1T[D0,D1,D2,D3]
DistTensor<double> r_bmfe_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//r_bmfe_part1_0[D0,D1,D2,D3]
DistTensor<double> r_bmfe_part1_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//r_bmfe_part1_1[D0,D1,D2,D3]
DistTensor<double> r_bmfe_part1_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//r_bmfe_part1_1[D0,D1,D2,D3]
DistTensor<double> r_bmfe_part1_1_perm1230__D_1__D_2__D_3__D_0( dist__D_0__D_1__D_2__D_3, g );
r_bmfe_part1_1_perm1230__D_1__D_2__D_3__D_0.SetLocalPermutation( perm_1_2_3_0 );
	//r_bmfe_part1_2[D0,D1,D2,D3]
DistTensor<double> r_bmfe_part1_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//t_fj[D01,D23]
DistTensor<double> t_fj__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//t_fj_part1B[D01,D23]
DistTensor<double> t_fj_part1B__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//t_fj_part1T[D01,D23]
DistTensor<double> t_fj_part1T__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//t_fj_part1_0[D01,D23]
DistTensor<double> t_fj_part1_0__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//t_fj_part1_1[D01,D23]
DistTensor<double> t_fj_part1_1__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//t_fj_part1_1[D0,*]
DistTensor<double> t_fj_part1_1__D_0__S( dist__D_0__S, g );
	//t_fj_part1_1[D0,*]
DistTensor<double> t_fj_part1_1_perm10__S__D_0( dist__D_0__S, g );
t_fj_part1_1_perm10__S__D_0.SetLocalPermutation( perm_1_0 );
	//t_fj_part1_2[D01,D23]
DistTensor<double> t_fj_part1_2__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//temp1[D0,D1,D2,D3]
DistTensor<double> temp1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp1_part2B[D0,D1,D2,D3]
DistTensor<double> temp1_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp1_part2T[D0,D1,D2,D3]
DistTensor<double> temp1_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp1_part2_0[D0,D1,D2,D3]
DistTensor<double> temp1_part2_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp1_part2_1[D0,D1,D2,D3]
DistTensor<double> temp1_part2_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp1_part2_1_part3B[D0,D1,D2,D3]
DistTensor<double> temp1_part2_1_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp1_part2_1_part3T[D0,D1,D2,D3]
DistTensor<double> temp1_part2_1_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp1_part2_1_part3_0[D0,D1,D2,D3]
DistTensor<double> temp1_part2_1_part3_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp1_part2_1_part3_1[D0,D1,D2,D3]
DistTensor<double> temp1_part2_1_part3_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp1_part2_1_part3_2[D0,D1,D2,D3]
DistTensor<double> temp1_part2_1_part3_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp1_part2_2[D0,D1,D2,D3]
DistTensor<double> temp1_part2_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp1_part3B[D0,D1,D2,D3]
DistTensor<double> temp1_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp1_part3T[D0,D1,D2,D3]
DistTensor<double> temp1_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp1_part3_0[D0,D1,D2,D3]
DistTensor<double> temp1_part3_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp1_part3_1[D0,D1,D2,D3]
DistTensor<double> temp1_part3_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp1_part3_1_part2B[D0,D1,D2,D3]
DistTensor<double> temp1_part3_1_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp1_part3_1_part2T[D0,D1,D2,D3]
DistTensor<double> temp1_part3_1_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp1_part3_1_part2_0[D0,D1,D2,D3]
DistTensor<double> temp1_part3_1_part2_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp1_part3_1_part2_1[D0,D1,D2,D3]
DistTensor<double> temp1_part3_1_part2_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp1_part3_1_part2_1[D0,D1,D3,D2]
DistTensor<double> temp1_part3_1_part2_1__D_0__D_1__D_3__D_2( dist__D_0__D_1__D_3__D_2, g );
	//temp1_part3_1_part2_2[D0,D1,D2,D3]
DistTensor<double> temp1_part3_1_part2_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp1_part3_2[D0,D1,D2,D3]
DistTensor<double> temp1_part3_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp1[D0,D1,D2,D3]
DistTensor<double> temp1_perm1302__D_1__D_3__D_0__D_2( dist__D_0__D_1__D_2__D_3, g );
temp1_perm1302__D_1__D_3__D_0__D_2.SetLocalPermutation( perm_1_3_0_2 );
	//temp2[D0,D1,D2,D3]
DistTensor<double> temp2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp2_part2B[D0,D1,D2,D3]
DistTensor<double> temp2_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp2_part2T[D0,D1,D2,D3]
DistTensor<double> temp2_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp2_part2_0[D0,D1,D2,D3]
DistTensor<double> temp2_part2_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp2_part2_1[D0,D1,D2,D3]
DistTensor<double> temp2_part2_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp2_part2_1_part3B[D0,D1,D2,D3]
DistTensor<double> temp2_part2_1_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp2_part2_1_part3T[D0,D1,D2,D3]
DistTensor<double> temp2_part2_1_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp2_part2_1_part3_0[D0,D1,D2,D3]
DistTensor<double> temp2_part2_1_part3_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp2_part2_1_part3_1[D0,D1,D2,D3]
DistTensor<double> temp2_part2_1_part3_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp2_part2_1_part3_2[D0,D1,D2,D3]
DistTensor<double> temp2_part2_1_part3_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp2_part2_2[D0,D1,D2,D3]
DistTensor<double> temp2_part2_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp2[D0,D1,D2,D3]
DistTensor<double> temp2_perm3102__D_3__D_1__D_0__D_2( dist__D_0__D_1__D_2__D_3, g );
temp2_perm3102__D_3__D_1__D_0__D_2.SetLocalPermutation( perm_3_1_0_2 );
	//v_femn[D0,D1,D2,D3]
DistTensor<double> v_femn__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//y_abef[D0,D1,D2,D3]
DistTensor<double> y_abef__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
// v_femn has 4 dims
//	Starting distribution: [D0,D1,D2,D3] or _D_0__D_1__D_2__D_3
ObjShape v_femn__D_0__D_1__D_2__D_3_tempShape;
v_femn__D_0__D_1__D_2__D_3_tempShape.push_back( n_v );
v_femn__D_0__D_1__D_2__D_3_tempShape.push_back( n_v );
v_femn__D_0__D_1__D_2__D_3_tempShape.push_back( n_o );
v_femn__D_0__D_1__D_2__D_3_tempShape.push_back( n_o );
v_femn__D_0__D_1__D_2__D_3.ResizeTo( v_femn__D_0__D_1__D_2__D_3_tempShape );
MakeUniform( v_femn__D_0__D_1__D_2__D_3 );
// Q_mnij has 4 dims
//	Starting distribution: [D0,D1,D2,D3] or _D_0__D_1__D_2__D_3
ObjShape Q_mnij__D_0__D_1__D_2__D_3_tempShape;
Q_mnij__D_0__D_1__D_2__D_3_tempShape.push_back( n_o );
Q_mnij__D_0__D_1__D_2__D_3_tempShape.push_back( n_o );
Q_mnij__D_0__D_1__D_2__D_3_tempShape.push_back( n_o );
Q_mnij__D_0__D_1__D_2__D_3_tempShape.push_back( n_o );
Q_mnij__D_0__D_1__D_2__D_3.ResizeTo( Q_mnij__D_0__D_1__D_2__D_3_tempShape );
MakeUniform( Q_mnij__D_0__D_1__D_2__D_3 );
// Tau_efmn has 4 dims
//	Starting distribution: [D0,D1,D2,D3] or _D_0__D_1__D_2__D_3
ObjShape Tau_efmn__D_0__D_1__D_2__D_3_tempShape;
Tau_efmn__D_0__D_1__D_2__D_3_tempShape.push_back( n_v );
Tau_efmn__D_0__D_1__D_2__D_3_tempShape.push_back( n_v );
Tau_efmn__D_0__D_1__D_2__D_3_tempShape.push_back( n_o );
Tau_efmn__D_0__D_1__D_2__D_3_tempShape.push_back( n_o );
Tau_efmn__D_0__D_1__D_2__D_3.ResizeTo( Tau_efmn__D_0__D_1__D_2__D_3_tempShape );
MakeUniform( Tau_efmn__D_0__D_1__D_2__D_3 );
// y_abef has 4 dims
//	Starting distribution: [D0,D1,D2,D3] or _D_0__D_1__D_2__D_3
ObjShape y_abef__D_0__D_1__D_2__D_3_tempShape;
y_abef__D_0__D_1__D_2__D_3_tempShape.push_back( n_v );
y_abef__D_0__D_1__D_2__D_3_tempShape.push_back( n_v );
y_abef__D_0__D_1__D_2__D_3_tempShape.push_back( n_v );
y_abef__D_0__D_1__D_2__D_3_tempShape.push_back( n_v );
y_abef__D_0__D_1__D_2__D_3.ResizeTo( y_abef__D_0__D_1__D_2__D_3_tempShape );
MakeUniform( y_abef__D_0__D_1__D_2__D_3 );
// r_bmfe has 4 dims
//	Starting distribution: [D0,D1,D2,D3] or _D_0__D_1__D_2__D_3
ObjShape r_bmfe__D_0__D_1__D_2__D_3_tempShape;
r_bmfe__D_0__D_1__D_2__D_3_tempShape.push_back( n_v );
r_bmfe__D_0__D_1__D_2__D_3_tempShape.push_back( n_o );
r_bmfe__D_0__D_1__D_2__D_3_tempShape.push_back( n_v );
r_bmfe__D_0__D_1__D_2__D_3_tempShape.push_back( n_v );
r_bmfe__D_0__D_1__D_2__D_3.ResizeTo( r_bmfe__D_0__D_1__D_2__D_3_tempShape );
MakeUniform( r_bmfe__D_0__D_1__D_2__D_3 );
// t_fj has 2 dims
//	Starting distribution: [D01,D23] or _D_0_1__D_2_3
ObjShape t_fj__D_0_1__D_2_3_tempShape;
t_fj__D_0_1__D_2_3_tempShape.push_back( n_v );
t_fj__D_0_1__D_2_3_tempShape.push_back( n_o );
t_fj__D_0_1__D_2_3.ResizeTo( t_fj__D_0_1__D_2_3_tempShape );
MakeUniform( t_fj__D_0_1__D_2_3 );
// P_ijmb has 4 dims
//	Starting distribution: [D0,D1,D2,D3] or _D_0__D_1__D_2__D_3
ObjShape P_ijmb__D_0__D_1__D_2__D_3_tempShape;
P_ijmb__D_0__D_1__D_2__D_3_tempShape.push_back( n_o );
P_ijmb__D_0__D_1__D_2__D_3_tempShape.push_back( n_o );
P_ijmb__D_0__D_1__D_2__D_3_tempShape.push_back( n_o );
P_ijmb__D_0__D_1__D_2__D_3_tempShape.push_back( n_v );
P_ijmb__D_0__D_1__D_2__D_3.ResizeTo( P_ijmb__D_0__D_1__D_2__D_3_tempShape );
MakeUniform( P_ijmb__D_0__D_1__D_2__D_3 );
// F_ae has 2 dims
//	Starting distribution: [D01,D23] or _D_0_1__D_2_3
ObjShape F_ae__D_0_1__D_2_3_tempShape;
F_ae__D_0_1__D_2_3_tempShape.push_back( n_v );
F_ae__D_0_1__D_2_3_tempShape.push_back( n_v );
F_ae__D_0_1__D_2_3.ResizeTo( F_ae__D_0_1__D_2_3_tempShape );
MakeUniform( F_ae__D_0_1__D_2_3 );
// T_bfnj has 4 dims
//	Starting distribution: [D0,D1,D2,D3] or _D_0__D_1__D_2__D_3
ObjShape T_bfnj__D_0__D_1__D_2__D_3_tempShape;
T_bfnj__D_0__D_1__D_2__D_3_tempShape.push_back( n_v );
T_bfnj__D_0__D_1__D_2__D_3_tempShape.push_back( n_v );
T_bfnj__D_0__D_1__D_2__D_3_tempShape.push_back( n_o );
T_bfnj__D_0__D_1__D_2__D_3_tempShape.push_back( n_o );
T_bfnj__D_0__D_1__D_2__D_3.ResizeTo( T_bfnj__D_0__D_1__D_2__D_3_tempShape );
MakeUniform( T_bfnj__D_0__D_1__D_2__D_3 );
// G_mi has 2 dims
//	Starting distribution: [D01,D23] or _D_0_1__D_2_3
ObjShape G_mi__D_0_1__D_2_3_tempShape;
G_mi__D_0_1__D_2_3_tempShape.push_back( n_o );
G_mi__D_0_1__D_2_3_tempShape.push_back( n_o );
G_mi__D_0_1__D_2_3.ResizeTo( G_mi__D_0_1__D_2_3_tempShape );
MakeUniform( G_mi__D_0_1__D_2_3 );
// W_bmje has 4 dims
//	Starting distribution: [D0,D1,D2,D3] or _D_0__D_1__D_2__D_3
ObjShape W_bmje__D_0__D_1__D_2__D_3_tempShape;
W_bmje__D_0__D_1__D_2__D_3_tempShape.push_back( n_v );
W_bmje__D_0__D_1__D_2__D_3_tempShape.push_back( n_o );
W_bmje__D_0__D_1__D_2__D_3_tempShape.push_back( n_o );
W_bmje__D_0__D_1__D_2__D_3_tempShape.push_back( n_v );
W_bmje__D_0__D_1__D_2__D_3.ResizeTo( W_bmje__D_0__D_1__D_2__D_3_tempShape );
MakeUniform( W_bmje__D_0__D_1__D_2__D_3 );
tempShape = T_bfnj__D_0__D_1__D_2__D_3.Shape();
temp2__D_0__D_1__D_2__D_3.ResizeTo( tempShape );
// X_bmej has 4 dims
//	Starting distribution: [D0,D1,D2,D3] or _D_0__D_1__D_2__D_3
ObjShape X_bmej__D_0__D_1__D_2__D_3_tempShape;
X_bmej__D_0__D_1__D_2__D_3_tempShape.push_back( n_v );
X_bmej__D_0__D_1__D_2__D_3_tempShape.push_back( n_o );
X_bmej__D_0__D_1__D_2__D_3_tempShape.push_back( n_v );
X_bmej__D_0__D_1__D_2__D_3_tempShape.push_back( n_o );
X_bmej__D_0__D_1__D_2__D_3.ResizeTo( X_bmej__D_0__D_1__D_2__D_3_tempShape );
MakeUniform( X_bmej__D_0__D_1__D_2__D_3 );
// Z_abij has 4 dims
//	Starting distribution: [D0,D1,D2,D3] or _D_0__D_1__D_2__D_3
ObjShape Z_abij__D_0__D_1__D_2__D_3_tempShape;
Z_abij__D_0__D_1__D_2__D_3_tempShape.push_back( n_v );
Z_abij__D_0__D_1__D_2__D_3_tempShape.push_back( n_v );
Z_abij__D_0__D_1__D_2__D_3_tempShape.push_back( n_o );
Z_abij__D_0__D_1__D_2__D_3_tempShape.push_back( n_o );
Z_abij__D_0__D_1__D_2__D_3.ResizeTo( Z_abij__D_0__D_1__D_2__D_3_tempShape );
MakeUniform( Z_abij__D_0__D_1__D_2__D_3 );
tempShape = Z_abij__D_0__D_1__D_2__D_3.Shape();
accum__D_0__D_1__D_2__D_3.ResizeTo( tempShape );
tempShape = Z_abij__D_0__D_1__D_2__D_3.Shape();
temp1__D_0__D_1__D_2__D_3.ResizeTo( tempShape );
//**** (out of 1)

//******************************
//* Load tensors
//******************************
////////////////////////////////
//Performance testing
////////////////////////////////
std::stringstream fullName;
#ifdef CORRECTNESS
DistTensor<T> check(dist__D_0__D_1__D_2__D_3, g);
check.ResizeTo(Z_abij__D_0__D_1__D_2__D_3.Shape());
Read(v_femn__D_0__D_1__D_2__D_3, "ccsd_terms/term_v_small", BINARY_FLAT, false);
Read(y_abef__D_0__D_1__D_2__D_3, "ccsd_terms/term_y_small", BINARY_FLAT, false);
Read(r_bmfe__D_0__D_1__D_2__D_3, "ccsd_terms/term_r_small", BINARY_FLAT, false);
fullName.str("");
fullName.clear();
fullName << "ccsd_terms/term_Q_iter" << testIter;
Read(Q_mnij__D_0__D_1__D_2__D_3, fullName.str(), BINARY_FLAT, false);
fullName.str("");
fullName.clear();
fullName << "ccsd_terms/term_Tau_iter" << testIter;
Read(Tau_efmn__D_0__D_1__D_2__D_3, fullName.str(), BINARY_FLAT, false);
fullName.str("");
fullName.clear();
fullName << "ccsd_terms/term_t_small_iter" << testIter;
Read(t_fj__D_0_1__D_2_3, fullName.str(), BINARY_FLAT, false);
fullName.str("");
fullName.clear();
fullName << "ccsd_terms/term_P_iter" << testIter;
Read(P_ijmb__D_0__D_1__D_2__D_3, fullName.str(), BINARY_FLAT, false);
fullName.str("");
fullName.clear();
fullName << "ccsd_terms/term_F_iter" << testIter;
Read(F_ae__D_0_1__D_2_3, fullName.str(), BINARY_FLAT, false);
fullName.str("");
fullName.clear();
fullName << "ccsd_terms/term_T_iter" << testIter;
Read(T_bfnj__D_0__D_1__D_2__D_3, fullName.str(), BINARY_FLAT, false);
fullName.str("");
fullName.clear();
fullName << "ccsd_terms/term_G_iter" << testIter;
Read(G_mi__D_0_1__D_2_3, fullName.str(), BINARY_FLAT, false);
fullName.str("");
fullName.clear();
fullName << "ccsd_terms/term_W_iter" << testIter;
Read(W_bmje__D_0__D_1__D_2__D_3, fullName.str(), BINARY_FLAT, false);
fullName.str("");
fullName.clear();
fullName << "ccsd_terms/term_X_iter" << testIter;
Read(X_bmej__D_0__D_1__D_2__D_3, fullName.str(), BINARY_FLAT, false);
fullName.str("");
fullName.clear();
fullName << "ccsd_terms/term_Z_iter" << testIter;
Read(check, fullName.str(), BINARY_FLAT, false);
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


	tempShape = temp1__D_0__D_1__D_2__D_3.Shape();
	temp1_perm1302__D_1__D_3__D_0__D_2.ResizeTo( tempShape );
	Scal( 0.0, temp1_perm1302__D_1__D_3__D_0__D_2 );
	//**** (out of 1)
	//**** Is real	0 shadows
		//Outputs:
		//  temp2__D_0__D_1__D_2__D_3
	PartitionDown(T_bfnj__D_0__D_1__D_2__D_3, T_bfnj_part2T__D_0__D_1__D_2__D_3, T_bfnj_part2B__D_0__D_1__D_2__D_3, 2, 0);
	PartitionDown(T_bfnj__D_0__D_1__D_2__D_3, T_bfnj_part3T__D_0__D_1__D_2__D_3, T_bfnj_part3B__D_0__D_1__D_2__D_3, 3, 0);
	PartitionDown(temp2__D_0__D_1__D_2__D_3, temp2_part2T__D_0__D_1__D_2__D_3, temp2_part2B__D_0__D_1__D_2__D_3, 2, 0);
	while(temp2_part2T__D_0__D_1__D_2__D_3.Dimension(2) < temp2__D_0__D_1__D_2__D_3.Dimension(2))
	{
		RepartitionDown
		( T_bfnj_part2T__D_0__D_1__D_2__D_3,  T_bfnj_part2_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       T_bfnj_part2_1__D_0__D_1__D_2__D_3,
		  T_bfnj_part2B__D_0__D_1__D_2__D_3, T_bfnj_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );
		RepartitionDown
		( T_bfnj_part3T__D_0__D_1__D_2__D_3,  T_bfnj_part3_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       T_bfnj_part3_1__D_0__D_1__D_2__D_3,
		  T_bfnj_part3B__D_0__D_1__D_2__D_3, T_bfnj_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );
		RepartitionDown
		( temp2_part2T__D_0__D_1__D_2__D_3,  temp2_part2_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       temp2_part2_1__D_0__D_1__D_2__D_3,
		  temp2_part2B__D_0__D_1__D_2__D_3, temp2_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );

		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  temp2_part2_1__D_0__D_1__D_2__D_3
		PartitionDown(T_bfnj_part2_1__D_0__D_1__D_2__D_3, T_bfnj_part2_1_part3T__D_0__D_1__D_2__D_3, T_bfnj_part2_1_part3B__D_0__D_1__D_2__D_3, 3, 0);
		PartitionDown(T_bfnj_part3_1__D_0__D_1__D_2__D_3, T_bfnj_part3_1_part2T__D_0__D_1__D_2__D_3, T_bfnj_part3_1_part2B__D_0__D_1__D_2__D_3, 2, 0);
		PartitionDown(temp2_part2_1__D_0__D_1__D_2__D_3, temp2_part2_1_part3T__D_0__D_1__D_2__D_3, temp2_part2_1_part3B__D_0__D_1__D_2__D_3, 3, 0);
		while(temp2_part2_1_part3T__D_0__D_1__D_2__D_3.Dimension(3) < temp2_part2_1__D_0__D_1__D_2__D_3.Dimension(3))
		{
			RepartitionDown
			( T_bfnj_part2_1_part3T__D_0__D_1__D_2__D_3,  T_bfnj_part2_1_part3_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       T_bfnj_part2_1_part3_1__D_0__D_1__D_2__D_3,
			  T_bfnj_part2_1_part3B__D_0__D_1__D_2__D_3, T_bfnj_part2_1_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );
			RepartitionDown
			( T_bfnj_part3_1_part2T__D_0__D_1__D_2__D_3,  T_bfnj_part3_1_part2_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       T_bfnj_part3_1_part2_1__D_0__D_1__D_2__D_3,
			  T_bfnj_part3_1_part2B__D_0__D_1__D_2__D_3, T_bfnj_part3_1_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );
			RepartitionDown
			( temp2_part2_1_part3T__D_0__D_1__D_2__D_3,  temp2_part2_1_part3_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       temp2_part2_1_part3_1__D_0__D_1__D_2__D_3,
			  temp2_part2_1_part3B__D_0__D_1__D_2__D_3, temp2_part2_1_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );

			   // T_bfnj_part3_1_part2_1[D0,D1,D3,D2] <- T_bfnj_part3_1_part2_1[D0,D1,D2,D3]
			T_bfnj_part3_1_part2_1__D_0__D_1__D_3__D_2.AlignModesWith( modes_0_1_2_3, T_bfnj_part2_1_part3_1__D_0__D_1__D_2__D_3, modes_0_1_3_2 );
			T_bfnj_part3_1_part2_1__D_0__D_1__D_3__D_2.AllToAllRedistFrom( T_bfnj_part3_1_part2_1__D_0__D_1__D_2__D_3, modes_2_3 );
			YAxpPx( 2.0, T_bfnj_part2_1_part3_1__D_0__D_1__D_2__D_3, -1.0, T_bfnj_part3_1_part2_1__D_0__D_1__D_3__D_2, perm_0_1_3_2, temp2_part2_1_part3_1__D_0__D_1__D_2__D_3 );
			T_bfnj_part3_1_part2_1__D_0__D_1__D_3__D_2.EmptyData();

			SlidePartitionDown
			( T_bfnj_part2_1_part3T__D_0__D_1__D_2__D_3,  T_bfnj_part2_1_part3_0__D_0__D_1__D_2__D_3,
			       T_bfnj_part2_1_part3_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  T_bfnj_part2_1_part3B__D_0__D_1__D_2__D_3, T_bfnj_part2_1_part3_2__D_0__D_1__D_2__D_3, 3 );
			SlidePartitionDown
			( T_bfnj_part3_1_part2T__D_0__D_1__D_2__D_3,  T_bfnj_part3_1_part2_0__D_0__D_1__D_2__D_3,
			       T_bfnj_part3_1_part2_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  T_bfnj_part3_1_part2B__D_0__D_1__D_2__D_3, T_bfnj_part3_1_part2_2__D_0__D_1__D_2__D_3, 2 );
			SlidePartitionDown
			( temp2_part2_1_part3T__D_0__D_1__D_2__D_3,  temp2_part2_1_part3_0__D_0__D_1__D_2__D_3,
			       temp2_part2_1_part3_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  temp2_part2_1_part3B__D_0__D_1__D_2__D_3, temp2_part2_1_part3_2__D_0__D_1__D_2__D_3, 3 );

		}
		//****

		SlidePartitionDown
		( T_bfnj_part2T__D_0__D_1__D_2__D_3,  T_bfnj_part2_0__D_0__D_1__D_2__D_3,
		       T_bfnj_part2_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  T_bfnj_part2B__D_0__D_1__D_2__D_3, T_bfnj_part2_2__D_0__D_1__D_2__D_3, 2 );
		SlidePartitionDown
		( T_bfnj_part3T__D_0__D_1__D_2__D_3,  T_bfnj_part3_0__D_0__D_1__D_2__D_3,
		       T_bfnj_part3_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  T_bfnj_part3B__D_0__D_1__D_2__D_3, T_bfnj_part3_2__D_0__D_1__D_2__D_3, 3 );
		SlidePartitionDown
		( temp2_part2T__D_0__D_1__D_2__D_3,  temp2_part2_0__D_0__D_1__D_2__D_3,
		       temp2_part2_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  temp2_part2B__D_0__D_1__D_2__D_3, temp2_part2_2__D_0__D_1__D_2__D_3, 2 );

	}
	//****
	//**** (out of 1)
	//**** Is real	0 shadows
		//Outputs:
		//  temp1_perm1302__D_1__D_3__D_0__D_2
	PartitionDown(X_bmej__D_0__D_1__D_2__D_3, X_bmej_part2T__D_0__D_1__D_2__D_3, X_bmej_part2B__D_0__D_1__D_2__D_3, 2, 0);
	PartitionDown(T_bfnj__D_0__D_1__D_2__D_3, T_bfnj_part1T__D_0__D_1__D_2__D_3, T_bfnj_part1B__D_0__D_1__D_2__D_3, 1, 0);
	while(X_bmej_part2T__D_0__D_1__D_2__D_3.Dimension(2) < X_bmej__D_0__D_1__D_2__D_3.Dimension(2))
	{
		RepartitionDown
		( X_bmej_part2T__D_0__D_1__D_2__D_3,  X_bmej_part2_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       X_bmej_part2_1__D_0__D_1__D_2__D_3,
		  X_bmej_part2B__D_0__D_1__D_2__D_3, X_bmej_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );
		RepartitionDown
		( T_bfnj_part1T__D_0__D_1__D_2__D_3,  T_bfnj_part1_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       T_bfnj_part1_1__D_0__D_1__D_2__D_3,
		  T_bfnj_part1B__D_0__D_1__D_2__D_3, T_bfnj_part1_2__D_0__D_1__D_2__D_3, 1, blkSize );

		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  temp1_perm1302__D_1__D_3__D_0__D_2
		PartitionDown(X_bmej_part2_1__D_0__D_1__D_2__D_3, X_bmej_part2_1_part1T__D_0__D_1__D_2__D_3, X_bmej_part2_1_part1B__D_0__D_1__D_2__D_3, 1, 0);
		PartitionDown(T_bfnj_part1_1__D_0__D_1__D_2__D_3, T_bfnj_part1_1_part2T__D_0__D_1__D_2__D_3, T_bfnj_part1_1_part2B__D_0__D_1__D_2__D_3, 2, 0);
		while(X_bmej_part2_1_part1T__D_0__D_1__D_2__D_3.Dimension(1) < X_bmej_part2_1__D_0__D_1__D_2__D_3.Dimension(1))
		{
			RepartitionDown
			( X_bmej_part2_1_part1T__D_0__D_1__D_2__D_3,  X_bmej_part2_1_part1_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       X_bmej_part2_1_part1_1__D_0__D_1__D_2__D_3,
			  X_bmej_part2_1_part1B__D_0__D_1__D_2__D_3, X_bmej_part2_1_part1_2__D_0__D_1__D_2__D_3, 1, blkSize );
			RepartitionDown
			( T_bfnj_part1_1_part2T__D_0__D_1__D_2__D_3,  T_bfnj_part1_1_part2_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       T_bfnj_part1_1_part2_1__D_0__D_1__D_2__D_3,
			  T_bfnj_part1_1_part2B__D_0__D_1__D_2__D_3, T_bfnj_part1_1_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );

			   // X_bmej_part2_1_part1_1[D1,D0,D2,D3] <- X_bmej_part2_1_part1_1[D0,D1,D2,D3]
			X_bmej_part2_1_part1_1__D_1__D_0__D_2__D_3.AlignModesWith( modes_0_3, temp1__D_0__D_1__D_2__D_3, modes_1_3 );
			X_bmej_part2_1_part1_1__D_1__D_0__D_2__D_3.AllToAllRedistFrom( X_bmej_part2_1_part1_1__D_0__D_1__D_2__D_3, modes_0_1 );
			   // T_bfnj_part1_1_part2_1[D0,D1,D3,D2] <- T_bfnj_part1_1_part2_1[D0,D1,D2,D3]
			T_bfnj_part1_1_part2_1__D_0__D_1__D_3__D_2.AlignModesWith( modes_0_3, temp1__D_0__D_1__D_2__D_3, modes_0_2 );
			T_bfnj_part1_1_part2_1__D_0__D_1__D_3__D_2.AllToAllRedistFrom( T_bfnj_part1_1_part2_1__D_0__D_1__D_2__D_3, modes_2_3 );
			   // X_bmej_part2_1_part1_1[D1,*,*,D3] <- X_bmej_part2_1_part1_1[D1,D0,D2,D3]
			X_bmej_part2_1_part1_1_perm0312__D_1__D_3__S__S.AlignModesWith( modes_0_3, temp1__D_0__D_1__D_2__D_3, modes_1_3 );
			X_bmej_part2_1_part1_1_perm0312__D_1__D_3__S__S.AllGatherRedistFrom( X_bmej_part2_1_part1_1__D_1__D_0__D_2__D_3, modes_0_2 );
			X_bmej_part2_1_part1_1__D_1__D_0__D_2__D_3.EmptyData();
			   // T_bfnj_part1_1_part2_1[D0,*,*,D2] <- T_bfnj_part1_1_part2_1[D0,D1,D3,D2]
			T_bfnj_part1_1_part2_1_perm2103__S__S__D_0__D_2.AlignModesWith( modes_0_3, temp1__D_0__D_1__D_2__D_3, modes_0_2 );
			T_bfnj_part1_1_part2_1_perm2103__S__S__D_0__D_2.AllGatherRedistFrom( T_bfnj_part1_1_part2_1__D_0__D_1__D_3__D_2, modes_1_3 );
			   // 1.0 * X_bmej_part2_1_part1_1[D1,*,*,D3]_bjme * T_bfnj_part1_1_part2_1[D0,*,*,D2]_meai + 1.0 * temp1[D0,D1,D2,D3]_bjai
			LocalContractAndLocalEliminate(1.0, X_bmej_part2_1_part1_1_perm0312__D_1__D_3__S__S.LockedTensor(), indices_bjme, false,
				T_bfnj_part1_1_part2_1_perm2103__S__S__D_0__D_2.LockedTensor(), indices_meai, false,
				1.0, temp1_perm1302__D_1__D_3__D_0__D_2.Tensor(), indices_bjai, false);
			T_bfnj_part1_1_part2_1_perm2103__S__S__D_0__D_2.EmptyData();
			X_bmej_part2_1_part1_1_perm0312__D_1__D_3__S__S.EmptyData();
			T_bfnj_part1_1_part2_1__D_0__D_1__D_3__D_2.EmptyData();

			SlidePartitionDown
			( X_bmej_part2_1_part1T__D_0__D_1__D_2__D_3,  X_bmej_part2_1_part1_0__D_0__D_1__D_2__D_3,
			       X_bmej_part2_1_part1_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  X_bmej_part2_1_part1B__D_0__D_1__D_2__D_3, X_bmej_part2_1_part1_2__D_0__D_1__D_2__D_3, 1 );
			SlidePartitionDown
			( T_bfnj_part1_1_part2T__D_0__D_1__D_2__D_3,  T_bfnj_part1_1_part2_0__D_0__D_1__D_2__D_3,
			       T_bfnj_part1_1_part2_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  T_bfnj_part1_1_part2B__D_0__D_1__D_2__D_3, T_bfnj_part1_1_part2_2__D_0__D_1__D_2__D_3, 2 );

		}
		//****

		SlidePartitionDown
		( X_bmej_part2T__D_0__D_1__D_2__D_3,  X_bmej_part2_0__D_0__D_1__D_2__D_3,
		       X_bmej_part2_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  X_bmej_part2B__D_0__D_1__D_2__D_3, X_bmej_part2_2__D_0__D_1__D_2__D_3, 2 );
		SlidePartitionDown
		( T_bfnj_part1T__D_0__D_1__D_2__D_3,  T_bfnj_part1_0__D_0__D_1__D_2__D_3,
		       T_bfnj_part1_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  T_bfnj_part1B__D_0__D_1__D_2__D_3, T_bfnj_part1_2__D_0__D_1__D_2__D_3, 1 );

	}
	X_bmej__D_0__D_1__D_2__D_3.EmptyData();
	X_bmej__D_0__D_1__D_2__D_3.EmptyData();
	//****
	Permute( temp2__D_0__D_1__D_2__D_3, temp2_perm3102__D_3__D_1__D_0__D_2 );
	temp2__D_0__D_1__D_2__D_3.EmptyData();
	Permute( temp1_perm1302__D_1__D_3__D_0__D_2, temp1__D_0__D_1__D_2__D_3 );
	temp1_perm1302__D_1__D_3__D_0__D_2.EmptyData();
	//**** (out of 1)
	//**** Is real	0 shadows
		//Outputs:
		//  accum__D_0__D_1__D_2__D_3
	PartitionDown(temp1__D_0__D_1__D_2__D_3, temp1_part2T__D_0__D_1__D_2__D_3, temp1_part2B__D_0__D_1__D_2__D_3, 2, 0);
	PartitionDown(temp1__D_0__D_1__D_2__D_3, temp1_part3T__D_0__D_1__D_2__D_3, temp1_part3B__D_0__D_1__D_2__D_3, 3, 0);
	PartitionDown(accum__D_0__D_1__D_2__D_3, accum_part2T__D_0__D_1__D_2__D_3, accum_part2B__D_0__D_1__D_2__D_3, 2, 0);
	while(accum_part2T__D_0__D_1__D_2__D_3.Dimension(2) < accum__D_0__D_1__D_2__D_3.Dimension(2))
	{
		RepartitionDown
		( temp1_part2T__D_0__D_1__D_2__D_3,  temp1_part2_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       temp1_part2_1__D_0__D_1__D_2__D_3,
		  temp1_part2B__D_0__D_1__D_2__D_3, temp1_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );
		RepartitionDown
		( temp1_part3T__D_0__D_1__D_2__D_3,  temp1_part3_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       temp1_part3_1__D_0__D_1__D_2__D_3,
		  temp1_part3B__D_0__D_1__D_2__D_3, temp1_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );
		RepartitionDown
		( accum_part2T__D_0__D_1__D_2__D_3,  accum_part2_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       accum_part2_1__D_0__D_1__D_2__D_3,
		  accum_part2B__D_0__D_1__D_2__D_3, accum_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );

		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  accum_part2_1__D_0__D_1__D_2__D_3
		PartitionDown(temp1_part2_1__D_0__D_1__D_2__D_3, temp1_part2_1_part3T__D_0__D_1__D_2__D_3, temp1_part2_1_part3B__D_0__D_1__D_2__D_3, 3, 0);
		PartitionDown(temp1_part3_1__D_0__D_1__D_2__D_3, temp1_part3_1_part2T__D_0__D_1__D_2__D_3, temp1_part3_1_part2B__D_0__D_1__D_2__D_3, 2, 0);
		PartitionDown(accum_part2_1__D_0__D_1__D_2__D_3, accum_part2_1_part3T__D_0__D_1__D_2__D_3, accum_part2_1_part3B__D_0__D_1__D_2__D_3, 3, 0);
		while(accum_part2_1_part3T__D_0__D_1__D_2__D_3.Dimension(3) < accum_part2_1__D_0__D_1__D_2__D_3.Dimension(3))
		{
			RepartitionDown
			( temp1_part2_1_part3T__D_0__D_1__D_2__D_3,  temp1_part2_1_part3_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       temp1_part2_1_part3_1__D_0__D_1__D_2__D_3,
			  temp1_part2_1_part3B__D_0__D_1__D_2__D_3, temp1_part2_1_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );
			RepartitionDown
			( temp1_part3_1_part2T__D_0__D_1__D_2__D_3,  temp1_part3_1_part2_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       temp1_part3_1_part2_1__D_0__D_1__D_2__D_3,
			  temp1_part3_1_part2B__D_0__D_1__D_2__D_3, temp1_part3_1_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );
			RepartitionDown
			( accum_part2_1_part3T__D_0__D_1__D_2__D_3,  accum_part2_1_part3_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       accum_part2_1_part3_1__D_0__D_1__D_2__D_3,
			  accum_part2_1_part3B__D_0__D_1__D_2__D_3, accum_part2_1_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );

			   // temp1_part3_1_part2_1[D0,D1,D3,D2] <- temp1_part3_1_part2_1[D0,D1,D2,D3]
			temp1_part3_1_part2_1__D_0__D_1__D_3__D_2.AlignModesWith( modes_0_1_2_3, temp1_part2_1_part3_1__D_0__D_1__D_2__D_3, modes_0_1_3_2 );
			temp1_part3_1_part2_1__D_0__D_1__D_3__D_2.AllToAllRedistFrom( temp1_part3_1_part2_1__D_0__D_1__D_2__D_3, modes_2_3 );
			YAxpPx( -0.5, temp1_part2_1_part3_1__D_0__D_1__D_2__D_3, -1.0, temp1_part3_1_part2_1__D_0__D_1__D_3__D_2, perm_0_1_3_2, accum_part2_1_part3_1__D_0__D_1__D_2__D_3 );
			temp1_part3_1_part2_1__D_0__D_1__D_3__D_2.EmptyData();

			SlidePartitionDown
			( temp1_part2_1_part3T__D_0__D_1__D_2__D_3,  temp1_part2_1_part3_0__D_0__D_1__D_2__D_3,
			       temp1_part2_1_part3_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  temp1_part2_1_part3B__D_0__D_1__D_2__D_3, temp1_part2_1_part3_2__D_0__D_1__D_2__D_3, 3 );
			SlidePartitionDown
			( temp1_part3_1_part2T__D_0__D_1__D_2__D_3,  temp1_part3_1_part2_0__D_0__D_1__D_2__D_3,
			       temp1_part3_1_part2_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  temp1_part3_1_part2B__D_0__D_1__D_2__D_3, temp1_part3_1_part2_2__D_0__D_1__D_2__D_3, 2 );
			SlidePartitionDown
			( accum_part2_1_part3T__D_0__D_1__D_2__D_3,  accum_part2_1_part3_0__D_0__D_1__D_2__D_3,
			       accum_part2_1_part3_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  accum_part2_1_part3B__D_0__D_1__D_2__D_3, accum_part2_1_part3_2__D_0__D_1__D_2__D_3, 3 );

		}
		//****

		SlidePartitionDown
		( temp1_part2T__D_0__D_1__D_2__D_3,  temp1_part2_0__D_0__D_1__D_2__D_3,
		       temp1_part2_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  temp1_part2B__D_0__D_1__D_2__D_3, temp1_part2_2__D_0__D_1__D_2__D_3, 2 );
		SlidePartitionDown
		( temp1_part3T__D_0__D_1__D_2__D_3,  temp1_part3_0__D_0__D_1__D_2__D_3,
		       temp1_part3_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  temp1_part3B__D_0__D_1__D_2__D_3, temp1_part3_2__D_0__D_1__D_2__D_3, 3 );
		SlidePartitionDown
		( accum_part2T__D_0__D_1__D_2__D_3,  accum_part2_0__D_0__D_1__D_2__D_3,
		       accum_part2_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  accum_part2B__D_0__D_1__D_2__D_3, accum_part2_2__D_0__D_1__D_2__D_3, 2 );

	}
	temp1__D_0__D_1__D_2__D_3.EmptyData();
	temp1__D_0__D_1__D_2__D_3.EmptyData();
	temp1__D_0__D_1__D_2__D_3.EmptyData();
	temp1__D_0__D_1__D_2__D_3.EmptyData();
	//****
	//**** (out of 1)
	//**** Is real	0 shadows
		//Outputs:
		//  accum__D_0__D_1__D_2__D_3
	PartitionDown(W_bmje__D_0__D_1__D_2__D_3, W_bmje_part0T__D_0__D_1__D_2__D_3, W_bmje_part0B__D_0__D_1__D_2__D_3, 0, 0);
	PartitionDown(accum__D_0__D_1__D_2__D_3, accum_part1T__D_0__D_1__D_2__D_3, accum_part1B__D_0__D_1__D_2__D_3, 1, 0);
	while(accum_part1T__D_0__D_1__D_2__D_3.Dimension(1) < accum__D_0__D_1__D_2__D_3.Dimension(1))
	{
		RepartitionDown
		( W_bmje_part0T__D_0__D_1__D_2__D_3,  W_bmje_part0_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       W_bmje_part0_1__D_0__D_1__D_2__D_3,
		  W_bmje_part0B__D_0__D_1__D_2__D_3, W_bmje_part0_2__D_0__D_1__D_2__D_3, 0, blkSize );
		RepartitionDown
		( accum_part1T__D_0__D_1__D_2__D_3,  accum_part1_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       accum_part1_1__D_0__D_1__D_2__D_3,
		  accum_part1B__D_0__D_1__D_2__D_3, accum_part1_2__D_0__D_1__D_2__D_3, 1, blkSize );

		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  accum_part1_1__D_0__D_1__D_2__D_3
		PartitionDown(W_bmje_part0_1__D_0__D_1__D_2__D_3, W_bmje_part0_1_part2T__D_0__D_1__D_2__D_3, W_bmje_part0_1_part2B__D_0__D_1__D_2__D_3, 2, 0);
		PartitionDown(accum_part1_1__D_0__D_1__D_2__D_3, accum_part1_1_part3T__D_0__D_1__D_2__D_3, accum_part1_1_part3B__D_0__D_1__D_2__D_3, 3, 0);
		while(accum_part1_1_part3T__D_0__D_1__D_2__D_3.Dimension(3) < accum_part1_1__D_0__D_1__D_2__D_3.Dimension(3))
		{
			RepartitionDown
			( W_bmje_part0_1_part2T__D_0__D_1__D_2__D_3,  W_bmje_part0_1_part2_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       W_bmje_part0_1_part2_1__D_0__D_1__D_2__D_3,
			  W_bmje_part0_1_part2B__D_0__D_1__D_2__D_3, W_bmje_part0_1_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );
			RepartitionDown
			( accum_part1_1_part3T__D_0__D_1__D_2__D_3,  accum_part1_1_part3_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       accum_part1_1_part3_1__D_0__D_1__D_2__D_3,
			  accum_part1_1_part3B__D_0__D_1__D_2__D_3, accum_part1_1_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );

			tempShape = accum_part1_1_part3_1__D_0__D_1__D_2__D_3.Shape();
			tempShape.push_back( g.Shape()[1] );
			tempShape.push_back( g.Shape()[3] );
			accum_part1_1_part3_1_perm130254__S__S__D_0__D_2__D_3__D_1.ResizeTo( tempShape );
			   // W_bmje_part0_1_part2_1[D0,D3,D2,D1] <- W_bmje_part0_1_part2_1[D0,D1,D2,D3]
			W_bmje_part0_1_part2_1__D_0__D_3__D_2__D_1.AlignModesWith( modes_1_3, temp2__D_0__D_1__D_2__D_3, modes_3_1 );
			W_bmje_part0_1_part2_1__D_0__D_3__D_2__D_1.AllToAllRedistFrom( W_bmje_part0_1_part2_1__D_0__D_1__D_2__D_3, modes_1_3 );
			   // W_bmje_part0_1_part2_1[*,D3,*,D1] <- W_bmje_part0_1_part2_1[D0,D3,D2,D1]
			W_bmje_part0_1_part2_1_perm0213__S__S__D_3__D_1.AlignModesWith( modes_1_3, temp2__D_0__D_1__D_2__D_3, modes_3_1 );
			W_bmje_part0_1_part2_1_perm0213__S__S__D_3__D_1.AllGatherRedistFrom( W_bmje_part0_1_part2_1__D_0__D_3__D_2__D_1, modes_0_2 );
			   // 0.5 * W_bmje_part0_1_part2_1[*,D3,*,D1]_bjme * temp2[D0,D1,D2,D3]_meai + 0.0 * accum_part1_1_part3_1[D0,*,D2,*,D1,D3]_bjaime
			LocalContract(0.5, W_bmje_part0_1_part2_1_perm0213__S__S__D_3__D_1.LockedTensor(), indices_bjme, false,
				temp2_perm3102__D_3__D_1__D_0__D_2.LockedTensor(), indices_meai, false,
				0.0, accum_part1_1_part3_1_perm130254__S__S__D_0__D_2__D_3__D_1.Tensor(), indices_bjaime, false);
			   // accum_part1_1_part3_1[D0,D1,D2,D3] <- accum_part1_1_part3_1[D0,*,D2,*,D1,D3] (with SumScatter on (D1)(D3))
			accum_part1_1_part3_1__D_0__D_1__D_2__D_3.ReduceScatterUpdateRedistFrom( accum_part1_1_part3_1_perm130254__S__S__D_0__D_2__D_3__D_1, 1.0, modes_5_4 );
			accum_part1_1_part3_1_perm130254__S__S__D_0__D_2__D_3__D_1.EmptyData();
			W_bmje_part0_1_part2_1_perm0213__S__S__D_3__D_1.EmptyData();
			W_bmje_part0_1_part2_1__D_0__D_3__D_2__D_1.EmptyData();

			SlidePartitionDown
			( W_bmje_part0_1_part2T__D_0__D_1__D_2__D_3,  W_bmje_part0_1_part2_0__D_0__D_1__D_2__D_3,
			       W_bmje_part0_1_part2_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  W_bmje_part0_1_part2B__D_0__D_1__D_2__D_3, W_bmje_part0_1_part2_2__D_0__D_1__D_2__D_3, 2 );
			SlidePartitionDown
			( accum_part1_1_part3T__D_0__D_1__D_2__D_3,  accum_part1_1_part3_0__D_0__D_1__D_2__D_3,
			       accum_part1_1_part3_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  accum_part1_1_part3B__D_0__D_1__D_2__D_3, accum_part1_1_part3_2__D_0__D_1__D_2__D_3, 3 );

		}
		//****

		SlidePartitionDown
		( W_bmje_part0T__D_0__D_1__D_2__D_3,  W_bmje_part0_0__D_0__D_1__D_2__D_3,
		       W_bmje_part0_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  W_bmje_part0B__D_0__D_1__D_2__D_3, W_bmje_part0_2__D_0__D_1__D_2__D_3, 0 );
		SlidePartitionDown
		( accum_part1T__D_0__D_1__D_2__D_3,  accum_part1_0__D_0__D_1__D_2__D_3,
		       accum_part1_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  accum_part1B__D_0__D_1__D_2__D_3, accum_part1_2__D_0__D_1__D_2__D_3, 1 );

	}
	W_bmje__D_0__D_1__D_2__D_3.EmptyData();
	temp2_perm3102__D_3__D_1__D_0__D_2.EmptyData();
	W_bmje__D_0__D_1__D_2__D_3.EmptyData();
	temp2_perm3102__D_3__D_1__D_0__D_2.EmptyData();
	//****
	//**** (out of 1)
	//**** Is real	0 shadows
		//Outputs:
		//  accum__D_0__D_1__D_2__D_3
	PartitionDown(T_bfnj__D_0__D_1__D_2__D_3, T_bfnj_part3T__D_0__D_1__D_2__D_3, T_bfnj_part3B__D_0__D_1__D_2__D_3, 3, 0);
	PartitionDown(accum__D_0__D_1__D_2__D_3, accum_part3T__D_0__D_1__D_2__D_3, accum_part3B__D_0__D_1__D_2__D_3, 3, 0);
	while(accum_part3T__D_0__D_1__D_2__D_3.Dimension(3) < accum__D_0__D_1__D_2__D_3.Dimension(3))
	{
		RepartitionDown
		( T_bfnj_part3T__D_0__D_1__D_2__D_3,  T_bfnj_part3_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       T_bfnj_part3_1__D_0__D_1__D_2__D_3,
		  T_bfnj_part3B__D_0__D_1__D_2__D_3, T_bfnj_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );
		RepartitionDown
		( accum_part3T__D_0__D_1__D_2__D_3,  accum_part3_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       accum_part3_1__D_0__D_1__D_2__D_3,
		  accum_part3B__D_0__D_1__D_2__D_3, accum_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );

		Permute( accum_part3_1__D_0__D_1__D_2__D_3, accum_part3_1_perm2013__D_2__D_0__D_1__D_3 );
		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  accum_part3_1_perm2013__D_2__D_0__D_1__D_3
		PartitionDown(G_mi__D_0_1__D_2_3, G_mi_part0T__D_0_1__D_2_3, G_mi_part0B__D_0_1__D_2_3, 0, 0);
		PartitionDown(T_bfnj_part3_1__D_0__D_1__D_2__D_3, T_bfnj_part3_1_part2T__D_0__D_1__D_2__D_3, T_bfnj_part3_1_part2B__D_0__D_1__D_2__D_3, 2, 0);
		while(G_mi_part0T__D_0_1__D_2_3.Dimension(0) < G_mi__D_0_1__D_2_3.Dimension(0))
		{
			RepartitionDown
			( G_mi_part0T__D_0_1__D_2_3,  G_mi_part0_0__D_0_1__D_2_3,
			  /**/ /**/
			       G_mi_part0_1__D_0_1__D_2_3,
			  G_mi_part0B__D_0_1__D_2_3, G_mi_part0_2__D_0_1__D_2_3, 0, blkSize );
			RepartitionDown
			( T_bfnj_part3_1_part2T__D_0__D_1__D_2__D_3,  T_bfnj_part3_1_part2_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       T_bfnj_part3_1_part2_1__D_0__D_1__D_2__D_3,
			  T_bfnj_part3_1_part2B__D_0__D_1__D_2__D_3, T_bfnj_part3_1_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );

			   // G_mi_part0_1[*,D2] <- G_mi_part0_1[D01,D23]
			G_mi_part0_1_perm10__D_2__S.AlignModesWith( modes_1, accum_part3_1__D_0__D_1__D_2__D_3, modes_2 );
			G_mi_part0_1_perm10__D_2__S.AllGatherRedistFrom( G_mi_part0_1__D_0_1__D_2_3, modes_0_1_3 );
			   // T_bfnj_part3_1_part2_1[D0,D1,*,D3] <- T_bfnj_part3_1_part2_1[D0,D1,D2,D3]
			T_bfnj_part3_1_part2_1_perm2013__S__D_0__D_1__D_3.AlignModesWith( modes_0_1_3, accum_part3_1__D_0__D_1__D_2__D_3, modes_0_1_3 );
			T_bfnj_part3_1_part2_1_perm2013__S__D_0__D_1__D_3.AllGatherRedistFrom( T_bfnj_part3_1_part2_1__D_0__D_1__D_2__D_3, modes_2 );
			   // -1.0 * G_mi_part0_1[*,D2]_im * T_bfnj_part3_1_part2_1[D0,D1,*,D3]_mabj + 1.0 * accum_part3_1[D0,D1,D2,D3]_iabj
			LocalContractAndLocalEliminate(-1.0, G_mi_part0_1_perm10__D_2__S.LockedTensor(), indices_im, false,
				T_bfnj_part3_1_part2_1_perm2013__S__D_0__D_1__D_3.LockedTensor(), indices_mabj, false,
				1.0, accum_part3_1_perm2013__D_2__D_0__D_1__D_3.Tensor(), indices_iabj, false);
			G_mi_part0_1_perm10__D_2__S.EmptyData();
			T_bfnj_part3_1_part2_1_perm2013__S__D_0__D_1__D_3.EmptyData();

			SlidePartitionDown
			( G_mi_part0T__D_0_1__D_2_3,  G_mi_part0_0__D_0_1__D_2_3,
			       G_mi_part0_1__D_0_1__D_2_3,
			  /**/ /**/
			  G_mi_part0B__D_0_1__D_2_3, G_mi_part0_2__D_0_1__D_2_3, 0 );
			SlidePartitionDown
			( T_bfnj_part3_1_part2T__D_0__D_1__D_2__D_3,  T_bfnj_part3_1_part2_0__D_0__D_1__D_2__D_3,
			       T_bfnj_part3_1_part2_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  T_bfnj_part3_1_part2B__D_0__D_1__D_2__D_3, T_bfnj_part3_1_part2_2__D_0__D_1__D_2__D_3, 2 );

		}
		//****
		Permute( accum_part3_1_perm2013__D_2__D_0__D_1__D_3, accum_part3_1__D_0__D_1__D_2__D_3 );
		accum_part3_1_perm2013__D_2__D_0__D_1__D_3.EmptyData();

		SlidePartitionDown
		( T_bfnj_part3T__D_0__D_1__D_2__D_3,  T_bfnj_part3_0__D_0__D_1__D_2__D_3,
		       T_bfnj_part3_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  T_bfnj_part3B__D_0__D_1__D_2__D_3, T_bfnj_part3_2__D_0__D_1__D_2__D_3, 3 );
		SlidePartitionDown
		( accum_part3T__D_0__D_1__D_2__D_3,  accum_part3_0__D_0__D_1__D_2__D_3,
		       accum_part3_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  accum_part3B__D_0__D_1__D_2__D_3, accum_part3_2__D_0__D_1__D_2__D_3, 3 );

	}
	G_mi__D_0_1__D_2_3.EmptyData();
	G_mi__D_0_1__D_2_3.EmptyData();
	//****
	//**** (out of 1)
	//**** Is real	0 shadows
		//Outputs:
		//  accum__D_0__D_1__D_2__D_3
	PartitionDown(t_fj__D_0_1__D_2_3, t_fj_part1T__D_0_1__D_2_3, t_fj_part1B__D_0_1__D_2_3, 1, 0);
	PartitionDown(P_ijmb__D_0__D_1__D_2__D_3, P_ijmb_part0T__D_0__D_1__D_2__D_3, P_ijmb_part0B__D_0__D_1__D_2__D_3, 0, 0);
	PartitionDown(T_bfnj__D_0__D_1__D_2__D_3, T_bfnj_part2T__D_0__D_1__D_2__D_3, T_bfnj_part2B__D_0__D_1__D_2__D_3, 2, 0);
	PartitionDown(accum__D_0__D_1__D_2__D_3, accum_part2T__D_0__D_1__D_2__D_3, accum_part2B__D_0__D_1__D_2__D_3, 2, 0);
	while(accum_part2T__D_0__D_1__D_2__D_3.Dimension(2) < accum__D_0__D_1__D_2__D_3.Dimension(2))
	{
		RepartitionDown
		( t_fj_part1T__D_0_1__D_2_3,  t_fj_part1_0__D_0_1__D_2_3,
		  /**/ /**/
		       t_fj_part1_1__D_0_1__D_2_3,
		  t_fj_part1B__D_0_1__D_2_3, t_fj_part1_2__D_0_1__D_2_3, 1, blkSize );
		RepartitionDown
		( P_ijmb_part0T__D_0__D_1__D_2__D_3,  P_ijmb_part0_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       P_ijmb_part0_1__D_0__D_1__D_2__D_3,
		  P_ijmb_part0B__D_0__D_1__D_2__D_3, P_ijmb_part0_2__D_0__D_1__D_2__D_3, 0, blkSize );
		RepartitionDown
		( T_bfnj_part2T__D_0__D_1__D_2__D_3,  T_bfnj_part2_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       T_bfnj_part2_1__D_0__D_1__D_2__D_3,
		  T_bfnj_part2B__D_0__D_1__D_2__D_3, T_bfnj_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );
		RepartitionDown
		( accum_part2T__D_0__D_1__D_2__D_3,  accum_part2_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       accum_part2_1__D_0__D_1__D_2__D_3,
		  accum_part2B__D_0__D_1__D_2__D_3, accum_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );

		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  accum_part2_1__D_0__D_1__D_2__D_3
		PartitionDown(T_bfnj_part2_1__D_0__D_1__D_2__D_3, T_bfnj_part2_1_part3T__D_0__D_1__D_2__D_3, T_bfnj_part2_1_part3B__D_0__D_1__D_2__D_3, 3, 0);
		PartitionDown(accum_part2_1__D_0__D_1__D_2__D_3, accum_part2_1_part3T__D_0__D_1__D_2__D_3, accum_part2_1_part3B__D_0__D_1__D_2__D_3, 3, 0);
		while(accum_part2_1_part3T__D_0__D_1__D_2__D_3.Dimension(3) < accum_part2_1__D_0__D_1__D_2__D_3.Dimension(3))
		{
			RepartitionDown
			( T_bfnj_part2_1_part3T__D_0__D_1__D_2__D_3,  T_bfnj_part2_1_part3_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       T_bfnj_part2_1_part3_1__D_0__D_1__D_2__D_3,
			  T_bfnj_part2_1_part3B__D_0__D_1__D_2__D_3, T_bfnj_part2_1_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );
			RepartitionDown
			( accum_part2_1_part3T__D_0__D_1__D_2__D_3,  accum_part2_1_part3_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       accum_part2_1_part3_1__D_0__D_1__D_2__D_3,
			  accum_part2_1_part3B__D_0__D_1__D_2__D_3, accum_part2_1_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );

			   // F_ae[D0,*] <- F_ae[D01,D23]
			F_ae__D_0__S.AlignModesWith( modes_0, accum_part2_1_part3_1__D_0__D_1__D_2__D_3, modes_0 );
			F_ae__D_0__S.AllGatherRedistFrom( F_ae__D_0_1__D_2_3, modes_1_2_3 );
			   // T_bfnj_part2_1_part3_1[*,D1,D2,D3] <- T_bfnj_part2_1_part3_1[D0,D1,D2,D3]
			T_bfnj_part2_1_part3_1__S__D_1__D_2__D_3.AlignModesWith( modes_1_2_3, accum_part2_1_part3_1__D_0__D_1__D_2__D_3, modes_1_2_3 );
			T_bfnj_part2_1_part3_1__S__D_1__D_2__D_3.AllGatherRedistFrom( T_bfnj_part2_1_part3_1__D_0__D_1__D_2__D_3, modes_0 );
			   // 1.0 * F_ae[D0,*]_ae * T_bfnj_part2_1_part3_1[*,D1,D2,D3]_ebij + 1.0 * accum_part2_1_part3_1[D0,D1,D2,D3]_abij
			LocalContractAndLocalEliminate(1.0, F_ae__D_0__S.LockedTensor(), indices_ae, false,
				T_bfnj_part2_1_part3_1__S__D_1__D_2__D_3.LockedTensor(), indices_ebij, false,
				1.0, accum_part2_1_part3_1__D_0__D_1__D_2__D_3.Tensor(), indices_abij, false);
			F_ae__D_0__S.EmptyData();
			T_bfnj_part2_1_part3_1__S__D_1__D_2__D_3.EmptyData();

			SlidePartitionDown
			( T_bfnj_part2_1_part3T__D_0__D_1__D_2__D_3,  T_bfnj_part2_1_part3_0__D_0__D_1__D_2__D_3,
			       T_bfnj_part2_1_part3_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  T_bfnj_part2_1_part3B__D_0__D_1__D_2__D_3, T_bfnj_part2_1_part3_2__D_0__D_1__D_2__D_3, 3 );
			SlidePartitionDown
			( accum_part2_1_part3T__D_0__D_1__D_2__D_3,  accum_part2_1_part3_0__D_0__D_1__D_2__D_3,
			       accum_part2_1_part3_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  accum_part2_1_part3B__D_0__D_1__D_2__D_3, accum_part2_1_part3_2__D_0__D_1__D_2__D_3, 3 );

		}
		//****
		Permute( accum_part2_1__D_0__D_1__D_2__D_3, accum_part2_1_perm2310__D_2__D_3__D_1__D_0 );
		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  accum_part2_1_perm2310__D_2__D_3__D_1__D_0
		PartitionDown(P_ijmb_part0_1__D_0__D_1__D_2__D_3, P_ijmb_part0_1_part2T__D_0__D_1__D_2__D_3, P_ijmb_part0_1_part2B__D_0__D_1__D_2__D_3, 2, 0);
		PartitionDown(t_fj__D_0_1__D_2_3, t_fj_part1T__D_0_1__D_2_3, t_fj_part1B__D_0_1__D_2_3, 1, 0);
		while(P_ijmb_part0_1_part2T__D_0__D_1__D_2__D_3.Dimension(2) < P_ijmb_part0_1__D_0__D_1__D_2__D_3.Dimension(2))
		{
			RepartitionDown
			( P_ijmb_part0_1_part2T__D_0__D_1__D_2__D_3,  P_ijmb_part0_1_part2_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       P_ijmb_part0_1_part2_1__D_0__D_1__D_2__D_3,
			  P_ijmb_part0_1_part2B__D_0__D_1__D_2__D_3, P_ijmb_part0_1_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );
			RepartitionDown
			( t_fj_part1T__D_0_1__D_2_3,  t_fj_part1_0__D_0_1__D_2_3,
			  /**/ /**/
			       t_fj_part1_1__D_0_1__D_2_3,
			  t_fj_part1B__D_0_1__D_2_3, t_fj_part1_2__D_0_1__D_2_3, 1, blkSize );

			   // P_ijmb_part0_1_part2_1[D0,D3,D2,D1] <- P_ijmb_part0_1_part2_1[D0,D1,D2,D3]
			P_ijmb_part0_1_part2_1__D_0__D_3__D_2__D_1.AlignModesWith( modes_0_1_3, accum_part2_1__D_0__D_1__D_2__D_3, modes_2_3_1 );
			P_ijmb_part0_1_part2_1__D_0__D_3__D_2__D_1.AllToAllRedistFrom( P_ijmb_part0_1_part2_1__D_0__D_1__D_2__D_3, modes_1_3 );
			   // P_ijmb_part0_1_part2_1[D2,D3,*,D1] <- P_ijmb_part0_1_part2_1[D0,D3,D2,D1]
			P_ijmb_part0_1_part2_1_perm0132__D_2__D_3__D_1__S.AlignModesWith( modes_0_1_3, accum_part2_1__D_0__D_1__D_2__D_3, modes_2_3_1 );
			P_ijmb_part0_1_part2_1_perm0132__D_2__D_3__D_1__S.AllToAllRedistFrom( P_ijmb_part0_1_part2_1__D_0__D_3__D_2__D_1, modes_0_2 );
			P_ijmb_part0_1_part2_1__D_0__D_3__D_2__D_1.EmptyData();
			   // t_fj_part1_1[D0,*] <- t_fj_part1_1[D01,D23]
			t_fj_part1_1_perm10__S__D_0.AlignModesWith( modes_0, accum_part2_1__D_0__D_1__D_2__D_3, modes_0 );
			t_fj_part1_1_perm10__S__D_0.AllGatherRedistFrom( t_fj_part1_1__D_0_1__D_2_3, modes_1_2_3 );
			   // -1.0 * P_ijmb_part0_1_part2_1[D2,D3,*,D1]_ijbm * t_fj_part1_1[D0,*]_ma + 1.0 * accum_part2_1[D0,D1,D2,D3]_ijba
			LocalContractAndLocalEliminate(-1.0, P_ijmb_part0_1_part2_1_perm0132__D_2__D_3__D_1__S.LockedTensor(), indices_ijbm, false,
				t_fj_part1_1_perm10__S__D_0.LockedTensor(), indices_ma, false,
				1.0, accum_part2_1_perm2310__D_2__D_3__D_1__D_0.Tensor(), indices_ijba, false);
			P_ijmb_part0_1_part2_1_perm0132__D_2__D_3__D_1__S.EmptyData();
			t_fj_part1_1_perm10__S__D_0.EmptyData();

			SlidePartitionDown
			( P_ijmb_part0_1_part2T__D_0__D_1__D_2__D_3,  P_ijmb_part0_1_part2_0__D_0__D_1__D_2__D_3,
			       P_ijmb_part0_1_part2_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  P_ijmb_part0_1_part2B__D_0__D_1__D_2__D_3, P_ijmb_part0_1_part2_2__D_0__D_1__D_2__D_3, 2 );
			SlidePartitionDown
			( t_fj_part1T__D_0_1__D_2_3,  t_fj_part1_0__D_0_1__D_2_3,
			       t_fj_part1_1__D_0_1__D_2_3,
			  /**/ /**/
			  t_fj_part1B__D_0_1__D_2_3, t_fj_part1_2__D_0_1__D_2_3, 1 );

		}
		//****
		Permute( accum_part2_1_perm2310__D_2__D_3__D_1__D_0, accum_part2_1__D_0__D_1__D_2__D_3 );
		accum_part2_1_perm2310__D_2__D_3__D_1__D_0.EmptyData();
		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  accum_part2_1__D_0__D_1__D_2__D_3
		PartitionDown(r_bmfe__D_0__D_1__D_2__D_3, r_bmfe_part1T__D_0__D_1__D_2__D_3, r_bmfe_part1B__D_0__D_1__D_2__D_3, 1, 0);
		PartitionDown(accum_part2_1__D_0__D_1__D_2__D_3, accum_part2_1_part3T__D_0__D_1__D_2__D_3, accum_part2_1_part3B__D_0__D_1__D_2__D_3, 3, 0);
		while(accum_part2_1_part3T__D_0__D_1__D_2__D_3.Dimension(3) < accum_part2_1__D_0__D_1__D_2__D_3.Dimension(3))
		{
			RepartitionDown
			( r_bmfe_part1T__D_0__D_1__D_2__D_3,  r_bmfe_part1_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       r_bmfe_part1_1__D_0__D_1__D_2__D_3,
			  r_bmfe_part1B__D_0__D_1__D_2__D_3, r_bmfe_part1_2__D_0__D_1__D_2__D_3, 1, blkSize );
			RepartitionDown
			( accum_part2_1_part3T__D_0__D_1__D_2__D_3,  accum_part2_1_part3_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       accum_part2_1_part3_1__D_0__D_1__D_2__D_3,
			  accum_part2_1_part3B__D_0__D_1__D_2__D_3, accum_part2_1_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );

			tempShape = accum_part2_1_part3_1__D_0__D_1__D_2__D_3.Shape();
			tempShape.push_back( g.Shape()[0] );
			accum_part2_1_part3_1_perm30124__D_1__D_2__D_3__S__D_0.ResizeTo( tempShape );
			tempShape = accum_part2_1_part3_1__D_0__D_1__D_2__D_3.Shape();
			accum_part2_1_part3_1_temp__D_2_0__D_3__S__D_1.ResizeTo( tempShape );
			   // t_fj_part1_1[D0,*] <- t_fj_part1_1[D01,D23]
			t_fj_part1_1__D_0__S.AlignModesWith( modes_0, r_bmfe_part1_1__D_0__D_1__D_2__D_3, modes_0 );
			t_fj_part1_1__D_0__S.AllGatherRedistFrom( t_fj_part1_1__D_0_1__D_2_3, modes_1_2_3 );
			Permute( r_bmfe_part1_1__D_0__D_1__D_2__D_3, r_bmfe_part1_1_perm1230__D_1__D_2__D_3__D_0 );
			   // 1.0 * r_bmfe_part1_1[D0,D1,D2,D3]_jabe * t_fj_part1_1[D0,*]_ei + 0.0 * accum_part2_1_part3_1[D2,D3,*,D1,D0]_jabie
			LocalContract(1.0, r_bmfe_part1_1_perm1230__D_1__D_2__D_3__D_0.LockedTensor(), indices_jabe, false,
				t_fj_part1_1__D_0__S.LockedTensor(), indices_ei, false,
				0.0, accum_part2_1_part3_1_perm30124__D_1__D_2__D_3__S__D_0.Tensor(), indices_jabie, false);
			   // accum_part2_1_part3_1_temp[D20,D3,*,D1] <- accum_part2_1_part3_1[D2,D3,*,D1,D0] (with SumScatter on D0)
			accum_part2_1_part3_1_temp__D_2_0__D_3__S__D_1.ReduceScatterRedistFrom( accum_part2_1_part3_1_perm30124__D_1__D_2__D_3__S__D_0, 4 );
			accum_part2_1_part3_1_perm30124__D_1__D_2__D_3__S__D_0.EmptyData();
			r_bmfe_part1_1_perm1230__D_1__D_2__D_3__D_0.EmptyData();
			t_fj_part1_1__D_0__S.EmptyData();
			   // accum_part2_1_part3_1_temp[D20,D1,*,D3] <- accum_part2_1_part3_1_temp[D20,D3,*,D1]
			accum_part2_1_part3_1_temp__D_2_0__D_1__S__D_3.AlignModesWith( modes_0_1_2_3, accum_part2_1_part3_1__D_0__D_1__D_2__D_3, modes_0_1_2_3 );
			accum_part2_1_part3_1_temp__D_2_0__D_1__S__D_3.AllToAllRedistFrom( accum_part2_1_part3_1_temp__D_2_0__D_3__S__D_1, modes_1_3 );
			accum_part2_1_part3_1_temp__D_2_0__D_3__S__D_1.EmptyData();
			   // accum_part2_1_part3_1_temp[D0,D1,D2,D3] <- accum_part2_1_part3_1_temp[D20,D1,*,D3]
			accum_part2_1_part3_1_temp__D_0__D_1__D_2__D_3.AlignModesWith( modes_0_1_2_3, accum_part2_1_part3_1__D_0__D_1__D_2__D_3, modes_0_1_2_3 );
			accum_part2_1_part3_1_temp__D_0__D_1__D_2__D_3.AllToAllRedistFrom( accum_part2_1_part3_1_temp__D_2_0__D_1__S__D_3, modes_0_2 );
			YxpBy( accum_part2_1_part3_1_temp__D_0__D_1__D_2__D_3, 1.0, accum_part2_1_part3_1__D_0__D_1__D_2__D_3 );
			accum_part2_1_part3_1_temp__D_0__D_1__D_2__D_3.EmptyData();
			accum_part2_1_part3_1_temp__D_2_0__D_1__S__D_3.EmptyData();

			SlidePartitionDown
			( r_bmfe_part1T__D_0__D_1__D_2__D_3,  r_bmfe_part1_0__D_0__D_1__D_2__D_3,
			       r_bmfe_part1_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  r_bmfe_part1B__D_0__D_1__D_2__D_3, r_bmfe_part1_2__D_0__D_1__D_2__D_3, 1 );
			SlidePartitionDown
			( accum_part2_1_part3T__D_0__D_1__D_2__D_3,  accum_part2_1_part3_0__D_0__D_1__D_2__D_3,
			       accum_part2_1_part3_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  accum_part2_1_part3B__D_0__D_1__D_2__D_3, accum_part2_1_part3_2__D_0__D_1__D_2__D_3, 3 );

		}
		//****

		SlidePartitionDown
		( t_fj_part1T__D_0_1__D_2_3,  t_fj_part1_0__D_0_1__D_2_3,
		       t_fj_part1_1__D_0_1__D_2_3,
		  /**/ /**/
		  t_fj_part1B__D_0_1__D_2_3, t_fj_part1_2__D_0_1__D_2_3, 1 );
		SlidePartitionDown
		( accum_part2T__D_0__D_1__D_2__D_3,  accum_part2_0__D_0__D_1__D_2__D_3,
		       accum_part2_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  accum_part2B__D_0__D_1__D_2__D_3, accum_part2_2__D_0__D_1__D_2__D_3, 2 );
		SlidePartitionDown
		( P_ijmb_part0T__D_0__D_1__D_2__D_3,  P_ijmb_part0_0__D_0__D_1__D_2__D_3,
		       P_ijmb_part0_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  P_ijmb_part0B__D_0__D_1__D_2__D_3, P_ijmb_part0_2__D_0__D_1__D_2__D_3, 0 );
		SlidePartitionDown
		( T_bfnj_part2T__D_0__D_1__D_2__D_3,  T_bfnj_part2_0__D_0__D_1__D_2__D_3,
		       T_bfnj_part2_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  T_bfnj_part2B__D_0__D_1__D_2__D_3, T_bfnj_part2_2__D_0__D_1__D_2__D_3, 2 );

	}
	r_bmfe__D_0__D_1__D_2__D_3.EmptyData();
	t_fj__D_0_1__D_2_3.EmptyData();
	P_ijmb__D_0__D_1__D_2__D_3.EmptyData();
	t_fj__D_0_1__D_2_3.EmptyData();
	F_ae__D_0_1__D_2_3.EmptyData();
	T_bfnj__D_0__D_1__D_2__D_3.EmptyData();
	r_bmfe__D_0__D_1__D_2__D_3.EmptyData();
	t_fj__D_0_1__D_2_3.EmptyData();
	P_ijmb__D_0__D_1__D_2__D_3.EmptyData();
	t_fj__D_0_1__D_2_3.EmptyData();
	F_ae__D_0_1__D_2_3.EmptyData();
	T_bfnj__D_0__D_1__D_2__D_3.EmptyData();
	//****
	//**** (out of 1)
	//**** Is real	0 shadows
		//Outputs:
		//  Z_abij__D_0__D_1__D_2__D_3
	PartitionDown(Tau_efmn__D_0__D_1__D_2__D_3, Tau_efmn_part2T__D_0__D_1__D_2__D_3, Tau_efmn_part2B__D_0__D_1__D_2__D_3, 2, 0);
	PartitionDown(accum__D_0__D_1__D_2__D_3, accum_part2T__D_0__D_1__D_2__D_3, accum_part2B__D_0__D_1__D_2__D_3, 2, 0);
	PartitionDown(accum__D_0__D_1__D_2__D_3, accum_part3T__D_0__D_1__D_2__D_3, accum_part3B__D_0__D_1__D_2__D_3, 3, 0);
	PartitionDown(Z_abij__D_0__D_1__D_2__D_3, Z_abij_part2T__D_0__D_1__D_2__D_3, Z_abij_part2B__D_0__D_1__D_2__D_3, 2, 0);
	while(Z_abij_part2T__D_0__D_1__D_2__D_3.Dimension(2) < Z_abij__D_0__D_1__D_2__D_3.Dimension(2))
	{
		RepartitionDown
		( Tau_efmn_part2T__D_0__D_1__D_2__D_3,  Tau_efmn_part2_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       Tau_efmn_part2_1__D_0__D_1__D_2__D_3,
		  Tau_efmn_part2B__D_0__D_1__D_2__D_3, Tau_efmn_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );
		RepartitionDown
		( accum_part2T__D_0__D_1__D_2__D_3,  accum_part2_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       accum_part2_1__D_0__D_1__D_2__D_3,
		  accum_part2B__D_0__D_1__D_2__D_3, accum_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );
		RepartitionDown
		( accum_part3T__D_0__D_1__D_2__D_3,  accum_part3_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       accum_part3_1__D_0__D_1__D_2__D_3,
		  accum_part3B__D_0__D_1__D_2__D_3, accum_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );
		RepartitionDown
		( Z_abij_part2T__D_0__D_1__D_2__D_3,  Z_abij_part2_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       Z_abij_part2_1__D_0__D_1__D_2__D_3,
		  Z_abij_part2B__D_0__D_1__D_2__D_3, Z_abij_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );

		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  Z_abij_part2_1__D_0__D_1__D_2__D_3
		PartitionDown(Tau_efmn_part2_1__D_0__D_1__D_2__D_3, Tau_efmn_part2_1_part3T__D_0__D_1__D_2__D_3, Tau_efmn_part2_1_part3B__D_0__D_1__D_2__D_3, 3, 0);
		PartitionDown(accum_part2_1__D_0__D_1__D_2__D_3, accum_part2_1_part3T__D_0__D_1__D_2__D_3, accum_part2_1_part3B__D_0__D_1__D_2__D_3, 3, 0);
		PartitionDown(accum_part3_1__D_0__D_1__D_2__D_3, accum_part3_1_part2T__D_0__D_1__D_2__D_3, accum_part3_1_part2B__D_0__D_1__D_2__D_3, 2, 0);
		PartitionDown(Z_abij_part2_1__D_0__D_1__D_2__D_3, Z_abij_part2_1_part3T__D_0__D_1__D_2__D_3, Z_abij_part2_1_part3B__D_0__D_1__D_2__D_3, 3, 0);
		while(Z_abij_part2_1_part3T__D_0__D_1__D_2__D_3.Dimension(3) < Z_abij_part2_1__D_0__D_1__D_2__D_3.Dimension(3))
		{
			RepartitionDown
			( Tau_efmn_part2_1_part3T__D_0__D_1__D_2__D_3,  Tau_efmn_part2_1_part3_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       Tau_efmn_part2_1_part3_1__D_0__D_1__D_2__D_3,
			  Tau_efmn_part2_1_part3B__D_0__D_1__D_2__D_3, Tau_efmn_part2_1_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );
			RepartitionDown
			( accum_part2_1_part3T__D_0__D_1__D_2__D_3,  accum_part2_1_part3_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       accum_part2_1_part3_1__D_0__D_1__D_2__D_3,
			  accum_part2_1_part3B__D_0__D_1__D_2__D_3, accum_part2_1_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );
			RepartitionDown
			( accum_part3_1_part2T__D_0__D_1__D_2__D_3,  accum_part3_1_part2_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       accum_part3_1_part2_1__D_0__D_1__D_2__D_3,
			  accum_part3_1_part2B__D_0__D_1__D_2__D_3, accum_part3_1_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );
			RepartitionDown
			( Z_abij_part2_1_part3T__D_0__D_1__D_2__D_3,  Z_abij_part2_1_part3_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       Z_abij_part2_1_part3_1__D_0__D_1__D_2__D_3,
			  Z_abij_part2_1_part3B__D_0__D_1__D_2__D_3, Z_abij_part2_1_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );

			   // accum_part3_1_part2_1[D0,D1,D3,D2] <- accum_part3_1_part2_1[D0,D1,D2,D3]
			accum_part3_1_part2_1__D_0__D_1__D_3__D_2.AlignModesWith( modes_0_1_2_3, accum_part2_1_part3_1__D_0__D_1__D_2__D_3, modes_1_0_3_2 );
			accum_part3_1_part2_1__D_0__D_1__D_3__D_2.AllToAllRedistFrom( accum_part3_1_part2_1__D_0__D_1__D_2__D_3, modes_2_3 );
			   // accum_part3_1_part2_1[D1,D0,D3,D2] <- accum_part3_1_part2_1[D0,D1,D3,D2]
			accum_part3_1_part2_1__D_1__D_0__D_3__D_2.AlignModesWith( modes_0_1_2_3, accum_part2_1_part3_1__D_0__D_1__D_2__D_3, modes_1_0_3_2 );
			accum_part3_1_part2_1__D_1__D_0__D_3__D_2.AllToAllRedistFrom( accum_part3_1_part2_1__D_0__D_1__D_3__D_2, modes_0_1 );
			YAxpPx( 1.0, accum_part2_1_part3_1__D_0__D_1__D_2__D_3, 1.0, accum_part3_1_part2_1__D_1__D_0__D_3__D_2, perm_1_0_3_2, Z_abij_part2_1_part3_1__D_0__D_1__D_2__D_3 );
			accum_part3_1_part2_1__D_1__D_0__D_3__D_2.EmptyData();
			accum_part3_1_part2_1__D_0__D_1__D_3__D_2.EmptyData();
			   // Tau_efmn_part2_1_part3_1[D2,D1,D0,D3] <- Tau_efmn_part2_1_part3_1[D0,D1,D2,D3]
			Tau_efmn_part2_1_part3_1__D_2__D_1__D_0__D_3.AlignModesWith( modes_0_1, y_abef__D_0__D_1__D_2__D_3, modes_2_3 );
			Tau_efmn_part2_1_part3_1__D_2__D_1__D_0__D_3.AllToAllRedistFrom( Tau_efmn_part2_1_part3_1__D_0__D_1__D_2__D_3, modes_0_2 );
			   // Tau_efmn_part2_1_part3_1[D2,D3,D0,D1] <- Tau_efmn_part2_1_part3_1[D2,D1,D0,D3]
			Tau_efmn_part2_1_part3_1__D_2__D_3__D_0__D_1.AlignModesWith( modes_0_1, y_abef__D_0__D_1__D_2__D_3, modes_2_3 );
			Tau_efmn_part2_1_part3_1__D_2__D_3__D_0__D_1.AllToAllRedistFrom( Tau_efmn_part2_1_part3_1__D_2__D_1__D_0__D_3, modes_1_3 );
			Tau_efmn_part2_1_part3_1__D_2__D_1__D_0__D_3.EmptyData();
			tempShape = Z_abij_part2_1_part3_1__D_0__D_1__D_2__D_3.Shape();
			tempShape.push_back( g.Shape()[2] );
			tempShape.push_back( g.Shape()[3] );
			Z_abij_part2_1_part3_1__D_0__D_1__S__S__D_2__D_3.ResizeTo( tempShape );
			   // Tau_efmn_part2_1_part3_1[D2,D3,*,*] <- Tau_efmn_part2_1_part3_1[D2,D3,D0,D1]
			Tau_efmn_part2_1_part3_1__D_2__D_3__S__S.AlignModesWith( modes_0_1, y_abef__D_0__D_1__D_2__D_3, modes_2_3 );
			Tau_efmn_part2_1_part3_1__D_2__D_3__S__S.AllGatherRedistFrom( Tau_efmn_part2_1_part3_1__D_2__D_3__D_0__D_1, modes_0_1 );
			   // 1.0 * y_abef[D0,D1,D2,D3]_abef * Tau_efmn_part2_1_part3_1[D2,D3,*,*]_efij + 0.0 * Z_abij_part2_1_part3_1[D0,D1,*,*,D2,D3]_abijef
			LocalContract(1.0, y_abef__D_0__D_1__D_2__D_3.LockedTensor(), indices_abef, false,
				Tau_efmn_part2_1_part3_1__D_2__D_3__S__S.LockedTensor(), indices_efij, false,
				0.0, Z_abij_part2_1_part3_1__D_0__D_1__S__S__D_2__D_3.Tensor(), indices_abijef, false);
			   // Z_abij_part2_1_part3_1[D0,D1,D2,D3] <- Z_abij_part2_1_part3_1[D0,D1,*,*,D2,D3] (with SumScatter on (D2)(D3))
			Z_abij_part2_1_part3_1__D_0__D_1__D_2__D_3.ReduceScatterUpdateRedistFrom( Z_abij_part2_1_part3_1__D_0__D_1__S__S__D_2__D_3, 1.0, modes_5_4 );
			Z_abij_part2_1_part3_1__D_0__D_1__S__S__D_2__D_3.EmptyData();
			Tau_efmn_part2_1_part3_1__D_2__D_3__S__S.EmptyData();
			Tau_efmn_part2_1_part3_1__D_2__D_3__D_0__D_1.EmptyData();

			SlidePartitionDown
			( Tau_efmn_part2_1_part3T__D_0__D_1__D_2__D_3,  Tau_efmn_part2_1_part3_0__D_0__D_1__D_2__D_3,
			       Tau_efmn_part2_1_part3_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  Tau_efmn_part2_1_part3B__D_0__D_1__D_2__D_3, Tau_efmn_part2_1_part3_2__D_0__D_1__D_2__D_3, 3 );
			SlidePartitionDown
			( Z_abij_part2_1_part3T__D_0__D_1__D_2__D_3,  Z_abij_part2_1_part3_0__D_0__D_1__D_2__D_3,
			       Z_abij_part2_1_part3_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  Z_abij_part2_1_part3B__D_0__D_1__D_2__D_3, Z_abij_part2_1_part3_2__D_0__D_1__D_2__D_3, 3 );
			SlidePartitionDown
			( accum_part2_1_part3T__D_0__D_1__D_2__D_3,  accum_part2_1_part3_0__D_0__D_1__D_2__D_3,
			       accum_part2_1_part3_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  accum_part2_1_part3B__D_0__D_1__D_2__D_3, accum_part2_1_part3_2__D_0__D_1__D_2__D_3, 3 );
			SlidePartitionDown
			( accum_part3_1_part2T__D_0__D_1__D_2__D_3,  accum_part3_1_part2_0__D_0__D_1__D_2__D_3,
			       accum_part3_1_part2_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  accum_part3_1_part2B__D_0__D_1__D_2__D_3, accum_part3_1_part2_2__D_0__D_1__D_2__D_3, 2 );

		}
		//****

		SlidePartitionDown
		( Tau_efmn_part2T__D_0__D_1__D_2__D_3,  Tau_efmn_part2_0__D_0__D_1__D_2__D_3,
		       Tau_efmn_part2_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  Tau_efmn_part2B__D_0__D_1__D_2__D_3, Tau_efmn_part2_2__D_0__D_1__D_2__D_3, 2 );
		SlidePartitionDown
		( Z_abij_part2T__D_0__D_1__D_2__D_3,  Z_abij_part2_0__D_0__D_1__D_2__D_3,
		       Z_abij_part2_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  Z_abij_part2B__D_0__D_1__D_2__D_3, Z_abij_part2_2__D_0__D_1__D_2__D_3, 2 );
		SlidePartitionDown
		( accum_part2T__D_0__D_1__D_2__D_3,  accum_part2_0__D_0__D_1__D_2__D_3,
		       accum_part2_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  accum_part2B__D_0__D_1__D_2__D_3, accum_part2_2__D_0__D_1__D_2__D_3, 2 );
		SlidePartitionDown
		( accum_part3T__D_0__D_1__D_2__D_3,  accum_part3_0__D_0__D_1__D_2__D_3,
		       accum_part3_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  accum_part3B__D_0__D_1__D_2__D_3, accum_part3_2__D_0__D_1__D_2__D_3, 3 );

	}
	y_abef__D_0__D_1__D_2__D_3.EmptyData();
	accum__D_0__D_1__D_2__D_3.EmptyData();
	accum__D_0__D_1__D_2__D_3.EmptyData();
	y_abef__D_0__D_1__D_2__D_3.EmptyData();
	accum__D_0__D_1__D_2__D_3.EmptyData();
	accum__D_0__D_1__D_2__D_3.EmptyData();
	//****
	Permute( Z_abij__D_0__D_1__D_2__D_3, Z_abij_perm2301__D_2__D_3__D_0__D_1 );
	//**** (out of 1)
	//**** Is real	0 shadows
		//Outputs:
		//  Z_abij_perm2301__D_2__D_3__D_0__D_1
	PartitionDown(Q_mnij__D_0__D_1__D_2__D_3, Q_mnij_part0T__D_0__D_1__D_2__D_3, Q_mnij_part0B__D_0__D_1__D_2__D_3, 0, 0);
	PartitionDown(Tau_efmn__D_0__D_1__D_2__D_3, Tau_efmn_part2T__D_0__D_1__D_2__D_3, Tau_efmn_part2B__D_0__D_1__D_2__D_3, 2, 0);
	while(Q_mnij_part0T__D_0__D_1__D_2__D_3.Dimension(0) < Q_mnij__D_0__D_1__D_2__D_3.Dimension(0))
	{
		RepartitionDown
		( Q_mnij_part0T__D_0__D_1__D_2__D_3,  Q_mnij_part0_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       Q_mnij_part0_1__D_0__D_1__D_2__D_3,
		  Q_mnij_part0B__D_0__D_1__D_2__D_3, Q_mnij_part0_2__D_0__D_1__D_2__D_3, 0, blkSize );
		RepartitionDown
		( Tau_efmn_part2T__D_0__D_1__D_2__D_3,  Tau_efmn_part2_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       Tau_efmn_part2_1__D_0__D_1__D_2__D_3,
		  Tau_efmn_part2B__D_0__D_1__D_2__D_3, Tau_efmn_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );

		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  Z_abij_perm2301__D_2__D_3__D_0__D_1
		PartitionDown(Q_mnij_part0_1__D_0__D_1__D_2__D_3, Q_mnij_part0_1_part1T__D_0__D_1__D_2__D_3, Q_mnij_part0_1_part1B__D_0__D_1__D_2__D_3, 1, 0);
		PartitionDown(Tau_efmn_part2_1__D_0__D_1__D_2__D_3, Tau_efmn_part2_1_part3T__D_0__D_1__D_2__D_3, Tau_efmn_part2_1_part3B__D_0__D_1__D_2__D_3, 3, 0);
		while(Q_mnij_part0_1_part1T__D_0__D_1__D_2__D_3.Dimension(1) < Q_mnij_part0_1__D_0__D_1__D_2__D_3.Dimension(1))
		{
			RepartitionDown
			( Q_mnij_part0_1_part1T__D_0__D_1__D_2__D_3,  Q_mnij_part0_1_part1_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       Q_mnij_part0_1_part1_1__D_0__D_1__D_2__D_3,
			  Q_mnij_part0_1_part1B__D_0__D_1__D_2__D_3, Q_mnij_part0_1_part1_2__D_0__D_1__D_2__D_3, 1, blkSize );
			RepartitionDown
			( Tau_efmn_part2_1_part3T__D_0__D_1__D_2__D_3,  Tau_efmn_part2_1_part3_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       Tau_efmn_part2_1_part3_1__D_0__D_1__D_2__D_3,
			  Tau_efmn_part2_1_part3B__D_0__D_1__D_2__D_3, Tau_efmn_part2_1_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );

			   // Q_mnij_part0_1_part1_1[*,*,D2,D3] <- Q_mnij_part0_1_part1_1[D0,D1,D2,D3]
			Q_mnij_part0_1_part1_1_perm2301__D_2__D_3__S__S.AlignModesWith( modes_2_3, Z_abij__D_0__D_1__D_2__D_3, modes_2_3 );
			Q_mnij_part0_1_part1_1_perm2301__D_2__D_3__S__S.AllGatherRedistFrom( Q_mnij_part0_1_part1_1__D_0__D_1__D_2__D_3, modes_0_1 );
			   // Tau_efmn_part2_1_part3_1[D0,D1,*,*] <- Tau_efmn_part2_1_part3_1[D0,D1,D2,D3]
			Tau_efmn_part2_1_part3_1_perm2301__S__S__D_0__D_1.AlignModesWith( modes_0_1, Z_abij__D_0__D_1__D_2__D_3, modes_0_1 );
			Tau_efmn_part2_1_part3_1_perm2301__S__S__D_0__D_1.AllGatherRedistFrom( Tau_efmn_part2_1_part3_1__D_0__D_1__D_2__D_3, modes_2_3 );
			   // 1.0 * Q_mnij_part0_1_part1_1[*,*,D2,D3]_ijmn * Tau_efmn_part2_1_part3_1[D0,D1,*,*]_mnab + 1.0 * Z_abij[D0,D1,D2,D3]_ijab
			LocalContractAndLocalEliminate(1.0, Q_mnij_part0_1_part1_1_perm2301__D_2__D_3__S__S.LockedTensor(), indices_ijmn, false,
				Tau_efmn_part2_1_part3_1_perm2301__S__S__D_0__D_1.LockedTensor(), indices_mnab, false,
				1.0, Z_abij_perm2301__D_2__D_3__D_0__D_1.Tensor(), indices_ijab, false);
			Q_mnij_part0_1_part1_1_perm2301__D_2__D_3__S__S.EmptyData();
			Tau_efmn_part2_1_part3_1_perm2301__S__S__D_0__D_1.EmptyData();

			SlidePartitionDown
			( Q_mnij_part0_1_part1T__D_0__D_1__D_2__D_3,  Q_mnij_part0_1_part1_0__D_0__D_1__D_2__D_3,
			       Q_mnij_part0_1_part1_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  Q_mnij_part0_1_part1B__D_0__D_1__D_2__D_3, Q_mnij_part0_1_part1_2__D_0__D_1__D_2__D_3, 1 );
			SlidePartitionDown
			( Tau_efmn_part2_1_part3T__D_0__D_1__D_2__D_3,  Tau_efmn_part2_1_part3_0__D_0__D_1__D_2__D_3,
			       Tau_efmn_part2_1_part3_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  Tau_efmn_part2_1_part3B__D_0__D_1__D_2__D_3, Tau_efmn_part2_1_part3_2__D_0__D_1__D_2__D_3, 3 );

		}
		//****

		SlidePartitionDown
		( Q_mnij_part0T__D_0__D_1__D_2__D_3,  Q_mnij_part0_0__D_0__D_1__D_2__D_3,
		       Q_mnij_part0_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  Q_mnij_part0B__D_0__D_1__D_2__D_3, Q_mnij_part0_2__D_0__D_1__D_2__D_3, 0 );
		SlidePartitionDown
		( Tau_efmn_part2T__D_0__D_1__D_2__D_3,  Tau_efmn_part2_0__D_0__D_1__D_2__D_3,
		       Tau_efmn_part2_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  Tau_efmn_part2B__D_0__D_1__D_2__D_3, Tau_efmn_part2_2__D_0__D_1__D_2__D_3, 2 );

	}
	Q_mnij__D_0__D_1__D_2__D_3.EmptyData();
	Tau_efmn__D_0__D_1__D_2__D_3.EmptyData();
	Q_mnij__D_0__D_1__D_2__D_3.EmptyData();
	Tau_efmn__D_0__D_1__D_2__D_3.EmptyData();
	//****
	Permute( Z_abij_perm2301__D_2__D_3__D_0__D_1, Z_abij__D_0__D_1__D_2__D_3 );
	Z_abij_perm2301__D_2__D_3__D_0__D_1.EmptyData();


Q_mnij__D_0__D_1__D_2__D_3.EmptyData();
Tau_efmn__D_0__D_1__D_2__D_3.EmptyData();
y_abef__D_0__D_1__D_2__D_3.EmptyData();
r_bmfe__D_0__D_1__D_2__D_3.EmptyData();
t_fj__D_0_1__D_2_3.EmptyData();
P_ijmb__D_0__D_1__D_2__D_3.EmptyData();
F_ae__D_0_1__D_2_3.EmptyData();
T_bfnj__D_0__D_1__D_2__D_3.EmptyData();
G_mi__D_0_1__D_2_3.EmptyData();
W_bmje__D_0__D_1__D_2__D_3.EmptyData();
temp2__D_0__D_1__D_2__D_3.EmptyData();
accum__D_0__D_1__D_2__D_3.EmptyData();
X_bmej__D_0__D_1__D_2__D_3.EmptyData();
temp1__D_0__D_1__D_2__D_3.EmptyData();
//****
//**** (out of 1)

	Yxpy( v_femn__D_0__D_1__D_2__D_3, Z_abij__D_0__D_1__D_2__D_3 );
	v_femn__D_0__D_1__D_2__D_3.EmptyData();


v_femn__D_0__D_1__D_2__D_3.EmptyData();
//****

/*****************************************/

    /*****************************************/
    mpi::Barrier(g.OwningComm());
    runTime = mpi::Time() - startTime;
    double flops = pow(n_o, 2) * pow(n_v, 2) * (11 + 2 * pow(n_o + n_v, 2));
    gflops = flops / (1e9 * runTime);

    //****

    double norm = 1.0;
#ifdef CORRECTNESS
    DistTensor<double> diff(dist__D_0__D_1__D_2__D_3, g);
    diff.ResizeTo(check);
    Diff(check, Z_abij__D_0__D_1__D_2__D_3, diff);
    norm = Norm(diff);
#endif

    //------------------------------------//

    //****

    if (commRank == 0)
        Timer::printTimers();

    //****
    if (commRank == 0) {
#ifdef CORRECTNESS
        std::cout << "NORM " << norm << std::endl;
#endif
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


