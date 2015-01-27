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
ObjShape overwrite_tmpShape_P;
TensorDistribution dist__S__S__D_1_2__D_0_3 = tmen::StringToTensorDist("[(),(),(1,2),(0,3)]");
TensorDistribution dist__S__S__D_2_1__D_0_3 = tmen::StringToTensorDist("[(),(),(2,1),(0,3)]");
TensorDistribution dist__S__S__D_1__D_0__D_2__D_3 = tmen::StringToTensorDist("[(),(),(1),(0),(2),(3)]");
TensorDistribution dist__S__D_1__D_2__D_0_3 = tmen::StringToTensorDist("[(),(1),(2),(0,3)]");
TensorDistribution dist__S__D_2__D_1__D_0__D_3 = tmen::StringToTensorDist("[(),(2),(1),(0),(3)]");
TensorDistribution dist__S__D_2__D_1__D_0_3 = tmen::StringToTensorDist("[(),(2),(1),(0,3)]");
TensorDistribution dist__D_0__S__D_1_2__D_3 = tmen::StringToTensorDist("[(0),(),(1,2),(3)]");
TensorDistribution dist__D_0__D_1__D_2__D_3 = tmen::StringToTensorDist("[(0),(1),(2),(3)]");
TensorDistribution dist__D_2__S = tmen::StringToTensorDist("[(2),()]");
TensorDistribution dist__D_2__D_1__D_0__D_3 = tmen::StringToTensorDist("[(2),(1),(0),(3)]");
TensorDistribution dist__D_2__D_3__S__S = tmen::StringToTensorDist("[(2),(3),(),()]");
TensorDistribution dist__D_2__D_3__D_0__D_1 = tmen::StringToTensorDist("[(2),(3),(0),(1)]");
TensorDistribution dist__D_3__S__D_1_2__D_0 = tmen::StringToTensorDist("[(3),(),(1,2),(0)]");
TensorDistribution dist__D_3__S__D_1__D_0__D_2 = tmen::StringToTensorDist("[(3),(),(1),(0),(2)]");
TensorDistribution dist__D_3__S = tmen::StringToTensorDist("[(3),()]");
TensorDistribution dist__D_3__D_2 = tmen::StringToTensorDist("[(3),(2)]");
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
Permutation perm_3_2_1_0_4_5( 6 );
perm_3_2_1_0_4_5[0] = 3;
perm_3_2_1_0_4_5[1] = 2;
perm_3_2_1_0_4_5[2] = 1;
perm_3_2_1_0_4_5[3] = 0;
perm_3_2_1_0_4_5[4] = 4;
perm_3_2_1_0_4_5[5] = 5;
ModeArray modes_0( 1 );
modes_0[0] = 0;
ModeArray modes_0_1( 2 );
modes_0_1[0] = 0;
modes_0_1[1] = 1;
ModeArray modes_0_1_2( 3 );
modes_0_1_2[0] = 0;
modes_0_1_2[1] = 1;
modes_0_1_2[2] = 2;
ModeArray modes_0_1_2_3( 4 );
modes_0_1_2_3[0] = 0;
modes_0_1_2_3[1] = 1;
modes_0_1_2_3[2] = 2;
modes_0_1_2_3[3] = 3;
ModeArray modes_0_1_3( 3 );
modes_0_1_3[0] = 0;
modes_0_1_3[1] = 1;
modes_0_1_3[2] = 3;
ModeArray modes_0_2( 2 );
modes_0_2[0] = 0;
modes_0_2[1] = 2;
ModeArray modes_0_3( 2 );
modes_0_3[0] = 0;
modes_0_3[1] = 3;
ModeArray modes_1( 1 );
modes_1[0] = 1;
ModeArray modes_1_2( 2 );
modes_1_2[0] = 1;
modes_1_2[1] = 2;
ModeArray modes_1_3( 2 );
modes_1_3[0] = 1;
modes_1_3[1] = 3;
ModeArray modes_2( 1 );
modes_2[0] = 2;
ModeArray modes_2_3( 2 );
modes_2_3[0] = 2;
modes_2_3[1] = 3;
ModeArray modes_3( 1 );
modes_3[0] = 3;
ModeArray modes_5_4( 2 );
modes_5_4[0] = 5;
modes_5_4[1] = 4;
IndexArray indices_bmef( 4 );
indices_bmef[0] = 'b';
indices_bmef[1] = 'm';
indices_bmef[2] = 'e';
indices_bmef[3] = 'f';
IndexArray indices_bmej( 4 );
indices_bmej[0] = 'b';
indices_bmej[1] = 'm';
indices_bmej[2] = 'e';
indices_bmej[3] = 'j';
IndexArray indices_bmie( 4 );
indices_bmie[0] = 'b';
indices_bmie[1] = 'm';
indices_bmie[2] = 'i';
indices_bmie[3] = 'e';
IndexArray indices_bmijef( 6 );
indices_bmijef[0] = 'b';
indices_bmijef[1] = 'm';
indices_bmijef[2] = 'i';
indices_bmijef[3] = 'j';
indices_bmijef[4] = 'e';
indices_bmijef[5] = 'f';
IndexArray indices_efij( 4 );
indices_efij[0] = 'e';
indices_efij[1] = 'f';
indices_efij[2] = 'i';
indices_efij[3] = 'j';
IndexArray indices_ei( 2 );
indices_ei[0] = 'e';
indices_ei[1] = 'i';
IndexArray indices_ej( 2 );
indices_ej[0] = 'e';
indices_ej[1] = 'j';
IndexArray indices_jimbe( 5 );
indices_jimbe[0] = 'j';
indices_jimbe[1] = 'i';
indices_jimbe[2] = 'm';
indices_jimbe[3] = 'b';
indices_jimbe[4] = 'e';
	//P_jimb[D0,D1,D2,D3]
DistTensor<double> P_jimb__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//P_jimb_part0B[D0,D1,D2,D3]
DistTensor<double> P_jimb_part0B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//P_jimb_part0T[D0,D1,D2,D3]
DistTensor<double> P_jimb_part0T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//P_jimb_part0_0[D0,D1,D2,D3]
DistTensor<double> P_jimb_part0_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//P_jimb_part0_1[D0,D1,D2,D3]
DistTensor<double> P_jimb_part0_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//P_jimb_part0_1_part1B[D0,D1,D2,D3]
DistTensor<double> P_jimb_part0_1_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//P_jimb_part0_1_part1T[D0,D1,D2,D3]
DistTensor<double> P_jimb_part0_1_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//P_jimb_part0_1_part1_0[D0,D1,D2,D3]
DistTensor<double> P_jimb_part0_1_part1_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//P_jimb_part0_1_part1_1[D0,D1,D2,D3]
DistTensor<double> P_jimb_part0_1_part1_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//P_jimb_part0_1_part1_1[D3,*,D1,D0,D2]
DistTensor<double> P_jimb_part0_1_part1_1__D_3__S__D_1__D_0__D_2( dist__D_3__S__D_1__D_0__D_2, g );
	//P_jimb_part0_1_part1_1[*,D2,D1,D0,D3]
DistTensor<double> P_jimb_part0_1_part1_1__S__D_2__D_1__D_0__D_3( dist__S__D_2__D_1__D_0__D_3, g );
	//P_jimb_part0_1_part1_1[*,*,D1,D0,D2,D3]
DistTensor<double> P_jimb_part0_1_part1_1_perm321045__D_0__D_1__S__S__D_2__D_3( dist__S__S__D_1__D_0__D_2__D_3, g );
P_jimb_part0_1_part1_1_perm321045__D_0__D_1__S__S__D_2__D_3.SetLocalPermutation( perm_3_2_1_0_4_5 );
	//P_jimb_part0_1_part1_1_P_temp[D0,D1,D2,D3]
DistTensor<double> P_jimb_part0_1_part1_1_P_temp__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//P_jimb_part0_1_part1_1_P_temp[D0,*,D12,D3]
DistTensor<double> P_jimb_part0_1_part1_1_P_temp__D_0__S__D_1_2__D_3( dist__D_0__S__D_1_2__D_3, g );
	//P_jimb_part0_1_part1_1_P_temp[D3,*,D12,D0]
DistTensor<double> P_jimb_part0_1_part1_1_P_temp__D_3__S__D_1_2__D_0( dist__D_3__S__D_1_2__D_0, g );
	//P_jimb_part0_1_part1_1_P_temp[*,D1,D2,D03]
DistTensor<double> P_jimb_part0_1_part1_1_P_temp__S__D_1__D_2__D_0_3( dist__S__D_1__D_2__D_0_3, g );
	//P_jimb_part0_1_part1_1_P_temp[*,D2,D1,D03]
DistTensor<double> P_jimb_part0_1_part1_1_P_temp__S__D_2__D_1__D_0_3( dist__S__D_2__D_1__D_0_3, g );
	//P_jimb_part0_1_part1_1_P_temp[*,*,D12,D03]
DistTensor<double> P_jimb_part0_1_part1_1_P_temp__S__S__D_1_2__D_0_3( dist__S__S__D_1_2__D_0_3, g );
	//P_jimb_part0_1_part1_1_P_temp[*,*,D21,D03]
DistTensor<double> P_jimb_part0_1_part1_1_P_temp__S__S__D_2_1__D_0_3( dist__S__S__D_2_1__D_0_3, g );
	//P_jimb_part0_1_part1_2[D0,D1,D2,D3]
DistTensor<double> P_jimb_part0_1_part1_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//P_jimb_part0_2[D0,D1,D2,D3]
DistTensor<double> P_jimb_part0_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Tau_efmn[D0,D1,D2,D3]
DistTensor<double> Tau_efmn__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Tau_efmn_part3B[D0,D1,D2,D3]
DistTensor<double> Tau_efmn_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Tau_efmn_part3T[D0,D1,D2,D3]
DistTensor<double> Tau_efmn_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Tau_efmn_part3_0[D0,D1,D2,D3]
DistTensor<double> Tau_efmn_part3_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Tau_efmn_part3_1[D0,D1,D2,D3]
DistTensor<double> Tau_efmn_part3_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Tau_efmn_part3_1_part2B[D0,D1,D2,D3]
DistTensor<double> Tau_efmn_part3_1_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Tau_efmn_part3_1_part2T[D0,D1,D2,D3]
DistTensor<double> Tau_efmn_part3_1_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Tau_efmn_part3_1_part2_0[D0,D1,D2,D3]
DistTensor<double> Tau_efmn_part3_1_part2_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Tau_efmn_part3_1_part2_1[D0,D1,D2,D3]
DistTensor<double> Tau_efmn_part3_1_part2_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Tau_efmn_part3_1_part2_1[D2,D1,D0,D3]
DistTensor<double> Tau_efmn_part3_1_part2_1__D_2__D_1__D_0__D_3( dist__D_2__D_1__D_0__D_3, g );
	//Tau_efmn_part3_1_part2_1[D2,D3,D0,D1]
DistTensor<double> Tau_efmn_part3_1_part2_1__D_2__D_3__D_0__D_1( dist__D_2__D_3__D_0__D_1, g );
	//Tau_efmn_part3_1_part2_1[D2,D3,*,*]
DistTensor<double> Tau_efmn_part3_1_part2_1__D_2__D_3__S__S( dist__D_2__D_3__S__S, g );
	//Tau_efmn_part3_1_part2_2[D0,D1,D2,D3]
DistTensor<double> Tau_efmn_part3_1_part2_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Tau_efmn_part3_2[D0,D1,D2,D3]
DistTensor<double> Tau_efmn_part3_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//r_bmfe[D0,D1,D2,D3]
DistTensor<double> r_bmfe__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
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
	//t_fj_part1_1[D01,*]
DistTensor<double> t_fj_part1_1__D_0_1__S( dist__D_0_1__S, g );
	//t_fj_part1_1[D2,*]
DistTensor<double> t_fj_part1_1__D_2__S( dist__D_2__S, g );
	//t_fj_part1_1[D3,D2]
DistTensor<double> t_fj_part1_1__D_3__D_2( dist__D_3__D_2, g );
	//t_fj_part1_1[D3,*]
DistTensor<double> t_fj_part1_1__D_3__S( dist__D_3__S, g );
	//t_fj_part1_2[D01,D23]
DistTensor<double> t_fj_part1_2__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//u_mnje[D0,D1,D2,D3]
DistTensor<double> u_mnje__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//w_bmje[D0,D1,D2,D3]
DistTensor<double> w_bmje__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//w_bmje_part2B[D0,D1,D2,D3]
DistTensor<double> w_bmje_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//w_bmje_part2T[D0,D1,D2,D3]
DistTensor<double> w_bmje_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//w_bmje_part2_0[D0,D1,D2,D3]
DistTensor<double> w_bmje_part2_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//w_bmje_part2_1[D0,D1,D2,D3]
DistTensor<double> w_bmje_part2_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//w_bmje_part2_2[D0,D1,D2,D3]
DistTensor<double> w_bmje_part2_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//x_bmej[D0,D1,D2,D3]
DistTensor<double> x_bmej__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//x_bmej_part3B[D0,D1,D2,D3]
DistTensor<double> x_bmej_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//x_bmej_part3T[D0,D1,D2,D3]
DistTensor<double> x_bmej_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//x_bmej_part3_0[D0,D1,D2,D3]
DistTensor<double> x_bmej_part3_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//x_bmej_part3_1[D0,D1,D2,D3]
DistTensor<double> x_bmej_part3_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//x_bmej_part3_2[D0,D1,D2,D3]
DistTensor<double> x_bmej_part3_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
// r_bmfe has 4 dims
//	Starting distribution: [D0,D1,D2,D3] or _D_0__D_1__D_2__D_3
ObjShape r_bmfe__D_0__D_1__D_2__D_3_tmpShape_P( 4 );
r_bmfe__D_0__D_1__D_2__D_3_tmpShape_P[ 0 ] = n_v;
r_bmfe__D_0__D_1__D_2__D_3_tmpShape_P[ 1 ] = n_o;
r_bmfe__D_0__D_1__D_2__D_3_tmpShape_P[ 2 ] = n_v;
r_bmfe__D_0__D_1__D_2__D_3_tmpShape_P[ 3 ] = n_v;
r_bmfe__D_0__D_1__D_2__D_3.ResizeTo( r_bmfe__D_0__D_1__D_2__D_3_tmpShape_P );
MakeUniform( r_bmfe__D_0__D_1__D_2__D_3 );
// Tau_efmn has 4 dims
ObjShape Tau_efmn__D_0__D_1__D_2__D_3_tmpShape_P( 4 );
Tau_efmn__D_0__D_1__D_2__D_3_tmpShape_P[ 0 ] = n_v;
Tau_efmn__D_0__D_1__D_2__D_3_tmpShape_P[ 1 ] = n_v;
Tau_efmn__D_0__D_1__D_2__D_3_tmpShape_P[ 2 ] = n_o;
Tau_efmn__D_0__D_1__D_2__D_3_tmpShape_P[ 3 ] = n_o;
Tau_efmn__D_0__D_1__D_2__D_3.ResizeTo( Tau_efmn__D_0__D_1__D_2__D_3_tmpShape_P );
MakeUniform( Tau_efmn__D_0__D_1__D_2__D_3 );
// w_bmje has 4 dims
ObjShape w_bmje__D_0__D_1__D_2__D_3_tmpShape_P( 4 );
w_bmje__D_0__D_1__D_2__D_3_tmpShape_P[ 0 ] = n_v;
w_bmje__D_0__D_1__D_2__D_3_tmpShape_P[ 1 ] = n_o;
w_bmje__D_0__D_1__D_2__D_3_tmpShape_P[ 2 ] = n_o;
w_bmje__D_0__D_1__D_2__D_3_tmpShape_P[ 3 ] = n_v;
w_bmje__D_0__D_1__D_2__D_3.ResizeTo( w_bmje__D_0__D_1__D_2__D_3_tmpShape_P );
MakeUniform( w_bmje__D_0__D_1__D_2__D_3 );
// t_fj has 2 dims
//	Starting distribution: [D01,D23] or _D_0_1__D_2_3
ObjShape t_fj__D_0_1__D_2_3_tmpShape_P( 2 );
t_fj__D_0_1__D_2_3_tmpShape_P[ 0 ] = n_v;
t_fj__D_0_1__D_2_3_tmpShape_P[ 1 ] = n_o;
t_fj__D_0_1__D_2_3.ResizeTo( t_fj__D_0_1__D_2_3_tmpShape_P );
MakeUniform( t_fj__D_0_1__D_2_3 );
// x_bmej has 4 dims
ObjShape x_bmej__D_0__D_1__D_2__D_3_tmpShape_P( 4 );
x_bmej__D_0__D_1__D_2__D_3_tmpShape_P[ 0 ] = n_v;
x_bmej__D_0__D_1__D_2__D_3_tmpShape_P[ 1 ] = n_o;
x_bmej__D_0__D_1__D_2__D_3_tmpShape_P[ 2 ] = n_v;
x_bmej__D_0__D_1__D_2__D_3_tmpShape_P[ 3 ] = n_o;
x_bmej__D_0__D_1__D_2__D_3.ResizeTo( x_bmej__D_0__D_1__D_2__D_3_tmpShape_P );
MakeUniform( x_bmej__D_0__D_1__D_2__D_3 );
// u_mnje has 4 dims
ObjShape u_mnje__D_0__D_1__D_2__D_3_tmpShape_P( 4 );
u_mnje__D_0__D_1__D_2__D_3_tmpShape_P[ 0 ] = n_o;
u_mnje__D_0__D_1__D_2__D_3_tmpShape_P[ 1 ] = n_o;
u_mnje__D_0__D_1__D_2__D_3_tmpShape_P[ 2 ] = n_o;
u_mnje__D_0__D_1__D_2__D_3_tmpShape_P[ 3 ] = n_v;
u_mnje__D_0__D_1__D_2__D_3.ResizeTo( u_mnje__D_0__D_1__D_2__D_3_tmpShape_P );
MakeUniform( u_mnje__D_0__D_1__D_2__D_3 );
// P_jimb has 4 dims
ObjShape P_jimb__D_0__D_1__D_2__D_3_tmpShape_P( 4 );
P_jimb__D_0__D_1__D_2__D_3_tmpShape_P[ 0 ] = n_o;
P_jimb__D_0__D_1__D_2__D_3_tmpShape_P[ 1 ] = n_o;
P_jimb__D_0__D_1__D_2__D_3_tmpShape_P[ 2 ] = n_o;
P_jimb__D_0__D_1__D_2__D_3_tmpShape_P[ 3 ] = n_v;
P_jimb__D_0__D_1__D_2__D_3.ResizeTo( P_jimb__D_0__D_1__D_2__D_3_tmpShape_P );
MakeUniform( P_jimb__D_0__D_1__D_2__D_3 );
//**** (out of 1)
//END_DECL

//******************************
//* Load tensors
//******************************
////////////////////////////////
//Performance testing
////////////////////////////////
std::stringstream fullName;
#ifdef CORRECTNESS
DistTensor<T> check_P(dist__D_0__D_1__D_2__D_3, g);
check_P.ResizeTo(P_jimb__D_0__D_1__D_2__D_3.Shape());
fullName.str("");
fullName.clear();
fullName << "ccsd_terms/term_P_iter" << testIter;
Read(check_P, fullName.str(), BINARY_FLAT, false);
#endif
//******************************
//* Load tensors
//******************************
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

	P_jimb__D_0__D_1__D_2__D_3 = u_mnje__D_0__D_1__D_2__D_3;


//****
//**** (out of 1)

	//**** (out of 1)
	//**** Is real	0 shadows
		//Outputs:
		//  P_jimb__D_0__D_1__D_2__D_3
	PartitionDown(Tau_efmn__D_0__D_1__D_2__D_3, Tau_efmn_part3T__D_0__D_1__D_2__D_3, Tau_efmn_part3B__D_0__D_1__D_2__D_3, 3, 0);
	PartitionDown(t_fj__D_0_1__D_2_3, t_fj_part1T__D_0_1__D_2_3, t_fj_part1B__D_0_1__D_2_3, 1, 0);
	PartitionDown(x_bmej__D_0__D_1__D_2__D_3, x_bmej_part3T__D_0__D_1__D_2__D_3, x_bmej_part3B__D_0__D_1__D_2__D_3, 3, 0);
	PartitionDown(P_jimb__D_0__D_1__D_2__D_3, P_jimb_part0T__D_0__D_1__D_2__D_3, P_jimb_part0B__D_0__D_1__D_2__D_3, 0, 0);
	while(P_jimb_part0T__D_0__D_1__D_2__D_3.Dimension(0) < P_jimb__D_0__D_1__D_2__D_3.Dimension(0))
	{
		RepartitionDown
		( Tau_efmn_part3T__D_0__D_1__D_2__D_3,  Tau_efmn_part3_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       Tau_efmn_part3_1__D_0__D_1__D_2__D_3,
		  Tau_efmn_part3B__D_0__D_1__D_2__D_3, Tau_efmn_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );
		RepartitionDown
		( t_fj_part1T__D_0_1__D_2_3,  t_fj_part1_0__D_0_1__D_2_3,
		  /**/ /**/
		       t_fj_part1_1__D_0_1__D_2_3,
		  t_fj_part1B__D_0_1__D_2_3, t_fj_part1_2__D_0_1__D_2_3, 1, blkSize );
		RepartitionDown
		( x_bmej_part3T__D_0__D_1__D_2__D_3,  x_bmej_part3_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       x_bmej_part3_1__D_0__D_1__D_2__D_3,
		  x_bmej_part3B__D_0__D_1__D_2__D_3, x_bmej_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );
		RepartitionDown
		( P_jimb_part0T__D_0__D_1__D_2__D_3,  P_jimb_part0_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       P_jimb_part0_1__D_0__D_1__D_2__D_3,
		  P_jimb_part0B__D_0__D_1__D_2__D_3, P_jimb_part0_2__D_0__D_1__D_2__D_3, 0, blkSize );

		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  P_jimb_part0_1__D_0__D_1__D_2__D_3
		PartitionDown(Tau_efmn_part3_1__D_0__D_1__D_2__D_3, Tau_efmn_part3_1_part2T__D_0__D_1__D_2__D_3, Tau_efmn_part3_1_part2B__D_0__D_1__D_2__D_3, 2, 0);
		PartitionDown(w_bmje__D_0__D_1__D_2__D_3, w_bmje_part2T__D_0__D_1__D_2__D_3, w_bmje_part2B__D_0__D_1__D_2__D_3, 2, 0);
		PartitionDown(t_fj__D_0_1__D_2_3, t_fj_part1T__D_0_1__D_2_3, t_fj_part1B__D_0_1__D_2_3, 1, 0);
		PartitionDown(P_jimb_part0_1__D_0__D_1__D_2__D_3, P_jimb_part0_1_part1T__D_0__D_1__D_2__D_3, P_jimb_part0_1_part1B__D_0__D_1__D_2__D_3, 1, 0);
		while(P_jimb_part0_1_part1T__D_0__D_1__D_2__D_3.Dimension(1) < P_jimb_part0_1__D_0__D_1__D_2__D_3.Dimension(1))
		{
			RepartitionDown
			( Tau_efmn_part3_1_part2T__D_0__D_1__D_2__D_3,  Tau_efmn_part3_1_part2_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       Tau_efmn_part3_1_part2_1__D_0__D_1__D_2__D_3,
			  Tau_efmn_part3_1_part2B__D_0__D_1__D_2__D_3, Tau_efmn_part3_1_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );
			RepartitionDown
			( w_bmje_part2T__D_0__D_1__D_2__D_3,  w_bmje_part2_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       w_bmje_part2_1__D_0__D_1__D_2__D_3,
			  w_bmje_part2B__D_0__D_1__D_2__D_3, w_bmje_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );
			RepartitionDown
			( t_fj_part1T__D_0_1__D_2_3,  t_fj_part1_0__D_0_1__D_2_3,
			  /**/ /**/
			       t_fj_part1_1__D_0_1__D_2_3,
			  t_fj_part1B__D_0_1__D_2_3, t_fj_part1_2__D_0_1__D_2_3, 1, blkSize );
			RepartitionDown
			( P_jimb_part0_1_part1T__D_0__D_1__D_2__D_3,  P_jimb_part0_1_part1_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       P_jimb_part0_1_part1_1__D_0__D_1__D_2__D_3,
			  P_jimb_part0_1_part1B__D_0__D_1__D_2__D_3, P_jimb_part0_1_part1_2__D_0__D_1__D_2__D_3, 1, blkSize );

			overwrite_tmpShape_P = P_jimb_part0_1_part1_1__D_0__D_1__D_2__D_3.Shape();
			overwrite_tmpShape_P.push_back( g.Shape()[2] );
			P_jimb_part0_1_part1_1__D_3__S__D_1__D_0__D_2.ResizeTo( overwrite_tmpShape_P );
			overwrite_tmpShape_P = P_jimb_part0_1_part1_1__D_0__D_1__D_2__D_3.Shape();
			P_jimb_part0_1_part1_1_P_temp__D_3__S__D_1_2__D_0.ResizeTo( overwrite_tmpShape_P );
			   // Tau_efmn_part3_1_part2_1[D2,D1,D0,D3] <- Tau_efmn_part3_1_part2_1[D0,D1,D2,D3]
			Tau_efmn_part3_1_part2_1__D_2__D_1__D_0__D_3.AlignModesWith( modes_0_1, r_bmfe__D_0__D_1__D_2__D_3, modes_2_3 );
			Tau_efmn_part3_1_part2_1__D_2__D_1__D_0__D_3.AllToAllRedistFrom( Tau_efmn_part3_1_part2_1__D_0__D_1__D_2__D_3, modes_0_2 );
			   // Tau_efmn_part3_1_part2_1[D2,D3,D0,D1] <- Tau_efmn_part3_1_part2_1[D2,D1,D0,D3]
			Tau_efmn_part3_1_part2_1__D_2__D_3__D_0__D_1.AlignModesWith( modes_0_1, r_bmfe__D_0__D_1__D_2__D_3, modes_2_3 );
			Tau_efmn_part3_1_part2_1__D_2__D_3__D_0__D_1.AllToAllRedistFrom( Tau_efmn_part3_1_part2_1__D_2__D_1__D_0__D_3, modes_1_3 );
			Tau_efmn_part3_1_part2_1__D_2__D_1__D_0__D_3.EmptyData();
			   // t_fj_part1_1[D3,D2] <- t_fj_part1_1[D01,D23]
			t_fj_part1_1__D_3__D_2.AlignModesWith( modes_0, w_bmje_part2_1__D_0__D_1__D_2__D_3, modes_3 );
			t_fj_part1_1__D_3__D_2.AllToAllRedistFrom( t_fj_part1_1__D_0_1__D_2_3, modes_0_1_3 );
			   // t_fj_part1_1[D3,*] <- t_fj_part1_1[D3,D2]
			t_fj_part1_1__D_3__S.AlignModesWith( modes_0, w_bmje_part2_1__D_0__D_1__D_2__D_3, modes_3 );
			t_fj_part1_1__D_3__S.AllGatherRedistFrom( t_fj_part1_1__D_3__D_2, modes_2 );
			t_fj_part1_1__D_3__D_2.EmptyData();
			   // t_fj_part1_1[D01,*] <- t_fj_part1_1[D01,D23]
			t_fj_part1_1__D_0_1__S.AlignModesWith( modes_0, x_bmej_part3_1__D_0__D_1__D_2__D_3, modes_2 );
			t_fj_part1_1__D_0_1__S.AllGatherRedistFrom( t_fj_part1_1__D_0_1__D_2_3, modes_2_3 );
			   // t_fj_part1_1[D2,*] <- t_fj_part1_1[D01,*]
			t_fj_part1_1__D_2__S.AlignModesWith( modes_0, x_bmej_part3_1__D_0__D_1__D_2__D_3, modes_2 );
			t_fj_part1_1__D_2__S.AllToAllRedistFrom( t_fj_part1_1__D_0_1__S, modes_0_1_2 );
			   // 1.0 * x_bmej_part3_1[D0,D1,D2,D3]_bmej * t_fj_part1_1[D2,*]_ei + 0.0 * P_jimb_part0_1_part1_1[D3,*,D1,D0,D2]_jimbe
PROFILE_SECTION("COMPUTE");
PROFILE_FLOPS(2*prod(P_jimb_part0_1_part1_1__D_3__S__D_1__D_0__D_2.Shape())*x_bmej_part3_1__D_0__D_1__D_2__D_3.Dimension(2));
			LocalContract(1.0, x_bmej_part3_1__D_0__D_1__D_2__D_3.LockedTensor(), indices_bmej, true,
				t_fj_part1_1__D_2__S.LockedTensor(), indices_ei, true,
				0.0, P_jimb_part0_1_part1_1__D_3__S__D_1__D_0__D_2.Tensor(), indices_jimbe, true);
PROFILE_STOP;
			   // P_jimb_part0_1_part1_1_P_temp[D3,*,D12,D0] <- P_jimb_part0_1_part1_1[D3,*,D1,D0,D2] (with SumScatter on D2)
PROFILE_SECTION("COMPUTE");
PROFILE_FLOPS(prod(P_jimb_part0_1_part1_1__D_3__S__D_1__D_0__D_2.Shape()));
			P_jimb_part0_1_part1_1_P_temp__D_3__S__D_1_2__D_0.ReduceScatterRedistFrom( P_jimb_part0_1_part1_1__D_3__S__D_1__D_0__D_2, 4 );
PROFILE_STOP;
			P_jimb_part0_1_part1_1__D_3__S__D_1__D_0__D_2.EmptyData();
			t_fj_part1_1__D_2__S.EmptyData();
			t_fj_part1_1__D_0_1__S.EmptyData();
			   // P_jimb_part0_1_part1_1_P_temp[D0,*,D12,D3] <- P_jimb_part0_1_part1_1_P_temp[D3,*,D12,D0]
			P_jimb_part0_1_part1_1_P_temp__D_0__S__D_1_2__D_3.AlignModesWith( modes_0_1_2_3, P_jimb_part0_1_part1_1__D_0__D_1__D_2__D_3, modes_0_1_2_3 );
			P_jimb_part0_1_part1_1_P_temp__D_0__S__D_1_2__D_3.AllToAllRedistFrom( P_jimb_part0_1_part1_1_P_temp__D_3__S__D_1_2__D_0, modes_0_3 );
			P_jimb_part0_1_part1_1_P_temp__D_3__S__D_1_2__D_0.EmptyData();
			   // P_jimb_part0_1_part1_1_P_temp[D0,D1,D2,D3] <- P_jimb_part0_1_part1_1_P_temp[D0,*,D12,D3]
			P_jimb_part0_1_part1_1_P_temp__D_0__D_1__D_2__D_3.AlignModesWith( modes_0_1_2_3, P_jimb_part0_1_part1_1__D_0__D_1__D_2__D_3, modes_0_1_2_3 );
			P_jimb_part0_1_part1_1_P_temp__D_0__D_1__D_2__D_3.AllToAllRedistFrom( P_jimb_part0_1_part1_1_P_temp__D_0__S__D_1_2__D_3, modes_1_2 );
PROFILE_SECTION("COMPUTE");
PROFILE_FLOPS(2*prod(P_jimb_part0_1_part1_1_P_temp__D_0__D_1__D_2__D_3.Shape()));
			YxpBy( P_jimb_part0_1_part1_1_P_temp__D_0__D_1__D_2__D_3, 1.0, P_jimb_part0_1_part1_1__D_0__D_1__D_2__D_3 );
PROFILE_STOP;
			P_jimb_part0_1_part1_1_P_temp__D_0__S__D_1_2__D_3.EmptyData();
			overwrite_tmpShape_P = P_jimb_part0_1_part1_1__D_0__D_1__D_2__D_3.Shape();
			overwrite_tmpShape_P.push_back( g.Shape()[3] );
			P_jimb_part0_1_part1_1__S__D_2__D_1__D_0__D_3.ResizeTo( overwrite_tmpShape_P );
			   // 1.0 * w_bmje_part2_1[D0,D1,D2,D3]_bmie * t_fj_part1_1[D3,*]_ej + 0.0 * P_jimb_part0_1_part1_1[*,D2,D1,D0,D3]_jimbe
PROFILE_SECTION("COMPUTE");
PROFILE_FLOPS(2*prod(P_jimb_part0_1_part1_1__S__D_2__D_1__D_0__D_3.Shape())*w_bmje_part2_1__D_0__D_1__D_2__D_3.Dimension(3));
			LocalContract(1.0, w_bmje_part2_1__D_0__D_1__D_2__D_3.LockedTensor(), indices_bmie, true,
				t_fj_part1_1__D_3__S.LockedTensor(), indices_ej, true,
				0.0, P_jimb_part0_1_part1_1__S__D_2__D_1__D_0__D_3.Tensor(), indices_jimbe, true);
PROFILE_STOP;
			t_fj_part1_1__D_3__S.EmptyData();
			overwrite_tmpShape_P = P_jimb_part0_1_part1_1__D_0__D_1__D_2__D_3.Shape();
			P_jimb_part0_1_part1_1_P_temp__S__D_2__D_1__D_0_3.ResizeTo( overwrite_tmpShape_P );
			   // Tau_efmn_part3_1_part2_1[D2,D3,*,*] <- Tau_efmn_part3_1_part2_1[D2,D3,D0,D1]
			Tau_efmn_part3_1_part2_1__D_2__D_3__S__S.AlignModesWith( modes_0_1, r_bmfe__D_0__D_1__D_2__D_3, modes_2_3 );
			Tau_efmn_part3_1_part2_1__D_2__D_3__S__S.AllGatherRedistFrom( Tau_efmn_part3_1_part2_1__D_2__D_3__D_0__D_1, modes_0_1 );
			Tau_efmn_part3_1_part2_1__D_2__D_3__D_0__D_1.EmptyData();
			   // P_jimb_part0_1_part1_1_P_temp[*,D2,D1,D03] <- P_jimb_part0_1_part1_1[*,D2,D1,D0,D3] (with SumScatter on D3)
PROFILE_SECTION("COMPUTE");
PROFILE_FLOPS(prod(P_jimb_part0_1_part1_1__S__D_2__D_1__D_0__D_3.Shape()));
			P_jimb_part0_1_part1_1_P_temp__S__D_2__D_1__D_0_3.ReduceScatterRedistFrom( P_jimb_part0_1_part1_1__S__D_2__D_1__D_0__D_3, 4 );
PROFILE_STOP;
			P_jimb_part0_1_part1_1__S__D_2__D_1__D_0__D_3.EmptyData();
			   // P_jimb_part0_1_part1_1_P_temp[*,D1,D2,D03] <- P_jimb_part0_1_part1_1_P_temp[*,D2,D1,D03]
			P_jimb_part0_1_part1_1_P_temp__S__D_1__D_2__D_0_3.AlignModesWith( modes_0_1_2_3, P_jimb_part0_1_part1_1__D_0__D_1__D_2__D_3, modes_0_1_2_3 );
			P_jimb_part0_1_part1_1_P_temp__S__D_1__D_2__D_0_3.AllToAllRedistFrom( P_jimb_part0_1_part1_1_P_temp__S__D_2__D_1__D_0_3, modes_1_2 );
			P_jimb_part0_1_part1_1_P_temp__S__D_2__D_1__D_0_3.EmptyData();
			   // P_jimb_part0_1_part1_1_P_temp[D0,D1,D2,D3] <- P_jimb_part0_1_part1_1_P_temp[*,D1,D2,D03]
			P_jimb_part0_1_part1_1_P_temp__D_0__D_1__D_2__D_3.AlignModesWith( modes_0_1_2_3, P_jimb_part0_1_part1_1__D_0__D_1__D_2__D_3, modes_0_1_2_3 );
			P_jimb_part0_1_part1_1_P_temp__D_0__D_1__D_2__D_3.AllToAllRedistFrom( P_jimb_part0_1_part1_1_P_temp__S__D_1__D_2__D_0_3, modes_0_3 );
PROFILE_SECTION("COMPUTE");
PROFILE_FLOPS(2*prod(P_jimb_part0_1_part1_1_P_temp__D_0__D_1__D_2__D_3.Shape()));
			YxpBy( P_jimb_part0_1_part1_1_P_temp__D_0__D_1__D_2__D_3, 1.0, P_jimb_part0_1_part1_1__D_0__D_1__D_2__D_3 );
PROFILE_STOP;
			overwrite_tmpShape_P = P_jimb_part0_1_part1_1__D_0__D_1__D_2__D_3.Shape();
			overwrite_tmpShape_P.push_back( g.Shape()[2] );
			overwrite_tmpShape_P.push_back( g.Shape()[3] );
			P_jimb_part0_1_part1_1_perm321045__D_0__D_1__S__S__D_2__D_3.ResizeTo( overwrite_tmpShape_P );
			   // 1.0 * r_bmfe[D0,D1,D2,D3]_bmef * Tau_efmn_part3_1_part2_1[D2,D3,*,*]_efij + 0.0 * P_jimb_part0_1_part1_1[*,*,D1,D0,D2,D3]_bmijef
PROFILE_SECTION("COMPUTE");
PROFILE_FLOPS(2*prod(P_jimb_part0_1_part1_1_perm321045__D_0__D_1__S__S__D_2__D_3.Shape())*r_bmfe__D_0__D_1__D_2__D_3.Dimension(2)*r_bmfe__D_0__D_1__D_2__D_3.Dimension(3));
			LocalContract(1.0, r_bmfe__D_0__D_1__D_2__D_3.LockedTensor(), indices_bmef, false,
				Tau_efmn_part3_1_part2_1__D_2__D_3__S__S.LockedTensor(), indices_efij, false,
				0.0, P_jimb_part0_1_part1_1_perm321045__D_0__D_1__S__S__D_2__D_3.Tensor(), indices_bmijef, false);
PROFILE_STOP;
			Tau_efmn_part3_1_part2_1__D_2__D_3__S__S.EmptyData();
			overwrite_tmpShape_P = P_jimb_part0_1_part1_1__D_0__D_1__D_2__D_3.Shape();
			P_jimb_part0_1_part1_1_P_temp__S__S__D_1_2__D_0_3.ResizeTo( overwrite_tmpShape_P );
			   // P_jimb_part0_1_part1_1_P_temp[*,*,D12,D03] <- P_jimb_part0_1_part1_1[*,*,D1,D0,D2,D3] (with SumScatter on (D2)(D3))
PROFILE_SECTION("COMPUTE");
PROFILE_FLOPS(prod(P_jimb_part0_1_part1_1_perm321045__D_0__D_1__S__S__D_2__D_3.Shape()));
			P_jimb_part0_1_part1_1_P_temp__S__S__D_1_2__D_0_3.ReduceScatterRedistFrom( P_jimb_part0_1_part1_1_perm321045__D_0__D_1__S__S__D_2__D_3, modes_5_4 );
PROFILE_STOP;
			P_jimb_part0_1_part1_1_perm321045__D_0__D_1__S__S__D_2__D_3.EmptyData();
			   // P_jimb_part0_1_part1_1_P_temp[*,*,D21,D03] <- P_jimb_part0_1_part1_1_P_temp[*,*,D12,D03]
			P_jimb_part0_1_part1_1_P_temp__S__S__D_2_1__D_0_3.AlignModesWith( modes_0_1_2_3, P_jimb_part0_1_part1_1__D_0__D_1__D_2__D_3, modes_0_1_2_3 );
			P_jimb_part0_1_part1_1_P_temp__S__S__D_2_1__D_0_3.AllToAllRedistFrom( P_jimb_part0_1_part1_1_P_temp__S__S__D_1_2__D_0_3, modes_1_2 );
			P_jimb_part0_1_part1_1_P_temp__S__S__D_1_2__D_0_3.EmptyData();
			   // P_jimb_part0_1_part1_1_P_temp[*,D1,D2,D03] <- P_jimb_part0_1_part1_1_P_temp[*,*,D21,D03]
			P_jimb_part0_1_part1_1_P_temp__S__D_1__D_2__D_0_3.AlignModesWith( modes_0_1_2_3, P_jimb_part0_1_part1_1__D_0__D_1__D_2__D_3, modes_0_1_2_3 );
			P_jimb_part0_1_part1_1_P_temp__S__D_1__D_2__D_0_3.AllToAllRedistFrom( P_jimb_part0_1_part1_1_P_temp__S__S__D_2_1__D_0_3, modes_1 );
			P_jimb_part0_1_part1_1_P_temp__S__S__D_2_1__D_0_3.EmptyData();
			   // P_jimb_part0_1_part1_1_P_temp[D0,D1,D2,D3] <- P_jimb_part0_1_part1_1_P_temp[*,D1,D2,D03]
			P_jimb_part0_1_part1_1_P_temp__D_0__D_1__D_2__D_3.AlignModesWith( modes_0_1_2_3, P_jimb_part0_1_part1_1__D_0__D_1__D_2__D_3, modes_0_1_2_3 );
			P_jimb_part0_1_part1_1_P_temp__D_0__D_1__D_2__D_3.AllToAllRedistFrom( P_jimb_part0_1_part1_1_P_temp__S__D_1__D_2__D_0_3, modes_0_3 );
PROFILE_SECTION("COMPUTE");
PROFILE_FLOPS(2*prod(P_jimb_part0_1_part1_1_P_temp__D_0__D_1__D_2__D_3.Shape()));
			YxpBy( P_jimb_part0_1_part1_1_P_temp__D_0__D_1__D_2__D_3, 1.0, P_jimb_part0_1_part1_1__D_0__D_1__D_2__D_3 );
PROFILE_STOP;
			P_jimb_part0_1_part1_1_P_temp__D_0__D_1__D_2__D_3.EmptyData();
			P_jimb_part0_1_part1_1_P_temp__S__D_1__D_2__D_0_3.EmptyData();

			SlidePartitionDown
			( Tau_efmn_part3_1_part2T__D_0__D_1__D_2__D_3,  Tau_efmn_part3_1_part2_0__D_0__D_1__D_2__D_3,
			       Tau_efmn_part3_1_part2_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  Tau_efmn_part3_1_part2B__D_0__D_1__D_2__D_3, Tau_efmn_part3_1_part2_2__D_0__D_1__D_2__D_3, 2 );
			SlidePartitionDown
			( P_jimb_part0_1_part1T__D_0__D_1__D_2__D_3,  P_jimb_part0_1_part1_0__D_0__D_1__D_2__D_3,
			       P_jimb_part0_1_part1_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  P_jimb_part0_1_part1B__D_0__D_1__D_2__D_3, P_jimb_part0_1_part1_2__D_0__D_1__D_2__D_3, 1 );
			SlidePartitionDown
			( w_bmje_part2T__D_0__D_1__D_2__D_3,  w_bmje_part2_0__D_0__D_1__D_2__D_3,
			       w_bmje_part2_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  w_bmje_part2B__D_0__D_1__D_2__D_3, w_bmje_part2_2__D_0__D_1__D_2__D_3, 2 );
			SlidePartitionDown
			( t_fj_part1T__D_0_1__D_2_3,  t_fj_part1_0__D_0_1__D_2_3,
			       t_fj_part1_1__D_0_1__D_2_3,
			  /**/ /**/
			  t_fj_part1B__D_0_1__D_2_3, t_fj_part1_2__D_0_1__D_2_3, 1 );

		}
		//****

		SlidePartitionDown
		( Tau_efmn_part3T__D_0__D_1__D_2__D_3,  Tau_efmn_part3_0__D_0__D_1__D_2__D_3,
		       Tau_efmn_part3_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  Tau_efmn_part3B__D_0__D_1__D_2__D_3, Tau_efmn_part3_2__D_0__D_1__D_2__D_3, 3 );
		SlidePartitionDown
		( P_jimb_part0T__D_0__D_1__D_2__D_3,  P_jimb_part0_0__D_0__D_1__D_2__D_3,
		       P_jimb_part0_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  P_jimb_part0B__D_0__D_1__D_2__D_3, P_jimb_part0_2__D_0__D_1__D_2__D_3, 0 );
		SlidePartitionDown
		( t_fj_part1T__D_0_1__D_2_3,  t_fj_part1_0__D_0_1__D_2_3,
		       t_fj_part1_1__D_0_1__D_2_3,
		  /**/ /**/
		  t_fj_part1B__D_0_1__D_2_3, t_fj_part1_2__D_0_1__D_2_3, 1 );
		SlidePartitionDown
		( x_bmej_part3T__D_0__D_1__D_2__D_3,  x_bmej_part3_0__D_0__D_1__D_2__D_3,
		       x_bmej_part3_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  x_bmej_part3B__D_0__D_1__D_2__D_3, x_bmej_part3_2__D_0__D_1__D_2__D_3, 3 );

	}
	//****


//****

//END_CODE

    /*****************************************/
    mpi::Barrier(g.OwningComm());
    runTime = mpi::Time() - startTime;
    long long flops = Timer::nflops("COMPUTE");
    gflops = flops / (1e9 * runTime);
#ifdef CORRECTNESS
    DistTensor<double> diff_P(dist__D_0__D_1__D_2__D_3, g);
    diff_P.ResizeTo(check_P);
    Diff(check_P, P_jimb__D_0__D_1__D_2__D_3, diff_P);
   norm = 1.0;
   norm = Norm(diff_P);
   if (commRank == 0){
     std::cout << "NORM_P " << norm << std::endl;
   }
#endif

    //****

    //------------------------------------//

    //****

    if (commRank == 0)
        Timer::printTimers();

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

