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
Permutation perm_0_1;
perm_0_1.push_back(0);
perm_0_1.push_back(1);
Permutation perm_0_1_2_3;
perm_0_1_2_3.push_back(0);
perm_0_1_2_3.push_back(1);
perm_0_1_2_3.push_back(2);
perm_0_1_2_3.push_back(3);
Permutation perm_0_1_2_3_4;
perm_0_1_2_3_4.push_back(0);
perm_0_1_2_3_4.push_back(1);
perm_0_1_2_3_4.push_back(2);
perm_0_1_2_3_4.push_back(3);
perm_0_1_2_3_4.push_back(4);
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
Permutation perm_1_0;
perm_1_0.push_back(1);
perm_1_0.push_back(0);
Permutation perm_1_0_2_3;
perm_1_0_2_3.push_back(1);
perm_1_0_2_3.push_back(0);
perm_1_0_2_3.push_back(2);
perm_1_0_2_3.push_back(3);
Permutation perm_2_0_1_3;
perm_2_0_1_3.push_back(2);
perm_2_0_1_3.push_back(0);
perm_2_0_1_3.push_back(1);
perm_2_0_1_3.push_back(3);
ModeArray modes;
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
ModeArray modes_1;
modes_1.push_back(1);
ModeArray modes_1_0_2_3;
modes_1_0_2_3.push_back(1);
modes_1_0_2_3.push_back(0);
modes_1_0_2_3.push_back(2);
modes_1_0_2_3.push_back(3);
ModeArray modes_2;
modes_2.push_back(2);
ModeArray modes_2_3;
modes_2_3.push_back(2);
modes_2_3.push_back(3);
ModeArray modes_3;
modes_3.push_back(3);
ModeArray modes_4_3_2;
modes_4_3_2.push_back(4);
modes_4_3_2.push_back(3);
modes_4_3_2.push_back(2);
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
	//G_mi_part0B[D01,D23]
DistTensor<double> G_mi_part0B__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//G_mi_part0T[D01,D23]
DistTensor<double> G_mi_part0T__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//G_mi_part0_0[D01,D23]
DistTensor<double> G_mi_part0_0__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//G_mi_part0_1[D01,D23]
DistTensor<double> G_mi_part0_1__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//G_mi_part0_1[*,D2,D0,D1,D3]
DistTensor<double> G_mi_part0_1__S__D_2__D_0__D_1__D_3( dist__S__D_2__D_0__D_1__D_3, g );
	//G_mi_part0_2[D01,D23]
DistTensor<double> G_mi_part0_2__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//H_me[D01,D23]
DistTensor<double> H_me__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//H_me_part0B[D01,D23]
DistTensor<double> H_me_part0B__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//H_me_part0T[D01,D23]
DistTensor<double> H_me_part0T__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//H_me_part0_0[D01,D23]
DistTensor<double> H_me_part0_0__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//H_me_part0_1[D01,D23]
DistTensor<double> H_me_part0_1__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//H_me_part0_1_part1B[D01,D23]
DistTensor<double> H_me_part0_1_part1B__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//H_me_part0_1_part1T[D01,D23]
DistTensor<double> H_me_part0_1_part1T__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//H_me_part0_1_part1_0[D01,D23]
DistTensor<double> H_me_part0_1_part1_0__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//H_me_part0_1_part1_1[D01,D23]
DistTensor<double> H_me_part0_1_part1_1__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//H_me_part0_1_part1_1[D01,*]
DistTensor<double> H_me_part0_1_part1_1__D_0_1__S( dist__D_0_1__S, g );
	//H_me_part0_1_part1_2[D01,D23]
DistTensor<double> H_me_part0_1_part1_2__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//H_me_part0_2[D01,D23]
DistTensor<double> H_me_part0_2__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//T_bfnj[D0,D1,D2,D3]
DistTensor<double> T_bfnj__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//T_bfnj_part3B[D0,D1,D2,D3]
DistTensor<double> T_bfnj_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//T_bfnj_part3T[D0,D1,D2,D3]
DistTensor<double> T_bfnj_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//T_bfnj_part3_0[D0,D1,D2,D3]
DistTensor<double> T_bfnj_part3_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//T_bfnj_part3_1[D0,D1,D2,D3]
DistTensor<double> T_bfnj_part3_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//T_bfnj_part3_1[D0,D1,D2,D3]
DistTensor<double> T_bfnj_part3_1_perm0132__D_0__D_1__D_3__D_2( dist__D_0__D_1__D_2__D_3, g );
T_bfnj_part3_1_perm0132__D_0__D_1__D_3__D_2.SetLocalPermutation( perm_0_1_3_2 );
	//T_bfnj_part3_2[D0,D1,D2,D3]
DistTensor<double> T_bfnj_part3_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//t_fj[D01,D23]
DistTensor<double> t_fj__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//t_fj_part0B[D01,D23]
DistTensor<double> t_fj_part0B__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//t_fj_part0T[D01,D23]
DistTensor<double> t_fj_part0T__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//t_fj_part0_0[D01,D23]
DistTensor<double> t_fj_part0_0__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//t_fj_part0_1[D01,D23]
DistTensor<double> t_fj_part0_1__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//t_fj_part0_1[*,D23]
DistTensor<double> t_fj_part0_1__S__D_2_3( dist__S__D_2_3, g );
	//t_fj_part0_1_part1B[D01,D23]
DistTensor<double> t_fj_part0_1_part1B__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//t_fj_part0_1_part1T[D01,D23]
DistTensor<double> t_fj_part0_1_part1T__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//t_fj_part0_1_part1_0[D01,D23]
DistTensor<double> t_fj_part0_1_part1_0__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//t_fj_part0_1_part1_1[D01,D23]
DistTensor<double> t_fj_part0_1_part1_1__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//t_fj_part0_1_part1_1[*,*]
DistTensor<double> t_fj_part0_1_part1_1_perm10__S__S( dist__S__S, g );
t_fj_part0_1_part1_1_perm10__S__S.SetLocalPermutation( perm_1_0 );
	//t_fj_part0_1_part1_2[D01,D23]
DistTensor<double> t_fj_part0_1_part1_2__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//t_fj_part0_2[D01,D23]
DistTensor<double> t_fj_part0_2__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
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
	//temp1_part3_1_part2_1[D0,D1,*,D3]
DistTensor<double> temp1_part3_1_part2_1_perm2013__S__D_0__D_1__D_3( dist__D_0__D_1__S__D_3, g );
temp1_part3_1_part2_1_perm2013__S__D_0__D_1__D_3.SetLocalPermutation( perm_2_0_1_3 );
	//temp1_part3_1_part2_2[D0,D1,D2,D3]
DistTensor<double> temp1_part3_1_part2_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp1_part3_2[D0,D1,D2,D3]
DistTensor<double> temp1_part3_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp2[D0,D1,D2,D3]
DistTensor<double> temp2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp2_part0B[D0,D1,D2,D3]
DistTensor<double> temp2_part0B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp2_part0T[D0,D1,D2,D3]
DistTensor<double> temp2_part0T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp2_part0_0[D0,D1,D2,D3]
DistTensor<double> temp2_part0_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp2_part0_1[D0,D1,D2,D3]
DistTensor<double> temp2_part0_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp2_part0_1_part1B[D0,D1,D2,D3]
DistTensor<double> temp2_part0_1_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp2_part0_1_part1T[D0,D1,D2,D3]
DistTensor<double> temp2_part0_1_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp2_part0_1_part1_0[D0,D1,D2,D3]
DistTensor<double> temp2_part0_1_part1_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp2_part0_1_part1_1[D0,D1,D2,D3]
DistTensor<double> temp2_part0_1_part1_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp2_part0_1_part1_2[D0,D1,D2,D3]
DistTensor<double> temp2_part0_1_part1_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp2_part0_2[D0,D1,D2,D3]
DistTensor<double> temp2_part0_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp2_part3B[D0,D1,D2,D3]
DistTensor<double> temp2_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp2_part3T[D0,D1,D2,D3]
DistTensor<double> temp2_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp2_part3_0[D0,D1,D2,D3]
DistTensor<double> temp2_part3_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp2_part3_1[D0,D1,D2,D3]
DistTensor<double> temp2_part3_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp2_part3_1_part1B[D0,D1,D2,D3]
DistTensor<double> temp2_part3_1_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp2_part3_1_part1T[D0,D1,D2,D3]
DistTensor<double> temp2_part3_1_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp2_part3_1_part1_0[D0,D1,D2,D3]
DistTensor<double> temp2_part3_1_part1_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp2_part3_1_part1_1[D0,D1,D23,*]
DistTensor<double> temp2_part3_1_part1_1__D_0__D_1__D_2_3__S( dist__D_0__D_1__D_2_3__S, g );
	//temp2_part3_1_part1_1[D0,D1,D2,D3]
DistTensor<double> temp2_part3_1_part1_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp2_part3_1_part1_1[D0,D1,D2,*]
DistTensor<double> temp2_part3_1_part1_1__D_0__D_1__D_2__S( dist__D_0__D_1__D_2__S, g );
	//temp2_part3_1_part1_1[D0,*,D23,*]
DistTensor<double> temp2_part3_1_part1_1__D_0__S__D_2_3__S( dist__D_0__S__D_2_3__S, g );
	//temp2_part3_1_part1_1[D01,*,D23,*]
DistTensor<double> temp2_part3_1_part1_1_perm0213__D_0_1__D_2_3__S__S( dist__D_0_1__S__D_2_3__S, g );
temp2_part3_1_part1_1_perm0213__D_0_1__D_2_3__S__S.SetLocalPermutation( perm_0_2_1_3 );
	//temp2_part3_1_part1_2[D0,D1,D2,D3]
DistTensor<double> temp2_part3_1_part1_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp2_part3_2[D0,D1,D2,D3]
DistTensor<double> temp2_part3_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//u_mnje[D0,D1,D2,D3]
DistTensor<double> u_mnje__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//u_mnje_part0B[D0,D1,D2,D3]
DistTensor<double> u_mnje_part0B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//u_mnje_part0T[D0,D1,D2,D3]
DistTensor<double> u_mnje_part0T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//u_mnje_part0_0[D0,D1,D2,D3]
DistTensor<double> u_mnje_part0_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//u_mnje_part0_1[D0,D1,D2,D3]
DistTensor<double> u_mnje_part0_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//u_mnje_part0_1_part1B[D0,D1,D2,D3]
DistTensor<double> u_mnje_part0_1_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//u_mnje_part0_1_part1T[D0,D1,D2,D3]
DistTensor<double> u_mnje_part0_1_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//u_mnje_part0_1_part1_0[D0,D1,D2,D3]
DistTensor<double> u_mnje_part0_1_part1_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//u_mnje_part0_1_part1_1[D0,D1,D2,D3]
DistTensor<double> u_mnje_part0_1_part1_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//u_mnje_part0_1_part1_2[D0,D1,D2,D3]
DistTensor<double> u_mnje_part0_1_part1_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//u_mnje_part0_2[D0,D1,D2,D3]
DistTensor<double> u_mnje_part0_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//u_mnje_part1B[D0,D1,D2,D3]
DistTensor<double> u_mnje_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//u_mnje_part1T[D0,D1,D2,D3]
DistTensor<double> u_mnje_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//u_mnje_part1_0[D0,D1,D2,D3]
DistTensor<double> u_mnje_part1_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//u_mnje_part1_1[D0,D1,D2,D3]
DistTensor<double> u_mnje_part1_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//u_mnje_part1_1_part0B[D0,D1,D2,D3]
DistTensor<double> u_mnje_part1_1_part0B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//u_mnje_part1_1_part0T[D0,D1,D2,D3]
DistTensor<double> u_mnje_part1_1_part0T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//u_mnje_part1_1_part0_0[D0,D1,D2,D3]
DistTensor<double> u_mnje_part1_1_part0_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//u_mnje_part1_1_part0_1[D0,D1,D2,D3]
DistTensor<double> u_mnje_part1_1_part0_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//u_mnje_part1_1_part0_1[D1,D0,D2,D3]
DistTensor<double> u_mnje_part1_1_part0_1__D_1__D_0__D_2__D_3( dist__D_1__D_0__D_2__D_3, g );
	//u_mnje_part1_1_part0_2[D0,D1,D2,D3]
DistTensor<double> u_mnje_part1_1_part0_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//u_mnje_part1_2[D0,D1,D2,D3]
DistTensor<double> u_mnje_part1_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//v_femn[D0,D1,D2,D3]
DistTensor<double> v_femn__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//v_femn_part2B[D0,D1,D2,D3]
DistTensor<double> v_femn_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//v_femn_part2T[D0,D1,D2,D3]
DistTensor<double> v_femn_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//v_femn_part2_0[D0,D1,D2,D3]
DistTensor<double> v_femn_part2_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//v_femn_part2_1[D0,D1,D2,D3]
DistTensor<double> v_femn_part2_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//v_femn_part2_1_part3B[D0,D1,D2,D3]
DistTensor<double> v_femn_part2_1_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//v_femn_part2_1_part3T[D0,D1,D2,D3]
DistTensor<double> v_femn_part2_1_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//v_femn_part2_1_part3_0[D0,D1,D2,D3]
DistTensor<double> v_femn_part2_1_part3_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//v_femn_part2_1_part3_1[D0,D1,D2,D3]
DistTensor<double> v_femn_part2_1_part3_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//v_femn_part2_1_part3_2[D0,D1,D2,D3]
DistTensor<double> v_femn_part2_1_part3_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//v_femn_part2_2[D0,D1,D2,D3]
DistTensor<double> v_femn_part2_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//v_femn_part3B[D0,D1,D2,D3]
DistTensor<double> v_femn_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//v_femn_part3T[D0,D1,D2,D3]
DistTensor<double> v_femn_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//v_femn_part3_0[D0,D1,D2,D3]
DistTensor<double> v_femn_part3_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//v_femn_part3_1[D0,D1,D2,D3]
DistTensor<double> v_femn_part3_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//v_femn_part3_1_part2B[D0,D1,D2,D3]
DistTensor<double> v_femn_part3_1_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//v_femn_part3_1_part2T[D0,D1,D2,D3]
DistTensor<double> v_femn_part3_1_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//v_femn_part3_1_part2_0[D0,D1,D2,D3]
DistTensor<double> v_femn_part3_1_part2_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//v_femn_part3_1_part2_1[D0,D1,D2,D3]
DistTensor<double> v_femn_part3_1_part2_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//v_femn_part3_1_part2_1[D0,D1,D3,D2]
DistTensor<double> v_femn_part3_1_part2_1__D_0__D_1__D_3__D_2( dist__D_0__D_1__D_3__D_2, g );
	//v_femn_part3_1_part2_2[D0,D1,D2,D3]
DistTensor<double> v_femn_part3_1_part2_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//v_femn_part3_2[D0,D1,D2,D3]
DistTensor<double> v_femn_part3_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
// H_me has 2 dims
//	Starting distribution: [D01,D23] or _D_0_1__D_2_3
ObjShape H_me__D_0_1__D_2_3_tempShape;
H_me__D_0_1__D_2_3_tempShape.push_back( n_o );
H_me__D_0_1__D_2_3_tempShape.push_back( n_v );
H_me__D_0_1__D_2_3.ResizeTo( H_me__D_0_1__D_2_3_tempShape );
MakeUniform( H_me__D_0_1__D_2_3 );
// t_fj has 2 dims
//	Starting distribution: [D01,D23] or _D_0_1__D_2_3
ObjShape t_fj__D_0_1__D_2_3_tempShape;
t_fj__D_0_1__D_2_3_tempShape.push_back( n_v );
t_fj__D_0_1__D_2_3_tempShape.push_back( n_o );
t_fj__D_0_1__D_2_3.ResizeTo( t_fj__D_0_1__D_2_3_tempShape );
MakeUniform( t_fj__D_0_1__D_2_3 );
// u_mnje has 4 dims
//	Starting distribution: [D0,D1,D2,D3] or _D_0__D_1__D_2__D_3
ObjShape u_mnje__D_0__D_1__D_2__D_3_tempShape;
u_mnje__D_0__D_1__D_2__D_3_tempShape.push_back( n_o );
u_mnje__D_0__D_1__D_2__D_3_tempShape.push_back( n_o );
u_mnje__D_0__D_1__D_2__D_3_tempShape.push_back( n_o );
u_mnje__D_0__D_1__D_2__D_3_tempShape.push_back( n_v );
u_mnje__D_0__D_1__D_2__D_3.ResizeTo( u_mnje__D_0__D_1__D_2__D_3_tempShape );
MakeUniform( u_mnje__D_0__D_1__D_2__D_3 );
tempShape = u_mnje__D_0__D_1__D_2__D_3.Shape();
temp2__D_0__D_1__D_2__D_3.ResizeTo( tempShape );
// v_femn has 4 dims
//	Starting distribution: [D0,D1,D2,D3] or _D_0__D_1__D_2__D_3
ObjShape v_femn__D_0__D_1__D_2__D_3_tempShape;
v_femn__D_0__D_1__D_2__D_3_tempShape.push_back( n_v );
v_femn__D_0__D_1__D_2__D_3_tempShape.push_back( n_v );
v_femn__D_0__D_1__D_2__D_3_tempShape.push_back( n_o );
v_femn__D_0__D_1__D_2__D_3_tempShape.push_back( n_o );
v_femn__D_0__D_1__D_2__D_3.ResizeTo( v_femn__D_0__D_1__D_2__D_3_tempShape );
MakeUniform( v_femn__D_0__D_1__D_2__D_3 );
tempShape = v_femn__D_0__D_1__D_2__D_3.Shape();
temp1__D_0__D_1__D_2__D_3.ResizeTo( tempShape );
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
//**** (out of 1)

//******************************
//* Load tensors
//******************************
////////////////////////////////
//Performance testing
////////////////////////////////
std::stringstream fullName;
#ifdef CORRECTNESS
DistTensor<T> check(dist__D_0_1__D_2_3, g);
check.ResizeTo(G_mi__D_0_1__D_2_3.Shape());
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


	Scal( 0.0, G_mi__D_0_1__D_2_3 );
	//**** (out of 1)
	//**** Is real	0 shadows
		//Outputs:
		//  temp2__D_0__D_1__D_2__D_3
	PartitionDown(u_mnje__D_0__D_1__D_2__D_3, u_mnje_part0T__D_0__D_1__D_2__D_3, u_mnje_part0B__D_0__D_1__D_2__D_3, 0, 0);
	PartitionDown(u_mnje__D_0__D_1__D_2__D_3, u_mnje_part1T__D_0__D_1__D_2__D_3, u_mnje_part1B__D_0__D_1__D_2__D_3, 1, 0);
	PartitionDown(temp2__D_0__D_1__D_2__D_3, temp2_part0T__D_0__D_1__D_2__D_3, temp2_part0B__D_0__D_1__D_2__D_3, 0, 0);
	while(temp2_part0T__D_0__D_1__D_2__D_3.Dimension(0) < temp2__D_0__D_1__D_2__D_3.Dimension(0))
	{
		RepartitionDown
		( u_mnje_part0T__D_0__D_1__D_2__D_3,  u_mnje_part0_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       u_mnje_part0_1__D_0__D_1__D_2__D_3,
		  u_mnje_part0B__D_0__D_1__D_2__D_3, u_mnje_part0_2__D_0__D_1__D_2__D_3, 0, blkSize );
		RepartitionDown
		( u_mnje_part1T__D_0__D_1__D_2__D_3,  u_mnje_part1_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       u_mnje_part1_1__D_0__D_1__D_2__D_3,
		  u_mnje_part1B__D_0__D_1__D_2__D_3, u_mnje_part1_2__D_0__D_1__D_2__D_3, 1, blkSize );
		RepartitionDown
		( temp2_part0T__D_0__D_1__D_2__D_3,  temp2_part0_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       temp2_part0_1__D_0__D_1__D_2__D_3,
		  temp2_part0B__D_0__D_1__D_2__D_3, temp2_part0_2__D_0__D_1__D_2__D_3, 0, blkSize );

		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  temp2_part0_1__D_0__D_1__D_2__D_3
		PartitionDown(u_mnje_part0_1__D_0__D_1__D_2__D_3, u_mnje_part0_1_part1T__D_0__D_1__D_2__D_3, u_mnje_part0_1_part1B__D_0__D_1__D_2__D_3, 1, 0);
		PartitionDown(u_mnje_part1_1__D_0__D_1__D_2__D_3, u_mnje_part1_1_part0T__D_0__D_1__D_2__D_3, u_mnje_part1_1_part0B__D_0__D_1__D_2__D_3, 0, 0);
		PartitionDown(temp2_part0_1__D_0__D_1__D_2__D_3, temp2_part0_1_part1T__D_0__D_1__D_2__D_3, temp2_part0_1_part1B__D_0__D_1__D_2__D_3, 1, 0);
		while(temp2_part0_1_part1T__D_0__D_1__D_2__D_3.Dimension(1) < temp2_part0_1__D_0__D_1__D_2__D_3.Dimension(1))
		{
			RepartitionDown
			( u_mnje_part0_1_part1T__D_0__D_1__D_2__D_3,  u_mnje_part0_1_part1_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       u_mnje_part0_1_part1_1__D_0__D_1__D_2__D_3,
			  u_mnje_part0_1_part1B__D_0__D_1__D_2__D_3, u_mnje_part0_1_part1_2__D_0__D_1__D_2__D_3, 1, blkSize );
			RepartitionDown
			( u_mnje_part1_1_part0T__D_0__D_1__D_2__D_3,  u_mnje_part1_1_part0_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       u_mnje_part1_1_part0_1__D_0__D_1__D_2__D_3,
			  u_mnje_part1_1_part0B__D_0__D_1__D_2__D_3, u_mnje_part1_1_part0_2__D_0__D_1__D_2__D_3, 0, blkSize );
			RepartitionDown
			( temp2_part0_1_part1T__D_0__D_1__D_2__D_3,  temp2_part0_1_part1_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       temp2_part0_1_part1_1__D_0__D_1__D_2__D_3,
			  temp2_part0_1_part1B__D_0__D_1__D_2__D_3, temp2_part0_1_part1_2__D_0__D_1__D_2__D_3, 1, blkSize );

			   // u_mnje_part1_1_part0_1[D1,D0,D2,D3] <- u_mnje_part1_1_part0_1[D0,D1,D2,D3]
			u_mnje_part1_1_part0_1__D_1__D_0__D_2__D_3.AlignModesWith( modes_0_1_2_3, u_mnje_part0_1_part1_1__D_0__D_1__D_2__D_3, modes_1_0_2_3 );
			u_mnje_part1_1_part0_1__D_1__D_0__D_2__D_3.AllToAllRedistFrom( u_mnje_part1_1_part0_1__D_0__D_1__D_2__D_3, modes_0_1 );
			YAxpPx( 2.0, u_mnje_part0_1_part1_1__D_0__D_1__D_2__D_3, -1.0, u_mnje_part1_1_part0_1__D_1__D_0__D_2__D_3, perm_1_0_2_3, temp2_part0_1_part1_1__D_0__D_1__D_2__D_3 );
			u_mnje_part1_1_part0_1__D_1__D_0__D_2__D_3.EmptyData();

			SlidePartitionDown
			( u_mnje_part0_1_part1T__D_0__D_1__D_2__D_3,  u_mnje_part0_1_part1_0__D_0__D_1__D_2__D_3,
			       u_mnje_part0_1_part1_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  u_mnje_part0_1_part1B__D_0__D_1__D_2__D_3, u_mnje_part0_1_part1_2__D_0__D_1__D_2__D_3, 1 );
			SlidePartitionDown
			( u_mnje_part1_1_part0T__D_0__D_1__D_2__D_3,  u_mnje_part1_1_part0_0__D_0__D_1__D_2__D_3,
			       u_mnje_part1_1_part0_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  u_mnje_part1_1_part0B__D_0__D_1__D_2__D_3, u_mnje_part1_1_part0_2__D_0__D_1__D_2__D_3, 0 );
			SlidePartitionDown
			( temp2_part0_1_part1T__D_0__D_1__D_2__D_3,  temp2_part0_1_part1_0__D_0__D_1__D_2__D_3,
			       temp2_part0_1_part1_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  temp2_part0_1_part1B__D_0__D_1__D_2__D_3, temp2_part0_1_part1_2__D_0__D_1__D_2__D_3, 1 );

		}
		//****

		SlidePartitionDown
		( u_mnje_part0T__D_0__D_1__D_2__D_3,  u_mnje_part0_0__D_0__D_1__D_2__D_3,
		       u_mnje_part0_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  u_mnje_part0B__D_0__D_1__D_2__D_3, u_mnje_part0_2__D_0__D_1__D_2__D_3, 0 );
		SlidePartitionDown
		( u_mnje_part1T__D_0__D_1__D_2__D_3,  u_mnje_part1_0__D_0__D_1__D_2__D_3,
		       u_mnje_part1_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  u_mnje_part1B__D_0__D_1__D_2__D_3, u_mnje_part1_2__D_0__D_1__D_2__D_3, 1 );
		SlidePartitionDown
		( temp2_part0T__D_0__D_1__D_2__D_3,  temp2_part0_0__D_0__D_1__D_2__D_3,
		       temp2_part0_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  temp2_part0B__D_0__D_1__D_2__D_3, temp2_part0_2__D_0__D_1__D_2__D_3, 0 );

	}
	u_mnje__D_0__D_1__D_2__D_3.EmptyData();
	u_mnje__D_0__D_1__D_2__D_3.EmptyData();
	u_mnje__D_0__D_1__D_2__D_3.EmptyData();
	u_mnje__D_0__D_1__D_2__D_3.EmptyData();
	//****
	//**** (out of 1)
	//**** Is real	0 shadows
		//Outputs:
		//  temp1__D_0__D_1__D_2__D_3
	PartitionDown(v_femn__D_0__D_1__D_2__D_3, v_femn_part2T__D_0__D_1__D_2__D_3, v_femn_part2B__D_0__D_1__D_2__D_3, 2, 0);
	PartitionDown(v_femn__D_0__D_1__D_2__D_3, v_femn_part3T__D_0__D_1__D_2__D_3, v_femn_part3B__D_0__D_1__D_2__D_3, 3, 0);
	PartitionDown(temp1__D_0__D_1__D_2__D_3, temp1_part2T__D_0__D_1__D_2__D_3, temp1_part2B__D_0__D_1__D_2__D_3, 2, 0);
	while(temp1_part2T__D_0__D_1__D_2__D_3.Dimension(2) < temp1__D_0__D_1__D_2__D_3.Dimension(2))
	{
		RepartitionDown
		( v_femn_part2T__D_0__D_1__D_2__D_3,  v_femn_part2_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       v_femn_part2_1__D_0__D_1__D_2__D_3,
		  v_femn_part2B__D_0__D_1__D_2__D_3, v_femn_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );
		RepartitionDown
		( v_femn_part3T__D_0__D_1__D_2__D_3,  v_femn_part3_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       v_femn_part3_1__D_0__D_1__D_2__D_3,
		  v_femn_part3B__D_0__D_1__D_2__D_3, v_femn_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );
		RepartitionDown
		( temp1_part2T__D_0__D_1__D_2__D_3,  temp1_part2_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       temp1_part2_1__D_0__D_1__D_2__D_3,
		  temp1_part2B__D_0__D_1__D_2__D_3, temp1_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );

		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  temp1_part2_1__D_0__D_1__D_2__D_3
		PartitionDown(v_femn_part2_1__D_0__D_1__D_2__D_3, v_femn_part2_1_part3T__D_0__D_1__D_2__D_3, v_femn_part2_1_part3B__D_0__D_1__D_2__D_3, 3, 0);
		PartitionDown(v_femn_part3_1__D_0__D_1__D_2__D_3, v_femn_part3_1_part2T__D_0__D_1__D_2__D_3, v_femn_part3_1_part2B__D_0__D_1__D_2__D_3, 2, 0);
		PartitionDown(temp1_part2_1__D_0__D_1__D_2__D_3, temp1_part2_1_part3T__D_0__D_1__D_2__D_3, temp1_part2_1_part3B__D_0__D_1__D_2__D_3, 3, 0);
		while(temp1_part2_1_part3T__D_0__D_1__D_2__D_3.Dimension(3) < temp1_part2_1__D_0__D_1__D_2__D_3.Dimension(3))
		{
			RepartitionDown
			( v_femn_part2_1_part3T__D_0__D_1__D_2__D_3,  v_femn_part2_1_part3_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       v_femn_part2_1_part3_1__D_0__D_1__D_2__D_3,
			  v_femn_part2_1_part3B__D_0__D_1__D_2__D_3, v_femn_part2_1_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );
			RepartitionDown
			( v_femn_part3_1_part2T__D_0__D_1__D_2__D_3,  v_femn_part3_1_part2_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       v_femn_part3_1_part2_1__D_0__D_1__D_2__D_3,
			  v_femn_part3_1_part2B__D_0__D_1__D_2__D_3, v_femn_part3_1_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );
			RepartitionDown
			( temp1_part2_1_part3T__D_0__D_1__D_2__D_3,  temp1_part2_1_part3_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       temp1_part2_1_part3_1__D_0__D_1__D_2__D_3,
			  temp1_part2_1_part3B__D_0__D_1__D_2__D_3, temp1_part2_1_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );

			   // v_femn_part3_1_part2_1[D0,D1,D3,D2] <- v_femn_part3_1_part2_1[D0,D1,D2,D3]
			v_femn_part3_1_part2_1__D_0__D_1__D_3__D_2.AlignModesWith( modes_0_1_2_3, v_femn_part2_1_part3_1__D_0__D_1__D_2__D_3, modes_0_1_3_2 );
			v_femn_part3_1_part2_1__D_0__D_1__D_3__D_2.AllToAllRedistFrom( v_femn_part3_1_part2_1__D_0__D_1__D_2__D_3, modes_2_3 );
			YAxpPx( 2.0, v_femn_part2_1_part3_1__D_0__D_1__D_2__D_3, -1.0, v_femn_part3_1_part2_1__D_0__D_1__D_3__D_2, perm_0_1_3_2, temp1_part2_1_part3_1__D_0__D_1__D_2__D_3 );
			v_femn_part3_1_part2_1__D_0__D_1__D_3__D_2.EmptyData();

			SlidePartitionDown
			( v_femn_part2_1_part3T__D_0__D_1__D_2__D_3,  v_femn_part2_1_part3_0__D_0__D_1__D_2__D_3,
			       v_femn_part2_1_part3_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  v_femn_part2_1_part3B__D_0__D_1__D_2__D_3, v_femn_part2_1_part3_2__D_0__D_1__D_2__D_3, 3 );
			SlidePartitionDown
			( v_femn_part3_1_part2T__D_0__D_1__D_2__D_3,  v_femn_part3_1_part2_0__D_0__D_1__D_2__D_3,
			       v_femn_part3_1_part2_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  v_femn_part3_1_part2B__D_0__D_1__D_2__D_3, v_femn_part3_1_part2_2__D_0__D_1__D_2__D_3, 2 );
			SlidePartitionDown
			( temp1_part2_1_part3T__D_0__D_1__D_2__D_3,  temp1_part2_1_part3_0__D_0__D_1__D_2__D_3,
			       temp1_part2_1_part3_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  temp1_part2_1_part3B__D_0__D_1__D_2__D_3, temp1_part2_1_part3_2__D_0__D_1__D_2__D_3, 3 );

		}
		//****

		SlidePartitionDown
		( v_femn_part2T__D_0__D_1__D_2__D_3,  v_femn_part2_0__D_0__D_1__D_2__D_3,
		       v_femn_part2_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  v_femn_part2B__D_0__D_1__D_2__D_3, v_femn_part2_2__D_0__D_1__D_2__D_3, 2 );
		SlidePartitionDown
		( v_femn_part3T__D_0__D_1__D_2__D_3,  v_femn_part3_0__D_0__D_1__D_2__D_3,
		       v_femn_part3_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  v_femn_part3B__D_0__D_1__D_2__D_3, v_femn_part3_2__D_0__D_1__D_2__D_3, 3 );
		SlidePartitionDown
		( temp1_part2T__D_0__D_1__D_2__D_3,  temp1_part2_0__D_0__D_1__D_2__D_3,
		       temp1_part2_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  temp1_part2B__D_0__D_1__D_2__D_3, temp1_part2_2__D_0__D_1__D_2__D_3, 2 );

	}
	v_femn__D_0__D_1__D_2__D_3.EmptyData();
	v_femn__D_0__D_1__D_2__D_3.EmptyData();
	v_femn__D_0__D_1__D_2__D_3.EmptyData();
	v_femn__D_0__D_1__D_2__D_3.EmptyData();
	//****
	//**** (out of 1)
	//**** Is real	0 shadows
		//Outputs:
		//  G_mi__D_0_1__D_2_3
	PartitionDown(temp1__D_0__D_1__D_2__D_3, temp1_part3T__D_0__D_1__D_2__D_3, temp1_part3B__D_0__D_1__D_2__D_3, 3, 0);
	PartitionDown(T_bfnj__D_0__D_1__D_2__D_3, T_bfnj_part3T__D_0__D_1__D_2__D_3, T_bfnj_part3B__D_0__D_1__D_2__D_3, 3, 0);
	while(temp1_part3T__D_0__D_1__D_2__D_3.Dimension(3) < temp1__D_0__D_1__D_2__D_3.Dimension(3))
	{
		RepartitionDown
		( temp1_part3T__D_0__D_1__D_2__D_3,  temp1_part3_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       temp1_part3_1__D_0__D_1__D_2__D_3,
		  temp1_part3B__D_0__D_1__D_2__D_3, temp1_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );
		RepartitionDown
		( T_bfnj_part3T__D_0__D_1__D_2__D_3,  T_bfnj_part3_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       T_bfnj_part3_1__D_0__D_1__D_2__D_3,
		  T_bfnj_part3B__D_0__D_1__D_2__D_3, T_bfnj_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );

		Permute( T_bfnj_part3_1__D_0__D_1__D_2__D_3, T_bfnj_part3_1_perm0132__D_0__D_1__D_3__D_2 );
		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  G_mi__D_0_1__D_2_3
		PartitionDown(temp1_part3_1__D_0__D_1__D_2__D_3, temp1_part3_1_part2T__D_0__D_1__D_2__D_3, temp1_part3_1_part2B__D_0__D_1__D_2__D_3, 2, 0);
		PartitionDown(G_mi__D_0_1__D_2_3, G_mi_part0T__D_0_1__D_2_3, G_mi_part0B__D_0_1__D_2_3, 0, 0);
		while(G_mi_part0T__D_0_1__D_2_3.Dimension(0) < G_mi__D_0_1__D_2_3.Dimension(0))
		{
			RepartitionDown
			( temp1_part3_1_part2T__D_0__D_1__D_2__D_3,  temp1_part3_1_part2_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       temp1_part3_1_part2_1__D_0__D_1__D_2__D_3,
			  temp1_part3_1_part2B__D_0__D_1__D_2__D_3, temp1_part3_1_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );
			RepartitionDown
			( G_mi_part0T__D_0_1__D_2_3,  G_mi_part0_0__D_0_1__D_2_3,
			  /**/ /**/
			       G_mi_part0_1__D_0_1__D_2_3,
			  G_mi_part0B__D_0_1__D_2_3, G_mi_part0_2__D_0_1__D_2_3, 0, blkSize );

			tempShape = G_mi_part0_1__D_0_1__D_2_3.Shape();
			tempShape.push_back( g.Shape()[0] );
			tempShape.push_back( g.Shape()[1] );
			tempShape.push_back( g.Shape()[3] );
			G_mi_part0_1__S__D_2__D_0__D_1__D_3.ResizeTo( tempShape );
			   // temp1_part3_1_part2_1[D0,D1,*,D3] <- temp1_part3_1_part2_1[D0,D1,D2,D3]
			temp1_part3_1_part2_1_perm2013__S__D_0__D_1__D_3.AlignModesWith( modes_0_1_3, T_bfnj_part3_1__D_0__D_1__D_2__D_3, modes_0_1_3 );
			temp1_part3_1_part2_1_perm2013__S__D_0__D_1__D_3.AllGatherRedistFrom( temp1_part3_1_part2_1__D_0__D_1__D_2__D_3, modes_2 );
			   // 1.0 * temp1_part3_1_part2_1[D0,D1,*,D3]_mefn * T_bfnj_part3_1[D0,D1,D2,D3]_efni + 0.0 * G_mi_part0_1[*,D2,D0,D1,D3]_miefn
			LocalContract(1.0, temp1_part3_1_part2_1_perm2013__S__D_0__D_1__D_3.LockedTensor(), indices_mefn, false,
				T_bfnj_part3_1_perm0132__D_0__D_1__D_3__D_2.LockedTensor(), indices_efni, false,
				0.0, G_mi_part0_1__S__D_2__D_0__D_1__D_3.Tensor(), indices_miefn, false);
			   // G_mi_part0_1[D01,D23] <- G_mi_part0_1[*,D2,D0,D1,D3] (with SumScatter on (D0)(D1)(D3))
			G_mi_part0_1__D_0_1__D_2_3.ReduceScatterUpdateRedistFrom( G_mi_part0_1__S__D_2__D_0__D_1__D_3, 1.0, modes_4_3_2 );
			G_mi_part0_1__S__D_2__D_0__D_1__D_3.EmptyData();
			temp1_part3_1_part2_1_perm2013__S__D_0__D_1__D_3.EmptyData();

			SlidePartitionDown
			( temp1_part3_1_part2T__D_0__D_1__D_2__D_3,  temp1_part3_1_part2_0__D_0__D_1__D_2__D_3,
			       temp1_part3_1_part2_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  temp1_part3_1_part2B__D_0__D_1__D_2__D_3, temp1_part3_1_part2_2__D_0__D_1__D_2__D_3, 2 );
			SlidePartitionDown
			( G_mi_part0T__D_0_1__D_2_3,  G_mi_part0_0__D_0_1__D_2_3,
			       G_mi_part0_1__D_0_1__D_2_3,
			  /**/ /**/
			  G_mi_part0B__D_0_1__D_2_3, G_mi_part0_2__D_0_1__D_2_3, 0 );

		}
		T_bfnj_part3_1_perm0132__D_0__D_1__D_3__D_2.EmptyData();
		T_bfnj_part3_1_perm0132__D_0__D_1__D_3__D_2.EmptyData();
		//****

		SlidePartitionDown
		( temp1_part3T__D_0__D_1__D_2__D_3,  temp1_part3_0__D_0__D_1__D_2__D_3,
		       temp1_part3_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  temp1_part3B__D_0__D_1__D_2__D_3, temp1_part3_2__D_0__D_1__D_2__D_3, 3 );
		SlidePartitionDown
		( T_bfnj_part3T__D_0__D_1__D_2__D_3,  T_bfnj_part3_0__D_0__D_1__D_2__D_3,
		       T_bfnj_part3_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  T_bfnj_part3B__D_0__D_1__D_2__D_3, T_bfnj_part3_2__D_0__D_1__D_2__D_3, 3 );

	}
	temp1__D_0__D_1__D_2__D_3.EmptyData();
	T_bfnj__D_0__D_1__D_2__D_3.EmptyData();
	temp1__D_0__D_1__D_2__D_3.EmptyData();
	T_bfnj__D_0__D_1__D_2__D_3.EmptyData();
	//****
	//**** (out of 1)
	//**** Is real	0 shadows
		//Outputs:
		//  G_mi__D_0_1__D_2_3
	PartitionDown(temp2__D_0__D_1__D_2__D_3, temp2_part3T__D_0__D_1__D_2__D_3, temp2_part3B__D_0__D_1__D_2__D_3, 3, 0);
	PartitionDown(t_fj__D_0_1__D_2_3, t_fj_part0T__D_0_1__D_2_3, t_fj_part0B__D_0_1__D_2_3, 0, 0);
	while(temp2_part3T__D_0__D_1__D_2__D_3.Dimension(3) < temp2__D_0__D_1__D_2__D_3.Dimension(3))
	{
		RepartitionDown
		( temp2_part3T__D_0__D_1__D_2__D_3,  temp2_part3_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       temp2_part3_1__D_0__D_1__D_2__D_3,
		  temp2_part3B__D_0__D_1__D_2__D_3, temp2_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );
		RepartitionDown
		( t_fj_part0T__D_0_1__D_2_3,  t_fj_part0_0__D_0_1__D_2_3,
		  /**/ /**/
		       t_fj_part0_1__D_0_1__D_2_3,
		  t_fj_part0B__D_0_1__D_2_3, t_fj_part0_2__D_0_1__D_2_3, 0, blkSize );

		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  G_mi__D_0_1__D_2_3
		PartitionDown(temp2_part3_1__D_0__D_1__D_2__D_3, temp2_part3_1_part1T__D_0__D_1__D_2__D_3, temp2_part3_1_part1B__D_0__D_1__D_2__D_3, 1, 0);
		PartitionDown(t_fj_part0_1__D_0_1__D_2_3, t_fj_part0_1_part1T__D_0_1__D_2_3, t_fj_part0_1_part1B__D_0_1__D_2_3, 1, 0);
		while(temp2_part3_1_part1T__D_0__D_1__D_2__D_3.Dimension(1) < temp2_part3_1__D_0__D_1__D_2__D_3.Dimension(1))
		{
			RepartitionDown
			( temp2_part3_1_part1T__D_0__D_1__D_2__D_3,  temp2_part3_1_part1_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       temp2_part3_1_part1_1__D_0__D_1__D_2__D_3,
			  temp2_part3_1_part1B__D_0__D_1__D_2__D_3, temp2_part3_1_part1_2__D_0__D_1__D_2__D_3, 1, blkSize );
			RepartitionDown
			( t_fj_part0_1_part1T__D_0_1__D_2_3,  t_fj_part0_1_part1_0__D_0_1__D_2_3,
			  /**/ /**/
			       t_fj_part0_1_part1_1__D_0_1__D_2_3,
			  t_fj_part0_1_part1B__D_0_1__D_2_3, t_fj_part0_1_part1_2__D_0_1__D_2_3, 1, blkSize );

			   // temp2_part3_1_part1_1[D0,D1,D2,*] <- temp2_part3_1_part1_1[D0,D1,D2,D3]
			temp2_part3_1_part1_1__D_0__D_1__D_2__S.AlignModesWith( modes_0_2, G_mi__D_0_1__D_2_3, modes_0_1 );
			temp2_part3_1_part1_1__D_0__D_1__D_2__S.AllGatherRedistFrom( temp2_part3_1_part1_1__D_0__D_1__D_2__D_3, modes_3 );
			   // temp2_part3_1_part1_1[D0,D1,D23,*] <- temp2_part3_1_part1_1[D0,D1,D2,*]
			temp2_part3_1_part1_1__D_0__D_1__D_2_3__S.AlignModesWith( modes_0_2, G_mi__D_0_1__D_2_3, modes_0_1 );
			temp2_part3_1_part1_1__D_0__D_1__D_2_3__S.LocalRedistFrom( temp2_part3_1_part1_1__D_0__D_1__D_2__S );
			temp2_part3_1_part1_1__D_0__D_1__D_2__S.EmptyData();
			   // t_fj_part0_1_part1_1[*,*] <- t_fj_part0_1_part1_1[D01,D23]
			t_fj_part0_1_part1_1_perm10__S__S.AllGatherRedistFrom( t_fj_part0_1_part1_1__D_0_1__D_2_3, modes_0_1_2_3 );
			   // temp2_part3_1_part1_1[D0,*,D23,*] <- temp2_part3_1_part1_1[D0,D1,D23,*]
			temp2_part3_1_part1_1__D_0__S__D_2_3__S.AlignModesWith( modes_0_2, G_mi__D_0_1__D_2_3, modes_0_1 );
			temp2_part3_1_part1_1__D_0__S__D_2_3__S.AllGatherRedistFrom( temp2_part3_1_part1_1__D_0__D_1__D_2_3__S, modes_1 );
			temp2_part3_1_part1_1__D_0__D_1__D_2_3__S.EmptyData();
			   // temp2_part3_1_part1_1[D01,*,D23,*] <- temp2_part3_1_part1_1[D0,*,D23,*]
			temp2_part3_1_part1_1_perm0213__D_0_1__D_2_3__S__S.AlignModesWith( modes_0_2, G_mi__D_0_1__D_2_3, modes_0_1 );
			temp2_part3_1_part1_1_perm0213__D_0_1__D_2_3__S__S.LocalRedistFrom( temp2_part3_1_part1_1__D_0__S__D_2_3__S );
			   // 1.0 * temp2_part3_1_part1_1[D01,*,D23,*]_mine * t_fj_part0_1_part1_1[*,*]_ne + 1.0 * G_mi[D01,D23]_mi
			LocalContractAndLocalEliminate(1.0, temp2_part3_1_part1_1_perm0213__D_0_1__D_2_3__S__S.LockedTensor(), indices_mine, false,
				t_fj_part0_1_part1_1_perm10__S__S.LockedTensor(), indices_ne, false,
				1.0, G_mi__D_0_1__D_2_3.Tensor(), indices_mi, false);
			t_fj_part0_1_part1_1_perm10__S__S.EmptyData();
			temp2_part3_1_part1_1_perm0213__D_0_1__D_2_3__S__S.EmptyData();
			temp2_part3_1_part1_1__D_0__S__D_2_3__S.EmptyData();

			SlidePartitionDown
			( temp2_part3_1_part1T__D_0__D_1__D_2__D_3,  temp2_part3_1_part1_0__D_0__D_1__D_2__D_3,
			       temp2_part3_1_part1_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  temp2_part3_1_part1B__D_0__D_1__D_2__D_3, temp2_part3_1_part1_2__D_0__D_1__D_2__D_3, 1 );
			SlidePartitionDown
			( t_fj_part0_1_part1T__D_0_1__D_2_3,  t_fj_part0_1_part1_0__D_0_1__D_2_3,
			       t_fj_part0_1_part1_1__D_0_1__D_2_3,
			  /**/ /**/
			  t_fj_part0_1_part1B__D_0_1__D_2_3, t_fj_part0_1_part1_2__D_0_1__D_2_3, 1 );

		}
		//****

		SlidePartitionDown
		( temp2_part3T__D_0__D_1__D_2__D_3,  temp2_part3_0__D_0__D_1__D_2__D_3,
		       temp2_part3_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  temp2_part3B__D_0__D_1__D_2__D_3, temp2_part3_2__D_0__D_1__D_2__D_3, 3 );
		SlidePartitionDown
		( t_fj_part0T__D_0_1__D_2_3,  t_fj_part0_0__D_0_1__D_2_3,
		       t_fj_part0_1__D_0_1__D_2_3,
		  /**/ /**/
		  t_fj_part0B__D_0_1__D_2_3, t_fj_part0_2__D_0_1__D_2_3, 0 );

	}
	temp2__D_0__D_1__D_2__D_3.EmptyData();
	temp2__D_0__D_1__D_2__D_3.EmptyData();
	//****
	//**** (out of 1)
	//**** Is real	0 shadows
		//Outputs:
		//  G_mi__D_0_1__D_2_3
	PartitionDown(H_me__D_0_1__D_2_3, H_me_part0T__D_0_1__D_2_3, H_me_part0B__D_0_1__D_2_3, 0, 0);
	PartitionDown(G_mi__D_0_1__D_2_3, G_mi_part0T__D_0_1__D_2_3, G_mi_part0B__D_0_1__D_2_3, 0, 0);
	while(G_mi_part0T__D_0_1__D_2_3.Dimension(0) < G_mi__D_0_1__D_2_3.Dimension(0))
	{
		RepartitionDown
		( H_me_part0T__D_0_1__D_2_3,  H_me_part0_0__D_0_1__D_2_3,
		  /**/ /**/
		       H_me_part0_1__D_0_1__D_2_3,
		  H_me_part0B__D_0_1__D_2_3, H_me_part0_2__D_0_1__D_2_3, 0, blkSize );
		RepartitionDown
		( G_mi_part0T__D_0_1__D_2_3,  G_mi_part0_0__D_0_1__D_2_3,
		  /**/ /**/
		       G_mi_part0_1__D_0_1__D_2_3,
		  G_mi_part0B__D_0_1__D_2_3, G_mi_part0_2__D_0_1__D_2_3, 0, blkSize );

		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  G_mi_part0_1__D_0_1__D_2_3
		PartitionDown(H_me_part0_1__D_0_1__D_2_3, H_me_part0_1_part1T__D_0_1__D_2_3, H_me_part0_1_part1B__D_0_1__D_2_3, 1, 0);
		PartitionDown(t_fj__D_0_1__D_2_3, t_fj_part0T__D_0_1__D_2_3, t_fj_part0B__D_0_1__D_2_3, 0, 0);
		while(H_me_part0_1_part1T__D_0_1__D_2_3.Dimension(1) < H_me_part0_1__D_0_1__D_2_3.Dimension(1))
		{
			RepartitionDown
			( H_me_part0_1_part1T__D_0_1__D_2_3,  H_me_part0_1_part1_0__D_0_1__D_2_3,
			  /**/ /**/
			       H_me_part0_1_part1_1__D_0_1__D_2_3,
			  H_me_part0_1_part1B__D_0_1__D_2_3, H_me_part0_1_part1_2__D_0_1__D_2_3, 1, blkSize );
			RepartitionDown
			( t_fj_part0T__D_0_1__D_2_3,  t_fj_part0_0__D_0_1__D_2_3,
			  /**/ /**/
			       t_fj_part0_1__D_0_1__D_2_3,
			  t_fj_part0B__D_0_1__D_2_3, t_fj_part0_2__D_0_1__D_2_3, 0, blkSize );

			   // H_me_part0_1_part1_1[D01,*] <- H_me_part0_1_part1_1[D01,D23]
			H_me_part0_1_part1_1__D_0_1__S.AlignModesWith( modes_0, G_mi_part0_1__D_0_1__D_2_3, modes_0 );
			H_me_part0_1_part1_1__D_0_1__S.AllGatherRedistFrom( H_me_part0_1_part1_1__D_0_1__D_2_3, modes_2_3 );
			   // t_fj_part0_1[*,D23] <- t_fj_part0_1[D01,D23]
			t_fj_part0_1__S__D_2_3.AlignModesWith( modes_1, G_mi_part0_1__D_0_1__D_2_3, modes_1 );
			t_fj_part0_1__S__D_2_3.AllGatherRedistFrom( t_fj_part0_1__D_0_1__D_2_3, modes_0_1 );
			   // 1.0 * H_me_part0_1_part1_1[D01,*]_me * t_fj_part0_1[*,D23]_ei + 1.0 * G_mi_part0_1[D01,D23]_mi
			LocalContractAndLocalEliminate(1.0, H_me_part0_1_part1_1__D_0_1__S.LockedTensor(), indices_me, false,
				t_fj_part0_1__S__D_2_3.LockedTensor(), indices_ei, false,
				1.0, G_mi_part0_1__D_0_1__D_2_3.Tensor(), indices_mi, false);
			H_me_part0_1_part1_1__D_0_1__S.EmptyData();
			t_fj_part0_1__S__D_2_3.EmptyData();

			SlidePartitionDown
			( H_me_part0_1_part1T__D_0_1__D_2_3,  H_me_part0_1_part1_0__D_0_1__D_2_3,
			       H_me_part0_1_part1_1__D_0_1__D_2_3,
			  /**/ /**/
			  H_me_part0_1_part1B__D_0_1__D_2_3, H_me_part0_1_part1_2__D_0_1__D_2_3, 1 );
			SlidePartitionDown
			( t_fj_part0T__D_0_1__D_2_3,  t_fj_part0_0__D_0_1__D_2_3,
			       t_fj_part0_1__D_0_1__D_2_3,
			  /**/ /**/
			  t_fj_part0B__D_0_1__D_2_3, t_fj_part0_2__D_0_1__D_2_3, 0 );

		}
		//****

		SlidePartitionDown
		( H_me_part0T__D_0_1__D_2_3,  H_me_part0_0__D_0_1__D_2_3,
		       H_me_part0_1__D_0_1__D_2_3,
		  /**/ /**/
		  H_me_part0B__D_0_1__D_2_3, H_me_part0_2__D_0_1__D_2_3, 0 );
		SlidePartitionDown
		( G_mi_part0T__D_0_1__D_2_3,  G_mi_part0_0__D_0_1__D_2_3,
		       G_mi_part0_1__D_0_1__D_2_3,
		  /**/ /**/
		  G_mi_part0B__D_0_1__D_2_3, G_mi_part0_2__D_0_1__D_2_3, 0 );

	}
	H_me__D_0_1__D_2_3.EmptyData();
	t_fj__D_0_1__D_2_3.EmptyData();
	H_me__D_0_1__D_2_3.EmptyData();
	t_fj__D_0_1__D_2_3.EmptyData();
	//****


H_me__D_0_1__D_2_3.EmptyData();
t_fj__D_0_1__D_2_3.EmptyData();
u_mnje__D_0__D_1__D_2__D_3.EmptyData();
temp2__D_0__D_1__D_2__D_3.EmptyData();
T_bfnj__D_0__D_1__D_2__D_3.EmptyData();
v_femn__D_0__D_1__D_2__D_3.EmptyData();
temp1__D_0__D_1__D_2__D_3.EmptyData();
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
    DistTensor<double> diff(dist__D_0_1__D_2_3, g);
    diff.ResizeTo(check);
    Diff(check, G_mi__D_0_1__D_2_3, diff);
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


