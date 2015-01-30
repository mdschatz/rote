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
TensorDistribution dist__S__D_2_3 = tmen::StringToTensorDist("[(),(2,3)]");
TensorDistribution dist__S__D_1__D_2__D_3 = tmen::StringToTensorDist("[(),(1),(2),(3)]");
TensorDistribution dist__D_0__S__D_1__D_2__D_3 = tmen::StringToTensorDist("[(0),(),(1),(2),(3)]");
TensorDistribution dist__D_0__D_1__D_2__D_3 = tmen::StringToTensorDist("[(0),(1),(2),(3)]");
TensorDistribution dist__D_0__D_1__D_3__D_2 = tmen::StringToTensorDist("[(0),(1),(3),(2)]");
TensorDistribution dist__D_0__D_2__D_3__D_1 = tmen::StringToTensorDist("[(0),(2),(3),(1)]");
TensorDistribution dist__D_3_0__D_1_2 = tmen::StringToTensorDist("[(3,0),(1,2)]");
TensorDistribution dist__D_3_0__D_2_1 = tmen::StringToTensorDist("[(3,0),(2,1)]");
TensorDistribution dist__D_3__D_1 = tmen::StringToTensorDist("[(3),(1)]");
TensorDistribution dist__D_0_1__S = tmen::StringToTensorDist("[(0,1),()]");
TensorDistribution dist__D_0_1__D_2_3 = tmen::StringToTensorDist("[(0,1),(2,3)]");
TensorDistribution dist__D_0_3__D_2_1 = tmen::StringToTensorDist("[(0,3),(2,1)]");
Permutation perm_0_1( 2 );
perm_0_1[0] = 0;
perm_0_1[1] = 1;
Permutation perm_0_1_2_3( 4 );
perm_0_1_2_3[0] = 0;
perm_0_1_2_3[1] = 1;
perm_0_1_2_3[2] = 2;
perm_0_1_2_3[3] = 3;
Permutation perm_0_1_3_2( 4 );
perm_0_1_3_2[0] = 0;
perm_0_1_3_2[1] = 1;
perm_0_1_3_2[2] = 3;
perm_0_1_3_2[3] = 2;
Permutation perm_0_2_3_1( 4 );
perm_0_2_3_1[0] = 0;
perm_0_2_3_1[1] = 2;
perm_0_2_3_1[2] = 3;
perm_0_2_3_1[3] = 1;
Permutation perm_1_0( 2 );
perm_1_0[0] = 1;
perm_1_0[1] = 0;
Permutation perm_1_0_2_3_4( 5 );
perm_1_0_2_3_4[0] = 1;
perm_1_0_2_3_4[1] = 0;
perm_1_0_2_3_4[2] = 2;
perm_1_0_2_3_4[3] = 3;
perm_1_0_2_3_4[4] = 4;
Permutation perm_1_2_3_0( 4 );
perm_1_2_3_0[0] = 1;
perm_1_2_3_0[1] = 2;
perm_1_2_3_0[2] = 3;
perm_1_2_3_0[3] = 0;
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
ModeArray modes_1_2_3( 3 );
modes_1_2_3[0] = 1;
modes_1_2_3[1] = 2;
modes_1_2_3[2] = 3;
ModeArray modes_1_3( 2 );
modes_1_3[0] = 1;
modes_1_3[1] = 3;
ModeArray modes_2_1( 2 );
modes_2_1[0] = 2;
modes_2_1[1] = 1;
ModeArray modes_2_3( 2 );
modes_2_3[0] = 2;
modes_2_3[1] = 3;
ModeArray modes_3_1( 2 );
modes_3_1[0] = 3;
modes_3_1[1] = 1;
ModeArray modes_3_2( 2 );
modes_3_2[0] = 3;
modes_3_2[1] = 2;
ModeArray modes_4_3_2( 3 );
modes_4_3_2[0] = 4;
modes_4_3_2[1] = 3;
modes_4_3_2[2] = 2;
IndexArray indices_aefm( 4 );
indices_aefm[0] = 'a';
indices_aefm[1] = 'e';
indices_aefm[2] = 'f';
indices_aefm[3] = 'm';
IndexArray indices_ea( 2 );
indices_ea[0] = 'e';
indices_ea[1] = 'a';
IndexArray indices_eafmn( 5 );
indices_eafmn[0] = 'e';
indices_eafmn[1] = 'a';
indices_eafmn[2] = 'f';
indices_eafmn[3] = 'm';
indices_eafmn[4] = 'n';
IndexArray indices_efmn( 4 );
indices_efmn[0] = 'e';
indices_efmn[1] = 'f';
indices_efmn[2] = 'm';
indices_efmn[3] = 'n';
IndexArray indices_em( 2 );
indices_em[0] = 'e';
indices_em[1] = 'm';
IndexArray indices_fm( 2 );
indices_fm[0] = 'f';
indices_fm[1] = 'm';
IndexArray indices_fmna( 4 );
indices_fmna[0] = 'f';
indices_fmna[1] = 'm';
indices_fmna[2] = 'n';
indices_fmna[3] = 'a';
IndexArray indices_ma( 2 );
indices_ma[0] = 'm';
indices_ma[1] = 'a';
	//F_ae[D01,D23]
DistTensor<double> F_ae__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//F_ae_lvl0_part1B[D01,D23]
DistTensor<double> F_ae_lvl0_part1B__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//F_ae_lvl0_part1T[D01,D23]
DistTensor<double> F_ae_lvl0_part1T__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//F_ae_lvl1_part1B[D01,D23]
DistTensor<double> F_ae_lvl1_part1B__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//F_ae_lvl1_part1T[D01,D23]
DistTensor<double> F_ae_lvl1_part1T__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//F_ae_lvl1_part1_0[D01,D23]
DistTensor<double> F_ae_lvl1_part1_0__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//F_ae_lvl1_part1_1[D01,D23]
DistTensor<double> F_ae_lvl1_part1_1__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//F_ae_lvl1_part1_1[D0,D2,D3,D1]
DistTensor<double> F_ae_lvl1_part1_1__D_0__D_2__D_3__D_1( dist__D_0__D_2__D_3__D_1, g );
DistTensor<double> F_ae_lvl1_part1_1_perm10__D_2_3__D_0_1( dist__D_0_1__D_2_3, g );
F_ae_lvl1_part1_1_perm10__D_2_3__D_0_1.SetLocalPermutation( perm_1_0 );
	//F_ae_lvl1_part1_2[D01,D23]
DistTensor<double> F_ae_lvl1_part1_2__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//F_ae[D0,*,D1,D2,D3]
DistTensor<double> F_ae_perm10234__S__D_0__D_1__D_2__D_3( dist__D_0__S__D_1__D_2__D_3, g );
F_ae_perm10234__S__D_0__D_1__D_2__D_3.SetLocalPermutation( perm_1_0_2_3_4 );
	//Ftemp1[D0,D1,D2,D3]
DistTensor<double> Ftemp1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Ftemp1_lvl0_part2B[D0,D1,D2,D3]
DistTensor<double> Ftemp1_lvl0_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Ftemp1_lvl0_part2T[D0,D1,D2,D3]
DistTensor<double> Ftemp1_lvl0_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Ftemp1_lvl1_part2B[D0,D1,D2,D3]
DistTensor<double> Ftemp1_lvl1_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Ftemp1_lvl1_part2T[D0,D1,D2,D3]
DistTensor<double> Ftemp1_lvl1_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Ftemp1_lvl1_part2_0[D0,D1,D2,D3]
DistTensor<double> Ftemp1_lvl1_part2_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Ftemp1_lvl1_part2_1[D0,D1,D2,D3]
DistTensor<double> Ftemp1_lvl1_part2_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Ftemp1_lvl1_part2_1_lvl1_part3B[D0,D1,D2,D3]
DistTensor<double> Ftemp1_lvl1_part2_1_lvl1_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Ftemp1_lvl1_part2_1_lvl1_part3T[D0,D1,D2,D3]
DistTensor<double> Ftemp1_lvl1_part2_1_lvl1_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Ftemp1_lvl1_part2_1_lvl2_part3B[D0,D1,D2,D3]
DistTensor<double> Ftemp1_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Ftemp1_lvl1_part2_1_lvl2_part3T[D0,D1,D2,D3]
DistTensor<double> Ftemp1_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Ftemp1_lvl1_part2_1_lvl2_part3_0[D0,D1,D2,D3]
DistTensor<double> Ftemp1_lvl1_part2_1_lvl2_part3_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Ftemp1_lvl1_part2_1_lvl2_part3_1[D0,D1,D2,D3]
DistTensor<double> Ftemp1_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Ftemp1_lvl1_part2_1_lvl2_part3_1[*,D1,D2,D3]
DistTensor<double> Ftemp1_lvl1_part2_1_lvl2_part3_1__S__D_1__D_2__D_3( dist__S__D_1__D_2__D_3, g );
	//Ftemp1_lvl1_part2_1_lvl2_part3_2[D0,D1,D2,D3]
DistTensor<double> Ftemp1_lvl1_part2_1_lvl2_part3_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Ftemp1_lvl1_part2_2[D0,D1,D2,D3]
DistTensor<double> Ftemp1_lvl1_part2_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Ftemp2[D0,D1,D2,D3]
DistTensor<double> Ftemp2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Ftemp2_lvl0_part0B[D0,D1,D2,D3]
DistTensor<double> Ftemp2_lvl0_part0B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Ftemp2_lvl0_part0T[D0,D1,D2,D3]
DistTensor<double> Ftemp2_lvl0_part0T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Ftemp2_lvl0_part2B[D0,D1,D2,D3]
DistTensor<double> Ftemp2_lvl0_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Ftemp2_lvl0_part2T[D0,D1,D2,D3]
DistTensor<double> Ftemp2_lvl0_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Ftemp2_lvl1_part0B[D0,D1,D2,D3]
DistTensor<double> Ftemp2_lvl1_part0B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Ftemp2_lvl1_part0T[D0,D1,D2,D3]
DistTensor<double> Ftemp2_lvl1_part0T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Ftemp2_lvl1_part0_0[D0,D1,D2,D3]
DistTensor<double> Ftemp2_lvl1_part0_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Ftemp2_lvl1_part0_1[D0,D1,D2,D3]
DistTensor<double> Ftemp2_lvl1_part0_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Ftemp2_lvl1_part0_1_lvl1_part1B[D0,D1,D2,D3]
DistTensor<double> Ftemp2_lvl1_part0_1_lvl1_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Ftemp2_lvl1_part0_1_lvl1_part1T[D0,D1,D2,D3]
DistTensor<double> Ftemp2_lvl1_part0_1_lvl1_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Ftemp2_lvl1_part0_1_lvl2_part1B[D0,D1,D2,D3]
DistTensor<double> Ftemp2_lvl1_part0_1_lvl2_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Ftemp2_lvl1_part0_1_lvl2_part1T[D0,D1,D2,D3]
DistTensor<double> Ftemp2_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Ftemp2_lvl1_part0_1_lvl2_part1_0[D0,D1,D2,D3]
DistTensor<double> Ftemp2_lvl1_part0_1_lvl2_part1_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Ftemp2_lvl1_part0_1_lvl2_part1_1[D0,D1,D2,D3]
DistTensor<double> Ftemp2_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Ftemp2_lvl1_part0_1_lvl2_part1_2[D0,D1,D2,D3]
DistTensor<double> Ftemp2_lvl1_part0_1_lvl2_part1_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Ftemp2_lvl1_part0_2[D0,D1,D2,D3]
DistTensor<double> Ftemp2_lvl1_part0_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Ftemp2_lvl1_part2B[D0,D1,D2,D3]
DistTensor<double> Ftemp2_lvl1_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Ftemp2_lvl1_part2T[D0,D1,D2,D3]
DistTensor<double> Ftemp2_lvl1_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Ftemp2_lvl1_part2_0[D0,D1,D2,D3]
DistTensor<double> Ftemp2_lvl1_part2_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Ftemp2_lvl1_part2_1[D0,D1,D2,D3]
DistTensor<double> Ftemp2_lvl1_part2_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Ftemp2_lvl1_part2_1_lvl1_part1B[D0,D1,D2,D3]
DistTensor<double> Ftemp2_lvl1_part2_1_lvl1_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Ftemp2_lvl1_part2_1_lvl1_part1T[D0,D1,D2,D3]
DistTensor<double> Ftemp2_lvl1_part2_1_lvl1_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Ftemp2_lvl1_part2_1_lvl2_part1B[D0,D1,D2,D3]
DistTensor<double> Ftemp2_lvl1_part2_1_lvl2_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Ftemp2_lvl1_part2_1_lvl2_part1T[D0,D1,D2,D3]
DistTensor<double> Ftemp2_lvl1_part2_1_lvl2_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Ftemp2_lvl1_part2_1_lvl2_part1_0[D0,D1,D2,D3]
DistTensor<double> Ftemp2_lvl1_part2_1_lvl2_part1_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Ftemp2_lvl1_part2_1_lvl2_part1_1[D0,D1,D2,D3]
DistTensor<double> Ftemp2_lvl1_part2_1_lvl2_part1_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> Ftemp2_lvl1_part2_1_lvl2_part1_1_perm0231__D_0__D_2__D_3__D_1( dist__D_0__D_1__D_2__D_3, g );
Ftemp2_lvl1_part2_1_lvl2_part1_1_perm0231__D_0__D_2__D_3__D_1.SetLocalPermutation( perm_0_2_3_1 );
	//Ftemp2_lvl1_part2_1_lvl2_part1_2[D0,D1,D2,D3]
DistTensor<double> Ftemp2_lvl1_part2_1_lvl2_part1_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Ftemp2_lvl1_part2_2[D0,D1,D2,D3]
DistTensor<double> Ftemp2_lvl1_part2_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//H_me[D01,D23]
DistTensor<double> H_me__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//H_me_lvl0_part1B[D01,D23]
DistTensor<double> H_me_lvl0_part1B__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//H_me_lvl0_part1T[D01,D23]
DistTensor<double> H_me_lvl0_part1T__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//H_me_lvl1_part1B[D01,D23]
DistTensor<double> H_me_lvl1_part1B__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//H_me_lvl1_part1T[D01,D23]
DistTensor<double> H_me_lvl1_part1T__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//H_me_lvl1_part1_0[D01,D23]
DistTensor<double> H_me_lvl1_part1_0__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//H_me_lvl1_part1_1[D01,D23]
DistTensor<double> H_me_lvl1_part1_1__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//H_me_lvl1_part1_1_lvl1_part0B[D01,D23]
DistTensor<double> H_me_lvl1_part1_1_lvl1_part0B__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//H_me_lvl1_part1_1_lvl1_part0T[D01,D23]
DistTensor<double> H_me_lvl1_part1_1_lvl1_part0T__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//H_me_lvl1_part1_1_lvl2_part0B[D01,D23]
DistTensor<double> H_me_lvl1_part1_1_lvl2_part0B__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//H_me_lvl1_part1_1_lvl2_part0T[D01,D23]
DistTensor<double> H_me_lvl1_part1_1_lvl2_part0T__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//H_me_lvl1_part1_1_lvl2_part0_0[D01,D23]
DistTensor<double> H_me_lvl1_part1_1_lvl2_part0_0__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//H_me_lvl1_part1_1_lvl2_part0_1[D01,D23]
DistTensor<double> H_me_lvl1_part1_1_lvl2_part0_1__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//H_me_lvl1_part1_1_lvl2_part0_1[*,D23]
DistTensor<double> H_me_lvl1_part1_1_lvl2_part0_1_perm10__D_2_3__S( dist__S__D_2_3, g );
H_me_lvl1_part1_1_lvl2_part0_1_perm10__D_2_3__S.SetLocalPermutation( perm_1_0 );
	//H_me_lvl1_part1_1_lvl2_part0_2[D01,D23]
DistTensor<double> H_me_lvl1_part1_1_lvl2_part0_2__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//H_me_lvl1_part1_2[D01,D23]
DistTensor<double> H_me_lvl1_part1_2__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//T_bfnj[D0,D1,D2,D3]
DistTensor<double> T_bfnj__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//T_bfnj_lvl0_part2B[D0,D1,D2,D3]
DistTensor<double> T_bfnj_lvl0_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//T_bfnj_lvl0_part2T[D0,D1,D2,D3]
DistTensor<double> T_bfnj_lvl0_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
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
DistTensor<double> T_bfnj_lvl1_part2_1_lvl2_part3_1_perm1230__D_1__D_2__D_3__D_0( dist__D_0__D_1__D_2__D_3, g );
T_bfnj_lvl1_part2_1_lvl2_part3_1_perm1230__D_1__D_2__D_3__D_0.SetLocalPermutation( perm_1_2_3_0 );
	//T_bfnj_lvl1_part2_1_lvl2_part3_2[D0,D1,D2,D3]
DistTensor<double> T_bfnj_lvl1_part2_1_lvl2_part3_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//T_bfnj_lvl1_part2_2[D0,D1,D2,D3]
DistTensor<double> T_bfnj_lvl1_part2_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
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
	//t_fj_lvl2_part1_1[D03,D21]
DistTensor<double> t_fj_lvl2_part1_1__D_0_3__D_2_1( dist__D_0_3__D_2_1, g );
	//t_fj_lvl2_part1_1[D30,D12]
DistTensor<double> t_fj_lvl2_part1_1__D_3_0__D_1_2( dist__D_3_0__D_1_2, g );
	//t_fj_lvl2_part1_1[D30,D21]
DistTensor<double> t_fj_lvl2_part1_1__D_3_0__D_2_1( dist__D_3_0__D_2_1, g );
	//t_fj_lvl2_part1_1[D3,D1]
DistTensor<double> t_fj_lvl2_part1_1__D_3__D_1( dist__D_3__D_1, g );
	//t_fj_lvl2_part1_1[D01,*]
DistTensor<double> t_fj_lvl2_part1_1_perm10__S__D_0_1( dist__D_0_1__S, g );
t_fj_lvl2_part1_1_perm10__S__D_0_1.SetLocalPermutation( perm_1_0 );
	//t_fj_lvl2_part1_2[D01,D23]
DistTensor<double> t_fj_lvl2_part1_2__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
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
ObjShape H_me__D_0_1__D_2_3_tempShape( 2 );
H_me__D_0_1__D_2_3_tempShape[ 0 ] = n_o;
H_me__D_0_1__D_2_3_tempShape[ 1 ] = n_v;
H_me__D_0_1__D_2_3.ResizeTo( H_me__D_0_1__D_2_3_tempShape );
MakeUniform( H_me__D_0_1__D_2_3 );
// t_fj has 2 dims
ObjShape t_fj__D_0_1__D_2_3_tempShape( 2 );
t_fj__D_0_1__D_2_3_tempShape[ 0 ] = n_v;
t_fj__D_0_1__D_2_3_tempShape[ 1 ] = n_o;
t_fj__D_0_1__D_2_3.ResizeTo( t_fj__D_0_1__D_2_3_tempShape );
MakeUniform( t_fj__D_0_1__D_2_3 );
// r_bmfe has 4 dims
//	Starting distribution: [D0,D1,D2,D3] or _D_0__D_1__D_2__D_3
ObjShape r_bmfe__D_0__D_1__D_2__D_3_tempShape( 4 );
r_bmfe__D_0__D_1__D_2__D_3_tempShape[ 0 ] = n_v;
r_bmfe__D_0__D_1__D_2__D_3_tempShape[ 1 ] = n_o;
r_bmfe__D_0__D_1__D_2__D_3_tempShape[ 2 ] = n_v;
r_bmfe__D_0__D_1__D_2__D_3_tempShape[ 3 ] = n_v;
r_bmfe__D_0__D_1__D_2__D_3.ResizeTo( r_bmfe__D_0__D_1__D_2__D_3_tempShape );
MakeUniform( r_bmfe__D_0__D_1__D_2__D_3 );
// T_bfnj has 4 dims
ObjShape T_bfnj__D_0__D_1__D_2__D_3_tempShape( 4 );
T_bfnj__D_0__D_1__D_2__D_3_tempShape[ 0 ] = n_v;
T_bfnj__D_0__D_1__D_2__D_3_tempShape[ 1 ] = n_v;
T_bfnj__D_0__D_1__D_2__D_3_tempShape[ 2 ] = n_o;
T_bfnj__D_0__D_1__D_2__D_3_tempShape[ 3 ] = n_o;
T_bfnj__D_0__D_1__D_2__D_3.ResizeTo( T_bfnj__D_0__D_1__D_2__D_3_tempShape );
MakeUniform( T_bfnj__D_0__D_1__D_2__D_3 );
// F_ae has 2 dims
ObjShape F_ae__D_0_1__D_2_3_tempShape( 2 );
F_ae__D_0_1__D_2_3_tempShape[ 0 ] = n_v;
F_ae__D_0_1__D_2_3_tempShape[ 1 ] = n_v;
F_ae__D_0_1__D_2_3.ResizeTo( F_ae__D_0_1__D_2_3_tempShape );
MakeUniform( F_ae__D_0_1__D_2_3 );
// v_femn has 4 dims
ObjShape v_femn__D_0__D_1__D_2__D_3_tempShape( 4 );
v_femn__D_0__D_1__D_2__D_3_tempShape[ 0 ] = n_v;
v_femn__D_0__D_1__D_2__D_3_tempShape[ 1 ] = n_v;
v_femn__D_0__D_1__D_2__D_3_tempShape[ 2 ] = n_o;
v_femn__D_0__D_1__D_2__D_3_tempShape[ 3 ] = n_o;
v_femn__D_0__D_1__D_2__D_3.ResizeTo( v_femn__D_0__D_1__D_2__D_3_tempShape );
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
DistTensor<T> check_F(dist__D_0_1__D_2_3, g);
check_F.ResizeTo(F_ae__D_0_1__D_2_3.Shape());
Read(r_bmfe__D_0__D_1__D_2__D_3, "ccsd_terms/term_r_small", BINARY_FLAT, false);
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
fullName << "ccsd_terms/term_F_iter" << testIter;
Read(check_F, fullName.str(), BINARY_FLAT, false);
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
	//  F_ae__D_0_1__D_2_3

if(commRank == 0){
flops += prod(F_ae__D_0_1__D_2_3.Shape());
}
	Scal( 0.0, F_ae__D_0_1__D_2_3 );
	tempShape = r_bmfe__D_0__D_1__D_2__D_3.Shape();
	Ftemp2__D_0__D_1__D_2__D_3.ResizeTo( tempShape );
	//**** (out of 1)
	//**** Is real	0 shadows
		//Outputs:
		//  Ftemp2__D_0__D_1__D_2__D_3
	PartitionDown(r_bmfe__D_0__D_1__D_2__D_3, r_bmfe_lvl1_part0T__D_0__D_1__D_2__D_3, r_bmfe_lvl1_part0B__D_0__D_1__D_2__D_3, 0, 0);
	PartitionDown(Ftemp2__D_0__D_1__D_2__D_3, Ftemp2_lvl1_part0T__D_0__D_1__D_2__D_3, Ftemp2_lvl1_part0B__D_0__D_1__D_2__D_3, 0, 0);
	while(Ftemp2_lvl1_part0T__D_0__D_1__D_2__D_3.Dimension(0) < Ftemp2__D_0__D_1__D_2__D_3.Dimension(0))
	{
		RepartitionDown
		( r_bmfe_lvl1_part0T__D_0__D_1__D_2__D_3,  r_bmfe_lvl1_part0_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       r_bmfe_lvl1_part0_1__D_0__D_1__D_2__D_3,
		  r_bmfe_lvl1_part0B__D_0__D_1__D_2__D_3, r_bmfe_lvl1_part0_2__D_0__D_1__D_2__D_3, 0, blkSize );
		RepartitionDown
		( Ftemp2_lvl1_part0T__D_0__D_1__D_2__D_3,  Ftemp2_lvl1_part0_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       Ftemp2_lvl1_part0_1__D_0__D_1__D_2__D_3,
		  Ftemp2_lvl1_part0B__D_0__D_1__D_2__D_3, Ftemp2_lvl1_part0_2__D_0__D_1__D_2__D_3, 0, blkSize );

		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  Ftemp2_lvl1_part0_1__D_0__D_1__D_2__D_3
		PartitionDown(r_bmfe_lvl1_part0_1__D_0__D_1__D_2__D_3, r_bmfe_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3, r_bmfe_lvl1_part0_1_lvl2_part1B__D_0__D_1__D_2__D_3, 1, 0);
		PartitionDown(Ftemp2_lvl1_part0_1__D_0__D_1__D_2__D_3, Ftemp2_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3, Ftemp2_lvl1_part0_1_lvl2_part1B__D_0__D_1__D_2__D_3, 1, 0);
		while(Ftemp2_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3.Dimension(1) < Ftemp2_lvl1_part0_1__D_0__D_1__D_2__D_3.Dimension(1))
		{
			RepartitionDown
			( r_bmfe_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3,  r_bmfe_lvl1_part0_1_lvl2_part1_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       r_bmfe_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3,
			  r_bmfe_lvl1_part0_1_lvl2_part1B__D_0__D_1__D_2__D_3, r_bmfe_lvl1_part0_1_lvl2_part1_2__D_0__D_1__D_2__D_3, 1, blkSize );
			RepartitionDown
			( Ftemp2_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3,  Ftemp2_lvl1_part0_1_lvl2_part1_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       Ftemp2_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3,
			  Ftemp2_lvl1_part0_1_lvl2_part1B__D_0__D_1__D_2__D_3, Ftemp2_lvl1_part0_1_lvl2_part1_2__D_0__D_1__D_2__D_3, 1, blkSize );

			   // r_bmfe_lvl1_part0_1_lvl2_part1_1[D0,D1,D3,D2] <- r_bmfe_lvl1_part0_1_lvl2_part1_1[D0,D1,D2,D3]
			r_bmfe_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_3__D_2.AlignModesWith( modes_0_1_2_3, r_bmfe_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3, modes_0_1_3_2 );
			r_bmfe_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_3__D_2.AllToAllRedistFrom( r_bmfe_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3, modes_2_3 );
if(commRank == 0){
flops += 3*prod(r_bmfe_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3.Shape());
}
			YAxpPx( 2.0, r_bmfe_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3, -1.0, r_bmfe_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_3__D_2, perm_0_1_3_2, Ftemp2_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3 );
			r_bmfe_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_3__D_2.EmptyData();

			SlidePartitionDown
			( r_bmfe_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3,  r_bmfe_lvl1_part0_1_lvl2_part1_0__D_0__D_1__D_2__D_3,
			       r_bmfe_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  r_bmfe_lvl1_part0_1_lvl2_part1B__D_0__D_1__D_2__D_3, r_bmfe_lvl1_part0_1_lvl2_part1_2__D_0__D_1__D_2__D_3, 1 );
			SlidePartitionDown
			( Ftemp2_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3,  Ftemp2_lvl1_part0_1_lvl2_part1_0__D_0__D_1__D_2__D_3,
			       Ftemp2_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  Ftemp2_lvl1_part0_1_lvl2_part1B__D_0__D_1__D_2__D_3, Ftemp2_lvl1_part0_1_lvl2_part1_2__D_0__D_1__D_2__D_3, 1 );

		}
		//****

		SlidePartitionDown
		( r_bmfe_lvl1_part0T__D_0__D_1__D_2__D_3,  r_bmfe_lvl1_part0_0__D_0__D_1__D_2__D_3,
		       r_bmfe_lvl1_part0_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  r_bmfe_lvl1_part0B__D_0__D_1__D_2__D_3, r_bmfe_lvl1_part0_2__D_0__D_1__D_2__D_3, 0 );
		SlidePartitionDown
		( Ftemp2_lvl1_part0T__D_0__D_1__D_2__D_3,  Ftemp2_lvl1_part0_0__D_0__D_1__D_2__D_3,
		       Ftemp2_lvl1_part0_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  Ftemp2_lvl1_part0B__D_0__D_1__D_2__D_3, Ftemp2_lvl1_part0_2__D_0__D_1__D_2__D_3, 0 );

	}
	//****
	tempShape = v_femn__D_0__D_1__D_2__D_3.Shape();
	Ftemp1__D_0__D_1__D_2__D_3.ResizeTo( tempShape );
	//**** (out of 1)
	//**** Is real	0 shadows
		//Outputs:
		//  Ftemp1__D_0__D_1__D_2__D_3
	PartitionDown(v_femn__D_0__D_1__D_2__D_3, v_femn_lvl1_part2T__D_0__D_1__D_2__D_3, v_femn_lvl1_part2B__D_0__D_1__D_2__D_3, 2, 0);
	PartitionDown(v_femn__D_0__D_1__D_2__D_3, v_femn_lvl1_part3T__D_0__D_1__D_2__D_3, v_femn_lvl1_part3B__D_0__D_1__D_2__D_3, 3, 0);
	PartitionDown(Ftemp1__D_0__D_1__D_2__D_3, Ftemp1_lvl1_part2T__D_0__D_1__D_2__D_3, Ftemp1_lvl1_part2B__D_0__D_1__D_2__D_3, 2, 0);
	while(Ftemp1_lvl1_part2T__D_0__D_1__D_2__D_3.Dimension(2) < Ftemp1__D_0__D_1__D_2__D_3.Dimension(2))
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
		( Ftemp1_lvl1_part2T__D_0__D_1__D_2__D_3,  Ftemp1_lvl1_part2_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       Ftemp1_lvl1_part2_1__D_0__D_1__D_2__D_3,
		  Ftemp1_lvl1_part2B__D_0__D_1__D_2__D_3, Ftemp1_lvl1_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );

		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  Ftemp1_lvl1_part2_1__D_0__D_1__D_2__D_3
		PartitionDown(v_femn_lvl1_part2_1__D_0__D_1__D_2__D_3, v_femn_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3, v_femn_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, 3, 0);
		PartitionDown(v_femn_lvl1_part3_1__D_0__D_1__D_2__D_3, v_femn_lvl1_part3_1_lvl2_part2T__D_0__D_1__D_2__D_3, v_femn_lvl1_part3_1_lvl2_part2B__D_0__D_1__D_2__D_3, 2, 0);
		PartitionDown(Ftemp1_lvl1_part2_1__D_0__D_1__D_2__D_3, Ftemp1_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3, Ftemp1_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, 3, 0);
		while(Ftemp1_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3.Dimension(3) < Ftemp1_lvl1_part2_1__D_0__D_1__D_2__D_3.Dimension(3))
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
			( Ftemp1_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3,  Ftemp1_lvl1_part2_1_lvl2_part3_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       Ftemp1_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3,
			  Ftemp1_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, Ftemp1_lvl1_part2_1_lvl2_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );

			   // v_femn_lvl1_part3_1_lvl2_part2_1[D0,D1,D3,D2] <- v_femn_lvl1_part3_1_lvl2_part2_1[D0,D1,D2,D3]
			v_femn_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_3__D_2.AlignModesWith( modes_0_1_2_3, v_femn_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3, modes_0_1_3_2 );
			v_femn_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_3__D_2.AllToAllRedistFrom( v_femn_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_2__D_3, modes_2_3 );
if(commRank == 0){
flops += 3*prod(v_femn_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3.Shape());
}
			YAxpPx( 2.0, v_femn_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3, -1.0, v_femn_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_3__D_2, perm_0_1_3_2, Ftemp1_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3 );
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
			( Ftemp1_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3,  Ftemp1_lvl1_part2_1_lvl2_part3_0__D_0__D_1__D_2__D_3,
			       Ftemp1_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  Ftemp1_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, Ftemp1_lvl1_part2_1_lvl2_part3_2__D_0__D_1__D_2__D_3, 3 );

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
		( Ftemp1_lvl1_part2T__D_0__D_1__D_2__D_3,  Ftemp1_lvl1_part2_0__D_0__D_1__D_2__D_3,
		       Ftemp1_lvl1_part2_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  Ftemp1_lvl1_part2B__D_0__D_1__D_2__D_3, Ftemp1_lvl1_part2_2__D_0__D_1__D_2__D_3, 2 );

	}
	//****
	//**** (out of 1)
	//**** Is real	0 shadows
		//Outputs:
		//  F_ae__D_0_1__D_2_3
	PartitionDown(Ftemp1__D_0__D_1__D_2__D_3, Ftemp1_lvl1_part2T__D_0__D_1__D_2__D_3, Ftemp1_lvl1_part2B__D_0__D_1__D_2__D_3, 2, 0);
	PartitionDown(T_bfnj__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part2T__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part2B__D_0__D_1__D_2__D_3, 2, 0);
	while(Ftemp1_lvl1_part2T__D_0__D_1__D_2__D_3.Dimension(2) < Ftemp1__D_0__D_1__D_2__D_3.Dimension(2))
	{
		RepartitionDown
		( Ftemp1_lvl1_part2T__D_0__D_1__D_2__D_3,  Ftemp1_lvl1_part2_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       Ftemp1_lvl1_part2_1__D_0__D_1__D_2__D_3,
		  Ftemp1_lvl1_part2B__D_0__D_1__D_2__D_3, Ftemp1_lvl1_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );
		RepartitionDown
		( T_bfnj_lvl1_part2T__D_0__D_1__D_2__D_3,  T_bfnj_lvl1_part2_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       T_bfnj_lvl1_part2_1__D_0__D_1__D_2__D_3,
		  T_bfnj_lvl1_part2B__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );

		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  F_ae__D_0_1__D_2_3
		PartitionDown(Ftemp1_lvl1_part2_1__D_0__D_1__D_2__D_3, Ftemp1_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3, Ftemp1_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, 3, 0);
		PartitionDown(T_bfnj_lvl1_part2_1__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, 3, 0);
		while(Ftemp1_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3.Dimension(3) < Ftemp1_lvl1_part2_1__D_0__D_1__D_2__D_3.Dimension(3))
		{
			RepartitionDown
			( Ftemp1_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3,  Ftemp1_lvl1_part2_1_lvl2_part3_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       Ftemp1_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3,
			  Ftemp1_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, Ftemp1_lvl1_part2_1_lvl2_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );
			RepartitionDown
			( T_bfnj_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3,  T_bfnj_lvl1_part2_1_lvl2_part3_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       T_bfnj_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3,
			  T_bfnj_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part2_1_lvl2_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );

			   // Ftemp1_lvl1_part2_1_lvl2_part3_1[*,D1,D2,D3] <- Ftemp1_lvl1_part2_1_lvl2_part3_1[D0,D1,D2,D3]
			Ftemp1_lvl1_part2_1_lvl2_part3_1__S__D_1__D_2__D_3.AlignModesWith( modes_1_2_3, T_bfnj_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3, modes_1_2_3 );
			Ftemp1_lvl1_part2_1_lvl2_part3_1__S__D_1__D_2__D_3.AllGatherRedistFrom( Ftemp1_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3, modes_0 );
			Permute( T_bfnj_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part2_1_lvl2_part3_1_perm1230__D_1__D_2__D_3__D_0 );
			F_ae_perm10234__S__D_0__D_1__D_2__D_3.AlignModesWith( modes_0, T_bfnj_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3, modes_0 );
			tempShape = F_ae__D_0_1__D_2_3.Shape();
			tempShape.push_back( g.Shape()[1] );
			tempShape.push_back( g.Shape()[2] );
			tempShape.push_back( g.Shape()[3] );
			F_ae_perm10234__S__D_0__D_1__D_2__D_3.ResizeTo( tempShape );
			   // -1.0 * Ftemp1_lvl1_part2_1_lvl2_part3_1[*,D1,D2,D3]_efmn * T_bfnj_lvl1_part2_1_lvl2_part3_1[D0,D1,D2,D3]_fmna + 0.0 * F_ae[D0,*,D1,D2,D3]_eafmn
if(commRank == 0){
flops += 2*1*Ftemp1_lvl1_part2_1_lvl2_part3_1__S__D_1__D_2__D_3.Dimension(1)*Ftemp1_lvl1_part2_1_lvl2_part3_1__S__D_1__D_2__D_3.Dimension(2)*Ftemp1_lvl1_part2_1_lvl2_part3_1__S__D_1__D_2__D_3.Dimension(3)*1*Ftemp1_lvl1_part2_1_lvl2_part3_1__S__D_1__D_2__D_3.Dimension(0)*1*T_bfnj_lvl1_part2_1_lvl2_part3_1_perm1230__D_1__D_2__D_3__D_0.Dimension(0);
}
			LocalContract(-1.0, Ftemp1_lvl1_part2_1_lvl2_part3_1__S__D_1__D_2__D_3.LockedTensor(), indices_efmn, false,
				T_bfnj_lvl1_part2_1_lvl2_part3_1_perm1230__D_1__D_2__D_3__D_0.LockedTensor(), indices_fmna, false,
				0.0, F_ae_perm10234__S__D_0__D_1__D_2__D_3.Tensor(), indices_eafmn, false);
			Ftemp1_lvl1_part2_1_lvl2_part3_1__S__D_1__D_2__D_3.EmptyData();
			T_bfnj_lvl1_part2_1_lvl2_part3_1_perm1230__D_1__D_2__D_3__D_0.EmptyData();
			   // F_ae[D01,D23] <- F_ae[D0,*,D1,D2,D3] (with SumScatter on (D1)(D2)(D3))
			F_ae__D_0_1__D_2_3.ReduceScatterUpdateRedistFrom( F_ae_perm10234__S__D_0__D_1__D_2__D_3, 1.0, modes_4_3_2 );
			F_ae_perm10234__S__D_0__D_1__D_2__D_3.EmptyData();

			SlidePartitionDown
			( Ftemp1_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3,  Ftemp1_lvl1_part2_1_lvl2_part3_0__D_0__D_1__D_2__D_3,
			       Ftemp1_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  Ftemp1_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, Ftemp1_lvl1_part2_1_lvl2_part3_2__D_0__D_1__D_2__D_3, 3 );
			SlidePartitionDown
			( T_bfnj_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3,  T_bfnj_lvl1_part2_1_lvl2_part3_0__D_0__D_1__D_2__D_3,
			       T_bfnj_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  T_bfnj_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part2_1_lvl2_part3_2__D_0__D_1__D_2__D_3, 3 );

		}
		//****

		SlidePartitionDown
		( Ftemp1_lvl1_part2T__D_0__D_1__D_2__D_3,  Ftemp1_lvl1_part2_0__D_0__D_1__D_2__D_3,
		       Ftemp1_lvl1_part2_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  Ftemp1_lvl1_part2B__D_0__D_1__D_2__D_3, Ftemp1_lvl1_part2_2__D_0__D_1__D_2__D_3, 2 );
		SlidePartitionDown
		( T_bfnj_lvl1_part2T__D_0__D_1__D_2__D_3,  T_bfnj_lvl1_part2_0__D_0__D_1__D_2__D_3,
		       T_bfnj_lvl1_part2_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  T_bfnj_lvl1_part2B__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part2_2__D_0__D_1__D_2__D_3, 2 );

	}
	//****
	Ftemp1__D_0__D_1__D_2__D_3.EmptyData();
	//**** (out of 1)
	//**** Is real	0 shadows
		//Outputs:
		//  F_ae__D_0_1__D_2_3
	PartitionDown(Ftemp2__D_0__D_1__D_2__D_3, Ftemp2_lvl1_part2T__D_0__D_1__D_2__D_3, Ftemp2_lvl1_part2B__D_0__D_1__D_2__D_3, 2, 0);
	PartitionDown(F_ae__D_0_1__D_2_3, F_ae_lvl1_part1T__D_0_1__D_2_3, F_ae_lvl1_part1B__D_0_1__D_2_3, 1, 0);
	while(F_ae_lvl1_part1T__D_0_1__D_2_3.Dimension(1) < F_ae__D_0_1__D_2_3.Dimension(1))
	{
		RepartitionDown
		( Ftemp2_lvl1_part2T__D_0__D_1__D_2__D_3,  Ftemp2_lvl1_part2_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       Ftemp2_lvl1_part2_1__D_0__D_1__D_2__D_3,
		  Ftemp2_lvl1_part2B__D_0__D_1__D_2__D_3, Ftemp2_lvl1_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );
		RepartitionDown
		( F_ae_lvl1_part1T__D_0_1__D_2_3,  F_ae_lvl1_part1_0__D_0_1__D_2_3,
		  /**/ /**/
		       F_ae_lvl1_part1_1__D_0_1__D_2_3,
		  F_ae_lvl1_part1B__D_0_1__D_2_3, F_ae_lvl1_part1_2__D_0_1__D_2_3, 1, blkSize );

		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  F_ae_lvl1_part1_1__D_0_1__D_2_3
		PartitionDown(Ftemp2_lvl1_part2_1__D_0__D_1__D_2__D_3, Ftemp2_lvl1_part2_1_lvl2_part1T__D_0__D_1__D_2__D_3, Ftemp2_lvl1_part2_1_lvl2_part1B__D_0__D_1__D_2__D_3, 1, 0);
		PartitionDown(t_fj__D_0_1__D_2_3, t_fj_lvl2_part1T__D_0_1__D_2_3, t_fj_lvl2_part1B__D_0_1__D_2_3, 1, 0);
		while(Ftemp2_lvl1_part2_1_lvl2_part1T__D_0__D_1__D_2__D_3.Dimension(1) < Ftemp2_lvl1_part2_1__D_0__D_1__D_2__D_3.Dimension(1))
		{
			RepartitionDown
			( Ftemp2_lvl1_part2_1_lvl2_part1T__D_0__D_1__D_2__D_3,  Ftemp2_lvl1_part2_1_lvl2_part1_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       Ftemp2_lvl1_part2_1_lvl2_part1_1__D_0__D_1__D_2__D_3,
			  Ftemp2_lvl1_part2_1_lvl2_part1B__D_0__D_1__D_2__D_3, Ftemp2_lvl1_part2_1_lvl2_part1_2__D_0__D_1__D_2__D_3, 1, blkSize );
			RepartitionDown
			( t_fj_lvl2_part1T__D_0_1__D_2_3,  t_fj_lvl2_part1_0__D_0_1__D_2_3,
			  /**/ /**/
			       t_fj_lvl2_part1_1__D_0_1__D_2_3,
			  t_fj_lvl2_part1B__D_0_1__D_2_3, t_fj_lvl2_part1_2__D_0_1__D_2_3, 1, blkSize );

			Permute( Ftemp2_lvl1_part2_1_lvl2_part1_1__D_0__D_1__D_2__D_3, Ftemp2_lvl1_part2_1_lvl2_part1_1_perm0231__D_0__D_2__D_3__D_1 );
			   // t_fj_lvl2_part1_1[D03,D21] <- t_fj_lvl2_part1_1[D01,D23]
			t_fj_lvl2_part1_1__D_0_3__D_2_1.AlignModesWith( modes_0_1, Ftemp2_lvl1_part2_1_lvl2_part1_1__D_0__D_1__D_2__D_3, modes_3_1 );
			t_fj_lvl2_part1_1__D_0_3__D_2_1.AllToAllRedistFrom( t_fj_lvl2_part1_1__D_0_1__D_2_3, modes_1_3 );
			   // t_fj_lvl2_part1_1[D30,D21] <- t_fj_lvl2_part1_1[D03,D21]
			t_fj_lvl2_part1_1__D_3_0__D_2_1.AlignModesWith( modes_0_1, Ftemp2_lvl1_part2_1_lvl2_part1_1__D_0__D_1__D_2__D_3, modes_3_1 );
			t_fj_lvl2_part1_1__D_3_0__D_2_1.PermutationRedistFrom( t_fj_lvl2_part1_1__D_0_3__D_2_1, modes_0_3 );
			t_fj_lvl2_part1_1__D_0_3__D_2_1.EmptyData();
			   // t_fj_lvl2_part1_1[D30,D12] <- t_fj_lvl2_part1_1[D30,D21]
			t_fj_lvl2_part1_1__D_3_0__D_1_2.AlignModesWith( modes_0_1, Ftemp2_lvl1_part2_1_lvl2_part1_1__D_0__D_1__D_2__D_3, modes_3_1 );
			t_fj_lvl2_part1_1__D_3_0__D_1_2.PermutationRedistFrom( t_fj_lvl2_part1_1__D_3_0__D_2_1, modes_2_1 );
			t_fj_lvl2_part1_1__D_3_0__D_2_1.EmptyData();
			   // t_fj_lvl2_part1_1[D3,D1] <- t_fj_lvl2_part1_1[D30,D12]
			t_fj_lvl2_part1_1__D_3__D_1.AlignModesWith( modes_0_1, Ftemp2_lvl1_part2_1_lvl2_part1_1__D_0__D_1__D_2__D_3, modes_3_1 );
			t_fj_lvl2_part1_1__D_3__D_1.AllGatherRedistFrom( t_fj_lvl2_part1_1__D_3_0__D_1_2, modes_0_2 );
			t_fj_lvl2_part1_1__D_3_0__D_1_2.EmptyData();
			F_ae_lvl1_part1_1__D_0__D_2__D_3__D_1.AlignModesWith( modes_0_1, Ftemp2_lvl1_part2_1_lvl2_part1_1__D_0__D_1__D_2__D_3, modes_0_2 );
			tempShape = F_ae_lvl1_part1_1__D_0_1__D_2_3.Shape();
			tempShape.push_back( g.Shape()[3] );
			tempShape.push_back( g.Shape()[1] );
			F_ae_lvl1_part1_1__D_0__D_2__D_3__D_1.ResizeTo( tempShape );
			   // 1.0 * Ftemp2_lvl1_part2_1_lvl2_part1_1[D0,D1,D2,D3]_aefm * t_fj_lvl2_part1_1[D3,D1]_fm + 0.0 * F_ae_lvl1_part1_1[D0,D2,D3,D1]_aefm
if(commRank == 0){
flops += 2*1*Ftemp2_lvl1_part2_1_lvl2_part1_1_perm0231__D_0__D_2__D_3__D_1.Dimension(1)*Ftemp2_lvl1_part2_1_lvl2_part1_1_perm0231__D_0__D_2__D_3__D_1.Dimension(3)*1*Ftemp2_lvl1_part2_1_lvl2_part1_1_perm0231__D_0__D_2__D_3__D_1.Dimension(0)*Ftemp2_lvl1_part2_1_lvl2_part1_1_perm0231__D_0__D_2__D_3__D_1.Dimension(2)*1;
}
			LocalContract(1.0, Ftemp2_lvl1_part2_1_lvl2_part1_1_perm0231__D_0__D_2__D_3__D_1.LockedTensor(), indices_aefm, false,
				t_fj_lvl2_part1_1__D_3__D_1.LockedTensor(), indices_fm, false,
				0.0, F_ae_lvl1_part1_1__D_0__D_2__D_3__D_1.Tensor(), indices_aefm, false);
			Ftemp2_lvl1_part2_1_lvl2_part1_1_perm0231__D_0__D_2__D_3__D_1.EmptyData();
			t_fj_lvl2_part1_1__D_3__D_1.EmptyData();
			   // F_ae_lvl1_part1_1[D01,D23] <- F_ae_lvl1_part1_1[D0,D2,D3,D1] (with SumScatter on (D3)(D1))
			F_ae_lvl1_part1_1__D_0_1__D_2_3.ReduceScatterUpdateRedistFrom( F_ae_lvl1_part1_1__D_0__D_2__D_3__D_1, 1.0, modes_3_2 );
			F_ae_lvl1_part1_1__D_0__D_2__D_3__D_1.EmptyData();

			SlidePartitionDown
			( Ftemp2_lvl1_part2_1_lvl2_part1T__D_0__D_1__D_2__D_3,  Ftemp2_lvl1_part2_1_lvl2_part1_0__D_0__D_1__D_2__D_3,
			       Ftemp2_lvl1_part2_1_lvl2_part1_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  Ftemp2_lvl1_part2_1_lvl2_part1B__D_0__D_1__D_2__D_3, Ftemp2_lvl1_part2_1_lvl2_part1_2__D_0__D_1__D_2__D_3, 1 );
			SlidePartitionDown
			( t_fj_lvl2_part1T__D_0_1__D_2_3,  t_fj_lvl2_part1_0__D_0_1__D_2_3,
			       t_fj_lvl2_part1_1__D_0_1__D_2_3,
			  /**/ /**/
			  t_fj_lvl2_part1B__D_0_1__D_2_3, t_fj_lvl2_part1_2__D_0_1__D_2_3, 1 );

		}
		//****

		SlidePartitionDown
		( Ftemp2_lvl1_part2T__D_0__D_1__D_2__D_3,  Ftemp2_lvl1_part2_0__D_0__D_1__D_2__D_3,
		       Ftemp2_lvl1_part2_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  Ftemp2_lvl1_part2B__D_0__D_1__D_2__D_3, Ftemp2_lvl1_part2_2__D_0__D_1__D_2__D_3, 2 );
		SlidePartitionDown
		( F_ae_lvl1_part1T__D_0_1__D_2_3,  F_ae_lvl1_part1_0__D_0_1__D_2_3,
		       F_ae_lvl1_part1_1__D_0_1__D_2_3,
		  /**/ /**/
		  F_ae_lvl1_part1B__D_0_1__D_2_3, F_ae_lvl1_part1_2__D_0_1__D_2_3, 1 );

	}
	//****
	Ftemp2__D_0__D_1__D_2__D_3.EmptyData();
	//**** (out of 1)
	//**** Is real	0 shadows
		//Outputs:
		//  F_ae__D_0_1__D_2_3
	PartitionDown(H_me__D_0_1__D_2_3, H_me_lvl1_part1T__D_0_1__D_2_3, H_me_lvl1_part1B__D_0_1__D_2_3, 1, 0);
	PartitionDown(F_ae__D_0_1__D_2_3, F_ae_lvl1_part1T__D_0_1__D_2_3, F_ae_lvl1_part1B__D_0_1__D_2_3, 1, 0);
	while(F_ae_lvl1_part1T__D_0_1__D_2_3.Dimension(1) < F_ae__D_0_1__D_2_3.Dimension(1))
	{
		RepartitionDown
		( H_me_lvl1_part1T__D_0_1__D_2_3,  H_me_lvl1_part1_0__D_0_1__D_2_3,
		  /**/ /**/
		       H_me_lvl1_part1_1__D_0_1__D_2_3,
		  H_me_lvl1_part1B__D_0_1__D_2_3, H_me_lvl1_part1_2__D_0_1__D_2_3, 1, blkSize );
		RepartitionDown
		( F_ae_lvl1_part1T__D_0_1__D_2_3,  F_ae_lvl1_part1_0__D_0_1__D_2_3,
		  /**/ /**/
		       F_ae_lvl1_part1_1__D_0_1__D_2_3,
		  F_ae_lvl1_part1B__D_0_1__D_2_3, F_ae_lvl1_part1_2__D_0_1__D_2_3, 1, blkSize );

		Permute( F_ae_lvl1_part1_1__D_0_1__D_2_3, F_ae_lvl1_part1_1_perm10__D_2_3__D_0_1 );
		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  F_ae_lvl1_part1_1_perm10__D_2_3__D_0_1
		PartitionDown(H_me_lvl1_part1_1__D_0_1__D_2_3, H_me_lvl1_part1_1_lvl2_part0T__D_0_1__D_2_3, H_me_lvl1_part1_1_lvl2_part0B__D_0_1__D_2_3, 0, 0);
		PartitionDown(t_fj__D_0_1__D_2_3, t_fj_lvl2_part1T__D_0_1__D_2_3, t_fj_lvl2_part1B__D_0_1__D_2_3, 1, 0);
		while(H_me_lvl1_part1_1_lvl2_part0T__D_0_1__D_2_3.Dimension(0) < H_me_lvl1_part1_1__D_0_1__D_2_3.Dimension(0))
		{
			RepartitionDown
			( H_me_lvl1_part1_1_lvl2_part0T__D_0_1__D_2_3,  H_me_lvl1_part1_1_lvl2_part0_0__D_0_1__D_2_3,
			  /**/ /**/
			       H_me_lvl1_part1_1_lvl2_part0_1__D_0_1__D_2_3,
			  H_me_lvl1_part1_1_lvl2_part0B__D_0_1__D_2_3, H_me_lvl1_part1_1_lvl2_part0_2__D_0_1__D_2_3, 0, blkSize );
			RepartitionDown
			( t_fj_lvl2_part1T__D_0_1__D_2_3,  t_fj_lvl2_part1_0__D_0_1__D_2_3,
			  /**/ /**/
			       t_fj_lvl2_part1_1__D_0_1__D_2_3,
			  t_fj_lvl2_part1B__D_0_1__D_2_3, t_fj_lvl2_part1_2__D_0_1__D_2_3, 1, blkSize );

			   // H_me_lvl1_part1_1_lvl2_part0_1[*,D23] <- H_me_lvl1_part1_1_lvl2_part0_1[D01,D23]
			H_me_lvl1_part1_1_lvl2_part0_1_perm10__D_2_3__S.AlignModesWith( modes_1, F_ae_lvl1_part1_1__D_0_1__D_2_3, modes_1 );
			H_me_lvl1_part1_1_lvl2_part0_1_perm10__D_2_3__S.AllGatherRedistFrom( H_me_lvl1_part1_1_lvl2_part0_1__D_0_1__D_2_3, modes_0_1 );
			   // t_fj_lvl2_part1_1[D01,*] <- t_fj_lvl2_part1_1[D01,D23]
			t_fj_lvl2_part1_1_perm10__S__D_0_1.AlignModesWith( modes_0, F_ae_lvl1_part1_1__D_0_1__D_2_3, modes_0 );
			t_fj_lvl2_part1_1_perm10__S__D_0_1.AllGatherRedistFrom( t_fj_lvl2_part1_1__D_0_1__D_2_3, modes_2_3 );
			   // -1.0 * H_me_lvl1_part1_1_lvl2_part0_1[*,D23]_em * t_fj_lvl2_part1_1[D01,*]_ma + 1.0 * F_ae_lvl1_part1_1[D01,D23]_ea
if(commRank == 0){
flops += 2*1*H_me_lvl1_part1_1_lvl2_part0_1_perm10__D_2_3__S.Dimension(0)*1*H_me_lvl1_part1_1_lvl2_part0_1_perm10__D_2_3__S.Dimension(1)*1*t_fj_lvl2_part1_1_perm10__S__D_0_1.Dimension(0);
}
			LocalContractAndLocalEliminate(-1.0, H_me_lvl1_part1_1_lvl2_part0_1_perm10__D_2_3__S.LockedTensor(), indices_em, false,
				t_fj_lvl2_part1_1_perm10__S__D_0_1.LockedTensor(), indices_ma, false,
				1.0, F_ae_lvl1_part1_1_perm10__D_2_3__D_0_1.Tensor(), indices_ea, false);
			H_me_lvl1_part1_1_lvl2_part0_1_perm10__D_2_3__S.EmptyData();
			t_fj_lvl2_part1_1_perm10__S__D_0_1.EmptyData();

			SlidePartitionDown
			( H_me_lvl1_part1_1_lvl2_part0T__D_0_1__D_2_3,  H_me_lvl1_part1_1_lvl2_part0_0__D_0_1__D_2_3,
			       H_me_lvl1_part1_1_lvl2_part0_1__D_0_1__D_2_3,
			  /**/ /**/
			  H_me_lvl1_part1_1_lvl2_part0B__D_0_1__D_2_3, H_me_lvl1_part1_1_lvl2_part0_2__D_0_1__D_2_3, 0 );
			SlidePartitionDown
			( t_fj_lvl2_part1T__D_0_1__D_2_3,  t_fj_lvl2_part1_0__D_0_1__D_2_3,
			       t_fj_lvl2_part1_1__D_0_1__D_2_3,
			  /**/ /**/
			  t_fj_lvl2_part1B__D_0_1__D_2_3, t_fj_lvl2_part1_2__D_0_1__D_2_3, 1 );

		}
		//****
		Permute( F_ae_lvl1_part1_1_perm10__D_2_3__D_0_1, F_ae_lvl1_part1_1__D_0_1__D_2_3 );
		F_ae_lvl1_part1_1_perm10__D_2_3__D_0_1.EmptyData();

		SlidePartitionDown
		( H_me_lvl1_part1T__D_0_1__D_2_3,  H_me_lvl1_part1_0__D_0_1__D_2_3,
		       H_me_lvl1_part1_1__D_0_1__D_2_3,
		  /**/ /**/
		  H_me_lvl1_part1B__D_0_1__D_2_3, H_me_lvl1_part1_2__D_0_1__D_2_3, 1 );
		SlidePartitionDown
		( F_ae_lvl1_part1T__D_0_1__D_2_3,  F_ae_lvl1_part1_0__D_0_1__D_2_3,
		       F_ae_lvl1_part1_1__D_0_1__D_2_3,
		  /**/ /**/
		  F_ae_lvl1_part1B__D_0_1__D_2_3, F_ae_lvl1_part1_2__D_0_1__D_2_3, 1 );

	}
	//****


//****


//END_CODE

    /*****************************************/
    mpi::Barrier(g.OwningComm());
    runTime = mpi::Time() - startTime;
    gflops = flops / (1e9 * runTime);
#ifdef CORRECTNESS
    DistTensor<double> diff_F(dist__D_0_1__D_2_3, g);
    diff_F.ResizeTo(check_F);
    Diff(check_F, F_ae__D_0_1__D_2_3, diff_F);
   norm = 1.0;
   norm = Norm(diff_F);
   if (commRank == 0){
     std::cout << "NORM_F " << norm << std::endl;
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


