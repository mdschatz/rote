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
    const Int commRank = mpi::CommRank(mpi::COMM_WORLD);

//START_DECL
ObjShape tempShape;
TensorDistribution dist__D_0__D_1__D_2__D_3 = tmen::StringToTensorDist("[(0),(1),(2),(3)]");
TensorDistribution dist__D_0__D_2 = tmen::StringToTensorDist("[(0),(2)]");
TensorDistribution dist__D_1__D_3 = tmen::StringToTensorDist("[(1),(3)]");
TensorDistribution dist__D_0_1__D_2_3 = tmen::StringToTensorDist("[(0,1),(2,3)]");
TensorDistribution dist__D_0_1__D_3_2 = tmen::StringToTensorDist("[(0,1),(3,2)]");
TensorDistribution dist__D_1_0__D_3_2 = tmen::StringToTensorDist("[(1,0),(3,2)]");
Permutation perm_0_1( 2 );
perm_0_1[0] = 0;
perm_0_1[1] = 1;
Permutation perm_0_1_2_3( 4 );
perm_0_1_2_3[0] = 0;
perm_0_1_2_3[1] = 1;
perm_0_1_2_3[2] = 2;
perm_0_1_2_3[3] = 3;
Permutation perm_0_2_1_3( 4 );
perm_0_2_1_3[0] = 0;
perm_0_2_1_3[1] = 2;
perm_0_2_1_3[2] = 1;
perm_0_2_1_3[3] = 3;
ModeArray modes_0_1( 2 );
modes_0_1[0] = 0;
modes_0_1[1] = 1;
ModeArray modes_0_2( 2 );
modes_0_2[0] = 0;
modes_0_2[1] = 2;
ModeArray modes_1_3( 2 );
modes_1_3[0] = 1;
modes_1_3[1] = 3;
ModeArray modes_2_3( 2 );
modes_2_3[0] = 2;
modes_2_3[1] = 3;
IndexArray indices_em( 2 );
indices_em[0] = 'e';
indices_em[1] = 'm';
IndexArray indices_emfn( 4 );
indices_emfn[0] = 'e';
indices_emfn[1] = 'm';
indices_emfn[2] = 'f';
indices_emfn[3] = 'n';
IndexArray indices_fn( 2 );
indices_fn[0] = 'f';
indices_fn[1] = 'n';
	//T_bfnj[D0,D1,D2,D3]
DistTensor<double> T_bfnj__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
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
DistTensor<double> Tau_efmn_lvl1_part2_1_lvl2_part3_1_perm0213__D_0__D_2__D_1__D_3( dist__D_0__D_1__D_2__D_3, g );
Tau_efmn_lvl1_part2_1_lvl2_part3_1_perm0213__D_0__D_2__D_1__D_3.SetLocalPermutation( perm_0_2_1_3 );
	//Tau_efmn_lvl1_part2_1_lvl2_part3_2[D0,D1,D2,D3]
DistTensor<double> Tau_efmn_lvl1_part2_1_lvl2_part3_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Tau_efmn_lvl1_part2_2[D0,D1,D2,D3]
DistTensor<double> Tau_efmn_lvl1_part2_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
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
	//t_fj_lvl1_part1_1[D0,D2]
DistTensor<double> t_fj_lvl1_part1_1__D_0__D_2( dist__D_0__D_2, g );
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
	//t_fj_lvl2_part1_1[D01,D32]
DistTensor<double> t_fj_lvl2_part1_1__D_0_1__D_3_2( dist__D_0_1__D_3_2, g );
	//t_fj_lvl2_part1_1[D10,D32]
DistTensor<double> t_fj_lvl2_part1_1__D_1_0__D_3_2( dist__D_1_0__D_3_2, g );
	//t_fj_lvl2_part1_1[D1,D3]
DistTensor<double> t_fj_lvl2_part1_1__D_1__D_3( dist__D_1__D_3, g );
	//t_fj_lvl2_part1_2[D01,D23]
DistTensor<double> t_fj_lvl2_part1_2__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
// t_fj has 2 dims
//	Starting distribution: [D01,D23] or _D_0_1__D_2_3
ObjShape t_fj__D_0_1__D_2_3_tempShape( 2 );
t_fj__D_0_1__D_2_3_tempShape[ 0 ] = n_v;
t_fj__D_0_1__D_2_3_tempShape[ 1 ] = n_o;
t_fj__D_0_1__D_2_3.ResizeTo( t_fj__D_0_1__D_2_3_tempShape );
MakeUniform( t_fj__D_0_1__D_2_3 );
// T_bfnj has 4 dims
//	Starting distribution: [D0,D1,D2,D3] or _D_0__D_1__D_2__D_3
ObjShape T_bfnj__D_0__D_1__D_2__D_3_tempShape( 4 );
T_bfnj__D_0__D_1__D_2__D_3_tempShape[ 0 ] = n_v;
T_bfnj__D_0__D_1__D_2__D_3_tempShape[ 1 ] = n_v;
T_bfnj__D_0__D_1__D_2__D_3_tempShape[ 2 ] = n_o;
T_bfnj__D_0__D_1__D_2__D_3_tempShape[ 3 ] = n_o;
T_bfnj__D_0__D_1__D_2__D_3.ResizeTo( T_bfnj__D_0__D_1__D_2__D_3_tempShape );
MakeUniform( T_bfnj__D_0__D_1__D_2__D_3 );
// Tau_efmn has 4 dims
ObjShape Tau_efmn__D_0__D_1__D_2__D_3_tempShape( 4 );
Tau_efmn__D_0__D_1__D_2__D_3_tempShape[ 0 ] = n_v;
Tau_efmn__D_0__D_1__D_2__D_3_tempShape[ 1 ] = n_v;
Tau_efmn__D_0__D_1__D_2__D_3_tempShape[ 2 ] = n_o;
Tau_efmn__D_0__D_1__D_2__D_3_tempShape[ 3 ] = n_o;
Tau_efmn__D_0__D_1__D_2__D_3.ResizeTo( Tau_efmn__D_0__D_1__D_2__D_3_tempShape );
MakeUniform( Tau_efmn__D_0__D_1__D_2__D_3 );
TensorDistribution dist__S__D_0__D_3__S = tmen::StringToTensorDist("[(),(0),(3),()]");
TensorDistribution dist__S__D_2__S__D_1__D_0__D_3 = tmen::StringToTensorDist("[(),(2),(),(1),(0),(3)]");
TensorDistribution dist__D_0__S__D_2__D_1_3 = tmen::StringToTensorDist("[(0),(),(2),(1,3)]");
TensorDistribution dist__D_0__S = tmen::StringToTensorDist("[(0),()]");
TensorDistribution dist__D_0__D_1__S__D_2_3 = tmen::StringToTensorDist("[(0),(1),(),(2,3)]");
TensorDistribution dist__D_0__D_1__S__D_2__D_3 = tmen::StringToTensorDist("[(0),(1),(),(2),(3)]");
TensorDistribution dist__D_0__D_1__D_3__D_2 = tmen::StringToTensorDist("[(0),(1),(3),(2)]");
TensorDistribution dist__D_0__D_2__S__D_1_3 = tmen::StringToTensorDist("[(0),(2),(),(1,3)]");
TensorDistribution dist__D_1__S__D_2__D_3 = tmen::StringToTensorDist("[(1),(),(2),(3)]");
TensorDistribution dist__D_1__D_0__D_2__D_3 = tmen::StringToTensorDist("[(1),(0),(2),(3)]");
TensorDistribution dist__D_1__D_0__D_3__S = tmen::StringToTensorDist("[(1),(0),(3),()]");
TensorDistribution dist__D_0_1_3__D_2 = tmen::StringToTensorDist("[(0,1,3),(2)]");
TensorDistribution dist__D_3__S = tmen::StringToTensorDist("[(3),()]");
TensorDistribution dist__D_0_1__D_2 = tmen::StringToTensorDist("[(0,1),(2)]");
TensorDistribution dist__D_3_0_1__D_2 = tmen::StringToTensorDist("[(3,0,1),(2)]");
Permutation perm_0_1_3_2( 4 );
perm_0_1_3_2[0] = 0;
perm_0_1_3_2[1] = 1;
perm_0_1_3_2[2] = 3;
perm_0_1_3_2[3] = 2;
Permutation perm_0_1_3_2_4( 5 );
perm_0_1_3_2_4[0] = 0;
perm_0_1_3_2_4[1] = 1;
perm_0_1_3_2_4[2] = 3;
perm_0_1_3_2_4[3] = 2;
perm_0_1_3_2_4[4] = 4;
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
Permutation perm_3_0_1_2( 4 );
perm_3_0_1_2[0] = 3;
perm_3_0_1_2[1] = 0;
perm_3_0_1_2[2] = 1;
perm_3_0_1_2[3] = 2;
Permutation perm_3_1_0_2_4_5( 6 );
perm_3_1_0_2_4_5[0] = 3;
perm_3_1_0_2_4_5[1] = 1;
perm_3_1_0_2_4_5[2] = 0;
perm_3_1_0_2_4_5[3] = 2;
perm_3_1_0_2_4_5[4] = 4;
perm_3_1_0_2_4_5[5] = 5;
ModeArray modes_0( 1 );
modes_0[0] = 0;
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
ModeArray modes_0_1_3_2( 4 );
modes_0_1_3_2[0] = 0;
modes_0_1_3_2[1] = 1;
modes_0_1_3_2[2] = 3;
modes_0_1_3_2[3] = 2;
ModeArray modes_0_2_3( 3 );
modes_0_2_3[0] = 0;
modes_0_2_3[1] = 2;
modes_0_2_3[2] = 3;
ModeArray modes_0_3( 2 );
modes_0_3[0] = 0;
modes_0_3[1] = 3;
ModeArray modes_1( 1 );
modes_1[0] = 1;
ModeArray modes_1_0_2_3( 4 );
modes_1_0_2_3[0] = 1;
modes_1_0_2_3[1] = 0;
modes_1_0_2_3[2] = 2;
modes_1_0_2_3[3] = 3;
ModeArray modes_1_2( 2 );
modes_1_2[0] = 1;
modes_1_2[1] = 2;
ModeArray modes_1_2_3( 3 );
modes_1_2_3[0] = 1;
modes_1_2_3[1] = 2;
modes_1_2_3[2] = 3;
ModeArray modes_2( 1 );
modes_2[0] = 2;
ModeArray modes_2_1( 2 );
modes_2_1[0] = 2;
modes_2_1[1] = 1;
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
IndexArray indices_bmejf( 5 );
indices_bmejf[0] = 'b';
indices_bmejf[1] = 'm';
indices_bmejf[2] = 'e';
indices_bmejf[3] = 'j';
indices_bmejf[4] = 'f';
IndexArray indices_embjfn( 6 );
indices_embjfn[0] = 'e';
indices_embjfn[1] = 'm';
indices_embjfn[2] = 'b';
indices_embjfn[3] = 'j';
indices_embjfn[4] = 'f';
indices_embjfn[5] = 'n';
IndexArray indices_fj( 2 );
indices_fj[0] = 'f';
indices_fj[1] = 'j';
IndexArray indices_fnbj( 4 );
indices_fnbj[0] = 'f';
indices_fnbj[1] = 'n';
indices_fnbj[2] = 'b';
indices_fnbj[3] = 'j';
IndexArray indices_mjeb( 4 );
indices_mjeb[0] = 'm';
indices_mjeb[1] = 'j';
indices_mjeb[2] = 'e';
indices_mjeb[3] = 'b';
IndexArray indices_mjen( 4 );
indices_mjen[0] = 'm';
indices_mjen[1] = 'j';
indices_mjen[2] = 'e';
indices_mjen[3] = 'n';
IndexArray indices_nb( 2 );
indices_nb[0] = 'n';
indices_nb[1] = 'b';
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
	//W_bmje[D0,D1,D2,D3]
DistTensor<double> W_bmje__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_bmje_lvl0_part0B[D0,D1,D2,D3]
DistTensor<double> W_bmje_lvl0_part0B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_bmje_lvl0_part0T[D0,D1,D2,D3]
DistTensor<double> W_bmje_lvl0_part0T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_bmje_lvl0_part1B[D0,D1,D2,D3]
DistTensor<double> W_bmje_lvl0_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_bmje_lvl0_part1T[D0,D1,D2,D3]
DistTensor<double> W_bmje_lvl0_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_bmje_lvl1_part0B[D0,D1,D2,D3]
DistTensor<double> W_bmje_lvl1_part0B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_bmje_lvl1_part0T[D0,D1,D2,D3]
DistTensor<double> W_bmje_lvl1_part0T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_bmje_lvl1_part0_0[D0,D1,D2,D3]
DistTensor<double> W_bmje_lvl1_part0_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_bmje_lvl1_part0_1[D0,D1,D2,D3]
DistTensor<double> W_bmje_lvl1_part0_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_bmje_lvl1_part0_1_lvl1_part2B[D0,D1,D2,D3]
DistTensor<double> W_bmje_lvl1_part0_1_lvl1_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_bmje_lvl1_part0_1_lvl1_part2T[D0,D1,D2,D3]
DistTensor<double> W_bmje_lvl1_part0_1_lvl1_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_bmje_lvl1_part0_1_lvl2_part2B[D0,D1,D2,D3]
DistTensor<double> W_bmje_lvl1_part0_1_lvl2_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_bmje_lvl1_part0_1_lvl2_part2T[D0,D1,D2,D3]
DistTensor<double> W_bmje_lvl1_part0_1_lvl2_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_bmje_lvl1_part0_1_lvl2_part2_0[D0,D1,D2,D3]
DistTensor<double> W_bmje_lvl1_part0_1_lvl2_part2_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_bmje_lvl1_part0_1_lvl2_part2_1[D0,D1,D2,D3]
DistTensor<double> W_bmje_lvl1_part0_1_lvl2_part2_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_bmje_lvl1_part0_1_lvl2_part2_1[*,D2,*,D1,D0,D3]
DistTensor<double> W_bmje_lvl1_part0_1_lvl2_part2_1_perm310245__D_1__D_2__S__S__D_0__D_3( dist__S__D_2__S__D_1__D_0__D_3, g );
W_bmje_lvl1_part0_1_lvl2_part2_1_perm310245__D_1__D_2__S__S__D_0__D_3.SetLocalPermutation( perm_3_1_0_2_4_5 );
	//W_bmje_lvl1_part0_1_lvl2_part2_1_temp[D0,D1,D2,D3]
DistTensor<double> W_bmje_lvl1_part0_1_lvl2_part2_1_temp__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_bmje_lvl1_part0_1_lvl2_part2_1_temp[D0,D2,*,D13]
DistTensor<double> W_bmje_lvl1_part0_1_lvl2_part2_1_temp__D_0__D_2__S__D_1_3( dist__D_0__D_2__S__D_1_3, g );
	//W_bmje_lvl1_part0_1_lvl2_part2_1_temp[D0,*,D2,D13]
DistTensor<double> W_bmje_lvl1_part0_1_lvl2_part2_1_temp__D_0__S__D_2__D_1_3( dist__D_0__S__D_2__D_1_3, g );
	//W_bmje_lvl1_part0_1_lvl2_part2_2[D0,D1,D2,D3]
DistTensor<double> W_bmje_lvl1_part0_1_lvl2_part2_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_bmje_lvl1_part0_2[D0,D1,D2,D3]
DistTensor<double> W_bmje_lvl1_part0_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_bmje_lvl1_part1B[D0,D1,D2,D3]
DistTensor<double> W_bmje_lvl1_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_bmje_lvl1_part1T[D0,D1,D2,D3]
DistTensor<double> W_bmje_lvl1_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_bmje_lvl1_part1_0[D0,D1,D2,D3]
DistTensor<double> W_bmje_lvl1_part1_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_bmje_lvl1_part1_1[D0,D1,D2,D3]
DistTensor<double> W_bmje_lvl1_part1_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_bmje_lvl1_part1_1_lvl1_part2B[D0,D1,D2,D3]
DistTensor<double> W_bmje_lvl1_part1_1_lvl1_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_bmje_lvl1_part1_1_lvl1_part2T[D0,D1,D2,D3]
DistTensor<double> W_bmje_lvl1_part1_1_lvl1_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_bmje_lvl1_part1_1_lvl2_part2B[D0,D1,D2,D3]
DistTensor<double> W_bmje_lvl1_part1_1_lvl2_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_bmje_lvl1_part1_1_lvl2_part2T[D0,D1,D2,D3]
DistTensor<double> W_bmje_lvl1_part1_1_lvl2_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_bmje_lvl1_part1_1_lvl2_part2_0[D0,D1,D2,D3]
DistTensor<double> W_bmje_lvl1_part1_1_lvl2_part2_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_bmje_lvl1_part1_1_lvl2_part2_1[D0,D1,D2,D3]
DistTensor<double> W_bmje_lvl1_part1_1_lvl2_part2_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_bmje_lvl1_part1_1_lvl2_part2_1[D0,D1,*,D2,D3]
DistTensor<double> W_bmje_lvl1_part1_1_lvl2_part2_1_perm01324__D_0__D_1__D_2__S__D_3( dist__D_0__D_1__S__D_2__D_3, g );
W_bmje_lvl1_part1_1_lvl2_part2_1_perm01324__D_0__D_1__D_2__S__D_3.SetLocalPermutation( perm_0_1_3_2_4 );
	//W_bmje_lvl1_part1_1_lvl2_part2_1_temp[D0,D1,D2,D3]
DistTensor<double> W_bmje_lvl1_part1_1_lvl2_part2_1_temp__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_bmje_lvl1_part1_1_lvl2_part2_1_temp[D0,D1,*,D23]
DistTensor<double> W_bmje_lvl1_part1_1_lvl2_part2_1_temp__D_0__D_1__S__D_2_3( dist__D_0__D_1__S__D_2_3, g );
	//W_bmje_lvl1_part1_1_lvl2_part2_2[D0,D1,D2,D3]
DistTensor<double> W_bmje_lvl1_part1_1_lvl2_part2_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> W_bmje_lvl1_part1_1_perm1230__D_1__D_2__D_3__D_0( dist__D_0__D_1__D_2__D_3, g );
W_bmje_lvl1_part1_1_perm1230__D_1__D_2__D_3__D_0.SetLocalPermutation( perm_1_2_3_0 );
	//W_bmje_lvl1_part1_2[D0,D1,D2,D3]
DistTensor<double> W_bmje_lvl1_part1_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Wtemp1[D0,D1,D2,D3]
DistTensor<double> Wtemp1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Wtemp1_lvl0_part0B[D0,D1,D2,D3]
DistTensor<double> Wtemp1_lvl0_part0B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Wtemp1_lvl0_part0T[D0,D1,D2,D3]
DistTensor<double> Wtemp1_lvl0_part0T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Wtemp1_lvl0_part2B[D0,D1,D2,D3]
DistTensor<double> Wtemp1_lvl0_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Wtemp1_lvl0_part2T[D0,D1,D2,D3]
DistTensor<double> Wtemp1_lvl0_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Wtemp1_lvl1_part0B[D0,D1,D2,D3]
DistTensor<double> Wtemp1_lvl1_part0B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Wtemp1_lvl1_part0T[D0,D1,D2,D3]
DistTensor<double> Wtemp1_lvl1_part0T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Wtemp1_lvl1_part0_0[D0,D1,D2,D3]
DistTensor<double> Wtemp1_lvl1_part0_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Wtemp1_lvl1_part0_1[D0,D1,D2,D3]
DistTensor<double> Wtemp1_lvl1_part0_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Wtemp1_lvl1_part0_1_lvl1_part3B[D0,D1,D2,D3]
DistTensor<double> Wtemp1_lvl1_part0_1_lvl1_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Wtemp1_lvl1_part0_1_lvl1_part3T[D0,D1,D2,D3]
DistTensor<double> Wtemp1_lvl1_part0_1_lvl1_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Wtemp1_lvl1_part0_1_lvl2_part3B[D0,D1,D2,D3]
DistTensor<double> Wtemp1_lvl1_part0_1_lvl2_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Wtemp1_lvl1_part0_1_lvl2_part3T[D0,D1,D2,D3]
DistTensor<double> Wtemp1_lvl1_part0_1_lvl2_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Wtemp1_lvl1_part0_1_lvl2_part3_0[D0,D1,D2,D3]
DistTensor<double> Wtemp1_lvl1_part0_1_lvl2_part3_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Wtemp1_lvl1_part0_1_lvl2_part3_1[D0,D1,D2,D3]
DistTensor<double> Wtemp1_lvl1_part0_1_lvl2_part3_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Wtemp1_lvl1_part0_1_lvl2_part3_1[D1,D0,D2,D3]
DistTensor<double> Wtemp1_lvl1_part0_1_lvl2_part3_1__D_1__D_0__D_2__D_3( dist__D_1__D_0__D_2__D_3, g );
	//Wtemp1_lvl1_part0_1_lvl2_part3_1[D1,D0,D3,*]
DistTensor<double> Wtemp1_lvl1_part0_1_lvl2_part3_1__D_1__D_0__D_3__S( dist__D_1__D_0__D_3__S, g );
	//Wtemp1_lvl1_part0_1_lvl2_part3_1[*,D0,D3,*]
DistTensor<double> Wtemp1_lvl1_part0_1_lvl2_part3_1_perm1203__D_0__D_3__S__S( dist__S__D_0__D_3__S, g );
Wtemp1_lvl1_part0_1_lvl2_part3_1_perm1203__D_0__D_3__S__S.SetLocalPermutation( perm_1_2_0_3 );
	//Wtemp1_lvl1_part0_1_lvl2_part3_2[D0,D1,D2,D3]
DistTensor<double> Wtemp1_lvl1_part0_1_lvl2_part3_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Wtemp1_lvl1_part0_2[D0,D1,D2,D3]
DistTensor<double> Wtemp1_lvl1_part0_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Wtemp1_lvl1_part2B[D0,D1,D2,D3]
DistTensor<double> Wtemp1_lvl1_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Wtemp1_lvl1_part2T[D0,D1,D2,D3]
DistTensor<double> Wtemp1_lvl1_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Wtemp1_lvl1_part2_0[D0,D1,D2,D3]
DistTensor<double> Wtemp1_lvl1_part2_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Wtemp1_lvl1_part2_1[D0,D1,D2,D3]
DistTensor<double> Wtemp1_lvl1_part2_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Wtemp1_lvl1_part2_1_lvl1_part3B[D0,D1,D2,D3]
DistTensor<double> Wtemp1_lvl1_part2_1_lvl1_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Wtemp1_lvl1_part2_1_lvl1_part3T[D0,D1,D2,D3]
DistTensor<double> Wtemp1_lvl1_part2_1_lvl1_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Wtemp1_lvl1_part2_1_lvl2_part3B[D0,D1,D2,D3]
DistTensor<double> Wtemp1_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Wtemp1_lvl1_part2_1_lvl2_part3T[D0,D1,D2,D3]
DistTensor<double> Wtemp1_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Wtemp1_lvl1_part2_1_lvl2_part3_0[D0,D1,D2,D3]
DistTensor<double> Wtemp1_lvl1_part2_1_lvl2_part3_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Wtemp1_lvl1_part2_1_lvl2_part3_1[D0,D1,D2,D3]
DistTensor<double> Wtemp1_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Wtemp1_lvl1_part2_1_lvl2_part3_2[D0,D1,D2,D3]
DistTensor<double> Wtemp1_lvl1_part2_1_lvl2_part3_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Wtemp1_lvl1_part2_2[D0,D1,D2,D3]
DistTensor<double> Wtemp1_lvl1_part2_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Wtemp2[D0,D1,D2,D3]
DistTensor<double> Wtemp2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Wtemp2_lvl0_part2B[D0,D1,D2,D3]
DistTensor<double> Wtemp2_lvl0_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Wtemp2_lvl0_part2T[D0,D1,D2,D3]
DistTensor<double> Wtemp2_lvl0_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Wtemp2_lvl1_part2B[D0,D1,D2,D3]
DistTensor<double> Wtemp2_lvl1_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Wtemp2_lvl1_part2T[D0,D1,D2,D3]
DistTensor<double> Wtemp2_lvl1_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Wtemp2_lvl1_part2_0[D0,D1,D2,D3]
DistTensor<double> Wtemp2_lvl1_part2_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Wtemp2_lvl1_part2_1[D0,D1,D2,D3]
DistTensor<double> Wtemp2_lvl1_part2_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Wtemp2_lvl1_part2_1_lvl1_part3B[D0,D1,D2,D3]
DistTensor<double> Wtemp2_lvl1_part2_1_lvl1_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Wtemp2_lvl1_part2_1_lvl1_part3T[D0,D1,D2,D3]
DistTensor<double> Wtemp2_lvl1_part2_1_lvl1_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Wtemp2_lvl1_part2_1_lvl2_part3B[D0,D1,D2,D3]
DistTensor<double> Wtemp2_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Wtemp2_lvl1_part2_1_lvl2_part3T[D0,D1,D2,D3]
DistTensor<double> Wtemp2_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Wtemp2_lvl1_part2_1_lvl2_part3_0[D0,D1,D2,D3]
DistTensor<double> Wtemp2_lvl1_part2_1_lvl2_part3_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Wtemp2_lvl1_part2_1_lvl2_part3_1[D0,D1,D2,D3]
DistTensor<double> Wtemp2_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Wtemp2_lvl1_part2_1_lvl2_part3_2[D0,D1,D2,D3]
DistTensor<double> Wtemp2_lvl1_part2_1_lvl2_part3_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Wtemp2_lvl1_part2_2[D0,D1,D2,D3]
DistTensor<double> Wtemp2_lvl1_part2_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> Wtemp2_perm1203__D_1__D_2__D_0__D_3( dist__D_0__D_1__D_2__D_3, g );
Wtemp2_perm1203__D_1__D_2__D_0__D_3.SetLocalPermutation( perm_1_2_0_3 );
	//Wtemp3[D0,D1,D2,D3]
DistTensor<double> Wtemp3__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Wtemp3_lvl0_part0B[D0,D1,D2,D3]
DistTensor<double> Wtemp3_lvl0_part0B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Wtemp3_lvl0_part0T[D0,D1,D2,D3]
DistTensor<double> Wtemp3_lvl0_part0T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Wtemp3_lvl1_part0B[D0,D1,D2,D3]
DistTensor<double> Wtemp3_lvl1_part0B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Wtemp3_lvl1_part0T[D0,D1,D2,D3]
DistTensor<double> Wtemp3_lvl1_part0T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Wtemp3_lvl1_part0_0[D0,D1,D2,D3]
DistTensor<double> Wtemp3_lvl1_part0_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Wtemp3_lvl1_part0_1[D0,D1,D2,D3]
DistTensor<double> Wtemp3_lvl1_part0_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Wtemp3_lvl1_part0_1_lvl1_part1B[D0,D1,D2,D3]
DistTensor<double> Wtemp3_lvl1_part0_1_lvl1_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Wtemp3_lvl1_part0_1_lvl1_part1T[D0,D1,D2,D3]
DistTensor<double> Wtemp3_lvl1_part0_1_lvl1_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Wtemp3_lvl1_part0_1_lvl2_part1B[D0,D1,D2,D3]
DistTensor<double> Wtemp3_lvl1_part0_1_lvl2_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Wtemp3_lvl1_part0_1_lvl2_part1T[D0,D1,D2,D3]
DistTensor<double> Wtemp3_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Wtemp3_lvl1_part0_1_lvl2_part1_0[D0,D1,D2,D3]
DistTensor<double> Wtemp3_lvl1_part0_1_lvl2_part1_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Wtemp3_lvl1_part0_1_lvl2_part1_1[D0,D1,D2,D3]
DistTensor<double> Wtemp3_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Wtemp3_lvl1_part0_1_lvl2_part1_1[D1,*,D2,D3]
DistTensor<double> Wtemp3_lvl1_part0_1_lvl2_part1_1_perm0231__D_1__D_2__D_3__S( dist__D_1__S__D_2__D_3, g );
Wtemp3_lvl1_part0_1_lvl2_part1_1_perm0231__D_1__D_2__D_3__S.SetLocalPermutation( perm_0_2_3_1 );
	//Wtemp3_lvl1_part0_1_lvl2_part1_2[D0,D1,D2,D3]
DistTensor<double> Wtemp3_lvl1_part0_1_lvl2_part1_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Wtemp3_lvl1_part0_2[D0,D1,D2,D3]
DistTensor<double> Wtemp3_lvl1_part0_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Wtemp4[D0,D1,D2,D3]
DistTensor<double> Wtemp4__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Wtemp4_lvl0_part0B[D0,D1,D2,D3]
DistTensor<double> Wtemp4_lvl0_part0B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Wtemp4_lvl0_part0T[D0,D1,D2,D3]
DistTensor<double> Wtemp4_lvl0_part0T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Wtemp4_lvl0_part1B[D0,D1,D2,D3]
DistTensor<double> Wtemp4_lvl0_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Wtemp4_lvl0_part1T[D0,D1,D2,D3]
DistTensor<double> Wtemp4_lvl0_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Wtemp4_lvl1_part0B[D0,D1,D2,D3]
DistTensor<double> Wtemp4_lvl1_part0B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Wtemp4_lvl1_part0T[D0,D1,D2,D3]
DistTensor<double> Wtemp4_lvl1_part0T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Wtemp4_lvl1_part0_0[D0,D1,D2,D3]
DistTensor<double> Wtemp4_lvl1_part0_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Wtemp4_lvl1_part0_1[D0,D1,D2,D3]
DistTensor<double> Wtemp4_lvl1_part0_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Wtemp4_lvl1_part0_1_lvl1_part1B[D0,D1,D2,D3]
DistTensor<double> Wtemp4_lvl1_part0_1_lvl1_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Wtemp4_lvl1_part0_1_lvl1_part1T[D0,D1,D2,D3]
DistTensor<double> Wtemp4_lvl1_part0_1_lvl1_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Wtemp4_lvl1_part0_1_lvl2_part1B[D0,D1,D2,D3]
DistTensor<double> Wtemp4_lvl1_part0_1_lvl2_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Wtemp4_lvl1_part0_1_lvl2_part1T[D0,D1,D2,D3]
DistTensor<double> Wtemp4_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Wtemp4_lvl1_part0_1_lvl2_part1_0[D0,D1,D2,D3]
DistTensor<double> Wtemp4_lvl1_part0_1_lvl2_part1_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Wtemp4_lvl1_part0_1_lvl2_part1_1[D0,D1,D2,D3]
DistTensor<double> Wtemp4_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Wtemp4_lvl1_part0_1_lvl2_part1_2[D0,D1,D2,D3]
DistTensor<double> Wtemp4_lvl1_part0_1_lvl2_part1_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Wtemp4_lvl1_part0_2[D0,D1,D2,D3]
DistTensor<double> Wtemp4_lvl1_part0_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Wtemp4_lvl1_part1B[D0,D1,D2,D3]
DistTensor<double> Wtemp4_lvl1_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Wtemp4_lvl1_part1T[D0,D1,D2,D3]
DistTensor<double> Wtemp4_lvl1_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Wtemp4_lvl1_part1_0[D0,D1,D2,D3]
DistTensor<double> Wtemp4_lvl1_part1_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Wtemp4_lvl1_part1_1[D0,D1,D2,D3]
DistTensor<double> Wtemp4_lvl1_part1_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Wtemp4_lvl1_part1_2[D0,D1,D2,D3]
DistTensor<double> Wtemp4_lvl1_part1_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
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
	//t_fj_lvl2_part1_1[D013,D2]
DistTensor<double> t_fj_lvl2_part1_1__D_0_1_3__D_2( dist__D_0_1_3__D_2, g );
	//t_fj_lvl2_part1_1[D01,D2]
DistTensor<double> t_fj_lvl2_part1_1__D_0_1__D_2( dist__D_0_1__D_2, g );
	//t_fj_lvl2_part1_1[D301,D2]
DistTensor<double> t_fj_lvl2_part1_1__D_3_0_1__D_2( dist__D_3_0_1__D_2, g );
	//t_fj_lvl2_part1_1[D3,*]
DistTensor<double> t_fj_lvl2_part1_1__D_3__S( dist__D_3__S, g );
	//t_fj_lvl2_part1_1[D0,*]
DistTensor<double> t_fj_lvl2_part1_1_perm10__S__D_0( dist__D_0__S, g );
t_fj_lvl2_part1_1_perm10__S__D_0.SetLocalPermutation( perm_1_0 );
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
// r_bmfe has 4 dims
ObjShape r_bmfe__D_0__D_1__D_2__D_3_tempShape( 4 );
r_bmfe__D_0__D_1__D_2__D_3_tempShape[ 0 ] = n_v;
r_bmfe__D_0__D_1__D_2__D_3_tempShape[ 1 ] = n_o;
r_bmfe__D_0__D_1__D_2__D_3_tempShape[ 2 ] = n_v;
r_bmfe__D_0__D_1__D_2__D_3_tempShape[ 3 ] = n_v;
r_bmfe__D_0__D_1__D_2__D_3.ResizeTo( r_bmfe__D_0__D_1__D_2__D_3_tempShape );
MakeUniform( r_bmfe__D_0__D_1__D_2__D_3 );
// u_mnje has 4 dims
ObjShape u_mnje__D_0__D_1__D_2__D_3_tempShape( 4 );
u_mnje__D_0__D_1__D_2__D_3_tempShape[ 0 ] = n_o;
u_mnje__D_0__D_1__D_2__D_3_tempShape[ 1 ] = n_o;
u_mnje__D_0__D_1__D_2__D_3_tempShape[ 2 ] = n_o;
u_mnje__D_0__D_1__D_2__D_3_tempShape[ 3 ] = n_v;
u_mnje__D_0__D_1__D_2__D_3.ResizeTo( u_mnje__D_0__D_1__D_2__D_3_tempShape );
MakeUniform( u_mnje__D_0__D_1__D_2__D_3 );
// v_femn has 4 dims
ObjShape v_femn__D_0__D_1__D_2__D_3_tempShape( 4 );
v_femn__D_0__D_1__D_2__D_3_tempShape[ 0 ] = n_v;
v_femn__D_0__D_1__D_2__D_3_tempShape[ 1 ] = n_v;
v_femn__D_0__D_1__D_2__D_3_tempShape[ 2 ] = n_o;
v_femn__D_0__D_1__D_2__D_3_tempShape[ 3 ] = n_o;
v_femn__D_0__D_1__D_2__D_3.ResizeTo( v_femn__D_0__D_1__D_2__D_3_tempShape );
MakeUniform( v_femn__D_0__D_1__D_2__D_3 );
// w_bmje has 4 dims
ObjShape w_bmje__D_0__D_1__D_2__D_3_tempShape( 4 );
w_bmje__D_0__D_1__D_2__D_3_tempShape[ 0 ] = n_v;
w_bmje__D_0__D_1__D_2__D_3_tempShape[ 1 ] = n_o;
w_bmje__D_0__D_1__D_2__D_3_tempShape[ 2 ] = n_o;
w_bmje__D_0__D_1__D_2__D_3_tempShape[ 3 ] = n_v;
w_bmje__D_0__D_1__D_2__D_3.ResizeTo( w_bmje__D_0__D_1__D_2__D_3_tempShape );
MakeUniform( w_bmje__D_0__D_1__D_2__D_3 );
// x_bmej has 4 dims
ObjShape x_bmej__D_0__D_1__D_2__D_3_tempShape( 4 );
x_bmej__D_0__D_1__D_2__D_3_tempShape[ 0 ] = n_v;
x_bmej__D_0__D_1__D_2__D_3_tempShape[ 1 ] = n_o;
x_bmej__D_0__D_1__D_2__D_3_tempShape[ 2 ] = n_v;
x_bmej__D_0__D_1__D_2__D_3_tempShape[ 3 ] = n_o;
x_bmej__D_0__D_1__D_2__D_3.ResizeTo( x_bmej__D_0__D_1__D_2__D_3_tempShape );
MakeUniform( x_bmej__D_0__D_1__D_2__D_3 );
// W_bmje has 4 dims
ObjShape W_bmje__D_0__D_1__D_2__D_3_tempShape( 4 );
W_bmje__D_0__D_1__D_2__D_3_tempShape[ 0 ] = n_v;
W_bmje__D_0__D_1__D_2__D_3_tempShape[ 1 ] = n_o;
W_bmje__D_0__D_1__D_2__D_3_tempShape[ 2 ] = n_o;
W_bmje__D_0__D_1__D_2__D_3_tempShape[ 3 ] = n_v;
W_bmje__D_0__D_1__D_2__D_3.ResizeTo( W_bmje__D_0__D_1__D_2__D_3_tempShape );
MakeUniform( W_bmje__D_0__D_1__D_2__D_3 );
TensorDistribution dist__S__D_2__D_1__S__D_0__D_3 = tmen::StringToTensorDist("[(),(2),(1),(),(0),(3)]");
TensorDistribution dist__D_0__D_1__D_2__S__D_3 = tmen::StringToTensorDist("[(0),(1),(2),(),(3)]");
TensorDistribution dist__D_0__D_2__D_1__D_3 = tmen::StringToTensorDist("[(0),(2),(1),(3)]");
TensorDistribution dist__D_1__S__D_3__D_2 = tmen::StringToTensorDist("[(1),(),(3),(2)]");
Permutation perm_0_1_2_3_4( 5 );
perm_0_1_2_3_4[0] = 0;
perm_0_1_2_3_4[1] = 1;
perm_0_1_2_3_4[2] = 2;
perm_0_1_2_3_4[3] = 3;
perm_0_1_2_3_4[4] = 4;
Permutation perm_1_3_2_0( 4 );
perm_1_3_2_0[0] = 1;
perm_1_3_2_0[1] = 3;
perm_1_3_2_0[2] = 2;
perm_1_3_2_0[3] = 0;
Permutation perm_2_1_0_3_4_5( 6 );
perm_2_1_0_3_4_5[0] = 2;
perm_2_1_0_3_4_5[1] = 1;
perm_2_1_0_3_4_5[2] = 0;
perm_2_1_0_3_4_5[3] = 3;
perm_2_1_0_3_4_5[4] = 4;
perm_2_1_0_3_4_5[5] = 5;
Permutation perm_3_0_2_1( 4 );
perm_3_0_2_1[0] = 3;
perm_3_0_2_1[1] = 0;
perm_3_0_2_1[2] = 2;
perm_3_0_2_1[3] = 1;
ModeArray modes_1_3_2( 3 );
modes_1_3_2[0] = 1;
modes_1_3_2[1] = 3;
modes_1_3_2[2] = 2;
	//X_bmej[D0,D1,D2,D3]
DistTensor<double> X_bmej__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//X_bmej_lvl0_part0B[D0,D1,D2,D3]
DistTensor<double> X_bmej_lvl0_part0B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//X_bmej_lvl0_part0T[D0,D1,D2,D3]
DistTensor<double> X_bmej_lvl0_part0T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//X_bmej_lvl0_part1B[D0,D1,D2,D3]
DistTensor<double> X_bmej_lvl0_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//X_bmej_lvl0_part1T[D0,D1,D2,D3]
DistTensor<double> X_bmej_lvl0_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//X_bmej_lvl1_part0B[D0,D1,D2,D3]
DistTensor<double> X_bmej_lvl1_part0B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//X_bmej_lvl1_part0T[D0,D1,D2,D3]
DistTensor<double> X_bmej_lvl1_part0T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//X_bmej_lvl1_part0_0[D0,D1,D2,D3]
DistTensor<double> X_bmej_lvl1_part0_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//X_bmej_lvl1_part0_1[D0,D1,D2,D3]
DistTensor<double> X_bmej_lvl1_part0_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//X_bmej_lvl1_part0_1_lvl1_part3B[D0,D1,D2,D3]
DistTensor<double> X_bmej_lvl1_part0_1_lvl1_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//X_bmej_lvl1_part0_1_lvl1_part3T[D0,D1,D2,D3]
DistTensor<double> X_bmej_lvl1_part0_1_lvl1_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//X_bmej_lvl1_part0_1_lvl2_part3B[D0,D1,D2,D3]
DistTensor<double> X_bmej_lvl1_part0_1_lvl2_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//X_bmej_lvl1_part0_1_lvl2_part3T[D0,D1,D2,D3]
DistTensor<double> X_bmej_lvl1_part0_1_lvl2_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//X_bmej_lvl1_part0_1_lvl2_part3_0[D0,D1,D2,D3]
DistTensor<double> X_bmej_lvl1_part0_1_lvl2_part3_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//X_bmej_lvl1_part0_1_lvl2_part3_1[D0,D1,D2,D3]
DistTensor<double> X_bmej_lvl1_part0_1_lvl2_part3_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//X_bmej_lvl1_part0_1_lvl2_part3_1[*,D2,D1,*,D0,D3]
DistTensor<double> X_bmej_lvl1_part0_1_lvl2_part3_1_perm210345__D_1__D_2__S__S__D_0__D_3( dist__S__D_2__D_1__S__D_0__D_3, g );
X_bmej_lvl1_part0_1_lvl2_part3_1_perm210345__D_1__D_2__S__S__D_0__D_3.SetLocalPermutation( perm_2_1_0_3_4_5 );
	//X_bmej_lvl1_part0_1_lvl2_part3_1_temp[D0,D1,D2,D3]
DistTensor<double> X_bmej_lvl1_part0_1_lvl2_part3_1_temp__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//X_bmej_lvl1_part0_1_lvl2_part3_1_temp[D0,D2,D1,D3]
DistTensor<double> X_bmej_lvl1_part0_1_lvl2_part3_1_temp__D_0__D_2__D_1__D_3( dist__D_0__D_2__D_1__D_3, g );
	//X_bmej_lvl1_part0_1_lvl2_part3_2[D0,D1,D2,D3]
DistTensor<double> X_bmej_lvl1_part0_1_lvl2_part3_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//X_bmej_lvl1_part0_2[D0,D1,D2,D3]
DistTensor<double> X_bmej_lvl1_part0_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//X_bmej_lvl1_part1B[D0,D1,D2,D3]
DistTensor<double> X_bmej_lvl1_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//X_bmej_lvl1_part1T[D0,D1,D2,D3]
DistTensor<double> X_bmej_lvl1_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//X_bmej_lvl1_part1_0[D0,D1,D2,D3]
DistTensor<double> X_bmej_lvl1_part1_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//X_bmej_lvl1_part1_1[D0,D1,D2,D3]
DistTensor<double> X_bmej_lvl1_part1_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//X_bmej_lvl1_part1_1_lvl1_part3B[D0,D1,D2,D3]
DistTensor<double> X_bmej_lvl1_part1_1_lvl1_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//X_bmej_lvl1_part1_1_lvl1_part3T[D0,D1,D2,D3]
DistTensor<double> X_bmej_lvl1_part1_1_lvl1_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//X_bmej_lvl1_part1_1_lvl2_part3B[D0,D1,D2,D3]
DistTensor<double> X_bmej_lvl1_part1_1_lvl2_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//X_bmej_lvl1_part1_1_lvl2_part3T[D0,D1,D2,D3]
DistTensor<double> X_bmej_lvl1_part1_1_lvl2_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//X_bmej_lvl1_part1_1_lvl2_part3_0[D0,D1,D2,D3]
DistTensor<double> X_bmej_lvl1_part1_1_lvl2_part3_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//X_bmej_lvl1_part1_1_lvl2_part3_1[D0,D1,D2,D3]
DistTensor<double> X_bmej_lvl1_part1_1_lvl2_part3_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//X_bmej_lvl1_part1_1_lvl2_part3_1[D0,D1,D2,*,D3]
DistTensor<double> X_bmej_lvl1_part1_1_lvl2_part3_1__D_0__D_1__D_2__S__D_3( dist__D_0__D_1__D_2__S__D_3, g );
	//X_bmej_lvl1_part1_1_lvl2_part3_2[D0,D1,D2,D3]
DistTensor<double> X_bmej_lvl1_part1_1_lvl2_part3_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> X_bmej_lvl1_part1_1_perm1320__D_1__D_3__D_2__D_0( dist__D_0__D_1__D_2__D_3, g );
X_bmej_lvl1_part1_1_perm1320__D_1__D_3__D_2__D_0.SetLocalPermutation( perm_1_3_2_0 );
	//X_bmej_lvl1_part1_2[D0,D1,D2,D3]
DistTensor<double> X_bmej_lvl1_part1_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Xtemp1[D0,D1,D2,D3]
DistTensor<double> Xtemp1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Xtemp1_lvl0_part0B[D0,D1,D2,D3]
DistTensor<double> Xtemp1_lvl0_part0B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Xtemp1_lvl0_part0T[D0,D1,D2,D3]
DistTensor<double> Xtemp1_lvl0_part0T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Xtemp1_lvl1_part0B[D0,D1,D2,D3]
DistTensor<double> Xtemp1_lvl1_part0B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Xtemp1_lvl1_part0T[D0,D1,D2,D3]
DistTensor<double> Xtemp1_lvl1_part0T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Xtemp1_lvl1_part0_0[D0,D1,D2,D3]
DistTensor<double> Xtemp1_lvl1_part0_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Xtemp1_lvl1_part0_1[D0,D1,D2,D3]
DistTensor<double> Xtemp1_lvl1_part0_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Xtemp1_lvl1_part0_1_lvl1_part3B[D0,D1,D2,D3]
DistTensor<double> Xtemp1_lvl1_part0_1_lvl1_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Xtemp1_lvl1_part0_1_lvl1_part3T[D0,D1,D2,D3]
DistTensor<double> Xtemp1_lvl1_part0_1_lvl1_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Xtemp1_lvl1_part0_1_lvl2_part3B[D0,D1,D2,D3]
DistTensor<double> Xtemp1_lvl1_part0_1_lvl2_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Xtemp1_lvl1_part0_1_lvl2_part3T[D0,D1,D2,D3]
DistTensor<double> Xtemp1_lvl1_part0_1_lvl2_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Xtemp1_lvl1_part0_1_lvl2_part3_0[D0,D1,D2,D3]
DistTensor<double> Xtemp1_lvl1_part0_1_lvl2_part3_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Xtemp1_lvl1_part0_1_lvl2_part3_1[D0,D1,D2,D3]
DistTensor<double> Xtemp1_lvl1_part0_1_lvl2_part3_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Xtemp1_lvl1_part0_1_lvl2_part3_1[D1,D0,D2,D3]
DistTensor<double> Xtemp1_lvl1_part0_1_lvl2_part3_1__D_1__D_0__D_2__D_3( dist__D_1__D_0__D_2__D_3, g );
	//Xtemp1_lvl1_part0_1_lvl2_part3_1[D1,D0,D3,*]
DistTensor<double> Xtemp1_lvl1_part0_1_lvl2_part3_1__D_1__D_0__D_3__S( dist__D_1__D_0__D_3__S, g );
	//Xtemp1_lvl1_part0_1_lvl2_part3_1[*,D0,D3,*]
DistTensor<double> Xtemp1_lvl1_part0_1_lvl2_part3_1_perm1203__D_0__D_3__S__S( dist__S__D_0__D_3__S, g );
Xtemp1_lvl1_part0_1_lvl2_part3_1_perm1203__D_0__D_3__S__S.SetLocalPermutation( perm_1_2_0_3 );
	//Xtemp1_lvl1_part0_1_lvl2_part3_2[D0,D1,D2,D3]
DistTensor<double> Xtemp1_lvl1_part0_1_lvl2_part3_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Xtemp1_lvl1_part0_2[D0,D1,D2,D3]
DistTensor<double> Xtemp1_lvl1_part0_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//r_bmfe_lvl0_part1B[D0,D1,D2,D3]
DistTensor<double> r_bmfe_lvl0_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//r_bmfe_lvl0_part1T[D0,D1,D2,D3]
DistTensor<double> r_bmfe_lvl0_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//r_bmfe_lvl1_part1B[D0,D1,D2,D3]
DistTensor<double> r_bmfe_lvl1_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//r_bmfe_lvl1_part1T[D0,D1,D2,D3]
DistTensor<double> r_bmfe_lvl1_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//r_bmfe_lvl1_part1_0[D0,D1,D2,D3]
DistTensor<double> r_bmfe_lvl1_part1_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//r_bmfe_lvl1_part1_1[D0,D1,D2,D3]
DistTensor<double> r_bmfe_lvl1_part1_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//r_bmfe_lvl1_part1_2[D0,D1,D2,D3]
DistTensor<double> r_bmfe_lvl1_part1_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//u_mnje_lvl1_part0_1_lvl2_part1_1[D0,D1,D3,D2]
DistTensor<double> u_mnje_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_3__D_2( dist__D_0__D_1__D_3__D_2, g );
	//u_mnje_lvl1_part0_1_lvl2_part1_1[D1,*,D3,D2]
DistTensor<double> u_mnje_lvl1_part0_1_lvl2_part1_1_perm0231__D_1__D_3__D_2__S( dist__D_1__S__D_3__D_2, g );
u_mnje_lvl1_part0_1_lvl2_part1_1_perm0231__D_1__D_3__D_2__S.SetLocalPermutation( perm_0_2_3_1 );
DistTensor<double> v_femn_perm1203__D_1__D_2__D_0__D_3( dist__D_0__D_1__D_2__D_3, g );
v_femn_perm1203__D_1__D_2__D_0__D_3.SetLocalPermutation( perm_1_2_0_3 );
// X_bmej has 4 dims
ObjShape X_bmej__D_0__D_1__D_2__D_3_tempShape( 4 );
X_bmej__D_0__D_1__D_2__D_3_tempShape[ 0 ] = n_v;
X_bmej__D_0__D_1__D_2__D_3_tempShape[ 1 ] = n_o;
X_bmej__D_0__D_1__D_2__D_3_tempShape[ 2 ] = n_v;
X_bmej__D_0__D_1__D_2__D_3_tempShape[ 3 ] = n_o;
X_bmej__D_0__D_1__D_2__D_3.ResizeTo( X_bmej__D_0__D_1__D_2__D_3_tempShape );
MakeUniform( X_bmej__D_0__D_1__D_2__D_3 );
TensorDistribution dist__D_2_0__D_1__S__D_3 = tmen::StringToTensorDist("[(2,0),(1),(),(3)]");
TensorDistribution dist__D_2_0__D_3__S__D_1 = tmen::StringToTensorDist("[(2,0),(3),(),(1)]");
TensorDistribution dist__D_2__D_3__S__D_1__D_0 = tmen::StringToTensorDist("[(2),(3),(),(1),(0)]");
Permutation perm_3_0_1_2_4( 5 );
perm_3_0_1_2_4[0] = 3;
perm_3_0_1_2_4[1] = 0;
perm_3_0_1_2_4[2] = 1;
perm_3_0_1_2_4[3] = 2;
perm_3_0_1_2_4[4] = 4;
ModeArray modes_2_3_1( 3 );
modes_2_3_1[0] = 2;
modes_2_3_1[1] = 3;
modes_2_3_1[2] = 1;
IndexArray indices_emnf( 4 );
indices_emnf[0] = 'e';
indices_emnf[1] = 'm';
indices_emnf[2] = 'n';
indices_emnf[3] = 'f';
IndexArray indices_emnif( 5 );
indices_emnif[0] = 'e';
indices_emnif[1] = 'm';
indices_emnif[2] = 'n';
indices_emnif[3] = 'i';
indices_emnif[4] = 'f';
IndexArray indices_fi( 2 );
indices_fi[0] = 'f';
indices_fi[1] = 'i';
	//U_mnie[D0,D1,D2,D3]
DistTensor<double> U_mnie__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//U_mnie_lvl0_part0B[D0,D1,D2,D3]
DistTensor<double> U_mnie_lvl0_part0B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//U_mnie_lvl0_part0T[D0,D1,D2,D3]
DistTensor<double> U_mnie_lvl0_part0T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//U_mnie_lvl1_part0B[D0,D1,D2,D3]
DistTensor<double> U_mnie_lvl1_part0B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//U_mnie_lvl1_part0T[D0,D1,D2,D3]
DistTensor<double> U_mnie_lvl1_part0T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//U_mnie_lvl1_part0_0[D0,D1,D2,D3]
DistTensor<double> U_mnie_lvl1_part0_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//U_mnie_lvl1_part0_1[D0,D1,D2,D3]
DistTensor<double> U_mnie_lvl1_part0_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//U_mnie_lvl1_part0_1_lvl1_part2B[D0,D1,D2,D3]
DistTensor<double> U_mnie_lvl1_part0_1_lvl1_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//U_mnie_lvl1_part0_1_lvl1_part2T[D0,D1,D2,D3]
DistTensor<double> U_mnie_lvl1_part0_1_lvl1_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//U_mnie_lvl1_part0_1_lvl2_part2B[D0,D1,D2,D3]
DistTensor<double> U_mnie_lvl1_part0_1_lvl2_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//U_mnie_lvl1_part0_1_lvl2_part2T[D0,D1,D2,D3]
DistTensor<double> U_mnie_lvl1_part0_1_lvl2_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//U_mnie_lvl1_part0_1_lvl2_part2_0[D0,D1,D2,D3]
DistTensor<double> U_mnie_lvl1_part0_1_lvl2_part2_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//U_mnie_lvl1_part0_1_lvl2_part2_1[D0,D1,D2,D3]
DistTensor<double> U_mnie_lvl1_part0_1_lvl2_part2_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//U_mnie_lvl1_part0_1_lvl2_part2_1[D2,D3,*,D1,D0]
DistTensor<double> U_mnie_lvl1_part0_1_lvl2_part2_1_perm30124__D_1__D_2__D_3__S__D_0( dist__D_2__D_3__S__D_1__D_0, g );
U_mnie_lvl1_part0_1_lvl2_part2_1_perm30124__D_1__D_2__D_3__S__D_0.SetLocalPermutation( perm_3_0_1_2_4 );
	//U_mnie_lvl1_part0_1_lvl2_part2_1_temp[D0,D1,D2,D3]
DistTensor<double> U_mnie_lvl1_part0_1_lvl2_part2_1_temp__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//U_mnie_lvl1_part0_1_lvl2_part2_1_temp[D20,D1,*,D3]
DistTensor<double> U_mnie_lvl1_part0_1_lvl2_part2_1_temp__D_2_0__D_1__S__D_3( dist__D_2_0__D_1__S__D_3, g );
	//U_mnie_lvl1_part0_1_lvl2_part2_1_temp[D20,D3,*,D1]
DistTensor<double> U_mnie_lvl1_part0_1_lvl2_part2_1_temp__D_2_0__D_3__S__D_1( dist__D_2_0__D_3__S__D_1, g );
	//U_mnie_lvl1_part0_1_lvl2_part2_2[D0,D1,D2,D3]
DistTensor<double> U_mnie_lvl1_part0_1_lvl2_part2_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//U_mnie_lvl1_part0_2[D0,D1,D2,D3]
DistTensor<double> U_mnie_lvl1_part0_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> t_fj_lvl2_part1_1__D_0__S( dist__D_0__S, g );
DistTensor<double> v_femn_lvl1_part2_1_perm1230__D_1__D_2__D_3__D_0( dist__D_0__D_1__D_2__D_3, g );
v_femn_lvl1_part2_1_perm1230__D_1__D_2__D_3__D_0.SetLocalPermutation( perm_1_2_3_0 );
// U_mnie has 4 dims
ObjShape U_mnie__D_0__D_1__D_2__D_3_tempShape( 4 );
U_mnie__D_0__D_1__D_2__D_3_tempShape[ 0 ] = n_o;
U_mnie__D_0__D_1__D_2__D_3_tempShape[ 1 ] = n_o;
U_mnie__D_0__D_1__D_2__D_3_tempShape[ 2 ] = n_o;
U_mnie__D_0__D_1__D_2__D_3_tempShape[ 3 ] = n_v;
U_mnie__D_0__D_1__D_2__D_3.ResizeTo( U_mnie__D_0__D_1__D_2__D_3_tempShape );
MakeUniform( U_mnie__D_0__D_1__D_2__D_3 );
TensorDistribution dist__S__S__D_2__D_3__D_0__D_1 = tmen::StringToTensorDist("[(),(),(2),(3),(0),(1)]");
TensorDistribution dist__D_0__D_1__S__S = tmen::StringToTensorDist("[(0),(1),(),()]");
TensorDistribution dist__D_1__D_0__D_3__D_2 = tmen::StringToTensorDist("[(1),(0),(3),(2)]");
Permutation perm_0_1_2_3_4_5( 6 );
perm_0_1_2_3_4_5[0] = 0;
perm_0_1_2_3_4_5[1] = 1;
perm_0_1_2_3_4_5[2] = 2;
perm_0_1_2_3_4_5[3] = 3;
perm_0_1_2_3_4_5[4] = 4;
perm_0_1_2_3_4_5[5] = 5;
Permutation perm_1_0_3_2( 4 );
perm_1_0_3_2[0] = 1;
perm_1_0_3_2[1] = 0;
perm_1_0_3_2[2] = 3;
perm_1_0_3_2[3] = 2;
Permutation perm_2_3_0_1( 4 );
perm_2_3_0_1[0] = 2;
perm_2_3_0_1[1] = 3;
perm_2_3_0_1[2] = 0;
perm_2_3_0_1[3] = 1;
ModeArray modes_1_0_3_2( 4 );
modes_1_0_3_2[0] = 1;
modes_1_0_3_2[1] = 0;
modes_1_0_3_2[2] = 3;
modes_1_0_3_2[3] = 2;
IndexArray indices_efij( 4 );
indices_efij[0] = 'e';
indices_efij[1] = 'f';
indices_efij[2] = 'i';
indices_efij[3] = 'j';
IndexArray indices_ej( 2 );
indices_ej[0] = 'e';
indices_ej[1] = 'j';
IndexArray indices_mnef( 4 );
indices_mnef[0] = 'm';
indices_mnef[1] = 'n';
indices_mnef[2] = 'e';
indices_mnef[3] = 'f';
IndexArray indices_mnie( 4 );
indices_mnie[0] = 'm';
indices_mnie[1] = 'n';
indices_mnie[2] = 'i';
indices_mnie[3] = 'e';
IndexArray indices_mnije( 5 );
indices_mnije[0] = 'm';
indices_mnije[1] = 'n';
indices_mnije[2] = 'i';
indices_mnije[3] = 'j';
indices_mnije[4] = 'e';
IndexArray indices_mnijef( 6 );
indices_mnijef[0] = 'm';
indices_mnijef[1] = 'n';
indices_mnijef[2] = 'i';
indices_mnijef[3] = 'j';
indices_mnijef[4] = 'e';
indices_mnijef[5] = 'f';
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
	//Q_mnij_lvl1_part0_1_lvl2_part1_1[*,*,D2,D3,D0,D1]
DistTensor<double> Q_mnij_lvl1_part0_1_lvl2_part1_1__S__S__D_2__D_3__D_0__D_1( dist__S__S__D_2__D_3__D_0__D_1, g );
	//Q_mnij_lvl1_part0_1_lvl2_part1_2[D0,D1,D2,D3]
DistTensor<double> Q_mnij_lvl1_part0_1_lvl2_part1_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Q_mnij_lvl1_part0_2[D0,D1,D2,D3]
DistTensor<double> Q_mnij_lvl1_part0_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Qtemp1[D0,D1,D2,D3]
DistTensor<double> Qtemp1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Qtemp1_lvl0_part0B[D0,D1,D2,D3]
DistTensor<double> Qtemp1_lvl0_part0B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Qtemp1_lvl0_part0T[D0,D1,D2,D3]
DistTensor<double> Qtemp1_lvl0_part0T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Qtemp1_lvl0_part1B[D0,D1,D2,D3]
DistTensor<double> Qtemp1_lvl0_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Qtemp1_lvl0_part1T[D0,D1,D2,D3]
DistTensor<double> Qtemp1_lvl0_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Qtemp1_lvl1_part0B[D0,D1,D2,D3]
DistTensor<double> Qtemp1_lvl1_part0B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Qtemp1_lvl1_part0T[D0,D1,D2,D3]
DistTensor<double> Qtemp1_lvl1_part0T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Qtemp1_lvl1_part0_0[D0,D1,D2,D3]
DistTensor<double> Qtemp1_lvl1_part0_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Qtemp1_lvl1_part0_1[D0,D1,D2,D3]
DistTensor<double> Qtemp1_lvl1_part0_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Qtemp1_lvl1_part0_1_lvl1_part1B[D0,D1,D2,D3]
DistTensor<double> Qtemp1_lvl1_part0_1_lvl1_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Qtemp1_lvl1_part0_1_lvl1_part1T[D0,D1,D2,D3]
DistTensor<double> Qtemp1_lvl1_part0_1_lvl1_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Qtemp1_lvl1_part0_1_lvl2_part1B[D0,D1,D2,D3]
DistTensor<double> Qtemp1_lvl1_part0_1_lvl2_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Qtemp1_lvl1_part0_1_lvl2_part1T[D0,D1,D2,D3]
DistTensor<double> Qtemp1_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Qtemp1_lvl1_part0_1_lvl2_part1_0[D0,D1,D2,D3]
DistTensor<double> Qtemp1_lvl1_part0_1_lvl2_part1_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Qtemp1_lvl1_part0_1_lvl2_part1_1[D0,D1,D2,D3]
DistTensor<double> Qtemp1_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Qtemp1_lvl1_part0_1_lvl2_part1_2[D0,D1,D2,D3]
DistTensor<double> Qtemp1_lvl1_part0_1_lvl2_part1_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Qtemp1_lvl1_part0_2[D0,D1,D2,D3]
DistTensor<double> Qtemp1_lvl1_part0_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Qtemp1_lvl1_part1B[D0,D1,D2,D3]
DistTensor<double> Qtemp1_lvl1_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Qtemp1_lvl1_part1T[D0,D1,D2,D3]
DistTensor<double> Qtemp1_lvl1_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Qtemp1_lvl1_part1_0[D0,D1,D2,D3]
DistTensor<double> Qtemp1_lvl1_part1_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Qtemp1_lvl1_part1_1[D0,D1,D2,D3]
DistTensor<double> Qtemp1_lvl1_part1_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Qtemp1_lvl1_part1_1_lvl1_part0B[D0,D1,D2,D3]
DistTensor<double> Qtemp1_lvl1_part1_1_lvl1_part0B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Qtemp1_lvl1_part1_1_lvl1_part0T[D0,D1,D2,D3]
DistTensor<double> Qtemp1_lvl1_part1_1_lvl1_part0T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Qtemp1_lvl1_part1_1_lvl1_part3B[D0,D1,D2,D3]
DistTensor<double> Qtemp1_lvl1_part1_1_lvl1_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Qtemp1_lvl1_part1_1_lvl1_part3T[D0,D1,D2,D3]
DistTensor<double> Qtemp1_lvl1_part1_1_lvl1_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Qtemp1_lvl1_part1_1_lvl2_part0B[D0,D1,D2,D3]
DistTensor<double> Qtemp1_lvl1_part1_1_lvl2_part0B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Qtemp1_lvl1_part1_1_lvl2_part0T[D0,D1,D2,D3]
DistTensor<double> Qtemp1_lvl1_part1_1_lvl2_part0T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Qtemp1_lvl1_part1_1_lvl2_part0_0[D0,D1,D2,D3]
DistTensor<double> Qtemp1_lvl1_part1_1_lvl2_part0_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Qtemp1_lvl1_part1_1_lvl2_part0_1[D0,D1,D2,D3]
DistTensor<double> Qtemp1_lvl1_part1_1_lvl2_part0_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Qtemp1_lvl1_part1_1_lvl2_part0_1[D1,D0,D2,D3]
DistTensor<double> Qtemp1_lvl1_part1_1_lvl2_part0_1__D_1__D_0__D_2__D_3( dist__D_1__D_0__D_2__D_3, g );
	//Qtemp1_lvl1_part1_1_lvl2_part0_1[D1,D0,D3,D2]
DistTensor<double> Qtemp1_lvl1_part1_1_lvl2_part0_1__D_1__D_0__D_3__D_2( dist__D_1__D_0__D_3__D_2, g );
	//Qtemp1_lvl1_part1_1_lvl2_part0_2[D0,D1,D2,D3]
DistTensor<double> Qtemp1_lvl1_part1_1_lvl2_part0_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Qtemp1_lvl1_part1_1_lvl2_part3B[D0,D1,D2,D3]
DistTensor<double> Qtemp1_lvl1_part1_1_lvl2_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Qtemp1_lvl1_part1_1_lvl2_part3T[D0,D1,D2,D3]
DistTensor<double> Qtemp1_lvl1_part1_1_lvl2_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Qtemp1_lvl1_part1_1_lvl2_part3_0[D0,D1,D2,D3]
DistTensor<double> Qtemp1_lvl1_part1_1_lvl2_part3_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Qtemp1_lvl1_part1_1_lvl2_part3_1[D0,D1,D2,D3]
DistTensor<double> Qtemp1_lvl1_part1_1_lvl2_part3_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Qtemp1_lvl1_part1_1_lvl2_part3_1[D0,D1,D2,*,D3]
DistTensor<double> Qtemp1_lvl1_part1_1_lvl2_part3_1__D_0__D_1__D_2__S__D_3( dist__D_0__D_1__D_2__S__D_3, g );
	//Qtemp1_lvl1_part1_1_lvl2_part3_2[D0,D1,D2,D3]
DistTensor<double> Qtemp1_lvl1_part1_1_lvl2_part3_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Qtemp1_lvl1_part1_2[D0,D1,D2,D3]
DistTensor<double> Qtemp1_lvl1_part1_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//q_mnij[D0,D1,D2,D3]
DistTensor<double> q_mnij__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//v_femn_lvl1_part2_1_lvl2_part3_1[D0,D1,*,*]
DistTensor<double> v_femn_lvl1_part2_1_lvl2_part3_1_perm2301__S__S__D_0__D_1( dist__D_0__D_1__S__S, g );
v_femn_lvl1_part2_1_lvl2_part3_1_perm2301__S__S__D_0__D_1.SetLocalPermutation( perm_2_3_0_1 );
// q_mnij has 4 dims
ObjShape q_mnij__D_0__D_1__D_2__D_3_tempShape( 4 );
q_mnij__D_0__D_1__D_2__D_3_tempShape[ 0 ] = n_o;
q_mnij__D_0__D_1__D_2__D_3_tempShape[ 1 ] = n_o;
q_mnij__D_0__D_1__D_2__D_3_tempShape[ 2 ] = n_o;
q_mnij__D_0__D_1__D_2__D_3_tempShape[ 3 ] = n_o;
q_mnij__D_0__D_1__D_2__D_3.ResizeTo( q_mnij__D_0__D_1__D_2__D_3_tempShape );
MakeUniform( q_mnij__D_0__D_1__D_2__D_3 );
// Q_mnij has 4 dims
ObjShape Q_mnij__D_0__D_1__D_2__D_3_tempShape( 4 );
Q_mnij__D_0__D_1__D_2__D_3_tempShape[ 0 ] = n_o;
Q_mnij__D_0__D_1__D_2__D_3_tempShape[ 1 ] = n_o;
Q_mnij__D_0__D_1__D_2__D_3_tempShape[ 2 ] = n_o;
Q_mnij__D_0__D_1__D_2__D_3_tempShape[ 3 ] = n_o;
Q_mnij__D_0__D_1__D_2__D_3.ResizeTo( Q_mnij__D_0__D_1__D_2__D_3_tempShape );
MakeUniform( Q_mnij__D_0__D_1__D_2__D_3 );
TensorDistribution dist__S__S__D_1_2__D_0_3 = tmen::StringToTensorDist("[(),(),(1,2),(0,3)]");
TensorDistribution dist__S__S__D_2_1__D_0_3 = tmen::StringToTensorDist("[(),(),(2,1),(0,3)]");
TensorDistribution dist__S__S__D_1__D_0__D_2__D_3 = tmen::StringToTensorDist("[(),(),(1),(0),(2),(3)]");
TensorDistribution dist__S__D_1__D_2__D_0_3 = tmen::StringToTensorDist("[(),(1),(2),(0,3)]");
TensorDistribution dist__S__D_2__D_1__D_0__D_3 = tmen::StringToTensorDist("[(),(2),(1),(0),(3)]");
TensorDistribution dist__S__D_2__D_1__D_0_3 = tmen::StringToTensorDist("[(),(2),(1),(0,3)]");
TensorDistribution dist__D_0_1_2__S = tmen::StringToTensorDist("[(0,1,2),()]");
TensorDistribution dist__D_2__S = tmen::StringToTensorDist("[(2),()]");
TensorDistribution dist__D_2__D_1__D_0__D_3 = tmen::StringToTensorDist("[(2),(1),(0),(3)]");
TensorDistribution dist__D_2__D_3__S__S = tmen::StringToTensorDist("[(2),(3),(),()]");
TensorDistribution dist__D_2__D_3__D_0__S = tmen::StringToTensorDist("[(2),(3),(0),()]");
TensorDistribution dist__D_3__S__D_1_2__D_0 = tmen::StringToTensorDist("[(3),(),(1,2),(0)]");
TensorDistribution dist__D_3__S__D_2_1__D_0 = tmen::StringToTensorDist("[(3),(),(2,1),(0)]");
TensorDistribution dist__D_3__S__D_1__D_0__D_2 = tmen::StringToTensorDist("[(3),(),(1),(0),(2)]");
TensorDistribution dist__D_3__D_1__D_2__D_0 = tmen::StringToTensorDist("[(3),(1),(2),(0)]");
TensorDistribution dist__D_2_0_1__S = tmen::StringToTensorDist("[(2,0,1),()]");
TensorDistribution dist__D_0_1__S = tmen::StringToTensorDist("[(0,1),()]");
Permutation perm_3_2_0_1_4( 5 );
perm_3_2_0_1_4[0] = 3;
perm_3_2_0_1_4[1] = 2;
perm_3_2_0_1_4[2] = 0;
perm_3_2_0_1_4[3] = 1;
perm_3_2_0_1_4[4] = 4;
Permutation perm_3_2_1_0_4( 5 );
perm_3_2_1_0_4[0] = 3;
perm_3_2_1_0_4[1] = 2;
perm_3_2_1_0_4[2] = 1;
perm_3_2_1_0_4[3] = 0;
perm_3_2_1_0_4[4] = 4;
Permutation perm_3_2_1_0_4_5( 6 );
perm_3_2_1_0_4_5[0] = 3;
perm_3_2_1_0_4_5[1] = 2;
perm_3_2_1_0_4_5[2] = 1;
perm_3_2_1_0_4_5[3] = 0;
perm_3_2_1_0_4_5[4] = 4;
perm_3_2_1_0_4_5[5] = 5;
ModeArray modes_1_0( 2 );
modes_1_0[0] = 1;
modes_1_0[1] = 0;
ModeArray modes_2_1_0( 3 );
modes_2_1_0[0] = 2;
modes_2_1_0[1] = 1;
modes_2_1_0[2] = 0;
ModeArray modes_3_1_0( 3 );
modes_3_1_0[0] = 3;
modes_3_1_0[1] = 1;
modes_3_1_0[2] = 0;
IndexArray indices_bmie( 4 );
indices_bmie[0] = 'b';
indices_bmie[1] = 'm';
indices_bmie[2] = 'i';
indices_bmie[3] = 'e';
IndexArray indices_bmije( 5 );
indices_bmije[0] = 'b';
indices_bmije[1] = 'm';
indices_bmije[2] = 'i';
indices_bmije[3] = 'j';
indices_bmije[4] = 'e';
IndexArray indices_bmijef( 6 );
indices_bmijef[0] = 'b';
indices_bmijef[1] = 'm';
indices_bmijef[2] = 'i';
indices_bmijef[3] = 'j';
indices_bmijef[4] = 'e';
indices_bmijef[5] = 'f';
IndexArray indices_bmje( 4 );
indices_bmje[0] = 'b';
indices_bmje[1] = 'm';
indices_bmje[2] = 'j';
indices_bmje[3] = 'e';
IndexArray indices_bmjie( 5 );
indices_bmjie[0] = 'b';
indices_bmjie[1] = 'm';
indices_bmjie[2] = 'j';
indices_bmjie[3] = 'i';
indices_bmjie[4] = 'e';
IndexArray indices_ei( 2 );
indices_ei[0] = 'e';
indices_ei[1] = 'i';
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
	//P_jimb_lvl1_part0_1_lvl1_part1B[D0,D1,D2,D3]
DistTensor<double> P_jimb_lvl1_part0_1_lvl1_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//P_jimb_lvl1_part0_1_lvl1_part1T[D0,D1,D2,D3]
DistTensor<double> P_jimb_lvl1_part0_1_lvl1_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//P_jimb_lvl1_part0_1_lvl1_part2B[D0,D1,D2,D3]
DistTensor<double> P_jimb_lvl1_part0_1_lvl1_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//P_jimb_lvl1_part0_1_lvl1_part2T[D0,D1,D2,D3]
DistTensor<double> P_jimb_lvl1_part0_1_lvl1_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//P_jimb_lvl1_part0_1_lvl2_part1B[D0,D1,D2,D3]
DistTensor<double> P_jimb_lvl1_part0_1_lvl2_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//P_jimb_lvl1_part0_1_lvl2_part1T[D0,D1,D2,D3]
DistTensor<double> P_jimb_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//P_jimb_lvl1_part0_1_lvl2_part1_0[D0,D1,D2,D3]
DistTensor<double> P_jimb_lvl1_part0_1_lvl2_part1_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//P_jimb_lvl1_part0_1_lvl2_part1_1[D0,D1,D2,D3]
DistTensor<double> P_jimb_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//P_jimb_lvl1_part0_1_lvl2_part1_1[D3,*,D1,D0,D2]
DistTensor<double> P_jimb_lvl1_part0_1_lvl2_part1_1_perm32014__D_0__D_1__D_3__S__D_2( dist__D_3__S__D_1__D_0__D_2, g );
P_jimb_lvl1_part0_1_lvl2_part1_1_perm32014__D_0__D_1__D_3__S__D_2.SetLocalPermutation( perm_3_2_0_1_4 );
	//P_jimb_lvl1_part0_1_lvl2_part1_1[*,*,D1,D0,D2,D3]
DistTensor<double> P_jimb_lvl1_part0_1_lvl2_part1_1_perm321045__D_0__D_1__S__S__D_2__D_3( dist__S__S__D_1__D_0__D_2__D_3, g );
P_jimb_lvl1_part0_1_lvl2_part1_1_perm321045__D_0__D_1__S__S__D_2__D_3.SetLocalPermutation( perm_3_2_1_0_4_5 );
	//P_jimb_lvl1_part0_1_lvl2_part1_1_temp[D0,D1,D2,D3]
DistTensor<double> P_jimb_lvl1_part0_1_lvl2_part1_1_temp__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//P_jimb_lvl1_part0_1_lvl2_part1_1_temp[D3,D1,D2,D0]
DistTensor<double> P_jimb_lvl1_part0_1_lvl2_part1_1_temp__D_3__D_1__D_2__D_0( dist__D_3__D_1__D_2__D_0, g );
	//P_jimb_lvl1_part0_1_lvl2_part1_1_temp[D3,*,D12,D0]
DistTensor<double> P_jimb_lvl1_part0_1_lvl2_part1_1_temp__D_3__S__D_1_2__D_0( dist__D_3__S__D_1_2__D_0, g );
	//P_jimb_lvl1_part0_1_lvl2_part1_1_temp[D3,*,D21,D0]
DistTensor<double> P_jimb_lvl1_part0_1_lvl2_part1_1_temp__D_3__S__D_2_1__D_0( dist__D_3__S__D_2_1__D_0, g );
	//P_jimb_lvl1_part0_1_lvl2_part1_1_temp[*,D1,D2,D03]
DistTensor<double> P_jimb_lvl1_part0_1_lvl2_part1_1_temp__S__D_1__D_2__D_0_3( dist__S__D_1__D_2__D_0_3, g );
	//P_jimb_lvl1_part0_1_lvl2_part1_1_temp[*,*,D12,D03]
DistTensor<double> P_jimb_lvl1_part0_1_lvl2_part1_1_temp__S__S__D_1_2__D_0_3( dist__S__S__D_1_2__D_0_3, g );
	//P_jimb_lvl1_part0_1_lvl2_part1_1_temp[*,*,D21,D03]
DistTensor<double> P_jimb_lvl1_part0_1_lvl2_part1_1_temp__S__S__D_2_1__D_0_3( dist__S__S__D_2_1__D_0_3, g );
	//P_jimb_lvl1_part0_1_lvl2_part1_2[D0,D1,D2,D3]
DistTensor<double> P_jimb_lvl1_part0_1_lvl2_part1_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//P_jimb_lvl1_part0_1_lvl2_part2B[D0,D1,D2,D3]
DistTensor<double> P_jimb_lvl1_part0_1_lvl2_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//P_jimb_lvl1_part0_1_lvl2_part2T[D0,D1,D2,D3]
DistTensor<double> P_jimb_lvl1_part0_1_lvl2_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//P_jimb_lvl1_part0_1_lvl2_part2_0[D0,D1,D2,D3]
DistTensor<double> P_jimb_lvl1_part0_1_lvl2_part2_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//P_jimb_lvl1_part0_1_lvl2_part2_1[D0,D1,D2,D3]
DistTensor<double> P_jimb_lvl1_part0_1_lvl2_part2_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//P_jimb_lvl1_part0_1_lvl2_part2_1[*,D2,D1,D0,D3]
DistTensor<double> P_jimb_lvl1_part0_1_lvl2_part2_1_perm32104__D_0__D_1__D_2__S__D_3( dist__S__D_2__D_1__D_0__D_3, g );
P_jimb_lvl1_part0_1_lvl2_part2_1_perm32104__D_0__D_1__D_2__S__D_3.SetLocalPermutation( perm_3_2_1_0_4 );
	//P_jimb_lvl1_part0_1_lvl2_part2_1_temp[D0,D1,D2,D3]
DistTensor<double> P_jimb_lvl1_part0_1_lvl2_part2_1_temp__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//P_jimb_lvl1_part0_1_lvl2_part2_1_temp[*,D1,D2,D03]
DistTensor<double> P_jimb_lvl1_part0_1_lvl2_part2_1_temp__S__D_1__D_2__D_0_3( dist__S__D_1__D_2__D_0_3, g );
	//P_jimb_lvl1_part0_1_lvl2_part2_1_temp[*,D2,D1,D03]
DistTensor<double> P_jimb_lvl1_part0_1_lvl2_part2_1_temp__S__D_2__D_1__D_0_3( dist__S__D_2__D_1__D_0_3, g );
	//P_jimb_lvl1_part0_1_lvl2_part2_2[D0,D1,D2,D3]
DistTensor<double> P_jimb_lvl1_part0_1_lvl2_part2_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//P_jimb_lvl1_part0_2[D0,D1,D2,D3]
DistTensor<double> P_jimb_lvl1_part0_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
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
	//Tau_efmn_lvl1_part3_1_lvl2_part2_1[D2,D1,D0,D3]
DistTensor<double> Tau_efmn_lvl1_part3_1_lvl2_part2_1__D_2__D_1__D_0__D_3( dist__D_2__D_1__D_0__D_3, g );
	//Tau_efmn_lvl1_part3_1_lvl2_part2_1[D2,D3,D0,*]
DistTensor<double> Tau_efmn_lvl1_part3_1_lvl2_part2_1__D_2__D_3__D_0__S( dist__D_2__D_3__D_0__S, g );
	//Tau_efmn_lvl1_part3_1_lvl2_part2_1[D2,D3,*,*]
DistTensor<double> Tau_efmn_lvl1_part3_1_lvl2_part2_1__D_2__D_3__S__S( dist__D_2__D_3__S__S, g );
	//Tau_efmn_lvl1_part3_1_lvl2_part2_2[D0,D1,D2,D3]
DistTensor<double> Tau_efmn_lvl1_part3_1_lvl2_part2_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Tau_efmn_lvl1_part3_2[D0,D1,D2,D3]
DistTensor<double> Tau_efmn_lvl1_part3_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//t_fj_lvl1_part1_1[D013,D2]
DistTensor<double> t_fj_lvl1_part1_1__D_0_1_3__D_2( dist__D_0_1_3__D_2, g );
	//t_fj_lvl1_part1_1[D01,D2]
DistTensor<double> t_fj_lvl1_part1_1__D_0_1__D_2( dist__D_0_1__D_2, g );
	//t_fj_lvl1_part1_1[D301,D2]
DistTensor<double> t_fj_lvl1_part1_1__D_3_0_1__D_2( dist__D_3_0_1__D_2, g );
	//t_fj_lvl1_part1_1[D3,*]
DistTensor<double> t_fj_lvl1_part1_1__D_3__S( dist__D_3__S, g );
	//t_fj_lvl2_part1_1[D012,*]
DistTensor<double> t_fj_lvl2_part1_1__D_0_1_2__S( dist__D_0_1_2__S, g );
	//t_fj_lvl2_part1_1[D01,*]
DistTensor<double> t_fj_lvl2_part1_1__D_0_1__S( dist__D_0_1__S, g );
	//t_fj_lvl2_part1_1[D201,*]
DistTensor<double> t_fj_lvl2_part1_1__D_2_0_1__S( dist__D_2_0_1__S, g );
	//t_fj_lvl2_part1_1[D2,*]
DistTensor<double> t_fj_lvl2_part1_1__D_2__S( dist__D_2__S, g );
	//w_bmje_lvl2_part1B[D0,D1,D2,D3]
DistTensor<double> w_bmje_lvl2_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//w_bmje_lvl2_part1T[D0,D1,D2,D3]
DistTensor<double> w_bmje_lvl2_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//w_bmje_lvl2_part1_0[D0,D1,D2,D3]
DistTensor<double> w_bmje_lvl2_part1_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//w_bmje_lvl2_part1_1[D0,D1,D2,D3]
DistTensor<double> w_bmje_lvl2_part1_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//w_bmje_lvl2_part1_2[D0,D1,D2,D3]
DistTensor<double> w_bmje_lvl2_part1_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//x_bmej_lvl0_part3B[D0,D1,D2,D3]
DistTensor<double> x_bmej_lvl0_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//x_bmej_lvl0_part3T[D0,D1,D2,D3]
DistTensor<double> x_bmej_lvl0_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//x_bmej_lvl1_part3B[D0,D1,D2,D3]
DistTensor<double> x_bmej_lvl1_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//x_bmej_lvl1_part3T[D0,D1,D2,D3]
DistTensor<double> x_bmej_lvl1_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//x_bmej_lvl1_part3_0[D0,D1,D2,D3]
DistTensor<double> x_bmej_lvl1_part3_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//x_bmej_lvl1_part3_1[D0,D1,D2,D3]
DistTensor<double> x_bmej_lvl1_part3_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> x_bmej_lvl1_part3_1_perm0132__D_0__D_1__D_3__D_2( dist__D_0__D_1__D_2__D_3, g );
x_bmej_lvl1_part3_1_perm0132__D_0__D_1__D_3__D_2.SetLocalPermutation( perm_0_1_3_2 );
	//x_bmej_lvl1_part3_2[D0,D1,D2,D3]
DistTensor<double> x_bmej_lvl1_part3_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
// P_jimb has 4 dims
ObjShape P_jimb__D_0__D_1__D_2__D_3_tempShape( 4 );
P_jimb__D_0__D_1__D_2__D_3_tempShape[ 0 ] = n_o;
P_jimb__D_0__D_1__D_2__D_3_tempShape[ 1 ] = n_o;
P_jimb__D_0__D_1__D_2__D_3_tempShape[ 2 ] = n_o;
P_jimb__D_0__D_1__D_2__D_3_tempShape[ 3 ] = n_v;
P_jimb__D_0__D_1__D_2__D_3.ResizeTo( P_jimb__D_0__D_1__D_2__D_3_tempShape );
MakeUniform( P_jimb__D_0__D_1__D_2__D_3 );
TensorDistribution dist__D_1_2__D_0_3 = tmen::StringToTensorDist("[(1,2),(0,3)]");
TensorDistribution dist__D_2_1__D_0_3 = tmen::StringToTensorDist("[(2,1),(0,3)]");
TensorDistribution dist__D_1__D_2_3 = tmen::StringToTensorDist("[(1),(2,3)]");
TensorDistribution dist__D_1__D_0_3_2 = tmen::StringToTensorDist("[(1),(0,3,2)]");
TensorDistribution dist__D_1__D_2_3_0 = tmen::StringToTensorDist("[(1),(2,3,0)]");
TensorDistribution dist__D_1__D_0_3 = tmen::StringToTensorDist("[(1),(0,3)]");
TensorDistribution dist__D_2__D_0__D_1__D_3 = tmen::StringToTensorDist("[(2),(0),(1),(3)]");
TensorDistribution dist__D_1_0__D_2_3 = tmen::StringToTensorDist("[(1,0),(2,3)]");
ModeArray modes_0_3_2( 3 );
modes_0_3_2[0] = 0;
modes_0_3_2[1] = 3;
modes_0_3_2[2] = 2;
ModeArray modes_2_0( 2 );
modes_2_0[0] = 2;
modes_2_0[1] = 0;
ModeArray modes_3_2( 2 );
modes_3_2[0] = 3;
modes_3_2[1] = 2;
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
	//H_me_lvl1_part0_1[D2,D0,D1,D3]
DistTensor<double> H_me_lvl1_part0_1_perm1023__D_0__D_2__D_1__D_3( dist__D_2__D_0__D_1__D_3, g );
H_me_lvl1_part0_1_perm1023__D_0__D_2__D_1__D_3.SetLocalPermutation( perm_1_0_2_3 );
	//H_me_lvl1_part0_1_temp[D01,D23]
DistTensor<double> H_me_lvl1_part0_1_temp__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//H_me_lvl1_part0_1_temp[D10,D23]
DistTensor<double> H_me_lvl1_part0_1_temp__D_1_0__D_2_3( dist__D_1_0__D_2_3, g );
	//H_me_lvl1_part0_1_temp[D12,D03]
DistTensor<double> H_me_lvl1_part0_1_temp__D_1_2__D_0_3( dist__D_1_2__D_0_3, g );
	//H_me_lvl1_part0_1_temp[D1,D03]
DistTensor<double> H_me_lvl1_part0_1_temp__D_1__D_0_3( dist__D_1__D_0_3, g );
	//H_me_lvl1_part0_1_temp[D1,D032]
DistTensor<double> H_me_lvl1_part0_1_temp__D_1__D_0_3_2( dist__D_1__D_0_3_2, g );
	//H_me_lvl1_part0_1_temp[D1,D23]
DistTensor<double> H_me_lvl1_part0_1_temp__D_1__D_2_3( dist__D_1__D_2_3, g );
	//H_me_lvl1_part0_1_temp[D1,D230]
DistTensor<double> H_me_lvl1_part0_1_temp__D_1__D_2_3_0( dist__D_1__D_2_3_0, g );
	//H_me_lvl1_part0_1_temp[D21,D03]
DistTensor<double> H_me_lvl1_part0_1_temp__D_2_1__D_0_3( dist__D_2_1__D_0_3, g );
	//H_me_lvl1_part0_2[D01,D23]
DistTensor<double> H_me_lvl1_part0_2__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//Htemp1[D0,D1,D2,D3]
DistTensor<double> Htemp1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Htemp1_lvl0_part2B[D0,D1,D2,D3]
DistTensor<double> Htemp1_lvl0_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Htemp1_lvl0_part2T[D0,D1,D2,D3]
DistTensor<double> Htemp1_lvl0_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Htemp1_lvl1_part2B[D0,D1,D2,D3]
DistTensor<double> Htemp1_lvl1_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Htemp1_lvl1_part2T[D0,D1,D2,D3]
DistTensor<double> Htemp1_lvl1_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Htemp1_lvl1_part2_0[D0,D1,D2,D3]
DistTensor<double> Htemp1_lvl1_part2_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Htemp1_lvl1_part2_1[D0,D1,D2,D3]
DistTensor<double> Htemp1_lvl1_part2_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Htemp1_lvl1_part2_1_lvl1_part3B[D0,D1,D2,D3]
DistTensor<double> Htemp1_lvl1_part2_1_lvl1_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Htemp1_lvl1_part2_1_lvl1_part3T[D0,D1,D2,D3]
DistTensor<double> Htemp1_lvl1_part2_1_lvl1_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Htemp1_lvl1_part2_1_lvl2_part3B[D0,D1,D2,D3]
DistTensor<double> Htemp1_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Htemp1_lvl1_part2_1_lvl2_part3T[D0,D1,D2,D3]
DistTensor<double> Htemp1_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Htemp1_lvl1_part2_1_lvl2_part3_0[D0,D1,D2,D3]
DistTensor<double> Htemp1_lvl1_part2_1_lvl2_part3_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Htemp1_lvl1_part2_1_lvl2_part3_1[D0,D1,D2,D3]
DistTensor<double> Htemp1_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> Htemp1_lvl1_part2_1_lvl2_part3_1_perm0213__D_0__D_2__D_1__D_3( dist__D_0__D_1__D_2__D_3, g );
Htemp1_lvl1_part2_1_lvl2_part3_1_perm0213__D_0__D_2__D_1__D_3.SetLocalPermutation( perm_0_2_1_3 );
	//Htemp1_lvl1_part2_1_lvl2_part3_2[D0,D1,D2,D3]
DistTensor<double> Htemp1_lvl1_part2_1_lvl2_part3_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Htemp1_lvl1_part2_2[D0,D1,D2,D3]
DistTensor<double> Htemp1_lvl1_part2_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//t_fj_lvl2_part1_1[D10,D23]
DistTensor<double> t_fj_lvl2_part1_1__D_1_0__D_2_3( dist__D_1_0__D_2_3, g );
// H_me has 2 dims
ObjShape H_me__D_0_1__D_2_3_tempShape( 2 );
H_me__D_0_1__D_2_3_tempShape[ 0 ] = n_o;
H_me__D_0_1__D_2_3_tempShape[ 1 ] = n_v;
H_me__D_0_1__D_2_3.ResizeTo( H_me__D_0_1__D_2_3_tempShape );
MakeUniform( H_me__D_0_1__D_2_3 );
TensorDistribution dist__S__D_2_3 = tmen::StringToTensorDist("[(),(2,3)]");
TensorDistribution dist__S__D_1__D_2__D_3 = tmen::StringToTensorDist("[(),(1),(2),(3)]");
TensorDistribution dist__D_0__S__D_1__D_2__D_3 = tmen::StringToTensorDist("[(0),(),(1),(2),(3)]");
TensorDistribution dist__D_0__D_2__D_3__D_1 = tmen::StringToTensorDist("[(0),(2),(3),(1)]");
TensorDistribution dist__D_3_0__D_1_2 = tmen::StringToTensorDist("[(3,0),(1,2)]");
TensorDistribution dist__D_3_0__D_2_1 = tmen::StringToTensorDist("[(3,0),(2,1)]");
TensorDistribution dist__D_3__D_1 = tmen::StringToTensorDist("[(3),(1)]");
TensorDistribution dist__D_0_3__D_2_1 = tmen::StringToTensorDist("[(0,3),(2,1)]");
Permutation perm_1_0_2_3_4( 5 );
perm_1_0_2_3_4[0] = 1;
perm_1_0_2_3_4[1] = 0;
perm_1_0_2_3_4[2] = 2;
perm_1_0_2_3_4[3] = 3;
perm_1_0_2_3_4[4] = 4;
ModeArray modes_3_1( 2 );
modes_3_1[0] = 3;
modes_3_1[1] = 1;
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
DistTensor<double> T_bfnj_lvl1_part2_1_lvl2_part3_1_perm1230__D_1__D_2__D_3__D_0( dist__D_0__D_1__D_2__D_3, g );
T_bfnj_lvl1_part2_1_lvl2_part3_1_perm1230__D_1__D_2__D_3__D_0.SetLocalPermutation( perm_1_2_3_0 );
	//t_fj_lvl2_part1_1[D03,D21]
DistTensor<double> t_fj_lvl2_part1_1__D_0_3__D_2_1( dist__D_0_3__D_2_1, g );
	//t_fj_lvl2_part1_1[D30,D12]
DistTensor<double> t_fj_lvl2_part1_1__D_3_0__D_1_2( dist__D_3_0__D_1_2, g );
	//t_fj_lvl2_part1_1[D30,D21]
DistTensor<double> t_fj_lvl2_part1_1__D_3_0__D_2_1( dist__D_3_0__D_2_1, g );
	//t_fj_lvl2_part1_1[D3,D1]
DistTensor<double> t_fj_lvl2_part1_1__D_3__D_1( dist__D_3__D_1, g );
DistTensor<double> t_fj_lvl2_part1_1_perm10__S__D_0_1( dist__D_0_1__S, g );
t_fj_lvl2_part1_1_perm10__S__D_0_1.SetLocalPermutation( perm_1_0 );
// F_ae has 2 dims
ObjShape F_ae__D_0_1__D_2_3_tempShape( 2 );
F_ae__D_0_1__D_2_3_tempShape[ 0 ] = n_v;
F_ae__D_0_1__D_2_3_tempShape[ 1 ] = n_v;
F_ae__D_0_1__D_2_3.ResizeTo( F_ae__D_0_1__D_2_3_tempShape );
MakeUniform( F_ae__D_0_1__D_2_3 );
TensorDistribution dist__S__D_2__D_0__D_1__D_3 = tmen::StringToTensorDist("[(),(2),(0),(1),(3)]");
TensorDistribution dist__D_0__D_1__S__D_3 = tmen::StringToTensorDist("[(0),(1),(),(3)]");
TensorDistribution dist__D_0_1_2_3__S = tmen::StringToTensorDist("[(0,1,2,3),()]");
TensorDistribution dist__D_2_3__S = tmen::StringToTensorDist("[(2,3),()]");
TensorDistribution dist__D_2_3_0_1__S = tmen::StringToTensorDist("[(2,3,0,1),()]");
TensorDistribution dist__D_0_1__S__D_2_3 = tmen::StringToTensorDist("[(0,1),(),(2,3)]");
TensorDistribution dist__D_0_3__D_1_2 = tmen::StringToTensorDist("[(0,3),(1,2)]");
Permutation perm_0_1_2( 3 );
perm_0_1_2[0] = 0;
perm_0_1_2[1] = 1;
perm_0_1_2[2] = 2;
Permutation perm_2_0_1_3( 4 );
perm_2_0_1_3[0] = 2;
perm_2_0_1_3[1] = 0;
perm_2_0_1_3[2] = 1;
perm_2_0_1_3[3] = 3;
IndexArray indices_efni( 4 );
indices_efni[0] = 'e';
indices_efni[1] = 'f';
indices_efni[2] = 'n';
indices_efni[3] = 'i';
IndexArray indices_en( 2 );
indices_en[0] = 'e';
indices_en[1] = 'n';
IndexArray indices_me( 2 );
indices_me[0] = 'm';
indices_me[1] = 'e';
IndexArray indices_mefn( 4 );
indices_mefn[0] = 'm';
indices_mefn[1] = 'e';
indices_mefn[2] = 'f';
indices_mefn[3] = 'n';
IndexArray indices_mie( 3 );
indices_mie[0] = 'm';
indices_mie[1] = 'i';
indices_mie[2] = 'e';
IndexArray indices_miefn( 5 );
indices_miefn[0] = 'm';
indices_miefn[1] = 'i';
indices_miefn[2] = 'e';
indices_miefn[3] = 'f';
indices_miefn[4] = 'n';
IndexArray indices_mien( 4 );
indices_mien[0] = 'm';
indices_mien[1] = 'i';
indices_mien[2] = 'e';
indices_mien[3] = 'n';
	//G_mi[D01,D23]
DistTensor<double> G_mi__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//G_mi_lvl0_part0B[D01,D23]
DistTensor<double> G_mi_lvl0_part0B__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//G_mi_lvl0_part0T[D01,D23]
DistTensor<double> G_mi_lvl0_part0T__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//G_mi_lvl0_part1B[D01,D23]
DistTensor<double> G_mi_lvl0_part1B__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//G_mi_lvl0_part1T[D01,D23]
DistTensor<double> G_mi_lvl0_part1T__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//G_mi_lvl1_part0B[D01,D23]
DistTensor<double> G_mi_lvl1_part0B__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//G_mi_lvl1_part0T[D01,D23]
DistTensor<double> G_mi_lvl1_part0T__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//G_mi_lvl1_part0_0[D01,D23]
DistTensor<double> G_mi_lvl1_part0_0__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//G_mi_lvl1_part0_1[D01,D23]
DistTensor<double> G_mi_lvl1_part0_1__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//G_mi_lvl1_part0_1[*,D2,D0,D1,D3]
DistTensor<double> G_mi_lvl1_part0_1__S__D_2__D_0__D_1__D_3( dist__S__D_2__D_0__D_1__D_3, g );
	//G_mi_lvl1_part0_1_lvl1_part1B[D01,D23]
DistTensor<double> G_mi_lvl1_part0_1_lvl1_part1B__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//G_mi_lvl1_part0_1_lvl1_part1T[D01,D23]
DistTensor<double> G_mi_lvl1_part0_1_lvl1_part1T__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//G_mi_lvl1_part0_1_lvl2_part1B[D01,D23]
DistTensor<double> G_mi_lvl1_part0_1_lvl2_part1B__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//G_mi_lvl1_part0_1_lvl2_part1T[D01,D23]
DistTensor<double> G_mi_lvl1_part0_1_lvl2_part1T__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//G_mi_lvl1_part0_1_lvl2_part1_0[D01,D23]
DistTensor<double> G_mi_lvl1_part0_1_lvl2_part1_0__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//G_mi_lvl1_part0_1_lvl2_part1_1[D01,D23]
DistTensor<double> G_mi_lvl1_part0_1_lvl2_part1_1__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//G_mi_lvl1_part0_1_lvl2_part1_1[D01,*,D23]
DistTensor<double> G_mi_lvl1_part0_1_lvl2_part1_1__D_0_1__S__D_2_3( dist__D_0_1__S__D_2_3, g );
	//G_mi_lvl1_part0_1_lvl2_part1_2[D01,D23]
DistTensor<double> G_mi_lvl1_part0_1_lvl2_part1_2__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//G_mi_lvl1_part0_2[D01,D23]
DistTensor<double> G_mi_lvl1_part0_2__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//G_mi_lvl1_part1B[D01,D23]
DistTensor<double> G_mi_lvl1_part1B__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//G_mi_lvl1_part1T[D01,D23]
DistTensor<double> G_mi_lvl1_part1T__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//G_mi_lvl1_part1_0[D01,D23]
DistTensor<double> G_mi_lvl1_part1_0__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//G_mi_lvl1_part1_1[D01,D23]
DistTensor<double> G_mi_lvl1_part1_1__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//G_mi_lvl1_part1_1[D0,D2,D3,D1]
DistTensor<double> G_mi_lvl1_part1_1__D_0__D_2__D_3__D_1( dist__D_0__D_2__D_3__D_1, g );
	//G_mi_lvl1_part1_2[D01,D23]
DistTensor<double> G_mi_lvl1_part1_2__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//Gtemp1[D0,D1,D2,D3]
DistTensor<double> Gtemp1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Gtemp1_lvl0_part2B[D0,D1,D2,D3]
DistTensor<double> Gtemp1_lvl0_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Gtemp1_lvl0_part2T[D0,D1,D2,D3]
DistTensor<double> Gtemp1_lvl0_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Gtemp1_lvl1_part2B[D0,D1,D2,D3]
DistTensor<double> Gtemp1_lvl1_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Gtemp1_lvl1_part2T[D0,D1,D2,D3]
DistTensor<double> Gtemp1_lvl1_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Gtemp1_lvl1_part2_0[D0,D1,D2,D3]
DistTensor<double> Gtemp1_lvl1_part2_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Gtemp1_lvl1_part2_1[D0,D1,D2,D3]
DistTensor<double> Gtemp1_lvl1_part2_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Gtemp1_lvl1_part2_1_lvl1_part3B[D0,D1,D2,D3]
DistTensor<double> Gtemp1_lvl1_part2_1_lvl1_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Gtemp1_lvl1_part2_1_lvl1_part3T[D0,D1,D2,D3]
DistTensor<double> Gtemp1_lvl1_part2_1_lvl1_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Gtemp1_lvl1_part2_1_lvl2_part3B[D0,D1,D2,D3]
DistTensor<double> Gtemp1_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Gtemp1_lvl1_part2_1_lvl2_part3T[D0,D1,D2,D3]
DistTensor<double> Gtemp1_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Gtemp1_lvl1_part2_1_lvl2_part3_0[D0,D1,D2,D3]
DistTensor<double> Gtemp1_lvl1_part2_1_lvl2_part3_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Gtemp1_lvl1_part2_1_lvl2_part3_1[D0,D1,D2,D3]
DistTensor<double> Gtemp1_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Gtemp1_lvl1_part2_1_lvl2_part3_1[D0,D1,*,D3]
DistTensor<double> Gtemp1_lvl1_part2_1_lvl2_part3_1_perm2013__S__D_0__D_1__D_3( dist__D_0__D_1__S__D_3, g );
Gtemp1_lvl1_part2_1_lvl2_part3_1_perm2013__S__D_0__D_1__D_3.SetLocalPermutation( perm_2_0_1_3 );
	//Gtemp1_lvl1_part2_1_lvl2_part3_2[D0,D1,D2,D3]
DistTensor<double> Gtemp1_lvl1_part2_1_lvl2_part3_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Gtemp1_lvl1_part2_2[D0,D1,D2,D3]
DistTensor<double> Gtemp1_lvl1_part2_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Gtemp2[D0,D1,D2,D3]
DistTensor<double> Gtemp2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Gtemp2_lvl0_part0B[D0,D1,D2,D3]
DistTensor<double> Gtemp2_lvl0_part0B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Gtemp2_lvl0_part0T[D0,D1,D2,D3]
DistTensor<double> Gtemp2_lvl0_part0T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Gtemp2_lvl0_part2B[D0,D1,D2,D3]
DistTensor<double> Gtemp2_lvl0_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Gtemp2_lvl0_part2T[D0,D1,D2,D3]
DistTensor<double> Gtemp2_lvl0_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Gtemp2_lvl1_part0B[D0,D1,D2,D3]
DistTensor<double> Gtemp2_lvl1_part0B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Gtemp2_lvl1_part0T[D0,D1,D2,D3]
DistTensor<double> Gtemp2_lvl1_part0T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Gtemp2_lvl1_part0_0[D0,D1,D2,D3]
DistTensor<double> Gtemp2_lvl1_part0_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Gtemp2_lvl1_part0_1[D0,D1,D2,D3]
DistTensor<double> Gtemp2_lvl1_part0_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Gtemp2_lvl1_part0_1_lvl1_part1B[D0,D1,D2,D3]
DistTensor<double> Gtemp2_lvl1_part0_1_lvl1_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Gtemp2_lvl1_part0_1_lvl1_part1T[D0,D1,D2,D3]
DistTensor<double> Gtemp2_lvl1_part0_1_lvl1_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Gtemp2_lvl1_part0_1_lvl2_part1B[D0,D1,D2,D3]
DistTensor<double> Gtemp2_lvl1_part0_1_lvl2_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Gtemp2_lvl1_part0_1_lvl2_part1T[D0,D1,D2,D3]
DistTensor<double> Gtemp2_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Gtemp2_lvl1_part0_1_lvl2_part1_0[D0,D1,D2,D3]
DistTensor<double> Gtemp2_lvl1_part0_1_lvl2_part1_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Gtemp2_lvl1_part0_1_lvl2_part1_1[D0,D1,D2,D3]
DistTensor<double> Gtemp2_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Gtemp2_lvl1_part0_1_lvl2_part1_2[D0,D1,D2,D3]
DistTensor<double> Gtemp2_lvl1_part0_1_lvl2_part1_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Gtemp2_lvl1_part0_2[D0,D1,D2,D3]
DistTensor<double> Gtemp2_lvl1_part0_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Gtemp2_lvl1_part2B[D0,D1,D2,D3]
DistTensor<double> Gtemp2_lvl1_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Gtemp2_lvl1_part2T[D0,D1,D2,D3]
DistTensor<double> Gtemp2_lvl1_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Gtemp2_lvl1_part2_0[D0,D1,D2,D3]
DistTensor<double> Gtemp2_lvl1_part2_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Gtemp2_lvl1_part2_1[D0,D1,D2,D3]
DistTensor<double> Gtemp2_lvl1_part2_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Gtemp2_lvl1_part2_1_lvl1_part1B[D0,D1,D2,D3]
DistTensor<double> Gtemp2_lvl1_part2_1_lvl1_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Gtemp2_lvl1_part2_1_lvl1_part1T[D0,D1,D2,D3]
DistTensor<double> Gtemp2_lvl1_part2_1_lvl1_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Gtemp2_lvl1_part2_1_lvl2_part1B[D0,D1,D2,D3]
DistTensor<double> Gtemp2_lvl1_part2_1_lvl2_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Gtemp2_lvl1_part2_1_lvl2_part1T[D0,D1,D2,D3]
DistTensor<double> Gtemp2_lvl1_part2_1_lvl2_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Gtemp2_lvl1_part2_1_lvl2_part1_0[D0,D1,D2,D3]
DistTensor<double> Gtemp2_lvl1_part2_1_lvl2_part1_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Gtemp2_lvl1_part2_1_lvl2_part1_1[D0,D1,D2,D3]
DistTensor<double> Gtemp2_lvl1_part2_1_lvl2_part1_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> Gtemp2_lvl1_part2_1_lvl2_part1_1_perm0231__D_0__D_2__D_3__D_1( dist__D_0__D_1__D_2__D_3, g );
Gtemp2_lvl1_part2_1_lvl2_part1_1_perm0231__D_0__D_2__D_3__D_1.SetLocalPermutation( perm_0_2_3_1 );
	//Gtemp2_lvl1_part2_1_lvl2_part1_2[D0,D1,D2,D3]
DistTensor<double> Gtemp2_lvl1_part2_1_lvl2_part1_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Gtemp2_lvl1_part2_2[D0,D1,D2,D3]
DistTensor<double> Gtemp2_lvl1_part2_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//T_bfnj_lvl2_part3B[D0,D1,D2,D3]
DistTensor<double> T_bfnj_lvl2_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//T_bfnj_lvl2_part3T[D0,D1,D2,D3]
DistTensor<double> T_bfnj_lvl2_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//T_bfnj_lvl2_part3_0[D0,D1,D2,D3]
DistTensor<double> T_bfnj_lvl2_part3_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//T_bfnj_lvl2_part3_1[D0,D1,D2,D3]
DistTensor<double> T_bfnj_lvl2_part3_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> T_bfnj_lvl2_part3_1_perm0132__D_0__D_1__D_3__D_2( dist__D_0__D_1__D_2__D_3, g );
T_bfnj_lvl2_part3_1_perm0132__D_0__D_1__D_3__D_2.SetLocalPermutation( perm_0_1_3_2 );
	//T_bfnj_lvl2_part3_2[D0,D1,D2,D3]
DistTensor<double> T_bfnj_lvl2_part3_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//t_fj_lvl2_part1_1[D0123,*]
DistTensor<double> t_fj_lvl2_part1_1__D_0_1_2_3__S( dist__D_0_1_2_3__S, g );
	//t_fj_lvl2_part1_1[D03,D12]
DistTensor<double> t_fj_lvl2_part1_1__D_0_3__D_1_2( dist__D_0_3__D_1_2, g );
	//t_fj_lvl2_part1_1[D2301,*]
DistTensor<double> t_fj_lvl2_part1_1__D_2_3_0_1__S( dist__D_2_3_0_1__S, g );
	//t_fj_lvl2_part1_1[D23,*]
DistTensor<double> t_fj_lvl2_part1_1__D_2_3__S( dist__D_2_3__S, g );
// G_mi has 2 dims
ObjShape G_mi__D_0_1__D_2_3_tempShape( 2 );
G_mi__D_0_1__D_2_3_tempShape[ 0 ] = n_o;
G_mi__D_0_1__D_2_3_tempShape[ 1 ] = n_o;
G_mi__D_0_1__D_2_3.ResizeTo( G_mi__D_0_1__D_2_3_tempShape );
MakeUniform( G_mi__D_0_1__D_2_3 );
TensorDistribution dist__D_0__S__D_2__D_3__D_1 = tmen::StringToTensorDist("[(0),(),(2),(3),(1)]");
TensorDistribution dist__D_0__D_3__D_2__D_1 = tmen::StringToTensorDist("[(0),(3),(2),(1)]");
TensorDistribution dist__D_2__D_3__S__D_1 = tmen::StringToTensorDist("[(2),(3),(),(1)]");
TensorDistribution dist__D_2__D_3__D_0__D_1 = tmen::StringToTensorDist("[(2),(3),(0),(1)]");
Permutation perm_1_0_3_4_2( 5 );
perm_1_0_3_4_2[0] = 1;
perm_1_0_3_4_2[1] = 0;
perm_1_0_3_4_2[2] = 3;
perm_1_0_3_4_2[3] = 4;
perm_1_0_3_4_2[4] = 2;
Permutation perm_2_3_1_0( 4 );
perm_2_3_1_0[0] = 2;
perm_2_3_1_0[1] = 3;
perm_2_3_1_0[2] = 1;
perm_2_3_1_0[3] = 0;
IndexArray indices_aiefm( 5 );
indices_aiefm[0] = 'a';
indices_aiefm[1] = 'i';
indices_aiefm[2] = 'e';
indices_aiefm[3] = 'f';
indices_aiefm[4] = 'm';
IndexArray indices_aiem( 4 );
indices_aiem[0] = 'a';
indices_aiem[1] = 'i';
indices_aiem[2] = 'e';
indices_aiem[3] = 'm';
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
IndexArray indices_mnea( 4 );
indices_mnea[0] = 'm';
indices_mnea[1] = 'n';
indices_mnea[2] = 'e';
indices_mnea[3] = 'a';
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
	//H_me_lvl2_part0B[D01,D23]
DistTensor<double> H_me_lvl2_part0B__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//H_me_lvl2_part0T[D01,D23]
DistTensor<double> H_me_lvl2_part0T__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//H_me_lvl2_part0_0[D01,D23]
DistTensor<double> H_me_lvl2_part0_0__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//H_me_lvl2_part0_1[D01,D23]
DistTensor<double> H_me_lvl2_part0_1__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//H_me_lvl2_part0_1[D03,D12]
DistTensor<double> H_me_lvl2_part0_1__D_0_3__D_1_2( dist__D_0_3__D_1_2, g );
	//H_me_lvl2_part0_1[D03,D21]
DistTensor<double> H_me_lvl2_part0_1__D_0_3__D_2_1( dist__D_0_3__D_2_1, g );
	//H_me_lvl2_part0_1[D30,D12]
DistTensor<double> H_me_lvl2_part0_1__D_3_0__D_1_2( dist__D_3_0__D_1_2, g );
	//H_me_lvl2_part0_1[D3,D1]
DistTensor<double> H_me_lvl2_part0_1__D_3__D_1( dist__D_3__D_1, g );
	//H_me_lvl2_part0_2[D01,D23]
DistTensor<double> H_me_lvl2_part0_2__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
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
	//Tau_efmn_lvl1_part3_1_lvl2_part2_1[D2,D3,D0,D1]
DistTensor<double> Tau_efmn_lvl1_part3_1_lvl2_part2_1__D_2__D_3__D_0__D_1( dist__D_2__D_3__D_0__D_1, g );
	//Tau_efmn_lvl1_part3_1_lvl2_part2_1[D2,D3,*,D1]
DistTensor<double> Tau_efmn_lvl1_part3_1_lvl2_part2_1_perm0132__D_2__D_3__D_1__S( dist__D_2__D_3__S__D_1, g );
Tau_efmn_lvl1_part3_1_lvl2_part2_1_perm0132__D_2__D_3__D_1__S.SetLocalPermutation( perm_0_1_3_2 );
	//U_mnie_lvl0_part1B[D0,D1,D2,D3]
DistTensor<double> U_mnie_lvl0_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//U_mnie_lvl0_part1T[D0,D1,D2,D3]
DistTensor<double> U_mnie_lvl0_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
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
	//z_ai_lvl1_part1_1[D0,D2,D3,D1]
DistTensor<double> z_ai_lvl1_part1_1__D_0__D_2__D_3__D_1( dist__D_0__D_2__D_3__D_1, g );
	//z_ai_lvl1_part1_1[D0,D2,D1,D3]
DistTensor<double> z_ai_lvl1_part1_1_perm0132__D_0__D_2__D_3__D_1( dist__D_0__D_2__D_1__D_3, g );
z_ai_lvl1_part1_1_perm0132__D_0__D_2__D_3__D_1.SetLocalPermutation( perm_0_1_3_2 );
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
	//ztemp2[D0,D1,D2,D3]
DistTensor<double> ztemp2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//ztemp2_lvl0_part0B[D0,D1,D2,D3]
DistTensor<double> ztemp2_lvl0_part0B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//ztemp2_lvl0_part0T[D0,D1,D2,D3]
DistTensor<double> ztemp2_lvl0_part0T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//ztemp2_lvl0_part1B[D0,D1,D2,D3]
DistTensor<double> ztemp2_lvl0_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//ztemp2_lvl0_part1T[D0,D1,D2,D3]
DistTensor<double> ztemp2_lvl0_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//ztemp2_lvl1_part0B[D0,D1,D2,D3]
DistTensor<double> ztemp2_lvl1_part0B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//ztemp2_lvl1_part0T[D0,D1,D2,D3]
DistTensor<double> ztemp2_lvl1_part0T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//ztemp2_lvl1_part0_0[D0,D1,D2,D3]
DistTensor<double> ztemp2_lvl1_part0_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//ztemp2_lvl1_part0_1[D0,D1,D2,D3]
DistTensor<double> ztemp2_lvl1_part0_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//ztemp2_lvl1_part0_1_lvl1_part1B[D0,D1,D2,D3]
DistTensor<double> ztemp2_lvl1_part0_1_lvl1_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//ztemp2_lvl1_part0_1_lvl1_part1T[D0,D1,D2,D3]
DistTensor<double> ztemp2_lvl1_part0_1_lvl1_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//ztemp2_lvl1_part0_1_lvl2_part1B[D0,D1,D2,D3]
DistTensor<double> ztemp2_lvl1_part0_1_lvl2_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//ztemp2_lvl1_part0_1_lvl2_part1T[D0,D1,D2,D3]
DistTensor<double> ztemp2_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//ztemp2_lvl1_part0_1_lvl2_part1_0[D0,D1,D2,D3]
DistTensor<double> ztemp2_lvl1_part0_1_lvl2_part1_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//ztemp2_lvl1_part0_1_lvl2_part1_1[D0,D1,D2,D3]
DistTensor<double> ztemp2_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//ztemp2_lvl1_part0_1_lvl2_part1_2[D0,D1,D2,D3]
DistTensor<double> ztemp2_lvl1_part0_1_lvl2_part1_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//ztemp2_lvl1_part0_2[D0,D1,D2,D3]
DistTensor<double> ztemp2_lvl1_part0_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//ztemp2_lvl1_part1B[D0,D1,D2,D3]
DistTensor<double> ztemp2_lvl1_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//ztemp2_lvl1_part1T[D0,D1,D2,D3]
DistTensor<double> ztemp2_lvl1_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//ztemp2_lvl1_part1_0[D0,D1,D2,D3]
DistTensor<double> ztemp2_lvl1_part1_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//ztemp2_lvl1_part1_1[D0,D1,D2,D3]
DistTensor<double> ztemp2_lvl1_part1_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> ztemp2_lvl1_part1_1_perm0231__D_0__D_2__D_3__D_1( dist__D_0__D_1__D_2__D_3, g );
ztemp2_lvl1_part1_1_perm0231__D_0__D_2__D_3__D_1.SetLocalPermutation( perm_0_2_3_1 );
	//ztemp2_lvl1_part1_2[D0,D1,D2,D3]
DistTensor<double> ztemp2_lvl1_part1_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//ztemp3[D0,D1,D2,D3]
DistTensor<double> ztemp3__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//ztemp3_lvl0_part2B[D0,D1,D2,D3]
DistTensor<double> ztemp3_lvl0_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//ztemp3_lvl0_part2T[D0,D1,D2,D3]
DistTensor<double> ztemp3_lvl0_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//ztemp3_lvl1_part2B[D0,D1,D2,D3]
DistTensor<double> ztemp3_lvl1_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//ztemp3_lvl1_part2T[D0,D1,D2,D3]
DistTensor<double> ztemp3_lvl1_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//ztemp3_lvl1_part2_0[D0,D1,D2,D3]
DistTensor<double> ztemp3_lvl1_part2_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//ztemp3_lvl1_part2_1[D0,D1,D2,D3]
DistTensor<double> ztemp3_lvl1_part2_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//ztemp3_lvl1_part2_1_lvl1_part3B[D0,D1,D2,D3]
DistTensor<double> ztemp3_lvl1_part2_1_lvl1_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//ztemp3_lvl1_part2_1_lvl1_part3T[D0,D1,D2,D3]
DistTensor<double> ztemp3_lvl1_part2_1_lvl1_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//ztemp3_lvl1_part2_1_lvl2_part3B[D0,D1,D2,D3]
DistTensor<double> ztemp3_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//ztemp3_lvl1_part2_1_lvl2_part3T[D0,D1,D2,D3]
DistTensor<double> ztemp3_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//ztemp3_lvl1_part2_1_lvl2_part3_0[D0,D1,D2,D3]
DistTensor<double> ztemp3_lvl1_part2_1_lvl2_part3_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//ztemp3_lvl1_part2_1_lvl2_part3_1[D0,D1,D2,D3]
DistTensor<double> ztemp3_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> ztemp3_lvl1_part2_1_lvl2_part3_1_perm0231__D_0__D_2__D_3__D_1( dist__D_0__D_1__D_2__D_3, g );
ztemp3_lvl1_part2_1_lvl2_part3_1_perm0231__D_0__D_2__D_3__D_1.SetLocalPermutation( perm_0_2_3_1 );
	//ztemp3_lvl1_part2_1_lvl2_part3_2[D0,D1,D2,D3]
DistTensor<double> ztemp3_lvl1_part2_1_lvl2_part3_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//ztemp3_lvl1_part2_2[D0,D1,D2,D3]
DistTensor<double> ztemp3_lvl1_part2_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//ztemp4[D0,D1,D2,D3]
DistTensor<double> ztemp4__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//ztemp4_lvl0_part1B[D0,D1,D2,D3]
DistTensor<double> ztemp4_lvl0_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//ztemp4_lvl0_part1T[D0,D1,D2,D3]
DistTensor<double> ztemp4_lvl0_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//ztemp4_lvl0_part2B[D0,D1,D2,D3]
DistTensor<double> ztemp4_lvl0_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//ztemp4_lvl0_part2T[D0,D1,D2,D3]
DistTensor<double> ztemp4_lvl0_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//ztemp4_lvl1_part1B[D0,D1,D2,D3]
DistTensor<double> ztemp4_lvl1_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//ztemp4_lvl1_part1T[D0,D1,D2,D3]
DistTensor<double> ztemp4_lvl1_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//ztemp4_lvl1_part1_0[D0,D1,D2,D3]
DistTensor<double> ztemp4_lvl1_part1_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//ztemp4_lvl1_part1_1[D0,D1,D2,D3]
DistTensor<double> ztemp4_lvl1_part1_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//ztemp4_lvl1_part1_1_lvl1_part2B[D0,D1,D2,D3]
DistTensor<double> ztemp4_lvl1_part1_1_lvl1_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//ztemp4_lvl1_part1_1_lvl1_part2T[D0,D1,D2,D3]
DistTensor<double> ztemp4_lvl1_part1_1_lvl1_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//ztemp4_lvl1_part1_1_lvl2_part2B[D0,D1,D2,D3]
DistTensor<double> ztemp4_lvl1_part1_1_lvl2_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//ztemp4_lvl1_part1_1_lvl2_part2T[D0,D1,D2,D3]
DistTensor<double> ztemp4_lvl1_part1_1_lvl2_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//ztemp4_lvl1_part1_1_lvl2_part2_0[D0,D1,D2,D3]
DistTensor<double> ztemp4_lvl1_part1_1_lvl2_part2_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//ztemp4_lvl1_part1_1_lvl2_part2_1[D0,D1,D2,D3]
DistTensor<double> ztemp4_lvl1_part1_1_lvl2_part2_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//ztemp4_lvl1_part1_1_lvl2_part2_2[D0,D1,D2,D3]
DistTensor<double> ztemp4_lvl1_part1_1_lvl2_part2_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//ztemp4_lvl1_part1_2[D0,D1,D2,D3]
DistTensor<double> ztemp4_lvl1_part1_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//ztemp4_lvl1_part2B[D0,D1,D2,D3]
DistTensor<double> ztemp4_lvl1_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//ztemp4_lvl1_part2T[D0,D1,D2,D3]
DistTensor<double> ztemp4_lvl1_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//ztemp4_lvl1_part2_0[D0,D1,D2,D3]
DistTensor<double> ztemp4_lvl1_part2_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//ztemp4_lvl1_part2_1[D0,D1,D2,D3]
DistTensor<double> ztemp4_lvl1_part2_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//ztemp4_lvl1_part2_1_lvl1_part1B[D0,D1,D2,D3]
DistTensor<double> ztemp4_lvl1_part2_1_lvl1_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//ztemp4_lvl1_part2_1_lvl1_part1T[D0,D1,D2,D3]
DistTensor<double> ztemp4_lvl1_part2_1_lvl1_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//ztemp4_lvl1_part2_1_lvl2_part1B[D0,D1,D2,D3]
DistTensor<double> ztemp4_lvl1_part2_1_lvl2_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//ztemp4_lvl1_part2_1_lvl2_part1T[D0,D1,D2,D3]
DistTensor<double> ztemp4_lvl1_part2_1_lvl2_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//ztemp4_lvl1_part2_1_lvl2_part1_0[D0,D1,D2,D3]
DistTensor<double> ztemp4_lvl1_part2_1_lvl2_part1_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//ztemp4_lvl1_part2_1_lvl2_part1_1[D0,D1,D2,D3]
DistTensor<double> ztemp4_lvl1_part2_1_lvl2_part1_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> ztemp4_lvl1_part2_1_lvl2_part1_1_perm0231__D_0__D_2__D_3__D_1( dist__D_0__D_1__D_2__D_3, g );
ztemp4_lvl1_part2_1_lvl2_part1_1_perm0231__D_0__D_2__D_3__D_1.SetLocalPermutation( perm_0_2_3_1 );
	//ztemp4_lvl1_part2_1_lvl2_part1_2[D0,D1,D2,D3]
DistTensor<double> ztemp4_lvl1_part2_1_lvl2_part1_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//ztemp4_lvl1_part2_2[D0,D1,D2,D3]
DistTensor<double> ztemp4_lvl1_part2_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//ztemp5[D0,D1,D2,D3]
DistTensor<double> ztemp5__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//ztemp5_lvl0_part0B[D0,D1,D2,D3]
DistTensor<double> ztemp5_lvl0_part0B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//ztemp5_lvl0_part0T[D0,D1,D2,D3]
DistTensor<double> ztemp5_lvl0_part0T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//ztemp5_lvl0_part2B[D0,D1,D2,D3]
DistTensor<double> ztemp5_lvl0_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//ztemp5_lvl0_part2T[D0,D1,D2,D3]
DistTensor<double> ztemp5_lvl0_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//ztemp5_lvl1_part0B[D0,D1,D2,D3]
DistTensor<double> ztemp5_lvl1_part0B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//ztemp5_lvl1_part0T[D0,D1,D2,D3]
DistTensor<double> ztemp5_lvl1_part0T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//ztemp5_lvl1_part0_0[D0,D1,D2,D3]
DistTensor<double> ztemp5_lvl1_part0_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//ztemp5_lvl1_part0_1[D0,D1,D2,D3]
DistTensor<double> ztemp5_lvl1_part0_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//ztemp5_lvl1_part0_1_lvl1_part1B[D0,D1,D2,D3]
DistTensor<double> ztemp5_lvl1_part0_1_lvl1_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//ztemp5_lvl1_part0_1_lvl1_part1T[D0,D1,D2,D3]
DistTensor<double> ztemp5_lvl1_part0_1_lvl1_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//ztemp5_lvl1_part0_1_lvl2_part1B[D0,D1,D2,D3]
DistTensor<double> ztemp5_lvl1_part0_1_lvl2_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//ztemp5_lvl1_part0_1_lvl2_part1T[D0,D1,D2,D3]
DistTensor<double> ztemp5_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//ztemp5_lvl1_part0_1_lvl2_part1_0[D0,D1,D2,D3]
DistTensor<double> ztemp5_lvl1_part0_1_lvl2_part1_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//ztemp5_lvl1_part0_1_lvl2_part1_1[D0,D1,D2,D3]
DistTensor<double> ztemp5_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//ztemp5_lvl1_part0_1_lvl2_part1_2[D0,D1,D2,D3]
DistTensor<double> ztemp5_lvl1_part0_1_lvl2_part1_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//ztemp5_lvl1_part0_2[D0,D1,D2,D3]
DistTensor<double> ztemp5_lvl1_part0_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//ztemp5_lvl1_part2B[D0,D1,D2,D3]
DistTensor<double> ztemp5_lvl1_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//ztemp5_lvl1_part2T[D0,D1,D2,D3]
DistTensor<double> ztemp5_lvl1_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//ztemp5_lvl1_part2_0[D0,D1,D2,D3]
DistTensor<double> ztemp5_lvl1_part2_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//ztemp5_lvl1_part2_1[D0,D1,D2,D3]
DistTensor<double> ztemp5_lvl1_part2_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//ztemp5_lvl1_part2_1_lvl1_part0B[D0,D1,D2,D3]
DistTensor<double> ztemp5_lvl1_part2_1_lvl1_part0B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//ztemp5_lvl1_part2_1_lvl1_part0T[D0,D1,D2,D3]
DistTensor<double> ztemp5_lvl1_part2_1_lvl1_part0T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//ztemp5_lvl1_part2_1_lvl2_part0B[D0,D1,D2,D3]
DistTensor<double> ztemp5_lvl1_part2_1_lvl2_part0B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//ztemp5_lvl1_part2_1_lvl2_part0T[D0,D1,D2,D3]
DistTensor<double> ztemp5_lvl1_part2_1_lvl2_part0T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//ztemp5_lvl1_part2_1_lvl2_part0_0[D0,D1,D2,D3]
DistTensor<double> ztemp5_lvl1_part2_1_lvl2_part0_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//ztemp5_lvl1_part2_1_lvl2_part0_1[D0,D1,D2,D3]
DistTensor<double> ztemp5_lvl1_part2_1_lvl2_part0_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//ztemp5_lvl1_part2_1_lvl2_part0_1[D0,D3,D2,D1]
DistTensor<double> ztemp5_lvl1_part2_1_lvl2_part0_1__D_0__D_3__D_2__D_1( dist__D_0__D_3__D_2__D_1, g );
	//ztemp5_lvl1_part2_1_lvl2_part0_1[D2,D3,*,D1]
DistTensor<double> ztemp5_lvl1_part2_1_lvl2_part0_1_perm2013__S__D_2__D_3__D_1( dist__D_2__D_3__S__D_1, g );
ztemp5_lvl1_part2_1_lvl2_part0_1_perm2013__S__D_2__D_3__D_1.SetLocalPermutation( perm_2_0_1_3 );
	//ztemp5_lvl1_part2_1_lvl2_part0_2[D0,D1,D2,D3]
DistTensor<double> ztemp5_lvl1_part2_1_lvl2_part0_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//ztemp5_lvl1_part2_2[D0,D1,D2,D3]
DistTensor<double> ztemp5_lvl1_part2_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
// z_ai has 2 dims
ObjShape z_ai__D_0_1__D_2_3_tempShape( 2 );
z_ai__D_0_1__D_2_3_tempShape[ 0 ] = n_v;
z_ai__D_0_1__D_2_3_tempShape[ 1 ] = n_o;
z_ai__D_0_1__D_2_3.ResizeTo( z_ai__D_0_1__D_2_3_tempShape );
MakeUniform( z_ai__D_0_1__D_2_3 );
TensorDistribution dist__S__S__D_2__D_3 = tmen::StringToTensorDist("[(),(),(2),(3)]");
TensorDistribution dist__S__D_2 = tmen::StringToTensorDist("[(),(2)]");
TensorDistribution dist__D_0__S__S__D_2 = tmen::StringToTensorDist("[(0),(),(),(2)]");
TensorDistribution dist__D_0__S__D_2__S = tmen::StringToTensorDist("[(0),(),(2),()]");
TensorDistribution dist__D_0__D_1__S__S__D_2__D_3 = tmen::StringToTensorDist("[(0),(1),(),(),(2),(3)]");
TensorDistribution dist__D_0__D_1__S__D_2 = tmen::StringToTensorDist("[(0),(1),(),(2)]");
TensorDistribution dist__D_1__S__S__D_3 = tmen::StringToTensorDist("[(1),(),(),(3)]");
TensorDistribution dist__D_1__S__D_3__S = tmen::StringToTensorDist("[(1),(),(3),()]");
Permutation perm_0_3_1_2( 4 );
perm_0_3_1_2[0] = 0;
perm_0_3_1_2[1] = 3;
perm_0_3_1_2[2] = 1;
perm_0_3_1_2[3] = 2;
Permutation perm_1_3_0_2( 4 );
perm_1_3_0_2[0] = 1;
perm_1_3_0_2[1] = 3;
perm_1_3_0_2[2] = 0;
perm_1_3_0_2[3] = 2;
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
	//F_ae[D0,*]
DistTensor<double> F_ae__D_0__S( dist__D_0__S, g );
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
	//P_jimb_lvl1_part0_1_lvl2_part2_1[D0,D3,D2,D1]
DistTensor<double> P_jimb_lvl1_part0_1_lvl2_part2_1__D_0__D_3__D_2__D_1( dist__D_0__D_3__D_2__D_1, g );
	//P_jimb_lvl1_part0_1_lvl2_part2_1[D2,D3,*,D1]
DistTensor<double> P_jimb_lvl1_part0_1_lvl2_part2_1_perm0132__D_2__D_3__D_1__S( dist__D_2__D_3__S__D_1, g );
P_jimb_lvl1_part0_1_lvl2_part2_1_perm0132__D_2__D_3__D_1__S.SetLocalPermutation( perm_0_1_3_2 );
	//Q_mnij_lvl1_part0_1_lvl2_part1_1[*,*,D2,D3]
DistTensor<double> Q_mnij_lvl1_part0_1_lvl2_part1_1_perm2301__D_2__D_3__S__S( dist__S__S__D_2__D_3, g );
Q_mnij_lvl1_part0_1_lvl2_part1_1_perm2301__D_2__D_3__S__S.SetLocalPermutation( perm_2_3_0_1 );
	//T_bfnj_lvl0_part1B[D0,D1,D2,D3]
DistTensor<double> T_bfnj_lvl0_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//T_bfnj_lvl0_part1T[D0,D1,D2,D3]
DistTensor<double> T_bfnj_lvl0_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
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
	//T_bfnj_lvl1_part2_1_lvl2_part3_1[*,D1,D2,D3]
DistTensor<double> T_bfnj_lvl1_part2_1_lvl2_part3_1__S__D_1__D_2__D_3( dist__S__D_1__D_2__D_3, g );
	//T_bfnj_lvl1_part3_1_lvl2_part2_1[D0,D1,*,D3]
DistTensor<double> T_bfnj_lvl1_part3_1_lvl2_part2_1_perm2013__S__D_0__D_1__D_3( dist__D_0__D_1__S__D_3, g );
T_bfnj_lvl1_part3_1_lvl2_part2_1_perm2013__S__D_0__D_1__D_3.SetLocalPermutation( perm_2_0_1_3 );
	//Tau_efmn_lvl1_part2_1_lvl2_part3_1[D2,D1,D0,D3]
DistTensor<double> Tau_efmn_lvl1_part2_1_lvl2_part3_1__D_2__D_1__D_0__D_3( dist__D_2__D_1__D_0__D_3, g );
	//Tau_efmn_lvl1_part2_1_lvl2_part3_1[D2,D3,D0,*]
DistTensor<double> Tau_efmn_lvl1_part2_1_lvl2_part3_1__D_2__D_3__D_0__S( dist__D_2__D_3__D_0__S, g );
	//Tau_efmn_lvl1_part2_1_lvl2_part3_1[D2,D3,*,*]
DistTensor<double> Tau_efmn_lvl1_part2_1_lvl2_part3_1__D_2__D_3__S__S( dist__D_2__D_3__S__S, g );
	//Tau_efmn_lvl1_part2_1_lvl2_part3_1[D0,D1,*,*]
DistTensor<double> Tau_efmn_lvl1_part2_1_lvl2_part3_1_perm2301__S__S__D_0__D_1( dist__D_0__D_1__S__S, g );
Tau_efmn_lvl1_part2_1_lvl2_part3_1_perm2301__S__S__D_0__D_1.SetLocalPermutation( perm_2_3_0_1 );
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
	//t_fj_lvl1_part1_1[D0,*]
DistTensor<double> t_fj_lvl1_part1_1__D_0__S( dist__D_0__S, g );
	//y_abef[D0,D1,D2,D3]
DistTensor<double> y_abef__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
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
//END_DECL

//******************************
//* Load tensors
//******************************
////////////////////////////////
//Performance testing
////////////////////////////////
std::stringstream fullName;
#ifdef CORRECTNESS
DistTensor<T> check_Tau(dist__D_0__D_1__D_2__D_3, g);
check_Tau.ResizeTo(Tau_efmn__D_0__D_1__D_2__D_3.Shape());
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
fullName << "ccsd_terms/term_Tau_iter" << testIter;
Read(check_Tau, fullName.str(), BINARY_FLAT, false);
DistTensor<T> check_W(dist__D_0__D_1__D_2__D_3, g);
check_W.ResizeTo(W_bmje__D_0__D_1__D_2__D_3.Shape());
Read(r_bmfe__D_0__D_1__D_2__D_3, "ccsd_terms/term_r_small", BINARY_FLAT, false);
Read(u_mnje__D_0__D_1__D_2__D_3, "ccsd_terms/term_u_small", BINARY_FLAT, false);
Read(v_femn__D_0__D_1__D_2__D_3, "ccsd_terms/term_v_small", BINARY_FLAT, false);
Read(w_bmje__D_0__D_1__D_2__D_3, "ccsd_terms/term_w_small", BINARY_FLAT, false);
Read(x_bmej__D_0__D_1__D_2__D_3, "ccsd_terms/term_x_small", BINARY_FLAT, false);
fullName.str("");
fullName.clear();
fullName << "ccsd_terms/term_W_iter" << testIter;
Read(check_W, fullName.str(), BINARY_FLAT, false);
DistTensor<T> check_X(dist__D_0__D_1__D_2__D_3, g);
check_X.ResizeTo(X_bmej__D_0__D_1__D_2__D_3.Shape());
fullName.str("");
fullName.clear();
fullName << "ccsd_terms/term_X_iter" << testIter;
Read(check_X, fullName.str(), BINARY_FLAT, false);
DistTensor<T> check_U(dist__D_0__D_1__D_2__D_3, g);
check_U.ResizeTo(U_mnie__D_0__D_1__D_2__D_3.Shape());
fullName.str("");
fullName.clear();
fullName << "ccsd_terms/term_U_iter" << testIter;
Read(check_U, fullName.str(), BINARY_FLAT, false);
DistTensor<T> check_Q(dist__D_0__D_1__D_2__D_3, g);
check_Q.ResizeTo(Q_mnij__D_0__D_1__D_2__D_3.Shape());
Read(q_mnij__D_0__D_1__D_2__D_3, "ccsd_terms/term_q_small", BINARY_FLAT, false);
fullName.str("");
fullName.clear();
fullName << "ccsd_terms/term_Q_iter" << testIter;
Read(check_Q, fullName.str(), BINARY_FLAT, false);
DistTensor<T> check_P(dist__D_0__D_1__D_2__D_3, g);
check_P.ResizeTo(P_jimb__D_0__D_1__D_2__D_3.Shape());
fullName.str("");
fullName.clear();
fullName << "ccsd_terms/term_P_iter" << testIter;
Read(check_P, fullName.str(), BINARY_FLAT, false);
DistTensor<T> check_H(dist__D_0_1__D_2_3, g);
check_H.ResizeTo(H_me__D_0_1__D_2_3.Shape());
fullName.str("");
fullName.clear();
fullName << "ccsd_terms/term_H_iter" << testIter;
Read(check_H, fullName.str(), BINARY_FLAT, false);
DistTensor<T> check_F(dist__D_0_1__D_2_3, g);
check_F.ResizeTo(F_ae__D_0_1__D_2_3.Shape());
fullName.str("");
fullName.clear();
fullName << "ccsd_terms/term_F_iter" << testIter;
Read(check_F, fullName.str(), BINARY_FLAT, false);
DistTensor<T> check_G(dist__D_0_1__D_2_3, g);
check_G.ResizeTo(G_mi__D_0_1__D_2_3.Shape());
fullName.str("");
fullName.clear();
fullName << "ccsd_terms/term_G_iter" << testIter;
Read(check_G, fullName.str(), BINARY_FLAT, false);
DistTensor<T> check_z_small(dist__D_0_1__D_2_3, g);
check_z_small.ResizeTo(z_ai__D_0_1__D_2_3.Shape());
fullName.str("");
fullName.clear();
fullName << "ccsd_terms/term_z_small_iter" << testIter;
Read(check_z_small, fullName.str(), BINARY_FLAT, false);
DistTensor<T> check_Z(dist__D_0__D_1__D_2__D_3, g);
check_Z.ResizeTo(Z_abij__D_0__D_1__D_2__D_3.Shape());
Read(y_abef__D_0__D_1__D_2__D_3, "ccsd_terms/term_y_small", BINARY_FLAT, false);
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
	//  Tau_efmn__D_0__D_1__D_2__D_3

	Tau_efmn__D_0__D_1__D_2__D_3 = T_bfnj__D_0__D_1__D_2__D_3;


//****
//**** (out of 1)
//**** Is real	0 shadows
	//Outputs:
	//  Tau_efmn__D_0__D_1__D_2__D_3

	//**** (out of 1)
	//**** Is real	0 shadows
		//Outputs:
		//  Tau_efmn__D_0__D_1__D_2__D_3
	PartitionDown(t_fj__D_0_1__D_2_3, t_fj_lvl1_part1T__D_0_1__D_2_3, t_fj_lvl1_part1B__D_0_1__D_2_3, 1, 0);
	PartitionDown(Tau_efmn__D_0__D_1__D_2__D_3, Tau_efmn_lvl1_part2T__D_0__D_1__D_2__D_3, Tau_efmn_lvl1_part2B__D_0__D_1__D_2__D_3, 2, 0);
	while(Tau_efmn_lvl1_part2T__D_0__D_1__D_2__D_3.Dimension(2) < Tau_efmn__D_0__D_1__D_2__D_3.Dimension(2))
	{
		RepartitionDown
		( t_fj_lvl1_part1T__D_0_1__D_2_3,  t_fj_lvl1_part1_0__D_0_1__D_2_3,
		  /**/ /**/
		       t_fj_lvl1_part1_1__D_0_1__D_2_3,
		  t_fj_lvl1_part1B__D_0_1__D_2_3, t_fj_lvl1_part1_2__D_0_1__D_2_3, 1, blkSize );
		RepartitionDown
		( Tau_efmn_lvl1_part2T__D_0__D_1__D_2__D_3,  Tau_efmn_lvl1_part2_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       Tau_efmn_lvl1_part2_1__D_0__D_1__D_2__D_3,
		  Tau_efmn_lvl1_part2B__D_0__D_1__D_2__D_3, Tau_efmn_lvl1_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );

		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  Tau_efmn_lvl1_part2_1__D_0__D_1__D_2__D_3
		PartitionDown(t_fj__D_0_1__D_2_3, t_fj_lvl2_part1T__D_0_1__D_2_3, t_fj_lvl2_part1B__D_0_1__D_2_3, 1, 0);
		PartitionDown(Tau_efmn_lvl1_part2_1__D_0__D_1__D_2__D_3, Tau_efmn_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3, Tau_efmn_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, 3, 0);
		while(Tau_efmn_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3.Dimension(3) < Tau_efmn_lvl1_part2_1__D_0__D_1__D_2__D_3.Dimension(3))
		{
			RepartitionDown
			( t_fj_lvl2_part1T__D_0_1__D_2_3,  t_fj_lvl2_part1_0__D_0_1__D_2_3,
			  /**/ /**/
			       t_fj_lvl2_part1_1__D_0_1__D_2_3,
			  t_fj_lvl2_part1B__D_0_1__D_2_3, t_fj_lvl2_part1_2__D_0_1__D_2_3, 1, blkSize );
			RepartitionDown
			( Tau_efmn_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3,  Tau_efmn_lvl1_part2_1_lvl2_part3_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       Tau_efmn_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3,
			  Tau_efmn_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, Tau_efmn_lvl1_part2_1_lvl2_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );

			   // t_fj_lvl2_part1_1[D01,D32] <- t_fj_lvl2_part1_1[D01,D23]
			t_fj_lvl2_part1_1__D_0_1__D_3_2.AlignModesWith( modes_0_1, Tau_efmn_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3, modes_1_3 );
			t_fj_lvl2_part1_1__D_0_1__D_3_2.PermutationRedistFrom( t_fj_lvl2_part1_1__D_0_1__D_2_3, modes_2_3 );
			   // t_fj_lvl2_part1_1[D10,D32] <- t_fj_lvl2_part1_1[D01,D32]
			t_fj_lvl2_part1_1__D_1_0__D_3_2.AlignModesWith( modes_0_1, Tau_efmn_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3, modes_1_3 );
			t_fj_lvl2_part1_1__D_1_0__D_3_2.PermutationRedistFrom( t_fj_lvl2_part1_1__D_0_1__D_3_2, modes_0_1 );
			t_fj_lvl2_part1_1__D_0_1__D_3_2.EmptyData();
			   // t_fj_lvl2_part1_1[D1,D3] <- t_fj_lvl2_part1_1[D10,D32]
			t_fj_lvl2_part1_1__D_1__D_3.AlignModesWith( modes_0_1, Tau_efmn_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3, modes_1_3 );
			t_fj_lvl2_part1_1__D_1__D_3.AllGatherRedistFrom( t_fj_lvl2_part1_1__D_1_0__D_3_2, modes_0_2 );
			t_fj_lvl2_part1_1__D_1_0__D_3_2.EmptyData();
			   // t_fj_lvl1_part1_1[D0,D2] <- t_fj_lvl1_part1_1[D01,D23]
			t_fj_lvl1_part1_1__D_0__D_2.AlignModesWith( modes_0_1, Tau_efmn_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3, modes_0_2 );
			t_fj_lvl1_part1_1__D_0__D_2.AllGatherRedistFrom( t_fj_lvl1_part1_1__D_0_1__D_2_3, modes_1_3 );
			Permute( Tau_efmn_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3, Tau_efmn_lvl1_part2_1_lvl2_part3_1_perm0213__D_0__D_2__D_1__D_3 );
			   // 1.0 * t_fj_lvl1_part1_1[D0,D2]_em * t_fj_lvl2_part1_1[D1,D3]_fn + 1.0 * Tau_efmn_lvl1_part2_1_lvl2_part3_1[D0,D1,D2,D3]_emfn
			LocalContractAndLocalEliminate(1.0, t_fj_lvl1_part1_1__D_0__D_2.LockedTensor(), indices_em, false,
				t_fj_lvl2_part1_1__D_1__D_3.LockedTensor(), indices_fn, false,
				1.0, Tau_efmn_lvl1_part2_1_lvl2_part3_1_perm0213__D_0__D_2__D_1__D_3.Tensor(), indices_emfn, false);
			t_fj_lvl1_part1_1__D_0__D_2.EmptyData();
			t_fj_lvl2_part1_1__D_1__D_3.EmptyData();
			Permute( Tau_efmn_lvl1_part2_1_lvl2_part3_1_perm0213__D_0__D_2__D_1__D_3, Tau_efmn_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3 );
			Tau_efmn_lvl1_part2_1_lvl2_part3_1_perm0213__D_0__D_2__D_1__D_3.EmptyData();

			SlidePartitionDown
			( t_fj_lvl2_part1T__D_0_1__D_2_3,  t_fj_lvl2_part1_0__D_0_1__D_2_3,
			       t_fj_lvl2_part1_1__D_0_1__D_2_3,
			  /**/ /**/
			  t_fj_lvl2_part1B__D_0_1__D_2_3, t_fj_lvl2_part1_2__D_0_1__D_2_3, 1 );
			SlidePartitionDown
			( Tau_efmn_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3,  Tau_efmn_lvl1_part2_1_lvl2_part3_0__D_0__D_1__D_2__D_3,
			       Tau_efmn_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  Tau_efmn_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, Tau_efmn_lvl1_part2_1_lvl2_part3_2__D_0__D_1__D_2__D_3, 3 );

		}
		//****

		SlidePartitionDown
		( t_fj_lvl1_part1T__D_0_1__D_2_3,  t_fj_lvl1_part1_0__D_0_1__D_2_3,
		       t_fj_lvl1_part1_1__D_0_1__D_2_3,
		  /**/ /**/
		  t_fj_lvl1_part1B__D_0_1__D_2_3, t_fj_lvl1_part1_2__D_0_1__D_2_3, 1 );
		SlidePartitionDown
		( Tau_efmn_lvl1_part2T__D_0__D_1__D_2__D_3,  Tau_efmn_lvl1_part2_0__D_0__D_1__D_2__D_3,
		       Tau_efmn_lvl1_part2_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  Tau_efmn_lvl1_part2B__D_0__D_1__D_2__D_3, Tau_efmn_lvl1_part2_2__D_0__D_1__D_2__D_3, 2 );

	}
	//****


//****


//**** (out of 1)
//**** Is real	0 shadows
	//Outputs:
	//  W_bmje__D_0__D_1__D_2__D_3

	tempShape = u_mnje__D_0__D_1__D_2__D_3.Shape();
	Wtemp3__D_0__D_1__D_2__D_3.ResizeTo( tempShape );
	//**** (out of 1)
	//**** Is real	0 shadows
		//Outputs:
		//  Wtemp3__D_0__D_1__D_2__D_3
	PartitionDown(u_mnje__D_0__D_1__D_2__D_3, u_mnje_lvl1_part0T__D_0__D_1__D_2__D_3, u_mnje_lvl1_part0B__D_0__D_1__D_2__D_3, 0, 0);
	PartitionDown(u_mnje__D_0__D_1__D_2__D_3, u_mnje_lvl1_part1T__D_0__D_1__D_2__D_3, u_mnje_lvl1_part1B__D_0__D_1__D_2__D_3, 1, 0);
	PartitionDown(Wtemp3__D_0__D_1__D_2__D_3, Wtemp3_lvl1_part0T__D_0__D_1__D_2__D_3, Wtemp3_lvl1_part0B__D_0__D_1__D_2__D_3, 0, 0);
	while(Wtemp3_lvl1_part0T__D_0__D_1__D_2__D_3.Dimension(0) < Wtemp3__D_0__D_1__D_2__D_3.Dimension(0))
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
		( Wtemp3_lvl1_part0T__D_0__D_1__D_2__D_3,  Wtemp3_lvl1_part0_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       Wtemp3_lvl1_part0_1__D_0__D_1__D_2__D_3,
		  Wtemp3_lvl1_part0B__D_0__D_1__D_2__D_3, Wtemp3_lvl1_part0_2__D_0__D_1__D_2__D_3, 0, blkSize );

		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  Wtemp3_lvl1_part0_1__D_0__D_1__D_2__D_3
		PartitionDown(u_mnje_lvl1_part0_1__D_0__D_1__D_2__D_3, u_mnje_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3, u_mnje_lvl1_part0_1_lvl2_part1B__D_0__D_1__D_2__D_3, 1, 0);
		PartitionDown(u_mnje_lvl1_part1_1__D_0__D_1__D_2__D_3, u_mnje_lvl1_part1_1_lvl2_part0T__D_0__D_1__D_2__D_3, u_mnje_lvl1_part1_1_lvl2_part0B__D_0__D_1__D_2__D_3, 0, 0);
		PartitionDown(Wtemp3_lvl1_part0_1__D_0__D_1__D_2__D_3, Wtemp3_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3, Wtemp3_lvl1_part0_1_lvl2_part1B__D_0__D_1__D_2__D_3, 1, 0);
		while(Wtemp3_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3.Dimension(1) < Wtemp3_lvl1_part0_1__D_0__D_1__D_2__D_3.Dimension(1))
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
			( Wtemp3_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3,  Wtemp3_lvl1_part0_1_lvl2_part1_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       Wtemp3_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3,
			  Wtemp3_lvl1_part0_1_lvl2_part1B__D_0__D_1__D_2__D_3, Wtemp3_lvl1_part0_1_lvl2_part1_2__D_0__D_1__D_2__D_3, 1, blkSize );

			   // u_mnje_lvl1_part1_1_lvl2_part0_1[D1,D0,D2,D3] <- u_mnje_lvl1_part1_1_lvl2_part0_1[D0,D1,D2,D3]
			u_mnje_lvl1_part1_1_lvl2_part0_1__D_1__D_0__D_2__D_3.AlignModesWith( modes_0_1_2_3, u_mnje_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3, modes_1_0_2_3 );
			u_mnje_lvl1_part1_1_lvl2_part0_1__D_1__D_0__D_2__D_3.AllToAllRedistFrom( u_mnje_lvl1_part1_1_lvl2_part0_1__D_0__D_1__D_2__D_3, modes_0_1 );
			YAxpPx( -1.0, u_mnje_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3, 2.0, u_mnje_lvl1_part1_1_lvl2_part0_1__D_1__D_0__D_2__D_3, perm_1_0_2_3, Wtemp3_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3 );
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
			( Wtemp3_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3,  Wtemp3_lvl1_part0_1_lvl2_part1_0__D_0__D_1__D_2__D_3,
			       Wtemp3_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  Wtemp3_lvl1_part0_1_lvl2_part1B__D_0__D_1__D_2__D_3, Wtemp3_lvl1_part0_1_lvl2_part1_2__D_0__D_1__D_2__D_3, 1 );

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
		( Wtemp3_lvl1_part0T__D_0__D_1__D_2__D_3,  Wtemp3_lvl1_part0_0__D_0__D_1__D_2__D_3,
		       Wtemp3_lvl1_part0_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  Wtemp3_lvl1_part0B__D_0__D_1__D_2__D_3, Wtemp3_lvl1_part0_2__D_0__D_1__D_2__D_3, 0 );

	}
	//****
	tempShape = v_femn__D_0__D_1__D_2__D_3.Shape();
	Wtemp2__D_0__D_1__D_2__D_3.ResizeTo( tempShape );
	//**** (out of 1)
	//**** Is real	0 shadows
		//Outputs:
		//  Wtemp2__D_0__D_1__D_2__D_3
	PartitionDown(v_femn__D_0__D_1__D_2__D_3, v_femn_lvl1_part2T__D_0__D_1__D_2__D_3, v_femn_lvl1_part2B__D_0__D_1__D_2__D_3, 2, 0);
	PartitionDown(v_femn__D_0__D_1__D_2__D_3, v_femn_lvl1_part3T__D_0__D_1__D_2__D_3, v_femn_lvl1_part3B__D_0__D_1__D_2__D_3, 3, 0);
	PartitionDown(Wtemp2__D_0__D_1__D_2__D_3, Wtemp2_lvl1_part2T__D_0__D_1__D_2__D_3, Wtemp2_lvl1_part2B__D_0__D_1__D_2__D_3, 2, 0);
	while(Wtemp2_lvl1_part2T__D_0__D_1__D_2__D_3.Dimension(2) < Wtemp2__D_0__D_1__D_2__D_3.Dimension(2))
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
		( Wtemp2_lvl1_part2T__D_0__D_1__D_2__D_3,  Wtemp2_lvl1_part2_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       Wtemp2_lvl1_part2_1__D_0__D_1__D_2__D_3,
		  Wtemp2_lvl1_part2B__D_0__D_1__D_2__D_3, Wtemp2_lvl1_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );

		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  Wtemp2_lvl1_part2_1__D_0__D_1__D_2__D_3
		PartitionDown(v_femn_lvl1_part2_1__D_0__D_1__D_2__D_3, v_femn_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3, v_femn_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, 3, 0);
		PartitionDown(v_femn_lvl1_part3_1__D_0__D_1__D_2__D_3, v_femn_lvl1_part3_1_lvl2_part2T__D_0__D_1__D_2__D_3, v_femn_lvl1_part3_1_lvl2_part2B__D_0__D_1__D_2__D_3, 2, 0);
		PartitionDown(Wtemp2_lvl1_part2_1__D_0__D_1__D_2__D_3, Wtemp2_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3, Wtemp2_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, 3, 0);
		while(Wtemp2_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3.Dimension(3) < Wtemp2_lvl1_part2_1__D_0__D_1__D_2__D_3.Dimension(3))
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
			( Wtemp2_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3,  Wtemp2_lvl1_part2_1_lvl2_part3_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       Wtemp2_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3,
			  Wtemp2_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, Wtemp2_lvl1_part2_1_lvl2_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );

			   // v_femn_lvl1_part3_1_lvl2_part2_1[D0,D1,D3,D2] <- v_femn_lvl1_part3_1_lvl2_part2_1[D0,D1,D2,D3]
			v_femn_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_3__D_2.AlignModesWith( modes_0_1_2_3, v_femn_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3, modes_0_1_3_2 );
			v_femn_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_3__D_2.AllToAllRedistFrom( v_femn_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_2__D_3, modes_2_3 );
			YAxpPx( -1.0, v_femn_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3, 2.0, v_femn_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_3__D_2, perm_0_1_3_2, Wtemp2_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3 );
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
			( Wtemp2_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3,  Wtemp2_lvl1_part2_1_lvl2_part3_0__D_0__D_1__D_2__D_3,
			       Wtemp2_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  Wtemp2_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, Wtemp2_lvl1_part2_1_lvl2_part3_2__D_0__D_1__D_2__D_3, 3 );

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
		( Wtemp2_lvl1_part2T__D_0__D_1__D_2__D_3,  Wtemp2_lvl1_part2_0__D_0__D_1__D_2__D_3,
		       Wtemp2_lvl1_part2_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  Wtemp2_lvl1_part2B__D_0__D_1__D_2__D_3, Wtemp2_lvl1_part2_2__D_0__D_1__D_2__D_3, 2 );

	}
	//****
	tempShape = T_bfnj__D_0__D_1__D_2__D_3.Shape();
	Wtemp1__D_0__D_1__D_2__D_3.ResizeTo( tempShape );
	//**** (out of 1)
	//**** Is real	0 shadows
		//Outputs:
		//  Wtemp1__D_0__D_1__D_2__D_3
	PartitionDown(T_bfnj__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part2T__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part2B__D_0__D_1__D_2__D_3, 2, 0);
	PartitionDown(Tau_efmn__D_0__D_1__D_2__D_3, Tau_efmn_lvl1_part2T__D_0__D_1__D_2__D_3, Tau_efmn_lvl1_part2B__D_0__D_1__D_2__D_3, 2, 0);
	PartitionDown(T_bfnj__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part3T__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part3B__D_0__D_1__D_2__D_3, 3, 0);
	PartitionDown(Wtemp1__D_0__D_1__D_2__D_3, Wtemp1_lvl1_part2T__D_0__D_1__D_2__D_3, Wtemp1_lvl1_part2B__D_0__D_1__D_2__D_3, 2, 0);
	while(Wtemp1_lvl1_part2T__D_0__D_1__D_2__D_3.Dimension(2) < Wtemp1__D_0__D_1__D_2__D_3.Dimension(2))
	{
		RepartitionDown
		( T_bfnj_lvl1_part2T__D_0__D_1__D_2__D_3,  T_bfnj_lvl1_part2_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       T_bfnj_lvl1_part2_1__D_0__D_1__D_2__D_3,
		  T_bfnj_lvl1_part2B__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );
		RepartitionDown
		( Tau_efmn_lvl1_part2T__D_0__D_1__D_2__D_3,  Tau_efmn_lvl1_part2_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       Tau_efmn_lvl1_part2_1__D_0__D_1__D_2__D_3,
		  Tau_efmn_lvl1_part2B__D_0__D_1__D_2__D_3, Tau_efmn_lvl1_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );
		RepartitionDown
		( T_bfnj_lvl1_part3T__D_0__D_1__D_2__D_3,  T_bfnj_lvl1_part3_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       T_bfnj_lvl1_part3_1__D_0__D_1__D_2__D_3,
		  T_bfnj_lvl1_part3B__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );
		RepartitionDown
		( Wtemp1_lvl1_part2T__D_0__D_1__D_2__D_3,  Wtemp1_lvl1_part2_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       Wtemp1_lvl1_part2_1__D_0__D_1__D_2__D_3,
		  Wtemp1_lvl1_part2B__D_0__D_1__D_2__D_3, Wtemp1_lvl1_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );

		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  Wtemp1_lvl1_part2_1__D_0__D_1__D_2__D_3
		PartitionDown(T_bfnj_lvl1_part2_1__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, 3, 0);
		PartitionDown(Tau_efmn_lvl1_part2_1__D_0__D_1__D_2__D_3, Tau_efmn_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3, Tau_efmn_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, 3, 0);
		PartitionDown(T_bfnj_lvl1_part3_1__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part3_1_lvl2_part2T__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part3_1_lvl2_part2B__D_0__D_1__D_2__D_3, 2, 0);
		PartitionDown(Wtemp1_lvl1_part2_1__D_0__D_1__D_2__D_3, Wtemp1_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3, Wtemp1_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, 3, 0);
		while(Wtemp1_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3.Dimension(3) < Wtemp1_lvl1_part2_1__D_0__D_1__D_2__D_3.Dimension(3))
		{
			RepartitionDown
			( T_bfnj_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3,  T_bfnj_lvl1_part2_1_lvl2_part3_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       T_bfnj_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3,
			  T_bfnj_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part2_1_lvl2_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );
			RepartitionDown
			( Tau_efmn_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3,  Tau_efmn_lvl1_part2_1_lvl2_part3_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       Tau_efmn_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3,
			  Tau_efmn_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, Tau_efmn_lvl1_part2_1_lvl2_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );
			RepartitionDown
			( T_bfnj_lvl1_part3_1_lvl2_part2T__D_0__D_1__D_2__D_3,  T_bfnj_lvl1_part3_1_lvl2_part2_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       T_bfnj_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_2__D_3,
			  T_bfnj_lvl1_part3_1_lvl2_part2B__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part3_1_lvl2_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );
			RepartitionDown
			( Wtemp1_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3,  Wtemp1_lvl1_part2_1_lvl2_part3_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       Wtemp1_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3,
			  Wtemp1_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, Wtemp1_lvl1_part2_1_lvl2_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );

			   // T_bfnj_lvl1_part3_1_lvl2_part2_1[D0,D1,D3,D2] <- T_bfnj_lvl1_part3_1_lvl2_part2_1[D0,D1,D2,D3]
			T_bfnj_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_3__D_2.AlignModesWith( modes_0_1_2_3, T_bfnj_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3, modes_0_1_3_2 );
			T_bfnj_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_3__D_2.AllToAllRedistFrom( T_bfnj_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_2__D_3, modes_2_3 );
			ZAxpBypPx( 0.5, T_bfnj_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3, -1.0, Tau_efmn_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_3__D_2, perm_0_1_3_2, Wtemp1_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3 );
			T_bfnj_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_3__D_2.EmptyData();

			SlidePartitionDown
			( T_bfnj_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3,  T_bfnj_lvl1_part2_1_lvl2_part3_0__D_0__D_1__D_2__D_3,
			       T_bfnj_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  T_bfnj_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part2_1_lvl2_part3_2__D_0__D_1__D_2__D_3, 3 );
			SlidePartitionDown
			( Tau_efmn_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3,  Tau_efmn_lvl1_part2_1_lvl2_part3_0__D_0__D_1__D_2__D_3,
			       Tau_efmn_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  Tau_efmn_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, Tau_efmn_lvl1_part2_1_lvl2_part3_2__D_0__D_1__D_2__D_3, 3 );
			SlidePartitionDown
			( T_bfnj_lvl1_part3_1_lvl2_part2T__D_0__D_1__D_2__D_3,  T_bfnj_lvl1_part3_1_lvl2_part2_0__D_0__D_1__D_2__D_3,
			       T_bfnj_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  T_bfnj_lvl1_part3_1_lvl2_part2B__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part3_1_lvl2_part2_2__D_0__D_1__D_2__D_3, 2 );
			SlidePartitionDown
			( Wtemp1_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3,  Wtemp1_lvl1_part2_1_lvl2_part3_0__D_0__D_1__D_2__D_3,
			       Wtemp1_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  Wtemp1_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, Wtemp1_lvl1_part2_1_lvl2_part3_2__D_0__D_1__D_2__D_3, 3 );

		}
		//****

		SlidePartitionDown
		( T_bfnj_lvl1_part2T__D_0__D_1__D_2__D_3,  T_bfnj_lvl1_part2_0__D_0__D_1__D_2__D_3,
		       T_bfnj_lvl1_part2_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  T_bfnj_lvl1_part2B__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part2_2__D_0__D_1__D_2__D_3, 2 );
		SlidePartitionDown
		( Tau_efmn_lvl1_part2T__D_0__D_1__D_2__D_3,  Tau_efmn_lvl1_part2_0__D_0__D_1__D_2__D_3,
		       Tau_efmn_lvl1_part2_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  Tau_efmn_lvl1_part2B__D_0__D_1__D_2__D_3, Tau_efmn_lvl1_part2_2__D_0__D_1__D_2__D_3, 2 );
		SlidePartitionDown
		( T_bfnj_lvl1_part3T__D_0__D_1__D_2__D_3,  T_bfnj_lvl1_part3_0__D_0__D_1__D_2__D_3,
		       T_bfnj_lvl1_part3_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  T_bfnj_lvl1_part3B__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part3_2__D_0__D_1__D_2__D_3, 3 );
		SlidePartitionDown
		( Wtemp1_lvl1_part2T__D_0__D_1__D_2__D_3,  Wtemp1_lvl1_part2_0__D_0__D_1__D_2__D_3,
		       Wtemp1_lvl1_part2_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  Wtemp1_lvl1_part2B__D_0__D_1__D_2__D_3, Wtemp1_lvl1_part2_2__D_0__D_1__D_2__D_3, 2 );

	}
	//****
	//**** (out of 1)
	//**** Is real	0 shadows
		//Outputs:
		//  W_bmje__D_0__D_1__D_2__D_3
	PartitionDown(w_bmje__D_0__D_1__D_2__D_3, w_bmje_lvl1_part1T__D_0__D_1__D_2__D_3, w_bmje_lvl1_part1B__D_0__D_1__D_2__D_3, 1, 0);
	PartitionDown(x_bmej__D_0__D_1__D_2__D_3, x_bmej_lvl1_part1T__D_0__D_1__D_2__D_3, x_bmej_lvl1_part1B__D_0__D_1__D_2__D_3, 1, 0);
	PartitionDown(W_bmje__D_0__D_1__D_2__D_3, W_bmje_lvl1_part1T__D_0__D_1__D_2__D_3, W_bmje_lvl1_part1B__D_0__D_1__D_2__D_3, 1, 0);
	while(W_bmje_lvl1_part1T__D_0__D_1__D_2__D_3.Dimension(1) < W_bmje__D_0__D_1__D_2__D_3.Dimension(1))
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
		( W_bmje_lvl1_part1T__D_0__D_1__D_2__D_3,  W_bmje_lvl1_part1_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       W_bmje_lvl1_part1_1__D_0__D_1__D_2__D_3,
		  W_bmje_lvl1_part1B__D_0__D_1__D_2__D_3, W_bmje_lvl1_part1_2__D_0__D_1__D_2__D_3, 1, blkSize );

		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  W_bmje_lvl1_part1_1__D_0__D_1__D_2__D_3
		PartitionDown(w_bmje_lvl1_part1_1__D_0__D_1__D_2__D_3, w_bmje_lvl1_part1_1_lvl2_part2T__D_0__D_1__D_2__D_3, w_bmje_lvl1_part1_1_lvl2_part2B__D_0__D_1__D_2__D_3, 2, 0);
		PartitionDown(x_bmej_lvl1_part1_1__D_0__D_1__D_2__D_3, x_bmej_lvl1_part1_1_lvl2_part3T__D_0__D_1__D_2__D_3, x_bmej_lvl1_part1_1_lvl2_part3B__D_0__D_1__D_2__D_3, 3, 0);
		PartitionDown(W_bmje_lvl1_part1_1__D_0__D_1__D_2__D_3, W_bmje_lvl1_part1_1_lvl2_part2T__D_0__D_1__D_2__D_3, W_bmje_lvl1_part1_1_lvl2_part2B__D_0__D_1__D_2__D_3, 2, 0);
		while(W_bmje_lvl1_part1_1_lvl2_part2T__D_0__D_1__D_2__D_3.Dimension(2) < W_bmje_lvl1_part1_1__D_0__D_1__D_2__D_3.Dimension(2))
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
			( W_bmje_lvl1_part1_1_lvl2_part2T__D_0__D_1__D_2__D_3,  W_bmje_lvl1_part1_1_lvl2_part2_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       W_bmje_lvl1_part1_1_lvl2_part2_1__D_0__D_1__D_2__D_3,
			  W_bmje_lvl1_part1_1_lvl2_part2B__D_0__D_1__D_2__D_3, W_bmje_lvl1_part1_1_lvl2_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );

			   // x_bmej_lvl1_part1_1_lvl2_part3_1[D0,D1,D3,D2] <- x_bmej_lvl1_part1_1_lvl2_part3_1[D0,D1,D2,D3]
			x_bmej_lvl1_part1_1_lvl2_part3_1__D_0__D_1__D_3__D_2.AlignModesWith( modes_0_1_2_3, w_bmje_lvl1_part1_1_lvl2_part2_1__D_0__D_1__D_2__D_3, modes_0_1_3_2 );
			x_bmej_lvl1_part1_1_lvl2_part3_1__D_0__D_1__D_3__D_2.AllToAllRedistFrom( x_bmej_lvl1_part1_1_lvl2_part3_1__D_0__D_1__D_2__D_3, modes_2_3 );
			YAxpPx( 2.0, w_bmje_lvl1_part1_1_lvl2_part2_1__D_0__D_1__D_2__D_3, -1.0, x_bmej_lvl1_part1_1_lvl2_part3_1__D_0__D_1__D_3__D_2, perm_0_1_3_2, W_bmje_lvl1_part1_1_lvl2_part2_1__D_0__D_1__D_2__D_3 );
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
			( W_bmje_lvl1_part1_1_lvl2_part2T__D_0__D_1__D_2__D_3,  W_bmje_lvl1_part1_1_lvl2_part2_0__D_0__D_1__D_2__D_3,
			       W_bmje_lvl1_part1_1_lvl2_part2_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  W_bmje_lvl1_part1_1_lvl2_part2B__D_0__D_1__D_2__D_3, W_bmje_lvl1_part1_1_lvl2_part2_2__D_0__D_1__D_2__D_3, 2 );

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
		( W_bmje_lvl1_part1T__D_0__D_1__D_2__D_3,  W_bmje_lvl1_part1_0__D_0__D_1__D_2__D_3,
		       W_bmje_lvl1_part1_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  W_bmje_lvl1_part1B__D_0__D_1__D_2__D_3, W_bmje_lvl1_part1_2__D_0__D_1__D_2__D_3, 1 );

	}
	//****
	Permute( Wtemp2__D_0__D_1__D_2__D_3, Wtemp2_perm1203__D_1__D_2__D_0__D_3 );
	Wtemp2__D_0__D_1__D_2__D_3.EmptyData();
	//**** (out of 1)
	//**** Is real	0 shadows
		//Outputs:
		//  W_bmje__D_0__D_1__D_2__D_3
	PartitionDown(Wtemp1__D_0__D_1__D_2__D_3, Wtemp1_lvl1_part0T__D_0__D_1__D_2__D_3, Wtemp1_lvl1_part0B__D_0__D_1__D_2__D_3, 0, 0);
	PartitionDown(W_bmje__D_0__D_1__D_2__D_3, W_bmje_lvl1_part0T__D_0__D_1__D_2__D_3, W_bmje_lvl1_part0B__D_0__D_1__D_2__D_3, 0, 0);
	while(W_bmje_lvl1_part0T__D_0__D_1__D_2__D_3.Dimension(0) < W_bmje__D_0__D_1__D_2__D_3.Dimension(0))
	{
		RepartitionDown
		( Wtemp1_lvl1_part0T__D_0__D_1__D_2__D_3,  Wtemp1_lvl1_part0_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       Wtemp1_lvl1_part0_1__D_0__D_1__D_2__D_3,
		  Wtemp1_lvl1_part0B__D_0__D_1__D_2__D_3, Wtemp1_lvl1_part0_2__D_0__D_1__D_2__D_3, 0, blkSize );
		RepartitionDown
		( W_bmje_lvl1_part0T__D_0__D_1__D_2__D_3,  W_bmje_lvl1_part0_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       W_bmje_lvl1_part0_1__D_0__D_1__D_2__D_3,
		  W_bmje_lvl1_part0B__D_0__D_1__D_2__D_3, W_bmje_lvl1_part0_2__D_0__D_1__D_2__D_3, 0, blkSize );

		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  W_bmje_lvl1_part0_1__D_0__D_1__D_2__D_3
		PartitionDown(Wtemp1_lvl1_part0_1__D_0__D_1__D_2__D_3, Wtemp1_lvl1_part0_1_lvl2_part3T__D_0__D_1__D_2__D_3, Wtemp1_lvl1_part0_1_lvl2_part3B__D_0__D_1__D_2__D_3, 3, 0);
		PartitionDown(W_bmje_lvl1_part0_1__D_0__D_1__D_2__D_3, W_bmje_lvl1_part0_1_lvl2_part2T__D_0__D_1__D_2__D_3, W_bmje_lvl1_part0_1_lvl2_part2B__D_0__D_1__D_2__D_3, 2, 0);
		while(W_bmje_lvl1_part0_1_lvl2_part2T__D_0__D_1__D_2__D_3.Dimension(2) < W_bmje_lvl1_part0_1__D_0__D_1__D_2__D_3.Dimension(2))
		{
			RepartitionDown
			( Wtemp1_lvl1_part0_1_lvl2_part3T__D_0__D_1__D_2__D_3,  Wtemp1_lvl1_part0_1_lvl2_part3_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       Wtemp1_lvl1_part0_1_lvl2_part3_1__D_0__D_1__D_2__D_3,
			  Wtemp1_lvl1_part0_1_lvl2_part3B__D_0__D_1__D_2__D_3, Wtemp1_lvl1_part0_1_lvl2_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );
			RepartitionDown
			( W_bmje_lvl1_part0_1_lvl2_part2T__D_0__D_1__D_2__D_3,  W_bmje_lvl1_part0_1_lvl2_part2_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       W_bmje_lvl1_part0_1_lvl2_part2_1__D_0__D_1__D_2__D_3,
			  W_bmje_lvl1_part0_1_lvl2_part2B__D_0__D_1__D_2__D_3, W_bmje_lvl1_part0_1_lvl2_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );

			   // Wtemp1_lvl1_part0_1_lvl2_part3_1[D1,D0,D2,D3] <- Wtemp1_lvl1_part0_1_lvl2_part3_1[D0,D1,D2,D3]
			Wtemp1_lvl1_part0_1_lvl2_part3_1__D_1__D_0__D_2__D_3.AlignModesWith( modes_1_2, Wtemp2__D_0__D_1__D_2__D_3, modes_0_3 );
			Wtemp1_lvl1_part0_1_lvl2_part3_1__D_1__D_0__D_2__D_3.AllToAllRedistFrom( Wtemp1_lvl1_part0_1_lvl2_part3_1__D_0__D_1__D_2__D_3, modes_0_1 );
			   // Wtemp1_lvl1_part0_1_lvl2_part3_1[D1,D0,D3,*] <- Wtemp1_lvl1_part0_1_lvl2_part3_1[D1,D0,D2,D3]
			Wtemp1_lvl1_part0_1_lvl2_part3_1__D_1__D_0__D_3__S.AlignModesWith( modes_1_2, Wtemp2__D_0__D_1__D_2__D_3, modes_0_3 );
			Wtemp1_lvl1_part0_1_lvl2_part3_1__D_1__D_0__D_3__S.AllToAllRedistFrom( Wtemp1_lvl1_part0_1_lvl2_part3_1__D_1__D_0__D_2__D_3, modes_2_3 );
			Wtemp1_lvl1_part0_1_lvl2_part3_1__D_1__D_0__D_2__D_3.EmptyData();
			   // Wtemp1_lvl1_part0_1_lvl2_part3_1[*,D0,D3,*] <- Wtemp1_lvl1_part0_1_lvl2_part3_1[D1,D0,D3,*]
			Wtemp1_lvl1_part0_1_lvl2_part3_1_perm1203__D_0__D_3__S__S.AlignModesWith( modes_1_2, Wtemp2__D_0__D_1__D_2__D_3, modes_0_3 );
			Wtemp1_lvl1_part0_1_lvl2_part3_1_perm1203__D_0__D_3__S__S.AllGatherRedistFrom( Wtemp1_lvl1_part0_1_lvl2_part3_1__D_1__D_0__D_3__S, modes_1 );
			Wtemp1_lvl1_part0_1_lvl2_part3_1__D_1__D_0__D_3__S.EmptyData();
			W_bmje_lvl1_part0_1_lvl2_part2_1_perm310245__D_1__D_2__S__S__D_0__D_3.AlignModesWith( modes_1_3, Wtemp2__D_0__D_1__D_2__D_3, modes_2_1 );
			tempShape = W_bmje_lvl1_part0_1_lvl2_part2_1__D_0__D_1__D_2__D_3.Shape();
			tempShape.push_back( g.Shape()[0] );
			tempShape.push_back( g.Shape()[3] );
			W_bmje_lvl1_part0_1_lvl2_part2_1_perm310245__D_1__D_2__S__S__D_0__D_3.ResizeTo( tempShape );
			   // 1.0 * Wtemp2[D0,D1,D2,D3]_emfn * Wtemp1_lvl1_part0_1_lvl2_part3_1[*,D0,D3,*]_fnbj + 0.0 * W_bmje_lvl1_part0_1_lvl2_part2_1[*,D2,*,D1,D0,D3]_embjfn
			LocalContract(1.0, Wtemp2_perm1203__D_1__D_2__D_0__D_3.LockedTensor(), indices_emfn, false,
				Wtemp1_lvl1_part0_1_lvl2_part3_1_perm1203__D_0__D_3__S__S.LockedTensor(), indices_fnbj, false,
				0.0, W_bmje_lvl1_part0_1_lvl2_part2_1_perm310245__D_1__D_2__S__S__D_0__D_3.Tensor(), indices_embjfn, false);
			Wtemp1_lvl1_part0_1_lvl2_part3_1_perm1203__D_0__D_3__S__S.EmptyData();
			W_bmje_lvl1_part0_1_lvl2_part2_1_temp__D_0__D_2__S__D_1_3.AlignModesWith( modes_0_1_2_3, W_bmje_lvl1_part0_1_lvl2_part2_1_temp__D_0__D_1__D_2__D_3, modes_0_1_2_3 );
			tempShape = W_bmje_lvl1_part0_1_lvl2_part2_1__D_0__D_1__D_2__D_3.Shape();
			W_bmje_lvl1_part0_1_lvl2_part2_1_temp__D_0__D_2__S__D_1_3.ResizeTo( tempShape );
			   // W_bmje_lvl1_part0_1_lvl2_part2_1_temp[D0,D2,*,D13] <- W_bmje_lvl1_part0_1_lvl2_part2_1[*,D2,*,D1,D0,D3] (with SumScatter on (D0)(D3))
			W_bmje_lvl1_part0_1_lvl2_part2_1_temp__D_0__D_2__S__D_1_3.ReduceScatterRedistFrom( W_bmje_lvl1_part0_1_lvl2_part2_1_perm310245__D_1__D_2__S__S__D_0__D_3, modes_5_4 );
			W_bmje_lvl1_part0_1_lvl2_part2_1_perm310245__D_1__D_2__S__S__D_0__D_3.EmptyData();
			   // W_bmje_lvl1_part0_1_lvl2_part2_1_temp[D0,*,D2,D13] <- W_bmje_lvl1_part0_1_lvl2_part2_1_temp[D0,D2,*,D13]
			W_bmje_lvl1_part0_1_lvl2_part2_1_temp__D_0__S__D_2__D_1_3.AlignModesWith( modes_0_1_2_3, W_bmje_lvl1_part0_1_lvl2_part2_1__D_0__D_1__D_2__D_3, modes_0_1_2_3 );
			W_bmje_lvl1_part0_1_lvl2_part2_1_temp__D_0__S__D_2__D_1_3.AllToAllRedistFrom( W_bmje_lvl1_part0_1_lvl2_part2_1_temp__D_0__D_2__S__D_1_3, modes_2 );
			W_bmje_lvl1_part0_1_lvl2_part2_1_temp__D_0__D_2__S__D_1_3.EmptyData();
			   // W_bmje_lvl1_part0_1_lvl2_part2_1_temp[D0,D1,D2,D3] <- W_bmje_lvl1_part0_1_lvl2_part2_1_temp[D0,*,D2,D13]
			W_bmje_lvl1_part0_1_lvl2_part2_1_temp__D_0__D_1__D_2__D_3.AlignModesWith( modes_0_1_2_3, W_bmje_lvl1_part0_1_lvl2_part2_1__D_0__D_1__D_2__D_3, modes_0_1_2_3 );
			W_bmje_lvl1_part0_1_lvl2_part2_1_temp__D_0__D_1__D_2__D_3.AllToAllRedistFrom( W_bmje_lvl1_part0_1_lvl2_part2_1_temp__D_0__S__D_2__D_1_3, modes_1_3 );
			W_bmje_lvl1_part0_1_lvl2_part2_1_temp__D_0__S__D_2__D_1_3.EmptyData();
			YxpBy( W_bmje_lvl1_part0_1_lvl2_part2_1_temp__D_0__D_1__D_2__D_3, 1.0, W_bmje_lvl1_part0_1_lvl2_part2_1__D_0__D_1__D_2__D_3 );
			W_bmje_lvl1_part0_1_lvl2_part2_1_temp__D_0__D_1__D_2__D_3.EmptyData();

			SlidePartitionDown
			( Wtemp1_lvl1_part0_1_lvl2_part3T__D_0__D_1__D_2__D_3,  Wtemp1_lvl1_part0_1_lvl2_part3_0__D_0__D_1__D_2__D_3,
			       Wtemp1_lvl1_part0_1_lvl2_part3_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  Wtemp1_lvl1_part0_1_lvl2_part3B__D_0__D_1__D_2__D_3, Wtemp1_lvl1_part0_1_lvl2_part3_2__D_0__D_1__D_2__D_3, 3 );
			SlidePartitionDown
			( W_bmje_lvl1_part0_1_lvl2_part2T__D_0__D_1__D_2__D_3,  W_bmje_lvl1_part0_1_lvl2_part2_0__D_0__D_1__D_2__D_3,
			       W_bmje_lvl1_part0_1_lvl2_part2_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  W_bmje_lvl1_part0_1_lvl2_part2B__D_0__D_1__D_2__D_3, W_bmje_lvl1_part0_1_lvl2_part2_2__D_0__D_1__D_2__D_3, 2 );

		}
		//****

		SlidePartitionDown
		( Wtemp1_lvl1_part0T__D_0__D_1__D_2__D_3,  Wtemp1_lvl1_part0_0__D_0__D_1__D_2__D_3,
		       Wtemp1_lvl1_part0_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  Wtemp1_lvl1_part0B__D_0__D_1__D_2__D_3, Wtemp1_lvl1_part0_2__D_0__D_1__D_2__D_3, 0 );
		SlidePartitionDown
		( W_bmje_lvl1_part0T__D_0__D_1__D_2__D_3,  W_bmje_lvl1_part0_0__D_0__D_1__D_2__D_3,
		       W_bmje_lvl1_part0_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  W_bmje_lvl1_part0B__D_0__D_1__D_2__D_3, W_bmje_lvl1_part0_2__D_0__D_1__D_2__D_3, 0 );

	}
	//****
	Wtemp2_perm1203__D_1__D_2__D_0__D_3.EmptyData();
	Wtemp1__D_0__D_1__D_2__D_3.EmptyData();
	//**** (out of 1)
	//**** Is real	0 shadows
		//Outputs:
		//  W_bmje__D_0__D_1__D_2__D_3
	PartitionDown(Wtemp3__D_0__D_1__D_2__D_3, Wtemp3_lvl1_part0T__D_0__D_1__D_2__D_3, Wtemp3_lvl1_part0B__D_0__D_1__D_2__D_3, 0, 0);
	PartitionDown(W_bmje__D_0__D_1__D_2__D_3, W_bmje_lvl1_part1T__D_0__D_1__D_2__D_3, W_bmje_lvl1_part1B__D_0__D_1__D_2__D_3, 1, 0);
	while(W_bmje_lvl1_part1T__D_0__D_1__D_2__D_3.Dimension(1) < W_bmje__D_0__D_1__D_2__D_3.Dimension(1))
	{
		RepartitionDown
		( Wtemp3_lvl1_part0T__D_0__D_1__D_2__D_3,  Wtemp3_lvl1_part0_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       Wtemp3_lvl1_part0_1__D_0__D_1__D_2__D_3,
		  Wtemp3_lvl1_part0B__D_0__D_1__D_2__D_3, Wtemp3_lvl1_part0_2__D_0__D_1__D_2__D_3, 0, blkSize );
		RepartitionDown
		( W_bmje_lvl1_part1T__D_0__D_1__D_2__D_3,  W_bmje_lvl1_part1_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       W_bmje_lvl1_part1_1__D_0__D_1__D_2__D_3,
		  W_bmje_lvl1_part1B__D_0__D_1__D_2__D_3, W_bmje_lvl1_part1_2__D_0__D_1__D_2__D_3, 1, blkSize );

		Permute( W_bmje_lvl1_part1_1__D_0__D_1__D_2__D_3, W_bmje_lvl1_part1_1_perm1230__D_1__D_2__D_3__D_0 );
		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  W_bmje_lvl1_part1_1_perm1230__D_1__D_2__D_3__D_0
		PartitionDown(Wtemp3_lvl1_part0_1__D_0__D_1__D_2__D_3, Wtemp3_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3, Wtemp3_lvl1_part0_1_lvl2_part1B__D_0__D_1__D_2__D_3, 1, 0);
		PartitionDown(t_fj__D_0_1__D_2_3, t_fj_lvl2_part1T__D_0_1__D_2_3, t_fj_lvl2_part1B__D_0_1__D_2_3, 1, 0);
		while(Wtemp3_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3.Dimension(1) < Wtemp3_lvl1_part0_1__D_0__D_1__D_2__D_3.Dimension(1))
		{
			RepartitionDown
			( Wtemp3_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3,  Wtemp3_lvl1_part0_1_lvl2_part1_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       Wtemp3_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3,
			  Wtemp3_lvl1_part0_1_lvl2_part1B__D_0__D_1__D_2__D_3, Wtemp3_lvl1_part0_1_lvl2_part1_2__D_0__D_1__D_2__D_3, 1, blkSize );
			RepartitionDown
			( t_fj_lvl2_part1T__D_0_1__D_2_3,  t_fj_lvl2_part1_0__D_0_1__D_2_3,
			  /**/ /**/
			       t_fj_lvl2_part1_1__D_0_1__D_2_3,
			  t_fj_lvl2_part1B__D_0_1__D_2_3, t_fj_lvl2_part1_2__D_0_1__D_2_3, 1, blkSize );

			   // Wtemp3_lvl1_part0_1_lvl2_part1_1[D1,*,D2,D3] <- Wtemp3_lvl1_part0_1_lvl2_part1_1[D0,D1,D2,D3]
			Wtemp3_lvl1_part0_1_lvl2_part1_1_perm0231__D_1__D_2__D_3__S.AlignModesWith( modes_0_2_3, W_bmje_lvl1_part1_1__D_0__D_1__D_2__D_3, modes_1_2_3 );
			Wtemp3_lvl1_part0_1_lvl2_part1_1_perm0231__D_1__D_2__D_3__S.AllToAllRedistFrom( Wtemp3_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3, modes_0_1 );
			   // t_fj_lvl2_part1_1[D0,*] <- t_fj_lvl2_part1_1[D01,D23]
			t_fj_lvl2_part1_1_perm10__S__D_0.AlignModesWith( modes_0, W_bmje_lvl1_part1_1__D_0__D_1__D_2__D_3, modes_0 );
			t_fj_lvl2_part1_1_perm10__S__D_0.AllGatherRedistFrom( t_fj_lvl2_part1_1__D_0_1__D_2_3, modes_1_2_3 );
			   // -1.0 * Wtemp3_lvl1_part0_1_lvl2_part1_1[D1,*,D2,D3]_mjen * t_fj_lvl2_part1_1[D0,*]_nb + 1.0 * W_bmje_lvl1_part1_1[D0,D1,D2,D3]_mjeb
			LocalContractAndLocalEliminate(-1.0, Wtemp3_lvl1_part0_1_lvl2_part1_1_perm0231__D_1__D_2__D_3__S.LockedTensor(), indices_mjen, false,
				t_fj_lvl2_part1_1_perm10__S__D_0.LockedTensor(), indices_nb, false,
				1.0, W_bmje_lvl1_part1_1_perm1230__D_1__D_2__D_3__D_0.Tensor(), indices_mjeb, false);
			Wtemp3_lvl1_part0_1_lvl2_part1_1_perm0231__D_1__D_2__D_3__S.EmptyData();
			t_fj_lvl2_part1_1_perm10__S__D_0.EmptyData();

			SlidePartitionDown
			( Wtemp3_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3,  Wtemp3_lvl1_part0_1_lvl2_part1_0__D_0__D_1__D_2__D_3,
			       Wtemp3_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  Wtemp3_lvl1_part0_1_lvl2_part1B__D_0__D_1__D_2__D_3, Wtemp3_lvl1_part0_1_lvl2_part1_2__D_0__D_1__D_2__D_3, 1 );
			SlidePartitionDown
			( t_fj_lvl2_part1T__D_0_1__D_2_3,  t_fj_lvl2_part1_0__D_0_1__D_2_3,
			       t_fj_lvl2_part1_1__D_0_1__D_2_3,
			  /**/ /**/
			  t_fj_lvl2_part1B__D_0_1__D_2_3, t_fj_lvl2_part1_2__D_0_1__D_2_3, 1 );

		}
		//****
		Permute( W_bmje_lvl1_part1_1_perm1230__D_1__D_2__D_3__D_0, W_bmje_lvl1_part1_1__D_0__D_1__D_2__D_3 );
		W_bmje_lvl1_part1_1_perm1230__D_1__D_2__D_3__D_0.EmptyData();

		SlidePartitionDown
		( Wtemp3_lvl1_part0T__D_0__D_1__D_2__D_3,  Wtemp3_lvl1_part0_0__D_0__D_1__D_2__D_3,
		       Wtemp3_lvl1_part0_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  Wtemp3_lvl1_part0B__D_0__D_1__D_2__D_3, Wtemp3_lvl1_part0_2__D_0__D_1__D_2__D_3, 0 );
		SlidePartitionDown
		( W_bmje_lvl1_part1T__D_0__D_1__D_2__D_3,  W_bmje_lvl1_part1_0__D_0__D_1__D_2__D_3,
		       W_bmje_lvl1_part1_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  W_bmje_lvl1_part1B__D_0__D_1__D_2__D_3, W_bmje_lvl1_part1_2__D_0__D_1__D_2__D_3, 1 );

	}
	//****
	Wtemp3__D_0__D_1__D_2__D_3.EmptyData();
	tempShape = r_bmfe__D_0__D_1__D_2__D_3.Shape();
	Wtemp4__D_0__D_1__D_2__D_3.ResizeTo( tempShape );
	//**** (out of 1)
	//**** Is real	0 shadows
		//Outputs:
		//  Wtemp4__D_0__D_1__D_2__D_3
	PartitionDown(r_bmfe__D_0__D_1__D_2__D_3, r_bmfe_lvl1_part0T__D_0__D_1__D_2__D_3, r_bmfe_lvl1_part0B__D_0__D_1__D_2__D_3, 0, 0);
	PartitionDown(Wtemp4__D_0__D_1__D_2__D_3, Wtemp4_lvl1_part0T__D_0__D_1__D_2__D_3, Wtemp4_lvl1_part0B__D_0__D_1__D_2__D_3, 0, 0);
	while(Wtemp4_lvl1_part0T__D_0__D_1__D_2__D_3.Dimension(0) < Wtemp4__D_0__D_1__D_2__D_3.Dimension(0))
	{
		RepartitionDown
		( r_bmfe_lvl1_part0T__D_0__D_1__D_2__D_3,  r_bmfe_lvl1_part0_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       r_bmfe_lvl1_part0_1__D_0__D_1__D_2__D_3,
		  r_bmfe_lvl1_part0B__D_0__D_1__D_2__D_3, r_bmfe_lvl1_part0_2__D_0__D_1__D_2__D_3, 0, blkSize );
		RepartitionDown
		( Wtemp4_lvl1_part0T__D_0__D_1__D_2__D_3,  Wtemp4_lvl1_part0_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       Wtemp4_lvl1_part0_1__D_0__D_1__D_2__D_3,
		  Wtemp4_lvl1_part0B__D_0__D_1__D_2__D_3, Wtemp4_lvl1_part0_2__D_0__D_1__D_2__D_3, 0, blkSize );

		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  Wtemp4_lvl1_part0_1__D_0__D_1__D_2__D_3
		PartitionDown(r_bmfe_lvl1_part0_1__D_0__D_1__D_2__D_3, r_bmfe_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3, r_bmfe_lvl1_part0_1_lvl2_part1B__D_0__D_1__D_2__D_3, 1, 0);
		PartitionDown(Wtemp4_lvl1_part0_1__D_0__D_1__D_2__D_3, Wtemp4_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3, Wtemp4_lvl1_part0_1_lvl2_part1B__D_0__D_1__D_2__D_3, 1, 0);
		while(Wtemp4_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3.Dimension(1) < Wtemp4_lvl1_part0_1__D_0__D_1__D_2__D_3.Dimension(1))
		{
			RepartitionDown
			( r_bmfe_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3,  r_bmfe_lvl1_part0_1_lvl2_part1_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       r_bmfe_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3,
			  r_bmfe_lvl1_part0_1_lvl2_part1B__D_0__D_1__D_2__D_3, r_bmfe_lvl1_part0_1_lvl2_part1_2__D_0__D_1__D_2__D_3, 1, blkSize );
			RepartitionDown
			( Wtemp4_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3,  Wtemp4_lvl1_part0_1_lvl2_part1_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       Wtemp4_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3,
			  Wtemp4_lvl1_part0_1_lvl2_part1B__D_0__D_1__D_2__D_3, Wtemp4_lvl1_part0_1_lvl2_part1_2__D_0__D_1__D_2__D_3, 1, blkSize );

			   // r_bmfe_lvl1_part0_1_lvl2_part1_1[D0,D1,D3,D2] <- r_bmfe_lvl1_part0_1_lvl2_part1_1[D0,D1,D2,D3]
			r_bmfe_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_3__D_2.AlignModesWith( modes_0_1_2_3, r_bmfe_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3, modes_0_1_3_2 );
			r_bmfe_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_3__D_2.AllToAllRedistFrom( r_bmfe_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3, modes_2_3 );
			YAxpPx( -1.0, r_bmfe_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3, 2.0, r_bmfe_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_3__D_2, perm_0_1_3_2, Wtemp4_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3 );
			r_bmfe_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_3__D_2.EmptyData();

			SlidePartitionDown
			( r_bmfe_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3,  r_bmfe_lvl1_part0_1_lvl2_part1_0__D_0__D_1__D_2__D_3,
			       r_bmfe_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  r_bmfe_lvl1_part0_1_lvl2_part1B__D_0__D_1__D_2__D_3, r_bmfe_lvl1_part0_1_lvl2_part1_2__D_0__D_1__D_2__D_3, 1 );
			SlidePartitionDown
			( Wtemp4_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3,  Wtemp4_lvl1_part0_1_lvl2_part1_0__D_0__D_1__D_2__D_3,
			       Wtemp4_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  Wtemp4_lvl1_part0_1_lvl2_part1B__D_0__D_1__D_2__D_3, Wtemp4_lvl1_part0_1_lvl2_part1_2__D_0__D_1__D_2__D_3, 1 );

		}
		//****

		SlidePartitionDown
		( r_bmfe_lvl1_part0T__D_0__D_1__D_2__D_3,  r_bmfe_lvl1_part0_0__D_0__D_1__D_2__D_3,
		       r_bmfe_lvl1_part0_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  r_bmfe_lvl1_part0B__D_0__D_1__D_2__D_3, r_bmfe_lvl1_part0_2__D_0__D_1__D_2__D_3, 0 );
		SlidePartitionDown
		( Wtemp4_lvl1_part0T__D_0__D_1__D_2__D_3,  Wtemp4_lvl1_part0_0__D_0__D_1__D_2__D_3,
		       Wtemp4_lvl1_part0_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  Wtemp4_lvl1_part0B__D_0__D_1__D_2__D_3, Wtemp4_lvl1_part0_2__D_0__D_1__D_2__D_3, 0 );

	}
	//****
	//**** (out of 1)
	//**** Is real	0 shadows
		//Outputs:
		//  W_bmje__D_0__D_1__D_2__D_3
	PartitionDown(Wtemp4__D_0__D_1__D_2__D_3, Wtemp4_lvl1_part1T__D_0__D_1__D_2__D_3, Wtemp4_lvl1_part1B__D_0__D_1__D_2__D_3, 1, 0);
	PartitionDown(W_bmje__D_0__D_1__D_2__D_3, W_bmje_lvl1_part1T__D_0__D_1__D_2__D_3, W_bmje_lvl1_part1B__D_0__D_1__D_2__D_3, 1, 0);
	while(W_bmje_lvl1_part1T__D_0__D_1__D_2__D_3.Dimension(1) < W_bmje__D_0__D_1__D_2__D_3.Dimension(1))
	{
		RepartitionDown
		( Wtemp4_lvl1_part1T__D_0__D_1__D_2__D_3,  Wtemp4_lvl1_part1_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       Wtemp4_lvl1_part1_1__D_0__D_1__D_2__D_3,
		  Wtemp4_lvl1_part1B__D_0__D_1__D_2__D_3, Wtemp4_lvl1_part1_2__D_0__D_1__D_2__D_3, 1, blkSize );
		RepartitionDown
		( W_bmje_lvl1_part1T__D_0__D_1__D_2__D_3,  W_bmje_lvl1_part1_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       W_bmje_lvl1_part1_1__D_0__D_1__D_2__D_3,
		  W_bmje_lvl1_part1B__D_0__D_1__D_2__D_3, W_bmje_lvl1_part1_2__D_0__D_1__D_2__D_3, 1, blkSize );

		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  W_bmje_lvl1_part1_1__D_0__D_1__D_2__D_3
		PartitionDown(t_fj__D_0_1__D_2_3, t_fj_lvl2_part1T__D_0_1__D_2_3, t_fj_lvl2_part1B__D_0_1__D_2_3, 1, 0);
		PartitionDown(W_bmje_lvl1_part1_1__D_0__D_1__D_2__D_3, W_bmje_lvl1_part1_1_lvl2_part2T__D_0__D_1__D_2__D_3, W_bmje_lvl1_part1_1_lvl2_part2B__D_0__D_1__D_2__D_3, 2, 0);
		while(W_bmje_lvl1_part1_1_lvl2_part2T__D_0__D_1__D_2__D_3.Dimension(2) < W_bmje_lvl1_part1_1__D_0__D_1__D_2__D_3.Dimension(2))
		{
			RepartitionDown
			( t_fj_lvl2_part1T__D_0_1__D_2_3,  t_fj_lvl2_part1_0__D_0_1__D_2_3,
			  /**/ /**/
			       t_fj_lvl2_part1_1__D_0_1__D_2_3,
			  t_fj_lvl2_part1B__D_0_1__D_2_3, t_fj_lvl2_part1_2__D_0_1__D_2_3, 1, blkSize );
			RepartitionDown
			( W_bmje_lvl1_part1_1_lvl2_part2T__D_0__D_1__D_2__D_3,  W_bmje_lvl1_part1_1_lvl2_part2_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       W_bmje_lvl1_part1_1_lvl2_part2_1__D_0__D_1__D_2__D_3,
			  W_bmje_lvl1_part1_1_lvl2_part2B__D_0__D_1__D_2__D_3, W_bmje_lvl1_part1_1_lvl2_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );

			   // t_fj_lvl2_part1_1[D01,D2] <- t_fj_lvl2_part1_1[D01,D23]
			t_fj_lvl2_part1_1__D_0_1__D_2.AlignModesWith( modes_0, Wtemp4_lvl1_part1_1__D_0__D_1__D_2__D_3, modes_3 );
			t_fj_lvl2_part1_1__D_0_1__D_2.AllGatherRedistFrom( t_fj_lvl2_part1_1__D_0_1__D_2_3, modes_3 );
			   // t_fj_lvl2_part1_1[D013,D2] <- t_fj_lvl2_part1_1[D01,D2]
			t_fj_lvl2_part1_1__D_0_1_3__D_2.AlignModesWith( modes_0, Wtemp4_lvl1_part1_1__D_0__D_1__D_2__D_3, modes_3 );
			t_fj_lvl2_part1_1__D_0_1_3__D_2.LocalRedistFrom( t_fj_lvl2_part1_1__D_0_1__D_2 );
			t_fj_lvl2_part1_1__D_0_1__D_2.EmptyData();
			   // t_fj_lvl2_part1_1[D301,D2] <- t_fj_lvl2_part1_1[D013,D2]
			t_fj_lvl2_part1_1__D_3_0_1__D_2.AlignModesWith( modes_0, Wtemp4_lvl1_part1_1__D_0__D_1__D_2__D_3, modes_3 );
			t_fj_lvl2_part1_1__D_3_0_1__D_2.PermutationRedistFrom( t_fj_lvl2_part1_1__D_0_1_3__D_2, modes_0_1_3 );
			t_fj_lvl2_part1_1__D_0_1_3__D_2.EmptyData();
			   // t_fj_lvl2_part1_1[D3,*] <- t_fj_lvl2_part1_1[D301,D2]
			t_fj_lvl2_part1_1__D_3__S.AlignModesWith( modes_0, Wtemp4_lvl1_part1_1__D_0__D_1__D_2__D_3, modes_3 );
			t_fj_lvl2_part1_1__D_3__S.AllGatherRedistFrom( t_fj_lvl2_part1_1__D_3_0_1__D_2, modes_0_1_2 );
			t_fj_lvl2_part1_1__D_3_0_1__D_2.EmptyData();
			W_bmje_lvl1_part1_1_lvl2_part2_1_perm01324__D_0__D_1__D_2__S__D_3.AlignModesWith( modes_0_1_3, Wtemp4_lvl1_part1_1__D_0__D_1__D_2__D_3, modes_0_1_2 );
			tempShape = W_bmje_lvl1_part1_1_lvl2_part2_1__D_0__D_1__D_2__D_3.Shape();
			tempShape.push_back( g.Shape()[3] );
			W_bmje_lvl1_part1_1_lvl2_part2_1_perm01324__D_0__D_1__D_2__S__D_3.ResizeTo( tempShape );
			   // 1.0 * Wtemp4_lvl1_part1_1[D0,D1,D2,D3]_bmef * t_fj_lvl2_part1_1[D3,*]_fj + 0.0 * W_bmje_lvl1_part1_1_lvl2_part2_1[D0,D1,*,D2,D3]_bmejf
			LocalContract(1.0, Wtemp4_lvl1_part1_1__D_0__D_1__D_2__D_3.LockedTensor(), indices_bmef, false,
				t_fj_lvl2_part1_1__D_3__S.LockedTensor(), indices_fj, false,
				0.0, W_bmje_lvl1_part1_1_lvl2_part2_1_perm01324__D_0__D_1__D_2__S__D_3.Tensor(), indices_bmejf, false);
			t_fj_lvl2_part1_1__D_3__S.EmptyData();
			W_bmje_lvl1_part1_1_lvl2_part2_1_temp__D_0__D_1__S__D_2_3.AlignModesWith( modes_0_1_2_3, W_bmje_lvl1_part1_1_lvl2_part2_1_temp__D_0__D_1__D_2__D_3, modes_0_1_2_3 );
			tempShape = W_bmje_lvl1_part1_1_lvl2_part2_1__D_0__D_1__D_2__D_3.Shape();
			W_bmje_lvl1_part1_1_lvl2_part2_1_temp__D_0__D_1__S__D_2_3.ResizeTo( tempShape );
			   // W_bmje_lvl1_part1_1_lvl2_part2_1_temp[D0,D1,*,D23] <- W_bmje_lvl1_part1_1_lvl2_part2_1[D0,D1,*,D2,D3] (with SumScatter on D3)
			W_bmje_lvl1_part1_1_lvl2_part2_1_temp__D_0__D_1__S__D_2_3.ReduceScatterRedistFrom( W_bmje_lvl1_part1_1_lvl2_part2_1_perm01324__D_0__D_1__D_2__S__D_3, 4 );
			W_bmje_lvl1_part1_1_lvl2_part2_1_perm01324__D_0__D_1__D_2__S__D_3.EmptyData();
			   // W_bmje_lvl1_part1_1_lvl2_part2_1_temp[D0,D1,D2,D3] <- W_bmje_lvl1_part1_1_lvl2_part2_1_temp[D0,D1,*,D23]
			W_bmje_lvl1_part1_1_lvl2_part2_1_temp__D_0__D_1__D_2__D_3.AlignModesWith( modes_0_1_2_3, W_bmje_lvl1_part1_1_lvl2_part2_1__D_0__D_1__D_2__D_3, modes_0_1_2_3 );
			W_bmje_lvl1_part1_1_lvl2_part2_1_temp__D_0__D_1__D_2__D_3.AllToAllRedistFrom( W_bmje_lvl1_part1_1_lvl2_part2_1_temp__D_0__D_1__S__D_2_3, modes_2_3 );
			W_bmje_lvl1_part1_1_lvl2_part2_1_temp__D_0__D_1__S__D_2_3.EmptyData();
			YxpBy( W_bmje_lvl1_part1_1_lvl2_part2_1_temp__D_0__D_1__D_2__D_3, 1.0, W_bmje_lvl1_part1_1_lvl2_part2_1__D_0__D_1__D_2__D_3 );
			W_bmje_lvl1_part1_1_lvl2_part2_1_temp__D_0__D_1__D_2__D_3.EmptyData();

			SlidePartitionDown
			( t_fj_lvl2_part1T__D_0_1__D_2_3,  t_fj_lvl2_part1_0__D_0_1__D_2_3,
			       t_fj_lvl2_part1_1__D_0_1__D_2_3,
			  /**/ /**/
			  t_fj_lvl2_part1B__D_0_1__D_2_3, t_fj_lvl2_part1_2__D_0_1__D_2_3, 1 );
			SlidePartitionDown
			( W_bmje_lvl1_part1_1_lvl2_part2T__D_0__D_1__D_2__D_3,  W_bmje_lvl1_part1_1_lvl2_part2_0__D_0__D_1__D_2__D_3,
			       W_bmje_lvl1_part1_1_lvl2_part2_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  W_bmje_lvl1_part1_1_lvl2_part2B__D_0__D_1__D_2__D_3, W_bmje_lvl1_part1_1_lvl2_part2_2__D_0__D_1__D_2__D_3, 2 );

		}
		//****

		SlidePartitionDown
		( Wtemp4_lvl1_part1T__D_0__D_1__D_2__D_3,  Wtemp4_lvl1_part1_0__D_0__D_1__D_2__D_3,
		       Wtemp4_lvl1_part1_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  Wtemp4_lvl1_part1B__D_0__D_1__D_2__D_3, Wtemp4_lvl1_part1_2__D_0__D_1__D_2__D_3, 1 );
		SlidePartitionDown
		( W_bmje_lvl1_part1T__D_0__D_1__D_2__D_3,  W_bmje_lvl1_part1_0__D_0__D_1__D_2__D_3,
		       W_bmje_lvl1_part1_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  W_bmje_lvl1_part1B__D_0__D_1__D_2__D_3, W_bmje_lvl1_part1_2__D_0__D_1__D_2__D_3, 1 );

	}
	//****
	Wtemp4__D_0__D_1__D_2__D_3.EmptyData();


//****


//**** (out of 1)
//**** Is real	0 shadows
	//Outputs:
	//  X_bmej__D_0__D_1__D_2__D_3

	X_bmej__D_0__D_1__D_2__D_3 = x_bmej__D_0__D_1__D_2__D_3;


//****
//**** (out of 1)
//**** Is real	0 shadows
	//Outputs:
	//  Xtemp1__D_0__D_1__D_2__D_3

	tempShape = T_bfnj__D_0__D_1__D_2__D_3.Shape();
	Xtemp1__D_0__D_1__D_2__D_3.ResizeTo( tempShape );
	ZAxpBy( 1.0, Tau_efmn__D_0__D_1__D_2__D_3, -0.5, T_bfnj__D_0__D_1__D_2__D_3, Xtemp1__D_0__D_1__D_2__D_3 );


//****
//**** (out of 1)
//**** Is real	0 shadows
	//Outputs:
	//  X_bmej__D_0__D_1__D_2__D_3

	Permute( v_femn__D_0__D_1__D_2__D_3, v_femn_perm1203__D_1__D_2__D_0__D_3 );
	//**** (out of 1)
	//**** Is real	0 shadows
		//Outputs:
		//  X_bmej__D_0__D_1__D_2__D_3
	PartitionDown(Xtemp1__D_0__D_1__D_2__D_3, Xtemp1_lvl1_part0T__D_0__D_1__D_2__D_3, Xtemp1_lvl1_part0B__D_0__D_1__D_2__D_3, 0, 0);
	PartitionDown(X_bmej__D_0__D_1__D_2__D_3, X_bmej_lvl1_part0T__D_0__D_1__D_2__D_3, X_bmej_lvl1_part0B__D_0__D_1__D_2__D_3, 0, 0);
	while(X_bmej_lvl1_part0T__D_0__D_1__D_2__D_3.Dimension(0) < X_bmej__D_0__D_1__D_2__D_3.Dimension(0))
	{
		RepartitionDown
		( Xtemp1_lvl1_part0T__D_0__D_1__D_2__D_3,  Xtemp1_lvl1_part0_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       Xtemp1_lvl1_part0_1__D_0__D_1__D_2__D_3,
		  Xtemp1_lvl1_part0B__D_0__D_1__D_2__D_3, Xtemp1_lvl1_part0_2__D_0__D_1__D_2__D_3, 0, blkSize );
		RepartitionDown
		( X_bmej_lvl1_part0T__D_0__D_1__D_2__D_3,  X_bmej_lvl1_part0_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       X_bmej_lvl1_part0_1__D_0__D_1__D_2__D_3,
		  X_bmej_lvl1_part0B__D_0__D_1__D_2__D_3, X_bmej_lvl1_part0_2__D_0__D_1__D_2__D_3, 0, blkSize );

		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  X_bmej_lvl1_part0_1__D_0__D_1__D_2__D_3
		PartitionDown(Xtemp1_lvl1_part0_1__D_0__D_1__D_2__D_3, Xtemp1_lvl1_part0_1_lvl2_part3T__D_0__D_1__D_2__D_3, Xtemp1_lvl1_part0_1_lvl2_part3B__D_0__D_1__D_2__D_3, 3, 0);
		PartitionDown(X_bmej_lvl1_part0_1__D_0__D_1__D_2__D_3, X_bmej_lvl1_part0_1_lvl2_part3T__D_0__D_1__D_2__D_3, X_bmej_lvl1_part0_1_lvl2_part3B__D_0__D_1__D_2__D_3, 3, 0);
		while(X_bmej_lvl1_part0_1_lvl2_part3T__D_0__D_1__D_2__D_3.Dimension(3) < X_bmej_lvl1_part0_1__D_0__D_1__D_2__D_3.Dimension(3))
		{
			RepartitionDown
			( Xtemp1_lvl1_part0_1_lvl2_part3T__D_0__D_1__D_2__D_3,  Xtemp1_lvl1_part0_1_lvl2_part3_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       Xtemp1_lvl1_part0_1_lvl2_part3_1__D_0__D_1__D_2__D_3,
			  Xtemp1_lvl1_part0_1_lvl2_part3B__D_0__D_1__D_2__D_3, Xtemp1_lvl1_part0_1_lvl2_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );
			RepartitionDown
			( X_bmej_lvl1_part0_1_lvl2_part3T__D_0__D_1__D_2__D_3,  X_bmej_lvl1_part0_1_lvl2_part3_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       X_bmej_lvl1_part0_1_lvl2_part3_1__D_0__D_1__D_2__D_3,
			  X_bmej_lvl1_part0_1_lvl2_part3B__D_0__D_1__D_2__D_3, X_bmej_lvl1_part0_1_lvl2_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );

			   // Xtemp1_lvl1_part0_1_lvl2_part3_1[D1,D0,D2,D3] <- Xtemp1_lvl1_part0_1_lvl2_part3_1[D0,D1,D2,D3]
			Xtemp1_lvl1_part0_1_lvl2_part3_1__D_1__D_0__D_2__D_3.AlignModesWith( modes_1_2, v_femn__D_0__D_1__D_2__D_3, modes_0_3 );
			Xtemp1_lvl1_part0_1_lvl2_part3_1__D_1__D_0__D_2__D_3.AllToAllRedistFrom( Xtemp1_lvl1_part0_1_lvl2_part3_1__D_0__D_1__D_2__D_3, modes_0_1 );
			   // Xtemp1_lvl1_part0_1_lvl2_part3_1[D1,D0,D3,*] <- Xtemp1_lvl1_part0_1_lvl2_part3_1[D1,D0,D2,D3]
			Xtemp1_lvl1_part0_1_lvl2_part3_1__D_1__D_0__D_3__S.AlignModesWith( modes_1_2, v_femn__D_0__D_1__D_2__D_3, modes_0_3 );
			Xtemp1_lvl1_part0_1_lvl2_part3_1__D_1__D_0__D_3__S.AllToAllRedistFrom( Xtemp1_lvl1_part0_1_lvl2_part3_1__D_1__D_0__D_2__D_3, modes_2_3 );
			Xtemp1_lvl1_part0_1_lvl2_part3_1__D_1__D_0__D_2__D_3.EmptyData();
			   // Xtemp1_lvl1_part0_1_lvl2_part3_1[*,D0,D3,*] <- Xtemp1_lvl1_part0_1_lvl2_part3_1[D1,D0,D3,*]
			Xtemp1_lvl1_part0_1_lvl2_part3_1_perm1203__D_0__D_3__S__S.AlignModesWith( modes_1_2, v_femn__D_0__D_1__D_2__D_3, modes_0_3 );
			Xtemp1_lvl1_part0_1_lvl2_part3_1_perm1203__D_0__D_3__S__S.AllGatherRedistFrom( Xtemp1_lvl1_part0_1_lvl2_part3_1__D_1__D_0__D_3__S, modes_1 );
			Xtemp1_lvl1_part0_1_lvl2_part3_1__D_1__D_0__D_3__S.EmptyData();
			X_bmej_lvl1_part0_1_lvl2_part3_1_perm210345__D_1__D_2__S__S__D_0__D_3.AlignModesWith( modes_1_2, v_femn__D_0__D_1__D_2__D_3, modes_2_1 );
			tempShape = X_bmej_lvl1_part0_1_lvl2_part3_1__D_0__D_1__D_2__D_3.Shape();
			tempShape.push_back( g.Shape()[0] );
			tempShape.push_back( g.Shape()[3] );
			X_bmej_lvl1_part0_1_lvl2_part3_1_perm210345__D_1__D_2__S__S__D_0__D_3.ResizeTo( tempShape );
			   // -1.0 * v_femn[D0,D1,D2,D3]_emfn * Xtemp1_lvl1_part0_1_lvl2_part3_1[*,D0,D3,*]_fnbj + 0.0 * X_bmej_lvl1_part0_1_lvl2_part3_1[*,D2,D1,*,D0,D3]_embjfn
			LocalContract(-1.0, v_femn_perm1203__D_1__D_2__D_0__D_3.LockedTensor(), indices_emfn, false,
				Xtemp1_lvl1_part0_1_lvl2_part3_1_perm1203__D_0__D_3__S__S.LockedTensor(), indices_fnbj, false,
				0.0, X_bmej_lvl1_part0_1_lvl2_part3_1_perm210345__D_1__D_2__S__S__D_0__D_3.Tensor(), indices_embjfn, false);
			Xtemp1_lvl1_part0_1_lvl2_part3_1_perm1203__D_0__D_3__S__S.EmptyData();
			X_bmej_lvl1_part0_1_lvl2_part3_1_temp__D_0__D_2__D_1__D_3.AlignModesWith( modes_0_1_2_3, X_bmej_lvl1_part0_1_lvl2_part3_1_temp__D_0__D_1__D_2__D_3, modes_0_1_2_3 );
			tempShape = X_bmej_lvl1_part0_1_lvl2_part3_1__D_0__D_1__D_2__D_3.Shape();
			X_bmej_lvl1_part0_1_lvl2_part3_1_temp__D_0__D_2__D_1__D_3.ResizeTo( tempShape );
			   // X_bmej_lvl1_part0_1_lvl2_part3_1_temp[D0,D2,D1,D3] <- X_bmej_lvl1_part0_1_lvl2_part3_1[*,D2,D1,*,D0,D3] (with SumScatter on (D0)(D3))
			X_bmej_lvl1_part0_1_lvl2_part3_1_temp__D_0__D_2__D_1__D_3.ReduceScatterRedistFrom( X_bmej_lvl1_part0_1_lvl2_part3_1_perm210345__D_1__D_2__S__S__D_0__D_3, modes_5_4 );
			X_bmej_lvl1_part0_1_lvl2_part3_1_perm210345__D_1__D_2__S__S__D_0__D_3.EmptyData();
			   // X_bmej_lvl1_part0_1_lvl2_part3_1_temp[D0,D1,D2,D3] <- X_bmej_lvl1_part0_1_lvl2_part3_1_temp[D0,D2,D1,D3]
			X_bmej_lvl1_part0_1_lvl2_part3_1_temp__D_0__D_1__D_2__D_3.AlignModesWith( modes_0_1_2_3, X_bmej_lvl1_part0_1_lvl2_part3_1__D_0__D_1__D_2__D_3, modes_0_1_2_3 );
			X_bmej_lvl1_part0_1_lvl2_part3_1_temp__D_0__D_1__D_2__D_3.AllToAllRedistFrom( X_bmej_lvl1_part0_1_lvl2_part3_1_temp__D_0__D_2__D_1__D_3, modes_1_2 );
			X_bmej_lvl1_part0_1_lvl2_part3_1_temp__D_0__D_2__D_1__D_3.EmptyData();
			YxpBy( X_bmej_lvl1_part0_1_lvl2_part3_1_temp__D_0__D_1__D_2__D_3, 1.0, X_bmej_lvl1_part0_1_lvl2_part3_1__D_0__D_1__D_2__D_3 );
			X_bmej_lvl1_part0_1_lvl2_part3_1_temp__D_0__D_1__D_2__D_3.EmptyData();

			SlidePartitionDown
			( Xtemp1_lvl1_part0_1_lvl2_part3T__D_0__D_1__D_2__D_3,  Xtemp1_lvl1_part0_1_lvl2_part3_0__D_0__D_1__D_2__D_3,
			       Xtemp1_lvl1_part0_1_lvl2_part3_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  Xtemp1_lvl1_part0_1_lvl2_part3B__D_0__D_1__D_2__D_3, Xtemp1_lvl1_part0_1_lvl2_part3_2__D_0__D_1__D_2__D_3, 3 );
			SlidePartitionDown
			( X_bmej_lvl1_part0_1_lvl2_part3T__D_0__D_1__D_2__D_3,  X_bmej_lvl1_part0_1_lvl2_part3_0__D_0__D_1__D_2__D_3,
			       X_bmej_lvl1_part0_1_lvl2_part3_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  X_bmej_lvl1_part0_1_lvl2_part3B__D_0__D_1__D_2__D_3, X_bmej_lvl1_part0_1_lvl2_part3_2__D_0__D_1__D_2__D_3, 3 );

		}
		//****

		SlidePartitionDown
		( Xtemp1_lvl1_part0T__D_0__D_1__D_2__D_3,  Xtemp1_lvl1_part0_0__D_0__D_1__D_2__D_3,
		       Xtemp1_lvl1_part0_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  Xtemp1_lvl1_part0B__D_0__D_1__D_2__D_3, Xtemp1_lvl1_part0_2__D_0__D_1__D_2__D_3, 0 );
		SlidePartitionDown
		( X_bmej_lvl1_part0T__D_0__D_1__D_2__D_3,  X_bmej_lvl1_part0_0__D_0__D_1__D_2__D_3,
		       X_bmej_lvl1_part0_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  X_bmej_lvl1_part0B__D_0__D_1__D_2__D_3, X_bmej_lvl1_part0_2__D_0__D_1__D_2__D_3, 0 );

	}
	//****
	v_femn_perm1203__D_1__D_2__D_0__D_3.EmptyData();
	Xtemp1__D_0__D_1__D_2__D_3.EmptyData();
	//**** (out of 1)
	//**** Is real	0 shadows
		//Outputs:
		//  X_bmej__D_0__D_1__D_2__D_3
	PartitionDown(u_mnje__D_0__D_1__D_2__D_3, u_mnje_lvl1_part0T__D_0__D_1__D_2__D_3, u_mnje_lvl1_part0B__D_0__D_1__D_2__D_3, 0, 0);
	PartitionDown(X_bmej__D_0__D_1__D_2__D_3, X_bmej_lvl1_part1T__D_0__D_1__D_2__D_3, X_bmej_lvl1_part1B__D_0__D_1__D_2__D_3, 1, 0);
	while(X_bmej_lvl1_part1T__D_0__D_1__D_2__D_3.Dimension(1) < X_bmej__D_0__D_1__D_2__D_3.Dimension(1))
	{
		RepartitionDown
		( u_mnje_lvl1_part0T__D_0__D_1__D_2__D_3,  u_mnje_lvl1_part0_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       u_mnje_lvl1_part0_1__D_0__D_1__D_2__D_3,
		  u_mnje_lvl1_part0B__D_0__D_1__D_2__D_3, u_mnje_lvl1_part0_2__D_0__D_1__D_2__D_3, 0, blkSize );
		RepartitionDown
		( X_bmej_lvl1_part1T__D_0__D_1__D_2__D_3,  X_bmej_lvl1_part1_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       X_bmej_lvl1_part1_1__D_0__D_1__D_2__D_3,
		  X_bmej_lvl1_part1B__D_0__D_1__D_2__D_3, X_bmej_lvl1_part1_2__D_0__D_1__D_2__D_3, 1, blkSize );

		Permute( X_bmej_lvl1_part1_1__D_0__D_1__D_2__D_3, X_bmej_lvl1_part1_1_perm1320__D_1__D_3__D_2__D_0 );
		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  X_bmej_lvl1_part1_1_perm1320__D_1__D_3__D_2__D_0
		PartitionDown(u_mnje_lvl1_part0_1__D_0__D_1__D_2__D_3, u_mnje_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3, u_mnje_lvl1_part0_1_lvl2_part1B__D_0__D_1__D_2__D_3, 1, 0);
		PartitionDown(t_fj__D_0_1__D_2_3, t_fj_lvl2_part1T__D_0_1__D_2_3, t_fj_lvl2_part1B__D_0_1__D_2_3, 1, 0);
		while(u_mnje_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3.Dimension(1) < u_mnje_lvl1_part0_1__D_0__D_1__D_2__D_3.Dimension(1))
		{
			RepartitionDown
			( u_mnje_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3,  u_mnje_lvl1_part0_1_lvl2_part1_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       u_mnje_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3,
			  u_mnje_lvl1_part0_1_lvl2_part1B__D_0__D_1__D_2__D_3, u_mnje_lvl1_part0_1_lvl2_part1_2__D_0__D_1__D_2__D_3, 1, blkSize );
			RepartitionDown
			( t_fj_lvl2_part1T__D_0_1__D_2_3,  t_fj_lvl2_part1_0__D_0_1__D_2_3,
			  /**/ /**/
			       t_fj_lvl2_part1_1__D_0_1__D_2_3,
			  t_fj_lvl2_part1B__D_0_1__D_2_3, t_fj_lvl2_part1_2__D_0_1__D_2_3, 1, blkSize );

			   // t_fj_lvl2_part1_1[D0,*] <- t_fj_lvl2_part1_1[D01,D23]
			t_fj_lvl2_part1_1_perm10__S__D_0.AlignModesWith( modes_0, X_bmej_lvl1_part1_1__D_0__D_1__D_2__D_3, modes_0 );
			t_fj_lvl2_part1_1_perm10__S__D_0.AllGatherRedistFrom( t_fj_lvl2_part1_1__D_0_1__D_2_3, modes_1_2_3 );
			   // u_mnje_lvl1_part0_1_lvl2_part1_1[D0,D1,D3,D2] <- u_mnje_lvl1_part0_1_lvl2_part1_1[D0,D1,D2,D3]
			u_mnje_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_3__D_2.AlignModesWith( modes_0_2_3, X_bmej_lvl1_part1_1__D_0__D_1__D_2__D_3, modes_1_3_2 );
			u_mnje_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_3__D_2.AllToAllRedistFrom( u_mnje_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3, modes_2_3 );
			   // u_mnje_lvl1_part0_1_lvl2_part1_1[D1,*,D3,D2] <- u_mnje_lvl1_part0_1_lvl2_part1_1[D0,D1,D3,D2]
			u_mnje_lvl1_part0_1_lvl2_part1_1_perm0231__D_1__D_3__D_2__S.AlignModesWith( modes_0_2_3, X_bmej_lvl1_part1_1__D_0__D_1__D_2__D_3, modes_1_3_2 );
			u_mnje_lvl1_part0_1_lvl2_part1_1_perm0231__D_1__D_3__D_2__S.AllToAllRedistFrom( u_mnje_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_3__D_2, modes_0_1 );
			u_mnje_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_3__D_2.EmptyData();
			   // -1.0 * u_mnje_lvl1_part0_1_lvl2_part1_1[D1,*,D3,D2]_mjen * t_fj_lvl2_part1_1[D0,*]_nb + 1.0 * X_bmej_lvl1_part1_1[D0,D1,D2,D3]_mjeb
			LocalContractAndLocalEliminate(-1.0, u_mnje_lvl1_part0_1_lvl2_part1_1_perm0231__D_1__D_3__D_2__S.LockedTensor(), indices_mjen, false,
				t_fj_lvl2_part1_1_perm10__S__D_0.LockedTensor(), indices_nb, false,
				1.0, X_bmej_lvl1_part1_1_perm1320__D_1__D_3__D_2__D_0.Tensor(), indices_mjeb, false);
			u_mnje_lvl1_part0_1_lvl2_part1_1_perm0231__D_1__D_3__D_2__S.EmptyData();
			t_fj_lvl2_part1_1_perm10__S__D_0.EmptyData();

			SlidePartitionDown
			( u_mnje_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3,  u_mnje_lvl1_part0_1_lvl2_part1_0__D_0__D_1__D_2__D_3,
			       u_mnje_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  u_mnje_lvl1_part0_1_lvl2_part1B__D_0__D_1__D_2__D_3, u_mnje_lvl1_part0_1_lvl2_part1_2__D_0__D_1__D_2__D_3, 1 );
			SlidePartitionDown
			( t_fj_lvl2_part1T__D_0_1__D_2_3,  t_fj_lvl2_part1_0__D_0_1__D_2_3,
			       t_fj_lvl2_part1_1__D_0_1__D_2_3,
			  /**/ /**/
			  t_fj_lvl2_part1B__D_0_1__D_2_3, t_fj_lvl2_part1_2__D_0_1__D_2_3, 1 );

		}
		//****
		Permute( X_bmej_lvl1_part1_1_perm1320__D_1__D_3__D_2__D_0, X_bmej_lvl1_part1_1__D_0__D_1__D_2__D_3 );
		X_bmej_lvl1_part1_1_perm1320__D_1__D_3__D_2__D_0.EmptyData();

		SlidePartitionDown
		( u_mnje_lvl1_part0T__D_0__D_1__D_2__D_3,  u_mnje_lvl1_part0_0__D_0__D_1__D_2__D_3,
		       u_mnje_lvl1_part0_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  u_mnje_lvl1_part0B__D_0__D_1__D_2__D_3, u_mnje_lvl1_part0_2__D_0__D_1__D_2__D_3, 0 );
		SlidePartitionDown
		( X_bmej_lvl1_part1T__D_0__D_1__D_2__D_3,  X_bmej_lvl1_part1_0__D_0__D_1__D_2__D_3,
		       X_bmej_lvl1_part1_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  X_bmej_lvl1_part1B__D_0__D_1__D_2__D_3, X_bmej_lvl1_part1_2__D_0__D_1__D_2__D_3, 1 );

	}
	//****
	//**** (out of 1)
	//**** Is real	0 shadows
		//Outputs:
		//  X_bmej__D_0__D_1__D_2__D_3
	PartitionDown(r_bmfe__D_0__D_1__D_2__D_3, r_bmfe_lvl1_part1T__D_0__D_1__D_2__D_3, r_bmfe_lvl1_part1B__D_0__D_1__D_2__D_3, 1, 0);
	PartitionDown(X_bmej__D_0__D_1__D_2__D_3, X_bmej_lvl1_part1T__D_0__D_1__D_2__D_3, X_bmej_lvl1_part1B__D_0__D_1__D_2__D_3, 1, 0);
	while(X_bmej_lvl1_part1T__D_0__D_1__D_2__D_3.Dimension(1) < X_bmej__D_0__D_1__D_2__D_3.Dimension(1))
	{
		RepartitionDown
		( r_bmfe_lvl1_part1T__D_0__D_1__D_2__D_3,  r_bmfe_lvl1_part1_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       r_bmfe_lvl1_part1_1__D_0__D_1__D_2__D_3,
		  r_bmfe_lvl1_part1B__D_0__D_1__D_2__D_3, r_bmfe_lvl1_part1_2__D_0__D_1__D_2__D_3, 1, blkSize );
		RepartitionDown
		( X_bmej_lvl1_part1T__D_0__D_1__D_2__D_3,  X_bmej_lvl1_part1_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       X_bmej_lvl1_part1_1__D_0__D_1__D_2__D_3,
		  X_bmej_lvl1_part1B__D_0__D_1__D_2__D_3, X_bmej_lvl1_part1_2__D_0__D_1__D_2__D_3, 1, blkSize );

		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  X_bmej_lvl1_part1_1__D_0__D_1__D_2__D_3
		PartitionDown(t_fj__D_0_1__D_2_3, t_fj_lvl2_part1T__D_0_1__D_2_3, t_fj_lvl2_part1B__D_0_1__D_2_3, 1, 0);
		PartitionDown(X_bmej_lvl1_part1_1__D_0__D_1__D_2__D_3, X_bmej_lvl1_part1_1_lvl2_part3T__D_0__D_1__D_2__D_3, X_bmej_lvl1_part1_1_lvl2_part3B__D_0__D_1__D_2__D_3, 3, 0);
		while(X_bmej_lvl1_part1_1_lvl2_part3T__D_0__D_1__D_2__D_3.Dimension(3) < X_bmej_lvl1_part1_1__D_0__D_1__D_2__D_3.Dimension(3))
		{
			RepartitionDown
			( t_fj_lvl2_part1T__D_0_1__D_2_3,  t_fj_lvl2_part1_0__D_0_1__D_2_3,
			  /**/ /**/
			       t_fj_lvl2_part1_1__D_0_1__D_2_3,
			  t_fj_lvl2_part1B__D_0_1__D_2_3, t_fj_lvl2_part1_2__D_0_1__D_2_3, 1, blkSize );
			RepartitionDown
			( X_bmej_lvl1_part1_1_lvl2_part3T__D_0__D_1__D_2__D_3,  X_bmej_lvl1_part1_1_lvl2_part3_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       X_bmej_lvl1_part1_1_lvl2_part3_1__D_0__D_1__D_2__D_3,
			  X_bmej_lvl1_part1_1_lvl2_part3B__D_0__D_1__D_2__D_3, X_bmej_lvl1_part1_1_lvl2_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );

			   // t_fj_lvl2_part1_1[D01,D2] <- t_fj_lvl2_part1_1[D01,D23]
			t_fj_lvl2_part1_1__D_0_1__D_2.AlignModesWith( modes_0, r_bmfe_lvl1_part1_1__D_0__D_1__D_2__D_3, modes_3 );
			t_fj_lvl2_part1_1__D_0_1__D_2.AllGatherRedistFrom( t_fj_lvl2_part1_1__D_0_1__D_2_3, modes_3 );
			   // t_fj_lvl2_part1_1[D013,D2] <- t_fj_lvl2_part1_1[D01,D2]
			t_fj_lvl2_part1_1__D_0_1_3__D_2.AlignModesWith( modes_0, r_bmfe_lvl1_part1_1__D_0__D_1__D_2__D_3, modes_3 );
			t_fj_lvl2_part1_1__D_0_1_3__D_2.LocalRedistFrom( t_fj_lvl2_part1_1__D_0_1__D_2 );
			t_fj_lvl2_part1_1__D_0_1__D_2.EmptyData();
			   // t_fj_lvl2_part1_1[D301,D2] <- t_fj_lvl2_part1_1[D013,D2]
			t_fj_lvl2_part1_1__D_3_0_1__D_2.AlignModesWith( modes_0, r_bmfe_lvl1_part1_1__D_0__D_1__D_2__D_3, modes_3 );
			t_fj_lvl2_part1_1__D_3_0_1__D_2.PermutationRedistFrom( t_fj_lvl2_part1_1__D_0_1_3__D_2, modes_0_1_3 );
			t_fj_lvl2_part1_1__D_0_1_3__D_2.EmptyData();
			   // t_fj_lvl2_part1_1[D3,*] <- t_fj_lvl2_part1_1[D301,D2]
			t_fj_lvl2_part1_1__D_3__S.AlignModesWith( modes_0, r_bmfe_lvl1_part1_1__D_0__D_1__D_2__D_3, modes_3 );
			t_fj_lvl2_part1_1__D_3__S.AllGatherRedistFrom( t_fj_lvl2_part1_1__D_3_0_1__D_2, modes_0_1_2 );
			t_fj_lvl2_part1_1__D_3_0_1__D_2.EmptyData();
			X_bmej_lvl1_part1_1_lvl2_part3_1__D_0__D_1__D_2__S__D_3.AlignModesWith( modes_0_1_2, r_bmfe_lvl1_part1_1__D_0__D_1__D_2__D_3, modes_0_1_2 );
			tempShape = X_bmej_lvl1_part1_1_lvl2_part3_1__D_0__D_1__D_2__D_3.Shape();
			tempShape.push_back( g.Shape()[3] );
			X_bmej_lvl1_part1_1_lvl2_part3_1__D_0__D_1__D_2__S__D_3.ResizeTo( tempShape );
			   // 1.0 * r_bmfe_lvl1_part1_1[D0,D1,D2,D3]_bmef * t_fj_lvl2_part1_1[D3,*]_fj + 0.0 * X_bmej_lvl1_part1_1_lvl2_part3_1[D0,D1,D2,*,D3]_bmejf
			LocalContract(1.0, r_bmfe_lvl1_part1_1__D_0__D_1__D_2__D_3.LockedTensor(), indices_bmef, false,
				t_fj_lvl2_part1_1__D_3__S.LockedTensor(), indices_fj, false,
				0.0, X_bmej_lvl1_part1_1_lvl2_part3_1__D_0__D_1__D_2__S__D_3.Tensor(), indices_bmejf, false);
			t_fj_lvl2_part1_1__D_3__S.EmptyData();
			   // X_bmej_lvl1_part1_1_lvl2_part3_1[D0,D1,D2,D3] <- X_bmej_lvl1_part1_1_lvl2_part3_1[D0,D1,D2,*,D3] (with SumScatter on D3)
			X_bmej_lvl1_part1_1_lvl2_part3_1__D_0__D_1__D_2__D_3.ReduceScatterUpdateRedistFrom( X_bmej_lvl1_part1_1_lvl2_part3_1__D_0__D_1__D_2__S__D_3, 1.0, 4 );
			X_bmej_lvl1_part1_1_lvl2_part3_1__D_0__D_1__D_2__S__D_3.EmptyData();

			SlidePartitionDown
			( t_fj_lvl2_part1T__D_0_1__D_2_3,  t_fj_lvl2_part1_0__D_0_1__D_2_3,
			       t_fj_lvl2_part1_1__D_0_1__D_2_3,
			  /**/ /**/
			  t_fj_lvl2_part1B__D_0_1__D_2_3, t_fj_lvl2_part1_2__D_0_1__D_2_3, 1 );
			SlidePartitionDown
			( X_bmej_lvl1_part1_1_lvl2_part3T__D_0__D_1__D_2__D_3,  X_bmej_lvl1_part1_1_lvl2_part3_0__D_0__D_1__D_2__D_3,
			       X_bmej_lvl1_part1_1_lvl2_part3_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  X_bmej_lvl1_part1_1_lvl2_part3B__D_0__D_1__D_2__D_3, X_bmej_lvl1_part1_1_lvl2_part3_2__D_0__D_1__D_2__D_3, 3 );

		}
		//****

		SlidePartitionDown
		( r_bmfe_lvl1_part1T__D_0__D_1__D_2__D_3,  r_bmfe_lvl1_part1_0__D_0__D_1__D_2__D_3,
		       r_bmfe_lvl1_part1_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  r_bmfe_lvl1_part1B__D_0__D_1__D_2__D_3, r_bmfe_lvl1_part1_2__D_0__D_1__D_2__D_3, 1 );
		SlidePartitionDown
		( X_bmej_lvl1_part1T__D_0__D_1__D_2__D_3,  X_bmej_lvl1_part1_0__D_0__D_1__D_2__D_3,
		       X_bmej_lvl1_part1_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  X_bmej_lvl1_part1B__D_0__D_1__D_2__D_3, X_bmej_lvl1_part1_2__D_0__D_1__D_2__D_3, 1 );

	}
	//****


//****


//**** (out of 1)
//**** Is real	0 shadows
	//Outputs:
	//  U_mnie__D_0__D_1__D_2__D_3

	U_mnie__D_0__D_1__D_2__D_3 = u_mnje__D_0__D_1__D_2__D_3;


//****
//**** (out of 1)
//**** Is real	0 shadows
	//Outputs:
	//  U_mnie__D_0__D_1__D_2__D_3

	//**** (out of 1)
	//**** Is real	0 shadows
		//Outputs:
		//  U_mnie__D_0__D_1__D_2__D_3
	PartitionDown(v_femn__D_0__D_1__D_2__D_3, v_femn_lvl1_part2T__D_0__D_1__D_2__D_3, v_femn_lvl1_part2B__D_0__D_1__D_2__D_3, 2, 0);
	PartitionDown(U_mnie__D_0__D_1__D_2__D_3, U_mnie_lvl1_part0T__D_0__D_1__D_2__D_3, U_mnie_lvl1_part0B__D_0__D_1__D_2__D_3, 0, 0);
	while(U_mnie_lvl1_part0T__D_0__D_1__D_2__D_3.Dimension(0) < U_mnie__D_0__D_1__D_2__D_3.Dimension(0))
	{
		RepartitionDown
		( v_femn_lvl1_part2T__D_0__D_1__D_2__D_3,  v_femn_lvl1_part2_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       v_femn_lvl1_part2_1__D_0__D_1__D_2__D_3,
		  v_femn_lvl1_part2B__D_0__D_1__D_2__D_3, v_femn_lvl1_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );
		RepartitionDown
		( U_mnie_lvl1_part0T__D_0__D_1__D_2__D_3,  U_mnie_lvl1_part0_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       U_mnie_lvl1_part0_1__D_0__D_1__D_2__D_3,
		  U_mnie_lvl1_part0B__D_0__D_1__D_2__D_3, U_mnie_lvl1_part0_2__D_0__D_1__D_2__D_3, 0, blkSize );

		Permute( v_femn_lvl1_part2_1__D_0__D_1__D_2__D_3, v_femn_lvl1_part2_1_perm1230__D_1__D_2__D_3__D_0 );
		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  U_mnie_lvl1_part0_1__D_0__D_1__D_2__D_3
		PartitionDown(t_fj__D_0_1__D_2_3, t_fj_lvl2_part1T__D_0_1__D_2_3, t_fj_lvl2_part1B__D_0_1__D_2_3, 1, 0);
		PartitionDown(U_mnie_lvl1_part0_1__D_0__D_1__D_2__D_3, U_mnie_lvl1_part0_1_lvl2_part2T__D_0__D_1__D_2__D_3, U_mnie_lvl1_part0_1_lvl2_part2B__D_0__D_1__D_2__D_3, 2, 0);
		while(U_mnie_lvl1_part0_1_lvl2_part2T__D_0__D_1__D_2__D_3.Dimension(2) < U_mnie_lvl1_part0_1__D_0__D_1__D_2__D_3.Dimension(2))
		{
			RepartitionDown
			( t_fj_lvl2_part1T__D_0_1__D_2_3,  t_fj_lvl2_part1_0__D_0_1__D_2_3,
			  /**/ /**/
			       t_fj_lvl2_part1_1__D_0_1__D_2_3,
			  t_fj_lvl2_part1B__D_0_1__D_2_3, t_fj_lvl2_part1_2__D_0_1__D_2_3, 1, blkSize );
			RepartitionDown
			( U_mnie_lvl1_part0_1_lvl2_part2T__D_0__D_1__D_2__D_3,  U_mnie_lvl1_part0_1_lvl2_part2_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       U_mnie_lvl1_part0_1_lvl2_part2_1__D_0__D_1__D_2__D_3,
			  U_mnie_lvl1_part0_1_lvl2_part2B__D_0__D_1__D_2__D_3, U_mnie_lvl1_part0_1_lvl2_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );

			   // t_fj_lvl2_part1_1[D0,*] <- t_fj_lvl2_part1_1[D01,D23]
			t_fj_lvl2_part1_1__D_0__S.AlignModesWith( modes_0, v_femn_lvl1_part2_1__D_0__D_1__D_2__D_3, modes_0 );
			t_fj_lvl2_part1_1__D_0__S.AllGatherRedistFrom( t_fj_lvl2_part1_1__D_0_1__D_2_3, modes_1_2_3 );
			U_mnie_lvl1_part0_1_lvl2_part2_1_perm30124__D_1__D_2__D_3__S__D_0.AlignModesWith( modes_0_1_3, v_femn_lvl1_part2_1__D_0__D_1__D_2__D_3, modes_2_3_1 );
			tempShape = U_mnie_lvl1_part0_1_lvl2_part2_1__D_0__D_1__D_2__D_3.Shape();
			tempShape.push_back( g.Shape()[0] );
			U_mnie_lvl1_part0_1_lvl2_part2_1_perm30124__D_1__D_2__D_3__S__D_0.ResizeTo( tempShape );
			   // 1.0 * v_femn_lvl1_part2_1[D0,D1,D2,D3]_emnf * t_fj_lvl2_part1_1[D0,*]_fi + 0.0 * U_mnie_lvl1_part0_1_lvl2_part2_1[D2,D3,*,D1,D0]_emnif
			LocalContract(1.0, v_femn_lvl1_part2_1_perm1230__D_1__D_2__D_3__D_0.LockedTensor(), indices_emnf, false,
				t_fj_lvl2_part1_1__D_0__S.LockedTensor(), indices_fi, false,
				0.0, U_mnie_lvl1_part0_1_lvl2_part2_1_perm30124__D_1__D_2__D_3__S__D_0.Tensor(), indices_emnif, false);
			t_fj_lvl2_part1_1__D_0__S.EmptyData();
			U_mnie_lvl1_part0_1_lvl2_part2_1_temp__D_2_0__D_3__S__D_1.AlignModesWith( modes_0_1_2_3, U_mnie_lvl1_part0_1_lvl2_part2_1_temp__D_0__D_1__D_2__D_3, modes_0_1_2_3 );
			tempShape = U_mnie_lvl1_part0_1_lvl2_part2_1__D_0__D_1__D_2__D_3.Shape();
			U_mnie_lvl1_part0_1_lvl2_part2_1_temp__D_2_0__D_3__S__D_1.ResizeTo( tempShape );
			   // U_mnie_lvl1_part0_1_lvl2_part2_1_temp[D20,D3,*,D1] <- U_mnie_lvl1_part0_1_lvl2_part2_1[D2,D3,*,D1,D0] (with SumScatter on D0)
			U_mnie_lvl1_part0_1_lvl2_part2_1_temp__D_2_0__D_3__S__D_1.ReduceScatterRedistFrom( U_mnie_lvl1_part0_1_lvl2_part2_1_perm30124__D_1__D_2__D_3__S__D_0, 4 );
			U_mnie_lvl1_part0_1_lvl2_part2_1_perm30124__D_1__D_2__D_3__S__D_0.EmptyData();
			   // U_mnie_lvl1_part0_1_lvl2_part2_1_temp[D20,D1,*,D3] <- U_mnie_lvl1_part0_1_lvl2_part2_1_temp[D20,D3,*,D1]
			U_mnie_lvl1_part0_1_lvl2_part2_1_temp__D_2_0__D_1__S__D_3.AlignModesWith( modes_0_1_2_3, U_mnie_lvl1_part0_1_lvl2_part2_1__D_0__D_1__D_2__D_3, modes_0_1_2_3 );
			U_mnie_lvl1_part0_1_lvl2_part2_1_temp__D_2_0__D_1__S__D_3.AllToAllRedistFrom( U_mnie_lvl1_part0_1_lvl2_part2_1_temp__D_2_0__D_3__S__D_1, modes_1_3 );
			U_mnie_lvl1_part0_1_lvl2_part2_1_temp__D_2_0__D_3__S__D_1.EmptyData();
			   // U_mnie_lvl1_part0_1_lvl2_part2_1_temp[D0,D1,D2,D3] <- U_mnie_lvl1_part0_1_lvl2_part2_1_temp[D20,D1,*,D3]
			U_mnie_lvl1_part0_1_lvl2_part2_1_temp__D_0__D_1__D_2__D_3.AlignModesWith( modes_0_1_2_3, U_mnie_lvl1_part0_1_lvl2_part2_1__D_0__D_1__D_2__D_3, modes_0_1_2_3 );
			U_mnie_lvl1_part0_1_lvl2_part2_1_temp__D_0__D_1__D_2__D_3.AllToAllRedistFrom( U_mnie_lvl1_part0_1_lvl2_part2_1_temp__D_2_0__D_1__S__D_3, modes_0_2 );
			U_mnie_lvl1_part0_1_lvl2_part2_1_temp__D_2_0__D_1__S__D_3.EmptyData();
			YxpBy( U_mnie_lvl1_part0_1_lvl2_part2_1_temp__D_0__D_1__D_2__D_3, 1.0, U_mnie_lvl1_part0_1_lvl2_part2_1__D_0__D_1__D_2__D_3 );
			U_mnie_lvl1_part0_1_lvl2_part2_1_temp__D_0__D_1__D_2__D_3.EmptyData();

			SlidePartitionDown
			( t_fj_lvl2_part1T__D_0_1__D_2_3,  t_fj_lvl2_part1_0__D_0_1__D_2_3,
			       t_fj_lvl2_part1_1__D_0_1__D_2_3,
			  /**/ /**/
			  t_fj_lvl2_part1B__D_0_1__D_2_3, t_fj_lvl2_part1_2__D_0_1__D_2_3, 1 );
			SlidePartitionDown
			( U_mnie_lvl1_part0_1_lvl2_part2T__D_0__D_1__D_2__D_3,  U_mnie_lvl1_part0_1_lvl2_part2_0__D_0__D_1__D_2__D_3,
			       U_mnie_lvl1_part0_1_lvl2_part2_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  U_mnie_lvl1_part0_1_lvl2_part2B__D_0__D_1__D_2__D_3, U_mnie_lvl1_part0_1_lvl2_part2_2__D_0__D_1__D_2__D_3, 2 );

		}
		//****
		v_femn_lvl1_part2_1_perm1230__D_1__D_2__D_3__D_0.EmptyData();

		SlidePartitionDown
		( v_femn_lvl1_part2T__D_0__D_1__D_2__D_3,  v_femn_lvl1_part2_0__D_0__D_1__D_2__D_3,
		       v_femn_lvl1_part2_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  v_femn_lvl1_part2B__D_0__D_1__D_2__D_3, v_femn_lvl1_part2_2__D_0__D_1__D_2__D_3, 2 );
		SlidePartitionDown
		( U_mnie_lvl1_part0T__D_0__D_1__D_2__D_3,  U_mnie_lvl1_part0_0__D_0__D_1__D_2__D_3,
		       U_mnie_lvl1_part0_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  U_mnie_lvl1_part0B__D_0__D_1__D_2__D_3, U_mnie_lvl1_part0_2__D_0__D_1__D_2__D_3, 0 );

	}
	//****


//****


//**** (out of 1)
//**** Is real	0 shadows
	//Outputs:
	//  Q_mnij__D_0__D_1__D_2__D_3

	tempShape = Q_mnij__D_0__D_1__D_2__D_3.Shape();
	Qtemp1__D_0__D_1__D_2__D_3.ResizeTo( tempShape );
	//**** (out of 1)
	//**** Is real	0 shadows
		//Outputs:
		//  Qtemp1__D_0__D_1__D_2__D_3
	PartitionDown(u_mnje__D_0__D_1__D_2__D_3, u_mnje_lvl1_part1T__D_0__D_1__D_2__D_3, u_mnje_lvl1_part1B__D_0__D_1__D_2__D_3, 1, 0);
	PartitionDown(Qtemp1__D_0__D_1__D_2__D_3, Qtemp1_lvl1_part1T__D_0__D_1__D_2__D_3, Qtemp1_lvl1_part1B__D_0__D_1__D_2__D_3, 1, 0);
	while(Qtemp1_lvl1_part1T__D_0__D_1__D_2__D_3.Dimension(1) < Qtemp1__D_0__D_1__D_2__D_3.Dimension(1))
	{
		RepartitionDown
		( u_mnje_lvl1_part1T__D_0__D_1__D_2__D_3,  u_mnje_lvl1_part1_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       u_mnje_lvl1_part1_1__D_0__D_1__D_2__D_3,
		  u_mnje_lvl1_part1B__D_0__D_1__D_2__D_3, u_mnje_lvl1_part1_2__D_0__D_1__D_2__D_3, 1, blkSize );
		RepartitionDown
		( Qtemp1_lvl1_part1T__D_0__D_1__D_2__D_3,  Qtemp1_lvl1_part1_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       Qtemp1_lvl1_part1_1__D_0__D_1__D_2__D_3,
		  Qtemp1_lvl1_part1B__D_0__D_1__D_2__D_3, Qtemp1_lvl1_part1_2__D_0__D_1__D_2__D_3, 1, blkSize );

		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  Qtemp1_lvl1_part1_1__D_0__D_1__D_2__D_3
		PartitionDown(t_fj__D_0_1__D_2_3, t_fj_lvl2_part1T__D_0_1__D_2_3, t_fj_lvl2_part1B__D_0_1__D_2_3, 1, 0);
		PartitionDown(Qtemp1_lvl1_part1_1__D_0__D_1__D_2__D_3, Qtemp1_lvl1_part1_1_lvl2_part3T__D_0__D_1__D_2__D_3, Qtemp1_lvl1_part1_1_lvl2_part3B__D_0__D_1__D_2__D_3, 3, 0);
		while(Qtemp1_lvl1_part1_1_lvl2_part3T__D_0__D_1__D_2__D_3.Dimension(3) < Qtemp1_lvl1_part1_1__D_0__D_1__D_2__D_3.Dimension(3))
		{
			RepartitionDown
			( t_fj_lvl2_part1T__D_0_1__D_2_3,  t_fj_lvl2_part1_0__D_0_1__D_2_3,
			  /**/ /**/
			       t_fj_lvl2_part1_1__D_0_1__D_2_3,
			  t_fj_lvl2_part1B__D_0_1__D_2_3, t_fj_lvl2_part1_2__D_0_1__D_2_3, 1, blkSize );
			RepartitionDown
			( Qtemp1_lvl1_part1_1_lvl2_part3T__D_0__D_1__D_2__D_3,  Qtemp1_lvl1_part1_1_lvl2_part3_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       Qtemp1_lvl1_part1_1_lvl2_part3_1__D_0__D_1__D_2__D_3,
			  Qtemp1_lvl1_part1_1_lvl2_part3B__D_0__D_1__D_2__D_3, Qtemp1_lvl1_part1_1_lvl2_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );

			   // t_fj_lvl2_part1_1[D01,D2] <- t_fj_lvl2_part1_1[D01,D23]
			t_fj_lvl2_part1_1__D_0_1__D_2.AlignModesWith( modes_0, u_mnje_lvl1_part1_1__D_0__D_1__D_2__D_3, modes_3 );
			t_fj_lvl2_part1_1__D_0_1__D_2.AllGatherRedistFrom( t_fj_lvl2_part1_1__D_0_1__D_2_3, modes_3 );
			   // t_fj_lvl2_part1_1[D013,D2] <- t_fj_lvl2_part1_1[D01,D2]
			t_fj_lvl2_part1_1__D_0_1_3__D_2.AlignModesWith( modes_0, u_mnje_lvl1_part1_1__D_0__D_1__D_2__D_3, modes_3 );
			t_fj_lvl2_part1_1__D_0_1_3__D_2.LocalRedistFrom( t_fj_lvl2_part1_1__D_0_1__D_2 );
			t_fj_lvl2_part1_1__D_0_1__D_2.EmptyData();
			   // t_fj_lvl2_part1_1[D301,D2] <- t_fj_lvl2_part1_1[D013,D2]
			t_fj_lvl2_part1_1__D_3_0_1__D_2.AlignModesWith( modes_0, u_mnje_lvl1_part1_1__D_0__D_1__D_2__D_3, modes_3 );
			t_fj_lvl2_part1_1__D_3_0_1__D_2.PermutationRedistFrom( t_fj_lvl2_part1_1__D_0_1_3__D_2, modes_0_1_3 );
			t_fj_lvl2_part1_1__D_0_1_3__D_2.EmptyData();
			   // t_fj_lvl2_part1_1[D3,*] <- t_fj_lvl2_part1_1[D301,D2]
			t_fj_lvl2_part1_1__D_3__S.AlignModesWith( modes_0, u_mnje_lvl1_part1_1__D_0__D_1__D_2__D_3, modes_3 );
			t_fj_lvl2_part1_1__D_3__S.AllGatherRedistFrom( t_fj_lvl2_part1_1__D_3_0_1__D_2, modes_0_1_2 );
			t_fj_lvl2_part1_1__D_3_0_1__D_2.EmptyData();
			Qtemp1_lvl1_part1_1_lvl2_part3_1__D_0__D_1__D_2__S__D_3.AlignModesWith( modes_0_1_2, u_mnje_lvl1_part1_1__D_0__D_1__D_2__D_3, modes_0_1_2 );
			tempShape = Qtemp1_lvl1_part1_1_lvl2_part3_1__D_0__D_1__D_2__D_3.Shape();
			tempShape.push_back( g.Shape()[3] );
			Qtemp1_lvl1_part1_1_lvl2_part3_1__D_0__D_1__D_2__S__D_3.ResizeTo( tempShape );
			   // 1.0 * u_mnje_lvl1_part1_1[D0,D1,D2,D3]_mnie * t_fj_lvl2_part1_1[D3,*]_ej + 0.0 * Qtemp1_lvl1_part1_1_lvl2_part3_1[D0,D1,D2,*,D3]_mnije
			LocalContract(1.0, u_mnje_lvl1_part1_1__D_0__D_1__D_2__D_3.LockedTensor(), indices_mnie, false,
				t_fj_lvl2_part1_1__D_3__S.LockedTensor(), indices_ej, false,
				0.0, Qtemp1_lvl1_part1_1_lvl2_part3_1__D_0__D_1__D_2__S__D_3.Tensor(), indices_mnije, false);
			t_fj_lvl2_part1_1__D_3__S.EmptyData();
			   // Qtemp1_lvl1_part1_1_lvl2_part3_1[D0,D1,D2,D3] <- Qtemp1_lvl1_part1_1_lvl2_part3_1[D0,D1,D2,*,D3] (with SumScatter on D3)
			Qtemp1_lvl1_part1_1_lvl2_part3_1__D_0__D_1__D_2__D_3.ReduceScatterRedistFrom( Qtemp1_lvl1_part1_1_lvl2_part3_1__D_0__D_1__D_2__S__D_3, 4 );
			Qtemp1_lvl1_part1_1_lvl2_part3_1__D_0__D_1__D_2__S__D_3.EmptyData();

			SlidePartitionDown
			( t_fj_lvl2_part1T__D_0_1__D_2_3,  t_fj_lvl2_part1_0__D_0_1__D_2_3,
			       t_fj_lvl2_part1_1__D_0_1__D_2_3,
			  /**/ /**/
			  t_fj_lvl2_part1B__D_0_1__D_2_3, t_fj_lvl2_part1_2__D_0_1__D_2_3, 1 );
			SlidePartitionDown
			( Qtemp1_lvl1_part1_1_lvl2_part3T__D_0__D_1__D_2__D_3,  Qtemp1_lvl1_part1_1_lvl2_part3_0__D_0__D_1__D_2__D_3,
			       Qtemp1_lvl1_part1_1_lvl2_part3_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  Qtemp1_lvl1_part1_1_lvl2_part3B__D_0__D_1__D_2__D_3, Qtemp1_lvl1_part1_1_lvl2_part3_2__D_0__D_1__D_2__D_3, 3 );

		}
		//****

		SlidePartitionDown
		( u_mnje_lvl1_part1T__D_0__D_1__D_2__D_3,  u_mnje_lvl1_part1_0__D_0__D_1__D_2__D_3,
		       u_mnje_lvl1_part1_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  u_mnje_lvl1_part1B__D_0__D_1__D_2__D_3, u_mnje_lvl1_part1_2__D_0__D_1__D_2__D_3, 1 );
		SlidePartitionDown
		( Qtemp1_lvl1_part1T__D_0__D_1__D_2__D_3,  Qtemp1_lvl1_part1_0__D_0__D_1__D_2__D_3,
		       Qtemp1_lvl1_part1_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  Qtemp1_lvl1_part1B__D_0__D_1__D_2__D_3, Qtemp1_lvl1_part1_2__D_0__D_1__D_2__D_3, 1 );

	}
	//****
	//**** (out of 1)
	//**** Is real	0 shadows
		//Outputs:
		//  Q_mnij__D_0__D_1__D_2__D_3
	PartitionDown(Qtemp1__D_0__D_1__D_2__D_3, Qtemp1_lvl1_part0T__D_0__D_1__D_2__D_3, Qtemp1_lvl1_part0B__D_0__D_1__D_2__D_3, 0, 0);
	PartitionDown(Qtemp1__D_0__D_1__D_2__D_3, Qtemp1_lvl1_part1T__D_0__D_1__D_2__D_3, Qtemp1_lvl1_part1B__D_0__D_1__D_2__D_3, 1, 0);
	PartitionDown(Q_mnij__D_0__D_1__D_2__D_3, Q_mnij_lvl1_part0T__D_0__D_1__D_2__D_3, Q_mnij_lvl1_part0B__D_0__D_1__D_2__D_3, 0, 0);
	while(Q_mnij_lvl1_part0T__D_0__D_1__D_2__D_3.Dimension(0) < Q_mnij__D_0__D_1__D_2__D_3.Dimension(0))
	{
		RepartitionDown
		( Qtemp1_lvl1_part0T__D_0__D_1__D_2__D_3,  Qtemp1_lvl1_part0_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       Qtemp1_lvl1_part0_1__D_0__D_1__D_2__D_3,
		  Qtemp1_lvl1_part0B__D_0__D_1__D_2__D_3, Qtemp1_lvl1_part0_2__D_0__D_1__D_2__D_3, 0, blkSize );
		RepartitionDown
		( Qtemp1_lvl1_part1T__D_0__D_1__D_2__D_3,  Qtemp1_lvl1_part1_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       Qtemp1_lvl1_part1_1__D_0__D_1__D_2__D_3,
		  Qtemp1_lvl1_part1B__D_0__D_1__D_2__D_3, Qtemp1_lvl1_part1_2__D_0__D_1__D_2__D_3, 1, blkSize );
		RepartitionDown
		( Q_mnij_lvl1_part0T__D_0__D_1__D_2__D_3,  Q_mnij_lvl1_part0_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       Q_mnij_lvl1_part0_1__D_0__D_1__D_2__D_3,
		  Q_mnij_lvl1_part0B__D_0__D_1__D_2__D_3, Q_mnij_lvl1_part0_2__D_0__D_1__D_2__D_3, 0, blkSize );

		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  Q_mnij_lvl1_part0_1__D_0__D_1__D_2__D_3
		PartitionDown(Qtemp1_lvl1_part0_1__D_0__D_1__D_2__D_3, Qtemp1_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3, Qtemp1_lvl1_part0_1_lvl2_part1B__D_0__D_1__D_2__D_3, 1, 0);
		PartitionDown(Qtemp1_lvl1_part1_1__D_0__D_1__D_2__D_3, Qtemp1_lvl1_part1_1_lvl2_part0T__D_0__D_1__D_2__D_3, Qtemp1_lvl1_part1_1_lvl2_part0B__D_0__D_1__D_2__D_3, 0, 0);
		PartitionDown(Q_mnij_lvl1_part0_1__D_0__D_1__D_2__D_3, Q_mnij_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3, Q_mnij_lvl1_part0_1_lvl2_part1B__D_0__D_1__D_2__D_3, 1, 0);
		while(Q_mnij_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3.Dimension(1) < Q_mnij_lvl1_part0_1__D_0__D_1__D_2__D_3.Dimension(1))
		{
			RepartitionDown
			( Qtemp1_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3,  Qtemp1_lvl1_part0_1_lvl2_part1_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       Qtemp1_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3,
			  Qtemp1_lvl1_part0_1_lvl2_part1B__D_0__D_1__D_2__D_3, Qtemp1_lvl1_part0_1_lvl2_part1_2__D_0__D_1__D_2__D_3, 1, blkSize );
			RepartitionDown
			( Qtemp1_lvl1_part1_1_lvl2_part0T__D_0__D_1__D_2__D_3,  Qtemp1_lvl1_part1_1_lvl2_part0_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       Qtemp1_lvl1_part1_1_lvl2_part0_1__D_0__D_1__D_2__D_3,
			  Qtemp1_lvl1_part1_1_lvl2_part0B__D_0__D_1__D_2__D_3, Qtemp1_lvl1_part1_1_lvl2_part0_2__D_0__D_1__D_2__D_3, 0, blkSize );
			RepartitionDown
			( Q_mnij_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3,  Q_mnij_lvl1_part0_1_lvl2_part1_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       Q_mnij_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3,
			  Q_mnij_lvl1_part0_1_lvl2_part1B__D_0__D_1__D_2__D_3, Q_mnij_lvl1_part0_1_lvl2_part1_2__D_0__D_1__D_2__D_3, 1, blkSize );

			   // Qtemp1_lvl1_part1_1_lvl2_part0_1[D1,D0,D2,D3] <- Qtemp1_lvl1_part1_1_lvl2_part0_1[D0,D1,D2,D3]
			Qtemp1_lvl1_part1_1_lvl2_part0_1__D_1__D_0__D_2__D_3.AlignModesWith( modes_0_1_2_3, Qtemp1_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3, modes_1_0_3_2 );
			Qtemp1_lvl1_part1_1_lvl2_part0_1__D_1__D_0__D_2__D_3.AllToAllRedistFrom( Qtemp1_lvl1_part1_1_lvl2_part0_1__D_0__D_1__D_2__D_3, modes_0_1 );
			   // Qtemp1_lvl1_part1_1_lvl2_part0_1[D1,D0,D3,D2] <- Qtemp1_lvl1_part1_1_lvl2_part0_1[D1,D0,D2,D3]
			Qtemp1_lvl1_part1_1_lvl2_part0_1__D_1__D_0__D_3__D_2.AlignModesWith( modes_0_1_2_3, Qtemp1_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3, modes_1_0_3_2 );
			Qtemp1_lvl1_part1_1_lvl2_part0_1__D_1__D_0__D_3__D_2.AllToAllRedistFrom( Qtemp1_lvl1_part1_1_lvl2_part0_1__D_1__D_0__D_2__D_3, modes_2_3 );
			Qtemp1_lvl1_part1_1_lvl2_part0_1__D_1__D_0__D_2__D_3.EmptyData();
			YAxpPx( 1.0, Qtemp1_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3, 1.0, Qtemp1_lvl1_part1_1_lvl2_part0_1__D_1__D_0__D_3__D_2, perm_1_0_3_2, Q_mnij_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3 );
			Qtemp1_lvl1_part1_1_lvl2_part0_1__D_1__D_0__D_3__D_2.EmptyData();

			SlidePartitionDown
			( Qtemp1_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3,  Qtemp1_lvl1_part0_1_lvl2_part1_0__D_0__D_1__D_2__D_3,
			       Qtemp1_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  Qtemp1_lvl1_part0_1_lvl2_part1B__D_0__D_1__D_2__D_3, Qtemp1_lvl1_part0_1_lvl2_part1_2__D_0__D_1__D_2__D_3, 1 );
			SlidePartitionDown
			( Qtemp1_lvl1_part1_1_lvl2_part0T__D_0__D_1__D_2__D_3,  Qtemp1_lvl1_part1_1_lvl2_part0_0__D_0__D_1__D_2__D_3,
			       Qtemp1_lvl1_part1_1_lvl2_part0_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  Qtemp1_lvl1_part1_1_lvl2_part0B__D_0__D_1__D_2__D_3, Qtemp1_lvl1_part1_1_lvl2_part0_2__D_0__D_1__D_2__D_3, 0 );
			SlidePartitionDown
			( Q_mnij_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3,  Q_mnij_lvl1_part0_1_lvl2_part1_0__D_0__D_1__D_2__D_3,
			       Q_mnij_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  Q_mnij_lvl1_part0_1_lvl2_part1B__D_0__D_1__D_2__D_3, Q_mnij_lvl1_part0_1_lvl2_part1_2__D_0__D_1__D_2__D_3, 1 );

		}
		//****

		SlidePartitionDown
		( Qtemp1_lvl1_part0T__D_0__D_1__D_2__D_3,  Qtemp1_lvl1_part0_0__D_0__D_1__D_2__D_3,
		       Qtemp1_lvl1_part0_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  Qtemp1_lvl1_part0B__D_0__D_1__D_2__D_3, Qtemp1_lvl1_part0_2__D_0__D_1__D_2__D_3, 0 );
		SlidePartitionDown
		( Qtemp1_lvl1_part1T__D_0__D_1__D_2__D_3,  Qtemp1_lvl1_part1_0__D_0__D_1__D_2__D_3,
		       Qtemp1_lvl1_part1_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  Qtemp1_lvl1_part1B__D_0__D_1__D_2__D_3, Qtemp1_lvl1_part1_2__D_0__D_1__D_2__D_3, 1 );
		SlidePartitionDown
		( Q_mnij_lvl1_part0T__D_0__D_1__D_2__D_3,  Q_mnij_lvl1_part0_0__D_0__D_1__D_2__D_3,
		       Q_mnij_lvl1_part0_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  Q_mnij_lvl1_part0B__D_0__D_1__D_2__D_3, Q_mnij_lvl1_part0_2__D_0__D_1__D_2__D_3, 0 );

	}
	//****
	Qtemp1__D_0__D_1__D_2__D_3.EmptyData();
	Qtemp1__D_0__D_1__D_2__D_3.EmptyData();
	//**** (out of 1)
	//**** Is real	0 shadows
		//Outputs:
		//  Q_mnij__D_0__D_1__D_2__D_3
	PartitionDown(v_femn__D_0__D_1__D_2__D_3, v_femn_lvl1_part2T__D_0__D_1__D_2__D_3, v_femn_lvl1_part2B__D_0__D_1__D_2__D_3, 2, 0);
	PartitionDown(Q_mnij__D_0__D_1__D_2__D_3, Q_mnij_lvl1_part0T__D_0__D_1__D_2__D_3, Q_mnij_lvl1_part0B__D_0__D_1__D_2__D_3, 0, 0);
	while(Q_mnij_lvl1_part0T__D_0__D_1__D_2__D_3.Dimension(0) < Q_mnij__D_0__D_1__D_2__D_3.Dimension(0))
	{
		RepartitionDown
		( v_femn_lvl1_part2T__D_0__D_1__D_2__D_3,  v_femn_lvl1_part2_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       v_femn_lvl1_part2_1__D_0__D_1__D_2__D_3,
		  v_femn_lvl1_part2B__D_0__D_1__D_2__D_3, v_femn_lvl1_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );
		RepartitionDown
		( Q_mnij_lvl1_part0T__D_0__D_1__D_2__D_3,  Q_mnij_lvl1_part0_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       Q_mnij_lvl1_part0_1__D_0__D_1__D_2__D_3,
		  Q_mnij_lvl1_part0B__D_0__D_1__D_2__D_3, Q_mnij_lvl1_part0_2__D_0__D_1__D_2__D_3, 0, blkSize );

		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  Q_mnij_lvl1_part0_1__D_0__D_1__D_2__D_3
		PartitionDown(v_femn_lvl1_part2_1__D_0__D_1__D_2__D_3, v_femn_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3, v_femn_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, 3, 0);
		PartitionDown(Q_mnij_lvl1_part0_1__D_0__D_1__D_2__D_3, Q_mnij_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3, Q_mnij_lvl1_part0_1_lvl2_part1B__D_0__D_1__D_2__D_3, 1, 0);
		while(Q_mnij_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3.Dimension(1) < Q_mnij_lvl1_part0_1__D_0__D_1__D_2__D_3.Dimension(1))
		{
			RepartitionDown
			( v_femn_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3,  v_femn_lvl1_part2_1_lvl2_part3_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       v_femn_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3,
			  v_femn_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, v_femn_lvl1_part2_1_lvl2_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );
			RepartitionDown
			( Q_mnij_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3,  Q_mnij_lvl1_part0_1_lvl2_part1_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       Q_mnij_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3,
			  Q_mnij_lvl1_part0_1_lvl2_part1B__D_0__D_1__D_2__D_3, Q_mnij_lvl1_part0_1_lvl2_part1_2__D_0__D_1__D_2__D_3, 1, blkSize );

			   // v_femn_lvl1_part2_1_lvl2_part3_1[D0,D1,*,*] <- v_femn_lvl1_part2_1_lvl2_part3_1[D0,D1,D2,D3]
			v_femn_lvl1_part2_1_lvl2_part3_1_perm2301__S__S__D_0__D_1.AlignModesWith( modes_0_1, Tau_efmn__D_0__D_1__D_2__D_3, modes_0_1 );
			v_femn_lvl1_part2_1_lvl2_part3_1_perm2301__S__S__D_0__D_1.AllGatherRedistFrom( v_femn_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3, modes_2_3 );
			Q_mnij_lvl1_part0_1_lvl2_part1_1__S__S__D_2__D_3__D_0__D_1.AlignModesWith( modes_2_3, Tau_efmn__D_0__D_1__D_2__D_3, modes_2_3 );
			tempShape = Q_mnij_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3.Shape();
			tempShape.push_back( g.Shape()[0] );
			tempShape.push_back( g.Shape()[1] );
			Q_mnij_lvl1_part0_1_lvl2_part1_1__S__S__D_2__D_3__D_0__D_1.ResizeTo( tempShape );
			   // 1.0 * v_femn_lvl1_part2_1_lvl2_part3_1[D0,D1,*,*]_mnef * Tau_efmn[D0,D1,D2,D3]_efij + 0.0 * Q_mnij_lvl1_part0_1_lvl2_part1_1[*,*,D2,D3,D0,D1]_mnijef
			LocalContract(1.0, v_femn_lvl1_part2_1_lvl2_part3_1_perm2301__S__S__D_0__D_1.LockedTensor(), indices_mnef, false,
				Tau_efmn__D_0__D_1__D_2__D_3.LockedTensor(), indices_efij, false,
				0.0, Q_mnij_lvl1_part0_1_lvl2_part1_1__S__S__D_2__D_3__D_0__D_1.Tensor(), indices_mnijef, false);
			v_femn_lvl1_part2_1_lvl2_part3_1_perm2301__S__S__D_0__D_1.EmptyData();
			   // Q_mnij_lvl1_part0_1_lvl2_part1_1[D0,D1,D2,D3] <- Q_mnij_lvl1_part0_1_lvl2_part1_1[*,*,D2,D3,D0,D1] (with SumScatter on (D0)(D1))
			Q_mnij_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3.ReduceScatterUpdateRedistFrom( Q_mnij_lvl1_part0_1_lvl2_part1_1__S__S__D_2__D_3__D_0__D_1, 1.0, modes_5_4 );
			Q_mnij_lvl1_part0_1_lvl2_part1_1__S__S__D_2__D_3__D_0__D_1.EmptyData();

			SlidePartitionDown
			( v_femn_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3,  v_femn_lvl1_part2_1_lvl2_part3_0__D_0__D_1__D_2__D_3,
			       v_femn_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  v_femn_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, v_femn_lvl1_part2_1_lvl2_part3_2__D_0__D_1__D_2__D_3, 3 );
			SlidePartitionDown
			( Q_mnij_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3,  Q_mnij_lvl1_part0_1_lvl2_part1_0__D_0__D_1__D_2__D_3,
			       Q_mnij_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  Q_mnij_lvl1_part0_1_lvl2_part1B__D_0__D_1__D_2__D_3, Q_mnij_lvl1_part0_1_lvl2_part1_2__D_0__D_1__D_2__D_3, 1 );

		}
		//****

		SlidePartitionDown
		( v_femn_lvl1_part2T__D_0__D_1__D_2__D_3,  v_femn_lvl1_part2_0__D_0__D_1__D_2__D_3,
		       v_femn_lvl1_part2_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  v_femn_lvl1_part2B__D_0__D_1__D_2__D_3, v_femn_lvl1_part2_2__D_0__D_1__D_2__D_3, 2 );
		SlidePartitionDown
		( Q_mnij_lvl1_part0T__D_0__D_1__D_2__D_3,  Q_mnij_lvl1_part0_0__D_0__D_1__D_2__D_3,
		       Q_mnij_lvl1_part0_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  Q_mnij_lvl1_part0B__D_0__D_1__D_2__D_3, Q_mnij_lvl1_part0_2__D_0__D_1__D_2__D_3, 0 );

	}
	//****


//****
//**** (out of 1)
//**** Is real	0 shadows
	//Outputs:
	//  Q_mnij__D_0__D_1__D_2__D_3

	YAxpy( 1.0, q_mnij__D_0__D_1__D_2__D_3, Q_mnij__D_0__D_1__D_2__D_3 );


//****


//**** (out of 1)
//**** Is real	0 shadows
	//Outputs:
	//  P_jimb__D_0__D_1__D_2__D_3

	P_jimb__D_0__D_1__D_2__D_3 = u_mnje__D_0__D_1__D_2__D_3;


//****
//**** (out of 1)
//**** Is real	0 shadows
	//Outputs:
	//  P_jimb__D_0__D_1__D_2__D_3

	//**** (out of 1)
	//**** Is real	0 shadows
		//Outputs:
		//  P_jimb__D_0__D_1__D_2__D_3
	PartitionDown(x_bmej__D_0__D_1__D_2__D_3, x_bmej_lvl1_part3T__D_0__D_1__D_2__D_3, x_bmej_lvl1_part3B__D_0__D_1__D_2__D_3, 3, 0);
	PartitionDown(P_jimb__D_0__D_1__D_2__D_3, P_jimb_lvl1_part0T__D_0__D_1__D_2__D_3, P_jimb_lvl1_part0B__D_0__D_1__D_2__D_3, 0, 0);
	while(P_jimb_lvl1_part0T__D_0__D_1__D_2__D_3.Dimension(0) < P_jimb__D_0__D_1__D_2__D_3.Dimension(0))
	{
		RepartitionDown
		( x_bmej_lvl1_part3T__D_0__D_1__D_2__D_3,  x_bmej_lvl1_part3_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       x_bmej_lvl1_part3_1__D_0__D_1__D_2__D_3,
		  x_bmej_lvl1_part3B__D_0__D_1__D_2__D_3, x_bmej_lvl1_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );
		RepartitionDown
		( P_jimb_lvl1_part0T__D_0__D_1__D_2__D_3,  P_jimb_lvl1_part0_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       P_jimb_lvl1_part0_1__D_0__D_1__D_2__D_3,
		  P_jimb_lvl1_part0B__D_0__D_1__D_2__D_3, P_jimb_lvl1_part0_2__D_0__D_1__D_2__D_3, 0, blkSize );

		Permute( x_bmej_lvl1_part3_1__D_0__D_1__D_2__D_3, x_bmej_lvl1_part3_1_perm0132__D_0__D_1__D_3__D_2 );
		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  P_jimb_lvl1_part0_1__D_0__D_1__D_2__D_3
		PartitionDown(t_fj__D_0_1__D_2_3, t_fj_lvl2_part1T__D_0_1__D_2_3, t_fj_lvl2_part1B__D_0_1__D_2_3, 1, 0);
		PartitionDown(P_jimb_lvl1_part0_1__D_0__D_1__D_2__D_3, P_jimb_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3, P_jimb_lvl1_part0_1_lvl2_part1B__D_0__D_1__D_2__D_3, 1, 0);
		while(P_jimb_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3.Dimension(1) < P_jimb_lvl1_part0_1__D_0__D_1__D_2__D_3.Dimension(1))
		{
			RepartitionDown
			( t_fj_lvl2_part1T__D_0_1__D_2_3,  t_fj_lvl2_part1_0__D_0_1__D_2_3,
			  /**/ /**/
			       t_fj_lvl2_part1_1__D_0_1__D_2_3,
			  t_fj_lvl2_part1B__D_0_1__D_2_3, t_fj_lvl2_part1_2__D_0_1__D_2_3, 1, blkSize );
			RepartitionDown
			( P_jimb_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3,  P_jimb_lvl1_part0_1_lvl2_part1_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       P_jimb_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3,
			  P_jimb_lvl1_part0_1_lvl2_part1B__D_0__D_1__D_2__D_3, P_jimb_lvl1_part0_1_lvl2_part1_2__D_0__D_1__D_2__D_3, 1, blkSize );

			   // t_fj_lvl2_part1_1[D01,*] <- t_fj_lvl2_part1_1[D01,D23]
			t_fj_lvl2_part1_1__D_0_1__S.AlignModesWith( modes_0, x_bmej_lvl1_part3_1__D_0__D_1__D_2__D_3, modes_2 );
			t_fj_lvl2_part1_1__D_0_1__S.AllGatherRedistFrom( t_fj_lvl2_part1_1__D_0_1__D_2_3, modes_2_3 );
			   // t_fj_lvl2_part1_1[D012,*] <- t_fj_lvl2_part1_1[D01,*]
			t_fj_lvl2_part1_1__D_0_1_2__S.AlignModesWith( modes_0, x_bmej_lvl1_part3_1__D_0__D_1__D_2__D_3, modes_2 );
			t_fj_lvl2_part1_1__D_0_1_2__S.LocalRedistFrom( t_fj_lvl2_part1_1__D_0_1__S );
			t_fj_lvl2_part1_1__D_0_1__S.EmptyData();
			   // t_fj_lvl2_part1_1[D201,*] <- t_fj_lvl2_part1_1[D012,*]
			t_fj_lvl2_part1_1__D_2_0_1__S.AlignModesWith( modes_0, x_bmej_lvl1_part3_1__D_0__D_1__D_2__D_3, modes_2 );
			t_fj_lvl2_part1_1__D_2_0_1__S.PermutationRedistFrom( t_fj_lvl2_part1_1__D_0_1_2__S, modes_0_1_2 );
			t_fj_lvl2_part1_1__D_0_1_2__S.EmptyData();
			   // t_fj_lvl2_part1_1[D2,*] <- t_fj_lvl2_part1_1[D201,*]
			t_fj_lvl2_part1_1__D_2__S.AlignModesWith( modes_0, x_bmej_lvl1_part3_1__D_0__D_1__D_2__D_3, modes_2 );
			t_fj_lvl2_part1_1__D_2__S.AllGatherRedistFrom( t_fj_lvl2_part1_1__D_2_0_1__S, modes_0_1 );
			t_fj_lvl2_part1_1__D_2_0_1__S.EmptyData();
			P_jimb_lvl1_part0_1_lvl2_part1_1_perm32014__D_0__D_1__D_3__S__D_2.AlignModesWith( modes_0_2_3, x_bmej_lvl1_part3_1__D_0__D_1__D_2__D_3, modes_3_1_0 );
			tempShape = P_jimb_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3.Shape();
			tempShape.push_back( g.Shape()[2] );
			P_jimb_lvl1_part0_1_lvl2_part1_1_perm32014__D_0__D_1__D_3__S__D_2.ResizeTo( tempShape );
			   // 1.0 * x_bmej_lvl1_part3_1[D0,D1,D2,D3]_bmje * t_fj_lvl2_part1_1[D2,*]_ei + 0.0 * P_jimb_lvl1_part0_1_lvl2_part1_1[D3,*,D1,D0,D2]_bmjie
			LocalContract(1.0, x_bmej_lvl1_part3_1_perm0132__D_0__D_1__D_3__D_2.LockedTensor(), indices_bmje, false,
				t_fj_lvl2_part1_1__D_2__S.LockedTensor(), indices_ei, false,
				0.0, P_jimb_lvl1_part0_1_lvl2_part1_1_perm32014__D_0__D_1__D_3__S__D_2.Tensor(), indices_bmjie, false);
			t_fj_lvl2_part1_1__D_2__S.EmptyData();
			P_jimb_lvl1_part0_1_lvl2_part1_1_temp__D_3__S__D_1_2__D_0.AlignModesWith( modes_0_1_2_3, P_jimb_lvl1_part0_1_lvl2_part1_1_temp__D_0__D_1__D_2__D_3, modes_0_1_2_3 );
			tempShape = P_jimb_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3.Shape();
			P_jimb_lvl1_part0_1_lvl2_part1_1_temp__D_3__S__D_1_2__D_0.ResizeTo( tempShape );
			   // P_jimb_lvl1_part0_1_lvl2_part1_1_temp[D3,*,D12,D0] <- P_jimb_lvl1_part0_1_lvl2_part1_1[D3,*,D1,D0,D2] (with SumScatter on D2)
			P_jimb_lvl1_part0_1_lvl2_part1_1_temp__D_3__S__D_1_2__D_0.ReduceScatterRedistFrom( P_jimb_lvl1_part0_1_lvl2_part1_1_perm32014__D_0__D_1__D_3__S__D_2, 4 );
			P_jimb_lvl1_part0_1_lvl2_part1_1_perm32014__D_0__D_1__D_3__S__D_2.EmptyData();
			   // P_jimb_lvl1_part0_1_lvl2_part1_1_temp[D3,*,D21,D0] <- P_jimb_lvl1_part0_1_lvl2_part1_1_temp[D3,*,D12,D0]
			P_jimb_lvl1_part0_1_lvl2_part1_1_temp__D_3__S__D_2_1__D_0.AlignModesWith( modes_0_1_2_3, P_jimb_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3, modes_0_1_2_3 );
			P_jimb_lvl1_part0_1_lvl2_part1_1_temp__D_3__S__D_2_1__D_0.PermutationRedistFrom( P_jimb_lvl1_part0_1_lvl2_part1_1_temp__D_3__S__D_1_2__D_0, modes_1_2 );
			P_jimb_lvl1_part0_1_lvl2_part1_1_temp__D_3__S__D_1_2__D_0.EmptyData();
			   // P_jimb_lvl1_part0_1_lvl2_part1_1_temp[D3,D1,D2,D0] <- P_jimb_lvl1_part0_1_lvl2_part1_1_temp[D3,*,D21,D0]
			P_jimb_lvl1_part0_1_lvl2_part1_1_temp__D_3__D_1__D_2__D_0.AlignModesWith( modes_0_1_2_3, P_jimb_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3, modes_0_1_2_3 );
			P_jimb_lvl1_part0_1_lvl2_part1_1_temp__D_3__D_1__D_2__D_0.AllToAllRedistFrom( P_jimb_lvl1_part0_1_lvl2_part1_1_temp__D_3__S__D_2_1__D_0, modes_1 );
			P_jimb_lvl1_part0_1_lvl2_part1_1_temp__D_3__S__D_2_1__D_0.EmptyData();
			   // P_jimb_lvl1_part0_1_lvl2_part1_1_temp[D0,D1,D2,D3] <- P_jimb_lvl1_part0_1_lvl2_part1_1_temp[D3,D1,D2,D0]
			P_jimb_lvl1_part0_1_lvl2_part1_1_temp__D_0__D_1__D_2__D_3.AlignModesWith( modes_0_1_2_3, P_jimb_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3, modes_0_1_2_3 );
			P_jimb_lvl1_part0_1_lvl2_part1_1_temp__D_0__D_1__D_2__D_3.AllToAllRedistFrom( P_jimb_lvl1_part0_1_lvl2_part1_1_temp__D_3__D_1__D_2__D_0, modes_0_3 );
			P_jimb_lvl1_part0_1_lvl2_part1_1_temp__D_3__D_1__D_2__D_0.EmptyData();
			YxpBy( P_jimb_lvl1_part0_1_lvl2_part1_1_temp__D_0__D_1__D_2__D_3, 1.0, P_jimb_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3 );
			P_jimb_lvl1_part0_1_lvl2_part1_1_temp__D_0__D_1__D_2__D_3.EmptyData();

			SlidePartitionDown
			( t_fj_lvl2_part1T__D_0_1__D_2_3,  t_fj_lvl2_part1_0__D_0_1__D_2_3,
			       t_fj_lvl2_part1_1__D_0_1__D_2_3,
			  /**/ /**/
			  t_fj_lvl2_part1B__D_0_1__D_2_3, t_fj_lvl2_part1_2__D_0_1__D_2_3, 1 );
			SlidePartitionDown
			( P_jimb_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3,  P_jimb_lvl1_part0_1_lvl2_part1_0__D_0__D_1__D_2__D_3,
			       P_jimb_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  P_jimb_lvl1_part0_1_lvl2_part1B__D_0__D_1__D_2__D_3, P_jimb_lvl1_part0_1_lvl2_part1_2__D_0__D_1__D_2__D_3, 1 );

		}
		//****
		x_bmej_lvl1_part3_1_perm0132__D_0__D_1__D_3__D_2.EmptyData();

		SlidePartitionDown
		( x_bmej_lvl1_part3T__D_0__D_1__D_2__D_3,  x_bmej_lvl1_part3_0__D_0__D_1__D_2__D_3,
		       x_bmej_lvl1_part3_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  x_bmej_lvl1_part3B__D_0__D_1__D_2__D_3, x_bmej_lvl1_part3_2__D_0__D_1__D_2__D_3, 3 );
		SlidePartitionDown
		( P_jimb_lvl1_part0T__D_0__D_1__D_2__D_3,  P_jimb_lvl1_part0_0__D_0__D_1__D_2__D_3,
		       P_jimb_lvl1_part0_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  P_jimb_lvl1_part0B__D_0__D_1__D_2__D_3, P_jimb_lvl1_part0_2__D_0__D_1__D_2__D_3, 0 );

	}
	//****
	//**** (out of 1)
	//**** Is real	0 shadows
		//Outputs:
		//  P_jimb__D_0__D_1__D_2__D_3
	PartitionDown(t_fj__D_0_1__D_2_3, t_fj_lvl1_part1T__D_0_1__D_2_3, t_fj_lvl1_part1B__D_0_1__D_2_3, 1, 0);
	PartitionDown(P_jimb__D_0__D_1__D_2__D_3, P_jimb_lvl1_part0T__D_0__D_1__D_2__D_3, P_jimb_lvl1_part0B__D_0__D_1__D_2__D_3, 0, 0);
	while(P_jimb_lvl1_part0T__D_0__D_1__D_2__D_3.Dimension(0) < P_jimb__D_0__D_1__D_2__D_3.Dimension(0))
	{
		RepartitionDown
		( t_fj_lvl1_part1T__D_0_1__D_2_3,  t_fj_lvl1_part1_0__D_0_1__D_2_3,
		  /**/ /**/
		       t_fj_lvl1_part1_1__D_0_1__D_2_3,
		  t_fj_lvl1_part1B__D_0_1__D_2_3, t_fj_lvl1_part1_2__D_0_1__D_2_3, 1, blkSize );
		RepartitionDown
		( P_jimb_lvl1_part0T__D_0__D_1__D_2__D_3,  P_jimb_lvl1_part0_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       P_jimb_lvl1_part0_1__D_0__D_1__D_2__D_3,
		  P_jimb_lvl1_part0B__D_0__D_1__D_2__D_3, P_jimb_lvl1_part0_2__D_0__D_1__D_2__D_3, 0, blkSize );

		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  P_jimb_lvl1_part0_1__D_0__D_1__D_2__D_3
		PartitionDown(w_bmje__D_0__D_1__D_2__D_3, w_bmje_lvl2_part1T__D_0__D_1__D_2__D_3, w_bmje_lvl2_part1B__D_0__D_1__D_2__D_3, 1, 0);
		PartitionDown(P_jimb_lvl1_part0_1__D_0__D_1__D_2__D_3, P_jimb_lvl1_part0_1_lvl2_part2T__D_0__D_1__D_2__D_3, P_jimb_lvl1_part0_1_lvl2_part2B__D_0__D_1__D_2__D_3, 2, 0);
		while(P_jimb_lvl1_part0_1_lvl2_part2T__D_0__D_1__D_2__D_3.Dimension(2) < P_jimb_lvl1_part0_1__D_0__D_1__D_2__D_3.Dimension(2))
		{
			RepartitionDown
			( w_bmje_lvl2_part1T__D_0__D_1__D_2__D_3,  w_bmje_lvl2_part1_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       w_bmje_lvl2_part1_1__D_0__D_1__D_2__D_3,
			  w_bmje_lvl2_part1B__D_0__D_1__D_2__D_3, w_bmje_lvl2_part1_2__D_0__D_1__D_2__D_3, 1, blkSize );
			RepartitionDown
			( P_jimb_lvl1_part0_1_lvl2_part2T__D_0__D_1__D_2__D_3,  P_jimb_lvl1_part0_1_lvl2_part2_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       P_jimb_lvl1_part0_1_lvl2_part2_1__D_0__D_1__D_2__D_3,
			  P_jimb_lvl1_part0_1_lvl2_part2B__D_0__D_1__D_2__D_3, P_jimb_lvl1_part0_1_lvl2_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );

			   // t_fj_lvl1_part1_1[D01,D2] <- t_fj_lvl1_part1_1[D01,D23]
			t_fj_lvl1_part1_1__D_0_1__D_2.AlignModesWith( modes_0, w_bmje_lvl2_part1_1__D_0__D_1__D_2__D_3, modes_3 );
			t_fj_lvl1_part1_1__D_0_1__D_2.AllGatherRedistFrom( t_fj_lvl1_part1_1__D_0_1__D_2_3, modes_3 );
			   // t_fj_lvl1_part1_1[D013,D2] <- t_fj_lvl1_part1_1[D01,D2]
			t_fj_lvl1_part1_1__D_0_1_3__D_2.AlignModesWith( modes_0, w_bmje_lvl2_part1_1__D_0__D_1__D_2__D_3, modes_3 );
			t_fj_lvl1_part1_1__D_0_1_3__D_2.LocalRedistFrom( t_fj_lvl1_part1_1__D_0_1__D_2 );
			t_fj_lvl1_part1_1__D_0_1__D_2.EmptyData();
			   // t_fj_lvl1_part1_1[D301,D2] <- t_fj_lvl1_part1_1[D013,D2]
			t_fj_lvl1_part1_1__D_3_0_1__D_2.AlignModesWith( modes_0, w_bmje_lvl2_part1_1__D_0__D_1__D_2__D_3, modes_3 );
			t_fj_lvl1_part1_1__D_3_0_1__D_2.PermutationRedistFrom( t_fj_lvl1_part1_1__D_0_1_3__D_2, modes_0_1_3 );
			t_fj_lvl1_part1_1__D_0_1_3__D_2.EmptyData();
			   // t_fj_lvl1_part1_1[D3,*] <- t_fj_lvl1_part1_1[D301,D2]
			t_fj_lvl1_part1_1__D_3__S.AlignModesWith( modes_0, w_bmje_lvl2_part1_1__D_0__D_1__D_2__D_3, modes_3 );
			t_fj_lvl1_part1_1__D_3__S.AllGatherRedistFrom( t_fj_lvl1_part1_1__D_3_0_1__D_2, modes_0_1_2 );
			t_fj_lvl1_part1_1__D_3_0_1__D_2.EmptyData();
			P_jimb_lvl1_part0_1_lvl2_part2_1_perm32104__D_0__D_1__D_2__S__D_3.AlignModesWith( modes_1_2_3, w_bmje_lvl2_part1_1__D_0__D_1__D_2__D_3, modes_2_1_0 );
			tempShape = P_jimb_lvl1_part0_1_lvl2_part2_1__D_0__D_1__D_2__D_3.Shape();
			tempShape.push_back( g.Shape()[3] );
			P_jimb_lvl1_part0_1_lvl2_part2_1_perm32104__D_0__D_1__D_2__S__D_3.ResizeTo( tempShape );
			   // 1.0 * w_bmje_lvl2_part1_1[D0,D1,D2,D3]_bmie * t_fj_lvl1_part1_1[D3,*]_ej + 0.0 * P_jimb_lvl1_part0_1_lvl2_part2_1[*,D2,D1,D0,D3]_bmije
			LocalContract(1.0, w_bmje_lvl2_part1_1__D_0__D_1__D_2__D_3.LockedTensor(), indices_bmie, false,
				t_fj_lvl1_part1_1__D_3__S.LockedTensor(), indices_ej, false,
				0.0, P_jimb_lvl1_part0_1_lvl2_part2_1_perm32104__D_0__D_1__D_2__S__D_3.Tensor(), indices_bmije, false);
			t_fj_lvl1_part1_1__D_3__S.EmptyData();
			P_jimb_lvl1_part0_1_lvl2_part2_1_temp__S__D_2__D_1__D_0_3.AlignModesWith( modes_0_1_2_3, P_jimb_lvl1_part0_1_lvl2_part2_1_temp__D_0__D_1__D_2__D_3, modes_0_1_2_3 );
			tempShape = P_jimb_lvl1_part0_1_lvl2_part2_1__D_0__D_1__D_2__D_3.Shape();
			P_jimb_lvl1_part0_1_lvl2_part2_1_temp__S__D_2__D_1__D_0_3.ResizeTo( tempShape );
			   // P_jimb_lvl1_part0_1_lvl2_part2_1_temp[*,D2,D1,D03] <- P_jimb_lvl1_part0_1_lvl2_part2_1[*,D2,D1,D0,D3] (with SumScatter on D3)
			P_jimb_lvl1_part0_1_lvl2_part2_1_temp__S__D_2__D_1__D_0_3.ReduceScatterRedistFrom( P_jimb_lvl1_part0_1_lvl2_part2_1_perm32104__D_0__D_1__D_2__S__D_3, 4 );
			P_jimb_lvl1_part0_1_lvl2_part2_1_perm32104__D_0__D_1__D_2__S__D_3.EmptyData();
			   // P_jimb_lvl1_part0_1_lvl2_part2_1_temp[*,D1,D2,D03] <- P_jimb_lvl1_part0_1_lvl2_part2_1_temp[*,D2,D1,D03]
			P_jimb_lvl1_part0_1_lvl2_part2_1_temp__S__D_1__D_2__D_0_3.AlignModesWith( modes_0_1_2_3, P_jimb_lvl1_part0_1_lvl2_part2_1__D_0__D_1__D_2__D_3, modes_0_1_2_3 );
			P_jimb_lvl1_part0_1_lvl2_part2_1_temp__S__D_1__D_2__D_0_3.AllToAllRedistFrom( P_jimb_lvl1_part0_1_lvl2_part2_1_temp__S__D_2__D_1__D_0_3, modes_1_2 );
			P_jimb_lvl1_part0_1_lvl2_part2_1_temp__S__D_2__D_1__D_0_3.EmptyData();
			   // P_jimb_lvl1_part0_1_lvl2_part2_1_temp[D0,D1,D2,D3] <- P_jimb_lvl1_part0_1_lvl2_part2_1_temp[*,D1,D2,D03]
			P_jimb_lvl1_part0_1_lvl2_part2_1_temp__D_0__D_1__D_2__D_3.AlignModesWith( modes_0_1_2_3, P_jimb_lvl1_part0_1_lvl2_part2_1__D_0__D_1__D_2__D_3, modes_0_1_2_3 );
			P_jimb_lvl1_part0_1_lvl2_part2_1_temp__D_0__D_1__D_2__D_3.AllToAllRedistFrom( P_jimb_lvl1_part0_1_lvl2_part2_1_temp__S__D_1__D_2__D_0_3, modes_0_3 );
			P_jimb_lvl1_part0_1_lvl2_part2_1_temp__S__D_1__D_2__D_0_3.EmptyData();
			YxpBy( P_jimb_lvl1_part0_1_lvl2_part2_1_temp__D_0__D_1__D_2__D_3, 1.0, P_jimb_lvl1_part0_1_lvl2_part2_1__D_0__D_1__D_2__D_3 );
			P_jimb_lvl1_part0_1_lvl2_part2_1_temp__D_0__D_1__D_2__D_3.EmptyData();

			SlidePartitionDown
			( w_bmje_lvl2_part1T__D_0__D_1__D_2__D_3,  w_bmje_lvl2_part1_0__D_0__D_1__D_2__D_3,
			       w_bmje_lvl2_part1_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  w_bmje_lvl2_part1B__D_0__D_1__D_2__D_3, w_bmje_lvl2_part1_2__D_0__D_1__D_2__D_3, 1 );
			SlidePartitionDown
			( P_jimb_lvl1_part0_1_lvl2_part2T__D_0__D_1__D_2__D_3,  P_jimb_lvl1_part0_1_lvl2_part2_0__D_0__D_1__D_2__D_3,
			       P_jimb_lvl1_part0_1_lvl2_part2_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  P_jimb_lvl1_part0_1_lvl2_part2B__D_0__D_1__D_2__D_3, P_jimb_lvl1_part0_1_lvl2_part2_2__D_0__D_1__D_2__D_3, 2 );

		}
		//****

		SlidePartitionDown
		( t_fj_lvl1_part1T__D_0_1__D_2_3,  t_fj_lvl1_part1_0__D_0_1__D_2_3,
		       t_fj_lvl1_part1_1__D_0_1__D_2_3,
		  /**/ /**/
		  t_fj_lvl1_part1B__D_0_1__D_2_3, t_fj_lvl1_part1_2__D_0_1__D_2_3, 1 );
		SlidePartitionDown
		( P_jimb_lvl1_part0T__D_0__D_1__D_2__D_3,  P_jimb_lvl1_part0_0__D_0__D_1__D_2__D_3,
		       P_jimb_lvl1_part0_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  P_jimb_lvl1_part0B__D_0__D_1__D_2__D_3, P_jimb_lvl1_part0_2__D_0__D_1__D_2__D_3, 0 );

	}
	//****
	//**** (out of 1)
	//**** Is real	0 shadows
		//Outputs:
		//  P_jimb__D_0__D_1__D_2__D_3
	PartitionDown(Tau_efmn__D_0__D_1__D_2__D_3, Tau_efmn_lvl1_part3T__D_0__D_1__D_2__D_3, Tau_efmn_lvl1_part3B__D_0__D_1__D_2__D_3, 3, 0);
	PartitionDown(P_jimb__D_0__D_1__D_2__D_3, P_jimb_lvl1_part0T__D_0__D_1__D_2__D_3, P_jimb_lvl1_part0B__D_0__D_1__D_2__D_3, 0, 0);
	while(P_jimb_lvl1_part0T__D_0__D_1__D_2__D_3.Dimension(0) < P_jimb__D_0__D_1__D_2__D_3.Dimension(0))
	{
		RepartitionDown
		( Tau_efmn_lvl1_part3T__D_0__D_1__D_2__D_3,  Tau_efmn_lvl1_part3_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       Tau_efmn_lvl1_part3_1__D_0__D_1__D_2__D_3,
		  Tau_efmn_lvl1_part3B__D_0__D_1__D_2__D_3, Tau_efmn_lvl1_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );
		RepartitionDown
		( P_jimb_lvl1_part0T__D_0__D_1__D_2__D_3,  P_jimb_lvl1_part0_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       P_jimb_lvl1_part0_1__D_0__D_1__D_2__D_3,
		  P_jimb_lvl1_part0B__D_0__D_1__D_2__D_3, P_jimb_lvl1_part0_2__D_0__D_1__D_2__D_3, 0, blkSize );

		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  P_jimb_lvl1_part0_1__D_0__D_1__D_2__D_3
		PartitionDown(Tau_efmn_lvl1_part3_1__D_0__D_1__D_2__D_3, Tau_efmn_lvl1_part3_1_lvl2_part2T__D_0__D_1__D_2__D_3, Tau_efmn_lvl1_part3_1_lvl2_part2B__D_0__D_1__D_2__D_3, 2, 0);
		PartitionDown(P_jimb_lvl1_part0_1__D_0__D_1__D_2__D_3, P_jimb_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3, P_jimb_lvl1_part0_1_lvl2_part1B__D_0__D_1__D_2__D_3, 1, 0);
		while(P_jimb_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3.Dimension(1) < P_jimb_lvl1_part0_1__D_0__D_1__D_2__D_3.Dimension(1))
		{
			RepartitionDown
			( Tau_efmn_lvl1_part3_1_lvl2_part2T__D_0__D_1__D_2__D_3,  Tau_efmn_lvl1_part3_1_lvl2_part2_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       Tau_efmn_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_2__D_3,
			  Tau_efmn_lvl1_part3_1_lvl2_part2B__D_0__D_1__D_2__D_3, Tau_efmn_lvl1_part3_1_lvl2_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );
			RepartitionDown
			( P_jimb_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3,  P_jimb_lvl1_part0_1_lvl2_part1_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       P_jimb_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3,
			  P_jimb_lvl1_part0_1_lvl2_part1B__D_0__D_1__D_2__D_3, P_jimb_lvl1_part0_1_lvl2_part1_2__D_0__D_1__D_2__D_3, 1, blkSize );

			   // Tau_efmn_lvl1_part3_1_lvl2_part2_1[D2,D1,D0,D3] <- Tau_efmn_lvl1_part3_1_lvl2_part2_1[D0,D1,D2,D3]
			Tau_efmn_lvl1_part3_1_lvl2_part2_1__D_2__D_1__D_0__D_3.AlignModesWith( modes_0_1, r_bmfe__D_0__D_1__D_2__D_3, modes_2_3 );
			Tau_efmn_lvl1_part3_1_lvl2_part2_1__D_2__D_1__D_0__D_3.AllToAllRedistFrom( Tau_efmn_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_2__D_3, modes_0_2 );
			   // Tau_efmn_lvl1_part3_1_lvl2_part2_1[D2,D3,D0,*] <- Tau_efmn_lvl1_part3_1_lvl2_part2_1[D2,D1,D0,D3]
			Tau_efmn_lvl1_part3_1_lvl2_part2_1__D_2__D_3__D_0__S.AlignModesWith( modes_0_1, r_bmfe__D_0__D_1__D_2__D_3, modes_2_3 );
			Tau_efmn_lvl1_part3_1_lvl2_part2_1__D_2__D_3__D_0__S.AllToAllRedistFrom( Tau_efmn_lvl1_part3_1_lvl2_part2_1__D_2__D_1__D_0__D_3, modes_1_3 );
			Tau_efmn_lvl1_part3_1_lvl2_part2_1__D_2__D_1__D_0__D_3.EmptyData();
			   // Tau_efmn_lvl1_part3_1_lvl2_part2_1[D2,D3,*,*] <- Tau_efmn_lvl1_part3_1_lvl2_part2_1[D2,D3,D0,*]
			Tau_efmn_lvl1_part3_1_lvl2_part2_1__D_2__D_3__S__S.AlignModesWith( modes_0_1, r_bmfe__D_0__D_1__D_2__D_3, modes_2_3 );
			Tau_efmn_lvl1_part3_1_lvl2_part2_1__D_2__D_3__S__S.AllGatherRedistFrom( Tau_efmn_lvl1_part3_1_lvl2_part2_1__D_2__D_3__D_0__S, modes_0 );
			Tau_efmn_lvl1_part3_1_lvl2_part2_1__D_2__D_3__D_0__S.EmptyData();
			P_jimb_lvl1_part0_1_lvl2_part1_1_perm321045__D_0__D_1__S__S__D_2__D_3.AlignModesWith( modes_2_3, r_bmfe__D_0__D_1__D_2__D_3, modes_1_0 );
			tempShape = P_jimb_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3.Shape();
			tempShape.push_back( g.Shape()[2] );
			tempShape.push_back( g.Shape()[3] );
			P_jimb_lvl1_part0_1_lvl2_part1_1_perm321045__D_0__D_1__S__S__D_2__D_3.ResizeTo( tempShape );
			   // 1.0 * r_bmfe[D0,D1,D2,D3]_bmef * Tau_efmn_lvl1_part3_1_lvl2_part2_1[D2,D3,*,*]_efij + 0.0 * P_jimb_lvl1_part0_1_lvl2_part1_1[*,*,D1,D0,D2,D3]_bmijef
			LocalContract(1.0, r_bmfe__D_0__D_1__D_2__D_3.LockedTensor(), indices_bmef, false,
				Tau_efmn_lvl1_part3_1_lvl2_part2_1__D_2__D_3__S__S.LockedTensor(), indices_efij, false,
				0.0, P_jimb_lvl1_part0_1_lvl2_part1_1_perm321045__D_0__D_1__S__S__D_2__D_3.Tensor(), indices_bmijef, false);
			Tau_efmn_lvl1_part3_1_lvl2_part2_1__D_2__D_3__S__S.EmptyData();
			P_jimb_lvl1_part0_1_lvl2_part1_1_temp__S__S__D_1_2__D_0_3.AlignModesWith( modes_0_1_2_3, P_jimb_lvl1_part0_1_lvl2_part1_1_temp__D_0__D_1__D_2__D_3, modes_0_1_2_3 );
			tempShape = P_jimb_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3.Shape();
			P_jimb_lvl1_part0_1_lvl2_part1_1_temp__S__S__D_1_2__D_0_3.ResizeTo( tempShape );
			   // P_jimb_lvl1_part0_1_lvl2_part1_1_temp[*,*,D12,D03] <- P_jimb_lvl1_part0_1_lvl2_part1_1[*,*,D1,D0,D2,D3] (with SumScatter on (D2)(D3))
			P_jimb_lvl1_part0_1_lvl2_part1_1_temp__S__S__D_1_2__D_0_3.ReduceScatterRedistFrom( P_jimb_lvl1_part0_1_lvl2_part1_1_perm321045__D_0__D_1__S__S__D_2__D_3, modes_5_4 );
			P_jimb_lvl1_part0_1_lvl2_part1_1_perm321045__D_0__D_1__S__S__D_2__D_3.EmptyData();
			   // P_jimb_lvl1_part0_1_lvl2_part1_1_temp[*,*,D21,D03] <- P_jimb_lvl1_part0_1_lvl2_part1_1_temp[*,*,D12,D03]
			P_jimb_lvl1_part0_1_lvl2_part1_1_temp__S__S__D_2_1__D_0_3.AlignModesWith( modes_0_1_2_3, P_jimb_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3, modes_0_1_2_3 );
			P_jimb_lvl1_part0_1_lvl2_part1_1_temp__S__S__D_2_1__D_0_3.PermutationRedistFrom( P_jimb_lvl1_part0_1_lvl2_part1_1_temp__S__S__D_1_2__D_0_3, modes_1_2 );
			P_jimb_lvl1_part0_1_lvl2_part1_1_temp__S__S__D_1_2__D_0_3.EmptyData();
			   // P_jimb_lvl1_part0_1_lvl2_part1_1_temp[*,D1,D2,D03] <- P_jimb_lvl1_part0_1_lvl2_part1_1_temp[*,*,D21,D03]
			P_jimb_lvl1_part0_1_lvl2_part1_1_temp__S__D_1__D_2__D_0_3.AlignModesWith( modes_0_1_2_3, P_jimb_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3, modes_0_1_2_3 );
			P_jimb_lvl1_part0_1_lvl2_part1_1_temp__S__D_1__D_2__D_0_3.AllToAllRedistFrom( P_jimb_lvl1_part0_1_lvl2_part1_1_temp__S__S__D_2_1__D_0_3, modes_1 );
			P_jimb_lvl1_part0_1_lvl2_part1_1_temp__S__S__D_2_1__D_0_3.EmptyData();
			   // P_jimb_lvl1_part0_1_lvl2_part1_1_temp[D0,D1,D2,D3] <- P_jimb_lvl1_part0_1_lvl2_part1_1_temp[*,D1,D2,D03]
			P_jimb_lvl1_part0_1_lvl2_part1_1_temp__D_0__D_1__D_2__D_3.AlignModesWith( modes_0_1_2_3, P_jimb_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3, modes_0_1_2_3 );
			P_jimb_lvl1_part0_1_lvl2_part1_1_temp__D_0__D_1__D_2__D_3.AllToAllRedistFrom( P_jimb_lvl1_part0_1_lvl2_part1_1_temp__S__D_1__D_2__D_0_3, modes_0_3 );
			P_jimb_lvl1_part0_1_lvl2_part1_1_temp__S__D_1__D_2__D_0_3.EmptyData();
			YxpBy( P_jimb_lvl1_part0_1_lvl2_part1_1_temp__D_0__D_1__D_2__D_3, 1.0, P_jimb_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3 );
			P_jimb_lvl1_part0_1_lvl2_part1_1_temp__D_0__D_1__D_2__D_3.EmptyData();

			SlidePartitionDown
			( Tau_efmn_lvl1_part3_1_lvl2_part2T__D_0__D_1__D_2__D_3,  Tau_efmn_lvl1_part3_1_lvl2_part2_0__D_0__D_1__D_2__D_3,
			       Tau_efmn_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  Tau_efmn_lvl1_part3_1_lvl2_part2B__D_0__D_1__D_2__D_3, Tau_efmn_lvl1_part3_1_lvl2_part2_2__D_0__D_1__D_2__D_3, 2 );
			SlidePartitionDown
			( P_jimb_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3,  P_jimb_lvl1_part0_1_lvl2_part1_0__D_0__D_1__D_2__D_3,
			       P_jimb_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  P_jimb_lvl1_part0_1_lvl2_part1B__D_0__D_1__D_2__D_3, P_jimb_lvl1_part0_1_lvl2_part1_2__D_0__D_1__D_2__D_3, 1 );

		}
		//****

		SlidePartitionDown
		( Tau_efmn_lvl1_part3T__D_0__D_1__D_2__D_3,  Tau_efmn_lvl1_part3_0__D_0__D_1__D_2__D_3,
		       Tau_efmn_lvl1_part3_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  Tau_efmn_lvl1_part3B__D_0__D_1__D_2__D_3, Tau_efmn_lvl1_part3_2__D_0__D_1__D_2__D_3, 3 );
		SlidePartitionDown
		( P_jimb_lvl1_part0T__D_0__D_1__D_2__D_3,  P_jimb_lvl1_part0_0__D_0__D_1__D_2__D_3,
		       P_jimb_lvl1_part0_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  P_jimb_lvl1_part0B__D_0__D_1__D_2__D_3, P_jimb_lvl1_part0_2__D_0__D_1__D_2__D_3, 0 );

	}
	//****


//****


//**** (out of 1)
//**** Is real	0 shadows
	//Outputs:
	//  H_me__D_0_1__D_2_3

	tempShape = v_femn__D_0__D_1__D_2__D_3.Shape();
	Htemp1__D_0__D_1__D_2__D_3.ResizeTo( tempShape );
	//**** (out of 1)
	//**** Is real	0 shadows
		//Outputs:
		//  Htemp1__D_0__D_1__D_2__D_3
	PartitionDown(v_femn__D_0__D_1__D_2__D_3, v_femn_lvl1_part2T__D_0__D_1__D_2__D_3, v_femn_lvl1_part2B__D_0__D_1__D_2__D_3, 2, 0);
	PartitionDown(v_femn__D_0__D_1__D_2__D_3, v_femn_lvl1_part3T__D_0__D_1__D_2__D_3, v_femn_lvl1_part3B__D_0__D_1__D_2__D_3, 3, 0);
	PartitionDown(Htemp1__D_0__D_1__D_2__D_3, Htemp1_lvl1_part2T__D_0__D_1__D_2__D_3, Htemp1_lvl1_part2B__D_0__D_1__D_2__D_3, 2, 0);
	while(Htemp1_lvl1_part2T__D_0__D_1__D_2__D_3.Dimension(2) < Htemp1__D_0__D_1__D_2__D_3.Dimension(2))
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
		( Htemp1_lvl1_part2T__D_0__D_1__D_2__D_3,  Htemp1_lvl1_part2_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       Htemp1_lvl1_part2_1__D_0__D_1__D_2__D_3,
		  Htemp1_lvl1_part2B__D_0__D_1__D_2__D_3, Htemp1_lvl1_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );

		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  Htemp1_lvl1_part2_1__D_0__D_1__D_2__D_3
		PartitionDown(v_femn_lvl1_part2_1__D_0__D_1__D_2__D_3, v_femn_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3, v_femn_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, 3, 0);
		PartitionDown(v_femn_lvl1_part3_1__D_0__D_1__D_2__D_3, v_femn_lvl1_part3_1_lvl2_part2T__D_0__D_1__D_2__D_3, v_femn_lvl1_part3_1_lvl2_part2B__D_0__D_1__D_2__D_3, 2, 0);
		PartitionDown(Htemp1_lvl1_part2_1__D_0__D_1__D_2__D_3, Htemp1_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3, Htemp1_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, 3, 0);
		while(Htemp1_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3.Dimension(3) < Htemp1_lvl1_part2_1__D_0__D_1__D_2__D_3.Dimension(3))
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
			( Htemp1_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3,  Htemp1_lvl1_part2_1_lvl2_part3_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       Htemp1_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3,
			  Htemp1_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, Htemp1_lvl1_part2_1_lvl2_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );

			   // v_femn_lvl1_part3_1_lvl2_part2_1[D0,D1,D3,D2] <- v_femn_lvl1_part3_1_lvl2_part2_1[D0,D1,D2,D3]
			v_femn_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_3__D_2.AlignModesWith( modes_0_1_2_3, v_femn_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3, modes_0_1_3_2 );
			v_femn_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_3__D_2.AllToAllRedistFrom( v_femn_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_2__D_3, modes_2_3 );
			YAxpPx( 2.0, v_femn_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3, -1.0, v_femn_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_3__D_2, perm_0_1_3_2, Htemp1_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3 );
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
			( Htemp1_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3,  Htemp1_lvl1_part2_1_lvl2_part3_0__D_0__D_1__D_2__D_3,
			       Htemp1_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  Htemp1_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, Htemp1_lvl1_part2_1_lvl2_part3_2__D_0__D_1__D_2__D_3, 3 );

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
		( Htemp1_lvl1_part2T__D_0__D_1__D_2__D_3,  Htemp1_lvl1_part2_0__D_0__D_1__D_2__D_3,
		       Htemp1_lvl1_part2_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  Htemp1_lvl1_part2B__D_0__D_1__D_2__D_3, Htemp1_lvl1_part2_2__D_0__D_1__D_2__D_3, 2 );

	}
	//****
	//**** (out of 1)
	//**** Is real	0 shadows
		//Outputs:
		//  H_me__D_0_1__D_2_3
	PartitionDown(Htemp1__D_0__D_1__D_2__D_3, Htemp1_lvl1_part2T__D_0__D_1__D_2__D_3, Htemp1_lvl1_part2B__D_0__D_1__D_2__D_3, 2, 0);
	PartitionDown(H_me__D_0_1__D_2_3, H_me_lvl1_part0T__D_0_1__D_2_3, H_me_lvl1_part0B__D_0_1__D_2_3, 0, 0);
	while(H_me_lvl1_part0T__D_0_1__D_2_3.Dimension(0) < H_me__D_0_1__D_2_3.Dimension(0))
	{
		RepartitionDown
		( Htemp1_lvl1_part2T__D_0__D_1__D_2__D_3,  Htemp1_lvl1_part2_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       Htemp1_lvl1_part2_1__D_0__D_1__D_2__D_3,
		  Htemp1_lvl1_part2B__D_0__D_1__D_2__D_3, Htemp1_lvl1_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );
		RepartitionDown
		( H_me_lvl1_part0T__D_0_1__D_2_3,  H_me_lvl1_part0_0__D_0_1__D_2_3,
		  /**/ /**/
		       H_me_lvl1_part0_1__D_0_1__D_2_3,
		  H_me_lvl1_part0B__D_0_1__D_2_3, H_me_lvl1_part0_2__D_0_1__D_2_3, 0, blkSize );

		Scal( 0.0, H_me_lvl1_part0_1__D_0_1__D_2_3 );
		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  H_me_lvl1_part0_1__D_0_1__D_2_3
		PartitionDown(Htemp1_lvl1_part2_1__D_0__D_1__D_2__D_3, Htemp1_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3, Htemp1_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, 3, 0);
		PartitionDown(t_fj__D_0_1__D_2_3, t_fj_lvl2_part1T__D_0_1__D_2_3, t_fj_lvl2_part1B__D_0_1__D_2_3, 1, 0);
		while(Htemp1_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3.Dimension(3) < Htemp1_lvl1_part2_1__D_0__D_1__D_2__D_3.Dimension(3))
		{
			RepartitionDown
			( Htemp1_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3,  Htemp1_lvl1_part2_1_lvl2_part3_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       Htemp1_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3,
			  Htemp1_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, Htemp1_lvl1_part2_1_lvl2_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );
			RepartitionDown
			( t_fj_lvl2_part1T__D_0_1__D_2_3,  t_fj_lvl2_part1_0__D_0_1__D_2_3,
			  /**/ /**/
			       t_fj_lvl2_part1_1__D_0_1__D_2_3,
			  t_fj_lvl2_part1B__D_0_1__D_2_3, t_fj_lvl2_part1_2__D_0_1__D_2_3, 1, blkSize );

			   // t_fj_lvl2_part1_1[D10,D23] <- t_fj_lvl2_part1_1[D01,D23]
			t_fj_lvl2_part1_1__D_1_0__D_2_3.AlignModesWith( modes_0_1, Htemp1_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3, modes_1_3 );
			t_fj_lvl2_part1_1__D_1_0__D_2_3.PermutationRedistFrom( t_fj_lvl2_part1_1__D_0_1__D_2_3, modes_0_1 );
			   // t_fj_lvl2_part1_1[D10,D32] <- t_fj_lvl2_part1_1[D10,D23]
			t_fj_lvl2_part1_1__D_1_0__D_3_2.AlignModesWith( modes_0_1, Htemp1_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3, modes_1_3 );
			t_fj_lvl2_part1_1__D_1_0__D_3_2.PermutationRedistFrom( t_fj_lvl2_part1_1__D_1_0__D_2_3, modes_2_3 );
			t_fj_lvl2_part1_1__D_1_0__D_2_3.EmptyData();
			   // t_fj_lvl2_part1_1[D1,D3] <- t_fj_lvl2_part1_1[D10,D32]
			t_fj_lvl2_part1_1__D_1__D_3.AlignModesWith( modes_0_1, Htemp1_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3, modes_1_3 );
			t_fj_lvl2_part1_1__D_1__D_3.AllGatherRedistFrom( t_fj_lvl2_part1_1__D_1_0__D_3_2, modes_0_2 );
			t_fj_lvl2_part1_1__D_1_0__D_3_2.EmptyData();
			Permute( Htemp1_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3, Htemp1_lvl1_part2_1_lvl2_part3_1_perm0213__D_0__D_2__D_1__D_3 );
			H_me_lvl1_part0_1_perm1023__D_0__D_2__D_1__D_3.AlignModesWith( modes_0_1, Htemp1_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3, modes_2_0 );
			tempShape = H_me_lvl1_part0_1__D_0_1__D_2_3.Shape();
			tempShape.push_back( g.Shape()[1] );
			tempShape.push_back( g.Shape()[3] );
			H_me_lvl1_part0_1_perm1023__D_0__D_2__D_1__D_3.ResizeTo( tempShape );
			   // 1.0 * Htemp1_lvl1_part2_1_lvl2_part3_1[D0,D1,D2,D3]_emfn * t_fj_lvl2_part1_1[D1,D3]_fn + 0.0 * H_me_lvl1_part0_1[D2,D0,D1,D3]_emfn
			LocalContract(1.0, Htemp1_lvl1_part2_1_lvl2_part3_1_perm0213__D_0__D_2__D_1__D_3.LockedTensor(), indices_emfn, false,
				t_fj_lvl2_part1_1__D_1__D_3.LockedTensor(), indices_fn, false,
				0.0, H_me_lvl1_part0_1_perm1023__D_0__D_2__D_1__D_3.Tensor(), indices_emfn, false);
			Htemp1_lvl1_part2_1_lvl2_part3_1_perm0213__D_0__D_2__D_1__D_3.EmptyData();
			t_fj_lvl2_part1_1__D_1__D_3.EmptyData();
			H_me_lvl1_part0_1_temp__D_2_1__D_0_3.AlignModesWith( modes_0_1, H_me_lvl1_part0_1_temp__D_0_1__D_2_3, modes_0_1 );
			tempShape = H_me_lvl1_part0_1__D_0_1__D_2_3.Shape();
			H_me_lvl1_part0_1_temp__D_2_1__D_0_3.ResizeTo( tempShape );
			   // H_me_lvl1_part0_1_temp[D21,D03] <- H_me_lvl1_part0_1[D2,D0,D1,D3] (with SumScatter on (D1)(D3))
			H_me_lvl1_part0_1_temp__D_2_1__D_0_3.ReduceScatterRedistFrom( H_me_lvl1_part0_1_perm1023__D_0__D_2__D_1__D_3, modes_3_2 );
			H_me_lvl1_part0_1_perm1023__D_0__D_2__D_1__D_3.EmptyData();
			   // H_me_lvl1_part0_1_temp[D12,D03] <- H_me_lvl1_part0_1_temp[D21,D03]
			H_me_lvl1_part0_1_temp__D_1_2__D_0_3.AlignModesWith( modes_0_1, H_me_lvl1_part0_1__D_0_1__D_2_3, modes_0_1 );
			H_me_lvl1_part0_1_temp__D_1_2__D_0_3.PermutationRedistFrom( H_me_lvl1_part0_1_temp__D_2_1__D_0_3, modes_2_1 );
			H_me_lvl1_part0_1_temp__D_2_1__D_0_3.EmptyData();
			   // H_me_lvl1_part0_1_temp[D1,D03] <- H_me_lvl1_part0_1_temp[D12,D03]
			H_me_lvl1_part0_1_temp__D_1__D_0_3.AlignModesWith( modes_0_1, H_me_lvl1_part0_1__D_0_1__D_2_3, modes_0_1 );
			H_me_lvl1_part0_1_temp__D_1__D_0_3.AllGatherRedistFrom( H_me_lvl1_part0_1_temp__D_1_2__D_0_3, modes_2 );
			H_me_lvl1_part0_1_temp__D_1_2__D_0_3.EmptyData();
			   // H_me_lvl1_part0_1_temp[D1,D032] <- H_me_lvl1_part0_1_temp[D1,D03]
			H_me_lvl1_part0_1_temp__D_1__D_0_3_2.AlignModesWith( modes_0_1, H_me_lvl1_part0_1__D_0_1__D_2_3, modes_0_1 );
			H_me_lvl1_part0_1_temp__D_1__D_0_3_2.LocalRedistFrom( H_me_lvl1_part0_1_temp__D_1__D_0_3 );
			H_me_lvl1_part0_1_temp__D_1__D_0_3.EmptyData();
			   // H_me_lvl1_part0_1_temp[D1,D230] <- H_me_lvl1_part0_1_temp[D1,D032]
			H_me_lvl1_part0_1_temp__D_1__D_2_3_0.AlignModesWith( modes_0_1, H_me_lvl1_part0_1__D_0_1__D_2_3, modes_0_1 );
			H_me_lvl1_part0_1_temp__D_1__D_2_3_0.PermutationRedistFrom( H_me_lvl1_part0_1_temp__D_1__D_0_3_2, modes_0_3_2 );
			H_me_lvl1_part0_1_temp__D_1__D_0_3_2.EmptyData();
			   // H_me_lvl1_part0_1_temp[D1,D23] <- H_me_lvl1_part0_1_temp[D1,D230]
			H_me_lvl1_part0_1_temp__D_1__D_2_3.AlignModesWith( modes_0_1, H_me_lvl1_part0_1__D_0_1__D_2_3, modes_0_1 );
			H_me_lvl1_part0_1_temp__D_1__D_2_3.AllGatherRedistFrom( H_me_lvl1_part0_1_temp__D_1__D_2_3_0, modes_0 );
			H_me_lvl1_part0_1_temp__D_1__D_2_3_0.EmptyData();
			   // H_me_lvl1_part0_1_temp[D10,D23] <- H_me_lvl1_part0_1_temp[D1,D23]
			H_me_lvl1_part0_1_temp__D_1_0__D_2_3.AlignModesWith( modes_0_1, H_me_lvl1_part0_1__D_0_1__D_2_3, modes_0_1 );
			H_me_lvl1_part0_1_temp__D_1_0__D_2_3.LocalRedistFrom( H_me_lvl1_part0_1_temp__D_1__D_2_3 );
			H_me_lvl1_part0_1_temp__D_1__D_2_3.EmptyData();
			   // H_me_lvl1_part0_1_temp[D01,D23] <- H_me_lvl1_part0_1_temp[D10,D23]
			H_me_lvl1_part0_1_temp__D_0_1__D_2_3.AlignModesWith( modes_0_1, H_me_lvl1_part0_1__D_0_1__D_2_3, modes_0_1 );
			H_me_lvl1_part0_1_temp__D_0_1__D_2_3.PermutationRedistFrom( H_me_lvl1_part0_1_temp__D_1_0__D_2_3, modes_1_0 );
			H_me_lvl1_part0_1_temp__D_1_0__D_2_3.EmptyData();
			YxpBy( H_me_lvl1_part0_1_temp__D_0_1__D_2_3, 1.0, H_me_lvl1_part0_1__D_0_1__D_2_3 );
			H_me_lvl1_part0_1_temp__D_0_1__D_2_3.EmptyData();

			SlidePartitionDown
			( Htemp1_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3,  Htemp1_lvl1_part2_1_lvl2_part3_0__D_0__D_1__D_2__D_3,
			       Htemp1_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  Htemp1_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, Htemp1_lvl1_part2_1_lvl2_part3_2__D_0__D_1__D_2__D_3, 3 );
			SlidePartitionDown
			( t_fj_lvl2_part1T__D_0_1__D_2_3,  t_fj_lvl2_part1_0__D_0_1__D_2_3,
			       t_fj_lvl2_part1_1__D_0_1__D_2_3,
			  /**/ /**/
			  t_fj_lvl2_part1B__D_0_1__D_2_3, t_fj_lvl2_part1_2__D_0_1__D_2_3, 1 );

		}
		//****

		SlidePartitionDown
		( Htemp1_lvl1_part2T__D_0__D_1__D_2__D_3,  Htemp1_lvl1_part2_0__D_0__D_1__D_2__D_3,
		       Htemp1_lvl1_part2_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  Htemp1_lvl1_part2B__D_0__D_1__D_2__D_3, Htemp1_lvl1_part2_2__D_0__D_1__D_2__D_3, 2 );
		SlidePartitionDown
		( H_me_lvl1_part0T__D_0_1__D_2_3,  H_me_lvl1_part0_0__D_0_1__D_2_3,
		       H_me_lvl1_part0_1__D_0_1__D_2_3,
		  /**/ /**/
		  H_me_lvl1_part0B__D_0_1__D_2_3, H_me_lvl1_part0_2__D_0_1__D_2_3, 0 );

	}
	//****
	Htemp1__D_0__D_1__D_2__D_3.EmptyData();


//****


//**** (out of 1)
//**** Is real	0 shadows
	//Outputs:
	//  F_ae__D_0_1__D_2_3

	Scal( 0.0, F_ae__D_0_1__D_2_3 );
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
			Permute( Ftemp2_lvl1_part2_1_lvl2_part1_1__D_0__D_1__D_2__D_3, Ftemp2_lvl1_part2_1_lvl2_part1_1_perm0231__D_0__D_2__D_3__D_1 );
			F_ae_lvl1_part1_1__D_0__D_2__D_3__D_1.AlignModesWith( modes_0_1, Ftemp2_lvl1_part2_1_lvl2_part1_1__D_0__D_1__D_2__D_3, modes_0_2 );
			tempShape = F_ae_lvl1_part1_1__D_0_1__D_2_3.Shape();
			tempShape.push_back( g.Shape()[3] );
			tempShape.push_back( g.Shape()[1] );
			F_ae_lvl1_part1_1__D_0__D_2__D_3__D_1.ResizeTo( tempShape );
			   // 1.0 * Ftemp2_lvl1_part2_1_lvl2_part1_1[D0,D1,D2,D3]_aefm * t_fj_lvl2_part1_1[D3,D1]_fm + 0.0 * F_ae_lvl1_part1_1[D0,D2,D3,D1]_aefm
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


//**** (out of 1)
//**** Is real	0 shadows
	//Outputs:
	//  G_mi__D_0_1__D_2_3

	tempShape = v_femn__D_0__D_1__D_2__D_3.Shape();
	Gtemp1__D_0__D_1__D_2__D_3.ResizeTo( tempShape );
	//**** (out of 1)
	//**** Is real	0 shadows
		//Outputs:
		//  Gtemp1__D_0__D_1__D_2__D_3
	PartitionDown(v_femn__D_0__D_1__D_2__D_3, v_femn_lvl1_part2T__D_0__D_1__D_2__D_3, v_femn_lvl1_part2B__D_0__D_1__D_2__D_3, 2, 0);
	PartitionDown(v_femn__D_0__D_1__D_2__D_3, v_femn_lvl1_part3T__D_0__D_1__D_2__D_3, v_femn_lvl1_part3B__D_0__D_1__D_2__D_3, 3, 0);
	PartitionDown(Gtemp1__D_0__D_1__D_2__D_3, Gtemp1_lvl1_part2T__D_0__D_1__D_2__D_3, Gtemp1_lvl1_part2B__D_0__D_1__D_2__D_3, 2, 0);
	while(Gtemp1_lvl1_part2T__D_0__D_1__D_2__D_3.Dimension(2) < Gtemp1__D_0__D_1__D_2__D_3.Dimension(2))
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
		( Gtemp1_lvl1_part2T__D_0__D_1__D_2__D_3,  Gtemp1_lvl1_part2_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       Gtemp1_lvl1_part2_1__D_0__D_1__D_2__D_3,
		  Gtemp1_lvl1_part2B__D_0__D_1__D_2__D_3, Gtemp1_lvl1_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );

		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  Gtemp1_lvl1_part2_1__D_0__D_1__D_2__D_3
		PartitionDown(v_femn_lvl1_part2_1__D_0__D_1__D_2__D_3, v_femn_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3, v_femn_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, 3, 0);
		PartitionDown(v_femn_lvl1_part3_1__D_0__D_1__D_2__D_3, v_femn_lvl1_part3_1_lvl2_part2T__D_0__D_1__D_2__D_3, v_femn_lvl1_part3_1_lvl2_part2B__D_0__D_1__D_2__D_3, 2, 0);
		PartitionDown(Gtemp1_lvl1_part2_1__D_0__D_1__D_2__D_3, Gtemp1_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3, Gtemp1_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, 3, 0);
		while(Gtemp1_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3.Dimension(3) < Gtemp1_lvl1_part2_1__D_0__D_1__D_2__D_3.Dimension(3))
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
			( Gtemp1_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3,  Gtemp1_lvl1_part2_1_lvl2_part3_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       Gtemp1_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3,
			  Gtemp1_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, Gtemp1_lvl1_part2_1_lvl2_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );

			   // v_femn_lvl1_part3_1_lvl2_part2_1[D0,D1,D3,D2] <- v_femn_lvl1_part3_1_lvl2_part2_1[D0,D1,D2,D3]
			v_femn_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_3__D_2.AlignModesWith( modes_0_1_2_3, v_femn_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3, modes_0_1_3_2 );
			v_femn_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_3__D_2.AllToAllRedistFrom( v_femn_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_2__D_3, modes_2_3 );
			YAxpPx( 2.0, v_femn_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3, -1.0, v_femn_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_3__D_2, perm_0_1_3_2, Gtemp1_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3 );
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
			( Gtemp1_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3,  Gtemp1_lvl1_part2_1_lvl2_part3_0__D_0__D_1__D_2__D_3,
			       Gtemp1_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  Gtemp1_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, Gtemp1_lvl1_part2_1_lvl2_part3_2__D_0__D_1__D_2__D_3, 3 );

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
		( Gtemp1_lvl1_part2T__D_0__D_1__D_2__D_3,  Gtemp1_lvl1_part2_0__D_0__D_1__D_2__D_3,
		       Gtemp1_lvl1_part2_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  Gtemp1_lvl1_part2B__D_0__D_1__D_2__D_3, Gtemp1_lvl1_part2_2__D_0__D_1__D_2__D_3, 2 );

	}
	//****
	//**** (out of 1)
	//**** Is real	0 shadows
		//Outputs:
		//  G_mi__D_0_1__D_2_3
	PartitionDown(Gtemp1__D_0__D_1__D_2__D_3, Gtemp1_lvl1_part2T__D_0__D_1__D_2__D_3, Gtemp1_lvl1_part2B__D_0__D_1__D_2__D_3, 2, 0);
	PartitionDown(G_mi__D_0_1__D_2_3, G_mi_lvl1_part0T__D_0_1__D_2_3, G_mi_lvl1_part0B__D_0_1__D_2_3, 0, 0);
	while(G_mi_lvl1_part0T__D_0_1__D_2_3.Dimension(0) < G_mi__D_0_1__D_2_3.Dimension(0))
	{
		RepartitionDown
		( Gtemp1_lvl1_part2T__D_0__D_1__D_2__D_3,  Gtemp1_lvl1_part2_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       Gtemp1_lvl1_part2_1__D_0__D_1__D_2__D_3,
		  Gtemp1_lvl1_part2B__D_0__D_1__D_2__D_3, Gtemp1_lvl1_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );
		RepartitionDown
		( G_mi_lvl1_part0T__D_0_1__D_2_3,  G_mi_lvl1_part0_0__D_0_1__D_2_3,
		  /**/ /**/
		       G_mi_lvl1_part0_1__D_0_1__D_2_3,
		  G_mi_lvl1_part0B__D_0_1__D_2_3, G_mi_lvl1_part0_2__D_0_1__D_2_3, 0, blkSize );

		Scal( 0.0, G_mi_lvl1_part0_1__D_0_1__D_2_3 );
		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  G_mi_lvl1_part0_1__D_0_1__D_2_3
		PartitionDown(Gtemp1_lvl1_part2_1__D_0__D_1__D_2__D_3, Gtemp1_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3, Gtemp1_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, 3, 0);
		PartitionDown(T_bfnj__D_0__D_1__D_2__D_3, T_bfnj_lvl2_part3T__D_0__D_1__D_2__D_3, T_bfnj_lvl2_part3B__D_0__D_1__D_2__D_3, 3, 0);
		while(Gtemp1_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3.Dimension(3) < Gtemp1_lvl1_part2_1__D_0__D_1__D_2__D_3.Dimension(3))
		{
			RepartitionDown
			( Gtemp1_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3,  Gtemp1_lvl1_part2_1_lvl2_part3_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       Gtemp1_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3,
			  Gtemp1_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, Gtemp1_lvl1_part2_1_lvl2_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );
			RepartitionDown
			( T_bfnj_lvl2_part3T__D_0__D_1__D_2__D_3,  T_bfnj_lvl2_part3_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       T_bfnj_lvl2_part3_1__D_0__D_1__D_2__D_3,
			  T_bfnj_lvl2_part3B__D_0__D_1__D_2__D_3, T_bfnj_lvl2_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );

			   // Gtemp1_lvl1_part2_1_lvl2_part3_1[D0,D1,*,D3] <- Gtemp1_lvl1_part2_1_lvl2_part3_1[D0,D1,D2,D3]
			Gtemp1_lvl1_part2_1_lvl2_part3_1_perm2013__S__D_0__D_1__D_3.AlignModesWith( modes_0_1_3, T_bfnj_lvl2_part3_1__D_0__D_1__D_2__D_3, modes_0_1_3 );
			Gtemp1_lvl1_part2_1_lvl2_part3_1_perm2013__S__D_0__D_1__D_3.AllGatherRedistFrom( Gtemp1_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3, modes_2 );
			Permute( T_bfnj_lvl2_part3_1__D_0__D_1__D_2__D_3, T_bfnj_lvl2_part3_1_perm0132__D_0__D_1__D_3__D_2 );
			G_mi_lvl1_part0_1__S__D_2__D_0__D_1__D_3.AlignModesWith( modes_1, T_bfnj_lvl2_part3_1__D_0__D_1__D_2__D_3, modes_2 );
			tempShape = G_mi_lvl1_part0_1__D_0_1__D_2_3.Shape();
			tempShape.push_back( g.Shape()[0] );
			tempShape.push_back( g.Shape()[1] );
			tempShape.push_back( g.Shape()[3] );
			G_mi_lvl1_part0_1__S__D_2__D_0__D_1__D_3.ResizeTo( tempShape );
			   // 1.0 * Gtemp1_lvl1_part2_1_lvl2_part3_1[D0,D1,*,D3]_mefn * T_bfnj_lvl2_part3_1[D0,D1,D2,D3]_efni + 0.0 * G_mi_lvl1_part0_1[*,D2,D0,D1,D3]_miefn
			LocalContract(1.0, Gtemp1_lvl1_part2_1_lvl2_part3_1_perm2013__S__D_0__D_1__D_3.LockedTensor(), indices_mefn, false,
				T_bfnj_lvl2_part3_1_perm0132__D_0__D_1__D_3__D_2.LockedTensor(), indices_efni, false,
				0.0, G_mi_lvl1_part0_1__S__D_2__D_0__D_1__D_3.Tensor(), indices_miefn, false);
			Gtemp1_lvl1_part2_1_lvl2_part3_1_perm2013__S__D_0__D_1__D_3.EmptyData();
			T_bfnj_lvl2_part3_1_perm0132__D_0__D_1__D_3__D_2.EmptyData();
			   // G_mi_lvl1_part0_1[D01,D23] <- G_mi_lvl1_part0_1[*,D2,D0,D1,D3] (with SumScatter on (D0)(D1)(D3))
			G_mi_lvl1_part0_1__D_0_1__D_2_3.ReduceScatterUpdateRedistFrom( G_mi_lvl1_part0_1__S__D_2__D_0__D_1__D_3, 1.0, modes_4_3_2 );
			G_mi_lvl1_part0_1__S__D_2__D_0__D_1__D_3.EmptyData();

			SlidePartitionDown
			( Gtemp1_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3,  Gtemp1_lvl1_part2_1_lvl2_part3_0__D_0__D_1__D_2__D_3,
			       Gtemp1_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  Gtemp1_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, Gtemp1_lvl1_part2_1_lvl2_part3_2__D_0__D_1__D_2__D_3, 3 );
			SlidePartitionDown
			( T_bfnj_lvl2_part3T__D_0__D_1__D_2__D_3,  T_bfnj_lvl2_part3_0__D_0__D_1__D_2__D_3,
			       T_bfnj_lvl2_part3_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  T_bfnj_lvl2_part3B__D_0__D_1__D_2__D_3, T_bfnj_lvl2_part3_2__D_0__D_1__D_2__D_3, 3 );

		}
		//****

		SlidePartitionDown
		( Gtemp1_lvl1_part2T__D_0__D_1__D_2__D_3,  Gtemp1_lvl1_part2_0__D_0__D_1__D_2__D_3,
		       Gtemp1_lvl1_part2_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  Gtemp1_lvl1_part2B__D_0__D_1__D_2__D_3, Gtemp1_lvl1_part2_2__D_0__D_1__D_2__D_3, 2 );
		SlidePartitionDown
		( G_mi_lvl1_part0T__D_0_1__D_2_3,  G_mi_lvl1_part0_0__D_0_1__D_2_3,
		       G_mi_lvl1_part0_1__D_0_1__D_2_3,
		  /**/ /**/
		  G_mi_lvl1_part0B__D_0_1__D_2_3, G_mi_lvl1_part0_2__D_0_1__D_2_3, 0 );

	}
	//****
	Gtemp1__D_0__D_1__D_2__D_3.EmptyData();
	tempShape = u_mnje__D_0__D_1__D_2__D_3.Shape();
	Gtemp2__D_0__D_1__D_2__D_3.ResizeTo( tempShape );
	//**** (out of 1)
	//**** Is real	0 shadows
		//Outputs:
		//  Gtemp2__D_0__D_1__D_2__D_3
	PartitionDown(u_mnje__D_0__D_1__D_2__D_3, u_mnje_lvl1_part0T__D_0__D_1__D_2__D_3, u_mnje_lvl1_part0B__D_0__D_1__D_2__D_3, 0, 0);
	PartitionDown(u_mnje__D_0__D_1__D_2__D_3, u_mnje_lvl1_part1T__D_0__D_1__D_2__D_3, u_mnje_lvl1_part1B__D_0__D_1__D_2__D_3, 1, 0);
	PartitionDown(Gtemp2__D_0__D_1__D_2__D_3, Gtemp2_lvl1_part0T__D_0__D_1__D_2__D_3, Gtemp2_lvl1_part0B__D_0__D_1__D_2__D_3, 0, 0);
	while(Gtemp2_lvl1_part0T__D_0__D_1__D_2__D_3.Dimension(0) < Gtemp2__D_0__D_1__D_2__D_3.Dimension(0))
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
		( Gtemp2_lvl1_part0T__D_0__D_1__D_2__D_3,  Gtemp2_lvl1_part0_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       Gtemp2_lvl1_part0_1__D_0__D_1__D_2__D_3,
		  Gtemp2_lvl1_part0B__D_0__D_1__D_2__D_3, Gtemp2_lvl1_part0_2__D_0__D_1__D_2__D_3, 0, blkSize );

		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  Gtemp2_lvl1_part0_1__D_0__D_1__D_2__D_3
		PartitionDown(u_mnje_lvl1_part0_1__D_0__D_1__D_2__D_3, u_mnje_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3, u_mnje_lvl1_part0_1_lvl2_part1B__D_0__D_1__D_2__D_3, 1, 0);
		PartitionDown(u_mnje_lvl1_part1_1__D_0__D_1__D_2__D_3, u_mnje_lvl1_part1_1_lvl2_part0T__D_0__D_1__D_2__D_3, u_mnje_lvl1_part1_1_lvl2_part0B__D_0__D_1__D_2__D_3, 0, 0);
		PartitionDown(Gtemp2_lvl1_part0_1__D_0__D_1__D_2__D_3, Gtemp2_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3, Gtemp2_lvl1_part0_1_lvl2_part1B__D_0__D_1__D_2__D_3, 1, 0);
		while(Gtemp2_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3.Dimension(1) < Gtemp2_lvl1_part0_1__D_0__D_1__D_2__D_3.Dimension(1))
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
			( Gtemp2_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3,  Gtemp2_lvl1_part0_1_lvl2_part1_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       Gtemp2_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3,
			  Gtemp2_lvl1_part0_1_lvl2_part1B__D_0__D_1__D_2__D_3, Gtemp2_lvl1_part0_1_lvl2_part1_2__D_0__D_1__D_2__D_3, 1, blkSize );

			   // u_mnje_lvl1_part1_1_lvl2_part0_1[D1,D0,D2,D3] <- u_mnje_lvl1_part1_1_lvl2_part0_1[D0,D1,D2,D3]
			u_mnje_lvl1_part1_1_lvl2_part0_1__D_1__D_0__D_2__D_3.AlignModesWith( modes_0_1_2_3, u_mnje_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3, modes_1_0_2_3 );
			u_mnje_lvl1_part1_1_lvl2_part0_1__D_1__D_0__D_2__D_3.AllToAllRedistFrom( u_mnje_lvl1_part1_1_lvl2_part0_1__D_0__D_1__D_2__D_3, modes_0_1 );
			YAxpPx( 2.0, u_mnje_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3, -1.0, u_mnje_lvl1_part1_1_lvl2_part0_1__D_1__D_0__D_2__D_3, perm_1_0_2_3, Gtemp2_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3 );
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
			( Gtemp2_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3,  Gtemp2_lvl1_part0_1_lvl2_part1_0__D_0__D_1__D_2__D_3,
			       Gtemp2_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  Gtemp2_lvl1_part0_1_lvl2_part1B__D_0__D_1__D_2__D_3, Gtemp2_lvl1_part0_1_lvl2_part1_2__D_0__D_1__D_2__D_3, 1 );

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
		( Gtemp2_lvl1_part0T__D_0__D_1__D_2__D_3,  Gtemp2_lvl1_part0_0__D_0__D_1__D_2__D_3,
		       Gtemp2_lvl1_part0_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  Gtemp2_lvl1_part0B__D_0__D_1__D_2__D_3, Gtemp2_lvl1_part0_2__D_0__D_1__D_2__D_3, 0 );

	}
	//****
	//**** (out of 1)
	//**** Is real	0 shadows
		//Outputs:
		//  G_mi__D_0_1__D_2_3
	PartitionDown(Gtemp2__D_0__D_1__D_2__D_3, Gtemp2_lvl1_part2T__D_0__D_1__D_2__D_3, Gtemp2_lvl1_part2B__D_0__D_1__D_2__D_3, 2, 0);
	PartitionDown(G_mi__D_0_1__D_2_3, G_mi_lvl1_part1T__D_0_1__D_2_3, G_mi_lvl1_part1B__D_0_1__D_2_3, 1, 0);
	while(G_mi_lvl1_part1T__D_0_1__D_2_3.Dimension(1) < G_mi__D_0_1__D_2_3.Dimension(1))
	{
		RepartitionDown
		( Gtemp2_lvl1_part2T__D_0__D_1__D_2__D_3,  Gtemp2_lvl1_part2_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       Gtemp2_lvl1_part2_1__D_0__D_1__D_2__D_3,
		  Gtemp2_lvl1_part2B__D_0__D_1__D_2__D_3, Gtemp2_lvl1_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );
		RepartitionDown
		( G_mi_lvl1_part1T__D_0_1__D_2_3,  G_mi_lvl1_part1_0__D_0_1__D_2_3,
		  /**/ /**/
		       G_mi_lvl1_part1_1__D_0_1__D_2_3,
		  G_mi_lvl1_part1B__D_0_1__D_2_3, G_mi_lvl1_part1_2__D_0_1__D_2_3, 1, blkSize );

		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  G_mi_lvl1_part1_1__D_0_1__D_2_3
		PartitionDown(Gtemp2_lvl1_part2_1__D_0__D_1__D_2__D_3, Gtemp2_lvl1_part2_1_lvl2_part1T__D_0__D_1__D_2__D_3, Gtemp2_lvl1_part2_1_lvl2_part1B__D_0__D_1__D_2__D_3, 1, 0);
		PartitionDown(t_fj__D_0_1__D_2_3, t_fj_lvl2_part1T__D_0_1__D_2_3, t_fj_lvl2_part1B__D_0_1__D_2_3, 1, 0);
		while(Gtemp2_lvl1_part2_1_lvl2_part1T__D_0__D_1__D_2__D_3.Dimension(1) < Gtemp2_lvl1_part2_1__D_0__D_1__D_2__D_3.Dimension(1))
		{
			RepartitionDown
			( Gtemp2_lvl1_part2_1_lvl2_part1T__D_0__D_1__D_2__D_3,  Gtemp2_lvl1_part2_1_lvl2_part1_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       Gtemp2_lvl1_part2_1_lvl2_part1_1__D_0__D_1__D_2__D_3,
			  Gtemp2_lvl1_part2_1_lvl2_part1B__D_0__D_1__D_2__D_3, Gtemp2_lvl1_part2_1_lvl2_part1_2__D_0__D_1__D_2__D_3, 1, blkSize );
			RepartitionDown
			( t_fj_lvl2_part1T__D_0_1__D_2_3,  t_fj_lvl2_part1_0__D_0_1__D_2_3,
			  /**/ /**/
			       t_fj_lvl2_part1_1__D_0_1__D_2_3,
			  t_fj_lvl2_part1B__D_0_1__D_2_3, t_fj_lvl2_part1_2__D_0_1__D_2_3, 1, blkSize );

			   // t_fj_lvl2_part1_1[D03,D21] <- t_fj_lvl2_part1_1[D01,D23]
			t_fj_lvl2_part1_1__D_0_3__D_2_1.AlignModesWith( modes_0_1, Gtemp2_lvl1_part2_1_lvl2_part1_1__D_0__D_1__D_2__D_3, modes_3_1 );
			t_fj_lvl2_part1_1__D_0_3__D_2_1.AllToAllRedistFrom( t_fj_lvl2_part1_1__D_0_1__D_2_3, modes_1_3 );
			   // t_fj_lvl2_part1_1[D03,D12] <- t_fj_lvl2_part1_1[D03,D21]
			t_fj_lvl2_part1_1__D_0_3__D_1_2.AlignModesWith( modes_0_1, Gtemp2_lvl1_part2_1_lvl2_part1_1__D_0__D_1__D_2__D_3, modes_3_1 );
			t_fj_lvl2_part1_1__D_0_3__D_1_2.PermutationRedistFrom( t_fj_lvl2_part1_1__D_0_3__D_2_1, modes_2_1 );
			t_fj_lvl2_part1_1__D_0_3__D_2_1.EmptyData();
			   // t_fj_lvl2_part1_1[D30,D12] <- t_fj_lvl2_part1_1[D03,D12]
			t_fj_lvl2_part1_1__D_3_0__D_1_2.AlignModesWith( modes_0_1, Gtemp2_lvl1_part2_1_lvl2_part1_1__D_0__D_1__D_2__D_3, modes_3_1 );
			t_fj_lvl2_part1_1__D_3_0__D_1_2.PermutationRedistFrom( t_fj_lvl2_part1_1__D_0_3__D_1_2, modes_0_3 );
			t_fj_lvl2_part1_1__D_0_3__D_1_2.EmptyData();
			   // t_fj_lvl2_part1_1[D3,D1] <- t_fj_lvl2_part1_1[D30,D12]
			t_fj_lvl2_part1_1__D_3__D_1.AlignModesWith( modes_0_1, Gtemp2_lvl1_part2_1_lvl2_part1_1__D_0__D_1__D_2__D_3, modes_3_1 );
			t_fj_lvl2_part1_1__D_3__D_1.AllGatherRedistFrom( t_fj_lvl2_part1_1__D_3_0__D_1_2, modes_0_2 );
			t_fj_lvl2_part1_1__D_3_0__D_1_2.EmptyData();
			Permute( Gtemp2_lvl1_part2_1_lvl2_part1_1__D_0__D_1__D_2__D_3, Gtemp2_lvl1_part2_1_lvl2_part1_1_perm0231__D_0__D_2__D_3__D_1 );
			G_mi_lvl1_part1_1__D_0__D_2__D_3__D_1.AlignModesWith( modes_0_1, Gtemp2_lvl1_part2_1_lvl2_part1_1__D_0__D_1__D_2__D_3, modes_0_2 );
			tempShape = G_mi_lvl1_part1_1__D_0_1__D_2_3.Shape();
			tempShape.push_back( g.Shape()[3] );
			tempShape.push_back( g.Shape()[1] );
			G_mi_lvl1_part1_1__D_0__D_2__D_3__D_1.ResizeTo( tempShape );
			   // 1.0 * Gtemp2_lvl1_part2_1_lvl2_part1_1[D0,D1,D2,D3]_mien * t_fj_lvl2_part1_1[D3,D1]_en + 0.0 * G_mi_lvl1_part1_1[D0,D2,D3,D1]_mien
			LocalContract(1.0, Gtemp2_lvl1_part2_1_lvl2_part1_1_perm0231__D_0__D_2__D_3__D_1.LockedTensor(), indices_mien, false,
				t_fj_lvl2_part1_1__D_3__D_1.LockedTensor(), indices_en, false,
				0.0, G_mi_lvl1_part1_1__D_0__D_2__D_3__D_1.Tensor(), indices_mien, false);
			Gtemp2_lvl1_part2_1_lvl2_part1_1_perm0231__D_0__D_2__D_3__D_1.EmptyData();
			t_fj_lvl2_part1_1__D_3__D_1.EmptyData();
			   // G_mi_lvl1_part1_1[D01,D23] <- G_mi_lvl1_part1_1[D0,D2,D3,D1] (with SumScatter on (D3)(D1))
			G_mi_lvl1_part1_1__D_0_1__D_2_3.ReduceScatterUpdateRedistFrom( G_mi_lvl1_part1_1__D_0__D_2__D_3__D_1, 1.0, modes_3_2 );
			G_mi_lvl1_part1_1__D_0__D_2__D_3__D_1.EmptyData();

			SlidePartitionDown
			( Gtemp2_lvl1_part2_1_lvl2_part1T__D_0__D_1__D_2__D_3,  Gtemp2_lvl1_part2_1_lvl2_part1_0__D_0__D_1__D_2__D_3,
			       Gtemp2_lvl1_part2_1_lvl2_part1_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  Gtemp2_lvl1_part2_1_lvl2_part1B__D_0__D_1__D_2__D_3, Gtemp2_lvl1_part2_1_lvl2_part1_2__D_0__D_1__D_2__D_3, 1 );
			SlidePartitionDown
			( t_fj_lvl2_part1T__D_0_1__D_2_3,  t_fj_lvl2_part1_0__D_0_1__D_2_3,
			       t_fj_lvl2_part1_1__D_0_1__D_2_3,
			  /**/ /**/
			  t_fj_lvl2_part1B__D_0_1__D_2_3, t_fj_lvl2_part1_2__D_0_1__D_2_3, 1 );

		}
		//****

		SlidePartitionDown
		( Gtemp2_lvl1_part2T__D_0__D_1__D_2__D_3,  Gtemp2_lvl1_part2_0__D_0__D_1__D_2__D_3,
		       Gtemp2_lvl1_part2_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  Gtemp2_lvl1_part2B__D_0__D_1__D_2__D_3, Gtemp2_lvl1_part2_2__D_0__D_1__D_2__D_3, 2 );
		SlidePartitionDown
		( G_mi_lvl1_part1T__D_0_1__D_2_3,  G_mi_lvl1_part1_0__D_0_1__D_2_3,
		       G_mi_lvl1_part1_1__D_0_1__D_2_3,
		  /**/ /**/
		  G_mi_lvl1_part1B__D_0_1__D_2_3, G_mi_lvl1_part1_2__D_0_1__D_2_3, 1 );

	}
	//****
	Gtemp2__D_0__D_1__D_2__D_3.EmptyData();
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
		PartitionDown(t_fj__D_0_1__D_2_3, t_fj_lvl2_part1T__D_0_1__D_2_3, t_fj_lvl2_part1B__D_0_1__D_2_3, 1, 0);
		PartitionDown(G_mi_lvl1_part0_1__D_0_1__D_2_3, G_mi_lvl1_part0_1_lvl2_part1T__D_0_1__D_2_3, G_mi_lvl1_part0_1_lvl2_part1B__D_0_1__D_2_3, 1, 0);
		while(G_mi_lvl1_part0_1_lvl2_part1T__D_0_1__D_2_3.Dimension(1) < G_mi_lvl1_part0_1__D_0_1__D_2_3.Dimension(1))
		{
			RepartitionDown
			( t_fj_lvl2_part1T__D_0_1__D_2_3,  t_fj_lvl2_part1_0__D_0_1__D_2_3,
			  /**/ /**/
			       t_fj_lvl2_part1_1__D_0_1__D_2_3,
			  t_fj_lvl2_part1B__D_0_1__D_2_3, t_fj_lvl2_part1_2__D_0_1__D_2_3, 1, blkSize );
			RepartitionDown
			( G_mi_lvl1_part0_1_lvl2_part1T__D_0_1__D_2_3,  G_mi_lvl1_part0_1_lvl2_part1_0__D_0_1__D_2_3,
			  /**/ /**/
			       G_mi_lvl1_part0_1_lvl2_part1_1__D_0_1__D_2_3,
			  G_mi_lvl1_part0_1_lvl2_part1B__D_0_1__D_2_3, G_mi_lvl1_part0_1_lvl2_part1_2__D_0_1__D_2_3, 1, blkSize );

			   // t_fj_lvl2_part1_1[D01,*] <- t_fj_lvl2_part1_1[D01,D23]
			t_fj_lvl2_part1_1__D_0_1__S.AlignModesWith( modes_0, H_me_lvl1_part0_1__D_0_1__D_2_3, modes_1 );
			t_fj_lvl2_part1_1__D_0_1__S.AllGatherRedistFrom( t_fj_lvl2_part1_1__D_0_1__D_2_3, modes_2_3 );
			   // t_fj_lvl2_part1_1[D0123,*] <- t_fj_lvl2_part1_1[D01,*]
			t_fj_lvl2_part1_1__D_0_1_2_3__S.AlignModesWith( modes_0, H_me_lvl1_part0_1__D_0_1__D_2_3, modes_1 );
			t_fj_lvl2_part1_1__D_0_1_2_3__S.LocalRedistFrom( t_fj_lvl2_part1_1__D_0_1__S );
			t_fj_lvl2_part1_1__D_0_1__S.EmptyData();
			   // t_fj_lvl2_part1_1[D2301,*] <- t_fj_lvl2_part1_1[D0123,*]
			t_fj_lvl2_part1_1__D_2_3_0_1__S.AlignModesWith( modes_0, H_me_lvl1_part0_1__D_0_1__D_2_3, modes_1 );
			t_fj_lvl2_part1_1__D_2_3_0_1__S.PermutationRedistFrom( t_fj_lvl2_part1_1__D_0_1_2_3__S, modes_0_1_2_3 );
			t_fj_lvl2_part1_1__D_0_1_2_3__S.EmptyData();
			   // t_fj_lvl2_part1_1[D23,*] <- t_fj_lvl2_part1_1[D2301,*]
			t_fj_lvl2_part1_1__D_2_3__S.AlignModesWith( modes_0, H_me_lvl1_part0_1__D_0_1__D_2_3, modes_1 );
			t_fj_lvl2_part1_1__D_2_3__S.AllGatherRedistFrom( t_fj_lvl2_part1_1__D_2_3_0_1__S, modes_0_1 );
			t_fj_lvl2_part1_1__D_2_3_0_1__S.EmptyData();
			G_mi_lvl1_part0_1_lvl2_part1_1__D_0_1__S__D_2_3.AlignModesWith( modes_0, H_me_lvl1_part0_1__D_0_1__D_2_3, modes_0 );
			tempShape = G_mi_lvl1_part0_1_lvl2_part1_1__D_0_1__D_2_3.Shape();
			tempShape.push_back( g.Shape()[2] * g.Shape()[3] );
			G_mi_lvl1_part0_1_lvl2_part1_1__D_0_1__S__D_2_3.ResizeTo( tempShape );
			   // 1.0 * H_me_lvl1_part0_1[D01,D23]_me * t_fj_lvl2_part1_1[D23,*]_ei + 0.0 * G_mi_lvl1_part0_1_lvl2_part1_1[D01,*,D23]_mie
			LocalContract(1.0, H_me_lvl1_part0_1__D_0_1__D_2_3.LockedTensor(), indices_me, false,
				t_fj_lvl2_part1_1__D_2_3__S.LockedTensor(), indices_ei, false,
				0.0, G_mi_lvl1_part0_1_lvl2_part1_1__D_0_1__S__D_2_3.Tensor(), indices_mie, false);
			t_fj_lvl2_part1_1__D_2_3__S.EmptyData();
			   // G_mi_lvl1_part0_1_lvl2_part1_1[D01,D23] <- G_mi_lvl1_part0_1_lvl2_part1_1[D01,*,D23] (with SumScatter on D23)
			G_mi_lvl1_part0_1_lvl2_part1_1__D_0_1__D_2_3.ReduceScatterUpdateRedistFrom( G_mi_lvl1_part0_1_lvl2_part1_1__D_0_1__S__D_2_3, 1.0, 2 );
			G_mi_lvl1_part0_1_lvl2_part1_1__D_0_1__S__D_2_3.EmptyData();

			SlidePartitionDown
			( t_fj_lvl2_part1T__D_0_1__D_2_3,  t_fj_lvl2_part1_0__D_0_1__D_2_3,
			       t_fj_lvl2_part1_1__D_0_1__D_2_3,
			  /**/ /**/
			  t_fj_lvl2_part1B__D_0_1__D_2_3, t_fj_lvl2_part1_2__D_0_1__D_2_3, 1 );
			SlidePartitionDown
			( G_mi_lvl1_part0_1_lvl2_part1T__D_0_1__D_2_3,  G_mi_lvl1_part0_1_lvl2_part1_0__D_0_1__D_2_3,
			       G_mi_lvl1_part0_1_lvl2_part1_1__D_0_1__D_2_3,
			  /**/ /**/
			  G_mi_lvl1_part0_1_lvl2_part1B__D_0_1__D_2_3, G_mi_lvl1_part0_1_lvl2_part1_2__D_0_1__D_2_3, 1 );

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


//**** (out of 1)
//**** Is real	0 shadows
	//Outputs:
	//  z_ai__D_0_1__D_2_3

	Scal( 0.0, z_ai__D_0_1__D_2_3 );
	tempShape = w_bmje__D_0__D_1__D_2__D_3.Shape();
	ztemp4__D_0__D_1__D_2__D_3.ResizeTo( tempShape );
	//**** (out of 1)
	//**** Is real	0 shadows
		//Outputs:
		//  ztemp4__D_0__D_1__D_2__D_3
	PartitionDown(w_bmje__D_0__D_1__D_2__D_3, w_bmje_lvl1_part1T__D_0__D_1__D_2__D_3, w_bmje_lvl1_part1B__D_0__D_1__D_2__D_3, 1, 0);
	PartitionDown(x_bmej__D_0__D_1__D_2__D_3, x_bmej_lvl1_part1T__D_0__D_1__D_2__D_3, x_bmej_lvl1_part1B__D_0__D_1__D_2__D_3, 1, 0);
	PartitionDown(ztemp4__D_0__D_1__D_2__D_3, ztemp4_lvl1_part1T__D_0__D_1__D_2__D_3, ztemp4_lvl1_part1B__D_0__D_1__D_2__D_3, 1, 0);
	while(ztemp4_lvl1_part1T__D_0__D_1__D_2__D_3.Dimension(1) < ztemp4__D_0__D_1__D_2__D_3.Dimension(1))
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
		( ztemp4_lvl1_part1T__D_0__D_1__D_2__D_3,  ztemp4_lvl1_part1_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       ztemp4_lvl1_part1_1__D_0__D_1__D_2__D_3,
		  ztemp4_lvl1_part1B__D_0__D_1__D_2__D_3, ztemp4_lvl1_part1_2__D_0__D_1__D_2__D_3, 1, blkSize );

		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  ztemp4_lvl1_part1_1__D_0__D_1__D_2__D_3
		PartitionDown(w_bmje_lvl1_part1_1__D_0__D_1__D_2__D_3, w_bmje_lvl1_part1_1_lvl2_part2T__D_0__D_1__D_2__D_3, w_bmje_lvl1_part1_1_lvl2_part2B__D_0__D_1__D_2__D_3, 2, 0);
		PartitionDown(x_bmej_lvl1_part1_1__D_0__D_1__D_2__D_3, x_bmej_lvl1_part1_1_lvl2_part3T__D_0__D_1__D_2__D_3, x_bmej_lvl1_part1_1_lvl2_part3B__D_0__D_1__D_2__D_3, 3, 0);
		PartitionDown(ztemp4_lvl1_part1_1__D_0__D_1__D_2__D_3, ztemp4_lvl1_part1_1_lvl2_part2T__D_0__D_1__D_2__D_3, ztemp4_lvl1_part1_1_lvl2_part2B__D_0__D_1__D_2__D_3, 2, 0);
		while(ztemp4_lvl1_part1_1_lvl2_part2T__D_0__D_1__D_2__D_3.Dimension(2) < ztemp4_lvl1_part1_1__D_0__D_1__D_2__D_3.Dimension(2))
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
			( ztemp4_lvl1_part1_1_lvl2_part2T__D_0__D_1__D_2__D_3,  ztemp4_lvl1_part1_1_lvl2_part2_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       ztemp4_lvl1_part1_1_lvl2_part2_1__D_0__D_1__D_2__D_3,
			  ztemp4_lvl1_part1_1_lvl2_part2B__D_0__D_1__D_2__D_3, ztemp4_lvl1_part1_1_lvl2_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );

			   // x_bmej_lvl1_part1_1_lvl2_part3_1[D0,D1,D3,D2] <- x_bmej_lvl1_part1_1_lvl2_part3_1[D0,D1,D2,D3]
			x_bmej_lvl1_part1_1_lvl2_part3_1__D_0__D_1__D_3__D_2.AlignModesWith( modes_0_1_2_3, w_bmje_lvl1_part1_1_lvl2_part2_1__D_0__D_1__D_2__D_3, modes_0_1_3_2 );
			x_bmej_lvl1_part1_1_lvl2_part3_1__D_0__D_1__D_3__D_2.AllToAllRedistFrom( x_bmej_lvl1_part1_1_lvl2_part3_1__D_0__D_1__D_2__D_3, modes_2_3 );
			YAxpPx( 2.0, w_bmje_lvl1_part1_1_lvl2_part2_1__D_0__D_1__D_2__D_3, -1.0, x_bmej_lvl1_part1_1_lvl2_part3_1__D_0__D_1__D_3__D_2, perm_0_1_3_2, ztemp4_lvl1_part1_1_lvl2_part2_1__D_0__D_1__D_2__D_3 );
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
			( ztemp4_lvl1_part1_1_lvl2_part2T__D_0__D_1__D_2__D_3,  ztemp4_lvl1_part1_1_lvl2_part2_0__D_0__D_1__D_2__D_3,
			       ztemp4_lvl1_part1_1_lvl2_part2_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  ztemp4_lvl1_part1_1_lvl2_part2B__D_0__D_1__D_2__D_3, ztemp4_lvl1_part1_1_lvl2_part2_2__D_0__D_1__D_2__D_3, 2 );

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
		( ztemp4_lvl1_part1T__D_0__D_1__D_2__D_3,  ztemp4_lvl1_part1_0__D_0__D_1__D_2__D_3,
		       ztemp4_lvl1_part1_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  ztemp4_lvl1_part1B__D_0__D_1__D_2__D_3, ztemp4_lvl1_part1_2__D_0__D_1__D_2__D_3, 1 );

	}
	//****
	tempShape = T_bfnj__D_0__D_1__D_2__D_3.Shape();
	ztemp3__D_0__D_1__D_2__D_3.ResizeTo( tempShape );
	//**** (out of 1)
	//**** Is real	0 shadows
		//Outputs:
		//  ztemp3__D_0__D_1__D_2__D_3
	PartitionDown(T_bfnj__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part2T__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part2B__D_0__D_1__D_2__D_3, 2, 0);
	PartitionDown(T_bfnj__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part3T__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part3B__D_0__D_1__D_2__D_3, 3, 0);
	PartitionDown(ztemp3__D_0__D_1__D_2__D_3, ztemp3_lvl1_part2T__D_0__D_1__D_2__D_3, ztemp3_lvl1_part2B__D_0__D_1__D_2__D_3, 2, 0);
	while(ztemp3_lvl1_part2T__D_0__D_1__D_2__D_3.Dimension(2) < ztemp3__D_0__D_1__D_2__D_3.Dimension(2))
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
		( ztemp3_lvl1_part2T__D_0__D_1__D_2__D_3,  ztemp3_lvl1_part2_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       ztemp3_lvl1_part2_1__D_0__D_1__D_2__D_3,
		  ztemp3_lvl1_part2B__D_0__D_1__D_2__D_3, ztemp3_lvl1_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );

		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  ztemp3_lvl1_part2_1__D_0__D_1__D_2__D_3
		PartitionDown(T_bfnj_lvl1_part2_1__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, 3, 0);
		PartitionDown(T_bfnj_lvl1_part3_1__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part3_1_lvl2_part2T__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part3_1_lvl2_part2B__D_0__D_1__D_2__D_3, 2, 0);
		PartitionDown(ztemp3_lvl1_part2_1__D_0__D_1__D_2__D_3, ztemp3_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3, ztemp3_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, 3, 0);
		while(ztemp3_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3.Dimension(3) < ztemp3_lvl1_part2_1__D_0__D_1__D_2__D_3.Dimension(3))
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
			( ztemp3_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3,  ztemp3_lvl1_part2_1_lvl2_part3_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       ztemp3_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3,
			  ztemp3_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, ztemp3_lvl1_part2_1_lvl2_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );

			   // T_bfnj_lvl1_part3_1_lvl2_part2_1[D0,D1,D3,D2] <- T_bfnj_lvl1_part3_1_lvl2_part2_1[D0,D1,D2,D3]
			T_bfnj_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_3__D_2.AlignModesWith( modes_0_1_2_3, T_bfnj_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3, modes_0_1_3_2 );
			T_bfnj_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_3__D_2.AllToAllRedistFrom( T_bfnj_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_2__D_3, modes_2_3 );
			YAxpPx( 2.0, T_bfnj_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3, -1.0, T_bfnj_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_3__D_2, perm_0_1_3_2, ztemp3_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3 );
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
			( ztemp3_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3,  ztemp3_lvl1_part2_1_lvl2_part3_0__D_0__D_1__D_2__D_3,
			       ztemp3_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  ztemp3_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, ztemp3_lvl1_part2_1_lvl2_part3_2__D_0__D_1__D_2__D_3, 3 );

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
		( ztemp3_lvl1_part2T__D_0__D_1__D_2__D_3,  ztemp3_lvl1_part2_0__D_0__D_1__D_2__D_3,
		       ztemp3_lvl1_part2_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  ztemp3_lvl1_part2B__D_0__D_1__D_2__D_3, ztemp3_lvl1_part2_2__D_0__D_1__D_2__D_3, 2 );

	}
	//****
	tempShape = r_bmfe__D_0__D_1__D_2__D_3.Shape();
	ztemp2__D_0__D_1__D_2__D_3.ResizeTo( tempShape );
	//**** (out of 1)
	//**** Is real	0 shadows
		//Outputs:
		//  ztemp2__D_0__D_1__D_2__D_3
	PartitionDown(r_bmfe__D_0__D_1__D_2__D_3, r_bmfe_lvl1_part0T__D_0__D_1__D_2__D_3, r_bmfe_lvl1_part0B__D_0__D_1__D_2__D_3, 0, 0);
	PartitionDown(ztemp2__D_0__D_1__D_2__D_3, ztemp2_lvl1_part0T__D_0__D_1__D_2__D_3, ztemp2_lvl1_part0B__D_0__D_1__D_2__D_3, 0, 0);
	while(ztemp2_lvl1_part0T__D_0__D_1__D_2__D_3.Dimension(0) < ztemp2__D_0__D_1__D_2__D_3.Dimension(0))
	{
		RepartitionDown
		( r_bmfe_lvl1_part0T__D_0__D_1__D_2__D_3,  r_bmfe_lvl1_part0_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       r_bmfe_lvl1_part0_1__D_0__D_1__D_2__D_3,
		  r_bmfe_lvl1_part0B__D_0__D_1__D_2__D_3, r_bmfe_lvl1_part0_2__D_0__D_1__D_2__D_3, 0, blkSize );
		RepartitionDown
		( ztemp2_lvl1_part0T__D_0__D_1__D_2__D_3,  ztemp2_lvl1_part0_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       ztemp2_lvl1_part0_1__D_0__D_1__D_2__D_3,
		  ztemp2_lvl1_part0B__D_0__D_1__D_2__D_3, ztemp2_lvl1_part0_2__D_0__D_1__D_2__D_3, 0, blkSize );

		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  ztemp2_lvl1_part0_1__D_0__D_1__D_2__D_3
		PartitionDown(r_bmfe_lvl1_part0_1__D_0__D_1__D_2__D_3, r_bmfe_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3, r_bmfe_lvl1_part0_1_lvl2_part1B__D_0__D_1__D_2__D_3, 1, 0);
		PartitionDown(ztemp2_lvl1_part0_1__D_0__D_1__D_2__D_3, ztemp2_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3, ztemp2_lvl1_part0_1_lvl2_part1B__D_0__D_1__D_2__D_3, 1, 0);
		while(ztemp2_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3.Dimension(1) < ztemp2_lvl1_part0_1__D_0__D_1__D_2__D_3.Dimension(1))
		{
			RepartitionDown
			( r_bmfe_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3,  r_bmfe_lvl1_part0_1_lvl2_part1_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       r_bmfe_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3,
			  r_bmfe_lvl1_part0_1_lvl2_part1B__D_0__D_1__D_2__D_3, r_bmfe_lvl1_part0_1_lvl2_part1_2__D_0__D_1__D_2__D_3, 1, blkSize );
			RepartitionDown
			( ztemp2_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3,  ztemp2_lvl1_part0_1_lvl2_part1_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       ztemp2_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3,
			  ztemp2_lvl1_part0_1_lvl2_part1B__D_0__D_1__D_2__D_3, ztemp2_lvl1_part0_1_lvl2_part1_2__D_0__D_1__D_2__D_3, 1, blkSize );

			   // r_bmfe_lvl1_part0_1_lvl2_part1_1[D0,D1,D3,D2] <- r_bmfe_lvl1_part0_1_lvl2_part1_1[D0,D1,D2,D3]
			r_bmfe_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_3__D_2.AlignModesWith( modes_0_1_2_3, r_bmfe_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3, modes_0_1_3_2 );
			r_bmfe_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_3__D_2.AllToAllRedistFrom( r_bmfe_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3, modes_2_3 );
			YAxpPx( 2.0, r_bmfe_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3, -1.0, r_bmfe_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_3__D_2, perm_0_1_3_2, ztemp2_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3 );
			r_bmfe_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_3__D_2.EmptyData();

			SlidePartitionDown
			( r_bmfe_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3,  r_bmfe_lvl1_part0_1_lvl2_part1_0__D_0__D_1__D_2__D_3,
			       r_bmfe_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  r_bmfe_lvl1_part0_1_lvl2_part1B__D_0__D_1__D_2__D_3, r_bmfe_lvl1_part0_1_lvl2_part1_2__D_0__D_1__D_2__D_3, 1 );
			SlidePartitionDown
			( ztemp2_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3,  ztemp2_lvl1_part0_1_lvl2_part1_0__D_0__D_1__D_2__D_3,
			       ztemp2_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  ztemp2_lvl1_part0_1_lvl2_part1B__D_0__D_1__D_2__D_3, ztemp2_lvl1_part0_1_lvl2_part1_2__D_0__D_1__D_2__D_3, 1 );

		}
		//****

		SlidePartitionDown
		( r_bmfe_lvl1_part0T__D_0__D_1__D_2__D_3,  r_bmfe_lvl1_part0_0__D_0__D_1__D_2__D_3,
		       r_bmfe_lvl1_part0_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  r_bmfe_lvl1_part0B__D_0__D_1__D_2__D_3, r_bmfe_lvl1_part0_2__D_0__D_1__D_2__D_3, 0 );
		SlidePartitionDown
		( ztemp2_lvl1_part0T__D_0__D_1__D_2__D_3,  ztemp2_lvl1_part0_0__D_0__D_1__D_2__D_3,
		       ztemp2_lvl1_part0_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  ztemp2_lvl1_part0B__D_0__D_1__D_2__D_3, ztemp2_lvl1_part0_2__D_0__D_1__D_2__D_3, 0 );

	}
	//****
	//**** (out of 1)
	//**** Is real	0 shadows
		//Outputs:
		//  z_ai__D_0_1__D_2_3
	PartitionDown(ztemp2__D_0__D_1__D_2__D_3, ztemp2_lvl1_part1T__D_0__D_1__D_2__D_3, ztemp2_lvl1_part1B__D_0__D_1__D_2__D_3, 1, 0);
	PartitionDown(Tau_efmn__D_0__D_1__D_2__D_3, Tau_efmn_lvl1_part3T__D_0__D_1__D_2__D_3, Tau_efmn_lvl1_part3B__D_0__D_1__D_2__D_3, 3, 0);
	while(ztemp2_lvl1_part1T__D_0__D_1__D_2__D_3.Dimension(1) < ztemp2__D_0__D_1__D_2__D_3.Dimension(1))
	{
		RepartitionDown
		( ztemp2_lvl1_part1T__D_0__D_1__D_2__D_3,  ztemp2_lvl1_part1_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       ztemp2_lvl1_part1_1__D_0__D_1__D_2__D_3,
		  ztemp2_lvl1_part1B__D_0__D_1__D_2__D_3, ztemp2_lvl1_part1_2__D_0__D_1__D_2__D_3, 1, blkSize );
		RepartitionDown
		( Tau_efmn_lvl1_part3T__D_0__D_1__D_2__D_3,  Tau_efmn_lvl1_part3_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       Tau_efmn_lvl1_part3_1__D_0__D_1__D_2__D_3,
		  Tau_efmn_lvl1_part3B__D_0__D_1__D_2__D_3, Tau_efmn_lvl1_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );

		Permute( ztemp2_lvl1_part1_1__D_0__D_1__D_2__D_3, ztemp2_lvl1_part1_1_perm0231__D_0__D_2__D_3__D_1 );
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

			   // Tau_efmn_lvl1_part3_1_lvl2_part2_1[D2,D1,D0,D3] <- Tau_efmn_lvl1_part3_1_lvl2_part2_1[D0,D1,D2,D3]
			Tau_efmn_lvl1_part3_1_lvl2_part2_1__D_2__D_1__D_0__D_3.AlignModesWith( modes_0_1_3, ztemp2_lvl1_part1_1__D_0__D_1__D_2__D_3, modes_2_3_1 );
			Tau_efmn_lvl1_part3_1_lvl2_part2_1__D_2__D_1__D_0__D_3.AllToAllRedistFrom( Tau_efmn_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_2__D_3, modes_0_2 );
			   // Tau_efmn_lvl1_part3_1_lvl2_part2_1[D2,D3,D0,D1] <- Tau_efmn_lvl1_part3_1_lvl2_part2_1[D2,D1,D0,D3]
			Tau_efmn_lvl1_part3_1_lvl2_part2_1__D_2__D_3__D_0__D_1.AlignModesWith( modes_0_1_3, ztemp2_lvl1_part1_1__D_0__D_1__D_2__D_3, modes_2_3_1 );
			Tau_efmn_lvl1_part3_1_lvl2_part2_1__D_2__D_3__D_0__D_1.AllToAllRedistFrom( Tau_efmn_lvl1_part3_1_lvl2_part2_1__D_2__D_1__D_0__D_3, modes_1_3 );
			Tau_efmn_lvl1_part3_1_lvl2_part2_1__D_2__D_1__D_0__D_3.EmptyData();
			   // Tau_efmn_lvl1_part3_1_lvl2_part2_1[D2,D3,*,D1] <- Tau_efmn_lvl1_part3_1_lvl2_part2_1[D2,D3,D0,D1]
			Tau_efmn_lvl1_part3_1_lvl2_part2_1_perm0132__D_2__D_3__D_1__S.AlignModesWith( modes_0_1_3, ztemp2_lvl1_part1_1__D_0__D_1__D_2__D_3, modes_2_3_1 );
			Tau_efmn_lvl1_part3_1_lvl2_part2_1_perm0132__D_2__D_3__D_1__S.AllGatherRedistFrom( Tau_efmn_lvl1_part3_1_lvl2_part2_1__D_2__D_3__D_0__D_1, modes_0 );
			Tau_efmn_lvl1_part3_1_lvl2_part2_1__D_2__D_3__D_0__D_1.EmptyData();
			z_ai_lvl2_part1_1__D_0__S__D_2__D_3__D_1.AlignModesWith( modes_0, ztemp2_lvl1_part1_1__D_0__D_1__D_2__D_3, modes_0 );
			tempShape = z_ai_lvl2_part1_1__D_0_1__D_2_3.Shape();
			tempShape.push_back( g.Shape()[2] );
			tempShape.push_back( g.Shape()[3] );
			tempShape.push_back( g.Shape()[1] );
			z_ai_lvl2_part1_1__D_0__S__D_2__D_3__D_1.ResizeTo( tempShape );
			   // 1.0 * ztemp2_lvl1_part1_1[D0,D1,D2,D3]_aefm * Tau_efmn_lvl1_part3_1_lvl2_part2_1[D2,D3,*,D1]_efmi + 0.0 * z_ai_lvl2_part1_1[D0,*,D2,D3,D1]_aiefm
			LocalContract(1.0, ztemp2_lvl1_part1_1_perm0231__D_0__D_2__D_3__D_1.LockedTensor(), indices_aefm, false,
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
		ztemp2_lvl1_part1_1_perm0231__D_0__D_2__D_3__D_1.EmptyData();

		SlidePartitionDown
		( ztemp2_lvl1_part1T__D_0__D_1__D_2__D_3,  ztemp2_lvl1_part1_0__D_0__D_1__D_2__D_3,
		       ztemp2_lvl1_part1_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  ztemp2_lvl1_part1B__D_0__D_1__D_2__D_3, ztemp2_lvl1_part1_2__D_0__D_1__D_2__D_3, 1 );
		SlidePartitionDown
		( Tau_efmn_lvl1_part3T__D_0__D_1__D_2__D_3,  Tau_efmn_lvl1_part3_0__D_0__D_1__D_2__D_3,
		       Tau_efmn_lvl1_part3_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  Tau_efmn_lvl1_part3B__D_0__D_1__D_2__D_3, Tau_efmn_lvl1_part3_2__D_0__D_1__D_2__D_3, 3 );

	}
	//****
	ztemp2__D_0__D_1__D_2__D_3.EmptyData();
	//**** (out of 1)
	//**** Is real	0 shadows
		//Outputs:
		//  z_ai__D_0_1__D_2_3
	PartitionDown(ztemp3__D_0__D_1__D_2__D_3, ztemp3_lvl1_part2T__D_0__D_1__D_2__D_3, ztemp3_lvl1_part2B__D_0__D_1__D_2__D_3, 2, 0);
	PartitionDown(z_ai__D_0_1__D_2_3, z_ai_lvl1_part1T__D_0_1__D_2_3, z_ai_lvl1_part1B__D_0_1__D_2_3, 1, 0);
	while(z_ai_lvl1_part1T__D_0_1__D_2_3.Dimension(1) < z_ai__D_0_1__D_2_3.Dimension(1))
	{
		RepartitionDown
		( ztemp3_lvl1_part2T__D_0__D_1__D_2__D_3,  ztemp3_lvl1_part2_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       ztemp3_lvl1_part2_1__D_0__D_1__D_2__D_3,
		  ztemp3_lvl1_part2B__D_0__D_1__D_2__D_3, ztemp3_lvl1_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );
		RepartitionDown
		( z_ai_lvl1_part1T__D_0_1__D_2_3,  z_ai_lvl1_part1_0__D_0_1__D_2_3,
		  /**/ /**/
		       z_ai_lvl1_part1_1__D_0_1__D_2_3,
		  z_ai_lvl1_part1B__D_0_1__D_2_3, z_ai_lvl1_part1_2__D_0_1__D_2_3, 1, blkSize );

		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  z_ai_lvl1_part1_1__D_0_1__D_2_3
		PartitionDown(ztemp3_lvl1_part2_1__D_0__D_1__D_2__D_3, ztemp3_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3, ztemp3_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, 3, 0);
		PartitionDown(H_me__D_0_1__D_2_3, H_me_lvl2_part0T__D_0_1__D_2_3, H_me_lvl2_part0B__D_0_1__D_2_3, 0, 0);
		while(ztemp3_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3.Dimension(3) < ztemp3_lvl1_part2_1__D_0__D_1__D_2__D_3.Dimension(3))
		{
			RepartitionDown
			( ztemp3_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3,  ztemp3_lvl1_part2_1_lvl2_part3_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       ztemp3_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3,
			  ztemp3_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, ztemp3_lvl1_part2_1_lvl2_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );
			RepartitionDown
			( H_me_lvl2_part0T__D_0_1__D_2_3,  H_me_lvl2_part0_0__D_0_1__D_2_3,
			  /**/ /**/
			       H_me_lvl2_part0_1__D_0_1__D_2_3,
			  H_me_lvl2_part0B__D_0_1__D_2_3, H_me_lvl2_part0_2__D_0_1__D_2_3, 0, blkSize );

			   // H_me_lvl2_part0_1[D03,D21] <- H_me_lvl2_part0_1[D01,D23]
			H_me_lvl2_part0_1__D_0_3__D_2_1.AlignModesWith( modes_0_1, ztemp3_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3, modes_3_1 );
			H_me_lvl2_part0_1__D_0_3__D_2_1.AllToAllRedistFrom( H_me_lvl2_part0_1__D_0_1__D_2_3, modes_1_3 );
			   // H_me_lvl2_part0_1[D03,D12] <- H_me_lvl2_part0_1[D03,D21]
			H_me_lvl2_part0_1__D_0_3__D_1_2.AlignModesWith( modes_0_1, ztemp3_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3, modes_3_1 );
			H_me_lvl2_part0_1__D_0_3__D_1_2.PermutationRedistFrom( H_me_lvl2_part0_1__D_0_3__D_2_1, modes_2_1 );
			H_me_lvl2_part0_1__D_0_3__D_2_1.EmptyData();
			   // H_me_lvl2_part0_1[D30,D12] <- H_me_lvl2_part0_1[D03,D12]
			H_me_lvl2_part0_1__D_3_0__D_1_2.AlignModesWith( modes_0_1, ztemp3_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3, modes_3_1 );
			H_me_lvl2_part0_1__D_3_0__D_1_2.PermutationRedistFrom( H_me_lvl2_part0_1__D_0_3__D_1_2, modes_0_3 );
			H_me_lvl2_part0_1__D_0_3__D_1_2.EmptyData();
			   // H_me_lvl2_part0_1[D3,D1] <- H_me_lvl2_part0_1[D30,D12]
			H_me_lvl2_part0_1__D_3__D_1.AlignModesWith( modes_0_1, ztemp3_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3, modes_3_1 );
			H_me_lvl2_part0_1__D_3__D_1.AllGatherRedistFrom( H_me_lvl2_part0_1__D_3_0__D_1_2, modes_0_2 );
			H_me_lvl2_part0_1__D_3_0__D_1_2.EmptyData();
			Permute( ztemp3_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3, ztemp3_lvl1_part2_1_lvl2_part3_1_perm0231__D_0__D_2__D_3__D_1 );
			z_ai_lvl1_part1_1_perm0132__D_0__D_2__D_3__D_1.AlignModesWith( modes_0_1, ztemp3_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3, modes_0_2 );
			tempShape = z_ai_lvl1_part1_1__D_0_1__D_2_3.Shape();
			tempShape.push_back( g.Shape()[1] );
			tempShape.push_back( g.Shape()[3] );
			z_ai_lvl1_part1_1_perm0132__D_0__D_2__D_3__D_1.ResizeTo( tempShape );
			   // 1.0 * ztemp3_lvl1_part2_1_lvl2_part3_1[D0,D1,D2,D3]_aime * H_me_lvl2_part0_1[D3,D1]_me + 0.0 * z_ai_lvl1_part1_1[D0,D2,D1,D3]_aime
			LocalContract(1.0, ztemp3_lvl1_part2_1_lvl2_part3_1_perm0231__D_0__D_2__D_3__D_1.LockedTensor(), indices_aime, false,
				H_me_lvl2_part0_1__D_3__D_1.LockedTensor(), indices_me, false,
				0.0, z_ai_lvl1_part1_1_perm0132__D_0__D_2__D_3__D_1.Tensor(), indices_aime, false);
			ztemp3_lvl1_part2_1_lvl2_part3_1_perm0231__D_0__D_2__D_3__D_1.EmptyData();
			H_me_lvl2_part0_1__D_3__D_1.EmptyData();
			   // z_ai_lvl1_part1_1[D01,D23] <- z_ai_lvl1_part1_1[D0,D2,D1,D3] (with SumScatter on (D1)(D3))
			z_ai_lvl1_part1_1__D_0_1__D_2_3.ReduceScatterUpdateRedistFrom( z_ai_lvl1_part1_1_perm0132__D_0__D_2__D_3__D_1, 1.0, modes_3_2 );
			z_ai_lvl1_part1_1_perm0132__D_0__D_2__D_3__D_1.EmptyData();

			SlidePartitionDown
			( ztemp3_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3,  ztemp3_lvl1_part2_1_lvl2_part3_0__D_0__D_1__D_2__D_3,
			       ztemp3_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  ztemp3_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, ztemp3_lvl1_part2_1_lvl2_part3_2__D_0__D_1__D_2__D_3, 3 );
			SlidePartitionDown
			( H_me_lvl2_part0T__D_0_1__D_2_3,  H_me_lvl2_part0_0__D_0_1__D_2_3,
			       H_me_lvl2_part0_1__D_0_1__D_2_3,
			  /**/ /**/
			  H_me_lvl2_part0B__D_0_1__D_2_3, H_me_lvl2_part0_2__D_0_1__D_2_3, 0 );

		}
		//****

		SlidePartitionDown
		( ztemp3_lvl1_part2T__D_0__D_1__D_2__D_3,  ztemp3_lvl1_part2_0__D_0__D_1__D_2__D_3,
		       ztemp3_lvl1_part2_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  ztemp3_lvl1_part2B__D_0__D_1__D_2__D_3, ztemp3_lvl1_part2_2__D_0__D_1__D_2__D_3, 2 );
		SlidePartitionDown
		( z_ai_lvl1_part1T__D_0_1__D_2_3,  z_ai_lvl1_part1_0__D_0_1__D_2_3,
		       z_ai_lvl1_part1_1__D_0_1__D_2_3,
		  /**/ /**/
		  z_ai_lvl1_part1B__D_0_1__D_2_3, z_ai_lvl1_part1_2__D_0_1__D_2_3, 1 );

	}
	//****
	ztemp3__D_0__D_1__D_2__D_3.EmptyData();
	//**** (out of 1)
	//**** Is real	0 shadows
		//Outputs:
		//  z_ai__D_0_1__D_2_3
	PartitionDown(ztemp4__D_0__D_1__D_2__D_3, ztemp4_lvl1_part2T__D_0__D_1__D_2__D_3, ztemp4_lvl1_part2B__D_0__D_1__D_2__D_3, 2, 0);
	PartitionDown(z_ai__D_0_1__D_2_3, z_ai_lvl1_part1T__D_0_1__D_2_3, z_ai_lvl1_part1B__D_0_1__D_2_3, 1, 0);
	while(z_ai_lvl1_part1T__D_0_1__D_2_3.Dimension(1) < z_ai__D_0_1__D_2_3.Dimension(1))
	{
		RepartitionDown
		( ztemp4_lvl1_part2T__D_0__D_1__D_2__D_3,  ztemp4_lvl1_part2_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       ztemp4_lvl1_part2_1__D_0__D_1__D_2__D_3,
		  ztemp4_lvl1_part2B__D_0__D_1__D_2__D_3, ztemp4_lvl1_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );
		RepartitionDown
		( z_ai_lvl1_part1T__D_0_1__D_2_3,  z_ai_lvl1_part1_0__D_0_1__D_2_3,
		  /**/ /**/
		       z_ai_lvl1_part1_1__D_0_1__D_2_3,
		  z_ai_lvl1_part1B__D_0_1__D_2_3, z_ai_lvl1_part1_2__D_0_1__D_2_3, 1, blkSize );

		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  z_ai_lvl1_part1_1__D_0_1__D_2_3
		PartitionDown(ztemp4_lvl1_part2_1__D_0__D_1__D_2__D_3, ztemp4_lvl1_part2_1_lvl2_part1T__D_0__D_1__D_2__D_3, ztemp4_lvl1_part2_1_lvl2_part1B__D_0__D_1__D_2__D_3, 1, 0);
		PartitionDown(t_fj__D_0_1__D_2_3, t_fj_lvl2_part1T__D_0_1__D_2_3, t_fj_lvl2_part1B__D_0_1__D_2_3, 1, 0);
		while(ztemp4_lvl1_part2_1_lvl2_part1T__D_0__D_1__D_2__D_3.Dimension(1) < ztemp4_lvl1_part2_1__D_0__D_1__D_2__D_3.Dimension(1))
		{
			RepartitionDown
			( ztemp4_lvl1_part2_1_lvl2_part1T__D_0__D_1__D_2__D_3,  ztemp4_lvl1_part2_1_lvl2_part1_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       ztemp4_lvl1_part2_1_lvl2_part1_1__D_0__D_1__D_2__D_3,
			  ztemp4_lvl1_part2_1_lvl2_part1B__D_0__D_1__D_2__D_3, ztemp4_lvl1_part2_1_lvl2_part1_2__D_0__D_1__D_2__D_3, 1, blkSize );
			RepartitionDown
			( t_fj_lvl2_part1T__D_0_1__D_2_3,  t_fj_lvl2_part1_0__D_0_1__D_2_3,
			  /**/ /**/
			       t_fj_lvl2_part1_1__D_0_1__D_2_3,
			  t_fj_lvl2_part1B__D_0_1__D_2_3, t_fj_lvl2_part1_2__D_0_1__D_2_3, 1, blkSize );

			   // t_fj_lvl2_part1_1[D03,D21] <- t_fj_lvl2_part1_1[D01,D23]
			t_fj_lvl2_part1_1__D_0_3__D_2_1.AlignModesWith( modes_0_1, ztemp4_lvl1_part2_1_lvl2_part1_1__D_0__D_1__D_2__D_3, modes_3_1 );
			t_fj_lvl2_part1_1__D_0_3__D_2_1.AllToAllRedistFrom( t_fj_lvl2_part1_1__D_0_1__D_2_3, modes_1_3 );
			   // t_fj_lvl2_part1_1[D30,D21] <- t_fj_lvl2_part1_1[D03,D21]
			t_fj_lvl2_part1_1__D_3_0__D_2_1.AlignModesWith( modes_0_1, ztemp4_lvl1_part2_1_lvl2_part1_1__D_0__D_1__D_2__D_3, modes_3_1 );
			t_fj_lvl2_part1_1__D_3_0__D_2_1.PermutationRedistFrom( t_fj_lvl2_part1_1__D_0_3__D_2_1, modes_0_3 );
			t_fj_lvl2_part1_1__D_0_3__D_2_1.EmptyData();
			   // t_fj_lvl2_part1_1[D30,D12] <- t_fj_lvl2_part1_1[D30,D21]
			t_fj_lvl2_part1_1__D_3_0__D_1_2.AlignModesWith( modes_0_1, ztemp4_lvl1_part2_1_lvl2_part1_1__D_0__D_1__D_2__D_3, modes_3_1 );
			t_fj_lvl2_part1_1__D_3_0__D_1_2.PermutationRedistFrom( t_fj_lvl2_part1_1__D_3_0__D_2_1, modes_2_1 );
			t_fj_lvl2_part1_1__D_3_0__D_2_1.EmptyData();
			   // t_fj_lvl2_part1_1[D3,D1] <- t_fj_lvl2_part1_1[D30,D12]
			t_fj_lvl2_part1_1__D_3__D_1.AlignModesWith( modes_0_1, ztemp4_lvl1_part2_1_lvl2_part1_1__D_0__D_1__D_2__D_3, modes_3_1 );
			t_fj_lvl2_part1_1__D_3__D_1.AllGatherRedistFrom( t_fj_lvl2_part1_1__D_3_0__D_1_2, modes_0_2 );
			t_fj_lvl2_part1_1__D_3_0__D_1_2.EmptyData();
			Permute( ztemp4_lvl1_part2_1_lvl2_part1_1__D_0__D_1__D_2__D_3, ztemp4_lvl1_part2_1_lvl2_part1_1_perm0231__D_0__D_2__D_3__D_1 );
			z_ai_lvl1_part1_1__D_0__D_2__D_3__D_1.AlignModesWith( modes_0_1, ztemp4_lvl1_part2_1_lvl2_part1_1__D_0__D_1__D_2__D_3, modes_0_2 );
			tempShape = z_ai_lvl1_part1_1__D_0_1__D_2_3.Shape();
			tempShape.push_back( g.Shape()[3] );
			tempShape.push_back( g.Shape()[1] );
			z_ai_lvl1_part1_1__D_0__D_2__D_3__D_1.ResizeTo( tempShape );
			   // 1.0 * ztemp4_lvl1_part2_1_lvl2_part1_1[D0,D1,D2,D3]_aiem * t_fj_lvl2_part1_1[D3,D1]_em + 0.0 * z_ai_lvl1_part1_1[D0,D2,D3,D1]_aiem
			LocalContract(1.0, ztemp4_lvl1_part2_1_lvl2_part1_1_perm0231__D_0__D_2__D_3__D_1.LockedTensor(), indices_aiem, false,
				t_fj_lvl2_part1_1__D_3__D_1.LockedTensor(), indices_em, false,
				0.0, z_ai_lvl1_part1_1__D_0__D_2__D_3__D_1.Tensor(), indices_aiem, false);
			ztemp4_lvl1_part2_1_lvl2_part1_1_perm0231__D_0__D_2__D_3__D_1.EmptyData();
			t_fj_lvl2_part1_1__D_3__D_1.EmptyData();
			   // z_ai_lvl1_part1_1[D01,D23] <- z_ai_lvl1_part1_1[D0,D2,D3,D1] (with SumScatter on (D3)(D1))
			z_ai_lvl1_part1_1__D_0_1__D_2_3.ReduceScatterUpdateRedistFrom( z_ai_lvl1_part1_1__D_0__D_2__D_3__D_1, 1.0, modes_3_2 );
			z_ai_lvl1_part1_1__D_0__D_2__D_3__D_1.EmptyData();

			SlidePartitionDown
			( ztemp4_lvl1_part2_1_lvl2_part1T__D_0__D_1__D_2__D_3,  ztemp4_lvl1_part2_1_lvl2_part1_0__D_0__D_1__D_2__D_3,
			       ztemp4_lvl1_part2_1_lvl2_part1_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  ztemp4_lvl1_part2_1_lvl2_part1B__D_0__D_1__D_2__D_3, ztemp4_lvl1_part2_1_lvl2_part1_2__D_0__D_1__D_2__D_3, 1 );
			SlidePartitionDown
			( t_fj_lvl2_part1T__D_0_1__D_2_3,  t_fj_lvl2_part1_0__D_0_1__D_2_3,
			       t_fj_lvl2_part1_1__D_0_1__D_2_3,
			  /**/ /**/
			  t_fj_lvl2_part1B__D_0_1__D_2_3, t_fj_lvl2_part1_2__D_0_1__D_2_3, 1 );

		}
		//****

		SlidePartitionDown
		( ztemp4_lvl1_part2T__D_0__D_1__D_2__D_3,  ztemp4_lvl1_part2_0__D_0__D_1__D_2__D_3,
		       ztemp4_lvl1_part2_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  ztemp4_lvl1_part2B__D_0__D_1__D_2__D_3, ztemp4_lvl1_part2_2__D_0__D_1__D_2__D_3, 2 );
		SlidePartitionDown
		( z_ai_lvl1_part1T__D_0_1__D_2_3,  z_ai_lvl1_part1_0__D_0_1__D_2_3,
		       z_ai_lvl1_part1_1__D_0_1__D_2_3,
		  /**/ /**/
		  z_ai_lvl1_part1B__D_0_1__D_2_3, z_ai_lvl1_part1_2__D_0_1__D_2_3, 1 );

	}
	//****
	ztemp4__D_0__D_1__D_2__D_3.EmptyData();
	tempShape = U_mnie__D_0__D_1__D_2__D_3.Shape();
	ztemp5__D_0__D_1__D_2__D_3.ResizeTo( tempShape );
	//**** (out of 1)
	//**** Is real	0 shadows
		//Outputs:
		//  ztemp5__D_0__D_1__D_2__D_3
	PartitionDown(U_mnie__D_0__D_1__D_2__D_3, U_mnie_lvl1_part0T__D_0__D_1__D_2__D_3, U_mnie_lvl1_part0B__D_0__D_1__D_2__D_3, 0, 0);
	PartitionDown(U_mnie__D_0__D_1__D_2__D_3, U_mnie_lvl1_part1T__D_0__D_1__D_2__D_3, U_mnie_lvl1_part1B__D_0__D_1__D_2__D_3, 1, 0);
	PartitionDown(ztemp5__D_0__D_1__D_2__D_3, ztemp5_lvl1_part0T__D_0__D_1__D_2__D_3, ztemp5_lvl1_part0B__D_0__D_1__D_2__D_3, 0, 0);
	while(ztemp5_lvl1_part0T__D_0__D_1__D_2__D_3.Dimension(0) < ztemp5__D_0__D_1__D_2__D_3.Dimension(0))
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
		( ztemp5_lvl1_part0T__D_0__D_1__D_2__D_3,  ztemp5_lvl1_part0_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       ztemp5_lvl1_part0_1__D_0__D_1__D_2__D_3,
		  ztemp5_lvl1_part0B__D_0__D_1__D_2__D_3, ztemp5_lvl1_part0_2__D_0__D_1__D_2__D_3, 0, blkSize );

		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  ztemp5_lvl1_part0_1__D_0__D_1__D_2__D_3
		PartitionDown(U_mnie_lvl1_part0_1__D_0__D_1__D_2__D_3, U_mnie_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3, U_mnie_lvl1_part0_1_lvl2_part1B__D_0__D_1__D_2__D_3, 1, 0);
		PartitionDown(U_mnie_lvl1_part1_1__D_0__D_1__D_2__D_3, U_mnie_lvl1_part1_1_lvl2_part0T__D_0__D_1__D_2__D_3, U_mnie_lvl1_part1_1_lvl2_part0B__D_0__D_1__D_2__D_3, 0, 0);
		PartitionDown(ztemp5_lvl1_part0_1__D_0__D_1__D_2__D_3, ztemp5_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3, ztemp5_lvl1_part0_1_lvl2_part1B__D_0__D_1__D_2__D_3, 1, 0);
		while(ztemp5_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3.Dimension(1) < ztemp5_lvl1_part0_1__D_0__D_1__D_2__D_3.Dimension(1))
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
			( ztemp5_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3,  ztemp5_lvl1_part0_1_lvl2_part1_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       ztemp5_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3,
			  ztemp5_lvl1_part0_1_lvl2_part1B__D_0__D_1__D_2__D_3, ztemp5_lvl1_part0_1_lvl2_part1_2__D_0__D_1__D_2__D_3, 1, blkSize );

			   // U_mnie_lvl1_part1_1_lvl2_part0_1[D1,D0,D2,D3] <- U_mnie_lvl1_part1_1_lvl2_part0_1[D0,D1,D2,D3]
			U_mnie_lvl1_part1_1_lvl2_part0_1__D_1__D_0__D_2__D_3.AlignModesWith( modes_0_1_2_3, U_mnie_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3, modes_1_0_2_3 );
			U_mnie_lvl1_part1_1_lvl2_part0_1__D_1__D_0__D_2__D_3.AllToAllRedistFrom( U_mnie_lvl1_part1_1_lvl2_part0_1__D_0__D_1__D_2__D_3, modes_0_1 );
			YAxpPx( 2.0, U_mnie_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3, -1.0, U_mnie_lvl1_part1_1_lvl2_part0_1__D_1__D_0__D_2__D_3, perm_1_0_2_3, ztemp5_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3 );
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
			( ztemp5_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3,  ztemp5_lvl1_part0_1_lvl2_part1_0__D_0__D_1__D_2__D_3,
			       ztemp5_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  ztemp5_lvl1_part0_1_lvl2_part1B__D_0__D_1__D_2__D_3, ztemp5_lvl1_part0_1_lvl2_part1_2__D_0__D_1__D_2__D_3, 1 );

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
		( ztemp5_lvl1_part0T__D_0__D_1__D_2__D_3,  ztemp5_lvl1_part0_0__D_0__D_1__D_2__D_3,
		       ztemp5_lvl1_part0_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  ztemp5_lvl1_part0B__D_0__D_1__D_2__D_3, ztemp5_lvl1_part0_2__D_0__D_1__D_2__D_3, 0 );

	}
	//****
	//**** (out of 1)
	//**** Is real	0 shadows
		//Outputs:
		//  z_ai__D_0_1__D_2_3
	PartitionDown(ztemp5__D_0__D_1__D_2__D_3, ztemp5_lvl1_part2T__D_0__D_1__D_2__D_3, ztemp5_lvl1_part2B__D_0__D_1__D_2__D_3, 2, 0);
	PartitionDown(z_ai__D_0_1__D_2_3, z_ai_lvl1_part1T__D_0_1__D_2_3, z_ai_lvl1_part1B__D_0_1__D_2_3, 1, 0);
	while(z_ai_lvl1_part1T__D_0_1__D_2_3.Dimension(1) < z_ai__D_0_1__D_2_3.Dimension(1))
	{
		RepartitionDown
		( ztemp5_lvl1_part2T__D_0__D_1__D_2__D_3,  ztemp5_lvl1_part2_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       ztemp5_lvl1_part2_1__D_0__D_1__D_2__D_3,
		  ztemp5_lvl1_part2B__D_0__D_1__D_2__D_3, ztemp5_lvl1_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );
		RepartitionDown
		( z_ai_lvl1_part1T__D_0_1__D_2_3,  z_ai_lvl1_part1_0__D_0_1__D_2_3,
		  /**/ /**/
		       z_ai_lvl1_part1_1__D_0_1__D_2_3,
		  z_ai_lvl1_part1B__D_0_1__D_2_3, z_ai_lvl1_part1_2__D_0_1__D_2_3, 1, blkSize );

		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  z_ai_lvl1_part1_1__D_0_1__D_2_3
		PartitionDown(ztemp5_lvl1_part2_1__D_0__D_1__D_2__D_3, ztemp5_lvl1_part2_1_lvl2_part0T__D_0__D_1__D_2__D_3, ztemp5_lvl1_part2_1_lvl2_part0B__D_0__D_1__D_2__D_3, 0, 0);
		PartitionDown(T_bfnj__D_0__D_1__D_2__D_3, T_bfnj_lvl2_part2T__D_0__D_1__D_2__D_3, T_bfnj_lvl2_part2B__D_0__D_1__D_2__D_3, 2, 0);
		while(ztemp5_lvl1_part2_1_lvl2_part0T__D_0__D_1__D_2__D_3.Dimension(0) < ztemp5_lvl1_part2_1__D_0__D_1__D_2__D_3.Dimension(0))
		{
			RepartitionDown
			( ztemp5_lvl1_part2_1_lvl2_part0T__D_0__D_1__D_2__D_3,  ztemp5_lvl1_part2_1_lvl2_part0_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       ztemp5_lvl1_part2_1_lvl2_part0_1__D_0__D_1__D_2__D_3,
			  ztemp5_lvl1_part2_1_lvl2_part0B__D_0__D_1__D_2__D_3, ztemp5_lvl1_part2_1_lvl2_part0_2__D_0__D_1__D_2__D_3, 0, blkSize );
			RepartitionDown
			( T_bfnj_lvl2_part2T__D_0__D_1__D_2__D_3,  T_bfnj_lvl2_part2_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       T_bfnj_lvl2_part2_1__D_0__D_1__D_2__D_3,
			  T_bfnj_lvl2_part2B__D_0__D_1__D_2__D_3, T_bfnj_lvl2_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );

			   // ztemp5_lvl1_part2_1_lvl2_part0_1[D0,D3,D2,D1] <- ztemp5_lvl1_part2_1_lvl2_part0_1[D0,D1,D2,D3]
			ztemp5_lvl1_part2_1_lvl2_part0_1__D_0__D_3__D_2__D_1.AlignModesWith( modes_0_1_3, T_bfnj_lvl2_part2_1__D_0__D_1__D_2__D_3, modes_2_3_1 );
			ztemp5_lvl1_part2_1_lvl2_part0_1__D_0__D_3__D_2__D_1.AllToAllRedistFrom( ztemp5_lvl1_part2_1_lvl2_part0_1__D_0__D_1__D_2__D_3, modes_1_3 );
			   // ztemp5_lvl1_part2_1_lvl2_part0_1[D2,D3,*,D1] <- ztemp5_lvl1_part2_1_lvl2_part0_1[D0,D3,D2,D1]
			ztemp5_lvl1_part2_1_lvl2_part0_1_perm2013__S__D_2__D_3__D_1.AlignModesWith( modes_0_1_3, T_bfnj_lvl2_part2_1__D_0__D_1__D_2__D_3, modes_2_3_1 );
			ztemp5_lvl1_part2_1_lvl2_part0_1_perm2013__S__D_2__D_3__D_1.AllToAllRedistFrom( ztemp5_lvl1_part2_1_lvl2_part0_1__D_0__D_3__D_2__D_1, modes_0_2 );
			ztemp5_lvl1_part2_1_lvl2_part0_1__D_0__D_3__D_2__D_1.EmptyData();
			Permute( T_bfnj_lvl2_part2_1__D_0__D_1__D_2__D_3, T_bfnj_lvl2_part2_1_perm2310__D_2__D_3__D_1__D_0 );
			z_ai_lvl1_part1_1_perm10342__S__D_0__D_2__D_3__D_1.AlignModesWith( modes_0, T_bfnj_lvl2_part2_1__D_0__D_1__D_2__D_3, modes_0 );
			tempShape = z_ai_lvl1_part1_1__D_0_1__D_2_3.Shape();
			tempShape.push_back( g.Shape()[1] );
			tempShape.push_back( g.Shape()[2] );
			tempShape.push_back( g.Shape()[3] );
			z_ai_lvl1_part1_1_perm10342__S__D_0__D_2__D_3__D_1.ResizeTo( tempShape );
			   // -1.0 * ztemp5_lvl1_part2_1_lvl2_part0_1[D2,D3,*,D1]_imne * T_bfnj_lvl2_part2_1[D0,D1,D2,D3]_mnea + 0.0 * z_ai_lvl1_part1_1[D0,*,D1,D2,D3]_iamne
			LocalContract(-1.0, ztemp5_lvl1_part2_1_lvl2_part0_1_perm2013__S__D_2__D_3__D_1.LockedTensor(), indices_imne, false,
				T_bfnj_lvl2_part2_1_perm2310__D_2__D_3__D_1__D_0.LockedTensor(), indices_mnea, false,
				0.0, z_ai_lvl1_part1_1_perm10342__S__D_0__D_2__D_3__D_1.Tensor(), indices_iamne, false);
			ztemp5_lvl1_part2_1_lvl2_part0_1_perm2013__S__D_2__D_3__D_1.EmptyData();
			T_bfnj_lvl2_part2_1_perm2310__D_2__D_3__D_1__D_0.EmptyData();
			   // z_ai_lvl1_part1_1[D01,D23] <- z_ai_lvl1_part1_1[D0,*,D1,D2,D3] (with SumScatter on (D1)(D2)(D3))
			z_ai_lvl1_part1_1__D_0_1__D_2_3.ReduceScatterUpdateRedistFrom( z_ai_lvl1_part1_1_perm10342__S__D_0__D_2__D_3__D_1, 1.0, modes_4_3_2 );
			z_ai_lvl1_part1_1_perm10342__S__D_0__D_2__D_3__D_1.EmptyData();

			SlidePartitionDown
			( ztemp5_lvl1_part2_1_lvl2_part0T__D_0__D_1__D_2__D_3,  ztemp5_lvl1_part2_1_lvl2_part0_0__D_0__D_1__D_2__D_3,
			       ztemp5_lvl1_part2_1_lvl2_part0_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  ztemp5_lvl1_part2_1_lvl2_part0B__D_0__D_1__D_2__D_3, ztemp5_lvl1_part2_1_lvl2_part0_2__D_0__D_1__D_2__D_3, 0 );
			SlidePartitionDown
			( T_bfnj_lvl2_part2T__D_0__D_1__D_2__D_3,  T_bfnj_lvl2_part2_0__D_0__D_1__D_2__D_3,
			       T_bfnj_lvl2_part2_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  T_bfnj_lvl2_part2B__D_0__D_1__D_2__D_3, T_bfnj_lvl2_part2_2__D_0__D_1__D_2__D_3, 2 );

		}
		//****

		SlidePartitionDown
		( ztemp5_lvl1_part2T__D_0__D_1__D_2__D_3,  ztemp5_lvl1_part2_0__D_0__D_1__D_2__D_3,
		       ztemp5_lvl1_part2_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  ztemp5_lvl1_part2B__D_0__D_1__D_2__D_3, ztemp5_lvl1_part2_2__D_0__D_1__D_2__D_3, 2 );
		SlidePartitionDown
		( z_ai_lvl1_part1T__D_0_1__D_2_3,  z_ai_lvl1_part1_0__D_0_1__D_2_3,
		       z_ai_lvl1_part1_1__D_0_1__D_2_3,
		  /**/ /**/
		  z_ai_lvl1_part1B__D_0_1__D_2_3, z_ai_lvl1_part1_2__D_0_1__D_2_3, 1 );

	}
	//****
	ztemp5__D_0__D_1__D_2__D_3.EmptyData();
	//**** (out of 1)
	//**** Is real	0 shadows
		//Outputs:
		//  z_ai__D_0_1__D_2_3
	PartitionDown(G_mi__D_0_1__D_2_3, G_mi_lvl1_part1T__D_0_1__D_2_3, G_mi_lvl1_part1B__D_0_1__D_2_3, 1, 0);
	PartitionDown(z_ai__D_0_1__D_2_3, z_ai_lvl1_part1T__D_0_1__D_2_3, z_ai_lvl1_part1B__D_0_1__D_2_3, 1, 0);
	while(z_ai_lvl1_part1T__D_0_1__D_2_3.Dimension(1) < z_ai__D_0_1__D_2_3.Dimension(1))
	{
		RepartitionDown
		( G_mi_lvl1_part1T__D_0_1__D_2_3,  G_mi_lvl1_part1_0__D_0_1__D_2_3,
		  /**/ /**/
		       G_mi_lvl1_part1_1__D_0_1__D_2_3,
		  G_mi_lvl1_part1B__D_0_1__D_2_3, G_mi_lvl1_part1_2__D_0_1__D_2_3, 1, blkSize );
		RepartitionDown
		( z_ai_lvl1_part1T__D_0_1__D_2_3,  z_ai_lvl1_part1_0__D_0_1__D_2_3,
		  /**/ /**/
		       z_ai_lvl1_part1_1__D_0_1__D_2_3,
		  z_ai_lvl1_part1B__D_0_1__D_2_3, z_ai_lvl1_part1_2__D_0_1__D_2_3, 1, blkSize );

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

			   // G_mi_lvl1_part1_1_lvl2_part0_1[*,D23] <- G_mi_lvl1_part1_1_lvl2_part0_1[D01,D23]
			G_mi_lvl1_part1_1_lvl2_part0_1_perm10__D_2_3__S.AlignModesWith( modes_1, z_ai_lvl1_part1_1__D_0_1__D_2_3, modes_1 );
			G_mi_lvl1_part1_1_lvl2_part0_1_perm10__D_2_3__S.AllGatherRedistFrom( G_mi_lvl1_part1_1_lvl2_part0_1__D_0_1__D_2_3, modes_0_1 );
			   // t_fj_lvl2_part1_1[D01,*] <- t_fj_lvl2_part1_1[D01,D23]
			t_fj_lvl2_part1_1_perm10__S__D_0_1.AlignModesWith( modes_0, z_ai_lvl1_part1_1__D_0_1__D_2_3, modes_0 );
			t_fj_lvl2_part1_1_perm10__S__D_0_1.AllGatherRedistFrom( t_fj_lvl2_part1_1__D_0_1__D_2_3, modes_2_3 );
			   // -1.0 * G_mi_lvl1_part1_1_lvl2_part0_1[*,D23]_im * t_fj_lvl2_part1_1[D01,*]_ma + 1.0 * z_ai_lvl1_part1_1[D01,D23]_ia
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
		( G_mi_lvl1_part1T__D_0_1__D_2_3,  G_mi_lvl1_part1_0__D_0_1__D_2_3,
		       G_mi_lvl1_part1_1__D_0_1__D_2_3,
		  /**/ /**/
		  G_mi_lvl1_part1B__D_0_1__D_2_3, G_mi_lvl1_part1_2__D_0_1__D_2_3, 1 );
		SlidePartitionDown
		( z_ai_lvl1_part1T__D_0_1__D_2_3,  z_ai_lvl1_part1_0__D_0_1__D_2_3,
		       z_ai_lvl1_part1_1__D_0_1__D_2_3,
		  /**/ /**/
		  z_ai_lvl1_part1B__D_0_1__D_2_3, z_ai_lvl1_part1_2__D_0_1__D_2_3, 1 );

	}
	//****


//****


//**** (out of 1)
//**** Is real	0 shadows
	//Outputs:
	//  Z_abij__D_0__D_1__D_2__D_3

	tempShape = Z_abij__D_0__D_1__D_2__D_3.Shape();
	Ztemp1__D_0__D_1__D_2__D_3.ResizeTo( tempShape );
	tempShape = Ztemp1__D_0__D_1__D_2__D_3.Shape();
	Ztemp1_perm1302__D_1__D_3__D_0__D_2.ResizeTo( tempShape );
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

			   // T_bfnj_lvl1_part2_1_lvl2_part3_1[*,D1,D2,D3] <- T_bfnj_lvl1_part2_1_lvl2_part3_1[D0,D1,D2,D3]
			T_bfnj_lvl1_part2_1_lvl2_part3_1__S__D_1__D_2__D_3.AlignModesWith( modes_1_2_3, Zaccum_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3, modes_1_2_3 );
			T_bfnj_lvl1_part2_1_lvl2_part3_1__S__D_1__D_2__D_3.AllGatherRedistFrom( T_bfnj_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3, modes_0 );
			   // F_ae[D0,*] <- F_ae[D01,D23]
			F_ae__D_0__S.AlignModesWith( modes_0, Zaccum_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3, modes_0 );
			F_ae__D_0__S.AllGatherRedistFrom( F_ae__D_0_1__D_2_3, modes_1_2_3 );
			   // 1.0 * F_ae[D0,*]_ae * T_bfnj_lvl1_part2_1_lvl2_part3_1[*,D1,D2,D3]_ebij + 1.0 * Zaccum_lvl1_part2_1_lvl2_part3_1[D0,D1,D2,D3]_abij
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

			   // t_fj_lvl2_part1_1[D0,*] <- t_fj_lvl2_part1_1[D01,D23]
			t_fj_lvl2_part1_1_perm10__S__D_0.AlignModesWith( modes_0, Zaccum_lvl1_part2_1__D_0__D_1__D_2__D_3, modes_0 );
			t_fj_lvl2_part1_1_perm10__S__D_0.AllGatherRedistFrom( t_fj_lvl2_part1_1__D_0_1__D_2_3, modes_1_2_3 );
			   // P_jimb_lvl1_part0_1_lvl2_part2_1[D0,D3,D2,D1] <- P_jimb_lvl1_part0_1_lvl2_part2_1[D0,D1,D2,D3]
			P_jimb_lvl1_part0_1_lvl2_part2_1__D_0__D_3__D_2__D_1.AlignModesWith( modes_0_1_3, Zaccum_lvl1_part2_1__D_0__D_1__D_2__D_3, modes_2_3_1 );
			P_jimb_lvl1_part0_1_lvl2_part2_1__D_0__D_3__D_2__D_1.AllToAllRedistFrom( P_jimb_lvl1_part0_1_lvl2_part2_1__D_0__D_1__D_2__D_3, modes_1_3 );
			   // P_jimb_lvl1_part0_1_lvl2_part2_1[D2,D3,*,D1] <- P_jimb_lvl1_part0_1_lvl2_part2_1[D0,D3,D2,D1]
			P_jimb_lvl1_part0_1_lvl2_part2_1_perm0132__D_2__D_3__D_1__S.AlignModesWith( modes_0_1_3, Zaccum_lvl1_part2_1__D_0__D_1__D_2__D_3, modes_2_3_1 );
			P_jimb_lvl1_part0_1_lvl2_part2_1_perm0132__D_2__D_3__D_1__S.AllToAllRedistFrom( P_jimb_lvl1_part0_1_lvl2_part2_1__D_0__D_3__D_2__D_1, modes_0_2 );
			P_jimb_lvl1_part0_1_lvl2_part2_1__D_0__D_3__D_2__D_1.EmptyData();
			   // -1.0 * P_jimb_lvl1_part0_1_lvl2_part2_1[D2,D3,*,D1]_ijbm * t_fj_lvl2_part1_1[D0,*]_ma + 1.0 * Zaccum_lvl1_part2_1[D0,D1,D2,D3]_ijba
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

			   // t_fj_lvl1_part1_1[D0,*] <- t_fj_lvl1_part1_1[D01,D23]
			t_fj_lvl1_part1_1__D_0__S.AlignModesWith( modes_0, r_bmfe_lvl2_part1_1__D_0__D_1__D_2__D_3, modes_0 );
			t_fj_lvl1_part1_1__D_0__S.AllGatherRedistFrom( t_fj_lvl1_part1_1__D_0_1__D_2_3, modes_1_2_3 );
			Permute( r_bmfe_lvl2_part1_1__D_0__D_1__D_2__D_3, r_bmfe_lvl2_part1_1_perm1230__D_1__D_2__D_3__D_0 );
			Zaccum_lvl1_part2_1_lvl2_part3_1_perm30124__D_1__D_2__D_3__S__D_0.AlignModesWith( modes_0_1_3, r_bmfe_lvl2_part1_1__D_0__D_1__D_2__D_3, modes_2_3_1 );
			tempShape = Zaccum_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3.Shape();
			tempShape.push_back( g.Shape()[0] );
			Zaccum_lvl1_part2_1_lvl2_part3_1_perm30124__D_1__D_2__D_3__S__D_0.ResizeTo( tempShape );
			   // 1.0 * r_bmfe_lvl2_part1_1[D0,D1,D2,D3]_jabe * t_fj_lvl1_part1_1[D0,*]_ei + 0.0 * Zaccum_lvl1_part2_1_lvl2_part3_1[D2,D3,*,D1,D0]_jabie
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
     std::cout << "NORM_c " << norm << std::endl;
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


