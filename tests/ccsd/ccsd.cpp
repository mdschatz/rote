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
ObjShape overwrite_tmpShape_Tau;
TensorDistribution dist__D_0__D_1__D_2__D_3 = tmen::StringToTensorDist("[(0),(1),(2),(3)]");
TensorDistribution dist__D_0__D_2 = tmen::StringToTensorDist("[(0),(2)]");
TensorDistribution dist__D_1__D_3 = tmen::StringToTensorDist("[(1),(3)]");
TensorDistribution dist__D_0_1__D_2_3 = tmen::StringToTensorDist("[(0,1),(2,3)]");
TensorDistribution dist__D_0_1__D_3 = tmen::StringToTensorDist("[(0,1),(3)]");
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
	//t_fj_lvl2_part1_1[D01,D3]
DistTensor<double> t_fj_lvl2_part1_1__D_0_1__D_3( dist__D_0_1__D_3, g );
	//t_fj_lvl2_part1_1[D1,D3]
DistTensor<double> t_fj_lvl2_part1_1__D_1__D_3( dist__D_1__D_3, g );
	//t_fj_lvl2_part1_2[D01,D23]
DistTensor<double> t_fj_lvl2_part1_2__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
// t_fj has 2 dims
//	Starting distribution: [D01,D23] or _D_0_1__D_2_3
ObjShape t_fj__D_0_1__D_2_3_tmpShape_Tau( 2 );
t_fj__D_0_1__D_2_3_tmpShape_Tau[ 0 ] = n_v;
t_fj__D_0_1__D_2_3_tmpShape_Tau[ 1 ] = n_o;
t_fj__D_0_1__D_2_3.ResizeTo( t_fj__D_0_1__D_2_3_tmpShape_Tau );
MakeUniform( t_fj__D_0_1__D_2_3 );
// T_bfnj has 4 dims
//	Starting distribution: [D0,D1,D2,D3] or _D_0__D_1__D_2__D_3
ObjShape T_bfnj__D_0__D_1__D_2__D_3_tmpShape_Tau( 4 );
T_bfnj__D_0__D_1__D_2__D_3_tmpShape_Tau[ 0 ] = n_v;
T_bfnj__D_0__D_1__D_2__D_3_tmpShape_Tau[ 1 ] = n_v;
T_bfnj__D_0__D_1__D_2__D_3_tmpShape_Tau[ 2 ] = n_o;
T_bfnj__D_0__D_1__D_2__D_3_tmpShape_Tau[ 3 ] = n_o;
T_bfnj__D_0__D_1__D_2__D_3.ResizeTo( T_bfnj__D_0__D_1__D_2__D_3_tmpShape_Tau );
MakeUniform( T_bfnj__D_0__D_1__D_2__D_3 );
// Tau_efmn has 4 dims
ObjShape Tau_efmn__D_0__D_1__D_2__D_3_tmpShape_Tau( 4 );
Tau_efmn__D_0__D_1__D_2__D_3_tmpShape_Tau[ 0 ] = n_v;
Tau_efmn__D_0__D_1__D_2__D_3_tmpShape_Tau[ 1 ] = n_v;
Tau_efmn__D_0__D_1__D_2__D_3_tmpShape_Tau[ 2 ] = n_o;
Tau_efmn__D_0__D_1__D_2__D_3_tmpShape_Tau[ 3 ] = n_o;
Tau_efmn__D_0__D_1__D_2__D_3.ResizeTo( Tau_efmn__D_0__D_1__D_2__D_3_tmpShape_Tau );
MakeUniform( Tau_efmn__D_0__D_1__D_2__D_3 );
ObjShape overwrite_tmpShape_W;
TensorDistribution dist__S__D_3__D_1__S = tmen::StringToTensorDist("[(),(3),(1),()]");
TensorDistribution dist__D_0__S__S__D_2 = tmen::StringToTensorDist("[(0),(),(),(2)]");
TensorDistribution dist__D_0__S__D_2_1__D_3 = tmen::StringToTensorDist("[(0),(),(2,1),(3)]");
TensorDistribution dist__D_0__S__D_2__D_3 = tmen::StringToTensorDist("[(0),(),(2),(3)]");
TensorDistribution dist__D_0__S = tmen::StringToTensorDist("[(0),()]");
TensorDistribution dist__D_0__D_1__S__D_2_3 = tmen::StringToTensorDist("[(0),(1),(),(2,3)]");
TensorDistribution dist__D_0__D_1__S__D_2__D_3 = tmen::StringToTensorDist("[(0),(1),(),(2),(3)]");
TensorDistribution dist__D_0__D_1__D_3__D_2 = tmen::StringToTensorDist("[(0),(1),(3),(2)]");
TensorDistribution dist__D_0__D_3__D_1_2__S = tmen::StringToTensorDist("[(0),(3),(1,2),()]");
TensorDistribution dist__D_0__D_3__D_2_1__S = tmen::StringToTensorDist("[(0),(3),(2,1),()]");
TensorDistribution dist__D_1__S__D_2__D_3 = tmen::StringToTensorDist("[(1),(),(2),(3)]");
TensorDistribution dist__D_1__D_0__D_2__D_3 = tmen::StringToTensorDist("[(1),(0),(2),(3)]");
TensorDistribution dist__D_3__S = tmen::StringToTensorDist("[(3),()]");
TensorDistribution dist__D_3__D_2 = tmen::StringToTensorDist("[(3),(2)]");
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
Permutation perm_2_1_3_0( 4 );
perm_2_1_3_0[0] = 2;
perm_2_1_3_0[1] = 1;
perm_2_1_3_0[2] = 3;
perm_2_1_3_0[3] = 0;
Permutation perm_3_0_1_2( 4 );
perm_3_0_1_2[0] = 3;
perm_3_0_1_2[1] = 0;
perm_3_0_1_2[2] = 1;
perm_3_0_1_2[3] = 2;
Permutation perm_3_1_0_2( 4 );
perm_3_1_0_2[0] = 3;
perm_3_1_0_2[1] = 1;
perm_3_1_0_2[2] = 0;
perm_3_1_0_2[3] = 2;
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
ModeArray modes_3( 1 );
modes_3[0] = 3;
ModeArray modes_3_1( 2 );
modes_3_1[0] = 3;
modes_3_1[1] = 1;
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
IndexArray indices_embj( 4 );
indices_embj[0] = 'e';
indices_embj[1] = 'm';
indices_embj[2] = 'b';
indices_embj[3] = 'j';
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
	//W_bmje_lvl0_part1B[D0,D1,D2,D3]
DistTensor<double> W_bmje_lvl0_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_bmje_lvl0_part1T[D0,D1,D2,D3]
DistTensor<double> W_bmje_lvl0_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
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
	//W_bmje_lvl1_part1_1_lvl2_part2_1_W_temp[D0,D1,D2,D3]
DistTensor<double> W_bmje_lvl1_part1_1_lvl2_part2_1_W_temp__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_bmje_lvl1_part1_1_lvl2_part2_1_W_temp[D0,D1,*,D23]
DistTensor<double> W_bmje_lvl1_part1_1_lvl2_part2_1_W_temp__D_0__D_1__S__D_2_3( dist__D_0__D_1__S__D_2_3, g );
	//W_bmje_lvl1_part1_1_lvl2_part2_2[D0,D1,D2,D3]
DistTensor<double> W_bmje_lvl1_part1_1_lvl2_part2_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> W_bmje_lvl1_part1_1_perm1230__D_1__D_2__D_3__D_0( dist__D_0__D_1__D_2__D_3, g );
W_bmje_lvl1_part1_1_perm1230__D_1__D_2__D_3__D_0.SetLocalPermutation( perm_1_2_3_0 );
	//W_bmje_lvl1_part1_2[D0,D1,D2,D3]
DistTensor<double> W_bmje_lvl1_part1_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> W_bmje_perm3102__D_3__D_1__D_0__D_2( dist__D_0__D_1__D_2__D_3, g );
W_bmje_perm3102__D_3__D_1__D_0__D_2.SetLocalPermutation( perm_3_1_0_2 );
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
	//t_fj_lvl2_part1_1[D3,D2]
DistTensor<double> t_fj_lvl2_part1_1__D_3__D_2( dist__D_3__D_2, g );
	//t_fj_lvl2_part1_1[D3,*]
DistTensor<double> t_fj_lvl2_part1_1__D_3__S( dist__D_3__S, g );
	//t_fj_lvl2_part1_1[D0,*]
DistTensor<double> t_fj_lvl2_part1_1_perm10__S__D_0( dist__D_0__S, g );
t_fj_lvl2_part1_1_perm10__S__D_0.SetLocalPermutation( perm_1_0 );
	//W_temp1[D0,D1,D2,D3]
DistTensor<double> W_temp1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_temp1_lvl0_part1B[D0,D1,D2,D3]
DistTensor<double> W_temp1_lvl0_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_temp1_lvl0_part1T[D0,D1,D2,D3]
DistTensor<double> W_temp1_lvl0_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_temp1_lvl0_part2B[D0,D1,D2,D3]
DistTensor<double> W_temp1_lvl0_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_temp1_lvl0_part2T[D0,D1,D2,D3]
DistTensor<double> W_temp1_lvl0_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_temp1_lvl1_part1B[D0,D1,D2,D3]
DistTensor<double> W_temp1_lvl1_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_temp1_lvl1_part1T[D0,D1,D2,D3]
DistTensor<double> W_temp1_lvl1_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_temp1_lvl1_part1_0[D0,D1,D2,D3]
DistTensor<double> W_temp1_lvl1_part1_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_temp1_lvl1_part1_1[D0,D1,D2,D3]
DistTensor<double> W_temp1_lvl1_part1_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_temp1_lvl1_part1_1_lvl1_part2B[D0,D1,D2,D3]
DistTensor<double> W_temp1_lvl1_part1_1_lvl1_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_temp1_lvl1_part1_1_lvl1_part2T[D0,D1,D2,D3]
DistTensor<double> W_temp1_lvl1_part1_1_lvl1_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_temp1_lvl1_part1_1_lvl2_part2B[D0,D1,D2,D3]
DistTensor<double> W_temp1_lvl1_part1_1_lvl2_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_temp1_lvl1_part1_1_lvl2_part2T[D0,D1,D2,D3]
DistTensor<double> W_temp1_lvl1_part1_1_lvl2_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_temp1_lvl1_part1_1_lvl2_part2_0[D0,D1,D2,D3]
DistTensor<double> W_temp1_lvl1_part1_1_lvl2_part2_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_temp1_lvl1_part1_1_lvl2_part2_1[D0,D1,D2,D3]
DistTensor<double> W_temp1_lvl1_part1_1_lvl2_part2_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_temp1_lvl1_part1_1_lvl2_part2_1[D0,D1,D3,D2]
DistTensor<double> W_temp1_lvl1_part1_1_lvl2_part2_1__D_0__D_1__D_3__D_2( dist__D_0__D_1__D_3__D_2, g );
	//W_temp1_lvl1_part1_1_lvl2_part2_1[D0,*,*,D2]
DistTensor<double> W_temp1_lvl1_part1_1_lvl2_part2_1_perm1203__S__S__D_0__D_2( dist__D_0__S__S__D_2, g );
W_temp1_lvl1_part1_1_lvl2_part2_1_perm1203__S__S__D_0__D_2.SetLocalPermutation( perm_1_2_0_3 );
	//W_temp1_lvl1_part1_1_lvl2_part2_2[D0,D1,D2,D3]
DistTensor<double> W_temp1_lvl1_part1_1_lvl2_part2_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_temp1_lvl1_part1_2[D0,D1,D2,D3]
DistTensor<double> W_temp1_lvl1_part1_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_temp1_lvl1_part2B[D0,D1,D2,D3]
DistTensor<double> W_temp1_lvl1_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_temp1_lvl1_part2T[D0,D1,D2,D3]
DistTensor<double> W_temp1_lvl1_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_temp1_lvl1_part2_0[D0,D1,D2,D3]
DistTensor<double> W_temp1_lvl1_part2_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_temp1_lvl1_part2_1[D0,D1,D2,D3]
DistTensor<double> W_temp1_lvl1_part2_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_temp1_lvl1_part2_1_lvl1_part3B[D0,D1,D2,D3]
DistTensor<double> W_temp1_lvl1_part2_1_lvl1_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_temp1_lvl1_part2_1_lvl1_part3T[D0,D1,D2,D3]
DistTensor<double> W_temp1_lvl1_part2_1_lvl1_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_temp1_lvl1_part2_1_lvl2_part3B[D0,D1,D2,D3]
DistTensor<double> W_temp1_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_temp1_lvl1_part2_1_lvl2_part3T[D0,D1,D2,D3]
DistTensor<double> W_temp1_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_temp1_lvl1_part2_1_lvl2_part3_0[D0,D1,D2,D3]
DistTensor<double> W_temp1_lvl1_part2_1_lvl2_part3_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_temp1_lvl1_part2_1_lvl2_part3_1[D0,D1,D2,D3]
DistTensor<double> W_temp1_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_temp1_lvl1_part2_1_lvl2_part3_2[D0,D1,D2,D3]
DistTensor<double> W_temp1_lvl1_part2_1_lvl2_part3_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_temp1_lvl1_part2_2[D0,D1,D2,D3]
DistTensor<double> W_temp1_lvl1_part2_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_temp2[D0,D1,D2,D3]
DistTensor<double> W_temp2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_temp2_lvl0_part0B[D0,D1,D2,D3]
DistTensor<double> W_temp2_lvl0_part0B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_temp2_lvl0_part0T[D0,D1,D2,D3]
DistTensor<double> W_temp2_lvl0_part0T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_temp2_lvl0_part2B[D0,D1,D2,D3]
DistTensor<double> W_temp2_lvl0_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_temp2_lvl0_part2T[D0,D1,D2,D3]
DistTensor<double> W_temp2_lvl0_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_temp2_lvl1_part0B[D0,D1,D2,D3]
DistTensor<double> W_temp2_lvl1_part0B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_temp2_lvl1_part0T[D0,D1,D2,D3]
DistTensor<double> W_temp2_lvl1_part0T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_temp2_lvl1_part0_0[D0,D1,D2,D3]
DistTensor<double> W_temp2_lvl1_part0_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_temp2_lvl1_part0_1[D0,D1,D2,D3]
DistTensor<double> W_temp2_lvl1_part0_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_temp2_lvl1_part0_1_lvl1_part3B[D0,D1,D2,D3]
DistTensor<double> W_temp2_lvl1_part0_1_lvl1_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_temp2_lvl1_part0_1_lvl1_part3T[D0,D1,D2,D3]
DistTensor<double> W_temp2_lvl1_part0_1_lvl1_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_temp2_lvl1_part0_1_lvl2_part3B[D0,D1,D2,D3]
DistTensor<double> W_temp2_lvl1_part0_1_lvl2_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_temp2_lvl1_part0_1_lvl2_part3T[D0,D1,D2,D3]
DistTensor<double> W_temp2_lvl1_part0_1_lvl2_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_temp2_lvl1_part0_1_lvl2_part3_0[D0,D1,D2,D3]
DistTensor<double> W_temp2_lvl1_part0_1_lvl2_part3_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_temp2_lvl1_part0_1_lvl2_part3_1[D0,D1,D2,D3]
DistTensor<double> W_temp2_lvl1_part0_1_lvl2_part3_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_temp2_lvl1_part0_1_lvl2_part3_1[D0,D3,D12,*]
DistTensor<double> W_temp2_lvl1_part0_1_lvl2_part3_1__D_0__D_3__D_1_2__S( dist__D_0__D_3__D_1_2__S, g );
	//W_temp2_lvl1_part0_1_lvl2_part3_1[D0,D3,D21,*]
DistTensor<double> W_temp2_lvl1_part0_1_lvl2_part3_1__D_0__D_3__D_2_1__S( dist__D_0__D_3__D_2_1__S, g );
	//W_temp2_lvl1_part0_1_lvl2_part3_1[D0,*,D21,D3]
DistTensor<double> W_temp2_lvl1_part0_1_lvl2_part3_1__D_0__S__D_2_1__D_3( dist__D_0__S__D_2_1__D_3, g );
	//W_temp2_lvl1_part0_1_lvl2_part3_1[D0,*,D2,D3]
DistTensor<double> W_temp2_lvl1_part0_1_lvl2_part3_1__D_0__S__D_2__D_3( dist__D_0__S__D_2__D_3, g );
	//W_temp2_lvl1_part0_1_lvl2_part3_1[*,D3,D1,*]
DistTensor<double> W_temp2_lvl1_part0_1_lvl2_part3_1_perm1203__D_3__D_1__S__S( dist__S__D_3__D_1__S, g );
W_temp2_lvl1_part0_1_lvl2_part3_1_perm1203__D_3__D_1__S__S.SetLocalPermutation( perm_1_2_0_3 );
	//W_temp2_lvl1_part0_1_lvl2_part3_2[D0,D1,D2,D3]
DistTensor<double> W_temp2_lvl1_part0_1_lvl2_part3_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_temp2_lvl1_part0_2[D0,D1,D2,D3]
DistTensor<double> W_temp2_lvl1_part0_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_temp2_lvl1_part2B[D0,D1,D2,D3]
DistTensor<double> W_temp2_lvl1_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_temp2_lvl1_part2T[D0,D1,D2,D3]
DistTensor<double> W_temp2_lvl1_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_temp2_lvl1_part2_0[D0,D1,D2,D3]
DistTensor<double> W_temp2_lvl1_part2_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_temp2_lvl1_part2_1[D0,D1,D2,D3]
DistTensor<double> W_temp2_lvl1_part2_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_temp2_lvl1_part2_1_lvl1_part3B[D0,D1,D2,D3]
DistTensor<double> W_temp2_lvl1_part2_1_lvl1_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_temp2_lvl1_part2_1_lvl1_part3T[D0,D1,D2,D3]
DistTensor<double> W_temp2_lvl1_part2_1_lvl1_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_temp2_lvl1_part2_1_lvl2_part3B[D0,D1,D2,D3]
DistTensor<double> W_temp2_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_temp2_lvl1_part2_1_lvl2_part3T[D0,D1,D2,D3]
DistTensor<double> W_temp2_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_temp2_lvl1_part2_1_lvl2_part3_0[D0,D1,D2,D3]
DistTensor<double> W_temp2_lvl1_part2_1_lvl2_part3_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_temp2_lvl1_part2_1_lvl2_part3_1[D0,D1,D2,D3]
DistTensor<double> W_temp2_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_temp2_lvl1_part2_1_lvl2_part3_2[D0,D1,D2,D3]
DistTensor<double> W_temp2_lvl1_part2_1_lvl2_part3_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_temp2_lvl1_part2_2[D0,D1,D2,D3]
DistTensor<double> W_temp2_lvl1_part2_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_temp3_lvl1_part0_1_lvl2_part1_1[D0,D1,D2,D3]
DistTensor<double> W_temp3_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_temp3_lvl1_part0_1_lvl2_part1_1[D1,*,D2,D3]
DistTensor<double> W_temp3_lvl1_part0_1_lvl2_part1_1_perm0231__D_1__D_2__D_3__S( dist__D_1__S__D_2__D_3, g );
W_temp3_lvl1_part0_1_lvl2_part1_1_perm0231__D_1__D_2__D_3__S.SetLocalPermutation( perm_0_2_3_1 );
	//W_temp4[D0,D1,D2,D3]
DistTensor<double> W_temp4__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_temp4_lvl0_part0B[D0,D1,D2,D3]
DistTensor<double> W_temp4_lvl0_part0B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_temp4_lvl0_part0T[D0,D1,D2,D3]
DistTensor<double> W_temp4_lvl0_part0T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_temp4_lvl0_part1B[D0,D1,D2,D3]
DistTensor<double> W_temp4_lvl0_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_temp4_lvl0_part1T[D0,D1,D2,D3]
DistTensor<double> W_temp4_lvl0_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_temp4_lvl1_part0B[D0,D1,D2,D3]
DistTensor<double> W_temp4_lvl1_part0B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_temp4_lvl1_part0T[D0,D1,D2,D3]
DistTensor<double> W_temp4_lvl1_part0T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_temp4_lvl1_part0_0[D0,D1,D2,D3]
DistTensor<double> W_temp4_lvl1_part0_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_temp4_lvl1_part0_1[D0,D1,D2,D3]
DistTensor<double> W_temp4_lvl1_part0_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_temp4_lvl1_part0_1_lvl1_part1B[D0,D1,D2,D3]
DistTensor<double> W_temp4_lvl1_part0_1_lvl1_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_temp4_lvl1_part0_1_lvl1_part1T[D0,D1,D2,D3]
DistTensor<double> W_temp4_lvl1_part0_1_lvl1_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_temp4_lvl1_part0_1_lvl2_part1B[D0,D1,D2,D3]
DistTensor<double> W_temp4_lvl1_part0_1_lvl2_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_temp4_lvl1_part0_1_lvl2_part1T[D0,D1,D2,D3]
DistTensor<double> W_temp4_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_temp4_lvl1_part0_1_lvl2_part1_0[D0,D1,D2,D3]
DistTensor<double> W_temp4_lvl1_part0_1_lvl2_part1_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_temp4_lvl1_part0_1_lvl2_part1_1[D0,D1,D2,D3]
DistTensor<double> W_temp4_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_temp4_lvl1_part0_1_lvl2_part1_2[D0,D1,D2,D3]
DistTensor<double> W_temp4_lvl1_part0_1_lvl2_part1_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_temp4_lvl1_part0_2[D0,D1,D2,D3]
DistTensor<double> W_temp4_lvl1_part0_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_temp4_lvl1_part1B[D0,D1,D2,D3]
DistTensor<double> W_temp4_lvl1_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_temp4_lvl1_part1T[D0,D1,D2,D3]
DistTensor<double> W_temp4_lvl1_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_temp4_lvl1_part1_0[D0,D1,D2,D3]
DistTensor<double> W_temp4_lvl1_part1_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_temp4_lvl1_part1_1[D0,D1,D2,D3]
DistTensor<double> W_temp4_lvl1_part1_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_temp4_lvl1_part1_2[D0,D1,D2,D3]
DistTensor<double> W_temp4_lvl1_part1_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
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
ObjShape t_fj__D_0_1__D_2_3_tmpShape_W( 2 );
t_fj__D_0_1__D_2_3_tmpShape_W[ 0 ] = n_v;
t_fj__D_0_1__D_2_3_tmpShape_W[ 1 ] = n_o;
t_fj__D_0_1__D_2_3.ResizeTo( t_fj__D_0_1__D_2_3_tmpShape_W );
// r_bmfe has 4 dims
ObjShape r_bmfe__D_0__D_1__D_2__D_3_tmpShape_W( 4 );
r_bmfe__D_0__D_1__D_2__D_3_tmpShape_W[ 0 ] = n_v;
r_bmfe__D_0__D_1__D_2__D_3_tmpShape_W[ 1 ] = n_o;
r_bmfe__D_0__D_1__D_2__D_3_tmpShape_W[ 2 ] = n_v;
r_bmfe__D_0__D_1__D_2__D_3_tmpShape_W[ 3 ] = n_v;
r_bmfe__D_0__D_1__D_2__D_3.ResizeTo( r_bmfe__D_0__D_1__D_2__D_3_tmpShape_W );
MakeUniform( r_bmfe__D_0__D_1__D_2__D_3 );
// u_mnje has 4 dims
ObjShape u_mnje__D_0__D_1__D_2__D_3_tmpShape_W( 4 );
u_mnje__D_0__D_1__D_2__D_3_tmpShape_W[ 0 ] = n_o;
u_mnje__D_0__D_1__D_2__D_3_tmpShape_W[ 1 ] = n_o;
u_mnje__D_0__D_1__D_2__D_3_tmpShape_W[ 2 ] = n_o;
u_mnje__D_0__D_1__D_2__D_3_tmpShape_W[ 3 ] = n_v;
u_mnje__D_0__D_1__D_2__D_3.ResizeTo( u_mnje__D_0__D_1__D_2__D_3_tmpShape_W );
MakeUniform( u_mnje__D_0__D_1__D_2__D_3 );
// v_femn has 4 dims
ObjShape v_femn__D_0__D_1__D_2__D_3_tmpShape_W( 4 );
v_femn__D_0__D_1__D_2__D_3_tmpShape_W[ 0 ] = n_v;
v_femn__D_0__D_1__D_2__D_3_tmpShape_W[ 1 ] = n_v;
v_femn__D_0__D_1__D_2__D_3_tmpShape_W[ 2 ] = n_o;
v_femn__D_0__D_1__D_2__D_3_tmpShape_W[ 3 ] = n_o;
v_femn__D_0__D_1__D_2__D_3.ResizeTo( v_femn__D_0__D_1__D_2__D_3_tmpShape_W );
MakeUniform( v_femn__D_0__D_1__D_2__D_3 );
ObjShape T_bfnj__D_0__D_1__D_2__D_3_tmpShape_W( 4 );
T_bfnj__D_0__D_1__D_2__D_3_tmpShape_W[ 0 ] = n_v;
T_bfnj__D_0__D_1__D_2__D_3_tmpShape_W[ 1 ] = n_v;
T_bfnj__D_0__D_1__D_2__D_3_tmpShape_W[ 2 ] = n_o;
T_bfnj__D_0__D_1__D_2__D_3_tmpShape_W[ 3 ] = n_o;
T_bfnj__D_0__D_1__D_2__D_3.ResizeTo( T_bfnj__D_0__D_1__D_2__D_3_tmpShape_W );
ObjShape Tau_efmn__D_0__D_1__D_2__D_3_tmpShape_W( 4 );
Tau_efmn__D_0__D_1__D_2__D_3_tmpShape_W[ 0 ] = n_v;
Tau_efmn__D_0__D_1__D_2__D_3_tmpShape_W[ 1 ] = n_v;
Tau_efmn__D_0__D_1__D_2__D_3_tmpShape_W[ 2 ] = n_o;
Tau_efmn__D_0__D_1__D_2__D_3_tmpShape_W[ 3 ] = n_o;
Tau_efmn__D_0__D_1__D_2__D_3.ResizeTo( Tau_efmn__D_0__D_1__D_2__D_3_tmpShape_W );
// w_bmje has 4 dims
ObjShape w_bmje__D_0__D_1__D_2__D_3_tmpShape_W( 4 );
w_bmje__D_0__D_1__D_2__D_3_tmpShape_W[ 0 ] = n_v;
w_bmje__D_0__D_1__D_2__D_3_tmpShape_W[ 1 ] = n_o;
w_bmje__D_0__D_1__D_2__D_3_tmpShape_W[ 2 ] = n_o;
w_bmje__D_0__D_1__D_2__D_3_tmpShape_W[ 3 ] = n_v;
w_bmje__D_0__D_1__D_2__D_3.ResizeTo( w_bmje__D_0__D_1__D_2__D_3_tmpShape_W );
MakeUniform( w_bmje__D_0__D_1__D_2__D_3 );
// x_bmej has 4 dims
ObjShape x_bmej__D_0__D_1__D_2__D_3_tmpShape_W( 4 );
x_bmej__D_0__D_1__D_2__D_3_tmpShape_W[ 0 ] = n_v;
x_bmej__D_0__D_1__D_2__D_3_tmpShape_W[ 1 ] = n_o;
x_bmej__D_0__D_1__D_2__D_3_tmpShape_W[ 2 ] = n_v;
x_bmej__D_0__D_1__D_2__D_3_tmpShape_W[ 3 ] = n_o;
x_bmej__D_0__D_1__D_2__D_3.ResizeTo( x_bmej__D_0__D_1__D_2__D_3_tmpShape_W );
MakeUniform( x_bmej__D_0__D_1__D_2__D_3 );
// W_bmje has 4 dims
ObjShape W_bmje__D_0__D_1__D_2__D_3_tmpShape_W( 4 );
W_bmje__D_0__D_1__D_2__D_3_tmpShape_W[ 0 ] = n_v;
W_bmje__D_0__D_1__D_2__D_3_tmpShape_W[ 1 ] = n_o;
W_bmje__D_0__D_1__D_2__D_3_tmpShape_W[ 2 ] = n_o;
W_bmje__D_0__D_1__D_2__D_3_tmpShape_W[ 3 ] = n_v;
W_bmje__D_0__D_1__D_2__D_3.ResizeTo( W_bmje__D_0__D_1__D_2__D_3_tmpShape_W );
MakeUniform( W_bmje__D_0__D_1__D_2__D_3 );
ObjShape overwrite_tmpShape_X;
TensorDistribution dist__S__D_2__D_1__S = tmen::StringToTensorDist("[(),(2),(1),()]");
TensorDistribution dist__D_0__S__S__D_3 = tmen::StringToTensorDist("[(0),(),(),(3)]");
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
Permutation perm_2_1_0_3( 4 );
perm_2_1_0_3[0] = 2;
perm_2_1_0_3[1] = 1;
perm_2_1_0_3[2] = 0;
perm_2_1_0_3[3] = 3;
Permutation perm_3_0_2_1( 4 );
perm_3_0_2_1[0] = 3;
perm_3_0_2_1[1] = 0;
perm_3_0_2_1[2] = 2;
perm_3_0_2_1[3] = 1;
ModeArray modes_1_3_2( 3 );
modes_1_3_2[0] = 1;
modes_1_3_2[1] = 3;
modes_1_3_2[2] = 2;
ModeArray modes_2_1( 2 );
modes_2_1[0] = 2;
modes_2_1[1] = 1;
	//X_bmej[D0,D1,D2,D3]
DistTensor<double> X_bmej__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//X_bmej_lvl0_part1B[D0,D1,D2,D3]
DistTensor<double> X_bmej_lvl0_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//X_bmej_lvl0_part1T[D0,D1,D2,D3]
DistTensor<double> X_bmej_lvl0_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
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
DistTensor<double> X_bmej_perm2103__D_2__D_1__D_0__D_3( dist__D_0__D_1__D_2__D_3, g );
X_bmej_perm2103__D_2__D_1__D_0__D_3.SetLocalPermutation( perm_2_1_0_3 );
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
	//X_temp1[D0,D1,D2,D3]
DistTensor<double> X_temp1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//X_temp1_lvl0_part1B[D0,D1,D2,D3]
DistTensor<double> X_temp1_lvl0_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//X_temp1_lvl0_part1T[D0,D1,D2,D3]
DistTensor<double> X_temp1_lvl0_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//X_temp1_lvl1_part1B[D0,D1,D2,D3]
DistTensor<double> X_temp1_lvl1_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//X_temp1_lvl1_part1T[D0,D1,D2,D3]
DistTensor<double> X_temp1_lvl1_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//X_temp1_lvl1_part1_0[D0,D1,D2,D3]
DistTensor<double> X_temp1_lvl1_part1_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//X_temp1_lvl1_part1_1[D0,D1,D2,D3]
DistTensor<double> X_temp1_lvl1_part1_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//X_temp1_lvl1_part1_1_lvl1_part2B[D0,D1,D2,D3]
DistTensor<double> X_temp1_lvl1_part1_1_lvl1_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//X_temp1_lvl1_part1_1_lvl1_part2T[D0,D1,D2,D3]
DistTensor<double> X_temp1_lvl1_part1_1_lvl1_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//X_temp1_lvl1_part1_1_lvl2_part2B[D0,D1,D2,D3]
DistTensor<double> X_temp1_lvl1_part1_1_lvl2_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//X_temp1_lvl1_part1_1_lvl2_part2T[D0,D1,D2,D3]
DistTensor<double> X_temp1_lvl1_part1_1_lvl2_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//X_temp1_lvl1_part1_1_lvl2_part2_0[D0,D1,D2,D3]
DistTensor<double> X_temp1_lvl1_part1_1_lvl2_part2_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//X_temp1_lvl1_part1_1_lvl2_part2_1[D0,D1,D2,D3]
DistTensor<double> X_temp1_lvl1_part1_1_lvl2_part2_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//X_temp1_lvl1_part1_1_lvl2_part2_1[D0,*,*,D3]
DistTensor<double> X_temp1_lvl1_part1_1_lvl2_part2_1_perm1203__S__S__D_0__D_3( dist__D_0__S__S__D_3, g );
X_temp1_lvl1_part1_1_lvl2_part2_1_perm1203__S__S__D_0__D_3.SetLocalPermutation( perm_1_2_0_3 );
	//X_temp1_lvl1_part1_1_lvl2_part2_2[D0,D1,D2,D3]
DistTensor<double> X_temp1_lvl1_part1_1_lvl2_part2_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//X_temp1_lvl1_part1_2[D0,D1,D2,D3]
DistTensor<double> X_temp1_lvl1_part1_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//u_mnje_lvl1_part0_1_lvl2_part1_1[D0,D1,D3,D2]
DistTensor<double> u_mnje_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_3__D_2( dist__D_0__D_1__D_3__D_2, g );
	//u_mnje_lvl1_part0_1_lvl2_part1_1[D1,*,D3,D2]
DistTensor<double> u_mnje_lvl1_part0_1_lvl2_part1_1_perm0231__D_1__D_3__D_2__S( dist__D_1__S__D_3__D_2, g );
u_mnje_lvl1_part0_1_lvl2_part1_1_perm0231__D_1__D_3__D_2__S.SetLocalPermutation( perm_0_2_3_1 );
	//v_femn_lvl0_part0B[D0,D1,D2,D3]
DistTensor<double> v_femn_lvl0_part0B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//v_femn_lvl0_part0T[D0,D1,D2,D3]
DistTensor<double> v_femn_lvl0_part0T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//v_femn_lvl1_part0B[D0,D1,D2,D3]
DistTensor<double> v_femn_lvl1_part0B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//v_femn_lvl1_part0T[D0,D1,D2,D3]
DistTensor<double> v_femn_lvl1_part0T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//v_femn_lvl1_part0_0[D0,D1,D2,D3]
DistTensor<double> v_femn_lvl1_part0_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//v_femn_lvl1_part0_1[D0,D1,D2,D3]
DistTensor<double> v_femn_lvl1_part0_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//v_femn_lvl1_part0_1_lvl1_part3B[D0,D1,D2,D3]
DistTensor<double> v_femn_lvl1_part0_1_lvl1_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//v_femn_lvl1_part0_1_lvl1_part3T[D0,D1,D2,D3]
DistTensor<double> v_femn_lvl1_part0_1_lvl1_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//v_femn_lvl1_part0_1_lvl2_part3B[D0,D1,D2,D3]
DistTensor<double> v_femn_lvl1_part0_1_lvl2_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//v_femn_lvl1_part0_1_lvl2_part3T[D0,D1,D2,D3]
DistTensor<double> v_femn_lvl1_part0_1_lvl2_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//v_femn_lvl1_part0_1_lvl2_part3_0[D0,D1,D2,D3]
DistTensor<double> v_femn_lvl1_part0_1_lvl2_part3_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//v_femn_lvl1_part0_1_lvl2_part3_1[D0,D1,D2,D3]
DistTensor<double> v_femn_lvl1_part0_1_lvl2_part3_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//v_femn_lvl1_part0_1_lvl2_part3_1[D0,D2,D1,D3]
DistTensor<double> v_femn_lvl1_part0_1_lvl2_part3_1__D_0__D_2__D_1__D_3( dist__D_0__D_2__D_1__D_3, g );
	//v_femn_lvl1_part0_1_lvl2_part3_1[*,D2,D1,*]
DistTensor<double> v_femn_lvl1_part0_1_lvl2_part3_1_perm1203__D_2__D_1__S__S( dist__S__D_2__D_1__S, g );
v_femn_lvl1_part0_1_lvl2_part3_1_perm1203__D_2__D_1__S__S.SetLocalPermutation( perm_1_2_0_3 );
	//v_femn_lvl1_part0_1_lvl2_part3_2[D0,D1,D2,D3]
DistTensor<double> v_femn_lvl1_part0_1_lvl2_part3_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//v_femn_lvl1_part0_2[D0,D1,D2,D3]
DistTensor<double> v_femn_lvl1_part0_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
ObjShape r_bmfe__D_0__D_1__D_2__D_3_tmpShape_X( 4 );
r_bmfe__D_0__D_1__D_2__D_3_tmpShape_X[ 0 ] = n_v;
r_bmfe__D_0__D_1__D_2__D_3_tmpShape_X[ 1 ] = n_o;
r_bmfe__D_0__D_1__D_2__D_3_tmpShape_X[ 2 ] = n_v;
r_bmfe__D_0__D_1__D_2__D_3_tmpShape_X[ 3 ] = n_v;
r_bmfe__D_0__D_1__D_2__D_3.ResizeTo( r_bmfe__D_0__D_1__D_2__D_3_tmpShape_X );
ObjShape t_fj__D_0_1__D_2_3_tmpShape_X( 2 );
t_fj__D_0_1__D_2_3_tmpShape_X[ 0 ] = n_v;
t_fj__D_0_1__D_2_3_tmpShape_X[ 1 ] = n_o;
t_fj__D_0_1__D_2_3.ResizeTo( t_fj__D_0_1__D_2_3_tmpShape_X );
ObjShape u_mnje__D_0__D_1__D_2__D_3_tmpShape_X( 4 );
u_mnje__D_0__D_1__D_2__D_3_tmpShape_X[ 0 ] = n_o;
u_mnje__D_0__D_1__D_2__D_3_tmpShape_X[ 1 ] = n_o;
u_mnje__D_0__D_1__D_2__D_3_tmpShape_X[ 2 ] = n_o;
u_mnje__D_0__D_1__D_2__D_3_tmpShape_X[ 3 ] = n_v;
u_mnje__D_0__D_1__D_2__D_3.ResizeTo( u_mnje__D_0__D_1__D_2__D_3_tmpShape_X );
ObjShape v_femn__D_0__D_1__D_2__D_3_tmpShape_X( 4 );
v_femn__D_0__D_1__D_2__D_3_tmpShape_X[ 0 ] = n_v;
v_femn__D_0__D_1__D_2__D_3_tmpShape_X[ 1 ] = n_v;
v_femn__D_0__D_1__D_2__D_3_tmpShape_X[ 2 ] = n_o;
v_femn__D_0__D_1__D_2__D_3_tmpShape_X[ 3 ] = n_o;
v_femn__D_0__D_1__D_2__D_3.ResizeTo( v_femn__D_0__D_1__D_2__D_3_tmpShape_X );
ObjShape Tau_efmn__D_0__D_1__D_2__D_3_tmpShape_X( 4 );
Tau_efmn__D_0__D_1__D_2__D_3_tmpShape_X[ 0 ] = n_v;
Tau_efmn__D_0__D_1__D_2__D_3_tmpShape_X[ 1 ] = n_v;
Tau_efmn__D_0__D_1__D_2__D_3_tmpShape_X[ 2 ] = n_o;
Tau_efmn__D_0__D_1__D_2__D_3_tmpShape_X[ 3 ] = n_o;
Tau_efmn__D_0__D_1__D_2__D_3.ResizeTo( Tau_efmn__D_0__D_1__D_2__D_3_tmpShape_X );
ObjShape T_bfnj__D_0__D_1__D_2__D_3_tmpShape_X( 4 );
T_bfnj__D_0__D_1__D_2__D_3_tmpShape_X[ 0 ] = n_v;
T_bfnj__D_0__D_1__D_2__D_3_tmpShape_X[ 1 ] = n_v;
T_bfnj__D_0__D_1__D_2__D_3_tmpShape_X[ 2 ] = n_o;
T_bfnj__D_0__D_1__D_2__D_3_tmpShape_X[ 3 ] = n_o;
T_bfnj__D_0__D_1__D_2__D_3.ResizeTo( T_bfnj__D_0__D_1__D_2__D_3_tmpShape_X );
ObjShape x_bmej__D_0__D_1__D_2__D_3_tmpShape_X( 4 );
x_bmej__D_0__D_1__D_2__D_3_tmpShape_X[ 0 ] = n_v;
x_bmej__D_0__D_1__D_2__D_3_tmpShape_X[ 1 ] = n_o;
x_bmej__D_0__D_1__D_2__D_3_tmpShape_X[ 2 ] = n_v;
x_bmej__D_0__D_1__D_2__D_3_tmpShape_X[ 3 ] = n_o;
x_bmej__D_0__D_1__D_2__D_3.ResizeTo( x_bmej__D_0__D_1__D_2__D_3_tmpShape_X );
// X_bmej has 4 dims
ObjShape X_bmej__D_0__D_1__D_2__D_3_tmpShape_X( 4 );
X_bmej__D_0__D_1__D_2__D_3_tmpShape_X[ 0 ] = n_v;
X_bmej__D_0__D_1__D_2__D_3_tmpShape_X[ 1 ] = n_o;
X_bmej__D_0__D_1__D_2__D_3_tmpShape_X[ 2 ] = n_v;
X_bmej__D_0__D_1__D_2__D_3_tmpShape_X[ 3 ] = n_o;
X_bmej__D_0__D_1__D_2__D_3.ResizeTo( X_bmej__D_0__D_1__D_2__D_3_tmpShape_X );
MakeUniform( X_bmej__D_0__D_1__D_2__D_3 );
ObjShape overwrite_tmpShape_U;
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
	//U_mnie_lvl0_part1B[D0,D1,D2,D3]
DistTensor<double> U_mnie_lvl0_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//U_mnie_lvl0_part1T[D0,D1,D2,D3]
DistTensor<double> U_mnie_lvl0_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//U_mnie_lvl1_part1B[D0,D1,D2,D3]
DistTensor<double> U_mnie_lvl1_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//U_mnie_lvl1_part1T[D0,D1,D2,D3]
DistTensor<double> U_mnie_lvl1_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//U_mnie_lvl1_part1_0[D0,D1,D2,D3]
DistTensor<double> U_mnie_lvl1_part1_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//U_mnie_lvl1_part1_1[D0,D1,D2,D3]
DistTensor<double> U_mnie_lvl1_part1_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//U_mnie_lvl1_part1_1_lvl1_part2B[D0,D1,D2,D3]
DistTensor<double> U_mnie_lvl1_part1_1_lvl1_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//U_mnie_lvl1_part1_1_lvl1_part2T[D0,D1,D2,D3]
DistTensor<double> U_mnie_lvl1_part1_1_lvl1_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//U_mnie_lvl1_part1_1_lvl2_part2B[D0,D1,D2,D3]
DistTensor<double> U_mnie_lvl1_part1_1_lvl2_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//U_mnie_lvl1_part1_1_lvl2_part2T[D0,D1,D2,D3]
DistTensor<double> U_mnie_lvl1_part1_1_lvl2_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//U_mnie_lvl1_part1_1_lvl2_part2_0[D0,D1,D2,D3]
DistTensor<double> U_mnie_lvl1_part1_1_lvl2_part2_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//U_mnie_lvl1_part1_1_lvl2_part2_1[D0,D1,D2,D3]
DistTensor<double> U_mnie_lvl1_part1_1_lvl2_part2_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//U_mnie_lvl1_part1_1_lvl2_part2_1[D2,D3,*,D1,D0]
DistTensor<double> U_mnie_lvl1_part1_1_lvl2_part2_1_perm30124__D_1__D_2__D_3__S__D_0( dist__D_2__D_3__S__D_1__D_0, g );
U_mnie_lvl1_part1_1_lvl2_part2_1_perm30124__D_1__D_2__D_3__S__D_0.SetLocalPermutation( perm_3_0_1_2_4 );
	//U_mnie_lvl1_part1_1_lvl2_part2_1_U_temp[D0,D1,D2,D3]
DistTensor<double> U_mnie_lvl1_part1_1_lvl2_part2_1_U_temp__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//U_mnie_lvl1_part1_1_lvl2_part2_1_U_temp[D20,D1,*,D3]
DistTensor<double> U_mnie_lvl1_part1_1_lvl2_part2_1_U_temp__D_2_0__D_1__S__D_3( dist__D_2_0__D_1__S__D_3, g );
	//U_mnie_lvl1_part1_1_lvl2_part2_1_U_temp[D20,D3,*,D1]
DistTensor<double> U_mnie_lvl1_part1_1_lvl2_part2_1_U_temp__D_2_0__D_3__S__D_1( dist__D_2_0__D_3__S__D_1, g );
	//U_mnie_lvl1_part1_1_lvl2_part2_2[D0,D1,D2,D3]
DistTensor<double> U_mnie_lvl1_part1_1_lvl2_part2_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//U_mnie_lvl1_part1_2[D0,D1,D2,D3]
DistTensor<double> U_mnie_lvl1_part1_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> t_fj_lvl2_part1_1__D_0__S( dist__D_0__S, g );
DistTensor<double> v_femn_lvl1_part3_1_perm1230__D_1__D_2__D_3__D_0( dist__D_0__D_1__D_2__D_3, g );
v_femn_lvl1_part3_1_perm1230__D_1__D_2__D_3__D_0.SetLocalPermutation( perm_1_2_3_0 );
ObjShape v_femn__D_0__D_1__D_2__D_3_tmpShape_U( 4 );
v_femn__D_0__D_1__D_2__D_3_tmpShape_U[ 0 ] = n_v;
v_femn__D_0__D_1__D_2__D_3_tmpShape_U[ 1 ] = n_v;
v_femn__D_0__D_1__D_2__D_3_tmpShape_U[ 2 ] = n_o;
v_femn__D_0__D_1__D_2__D_3_tmpShape_U[ 3 ] = n_o;
v_femn__D_0__D_1__D_2__D_3.ResizeTo( v_femn__D_0__D_1__D_2__D_3_tmpShape_U );
ObjShape t_fj__D_0_1__D_2_3_tmpShape_U( 2 );
t_fj__D_0_1__D_2_3_tmpShape_U[ 0 ] = n_v;
t_fj__D_0_1__D_2_3_tmpShape_U[ 1 ] = n_o;
t_fj__D_0_1__D_2_3.ResizeTo( t_fj__D_0_1__D_2_3_tmpShape_U );
ObjShape u_mnje__D_0__D_1__D_2__D_3_tmpShape_U( 4 );
u_mnje__D_0__D_1__D_2__D_3_tmpShape_U[ 0 ] = n_o;
u_mnje__D_0__D_1__D_2__D_3_tmpShape_U[ 1 ] = n_o;
u_mnje__D_0__D_1__D_2__D_3_tmpShape_U[ 2 ] = n_o;
u_mnje__D_0__D_1__D_2__D_3_tmpShape_U[ 3 ] = n_v;
u_mnje__D_0__D_1__D_2__D_3.ResizeTo( u_mnje__D_0__D_1__D_2__D_3_tmpShape_U );
// U_mnie has 4 dims
ObjShape U_mnie__D_0__D_1__D_2__D_3_tmpShape_U( 4 );
U_mnie__D_0__D_1__D_2__D_3_tmpShape_U[ 0 ] = n_o;
U_mnie__D_0__D_1__D_2__D_3_tmpShape_U[ 1 ] = n_o;
U_mnie__D_0__D_1__D_2__D_3_tmpShape_U[ 2 ] = n_o;
U_mnie__D_0__D_1__D_2__D_3_tmpShape_U[ 3 ] = n_v;
U_mnie__D_0__D_1__D_2__D_3.ResizeTo( U_mnie__D_0__D_1__D_2__D_3_tmpShape_U );
MakeUniform( U_mnie__D_0__D_1__D_2__D_3 );
ObjShape overwrite_tmpShape_Q;
TensorDistribution dist__S__S__D_2__D_3__D_0__D_1 = tmen::StringToTensorDist("[(),(),(2),(3),(0),(1)]");
TensorDistribution dist__S__D_3 = tmen::StringToTensorDist("[(),(3)]");
TensorDistribution dist__D_0__D_1__S__S = tmen::StringToTensorDist("[(0),(1),(),()]");
TensorDistribution dist__D_0__D_1__D_2__S = tmen::StringToTensorDist("[(0),(1),(2),()]");
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
ModeArray modes_5_4( 2 );
modes_5_4[0] = 5;
modes_5_4[1] = 4;
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
IndexArray indices_mnij( 4 );
indices_mnij[0] = 'm';
indices_mnij[1] = 'n';
indices_mnij[2] = 'i';
indices_mnij[3] = 'j';
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
	//Q_mnij_lvl1_part0_1_lvl1_part2B[D0,D1,D2,D3]
DistTensor<double> Q_mnij_lvl1_part0_1_lvl1_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Q_mnij_lvl1_part0_1_lvl1_part2T[D0,D1,D2,D3]
DistTensor<double> Q_mnij_lvl1_part0_1_lvl1_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
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
	//Q_mnij_lvl1_part0_1_lvl2_part2B[D0,D1,D2,D3]
DistTensor<double> Q_mnij_lvl1_part0_1_lvl2_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Q_mnij_lvl1_part0_1_lvl2_part2T[D0,D1,D2,D3]
DistTensor<double> Q_mnij_lvl1_part0_1_lvl2_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Q_mnij_lvl1_part0_1_lvl2_part2_0[D0,D1,D2,D3]
DistTensor<double> Q_mnij_lvl1_part0_1_lvl2_part2_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Q_mnij_lvl1_part0_1_lvl2_part2_1[D0,D1,D2,D3]
DistTensor<double> Q_mnij_lvl1_part0_1_lvl2_part2_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Q_mnij_lvl1_part0_1_lvl2_part2_2[D0,D1,D2,D3]
DistTensor<double> Q_mnij_lvl1_part0_1_lvl2_part2_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Q_mnij_lvl1_part0_2[D0,D1,D2,D3]
DistTensor<double> Q_mnij_lvl1_part0_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//q_mnij[D0,D1,D2,D3]
DistTensor<double> q_mnij__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//t_fj[D01,D3]
DistTensor<double> t_fj__D_0_1__D_3( dist__D_0_1__D_3, g );
	//t_fj[*,D3]
DistTensor<double> t_fj__S__D_3( dist__S__D_3, g );
	//Q_temp1[D0,D1,D2,D3]
DistTensor<double> Q_temp1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Q_temp1_lvl0_part0B[D0,D1,D2,D3]
DistTensor<double> Q_temp1_lvl0_part0B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Q_temp1_lvl0_part0T[D0,D1,D2,D3]
DistTensor<double> Q_temp1_lvl0_part0T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Q_temp1_lvl0_part1B[D0,D1,D2,D3]
DistTensor<double> Q_temp1_lvl0_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Q_temp1_lvl0_part1T[D0,D1,D2,D3]
DistTensor<double> Q_temp1_lvl0_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Q_temp1_lvl1_part0B[D0,D1,D2,D3]
DistTensor<double> Q_temp1_lvl1_part0B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Q_temp1_lvl1_part0T[D0,D1,D2,D3]
DistTensor<double> Q_temp1_lvl1_part0T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Q_temp1_lvl1_part0_0[D0,D1,D2,D3]
DistTensor<double> Q_temp1_lvl1_part0_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Q_temp1_lvl1_part0_1[D0,D1,D2,D3]
DistTensor<double> Q_temp1_lvl1_part0_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Q_temp1_lvl1_part0_1_lvl1_part1B[D0,D1,D2,D3]
DistTensor<double> Q_temp1_lvl1_part0_1_lvl1_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Q_temp1_lvl1_part0_1_lvl1_part1T[D0,D1,D2,D3]
DistTensor<double> Q_temp1_lvl1_part0_1_lvl1_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Q_temp1_lvl1_part0_1_lvl1_part2B[D0,D1,D2,D3]
DistTensor<double> Q_temp1_lvl1_part0_1_lvl1_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Q_temp1_lvl1_part0_1_lvl1_part2T[D0,D1,D2,D3]
DistTensor<double> Q_temp1_lvl1_part0_1_lvl1_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Q_temp1_lvl1_part0_1_lvl2_part1B[D0,D1,D2,D3]
DistTensor<double> Q_temp1_lvl1_part0_1_lvl2_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Q_temp1_lvl1_part0_1_lvl2_part1T[D0,D1,D2,D3]
DistTensor<double> Q_temp1_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Q_temp1_lvl1_part0_1_lvl2_part1_0[D0,D1,D2,D3]
DistTensor<double> Q_temp1_lvl1_part0_1_lvl2_part1_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Q_temp1_lvl1_part0_1_lvl2_part1_1[D0,D1,D2,D3]
DistTensor<double> Q_temp1_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Q_temp1_lvl1_part0_1_lvl2_part1_2[D0,D1,D2,D3]
DistTensor<double> Q_temp1_lvl1_part0_1_lvl2_part1_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Q_temp1_lvl1_part0_1_lvl2_part2B[D0,D1,D2,D3]
DistTensor<double> Q_temp1_lvl1_part0_1_lvl2_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Q_temp1_lvl1_part0_1_lvl2_part2T[D0,D1,D2,D3]
DistTensor<double> Q_temp1_lvl1_part0_1_lvl2_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Q_temp1_lvl1_part0_1_lvl2_part2_0[D0,D1,D2,D3]
DistTensor<double> Q_temp1_lvl1_part0_1_lvl2_part2_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Q_temp1_lvl1_part0_1_lvl2_part2_1[D0,D1,D2,D3]
DistTensor<double> Q_temp1_lvl1_part0_1_lvl2_part2_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Q_temp1_lvl1_part0_1_lvl2_part2_2[D0,D1,D2,D3]
DistTensor<double> Q_temp1_lvl1_part0_1_lvl2_part2_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Q_temp1_lvl1_part0_2[D0,D1,D2,D3]
DistTensor<double> Q_temp1_lvl1_part0_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Q_temp1_lvl1_part1B[D0,D1,D2,D3]
DistTensor<double> Q_temp1_lvl1_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Q_temp1_lvl1_part1T[D0,D1,D2,D3]
DistTensor<double> Q_temp1_lvl1_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Q_temp1_lvl1_part1_0[D0,D1,D2,D3]
DistTensor<double> Q_temp1_lvl1_part1_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Q_temp1_lvl1_part1_1[D0,D1,D2,D3]
DistTensor<double> Q_temp1_lvl1_part1_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Q_temp1_lvl1_part1_1_lvl1_part3B[D0,D1,D2,D3]
DistTensor<double> Q_temp1_lvl1_part1_1_lvl1_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Q_temp1_lvl1_part1_1_lvl1_part3T[D0,D1,D2,D3]
DistTensor<double> Q_temp1_lvl1_part1_1_lvl1_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Q_temp1_lvl1_part1_1_lvl2_part3B[D0,D1,D2,D3]
DistTensor<double> Q_temp1_lvl1_part1_1_lvl2_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Q_temp1_lvl1_part1_1_lvl2_part3T[D0,D1,D2,D3]
DistTensor<double> Q_temp1_lvl1_part1_1_lvl2_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Q_temp1_lvl1_part1_1_lvl2_part3_0[D0,D1,D2,D3]
DistTensor<double> Q_temp1_lvl1_part1_1_lvl2_part3_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Q_temp1_lvl1_part1_1_lvl2_part3_1[D0,D1,D2,D3]
DistTensor<double> Q_temp1_lvl1_part1_1_lvl2_part3_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Q_temp1_lvl1_part1_1_lvl2_part3_1[D1,D0,D2,D3]
DistTensor<double> Q_temp1_lvl1_part1_1_lvl2_part3_1__D_1__D_0__D_2__D_3( dist__D_1__D_0__D_2__D_3, g );
	//Q_temp1_lvl1_part1_1_lvl2_part3_1[D1,D0,D3,D2]
DistTensor<double> Q_temp1_lvl1_part1_1_lvl2_part3_1__D_1__D_0__D_3__D_2( dist__D_1__D_0__D_3__D_2, g );
	//Q_temp1_lvl1_part1_1_lvl2_part3_2[D0,D1,D2,D3]
DistTensor<double> Q_temp1_lvl1_part1_1_lvl2_part3_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Q_temp1_lvl1_part1_2[D0,D1,D2,D3]
DistTensor<double> Q_temp1_lvl1_part1_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//u_mnje_lvl1_part0_1_lvl2_part1_1[D0,D1,D2,*]
DistTensor<double> u_mnje_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__S( dist__D_0__D_1__D_2__S, g );
	//v_femn_lvl1_part2_1_lvl2_part3_1[D0,D1,*,*]
DistTensor<double> v_femn_lvl1_part2_1_lvl2_part3_1_perm2301__S__S__D_0__D_1( dist__D_0__D_1__S__S, g );
v_femn_lvl1_part2_1_lvl2_part3_1_perm2301__S__S__D_0__D_1.SetLocalPermutation( perm_2_3_0_1 );
// q_mnij has 4 dims
ObjShape q_mnij__D_0__D_1__D_2__D_3_tmpShape_Q( 4 );
q_mnij__D_0__D_1__D_2__D_3_tmpShape_Q[ 0 ] = n_o;
q_mnij__D_0__D_1__D_2__D_3_tmpShape_Q[ 1 ] = n_o;
q_mnij__D_0__D_1__D_2__D_3_tmpShape_Q[ 2 ] = n_o;
q_mnij__D_0__D_1__D_2__D_3_tmpShape_Q[ 3 ] = n_o;
q_mnij__D_0__D_1__D_2__D_3.ResizeTo( q_mnij__D_0__D_1__D_2__D_3_tmpShape_Q );
MakeUniform( q_mnij__D_0__D_1__D_2__D_3 );
ObjShape v_femn__D_0__D_1__D_2__D_3_tmpShape_Q( 4 );
v_femn__D_0__D_1__D_2__D_3_tmpShape_Q[ 0 ] = n_v;
v_femn__D_0__D_1__D_2__D_3_tmpShape_Q[ 1 ] = n_v;
v_femn__D_0__D_1__D_2__D_3_tmpShape_Q[ 2 ] = n_o;
v_femn__D_0__D_1__D_2__D_3_tmpShape_Q[ 3 ] = n_o;
v_femn__D_0__D_1__D_2__D_3.ResizeTo( v_femn__D_0__D_1__D_2__D_3_tmpShape_Q );
ObjShape Tau_efmn__D_0__D_1__D_2__D_3_tmpShape_Q( 4 );
Tau_efmn__D_0__D_1__D_2__D_3_tmpShape_Q[ 0 ] = n_v;
Tau_efmn__D_0__D_1__D_2__D_3_tmpShape_Q[ 1 ] = n_v;
Tau_efmn__D_0__D_1__D_2__D_3_tmpShape_Q[ 2 ] = n_o;
Tau_efmn__D_0__D_1__D_2__D_3_tmpShape_Q[ 3 ] = n_o;
Tau_efmn__D_0__D_1__D_2__D_3.ResizeTo( Tau_efmn__D_0__D_1__D_2__D_3_tmpShape_Q );
// Q_mnij has 4 dims
ObjShape Q_mnij__D_0__D_1__D_2__D_3_tmpShape_Q( 4 );
Q_mnij__D_0__D_1__D_2__D_3_tmpShape_Q[ 0 ] = n_o;
Q_mnij__D_0__D_1__D_2__D_3_tmpShape_Q[ 1 ] = n_o;
Q_mnij__D_0__D_1__D_2__D_3_tmpShape_Q[ 2 ] = n_o;
Q_mnij__D_0__D_1__D_2__D_3_tmpShape_Q[ 3 ] = n_o;
Q_mnij__D_0__D_1__D_2__D_3.ResizeTo( Q_mnij__D_0__D_1__D_2__D_3_tmpShape_Q );
MakeUniform( Q_mnij__D_0__D_1__D_2__D_3 );
ObjShape u_mnje__D_0__D_1__D_2__D_3_tmpShape_Q( 4 );
u_mnje__D_0__D_1__D_2__D_3_tmpShape_Q[ 0 ] = n_o;
u_mnje__D_0__D_1__D_2__D_3_tmpShape_Q[ 1 ] = n_o;
u_mnje__D_0__D_1__D_2__D_3_tmpShape_Q[ 2 ] = n_o;
u_mnje__D_0__D_1__D_2__D_3_tmpShape_Q[ 3 ] = n_v;
u_mnje__D_0__D_1__D_2__D_3.ResizeTo( u_mnje__D_0__D_1__D_2__D_3_tmpShape_Q );
ObjShape t_fj__D_0_1__D_2_3_tmpShape_Q( 2 );
t_fj__D_0_1__D_2_3_tmpShape_Q[ 0 ] = n_v;
t_fj__D_0_1__D_2_3_tmpShape_Q[ 1 ] = n_o;
t_fj__D_0_1__D_2_3.ResizeTo( t_fj__D_0_1__D_2_3_tmpShape_Q );
ObjShape overwrite_tmpShape_P;
TensorDistribution dist__S__S__D_1_2__D_0_3 = tmen::StringToTensorDist("[(),(),(1,2),(0,3)]");
TensorDistribution dist__S__S__D_2_1__D_0_3 = tmen::StringToTensorDist("[(),(),(2,1),(0,3)]");
TensorDistribution dist__S__S__D_1__D_0__D_2__D_3 = tmen::StringToTensorDist("[(),(),(1),(0),(2),(3)]");
TensorDistribution dist__S__D_1__D_2__D_0_3 = tmen::StringToTensorDist("[(),(1),(2),(0,3)]");
TensorDistribution dist__S__D_2__D_1__D_0__D_3 = tmen::StringToTensorDist("[(),(2),(1),(0),(3)]");
TensorDistribution dist__S__D_2__D_1__D_0_3 = tmen::StringToTensorDist("[(),(2),(1),(0,3)]");
TensorDistribution dist__D_0__S__D_1_2__D_3 = tmen::StringToTensorDist("[(0),(),(1,2),(3)]");
TensorDistribution dist__D_2__S = tmen::StringToTensorDist("[(2),()]");
TensorDistribution dist__D_2__D_1__D_0__D_3 = tmen::StringToTensorDist("[(2),(1),(0),(3)]");
TensorDistribution dist__D_2__D_3__S__S = tmen::StringToTensorDist("[(2),(3),(),()]");
TensorDistribution dist__D_2__D_3__D_0__D_1 = tmen::StringToTensorDist("[(2),(3),(0),(1)]");
TensorDistribution dist__D_3__S__D_1_2__D_0 = tmen::StringToTensorDist("[(3),(),(1,2),(0)]");
TensorDistribution dist__D_3__S__D_1__D_0__D_2 = tmen::StringToTensorDist("[(3),(),(1),(0),(2)]");
TensorDistribution dist__D_0_1__S = tmen::StringToTensorDist("[(0,1),()]");
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
IndexArray indices_ei( 2 );
indices_ei[0] = 'e';
indices_ei[1] = 'i';
IndexArray indices_jimbe( 5 );
indices_jimbe[0] = 'j';
indices_jimbe[1] = 'i';
indices_jimbe[2] = 'm';
indices_jimbe[3] = 'b';
indices_jimbe[4] = 'e';
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
	//P_jimb_lvl1_part0_1_lvl2_part1B[D0,D1,D2,D3]
DistTensor<double> P_jimb_lvl1_part0_1_lvl2_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//P_jimb_lvl1_part0_1_lvl2_part1T[D0,D1,D2,D3]
DistTensor<double> P_jimb_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//P_jimb_lvl1_part0_1_lvl2_part1_0[D0,D1,D2,D3]
DistTensor<double> P_jimb_lvl1_part0_1_lvl2_part1_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//P_jimb_lvl1_part0_1_lvl2_part1_1[D0,D1,D2,D3]
DistTensor<double> P_jimb_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//P_jimb_lvl1_part0_1_lvl2_part1_1[D3,*,D1,D0,D2]
DistTensor<double> P_jimb_lvl1_part0_1_lvl2_part1_1__D_3__S__D_1__D_0__D_2( dist__D_3__S__D_1__D_0__D_2, g );
	//P_jimb_lvl1_part0_1_lvl2_part1_1[*,D2,D1,D0,D3]
DistTensor<double> P_jimb_lvl1_part0_1_lvl2_part1_1__S__D_2__D_1__D_0__D_3( dist__S__D_2__D_1__D_0__D_3, g );
	//P_jimb_lvl1_part0_1_lvl2_part1_1[*,*,D1,D0,D2,D3]
DistTensor<double> P_jimb_lvl1_part0_1_lvl2_part1_1_perm321045__D_0__D_1__S__S__D_2__D_3( dist__S__S__D_1__D_0__D_2__D_3, g );
P_jimb_lvl1_part0_1_lvl2_part1_1_perm321045__D_0__D_1__S__S__D_2__D_3.SetLocalPermutation( perm_3_2_1_0_4_5 );
	//P_jimb_lvl1_part0_1_lvl2_part1_1_P_temp[D0,D1,D2,D3]
DistTensor<double> P_jimb_lvl1_part0_1_lvl2_part1_1_P_temp__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//P_jimb_lvl1_part0_1_lvl2_part1_1_P_temp[D0,*,D12,D3]
DistTensor<double> P_jimb_lvl1_part0_1_lvl2_part1_1_P_temp__D_0__S__D_1_2__D_3( dist__D_0__S__D_1_2__D_3, g );
	//P_jimb_lvl1_part0_1_lvl2_part1_1_P_temp[D3,*,D12,D0]
DistTensor<double> P_jimb_lvl1_part0_1_lvl2_part1_1_P_temp__D_3__S__D_1_2__D_0( dist__D_3__S__D_1_2__D_0, g );
	//P_jimb_lvl1_part0_1_lvl2_part1_1_P_temp[*,D1,D2,D03]
DistTensor<double> P_jimb_lvl1_part0_1_lvl2_part1_1_P_temp__S__D_1__D_2__D_0_3( dist__S__D_1__D_2__D_0_3, g );
	//P_jimb_lvl1_part0_1_lvl2_part1_1_P_temp[*,D2,D1,D03]
DistTensor<double> P_jimb_lvl1_part0_1_lvl2_part1_1_P_temp__S__D_2__D_1__D_0_3( dist__S__D_2__D_1__D_0_3, g );
	//P_jimb_lvl1_part0_1_lvl2_part1_1_P_temp[*,*,D12,D03]
DistTensor<double> P_jimb_lvl1_part0_1_lvl2_part1_1_P_temp__S__S__D_1_2__D_0_3( dist__S__S__D_1_2__D_0_3, g );
	//P_jimb_lvl1_part0_1_lvl2_part1_1_P_temp[*,*,D21,D03]
DistTensor<double> P_jimb_lvl1_part0_1_lvl2_part1_1_P_temp__S__S__D_2_1__D_0_3( dist__S__S__D_2_1__D_0_3, g );
	//P_jimb_lvl1_part0_1_lvl2_part1_2[D0,D1,D2,D3]
DistTensor<double> P_jimb_lvl1_part0_1_lvl2_part1_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
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
	//Tau_efmn_lvl1_part3_1_lvl2_part2_1[D2,D3,D0,D1]
DistTensor<double> Tau_efmn_lvl1_part3_1_lvl2_part2_1__D_2__D_3__D_0__D_1( dist__D_2__D_3__D_0__D_1, g );
	//Tau_efmn_lvl1_part3_1_lvl2_part2_1[D2,D3,*,*]
DistTensor<double> Tau_efmn_lvl1_part3_1_lvl2_part2_1__D_2__D_3__S__S( dist__D_2__D_3__S__S, g );
	//Tau_efmn_lvl1_part3_1_lvl2_part2_2[D0,D1,D2,D3]
DistTensor<double> Tau_efmn_lvl1_part3_1_lvl2_part2_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Tau_efmn_lvl1_part3_2[D0,D1,D2,D3]
DistTensor<double> Tau_efmn_lvl1_part3_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//t_fj_lvl1_part1_1[D3,D2]
DistTensor<double> t_fj_lvl1_part1_1__D_3__D_2( dist__D_3__D_2, g );
	//t_fj_lvl1_part1_1[D3,*]
DistTensor<double> t_fj_lvl1_part1_1__D_3__S( dist__D_3__S, g );
	//t_fj_lvl2_part1_1[D01,*]
DistTensor<double> t_fj_lvl2_part1_1__D_0_1__S( dist__D_0_1__S, g );
	//t_fj_lvl2_part1_1[D2,*]
DistTensor<double> t_fj_lvl2_part1_1__D_2__S( dist__D_2__S, g );
	//w_bmje_lvl1_part2B[D0,D1,D2,D3]
DistTensor<double> w_bmje_lvl1_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//w_bmje_lvl1_part2T[D0,D1,D2,D3]
DistTensor<double> w_bmje_lvl1_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//w_bmje_lvl2_part2B[D0,D1,D2,D3]
DistTensor<double> w_bmje_lvl2_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//w_bmje_lvl2_part2T[D0,D1,D2,D3]
DistTensor<double> w_bmje_lvl2_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//w_bmje_lvl2_part2_0[D0,D1,D2,D3]
DistTensor<double> w_bmje_lvl2_part2_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//w_bmje_lvl2_part2_1[D0,D1,D2,D3]
DistTensor<double> w_bmje_lvl2_part2_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//w_bmje_lvl2_part2_2[D0,D1,D2,D3]
DistTensor<double> w_bmje_lvl2_part2_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
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
	//x_bmej_lvl1_part3_2[D0,D1,D2,D3]
DistTensor<double> x_bmej_lvl1_part3_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
ObjShape r_bmfe__D_0__D_1__D_2__D_3_tmpShape_P( 4 );
r_bmfe__D_0__D_1__D_2__D_3_tmpShape_P[ 0 ] = n_v;
r_bmfe__D_0__D_1__D_2__D_3_tmpShape_P[ 1 ] = n_o;
r_bmfe__D_0__D_1__D_2__D_3_tmpShape_P[ 2 ] = n_v;
r_bmfe__D_0__D_1__D_2__D_3_tmpShape_P[ 3 ] = n_v;
r_bmfe__D_0__D_1__D_2__D_3.ResizeTo( r_bmfe__D_0__D_1__D_2__D_3_tmpShape_P );
ObjShape Tau_efmn__D_0__D_1__D_2__D_3_tmpShape_P( 4 );
Tau_efmn__D_0__D_1__D_2__D_3_tmpShape_P[ 0 ] = n_v;
Tau_efmn__D_0__D_1__D_2__D_3_tmpShape_P[ 1 ] = n_v;
Tau_efmn__D_0__D_1__D_2__D_3_tmpShape_P[ 2 ] = n_o;
Tau_efmn__D_0__D_1__D_2__D_3_tmpShape_P[ 3 ] = n_o;
Tau_efmn__D_0__D_1__D_2__D_3.ResizeTo( Tau_efmn__D_0__D_1__D_2__D_3_tmpShape_P );
ObjShape w_bmje__D_0__D_1__D_2__D_3_tmpShape_P( 4 );
w_bmje__D_0__D_1__D_2__D_3_tmpShape_P[ 0 ] = n_v;
w_bmje__D_0__D_1__D_2__D_3_tmpShape_P[ 1 ] = n_o;
w_bmje__D_0__D_1__D_2__D_3_tmpShape_P[ 2 ] = n_o;
w_bmje__D_0__D_1__D_2__D_3_tmpShape_P[ 3 ] = n_v;
w_bmje__D_0__D_1__D_2__D_3.ResizeTo( w_bmje__D_0__D_1__D_2__D_3_tmpShape_P );
ObjShape t_fj__D_0_1__D_2_3_tmpShape_P( 2 );
t_fj__D_0_1__D_2_3_tmpShape_P[ 0 ] = n_v;
t_fj__D_0_1__D_2_3_tmpShape_P[ 1 ] = n_o;
t_fj__D_0_1__D_2_3.ResizeTo( t_fj__D_0_1__D_2_3_tmpShape_P );
ObjShape x_bmej__D_0__D_1__D_2__D_3_tmpShape_P( 4 );
x_bmej__D_0__D_1__D_2__D_3_tmpShape_P[ 0 ] = n_v;
x_bmej__D_0__D_1__D_2__D_3_tmpShape_P[ 1 ] = n_o;
x_bmej__D_0__D_1__D_2__D_3_tmpShape_P[ 2 ] = n_v;
x_bmej__D_0__D_1__D_2__D_3_tmpShape_P[ 3 ] = n_o;
x_bmej__D_0__D_1__D_2__D_3.ResizeTo( x_bmej__D_0__D_1__D_2__D_3_tmpShape_P );
ObjShape u_mnje__D_0__D_1__D_2__D_3_tmpShape_P( 4 );
u_mnje__D_0__D_1__D_2__D_3_tmpShape_P[ 0 ] = n_o;
u_mnje__D_0__D_1__D_2__D_3_tmpShape_P[ 1 ] = n_o;
u_mnje__D_0__D_1__D_2__D_3_tmpShape_P[ 2 ] = n_o;
u_mnje__D_0__D_1__D_2__D_3_tmpShape_P[ 3 ] = n_v;
u_mnje__D_0__D_1__D_2__D_3.ResizeTo( u_mnje__D_0__D_1__D_2__D_3_tmpShape_P );
// P_jimb has 4 dims
ObjShape P_jimb__D_0__D_1__D_2__D_3_tmpShape_P( 4 );
P_jimb__D_0__D_1__D_2__D_3_tmpShape_P[ 0 ] = n_o;
P_jimb__D_0__D_1__D_2__D_3_tmpShape_P[ 1 ] = n_o;
P_jimb__D_0__D_1__D_2__D_3_tmpShape_P[ 2 ] = n_o;
P_jimb__D_0__D_1__D_2__D_3_tmpShape_P[ 3 ] = n_v;
P_jimb__D_0__D_1__D_2__D_3.ResizeTo( P_jimb__D_0__D_1__D_2__D_3_tmpShape_P );
MakeUniform( P_jimb__D_0__D_1__D_2__D_3 );
ObjShape overwrite_tmpShape_H;
TensorDistribution dist__S__S = tmen::StringToTensorDist("[(),()]");
TensorDistribution dist__D_2_3__S__D_0_1__S = tmen::StringToTensorDist("[(2,3),(),(0,1),()]");
TensorDistribution dist__D_2_3__D_1__D_0__S = tmen::StringToTensorDist("[(2,3),(1),(0),()]");
ModeArray modes( 0 );
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
DistTensor<double> H_me_lvl2_part0_1_perm10__D_2_3__D_0_1( dist__D_0_1__D_2_3, g );
H_me_lvl2_part0_1_perm10__D_2_3__D_0_1.SetLocalPermutation( perm_1_0 );
	//H_me_lvl2_part0_2[D01,D23]
DistTensor<double> H_me_lvl2_part0_2__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//t_fj_lvl1_part1_1[*,*]
DistTensor<double> t_fj_lvl1_part1_1__S__S( dist__S__S, g );
	//H_temp1[D0,D1,D2,D3]
DistTensor<double> H_temp1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//H_temp1_lvl0_part2B[D0,D1,D2,D3]
DistTensor<double> H_temp1_lvl0_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//H_temp1_lvl0_part2T[D0,D1,D2,D3]
DistTensor<double> H_temp1_lvl0_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//H_temp1_lvl0_part3B[D0,D1,D2,D3]
DistTensor<double> H_temp1_lvl0_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//H_temp1_lvl0_part3T[D0,D1,D2,D3]
DistTensor<double> H_temp1_lvl0_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//H_temp1_lvl1_part2B[D0,D1,D2,D3]
DistTensor<double> H_temp1_lvl1_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//H_temp1_lvl1_part2T[D0,D1,D2,D3]
DistTensor<double> H_temp1_lvl1_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//H_temp1_lvl1_part2_0[D0,D1,D2,D3]
DistTensor<double> H_temp1_lvl1_part2_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//H_temp1_lvl1_part2_1[D0,D1,D2,D3]
DistTensor<double> H_temp1_lvl1_part2_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//H_temp1_lvl1_part2_1_lvl1_part3B[D0,D1,D2,D3]
DistTensor<double> H_temp1_lvl1_part2_1_lvl1_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//H_temp1_lvl1_part2_1_lvl1_part3T[D0,D1,D2,D3]
DistTensor<double> H_temp1_lvl1_part2_1_lvl1_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//H_temp1_lvl1_part2_1_lvl2_part3B[D0,D1,D2,D3]
DistTensor<double> H_temp1_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//H_temp1_lvl1_part2_1_lvl2_part3T[D0,D1,D2,D3]
DistTensor<double> H_temp1_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//H_temp1_lvl1_part2_1_lvl2_part3_0[D0,D1,D2,D3]
DistTensor<double> H_temp1_lvl1_part2_1_lvl2_part3_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//H_temp1_lvl1_part2_1_lvl2_part3_1[D0,D1,D2,D3]
DistTensor<double> H_temp1_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//H_temp1_lvl1_part2_1_lvl2_part3_2[D0,D1,D2,D3]
DistTensor<double> H_temp1_lvl1_part2_1_lvl2_part3_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//H_temp1_lvl1_part2_2[D0,D1,D2,D3]
DistTensor<double> H_temp1_lvl1_part2_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//H_temp1_lvl1_part3B[D0,D1,D2,D3]
DistTensor<double> H_temp1_lvl1_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//H_temp1_lvl1_part3T[D0,D1,D2,D3]
DistTensor<double> H_temp1_lvl1_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//H_temp1_lvl1_part3_0[D0,D1,D2,D3]
DistTensor<double> H_temp1_lvl1_part3_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//H_temp1_lvl1_part3_1[D0,D1,D2,D3]
DistTensor<double> H_temp1_lvl1_part3_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//H_temp1_lvl1_part3_1_lvl1_part2B[D0,D1,D2,D3]
DistTensor<double> H_temp1_lvl1_part3_1_lvl1_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//H_temp1_lvl1_part3_1_lvl1_part2T[D0,D1,D2,D3]
DistTensor<double> H_temp1_lvl1_part3_1_lvl1_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//H_temp1_lvl1_part3_1_lvl2_part2B[D0,D1,D2,D3]
DistTensor<double> H_temp1_lvl1_part3_1_lvl2_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//H_temp1_lvl1_part3_1_lvl2_part2T[D0,D1,D2,D3]
DistTensor<double> H_temp1_lvl1_part3_1_lvl2_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//H_temp1_lvl1_part3_1_lvl2_part2_0[D0,D1,D2,D3]
DistTensor<double> H_temp1_lvl1_part3_1_lvl2_part2_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//H_temp1_lvl1_part3_1_lvl2_part2_1[D0,D1,D2,D3]
DistTensor<double> H_temp1_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//H_temp1_lvl1_part3_1_lvl2_part2_1[D23,D1,D0,*]
DistTensor<double> H_temp1_lvl1_part3_1_lvl2_part2_1__D_2_3__D_1__D_0__S( dist__D_2_3__D_1__D_0__S, g );
	//H_temp1_lvl1_part3_1_lvl2_part2_1[D2,D1,D0,D3]
DistTensor<double> H_temp1_lvl1_part3_1_lvl2_part2_1__D_2__D_1__D_0__D_3( dist__D_2__D_1__D_0__D_3, g );
	//H_temp1_lvl1_part3_1_lvl2_part2_1[D23,*,D01,*]
DistTensor<double> H_temp1_lvl1_part3_1_lvl2_part2_1_perm0213__D_2_3__D_0_1__S__S( dist__D_2_3__S__D_0_1__S, g );
H_temp1_lvl1_part3_1_lvl2_part2_1_perm0213__D_2_3__D_0_1__S__S.SetLocalPermutation( perm_0_2_1_3 );
	//H_temp1_lvl1_part3_1_lvl2_part2_2[D0,D1,D2,D3]
DistTensor<double> H_temp1_lvl1_part3_1_lvl2_part2_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//H_temp1_lvl1_part3_2[D0,D1,D2,D3]
DistTensor<double> H_temp1_lvl1_part3_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
ObjShape t_fj__D_0_1__D_2_3_tmpShape_H( 2 );
t_fj__D_0_1__D_2_3_tmpShape_H[ 0 ] = n_v;
t_fj__D_0_1__D_2_3_tmpShape_H[ 1 ] = n_o;
t_fj__D_0_1__D_2_3.ResizeTo( t_fj__D_0_1__D_2_3_tmpShape_H );
// H_me has 2 dims
ObjShape H_me__D_0_1__D_2_3_tmpShape_H( 2 );
H_me__D_0_1__D_2_3_tmpShape_H[ 0 ] = n_o;
H_me__D_0_1__D_2_3_tmpShape_H[ 1 ] = n_v;
H_me__D_0_1__D_2_3.ResizeTo( H_me__D_0_1__D_2_3_tmpShape_H );
MakeUniform( H_me__D_0_1__D_2_3 );
ObjShape v_femn__D_0__D_1__D_2__D_3_tmpShape_H( 4 );
v_femn__D_0__D_1__D_2__D_3_tmpShape_H[ 0 ] = n_v;
v_femn__D_0__D_1__D_2__D_3_tmpShape_H[ 1 ] = n_v;
v_femn__D_0__D_1__D_2__D_3_tmpShape_H[ 2 ] = n_o;
v_femn__D_0__D_1__D_2__D_3_tmpShape_H[ 3 ] = n_o;
v_femn__D_0__D_1__D_2__D_3.ResizeTo( v_femn__D_0__D_1__D_2__D_3_tmpShape_H );
ObjShape overwrite_tmpShape_F;
TensorDistribution dist__S__D_2_3 = tmen::StringToTensorDist("[(),(2,3)]");
TensorDistribution dist__S__D_1__D_2__D_3 = tmen::StringToTensorDist("[(),(1),(2),(3)]");
TensorDistribution dist__D_0__S__D_1__D_2__D_3 = tmen::StringToTensorDist("[(0),(),(1),(2),(3)]");
TensorDistribution dist__D_0__D_1__D_2_3__S = tmen::StringToTensorDist("[(0),(1),(2,3),()]");
TensorDistribution dist__D_0_1__S__D_2_3__S = tmen::StringToTensorDist("[(0,1),(),(2,3),()]");
Permutation perm_1_0_2_3_4( 5 );
perm_1_0_2_3_4[0] = 1;
perm_1_0_2_3_4[1] = 0;
perm_1_0_2_3_4[2] = 2;
perm_1_0_2_3_4[3] = 3;
perm_1_0_2_3_4[4] = 4;
ModeArray modes_4_3_2( 3 );
modes_4_3_2[0] = 4;
modes_4_3_2[1] = 3;
modes_4_3_2[2] = 2;
IndexArray indices_ae( 2 );
indices_ae[0] = 'a';
indices_ae[1] = 'e';
IndexArray indices_aemf( 4 );
indices_aemf[0] = 'a';
indices_aemf[1] = 'e';
indices_aemf[2] = 'm';
indices_aemf[3] = 'f';
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
IndexArray indices_fmna( 4 );
indices_fmna[0] = 'f';
indices_fmna[1] = 'm';
indices_fmna[2] = 'n';
indices_fmna[3] = 'a';
IndexArray indices_ma( 2 );
indices_ma[0] = 'm';
indices_ma[1] = 'a';
IndexArray indices_mf( 2 );
indices_mf[0] = 'm';
indices_mf[1] = 'f';
	//F_ae[D01,D23]
DistTensor<double> F_ae__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//F_ae_lvl0_part0B[D01,D23]
DistTensor<double> F_ae_lvl0_part0B__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//F_ae_lvl0_part0T[D01,D23]
DistTensor<double> F_ae_lvl0_part0T__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//F_ae_lvl1_part0B[D01,D23]
DistTensor<double> F_ae_lvl1_part0B__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//F_ae_lvl1_part0T[D01,D23]
DistTensor<double> F_ae_lvl1_part0T__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//F_ae_lvl1_part0_0[D01,D23]
DistTensor<double> F_ae_lvl1_part0_0__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//F_ae_lvl1_part0_1[D01,D23]
DistTensor<double> F_ae_lvl1_part0_1__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
DistTensor<double> F_ae_lvl1_part0_1_perm10__D_2_3__D_0_1( dist__D_0_1__D_2_3, g );
F_ae_lvl1_part0_1_perm10__D_2_3__D_0_1.SetLocalPermutation( perm_1_0 );
	//F_ae_lvl1_part0_2[D01,D23]
DistTensor<double> F_ae_lvl1_part0_2__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//F_ae[D0,*,D1,D2,D3]
DistTensor<double> F_ae_perm10234__S__D_0__D_1__D_2__D_3( dist__D_0__S__D_1__D_2__D_3, g );
F_ae_perm10234__S__D_0__D_1__D_2__D_3.SetLocalPermutation( perm_1_0_2_3_4 );
	//H_me_lvl2_part0_1[*,D23]
DistTensor<double> H_me_lvl2_part0_1_perm10__D_2_3__S( dist__S__D_2_3, g );
H_me_lvl2_part0_1_perm10__D_2_3__S.SetLocalPermutation( perm_1_0 );
DistTensor<double> T_bfnj_lvl1_part2_1_lvl2_part3_1_perm1230__D_1__D_2__D_3__D_0( dist__D_0__D_1__D_2__D_3, g );
T_bfnj_lvl1_part2_1_lvl2_part3_1_perm1230__D_1__D_2__D_3__D_0.SetLocalPermutation( perm_1_2_3_0 );
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
	//t_fj_lvl1_part0_1_lvl2_part1_1[D01,*]
DistTensor<double> t_fj_lvl1_part0_1_lvl2_part1_1_perm10__S__D_0_1( dist__D_0_1__S, g );
t_fj_lvl1_part0_1_lvl2_part1_1_perm10__S__D_0_1.SetLocalPermutation( perm_1_0 );
	//t_fj_lvl1_part0_1_lvl2_part1_1[*,*]
DistTensor<double> t_fj_lvl1_part0_1_lvl2_part1_1_perm10__S__S( dist__S__S, g );
t_fj_lvl1_part0_1_lvl2_part1_1_perm10__S__S.SetLocalPermutation( perm_1_0 );
	//t_fj_lvl1_part0_1_lvl2_part1_2[D01,D23]
DistTensor<double> t_fj_lvl1_part0_1_lvl2_part1_2__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//t_fj_lvl1_part0_2[D01,D23]
DistTensor<double> t_fj_lvl1_part0_2__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//F_temp1[D0,D1,D2,D3]
DistTensor<double> F_temp1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//F_temp1_lvl0_part2B[D0,D1,D2,D3]
DistTensor<double> F_temp1_lvl0_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//F_temp1_lvl0_part2T[D0,D1,D2,D3]
DistTensor<double> F_temp1_lvl0_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//F_temp1_lvl1_part2B[D0,D1,D2,D3]
DistTensor<double> F_temp1_lvl1_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//F_temp1_lvl1_part2T[D0,D1,D2,D3]
DistTensor<double> F_temp1_lvl1_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//F_temp1_lvl1_part2_0[D0,D1,D2,D3]
DistTensor<double> F_temp1_lvl1_part2_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//F_temp1_lvl1_part2_1[D0,D1,D2,D3]
DistTensor<double> F_temp1_lvl1_part2_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//F_temp1_lvl1_part2_1_lvl1_part3B[D0,D1,D2,D3]
DistTensor<double> F_temp1_lvl1_part2_1_lvl1_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//F_temp1_lvl1_part2_1_lvl1_part3T[D0,D1,D2,D3]
DistTensor<double> F_temp1_lvl1_part2_1_lvl1_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//F_temp1_lvl1_part2_1_lvl2_part3B[D0,D1,D2,D3]
DistTensor<double> F_temp1_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//F_temp1_lvl1_part2_1_lvl2_part3T[D0,D1,D2,D3]
DistTensor<double> F_temp1_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//F_temp1_lvl1_part2_1_lvl2_part3_0[D0,D1,D2,D3]
DistTensor<double> F_temp1_lvl1_part2_1_lvl2_part3_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//F_temp1_lvl1_part2_1_lvl2_part3_1[D0,D1,D2,D3]
DistTensor<double> F_temp1_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//F_temp1_lvl1_part2_1_lvl2_part3_1[*,D1,D2,D3]
DistTensor<double> F_temp1_lvl1_part2_1_lvl2_part3_1__S__D_1__D_2__D_3( dist__S__D_1__D_2__D_3, g );
	//F_temp1_lvl1_part2_1_lvl2_part3_2[D0,D1,D2,D3]
DistTensor<double> F_temp1_lvl1_part2_1_lvl2_part3_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//F_temp1_lvl1_part2_2[D0,D1,D2,D3]
DistTensor<double> F_temp1_lvl1_part2_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//F_temp2[D0,D1,D2,D3]
DistTensor<double> F_temp2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//F_temp2_lvl0_part0B[D0,D1,D2,D3]
DistTensor<double> F_temp2_lvl0_part0B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//F_temp2_lvl0_part0T[D0,D1,D2,D3]
DistTensor<double> F_temp2_lvl0_part0T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//F_temp2_lvl0_part3B[D0,D1,D2,D3]
DistTensor<double> F_temp2_lvl0_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//F_temp2_lvl0_part3T[D0,D1,D2,D3]
DistTensor<double> F_temp2_lvl0_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//F_temp2_lvl1_part0B[D0,D1,D2,D3]
DistTensor<double> F_temp2_lvl1_part0B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//F_temp2_lvl1_part0T[D0,D1,D2,D3]
DistTensor<double> F_temp2_lvl1_part0T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//F_temp2_lvl1_part0_0[D0,D1,D2,D3]
DistTensor<double> F_temp2_lvl1_part0_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//F_temp2_lvl1_part0_1[D0,D1,D2,D3]
DistTensor<double> F_temp2_lvl1_part0_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//F_temp2_lvl1_part0_1_lvl1_part1B[D0,D1,D2,D3]
DistTensor<double> F_temp2_lvl1_part0_1_lvl1_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//F_temp2_lvl1_part0_1_lvl1_part1T[D0,D1,D2,D3]
DistTensor<double> F_temp2_lvl1_part0_1_lvl1_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//F_temp2_lvl1_part0_1_lvl2_part1B[D0,D1,D2,D3]
DistTensor<double> F_temp2_lvl1_part0_1_lvl2_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//F_temp2_lvl1_part0_1_lvl2_part1T[D0,D1,D2,D3]
DistTensor<double> F_temp2_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//F_temp2_lvl1_part0_1_lvl2_part1_0[D0,D1,D2,D3]
DistTensor<double> F_temp2_lvl1_part0_1_lvl2_part1_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//F_temp2_lvl1_part0_1_lvl2_part1_1[D0,D1,D2,D3]
DistTensor<double> F_temp2_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//F_temp2_lvl1_part0_1_lvl2_part1_2[D0,D1,D2,D3]
DistTensor<double> F_temp2_lvl1_part0_1_lvl2_part1_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//F_temp2_lvl1_part0_2[D0,D1,D2,D3]
DistTensor<double> F_temp2_lvl1_part0_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//F_temp2_lvl1_part3B[D0,D1,D2,D3]
DistTensor<double> F_temp2_lvl1_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//F_temp2_lvl1_part3T[D0,D1,D2,D3]
DistTensor<double> F_temp2_lvl1_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//F_temp2_lvl1_part3_0[D0,D1,D2,D3]
DistTensor<double> F_temp2_lvl1_part3_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//F_temp2_lvl1_part3_1[D0,D1,D2,D3]
DistTensor<double> F_temp2_lvl1_part3_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//F_temp2_lvl1_part3_1_lvl1_part1B[D0,D1,D2,D3]
DistTensor<double> F_temp2_lvl1_part3_1_lvl1_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//F_temp2_lvl1_part3_1_lvl1_part1T[D0,D1,D2,D3]
DistTensor<double> F_temp2_lvl1_part3_1_lvl1_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//F_temp2_lvl1_part3_1_lvl2_part1B[D0,D1,D2,D3]
DistTensor<double> F_temp2_lvl1_part3_1_lvl2_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//F_temp2_lvl1_part3_1_lvl2_part1T[D0,D1,D2,D3]
DistTensor<double> F_temp2_lvl1_part3_1_lvl2_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//F_temp2_lvl1_part3_1_lvl2_part1_0[D0,D1,D2,D3]
DistTensor<double> F_temp2_lvl1_part3_1_lvl2_part1_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//F_temp2_lvl1_part3_1_lvl2_part1_1[D0,D1,D23,*]
DistTensor<double> F_temp2_lvl1_part3_1_lvl2_part1_1__D_0__D_1__D_2_3__S( dist__D_0__D_1__D_2_3__S, g );
	//F_temp2_lvl1_part3_1_lvl2_part1_1[D0,D1,D2,D3]
DistTensor<double> F_temp2_lvl1_part3_1_lvl2_part1_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//F_temp2_lvl1_part3_1_lvl2_part1_1[D01,*,D23,*]
DistTensor<double> F_temp2_lvl1_part3_1_lvl2_part1_1_perm0213__D_0_1__D_2_3__S__S( dist__D_0_1__S__D_2_3__S, g );
F_temp2_lvl1_part3_1_lvl2_part1_1_perm0213__D_0_1__D_2_3__S__S.SetLocalPermutation( perm_0_2_1_3 );
	//F_temp2_lvl1_part3_1_lvl2_part1_2[D0,D1,D2,D3]
DistTensor<double> F_temp2_lvl1_part3_1_lvl2_part1_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//F_temp2_lvl1_part3_2[D0,D1,D2,D3]
DistTensor<double> F_temp2_lvl1_part3_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
ObjShape H_me__D_0_1__D_2_3_tmpShape_F( 2 );
H_me__D_0_1__D_2_3_tmpShape_F[ 0 ] = n_o;
H_me__D_0_1__D_2_3_tmpShape_F[ 1 ] = n_v;
H_me__D_0_1__D_2_3.ResizeTo( H_me__D_0_1__D_2_3_tmpShape_F );
ObjShape t_fj__D_0_1__D_2_3_tmpShape_F( 2 );
t_fj__D_0_1__D_2_3_tmpShape_F[ 0 ] = n_v;
t_fj__D_0_1__D_2_3_tmpShape_F[ 1 ] = n_o;
t_fj__D_0_1__D_2_3.ResizeTo( t_fj__D_0_1__D_2_3_tmpShape_F );
ObjShape r_bmfe__D_0__D_1__D_2__D_3_tmpShape_F( 4 );
r_bmfe__D_0__D_1__D_2__D_3_tmpShape_F[ 0 ] = n_v;
r_bmfe__D_0__D_1__D_2__D_3_tmpShape_F[ 1 ] = n_o;
r_bmfe__D_0__D_1__D_2__D_3_tmpShape_F[ 2 ] = n_v;
r_bmfe__D_0__D_1__D_2__D_3_tmpShape_F[ 3 ] = n_v;
r_bmfe__D_0__D_1__D_2__D_3.ResizeTo( r_bmfe__D_0__D_1__D_2__D_3_tmpShape_F );
ObjShape T_bfnj__D_0__D_1__D_2__D_3_tmpShape_F( 4 );
T_bfnj__D_0__D_1__D_2__D_3_tmpShape_F[ 0 ] = n_v;
T_bfnj__D_0__D_1__D_2__D_3_tmpShape_F[ 1 ] = n_v;
T_bfnj__D_0__D_1__D_2__D_3_tmpShape_F[ 2 ] = n_o;
T_bfnj__D_0__D_1__D_2__D_3_tmpShape_F[ 3 ] = n_o;
T_bfnj__D_0__D_1__D_2__D_3.ResizeTo( T_bfnj__D_0__D_1__D_2__D_3_tmpShape_F );
// F_ae has 2 dims
ObjShape F_ae__D_0_1__D_2_3_tmpShape_F( 2 );
F_ae__D_0_1__D_2_3_tmpShape_F[ 0 ] = n_v;
F_ae__D_0_1__D_2_3_tmpShape_F[ 1 ] = n_v;
F_ae__D_0_1__D_2_3.ResizeTo( F_ae__D_0_1__D_2_3_tmpShape_F );
MakeUniform( F_ae__D_0_1__D_2_3 );
ObjShape v_femn__D_0__D_1__D_2__D_3_tmpShape_F( 4 );
v_femn__D_0__D_1__D_2__D_3_tmpShape_F[ 0 ] = n_v;
v_femn__D_0__D_1__D_2__D_3_tmpShape_F[ 1 ] = n_v;
v_femn__D_0__D_1__D_2__D_3_tmpShape_F[ 2 ] = n_o;
v_femn__D_0__D_1__D_2__D_3_tmpShape_F[ 3 ] = n_o;
v_femn__D_0__D_1__D_2__D_3.ResizeTo( v_femn__D_0__D_1__D_2__D_3_tmpShape_F );
ObjShape overwrite_tmpShape_G;
TensorDistribution dist__S__D_2__D_0__D_1__D_3 = tmen::StringToTensorDist("[(),(2),(0),(1),(3)]");
TensorDistribution dist__D_0__S__D_2_3__S = tmen::StringToTensorDist("[(0),(),(2,3),()]");
TensorDistribution dist__D_0__D_1__S__D_3 = tmen::StringToTensorDist("[(0),(1),(),(3)]");
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
	//H_me_lvl0_part0B[D01,D23]
DistTensor<double> H_me_lvl0_part0B__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//H_me_lvl0_part0T[D01,D23]
DistTensor<double> H_me_lvl0_part0T__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
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
DistTensor<double> T_bfnj_lvl1_part3_1_perm0132__D_0__D_1__D_3__D_2( dist__D_0__D_1__D_2__D_3, g );
T_bfnj_lvl1_part3_1_perm0132__D_0__D_1__D_3__D_2.SetLocalPermutation( perm_0_1_3_2 );
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
ObjShape H_me__D_0_1__D_2_3_tmpShape_G( 2 );
H_me__D_0_1__D_2_3_tmpShape_G[ 0 ] = n_o;
H_me__D_0_1__D_2_3_tmpShape_G[ 1 ] = n_v;
H_me__D_0_1__D_2_3.ResizeTo( H_me__D_0_1__D_2_3_tmpShape_G );
ObjShape t_fj__D_0_1__D_2_3_tmpShape_G( 2 );
t_fj__D_0_1__D_2_3_tmpShape_G[ 0 ] = n_v;
t_fj__D_0_1__D_2_3_tmpShape_G[ 1 ] = n_o;
t_fj__D_0_1__D_2_3.ResizeTo( t_fj__D_0_1__D_2_3_tmpShape_G );
ObjShape u_mnje__D_0__D_1__D_2__D_3_tmpShape_G( 4 );
u_mnje__D_0__D_1__D_2__D_3_tmpShape_G[ 0 ] = n_o;
u_mnje__D_0__D_1__D_2__D_3_tmpShape_G[ 1 ] = n_o;
u_mnje__D_0__D_1__D_2__D_3_tmpShape_G[ 2 ] = n_o;
u_mnje__D_0__D_1__D_2__D_3_tmpShape_G[ 3 ] = n_v;
u_mnje__D_0__D_1__D_2__D_3.ResizeTo( u_mnje__D_0__D_1__D_2__D_3_tmpShape_G );
ObjShape T_bfnj__D_0__D_1__D_2__D_3_tmpShape_G( 4 );
T_bfnj__D_0__D_1__D_2__D_3_tmpShape_G[ 0 ] = n_v;
T_bfnj__D_0__D_1__D_2__D_3_tmpShape_G[ 1 ] = n_v;
T_bfnj__D_0__D_1__D_2__D_3_tmpShape_G[ 2 ] = n_o;
T_bfnj__D_0__D_1__D_2__D_3_tmpShape_G[ 3 ] = n_o;
T_bfnj__D_0__D_1__D_2__D_3.ResizeTo( T_bfnj__D_0__D_1__D_2__D_3_tmpShape_G );
// G_mi has 2 dims
ObjShape G_mi__D_0_1__D_2_3_tmpShape_G( 2 );
G_mi__D_0_1__D_2_3_tmpShape_G[ 0 ] = n_o;
G_mi__D_0_1__D_2_3_tmpShape_G[ 1 ] = n_o;
G_mi__D_0_1__D_2_3.ResizeTo( G_mi__D_0_1__D_2_3_tmpShape_G );
MakeUniform( G_mi__D_0_1__D_2_3 );
ObjShape v_femn__D_0__D_1__D_2__D_3_tmpShape_G( 4 );
v_femn__D_0__D_1__D_2__D_3_tmpShape_G[ 0 ] = n_v;
v_femn__D_0__D_1__D_2__D_3_tmpShape_G[ 1 ] = n_v;
v_femn__D_0__D_1__D_2__D_3_tmpShape_G[ 2 ] = n_o;
v_femn__D_0__D_1__D_2__D_3_tmpShape_G[ 3 ] = n_o;
v_femn__D_0__D_1__D_2__D_3.ResizeTo( v_femn__D_0__D_1__D_2__D_3_tmpShape_G );
ObjShape overwrite_tmpShape_z_small;
TensorDistribution dist__D_0__S__D_2__D_3__D_1 = tmen::StringToTensorDist("[(0),(),(2),(3),(1)]");
TensorDistribution dist__D_0__D_3__D_2__D_1 = tmen::StringToTensorDist("[(0),(3),(2),(1)]");
TensorDistribution dist__D_2__D_3__S__D_1 = tmen::StringToTensorDist("[(2),(3),(),(1)]");
TensorDistribution dist__D_0_1__S__D_2__D_3 = tmen::StringToTensorDist("[(0,1),(),(2),(3)]");
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
IndexArray indices_mnea( 4 );
indices_mnea[0] = 'm';
indices_mnea[1] = 'n';
indices_mnea[2] = 'e';
indices_mnea[3] = 'a';
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
	//H_me_lvl2_part0_1[*,*]
DistTensor<double> H_me_lvl2_part0_1__S__S( dist__S__S, g );
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
	//Tau_efmn_lvl1_part3_1_lvl2_part2_1[D0,D3,D2,D1]
DistTensor<double> Tau_efmn_lvl1_part3_1_lvl2_part2_1__D_0__D_3__D_2__D_1( dist__D_0__D_3__D_2__D_1, g );
	//Tau_efmn_lvl1_part3_1_lvl2_part2_1[D2,D3,*,D1]
DistTensor<double> Tau_efmn_lvl1_part3_1_lvl2_part2_1_perm0132__D_2__D_3__D_1__S( dist__D_2__D_3__S__D_1, g );
Tau_efmn_lvl1_part3_1_lvl2_part2_1_perm0132__D_2__D_3__D_1__S.SetLocalPermutation( perm_0_1_3_2 );
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
DistTensor<double> t_fj_lvl2_part1_1_perm10__S__D_0_1( dist__D_0_1__S, g );
t_fj_lvl2_part1_1_perm10__S__D_0_1.SetLocalPermutation( perm_1_0 );
	//t_fj_lvl2_part1_1[*,*]
DistTensor<double> t_fj_lvl2_part1_1_perm10__S__S( dist__S__S, g );
t_fj_lvl2_part1_1_perm10__S__S.SetLocalPermutation( perm_1_0 );
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
ObjShape G_mi__D_0_1__D_2_3_tmpShape_z_small( 2 );
G_mi__D_0_1__D_2_3_tmpShape_z_small[ 0 ] = n_o;
G_mi__D_0_1__D_2_3_tmpShape_z_small[ 1 ] = n_o;
G_mi__D_0_1__D_2_3.ResizeTo( G_mi__D_0_1__D_2_3_tmpShape_z_small );
ObjShape t_fj__D_0_1__D_2_3_tmpShape_z_small( 2 );
t_fj__D_0_1__D_2_3_tmpShape_z_small[ 0 ] = n_v;
t_fj__D_0_1__D_2_3_tmpShape_z_small[ 1 ] = n_o;
t_fj__D_0_1__D_2_3.ResizeTo( t_fj__D_0_1__D_2_3_tmpShape_z_small );
ObjShape T_bfnj__D_0__D_1__D_2__D_3_tmpShape_z_small( 4 );
T_bfnj__D_0__D_1__D_2__D_3_tmpShape_z_small[ 0 ] = n_v;
T_bfnj__D_0__D_1__D_2__D_3_tmpShape_z_small[ 1 ] = n_v;
T_bfnj__D_0__D_1__D_2__D_3_tmpShape_z_small[ 2 ] = n_o;
T_bfnj__D_0__D_1__D_2__D_3_tmpShape_z_small[ 3 ] = n_o;
T_bfnj__D_0__D_1__D_2__D_3.ResizeTo( T_bfnj__D_0__D_1__D_2__D_3_tmpShape_z_small );
ObjShape U_mnie__D_0__D_1__D_2__D_3_tmpShape_z_small( 4 );
U_mnie__D_0__D_1__D_2__D_3_tmpShape_z_small[ 0 ] = n_o;
U_mnie__D_0__D_1__D_2__D_3_tmpShape_z_small[ 1 ] = n_o;
U_mnie__D_0__D_1__D_2__D_3_tmpShape_z_small[ 2 ] = n_o;
U_mnie__D_0__D_1__D_2__D_3_tmpShape_z_small[ 3 ] = n_v;
U_mnie__D_0__D_1__D_2__D_3.ResizeTo( U_mnie__D_0__D_1__D_2__D_3_tmpShape_z_small );
ObjShape w_bmje__D_0__D_1__D_2__D_3_tmpShape_z_small( 4 );
w_bmje__D_0__D_1__D_2__D_3_tmpShape_z_small[ 0 ] = n_v;
w_bmje__D_0__D_1__D_2__D_3_tmpShape_z_small[ 1 ] = n_o;
w_bmje__D_0__D_1__D_2__D_3_tmpShape_z_small[ 2 ] = n_o;
w_bmje__D_0__D_1__D_2__D_3_tmpShape_z_small[ 3 ] = n_v;
w_bmje__D_0__D_1__D_2__D_3.ResizeTo( w_bmje__D_0__D_1__D_2__D_3_tmpShape_z_small );
ObjShape x_bmej__D_0__D_1__D_2__D_3_tmpShape_z_small( 4 );
x_bmej__D_0__D_1__D_2__D_3_tmpShape_z_small[ 0 ] = n_v;
x_bmej__D_0__D_1__D_2__D_3_tmpShape_z_small[ 1 ] = n_o;
x_bmej__D_0__D_1__D_2__D_3_tmpShape_z_small[ 2 ] = n_v;
x_bmej__D_0__D_1__D_2__D_3_tmpShape_z_small[ 3 ] = n_o;
x_bmej__D_0__D_1__D_2__D_3.ResizeTo( x_bmej__D_0__D_1__D_2__D_3_tmpShape_z_small );
ObjShape H_me__D_0_1__D_2_3_tmpShape_z_small( 2 );
H_me__D_0_1__D_2_3_tmpShape_z_small[ 0 ] = n_o;
H_me__D_0_1__D_2_3_tmpShape_z_small[ 1 ] = n_v;
H_me__D_0_1__D_2_3.ResizeTo( H_me__D_0_1__D_2_3_tmpShape_z_small );
ObjShape Tau_efmn__D_0__D_1__D_2__D_3_tmpShape_z_small( 4 );
Tau_efmn__D_0__D_1__D_2__D_3_tmpShape_z_small[ 0 ] = n_v;
Tau_efmn__D_0__D_1__D_2__D_3_tmpShape_z_small[ 1 ] = n_v;
Tau_efmn__D_0__D_1__D_2__D_3_tmpShape_z_small[ 2 ] = n_o;
Tau_efmn__D_0__D_1__D_2__D_3_tmpShape_z_small[ 3 ] = n_o;
Tau_efmn__D_0__D_1__D_2__D_3.ResizeTo( Tau_efmn__D_0__D_1__D_2__D_3_tmpShape_z_small );
// z_ai has 2 dims
ObjShape z_ai__D_0_1__D_2_3_tmpShape_z_small( 2 );
z_ai__D_0_1__D_2_3_tmpShape_z_small[ 0 ] = n_v;
z_ai__D_0_1__D_2_3_tmpShape_z_small[ 1 ] = n_o;
z_ai__D_0_1__D_2_3.ResizeTo( z_ai__D_0_1__D_2_3_tmpShape_z_small );
MakeUniform( z_ai__D_0_1__D_2_3 );
ObjShape r_bmfe__D_0__D_1__D_2__D_3_tmpShape_z_small( 4 );
r_bmfe__D_0__D_1__D_2__D_3_tmpShape_z_small[ 0 ] = n_v;
r_bmfe__D_0__D_1__D_2__D_3_tmpShape_z_small[ 1 ] = n_o;
r_bmfe__D_0__D_1__D_2__D_3_tmpShape_z_small[ 2 ] = n_v;
r_bmfe__D_0__D_1__D_2__D_3_tmpShape_z_small[ 3 ] = n_v;
r_bmfe__D_0__D_1__D_2__D_3.ResizeTo( r_bmfe__D_0__D_1__D_2__D_3_tmpShape_z_small );
ObjShape overwrite_tmpShape_Z;
TensorDistribution dist__S__S__D_2__D_3 = tmen::StringToTensorDist("[(),(),(2),(3)]");
TensorDistribution dist__S__D_2 = tmen::StringToTensorDist("[(),(2)]");
TensorDistribution dist__S__D_3__S__D_1 = tmen::StringToTensorDist("[(),(3),(),(1)]");
TensorDistribution dist__D_0__S__D_2__S__D_1__D_3 = tmen::StringToTensorDist("[(0),(),(2),(),(1),(3)]");
TensorDistribution dist__D_0__D_1__S__S__D_2__D_3 = tmen::StringToTensorDist("[(0),(1),(),(),(2),(3)]");
TensorDistribution dist__D_1__S__S__D_3 = tmen::StringToTensorDist("[(1),(),(),(3)]");
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
Permutation perm_1_3_0_2_5_4( 6 );
perm_1_3_0_2_5_4[0] = 1;
perm_1_3_0_2_5_4[1] = 3;
perm_1_3_0_2_5_4[2] = 0;
perm_1_3_0_2_5_4[3] = 2;
perm_1_3_0_2_5_4[4] = 5;
perm_1_3_0_2_5_4[5] = 4;
Permutation perm_2_0_3_1( 4 );
perm_2_0_3_1[0] = 2;
perm_2_0_3_1[1] = 0;
perm_2_0_3_1[2] = 3;
perm_2_0_3_1[3] = 1;
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
	//G_mi_lvl2_part0_1[*,D2]
DistTensor<double> G_mi_lvl2_part0_1_perm10__D_2__S( dist__S__D_2, g );
G_mi_lvl2_part0_1_perm10__D_2__S.SetLocalPermutation( perm_1_0 );
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
	//T_bfnj_lvl1_part1_1_lvl2_part2_1[D0,D1,D3,D2]
DistTensor<double> T_bfnj_lvl1_part1_1_lvl2_part2_1__D_0__D_1__D_3__D_2( dist__D_0__D_1__D_3__D_2, g );
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
	//Tau_efmn_lvl1_part2_1_lvl2_part3_1[D2,D3,D0,D1]
DistTensor<double> Tau_efmn_lvl1_part2_1_lvl2_part3_1__D_2__D_3__D_0__D_1( dist__D_2__D_3__D_0__D_1, g );
	//Tau_efmn_lvl1_part2_1_lvl2_part3_1[D2,D3,*,*]
DistTensor<double> Tau_efmn_lvl1_part2_1_lvl2_part3_1__D_2__D_3__S__S( dist__D_2__D_3__S__S, g );
	//Tau_efmn_lvl1_part2_1_lvl2_part3_1[D0,D1,*,*]
DistTensor<double> Tau_efmn_lvl1_part2_1_lvl2_part3_1_perm2301__S__S__D_0__D_1( dist__D_0__D_1__S__S, g );
Tau_efmn_lvl1_part2_1_lvl2_part3_1_perm2301__S__S__D_0__D_1.SetLocalPermutation( perm_2_3_0_1 );
	//W_bmje_lvl0_part0B[D0,D1,D2,D3]
DistTensor<double> W_bmje_lvl0_part0B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_bmje_lvl0_part0T[D0,D1,D2,D3]
DistTensor<double> W_bmje_lvl0_part0T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
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
	//W_bmje_lvl1_part0_1_lvl2_part2_1[D0,D3,D2,D1]
DistTensor<double> W_bmje_lvl1_part0_1_lvl2_part2_1__D_0__D_3__D_2__D_1( dist__D_0__D_3__D_2__D_1, g );
	//W_bmje_lvl1_part0_1_lvl2_part2_1[*,D3,*,D1]
DistTensor<double> W_bmje_lvl1_part0_1_lvl2_part2_1_perm0213__S__S__D_3__D_1( dist__S__D_3__S__D_1, g );
W_bmje_lvl1_part0_1_lvl2_part2_1_perm0213__S__S__D_3__D_1.SetLocalPermutation( perm_0_2_1_3 );
	//W_bmje_lvl1_part0_1_lvl2_part2_2[D0,D1,D2,D3]
DistTensor<double> W_bmje_lvl1_part0_1_lvl2_part2_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_bmje_lvl1_part0_2[D0,D1,D2,D3]
DistTensor<double> W_bmje_lvl1_part0_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
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
	//X_bmej_lvl1_part2_1_lvl2_part1_1[D1,D0,D2,D3]
DistTensor<double> X_bmej_lvl1_part2_1_lvl2_part1_1__D_1__D_0__D_2__D_3( dist__D_1__D_0__D_2__D_3, g );
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
	//accum[D0,D1,D2,D3]
DistTensor<double> accum__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//accum_lvl0_part1B[D0,D1,D2,D3]
DistTensor<double> accum_lvl0_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//accum_lvl0_part1T[D0,D1,D2,D3]
DistTensor<double> accum_lvl0_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//accum_lvl0_part2B[D0,D1,D2,D3]
DistTensor<double> accum_lvl0_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//accum_lvl0_part2T[D0,D1,D2,D3]
DistTensor<double> accum_lvl0_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//accum_lvl0_part3B[D0,D1,D2,D3]
DistTensor<double> accum_lvl0_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//accum_lvl0_part3T[D0,D1,D2,D3]
DistTensor<double> accum_lvl0_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//accum_lvl1_part1B[D0,D1,D2,D3]
DistTensor<double> accum_lvl1_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//accum_lvl1_part1T[D0,D1,D2,D3]
DistTensor<double> accum_lvl1_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//accum_lvl1_part1_0[D0,D1,D2,D3]
DistTensor<double> accum_lvl1_part1_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//accum_lvl1_part1_1[D0,D1,D2,D3]
DistTensor<double> accum_lvl1_part1_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//accum_lvl1_part1_1_lvl1_part3B[D0,D1,D2,D3]
DistTensor<double> accum_lvl1_part1_1_lvl1_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//accum_lvl1_part1_1_lvl1_part3T[D0,D1,D2,D3]
DistTensor<double> accum_lvl1_part1_1_lvl1_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//accum_lvl1_part1_1_lvl2_part3B[D0,D1,D2,D3]
DistTensor<double> accum_lvl1_part1_1_lvl2_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//accum_lvl1_part1_1_lvl2_part3T[D0,D1,D2,D3]
DistTensor<double> accum_lvl1_part1_1_lvl2_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//accum_lvl1_part1_1_lvl2_part3_0[D0,D1,D2,D3]
DistTensor<double> accum_lvl1_part1_1_lvl2_part3_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//accum_lvl1_part1_1_lvl2_part3_1[D0,D1,D2,D3]
DistTensor<double> accum_lvl1_part1_1_lvl2_part3_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//accum_lvl1_part1_1_lvl2_part3_1[D0,*,D2,*,D1,D3]
DistTensor<double> accum_lvl1_part1_1_lvl2_part3_1_perm130254__S__S__D_0__D_2__D_3__D_1( dist__D_0__S__D_2__S__D_1__D_3, g );
accum_lvl1_part1_1_lvl2_part3_1_perm130254__S__S__D_0__D_2__D_3__D_1.SetLocalPermutation( perm_1_3_0_2_5_4 );
	//accum_lvl1_part1_1_lvl2_part3_2[D0,D1,D2,D3]
DistTensor<double> accum_lvl1_part1_1_lvl2_part3_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//accum_lvl1_part1_2[D0,D1,D2,D3]
DistTensor<double> accum_lvl1_part1_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//accum_lvl1_part2B[D0,D1,D2,D3]
DistTensor<double> accum_lvl1_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//accum_lvl1_part2T[D0,D1,D2,D3]
DistTensor<double> accum_lvl1_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//accum_lvl1_part2_0[D0,D1,D2,D3]
DistTensor<double> accum_lvl1_part2_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//accum_lvl1_part2_1[D0,D1,D2,D3]
DistTensor<double> accum_lvl1_part2_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//accum_lvl1_part2_1_lvl1_part3B[D0,D1,D2,D3]
DistTensor<double> accum_lvl1_part2_1_lvl1_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//accum_lvl1_part2_1_lvl1_part3T[D0,D1,D2,D3]
DistTensor<double> accum_lvl1_part2_1_lvl1_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//accum_lvl1_part2_1_lvl2_part3B[D0,D1,D2,D3]
DistTensor<double> accum_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//accum_lvl1_part2_1_lvl2_part3T[D0,D1,D2,D3]
DistTensor<double> accum_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//accum_lvl1_part2_1_lvl2_part3_0[D0,D1,D2,D3]
DistTensor<double> accum_lvl1_part2_1_lvl2_part3_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//accum_lvl1_part2_1_lvl2_part3_1[D0,D1,D2,D3]
DistTensor<double> accum_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//accum_lvl1_part2_1_lvl2_part3_1[D2,D3,*,D1,D0]
DistTensor<double> accum_lvl1_part2_1_lvl2_part3_1_perm30124__D_1__D_2__D_3__S__D_0( dist__D_2__D_3__S__D_1__D_0, g );
accum_lvl1_part2_1_lvl2_part3_1_perm30124__D_1__D_2__D_3__S__D_0.SetLocalPermutation( perm_3_0_1_2_4 );
	//accum_lvl1_part2_1_lvl2_part3_1_Z_temp[D0,D1,D2,D3]
DistTensor<double> accum_lvl1_part2_1_lvl2_part3_1_Z_temp__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//accum_lvl1_part2_1_lvl2_part3_1_Z_temp[D20,D1,*,D3]
DistTensor<double> accum_lvl1_part2_1_lvl2_part3_1_Z_temp__D_2_0__D_1__S__D_3( dist__D_2_0__D_1__S__D_3, g );
	//accum_lvl1_part2_1_lvl2_part3_1_Z_temp[D20,D3,*,D1]
DistTensor<double> accum_lvl1_part2_1_lvl2_part3_1_Z_temp__D_2_0__D_3__S__D_1( dist__D_2_0__D_3__S__D_1, g );
	//accum_lvl1_part2_1_lvl2_part3_2[D0,D1,D2,D3]
DistTensor<double> accum_lvl1_part2_1_lvl2_part3_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> accum_lvl1_part2_1_perm2310__D_2__D_3__D_1__D_0( dist__D_0__D_1__D_2__D_3, g );
accum_lvl1_part2_1_perm2310__D_2__D_3__D_1__D_0.SetLocalPermutation( perm_2_3_1_0 );
	//accum_lvl1_part2_2[D0,D1,D2,D3]
DistTensor<double> accum_lvl1_part2_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//accum_lvl1_part3B[D0,D1,D2,D3]
DistTensor<double> accum_lvl1_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//accum_lvl1_part3T[D0,D1,D2,D3]
DistTensor<double> accum_lvl1_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//accum_lvl1_part3_0[D0,D1,D2,D3]
DistTensor<double> accum_lvl1_part3_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//accum_lvl1_part3_1[D0,D1,D2,D3]
DistTensor<double> accum_lvl1_part3_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//accum_lvl1_part3_1_lvl1_part2B[D0,D1,D2,D3]
DistTensor<double> accum_lvl1_part3_1_lvl1_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//accum_lvl1_part3_1_lvl1_part2T[D0,D1,D2,D3]
DistTensor<double> accum_lvl1_part3_1_lvl1_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//accum_lvl1_part3_1_lvl2_part2B[D0,D1,D2,D3]
DistTensor<double> accum_lvl1_part3_1_lvl2_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//accum_lvl1_part3_1_lvl2_part2T[D0,D1,D2,D3]
DistTensor<double> accum_lvl1_part3_1_lvl2_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//accum_lvl1_part3_1_lvl2_part2_0[D0,D1,D2,D3]
DistTensor<double> accum_lvl1_part3_1_lvl2_part2_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//accum_lvl1_part3_1_lvl2_part2_1[D0,D1,D2,D3]
DistTensor<double> accum_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//accum_lvl1_part3_1_lvl2_part2_1[D1,D0,D2,D3]
DistTensor<double> accum_lvl1_part3_1_lvl2_part2_1__D_1__D_0__D_2__D_3( dist__D_1__D_0__D_2__D_3, g );
	//accum_lvl1_part3_1_lvl2_part2_1[D1,D0,D3,D2]
DistTensor<double> accum_lvl1_part3_1_lvl2_part2_1__D_1__D_0__D_3__D_2( dist__D_1__D_0__D_3__D_2, g );
	//accum_lvl1_part3_1_lvl2_part2_2[D0,D1,D2,D3]
DistTensor<double> accum_lvl1_part3_1_lvl2_part2_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> accum_lvl1_part3_1_perm2013__D_2__D_0__D_1__D_3( dist__D_0__D_1__D_2__D_3, g );
accum_lvl1_part3_1_perm2013__D_2__D_0__D_1__D_3.SetLocalPermutation( perm_2_0_1_3 );
	//accum_lvl1_part3_2[D0,D1,D2,D3]
DistTensor<double> accum_lvl1_part3_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
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
	//Z_temp1[D0,D1,D2,D3]
DistTensor<double> Z_temp1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Z_temp1_lvl0_part2B[D0,D1,D2,D3]
DistTensor<double> Z_temp1_lvl0_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Z_temp1_lvl0_part2T[D0,D1,D2,D3]
DistTensor<double> Z_temp1_lvl0_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Z_temp1_lvl0_part3B[D0,D1,D2,D3]
DistTensor<double> Z_temp1_lvl0_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Z_temp1_lvl0_part3T[D0,D1,D2,D3]
DistTensor<double> Z_temp1_lvl0_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Z_temp1_lvl1_part2B[D0,D1,D2,D3]
DistTensor<double> Z_temp1_lvl1_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Z_temp1_lvl1_part2T[D0,D1,D2,D3]
DistTensor<double> Z_temp1_lvl1_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Z_temp1_lvl1_part2_0[D0,D1,D2,D3]
DistTensor<double> Z_temp1_lvl1_part2_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Z_temp1_lvl1_part2_1[D0,D1,D2,D3]
DistTensor<double> Z_temp1_lvl1_part2_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Z_temp1_lvl1_part2_1_lvl1_part3B[D0,D1,D2,D3]
DistTensor<double> Z_temp1_lvl1_part2_1_lvl1_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Z_temp1_lvl1_part2_1_lvl1_part3T[D0,D1,D2,D3]
DistTensor<double> Z_temp1_lvl1_part2_1_lvl1_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Z_temp1_lvl1_part2_1_lvl2_part3B[D0,D1,D2,D3]
DistTensor<double> Z_temp1_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Z_temp1_lvl1_part2_1_lvl2_part3T[D0,D1,D2,D3]
DistTensor<double> Z_temp1_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Z_temp1_lvl1_part2_1_lvl2_part3_0[D0,D1,D2,D3]
DistTensor<double> Z_temp1_lvl1_part2_1_lvl2_part3_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Z_temp1_lvl1_part2_1_lvl2_part3_1[D0,D1,D2,D3]
DistTensor<double> Z_temp1_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Z_temp1_lvl1_part2_1_lvl2_part3_2[D0,D1,D2,D3]
DistTensor<double> Z_temp1_lvl1_part2_1_lvl2_part3_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Z_temp1_lvl1_part2_2[D0,D1,D2,D3]
DistTensor<double> Z_temp1_lvl1_part2_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Z_temp1_lvl1_part3B[D0,D1,D2,D3]
DistTensor<double> Z_temp1_lvl1_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Z_temp1_lvl1_part3T[D0,D1,D2,D3]
DistTensor<double> Z_temp1_lvl1_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Z_temp1_lvl1_part3_0[D0,D1,D2,D3]
DistTensor<double> Z_temp1_lvl1_part3_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Z_temp1_lvl1_part3_1[D0,D1,D2,D3]
DistTensor<double> Z_temp1_lvl1_part3_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Z_temp1_lvl1_part3_1_lvl1_part2B[D0,D1,D2,D3]
DistTensor<double> Z_temp1_lvl1_part3_1_lvl1_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Z_temp1_lvl1_part3_1_lvl1_part2T[D0,D1,D2,D3]
DistTensor<double> Z_temp1_lvl1_part3_1_lvl1_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Z_temp1_lvl1_part3_1_lvl2_part2B[D0,D1,D2,D3]
DistTensor<double> Z_temp1_lvl1_part3_1_lvl2_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Z_temp1_lvl1_part3_1_lvl2_part2T[D0,D1,D2,D3]
DistTensor<double> Z_temp1_lvl1_part3_1_lvl2_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Z_temp1_lvl1_part3_1_lvl2_part2_0[D0,D1,D2,D3]
DistTensor<double> Z_temp1_lvl1_part3_1_lvl2_part2_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Z_temp1_lvl1_part3_1_lvl2_part2_1[D0,D1,D2,D3]
DistTensor<double> Z_temp1_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Z_temp1_lvl1_part3_1_lvl2_part2_1[D0,D1,D3,D2]
DistTensor<double> Z_temp1_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_3__D_2( dist__D_0__D_1__D_3__D_2, g );
	//Z_temp1_lvl1_part3_1_lvl2_part2_2[D0,D1,D2,D3]
DistTensor<double> Z_temp1_lvl1_part3_1_lvl2_part2_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Z_temp1_lvl1_part3_2[D0,D1,D2,D3]
DistTensor<double> Z_temp1_lvl1_part3_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> Z_temp1_perm1302__D_1__D_3__D_0__D_2( dist__D_0__D_1__D_2__D_3, g );
Z_temp1_perm1302__D_1__D_3__D_0__D_2.SetLocalPermutation( perm_1_3_0_2 );
	//Z_temp2[D0,D1,D2,D3]
DistTensor<double> Z_temp2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Z_temp2_lvl0_part2B[D0,D1,D2,D3]
DistTensor<double> Z_temp2_lvl0_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Z_temp2_lvl0_part2T[D0,D1,D2,D3]
DistTensor<double> Z_temp2_lvl0_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Z_temp2_lvl1_part2B[D0,D1,D2,D3]
DistTensor<double> Z_temp2_lvl1_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Z_temp2_lvl1_part2T[D0,D1,D2,D3]
DistTensor<double> Z_temp2_lvl1_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Z_temp2_lvl1_part2_0[D0,D1,D2,D3]
DistTensor<double> Z_temp2_lvl1_part2_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Z_temp2_lvl1_part2_1[D0,D1,D2,D3]
DistTensor<double> Z_temp2_lvl1_part2_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Z_temp2_lvl1_part2_1_lvl1_part3B[D0,D1,D2,D3]
DistTensor<double> Z_temp2_lvl1_part2_1_lvl1_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Z_temp2_lvl1_part2_1_lvl1_part3T[D0,D1,D2,D3]
DistTensor<double> Z_temp2_lvl1_part2_1_lvl1_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Z_temp2_lvl1_part2_1_lvl2_part3B[D0,D1,D2,D3]
DistTensor<double> Z_temp2_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Z_temp2_lvl1_part2_1_lvl2_part3T[D0,D1,D2,D3]
DistTensor<double> Z_temp2_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Z_temp2_lvl1_part2_1_lvl2_part3_0[D0,D1,D2,D3]
DistTensor<double> Z_temp2_lvl1_part2_1_lvl2_part3_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Z_temp2_lvl1_part2_1_lvl2_part3_1[D0,D1,D2,D3]
DistTensor<double> Z_temp2_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Z_temp2_lvl1_part2_1_lvl2_part3_2[D0,D1,D2,D3]
DistTensor<double> Z_temp2_lvl1_part2_1_lvl2_part3_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Z_temp2_lvl1_part2_2[D0,D1,D2,D3]
DistTensor<double> Z_temp2_lvl1_part2_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> Z_temp2_perm3102__D_3__D_1__D_0__D_2( dist__D_0__D_1__D_2__D_3, g );
Z_temp2_perm3102__D_3__D_1__D_0__D_2.SetLocalPermutation( perm_3_1_0_2 );
	//y_abef[D0,D1,D2,D3]
DistTensor<double> y_abef__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
ObjShape v_femn__D_0__D_1__D_2__D_3_tmpShape_Z( 4 );
v_femn__D_0__D_1__D_2__D_3_tmpShape_Z[ 0 ] = n_v;
v_femn__D_0__D_1__D_2__D_3_tmpShape_Z[ 1 ] = n_v;
v_femn__D_0__D_1__D_2__D_3_tmpShape_Z[ 2 ] = n_o;
v_femn__D_0__D_1__D_2__D_3_tmpShape_Z[ 3 ] = n_o;
v_femn__D_0__D_1__D_2__D_3.ResizeTo( v_femn__D_0__D_1__D_2__D_3_tmpShape_Z );
ObjShape Q_mnij__D_0__D_1__D_2__D_3_tmpShape_Z( 4 );
Q_mnij__D_0__D_1__D_2__D_3_tmpShape_Z[ 0 ] = n_o;
Q_mnij__D_0__D_1__D_2__D_3_tmpShape_Z[ 1 ] = n_o;
Q_mnij__D_0__D_1__D_2__D_3_tmpShape_Z[ 2 ] = n_o;
Q_mnij__D_0__D_1__D_2__D_3_tmpShape_Z[ 3 ] = n_o;
Q_mnij__D_0__D_1__D_2__D_3.ResizeTo( Q_mnij__D_0__D_1__D_2__D_3_tmpShape_Z );
ObjShape Tau_efmn__D_0__D_1__D_2__D_3_tmpShape_Z( 4 );
Tau_efmn__D_0__D_1__D_2__D_3_tmpShape_Z[ 0 ] = n_v;
Tau_efmn__D_0__D_1__D_2__D_3_tmpShape_Z[ 1 ] = n_v;
Tau_efmn__D_0__D_1__D_2__D_3_tmpShape_Z[ 2 ] = n_o;
Tau_efmn__D_0__D_1__D_2__D_3_tmpShape_Z[ 3 ] = n_o;
Tau_efmn__D_0__D_1__D_2__D_3.ResizeTo( Tau_efmn__D_0__D_1__D_2__D_3_tmpShape_Z );
// y_abef has 4 dims
ObjShape y_abef__D_0__D_1__D_2__D_3_tmpShape_Z( 4 );
y_abef__D_0__D_1__D_2__D_3_tmpShape_Z[ 0 ] = n_v;
y_abef__D_0__D_1__D_2__D_3_tmpShape_Z[ 1 ] = n_v;
y_abef__D_0__D_1__D_2__D_3_tmpShape_Z[ 2 ] = n_v;
y_abef__D_0__D_1__D_2__D_3_tmpShape_Z[ 3 ] = n_v;
y_abef__D_0__D_1__D_2__D_3.ResizeTo( y_abef__D_0__D_1__D_2__D_3_tmpShape_Z );
MakeUniform( y_abef__D_0__D_1__D_2__D_3 );
// Z_abij has 4 dims
ObjShape Z_abij__D_0__D_1__D_2__D_3_tmpShape_Z( 4 );
Z_abij__D_0__D_1__D_2__D_3_tmpShape_Z[ 0 ] = n_v;
Z_abij__D_0__D_1__D_2__D_3_tmpShape_Z[ 1 ] = n_v;
Z_abij__D_0__D_1__D_2__D_3_tmpShape_Z[ 2 ] = n_o;
Z_abij__D_0__D_1__D_2__D_3_tmpShape_Z[ 3 ] = n_o;
Z_abij__D_0__D_1__D_2__D_3.ResizeTo( Z_abij__D_0__D_1__D_2__D_3_tmpShape_Z );
MakeUniform( Z_abij__D_0__D_1__D_2__D_3 );
ObjShape r_bmfe__D_0__D_1__D_2__D_3_tmpShape_Z( 4 );
r_bmfe__D_0__D_1__D_2__D_3_tmpShape_Z[ 0 ] = n_v;
r_bmfe__D_0__D_1__D_2__D_3_tmpShape_Z[ 1 ] = n_o;
r_bmfe__D_0__D_1__D_2__D_3_tmpShape_Z[ 2 ] = n_v;
r_bmfe__D_0__D_1__D_2__D_3_tmpShape_Z[ 3 ] = n_v;
r_bmfe__D_0__D_1__D_2__D_3.ResizeTo( r_bmfe__D_0__D_1__D_2__D_3_tmpShape_Z );
ObjShape t_fj__D_0_1__D_2_3_tmpShape_Z( 2 );
t_fj__D_0_1__D_2_3_tmpShape_Z[ 0 ] = n_v;
t_fj__D_0_1__D_2_3_tmpShape_Z[ 1 ] = n_o;
t_fj__D_0_1__D_2_3.ResizeTo( t_fj__D_0_1__D_2_3_tmpShape_Z );
ObjShape P_jimb__D_0__D_1__D_2__D_3_tmpShape_Z( 4 );
P_jimb__D_0__D_1__D_2__D_3_tmpShape_Z[ 0 ] = n_o;
P_jimb__D_0__D_1__D_2__D_3_tmpShape_Z[ 1 ] = n_o;
P_jimb__D_0__D_1__D_2__D_3_tmpShape_Z[ 2 ] = n_o;
P_jimb__D_0__D_1__D_2__D_3_tmpShape_Z[ 3 ] = n_v;
P_jimb__D_0__D_1__D_2__D_3.ResizeTo( P_jimb__D_0__D_1__D_2__D_3_tmpShape_Z );
ObjShape F_ae__D_0_1__D_2_3_tmpShape_Z( 2 );
F_ae__D_0_1__D_2_3_tmpShape_Z[ 0 ] = n_v;
F_ae__D_0_1__D_2_3_tmpShape_Z[ 1 ] = n_v;
F_ae__D_0_1__D_2_3.ResizeTo( F_ae__D_0_1__D_2_3_tmpShape_Z );
ObjShape T_bfnj__D_0__D_1__D_2__D_3_tmpShape_Z( 4 );
T_bfnj__D_0__D_1__D_2__D_3_tmpShape_Z[ 0 ] = n_v;
T_bfnj__D_0__D_1__D_2__D_3_tmpShape_Z[ 1 ] = n_v;
T_bfnj__D_0__D_1__D_2__D_3_tmpShape_Z[ 2 ] = n_o;
T_bfnj__D_0__D_1__D_2__D_3_tmpShape_Z[ 3 ] = n_o;
T_bfnj__D_0__D_1__D_2__D_3.ResizeTo( T_bfnj__D_0__D_1__D_2__D_3_tmpShape_Z );
ObjShape G_mi__D_0_1__D_2_3_tmpShape_Z( 2 );
G_mi__D_0_1__D_2_3_tmpShape_Z[ 0 ] = n_o;
G_mi__D_0_1__D_2_3_tmpShape_Z[ 1 ] = n_o;
G_mi__D_0_1__D_2_3.ResizeTo( G_mi__D_0_1__D_2_3_tmpShape_Z );
ObjShape W_bmje__D_0__D_1__D_2__D_3_tmpShape_Z( 4 );
W_bmje__D_0__D_1__D_2__D_3_tmpShape_Z[ 0 ] = n_v;
W_bmje__D_0__D_1__D_2__D_3_tmpShape_Z[ 1 ] = n_o;
W_bmje__D_0__D_1__D_2__D_3_tmpShape_Z[ 2 ] = n_o;
W_bmje__D_0__D_1__D_2__D_3_tmpShape_Z[ 3 ] = n_v;
W_bmje__D_0__D_1__D_2__D_3.ResizeTo( W_bmje__D_0__D_1__D_2__D_3_tmpShape_Z );
ObjShape X_bmej__D_0__D_1__D_2__D_3_tmpShape_Z( 4 );
X_bmej__D_0__D_1__D_2__D_3_tmpShape_Z[ 0 ] = n_v;
X_bmej__D_0__D_1__D_2__D_3_tmpShape_Z[ 1 ] = n_o;
X_bmej__D_0__D_1__D_2__D_3_tmpShape_Z[ 2 ] = n_v;
X_bmej__D_0__D_1__D_2__D_3_tmpShape_Z[ 3 ] = n_o;
X_bmej__D_0__D_1__D_2__D_3.ResizeTo( X_bmej__D_0__D_1__D_2__D_3_tmpShape_Z );
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

			Permute( Tau_efmn_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3, Tau_efmn_lvl1_part2_1_lvl2_part3_1_perm0213__D_0__D_2__D_1__D_3 );
			   // t_fj_lvl2_part1_1[D01,D3] <- t_fj_lvl2_part1_1[D01,D23]
			t_fj_lvl2_part1_1__D_0_1__D_3.AlignModesWith( modes_0_1, Tau_efmn_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3, modes_1_3 );
			t_fj_lvl2_part1_1__D_0_1__D_3.AllToAllRedistFrom( t_fj_lvl2_part1_1__D_0_1__D_2_3, modes_2_3 );
			   // t_fj_lvl2_part1_1[D1,D3] <- t_fj_lvl2_part1_1[D01,D3]
			t_fj_lvl2_part1_1__D_1__D_3.AlignModesWith( modes_0_1, Tau_efmn_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3, modes_1_3 );
			t_fj_lvl2_part1_1__D_1__D_3.AllToAllRedistFrom( t_fj_lvl2_part1_1__D_0_1__D_3, modes_0_1 );
			t_fj_lvl2_part1_1__D_0_1__D_3.EmptyData();
			   // t_fj_lvl1_part1_1[D0,D2] <- t_fj_lvl1_part1_1[D01,D23]
			t_fj_lvl1_part1_1__D_0__D_2.AlignModesWith( modes_0_1, Tau_efmn_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3, modes_0_2 );
			t_fj_lvl1_part1_1__D_0__D_2.AllGatherRedistFrom( t_fj_lvl1_part1_1__D_0_1__D_2_3, modes_1_3 );
			   // 1.0 * t_fj_lvl1_part1_1[D0,D2]_em * t_fj_lvl2_part1_1[D1,D3]_fn + 1.0 * Tau_efmn_lvl1_part2_1_lvl2_part3_1[D0,D1,D2,D3]_emfn
PROFILE_SECTION("COMPUTE");
PROFILE_FLOPS(2*prod(Tau_efmn_lvl1_part2_1_lvl2_part3_1_perm0213__D_0__D_2__D_1__D_3.Shape())*1);
			LocalContractAndLocalEliminate(1.0, t_fj_lvl1_part1_1__D_0__D_2.LockedTensor(), indices_em, false,
				t_fj_lvl2_part1_1__D_1__D_3.LockedTensor(), indices_fn, false,
				1.0, Tau_efmn_lvl1_part2_1_lvl2_part3_1_perm0213__D_0__D_2__D_1__D_3.Tensor(), indices_emfn, false);
PROFILE_STOP;
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


#ifdef CORRECTNESS
    DistTensor<double> diff_Tau(dist__D_0__D_1__D_2__D_3, g);
    diff_Tau.ResizeTo(check_Tau);
    Diff(check_Tau, Tau_efmn__D_0__D_1__D_2__D_3, diff_Tau);
   norm = 1.0;
   norm = Norm(diff_Tau);
   if (commRank == 0){
     std::cout << "NORM_Tau " << norm << std::endl;
   }
#endif
overwrite_tmpShape_W = T_bfnj__D_0__D_1__D_2__D_3.Shape();
W_temp1__D_0__D_1__D_2__D_3.ResizeTo( overwrite_tmpShape_W );
overwrite_tmpShape_W = v_femn__D_0__D_1__D_2__D_3.Shape();
W_temp2__D_0__D_1__D_2__D_3.ResizeTo( overwrite_tmpShape_W );
overwrite_tmpShape_W = r_bmfe__D_0__D_1__D_2__D_3.Shape();
W_temp4__D_0__D_1__D_2__D_3.ResizeTo( overwrite_tmpShape_W );
//**** (out of 1)
//**** Is real	0 shadows
	//Outputs:
	//  W_bmje__D_0__D_1__D_2__D_3

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
PROFILE_SECTION("COMPUTE");
PROFILE_FLOPS(3*prod(w_bmje_lvl1_part1_1_lvl2_part2_1__D_0__D_1__D_2__D_3.Shape()));
			YAxpPx( 2.0, w_bmje_lvl1_part1_1_lvl2_part2_1__D_0__D_1__D_2__D_3, -1.0, x_bmej_lvl1_part1_1_lvl2_part3_1__D_0__D_1__D_3__D_2, perm_0_1_3_2, W_bmje_lvl1_part1_1_lvl2_part2_1__D_0__D_1__D_2__D_3 );
PROFILE_STOP;
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
	Permute( W_bmje__D_0__D_1__D_2__D_3, W_bmje_perm3102__D_3__D_1__D_0__D_2 );
	//**** (out of 1)
	//**** Is real	0 shadows
		//Outputs:
		//  W_temp1__D_0__D_1__D_2__D_3
	PartitionDown(T_bfnj__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part2T__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part2B__D_0__D_1__D_2__D_3, 2, 0);
	PartitionDown(Tau_efmn__D_0__D_1__D_2__D_3, Tau_efmn_lvl1_part2T__D_0__D_1__D_2__D_3, Tau_efmn_lvl1_part2B__D_0__D_1__D_2__D_3, 2, 0);
	PartitionDown(T_bfnj__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part3T__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part3B__D_0__D_1__D_2__D_3, 3, 0);
	PartitionDown(W_temp1__D_0__D_1__D_2__D_3, W_temp1_lvl1_part2T__D_0__D_1__D_2__D_3, W_temp1_lvl1_part2B__D_0__D_1__D_2__D_3, 2, 0);
	while(W_temp1_lvl1_part2T__D_0__D_1__D_2__D_3.Dimension(2) < W_temp1__D_0__D_1__D_2__D_3.Dimension(2))
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
		( W_temp1_lvl1_part2T__D_0__D_1__D_2__D_3,  W_temp1_lvl1_part2_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       W_temp1_lvl1_part2_1__D_0__D_1__D_2__D_3,
		  W_temp1_lvl1_part2B__D_0__D_1__D_2__D_3, W_temp1_lvl1_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );

		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  W_temp1_lvl1_part2_1__D_0__D_1__D_2__D_3
		PartitionDown(T_bfnj_lvl1_part2_1__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, 3, 0);
		PartitionDown(Tau_efmn_lvl1_part2_1__D_0__D_1__D_2__D_3, Tau_efmn_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3, Tau_efmn_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, 3, 0);
		PartitionDown(T_bfnj_lvl1_part3_1__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part3_1_lvl2_part2T__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part3_1_lvl2_part2B__D_0__D_1__D_2__D_3, 2, 0);
		PartitionDown(W_temp1_lvl1_part2_1__D_0__D_1__D_2__D_3, W_temp1_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3, W_temp1_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, 3, 0);
		while(W_temp1_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3.Dimension(3) < W_temp1_lvl1_part2_1__D_0__D_1__D_2__D_3.Dimension(3))
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
			( W_temp1_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3,  W_temp1_lvl1_part2_1_lvl2_part3_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       W_temp1_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3,
			  W_temp1_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, W_temp1_lvl1_part2_1_lvl2_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );

			   // T_bfnj_lvl1_part3_1_lvl2_part2_1[D0,D1,D3,D2] <- T_bfnj_lvl1_part3_1_lvl2_part2_1[D0,D1,D2,D3]
			T_bfnj_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_3__D_2.AlignModesWith( modes_0_1_2_3, T_bfnj_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3, modes_0_1_3_2 );
			T_bfnj_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_3__D_2.AllToAllRedistFrom( T_bfnj_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_2__D_3, modes_2_3 );
PROFILE_SECTION("COMPUTE");
PROFILE_FLOPS(4*prod(T_bfnj_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3.Shape()));
			ZAxpBypPx( 0.5, T_bfnj_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3, -1.0, Tau_efmn_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_3__D_2, perm_0_1_3_2, W_temp1_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3 );
PROFILE_STOP;
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
			( W_temp1_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3,  W_temp1_lvl1_part2_1_lvl2_part3_0__D_0__D_1__D_2__D_3,
			       W_temp1_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  W_temp1_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, W_temp1_lvl1_part2_1_lvl2_part3_2__D_0__D_1__D_2__D_3, 3 );

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
		( W_temp1_lvl1_part2T__D_0__D_1__D_2__D_3,  W_temp1_lvl1_part2_0__D_0__D_1__D_2__D_3,
		       W_temp1_lvl1_part2_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  W_temp1_lvl1_part2B__D_0__D_1__D_2__D_3, W_temp1_lvl1_part2_2__D_0__D_1__D_2__D_3, 2 );

	}
	//****
	//**** (out of 1)
	//**** Is real	0 shadows
		//Outputs:
		//  W_temp2__D_0__D_1__D_2__D_3
	PartitionDown(v_femn__D_0__D_1__D_2__D_3, v_femn_lvl1_part2T__D_0__D_1__D_2__D_3, v_femn_lvl1_part2B__D_0__D_1__D_2__D_3, 2, 0);
	PartitionDown(v_femn__D_0__D_1__D_2__D_3, v_femn_lvl1_part3T__D_0__D_1__D_2__D_3, v_femn_lvl1_part3B__D_0__D_1__D_2__D_3, 3, 0);
	PartitionDown(W_temp2__D_0__D_1__D_2__D_3, W_temp2_lvl1_part2T__D_0__D_1__D_2__D_3, W_temp2_lvl1_part2B__D_0__D_1__D_2__D_3, 2, 0);
	while(W_temp2_lvl1_part2T__D_0__D_1__D_2__D_3.Dimension(2) < W_temp2__D_0__D_1__D_2__D_3.Dimension(2))
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
		( W_temp2_lvl1_part2T__D_0__D_1__D_2__D_3,  W_temp2_lvl1_part2_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       W_temp2_lvl1_part2_1__D_0__D_1__D_2__D_3,
		  W_temp2_lvl1_part2B__D_0__D_1__D_2__D_3, W_temp2_lvl1_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );

		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  W_temp2_lvl1_part2_1__D_0__D_1__D_2__D_3
		PartitionDown(v_femn_lvl1_part2_1__D_0__D_1__D_2__D_3, v_femn_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3, v_femn_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, 3, 0);
		PartitionDown(v_femn_lvl1_part3_1__D_0__D_1__D_2__D_3, v_femn_lvl1_part3_1_lvl2_part2T__D_0__D_1__D_2__D_3, v_femn_lvl1_part3_1_lvl2_part2B__D_0__D_1__D_2__D_3, 2, 0);
		PartitionDown(W_temp2_lvl1_part2_1__D_0__D_1__D_2__D_3, W_temp2_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3, W_temp2_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, 3, 0);
		while(W_temp2_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3.Dimension(3) < W_temp2_lvl1_part2_1__D_0__D_1__D_2__D_3.Dimension(3))
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
			( W_temp2_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3,  W_temp2_lvl1_part2_1_lvl2_part3_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       W_temp2_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3,
			  W_temp2_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, W_temp2_lvl1_part2_1_lvl2_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );

			   // v_femn_lvl1_part3_1_lvl2_part2_1[D0,D1,D3,D2] <- v_femn_lvl1_part3_1_lvl2_part2_1[D0,D1,D2,D3]
			v_femn_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_3__D_2.AlignModesWith( modes_0_1_2_3, v_femn_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3, modes_0_1_3_2 );
			v_femn_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_3__D_2.AllToAllRedistFrom( v_femn_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_2__D_3, modes_2_3 );
PROFILE_SECTION("COMPUTE");
PROFILE_FLOPS(3*prod(v_femn_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3.Shape()));
			YAxpPx( -1.0, v_femn_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3, 2.0, v_femn_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_3__D_2, perm_0_1_3_2, W_temp2_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3 );
PROFILE_STOP;
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
			( W_temp2_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3,  W_temp2_lvl1_part2_1_lvl2_part3_0__D_0__D_1__D_2__D_3,
			       W_temp2_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  W_temp2_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, W_temp2_lvl1_part2_1_lvl2_part3_2__D_0__D_1__D_2__D_3, 3 );

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
		( W_temp2_lvl1_part2T__D_0__D_1__D_2__D_3,  W_temp2_lvl1_part2_0__D_0__D_1__D_2__D_3,
		       W_temp2_lvl1_part2_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  W_temp2_lvl1_part2B__D_0__D_1__D_2__D_3, W_temp2_lvl1_part2_2__D_0__D_1__D_2__D_3, 2 );

	}
	//****
	//**** (out of 1)
	//**** Is real	0 shadows
		//Outputs:
		//  W_bmje_perm3102__D_3__D_1__D_0__D_2
	PartitionDown(W_temp2__D_0__D_1__D_2__D_3, W_temp2_lvl1_part0T__D_0__D_1__D_2__D_3, W_temp2_lvl1_part0B__D_0__D_1__D_2__D_3, 0, 0);
	PartitionDown(W_temp1__D_0__D_1__D_2__D_3, W_temp1_lvl1_part1T__D_0__D_1__D_2__D_3, W_temp1_lvl1_part1B__D_0__D_1__D_2__D_3, 1, 0);
	while(W_temp2_lvl1_part0T__D_0__D_1__D_2__D_3.Dimension(0) < W_temp2__D_0__D_1__D_2__D_3.Dimension(0))
	{
		RepartitionDown
		( W_temp2_lvl1_part0T__D_0__D_1__D_2__D_3,  W_temp2_lvl1_part0_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       W_temp2_lvl1_part0_1__D_0__D_1__D_2__D_3,
		  W_temp2_lvl1_part0B__D_0__D_1__D_2__D_3, W_temp2_lvl1_part0_2__D_0__D_1__D_2__D_3, 0, blkSize );
		RepartitionDown
		( W_temp1_lvl1_part1T__D_0__D_1__D_2__D_3,  W_temp1_lvl1_part1_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       W_temp1_lvl1_part1_1__D_0__D_1__D_2__D_3,
		  W_temp1_lvl1_part1B__D_0__D_1__D_2__D_3, W_temp1_lvl1_part1_2__D_0__D_1__D_2__D_3, 1, blkSize );

		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  W_bmje_perm3102__D_3__D_1__D_0__D_2
		PartitionDown(W_temp2_lvl1_part0_1__D_0__D_1__D_2__D_3, W_temp2_lvl1_part0_1_lvl2_part3T__D_0__D_1__D_2__D_3, W_temp2_lvl1_part0_1_lvl2_part3B__D_0__D_1__D_2__D_3, 3, 0);
		PartitionDown(W_temp1_lvl1_part1_1__D_0__D_1__D_2__D_3, W_temp1_lvl1_part1_1_lvl2_part2T__D_0__D_1__D_2__D_3, W_temp1_lvl1_part1_1_lvl2_part2B__D_0__D_1__D_2__D_3, 2, 0);
		while(W_temp2_lvl1_part0_1_lvl2_part3T__D_0__D_1__D_2__D_3.Dimension(3) < W_temp2_lvl1_part0_1__D_0__D_1__D_2__D_3.Dimension(3))
		{
			RepartitionDown
			( W_temp2_lvl1_part0_1_lvl2_part3T__D_0__D_1__D_2__D_3,  W_temp2_lvl1_part0_1_lvl2_part3_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       W_temp2_lvl1_part0_1_lvl2_part3_1__D_0__D_1__D_2__D_3,
			  W_temp2_lvl1_part0_1_lvl2_part3B__D_0__D_1__D_2__D_3, W_temp2_lvl1_part0_1_lvl2_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );
			RepartitionDown
			( W_temp1_lvl1_part1_1_lvl2_part2T__D_0__D_1__D_2__D_3,  W_temp1_lvl1_part1_1_lvl2_part2_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       W_temp1_lvl1_part1_1_lvl2_part2_1__D_0__D_1__D_2__D_3,
			  W_temp1_lvl1_part1_1_lvl2_part2B__D_0__D_1__D_2__D_3, W_temp1_lvl1_part1_1_lvl2_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );

			   // W_temp1_lvl1_part1_1_lvl2_part2_1[D0,D1,D3,D2] <- W_temp1_lvl1_part1_1_lvl2_part2_1[D0,D1,D2,D3]
			W_temp1_lvl1_part1_1_lvl2_part2_1__D_0__D_1__D_3__D_2.AlignModesWith( modes_0_3, W_bmje__D_0__D_1__D_2__D_3, modes_0_2 );
			W_temp1_lvl1_part1_1_lvl2_part2_1__D_0__D_1__D_3__D_2.AllToAllRedistFrom( W_temp1_lvl1_part1_1_lvl2_part2_1__D_0__D_1__D_2__D_3, modes_2_3 );
			   // W_temp1_lvl1_part1_1_lvl2_part2_1[D0,*,*,D2] <- W_temp1_lvl1_part1_1_lvl2_part2_1[D0,D1,D3,D2]
			W_temp1_lvl1_part1_1_lvl2_part2_1_perm1203__S__S__D_0__D_2.AlignModesWith( modes_0_3, W_bmje__D_0__D_1__D_2__D_3, modes_0_2 );
			W_temp1_lvl1_part1_1_lvl2_part2_1_perm1203__S__S__D_0__D_2.AllGatherRedistFrom( W_temp1_lvl1_part1_1_lvl2_part2_1__D_0__D_1__D_3__D_2, modes_1_3 );
			W_temp1_lvl1_part1_1_lvl2_part2_1__D_0__D_1__D_3__D_2.EmptyData();
			   // W_temp2_lvl1_part0_1_lvl2_part3_1[D0,*,D2,D3] <- W_temp2_lvl1_part0_1_lvl2_part3_1[D0,D1,D2,D3]
			W_temp2_lvl1_part0_1_lvl2_part3_1__D_0__S__D_2__D_3.AlignModesWith( modes_1_2, W_bmje__D_0__D_1__D_2__D_3, modes_3_1 );
			W_temp2_lvl1_part0_1_lvl2_part3_1__D_0__S__D_2__D_3.AllGatherRedistFrom( W_temp2_lvl1_part0_1_lvl2_part3_1__D_0__D_1__D_2__D_3, modes_1 );
			   // W_temp2_lvl1_part0_1_lvl2_part3_1[D0,*,D21,D3] <- W_temp2_lvl1_part0_1_lvl2_part3_1[D0,*,D2,D3]
			W_temp2_lvl1_part0_1_lvl2_part3_1__D_0__S__D_2_1__D_3.AlignModesWith( modes_1_2, W_bmje__D_0__D_1__D_2__D_3, modes_3_1 );
			W_temp2_lvl1_part0_1_lvl2_part3_1__D_0__S__D_2_1__D_3.LocalRedistFrom( W_temp2_lvl1_part0_1_lvl2_part3_1__D_0__S__D_2__D_3 );
			W_temp2_lvl1_part0_1_lvl2_part3_1__D_0__S__D_2__D_3.EmptyData();
			   // W_temp2_lvl1_part0_1_lvl2_part3_1[D0,D3,D21,*] <- W_temp2_lvl1_part0_1_lvl2_part3_1[D0,*,D21,D3]
			W_temp2_lvl1_part0_1_lvl2_part3_1__D_0__D_3__D_2_1__S.AlignModesWith( modes_1_2, W_bmje__D_0__D_1__D_2__D_3, modes_3_1 );
			W_temp2_lvl1_part0_1_lvl2_part3_1__D_0__D_3__D_2_1__S.AllToAllRedistFrom( W_temp2_lvl1_part0_1_lvl2_part3_1__D_0__S__D_2_1__D_3, modes_3 );
			W_temp2_lvl1_part0_1_lvl2_part3_1__D_0__S__D_2_1__D_3.EmptyData();
			   // W_temp2_lvl1_part0_1_lvl2_part3_1[D0,D3,D12,*] <- W_temp2_lvl1_part0_1_lvl2_part3_1[D0,D3,D21,*]
			W_temp2_lvl1_part0_1_lvl2_part3_1__D_0__D_3__D_1_2__S.AlignModesWith( modes_1_2, W_bmje__D_0__D_1__D_2__D_3, modes_3_1 );
			W_temp2_lvl1_part0_1_lvl2_part3_1__D_0__D_3__D_1_2__S.AllToAllRedistFrom( W_temp2_lvl1_part0_1_lvl2_part3_1__D_0__D_3__D_2_1__S, modes_1_2 );
			W_temp2_lvl1_part0_1_lvl2_part3_1__D_0__D_3__D_2_1__S.EmptyData();
			   // W_temp2_lvl1_part0_1_lvl2_part3_1[*,D3,D1,*] <- W_temp2_lvl1_part0_1_lvl2_part3_1[D0,D3,D12,*]
			W_temp2_lvl1_part0_1_lvl2_part3_1_perm1203__D_3__D_1__S__S.AlignModesWith( modes_1_2, W_bmje__D_0__D_1__D_2__D_3, modes_3_1 );
			W_temp2_lvl1_part0_1_lvl2_part3_1_perm1203__D_3__D_1__S__S.AllGatherRedistFrom( W_temp2_lvl1_part0_1_lvl2_part3_1__D_0__D_3__D_1_2__S, modes_0_2 );
			W_temp2_lvl1_part0_1_lvl2_part3_1__D_0__D_3__D_1_2__S.EmptyData();
			   // 1.0 * W_temp2_lvl1_part0_1_lvl2_part3_1[*,D3,D1,*]_emfn * W_temp1_lvl1_part1_1_lvl2_part2_1[D0,*,*,D2]_fnbj + 1.0 * W_bmje[D0,D1,D2,D3]_embj
PROFILE_SECTION("COMPUTE");
PROFILE_FLOPS(2*prod(W_bmje_perm3102__D_3__D_1__D_0__D_2.Shape())*W_temp2_lvl1_part0_1_lvl2_part3_1_perm1203__D_3__D_1__S__S.Dimension(0)*W_temp2_lvl1_part0_1_lvl2_part3_1_perm1203__D_3__D_1__S__S.Dimension(3));
			LocalContractAndLocalEliminate(1.0, W_temp2_lvl1_part0_1_lvl2_part3_1_perm1203__D_3__D_1__S__S.LockedTensor(), indices_emfn, false,
				W_temp1_lvl1_part1_1_lvl2_part2_1_perm1203__S__S__D_0__D_2.LockedTensor(), indices_fnbj, false,
				1.0, W_bmje_perm3102__D_3__D_1__D_0__D_2.Tensor(), indices_embj, false);
PROFILE_STOP;
			W_temp2_lvl1_part0_1_lvl2_part3_1_perm1203__D_3__D_1__S__S.EmptyData();
			W_temp1_lvl1_part1_1_lvl2_part2_1_perm1203__S__S__D_0__D_2.EmptyData();

			SlidePartitionDown
			( W_temp2_lvl1_part0_1_lvl2_part3T__D_0__D_1__D_2__D_3,  W_temp2_lvl1_part0_1_lvl2_part3_0__D_0__D_1__D_2__D_3,
			       W_temp2_lvl1_part0_1_lvl2_part3_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  W_temp2_lvl1_part0_1_lvl2_part3B__D_0__D_1__D_2__D_3, W_temp2_lvl1_part0_1_lvl2_part3_2__D_0__D_1__D_2__D_3, 3 );
			SlidePartitionDown
			( W_temp1_lvl1_part1_1_lvl2_part2T__D_0__D_1__D_2__D_3,  W_temp1_lvl1_part1_1_lvl2_part2_0__D_0__D_1__D_2__D_3,
			       W_temp1_lvl1_part1_1_lvl2_part2_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  W_temp1_lvl1_part1_1_lvl2_part2B__D_0__D_1__D_2__D_3, W_temp1_lvl1_part1_1_lvl2_part2_2__D_0__D_1__D_2__D_3, 2 );

		}
		//****

		SlidePartitionDown
		( W_temp2_lvl1_part0T__D_0__D_1__D_2__D_3,  W_temp2_lvl1_part0_0__D_0__D_1__D_2__D_3,
		       W_temp2_lvl1_part0_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  W_temp2_lvl1_part0B__D_0__D_1__D_2__D_3, W_temp2_lvl1_part0_2__D_0__D_1__D_2__D_3, 0 );
		SlidePartitionDown
		( W_temp1_lvl1_part1T__D_0__D_1__D_2__D_3,  W_temp1_lvl1_part1_0__D_0__D_1__D_2__D_3,
		       W_temp1_lvl1_part1_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  W_temp1_lvl1_part1B__D_0__D_1__D_2__D_3, W_temp1_lvl1_part1_2__D_0__D_1__D_2__D_3, 1 );

	}
	//****
	W_temp2__D_0__D_1__D_2__D_3.EmptyData();
	W_temp1__D_0__D_1__D_2__D_3.EmptyData();
	Permute( W_bmje_perm3102__D_3__D_1__D_0__D_2, W_bmje__D_0__D_1__D_2__D_3 );
	W_bmje_perm3102__D_3__D_1__D_0__D_2.EmptyData();
	//**** (out of 1)
	//**** Is real	0 shadows
		//Outputs:
		//  W_temp4__D_0__D_1__D_2__D_3
	PartitionDown(r_bmfe__D_0__D_1__D_2__D_3, r_bmfe_lvl1_part0T__D_0__D_1__D_2__D_3, r_bmfe_lvl1_part0B__D_0__D_1__D_2__D_3, 0, 0);
	PartitionDown(W_temp4__D_0__D_1__D_2__D_3, W_temp4_lvl1_part0T__D_0__D_1__D_2__D_3, W_temp4_lvl1_part0B__D_0__D_1__D_2__D_3, 0, 0);
	while(W_temp4_lvl1_part0T__D_0__D_1__D_2__D_3.Dimension(0) < W_temp4__D_0__D_1__D_2__D_3.Dimension(0))
	{
		RepartitionDown
		( r_bmfe_lvl1_part0T__D_0__D_1__D_2__D_3,  r_bmfe_lvl1_part0_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       r_bmfe_lvl1_part0_1__D_0__D_1__D_2__D_3,
		  r_bmfe_lvl1_part0B__D_0__D_1__D_2__D_3, r_bmfe_lvl1_part0_2__D_0__D_1__D_2__D_3, 0, blkSize );
		RepartitionDown
		( W_temp4_lvl1_part0T__D_0__D_1__D_2__D_3,  W_temp4_lvl1_part0_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       W_temp4_lvl1_part0_1__D_0__D_1__D_2__D_3,
		  W_temp4_lvl1_part0B__D_0__D_1__D_2__D_3, W_temp4_lvl1_part0_2__D_0__D_1__D_2__D_3, 0, blkSize );

		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  W_temp4_lvl1_part0_1__D_0__D_1__D_2__D_3
		PartitionDown(r_bmfe_lvl1_part0_1__D_0__D_1__D_2__D_3, r_bmfe_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3, r_bmfe_lvl1_part0_1_lvl2_part1B__D_0__D_1__D_2__D_3, 1, 0);
		PartitionDown(W_temp4_lvl1_part0_1__D_0__D_1__D_2__D_3, W_temp4_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3, W_temp4_lvl1_part0_1_lvl2_part1B__D_0__D_1__D_2__D_3, 1, 0);
		while(W_temp4_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3.Dimension(1) < W_temp4_lvl1_part0_1__D_0__D_1__D_2__D_3.Dimension(1))
		{
			RepartitionDown
			( r_bmfe_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3,  r_bmfe_lvl1_part0_1_lvl2_part1_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       r_bmfe_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3,
			  r_bmfe_lvl1_part0_1_lvl2_part1B__D_0__D_1__D_2__D_3, r_bmfe_lvl1_part0_1_lvl2_part1_2__D_0__D_1__D_2__D_3, 1, blkSize );
			RepartitionDown
			( W_temp4_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3,  W_temp4_lvl1_part0_1_lvl2_part1_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       W_temp4_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3,
			  W_temp4_lvl1_part0_1_lvl2_part1B__D_0__D_1__D_2__D_3, W_temp4_lvl1_part0_1_lvl2_part1_2__D_0__D_1__D_2__D_3, 1, blkSize );

			   // r_bmfe_lvl1_part0_1_lvl2_part1_1[D0,D1,D3,D2] <- r_bmfe_lvl1_part0_1_lvl2_part1_1[D0,D1,D2,D3]
			r_bmfe_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_3__D_2.AlignModesWith( modes_0_1_2_3, r_bmfe_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3, modes_0_1_3_2 );
			r_bmfe_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_3__D_2.AllToAllRedistFrom( r_bmfe_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3, modes_2_3 );
PROFILE_SECTION("COMPUTE");
PROFILE_FLOPS(3*prod(r_bmfe_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3.Shape()));
			YAxpPx( -1.0, r_bmfe_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3, 2.0, r_bmfe_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_3__D_2, perm_0_1_3_2, W_temp4_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3 );
PROFILE_STOP;
			r_bmfe_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_3__D_2.EmptyData();

			SlidePartitionDown
			( r_bmfe_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3,  r_bmfe_lvl1_part0_1_lvl2_part1_0__D_0__D_1__D_2__D_3,
			       r_bmfe_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  r_bmfe_lvl1_part0_1_lvl2_part1B__D_0__D_1__D_2__D_3, r_bmfe_lvl1_part0_1_lvl2_part1_2__D_0__D_1__D_2__D_3, 1 );
			SlidePartitionDown
			( W_temp4_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3,  W_temp4_lvl1_part0_1_lvl2_part1_0__D_0__D_1__D_2__D_3,
			       W_temp4_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  W_temp4_lvl1_part0_1_lvl2_part1B__D_0__D_1__D_2__D_3, W_temp4_lvl1_part0_1_lvl2_part1_2__D_0__D_1__D_2__D_3, 1 );

		}
		//****

		SlidePartitionDown
		( r_bmfe_lvl1_part0T__D_0__D_1__D_2__D_3,  r_bmfe_lvl1_part0_0__D_0__D_1__D_2__D_3,
		       r_bmfe_lvl1_part0_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  r_bmfe_lvl1_part0B__D_0__D_1__D_2__D_3, r_bmfe_lvl1_part0_2__D_0__D_1__D_2__D_3, 0 );
		SlidePartitionDown
		( W_temp4_lvl1_part0T__D_0__D_1__D_2__D_3,  W_temp4_lvl1_part0_0__D_0__D_1__D_2__D_3,
		       W_temp4_lvl1_part0_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  W_temp4_lvl1_part0B__D_0__D_1__D_2__D_3, W_temp4_lvl1_part0_2__D_0__D_1__D_2__D_3, 0 );

	}
	//****
	//**** (out of 1)
	//**** Is real	0 shadows
		//Outputs:
		//  W_bmje__D_0__D_1__D_2__D_3
	PartitionDown(W_temp4__D_0__D_1__D_2__D_3, W_temp4_lvl1_part1T__D_0__D_1__D_2__D_3, W_temp4_lvl1_part1B__D_0__D_1__D_2__D_3, 1, 0);
	PartitionDown(W_bmje__D_0__D_1__D_2__D_3, W_bmje_lvl1_part1T__D_0__D_1__D_2__D_3, W_bmje_lvl1_part1B__D_0__D_1__D_2__D_3, 1, 0);
	PartitionDown(u_mnje__D_0__D_1__D_2__D_3, u_mnje_lvl1_part0T__D_0__D_1__D_2__D_3, u_mnje_lvl1_part0B__D_0__D_1__D_2__D_3, 0, 0);
	PartitionDown(u_mnje__D_0__D_1__D_2__D_3, u_mnje_lvl1_part1T__D_0__D_1__D_2__D_3, u_mnje_lvl1_part1B__D_0__D_1__D_2__D_3, 1, 0);
	while(W_bmje_lvl1_part1T__D_0__D_1__D_2__D_3.Dimension(1) < W_bmje__D_0__D_1__D_2__D_3.Dimension(1))
	{
		RepartitionDown
		( W_temp4_lvl1_part1T__D_0__D_1__D_2__D_3,  W_temp4_lvl1_part1_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       W_temp4_lvl1_part1_1__D_0__D_1__D_2__D_3,
		  W_temp4_lvl1_part1B__D_0__D_1__D_2__D_3, W_temp4_lvl1_part1_2__D_0__D_1__D_2__D_3, 1, blkSize );
		RepartitionDown
		( W_bmje_lvl1_part1T__D_0__D_1__D_2__D_3,  W_bmje_lvl1_part1_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       W_bmje_lvl1_part1_1__D_0__D_1__D_2__D_3,
		  W_bmje_lvl1_part1B__D_0__D_1__D_2__D_3, W_bmje_lvl1_part1_2__D_0__D_1__D_2__D_3, 1, blkSize );
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

		Permute( W_bmje_lvl1_part1_1__D_0__D_1__D_2__D_3, W_bmje_lvl1_part1_1_perm1230__D_1__D_2__D_3__D_0 );
		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  W_bmje_lvl1_part1_1_perm1230__D_1__D_2__D_3__D_0
		PartitionDown(t_fj__D_0_1__D_2_3, t_fj_lvl2_part1T__D_0_1__D_2_3, t_fj_lvl2_part1B__D_0_1__D_2_3, 1, 0);
		PartitionDown(u_mnje_lvl1_part0_1__D_0__D_1__D_2__D_3, u_mnje_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3, u_mnje_lvl1_part0_1_lvl2_part1B__D_0__D_1__D_2__D_3, 1, 0);
		PartitionDown(u_mnje_lvl1_part1_1__D_0__D_1__D_2__D_3, u_mnje_lvl1_part1_1_lvl2_part0T__D_0__D_1__D_2__D_3, u_mnje_lvl1_part1_1_lvl2_part0B__D_0__D_1__D_2__D_3, 0, 0);
		while(t_fj_lvl2_part1T__D_0_1__D_2_3.Dimension(1) < t_fj__D_0_1__D_2_3.Dimension(1))
		{
			RepartitionDown
			( t_fj_lvl2_part1T__D_0_1__D_2_3,  t_fj_lvl2_part1_0__D_0_1__D_2_3,
			  /**/ /**/
			       t_fj_lvl2_part1_1__D_0_1__D_2_3,
			  t_fj_lvl2_part1B__D_0_1__D_2_3, t_fj_lvl2_part1_2__D_0_1__D_2_3, 1, blkSize );
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

			   // t_fj_lvl2_part1_1[D0,*] <- t_fj_lvl2_part1_1[D01,D23]
			t_fj_lvl2_part1_1_perm10__S__D_0.AlignModesWith( modes_0, W_bmje_lvl1_part1_1__D_0__D_1__D_2__D_3, modes_0 );
			t_fj_lvl2_part1_1_perm10__S__D_0.AllGatherRedistFrom( t_fj_lvl2_part1_1__D_0_1__D_2_3, modes_1_2_3 );
			W_temp3_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3.AlignModesWith( modes_0_1_2_3, u_mnje_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3, modes_0_1_2_3 );
			overwrite_tmpShape_W = u_mnje_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3.Shape();
			W_temp3_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3.ResizeTo( overwrite_tmpShape_W );
			   // u_mnje_lvl1_part1_1_lvl2_part0_1[D1,D0,D2,D3] <- u_mnje_lvl1_part1_1_lvl2_part0_1[D0,D1,D2,D3]
			u_mnje_lvl1_part1_1_lvl2_part0_1__D_1__D_0__D_2__D_3.AlignModesWith( modes_0_1_2_3, u_mnje_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3, modes_1_0_2_3 );
			u_mnje_lvl1_part1_1_lvl2_part0_1__D_1__D_0__D_2__D_3.AllToAllRedistFrom( u_mnje_lvl1_part1_1_lvl2_part0_1__D_0__D_1__D_2__D_3, modes_0_1 );
PROFILE_SECTION("COMPUTE");
PROFILE_FLOPS(3*prod(u_mnje_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3.Shape()));
			YAxpPx( -1.0, u_mnje_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3, 2.0, u_mnje_lvl1_part1_1_lvl2_part0_1__D_1__D_0__D_2__D_3, perm_1_0_2_3, W_temp3_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3 );
PROFILE_STOP;
			u_mnje_lvl1_part1_1_lvl2_part0_1__D_1__D_0__D_2__D_3.EmptyData();
			   // W_temp3_lvl1_part0_1_lvl2_part1_1[D1,*,D2,D3] <- W_temp3_lvl1_part0_1_lvl2_part1_1[D0,D1,D2,D3]
			W_temp3_lvl1_part0_1_lvl2_part1_1_perm0231__D_1__D_2__D_3__S.AlignModesWith( modes_0_2_3, W_bmje_lvl1_part1_1__D_0__D_1__D_2__D_3, modes_1_2_3 );
			W_temp3_lvl1_part0_1_lvl2_part1_1_perm0231__D_1__D_2__D_3__S.AllToAllRedistFrom( W_temp3_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3, modes_0_1 );
			W_temp3_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3.EmptyData();
			   // -1.0 * W_temp3_lvl1_part0_1_lvl2_part1_1[D1,*,D2,D3]_mjen * t_fj_lvl2_part1_1[D0,*]_nb + 1.0 * W_bmje_lvl1_part1_1[D0,D1,D2,D3]_mjeb
PROFILE_SECTION("COMPUTE");
PROFILE_FLOPS(2*prod(W_bmje_lvl1_part1_1_perm1230__D_1__D_2__D_3__D_0.Shape())*W_temp3_lvl1_part0_1_lvl2_part1_1_perm0231__D_1__D_2__D_3__S.Dimension(1));
			LocalContractAndLocalEliminate(-1.0, W_temp3_lvl1_part0_1_lvl2_part1_1_perm0231__D_1__D_2__D_3__S.LockedTensor(), indices_mjen, false,
				t_fj_lvl2_part1_1_perm10__S__D_0.LockedTensor(), indices_nb, false,
				1.0, W_bmje_lvl1_part1_1_perm1230__D_1__D_2__D_3__D_0.Tensor(), indices_mjeb, false);
PROFILE_STOP;
			W_temp3_lvl1_part0_1_lvl2_part1_1_perm0231__D_1__D_2__D_3__S.EmptyData();
			t_fj_lvl2_part1_1_perm10__S__D_0.EmptyData();

			SlidePartitionDown
			( t_fj_lvl2_part1T__D_0_1__D_2_3,  t_fj_lvl2_part1_0__D_0_1__D_2_3,
			       t_fj_lvl2_part1_1__D_0_1__D_2_3,
			  /**/ /**/
			  t_fj_lvl2_part1B__D_0_1__D_2_3, t_fj_lvl2_part1_2__D_0_1__D_2_3, 1 );
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

		}
		//****
		Permute( W_bmje_lvl1_part1_1_perm1230__D_1__D_2__D_3__D_0, W_bmje_lvl1_part1_1__D_0__D_1__D_2__D_3 );
		W_bmje_lvl1_part1_1_perm1230__D_1__D_2__D_3__D_0.EmptyData();
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

			W_bmje_lvl1_part1_1_lvl2_part2_1_W_temp__D_0__D_1__S__D_2_3.AlignModesWith( modes_0_1_2_3, W_bmje_lvl1_part1_1_lvl2_part2_1_W_temp__D_0__D_1__D_2__D_3, modes_0_1_2_3 );
			overwrite_tmpShape_W = W_bmje_lvl1_part1_1_lvl2_part2_1__D_0__D_1__D_2__D_3.Shape();
			W_bmje_lvl1_part1_1_lvl2_part2_1_W_temp__D_0__D_1__S__D_2_3.ResizeTo( overwrite_tmpShape_W );
			   // t_fj_lvl2_part1_1[D3,D2] <- t_fj_lvl2_part1_1[D01,D23]
			t_fj_lvl2_part1_1__D_3__D_2.AlignModesWith( modes_0, W_temp4_lvl1_part1_1__D_0__D_1__D_2__D_3, modes_3 );
			t_fj_lvl2_part1_1__D_3__D_2.AllToAllRedistFrom( t_fj_lvl2_part1_1__D_0_1__D_2_3, modes_0_1_3 );
			   // t_fj_lvl2_part1_1[D3,*] <- t_fj_lvl2_part1_1[D3,D2]
			t_fj_lvl2_part1_1__D_3__S.AlignModesWith( modes_0, W_temp4_lvl1_part1_1__D_0__D_1__D_2__D_3, modes_3 );
			t_fj_lvl2_part1_1__D_3__S.AllGatherRedistFrom( t_fj_lvl2_part1_1__D_3__D_2, modes_2 );
			t_fj_lvl2_part1_1__D_3__D_2.EmptyData();
			W_bmje_lvl1_part1_1_lvl2_part2_1_perm01324__D_0__D_1__D_2__S__D_3.AlignModesWith( modes_0_1_3, W_temp4_lvl1_part1_1__D_0__D_1__D_2__D_3, modes_0_1_2 );
			overwrite_tmpShape_W = W_bmje_lvl1_part1_1_lvl2_part2_1__D_0__D_1__D_2__D_3.Shape();
			overwrite_tmpShape_W.push_back( g.Shape()[3] );
			W_bmje_lvl1_part1_1_lvl2_part2_1_perm01324__D_0__D_1__D_2__S__D_3.ResizeTo( overwrite_tmpShape_W );
			   // 1.0 * W_temp4_lvl1_part1_1[D0,D1,D2,D3]_bmef * t_fj_lvl2_part1_1[D3,*]_fj + 0.0 * W_bmje_lvl1_part1_1_lvl2_part2_1[D0,D1,*,D2,D3]_bmejf
PROFILE_SECTION("COMPUTE");
PROFILE_FLOPS(2*prod(W_bmje_lvl1_part1_1_lvl2_part2_1_perm01324__D_0__D_1__D_2__S__D_3.Shape())*W_temp4_lvl1_part1_1__D_0__D_1__D_2__D_3.Dimension(3));
			LocalContract(1.0, W_temp4_lvl1_part1_1__D_0__D_1__D_2__D_3.LockedTensor(), indices_bmef, false,
				t_fj_lvl2_part1_1__D_3__S.LockedTensor(), indices_fj, false,
				0.0, W_bmje_lvl1_part1_1_lvl2_part2_1_perm01324__D_0__D_1__D_2__S__D_3.Tensor(), indices_bmejf, false);
PROFILE_STOP;
			t_fj_lvl2_part1_1__D_3__S.EmptyData();
			   // W_bmje_lvl1_part1_1_lvl2_part2_1_W_temp[D0,D1,*,D23] <- W_bmje_lvl1_part1_1_lvl2_part2_1[D0,D1,*,D2,D3] (with SumScatter on D3)
PROFILE_SECTION("COMPUTE");
PROFILE_FLOPS(prod(W_bmje_lvl1_part1_1_lvl2_part2_1_perm01324__D_0__D_1__D_2__S__D_3.Shape()));
			W_bmje_lvl1_part1_1_lvl2_part2_1_W_temp__D_0__D_1__S__D_2_3.ReduceScatterRedistFrom( W_bmje_lvl1_part1_1_lvl2_part2_1_perm01324__D_0__D_1__D_2__S__D_3, 4 );
PROFILE_STOP;
			W_bmje_lvl1_part1_1_lvl2_part2_1_perm01324__D_0__D_1__D_2__S__D_3.EmptyData();
			   // W_bmje_lvl1_part1_1_lvl2_part2_1_W_temp[D0,D1,D2,D3] <- W_bmje_lvl1_part1_1_lvl2_part2_1_W_temp[D0,D1,*,D23]
			W_bmje_lvl1_part1_1_lvl2_part2_1_W_temp__D_0__D_1__D_2__D_3.AlignModesWith( modes_0_1_2_3, W_bmje_lvl1_part1_1_lvl2_part2_1__D_0__D_1__D_2__D_3, modes_0_1_2_3 );
			W_bmje_lvl1_part1_1_lvl2_part2_1_W_temp__D_0__D_1__D_2__D_3.AllToAllRedistFrom( W_bmje_lvl1_part1_1_lvl2_part2_1_W_temp__D_0__D_1__S__D_2_3, modes_2_3 );
			W_bmje_lvl1_part1_1_lvl2_part2_1_W_temp__D_0__D_1__S__D_2_3.EmptyData();
PROFILE_SECTION("COMPUTE");
PROFILE_FLOPS(2*prod(W_bmje_lvl1_part1_1_lvl2_part2_1_W_temp__D_0__D_1__D_2__D_3.Shape()));
			YxpBy( W_bmje_lvl1_part1_1_lvl2_part2_1_W_temp__D_0__D_1__D_2__D_3, 1.0, W_bmje_lvl1_part1_1_lvl2_part2_1__D_0__D_1__D_2__D_3 );
PROFILE_STOP;
			W_bmje_lvl1_part1_1_lvl2_part2_1_W_temp__D_0__D_1__D_2__D_3.EmptyData();

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
		( W_temp4_lvl1_part1T__D_0__D_1__D_2__D_3,  W_temp4_lvl1_part1_0__D_0__D_1__D_2__D_3,
		       W_temp4_lvl1_part1_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  W_temp4_lvl1_part1B__D_0__D_1__D_2__D_3, W_temp4_lvl1_part1_2__D_0__D_1__D_2__D_3, 1 );
		SlidePartitionDown
		( W_bmje_lvl1_part1T__D_0__D_1__D_2__D_3,  W_bmje_lvl1_part1_0__D_0__D_1__D_2__D_3,
		       W_bmje_lvl1_part1_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  W_bmje_lvl1_part1B__D_0__D_1__D_2__D_3, W_bmje_lvl1_part1_2__D_0__D_1__D_2__D_3, 1 );
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

	}
	//****
	W_temp4__D_0__D_1__D_2__D_3.EmptyData();


//****


#ifdef CORRECTNESS
    DistTensor<double> diff_W(dist__D_0__D_1__D_2__D_3, g);
    diff_W.ResizeTo(check_W);
    Diff(check_W, W_bmje__D_0__D_1__D_2__D_3, diff_W);
   norm = 1.0;
   norm = Norm(diff_W);
   if (commRank == 0){
     std::cout << "NORM_W " << norm << std::endl;
   }
#endif
//**** (out of 1)
//**** Is real	0 shadows
	//Outputs:
	//  X_bmej__D_0__D_1__D_2__D_3

	X_bmej__D_0__D_1__D_2__D_3 = x_bmej__D_0__D_1__D_2__D_3;


//****
overwrite_tmpShape_X = T_bfnj__D_0__D_1__D_2__D_3.Shape();
X_temp1__D_0__D_1__D_2__D_3.ResizeTo( overwrite_tmpShape_X );
//**** (out of 1)
//**** Is real	0 shadows
	//Outputs:
	//  X_temp1__D_0__D_1__D_2__D_3

	ZAxpBy( 1.0, Tau_efmn__D_0__D_1__D_2__D_3, -0.5, T_bfnj__D_0__D_1__D_2__D_3, X_temp1__D_0__D_1__D_2__D_3 );


//****
//**** (out of 1)
//**** Is real	0 shadows
	//Outputs:
	//  X_bmej__D_0__D_1__D_2__D_3

	Permute( X_bmej__D_0__D_1__D_2__D_3, X_bmej_perm2103__D_2__D_1__D_0__D_3 );
	//**** (out of 1)
	//**** Is real	0 shadows
		//Outputs:
		//  X_bmej_perm2103__D_2__D_1__D_0__D_3
	PartitionDown(v_femn__D_0__D_1__D_2__D_3, v_femn_lvl1_part0T__D_0__D_1__D_2__D_3, v_femn_lvl1_part0B__D_0__D_1__D_2__D_3, 0, 0);
	PartitionDown(X_temp1__D_0__D_1__D_2__D_3, X_temp1_lvl1_part1T__D_0__D_1__D_2__D_3, X_temp1_lvl1_part1B__D_0__D_1__D_2__D_3, 1, 0);
	while(v_femn_lvl1_part0T__D_0__D_1__D_2__D_3.Dimension(0) < v_femn__D_0__D_1__D_2__D_3.Dimension(0))
	{
		RepartitionDown
		( v_femn_lvl1_part0T__D_0__D_1__D_2__D_3,  v_femn_lvl1_part0_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       v_femn_lvl1_part0_1__D_0__D_1__D_2__D_3,
		  v_femn_lvl1_part0B__D_0__D_1__D_2__D_3, v_femn_lvl1_part0_2__D_0__D_1__D_2__D_3, 0, blkSize );
		RepartitionDown
		( X_temp1_lvl1_part1T__D_0__D_1__D_2__D_3,  X_temp1_lvl1_part1_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       X_temp1_lvl1_part1_1__D_0__D_1__D_2__D_3,
		  X_temp1_lvl1_part1B__D_0__D_1__D_2__D_3, X_temp1_lvl1_part1_2__D_0__D_1__D_2__D_3, 1, blkSize );

		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  X_bmej_perm2103__D_2__D_1__D_0__D_3
		PartitionDown(v_femn_lvl1_part0_1__D_0__D_1__D_2__D_3, v_femn_lvl1_part0_1_lvl2_part3T__D_0__D_1__D_2__D_3, v_femn_lvl1_part0_1_lvl2_part3B__D_0__D_1__D_2__D_3, 3, 0);
		PartitionDown(X_temp1_lvl1_part1_1__D_0__D_1__D_2__D_3, X_temp1_lvl1_part1_1_lvl2_part2T__D_0__D_1__D_2__D_3, X_temp1_lvl1_part1_1_lvl2_part2B__D_0__D_1__D_2__D_3, 2, 0);
		while(v_femn_lvl1_part0_1_lvl2_part3T__D_0__D_1__D_2__D_3.Dimension(3) < v_femn_lvl1_part0_1__D_0__D_1__D_2__D_3.Dimension(3))
		{
			RepartitionDown
			( v_femn_lvl1_part0_1_lvl2_part3T__D_0__D_1__D_2__D_3,  v_femn_lvl1_part0_1_lvl2_part3_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       v_femn_lvl1_part0_1_lvl2_part3_1__D_0__D_1__D_2__D_3,
			  v_femn_lvl1_part0_1_lvl2_part3B__D_0__D_1__D_2__D_3, v_femn_lvl1_part0_1_lvl2_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );
			RepartitionDown
			( X_temp1_lvl1_part1_1_lvl2_part2T__D_0__D_1__D_2__D_3,  X_temp1_lvl1_part1_1_lvl2_part2_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       X_temp1_lvl1_part1_1_lvl2_part2_1__D_0__D_1__D_2__D_3,
			  X_temp1_lvl1_part1_1_lvl2_part2B__D_0__D_1__D_2__D_3, X_temp1_lvl1_part1_1_lvl2_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );

			   // X_temp1_lvl1_part1_1_lvl2_part2_1[D0,*,*,D3] <- X_temp1_lvl1_part1_1_lvl2_part2_1[D0,D1,D2,D3]
			X_temp1_lvl1_part1_1_lvl2_part2_1_perm1203__S__S__D_0__D_3.AlignModesWith( modes_0_3, X_bmej__D_0__D_1__D_2__D_3, modes_0_3 );
			X_temp1_lvl1_part1_1_lvl2_part2_1_perm1203__S__S__D_0__D_3.AllGatherRedistFrom( X_temp1_lvl1_part1_1_lvl2_part2_1__D_0__D_1__D_2__D_3, modes_1_2 );
			   // v_femn_lvl1_part0_1_lvl2_part3_1[D0,D2,D1,D3] <- v_femn_lvl1_part0_1_lvl2_part3_1[D0,D1,D2,D3]
			v_femn_lvl1_part0_1_lvl2_part3_1__D_0__D_2__D_1__D_3.AlignModesWith( modes_1_2, X_bmej__D_0__D_1__D_2__D_3, modes_2_1 );
			v_femn_lvl1_part0_1_lvl2_part3_1__D_0__D_2__D_1__D_3.AllToAllRedistFrom( v_femn_lvl1_part0_1_lvl2_part3_1__D_0__D_1__D_2__D_3, modes_1_2 );
			   // v_femn_lvl1_part0_1_lvl2_part3_1[*,D2,D1,*] <- v_femn_lvl1_part0_1_lvl2_part3_1[D0,D2,D1,D3]
			v_femn_lvl1_part0_1_lvl2_part3_1_perm1203__D_2__D_1__S__S.AlignModesWith( modes_1_2, X_bmej__D_0__D_1__D_2__D_3, modes_2_1 );
			v_femn_lvl1_part0_1_lvl2_part3_1_perm1203__D_2__D_1__S__S.AllGatherRedistFrom( v_femn_lvl1_part0_1_lvl2_part3_1__D_0__D_2__D_1__D_3, modes_0_3 );
			v_femn_lvl1_part0_1_lvl2_part3_1__D_0__D_2__D_1__D_3.EmptyData();
			   // -1.0 * v_femn_lvl1_part0_1_lvl2_part3_1[*,D2,D1,*]_emfn * X_temp1_lvl1_part1_1_lvl2_part2_1[D0,*,*,D3]_fnbj + 1.0 * X_bmej[D0,D1,D2,D3]_embj
PROFILE_SECTION("COMPUTE");
PROFILE_FLOPS(2*prod(X_bmej_perm2103__D_2__D_1__D_0__D_3.Shape())*v_femn_lvl1_part0_1_lvl2_part3_1_perm1203__D_2__D_1__S__S.Dimension(0)*v_femn_lvl1_part0_1_lvl2_part3_1_perm1203__D_2__D_1__S__S.Dimension(3));
			LocalContractAndLocalEliminate(-1.0, v_femn_lvl1_part0_1_lvl2_part3_1_perm1203__D_2__D_1__S__S.LockedTensor(), indices_emfn, false,
				X_temp1_lvl1_part1_1_lvl2_part2_1_perm1203__S__S__D_0__D_3.LockedTensor(), indices_fnbj, false,
				1.0, X_bmej_perm2103__D_2__D_1__D_0__D_3.Tensor(), indices_embj, false);
PROFILE_STOP;
			v_femn_lvl1_part0_1_lvl2_part3_1_perm1203__D_2__D_1__S__S.EmptyData();
			X_temp1_lvl1_part1_1_lvl2_part2_1_perm1203__S__S__D_0__D_3.EmptyData();

			SlidePartitionDown
			( v_femn_lvl1_part0_1_lvl2_part3T__D_0__D_1__D_2__D_3,  v_femn_lvl1_part0_1_lvl2_part3_0__D_0__D_1__D_2__D_3,
			       v_femn_lvl1_part0_1_lvl2_part3_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  v_femn_lvl1_part0_1_lvl2_part3B__D_0__D_1__D_2__D_3, v_femn_lvl1_part0_1_lvl2_part3_2__D_0__D_1__D_2__D_3, 3 );
			SlidePartitionDown
			( X_temp1_lvl1_part1_1_lvl2_part2T__D_0__D_1__D_2__D_3,  X_temp1_lvl1_part1_1_lvl2_part2_0__D_0__D_1__D_2__D_3,
			       X_temp1_lvl1_part1_1_lvl2_part2_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  X_temp1_lvl1_part1_1_lvl2_part2B__D_0__D_1__D_2__D_3, X_temp1_lvl1_part1_1_lvl2_part2_2__D_0__D_1__D_2__D_3, 2 );

		}
		//****

		SlidePartitionDown
		( v_femn_lvl1_part0T__D_0__D_1__D_2__D_3,  v_femn_lvl1_part0_0__D_0__D_1__D_2__D_3,
		       v_femn_lvl1_part0_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  v_femn_lvl1_part0B__D_0__D_1__D_2__D_3, v_femn_lvl1_part0_2__D_0__D_1__D_2__D_3, 0 );
		SlidePartitionDown
		( X_temp1_lvl1_part1T__D_0__D_1__D_2__D_3,  X_temp1_lvl1_part1_0__D_0__D_1__D_2__D_3,
		       X_temp1_lvl1_part1_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  X_temp1_lvl1_part1B__D_0__D_1__D_2__D_3, X_temp1_lvl1_part1_2__D_0__D_1__D_2__D_3, 1 );

	}
	//****
	X_temp1__D_0__D_1__D_2__D_3.EmptyData();
	Permute( X_bmej_perm2103__D_2__D_1__D_0__D_3, X_bmej__D_0__D_1__D_2__D_3 );
	X_bmej_perm2103__D_2__D_1__D_0__D_3.EmptyData();
	//**** (out of 1)
	//**** Is real	0 shadows
		//Outputs:
		//  X_bmej__D_0__D_1__D_2__D_3
	PartitionDown(r_bmfe__D_0__D_1__D_2__D_3, r_bmfe_lvl1_part1T__D_0__D_1__D_2__D_3, r_bmfe_lvl1_part1B__D_0__D_1__D_2__D_3, 1, 0);
	PartitionDown(u_mnje__D_0__D_1__D_2__D_3, u_mnje_lvl1_part0T__D_0__D_1__D_2__D_3, u_mnje_lvl1_part0B__D_0__D_1__D_2__D_3, 0, 0);
	PartitionDown(X_bmej__D_0__D_1__D_2__D_3, X_bmej_lvl1_part1T__D_0__D_1__D_2__D_3, X_bmej_lvl1_part1B__D_0__D_1__D_2__D_3, 1, 0);
	while(X_bmej_lvl1_part1T__D_0__D_1__D_2__D_3.Dimension(1) < X_bmej__D_0__D_1__D_2__D_3.Dimension(1))
	{
		RepartitionDown
		( r_bmfe_lvl1_part1T__D_0__D_1__D_2__D_3,  r_bmfe_lvl1_part1_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       r_bmfe_lvl1_part1_1__D_0__D_1__D_2__D_3,
		  r_bmfe_lvl1_part1B__D_0__D_1__D_2__D_3, r_bmfe_lvl1_part1_2__D_0__D_1__D_2__D_3, 1, blkSize );
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
PROFILE_SECTION("COMPUTE");
PROFILE_FLOPS(2*prod(X_bmej_lvl1_part1_1_perm1320__D_1__D_3__D_2__D_0.Shape())*u_mnje_lvl1_part0_1_lvl2_part1_1_perm0231__D_1__D_3__D_2__S.Dimension(1));
			LocalContractAndLocalEliminate(-1.0, u_mnje_lvl1_part0_1_lvl2_part1_1_perm0231__D_1__D_3__D_2__S.LockedTensor(), indices_mjen, false,
				t_fj_lvl2_part1_1_perm10__S__D_0.LockedTensor(), indices_nb, false,
				1.0, X_bmej_lvl1_part1_1_perm1320__D_1__D_3__D_2__D_0.Tensor(), indices_mjeb, false);
PROFILE_STOP;
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

			   // t_fj_lvl2_part1_1[D3,D2] <- t_fj_lvl2_part1_1[D01,D23]
			t_fj_lvl2_part1_1__D_3__D_2.AlignModesWith( modes_0, r_bmfe_lvl1_part1_1__D_0__D_1__D_2__D_3, modes_3 );
			t_fj_lvl2_part1_1__D_3__D_2.AllToAllRedistFrom( t_fj_lvl2_part1_1__D_0_1__D_2_3, modes_0_1_3 );
			   // t_fj_lvl2_part1_1[D3,*] <- t_fj_lvl2_part1_1[D3,D2]
			t_fj_lvl2_part1_1__D_3__S.AlignModesWith( modes_0, r_bmfe_lvl1_part1_1__D_0__D_1__D_2__D_3, modes_3 );
			t_fj_lvl2_part1_1__D_3__S.AllGatherRedistFrom( t_fj_lvl2_part1_1__D_3__D_2, modes_2 );
			t_fj_lvl2_part1_1__D_3__D_2.EmptyData();
			X_bmej_lvl1_part1_1_lvl2_part3_1__D_0__D_1__D_2__S__D_3.AlignModesWith( modes_0_1_2, r_bmfe_lvl1_part1_1__D_0__D_1__D_2__D_3, modes_0_1_2 );
			overwrite_tmpShape_X = X_bmej_lvl1_part1_1_lvl2_part3_1__D_0__D_1__D_2__D_3.Shape();
			overwrite_tmpShape_X.push_back( g.Shape()[3] );
			X_bmej_lvl1_part1_1_lvl2_part3_1__D_0__D_1__D_2__S__D_3.ResizeTo( overwrite_tmpShape_X );
			   // 1.0 * r_bmfe_lvl1_part1_1[D0,D1,D2,D3]_bmef * t_fj_lvl2_part1_1[D3,*]_fj + 0.0 * X_bmej_lvl1_part1_1_lvl2_part3_1[D0,D1,D2,*,D3]_bmejf
PROFILE_SECTION("COMPUTE");
PROFILE_FLOPS(2*prod(X_bmej_lvl1_part1_1_lvl2_part3_1__D_0__D_1__D_2__S__D_3.Shape())*r_bmfe_lvl1_part1_1__D_0__D_1__D_2__D_3.Dimension(3));
			LocalContract(1.0, r_bmfe_lvl1_part1_1__D_0__D_1__D_2__D_3.LockedTensor(), indices_bmef, false,
				t_fj_lvl2_part1_1__D_3__S.LockedTensor(), indices_fj, false,
				0.0, X_bmej_lvl1_part1_1_lvl2_part3_1__D_0__D_1__D_2__S__D_3.Tensor(), indices_bmejf, false);
PROFILE_STOP;
			t_fj_lvl2_part1_1__D_3__S.EmptyData();
			   // X_bmej_lvl1_part1_1_lvl2_part3_1[D0,D1,D2,D3] <- X_bmej_lvl1_part1_1_lvl2_part3_1[D0,D1,D2,*,D3] (with SumScatter on D3)
PROFILE_SECTION("COMPUTE");
PROFILE_FLOPS(2*prod(X_bmej_lvl1_part1_1_lvl2_part3_1__D_0__D_1__D_2__S__D_3.Shape()));
			X_bmej_lvl1_part1_1_lvl2_part3_1__D_0__D_1__D_2__D_3.ReduceScatterUpdateRedistFrom( X_bmej_lvl1_part1_1_lvl2_part3_1__D_0__D_1__D_2__S__D_3, 1.0, 4 );
PROFILE_STOP;
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
		SlidePartitionDown
		( u_mnje_lvl1_part0T__D_0__D_1__D_2__D_3,  u_mnje_lvl1_part0_0__D_0__D_1__D_2__D_3,
		       u_mnje_lvl1_part0_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  u_mnje_lvl1_part0B__D_0__D_1__D_2__D_3, u_mnje_lvl1_part0_2__D_0__D_1__D_2__D_3, 0 );

	}
	//****


//****


#ifdef CORRECTNESS
    DistTensor<double> diff_X(dist__D_0__D_1__D_2__D_3, g);
    diff_X.ResizeTo(check_X);
    Diff(check_X, X_bmej__D_0__D_1__D_2__D_3, diff_X);
   norm = 1.0;
   norm = Norm(diff_X);
   if (commRank == 0){
     std::cout << "NORM_X " << norm << std::endl;
   }
#endif
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
	PartitionDown(v_femn__D_0__D_1__D_2__D_3, v_femn_lvl1_part3T__D_0__D_1__D_2__D_3, v_femn_lvl1_part3B__D_0__D_1__D_2__D_3, 3, 0);
	PartitionDown(U_mnie__D_0__D_1__D_2__D_3, U_mnie_lvl1_part1T__D_0__D_1__D_2__D_3, U_mnie_lvl1_part1B__D_0__D_1__D_2__D_3, 1, 0);
	while(U_mnie_lvl1_part1T__D_0__D_1__D_2__D_3.Dimension(1) < U_mnie__D_0__D_1__D_2__D_3.Dimension(1))
	{
		RepartitionDown
		( v_femn_lvl1_part3T__D_0__D_1__D_2__D_3,  v_femn_lvl1_part3_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       v_femn_lvl1_part3_1__D_0__D_1__D_2__D_3,
		  v_femn_lvl1_part3B__D_0__D_1__D_2__D_3, v_femn_lvl1_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );
		RepartitionDown
		( U_mnie_lvl1_part1T__D_0__D_1__D_2__D_3,  U_mnie_lvl1_part1_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       U_mnie_lvl1_part1_1__D_0__D_1__D_2__D_3,
		  U_mnie_lvl1_part1B__D_0__D_1__D_2__D_3, U_mnie_lvl1_part1_2__D_0__D_1__D_2__D_3, 1, blkSize );

		Permute( v_femn_lvl1_part3_1__D_0__D_1__D_2__D_3, v_femn_lvl1_part3_1_perm1230__D_1__D_2__D_3__D_0 );
		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  U_mnie_lvl1_part1_1__D_0__D_1__D_2__D_3
		PartitionDown(t_fj__D_0_1__D_2_3, t_fj_lvl2_part1T__D_0_1__D_2_3, t_fj_lvl2_part1B__D_0_1__D_2_3, 1, 0);
		PartitionDown(U_mnie_lvl1_part1_1__D_0__D_1__D_2__D_3, U_mnie_lvl1_part1_1_lvl2_part2T__D_0__D_1__D_2__D_3, U_mnie_lvl1_part1_1_lvl2_part2B__D_0__D_1__D_2__D_3, 2, 0);
		while(U_mnie_lvl1_part1_1_lvl2_part2T__D_0__D_1__D_2__D_3.Dimension(2) < U_mnie_lvl1_part1_1__D_0__D_1__D_2__D_3.Dimension(2))
		{
			RepartitionDown
			( t_fj_lvl2_part1T__D_0_1__D_2_3,  t_fj_lvl2_part1_0__D_0_1__D_2_3,
			  /**/ /**/
			       t_fj_lvl2_part1_1__D_0_1__D_2_3,
			  t_fj_lvl2_part1B__D_0_1__D_2_3, t_fj_lvl2_part1_2__D_0_1__D_2_3, 1, blkSize );
			RepartitionDown
			( U_mnie_lvl1_part1_1_lvl2_part2T__D_0__D_1__D_2__D_3,  U_mnie_lvl1_part1_1_lvl2_part2_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       U_mnie_lvl1_part1_1_lvl2_part2_1__D_0__D_1__D_2__D_3,
			  U_mnie_lvl1_part1_1_lvl2_part2B__D_0__D_1__D_2__D_3, U_mnie_lvl1_part1_1_lvl2_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );

			U_mnie_lvl1_part1_1_lvl2_part2_1_U_temp__D_2_0__D_3__S__D_1.AlignModesWith( modes_0_1_2_3, U_mnie_lvl1_part1_1_lvl2_part2_1_U_temp__D_0__D_1__D_2__D_3, modes_0_1_2_3 );
			overwrite_tmpShape_U = U_mnie_lvl1_part1_1_lvl2_part2_1__D_0__D_1__D_2__D_3.Shape();
			U_mnie_lvl1_part1_1_lvl2_part2_1_U_temp__D_2_0__D_3__S__D_1.ResizeTo( overwrite_tmpShape_U );
			   // t_fj_lvl2_part1_1[D0,*] <- t_fj_lvl2_part1_1[D01,D23]
			t_fj_lvl2_part1_1__D_0__S.AlignModesWith( modes_0, v_femn_lvl1_part3_1__D_0__D_1__D_2__D_3, modes_0 );
			t_fj_lvl2_part1_1__D_0__S.AllGatherRedistFrom( t_fj_lvl2_part1_1__D_0_1__D_2_3, modes_1_2_3 );
			U_mnie_lvl1_part1_1_lvl2_part2_1_perm30124__D_1__D_2__D_3__S__D_0.AlignModesWith( modes_0_1_3, v_femn_lvl1_part3_1__D_0__D_1__D_2__D_3, modes_2_3_1 );
			overwrite_tmpShape_U = U_mnie_lvl1_part1_1_lvl2_part2_1__D_0__D_1__D_2__D_3.Shape();
			overwrite_tmpShape_U.push_back( g.Shape()[0] );
			U_mnie_lvl1_part1_1_lvl2_part2_1_perm30124__D_1__D_2__D_3__S__D_0.ResizeTo( overwrite_tmpShape_U );
			   // 1.0 * v_femn_lvl1_part3_1[D0,D1,D2,D3]_emnf * t_fj_lvl2_part1_1[D0,*]_fi + 0.0 * U_mnie_lvl1_part1_1_lvl2_part2_1[D2,D3,*,D1,D0]_emnif
PROFILE_SECTION("COMPUTE");
PROFILE_FLOPS(2*prod(U_mnie_lvl1_part1_1_lvl2_part2_1_perm30124__D_1__D_2__D_3__S__D_0.Shape())*v_femn_lvl1_part3_1_perm1230__D_1__D_2__D_3__D_0.Dimension(0));
			LocalContract(1.0, v_femn_lvl1_part3_1_perm1230__D_1__D_2__D_3__D_0.LockedTensor(), indices_emnf, false,
				t_fj_lvl2_part1_1__D_0__S.LockedTensor(), indices_fi, false,
				0.0, U_mnie_lvl1_part1_1_lvl2_part2_1_perm30124__D_1__D_2__D_3__S__D_0.Tensor(), indices_emnif, false);
PROFILE_STOP;
			t_fj_lvl2_part1_1__D_0__S.EmptyData();
			   // U_mnie_lvl1_part1_1_lvl2_part2_1_U_temp[D20,D3,*,D1] <- U_mnie_lvl1_part1_1_lvl2_part2_1[D2,D3,*,D1,D0] (with SumScatter on D0)
PROFILE_SECTION("COMPUTE");
PROFILE_FLOPS(prod(U_mnie_lvl1_part1_1_lvl2_part2_1_perm30124__D_1__D_2__D_3__S__D_0.Shape()));
			U_mnie_lvl1_part1_1_lvl2_part2_1_U_temp__D_2_0__D_3__S__D_1.ReduceScatterRedistFrom( U_mnie_lvl1_part1_1_lvl2_part2_1_perm30124__D_1__D_2__D_3__S__D_0, 4 );
PROFILE_STOP;
			U_mnie_lvl1_part1_1_lvl2_part2_1_perm30124__D_1__D_2__D_3__S__D_0.EmptyData();
			   // U_mnie_lvl1_part1_1_lvl2_part2_1_U_temp[D20,D1,*,D3] <- U_mnie_lvl1_part1_1_lvl2_part2_1_U_temp[D20,D3,*,D1]
			U_mnie_lvl1_part1_1_lvl2_part2_1_U_temp__D_2_0__D_1__S__D_3.AlignModesWith( modes_0_1_2_3, U_mnie_lvl1_part1_1_lvl2_part2_1__D_0__D_1__D_2__D_3, modes_0_1_2_3 );
			U_mnie_lvl1_part1_1_lvl2_part2_1_U_temp__D_2_0__D_1__S__D_3.AllToAllRedistFrom( U_mnie_lvl1_part1_1_lvl2_part2_1_U_temp__D_2_0__D_3__S__D_1, modes_1_3 );
			U_mnie_lvl1_part1_1_lvl2_part2_1_U_temp__D_2_0__D_3__S__D_1.EmptyData();
			   // U_mnie_lvl1_part1_1_lvl2_part2_1_U_temp[D0,D1,D2,D3] <- U_mnie_lvl1_part1_1_lvl2_part2_1_U_temp[D20,D1,*,D3]
			U_mnie_lvl1_part1_1_lvl2_part2_1_U_temp__D_0__D_1__D_2__D_3.AlignModesWith( modes_0_1_2_3, U_mnie_lvl1_part1_1_lvl2_part2_1__D_0__D_1__D_2__D_3, modes_0_1_2_3 );
			U_mnie_lvl1_part1_1_lvl2_part2_1_U_temp__D_0__D_1__D_2__D_3.AllToAllRedistFrom( U_mnie_lvl1_part1_1_lvl2_part2_1_U_temp__D_2_0__D_1__S__D_3, modes_0_2 );
			U_mnie_lvl1_part1_1_lvl2_part2_1_U_temp__D_2_0__D_1__S__D_3.EmptyData();
PROFILE_SECTION("COMPUTE");
PROFILE_FLOPS(2*prod(U_mnie_lvl1_part1_1_lvl2_part2_1_U_temp__D_0__D_1__D_2__D_3.Shape()));
			YxpBy( U_mnie_lvl1_part1_1_lvl2_part2_1_U_temp__D_0__D_1__D_2__D_3, 1.0, U_mnie_lvl1_part1_1_lvl2_part2_1__D_0__D_1__D_2__D_3 );
PROFILE_STOP;
			U_mnie_lvl1_part1_1_lvl2_part2_1_U_temp__D_0__D_1__D_2__D_3.EmptyData();

			SlidePartitionDown
			( t_fj_lvl2_part1T__D_0_1__D_2_3,  t_fj_lvl2_part1_0__D_0_1__D_2_3,
			       t_fj_lvl2_part1_1__D_0_1__D_2_3,
			  /**/ /**/
			  t_fj_lvl2_part1B__D_0_1__D_2_3, t_fj_lvl2_part1_2__D_0_1__D_2_3, 1 );
			SlidePartitionDown
			( U_mnie_lvl1_part1_1_lvl2_part2T__D_0__D_1__D_2__D_3,  U_mnie_lvl1_part1_1_lvl2_part2_0__D_0__D_1__D_2__D_3,
			       U_mnie_lvl1_part1_1_lvl2_part2_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  U_mnie_lvl1_part1_1_lvl2_part2B__D_0__D_1__D_2__D_3, U_mnie_lvl1_part1_1_lvl2_part2_2__D_0__D_1__D_2__D_3, 2 );

		}
		//****
		v_femn_lvl1_part3_1_perm1230__D_1__D_2__D_3__D_0.EmptyData();

		SlidePartitionDown
		( v_femn_lvl1_part3T__D_0__D_1__D_2__D_3,  v_femn_lvl1_part3_0__D_0__D_1__D_2__D_3,
		       v_femn_lvl1_part3_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  v_femn_lvl1_part3B__D_0__D_1__D_2__D_3, v_femn_lvl1_part3_2__D_0__D_1__D_2__D_3, 3 );
		SlidePartitionDown
		( U_mnie_lvl1_part1T__D_0__D_1__D_2__D_3,  U_mnie_lvl1_part1_0__D_0__D_1__D_2__D_3,
		       U_mnie_lvl1_part1_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  U_mnie_lvl1_part1B__D_0__D_1__D_2__D_3, U_mnie_lvl1_part1_2__D_0__D_1__D_2__D_3, 1 );

	}
	//****


//****


#ifdef CORRECTNESS
    DistTensor<double> diff_U(dist__D_0__D_1__D_2__D_3, g);
    diff_U.ResizeTo(check_U);
    Diff(check_U, U_mnie__D_0__D_1__D_2__D_3, diff_U);
   norm = 1.0;
   norm = Norm(diff_U);
   if (commRank == 0){
     std::cout << "NORM_U " << norm << std::endl;
   }
#endif
overwrite_tmpShape_Q = Q_mnij__D_0__D_1__D_2__D_3.Shape();
Q_temp1__D_0__D_1__D_2__D_3.ResizeTo( overwrite_tmpShape_Q );
//**** (out of 1)
//**** Is real	0 shadows
	//Outputs:
	//  Q_mnij__D_0__D_1__D_2__D_3

	//**** (out of 1)
	//**** Is real	0 shadows
		//Outputs:
		//  Q_temp1__D_0__D_1__D_2__D_3
	PartitionDown(u_mnje__D_0__D_1__D_2__D_3, u_mnje_lvl1_part0T__D_0__D_1__D_2__D_3, u_mnje_lvl1_part0B__D_0__D_1__D_2__D_3, 0, 0);
	PartitionDown(Q_temp1__D_0__D_1__D_2__D_3, Q_temp1_lvl1_part0T__D_0__D_1__D_2__D_3, Q_temp1_lvl1_part0B__D_0__D_1__D_2__D_3, 0, 0);
	while(Q_temp1_lvl1_part0T__D_0__D_1__D_2__D_3.Dimension(0) < Q_temp1__D_0__D_1__D_2__D_3.Dimension(0))
	{
		RepartitionDown
		( u_mnje_lvl1_part0T__D_0__D_1__D_2__D_3,  u_mnje_lvl1_part0_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       u_mnje_lvl1_part0_1__D_0__D_1__D_2__D_3,
		  u_mnje_lvl1_part0B__D_0__D_1__D_2__D_3, u_mnje_lvl1_part0_2__D_0__D_1__D_2__D_3, 0, blkSize );
		RepartitionDown
		( Q_temp1_lvl1_part0T__D_0__D_1__D_2__D_3,  Q_temp1_lvl1_part0_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       Q_temp1_lvl1_part0_1__D_0__D_1__D_2__D_3,
		  Q_temp1_lvl1_part0B__D_0__D_1__D_2__D_3, Q_temp1_lvl1_part0_2__D_0__D_1__D_2__D_3, 0, blkSize );

		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  Q_temp1_lvl1_part0_1__D_0__D_1__D_2__D_3
		PartitionDown(u_mnje_lvl1_part0_1__D_0__D_1__D_2__D_3, u_mnje_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3, u_mnje_lvl1_part0_1_lvl2_part1B__D_0__D_1__D_2__D_3, 1, 0);
		PartitionDown(Q_temp1_lvl1_part0_1__D_0__D_1__D_2__D_3, Q_temp1_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3, Q_temp1_lvl1_part0_1_lvl2_part1B__D_0__D_1__D_2__D_3, 1, 0);
		while(Q_temp1_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3.Dimension(1) < Q_temp1_lvl1_part0_1__D_0__D_1__D_2__D_3.Dimension(1))
		{
			RepartitionDown
			( u_mnje_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3,  u_mnje_lvl1_part0_1_lvl2_part1_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       u_mnje_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3,
			  u_mnje_lvl1_part0_1_lvl2_part1B__D_0__D_1__D_2__D_3, u_mnje_lvl1_part0_1_lvl2_part1_2__D_0__D_1__D_2__D_3, 1, blkSize );
			RepartitionDown
			( Q_temp1_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3,  Q_temp1_lvl1_part0_1_lvl2_part1_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       Q_temp1_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3,
			  Q_temp1_lvl1_part0_1_lvl2_part1B__D_0__D_1__D_2__D_3, Q_temp1_lvl1_part0_1_lvl2_part1_2__D_0__D_1__D_2__D_3, 1, blkSize );

			   // t_fj[D01,D3] <- t_fj[D01,D23]
			t_fj__D_0_1__D_3.AlignModesWith( modes_1, Q_temp1_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3, modes_3 );
			t_fj__D_0_1__D_3.AllToAllRedistFrom( t_fj__D_0_1__D_2_3, modes_2_3 );
			   // t_fj[*,D3] <- t_fj[D01,D3]
			t_fj__S__D_3.AlignModesWith( modes_1, Q_temp1_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3, modes_3 );
			t_fj__S__D_3.AllGatherRedistFrom( t_fj__D_0_1__D_3, modes_0_1 );
			t_fj__D_0_1__D_3.EmptyData();
			   // u_mnje_lvl1_part0_1_lvl2_part1_1[D0,D1,D2,*] <- u_mnje_lvl1_part0_1_lvl2_part1_1[D0,D1,D2,D3]
			u_mnje_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__S.AlignModesWith( modes_0_1_2, Q_temp1_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3, modes_0_1_2 );
			u_mnje_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__S.AllGatherRedistFrom( u_mnje_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3, modes_3 );
			   // 1.0 * u_mnje_lvl1_part0_1_lvl2_part1_1[D0,D1,D2,*]_mnie * t_fj[*,D3]_ej + 0.0 * Q_temp1_lvl1_part0_1_lvl2_part1_1[D0,D1,D2,D3]_mnij
PROFILE_SECTION("COMPUTE");
PROFILE_FLOPS(2*prod(Q_temp1_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3.Shape())*u_mnje_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__S.Dimension(3));
			LocalContractAndLocalEliminate(1.0, u_mnje_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__S.LockedTensor(), indices_mnie, false,
				t_fj__S__D_3.LockedTensor(), indices_ej, false,
				0.0, Q_temp1_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3.Tensor(), indices_mnij, false);
PROFILE_STOP;
			u_mnje_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__S.EmptyData();
			t_fj__S__D_3.EmptyData();

			SlidePartitionDown
			( u_mnje_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3,  u_mnje_lvl1_part0_1_lvl2_part1_0__D_0__D_1__D_2__D_3,
			       u_mnje_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  u_mnje_lvl1_part0_1_lvl2_part1B__D_0__D_1__D_2__D_3, u_mnje_lvl1_part0_1_lvl2_part1_2__D_0__D_1__D_2__D_3, 1 );
			SlidePartitionDown
			( Q_temp1_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3,  Q_temp1_lvl1_part0_1_lvl2_part1_0__D_0__D_1__D_2__D_3,
			       Q_temp1_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  Q_temp1_lvl1_part0_1_lvl2_part1B__D_0__D_1__D_2__D_3, Q_temp1_lvl1_part0_1_lvl2_part1_2__D_0__D_1__D_2__D_3, 1 );

		}
		//****

		SlidePartitionDown
		( u_mnje_lvl1_part0T__D_0__D_1__D_2__D_3,  u_mnje_lvl1_part0_0__D_0__D_1__D_2__D_3,
		       u_mnje_lvl1_part0_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  u_mnje_lvl1_part0B__D_0__D_1__D_2__D_3, u_mnje_lvl1_part0_2__D_0__D_1__D_2__D_3, 0 );
		SlidePartitionDown
		( Q_temp1_lvl1_part0T__D_0__D_1__D_2__D_3,  Q_temp1_lvl1_part0_0__D_0__D_1__D_2__D_3,
		       Q_temp1_lvl1_part0_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  Q_temp1_lvl1_part0B__D_0__D_1__D_2__D_3, Q_temp1_lvl1_part0_2__D_0__D_1__D_2__D_3, 0 );

	}
	//****
	//**** (out of 1)
	//**** Is real	0 shadows
		//Outputs:
		//  Q_mnij__D_0__D_1__D_2__D_3
	PartitionDown(v_femn__D_0__D_1__D_2__D_3, v_femn_lvl1_part2T__D_0__D_1__D_2__D_3, v_femn_lvl1_part2B__D_0__D_1__D_2__D_3, 2, 0);
	PartitionDown(Q_temp1__D_0__D_1__D_2__D_3, Q_temp1_lvl1_part0T__D_0__D_1__D_2__D_3, Q_temp1_lvl1_part0B__D_0__D_1__D_2__D_3, 0, 0);
	PartitionDown(Q_temp1__D_0__D_1__D_2__D_3, Q_temp1_lvl1_part1T__D_0__D_1__D_2__D_3, Q_temp1_lvl1_part1B__D_0__D_1__D_2__D_3, 1, 0);
	PartitionDown(Q_mnij__D_0__D_1__D_2__D_3, Q_mnij_lvl1_part0T__D_0__D_1__D_2__D_3, Q_mnij_lvl1_part0B__D_0__D_1__D_2__D_3, 0, 0);
	while(Q_mnij_lvl1_part0T__D_0__D_1__D_2__D_3.Dimension(0) < Q_mnij__D_0__D_1__D_2__D_3.Dimension(0))
	{
		RepartitionDown
		( v_femn_lvl1_part2T__D_0__D_1__D_2__D_3,  v_femn_lvl1_part2_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       v_femn_lvl1_part2_1__D_0__D_1__D_2__D_3,
		  v_femn_lvl1_part2B__D_0__D_1__D_2__D_3, v_femn_lvl1_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );
		RepartitionDown
		( Q_temp1_lvl1_part0T__D_0__D_1__D_2__D_3,  Q_temp1_lvl1_part0_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       Q_temp1_lvl1_part0_1__D_0__D_1__D_2__D_3,
		  Q_temp1_lvl1_part0B__D_0__D_1__D_2__D_3, Q_temp1_lvl1_part0_2__D_0__D_1__D_2__D_3, 0, blkSize );
		RepartitionDown
		( Q_temp1_lvl1_part1T__D_0__D_1__D_2__D_3,  Q_temp1_lvl1_part1_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       Q_temp1_lvl1_part1_1__D_0__D_1__D_2__D_3,
		  Q_temp1_lvl1_part1B__D_0__D_1__D_2__D_3, Q_temp1_lvl1_part1_2__D_0__D_1__D_2__D_3, 1, blkSize );
		RepartitionDown
		( Q_mnij_lvl1_part0T__D_0__D_1__D_2__D_3,  Q_mnij_lvl1_part0_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       Q_mnij_lvl1_part0_1__D_0__D_1__D_2__D_3,
		  Q_mnij_lvl1_part0B__D_0__D_1__D_2__D_3, Q_mnij_lvl1_part0_2__D_0__D_1__D_2__D_3, 0, blkSize );

		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  Q_mnij_lvl1_part0_1__D_0__D_1__D_2__D_3
		PartitionDown(Q_temp1_lvl1_part0_1__D_0__D_1__D_2__D_3, Q_temp1_lvl1_part0_1_lvl2_part2T__D_0__D_1__D_2__D_3, Q_temp1_lvl1_part0_1_lvl2_part2B__D_0__D_1__D_2__D_3, 2, 0);
		PartitionDown(Q_temp1_lvl1_part1_1__D_0__D_1__D_2__D_3, Q_temp1_lvl1_part1_1_lvl2_part3T__D_0__D_1__D_2__D_3, Q_temp1_lvl1_part1_1_lvl2_part3B__D_0__D_1__D_2__D_3, 3, 0);
		PartitionDown(Q_mnij_lvl1_part0_1__D_0__D_1__D_2__D_3, Q_mnij_lvl1_part0_1_lvl2_part2T__D_0__D_1__D_2__D_3, Q_mnij_lvl1_part0_1_lvl2_part2B__D_0__D_1__D_2__D_3, 2, 0);
		while(Q_mnij_lvl1_part0_1_lvl2_part2T__D_0__D_1__D_2__D_3.Dimension(2) < Q_mnij_lvl1_part0_1__D_0__D_1__D_2__D_3.Dimension(2))
		{
			RepartitionDown
			( Q_temp1_lvl1_part0_1_lvl2_part2T__D_0__D_1__D_2__D_3,  Q_temp1_lvl1_part0_1_lvl2_part2_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       Q_temp1_lvl1_part0_1_lvl2_part2_1__D_0__D_1__D_2__D_3,
			  Q_temp1_lvl1_part0_1_lvl2_part2B__D_0__D_1__D_2__D_3, Q_temp1_lvl1_part0_1_lvl2_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );
			RepartitionDown
			( Q_temp1_lvl1_part1_1_lvl2_part3T__D_0__D_1__D_2__D_3,  Q_temp1_lvl1_part1_1_lvl2_part3_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       Q_temp1_lvl1_part1_1_lvl2_part3_1__D_0__D_1__D_2__D_3,
			  Q_temp1_lvl1_part1_1_lvl2_part3B__D_0__D_1__D_2__D_3, Q_temp1_lvl1_part1_1_lvl2_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );
			RepartitionDown
			( Q_mnij_lvl1_part0_1_lvl2_part2T__D_0__D_1__D_2__D_3,  Q_mnij_lvl1_part0_1_lvl2_part2_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       Q_mnij_lvl1_part0_1_lvl2_part2_1__D_0__D_1__D_2__D_3,
			  Q_mnij_lvl1_part0_1_lvl2_part2B__D_0__D_1__D_2__D_3, Q_mnij_lvl1_part0_1_lvl2_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );

			   // Q_temp1_lvl1_part1_1_lvl2_part3_1[D1,D0,D2,D3] <- Q_temp1_lvl1_part1_1_lvl2_part3_1[D0,D1,D2,D3]
			Q_temp1_lvl1_part1_1_lvl2_part3_1__D_1__D_0__D_2__D_3.AlignModesWith( modes_0_1_2_3, Q_temp1_lvl1_part0_1_lvl2_part2_1__D_0__D_1__D_2__D_3, modes_1_0_3_2 );
			Q_temp1_lvl1_part1_1_lvl2_part3_1__D_1__D_0__D_2__D_3.AllToAllRedistFrom( Q_temp1_lvl1_part1_1_lvl2_part3_1__D_0__D_1__D_2__D_3, modes_0_1 );
			   // Q_temp1_lvl1_part1_1_lvl2_part3_1[D1,D0,D3,D2] <- Q_temp1_lvl1_part1_1_lvl2_part3_1[D1,D0,D2,D3]
			Q_temp1_lvl1_part1_1_lvl2_part3_1__D_1__D_0__D_3__D_2.AlignModesWith( modes_0_1_2_3, Q_temp1_lvl1_part0_1_lvl2_part2_1__D_0__D_1__D_2__D_3, modes_1_0_3_2 );
			Q_temp1_lvl1_part1_1_lvl2_part3_1__D_1__D_0__D_3__D_2.AllToAllRedistFrom( Q_temp1_lvl1_part1_1_lvl2_part3_1__D_1__D_0__D_2__D_3, modes_2_3 );
			Q_temp1_lvl1_part1_1_lvl2_part3_1__D_1__D_0__D_2__D_3.EmptyData();
PROFILE_SECTION("COMPUTE");
PROFILE_FLOPS(3*prod(Q_temp1_lvl1_part0_1_lvl2_part2_1__D_0__D_1__D_2__D_3.Shape()));
			YAxpPx( 1.0, Q_temp1_lvl1_part0_1_lvl2_part2_1__D_0__D_1__D_2__D_3, 1.0, Q_temp1_lvl1_part1_1_lvl2_part3_1__D_1__D_0__D_3__D_2, perm_1_0_3_2, Q_mnij_lvl1_part0_1_lvl2_part2_1__D_0__D_1__D_2__D_3 );
PROFILE_STOP;
			Q_temp1_lvl1_part1_1_lvl2_part3_1__D_1__D_0__D_3__D_2.EmptyData();

			SlidePartitionDown
			( Q_temp1_lvl1_part0_1_lvl2_part2T__D_0__D_1__D_2__D_3,  Q_temp1_lvl1_part0_1_lvl2_part2_0__D_0__D_1__D_2__D_3,
			       Q_temp1_lvl1_part0_1_lvl2_part2_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  Q_temp1_lvl1_part0_1_lvl2_part2B__D_0__D_1__D_2__D_3, Q_temp1_lvl1_part0_1_lvl2_part2_2__D_0__D_1__D_2__D_3, 2 );
			SlidePartitionDown
			( Q_temp1_lvl1_part1_1_lvl2_part3T__D_0__D_1__D_2__D_3,  Q_temp1_lvl1_part1_1_lvl2_part3_0__D_0__D_1__D_2__D_3,
			       Q_temp1_lvl1_part1_1_lvl2_part3_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  Q_temp1_lvl1_part1_1_lvl2_part3B__D_0__D_1__D_2__D_3, Q_temp1_lvl1_part1_1_lvl2_part3_2__D_0__D_1__D_2__D_3, 3 );
			SlidePartitionDown
			( Q_mnij_lvl1_part0_1_lvl2_part2T__D_0__D_1__D_2__D_3,  Q_mnij_lvl1_part0_1_lvl2_part2_0__D_0__D_1__D_2__D_3,
			       Q_mnij_lvl1_part0_1_lvl2_part2_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  Q_mnij_lvl1_part0_1_lvl2_part2B__D_0__D_1__D_2__D_3, Q_mnij_lvl1_part0_1_lvl2_part2_2__D_0__D_1__D_2__D_3, 2 );

		}
		//****
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
			overwrite_tmpShape_Q = Q_mnij_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3.Shape();
			overwrite_tmpShape_Q.push_back( g.Shape()[0] );
			overwrite_tmpShape_Q.push_back( g.Shape()[1] );
			Q_mnij_lvl1_part0_1_lvl2_part1_1__S__S__D_2__D_3__D_0__D_1.ResizeTo( overwrite_tmpShape_Q );
			   // 1.0 * v_femn_lvl1_part2_1_lvl2_part3_1[D0,D1,*,*]_mnef * Tau_efmn[D0,D1,D2,D3]_efij + 0.0 * Q_mnij_lvl1_part0_1_lvl2_part1_1[*,*,D2,D3,D0,D1]_mnijef
PROFILE_SECTION("COMPUTE");
PROFILE_FLOPS(2*prod(Q_mnij_lvl1_part0_1_lvl2_part1_1__S__S__D_2__D_3__D_0__D_1.Shape())*v_femn_lvl1_part2_1_lvl2_part3_1_perm2301__S__S__D_0__D_1.Dimension(0)*v_femn_lvl1_part2_1_lvl2_part3_1_perm2301__S__S__D_0__D_1.Dimension(1));
			LocalContract(1.0, v_femn_lvl1_part2_1_lvl2_part3_1_perm2301__S__S__D_0__D_1.LockedTensor(), indices_mnef, false,
				Tau_efmn__D_0__D_1__D_2__D_3.LockedTensor(), indices_efij, false,
				0.0, Q_mnij_lvl1_part0_1_lvl2_part1_1__S__S__D_2__D_3__D_0__D_1.Tensor(), indices_mnijef, false);
PROFILE_STOP;
			v_femn_lvl1_part2_1_lvl2_part3_1_perm2301__S__S__D_0__D_1.EmptyData();
			   // Q_mnij_lvl1_part0_1_lvl2_part1_1[D0,D1,D2,D3] <- Q_mnij_lvl1_part0_1_lvl2_part1_1[*,*,D2,D3,D0,D1] (with SumScatter on (D0)(D1))
PROFILE_SECTION("COMPUTE");
PROFILE_FLOPS(2*prod(Q_mnij_lvl1_part0_1_lvl2_part1_1__S__S__D_2__D_3__D_0__D_1.Shape()));
			Q_mnij_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3.ReduceScatterUpdateRedistFrom( Q_mnij_lvl1_part0_1_lvl2_part1_1__S__S__D_2__D_3__D_0__D_1, 1.0, modes_5_4 );
PROFILE_STOP;
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
		SlidePartitionDown
		( Q_temp1_lvl1_part0T__D_0__D_1__D_2__D_3,  Q_temp1_lvl1_part0_0__D_0__D_1__D_2__D_3,
		       Q_temp1_lvl1_part0_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  Q_temp1_lvl1_part0B__D_0__D_1__D_2__D_3, Q_temp1_lvl1_part0_2__D_0__D_1__D_2__D_3, 0 );
		SlidePartitionDown
		( Q_temp1_lvl1_part1T__D_0__D_1__D_2__D_3,  Q_temp1_lvl1_part1_0__D_0__D_1__D_2__D_3,
		       Q_temp1_lvl1_part1_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  Q_temp1_lvl1_part1B__D_0__D_1__D_2__D_3, Q_temp1_lvl1_part1_2__D_0__D_1__D_2__D_3, 1 );

	}
	//****
	Q_temp1__D_0__D_1__D_2__D_3.EmptyData();
	Q_temp1__D_0__D_1__D_2__D_3.EmptyData();


//****
//**** (out of 1)
//**** Is real	0 shadows
	//Outputs:
	//  Q_mnij__D_0__D_1__D_2__D_3

PROFILE_SECTION("COMPUTE");
PROFILE_FLOPS(2*prod(q_mnij__D_0__D_1__D_2__D_3.Shape()));
	YAxpy( 1.0, q_mnij__D_0__D_1__D_2__D_3, Q_mnij__D_0__D_1__D_2__D_3 );
PROFILE_STOP;


//****


#ifdef CORRECTNESS
    DistTensor<double> diff_Q(dist__D_0__D_1__D_2__D_3, g);
    diff_Q.ResizeTo(check_Q);
    Diff(check_Q, Q_mnij__D_0__D_1__D_2__D_3, diff_Q);
   norm = 1.0;
   norm = Norm(diff_Q);
   if (commRank == 0){
     std::cout << "NORM_Q " << norm << std::endl;
   }
#endif
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
	PartitionDown(Tau_efmn__D_0__D_1__D_2__D_3, Tau_efmn_lvl1_part3T__D_0__D_1__D_2__D_3, Tau_efmn_lvl1_part3B__D_0__D_1__D_2__D_3, 3, 0);
	PartitionDown(t_fj__D_0_1__D_2_3, t_fj_lvl1_part1T__D_0_1__D_2_3, t_fj_lvl1_part1B__D_0_1__D_2_3, 1, 0);
	PartitionDown(x_bmej__D_0__D_1__D_2__D_3, x_bmej_lvl1_part3T__D_0__D_1__D_2__D_3, x_bmej_lvl1_part3B__D_0__D_1__D_2__D_3, 3, 0);
	PartitionDown(P_jimb__D_0__D_1__D_2__D_3, P_jimb_lvl1_part0T__D_0__D_1__D_2__D_3, P_jimb_lvl1_part0B__D_0__D_1__D_2__D_3, 0, 0);
	while(P_jimb_lvl1_part0T__D_0__D_1__D_2__D_3.Dimension(0) < P_jimb__D_0__D_1__D_2__D_3.Dimension(0))
	{
		RepartitionDown
		( Tau_efmn_lvl1_part3T__D_0__D_1__D_2__D_3,  Tau_efmn_lvl1_part3_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       Tau_efmn_lvl1_part3_1__D_0__D_1__D_2__D_3,
		  Tau_efmn_lvl1_part3B__D_0__D_1__D_2__D_3, Tau_efmn_lvl1_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );
		RepartitionDown
		( t_fj_lvl1_part1T__D_0_1__D_2_3,  t_fj_lvl1_part1_0__D_0_1__D_2_3,
		  /**/ /**/
		       t_fj_lvl1_part1_1__D_0_1__D_2_3,
		  t_fj_lvl1_part1B__D_0_1__D_2_3, t_fj_lvl1_part1_2__D_0_1__D_2_3, 1, blkSize );
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

		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  P_jimb_lvl1_part0_1__D_0__D_1__D_2__D_3
		PartitionDown(Tau_efmn_lvl1_part3_1__D_0__D_1__D_2__D_3, Tau_efmn_lvl1_part3_1_lvl2_part2T__D_0__D_1__D_2__D_3, Tau_efmn_lvl1_part3_1_lvl2_part2B__D_0__D_1__D_2__D_3, 2, 0);
		PartitionDown(w_bmje__D_0__D_1__D_2__D_3, w_bmje_lvl2_part2T__D_0__D_1__D_2__D_3, w_bmje_lvl2_part2B__D_0__D_1__D_2__D_3, 2, 0);
		PartitionDown(t_fj__D_0_1__D_2_3, t_fj_lvl2_part1T__D_0_1__D_2_3, t_fj_lvl2_part1B__D_0_1__D_2_3, 1, 0);
		PartitionDown(P_jimb_lvl1_part0_1__D_0__D_1__D_2__D_3, P_jimb_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3, P_jimb_lvl1_part0_1_lvl2_part1B__D_0__D_1__D_2__D_3, 1, 0);
		while(P_jimb_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3.Dimension(1) < P_jimb_lvl1_part0_1__D_0__D_1__D_2__D_3.Dimension(1))
		{
			RepartitionDown
			( Tau_efmn_lvl1_part3_1_lvl2_part2T__D_0__D_1__D_2__D_3,  Tau_efmn_lvl1_part3_1_lvl2_part2_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       Tau_efmn_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_2__D_3,
			  Tau_efmn_lvl1_part3_1_lvl2_part2B__D_0__D_1__D_2__D_3, Tau_efmn_lvl1_part3_1_lvl2_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );
			RepartitionDown
			( w_bmje_lvl2_part2T__D_0__D_1__D_2__D_3,  w_bmje_lvl2_part2_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       w_bmje_lvl2_part2_1__D_0__D_1__D_2__D_3,
			  w_bmje_lvl2_part2B__D_0__D_1__D_2__D_3, w_bmje_lvl2_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );
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

			   // Tau_efmn_lvl1_part3_1_lvl2_part2_1[D2,D1,D0,D3] <- Tau_efmn_lvl1_part3_1_lvl2_part2_1[D0,D1,D2,D3]
			Tau_efmn_lvl1_part3_1_lvl2_part2_1__D_2__D_1__D_0__D_3.AlignModesWith( modes_0_1, r_bmfe__D_0__D_1__D_2__D_3, modes_2_3 );
			Tau_efmn_lvl1_part3_1_lvl2_part2_1__D_2__D_1__D_0__D_3.AllToAllRedistFrom( Tau_efmn_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_2__D_3, modes_0_2 );
			   // Tau_efmn_lvl1_part3_1_lvl2_part2_1[D2,D3,D0,D1] <- Tau_efmn_lvl1_part3_1_lvl2_part2_1[D2,D1,D0,D3]
			Tau_efmn_lvl1_part3_1_lvl2_part2_1__D_2__D_3__D_0__D_1.AlignModesWith( modes_0_1, r_bmfe__D_0__D_1__D_2__D_3, modes_2_3 );
			Tau_efmn_lvl1_part3_1_lvl2_part2_1__D_2__D_3__D_0__D_1.AllToAllRedistFrom( Tau_efmn_lvl1_part3_1_lvl2_part2_1__D_2__D_1__D_0__D_3, modes_1_3 );
			Tau_efmn_lvl1_part3_1_lvl2_part2_1__D_2__D_1__D_0__D_3.EmptyData();
			   // Tau_efmn_lvl1_part3_1_lvl2_part2_1[D2,D3,*,*] <- Tau_efmn_lvl1_part3_1_lvl2_part2_1[D2,D3,D0,D1]
			Tau_efmn_lvl1_part3_1_lvl2_part2_1__D_2__D_3__S__S.AlignModesWith( modes_0_1, r_bmfe__D_0__D_1__D_2__D_3, modes_2_3 );
			Tau_efmn_lvl1_part3_1_lvl2_part2_1__D_2__D_3__S__S.AllGatherRedistFrom( Tau_efmn_lvl1_part3_1_lvl2_part2_1__D_2__D_3__D_0__D_1, modes_0_1 );
			Tau_efmn_lvl1_part3_1_lvl2_part2_1__D_2__D_3__D_0__D_1.EmptyData();
			P_jimb_lvl1_part0_1_lvl2_part1_1_P_temp__D_3__S__D_1_2__D_0.AlignModesWith( modes_0_1_2_3, P_jimb_lvl1_part0_1_lvl2_part1_1_P_temp__D_0__D_1__D_2__D_3, modes_0_1_2_3 );
			overwrite_tmpShape_P = P_jimb_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3.Shape();
			P_jimb_lvl1_part0_1_lvl2_part1_1_P_temp__D_3__S__D_1_2__D_0.ResizeTo( overwrite_tmpShape_P );
			P_jimb_lvl1_part0_1_lvl2_part1_1__D_3__S__D_1__D_0__D_2.AlignModesWith( modes_0_2_3, x_bmej_lvl1_part3_1__D_0__D_1__D_2__D_3, modes_3_1_0 );
			overwrite_tmpShape_P = P_jimb_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3.Shape();
			overwrite_tmpShape_P.push_back( g.Shape()[2] );
			P_jimb_lvl1_part0_1_lvl2_part1_1__D_3__S__D_1__D_0__D_2.ResizeTo( overwrite_tmpShape_P );
			   // t_fj_lvl2_part1_1[D01,*] <- t_fj_lvl2_part1_1[D01,D23]
			t_fj_lvl2_part1_1__D_0_1__S.AlignModesWith( modes_0, x_bmej_lvl1_part3_1__D_0__D_1__D_2__D_3, modes_2 );
			t_fj_lvl2_part1_1__D_0_1__S.AllGatherRedistFrom( t_fj_lvl2_part1_1__D_0_1__D_2_3, modes_2_3 );
			   // t_fj_lvl2_part1_1[D2,*] <- t_fj_lvl2_part1_1[D01,*]
			t_fj_lvl2_part1_1__D_2__S.AlignModesWith( modes_0, x_bmej_lvl1_part3_1__D_0__D_1__D_2__D_3, modes_2 );
			t_fj_lvl2_part1_1__D_2__S.AllToAllRedistFrom( t_fj_lvl2_part1_1__D_0_1__S, modes_0_1_2 );
			t_fj_lvl2_part1_1__D_0_1__S.EmptyData();
			   // 1.0 * x_bmej_lvl1_part3_1[D0,D1,D2,D3]_bmej * t_fj_lvl2_part1_1[D2,*]_ei + 0.0 * P_jimb_lvl1_part0_1_lvl2_part1_1[D3,*,D1,D0,D2]_jimbe
PROFILE_SECTION("COMPUTE");
PROFILE_FLOPS(2*prod(P_jimb_lvl1_part0_1_lvl2_part1_1__D_3__S__D_1__D_0__D_2.Shape())*x_bmej_lvl1_part3_1__D_0__D_1__D_2__D_3.Dimension(2));
			LocalContract(1.0, x_bmej_lvl1_part3_1__D_0__D_1__D_2__D_3.LockedTensor(), indices_bmej, true,
				t_fj_lvl2_part1_1__D_2__S.LockedTensor(), indices_ei, true,
				0.0, P_jimb_lvl1_part0_1_lvl2_part1_1__D_3__S__D_1__D_0__D_2.Tensor(), indices_jimbe, true);
PROFILE_STOP;
			t_fj_lvl2_part1_1__D_2__S.EmptyData();
			   // P_jimb_lvl1_part0_1_lvl2_part1_1_P_temp[D3,*,D12,D0] <- P_jimb_lvl1_part0_1_lvl2_part1_1[D3,*,D1,D0,D2] (with SumScatter on D2)
PROFILE_SECTION("COMPUTE");
PROFILE_FLOPS(prod(P_jimb_lvl1_part0_1_lvl2_part1_1__D_3__S__D_1__D_0__D_2.Shape()));
			P_jimb_lvl1_part0_1_lvl2_part1_1_P_temp__D_3__S__D_1_2__D_0.ReduceScatterRedistFrom( P_jimb_lvl1_part0_1_lvl2_part1_1__D_3__S__D_1__D_0__D_2, 4 );
PROFILE_STOP;
			P_jimb_lvl1_part0_1_lvl2_part1_1__D_3__S__D_1__D_0__D_2.EmptyData();
			   // P_jimb_lvl1_part0_1_lvl2_part1_1_P_temp[D0,*,D12,D3] <- P_jimb_lvl1_part0_1_lvl2_part1_1_P_temp[D3,*,D12,D0]
			P_jimb_lvl1_part0_1_lvl2_part1_1_P_temp__D_0__S__D_1_2__D_3.AlignModesWith( modes_0_1_2_3, P_jimb_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3, modes_0_1_2_3 );
			P_jimb_lvl1_part0_1_lvl2_part1_1_P_temp__D_0__S__D_1_2__D_3.AllToAllRedistFrom( P_jimb_lvl1_part0_1_lvl2_part1_1_P_temp__D_3__S__D_1_2__D_0, modes_0_3 );
			P_jimb_lvl1_part0_1_lvl2_part1_1_P_temp__D_3__S__D_1_2__D_0.EmptyData();
			   // P_jimb_lvl1_part0_1_lvl2_part1_1_P_temp[D0,D1,D2,D3] <- P_jimb_lvl1_part0_1_lvl2_part1_1_P_temp[D0,*,D12,D3]
			P_jimb_lvl1_part0_1_lvl2_part1_1_P_temp__D_0__D_1__D_2__D_3.AlignModesWith( modes_0_1_2_3, P_jimb_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3, modes_0_1_2_3 );
			P_jimb_lvl1_part0_1_lvl2_part1_1_P_temp__D_0__D_1__D_2__D_3.AllToAllRedistFrom( P_jimb_lvl1_part0_1_lvl2_part1_1_P_temp__D_0__S__D_1_2__D_3, modes_1_2 );
			P_jimb_lvl1_part0_1_lvl2_part1_1_P_temp__D_0__S__D_1_2__D_3.EmptyData();
PROFILE_SECTION("COMPUTE");
PROFILE_FLOPS(2*prod(P_jimb_lvl1_part0_1_lvl2_part1_1_P_temp__D_0__D_1__D_2__D_3.Shape()));
			YxpBy( P_jimb_lvl1_part0_1_lvl2_part1_1_P_temp__D_0__D_1__D_2__D_3, 1.0, P_jimb_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3 );
PROFILE_STOP;
			P_jimb_lvl1_part0_1_lvl2_part1_1_P_temp__S__D_2__D_1__D_0_3.AlignModesWith( modes_0_1_2_3, P_jimb_lvl1_part0_1_lvl2_part1_1_P_temp__D_0__D_1__D_2__D_3, modes_0_1_2_3 );
			overwrite_tmpShape_P = P_jimb_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3.Shape();
			P_jimb_lvl1_part0_1_lvl2_part1_1_P_temp__S__D_2__D_1__D_0_3.ResizeTo( overwrite_tmpShape_P );
			P_jimb_lvl1_part0_1_lvl2_part1_1__S__D_2__D_1__D_0__D_3.AlignModesWith( modes_1_2_3, w_bmje_lvl2_part2_1__D_0__D_1__D_2__D_3, modes_2_1_0 );
			overwrite_tmpShape_P = P_jimb_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3.Shape();
			overwrite_tmpShape_P.push_back( g.Shape()[3] );
			P_jimb_lvl1_part0_1_lvl2_part1_1__S__D_2__D_1__D_0__D_3.ResizeTo( overwrite_tmpShape_P );
			   // t_fj_lvl1_part1_1[D3,D2] <- t_fj_lvl1_part1_1[D01,D23]
			t_fj_lvl1_part1_1__D_3__D_2.AlignModesWith( modes_0, w_bmje_lvl2_part2_1__D_0__D_1__D_2__D_3, modes_3 );
			t_fj_lvl1_part1_1__D_3__D_2.AllToAllRedistFrom( t_fj_lvl1_part1_1__D_0_1__D_2_3, modes_0_1_3 );
			   // t_fj_lvl1_part1_1[D3,*] <- t_fj_lvl1_part1_1[D3,D2]
			t_fj_lvl1_part1_1__D_3__S.AlignModesWith( modes_0, w_bmje_lvl2_part2_1__D_0__D_1__D_2__D_3, modes_3 );
			t_fj_lvl1_part1_1__D_3__S.AllGatherRedistFrom( t_fj_lvl1_part1_1__D_3__D_2, modes_2 );
			t_fj_lvl1_part1_1__D_3__D_2.EmptyData();
			   // 1.0 * w_bmje_lvl2_part2_1[D0,D1,D2,D3]_bmie * t_fj_lvl1_part1_1[D3,*]_ej + 0.0 * P_jimb_lvl1_part0_1_lvl2_part1_1[*,D2,D1,D0,D3]_jimbe
PROFILE_SECTION("COMPUTE");
PROFILE_FLOPS(2*prod(P_jimb_lvl1_part0_1_lvl2_part1_1__S__D_2__D_1__D_0__D_3.Shape())*w_bmje_lvl2_part2_1__D_0__D_1__D_2__D_3.Dimension(3));
			LocalContract(1.0, w_bmje_lvl2_part2_1__D_0__D_1__D_2__D_3.LockedTensor(), indices_bmie, true,
				t_fj_lvl1_part1_1__D_3__S.LockedTensor(), indices_ej, true,
				0.0, P_jimb_lvl1_part0_1_lvl2_part1_1__S__D_2__D_1__D_0__D_3.Tensor(), indices_jimbe, true);
PROFILE_STOP;
			t_fj_lvl1_part1_1__D_3__S.EmptyData();
			   // P_jimb_lvl1_part0_1_lvl2_part1_1_P_temp[*,D2,D1,D03] <- P_jimb_lvl1_part0_1_lvl2_part1_1[*,D2,D1,D0,D3] (with SumScatter on D3)
PROFILE_SECTION("COMPUTE");
PROFILE_FLOPS(prod(P_jimb_lvl1_part0_1_lvl2_part1_1__S__D_2__D_1__D_0__D_3.Shape()));
			P_jimb_lvl1_part0_1_lvl2_part1_1_P_temp__S__D_2__D_1__D_0_3.ReduceScatterRedistFrom( P_jimb_lvl1_part0_1_lvl2_part1_1__S__D_2__D_1__D_0__D_3, 4 );
PROFILE_STOP;
			P_jimb_lvl1_part0_1_lvl2_part1_1__S__D_2__D_1__D_0__D_3.EmptyData();
			   // P_jimb_lvl1_part0_1_lvl2_part1_1_P_temp[*,D1,D2,D03] <- P_jimb_lvl1_part0_1_lvl2_part1_1_P_temp[*,D2,D1,D03]
			P_jimb_lvl1_part0_1_lvl2_part1_1_P_temp__S__D_1__D_2__D_0_3.AlignModesWith( modes_0_1_2_3, P_jimb_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3, modes_0_1_2_3 );
			P_jimb_lvl1_part0_1_lvl2_part1_1_P_temp__S__D_1__D_2__D_0_3.AllToAllRedistFrom( P_jimb_lvl1_part0_1_lvl2_part1_1_P_temp__S__D_2__D_1__D_0_3, modes_1_2 );
			P_jimb_lvl1_part0_1_lvl2_part1_1_P_temp__S__D_2__D_1__D_0_3.EmptyData();
			   // P_jimb_lvl1_part0_1_lvl2_part1_1_P_temp[D0,D1,D2,D3] <- P_jimb_lvl1_part0_1_lvl2_part1_1_P_temp[*,D1,D2,D03]
			P_jimb_lvl1_part0_1_lvl2_part1_1_P_temp__D_0__D_1__D_2__D_3.AlignModesWith( modes_0_1_2_3, P_jimb_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3, modes_0_1_2_3 );
			P_jimb_lvl1_part0_1_lvl2_part1_1_P_temp__D_0__D_1__D_2__D_3.AllToAllRedistFrom( P_jimb_lvl1_part0_1_lvl2_part1_1_P_temp__S__D_1__D_2__D_0_3, modes_0_3 );
PROFILE_SECTION("COMPUTE");
PROFILE_FLOPS(2*prod(P_jimb_lvl1_part0_1_lvl2_part1_1_P_temp__D_0__D_1__D_2__D_3.Shape()));
			YxpBy( P_jimb_lvl1_part0_1_lvl2_part1_1_P_temp__D_0__D_1__D_2__D_3, 1.0, P_jimb_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3 );
PROFILE_STOP;
			P_jimb_lvl1_part0_1_lvl2_part1_1_P_temp__S__S__D_1_2__D_0_3.AlignModesWith( modes_0_1_2_3, P_jimb_lvl1_part0_1_lvl2_part1_1_P_temp__D_0__D_1__D_2__D_3, modes_0_1_2_3 );
			overwrite_tmpShape_P = P_jimb_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3.Shape();
			P_jimb_lvl1_part0_1_lvl2_part1_1_P_temp__S__S__D_1_2__D_0_3.ResizeTo( overwrite_tmpShape_P );
			P_jimb_lvl1_part0_1_lvl2_part1_1_perm321045__D_0__D_1__S__S__D_2__D_3.AlignModesWith( modes_2_3, r_bmfe__D_0__D_1__D_2__D_3, modes_1_0 );
			overwrite_tmpShape_P = P_jimb_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3.Shape();
			overwrite_tmpShape_P.push_back( g.Shape()[2] );
			overwrite_tmpShape_P.push_back( g.Shape()[3] );
			P_jimb_lvl1_part0_1_lvl2_part1_1_perm321045__D_0__D_1__S__S__D_2__D_3.ResizeTo( overwrite_tmpShape_P );
			   // 1.0 * r_bmfe[D0,D1,D2,D3]_bmef * Tau_efmn_lvl1_part3_1_lvl2_part2_1[D2,D3,*,*]_efij + 0.0 * P_jimb_lvl1_part0_1_lvl2_part1_1[*,*,D1,D0,D2,D3]_bmijef
PROFILE_SECTION("COMPUTE");
PROFILE_FLOPS(2*prod(P_jimb_lvl1_part0_1_lvl2_part1_1_perm321045__D_0__D_1__S__S__D_2__D_3.Shape())*r_bmfe__D_0__D_1__D_2__D_3.Dimension(2)*r_bmfe__D_0__D_1__D_2__D_3.Dimension(3));
			LocalContract(1.0, r_bmfe__D_0__D_1__D_2__D_3.LockedTensor(), indices_bmef, false,
				Tau_efmn_lvl1_part3_1_lvl2_part2_1__D_2__D_3__S__S.LockedTensor(), indices_efij, false,
				0.0, P_jimb_lvl1_part0_1_lvl2_part1_1_perm321045__D_0__D_1__S__S__D_2__D_3.Tensor(), indices_bmijef, false);
PROFILE_STOP;
			Tau_efmn_lvl1_part3_1_lvl2_part2_1__D_2__D_3__S__S.EmptyData();
			   // P_jimb_lvl1_part0_1_lvl2_part1_1_P_temp[*,*,D12,D03] <- P_jimb_lvl1_part0_1_lvl2_part1_1[*,*,D1,D0,D2,D3] (with SumScatter on (D2)(D3))
PROFILE_SECTION("COMPUTE");
PROFILE_FLOPS(prod(P_jimb_lvl1_part0_1_lvl2_part1_1_perm321045__D_0__D_1__S__S__D_2__D_3.Shape()));
			P_jimb_lvl1_part0_1_lvl2_part1_1_P_temp__S__S__D_1_2__D_0_3.ReduceScatterRedistFrom( P_jimb_lvl1_part0_1_lvl2_part1_1_perm321045__D_0__D_1__S__S__D_2__D_3, modes_5_4 );
PROFILE_STOP;
			P_jimb_lvl1_part0_1_lvl2_part1_1_perm321045__D_0__D_1__S__S__D_2__D_3.EmptyData();
			   // P_jimb_lvl1_part0_1_lvl2_part1_1_P_temp[*,*,D21,D03] <- P_jimb_lvl1_part0_1_lvl2_part1_1_P_temp[*,*,D12,D03]
			P_jimb_lvl1_part0_1_lvl2_part1_1_P_temp__S__S__D_2_1__D_0_3.AlignModesWith( modes_0_1_2_3, P_jimb_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3, modes_0_1_2_3 );
			P_jimb_lvl1_part0_1_lvl2_part1_1_P_temp__S__S__D_2_1__D_0_3.AllToAllRedistFrom( P_jimb_lvl1_part0_1_lvl2_part1_1_P_temp__S__S__D_1_2__D_0_3, modes_1_2 );
			P_jimb_lvl1_part0_1_lvl2_part1_1_P_temp__S__S__D_1_2__D_0_3.EmptyData();
			   // P_jimb_lvl1_part0_1_lvl2_part1_1_P_temp[*,D1,D2,D03] <- P_jimb_lvl1_part0_1_lvl2_part1_1_P_temp[*,*,D21,D03]
			P_jimb_lvl1_part0_1_lvl2_part1_1_P_temp__S__D_1__D_2__D_0_3.AlignModesWith( modes_0_1_2_3, P_jimb_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3, modes_0_1_2_3 );
			P_jimb_lvl1_part0_1_lvl2_part1_1_P_temp__S__D_1__D_2__D_0_3.AllToAllRedistFrom( P_jimb_lvl1_part0_1_lvl2_part1_1_P_temp__S__S__D_2_1__D_0_3, modes_1 );
			P_jimb_lvl1_part0_1_lvl2_part1_1_P_temp__S__S__D_2_1__D_0_3.EmptyData();
			   // P_jimb_lvl1_part0_1_lvl2_part1_1_P_temp[D0,D1,D2,D3] <- P_jimb_lvl1_part0_1_lvl2_part1_1_P_temp[*,D1,D2,D03]
			P_jimb_lvl1_part0_1_lvl2_part1_1_P_temp__D_0__D_1__D_2__D_3.AlignModesWith( modes_0_1_2_3, P_jimb_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3, modes_0_1_2_3 );
			P_jimb_lvl1_part0_1_lvl2_part1_1_P_temp__D_0__D_1__D_2__D_3.AllToAllRedistFrom( P_jimb_lvl1_part0_1_lvl2_part1_1_P_temp__S__D_1__D_2__D_0_3, modes_0_3 );
			P_jimb_lvl1_part0_1_lvl2_part1_1_P_temp__S__D_1__D_2__D_0_3.EmptyData();
PROFILE_SECTION("COMPUTE");
PROFILE_FLOPS(2*prod(P_jimb_lvl1_part0_1_lvl2_part1_1_P_temp__D_0__D_1__D_2__D_3.Shape()));
			YxpBy( P_jimb_lvl1_part0_1_lvl2_part1_1_P_temp__D_0__D_1__D_2__D_3, 1.0, P_jimb_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3 );
PROFILE_STOP;
			P_jimb_lvl1_part0_1_lvl2_part1_1_P_temp__D_0__D_1__D_2__D_3.EmptyData();

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
			SlidePartitionDown
			( w_bmje_lvl2_part2T__D_0__D_1__D_2__D_3,  w_bmje_lvl2_part2_0__D_0__D_1__D_2__D_3,
			       w_bmje_lvl2_part2_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  w_bmje_lvl2_part2B__D_0__D_1__D_2__D_3, w_bmje_lvl2_part2_2__D_0__D_1__D_2__D_3, 2 );
			SlidePartitionDown
			( t_fj_lvl2_part1T__D_0_1__D_2_3,  t_fj_lvl2_part1_0__D_0_1__D_2_3,
			       t_fj_lvl2_part1_1__D_0_1__D_2_3,
			  /**/ /**/
			  t_fj_lvl2_part1B__D_0_1__D_2_3, t_fj_lvl2_part1_2__D_0_1__D_2_3, 1 );

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
		SlidePartitionDown
		( t_fj_lvl1_part1T__D_0_1__D_2_3,  t_fj_lvl1_part1_0__D_0_1__D_2_3,
		       t_fj_lvl1_part1_1__D_0_1__D_2_3,
		  /**/ /**/
		  t_fj_lvl1_part1B__D_0_1__D_2_3, t_fj_lvl1_part1_2__D_0_1__D_2_3, 1 );
		SlidePartitionDown
		( x_bmej_lvl1_part3T__D_0__D_1__D_2__D_3,  x_bmej_lvl1_part3_0__D_0__D_1__D_2__D_3,
		       x_bmej_lvl1_part3_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  x_bmej_lvl1_part3B__D_0__D_1__D_2__D_3, x_bmej_lvl1_part3_2__D_0__D_1__D_2__D_3, 3 );

	}
	//****


//****


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
overwrite_tmpShape_H = v_femn__D_0__D_1__D_2__D_3.Shape();
H_temp1__D_0__D_1__D_2__D_3.ResizeTo( overwrite_tmpShape_H );
//**** (out of 1)
//**** Is real	0 shadows
	//Outputs:
	//  H_me__D_0_1__D_2_3

	//**** (out of 1)
	//**** Is real	0 shadows
		//Outputs:
		//  H_temp1__D_0__D_1__D_2__D_3
	PartitionDown(v_femn__D_0__D_1__D_2__D_3, v_femn_lvl1_part2T__D_0__D_1__D_2__D_3, v_femn_lvl1_part2B__D_0__D_1__D_2__D_3, 2, 0);
	PartitionDown(v_femn__D_0__D_1__D_2__D_3, v_femn_lvl1_part3T__D_0__D_1__D_2__D_3, v_femn_lvl1_part3B__D_0__D_1__D_2__D_3, 3, 0);
	PartitionDown(H_temp1__D_0__D_1__D_2__D_3, H_temp1_lvl1_part2T__D_0__D_1__D_2__D_3, H_temp1_lvl1_part2B__D_0__D_1__D_2__D_3, 2, 0);
	while(H_temp1_lvl1_part2T__D_0__D_1__D_2__D_3.Dimension(2) < H_temp1__D_0__D_1__D_2__D_3.Dimension(2))
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
		( H_temp1_lvl1_part2T__D_0__D_1__D_2__D_3,  H_temp1_lvl1_part2_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       H_temp1_lvl1_part2_1__D_0__D_1__D_2__D_3,
		  H_temp1_lvl1_part2B__D_0__D_1__D_2__D_3, H_temp1_lvl1_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );

		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  H_temp1_lvl1_part2_1__D_0__D_1__D_2__D_3
		PartitionDown(v_femn_lvl1_part2_1__D_0__D_1__D_2__D_3, v_femn_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3, v_femn_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, 3, 0);
		PartitionDown(v_femn_lvl1_part3_1__D_0__D_1__D_2__D_3, v_femn_lvl1_part3_1_lvl2_part2T__D_0__D_1__D_2__D_3, v_femn_lvl1_part3_1_lvl2_part2B__D_0__D_1__D_2__D_3, 2, 0);
		PartitionDown(H_temp1_lvl1_part2_1__D_0__D_1__D_2__D_3, H_temp1_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3, H_temp1_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, 3, 0);
		while(H_temp1_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3.Dimension(3) < H_temp1_lvl1_part2_1__D_0__D_1__D_2__D_3.Dimension(3))
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
			( H_temp1_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3,  H_temp1_lvl1_part2_1_lvl2_part3_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       H_temp1_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3,
			  H_temp1_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, H_temp1_lvl1_part2_1_lvl2_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );

			   // v_femn_lvl1_part3_1_lvl2_part2_1[D0,D1,D3,D2] <- v_femn_lvl1_part3_1_lvl2_part2_1[D0,D1,D2,D3]
			v_femn_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_3__D_2.AlignModesWith( modes_0_1_2_3, v_femn_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3, modes_0_1_3_2 );
			v_femn_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_3__D_2.AllToAllRedistFrom( v_femn_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_2__D_3, modes_2_3 );
PROFILE_SECTION("COMPUTE");
PROFILE_FLOPS(3*prod(v_femn_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3.Shape()));
			YAxpPx( 2.0, v_femn_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3, -1.0, v_femn_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_3__D_2, perm_0_1_3_2, H_temp1_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3 );
PROFILE_STOP;
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
			( H_temp1_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3,  H_temp1_lvl1_part2_1_lvl2_part3_0__D_0__D_1__D_2__D_3,
			       H_temp1_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  H_temp1_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, H_temp1_lvl1_part2_1_lvl2_part3_2__D_0__D_1__D_2__D_3, 3 );

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
		( H_temp1_lvl1_part2T__D_0__D_1__D_2__D_3,  H_temp1_lvl1_part2_0__D_0__D_1__D_2__D_3,
		       H_temp1_lvl1_part2_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  H_temp1_lvl1_part2B__D_0__D_1__D_2__D_3, H_temp1_lvl1_part2_2__D_0__D_1__D_2__D_3, 2 );

	}
	//****
PROFILE_SECTION("COMPUTE");
PROFILE_FLOPS(prod(H_me__D_0_1__D_2_3.Shape()));
	Scal( 0.0, H_me__D_0_1__D_2_3 );
PROFILE_STOP;
	//**** (out of 1)
	//**** Is real	0 shadows
		//Outputs:
		//  H_me__D_0_1__D_2_3
	PartitionDown(H_temp1__D_0__D_1__D_2__D_3, H_temp1_lvl1_part3T__D_0__D_1__D_2__D_3, H_temp1_lvl1_part3B__D_0__D_1__D_2__D_3, 3, 0);
	PartitionDown(t_fj__D_0_1__D_2_3, t_fj_lvl1_part1T__D_0_1__D_2_3, t_fj_lvl1_part1B__D_0_1__D_2_3, 1, 0);
	while(H_temp1_lvl1_part3T__D_0__D_1__D_2__D_3.Dimension(3) < H_temp1__D_0__D_1__D_2__D_3.Dimension(3))
	{
		RepartitionDown
		( H_temp1_lvl1_part3T__D_0__D_1__D_2__D_3,  H_temp1_lvl1_part3_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       H_temp1_lvl1_part3_1__D_0__D_1__D_2__D_3,
		  H_temp1_lvl1_part3B__D_0__D_1__D_2__D_3, H_temp1_lvl1_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );
		RepartitionDown
		( t_fj_lvl1_part1T__D_0_1__D_2_3,  t_fj_lvl1_part1_0__D_0_1__D_2_3,
		  /**/ /**/
		       t_fj_lvl1_part1_1__D_0_1__D_2_3,
		  t_fj_lvl1_part1B__D_0_1__D_2_3, t_fj_lvl1_part1_2__D_0_1__D_2_3, 1, blkSize );

		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  H_me__D_0_1__D_2_3
		PartitionDown(H_temp1_lvl1_part3_1__D_0__D_1__D_2__D_3, H_temp1_lvl1_part3_1_lvl2_part2T__D_0__D_1__D_2__D_3, H_temp1_lvl1_part3_1_lvl2_part2B__D_0__D_1__D_2__D_3, 2, 0);
		PartitionDown(H_me__D_0_1__D_2_3, H_me_lvl2_part0T__D_0_1__D_2_3, H_me_lvl2_part0B__D_0_1__D_2_3, 0, 0);
		while(H_me_lvl2_part0T__D_0_1__D_2_3.Dimension(0) < H_me__D_0_1__D_2_3.Dimension(0))
		{
			RepartitionDown
			( H_temp1_lvl1_part3_1_lvl2_part2T__D_0__D_1__D_2__D_3,  H_temp1_lvl1_part3_1_lvl2_part2_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       H_temp1_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_2__D_3,
			  H_temp1_lvl1_part3_1_lvl2_part2B__D_0__D_1__D_2__D_3, H_temp1_lvl1_part3_1_lvl2_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );
			RepartitionDown
			( H_me_lvl2_part0T__D_0_1__D_2_3,  H_me_lvl2_part0_0__D_0_1__D_2_3,
			  /**/ /**/
			       H_me_lvl2_part0_1__D_0_1__D_2_3,
			  H_me_lvl2_part0B__D_0_1__D_2_3, H_me_lvl2_part0_2__D_0_1__D_2_3, 0, blkSize );

			Permute( H_me_lvl2_part0_1__D_0_1__D_2_3, H_me_lvl2_part0_1_perm10__D_2_3__D_0_1 );
			   // t_fj_lvl1_part1_1[*,*] <- t_fj_lvl1_part1_1[D01,D23]
			t_fj_lvl1_part1_1__S__S.AllGatherRedistFrom( t_fj_lvl1_part1_1__D_0_1__D_2_3, modes_0_1_2_3 );
			   // H_temp1_lvl1_part3_1_lvl2_part2_1[D2,D1,D0,D3] <- H_temp1_lvl1_part3_1_lvl2_part2_1[D0,D1,D2,D3]
			H_temp1_lvl1_part3_1_lvl2_part2_1__D_2__D_1__D_0__D_3.AlignModesWith( modes_0_2, H_me_lvl2_part0_1__D_0_1__D_2_3, modes_1_0 );
			H_temp1_lvl1_part3_1_lvl2_part2_1__D_2__D_1__D_0__D_3.AllToAllRedistFrom( H_temp1_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_2__D_3, modes_0_2 );
			   // H_temp1_lvl1_part3_1_lvl2_part2_1[D23,D1,D0,*] <- H_temp1_lvl1_part3_1_lvl2_part2_1[D2,D1,D0,D3]
			H_temp1_lvl1_part3_1_lvl2_part2_1__D_2_3__D_1__D_0__S.AlignModesWith( modes_0_2, H_me_lvl2_part0_1__D_0_1__D_2_3, modes_1_0 );
			H_temp1_lvl1_part3_1_lvl2_part2_1__D_2_3__D_1__D_0__S.AllToAllRedistFrom( H_temp1_lvl1_part3_1_lvl2_part2_1__D_2__D_1__D_0__D_3, modes_3 );
			H_temp1_lvl1_part3_1_lvl2_part2_1__D_2__D_1__D_0__D_3.EmptyData();
			   // H_temp1_lvl1_part3_1_lvl2_part2_1[D23,*,D01,*] <- H_temp1_lvl1_part3_1_lvl2_part2_1[D23,D1,D0,*]
			H_temp1_lvl1_part3_1_lvl2_part2_1_perm0213__D_2_3__D_0_1__S__S.AlignModesWith( modes_0_2, H_me_lvl2_part0_1__D_0_1__D_2_3, modes_1_0 );
			H_temp1_lvl1_part3_1_lvl2_part2_1_perm0213__D_2_3__D_0_1__S__S.AllToAllRedistFrom( H_temp1_lvl1_part3_1_lvl2_part2_1__D_2_3__D_1__D_0__S, modes_1 );
			H_temp1_lvl1_part3_1_lvl2_part2_1__D_2_3__D_1__D_0__S.EmptyData();
			   // 1.0 * H_temp1_lvl1_part3_1_lvl2_part2_1[D23,*,D01,*]_emfn * t_fj_lvl1_part1_1[*,*]_fn + 1.0 * H_me_lvl2_part0_1[D01,D23]_em
PROFILE_SECTION("COMPUTE");
PROFILE_FLOPS(2*prod(H_me_lvl2_part0_1_perm10__D_2_3__D_0_1.Shape())*H_temp1_lvl1_part3_1_lvl2_part2_1_perm0213__D_2_3__D_0_1__S__S.Dimension(1)*H_temp1_lvl1_part3_1_lvl2_part2_1_perm0213__D_2_3__D_0_1__S__S.Dimension(3));
			LocalContractAndLocalEliminate(1.0, H_temp1_lvl1_part3_1_lvl2_part2_1_perm0213__D_2_3__D_0_1__S__S.LockedTensor(), indices_emfn, false,
				t_fj_lvl1_part1_1__S__S.LockedTensor(), indices_fn, false,
				1.0, H_me_lvl2_part0_1_perm10__D_2_3__D_0_1.Tensor(), indices_em, false);
PROFILE_STOP;
			H_temp1_lvl1_part3_1_lvl2_part2_1_perm0213__D_2_3__D_0_1__S__S.EmptyData();
			t_fj_lvl1_part1_1__S__S.EmptyData();
			Permute( H_me_lvl2_part0_1_perm10__D_2_3__D_0_1, H_me_lvl2_part0_1__D_0_1__D_2_3 );
			H_me_lvl2_part0_1_perm10__D_2_3__D_0_1.EmptyData();

			SlidePartitionDown
			( H_temp1_lvl1_part3_1_lvl2_part2T__D_0__D_1__D_2__D_3,  H_temp1_lvl1_part3_1_lvl2_part2_0__D_0__D_1__D_2__D_3,
			       H_temp1_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  H_temp1_lvl1_part3_1_lvl2_part2B__D_0__D_1__D_2__D_3, H_temp1_lvl1_part3_1_lvl2_part2_2__D_0__D_1__D_2__D_3, 2 );
			SlidePartitionDown
			( H_me_lvl2_part0T__D_0_1__D_2_3,  H_me_lvl2_part0_0__D_0_1__D_2_3,
			       H_me_lvl2_part0_1__D_0_1__D_2_3,
			  /**/ /**/
			  H_me_lvl2_part0B__D_0_1__D_2_3, H_me_lvl2_part0_2__D_0_1__D_2_3, 0 );

		}
		//****

		SlidePartitionDown
		( H_temp1_lvl1_part3T__D_0__D_1__D_2__D_3,  H_temp1_lvl1_part3_0__D_0__D_1__D_2__D_3,
		       H_temp1_lvl1_part3_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  H_temp1_lvl1_part3B__D_0__D_1__D_2__D_3, H_temp1_lvl1_part3_2__D_0__D_1__D_2__D_3, 3 );
		SlidePartitionDown
		( t_fj_lvl1_part1T__D_0_1__D_2_3,  t_fj_lvl1_part1_0__D_0_1__D_2_3,
		       t_fj_lvl1_part1_1__D_0_1__D_2_3,
		  /**/ /**/
		  t_fj_lvl1_part1B__D_0_1__D_2_3, t_fj_lvl1_part1_2__D_0_1__D_2_3, 1 );

	}
	//****
	H_temp1__D_0__D_1__D_2__D_3.EmptyData();


//****


#ifdef CORRECTNESS
    DistTensor<double> diff_H(dist__D_0_1__D_2_3, g);
    diff_H.ResizeTo(check_H);
    Diff(check_H, H_me__D_0_1__D_2_3, diff_H);
   norm = 1.0;
   norm = Norm(diff_H);
   if (commRank == 0){
     std::cout << "NORM_H " << norm << std::endl;
   }
#endif
overwrite_tmpShape_F = r_bmfe__D_0__D_1__D_2__D_3.Shape();
F_temp2__D_0__D_1__D_2__D_3.ResizeTo( overwrite_tmpShape_F );
//**** (out of 1)
//**** Is real	0 shadows
	//Outputs:
	//  F_ae__D_0_1__D_2_3

	overwrite_tmpShape_F = v_femn__D_0__D_1__D_2__D_3.Shape();
	F_temp1__D_0__D_1__D_2__D_3.ResizeTo( overwrite_tmpShape_F );
PROFILE_SECTION("COMPUTE");
PROFILE_FLOPS(prod(F_ae__D_0_1__D_2_3.Shape()));
	Scal( 0.0, F_ae__D_0_1__D_2_3 );
PROFILE_STOP;
	//**** (out of 1)
	//**** Is real	0 shadows
		//Outputs:
		//  F_ae__D_0_1__D_2_3
	PartitionDown(T_bfnj__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part2T__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part2B__D_0__D_1__D_2__D_3, 2, 0);
	PartitionDown(v_femn__D_0__D_1__D_2__D_3, v_femn_lvl1_part2T__D_0__D_1__D_2__D_3, v_femn_lvl1_part2B__D_0__D_1__D_2__D_3, 2, 0);
	PartitionDown(v_femn__D_0__D_1__D_2__D_3, v_femn_lvl1_part3T__D_0__D_1__D_2__D_3, v_femn_lvl1_part3B__D_0__D_1__D_2__D_3, 3, 0);
	PartitionDown(F_temp1__D_0__D_1__D_2__D_3, F_temp1_lvl1_part2T__D_0__D_1__D_2__D_3, F_temp1_lvl1_part2B__D_0__D_1__D_2__D_3, 2, 0);
	while(F_temp1_lvl1_part2T__D_0__D_1__D_2__D_3.Dimension(2) < F_temp1__D_0__D_1__D_2__D_3.Dimension(2))
	{
		RepartitionDown
		( T_bfnj_lvl1_part2T__D_0__D_1__D_2__D_3,  T_bfnj_lvl1_part2_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       T_bfnj_lvl1_part2_1__D_0__D_1__D_2__D_3,
		  T_bfnj_lvl1_part2B__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );
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
		( F_temp1_lvl1_part2T__D_0__D_1__D_2__D_3,  F_temp1_lvl1_part2_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       F_temp1_lvl1_part2_1__D_0__D_1__D_2__D_3,
		  F_temp1_lvl1_part2B__D_0__D_1__D_2__D_3, F_temp1_lvl1_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );

		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  F_temp1_lvl1_part2_1__D_0__D_1__D_2__D_3
			//  F_ae__D_0_1__D_2_3
		PartitionDown(T_bfnj_lvl1_part2_1__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, 3, 0);
		PartitionDown(v_femn_lvl1_part2_1__D_0__D_1__D_2__D_3, v_femn_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3, v_femn_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, 3, 0);
		PartitionDown(v_femn_lvl1_part3_1__D_0__D_1__D_2__D_3, v_femn_lvl1_part3_1_lvl2_part2T__D_0__D_1__D_2__D_3, v_femn_lvl1_part3_1_lvl2_part2B__D_0__D_1__D_2__D_3, 2, 0);
		PartitionDown(F_temp1_lvl1_part2_1__D_0__D_1__D_2__D_3, F_temp1_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3, F_temp1_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, 3, 0);
		while(F_temp1_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3.Dimension(3) < F_temp1_lvl1_part2_1__D_0__D_1__D_2__D_3.Dimension(3))
		{
			RepartitionDown
			( T_bfnj_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3,  T_bfnj_lvl1_part2_1_lvl2_part3_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       T_bfnj_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3,
			  T_bfnj_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part2_1_lvl2_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );
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
			( F_temp1_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3,  F_temp1_lvl1_part2_1_lvl2_part3_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       F_temp1_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3,
			  F_temp1_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, F_temp1_lvl1_part2_1_lvl2_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );

			Permute( T_bfnj_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part2_1_lvl2_part3_1_perm1230__D_1__D_2__D_3__D_0 );
			   // v_femn_lvl1_part3_1_lvl2_part2_1[D0,D1,D3,D2] <- v_femn_lvl1_part3_1_lvl2_part2_1[D0,D1,D2,D3]
			v_femn_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_3__D_2.AlignModesWith( modes_0_1_2_3, v_femn_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3, modes_0_1_3_2 );
			v_femn_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_3__D_2.AllToAllRedistFrom( v_femn_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_2__D_3, modes_2_3 );
PROFILE_SECTION("COMPUTE");
PROFILE_FLOPS(3*prod(v_femn_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3.Shape()));
			YAxpPx( 2.0, v_femn_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3, -1.0, v_femn_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_3__D_2, perm_0_1_3_2, F_temp1_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3 );
PROFILE_STOP;
			v_femn_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_3__D_2.EmptyData();
			   // F_temp1_lvl1_part2_1_lvl2_part3_1[*,D1,D2,D3] <- F_temp1_lvl1_part2_1_lvl2_part3_1[D0,D1,D2,D3]
			F_temp1_lvl1_part2_1_lvl2_part3_1__S__D_1__D_2__D_3.AlignModesWith( modes_1_2_3, T_bfnj_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3, modes_1_2_3 );
			F_temp1_lvl1_part2_1_lvl2_part3_1__S__D_1__D_2__D_3.AllGatherRedistFrom( F_temp1_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3, modes_0 );
			F_ae_perm10234__S__D_0__D_1__D_2__D_3.AlignModesWith( modes_0, T_bfnj_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3, modes_0 );
			overwrite_tmpShape_F = F_ae__D_0_1__D_2_3.Shape();
			overwrite_tmpShape_F.push_back( g.Shape()[1] );
			overwrite_tmpShape_F.push_back( g.Shape()[2] );
			overwrite_tmpShape_F.push_back( g.Shape()[3] );
			F_ae_perm10234__S__D_0__D_1__D_2__D_3.ResizeTo( overwrite_tmpShape_F );
			   // -1.0 * F_temp1_lvl1_part2_1_lvl2_part3_1[*,D1,D2,D3]_efmn * T_bfnj_lvl1_part2_1_lvl2_part3_1[D0,D1,D2,D3]_fmna + 0.0 * F_ae[D0,*,D1,D2,D3]_eafmn
PROFILE_SECTION("COMPUTE");
PROFILE_FLOPS(2*prod(F_ae_perm10234__S__D_0__D_1__D_2__D_3.Shape())*F_temp1_lvl1_part2_1_lvl2_part3_1__S__D_1__D_2__D_3.Dimension(1)*F_temp1_lvl1_part2_1_lvl2_part3_1__S__D_1__D_2__D_3.Dimension(2)*F_temp1_lvl1_part2_1_lvl2_part3_1__S__D_1__D_2__D_3.Dimension(3));
			LocalContract(-1.0, F_temp1_lvl1_part2_1_lvl2_part3_1__S__D_1__D_2__D_3.LockedTensor(), indices_efmn, false,
				T_bfnj_lvl1_part2_1_lvl2_part3_1_perm1230__D_1__D_2__D_3__D_0.LockedTensor(), indices_fmna, false,
				0.0, F_ae_perm10234__S__D_0__D_1__D_2__D_3.Tensor(), indices_eafmn, false);
PROFILE_STOP;
			F_temp1_lvl1_part2_1_lvl2_part3_1__S__D_1__D_2__D_3.EmptyData();
			T_bfnj_lvl1_part2_1_lvl2_part3_1_perm1230__D_1__D_2__D_3__D_0.EmptyData();
			   // F_ae[D01,D23] <- F_ae[D0,*,D1,D2,D3] (with SumScatter on (D1)(D2)(D3))
PROFILE_SECTION("COMPUTE");
PROFILE_FLOPS(2*prod(F_ae_perm10234__S__D_0__D_1__D_2__D_3.Shape()));
			F_ae__D_0_1__D_2_3.ReduceScatterUpdateRedistFrom( F_ae_perm10234__S__D_0__D_1__D_2__D_3, 1.0, modes_4_3_2 );
PROFILE_STOP;
			F_ae_perm10234__S__D_0__D_1__D_2__D_3.EmptyData();

			SlidePartitionDown
			( F_temp1_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3,  F_temp1_lvl1_part2_1_lvl2_part3_0__D_0__D_1__D_2__D_3,
			       F_temp1_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  F_temp1_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, F_temp1_lvl1_part2_1_lvl2_part3_2__D_0__D_1__D_2__D_3, 3 );
			SlidePartitionDown
			( T_bfnj_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3,  T_bfnj_lvl1_part2_1_lvl2_part3_0__D_0__D_1__D_2__D_3,
			       T_bfnj_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  T_bfnj_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part2_1_lvl2_part3_2__D_0__D_1__D_2__D_3, 3 );
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

		}
		//****

		SlidePartitionDown
		( F_temp1_lvl1_part2T__D_0__D_1__D_2__D_3,  F_temp1_lvl1_part2_0__D_0__D_1__D_2__D_3,
		       F_temp1_lvl1_part2_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  F_temp1_lvl1_part2B__D_0__D_1__D_2__D_3, F_temp1_lvl1_part2_2__D_0__D_1__D_2__D_3, 2 );
		SlidePartitionDown
		( T_bfnj_lvl1_part2T__D_0__D_1__D_2__D_3,  T_bfnj_lvl1_part2_0__D_0__D_1__D_2__D_3,
		       T_bfnj_lvl1_part2_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  T_bfnj_lvl1_part2B__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part2_2__D_0__D_1__D_2__D_3, 2 );
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

	}
	//****
	F_temp1__D_0__D_1__D_2__D_3.EmptyData();
	//**** (out of 1)
	//**** Is real	0 shadows
		//Outputs:
		//  F_temp2__D_0__D_1__D_2__D_3
	PartitionDown(r_bmfe__D_0__D_1__D_2__D_3, r_bmfe_lvl1_part0T__D_0__D_1__D_2__D_3, r_bmfe_lvl1_part0B__D_0__D_1__D_2__D_3, 0, 0);
	PartitionDown(F_temp2__D_0__D_1__D_2__D_3, F_temp2_lvl1_part0T__D_0__D_1__D_2__D_3, F_temp2_lvl1_part0B__D_0__D_1__D_2__D_3, 0, 0);
	while(F_temp2_lvl1_part0T__D_0__D_1__D_2__D_3.Dimension(0) < F_temp2__D_0__D_1__D_2__D_3.Dimension(0))
	{
		RepartitionDown
		( r_bmfe_lvl1_part0T__D_0__D_1__D_2__D_3,  r_bmfe_lvl1_part0_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       r_bmfe_lvl1_part0_1__D_0__D_1__D_2__D_3,
		  r_bmfe_lvl1_part0B__D_0__D_1__D_2__D_3, r_bmfe_lvl1_part0_2__D_0__D_1__D_2__D_3, 0, blkSize );
		RepartitionDown
		( F_temp2_lvl1_part0T__D_0__D_1__D_2__D_3,  F_temp2_lvl1_part0_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       F_temp2_lvl1_part0_1__D_0__D_1__D_2__D_3,
		  F_temp2_lvl1_part0B__D_0__D_1__D_2__D_3, F_temp2_lvl1_part0_2__D_0__D_1__D_2__D_3, 0, blkSize );

		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  F_temp2_lvl1_part0_1__D_0__D_1__D_2__D_3
		PartitionDown(r_bmfe_lvl1_part0_1__D_0__D_1__D_2__D_3, r_bmfe_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3, r_bmfe_lvl1_part0_1_lvl2_part1B__D_0__D_1__D_2__D_3, 1, 0);
		PartitionDown(F_temp2_lvl1_part0_1__D_0__D_1__D_2__D_3, F_temp2_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3, F_temp2_lvl1_part0_1_lvl2_part1B__D_0__D_1__D_2__D_3, 1, 0);
		while(F_temp2_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3.Dimension(1) < F_temp2_lvl1_part0_1__D_0__D_1__D_2__D_3.Dimension(1))
		{
			RepartitionDown
			( r_bmfe_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3,  r_bmfe_lvl1_part0_1_lvl2_part1_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       r_bmfe_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3,
			  r_bmfe_lvl1_part0_1_lvl2_part1B__D_0__D_1__D_2__D_3, r_bmfe_lvl1_part0_1_lvl2_part1_2__D_0__D_1__D_2__D_3, 1, blkSize );
			RepartitionDown
			( F_temp2_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3,  F_temp2_lvl1_part0_1_lvl2_part1_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       F_temp2_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3,
			  F_temp2_lvl1_part0_1_lvl2_part1B__D_0__D_1__D_2__D_3, F_temp2_lvl1_part0_1_lvl2_part1_2__D_0__D_1__D_2__D_3, 1, blkSize );

			   // r_bmfe_lvl1_part0_1_lvl2_part1_1[D0,D1,D3,D2] <- r_bmfe_lvl1_part0_1_lvl2_part1_1[D0,D1,D2,D3]
			r_bmfe_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_3__D_2.AlignModesWith( modes_0_1_2_3, r_bmfe_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3, modes_0_1_3_2 );
			r_bmfe_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_3__D_2.AllToAllRedistFrom( r_bmfe_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3, modes_2_3 );
PROFILE_SECTION("COMPUTE");
PROFILE_FLOPS(3*prod(r_bmfe_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3.Shape()));
			YAxpPx( 2.0, r_bmfe_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3, -1.0, r_bmfe_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_3__D_2, perm_0_1_3_2, F_temp2_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3 );
PROFILE_STOP;
			r_bmfe_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_3__D_2.EmptyData();

			SlidePartitionDown
			( r_bmfe_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3,  r_bmfe_lvl1_part0_1_lvl2_part1_0__D_0__D_1__D_2__D_3,
			       r_bmfe_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  r_bmfe_lvl1_part0_1_lvl2_part1B__D_0__D_1__D_2__D_3, r_bmfe_lvl1_part0_1_lvl2_part1_2__D_0__D_1__D_2__D_3, 1 );
			SlidePartitionDown
			( F_temp2_lvl1_part0_1_lvl2_part1T__D_0__D_1__D_2__D_3,  F_temp2_lvl1_part0_1_lvl2_part1_0__D_0__D_1__D_2__D_3,
			       F_temp2_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  F_temp2_lvl1_part0_1_lvl2_part1B__D_0__D_1__D_2__D_3, F_temp2_lvl1_part0_1_lvl2_part1_2__D_0__D_1__D_2__D_3, 1 );

		}
		//****

		SlidePartitionDown
		( r_bmfe_lvl1_part0T__D_0__D_1__D_2__D_3,  r_bmfe_lvl1_part0_0__D_0__D_1__D_2__D_3,
		       r_bmfe_lvl1_part0_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  r_bmfe_lvl1_part0B__D_0__D_1__D_2__D_3, r_bmfe_lvl1_part0_2__D_0__D_1__D_2__D_3, 0 );
		SlidePartitionDown
		( F_temp2_lvl1_part0T__D_0__D_1__D_2__D_3,  F_temp2_lvl1_part0_0__D_0__D_1__D_2__D_3,
		       F_temp2_lvl1_part0_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  F_temp2_lvl1_part0B__D_0__D_1__D_2__D_3, F_temp2_lvl1_part0_2__D_0__D_1__D_2__D_3, 0 );

	}
	//****
	//**** (out of 1)
	//**** Is real	0 shadows
		//Outputs:
		//  F_ae__D_0_1__D_2_3
	PartitionDown(F_temp2__D_0__D_1__D_2__D_3, F_temp2_lvl1_part3T__D_0__D_1__D_2__D_3, F_temp2_lvl1_part3B__D_0__D_1__D_2__D_3, 3, 0);
	PartitionDown(t_fj__D_0_1__D_2_3, t_fj_lvl1_part0T__D_0_1__D_2_3, t_fj_lvl1_part0B__D_0_1__D_2_3, 0, 0);
	while(F_temp2_lvl1_part3T__D_0__D_1__D_2__D_3.Dimension(3) < F_temp2__D_0__D_1__D_2__D_3.Dimension(3))
	{
		RepartitionDown
		( F_temp2_lvl1_part3T__D_0__D_1__D_2__D_3,  F_temp2_lvl1_part3_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       F_temp2_lvl1_part3_1__D_0__D_1__D_2__D_3,
		  F_temp2_lvl1_part3B__D_0__D_1__D_2__D_3, F_temp2_lvl1_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );
		RepartitionDown
		( t_fj_lvl1_part0T__D_0_1__D_2_3,  t_fj_lvl1_part0_0__D_0_1__D_2_3,
		  /**/ /**/
		       t_fj_lvl1_part0_1__D_0_1__D_2_3,
		  t_fj_lvl1_part0B__D_0_1__D_2_3, t_fj_lvl1_part0_2__D_0_1__D_2_3, 0, blkSize );

		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  F_ae__D_0_1__D_2_3
		PartitionDown(F_temp2_lvl1_part3_1__D_0__D_1__D_2__D_3, F_temp2_lvl1_part3_1_lvl2_part1T__D_0__D_1__D_2__D_3, F_temp2_lvl1_part3_1_lvl2_part1B__D_0__D_1__D_2__D_3, 1, 0);
		PartitionDown(t_fj_lvl1_part0_1__D_0_1__D_2_3, t_fj_lvl1_part0_1_lvl2_part1T__D_0_1__D_2_3, t_fj_lvl1_part0_1_lvl2_part1B__D_0_1__D_2_3, 1, 0);
		while(F_temp2_lvl1_part3_1_lvl2_part1T__D_0__D_1__D_2__D_3.Dimension(1) < F_temp2_lvl1_part3_1__D_0__D_1__D_2__D_3.Dimension(1))
		{
			RepartitionDown
			( F_temp2_lvl1_part3_1_lvl2_part1T__D_0__D_1__D_2__D_3,  F_temp2_lvl1_part3_1_lvl2_part1_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       F_temp2_lvl1_part3_1_lvl2_part1_1__D_0__D_1__D_2__D_3,
			  F_temp2_lvl1_part3_1_lvl2_part1B__D_0__D_1__D_2__D_3, F_temp2_lvl1_part3_1_lvl2_part1_2__D_0__D_1__D_2__D_3, 1, blkSize );
			RepartitionDown
			( t_fj_lvl1_part0_1_lvl2_part1T__D_0_1__D_2_3,  t_fj_lvl1_part0_1_lvl2_part1_0__D_0_1__D_2_3,
			  /**/ /**/
			       t_fj_lvl1_part0_1_lvl2_part1_1__D_0_1__D_2_3,
			  t_fj_lvl1_part0_1_lvl2_part1B__D_0_1__D_2_3, t_fj_lvl1_part0_1_lvl2_part1_2__D_0_1__D_2_3, 1, blkSize );

			   // t_fj_lvl1_part0_1_lvl2_part1_1[*,*] <- t_fj_lvl1_part0_1_lvl2_part1_1[D01,D23]
			t_fj_lvl1_part0_1_lvl2_part1_1_perm10__S__S.AllGatherRedistFrom( t_fj_lvl1_part0_1_lvl2_part1_1__D_0_1__D_2_3, modes_0_1_2_3 );
			   // F_temp2_lvl1_part3_1_lvl2_part1_1[D0,D1,D23,*] <- F_temp2_lvl1_part3_1_lvl2_part1_1[D0,D1,D2,D3]
			F_temp2_lvl1_part3_1_lvl2_part1_1__D_0__D_1__D_2_3__S.AlignModesWith( modes_0_2, F_ae__D_0_1__D_2_3, modes_0_1 );
			F_temp2_lvl1_part3_1_lvl2_part1_1__D_0__D_1__D_2_3__S.AllToAllRedistFrom( F_temp2_lvl1_part3_1_lvl2_part1_1__D_0__D_1__D_2__D_3, modes_3 );
			   // F_temp2_lvl1_part3_1_lvl2_part1_1[D01,*,D23,*] <- F_temp2_lvl1_part3_1_lvl2_part1_1[D0,D1,D23,*]
			F_temp2_lvl1_part3_1_lvl2_part1_1_perm0213__D_0_1__D_2_3__S__S.AlignModesWith( modes_0_2, F_ae__D_0_1__D_2_3, modes_0_1 );
			F_temp2_lvl1_part3_1_lvl2_part1_1_perm0213__D_0_1__D_2_3__S__S.AllToAllRedistFrom( F_temp2_lvl1_part3_1_lvl2_part1_1__D_0__D_1__D_2_3__S, modes_1 );
			F_temp2_lvl1_part3_1_lvl2_part1_1__D_0__D_1__D_2_3__S.EmptyData();
			   // 1.0 * F_temp2_lvl1_part3_1_lvl2_part1_1[D01,*,D23,*]_aemf * t_fj_lvl1_part0_1_lvl2_part1_1[*,*]_mf + 1.0 * F_ae[D01,D23]_ae
PROFILE_SECTION("COMPUTE");
PROFILE_FLOPS(2*prod(F_ae__D_0_1__D_2_3.Shape())*F_temp2_lvl1_part3_1_lvl2_part1_1_perm0213__D_0_1__D_2_3__S__S.Dimension(1)*F_temp2_lvl1_part3_1_lvl2_part1_1_perm0213__D_0_1__D_2_3__S__S.Dimension(3));
			LocalContractAndLocalEliminate(1.0, F_temp2_lvl1_part3_1_lvl2_part1_1_perm0213__D_0_1__D_2_3__S__S.LockedTensor(), indices_aemf, false,
				t_fj_lvl1_part0_1_lvl2_part1_1_perm10__S__S.LockedTensor(), indices_mf, false,
				1.0, F_ae__D_0_1__D_2_3.Tensor(), indices_ae, false);
PROFILE_STOP;
			F_temp2_lvl1_part3_1_lvl2_part1_1_perm0213__D_0_1__D_2_3__S__S.EmptyData();
			t_fj_lvl1_part0_1_lvl2_part1_1_perm10__S__S.EmptyData();

			SlidePartitionDown
			( F_temp2_lvl1_part3_1_lvl2_part1T__D_0__D_1__D_2__D_3,  F_temp2_lvl1_part3_1_lvl2_part1_0__D_0__D_1__D_2__D_3,
			       F_temp2_lvl1_part3_1_lvl2_part1_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  F_temp2_lvl1_part3_1_lvl2_part1B__D_0__D_1__D_2__D_3, F_temp2_lvl1_part3_1_lvl2_part1_2__D_0__D_1__D_2__D_3, 1 );
			SlidePartitionDown
			( t_fj_lvl1_part0_1_lvl2_part1T__D_0_1__D_2_3,  t_fj_lvl1_part0_1_lvl2_part1_0__D_0_1__D_2_3,
			       t_fj_lvl1_part0_1_lvl2_part1_1__D_0_1__D_2_3,
			  /**/ /**/
			  t_fj_lvl1_part0_1_lvl2_part1B__D_0_1__D_2_3, t_fj_lvl1_part0_1_lvl2_part1_2__D_0_1__D_2_3, 1 );

		}
		//****

		SlidePartitionDown
		( F_temp2_lvl1_part3T__D_0__D_1__D_2__D_3,  F_temp2_lvl1_part3_0__D_0__D_1__D_2__D_3,
		       F_temp2_lvl1_part3_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  F_temp2_lvl1_part3B__D_0__D_1__D_2__D_3, F_temp2_lvl1_part3_2__D_0__D_1__D_2__D_3, 3 );
		SlidePartitionDown
		( t_fj_lvl1_part0T__D_0_1__D_2_3,  t_fj_lvl1_part0_0__D_0_1__D_2_3,
		       t_fj_lvl1_part0_1__D_0_1__D_2_3,
		  /**/ /**/
		  t_fj_lvl1_part0B__D_0_1__D_2_3, t_fj_lvl1_part0_2__D_0_1__D_2_3, 0 );

	}
	//****
	F_temp2__D_0__D_1__D_2__D_3.EmptyData();
	//**** (out of 1)
	//**** Is real	0 shadows
		//Outputs:
		//  F_ae__D_0_1__D_2_3
	PartitionDown(t_fj__D_0_1__D_2_3, t_fj_lvl1_part0T__D_0_1__D_2_3, t_fj_lvl1_part0B__D_0_1__D_2_3, 0, 0);
	PartitionDown(F_ae__D_0_1__D_2_3, F_ae_lvl1_part0T__D_0_1__D_2_3, F_ae_lvl1_part0B__D_0_1__D_2_3, 0, 0);
	while(F_ae_lvl1_part0T__D_0_1__D_2_3.Dimension(0) < F_ae__D_0_1__D_2_3.Dimension(0))
	{
		RepartitionDown
		( t_fj_lvl1_part0T__D_0_1__D_2_3,  t_fj_lvl1_part0_0__D_0_1__D_2_3,
		  /**/ /**/
		       t_fj_lvl1_part0_1__D_0_1__D_2_3,
		  t_fj_lvl1_part0B__D_0_1__D_2_3, t_fj_lvl1_part0_2__D_0_1__D_2_3, 0, blkSize );
		RepartitionDown
		( F_ae_lvl1_part0T__D_0_1__D_2_3,  F_ae_lvl1_part0_0__D_0_1__D_2_3,
		  /**/ /**/
		       F_ae_lvl1_part0_1__D_0_1__D_2_3,
		  F_ae_lvl1_part0B__D_0_1__D_2_3, F_ae_lvl1_part0_2__D_0_1__D_2_3, 0, blkSize );

		Permute( F_ae_lvl1_part0_1__D_0_1__D_2_3, F_ae_lvl1_part0_1_perm10__D_2_3__D_0_1 );
		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  F_ae_lvl1_part0_1_perm10__D_2_3__D_0_1
		PartitionDown(H_me__D_0_1__D_2_3, H_me_lvl2_part0T__D_0_1__D_2_3, H_me_lvl2_part0B__D_0_1__D_2_3, 0, 0);
		PartitionDown(t_fj_lvl1_part0_1__D_0_1__D_2_3, t_fj_lvl1_part0_1_lvl2_part1T__D_0_1__D_2_3, t_fj_lvl1_part0_1_lvl2_part1B__D_0_1__D_2_3, 1, 0);
		while(H_me_lvl2_part0T__D_0_1__D_2_3.Dimension(0) < H_me__D_0_1__D_2_3.Dimension(0))
		{
			RepartitionDown
			( H_me_lvl2_part0T__D_0_1__D_2_3,  H_me_lvl2_part0_0__D_0_1__D_2_3,
			  /**/ /**/
			       H_me_lvl2_part0_1__D_0_1__D_2_3,
			  H_me_lvl2_part0B__D_0_1__D_2_3, H_me_lvl2_part0_2__D_0_1__D_2_3, 0, blkSize );
			RepartitionDown
			( t_fj_lvl1_part0_1_lvl2_part1T__D_0_1__D_2_3,  t_fj_lvl1_part0_1_lvl2_part1_0__D_0_1__D_2_3,
			  /**/ /**/
			       t_fj_lvl1_part0_1_lvl2_part1_1__D_0_1__D_2_3,
			  t_fj_lvl1_part0_1_lvl2_part1B__D_0_1__D_2_3, t_fj_lvl1_part0_1_lvl2_part1_2__D_0_1__D_2_3, 1, blkSize );

			   // t_fj_lvl1_part0_1_lvl2_part1_1[D01,*] <- t_fj_lvl1_part0_1_lvl2_part1_1[D01,D23]
			t_fj_lvl1_part0_1_lvl2_part1_1_perm10__S__D_0_1.AlignModesWith( modes_0, F_ae_lvl1_part0_1__D_0_1__D_2_3, modes_0 );
			t_fj_lvl1_part0_1_lvl2_part1_1_perm10__S__D_0_1.AllGatherRedistFrom( t_fj_lvl1_part0_1_lvl2_part1_1__D_0_1__D_2_3, modes_2_3 );
			   // H_me_lvl2_part0_1[*,D23] <- H_me_lvl2_part0_1[D01,D23]
			H_me_lvl2_part0_1_perm10__D_2_3__S.AlignModesWith( modes_1, F_ae_lvl1_part0_1__D_0_1__D_2_3, modes_1 );
			H_me_lvl2_part0_1_perm10__D_2_3__S.AllGatherRedistFrom( H_me_lvl2_part0_1__D_0_1__D_2_3, modes_0_1 );
			   // -1.0 * H_me_lvl2_part0_1[*,D23]_em * t_fj_lvl1_part0_1_lvl2_part1_1[D01,*]_ma + 1.0 * F_ae_lvl1_part0_1[D01,D23]_ea
PROFILE_SECTION("COMPUTE");
PROFILE_FLOPS(2*prod(F_ae_lvl1_part0_1_perm10__D_2_3__D_0_1.Shape())*H_me_lvl2_part0_1_perm10__D_2_3__S.Dimension(0));
			LocalContractAndLocalEliminate(-1.0, H_me_lvl2_part0_1_perm10__D_2_3__S.LockedTensor(), indices_em, false,
				t_fj_lvl1_part0_1_lvl2_part1_1_perm10__S__D_0_1.LockedTensor(), indices_ma, false,
				1.0, F_ae_lvl1_part0_1_perm10__D_2_3__D_0_1.Tensor(), indices_ea, false);
PROFILE_STOP;
			H_me_lvl2_part0_1_perm10__D_2_3__S.EmptyData();
			t_fj_lvl1_part0_1_lvl2_part1_1_perm10__S__D_0_1.EmptyData();

			SlidePartitionDown
			( H_me_lvl2_part0T__D_0_1__D_2_3,  H_me_lvl2_part0_0__D_0_1__D_2_3,
			       H_me_lvl2_part0_1__D_0_1__D_2_3,
			  /**/ /**/
			  H_me_lvl2_part0B__D_0_1__D_2_3, H_me_lvl2_part0_2__D_0_1__D_2_3, 0 );
			SlidePartitionDown
			( t_fj_lvl1_part0_1_lvl2_part1T__D_0_1__D_2_3,  t_fj_lvl1_part0_1_lvl2_part1_0__D_0_1__D_2_3,
			       t_fj_lvl1_part0_1_lvl2_part1_1__D_0_1__D_2_3,
			  /**/ /**/
			  t_fj_lvl1_part0_1_lvl2_part1B__D_0_1__D_2_3, t_fj_lvl1_part0_1_lvl2_part1_2__D_0_1__D_2_3, 1 );

		}
		//****
		Permute( F_ae_lvl1_part0_1_perm10__D_2_3__D_0_1, F_ae_lvl1_part0_1__D_0_1__D_2_3 );
		F_ae_lvl1_part0_1_perm10__D_2_3__D_0_1.EmptyData();

		SlidePartitionDown
		( t_fj_lvl1_part0T__D_0_1__D_2_3,  t_fj_lvl1_part0_0__D_0_1__D_2_3,
		       t_fj_lvl1_part0_1__D_0_1__D_2_3,
		  /**/ /**/
		  t_fj_lvl1_part0B__D_0_1__D_2_3, t_fj_lvl1_part0_2__D_0_1__D_2_3, 0 );
		SlidePartitionDown
		( F_ae_lvl1_part0T__D_0_1__D_2_3,  F_ae_lvl1_part0_0__D_0_1__D_2_3,
		       F_ae_lvl1_part0_1__D_0_1__D_2_3,
		  /**/ /**/
		  F_ae_lvl1_part0B__D_0_1__D_2_3, F_ae_lvl1_part0_2__D_0_1__D_2_3, 0 );

	}
	//****


//****


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
overwrite_tmpShape_G = v_femn__D_0__D_1__D_2__D_3.Shape();
G_temp1__D_0__D_1__D_2__D_3.ResizeTo( overwrite_tmpShape_G );
overwrite_tmpShape_G = u_mnje__D_0__D_1__D_2__D_3.Shape();
G_temp2__D_0__D_1__D_2__D_3.ResizeTo( overwrite_tmpShape_G );
//**** (out of 1)
//**** Is real	0 shadows
	//Outputs:
	//  G_mi__D_0_1__D_2_3

PROFILE_SECTION("COMPUTE");
PROFILE_FLOPS(prod(G_mi__D_0_1__D_2_3.Shape()));
	Scal( 0.0, G_mi__D_0_1__D_2_3 );
PROFILE_STOP;
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
PROFILE_SECTION("COMPUTE");
PROFILE_FLOPS(3*prod(v_femn_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3.Shape()));
			YAxpPx( 2.0, v_femn_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3, -1.0, v_femn_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_3__D_2, perm_0_1_3_2, G_temp1_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3 );
PROFILE_STOP;
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
PROFILE_SECTION("COMPUTE");
PROFILE_FLOPS(2*prod(G_mi_lvl2_part0_1__S__D_2__D_0__D_1__D_3.Shape())*G_temp1_lvl1_part3_1_lvl2_part2_1_perm2013__S__D_0__D_1__D_3.Dimension(1)*G_temp1_lvl1_part3_1_lvl2_part2_1_perm2013__S__D_0__D_1__D_3.Dimension(0)*G_temp1_lvl1_part3_1_lvl2_part2_1_perm2013__S__D_0__D_1__D_3.Dimension(3));
			LocalContract(1.0, G_temp1_lvl1_part3_1_lvl2_part2_1_perm2013__S__D_0__D_1__D_3.LockedTensor(), indices_mefn, false,
				T_bfnj_lvl1_part3_1_perm0132__D_0__D_1__D_3__D_2.LockedTensor(), indices_efni, false,
				0.0, G_mi_lvl2_part0_1__S__D_2__D_0__D_1__D_3.Tensor(), indices_miefn, false);
PROFILE_STOP;
			G_temp1_lvl1_part3_1_lvl2_part2_1_perm2013__S__D_0__D_1__D_3.EmptyData();
			   // G_mi_lvl2_part0_1[D01,D23] <- G_mi_lvl2_part0_1[*,D2,D0,D1,D3] (with SumScatter on (D0)(D1)(D3))
PROFILE_SECTION("COMPUTE");
PROFILE_FLOPS(2*prod(G_mi_lvl2_part0_1__S__D_2__D_0__D_1__D_3.Shape()));
			G_mi_lvl2_part0_1__D_0_1__D_2_3.ReduceScatterUpdateRedistFrom( G_mi_lvl2_part0_1__S__D_2__D_0__D_1__D_3, 1.0, modes_4_3_2 );
PROFILE_STOP;
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
PROFILE_SECTION("COMPUTE");
PROFILE_FLOPS(3*prod(u_mnje_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3.Shape()));
			YAxpPx( 2.0, u_mnje_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3, -1.0, u_mnje_lvl1_part1_1_lvl2_part0_1__D_1__D_0__D_2__D_3, perm_1_0_2_3, G_temp2_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3 );
PROFILE_STOP;
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
PROFILE_SECTION("COMPUTE");
PROFILE_FLOPS(2*prod(G_mi__D_0_1__D_2_3.Shape())*G_temp2_lvl1_part3_1_lvl2_part1_1_perm0213__D_0_1__D_2_3__S__S.Dimension(3)*G_temp2_lvl1_part3_1_lvl2_part1_1_perm0213__D_0_1__D_2_3__S__S.Dimension(1));
			LocalContractAndLocalEliminate(1.0, G_temp2_lvl1_part3_1_lvl2_part1_1_perm0213__D_0_1__D_2_3__S__S.LockedTensor(), indices_mine, false,
				t_fj_lvl1_part0_1_lvl2_part1_1_perm10__S__S.LockedTensor(), indices_ne, false,
				1.0, G_mi__D_0_1__D_2_3.Tensor(), indices_mi, false);
PROFILE_STOP;
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
PROFILE_SECTION("COMPUTE");
PROFILE_FLOPS(2*prod(G_mi_lvl1_part0_1__D_0_1__D_2_3.Shape())*H_me_lvl1_part0_1_lvl2_part1_1__D_0_1__S.Dimension(1));
			LocalContractAndLocalEliminate(1.0, H_me_lvl1_part0_1_lvl2_part1_1__D_0_1__S.LockedTensor(), indices_me, false,
				t_fj_lvl2_part0_1__S__D_2_3.LockedTensor(), indices_ei, false,
				1.0, G_mi_lvl1_part0_1__D_0_1__D_2_3.Tensor(), indices_mi, false);
PROFILE_STOP;
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

PROFILE_SECTION("COMPUTE");
PROFILE_FLOPS(prod(z_ai__D_0_1__D_2_3.Shape()));
	Scal( 0.0, z_ai__D_0_1__D_2_3 );
PROFILE_STOP;
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
PROFILE_SECTION("COMPUTE");
PROFILE_FLOPS(3*prod(r_bmfe_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3.Shape()));
			YAxpPx( 2.0, r_bmfe_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3, -1.0, r_bmfe_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_3__D_2, perm_0_1_3_2, z_small_temp2_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3 );
PROFILE_STOP;
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
PROFILE_SECTION("COMPUTE");
PROFILE_FLOPS(2*prod(z_ai_lvl2_part1_1__D_0__S__D_2__D_3__D_1.Shape())*z_small_temp2_lvl1_part1_1_perm0231__D_0__D_2__D_3__D_1.Dimension(2)*z_small_temp2_lvl1_part1_1_perm0231__D_0__D_2__D_3__D_1.Dimension(1)*z_small_temp2_lvl1_part1_1_perm0231__D_0__D_2__D_3__D_1.Dimension(3));
			LocalContract(1.0, z_small_temp2_lvl1_part1_1_perm0231__D_0__D_2__D_3__D_1.LockedTensor(), indices_aefm, false,
				Tau_efmn_lvl1_part3_1_lvl2_part2_1_perm0132__D_2__D_3__D_1__S.LockedTensor(), indices_efmi, false,
				0.0, z_ai_lvl2_part1_1__D_0__S__D_2__D_3__D_1.Tensor(), indices_aiefm, false);
PROFILE_STOP;
			Tau_efmn_lvl1_part3_1_lvl2_part2_1_perm0132__D_2__D_3__D_1__S.EmptyData();
			   // z_ai_lvl2_part1_1[D01,D23] <- z_ai_lvl2_part1_1[D0,*,D2,D3,D1] (with SumScatter on (D2)(D3)(D1))
PROFILE_SECTION("COMPUTE");
PROFILE_FLOPS(2*prod(z_ai_lvl2_part1_1__D_0__S__D_2__D_3__D_1.Shape()));
			z_ai_lvl2_part1_1__D_0_1__D_2_3.ReduceScatterUpdateRedistFrom( z_ai_lvl2_part1_1__D_0__S__D_2__D_3__D_1, 1.0, modes_4_3_2 );
PROFILE_STOP;
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
PROFILE_SECTION("COMPUTE");
PROFILE_FLOPS(3*prod(w_bmje_lvl1_part1_1_lvl2_part2_1__D_0__D_1__D_2__D_3.Shape()));
			YAxpPx( 2.0, w_bmje_lvl1_part1_1_lvl2_part2_1__D_0__D_1__D_2__D_3, -1.0, x_bmej_lvl1_part1_1_lvl2_part3_1__D_0__D_1__D_3__D_2, perm_0_1_3_2, z_small_temp4_lvl1_part1_1_lvl2_part2_1__D_0__D_1__D_2__D_3 );
PROFILE_STOP;
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
PROFILE_SECTION("COMPUTE");
PROFILE_FLOPS(3*prod(U_mnie_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3.Shape()));
			YAxpPx( 2.0, U_mnie_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3, -1.0, U_mnie_lvl1_part1_1_lvl2_part0_1__D_1__D_0__D_2__D_3, perm_1_0_2_3, z_small_temp5_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3 );
PROFILE_STOP;
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
PROFILE_SECTION("COMPUTE");
PROFILE_FLOPS(3*prod(T_bfnj_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3.Shape()));
			YAxpPx( 2.0, T_bfnj_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3, -1.0, T_bfnj_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_3__D_2, perm_0_1_3_2, z_small_temp3_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3 );
PROFILE_STOP;
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
PROFILE_SECTION("COMPUTE");
PROFILE_FLOPS(2*prod(z_ai_lvl1_part1_1__D_0_1__D_2_3.Shape())*z_small_temp3_lvl1_part2_1_lvl2_part3_1_perm0231__D_0_1__D_2_3__S__S.Dimension(3)*z_small_temp3_lvl1_part2_1_lvl2_part3_1_perm0231__D_0_1__D_2_3__S__S.Dimension(1));
			LocalContractAndLocalEliminate(1.0, z_small_temp3_lvl1_part2_1_lvl2_part3_1_perm0231__D_0_1__D_2_3__S__S.LockedTensor(), indices_aime, false,
				H_me_lvl2_part0_1__S__S.LockedTensor(), indices_me, false,
				1.0, z_ai_lvl1_part1_1__D_0_1__D_2_3.Tensor(), indices_ai, false);
PROFILE_STOP;
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
PROFILE_SECTION("COMPUTE");
PROFILE_FLOPS(2*prod(z_ai_lvl1_part1_1__D_0_1__D_2_3.Shape())*z_small_temp4_lvl1_part2_1_lvl2_part1_1_perm0213__D_0_1__D_2_3__S__S.Dimension(1)*z_small_temp4_lvl1_part2_1_lvl2_part1_1_perm0213__D_0_1__D_2_3__S__S.Dimension(3));
			LocalContractAndLocalEliminate(1.0, z_small_temp4_lvl1_part2_1_lvl2_part1_1_perm0213__D_0_1__D_2_3__S__S.LockedTensor(), indices_aime, false,
				t_fj_lvl2_part1_1_perm10__S__S.LockedTensor(), indices_me, false,
				1.0, z_ai_lvl1_part1_1__D_0_1__D_2_3.Tensor(), indices_ai, false);
PROFILE_STOP;
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
PROFILE_SECTION("COMPUTE");
PROFILE_FLOPS(2*prod(z_ai_lvl1_part1_1_perm10342__S__D_0__D_2__D_3__D_1.Shape())*z_small_temp5_lvl1_part2_1_lvl2_part0_1_perm2013__S__D_2__D_3__D_1.Dimension(0)*z_small_temp5_lvl1_part2_1_lvl2_part0_1_perm2013__S__D_2__D_3__D_1.Dimension(3)*z_small_temp5_lvl1_part2_1_lvl2_part0_1_perm2013__S__D_2__D_3__D_1.Dimension(1));
			LocalContract(-1.0, z_small_temp5_lvl1_part2_1_lvl2_part0_1_perm2013__S__D_2__D_3__D_1.LockedTensor(), indices_imne, false,
				T_bfnj_lvl2_part2_1_perm2310__D_2__D_3__D_1__D_0.LockedTensor(), indices_mnea, false,
				0.0, z_ai_lvl1_part1_1_perm10342__S__D_0__D_2__D_3__D_1.Tensor(), indices_iamne, false);
PROFILE_STOP;
			z_small_temp5_lvl1_part2_1_lvl2_part0_1_perm2013__S__D_2__D_3__D_1.EmptyData();
			T_bfnj_lvl2_part2_1_perm2310__D_2__D_3__D_1__D_0.EmptyData();
			   // z_ai_lvl1_part1_1[D01,D23] <- z_ai_lvl1_part1_1[D0,*,D1,D2,D3] (with SumScatter on (D1)(D2)(D3))
PROFILE_SECTION("COMPUTE");
PROFILE_FLOPS(2*prod(z_ai_lvl1_part1_1_perm10342__S__D_0__D_2__D_3__D_1.Shape()));
			z_ai_lvl1_part1_1__D_0_1__D_2_3.ReduceScatterUpdateRedistFrom( z_ai_lvl1_part1_1_perm10342__S__D_0__D_2__D_3__D_1, 1.0, modes_4_3_2 );
PROFILE_STOP;
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
PROFILE_SECTION("COMPUTE");
PROFILE_FLOPS(2*prod(z_ai_lvl1_part1_1_perm10__D_2_3__D_0_1.Shape())*G_mi_lvl1_part1_1_lvl2_part0_1_perm10__D_2_3__S.Dimension(0));
			LocalContractAndLocalEliminate(-1.0, G_mi_lvl1_part1_1_lvl2_part0_1_perm10__D_2_3__S.LockedTensor(), indices_im, false,
				t_fj_lvl2_part1_1_perm10__S__D_0_1.LockedTensor(), indices_ma, false,
				1.0, z_ai_lvl1_part1_1_perm10__D_2_3__D_0_1.Tensor(), indices_ia, false);
PROFILE_STOP;
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
overwrite_tmpShape_Z = T_bfnj__D_0__D_1__D_2__D_3.Shape();
Z_temp2__D_0__D_1__D_2__D_3.ResizeTo( overwrite_tmpShape_Z );
overwrite_tmpShape_Z = Z_abij__D_0__D_1__D_2__D_3.Shape();
accum__D_0__D_1__D_2__D_3.ResizeTo( overwrite_tmpShape_Z );
overwrite_tmpShape_Z = Z_abij__D_0__D_1__D_2__D_3.Shape();
Z_temp1__D_0__D_1__D_2__D_3.ResizeTo( overwrite_tmpShape_Z );
//**** (out of 1)
//**** Is real	0 shadows
	//Outputs:
	//  Z_abij__D_0__D_1__D_2__D_3

	overwrite_tmpShape_Z = Z_temp1__D_0__D_1__D_2__D_3.Shape();
	Z_temp1_perm1302__D_1__D_3__D_0__D_2.ResizeTo( overwrite_tmpShape_Z );
PROFILE_SECTION("COMPUTE");
PROFILE_FLOPS(prod(Z_temp1_perm1302__D_1__D_3__D_0__D_2.Shape()));
	Scal( 0.0, Z_temp1_perm1302__D_1__D_3__D_0__D_2 );
PROFILE_STOP;
	//**** (out of 1)
	//**** Is real	0 shadows
		//Outputs:
		//  Z_temp1_perm1302__D_1__D_3__D_0__D_2
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
			//  Z_temp1_perm1302__D_1__D_3__D_0__D_2
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

			   // T_bfnj_lvl1_part1_1_lvl2_part2_1[D0,D1,D3,D2] <- T_bfnj_lvl1_part1_1_lvl2_part2_1[D0,D1,D2,D3]
			T_bfnj_lvl1_part1_1_lvl2_part2_1__D_0__D_1__D_3__D_2.AlignModesWith( modes_0_3, Z_temp1__D_0__D_1__D_2__D_3, modes_0_2 );
			T_bfnj_lvl1_part1_1_lvl2_part2_1__D_0__D_1__D_3__D_2.AllToAllRedistFrom( T_bfnj_lvl1_part1_1_lvl2_part2_1__D_0__D_1__D_2__D_3, modes_2_3 );
			   // T_bfnj_lvl1_part1_1_lvl2_part2_1[D0,*,*,D2] <- T_bfnj_lvl1_part1_1_lvl2_part2_1[D0,D1,D3,D2]
			T_bfnj_lvl1_part1_1_lvl2_part2_1_perm2103__S__S__D_0__D_2.AlignModesWith( modes_0_3, Z_temp1__D_0__D_1__D_2__D_3, modes_0_2 );
			T_bfnj_lvl1_part1_1_lvl2_part2_1_perm2103__S__S__D_0__D_2.AllGatherRedistFrom( T_bfnj_lvl1_part1_1_lvl2_part2_1__D_0__D_1__D_3__D_2, modes_1_3 );
			T_bfnj_lvl1_part1_1_lvl2_part2_1__D_0__D_1__D_3__D_2.EmptyData();
			   // X_bmej_lvl1_part2_1_lvl2_part1_1[D1,D0,D2,D3] <- X_bmej_lvl1_part2_1_lvl2_part1_1[D0,D1,D2,D3]
			X_bmej_lvl1_part2_1_lvl2_part1_1__D_1__D_0__D_2__D_3.AlignModesWith( modes_0_3, Z_temp1__D_0__D_1__D_2__D_3, modes_1_3 );
			X_bmej_lvl1_part2_1_lvl2_part1_1__D_1__D_0__D_2__D_3.AllToAllRedistFrom( X_bmej_lvl1_part2_1_lvl2_part1_1__D_0__D_1__D_2__D_3, modes_0_1 );
			   // X_bmej_lvl1_part2_1_lvl2_part1_1[D1,*,*,D3] <- X_bmej_lvl1_part2_1_lvl2_part1_1[D1,D0,D2,D3]
			X_bmej_lvl1_part2_1_lvl2_part1_1_perm0312__D_1__D_3__S__S.AlignModesWith( modes_0_3, Z_temp1__D_0__D_1__D_2__D_3, modes_1_3 );
			X_bmej_lvl1_part2_1_lvl2_part1_1_perm0312__D_1__D_3__S__S.AllGatherRedistFrom( X_bmej_lvl1_part2_1_lvl2_part1_1__D_1__D_0__D_2__D_3, modes_0_2 );
			X_bmej_lvl1_part2_1_lvl2_part1_1__D_1__D_0__D_2__D_3.EmptyData();
			   // 1.0 * X_bmej_lvl1_part2_1_lvl2_part1_1[D1,*,*,D3]_bjme * T_bfnj_lvl1_part1_1_lvl2_part2_1[D0,*,*,D2]_meai + 1.0 * Z_temp1[D0,D1,D2,D3]_bjai
PROFILE_SECTION("COMPUTE");
PROFILE_FLOPS(2*prod(Z_temp1_perm1302__D_1__D_3__D_0__D_2.Shape())*X_bmej_lvl1_part2_1_lvl2_part1_1_perm0312__D_1__D_3__S__S.Dimension(1)*X_bmej_lvl1_part2_1_lvl2_part1_1_perm0312__D_1__D_3__S__S.Dimension(2));
			LocalContractAndLocalEliminate(1.0, X_bmej_lvl1_part2_1_lvl2_part1_1_perm0312__D_1__D_3__S__S.LockedTensor(), indices_bjme, false,
				T_bfnj_lvl1_part1_1_lvl2_part2_1_perm2103__S__S__D_0__D_2.LockedTensor(), indices_meai, false,
				1.0, Z_temp1_perm1302__D_1__D_3__D_0__D_2.Tensor(), indices_bjai, false);
PROFILE_STOP;
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
	Permute( Z_temp1_perm1302__D_1__D_3__D_0__D_2, Z_temp1__D_0__D_1__D_2__D_3 );
	Z_temp1_perm1302__D_1__D_3__D_0__D_2.EmptyData();
	//**** (out of 1)
	//**** Is real	0 shadows
		//Outputs:
		//  accum__D_0__D_1__D_2__D_3
	PartitionDown(Z_temp1__D_0__D_1__D_2__D_3, Z_temp1_lvl1_part2T__D_0__D_1__D_2__D_3, Z_temp1_lvl1_part2B__D_0__D_1__D_2__D_3, 2, 0);
	PartitionDown(Z_temp1__D_0__D_1__D_2__D_3, Z_temp1_lvl1_part3T__D_0__D_1__D_2__D_3, Z_temp1_lvl1_part3B__D_0__D_1__D_2__D_3, 3, 0);
	PartitionDown(accum__D_0__D_1__D_2__D_3, accum_lvl1_part2T__D_0__D_1__D_2__D_3, accum_lvl1_part2B__D_0__D_1__D_2__D_3, 2, 0);
	while(accum_lvl1_part2T__D_0__D_1__D_2__D_3.Dimension(2) < accum__D_0__D_1__D_2__D_3.Dimension(2))
	{
		RepartitionDown
		( Z_temp1_lvl1_part2T__D_0__D_1__D_2__D_3,  Z_temp1_lvl1_part2_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       Z_temp1_lvl1_part2_1__D_0__D_1__D_2__D_3,
		  Z_temp1_lvl1_part2B__D_0__D_1__D_2__D_3, Z_temp1_lvl1_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );
		RepartitionDown
		( Z_temp1_lvl1_part3T__D_0__D_1__D_2__D_3,  Z_temp1_lvl1_part3_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       Z_temp1_lvl1_part3_1__D_0__D_1__D_2__D_3,
		  Z_temp1_lvl1_part3B__D_0__D_1__D_2__D_3, Z_temp1_lvl1_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );
		RepartitionDown
		( accum_lvl1_part2T__D_0__D_1__D_2__D_3,  accum_lvl1_part2_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       accum_lvl1_part2_1__D_0__D_1__D_2__D_3,
		  accum_lvl1_part2B__D_0__D_1__D_2__D_3, accum_lvl1_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );

		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  accum_lvl1_part2_1__D_0__D_1__D_2__D_3
		PartitionDown(Z_temp1_lvl1_part2_1__D_0__D_1__D_2__D_3, Z_temp1_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3, Z_temp1_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, 3, 0);
		PartitionDown(Z_temp1_lvl1_part3_1__D_0__D_1__D_2__D_3, Z_temp1_lvl1_part3_1_lvl2_part2T__D_0__D_1__D_2__D_3, Z_temp1_lvl1_part3_1_lvl2_part2B__D_0__D_1__D_2__D_3, 2, 0);
		PartitionDown(accum_lvl1_part2_1__D_0__D_1__D_2__D_3, accum_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3, accum_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, 3, 0);
		while(accum_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3.Dimension(3) < accum_lvl1_part2_1__D_0__D_1__D_2__D_3.Dimension(3))
		{
			RepartitionDown
			( Z_temp1_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3,  Z_temp1_lvl1_part2_1_lvl2_part3_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       Z_temp1_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3,
			  Z_temp1_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, Z_temp1_lvl1_part2_1_lvl2_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );
			RepartitionDown
			( Z_temp1_lvl1_part3_1_lvl2_part2T__D_0__D_1__D_2__D_3,  Z_temp1_lvl1_part3_1_lvl2_part2_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       Z_temp1_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_2__D_3,
			  Z_temp1_lvl1_part3_1_lvl2_part2B__D_0__D_1__D_2__D_3, Z_temp1_lvl1_part3_1_lvl2_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );
			RepartitionDown
			( accum_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3,  accum_lvl1_part2_1_lvl2_part3_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       accum_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3,
			  accum_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, accum_lvl1_part2_1_lvl2_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );

			   // Z_temp1_lvl1_part3_1_lvl2_part2_1[D0,D1,D3,D2] <- Z_temp1_lvl1_part3_1_lvl2_part2_1[D0,D1,D2,D3]
			Z_temp1_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_3__D_2.AlignModesWith( modes_0_1_2_3, Z_temp1_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3, modes_0_1_3_2 );
			Z_temp1_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_3__D_2.AllToAllRedistFrom( Z_temp1_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_2__D_3, modes_2_3 );
PROFILE_SECTION("COMPUTE");
PROFILE_FLOPS(3*prod(Z_temp1_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3.Shape()));
			YAxpPx( -0.5, Z_temp1_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3, -1.0, Z_temp1_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_3__D_2, perm_0_1_3_2, accum_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3 );
PROFILE_STOP;
			Z_temp1_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_3__D_2.EmptyData();

			SlidePartitionDown
			( Z_temp1_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3,  Z_temp1_lvl1_part2_1_lvl2_part3_0__D_0__D_1__D_2__D_3,
			       Z_temp1_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  Z_temp1_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, Z_temp1_lvl1_part2_1_lvl2_part3_2__D_0__D_1__D_2__D_3, 3 );
			SlidePartitionDown
			( Z_temp1_lvl1_part3_1_lvl2_part2T__D_0__D_1__D_2__D_3,  Z_temp1_lvl1_part3_1_lvl2_part2_0__D_0__D_1__D_2__D_3,
			       Z_temp1_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  Z_temp1_lvl1_part3_1_lvl2_part2B__D_0__D_1__D_2__D_3, Z_temp1_lvl1_part3_1_lvl2_part2_2__D_0__D_1__D_2__D_3, 2 );
			SlidePartitionDown
			( accum_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3,  accum_lvl1_part2_1_lvl2_part3_0__D_0__D_1__D_2__D_3,
			       accum_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  accum_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, accum_lvl1_part2_1_lvl2_part3_2__D_0__D_1__D_2__D_3, 3 );

		}
		//****

		SlidePartitionDown
		( Z_temp1_lvl1_part2T__D_0__D_1__D_2__D_3,  Z_temp1_lvl1_part2_0__D_0__D_1__D_2__D_3,
		       Z_temp1_lvl1_part2_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  Z_temp1_lvl1_part2B__D_0__D_1__D_2__D_3, Z_temp1_lvl1_part2_2__D_0__D_1__D_2__D_3, 2 );
		SlidePartitionDown
		( Z_temp1_lvl1_part3T__D_0__D_1__D_2__D_3,  Z_temp1_lvl1_part3_0__D_0__D_1__D_2__D_3,
		       Z_temp1_lvl1_part3_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  Z_temp1_lvl1_part3B__D_0__D_1__D_2__D_3, Z_temp1_lvl1_part3_2__D_0__D_1__D_2__D_3, 3 );
		SlidePartitionDown
		( accum_lvl1_part2T__D_0__D_1__D_2__D_3,  accum_lvl1_part2_0__D_0__D_1__D_2__D_3,
		       accum_lvl1_part2_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  accum_lvl1_part2B__D_0__D_1__D_2__D_3, accum_lvl1_part2_2__D_0__D_1__D_2__D_3, 2 );

	}
	//****
	Z_temp1__D_0__D_1__D_2__D_3.EmptyData();
	Z_temp1__D_0__D_1__D_2__D_3.EmptyData();
	//**** (out of 1)
	//**** Is real	0 shadows
		//Outputs:
		//  Z_temp2__D_0__D_1__D_2__D_3
	PartitionDown(T_bfnj__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part2T__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part2B__D_0__D_1__D_2__D_3, 2, 0);
	PartitionDown(T_bfnj__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part3T__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part3B__D_0__D_1__D_2__D_3, 3, 0);
	PartitionDown(Z_temp2__D_0__D_1__D_2__D_3, Z_temp2_lvl1_part2T__D_0__D_1__D_2__D_3, Z_temp2_lvl1_part2B__D_0__D_1__D_2__D_3, 2, 0);
	while(Z_temp2_lvl1_part2T__D_0__D_1__D_2__D_3.Dimension(2) < Z_temp2__D_0__D_1__D_2__D_3.Dimension(2))
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
		( Z_temp2_lvl1_part2T__D_0__D_1__D_2__D_3,  Z_temp2_lvl1_part2_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       Z_temp2_lvl1_part2_1__D_0__D_1__D_2__D_3,
		  Z_temp2_lvl1_part2B__D_0__D_1__D_2__D_3, Z_temp2_lvl1_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );

		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  Z_temp2_lvl1_part2_1__D_0__D_1__D_2__D_3
		PartitionDown(T_bfnj_lvl1_part2_1__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, 3, 0);
		PartitionDown(T_bfnj_lvl1_part3_1__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part3_1_lvl2_part2T__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part3_1_lvl2_part2B__D_0__D_1__D_2__D_3, 2, 0);
		PartitionDown(Z_temp2_lvl1_part2_1__D_0__D_1__D_2__D_3, Z_temp2_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3, Z_temp2_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, 3, 0);
		while(Z_temp2_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3.Dimension(3) < Z_temp2_lvl1_part2_1__D_0__D_1__D_2__D_3.Dimension(3))
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
			( Z_temp2_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3,  Z_temp2_lvl1_part2_1_lvl2_part3_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       Z_temp2_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3,
			  Z_temp2_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, Z_temp2_lvl1_part2_1_lvl2_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );

			   // T_bfnj_lvl1_part3_1_lvl2_part2_1[D0,D1,D3,D2] <- T_bfnj_lvl1_part3_1_lvl2_part2_1[D0,D1,D2,D3]
			T_bfnj_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_3__D_2.AlignModesWith( modes_0_1_2_3, T_bfnj_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3, modes_0_1_3_2 );
			T_bfnj_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_3__D_2.AllToAllRedistFrom( T_bfnj_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_2__D_3, modes_2_3 );
PROFILE_SECTION("COMPUTE");
PROFILE_FLOPS(3*prod(T_bfnj_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3.Shape()));
			YAxpPx( 2.0, T_bfnj_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3, -1.0, T_bfnj_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_3__D_2, perm_0_1_3_2, Z_temp2_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3 );
PROFILE_STOP;
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
			( Z_temp2_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3,  Z_temp2_lvl1_part2_1_lvl2_part3_0__D_0__D_1__D_2__D_3,
			       Z_temp2_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  Z_temp2_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, Z_temp2_lvl1_part2_1_lvl2_part3_2__D_0__D_1__D_2__D_3, 3 );

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
		( Z_temp2_lvl1_part2T__D_0__D_1__D_2__D_3,  Z_temp2_lvl1_part2_0__D_0__D_1__D_2__D_3,
		       Z_temp2_lvl1_part2_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  Z_temp2_lvl1_part2B__D_0__D_1__D_2__D_3, Z_temp2_lvl1_part2_2__D_0__D_1__D_2__D_3, 2 );

	}
	//****
	Permute( Z_temp2__D_0__D_1__D_2__D_3, Z_temp2_perm3102__D_3__D_1__D_0__D_2 );
	Z_temp2__D_0__D_1__D_2__D_3.EmptyData();
	//**** (out of 1)
	//**** Is real	0 shadows
		//Outputs:
		//  accum__D_0__D_1__D_2__D_3
	PartitionDown(W_bmje__D_0__D_1__D_2__D_3, W_bmje_lvl1_part0T__D_0__D_1__D_2__D_3, W_bmje_lvl1_part0B__D_0__D_1__D_2__D_3, 0, 0);
	PartitionDown(accum__D_0__D_1__D_2__D_3, accum_lvl1_part1T__D_0__D_1__D_2__D_3, accum_lvl1_part1B__D_0__D_1__D_2__D_3, 1, 0);
	while(accum_lvl1_part1T__D_0__D_1__D_2__D_3.Dimension(1) < accum__D_0__D_1__D_2__D_3.Dimension(1))
	{
		RepartitionDown
		( W_bmje_lvl1_part0T__D_0__D_1__D_2__D_3,  W_bmje_lvl1_part0_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       W_bmje_lvl1_part0_1__D_0__D_1__D_2__D_3,
		  W_bmje_lvl1_part0B__D_0__D_1__D_2__D_3, W_bmje_lvl1_part0_2__D_0__D_1__D_2__D_3, 0, blkSize );
		RepartitionDown
		( accum_lvl1_part1T__D_0__D_1__D_2__D_3,  accum_lvl1_part1_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       accum_lvl1_part1_1__D_0__D_1__D_2__D_3,
		  accum_lvl1_part1B__D_0__D_1__D_2__D_3, accum_lvl1_part1_2__D_0__D_1__D_2__D_3, 1, blkSize );

		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  accum_lvl1_part1_1__D_0__D_1__D_2__D_3
		PartitionDown(W_bmje_lvl1_part0_1__D_0__D_1__D_2__D_3, W_bmje_lvl1_part0_1_lvl2_part2T__D_0__D_1__D_2__D_3, W_bmje_lvl1_part0_1_lvl2_part2B__D_0__D_1__D_2__D_3, 2, 0);
		PartitionDown(accum_lvl1_part1_1__D_0__D_1__D_2__D_3, accum_lvl1_part1_1_lvl2_part3T__D_0__D_1__D_2__D_3, accum_lvl1_part1_1_lvl2_part3B__D_0__D_1__D_2__D_3, 3, 0);
		while(accum_lvl1_part1_1_lvl2_part3T__D_0__D_1__D_2__D_3.Dimension(3) < accum_lvl1_part1_1__D_0__D_1__D_2__D_3.Dimension(3))
		{
			RepartitionDown
			( W_bmje_lvl1_part0_1_lvl2_part2T__D_0__D_1__D_2__D_3,  W_bmje_lvl1_part0_1_lvl2_part2_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       W_bmje_lvl1_part0_1_lvl2_part2_1__D_0__D_1__D_2__D_3,
			  W_bmje_lvl1_part0_1_lvl2_part2B__D_0__D_1__D_2__D_3, W_bmje_lvl1_part0_1_lvl2_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );
			RepartitionDown
			( accum_lvl1_part1_1_lvl2_part3T__D_0__D_1__D_2__D_3,  accum_lvl1_part1_1_lvl2_part3_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       accum_lvl1_part1_1_lvl2_part3_1__D_0__D_1__D_2__D_3,
			  accum_lvl1_part1_1_lvl2_part3B__D_0__D_1__D_2__D_3, accum_lvl1_part1_1_lvl2_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );

			   // W_bmje_lvl1_part0_1_lvl2_part2_1[D0,D3,D2,D1] <- W_bmje_lvl1_part0_1_lvl2_part2_1[D0,D1,D2,D3]
			W_bmje_lvl1_part0_1_lvl2_part2_1__D_0__D_3__D_2__D_1.AlignModesWith( modes_1_3, Z_temp2__D_0__D_1__D_2__D_3, modes_3_1 );
			W_bmje_lvl1_part0_1_lvl2_part2_1__D_0__D_3__D_2__D_1.AllToAllRedistFrom( W_bmje_lvl1_part0_1_lvl2_part2_1__D_0__D_1__D_2__D_3, modes_1_3 );
			   // W_bmje_lvl1_part0_1_lvl2_part2_1[*,D3,*,D1] <- W_bmje_lvl1_part0_1_lvl2_part2_1[D0,D3,D2,D1]
			W_bmje_lvl1_part0_1_lvl2_part2_1_perm0213__S__S__D_3__D_1.AlignModesWith( modes_1_3, Z_temp2__D_0__D_1__D_2__D_3, modes_3_1 );
			W_bmje_lvl1_part0_1_lvl2_part2_1_perm0213__S__S__D_3__D_1.AllGatherRedistFrom( W_bmje_lvl1_part0_1_lvl2_part2_1__D_0__D_3__D_2__D_1, modes_0_2 );
			W_bmje_lvl1_part0_1_lvl2_part2_1__D_0__D_3__D_2__D_1.EmptyData();
			accum_lvl1_part1_1_lvl2_part3_1_perm130254__S__S__D_0__D_2__D_3__D_1.AlignModesWith( modes_0_2, Z_temp2__D_0__D_1__D_2__D_3, modes_0_2 );
			overwrite_tmpShape_Z = accum_lvl1_part1_1_lvl2_part3_1__D_0__D_1__D_2__D_3.Shape();
			overwrite_tmpShape_Z.push_back( g.Shape()[1] );
			overwrite_tmpShape_Z.push_back( g.Shape()[3] );
			accum_lvl1_part1_1_lvl2_part3_1_perm130254__S__S__D_0__D_2__D_3__D_1.ResizeTo( overwrite_tmpShape_Z );
			   // 0.5 * W_bmje_lvl1_part0_1_lvl2_part2_1[*,D3,*,D1]_bjme * Z_temp2[D0,D1,D2,D3]_meai + 0.0 * accum_lvl1_part1_1_lvl2_part3_1[D0,*,D2,*,D1,D3]_bjaime
PROFILE_SECTION("COMPUTE");
PROFILE_FLOPS(2*prod(accum_lvl1_part1_1_lvl2_part3_1_perm130254__S__S__D_0__D_2__D_3__D_1.Shape())*W_bmje_lvl1_part0_1_lvl2_part2_1_perm0213__S__S__D_3__D_1.Dimension(1)*W_bmje_lvl1_part0_1_lvl2_part2_1_perm0213__S__S__D_3__D_1.Dimension(3));
			LocalContract(0.5, W_bmje_lvl1_part0_1_lvl2_part2_1_perm0213__S__S__D_3__D_1.LockedTensor(), indices_bjme, false,
				Z_temp2_perm3102__D_3__D_1__D_0__D_2.LockedTensor(), indices_meai, false,
				0.0, accum_lvl1_part1_1_lvl2_part3_1_perm130254__S__S__D_0__D_2__D_3__D_1.Tensor(), indices_bjaime, false);
PROFILE_STOP;
			W_bmje_lvl1_part0_1_lvl2_part2_1_perm0213__S__S__D_3__D_1.EmptyData();
			   // accum_lvl1_part1_1_lvl2_part3_1[D0,D1,D2,D3] <- accum_lvl1_part1_1_lvl2_part3_1[D0,*,D2,*,D1,D3] (with SumScatter on (D1)(D3))
PROFILE_SECTION("COMPUTE");
PROFILE_FLOPS(2*prod(accum_lvl1_part1_1_lvl2_part3_1_perm130254__S__S__D_0__D_2__D_3__D_1.Shape()));
			accum_lvl1_part1_1_lvl2_part3_1__D_0__D_1__D_2__D_3.ReduceScatterUpdateRedistFrom( accum_lvl1_part1_1_lvl2_part3_1_perm130254__S__S__D_0__D_2__D_3__D_1, 1.0, modes_5_4 );
PROFILE_STOP;
			accum_lvl1_part1_1_lvl2_part3_1_perm130254__S__S__D_0__D_2__D_3__D_1.EmptyData();

			SlidePartitionDown
			( W_bmje_lvl1_part0_1_lvl2_part2T__D_0__D_1__D_2__D_3,  W_bmje_lvl1_part0_1_lvl2_part2_0__D_0__D_1__D_2__D_3,
			       W_bmje_lvl1_part0_1_lvl2_part2_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  W_bmje_lvl1_part0_1_lvl2_part2B__D_0__D_1__D_2__D_3, W_bmje_lvl1_part0_1_lvl2_part2_2__D_0__D_1__D_2__D_3, 2 );
			SlidePartitionDown
			( accum_lvl1_part1_1_lvl2_part3T__D_0__D_1__D_2__D_3,  accum_lvl1_part1_1_lvl2_part3_0__D_0__D_1__D_2__D_3,
			       accum_lvl1_part1_1_lvl2_part3_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  accum_lvl1_part1_1_lvl2_part3B__D_0__D_1__D_2__D_3, accum_lvl1_part1_1_lvl2_part3_2__D_0__D_1__D_2__D_3, 3 );

		}
		//****

		SlidePartitionDown
		( W_bmje_lvl1_part0T__D_0__D_1__D_2__D_3,  W_bmje_lvl1_part0_0__D_0__D_1__D_2__D_3,
		       W_bmje_lvl1_part0_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  W_bmje_lvl1_part0B__D_0__D_1__D_2__D_3, W_bmje_lvl1_part0_2__D_0__D_1__D_2__D_3, 0 );
		SlidePartitionDown
		( accum_lvl1_part1T__D_0__D_1__D_2__D_3,  accum_lvl1_part1_0__D_0__D_1__D_2__D_3,
		       accum_lvl1_part1_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  accum_lvl1_part1B__D_0__D_1__D_2__D_3, accum_lvl1_part1_2__D_0__D_1__D_2__D_3, 1 );

	}
	//****
	Z_temp2_perm3102__D_3__D_1__D_0__D_2.EmptyData();
	//**** (out of 1)
	//**** Is real	0 shadows
		//Outputs:
		//  accum__D_0__D_1__D_2__D_3
	PartitionDown(T_bfnj__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part3T__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part3B__D_0__D_1__D_2__D_3, 3, 0);
	PartitionDown(accum__D_0__D_1__D_2__D_3, accum_lvl1_part3T__D_0__D_1__D_2__D_3, accum_lvl1_part3B__D_0__D_1__D_2__D_3, 3, 0);
	while(accum_lvl1_part3T__D_0__D_1__D_2__D_3.Dimension(3) < accum__D_0__D_1__D_2__D_3.Dimension(3))
	{
		RepartitionDown
		( T_bfnj_lvl1_part3T__D_0__D_1__D_2__D_3,  T_bfnj_lvl1_part3_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       T_bfnj_lvl1_part3_1__D_0__D_1__D_2__D_3,
		  T_bfnj_lvl1_part3B__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );
		RepartitionDown
		( accum_lvl1_part3T__D_0__D_1__D_2__D_3,  accum_lvl1_part3_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       accum_lvl1_part3_1__D_0__D_1__D_2__D_3,
		  accum_lvl1_part3B__D_0__D_1__D_2__D_3, accum_lvl1_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );

		Permute( accum_lvl1_part3_1__D_0__D_1__D_2__D_3, accum_lvl1_part3_1_perm2013__D_2__D_0__D_1__D_3 );
		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  accum_lvl1_part3_1_perm2013__D_2__D_0__D_1__D_3
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

			   // T_bfnj_lvl1_part3_1_lvl2_part2_1[D0,D1,*,D3] <- T_bfnj_lvl1_part3_1_lvl2_part2_1[D0,D1,D2,D3]
			T_bfnj_lvl1_part3_1_lvl2_part2_1_perm2013__S__D_0__D_1__D_3.AlignModesWith( modes_0_1_3, accum_lvl1_part3_1__D_0__D_1__D_2__D_3, modes_0_1_3 );
			T_bfnj_lvl1_part3_1_lvl2_part2_1_perm2013__S__D_0__D_1__D_3.AllGatherRedistFrom( T_bfnj_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_2__D_3, modes_2 );
			   // G_mi_lvl2_part0_1[*,D2] <- G_mi_lvl2_part0_1[D01,D23]
			G_mi_lvl2_part0_1_perm10__D_2__S.AlignModesWith( modes_1, accum_lvl1_part3_1__D_0__D_1__D_2__D_3, modes_2 );
			G_mi_lvl2_part0_1_perm10__D_2__S.AllGatherRedistFrom( G_mi_lvl2_part0_1__D_0_1__D_2_3, modes_0_1_3 );
			   // -1.0 * G_mi_lvl2_part0_1[*,D2]_im * T_bfnj_lvl1_part3_1_lvl2_part2_1[D0,D1,*,D3]_mabj + 1.0 * accum_lvl1_part3_1[D0,D1,D2,D3]_iabj
PROFILE_SECTION("COMPUTE");
PROFILE_FLOPS(2*prod(accum_lvl1_part3_1_perm2013__D_2__D_0__D_1__D_3.Shape())*G_mi_lvl2_part0_1_perm10__D_2__S.Dimension(0));
			LocalContractAndLocalEliminate(-1.0, G_mi_lvl2_part0_1_perm10__D_2__S.LockedTensor(), indices_im, false,
				T_bfnj_lvl1_part3_1_lvl2_part2_1_perm2013__S__D_0__D_1__D_3.LockedTensor(), indices_mabj, false,
				1.0, accum_lvl1_part3_1_perm2013__D_2__D_0__D_1__D_3.Tensor(), indices_iabj, false);
PROFILE_STOP;
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
		Permute( accum_lvl1_part3_1_perm2013__D_2__D_0__D_1__D_3, accum_lvl1_part3_1__D_0__D_1__D_2__D_3 );
		accum_lvl1_part3_1_perm2013__D_2__D_0__D_1__D_3.EmptyData();

		SlidePartitionDown
		( T_bfnj_lvl1_part3T__D_0__D_1__D_2__D_3,  T_bfnj_lvl1_part3_0__D_0__D_1__D_2__D_3,
		       T_bfnj_lvl1_part3_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  T_bfnj_lvl1_part3B__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part3_2__D_0__D_1__D_2__D_3, 3 );
		SlidePartitionDown
		( accum_lvl1_part3T__D_0__D_1__D_2__D_3,  accum_lvl1_part3_0__D_0__D_1__D_2__D_3,
		       accum_lvl1_part3_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  accum_lvl1_part3B__D_0__D_1__D_2__D_3, accum_lvl1_part3_2__D_0__D_1__D_2__D_3, 3 );

	}
	//****
	//**** (out of 1)
	//**** Is real	0 shadows
		//Outputs:
		//  accum__D_0__D_1__D_2__D_3
	PartitionDown(t_fj__D_0_1__D_2_3, t_fj_lvl1_part1T__D_0_1__D_2_3, t_fj_lvl1_part1B__D_0_1__D_2_3, 1, 0);
	PartitionDown(P_jimb__D_0__D_1__D_2__D_3, P_jimb_lvl1_part0T__D_0__D_1__D_2__D_3, P_jimb_lvl1_part0B__D_0__D_1__D_2__D_3, 0, 0);
	PartitionDown(T_bfnj__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part2T__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part2B__D_0__D_1__D_2__D_3, 2, 0);
	PartitionDown(accum__D_0__D_1__D_2__D_3, accum_lvl1_part2T__D_0__D_1__D_2__D_3, accum_lvl1_part2B__D_0__D_1__D_2__D_3, 2, 0);
	while(accum_lvl1_part2T__D_0__D_1__D_2__D_3.Dimension(2) < accum__D_0__D_1__D_2__D_3.Dimension(2))
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
		RepartitionDown
		( T_bfnj_lvl1_part2T__D_0__D_1__D_2__D_3,  T_bfnj_lvl1_part2_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       T_bfnj_lvl1_part2_1__D_0__D_1__D_2__D_3,
		  T_bfnj_lvl1_part2B__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );
		RepartitionDown
		( accum_lvl1_part2T__D_0__D_1__D_2__D_3,  accum_lvl1_part2_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       accum_lvl1_part2_1__D_0__D_1__D_2__D_3,
		  accum_lvl1_part2B__D_0__D_1__D_2__D_3, accum_lvl1_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );

		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  accum_lvl1_part2_1__D_0__D_1__D_2__D_3
		PartitionDown(T_bfnj_lvl1_part2_1__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, 3, 0);
		PartitionDown(accum_lvl1_part2_1__D_0__D_1__D_2__D_3, accum_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3, accum_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, 3, 0);
		while(accum_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3.Dimension(3) < accum_lvl1_part2_1__D_0__D_1__D_2__D_3.Dimension(3))
		{
			RepartitionDown
			( T_bfnj_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3,  T_bfnj_lvl1_part2_1_lvl2_part3_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       T_bfnj_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3,
			  T_bfnj_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part2_1_lvl2_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );
			RepartitionDown
			( accum_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3,  accum_lvl1_part2_1_lvl2_part3_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       accum_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3,
			  accum_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, accum_lvl1_part2_1_lvl2_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );

			   // T_bfnj_lvl1_part2_1_lvl2_part3_1[*,D1,D2,D3] <- T_bfnj_lvl1_part2_1_lvl2_part3_1[D0,D1,D2,D3]
			T_bfnj_lvl1_part2_1_lvl2_part3_1__S__D_1__D_2__D_3.AlignModesWith( modes_1_2_3, accum_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3, modes_1_2_3 );
			T_bfnj_lvl1_part2_1_lvl2_part3_1__S__D_1__D_2__D_3.AllGatherRedistFrom( T_bfnj_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3, modes_0 );
			   // F_ae[D0,*] <- F_ae[D01,D23]
			F_ae__D_0__S.AlignModesWith( modes_0, accum_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3, modes_0 );
			F_ae__D_0__S.AllGatherRedistFrom( F_ae__D_0_1__D_2_3, modes_1_2_3 );
			   // 1.0 * F_ae[D0,*]_ae * T_bfnj_lvl1_part2_1_lvl2_part3_1[*,D1,D2,D3]_ebij + 1.0 * accum_lvl1_part2_1_lvl2_part3_1[D0,D1,D2,D3]_abij
PROFILE_SECTION("COMPUTE");
PROFILE_FLOPS(2*prod(accum_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3.Shape())*F_ae__D_0__S.Dimension(1));
			LocalContractAndLocalEliminate(1.0, F_ae__D_0__S.LockedTensor(), indices_ae, false,
				T_bfnj_lvl1_part2_1_lvl2_part3_1__S__D_1__D_2__D_3.LockedTensor(), indices_ebij, false,
				1.0, accum_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3.Tensor(), indices_abij, false);
PROFILE_STOP;
			F_ae__D_0__S.EmptyData();
			T_bfnj_lvl1_part2_1_lvl2_part3_1__S__D_1__D_2__D_3.EmptyData();

			SlidePartitionDown
			( T_bfnj_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3,  T_bfnj_lvl1_part2_1_lvl2_part3_0__D_0__D_1__D_2__D_3,
			       T_bfnj_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  T_bfnj_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part2_1_lvl2_part3_2__D_0__D_1__D_2__D_3, 3 );
			SlidePartitionDown
			( accum_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3,  accum_lvl1_part2_1_lvl2_part3_0__D_0__D_1__D_2__D_3,
			       accum_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  accum_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, accum_lvl1_part2_1_lvl2_part3_2__D_0__D_1__D_2__D_3, 3 );

		}
		//****
		Permute( accum_lvl1_part2_1__D_0__D_1__D_2__D_3, accum_lvl1_part2_1_perm2310__D_2__D_3__D_1__D_0 );
		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  accum_lvl1_part2_1_perm2310__D_2__D_3__D_1__D_0
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
			t_fj_lvl2_part1_1_perm10__S__D_0.AlignModesWith( modes_0, accum_lvl1_part2_1__D_0__D_1__D_2__D_3, modes_0 );
			t_fj_lvl2_part1_1_perm10__S__D_0.AllGatherRedistFrom( t_fj_lvl2_part1_1__D_0_1__D_2_3, modes_1_2_3 );
			   // P_jimb_lvl1_part0_1_lvl2_part2_1[D0,D3,D2,D1] <- P_jimb_lvl1_part0_1_lvl2_part2_1[D0,D1,D2,D3]
			P_jimb_lvl1_part0_1_lvl2_part2_1__D_0__D_3__D_2__D_1.AlignModesWith( modes_0_1_3, accum_lvl1_part2_1__D_0__D_1__D_2__D_3, modes_2_3_1 );
			P_jimb_lvl1_part0_1_lvl2_part2_1__D_0__D_3__D_2__D_1.AllToAllRedistFrom( P_jimb_lvl1_part0_1_lvl2_part2_1__D_0__D_1__D_2__D_3, modes_1_3 );
			   // P_jimb_lvl1_part0_1_lvl2_part2_1[D2,D3,*,D1] <- P_jimb_lvl1_part0_1_lvl2_part2_1[D0,D3,D2,D1]
			P_jimb_lvl1_part0_1_lvl2_part2_1_perm0132__D_2__D_3__D_1__S.AlignModesWith( modes_0_1_3, accum_lvl1_part2_1__D_0__D_1__D_2__D_3, modes_2_3_1 );
			P_jimb_lvl1_part0_1_lvl2_part2_1_perm0132__D_2__D_3__D_1__S.AllToAllRedistFrom( P_jimb_lvl1_part0_1_lvl2_part2_1__D_0__D_3__D_2__D_1, modes_0_2 );
			P_jimb_lvl1_part0_1_lvl2_part2_1__D_0__D_3__D_2__D_1.EmptyData();
			   // -1.0 * P_jimb_lvl1_part0_1_lvl2_part2_1[D2,D3,*,D1]_ijbm * t_fj_lvl2_part1_1[D0,*]_ma + 1.0 * accum_lvl1_part2_1[D0,D1,D2,D3]_ijba
PROFILE_SECTION("COMPUTE");
PROFILE_FLOPS(2*prod(accum_lvl1_part2_1_perm2310__D_2__D_3__D_1__D_0.Shape())*P_jimb_lvl1_part0_1_lvl2_part2_1_perm0132__D_2__D_3__D_1__S.Dimension(2));
			LocalContractAndLocalEliminate(-1.0, P_jimb_lvl1_part0_1_lvl2_part2_1_perm0132__D_2__D_3__D_1__S.LockedTensor(), indices_ijbm, false,
				t_fj_lvl2_part1_1_perm10__S__D_0.LockedTensor(), indices_ma, false,
				1.0, accum_lvl1_part2_1_perm2310__D_2__D_3__D_1__D_0.Tensor(), indices_ijba, false);
PROFILE_STOP;
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
		Permute( accum_lvl1_part2_1_perm2310__D_2__D_3__D_1__D_0, accum_lvl1_part2_1__D_0__D_1__D_2__D_3 );
		accum_lvl1_part2_1_perm2310__D_2__D_3__D_1__D_0.EmptyData();
		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  accum_lvl1_part2_1__D_0__D_1__D_2__D_3
		PartitionDown(r_bmfe__D_0__D_1__D_2__D_3, r_bmfe_lvl2_part1T__D_0__D_1__D_2__D_3, r_bmfe_lvl2_part1B__D_0__D_1__D_2__D_3, 1, 0);
		PartitionDown(accum_lvl1_part2_1__D_0__D_1__D_2__D_3, accum_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3, accum_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, 3, 0);
		while(accum_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3.Dimension(3) < accum_lvl1_part2_1__D_0__D_1__D_2__D_3.Dimension(3))
		{
			RepartitionDown
			( r_bmfe_lvl2_part1T__D_0__D_1__D_2__D_3,  r_bmfe_lvl2_part1_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       r_bmfe_lvl2_part1_1__D_0__D_1__D_2__D_3,
			  r_bmfe_lvl2_part1B__D_0__D_1__D_2__D_3, r_bmfe_lvl2_part1_2__D_0__D_1__D_2__D_3, 1, blkSize );
			RepartitionDown
			( accum_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3,  accum_lvl1_part2_1_lvl2_part3_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       accum_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3,
			  accum_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, accum_lvl1_part2_1_lvl2_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );

			accum_lvl1_part2_1_lvl2_part3_1_Z_temp__D_2_0__D_3__S__D_1.AlignModesWith( modes_0_1_2_3, accum_lvl1_part2_1_lvl2_part3_1_Z_temp__D_0__D_1__D_2__D_3, modes_0_1_2_3 );
			overwrite_tmpShape_Z = accum_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3.Shape();
			accum_lvl1_part2_1_lvl2_part3_1_Z_temp__D_2_0__D_3__S__D_1.ResizeTo( overwrite_tmpShape_Z );
			   // t_fj_lvl1_part1_1[D0,*] <- t_fj_lvl1_part1_1[D01,D23]
			t_fj_lvl1_part1_1__D_0__S.AlignModesWith( modes_0, r_bmfe_lvl2_part1_1__D_0__D_1__D_2__D_3, modes_0 );
			t_fj_lvl1_part1_1__D_0__S.AllGatherRedistFrom( t_fj_lvl1_part1_1__D_0_1__D_2_3, modes_1_2_3 );
			Permute( r_bmfe_lvl2_part1_1__D_0__D_1__D_2__D_3, r_bmfe_lvl2_part1_1_perm1230__D_1__D_2__D_3__D_0 );
			accum_lvl1_part2_1_lvl2_part3_1_perm30124__D_1__D_2__D_3__S__D_0.AlignModesWith( modes_0_1_3, r_bmfe_lvl2_part1_1__D_0__D_1__D_2__D_3, modes_2_3_1 );
			overwrite_tmpShape_Z = accum_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3.Shape();
			overwrite_tmpShape_Z.push_back( g.Shape()[0] );
			accum_lvl1_part2_1_lvl2_part3_1_perm30124__D_1__D_2__D_3__S__D_0.ResizeTo( overwrite_tmpShape_Z );
			   // 1.0 * r_bmfe_lvl2_part1_1[D0,D1,D2,D3]_jabe * t_fj_lvl1_part1_1[D0,*]_ei + 0.0 * accum_lvl1_part2_1_lvl2_part3_1[D2,D3,*,D1,D0]_jabie
PROFILE_SECTION("COMPUTE");
PROFILE_FLOPS(2*prod(accum_lvl1_part2_1_lvl2_part3_1_perm30124__D_1__D_2__D_3__S__D_0.Shape())*r_bmfe_lvl2_part1_1_perm1230__D_1__D_2__D_3__D_0.Dimension(0));
			LocalContract(1.0, r_bmfe_lvl2_part1_1_perm1230__D_1__D_2__D_3__D_0.LockedTensor(), indices_jabe, false,
				t_fj_lvl1_part1_1__D_0__S.LockedTensor(), indices_ei, false,
				0.0, accum_lvl1_part2_1_lvl2_part3_1_perm30124__D_1__D_2__D_3__S__D_0.Tensor(), indices_jabie, false);
PROFILE_STOP;
			r_bmfe_lvl2_part1_1_perm1230__D_1__D_2__D_3__D_0.EmptyData();
			t_fj_lvl1_part1_1__D_0__S.EmptyData();
			   // accum_lvl1_part2_1_lvl2_part3_1_Z_temp[D20,D3,*,D1] <- accum_lvl1_part2_1_lvl2_part3_1[D2,D3,*,D1,D0] (with SumScatter on D0)
PROFILE_SECTION("COMPUTE");
PROFILE_FLOPS(prod(accum_lvl1_part2_1_lvl2_part3_1_perm30124__D_1__D_2__D_3__S__D_0.Shape()));
			accum_lvl1_part2_1_lvl2_part3_1_Z_temp__D_2_0__D_3__S__D_1.ReduceScatterRedistFrom( accum_lvl1_part2_1_lvl2_part3_1_perm30124__D_1__D_2__D_3__S__D_0, 4 );
PROFILE_STOP;
			accum_lvl1_part2_1_lvl2_part3_1_perm30124__D_1__D_2__D_3__S__D_0.EmptyData();
			   // accum_lvl1_part2_1_lvl2_part3_1_Z_temp[D20,D1,*,D3] <- accum_lvl1_part2_1_lvl2_part3_1_Z_temp[D20,D3,*,D1]
			accum_lvl1_part2_1_lvl2_part3_1_Z_temp__D_2_0__D_1__S__D_3.AlignModesWith( modes_0_1_2_3, accum_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3, modes_0_1_2_3 );
			accum_lvl1_part2_1_lvl2_part3_1_Z_temp__D_2_0__D_1__S__D_3.AllToAllRedistFrom( accum_lvl1_part2_1_lvl2_part3_1_Z_temp__D_2_0__D_3__S__D_1, modes_1_3 );
			accum_lvl1_part2_1_lvl2_part3_1_Z_temp__D_2_0__D_3__S__D_1.EmptyData();
			   // accum_lvl1_part2_1_lvl2_part3_1_Z_temp[D0,D1,D2,D3] <- accum_lvl1_part2_1_lvl2_part3_1_Z_temp[D20,D1,*,D3]
			accum_lvl1_part2_1_lvl2_part3_1_Z_temp__D_0__D_1__D_2__D_3.AlignModesWith( modes_0_1_2_3, accum_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3, modes_0_1_2_3 );
			accum_lvl1_part2_1_lvl2_part3_1_Z_temp__D_0__D_1__D_2__D_3.AllToAllRedistFrom( accum_lvl1_part2_1_lvl2_part3_1_Z_temp__D_2_0__D_1__S__D_3, modes_0_2 );
			accum_lvl1_part2_1_lvl2_part3_1_Z_temp__D_2_0__D_1__S__D_3.EmptyData();
PROFILE_SECTION("COMPUTE");
PROFILE_FLOPS(2*prod(accum_lvl1_part2_1_lvl2_part3_1_Z_temp__D_0__D_1__D_2__D_3.Shape()));
			YxpBy( accum_lvl1_part2_1_lvl2_part3_1_Z_temp__D_0__D_1__D_2__D_3, 1.0, accum_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3 );
PROFILE_STOP;
			accum_lvl1_part2_1_lvl2_part3_1_Z_temp__D_0__D_1__D_2__D_3.EmptyData();

			SlidePartitionDown
			( r_bmfe_lvl2_part1T__D_0__D_1__D_2__D_3,  r_bmfe_lvl2_part1_0__D_0__D_1__D_2__D_3,
			       r_bmfe_lvl2_part1_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  r_bmfe_lvl2_part1B__D_0__D_1__D_2__D_3, r_bmfe_lvl2_part1_2__D_0__D_1__D_2__D_3, 1 );
			SlidePartitionDown
			( accum_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3,  accum_lvl1_part2_1_lvl2_part3_0__D_0__D_1__D_2__D_3,
			       accum_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  accum_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, accum_lvl1_part2_1_lvl2_part3_2__D_0__D_1__D_2__D_3, 3 );

		}
		//****

		SlidePartitionDown
		( t_fj_lvl1_part1T__D_0_1__D_2_3,  t_fj_lvl1_part1_0__D_0_1__D_2_3,
		       t_fj_lvl1_part1_1__D_0_1__D_2_3,
		  /**/ /**/
		  t_fj_lvl1_part1B__D_0_1__D_2_3, t_fj_lvl1_part1_2__D_0_1__D_2_3, 1 );
		SlidePartitionDown
		( accum_lvl1_part2T__D_0__D_1__D_2__D_3,  accum_lvl1_part2_0__D_0__D_1__D_2__D_3,
		       accum_lvl1_part2_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  accum_lvl1_part2B__D_0__D_1__D_2__D_3, accum_lvl1_part2_2__D_0__D_1__D_2__D_3, 2 );
		SlidePartitionDown
		( P_jimb_lvl1_part0T__D_0__D_1__D_2__D_3,  P_jimb_lvl1_part0_0__D_0__D_1__D_2__D_3,
		       P_jimb_lvl1_part0_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  P_jimb_lvl1_part0B__D_0__D_1__D_2__D_3, P_jimb_lvl1_part0_2__D_0__D_1__D_2__D_3, 0 );
		SlidePartitionDown
		( T_bfnj_lvl1_part2T__D_0__D_1__D_2__D_3,  T_bfnj_lvl1_part2_0__D_0__D_1__D_2__D_3,
		       T_bfnj_lvl1_part2_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  T_bfnj_lvl1_part2B__D_0__D_1__D_2__D_3, T_bfnj_lvl1_part2_2__D_0__D_1__D_2__D_3, 2 );

	}
	//****
	//**** (out of 1)
	//**** Is real	0 shadows
		//Outputs:
		//  Z_abij__D_0__D_1__D_2__D_3
	PartitionDown(Tau_efmn__D_0__D_1__D_2__D_3, Tau_efmn_lvl1_part2T__D_0__D_1__D_2__D_3, Tau_efmn_lvl1_part2B__D_0__D_1__D_2__D_3, 2, 0);
	PartitionDown(accum__D_0__D_1__D_2__D_3, accum_lvl1_part2T__D_0__D_1__D_2__D_3, accum_lvl1_part2B__D_0__D_1__D_2__D_3, 2, 0);
	PartitionDown(accum__D_0__D_1__D_2__D_3, accum_lvl1_part3T__D_0__D_1__D_2__D_3, accum_lvl1_part3B__D_0__D_1__D_2__D_3, 3, 0);
	PartitionDown(Z_abij__D_0__D_1__D_2__D_3, Z_abij_lvl1_part2T__D_0__D_1__D_2__D_3, Z_abij_lvl1_part2B__D_0__D_1__D_2__D_3, 2, 0);
	while(Z_abij_lvl1_part2T__D_0__D_1__D_2__D_3.Dimension(2) < Z_abij__D_0__D_1__D_2__D_3.Dimension(2))
	{
		RepartitionDown
		( Tau_efmn_lvl1_part2T__D_0__D_1__D_2__D_3,  Tau_efmn_lvl1_part2_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       Tau_efmn_lvl1_part2_1__D_0__D_1__D_2__D_3,
		  Tau_efmn_lvl1_part2B__D_0__D_1__D_2__D_3, Tau_efmn_lvl1_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );
		RepartitionDown
		( accum_lvl1_part2T__D_0__D_1__D_2__D_3,  accum_lvl1_part2_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       accum_lvl1_part2_1__D_0__D_1__D_2__D_3,
		  accum_lvl1_part2B__D_0__D_1__D_2__D_3, accum_lvl1_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );
		RepartitionDown
		( accum_lvl1_part3T__D_0__D_1__D_2__D_3,  accum_lvl1_part3_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       accum_lvl1_part3_1__D_0__D_1__D_2__D_3,
		  accum_lvl1_part3B__D_0__D_1__D_2__D_3, accum_lvl1_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );
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
		PartitionDown(accum_lvl1_part2_1__D_0__D_1__D_2__D_3, accum_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3, accum_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, 3, 0);
		PartitionDown(accum_lvl1_part3_1__D_0__D_1__D_2__D_3, accum_lvl1_part3_1_lvl2_part2T__D_0__D_1__D_2__D_3, accum_lvl1_part3_1_lvl2_part2B__D_0__D_1__D_2__D_3, 2, 0);
		PartitionDown(Z_abij_lvl1_part2_1__D_0__D_1__D_2__D_3, Z_abij_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3, Z_abij_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, 3, 0);
		while(Z_abij_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3.Dimension(3) < Z_abij_lvl1_part2_1__D_0__D_1__D_2__D_3.Dimension(3))
		{
			RepartitionDown
			( Tau_efmn_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3,  Tau_efmn_lvl1_part2_1_lvl2_part3_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       Tau_efmn_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3,
			  Tau_efmn_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, Tau_efmn_lvl1_part2_1_lvl2_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );
			RepartitionDown
			( accum_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3,  accum_lvl1_part2_1_lvl2_part3_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       accum_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3,
			  accum_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, accum_lvl1_part2_1_lvl2_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );
			RepartitionDown
			( accum_lvl1_part3_1_lvl2_part2T__D_0__D_1__D_2__D_3,  accum_lvl1_part3_1_lvl2_part2_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       accum_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_2__D_3,
			  accum_lvl1_part3_1_lvl2_part2B__D_0__D_1__D_2__D_3, accum_lvl1_part3_1_lvl2_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );
			RepartitionDown
			( Z_abij_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3,  Z_abij_lvl1_part2_1_lvl2_part3_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       Z_abij_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3,
			  Z_abij_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, Z_abij_lvl1_part2_1_lvl2_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );

			   // Tau_efmn_lvl1_part2_1_lvl2_part3_1[D2,D1,D0,D3] <- Tau_efmn_lvl1_part2_1_lvl2_part3_1[D0,D1,D2,D3]
			Tau_efmn_lvl1_part2_1_lvl2_part3_1__D_2__D_1__D_0__D_3.AlignModesWith( modes_0_1, y_abef__D_0__D_1__D_2__D_3, modes_2_3 );
			Tau_efmn_lvl1_part2_1_lvl2_part3_1__D_2__D_1__D_0__D_3.AllToAllRedistFrom( Tau_efmn_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3, modes_0_2 );
			   // Tau_efmn_lvl1_part2_1_lvl2_part3_1[D2,D3,D0,D1] <- Tau_efmn_lvl1_part2_1_lvl2_part3_1[D2,D1,D0,D3]
			Tau_efmn_lvl1_part2_1_lvl2_part3_1__D_2__D_3__D_0__D_1.AlignModesWith( modes_0_1, y_abef__D_0__D_1__D_2__D_3, modes_2_3 );
			Tau_efmn_lvl1_part2_1_lvl2_part3_1__D_2__D_3__D_0__D_1.AllToAllRedistFrom( Tau_efmn_lvl1_part2_1_lvl2_part3_1__D_2__D_1__D_0__D_3, modes_1_3 );
			Tau_efmn_lvl1_part2_1_lvl2_part3_1__D_2__D_1__D_0__D_3.EmptyData();
			   // Tau_efmn_lvl1_part2_1_lvl2_part3_1[D2,D3,*,*] <- Tau_efmn_lvl1_part2_1_lvl2_part3_1[D2,D3,D0,D1]
			Tau_efmn_lvl1_part2_1_lvl2_part3_1__D_2__D_3__S__S.AlignModesWith( modes_0_1, y_abef__D_0__D_1__D_2__D_3, modes_2_3 );
			Tau_efmn_lvl1_part2_1_lvl2_part3_1__D_2__D_3__S__S.AllGatherRedistFrom( Tau_efmn_lvl1_part2_1_lvl2_part3_1__D_2__D_3__D_0__D_1, modes_0_1 );
			Tau_efmn_lvl1_part2_1_lvl2_part3_1__D_2__D_3__D_0__D_1.EmptyData();
			   // accum_lvl1_part3_1_lvl2_part2_1[D1,D0,D2,D3] <- accum_lvl1_part3_1_lvl2_part2_1[D0,D1,D2,D3]
			accum_lvl1_part3_1_lvl2_part2_1__D_1__D_0__D_2__D_3.AlignModesWith( modes_0_1_2_3, accum_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3, modes_1_0_3_2 );
			accum_lvl1_part3_1_lvl2_part2_1__D_1__D_0__D_2__D_3.AllToAllRedistFrom( accum_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_2__D_3, modes_0_1 );
			   // accum_lvl1_part3_1_lvl2_part2_1[D1,D0,D3,D2] <- accum_lvl1_part3_1_lvl2_part2_1[D1,D0,D2,D3]
			accum_lvl1_part3_1_lvl2_part2_1__D_1__D_0__D_3__D_2.AlignModesWith( modes_0_1_2_3, accum_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3, modes_1_0_3_2 );
			accum_lvl1_part3_1_lvl2_part2_1__D_1__D_0__D_3__D_2.AllToAllRedistFrom( accum_lvl1_part3_1_lvl2_part2_1__D_1__D_0__D_2__D_3, modes_2_3 );
			accum_lvl1_part3_1_lvl2_part2_1__D_1__D_0__D_2__D_3.EmptyData();
PROFILE_SECTION("COMPUTE");
PROFILE_FLOPS(3*prod(accum_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3.Shape()));
			YAxpPx( 1.0, accum_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3, 1.0, accum_lvl1_part3_1_lvl2_part2_1__D_1__D_0__D_3__D_2, perm_1_0_3_2, Z_abij_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3 );
PROFILE_STOP;
			accum_lvl1_part3_1_lvl2_part2_1__D_1__D_0__D_3__D_2.EmptyData();
			Z_abij_lvl1_part2_1_lvl2_part3_1__D_0__D_1__S__S__D_2__D_3.AlignModesWith( modes_0_1, y_abef__D_0__D_1__D_2__D_3, modes_0_1 );
			overwrite_tmpShape_Z = Z_abij_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3.Shape();
			overwrite_tmpShape_Z.push_back( g.Shape()[2] );
			overwrite_tmpShape_Z.push_back( g.Shape()[3] );
			Z_abij_lvl1_part2_1_lvl2_part3_1__D_0__D_1__S__S__D_2__D_3.ResizeTo( overwrite_tmpShape_Z );
			   // 1.0 * y_abef[D0,D1,D2,D3]_abef * Tau_efmn_lvl1_part2_1_lvl2_part3_1[D2,D3,*,*]_efij + 0.0 * Z_abij_lvl1_part2_1_lvl2_part3_1[D0,D1,*,*,D2,D3]_abijef
PROFILE_SECTION("COMPUTE");
PROFILE_FLOPS(2*prod(Z_abij_lvl1_part2_1_lvl2_part3_1__D_0__D_1__S__S__D_2__D_3.Shape())*y_abef__D_0__D_1__D_2__D_3.Dimension(2)*y_abef__D_0__D_1__D_2__D_3.Dimension(3));
			LocalContract(1.0, y_abef__D_0__D_1__D_2__D_3.LockedTensor(), indices_abef, false,
				Tau_efmn_lvl1_part2_1_lvl2_part3_1__D_2__D_3__S__S.LockedTensor(), indices_efij, false,
				0.0, Z_abij_lvl1_part2_1_lvl2_part3_1__D_0__D_1__S__S__D_2__D_3.Tensor(), indices_abijef, false);
PROFILE_STOP;
			Tau_efmn_lvl1_part2_1_lvl2_part3_1__D_2__D_3__S__S.EmptyData();
			   // Z_abij_lvl1_part2_1_lvl2_part3_1[D0,D1,D2,D3] <- Z_abij_lvl1_part2_1_lvl2_part3_1[D0,D1,*,*,D2,D3] (with SumScatter on (D2)(D3))
PROFILE_SECTION("COMPUTE");
PROFILE_FLOPS(2*prod(Z_abij_lvl1_part2_1_lvl2_part3_1__D_0__D_1__S__S__D_2__D_3.Shape()));
			Z_abij_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3.ReduceScatterUpdateRedistFrom( Z_abij_lvl1_part2_1_lvl2_part3_1__D_0__D_1__S__S__D_2__D_3, 1.0, modes_5_4 );
PROFILE_STOP;
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
			SlidePartitionDown
			( accum_lvl1_part2_1_lvl2_part3T__D_0__D_1__D_2__D_3,  accum_lvl1_part2_1_lvl2_part3_0__D_0__D_1__D_2__D_3,
			       accum_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  accum_lvl1_part2_1_lvl2_part3B__D_0__D_1__D_2__D_3, accum_lvl1_part2_1_lvl2_part3_2__D_0__D_1__D_2__D_3, 3 );
			SlidePartitionDown
			( accum_lvl1_part3_1_lvl2_part2T__D_0__D_1__D_2__D_3,  accum_lvl1_part3_1_lvl2_part2_0__D_0__D_1__D_2__D_3,
			       accum_lvl1_part3_1_lvl2_part2_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  accum_lvl1_part3_1_lvl2_part2B__D_0__D_1__D_2__D_3, accum_lvl1_part3_1_lvl2_part2_2__D_0__D_1__D_2__D_3, 2 );

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
		SlidePartitionDown
		( accum_lvl1_part2T__D_0__D_1__D_2__D_3,  accum_lvl1_part2_0__D_0__D_1__D_2__D_3,
		       accum_lvl1_part2_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  accum_lvl1_part2B__D_0__D_1__D_2__D_3, accum_lvl1_part2_2__D_0__D_1__D_2__D_3, 2 );
		SlidePartitionDown
		( accum_lvl1_part3T__D_0__D_1__D_2__D_3,  accum_lvl1_part3_0__D_0__D_1__D_2__D_3,
		       accum_lvl1_part3_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  accum_lvl1_part3B__D_0__D_1__D_2__D_3, accum_lvl1_part3_2__D_0__D_1__D_2__D_3, 3 );

	}
	//****
	accum__D_0__D_1__D_2__D_3.EmptyData();
	accum__D_0__D_1__D_2__D_3.EmptyData();
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

			   // Tau_efmn_lvl1_part2_1_lvl2_part3_1[D0,D1,*,*] <- Tau_efmn_lvl1_part2_1_lvl2_part3_1[D0,D1,D2,D3]
			Tau_efmn_lvl1_part2_1_lvl2_part3_1_perm2301__S__S__D_0__D_1.AlignModesWith( modes_0_1, Z_abij__D_0__D_1__D_2__D_3, modes_0_1 );
			Tau_efmn_lvl1_part2_1_lvl2_part3_1_perm2301__S__S__D_0__D_1.AllGatherRedistFrom( Tau_efmn_lvl1_part2_1_lvl2_part3_1__D_0__D_1__D_2__D_3, modes_2_3 );
			   // Q_mnij_lvl1_part0_1_lvl2_part1_1[*,*,D2,D3] <- Q_mnij_lvl1_part0_1_lvl2_part1_1[D0,D1,D2,D3]
			Q_mnij_lvl1_part0_1_lvl2_part1_1_perm2301__D_2__D_3__S__S.AlignModesWith( modes_2_3, Z_abij__D_0__D_1__D_2__D_3, modes_2_3 );
			Q_mnij_lvl1_part0_1_lvl2_part1_1_perm2301__D_2__D_3__S__S.AllGatherRedistFrom( Q_mnij_lvl1_part0_1_lvl2_part1_1__D_0__D_1__D_2__D_3, modes_0_1 );
			   // 1.0 * Q_mnij_lvl1_part0_1_lvl2_part1_1[*,*,D2,D3]_ijmn * Tau_efmn_lvl1_part2_1_lvl2_part3_1[D0,D1,*,*]_mnab + 1.0 * Z_abij[D0,D1,D2,D3]_ijab
PROFILE_SECTION("COMPUTE");
PROFILE_FLOPS(2*prod(Z_abij_perm2301__D_2__D_3__D_0__D_1.Shape())*Q_mnij_lvl1_part0_1_lvl2_part1_1_perm2301__D_2__D_3__S__S.Dimension(0)*Q_mnij_lvl1_part0_1_lvl2_part1_1_perm2301__D_2__D_3__S__S.Dimension(1));
			LocalContractAndLocalEliminate(1.0, Q_mnij_lvl1_part0_1_lvl2_part1_1_perm2301__D_2__D_3__S__S.LockedTensor(), indices_ijmn, false,
				Tau_efmn_lvl1_part2_1_lvl2_part3_1_perm2301__S__S__D_0__D_1.LockedTensor(), indices_mnab, false,
				1.0, Z_abij_perm2301__D_2__D_3__D_0__D_1.Tensor(), indices_ijab, false);
PROFILE_STOP;
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

PROFILE_SECTION("COMPUTE");
PROFILE_FLOPS(prod(v_femn__D_0__D_1__D_2__D_3.Shape()));
	Yxpy( v_femn__D_0__D_1__D_2__D_3, Z_abij__D_0__D_1__D_2__D_3 );
PROFILE_STOP;


//****


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
//END_CODE

    /*****************************************/
    mpi::Barrier(g.OwningComm());
    runTime = mpi::Time() - startTime;
    long long flops = Timer::nflops("COMPUTE");
    gflops = flops / (1e9 * runTime);
#ifdef CORRECTNESS
//    DistTensor<double> diff_Z(dist__D_0__D_1__D_2__D_3, g);
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


