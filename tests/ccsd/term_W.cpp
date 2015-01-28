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
ObjShape overwrite_tmpShape_W;
TensorDistribution dist__S__D_3__D_1__S = tmen::StringToTensorDist("[(),(3),(1),()]");
TensorDistribution dist__D_0__S__S__D_2 = tmen::StringToTensorDist("[(0),(),(),(2)]");
TensorDistribution dist__D_0__S__D_2_1__D_3 = tmen::StringToTensorDist("[(0),(),(2,1),(3)]");
TensorDistribution dist__D_0__S__D_2__D_3 = tmen::StringToTensorDist("[(0),(),(2),(3)]");
TensorDistribution dist__D_0__S = tmen::StringToTensorDist("[(0),()]");
TensorDistribution dist__D_0__D_1__S__D_2_3 = tmen::StringToTensorDist("[(0),(1),(),(2,3)]");
TensorDistribution dist__D_0__D_1__S__D_2__D_3 = tmen::StringToTensorDist("[(0),(1),(),(2),(3)]");
TensorDistribution dist__D_0__D_1__D_2__D_3 = tmen::StringToTensorDist("[(0),(1),(2),(3)]");
TensorDistribution dist__D_0__D_1__D_3__D_2 = tmen::StringToTensorDist("[(0),(1),(3),(2)]");
TensorDistribution dist__D_0__D_3__D_1_2__S = tmen::StringToTensorDist("[(0),(3),(1,2),()]");
TensorDistribution dist__D_0__D_3__D_2_1__S = tmen::StringToTensorDist("[(0),(3),(2,1),()]");
TensorDistribution dist__D_1__S__D_2__D_3 = tmen::StringToTensorDist("[(1),(),(2),(3)]");
TensorDistribution dist__D_1__D_0__D_2__D_3 = tmen::StringToTensorDist("[(1),(0),(2),(3)]");
TensorDistribution dist__D_3__S = tmen::StringToTensorDist("[(3),()]");
TensorDistribution dist__D_3__D_2 = tmen::StringToTensorDist("[(3),(2)]");
TensorDistribution dist__D_0_1__D_2_3 = tmen::StringToTensorDist("[(0,1),(2,3)]");
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
ModeArray modes_0_1_3_2( 4 );
modes_0_1_3_2[0] = 0;
modes_0_1_3_2[1] = 1;
modes_0_1_3_2[2] = 3;
modes_0_1_3_2[3] = 2;
ModeArray modes_0_2( 2 );
modes_0_2[0] = 0;
modes_0_2[1] = 2;
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
IndexArray indices_emfn( 4 );
indices_emfn[0] = 'e';
indices_emfn[1] = 'm';
indices_emfn[2] = 'f';
indices_emfn[3] = 'n';
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
	//Tau_efmn_lvl1_part2_1_lvl2_part3_2[D0,D1,D2,D3]
DistTensor<double> Tau_efmn_lvl1_part2_1_lvl2_part3_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Tau_efmn_lvl1_part2_2[D0,D1,D2,D3]
DistTensor<double> Tau_efmn_lvl1_part2_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
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
	//t_fj_lvl2_part1_1[D3,D2]
DistTensor<double> t_fj_lvl2_part1_1__D_3__D_2( dist__D_3__D_2, g );
	//t_fj_lvl2_part1_1[D3,*]
DistTensor<double> t_fj_lvl2_part1_1__D_3__S( dist__D_3__S, g );
	//t_fj_lvl2_part1_1[D0,*]
DistTensor<double> t_fj_lvl2_part1_1_perm10__S__D_0( dist__D_0__S, g );
t_fj_lvl2_part1_1_perm10__S__D_0.SetLocalPermutation( perm_1_0 );
	//t_fj_lvl2_part1_2[D01,D23]
DistTensor<double> t_fj_lvl2_part1_2__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
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
// t_fj has 2 dims
//	Starting distribution: [D01,D23] or _D_0_1__D_2_3
ObjShape t_fj__D_0_1__D_2_3_tmpShape_W( 2 );
t_fj__D_0_1__D_2_3_tmpShape_W[ 0 ] = n_v;
t_fj__D_0_1__D_2_3_tmpShape_W[ 1 ] = n_o;
t_fj__D_0_1__D_2_3.ResizeTo( t_fj__D_0_1__D_2_3_tmpShape_W );
MakeUniform( t_fj__D_0_1__D_2_3 );
// r_bmfe has 4 dims
//	Starting distribution: [D0,D1,D2,D3] or _D_0__D_1__D_2__D_3
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
// T_bfnj has 4 dims
ObjShape T_bfnj__D_0__D_1__D_2__D_3_tmpShape_W( 4 );
T_bfnj__D_0__D_1__D_2__D_3_tmpShape_W[ 0 ] = n_v;
T_bfnj__D_0__D_1__D_2__D_3_tmpShape_W[ 1 ] = n_v;
T_bfnj__D_0__D_1__D_2__D_3_tmpShape_W[ 2 ] = n_o;
T_bfnj__D_0__D_1__D_2__D_3_tmpShape_W[ 3 ] = n_o;
T_bfnj__D_0__D_1__D_2__D_3.ResizeTo( T_bfnj__D_0__D_1__D_2__D_3_tmpShape_W );
MakeUniform( T_bfnj__D_0__D_1__D_2__D_3 );
// Tau_efmn has 4 dims
ObjShape Tau_efmn__D_0__D_1__D_2__D_3_tmpShape_W( 4 );
Tau_efmn__D_0__D_1__D_2__D_3_tmpShape_W[ 0 ] = n_v;
Tau_efmn__D_0__D_1__D_2__D_3_tmpShape_W[ 1 ] = n_v;
Tau_efmn__D_0__D_1__D_2__D_3_tmpShape_W[ 2 ] = n_o;
Tau_efmn__D_0__D_1__D_2__D_3_tmpShape_W[ 3 ] = n_o;
Tau_efmn__D_0__D_1__D_2__D_3.ResizeTo( Tau_efmn__D_0__D_1__D_2__D_3_tmpShape_W );
MakeUniform( Tau_efmn__D_0__D_1__D_2__D_3 );
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
//END_DECL

//******************************
//* Load tensors
//******************************
////////////////////////////////
//Performance testing
////////////////////////////////
std::stringstream fullName;
#ifdef CORRECTNESS
DistTensor<T> check_W(dist__D_0__D_1__D_2__D_3, g);
check_W.ResizeTo(W_bmje__D_0__D_1__D_2__D_3.Shape());
Read(r_bmfe__D_0__D_1__D_2__D_3, "ccsd_terms/term_r_small", BINARY_FLAT, false);
Read(u_mnje__D_0__D_1__D_2__D_3, "ccsd_terms/term_u_small", BINARY_FLAT, false);
Read(v_femn__D_0__D_1__D_2__D_3, "ccsd_terms/term_v_small", BINARY_FLAT, false);
Read(w_bmje__D_0__D_1__D_2__D_3, "ccsd_terms/term_w_small", BINARY_FLAT, false);
Read(x_bmej__D_0__D_1__D_2__D_3, "ccsd_terms/term_x_small", BINARY_FLAT, false);
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
fullName << "ccsd_terms/term_W_iter" << testIter;
Read(check_W, fullName.str(), BINARY_FLAT, false);
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


//END_CODE

    /*****************************************/
    mpi::Barrier(g.OwningComm());
    runTime = mpi::Time() - startTime;
    long long flops = Timer::nflops("COMPUTE");
    gflops = flops / (1e9 * runTime);
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


