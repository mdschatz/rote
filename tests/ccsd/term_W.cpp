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
Permutation perm_0_1;
perm_0_1.push_back(0);
perm_0_1.push_back(1);
Permutation perm_0_1_2_3;
perm_0_1_2_3.push_back(0);
perm_0_1_2_3.push_back(1);
perm_0_1_2_3.push_back(2);
perm_0_1_2_3.push_back(3);
Permutation perm_0_1_3_2;
perm_0_1_3_2.push_back(0);
perm_0_1_3_2.push_back(1);
perm_0_1_3_2.push_back(3);
perm_0_1_3_2.push_back(2);
Permutation perm_0_1_3_2_4;
perm_0_1_3_2_4.push_back(0);
perm_0_1_3_2_4.push_back(1);
perm_0_1_3_2_4.push_back(3);
perm_0_1_3_2_4.push_back(2);
perm_0_1_3_2_4.push_back(4);
Permutation perm_0_2_3_1;
perm_0_2_3_1.push_back(0);
perm_0_2_3_1.push_back(2);
perm_0_2_3_1.push_back(3);
perm_0_2_3_1.push_back(1);
Permutation perm_1_0;
perm_1_0.push_back(1);
perm_1_0.push_back(0);
Permutation perm_1_0_2_3;
perm_1_0_2_3.push_back(1);
perm_1_0_2_3.push_back(0);
perm_1_0_2_3.push_back(2);
perm_1_0_2_3.push_back(3);
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
Permutation perm_2_1_3_0;
perm_2_1_3_0.push_back(2);
perm_2_1_3_0.push_back(1);
perm_2_1_3_0.push_back(3);
perm_2_1_3_0.push_back(0);
Permutation perm_3_0_1_2;
perm_3_0_1_2.push_back(3);
perm_3_0_1_2.push_back(0);
perm_3_0_1_2.push_back(1);
perm_3_0_1_2.push_back(2);
Permutation perm_3_1_0_2;
perm_3_1_0_2.push_back(3);
perm_3_1_0_2.push_back(1);
perm_3_1_0_2.push_back(0);
perm_3_1_0_2.push_back(2);
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
ModeArray modes_0_2_3;
modes_0_2_3.push_back(0);
modes_0_2_3.push_back(2);
modes_0_2_3.push_back(3);
ModeArray modes_0_3;
modes_0_3.push_back(0);
modes_0_3.push_back(3);
ModeArray modes_1;
modes_1.push_back(1);
ModeArray modes_1_0_2_3;
modes_1_0_2_3.push_back(1);
modes_1_0_2_3.push_back(0);
modes_1_0_2_3.push_back(2);
modes_1_0_2_3.push_back(3);
ModeArray modes_1_2;
modes_1_2.push_back(1);
modes_1_2.push_back(2);
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
ModeArray modes_3;
modes_3.push_back(3);
ModeArray modes_3_1;
modes_3_1.push_back(3);
modes_3_1.push_back(1);
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
	//Tau_efmn_part2_1_part3_2[D0,D1,D2,D3]
DistTensor<double> Tau_efmn_part2_1_part3_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Tau_efmn_part2_2[D0,D1,D2,D3]
DistTensor<double> Tau_efmn_part2_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_bmje[D0,D1,D2,D3]
DistTensor<double> W_bmje__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_bmje_part1B[D0,D1,D2,D3]
DistTensor<double> W_bmje_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_bmje_part1T[D0,D1,D2,D3]
DistTensor<double> W_bmje_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_bmje_part1_0[D0,D1,D2,D3]
DistTensor<double> W_bmje_part1_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_bmje_part1_1[D0,D1,D2,D3]
DistTensor<double> W_bmje_part1_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_bmje_part1_1_part2B[D0,D1,D2,D3]
DistTensor<double> W_bmje_part1_1_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_bmje_part1_1_part2T[D0,D1,D2,D3]
DistTensor<double> W_bmje_part1_1_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_bmje_part1_1_part2_0[D0,D1,D2,D3]
DistTensor<double> W_bmje_part1_1_part2_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_bmje_part1_1_part2_1[D0,D1,D2,D3]
DistTensor<double> W_bmje_part1_1_part2_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_bmje_part1_1_part2_1[D0,D1,*,D2,D3]
DistTensor<double> W_bmje_part1_1_part2_1_perm01324__D_0__D_1__D_2__S__D_3( dist__D_0__D_1__S__D_2__D_3, g );
W_bmje_part1_1_part2_1_perm01324__D_0__D_1__D_2__S__D_3.SetLocalPermutation( perm_0_1_3_2_4 );
	//W_bmje_part1_1_part2_1_temp[D0,D1,D2,D3]
DistTensor<double> W_bmje_part1_1_part2_1_temp__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_bmje_part1_1_part2_1_temp[D0,D1,*,D23]
DistTensor<double> W_bmje_part1_1_part2_1_temp__D_0__D_1__S__D_2_3( dist__D_0__D_1__S__D_2_3, g );
	//W_bmje_part1_1_part2_2[D0,D1,D2,D3]
DistTensor<double> W_bmje_part1_1_part2_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_bmje_part1_1[D0,D1,D2,D3]
DistTensor<double> W_bmje_part1_1_perm1230__D_1__D_2__D_3__D_0( dist__D_0__D_1__D_2__D_3, g );
W_bmje_part1_1_perm1230__D_1__D_2__D_3__D_0.SetLocalPermutation( perm_1_2_3_0 );
	//W_bmje_part1_2[D0,D1,D2,D3]
DistTensor<double> W_bmje_part1_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_bmje[D0,D1,D2,D3]
DistTensor<double> W_bmje_perm3102__D_3__D_1__D_0__D_2( dist__D_0__D_1__D_2__D_3, g );
W_bmje_perm3102__D_3__D_1__D_0__D_2.SetLocalPermutation( perm_3_1_0_2 );
	//r_bmfe[D0,D1,D2,D3]
DistTensor<double> r_bmfe__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//r_bmfe_part0B[D0,D1,D2,D3]
DistTensor<double> r_bmfe_part0B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//r_bmfe_part0T[D0,D1,D2,D3]
DistTensor<double> r_bmfe_part0T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//r_bmfe_part0_0[D0,D1,D2,D3]
DistTensor<double> r_bmfe_part0_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//r_bmfe_part0_1[D0,D1,D2,D3]
DistTensor<double> r_bmfe_part0_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//r_bmfe_part0_1_part1B[D0,D1,D2,D3]
DistTensor<double> r_bmfe_part0_1_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//r_bmfe_part0_1_part1T[D0,D1,D2,D3]
DistTensor<double> r_bmfe_part0_1_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//r_bmfe_part0_1_part1_0[D0,D1,D2,D3]
DistTensor<double> r_bmfe_part0_1_part1_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//r_bmfe_part0_1_part1_1[D0,D1,D2,D3]
DistTensor<double> r_bmfe_part0_1_part1_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//r_bmfe_part0_1_part1_1[D0,D1,D3,D2]
DistTensor<double> r_bmfe_part0_1_part1_1__D_0__D_1__D_3__D_2( dist__D_0__D_1__D_3__D_2, g );
	//r_bmfe_part0_1_part1_2[D0,D1,D2,D3]
DistTensor<double> r_bmfe_part0_1_part1_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//r_bmfe_part0_2[D0,D1,D2,D3]
DistTensor<double> r_bmfe_part0_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
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
	//t_fj_part1_1[D3,D2]
DistTensor<double> t_fj_part1_1__D_3__D_2( dist__D_3__D_2, g );
	//t_fj_part1_1[D3,*]
DistTensor<double> t_fj_part1_1__D_3__S( dist__D_3__S, g );
	//t_fj_part1_1[D0,*]
DistTensor<double> t_fj_part1_1_perm10__S__D_0( dist__D_0__S, g );
t_fj_part1_1_perm10__S__D_0.SetLocalPermutation( perm_1_0 );
	//t_fj_part1_2[D01,D23]
DistTensor<double> t_fj_part1_2__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//temp1[D0,D1,D2,D3]
DistTensor<double> temp1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp1_part1B[D0,D1,D2,D3]
DistTensor<double> temp1_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp1_part1T[D0,D1,D2,D3]
DistTensor<double> temp1_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp1_part1_0[D0,D1,D2,D3]
DistTensor<double> temp1_part1_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp1_part1_1[D0,D1,D2,D3]
DistTensor<double> temp1_part1_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp1_part1_1_part2B[D0,D1,D2,D3]
DistTensor<double> temp1_part1_1_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp1_part1_1_part2T[D0,D1,D2,D3]
DistTensor<double> temp1_part1_1_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp1_part1_1_part2_0[D0,D1,D2,D3]
DistTensor<double> temp1_part1_1_part2_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp1_part1_1_part2_1[D0,D1,D2,D3]
DistTensor<double> temp1_part1_1_part2_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp1_part1_1_part2_1[D0,D1,D3,D2]
DistTensor<double> temp1_part1_1_part2_1__D_0__D_1__D_3__D_2( dist__D_0__D_1__D_3__D_2, g );
	//temp1_part1_1_part2_1[D0,*,*,D2]
DistTensor<double> temp1_part1_1_part2_1_perm1203__S__S__D_0__D_2( dist__D_0__S__S__D_2, g );
temp1_part1_1_part2_1_perm1203__S__S__D_0__D_2.SetLocalPermutation( perm_1_2_0_3 );
	//temp1_part1_1_part2_2[D0,D1,D2,D3]
DistTensor<double> temp1_part1_1_part2_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp1_part1_2[D0,D1,D2,D3]
DistTensor<double> temp1_part1_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
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
	//temp2_part0_1_part3B[D0,D1,D2,D3]
DistTensor<double> temp2_part0_1_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp2_part0_1_part3T[D0,D1,D2,D3]
DistTensor<double> temp2_part0_1_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp2_part0_1_part3_0[D0,D1,D2,D3]
DistTensor<double> temp2_part0_1_part3_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp2_part0_1_part3_1[D0,D1,D2,D3]
DistTensor<double> temp2_part0_1_part3_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp2_part0_1_part3_1[D0,D3,D12,*]
DistTensor<double> temp2_part0_1_part3_1__D_0__D_3__D_1_2__S( dist__D_0__D_3__D_1_2__S, g );
	//temp2_part0_1_part3_1[D0,D3,D21,*]
DistTensor<double> temp2_part0_1_part3_1__D_0__D_3__D_2_1__S( dist__D_0__D_3__D_2_1__S, g );
	//temp2_part0_1_part3_1[D0,*,D21,D3]
DistTensor<double> temp2_part0_1_part3_1__D_0__S__D_2_1__D_3( dist__D_0__S__D_2_1__D_3, g );
	//temp2_part0_1_part3_1[D0,*,D2,D3]
DistTensor<double> temp2_part0_1_part3_1__D_0__S__D_2__D_3( dist__D_0__S__D_2__D_3, g );
	//temp2_part0_1_part3_1[*,D3,D1,*]
DistTensor<double> temp2_part0_1_part3_1_perm1203__D_3__D_1__S__S( dist__S__D_3__D_1__S, g );
temp2_part0_1_part3_1_perm1203__D_3__D_1__S__S.SetLocalPermutation( perm_1_2_0_3 );
	//temp2_part0_1_part3_2[D0,D1,D2,D3]
DistTensor<double> temp2_part0_1_part3_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp2_part0_2[D0,D1,D2,D3]
DistTensor<double> temp2_part0_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
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
	//temp3_part0_1_part1_1[D0,D1,D2,D3]
DistTensor<double> temp3_part0_1_part1_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp3_part0_1_part1_1[D1,*,D2,D3]
DistTensor<double> temp3_part0_1_part1_1_perm0231__D_1__D_2__D_3__S( dist__D_1__S__D_2__D_3, g );
temp3_part0_1_part1_1_perm0231__D_1__D_2__D_3__S.SetLocalPermutation( perm_0_2_3_1 );
	//temp4[D0,D1,D2,D3]
DistTensor<double> temp4__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp4_part0B[D0,D1,D2,D3]
DistTensor<double> temp4_part0B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp4_part0T[D0,D1,D2,D3]
DistTensor<double> temp4_part0T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp4_part0_0[D0,D1,D2,D3]
DistTensor<double> temp4_part0_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp4_part0_1[D0,D1,D2,D3]
DistTensor<double> temp4_part0_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp4_part0_1_part1B[D0,D1,D2,D3]
DistTensor<double> temp4_part0_1_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp4_part0_1_part1T[D0,D1,D2,D3]
DistTensor<double> temp4_part0_1_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp4_part0_1_part1_0[D0,D1,D2,D3]
DistTensor<double> temp4_part0_1_part1_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp4_part0_1_part1_1[D0,D1,D2,D3]
DistTensor<double> temp4_part0_1_part1_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp4_part0_1_part1_2[D0,D1,D2,D3]
DistTensor<double> temp4_part0_1_part1_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp4_part0_2[D0,D1,D2,D3]
DistTensor<double> temp4_part0_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp4_part1B[D0,D1,D2,D3]
DistTensor<double> temp4_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp4_part1T[D0,D1,D2,D3]
DistTensor<double> temp4_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp4_part1_0[D0,D1,D2,D3]
DistTensor<double> temp4_part1_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp4_part1_1[D0,D1,D2,D3]
DistTensor<double> temp4_part1_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp4_part1_2[D0,D1,D2,D3]
DistTensor<double> temp4_part1_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
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
	//w_bmje[D0,D1,D2,D3]
DistTensor<double> w_bmje__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//w_bmje_part1B[D0,D1,D2,D3]
DistTensor<double> w_bmje_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//w_bmje_part1T[D0,D1,D2,D3]
DistTensor<double> w_bmje_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//w_bmje_part1_0[D0,D1,D2,D3]
DistTensor<double> w_bmje_part1_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//w_bmje_part1_1[D0,D1,D2,D3]
DistTensor<double> w_bmje_part1_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//w_bmje_part1_1_part2B[D0,D1,D2,D3]
DistTensor<double> w_bmje_part1_1_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//w_bmje_part1_1_part2T[D0,D1,D2,D3]
DistTensor<double> w_bmje_part1_1_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//w_bmje_part1_1_part2_0[D0,D1,D2,D3]
DistTensor<double> w_bmje_part1_1_part2_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//w_bmje_part1_1_part2_1[D0,D1,D2,D3]
DistTensor<double> w_bmje_part1_1_part2_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//w_bmje_part1_1_part2_2[D0,D1,D2,D3]
DistTensor<double> w_bmje_part1_1_part2_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//w_bmje_part1_2[D0,D1,D2,D3]
DistTensor<double> w_bmje_part1_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//x_bmej[D0,D1,D2,D3]
DistTensor<double> x_bmej__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//x_bmej_part1B[D0,D1,D2,D3]
DistTensor<double> x_bmej_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//x_bmej_part1T[D0,D1,D2,D3]
DistTensor<double> x_bmej_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//x_bmej_part1_0[D0,D1,D2,D3]
DistTensor<double> x_bmej_part1_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//x_bmej_part1_1[D0,D1,D2,D3]
DistTensor<double> x_bmej_part1_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//x_bmej_part1_1_part3B[D0,D1,D2,D3]
DistTensor<double> x_bmej_part1_1_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//x_bmej_part1_1_part3T[D0,D1,D2,D3]
DistTensor<double> x_bmej_part1_1_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//x_bmej_part1_1_part3_0[D0,D1,D2,D3]
DistTensor<double> x_bmej_part1_1_part3_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//x_bmej_part1_1_part3_1[D0,D1,D2,D3]
DistTensor<double> x_bmej_part1_1_part3_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//x_bmej_part1_1_part3_1[D0,D1,D3,D2]
DistTensor<double> x_bmej_part1_1_part3_1__D_0__D_1__D_3__D_2( dist__D_0__D_1__D_3__D_2, g );
	//x_bmej_part1_1_part3_2[D0,D1,D2,D3]
DistTensor<double> x_bmej_part1_1_part3_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//x_bmej_part1_2[D0,D1,D2,D3]
DistTensor<double> x_bmej_part1_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
// r_bmfe has 4 dims
//	Starting distribution: [D0,D1,D2,D3] or _D_0__D_1__D_2__D_3
ObjShape r_bmfe__D_0__D_1__D_2__D_3_tempShape;
r_bmfe__D_0__D_1__D_2__D_3_tempShape.push_back( n_v );
r_bmfe__D_0__D_1__D_2__D_3_tempShape.push_back( n_o );
r_bmfe__D_0__D_1__D_2__D_3_tempShape.push_back( n_v );
r_bmfe__D_0__D_1__D_2__D_3_tempShape.push_back( n_v );
r_bmfe__D_0__D_1__D_2__D_3.ResizeTo( r_bmfe__D_0__D_1__D_2__D_3_tempShape );
MakeUniform( r_bmfe__D_0__D_1__D_2__D_3 );
tempShape = r_bmfe__D_0__D_1__D_2__D_3.Shape();
temp4__D_0__D_1__D_2__D_3.ResizeTo( tempShape );
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
temp2__D_0__D_1__D_2__D_3.ResizeTo( tempShape );
// T_bfnj has 4 dims
//	Starting distribution: [D0,D1,D2,D3] or _D_0__D_1__D_2__D_3
ObjShape T_bfnj__D_0__D_1__D_2__D_3_tempShape;
T_bfnj__D_0__D_1__D_2__D_3_tempShape.push_back( n_v );
T_bfnj__D_0__D_1__D_2__D_3_tempShape.push_back( n_v );
T_bfnj__D_0__D_1__D_2__D_3_tempShape.push_back( n_o );
T_bfnj__D_0__D_1__D_2__D_3_tempShape.push_back( n_o );
T_bfnj__D_0__D_1__D_2__D_3.ResizeTo( T_bfnj__D_0__D_1__D_2__D_3_tempShape );
MakeUniform( T_bfnj__D_0__D_1__D_2__D_3 );
// Tau_efmn has 4 dims
//	Starting distribution: [D0,D1,D2,D3] or _D_0__D_1__D_2__D_3
ObjShape Tau_efmn__D_0__D_1__D_2__D_3_tempShape;
Tau_efmn__D_0__D_1__D_2__D_3_tempShape.push_back( n_v );
Tau_efmn__D_0__D_1__D_2__D_3_tempShape.push_back( n_v );
Tau_efmn__D_0__D_1__D_2__D_3_tempShape.push_back( n_o );
Tau_efmn__D_0__D_1__D_2__D_3_tempShape.push_back( n_o );
Tau_efmn__D_0__D_1__D_2__D_3.ResizeTo( Tau_efmn__D_0__D_1__D_2__D_3_tempShape );
MakeUniform( Tau_efmn__D_0__D_1__D_2__D_3 );
tempShape = T_bfnj__D_0__D_1__D_2__D_3.Shape();
temp1__D_0__D_1__D_2__D_3.ResizeTo( tempShape );
// w_bmje has 4 dims
//	Starting distribution: [D0,D1,D2,D3] or _D_0__D_1__D_2__D_3
ObjShape w_bmje__D_0__D_1__D_2__D_3_tempShape;
w_bmje__D_0__D_1__D_2__D_3_tempShape.push_back( n_v );
w_bmje__D_0__D_1__D_2__D_3_tempShape.push_back( n_o );
w_bmje__D_0__D_1__D_2__D_3_tempShape.push_back( n_o );
w_bmje__D_0__D_1__D_2__D_3_tempShape.push_back( n_v );
w_bmje__D_0__D_1__D_2__D_3.ResizeTo( w_bmje__D_0__D_1__D_2__D_3_tempShape );
MakeUniform( w_bmje__D_0__D_1__D_2__D_3 );
// x_bmej has 4 dims
//	Starting distribution: [D0,D1,D2,D3] or _D_0__D_1__D_2__D_3
ObjShape x_bmej__D_0__D_1__D_2__D_3_tempShape;
x_bmej__D_0__D_1__D_2__D_3_tempShape.push_back( n_v );
x_bmej__D_0__D_1__D_2__D_3_tempShape.push_back( n_o );
x_bmej__D_0__D_1__D_2__D_3_tempShape.push_back( n_v );
x_bmej__D_0__D_1__D_2__D_3_tempShape.push_back( n_o );
x_bmej__D_0__D_1__D_2__D_3.ResizeTo( x_bmej__D_0__D_1__D_2__D_3_tempShape );
MakeUniform( x_bmej__D_0__D_1__D_2__D_3 );
// W_bmje has 4 dims
//	Starting distribution: [D0,D1,D2,D3] or _D_0__D_1__D_2__D_3
ObjShape W_bmje__D_0__D_1__D_2__D_3_tempShape;
W_bmje__D_0__D_1__D_2__D_3_tempShape.push_back( n_v );
W_bmje__D_0__D_1__D_2__D_3_tempShape.push_back( n_o );
W_bmje__D_0__D_1__D_2__D_3_tempShape.push_back( n_o );
W_bmje__D_0__D_1__D_2__D_3_tempShape.push_back( n_v );
W_bmje__D_0__D_1__D_2__D_3.ResizeTo( W_bmje__D_0__D_1__D_2__D_3_tempShape );
MakeUniform( W_bmje__D_0__D_1__D_2__D_3 );
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
check.ResizeTo(W_bmje__D_0__D_1__D_2__D_3.Shape());
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
fullName << "ccsd_terms/term_T_iter" << testIter;
Read(T_bfnj__D_0__D_1__D_2__D_3, fullName.str(), BINARY_FLAT, false);
fullName.str("");
fullName.clear();
fullName << "ccsd_terms/term_Tau_iter" << testIter;
Read(Tau_efmn__D_0__D_1__D_2__D_3, fullName.str(), BINARY_FLAT, false);
fullName.str("");
fullName.clear();
fullName << "ccsd_terms/term_W_iter" << testIter;
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


	//**** (out of 1)
	//**** Is real	0 shadows
		//Outputs:
		//  temp4__D_0__D_1__D_2__D_3
	PartitionDown(r_bmfe__D_0__D_1__D_2__D_3, r_bmfe_part0T__D_0__D_1__D_2__D_3, r_bmfe_part0B__D_0__D_1__D_2__D_3, 0, 0);
	PartitionDown(temp4__D_0__D_1__D_2__D_3, temp4_part0T__D_0__D_1__D_2__D_3, temp4_part0B__D_0__D_1__D_2__D_3, 0, 0);
	while(temp4_part0T__D_0__D_1__D_2__D_3.Dimension(0) < temp4__D_0__D_1__D_2__D_3.Dimension(0))
	{
		RepartitionDown
		( r_bmfe_part0T__D_0__D_1__D_2__D_3,  r_bmfe_part0_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       r_bmfe_part0_1__D_0__D_1__D_2__D_3,
		  r_bmfe_part0B__D_0__D_1__D_2__D_3, r_bmfe_part0_2__D_0__D_1__D_2__D_3, 0, blkSize );
		RepartitionDown
		( temp4_part0T__D_0__D_1__D_2__D_3,  temp4_part0_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       temp4_part0_1__D_0__D_1__D_2__D_3,
		  temp4_part0B__D_0__D_1__D_2__D_3, temp4_part0_2__D_0__D_1__D_2__D_3, 0, blkSize );

		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  temp4_part0_1__D_0__D_1__D_2__D_3
		PartitionDown(r_bmfe_part0_1__D_0__D_1__D_2__D_3, r_bmfe_part0_1_part1T__D_0__D_1__D_2__D_3, r_bmfe_part0_1_part1B__D_0__D_1__D_2__D_3, 1, 0);
		PartitionDown(temp4_part0_1__D_0__D_1__D_2__D_3, temp4_part0_1_part1T__D_0__D_1__D_2__D_3, temp4_part0_1_part1B__D_0__D_1__D_2__D_3, 1, 0);
		while(temp4_part0_1_part1T__D_0__D_1__D_2__D_3.Dimension(1) < temp4_part0_1__D_0__D_1__D_2__D_3.Dimension(1))
		{
			RepartitionDown
			( r_bmfe_part0_1_part1T__D_0__D_1__D_2__D_3,  r_bmfe_part0_1_part1_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       r_bmfe_part0_1_part1_1__D_0__D_1__D_2__D_3,
			  r_bmfe_part0_1_part1B__D_0__D_1__D_2__D_3, r_bmfe_part0_1_part1_2__D_0__D_1__D_2__D_3, 1, blkSize );
			RepartitionDown
			( temp4_part0_1_part1T__D_0__D_1__D_2__D_3,  temp4_part0_1_part1_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       temp4_part0_1_part1_1__D_0__D_1__D_2__D_3,
			  temp4_part0_1_part1B__D_0__D_1__D_2__D_3, temp4_part0_1_part1_2__D_0__D_1__D_2__D_3, 1, blkSize );

			   // r_bmfe_part0_1_part1_1[D0,D1,D3,D2] <- r_bmfe_part0_1_part1_1[D0,D1,D2,D3]
			r_bmfe_part0_1_part1_1__D_0__D_1__D_3__D_2.AlignModesWith( modes_0_1_2_3, r_bmfe_part0_1_part1_1__D_0__D_1__D_2__D_3, modes_0_1_3_2 );
			r_bmfe_part0_1_part1_1__D_0__D_1__D_3__D_2.AllToAllRedistFrom( r_bmfe_part0_1_part1_1__D_0__D_1__D_2__D_3, modes_2_3 );
			YAxpPx( -1.0, r_bmfe_part0_1_part1_1__D_0__D_1__D_2__D_3, 2.0, r_bmfe_part0_1_part1_1__D_0__D_1__D_3__D_2, perm_0_1_3_2, temp4_part0_1_part1_1__D_0__D_1__D_2__D_3 );
			r_bmfe_part0_1_part1_1__D_0__D_1__D_3__D_2.EmptyData();

			SlidePartitionDown
			( r_bmfe_part0_1_part1T__D_0__D_1__D_2__D_3,  r_bmfe_part0_1_part1_0__D_0__D_1__D_2__D_3,
			       r_bmfe_part0_1_part1_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  r_bmfe_part0_1_part1B__D_0__D_1__D_2__D_3, r_bmfe_part0_1_part1_2__D_0__D_1__D_2__D_3, 1 );
			SlidePartitionDown
			( temp4_part0_1_part1T__D_0__D_1__D_2__D_3,  temp4_part0_1_part1_0__D_0__D_1__D_2__D_3,
			       temp4_part0_1_part1_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  temp4_part0_1_part1B__D_0__D_1__D_2__D_3, temp4_part0_1_part1_2__D_0__D_1__D_2__D_3, 1 );

		}
		//****

		SlidePartitionDown
		( r_bmfe_part0T__D_0__D_1__D_2__D_3,  r_bmfe_part0_0__D_0__D_1__D_2__D_3,
		       r_bmfe_part0_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  r_bmfe_part0B__D_0__D_1__D_2__D_3, r_bmfe_part0_2__D_0__D_1__D_2__D_3, 0 );
		SlidePartitionDown
		( temp4_part0T__D_0__D_1__D_2__D_3,  temp4_part0_0__D_0__D_1__D_2__D_3,
		       temp4_part0_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  temp4_part0B__D_0__D_1__D_2__D_3, temp4_part0_2__D_0__D_1__D_2__D_3, 0 );

	}
	r_bmfe__D_0__D_1__D_2__D_3.EmptyData();
	r_bmfe__D_0__D_1__D_2__D_3.EmptyData();
	//****
	//**** (out of 1)
	//**** Is real	0 shadows
		//Outputs:
		//  temp2__D_0__D_1__D_2__D_3
	PartitionDown(v_femn__D_0__D_1__D_2__D_3, v_femn_part2T__D_0__D_1__D_2__D_3, v_femn_part2B__D_0__D_1__D_2__D_3, 2, 0);
	PartitionDown(v_femn__D_0__D_1__D_2__D_3, v_femn_part3T__D_0__D_1__D_2__D_3, v_femn_part3B__D_0__D_1__D_2__D_3, 3, 0);
	PartitionDown(temp2__D_0__D_1__D_2__D_3, temp2_part2T__D_0__D_1__D_2__D_3, temp2_part2B__D_0__D_1__D_2__D_3, 2, 0);
	while(temp2_part2T__D_0__D_1__D_2__D_3.Dimension(2) < temp2__D_0__D_1__D_2__D_3.Dimension(2))
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
		( temp2_part2T__D_0__D_1__D_2__D_3,  temp2_part2_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       temp2_part2_1__D_0__D_1__D_2__D_3,
		  temp2_part2B__D_0__D_1__D_2__D_3, temp2_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );

		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  temp2_part2_1__D_0__D_1__D_2__D_3
		PartitionDown(v_femn_part2_1__D_0__D_1__D_2__D_3, v_femn_part2_1_part3T__D_0__D_1__D_2__D_3, v_femn_part2_1_part3B__D_0__D_1__D_2__D_3, 3, 0);
		PartitionDown(v_femn_part3_1__D_0__D_1__D_2__D_3, v_femn_part3_1_part2T__D_0__D_1__D_2__D_3, v_femn_part3_1_part2B__D_0__D_1__D_2__D_3, 2, 0);
		PartitionDown(temp2_part2_1__D_0__D_1__D_2__D_3, temp2_part2_1_part3T__D_0__D_1__D_2__D_3, temp2_part2_1_part3B__D_0__D_1__D_2__D_3, 3, 0);
		while(temp2_part2_1_part3T__D_0__D_1__D_2__D_3.Dimension(3) < temp2_part2_1__D_0__D_1__D_2__D_3.Dimension(3))
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
			( temp2_part2_1_part3T__D_0__D_1__D_2__D_3,  temp2_part2_1_part3_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       temp2_part2_1_part3_1__D_0__D_1__D_2__D_3,
			  temp2_part2_1_part3B__D_0__D_1__D_2__D_3, temp2_part2_1_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );

			   // v_femn_part3_1_part2_1[D0,D1,D3,D2] <- v_femn_part3_1_part2_1[D0,D1,D2,D3]
			v_femn_part3_1_part2_1__D_0__D_1__D_3__D_2.AlignModesWith( modes_0_1_2_3, v_femn_part2_1_part3_1__D_0__D_1__D_2__D_3, modes_0_1_3_2 );
			v_femn_part3_1_part2_1__D_0__D_1__D_3__D_2.AllToAllRedistFrom( v_femn_part3_1_part2_1__D_0__D_1__D_2__D_3, modes_2_3 );
			YAxpPx( -1.0, v_femn_part2_1_part3_1__D_0__D_1__D_2__D_3, 2.0, v_femn_part3_1_part2_1__D_0__D_1__D_3__D_2, perm_0_1_3_2, temp2_part2_1_part3_1__D_0__D_1__D_2__D_3 );
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
			( temp2_part2_1_part3T__D_0__D_1__D_2__D_3,  temp2_part2_1_part3_0__D_0__D_1__D_2__D_3,
			       temp2_part2_1_part3_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  temp2_part2_1_part3B__D_0__D_1__D_2__D_3, temp2_part2_1_part3_2__D_0__D_1__D_2__D_3, 3 );

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
		( temp2_part2T__D_0__D_1__D_2__D_3,  temp2_part2_0__D_0__D_1__D_2__D_3,
		       temp2_part2_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  temp2_part2B__D_0__D_1__D_2__D_3, temp2_part2_2__D_0__D_1__D_2__D_3, 2 );

	}
	v_femn__D_0__D_1__D_2__D_3.EmptyData();
	v_femn__D_0__D_1__D_2__D_3.EmptyData();
	v_femn__D_0__D_1__D_2__D_3.EmptyData();
	v_femn__D_0__D_1__D_2__D_3.EmptyData();
	//****
	//**** (out of 1)
	//**** Is real	0 shadows
		//Outputs:
		//  temp1__D_0__D_1__D_2__D_3
	PartitionDown(T_bfnj__D_0__D_1__D_2__D_3, T_bfnj_part2T__D_0__D_1__D_2__D_3, T_bfnj_part2B__D_0__D_1__D_2__D_3, 2, 0);
	PartitionDown(Tau_efmn__D_0__D_1__D_2__D_3, Tau_efmn_part2T__D_0__D_1__D_2__D_3, Tau_efmn_part2B__D_0__D_1__D_2__D_3, 2, 0);
	PartitionDown(T_bfnj__D_0__D_1__D_2__D_3, T_bfnj_part3T__D_0__D_1__D_2__D_3, T_bfnj_part3B__D_0__D_1__D_2__D_3, 3, 0);
	PartitionDown(temp1__D_0__D_1__D_2__D_3, temp1_part2T__D_0__D_1__D_2__D_3, temp1_part2B__D_0__D_1__D_2__D_3, 2, 0);
	while(temp1_part2T__D_0__D_1__D_2__D_3.Dimension(2) < temp1__D_0__D_1__D_2__D_3.Dimension(2))
	{
		RepartitionDown
		( T_bfnj_part2T__D_0__D_1__D_2__D_3,  T_bfnj_part2_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       T_bfnj_part2_1__D_0__D_1__D_2__D_3,
		  T_bfnj_part2B__D_0__D_1__D_2__D_3, T_bfnj_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );
		RepartitionDown
		( Tau_efmn_part2T__D_0__D_1__D_2__D_3,  Tau_efmn_part2_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       Tau_efmn_part2_1__D_0__D_1__D_2__D_3,
		  Tau_efmn_part2B__D_0__D_1__D_2__D_3, Tau_efmn_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );
		RepartitionDown
		( T_bfnj_part3T__D_0__D_1__D_2__D_3,  T_bfnj_part3_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       T_bfnj_part3_1__D_0__D_1__D_2__D_3,
		  T_bfnj_part3B__D_0__D_1__D_2__D_3, T_bfnj_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );
		RepartitionDown
		( temp1_part2T__D_0__D_1__D_2__D_3,  temp1_part2_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       temp1_part2_1__D_0__D_1__D_2__D_3,
		  temp1_part2B__D_0__D_1__D_2__D_3, temp1_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );

		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  temp1_part2_1__D_0__D_1__D_2__D_3
		PartitionDown(T_bfnj_part2_1__D_0__D_1__D_2__D_3, T_bfnj_part2_1_part3T__D_0__D_1__D_2__D_3, T_bfnj_part2_1_part3B__D_0__D_1__D_2__D_3, 3, 0);
		PartitionDown(Tau_efmn_part2_1__D_0__D_1__D_2__D_3, Tau_efmn_part2_1_part3T__D_0__D_1__D_2__D_3, Tau_efmn_part2_1_part3B__D_0__D_1__D_2__D_3, 3, 0);
		PartitionDown(T_bfnj_part3_1__D_0__D_1__D_2__D_3, T_bfnj_part3_1_part2T__D_0__D_1__D_2__D_3, T_bfnj_part3_1_part2B__D_0__D_1__D_2__D_3, 2, 0);
		PartitionDown(temp1_part2_1__D_0__D_1__D_2__D_3, temp1_part2_1_part3T__D_0__D_1__D_2__D_3, temp1_part2_1_part3B__D_0__D_1__D_2__D_3, 3, 0);
		while(temp1_part2_1_part3T__D_0__D_1__D_2__D_3.Dimension(3) < temp1_part2_1__D_0__D_1__D_2__D_3.Dimension(3))
		{
			RepartitionDown
			( T_bfnj_part2_1_part3T__D_0__D_1__D_2__D_3,  T_bfnj_part2_1_part3_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       T_bfnj_part2_1_part3_1__D_0__D_1__D_2__D_3,
			  T_bfnj_part2_1_part3B__D_0__D_1__D_2__D_3, T_bfnj_part2_1_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );
			RepartitionDown
			( Tau_efmn_part2_1_part3T__D_0__D_1__D_2__D_3,  Tau_efmn_part2_1_part3_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       Tau_efmn_part2_1_part3_1__D_0__D_1__D_2__D_3,
			  Tau_efmn_part2_1_part3B__D_0__D_1__D_2__D_3, Tau_efmn_part2_1_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );
			RepartitionDown
			( T_bfnj_part3_1_part2T__D_0__D_1__D_2__D_3,  T_bfnj_part3_1_part2_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       T_bfnj_part3_1_part2_1__D_0__D_1__D_2__D_3,
			  T_bfnj_part3_1_part2B__D_0__D_1__D_2__D_3, T_bfnj_part3_1_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );
			RepartitionDown
			( temp1_part2_1_part3T__D_0__D_1__D_2__D_3,  temp1_part2_1_part3_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       temp1_part2_1_part3_1__D_0__D_1__D_2__D_3,
			  temp1_part2_1_part3B__D_0__D_1__D_2__D_3, temp1_part2_1_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );

			   // T_bfnj_part3_1_part2_1[D0,D1,D3,D2] <- T_bfnj_part3_1_part2_1[D0,D1,D2,D3]
			T_bfnj_part3_1_part2_1__D_0__D_1__D_3__D_2.AlignModesWith( modes_0_1_2_3, T_bfnj_part2_1_part3_1__D_0__D_1__D_2__D_3, modes_0_1_3_2 );
			T_bfnj_part3_1_part2_1__D_0__D_1__D_3__D_2.AllToAllRedistFrom( T_bfnj_part3_1_part2_1__D_0__D_1__D_2__D_3, modes_2_3 );
			ZAxpBypPx( 0.5, T_bfnj_part2_1_part3_1__D_0__D_1__D_2__D_3, -1.0, Tau_efmn_part2_1_part3_1__D_0__D_1__D_2__D_3, T_bfnj_part3_1_part2_1__D_0__D_1__D_3__D_2, perm_0_1_3_2, temp1_part2_1_part3_1__D_0__D_1__D_2__D_3 );
			T_bfnj_part3_1_part2_1__D_0__D_1__D_3__D_2.EmptyData();

			SlidePartitionDown
			( T_bfnj_part2_1_part3T__D_0__D_1__D_2__D_3,  T_bfnj_part2_1_part3_0__D_0__D_1__D_2__D_3,
			       T_bfnj_part2_1_part3_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  T_bfnj_part2_1_part3B__D_0__D_1__D_2__D_3, T_bfnj_part2_1_part3_2__D_0__D_1__D_2__D_3, 3 );
			SlidePartitionDown
			( Tau_efmn_part2_1_part3T__D_0__D_1__D_2__D_3,  Tau_efmn_part2_1_part3_0__D_0__D_1__D_2__D_3,
			       Tau_efmn_part2_1_part3_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  Tau_efmn_part2_1_part3B__D_0__D_1__D_2__D_3, Tau_efmn_part2_1_part3_2__D_0__D_1__D_2__D_3, 3 );
			SlidePartitionDown
			( T_bfnj_part3_1_part2T__D_0__D_1__D_2__D_3,  T_bfnj_part3_1_part2_0__D_0__D_1__D_2__D_3,
			       T_bfnj_part3_1_part2_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  T_bfnj_part3_1_part2B__D_0__D_1__D_2__D_3, T_bfnj_part3_1_part2_2__D_0__D_1__D_2__D_3, 2 );
			SlidePartitionDown
			( temp1_part2_1_part3T__D_0__D_1__D_2__D_3,  temp1_part2_1_part3_0__D_0__D_1__D_2__D_3,
			       temp1_part2_1_part3_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  temp1_part2_1_part3B__D_0__D_1__D_2__D_3, temp1_part2_1_part3_2__D_0__D_1__D_2__D_3, 3 );

		}
		//****

		SlidePartitionDown
		( T_bfnj_part2T__D_0__D_1__D_2__D_3,  T_bfnj_part2_0__D_0__D_1__D_2__D_3,
		       T_bfnj_part2_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  T_bfnj_part2B__D_0__D_1__D_2__D_3, T_bfnj_part2_2__D_0__D_1__D_2__D_3, 2 );
		SlidePartitionDown
		( Tau_efmn_part2T__D_0__D_1__D_2__D_3,  Tau_efmn_part2_0__D_0__D_1__D_2__D_3,
		       Tau_efmn_part2_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  Tau_efmn_part2B__D_0__D_1__D_2__D_3, Tau_efmn_part2_2__D_0__D_1__D_2__D_3, 2 );
		SlidePartitionDown
		( T_bfnj_part3T__D_0__D_1__D_2__D_3,  T_bfnj_part3_0__D_0__D_1__D_2__D_3,
		       T_bfnj_part3_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  T_bfnj_part3B__D_0__D_1__D_2__D_3, T_bfnj_part3_2__D_0__D_1__D_2__D_3, 3 );
		SlidePartitionDown
		( temp1_part2T__D_0__D_1__D_2__D_3,  temp1_part2_0__D_0__D_1__D_2__D_3,
		       temp1_part2_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  temp1_part2B__D_0__D_1__D_2__D_3, temp1_part2_2__D_0__D_1__D_2__D_3, 2 );

	}
	T_bfnj__D_0__D_1__D_2__D_3.EmptyData();
	Tau_efmn__D_0__D_1__D_2__D_3.EmptyData();
	T_bfnj__D_0__D_1__D_2__D_3.EmptyData();
	T_bfnj__D_0__D_1__D_2__D_3.EmptyData();
	Tau_efmn__D_0__D_1__D_2__D_3.EmptyData();
	T_bfnj__D_0__D_1__D_2__D_3.EmptyData();
	//****
	//**** (out of 1)
	//**** Is real	0 shadows
		//Outputs:
		//  W_bmje__D_0__D_1__D_2__D_3
	PartitionDown(w_bmje__D_0__D_1__D_2__D_3, w_bmje_part1T__D_0__D_1__D_2__D_3, w_bmje_part1B__D_0__D_1__D_2__D_3, 1, 0);
	PartitionDown(x_bmej__D_0__D_1__D_2__D_3, x_bmej_part1T__D_0__D_1__D_2__D_3, x_bmej_part1B__D_0__D_1__D_2__D_3, 1, 0);
	PartitionDown(W_bmje__D_0__D_1__D_2__D_3, W_bmje_part1T__D_0__D_1__D_2__D_3, W_bmje_part1B__D_0__D_1__D_2__D_3, 1, 0);
	while(W_bmje_part1T__D_0__D_1__D_2__D_3.Dimension(1) < W_bmje__D_0__D_1__D_2__D_3.Dimension(1))
	{
		RepartitionDown
		( w_bmje_part1T__D_0__D_1__D_2__D_3,  w_bmje_part1_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       w_bmje_part1_1__D_0__D_1__D_2__D_3,
		  w_bmje_part1B__D_0__D_1__D_2__D_3, w_bmje_part1_2__D_0__D_1__D_2__D_3, 1, blkSize );
		RepartitionDown
		( x_bmej_part1T__D_0__D_1__D_2__D_3,  x_bmej_part1_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       x_bmej_part1_1__D_0__D_1__D_2__D_3,
		  x_bmej_part1B__D_0__D_1__D_2__D_3, x_bmej_part1_2__D_0__D_1__D_2__D_3, 1, blkSize );
		RepartitionDown
		( W_bmje_part1T__D_0__D_1__D_2__D_3,  W_bmje_part1_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       W_bmje_part1_1__D_0__D_1__D_2__D_3,
		  W_bmje_part1B__D_0__D_1__D_2__D_3, W_bmje_part1_2__D_0__D_1__D_2__D_3, 1, blkSize );

		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  W_bmje_part1_1__D_0__D_1__D_2__D_3
		PartitionDown(w_bmje_part1_1__D_0__D_1__D_2__D_3, w_bmje_part1_1_part2T__D_0__D_1__D_2__D_3, w_bmje_part1_1_part2B__D_0__D_1__D_2__D_3, 2, 0);
		PartitionDown(x_bmej_part1_1__D_0__D_1__D_2__D_3, x_bmej_part1_1_part3T__D_0__D_1__D_2__D_3, x_bmej_part1_1_part3B__D_0__D_1__D_2__D_3, 3, 0);
		PartitionDown(W_bmje_part1_1__D_0__D_1__D_2__D_3, W_bmje_part1_1_part2T__D_0__D_1__D_2__D_3, W_bmje_part1_1_part2B__D_0__D_1__D_2__D_3, 2, 0);
		while(W_bmje_part1_1_part2T__D_0__D_1__D_2__D_3.Dimension(2) < W_bmje_part1_1__D_0__D_1__D_2__D_3.Dimension(2))
		{
			RepartitionDown
			( w_bmje_part1_1_part2T__D_0__D_1__D_2__D_3,  w_bmje_part1_1_part2_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       w_bmje_part1_1_part2_1__D_0__D_1__D_2__D_3,
			  w_bmje_part1_1_part2B__D_0__D_1__D_2__D_3, w_bmje_part1_1_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );
			RepartitionDown
			( x_bmej_part1_1_part3T__D_0__D_1__D_2__D_3,  x_bmej_part1_1_part3_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       x_bmej_part1_1_part3_1__D_0__D_1__D_2__D_3,
			  x_bmej_part1_1_part3B__D_0__D_1__D_2__D_3, x_bmej_part1_1_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );
			RepartitionDown
			( W_bmje_part1_1_part2T__D_0__D_1__D_2__D_3,  W_bmje_part1_1_part2_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       W_bmje_part1_1_part2_1__D_0__D_1__D_2__D_3,
			  W_bmje_part1_1_part2B__D_0__D_1__D_2__D_3, W_bmje_part1_1_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );

			   // x_bmej_part1_1_part3_1[D0,D1,D3,D2] <- x_bmej_part1_1_part3_1[D0,D1,D2,D3]
			x_bmej_part1_1_part3_1__D_0__D_1__D_3__D_2.AlignModesWith( modes_0_1_2_3, w_bmje_part1_1_part2_1__D_0__D_1__D_2__D_3, modes_0_1_3_2 );
			x_bmej_part1_1_part3_1__D_0__D_1__D_3__D_2.AllToAllRedistFrom( x_bmej_part1_1_part3_1__D_0__D_1__D_2__D_3, modes_2_3 );
			YAxpPx( 2.0, w_bmje_part1_1_part2_1__D_0__D_1__D_2__D_3, -0.5, x_bmej_part1_1_part3_1__D_0__D_1__D_3__D_2, perm_0_1_3_2, W_bmje_part1_1_part2_1__D_0__D_1__D_2__D_3 );
			x_bmej_part1_1_part3_1__D_0__D_1__D_3__D_2.EmptyData();

			SlidePartitionDown
			( w_bmje_part1_1_part2T__D_0__D_1__D_2__D_3,  w_bmje_part1_1_part2_0__D_0__D_1__D_2__D_3,
			       w_bmje_part1_1_part2_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  w_bmje_part1_1_part2B__D_0__D_1__D_2__D_3, w_bmje_part1_1_part2_2__D_0__D_1__D_2__D_3, 2 );
			SlidePartitionDown
			( x_bmej_part1_1_part3T__D_0__D_1__D_2__D_3,  x_bmej_part1_1_part3_0__D_0__D_1__D_2__D_3,
			       x_bmej_part1_1_part3_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  x_bmej_part1_1_part3B__D_0__D_1__D_2__D_3, x_bmej_part1_1_part3_2__D_0__D_1__D_2__D_3, 3 );
			SlidePartitionDown
			( W_bmje_part1_1_part2T__D_0__D_1__D_2__D_3,  W_bmje_part1_1_part2_0__D_0__D_1__D_2__D_3,
			       W_bmje_part1_1_part2_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  W_bmje_part1_1_part2B__D_0__D_1__D_2__D_3, W_bmje_part1_1_part2_2__D_0__D_1__D_2__D_3, 2 );

		}
		//****

		SlidePartitionDown
		( w_bmje_part1T__D_0__D_1__D_2__D_3,  w_bmje_part1_0__D_0__D_1__D_2__D_3,
		       w_bmje_part1_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  w_bmje_part1B__D_0__D_1__D_2__D_3, w_bmje_part1_2__D_0__D_1__D_2__D_3, 1 );
		SlidePartitionDown
		( x_bmej_part1T__D_0__D_1__D_2__D_3,  x_bmej_part1_0__D_0__D_1__D_2__D_3,
		       x_bmej_part1_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  x_bmej_part1B__D_0__D_1__D_2__D_3, x_bmej_part1_2__D_0__D_1__D_2__D_3, 1 );
		SlidePartitionDown
		( W_bmje_part1T__D_0__D_1__D_2__D_3,  W_bmje_part1_0__D_0__D_1__D_2__D_3,
		       W_bmje_part1_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  W_bmje_part1B__D_0__D_1__D_2__D_3, W_bmje_part1_2__D_0__D_1__D_2__D_3, 1 );

	}
	w_bmje__D_0__D_1__D_2__D_3.EmptyData();
	x_bmej__D_0__D_1__D_2__D_3.EmptyData();
	w_bmje__D_0__D_1__D_2__D_3.EmptyData();
	x_bmej__D_0__D_1__D_2__D_3.EmptyData();
	//****
	tempShape = W_bmje__D_0__D_1__D_2__D_3.Shape();
	W_bmje_perm3102__D_3__D_1__D_0__D_2.ResizeTo( tempShape );
	Scal( 0.0, W_bmje_perm3102__D_3__D_1__D_0__D_2 );
	//**** (out of 1)
	//**** Is real	0 shadows
		//Outputs:
		//  W_bmje_perm3102__D_3__D_1__D_0__D_2
	PartitionDown(temp2__D_0__D_1__D_2__D_3, temp2_part0T__D_0__D_1__D_2__D_3, temp2_part0B__D_0__D_1__D_2__D_3, 0, 0);
	PartitionDown(temp1__D_0__D_1__D_2__D_3, temp1_part1T__D_0__D_1__D_2__D_3, temp1_part1B__D_0__D_1__D_2__D_3, 1, 0);
	while(temp2_part0T__D_0__D_1__D_2__D_3.Dimension(0) < temp2__D_0__D_1__D_2__D_3.Dimension(0))
	{
		RepartitionDown
		( temp2_part0T__D_0__D_1__D_2__D_3,  temp2_part0_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       temp2_part0_1__D_0__D_1__D_2__D_3,
		  temp2_part0B__D_0__D_1__D_2__D_3, temp2_part0_2__D_0__D_1__D_2__D_3, 0, blkSize );
		RepartitionDown
		( temp1_part1T__D_0__D_1__D_2__D_3,  temp1_part1_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       temp1_part1_1__D_0__D_1__D_2__D_3,
		  temp1_part1B__D_0__D_1__D_2__D_3, temp1_part1_2__D_0__D_1__D_2__D_3, 1, blkSize );

		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  W_bmje_perm3102__D_3__D_1__D_0__D_2
		PartitionDown(temp2_part0_1__D_0__D_1__D_2__D_3, temp2_part0_1_part3T__D_0__D_1__D_2__D_3, temp2_part0_1_part3B__D_0__D_1__D_2__D_3, 3, 0);
		PartitionDown(temp1_part1_1__D_0__D_1__D_2__D_3, temp1_part1_1_part2T__D_0__D_1__D_2__D_3, temp1_part1_1_part2B__D_0__D_1__D_2__D_3, 2, 0);
		while(temp2_part0_1_part3T__D_0__D_1__D_2__D_3.Dimension(3) < temp2_part0_1__D_0__D_1__D_2__D_3.Dimension(3))
		{
			RepartitionDown
			( temp2_part0_1_part3T__D_0__D_1__D_2__D_3,  temp2_part0_1_part3_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       temp2_part0_1_part3_1__D_0__D_1__D_2__D_3,
			  temp2_part0_1_part3B__D_0__D_1__D_2__D_3, temp2_part0_1_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );
			RepartitionDown
			( temp1_part1_1_part2T__D_0__D_1__D_2__D_3,  temp1_part1_1_part2_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       temp1_part1_1_part2_1__D_0__D_1__D_2__D_3,
			  temp1_part1_1_part2B__D_0__D_1__D_2__D_3, temp1_part1_1_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );

			   // temp2_part0_1_part3_1[D0,*,D2,D3] <- temp2_part0_1_part3_1[D0,D1,D2,D3]
			temp2_part0_1_part3_1__D_0__S__D_2__D_3.AlignModesWith( modes_1_2, W_bmje__D_0__D_1__D_2__D_3, modes_3_1 );
			temp2_part0_1_part3_1__D_0__S__D_2__D_3.AllGatherRedistFrom( temp2_part0_1_part3_1__D_0__D_1__D_2__D_3, modes_1 );
			   // temp2_part0_1_part3_1[D0,*,D21,D3] <- temp2_part0_1_part3_1[D0,*,D2,D3]
			temp2_part0_1_part3_1__D_0__S__D_2_1__D_3.AlignModesWith( modes_1_2, W_bmje__D_0__D_1__D_2__D_3, modes_3_1 );
			temp2_part0_1_part3_1__D_0__S__D_2_1__D_3.LocalRedistFrom( temp2_part0_1_part3_1__D_0__S__D_2__D_3 );
			temp2_part0_1_part3_1__D_0__S__D_2__D_3.EmptyData();
			   // temp2_part0_1_part3_1[D0,D3,D21,*] <- temp2_part0_1_part3_1[D0,*,D21,D3]
			temp2_part0_1_part3_1__D_0__D_3__D_2_1__S.AlignModesWith( modes_1_2, W_bmje__D_0__D_1__D_2__D_3, modes_3_1 );
			temp2_part0_1_part3_1__D_0__D_3__D_2_1__S.AllToAllRedistFrom( temp2_part0_1_part3_1__D_0__S__D_2_1__D_3, modes_3 );
			temp2_part0_1_part3_1__D_0__S__D_2_1__D_3.EmptyData();
			   // temp1_part1_1_part2_1[D0,D1,D3,D2] <- temp1_part1_1_part2_1[D0,D1,D2,D3]
			temp1_part1_1_part2_1__D_0__D_1__D_3__D_2.AlignModesWith( modes_0_3, W_bmje__D_0__D_1__D_2__D_3, modes_0_2 );
			temp1_part1_1_part2_1__D_0__D_1__D_3__D_2.AllToAllRedistFrom( temp1_part1_1_part2_1__D_0__D_1__D_2__D_3, modes_2_3 );
			   // temp1_part1_1_part2_1[D0,*,*,D2] <- temp1_part1_1_part2_1[D0,D1,D3,D2]
			temp1_part1_1_part2_1_perm1203__S__S__D_0__D_2.AlignModesWith( modes_0_3, W_bmje__D_0__D_1__D_2__D_3, modes_0_2 );
			temp1_part1_1_part2_1_perm1203__S__S__D_0__D_2.AllGatherRedistFrom( temp1_part1_1_part2_1__D_0__D_1__D_3__D_2, modes_1_3 );
			temp1_part1_1_part2_1__D_0__D_1__D_3__D_2.EmptyData();
			   // temp2_part0_1_part3_1[D0,D3,D12,*] <- temp2_part0_1_part3_1[D0,D3,D21,*]
			temp2_part0_1_part3_1__D_0__D_3__D_1_2__S.AlignModesWith( modes_1_2, W_bmje__D_0__D_1__D_2__D_3, modes_3_1 );
			temp2_part0_1_part3_1__D_0__D_3__D_1_2__S.AllToAllRedistFrom( temp2_part0_1_part3_1__D_0__D_3__D_2_1__S, modes_1_2 );
			temp2_part0_1_part3_1__D_0__D_3__D_2_1__S.EmptyData();
			   // temp2_part0_1_part3_1[*,D3,D1,*] <- temp2_part0_1_part3_1[D0,D3,D12,*]
			temp2_part0_1_part3_1_perm1203__D_3__D_1__S__S.AlignModesWith( modes_1_2, W_bmje__D_0__D_1__D_2__D_3, modes_3_1 );
			temp2_part0_1_part3_1_perm1203__D_3__D_1__S__S.AllGatherRedistFrom( temp2_part0_1_part3_1__D_0__D_3__D_1_2__S, modes_0_2 );
			   // 1.0 * temp2_part0_1_part3_1[*,D3,D1,*]_emfn * temp1_part1_1_part2_1[D0,*,*,D2]_fnbj + 1.0 * W_bmje[D0,D1,D2,D3]_embj
			LocalContractAndLocalEliminate(1.0, temp2_part0_1_part3_1_perm1203__D_3__D_1__S__S.LockedTensor(), indices_emfn, false,
				temp1_part1_1_part2_1_perm1203__S__S__D_0__D_2.LockedTensor(), indices_fnbj, false,
				1.0, W_bmje_perm3102__D_3__D_1__D_0__D_2.Tensor(), indices_embj, false);
			temp1_part1_1_part2_1_perm1203__S__S__D_0__D_2.EmptyData();
			temp2_part0_1_part3_1_perm1203__D_3__D_1__S__S.EmptyData();
			temp2_part0_1_part3_1__D_0__D_3__D_1_2__S.EmptyData();

			SlidePartitionDown
			( temp2_part0_1_part3T__D_0__D_1__D_2__D_3,  temp2_part0_1_part3_0__D_0__D_1__D_2__D_3,
			       temp2_part0_1_part3_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  temp2_part0_1_part3B__D_0__D_1__D_2__D_3, temp2_part0_1_part3_2__D_0__D_1__D_2__D_3, 3 );
			SlidePartitionDown
			( temp1_part1_1_part2T__D_0__D_1__D_2__D_3,  temp1_part1_1_part2_0__D_0__D_1__D_2__D_3,
			       temp1_part1_1_part2_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  temp1_part1_1_part2B__D_0__D_1__D_2__D_3, temp1_part1_1_part2_2__D_0__D_1__D_2__D_3, 2 );

		}
		//****

		SlidePartitionDown
		( temp2_part0T__D_0__D_1__D_2__D_3,  temp2_part0_0__D_0__D_1__D_2__D_3,
		       temp2_part0_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  temp2_part0B__D_0__D_1__D_2__D_3, temp2_part0_2__D_0__D_1__D_2__D_3, 0 );
		SlidePartitionDown
		( temp1_part1T__D_0__D_1__D_2__D_3,  temp1_part1_0__D_0__D_1__D_2__D_3,
		       temp1_part1_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  temp1_part1B__D_0__D_1__D_2__D_3, temp1_part1_2__D_0__D_1__D_2__D_3, 1 );

	}
	temp2__D_0__D_1__D_2__D_3.EmptyData();
	temp1__D_0__D_1__D_2__D_3.EmptyData();
	temp2__D_0__D_1__D_2__D_3.EmptyData();
	temp1__D_0__D_1__D_2__D_3.EmptyData();
	//****
	Permute( W_bmje_perm3102__D_3__D_1__D_0__D_2, W_bmje__D_0__D_1__D_2__D_3 );
	W_bmje_perm3102__D_3__D_1__D_0__D_2.EmptyData();
	Print(W_bmje__D_0__D_1__D_2__D_3, "Wafterv");
	//**** (out of 1)
	//**** Is real	0 shadows
		//Outputs:
		//  W_bmje__D_0__D_1__D_2__D_3
	PartitionDown(temp4__D_0__D_1__D_2__D_3, temp4_part1T__D_0__D_1__D_2__D_3, temp4_part1B__D_0__D_1__D_2__D_3, 1, 0);
	PartitionDown(W_bmje__D_0__D_1__D_2__D_3, W_bmje_part1T__D_0__D_1__D_2__D_3, W_bmje_part1B__D_0__D_1__D_2__D_3, 1, 0);
	PartitionDown(u_mnje__D_0__D_1__D_2__D_3, u_mnje_part0T__D_0__D_1__D_2__D_3, u_mnje_part0B__D_0__D_1__D_2__D_3, 0, 0);
	PartitionDown(u_mnje__D_0__D_1__D_2__D_3, u_mnje_part1T__D_0__D_1__D_2__D_3, u_mnje_part1B__D_0__D_1__D_2__D_3, 1, 0);
	while(W_bmje_part1T__D_0__D_1__D_2__D_3.Dimension(1) < W_bmje__D_0__D_1__D_2__D_3.Dimension(1))
	{
		RepartitionDown
		( temp4_part1T__D_0__D_1__D_2__D_3,  temp4_part1_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       temp4_part1_1__D_0__D_1__D_2__D_3,
		  temp4_part1B__D_0__D_1__D_2__D_3, temp4_part1_2__D_0__D_1__D_2__D_3, 1, blkSize );
		RepartitionDown
		( W_bmje_part1T__D_0__D_1__D_2__D_3,  W_bmje_part1_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       W_bmje_part1_1__D_0__D_1__D_2__D_3,
		  W_bmje_part1B__D_0__D_1__D_2__D_3, W_bmje_part1_2__D_0__D_1__D_2__D_3, 1, blkSize );
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

		Permute( W_bmje_part1_1__D_0__D_1__D_2__D_3, W_bmje_part1_1_perm1230__D_1__D_2__D_3__D_0 );
		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  W_bmje_part1_1_perm1230__D_1__D_2__D_3__D_0
		PartitionDown(t_fj__D_0_1__D_2_3, t_fj_part1T__D_0_1__D_2_3, t_fj_part1B__D_0_1__D_2_3, 1, 0);
		PartitionDown(u_mnje_part0_1__D_0__D_1__D_2__D_3, u_mnje_part0_1_part1T__D_0__D_1__D_2__D_3, u_mnje_part0_1_part1B__D_0__D_1__D_2__D_3, 1, 0);
		PartitionDown(u_mnje_part1_1__D_0__D_1__D_2__D_3, u_mnje_part1_1_part0T__D_0__D_1__D_2__D_3, u_mnje_part1_1_part0B__D_0__D_1__D_2__D_3, 0, 0);
		while(t_fj_part1T__D_0_1__D_2_3.Dimension(1) < t_fj__D_0_1__D_2_3.Dimension(1))
		{
			RepartitionDown
			( t_fj_part1T__D_0_1__D_2_3,  t_fj_part1_0__D_0_1__D_2_3,
			  /**/ /**/
			       t_fj_part1_1__D_0_1__D_2_3,
			  t_fj_part1B__D_0_1__D_2_3, t_fj_part1_2__D_0_1__D_2_3, 1, blkSize );
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

			   // u_mnje_part1_1_part0_1[D1,D0,D2,D3] <- u_mnje_part1_1_part0_1[D0,D1,D2,D3]
			u_mnje_part1_1_part0_1__D_1__D_0__D_2__D_3.AlignModesWith( modes_0_1_2_3, u_mnje_part0_1_part1_1__D_0__D_1__D_2__D_3, modes_1_0_2_3 );
			u_mnje_part1_1_part0_1__D_1__D_0__D_2__D_3.AllToAllRedistFrom( u_mnje_part1_1_part0_1__D_0__D_1__D_2__D_3, modes_0_1 );
			tempShape = u_mnje_part0_1_part1_1__D_0__D_1__D_2__D_3.Shape();
			temp3_part0_1_part1_1__D_0__D_1__D_2__D_3.ResizeTo( tempShape );
			   // t_fj_part1_1[D0,*] <- t_fj_part1_1[D01,D23]
			t_fj_part1_1_perm10__S__D_0.AlignModesWith( modes_0, W_bmje_part1_1__D_0__D_1__D_2__D_3, modes_0 );
			t_fj_part1_1_perm10__S__D_0.AllGatherRedistFrom( t_fj_part1_1__D_0_1__D_2_3, modes_1_2_3 );
			YAxpPx( -1.0, u_mnje_part0_1_part1_1__D_0__D_1__D_2__D_3, 2.0, u_mnje_part1_1_part0_1__D_1__D_0__D_2__D_3, perm_1_0_2_3, temp3_part0_1_part1_1__D_0__D_1__D_2__D_3 );
			u_mnje_part1_1_part0_1__D_1__D_0__D_2__D_3.EmptyData();
			   // temp3_part0_1_part1_1[D1,*,D2,D3] <- temp3_part0_1_part1_1[D0,D1,D2,D3]
			temp3_part0_1_part1_1_perm0231__D_1__D_2__D_3__S.AlignModesWith( modes_0_2_3, W_bmje_part1_1__D_0__D_1__D_2__D_3, modes_1_2_3 );
			temp3_part0_1_part1_1_perm0231__D_1__D_2__D_3__S.AllToAllRedistFrom( temp3_part0_1_part1_1__D_0__D_1__D_2__D_3, modes_0_1 );
			   // -1.0 * temp3_part0_1_part1_1[D1,*,D2,D3]_mjen * t_fj_part1_1[D0,*]_nb + 1.0 * W_bmje_part1_1[D0,D1,D2,D3]_mjeb
			LocalContractAndLocalEliminate(-1.0, temp3_part0_1_part1_1_perm0231__D_1__D_2__D_3__S.LockedTensor(), indices_mjen, false,
				t_fj_part1_1_perm10__S__D_0.LockedTensor(), indices_nb, false,
				1.0, W_bmje_part1_1_perm1230__D_1__D_2__D_3__D_0.Tensor(), indices_mjeb, false);
			t_fj_part1_1_perm10__S__D_0.EmptyData();
			temp3_part0_1_part1_1_perm0231__D_1__D_2__D_3__S.EmptyData();
			temp3_part0_1_part1_1__D_0__D_1__D_2__D_3.EmptyData();

			SlidePartitionDown
			( t_fj_part1T__D_0_1__D_2_3,  t_fj_part1_0__D_0_1__D_2_3,
			       t_fj_part1_1__D_0_1__D_2_3,
			  /**/ /**/
			  t_fj_part1B__D_0_1__D_2_3, t_fj_part1_2__D_0_1__D_2_3, 1 );
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

		}
		//****
		Permute( W_bmje_part1_1_perm1230__D_1__D_2__D_3__D_0, W_bmje_part1_1__D_0__D_1__D_2__D_3 );
		Print(W_bmje__D_0__D_1__D_2__D_3, "Wafteru");
		W_bmje_part1_1_perm1230__D_1__D_2__D_3__D_0.EmptyData();
		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  W_bmje_part1_1__D_0__D_1__D_2__D_3
		PartitionDown(t_fj__D_0_1__D_2_3, t_fj_part1T__D_0_1__D_2_3, t_fj_part1B__D_0_1__D_2_3, 1, 0);
		PartitionDown(W_bmje_part1_1__D_0__D_1__D_2__D_3, W_bmje_part1_1_part2T__D_0__D_1__D_2__D_3, W_bmje_part1_1_part2B__D_0__D_1__D_2__D_3, 2, 0);
		while(W_bmje_part1_1_part2T__D_0__D_1__D_2__D_3.Dimension(2) < W_bmje_part1_1__D_0__D_1__D_2__D_3.Dimension(2))
		{
			RepartitionDown
			( t_fj_part1T__D_0_1__D_2_3,  t_fj_part1_0__D_0_1__D_2_3,
			  /**/ /**/
			       t_fj_part1_1__D_0_1__D_2_3,
			  t_fj_part1B__D_0_1__D_2_3, t_fj_part1_2__D_0_1__D_2_3, 1, blkSize );
			RepartitionDown
			( W_bmje_part1_1_part2T__D_0__D_1__D_2__D_3,  W_bmje_part1_1_part2_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       W_bmje_part1_1_part2_1__D_0__D_1__D_2__D_3,
			  W_bmje_part1_1_part2B__D_0__D_1__D_2__D_3, W_bmje_part1_1_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );

			tempShape = W_bmje_part1_1_part2_1__D_0__D_1__D_2__D_3.Shape();
			tempShape.push_back( g.Shape()[3] );
			W_bmje_part1_1_part2_1_perm01324__D_0__D_1__D_2__S__D_3.ResizeTo( tempShape );
			tempShape = W_bmje_part1_1_part2_1__D_0__D_1__D_2__D_3.Shape();
			W_bmje_part1_1_part2_1_temp__D_0__D_1__S__D_2_3.ResizeTo( tempShape );
			   // t_fj_part1_1[D3,D2] <- t_fj_part1_1[D01,D23]
			t_fj_part1_1__D_3__D_2.AlignModesWith( modes_0, temp4_part1_1__D_0__D_1__D_2__D_3, modes_3 );
			t_fj_part1_1__D_3__D_2.AllToAllRedistFrom( t_fj_part1_1__D_0_1__D_2_3, modes_0_1_3 );
			   // t_fj_part1_1[D3,*] <- t_fj_part1_1[D3,D2]
			t_fj_part1_1__D_3__S.AlignModesWith( modes_0, temp4_part1_1__D_0__D_1__D_2__D_3, modes_3 );
			t_fj_part1_1__D_3__S.AllGatherRedistFrom( t_fj_part1_1__D_3__D_2, modes_2 );
			   // 1.0 * temp4_part1_1[D0,D1,D2,D3]_bmef * t_fj_part1_1[D3,*]_fj + 0.0 * W_bmje_part1_1_part2_1[D0,D1,*,D2,D3]_bmejf
			LocalContract(1.0, temp4_part1_1__D_0__D_1__D_2__D_3.LockedTensor(), indices_bmef, false,
				t_fj_part1_1__D_3__S.LockedTensor(), indices_fj, false,
				0.0, W_bmje_part1_1_part2_1_perm01324__D_0__D_1__D_2__S__D_3.Tensor(), indices_bmejf, false);
			   // W_bmje_part1_1_part2_1_temp[D0,D1,*,D23] <- W_bmje_part1_1_part2_1[D0,D1,*,D2,D3] (with SumScatter on D3)
			W_bmje_part1_1_part2_1_temp__D_0__D_1__S__D_2_3.ReduceScatterRedistFrom( W_bmje_part1_1_part2_1_perm01324__D_0__D_1__D_2__S__D_3, 4 );
			W_bmje_part1_1_part2_1_perm01324__D_0__D_1__D_2__S__D_3.EmptyData();
			t_fj_part1_1__D_3__S.EmptyData();
			t_fj_part1_1__D_3__D_2.EmptyData();
			   // W_bmje_part1_1_part2_1_temp[D0,D1,D2,D3] <- W_bmje_part1_1_part2_1_temp[D0,D1,*,D23]
			W_bmje_part1_1_part2_1_temp__D_0__D_1__D_2__D_3.AlignModesWith( modes_0_1_2_3, W_bmje_part1_1_part2_1__D_0__D_1__D_2__D_3, modes_0_1_2_3 );
			W_bmje_part1_1_part2_1_temp__D_0__D_1__D_2__D_3.AllToAllRedistFrom( W_bmje_part1_1_part2_1_temp__D_0__D_1__S__D_2_3, modes_2_3 );
			YxpBy( W_bmje_part1_1_part2_1_temp__D_0__D_1__D_2__D_3, 1.0, W_bmje_part1_1_part2_1__D_0__D_1__D_2__D_3 );
			W_bmje_part1_1_part2_1_temp__D_0__D_1__D_2__D_3.EmptyData();
			W_bmje_part1_1_part2_1_temp__D_0__D_1__S__D_2_3.EmptyData();

			SlidePartitionDown
			( t_fj_part1T__D_0_1__D_2_3,  t_fj_part1_0__D_0_1__D_2_3,
			       t_fj_part1_1__D_0_1__D_2_3,
			  /**/ /**/
			  t_fj_part1B__D_0_1__D_2_3, t_fj_part1_2__D_0_1__D_2_3, 1 );
			SlidePartitionDown
			( W_bmje_part1_1_part2T__D_0__D_1__D_2__D_3,  W_bmje_part1_1_part2_0__D_0__D_1__D_2__D_3,
			       W_bmje_part1_1_part2_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  W_bmje_part1_1_part2B__D_0__D_1__D_2__D_3, W_bmje_part1_1_part2_2__D_0__D_1__D_2__D_3, 2 );

		}
		//****

		SlidePartitionDown
		( temp4_part1T__D_0__D_1__D_2__D_3,  temp4_part1_0__D_0__D_1__D_2__D_3,
		       temp4_part1_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  temp4_part1B__D_0__D_1__D_2__D_3, temp4_part1_2__D_0__D_1__D_2__D_3, 1 );
		SlidePartitionDown
		( W_bmje_part1T__D_0__D_1__D_2__D_3,  W_bmje_part1_0__D_0__D_1__D_2__D_3,
		       W_bmje_part1_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  W_bmje_part1B__D_0__D_1__D_2__D_3, W_bmje_part1_2__D_0__D_1__D_2__D_3, 1 );
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

	}
	temp4__D_0__D_1__D_2__D_3.EmptyData();
	t_fj__D_0_1__D_2_3.EmptyData();
	u_mnje__D_0__D_1__D_2__D_3.EmptyData();
	u_mnje__D_0__D_1__D_2__D_3.EmptyData();
	temp4__D_0__D_1__D_2__D_3.EmptyData();
	t_fj__D_0_1__D_2_3.EmptyData();
	u_mnje__D_0__D_1__D_2__D_3.EmptyData();
	u_mnje__D_0__D_1__D_2__D_3.EmptyData();
	//****


t_fj__D_0_1__D_2_3.EmptyData();
r_bmfe__D_0__D_1__D_2__D_3.EmptyData();
temp4__D_0__D_1__D_2__D_3.EmptyData();
u_mnje__D_0__D_1__D_2__D_3.EmptyData();
v_femn__D_0__D_1__D_2__D_3.EmptyData();
temp2__D_0__D_1__D_2__D_3.EmptyData();
T_bfnj__D_0__D_1__D_2__D_3.EmptyData();
Tau_efmn__D_0__D_1__D_2__D_3.EmptyData();
temp1__D_0__D_1__D_2__D_3.EmptyData();
w_bmje__D_0__D_1__D_2__D_3.EmptyData();
x_bmej__D_0__D_1__D_2__D_3.EmptyData();
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
    Print(W_bmje__D_0__D_1__D_2__D_3, "Wafterr");
    diff.ResizeTo(check);
    Diff(check, W_bmje__D_0__D_1__D_2__D_3, diff);
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


