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
#include "rote.hpp"

#ifdef PROFILE
#ifdef BGQ
#include <spi/include/kernel/memory.h>
#endif
#endif

using namespace rote;
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
    Unsigned i;
    const Int commRank = mpi::CommRank(mpi::COMM_WORLD);

//START_DECL
ObjShape tempShape;
TensorDistribution dist__D_0__D_1__D_2__D_3 = rote::StringToTensorDist("[(0),(1),(2),(3)]");
TensorDistribution dist__D_0__D_2 = rote::StringToTensorDist("[(0),(2)]");
TensorDistribution dist__D_1__D_3 = rote::StringToTensorDist("[(1),(3)]");
TensorDistribution dist__D_0_1__D_2_3 = rote::StringToTensorDist("[(0,1),(2,3)]");
TensorDistribution dist__D_1_0__D_3_2 = rote::StringToTensorDist("[(1,0),(3,2)]");
Permutation perm_0_1( 2 );
Permutation perm_0_1_2_3( 4 );
Permutation perm_0_2_1_3 = {0,2,1,3};
ModeArray modes( 0 );
ModeArray modes_0_1( 2 );
modes_0_1[0] = 0;
modes_0_1[1] = 1;
ModeArray modes_0_1_2_3( 4 );
modes_0_1_2_3[0] = 0;
modes_0_1_2_3[1] = 1;
modes_0_1_2_3[2] = 2;
modes_0_1_2_3[3] = 3;
ModeArray modes_0_2( 2 );
modes_0_2[0] = 0;
modes_0_2[1] = 2;
ModeArray modes_1_3( 2 );
modes_1_3[0] = 1;
modes_1_3[1] = 3;
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
DistTensor<double> Tau_efmn_perm0213__D_0__D_2__D_1__D_3( dist__D_0__D_1__D_2__D_3, g );
Tau_efmn_perm0213__D_0__D_2__D_1__D_3.SetLocalPermutation( perm_0_2_1_3 );
	//t_fj[D01,D23]
DistTensor<double> t_fj__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//t_fj[D0,D2]
DistTensor<double> t_fj__D_0__D_2( dist__D_0__D_2, g );
	//t_fj[D10,D32]
DistTensor<double> t_fj__D_1_0__D_3_2( dist__D_1_0__D_3_2, g );
	//t_fj[D1,D3]
DistTensor<double> t_fj__D_1__D_3( dist__D_1__D_3, g );
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

//**** (out of 1)
//**** Is real	0 shadows
	//Outputs:
	//  Tau_efmn__D_0__D_1__D_2__D_3
TensorDistribution dist__S__D_3__D_1__S = rote::StringToTensorDist("[(),(3),(1),()]");
TensorDistribution dist__D_0__S__S__D_2 = rote::StringToTensorDist("[(0),(),(),(2)]");
TensorDistribution dist__D_0__S = rote::StringToTensorDist("[(0),()]");
TensorDistribution dist__D_0__D_1__S__D_2_3 = rote::StringToTensorDist("[(0),(1),(),(2,3)]");
TensorDistribution dist__D_0__D_1__S__D_3_2 = rote::StringToTensorDist("[(0),(1),(),(3,2)]");
TensorDistribution dist__D_0__D_1__S__D_2__D_3 = rote::StringToTensorDist("[(0),(1),(),(2),(3)]");
TensorDistribution dist__D_0__D_1__D_3__D_2 = rote::StringToTensorDist("[(0),(1),(3),(2)]");
TensorDistribution dist__D_0__D_3__D_1__D_2 = rote::StringToTensorDist("[(0),(3),(1),(2)]");
TensorDistribution dist__D_1__S__D_2__D_3 = rote::StringToTensorDist("[(1),(),(2),(3)]");
TensorDistribution dist__D_1__D_0__D_2__D_3 = rote::StringToTensorDist("[(1),(0),(2),(3)]");
TensorDistribution dist__D_0_1_3__S = rote::StringToTensorDist("[(0,1,3),()]");
TensorDistribution dist__D_3__S = rote::StringToTensorDist("[(3),()]");
TensorDistribution dist__D_0_1__S = rote::StringToTensorDist("[(0,1),()]");
TensorDistribution dist__D_3_0_1__S = rote::StringToTensorDist("[(3,0,1),()]");
Permutation perm_0_1_3_2 = {0,1,3,2};
Permutation perm_0_1_3_2_4 = {0,1,3,2,4};
Permutation perm_0_2_3_1 = {0,2,3,1};
Permutation perm_0_3_1_2 = {0,3,1,2};
Permutation perm_1_0 = {1,0};
Permutation perm_1_0_2_3 = {1,0,2,3};
Permutation perm_1_2_0_3 = {1,2,0,3};
Permutation perm_1_2_3_0 = {1,2,3,0};
Permutation perm_1_3_0_2 = {1,3,0,2};
Permutation perm_2_1_0_3 = {2,1,0,3};
Permutation perm_3_0_1_2 = {3,0,1,2};
ModeArray modes_0( 1 );
modes_0[0] = 0;
ModeArray modes_0_1_2( 3 );
modes_0_1_2[0] = 0;
modes_0_1_2[1] = 1;
modes_0_1_2[2] = 2;
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
IndexArray indices_fj( 2 );
indices_fj[0] = 'f';
indices_fj[1] = 'j';
IndexArray indices_fnbj( 4 );
indices_fnbj[0] = 'f';
indices_fnbj[1] = 'n';
indices_fnbj[2] = 'b';
indices_fnbj[3] = 'j';
IndexArray indices_mebj( 4 );
indices_mebj[0] = 'm';
indices_mebj[1] = 'e';
indices_mebj[2] = 'b';
indices_mebj[3] = 'j';
IndexArray indices_mefn( 4 );
indices_mefn[0] = 'm';
indices_mefn[1] = 'e';
indices_mefn[2] = 'f';
indices_mefn[3] = 'n';
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
	//T_bfnj[D0,D1,D3,D2]
DistTensor<double> T_bfnj__D_0__D_1__D_3__D_2( dist__D_0__D_1__D_3__D_2, g );
	//W_bmje[D0,D1,D2,D3]
DistTensor<double> W_bmje__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_bmje[D0,D1,*,D2,D3]
DistTensor<double> W_bmje_perm01324__D_0__D_1__D_2__S__D_3( dist__D_0__D_1__S__D_2__D_3, g );
W_bmje_perm01324__D_0__D_1__D_2__S__D_3.SetLocalPermutation( perm_0_1_3_2_4 );
DistTensor<double> W_bmje_perm1230__D_1__D_2__D_3__D_0( dist__D_0__D_1__D_2__D_3, g );
W_bmje_perm1230__D_1__D_2__D_3__D_0.SetLocalPermutation( perm_1_2_3_0 );
DistTensor<double> W_bmje_perm1302__D_1__D_3__D_0__D_2( dist__D_0__D_1__D_2__D_3, g );
W_bmje_perm1302__D_1__D_3__D_0__D_2.SetLocalPermutation( perm_1_3_0_2 );
	//W_bmje_temp[D0,D1,D2,D3]
DistTensor<double> W_bmje_temp__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//W_bmje_temp[D0,D1,*,D23]
DistTensor<double> W_bmje_temp__D_0__D_1__S__D_2_3( dist__D_0__D_1__S__D_2_3, g );
	//W_bmje_temp[D0,D1,*,D32]
DistTensor<double> W_bmje_temp__D_0__D_1__S__D_3_2( dist__D_0__D_1__S__D_3_2, g );
	//Wtemp1[D0,D1,D2,D3]
DistTensor<double> Wtemp1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Wtemp1[D0,D1,D3,D2]
DistTensor<double> Wtemp1__D_0__D_1__D_3__D_2( dist__D_0__D_1__D_3__D_2, g );
	//Wtemp1[D0,*,*,D2]
DistTensor<double> Wtemp1_perm1203__S__S__D_0__D_2( dist__D_0__S__S__D_2, g );
Wtemp1_perm1203__S__S__D_0__D_2.SetLocalPermutation( perm_1_2_0_3 );
	//Wtemp2[D0,D1,D2,D3]
DistTensor<double> Wtemp2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Wtemp2[D0,D3,D1,D2]
DistTensor<double> Wtemp2__D_0__D_3__D_1__D_2( dist__D_0__D_3__D_1__D_2, g );
	//Wtemp2[*,D3,D1,*]
DistTensor<double> Wtemp2_perm2103__D_1__D_3__S__S( dist__S__D_3__D_1__S, g );
Wtemp2_perm2103__D_1__D_3__S__S.SetLocalPermutation( perm_2_1_0_3 );
	//Wtemp3[D0,D1,D2,D3]
DistTensor<double> Wtemp3__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Wtemp3[D1,D0,D2,D3]
DistTensor<double> Wtemp3__D_1__D_0__D_2__D_3( dist__D_1__D_0__D_2__D_3, g );
	//Wtemp3[D1,*,D2,D3]
DistTensor<double> Wtemp3_perm0231__D_1__D_2__D_3__S( dist__D_1__S__D_2__D_3, g );
Wtemp3_perm0231__D_1__D_2__D_3__S.SetLocalPermutation( perm_0_2_3_1 );
	//Wtemp4[D0,D1,D2,D3]
DistTensor<double> Wtemp4__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//r_bmfe[D0,D1,D2,D3]
DistTensor<double> r_bmfe__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//r_bmfe[D0,D1,D3,D2]
DistTensor<double> r_bmfe__D_0__D_1__D_3__D_2( dist__D_0__D_1__D_3__D_2, g );
	//t_fj[D013,*]
DistTensor<double> t_fj__D_0_1_3__S( dist__D_0_1_3__S, g );
	//t_fj[D01,*]
DistTensor<double> t_fj__D_0_1__S( dist__D_0_1__S, g );
	//t_fj[D301,*]
DistTensor<double> t_fj__D_3_0_1__S( dist__D_3_0_1__S, g );
	//t_fj[D3,*]
DistTensor<double> t_fj__D_3__S( dist__D_3__S, g );
	//t_fj[D0,*]
DistTensor<double> t_fj_perm10__S__D_0( dist__D_0__S, g );
t_fj_perm10__S__D_0.SetLocalPermutation( perm_1_0 );
	//u_mnje[D0,D1,D2,D3]
DistTensor<double> u_mnje__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//u_mnje[D1,D0,D2,D3]
DistTensor<double> u_mnje__D_1__D_0__D_2__D_3( dist__D_1__D_0__D_2__D_3, g );
	//v_femn[D0,D1,D2,D3]
DistTensor<double> v_femn__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//v_femn[D0,D1,D3,D2]
DistTensor<double> v_femn__D_0__D_1__D_3__D_2( dist__D_0__D_1__D_3__D_2, g );
	//w_bmje[D0,D1,D2,D3]
DistTensor<double> w_bmje__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//x_bmej[D0,D1,D2,D3]
DistTensor<double> x_bmej__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//x_bmej[D0,D1,D3,D2]
DistTensor<double> x_bmej__D_0__D_1__D_3__D_2( dist__D_0__D_1__D_3__D_2, g );
DistTensor<double> Wtemp2_lvl0_part0T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> Wtemp2_lvl0_part0B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> Wtemp1_lvl0_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> Wtemp1_lvl0_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> Wtemp2_lvl0_part0_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> Wtemp2_lvl0_part0_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> Wtemp2_lvl0_part0_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> Wtemp1_lvl0_part1_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> Wtemp1_lvl0_part1_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> Wtemp1_lvl0_part1_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> Wtemp2_lvl0_part0_1_lvl1_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> Wtemp2_lvl0_part0_1_lvl1_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> Wtemp1_lvl0_part1_1_lvl1_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> Wtemp1_lvl0_part1_1_lvl1_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> Wtemp2_lvl0_part0_1_lvl1_part3_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> Wtemp2_lvl0_part0_1_lvl1_part3_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> Wtemp2_lvl0_part0_1_lvl1_part3_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> Wtemp1_lvl0_part1_1_lvl1_part2_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> Wtemp1_lvl0_part1_1_lvl1_part2_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> Wtemp1_lvl0_part1_1_lvl1_part2_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> Wtemp2_lvl0_part0_1_lvl1_part3_1__D_0__D_3__D_1__D_2( dist__D_0__D_3__D_1__D_2, g );
DistTensor<double> Wtemp1_lvl0_part1_1_lvl1_part2_1__D_0__D_1__D_3__D_2( dist__D_0__D_1__D_3__D_2, g );
DistTensor<double> Wtemp1_lvl0_part1_1_lvl1_part2_1_perm1203__S__S__D_0__D_2( dist__D_0__S__S__D_2, g );
Wtemp1_lvl0_part1_1_lvl1_part2_1_perm1203__S__S__D_0__D_2.SetLocalPermutation( perm_1_2_0_3 );
DistTensor<double> Wtemp2_lvl0_part0_1_lvl1_part3_1_perm2103__D_1__D_3__S__S( dist__S__D_3__D_1__S, g );
Wtemp2_lvl0_part0_1_lvl1_part3_1_perm2103__D_1__D_3__S__S.SetLocalPermutation( perm_2_1_0_3 );
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
	//  Wtemp3__D_0__D_1__D_2__D_3
TensorDistribution dist__S__D_2__D_1__S = rote::StringToTensorDist("[(),(2),(1),()]");
TensorDistribution dist__D_0__S__S__D_3 = rote::StringToTensorDist("[(0),(),(),(3)]");
TensorDistribution dist__D_0__D_1__D_2__S__D_3 = rote::StringToTensorDist("[(0),(1),(2),(),(3)]");
TensorDistribution dist__D_0__D_2__D_1__D_3 = rote::StringToTensorDist("[(0),(2),(1),(3)]");
TensorDistribution dist__D_1__S__D_3__D_2 = rote::StringToTensorDist("[(1),(),(3),(2)]");
TensorDistribution dist__D_1__D_0__D_3__D_2 = rote::StringToTensorDist("[(1),(0),(3),(2)]");
Permutation perm_0_1_2_3_4( 5 );
Permutation perm_0_3_2_1 = {0,3,2,1};
ModeArray modes_1_3_2( 3 );
modes_1_3_2[0] = 1;
modes_1_3_2[1] = 3;
modes_1_3_2[2] = 2;
ModeArray modes_2_1( 2 );
modes_2_1[0] = 2;
modes_2_1[1] = 1;
IndexArray indices_mejb( 4 );
indices_mejb[0] = 'm';
indices_mejb[1] = 'e';
indices_mejb[2] = 'j';
indices_mejb[3] = 'b';
IndexArray indices_mejn( 4 );
indices_mejn[0] = 'm';
indices_mejn[1] = 'e';
indices_mejn[2] = 'j';
indices_mejn[3] = 'n';
	//X_bmej[D0,D1,D2,D3]
DistTensor<double> X_bmej__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//X_bmej[D0,D1,D2,*,D3]
DistTensor<double> X_bmej__D_0__D_1__D_2__S__D_3( dist__D_0__D_1__D_2__S__D_3, g );
DistTensor<double> X_bmej_perm1203__D_1__D_2__D_0__D_3( dist__D_0__D_1__D_2__D_3, g );
X_bmej_perm1203__D_1__D_2__D_0__D_3.SetLocalPermutation( perm_1_2_0_3 );
DistTensor<double> X_bmej_perm1230__D_1__D_2__D_3__D_0( dist__D_0__D_1__D_2__D_3, g );
X_bmej_perm1230__D_1__D_2__D_3__D_0.SetLocalPermutation( perm_1_2_3_0 );
	//Xtemp1[D0,D1,D2,D3]
DistTensor<double> Xtemp1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Xtemp1[D0,*,*,D3]
DistTensor<double> Xtemp1_perm1203__S__S__D_0__D_3( dist__D_0__S__S__D_3, g );
Xtemp1_perm1203__S__S__D_0__D_3.SetLocalPermutation( perm_1_2_0_3 );
	//u_mnje[D1,D0,D3,D2]
DistTensor<double> u_mnje__D_1__D_0__D_3__D_2( dist__D_1__D_0__D_3__D_2, g );
	//u_mnje[D1,*,D3,D2]
DistTensor<double> u_mnje_perm0321__D_1__D_2__D_3__S( dist__D_1__S__D_3__D_2, g );
u_mnje_perm0321__D_1__D_2__D_3__S.SetLocalPermutation( perm_0_3_2_1 );
	//v_femn[D0,D2,D1,D3]
DistTensor<double> v_femn__D_0__D_2__D_1__D_3( dist__D_0__D_2__D_1__D_3, g );
	//v_femn[*,D2,D1,*]
DistTensor<double> v_femn_perm2103__D_1__D_2__S__S( dist__S__D_2__D_1__S, g );
v_femn_perm2103__D_1__D_2__S__S.SetLocalPermutation( perm_2_1_0_3 );
DistTensor<double> v_femn_lvl0_part0T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> v_femn_lvl0_part0B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> Xtemp1_lvl0_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> Xtemp1_lvl0_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> v_femn_lvl0_part0_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> v_femn_lvl0_part0_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> v_femn_lvl0_part0_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> Xtemp1_lvl0_part1_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> Xtemp1_lvl0_part1_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> Xtemp1_lvl0_part1_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> v_femn_lvl0_part0_1_lvl1_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> v_femn_lvl0_part0_1_lvl1_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> Xtemp1_lvl0_part1_1_lvl1_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> Xtemp1_lvl0_part1_1_lvl1_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> v_femn_lvl0_part0_1_lvl1_part3_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> v_femn_lvl0_part0_1_lvl1_part3_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> v_femn_lvl0_part0_1_lvl1_part3_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> Xtemp1_lvl0_part1_1_lvl1_part2_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> Xtemp1_lvl0_part1_1_lvl1_part2_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> Xtemp1_lvl0_part1_1_lvl1_part2_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> v_femn_lvl0_part0_1_lvl1_part3_1_perm2103__D_1__D_2__S__S( dist__S__D_2__D_1__S, g );
v_femn_lvl0_part0_1_lvl1_part3_1_perm2103__D_1__D_2__S__S.SetLocalPermutation( perm_2_1_0_3 );
DistTensor<double> Xtemp1_lvl0_part1_1_lvl1_part2_1_perm1203__S__S__D_0__D_3( dist__D_0__S__S__D_3, g );
Xtemp1_lvl0_part1_1_lvl1_part2_1_perm1203__S__S__D_0__D_3.SetLocalPermutation( perm_1_2_0_3 );
DistTensor<double> v_femn_lvl0_part0_1_lvl1_part3_1__D_0__D_2__D_1__D_3( dist__D_0__D_2__D_1__D_3, g );
// X_bmej has 4 dims
ObjShape X_bmej__D_0__D_1__D_2__D_3_tempShape( 4 );
X_bmej__D_0__D_1__D_2__D_3_tempShape[ 0 ] = n_v;
X_bmej__D_0__D_1__D_2__D_3_tempShape[ 1 ] = n_o;
X_bmej__D_0__D_1__D_2__D_3_tempShape[ 2 ] = n_v;
X_bmej__D_0__D_1__D_2__D_3_tempShape[ 3 ] = n_o;
X_bmej__D_0__D_1__D_2__D_3.ResizeTo( X_bmej__D_0__D_1__D_2__D_3_tempShape );
MakeUniform( X_bmej__D_0__D_1__D_2__D_3 );
	//  X_bmej__D_0__D_1__D_2__D_3
TensorDistribution dist__D_2_0__D_3__S__D_1 = rote::StringToTensorDist("[(2,0),(3),(),(1)]");
TensorDistribution dist__D_2__D_3__S__D_1__D_0 = rote::StringToTensorDist("[(2),(3),(),(1),(0)]");
TensorDistribution dist__D_0_2__D_1__S__D_3 = rote::StringToTensorDist("[(0,2),(1),(),(3)]");
Permutation perm_2_3_1_0 = {2,3,1,0};
ModeArray modes_2_0_3_1( 4 );
modes_2_0_3_1[0] = 2;
modes_2_0_3_1[1] = 0;
modes_2_0_3_1[2] = 3;
modes_2_0_3_1[3] = 1;
ModeArray modes_2_3_1( 3 );
modes_2_3_1[0] = 2;
modes_2_3_1[1] = 3;
modes_2_3_1[2] = 1;
IndexArray indices_fi( 2 );
indices_fi[0] = 'f';
indices_fi[1] = 'i';
IndexArray indices_mnef( 4 );
indices_mnef[0] = 'm';
indices_mnef[1] = 'n';
indices_mnef[2] = 'e';
indices_mnef[3] = 'f';
IndexArray indices_mneif( 5 );
indices_mneif[0] = 'm';
indices_mneif[1] = 'n';
indices_mneif[2] = 'e';
indices_mneif[3] = 'i';
indices_mneif[4] = 'f';
	//U_mnie[D0,D1,D2,D3]
DistTensor<double> U_mnie__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//U_mnie[D2,D3,*,D1,D0]
DistTensor<double> U_mnie_perm01324__D_2__D_3__D_1__S__D_0( dist__D_2__D_3__S__D_1__D_0, g );
U_mnie_perm01324__D_2__D_3__D_1__S__D_0.SetLocalPermutation( perm_0_1_3_2_4 );
	//U_mnie_temp[D02,D1,*,D3]
DistTensor<double> U_mnie_temp__D_0_2__D_1__S__D_3( dist__D_0_2__D_1__S__D_3, g );
	//U_mnie_temp[D0,D1,D2,D3]
DistTensor<double> U_mnie_temp__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//U_mnie_temp[D20,D3,*,D1]
DistTensor<double> U_mnie_temp__D_2_0__D_3__S__D_1( dist__D_2_0__D_3__S__D_1, g );
DistTensor<double> t_fj__D_0__S( dist__D_0__S, g );
DistTensor<double> v_femn_perm2310__D_2__D_3__D_1__D_0( dist__D_0__D_1__D_2__D_3, g );
v_femn_perm2310__D_2__D_3__D_1__D_0.SetLocalPermutation( perm_2_3_1_0 );
// U_mnie has 4 dims
ObjShape U_mnie__D_0__D_1__D_2__D_3_tempShape( 4 );
U_mnie__D_0__D_1__D_2__D_3_tempShape[ 0 ] = n_o;
U_mnie__D_0__D_1__D_2__D_3_tempShape[ 1 ] = n_o;
U_mnie__D_0__D_1__D_2__D_3_tempShape[ 2 ] = n_o;
U_mnie__D_0__D_1__D_2__D_3_tempShape[ 3 ] = n_v;
U_mnie__D_0__D_1__D_2__D_3.ResizeTo( U_mnie__D_0__D_1__D_2__D_3_tempShape );
MakeUniform( U_mnie__D_0__D_1__D_2__D_3 );
	//  U_mnie__D_0__D_1__D_2__D_3
TensorDistribution dist__S__S__D_2__D_3__D_0__D_1 = rote::StringToTensorDist("[(),(),(2),(3),(0),(1)]");
TensorDistribution dist__D_0__D_1__S__S = rote::StringToTensorDist("[(0),(1),(),()]");
TensorDistribution dist__D_0_1_3__D_2 = rote::StringToTensorDist("[(0,1,3),(2)]");
TensorDistribution dist__D_0_1__D_2 = rote::StringToTensorDist("[(0,1),(2)]");
TensorDistribution dist__D_3_0_1__D_2 = rote::StringToTensorDist("[(3,0,1),(2)]");
Permutation perm_0_1_2_3_4_5( 6 );
Permutation perm_1_0_3_2 = {1,0,3,2};
Permutation perm_2_3_0_1 = {2,3,0,1};
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
	//Q_mnij[D0,D1,D2,*,D3]
DistTensor<double> Q_mnij__D_0__D_1__D_2__S__D_3( dist__D_0__D_1__D_2__S__D_3, g );
	//Q_mnij[*,*,D2,D3,D0,D1]
DistTensor<double> Q_mnij__S__S__D_2__D_3__D_0__D_1( dist__S__S__D_2__D_3__D_0__D_1, g );
	//Qtemp1[D0,D1,D2,D3]
DistTensor<double> Qtemp1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Qtemp1[D1,D0,D3,D2]
DistTensor<double> Qtemp1__D_1__D_0__D_3__D_2( dist__D_1__D_0__D_3__D_2, g );
	//q_mnij[D0,D1,D2,D3]
DistTensor<double> q_mnij__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//t_fj[D013,D2]
DistTensor<double> t_fj__D_0_1_3__D_2( dist__D_0_1_3__D_2, g );
	//t_fj[D01,D2]
DistTensor<double> t_fj__D_0_1__D_2( dist__D_0_1__D_2, g );
	//t_fj[D301,D2]
DistTensor<double> t_fj__D_3_0_1__D_2( dist__D_3_0_1__D_2, g );
	//v_femn[D0,D1,*,*]
DistTensor<double> v_femn_perm2301__S__S__D_0__D_1( dist__D_0__D_1__S__S, g );
v_femn_perm2301__S__S__D_0__D_1.SetLocalPermutation( perm_2_3_0_1 );
DistTensor<double> v_femn_lvl0_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> v_femn_lvl0_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> Q_mnij_lvl0_part0T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> Q_mnij_lvl0_part0B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> v_femn_lvl0_part2_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> v_femn_lvl0_part2_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> v_femn_lvl0_part2_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> Q_mnij_lvl0_part0_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> Q_mnij_lvl0_part0_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> Q_mnij_lvl0_part0_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> v_femn_lvl0_part2_1_lvl1_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> v_femn_lvl0_part2_1_lvl1_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> Q_mnij_lvl0_part0_1_lvl1_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> Q_mnij_lvl0_part0_1_lvl1_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> v_femn_lvl0_part2_1_lvl1_part3_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> v_femn_lvl0_part2_1_lvl1_part3_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> v_femn_lvl0_part2_1_lvl1_part3_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> Q_mnij_lvl0_part0_1_lvl1_part1_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> Q_mnij_lvl0_part0_1_lvl1_part1_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> Q_mnij_lvl0_part0_1_lvl1_part1_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> Q_mnij_lvl0_part0_1_lvl1_part1_1__S__S__D_2__D_3__D_0__D_1( dist__S__S__D_2__D_3__D_0__D_1, g );
DistTensor<double> v_femn_lvl0_part2_1_lvl1_part3_1_perm2301__S__S__D_0__D_1( dist__D_0__D_1__S__S, g );
v_femn_lvl0_part2_1_lvl1_part3_1_perm2301__S__S__D_0__D_1.SetLocalPermutation( perm_2_3_0_1 );
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
	//  Qtemp1__D_0__D_1__D_2__D_3
TensorDistribution dist__S__S__D_1_2__D_0_3 = rote::StringToTensorDist("[(),(),(1,2),(0,3)]");
TensorDistribution dist__S__S__D_2_1__D_3_0 = rote::StringToTensorDist("[(),(),(2,1),(3,0)]");
TensorDistribution dist__S__S__D_1__D_0__D_2__D_3 = rote::StringToTensorDist("[(),(),(1),(0),(2),(3)]");
TensorDistribution dist__S__D_1__D_2__D_3_0 = rote::StringToTensorDist("[(),(1),(2),(3,0)]");
TensorDistribution dist__S__D_2__D_1__D_0__D_3 = rote::StringToTensorDist("[(),(2),(1),(0),(3)]");
TensorDistribution dist__S__D_2__D_1__D_0_3 = rote::StringToTensorDist("[(),(2),(1),(0,3)]");
TensorDistribution dist__D_0__S__D_2_1__D_3 = rote::StringToTensorDist("[(0),(),(2,1),(3)]");
TensorDistribution dist__D_0_1_2__S = rote::StringToTensorDist("[(0,1,2),()]");
TensorDistribution dist__D_2__S = rote::StringToTensorDist("[(2),()]");
TensorDistribution dist__D_2__D_3__S__S = rote::StringToTensorDist("[(2),(3),(),()]");
TensorDistribution dist__D_2__D_3__D_0__D_1 = rote::StringToTensorDist("[(2),(3),(0),(1)]");
TensorDistribution dist__D_3__S__D_1_2__D_0 = rote::StringToTensorDist("[(3),(),(1,2),(0)]");
TensorDistribution dist__D_3__S__D_1__D_0__D_2 = rote::StringToTensorDist("[(3),(),(1),(0),(2)]");
TensorDistribution dist__D_2_0_1__S = rote::StringToTensorDist("[(2,0,1),()]");
Permutation perm_0_2_3_1_4 = {0,2,3,1,4};
Permutation perm_3_1_0_2 = {3,1,0,2};
Permutation perm_3_2_1_0_4 = {3,2,1,0,4};
Permutation perm_3_2_1_0_4_5 = {3,2,1,0,4,5};
ModeArray modes_0_2_1_3( 4 );
modes_0_2_1_3[0] = 0;
modes_0_2_1_3[1] = 2;
modes_0_2_1_3[2] = 1;
modes_0_2_1_3[3] = 3;
ModeArray modes_1_0( 2 );
modes_1_0[0] = 1;
modes_1_0[1] = 0;
ModeArray modes_1_2_0_3( 4 );
modes_1_2_0_3[0] = 1;
modes_1_2_0_3[1] = 2;
modes_1_2_0_3[2] = 0;
modes_1_2_0_3[3] = 3;
ModeArray modes_2_1_0( 3 );
modes_2_1_0[0] = 2;
modes_2_1_0[1] = 1;
modes_2_1_0[2] = 0;
ModeArray modes_2_1_0_3( 4 );
modes_2_1_0_3[0] = 2;
modes_2_1_0_3[1] = 1;
modes_2_1_0_3[2] = 0;
modes_2_1_0_3[3] = 3;
ModeArray modes_3_0_1_2( 4 );
modes_3_0_1_2[0] = 3;
modes_3_0_1_2[1] = 0;
modes_3_0_1_2[2] = 1;
modes_3_0_1_2[3] = 2;
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
IndexArray indices_ei( 2 );
indices_ei[0] = 'e';
indices_ei[1] = 'i';
IndexArray indices_jmbe( 4 );
indices_jmbe[0] = 'j';
indices_jmbe[1] = 'm';
indices_jmbe[2] = 'b';
indices_jmbe[3] = 'e';
IndexArray indices_jmbie( 5 );
indices_jmbie[0] = 'j';
indices_jmbie[1] = 'm';
indices_jmbie[2] = 'b';
indices_jmbie[3] = 'i';
indices_jmbie[4] = 'e';
	//P_jimb[D0,D1,D2,D3]
DistTensor<double> P_jimb__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//P_jimb[D3,*,D1,D0,D2]
DistTensor<double> P_jimb_perm02314__D_3__D_1__D_0__S__D_2( dist__D_3__S__D_1__D_0__D_2, g );
P_jimb_perm02314__D_3__D_1__D_0__S__D_2.SetLocalPermutation( perm_0_2_3_1_4 );
	//P_jimb[*,*,D1,D0,D2,D3]
DistTensor<double> P_jimb_perm321045__D_0__D_1__S__S__D_2__D_3( dist__S__S__D_1__D_0__D_2__D_3, g );
P_jimb_perm321045__D_0__D_1__S__S__D_2__D_3.SetLocalPermutation( perm_3_2_1_0_4_5 );
	//P_jimb[*,D2,D1,D0,D3]
DistTensor<double> P_jimb_perm32104__D_0__D_1__D_2__S__D_3( dist__S__D_2__D_1__D_0__D_3, g );
P_jimb_perm32104__D_0__D_1__D_2__S__D_3.SetLocalPermutation( perm_3_2_1_0_4 );
	//P_jimb_temp[D0,D1,D2,D3]
DistTensor<double> P_jimb_temp__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//P_jimb_temp[D0,*,D21,D3]
DistTensor<double> P_jimb_temp__D_0__S__D_2_1__D_3( dist__D_0__S__D_2_1__D_3, g );
	//P_jimb_temp[D3,*,D12,D0]
DistTensor<double> P_jimb_temp__D_3__S__D_1_2__D_0( dist__D_3__S__D_1_2__D_0, g );
	//P_jimb_temp[*,D1,D2,D30]
DistTensor<double> P_jimb_temp__S__D_1__D_2__D_3_0( dist__S__D_1__D_2__D_3_0, g );
	//P_jimb_temp[*,D2,D1,D03]
DistTensor<double> P_jimb_temp__S__D_2__D_1__D_0_3( dist__S__D_2__D_1__D_0_3, g );
	//P_jimb_temp[*,*,D12,D03]
DistTensor<double> P_jimb_temp__S__S__D_1_2__D_0_3( dist__S__S__D_1_2__D_0_3, g );
	//P_jimb_temp[*,*,D21,D30]
DistTensor<double> P_jimb_temp__S__S__D_2_1__D_3_0( dist__S__S__D_2_1__D_3_0, g );
	//Tau_efmn[D2,D3,D0,D1]
DistTensor<double> Tau_efmn__D_2__D_3__D_0__D_1( dist__D_2__D_3__D_0__D_1, g );
	//Tau_efmn[D2,D3,*,*]
DistTensor<double> Tau_efmn__D_2__D_3__S__S( dist__D_2__D_3__S__S, g );
	//t_fj[D012,*]
DistTensor<double> t_fj__D_0_1_2__S( dist__D_0_1_2__S, g );
	//t_fj[D201,*]
DistTensor<double> t_fj__D_2_0_1__S( dist__D_2_0_1__S, g );
	//t_fj[D2,*]
DistTensor<double> t_fj__D_2__S( dist__D_2__S, g );
DistTensor<double> x_bmej_perm3102__D_3__D_1__D_0__D_2( dist__D_0__D_1__D_2__D_3, g );
x_bmej_perm3102__D_3__D_1__D_0__D_2.SetLocalPermutation( perm_3_1_0_2 );
DistTensor<double> Tau_efmn_lvl0_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> Tau_efmn_lvl0_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> P_jimb_lvl0_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> P_jimb_lvl0_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> Tau_efmn_lvl0_part2_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> Tau_efmn_lvl0_part2_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> Tau_efmn_lvl0_part2_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> P_jimb_lvl0_part1_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> P_jimb_lvl0_part1_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> P_jimb_lvl0_part1_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> Tau_efmn_lvl0_part2_1_lvl1_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> Tau_efmn_lvl0_part2_1_lvl1_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> P_jimb_lvl0_part1_1_lvl1_part0T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> P_jimb_lvl0_part1_1_lvl1_part0B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> Tau_efmn_lvl0_part2_1_lvl1_part3_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> Tau_efmn_lvl0_part2_1_lvl1_part3_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> Tau_efmn_lvl0_part2_1_lvl1_part3_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> P_jimb_lvl0_part1_1_lvl1_part0_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> P_jimb_lvl0_part1_1_lvl1_part0_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> P_jimb_lvl0_part1_1_lvl1_part0_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> P_jimb_temp_lvl0_part1_1_lvl1_part0_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> P_jimb_lvl0_part1_1_lvl1_part0_1_perm321045__D_0__D_1__S__S__D_2__D_3( dist__S__S__D_1__D_0__D_2__D_3, g );
P_jimb_lvl0_part1_1_lvl1_part0_1_perm321045__D_0__D_1__S__S__D_2__D_3.SetLocalPermutation( perm_3_2_1_0_4_5 );
DistTensor<double> Tau_efmn_lvl0_part2_1_lvl1_part3_1__D_2__D_3__S__S( dist__D_2__D_3__S__S, g );
DistTensor<double> Tau_efmn_lvl0_part2_1_lvl1_part3_1__D_2__D_3__D_0__D_1( dist__D_2__D_3__D_0__D_1, g );
DistTensor<double> P_jimb_temp_lvl0_part1_1_lvl1_part0_1__S__S__D_1_2__D_0_3( dist__S__S__D_1_2__D_0_3, g );
DistTensor<double> P_jimb_temp_lvl0_part1_1_lvl1_part0_1__S__D_1__D_2__D_3_0( dist__S__D_1__D_2__D_3_0, g );
DistTensor<double> P_jimb_temp_lvl0_part1_1_lvl1_part0_1__S__S__D_2_1__D_3_0( dist__S__S__D_2_1__D_3_0, g );
// P_jimb has 4 dims
ObjShape P_jimb__D_0__D_1__D_2__D_3_tempShape( 4 );
P_jimb__D_0__D_1__D_2__D_3_tempShape[ 0 ] = n_o;
P_jimb__D_0__D_1__D_2__D_3_tempShape[ 1 ] = n_o;
P_jimb__D_0__D_1__D_2__D_3_tempShape[ 2 ] = n_o;
P_jimb__D_0__D_1__D_2__D_3_tempShape[ 3 ] = n_v;
P_jimb__D_0__D_1__D_2__D_3.ResizeTo( P_jimb__D_0__D_1__D_2__D_3_tempShape );
MakeUniform( P_jimb__D_0__D_1__D_2__D_3 );
	//  P_jimb__D_0__D_1__D_2__D_3
TensorDistribution dist__D_1_2__D_3_0 = rote::StringToTensorDist("[(1,2),(3,0)]");
TensorDistribution dist__D_2_1__D_0_3 = rote::StringToTensorDist("[(2,1),(0,3)]");
TensorDistribution dist__D_2__D_0__D_1__D_3 = rote::StringToTensorDist("[(2),(0),(1),(3)]");
Permutation perm_2_0_1_3 = {2,0,1,3};
ModeArray modes_2_0( 2 );
modes_2_0[0] = 2;
modes_2_0[1] = 0;
ModeArray modes_3_2( 2 );
modes_3_2[0] = 3;
modes_3_2[1] = 2;
	//H_me[D01,D23]
DistTensor<double> H_me__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//H_me[D10,D32]
DistTensor<double> H_me__D_1_0__D_3_2( dist__D_1_0__D_3_2, g );
	//H_me[D12,D30]
DistTensor<double> H_me__D_1_2__D_3_0( dist__D_1_2__D_3_0, g );
	//H_me[D21,D03]
DistTensor<double> H_me__D_2_1__D_0_3( dist__D_2_1__D_0_3, g );
	//H_me[D2,D0,D1,D3]
DistTensor<double> H_me__D_2__D_0__D_1__D_3( dist__D_2__D_0__D_1__D_3, g );
	//Htemp1[D0,D1,D2,D3]
DistTensor<double> Htemp1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> Htemp1_perm2013__D_2__D_0__D_1__D_3( dist__D_0__D_1__D_2__D_3, g );
Htemp1_perm2013__D_2__D_0__D_1__D_3.SetLocalPermutation( perm_2_0_1_3 );
// H_me has 2 dims
ObjShape H_me__D_0_1__D_2_3_tempShape( 2 );
H_me__D_0_1__D_2_3_tempShape[ 0 ] = n_o;
H_me__D_0_1__D_2_3_tempShape[ 1 ] = n_v;
H_me__D_0_1__D_2_3.ResizeTo( H_me__D_0_1__D_2_3_tempShape );
MakeUniform( H_me__D_0_1__D_2_3 );
	//  Htemp1__D_0__D_1__D_2__D_3
TensorDistribution dist__S__D_2_3 = rote::StringToTensorDist("[(),(2,3)]");
TensorDistribution dist__S__D_1__D_2__D_3 = rote::StringToTensorDist("[(),(1),(2),(3)]");
TensorDistribution dist__D_0__S__D_1__D_2__D_3 = rote::StringToTensorDist("[(0),(),(1),(2),(3)]");
TensorDistribution dist__D_0__D_2__D_3__D_1 = rote::StringToTensorDist("[(0),(2),(3),(1)]");
TensorDistribution dist__D_3_0__D_1_2 = rote::StringToTensorDist("[(3,0),(1,2)]");
TensorDistribution dist__D_3__D_1 = rote::StringToTensorDist("[(3),(1)]");
TensorDistribution dist__D_0_3__D_2_1 = rote::StringToTensorDist("[(0,3),(2,1)]");
Permutation perm_1_0_2_3_4 = {1,0,2,3,4};
ModeArray modes_0_3_2_1( 4 );
modes_0_3_2_1[0] = 0;
modes_0_3_2_1[1] = 3;
modes_0_3_2_1[2] = 2;
modes_0_3_2_1[3] = 1;
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
	//F_ae[D0,D2,D3,D1]
DistTensor<double> F_ae__D_0__D_2__D_3__D_1( dist__D_0__D_2__D_3__D_1, g );
	//F_ae[D0,*,D1,D2,D3]
DistTensor<double> F_ae_perm10234__S__D_0__D_1__D_2__D_3( dist__D_0__S__D_1__D_2__D_3, g );
F_ae_perm10234__S__D_0__D_1__D_2__D_3.SetLocalPermutation( perm_1_0_2_3_4 );
DistTensor<double> F_ae_perm10__D_2_3__D_0_1( dist__D_0_1__D_2_3, g );
F_ae_perm10__D_2_3__D_0_1.SetLocalPermutation( perm_1_0 );
	//Ftemp1[D0,D1,D2,D3]
DistTensor<double> Ftemp1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Ftemp1[*,D1,D2,D3]
DistTensor<double> Ftemp1__S__D_1__D_2__D_3( dist__S__D_1__D_2__D_3, g );
	//Ftemp2[D0,D1,D2,D3]
DistTensor<double> Ftemp2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> Ftemp2_perm0231__D_0__D_2__D_3__D_1( dist__D_0__D_1__D_2__D_3, g );
Ftemp2_perm0231__D_0__D_2__D_3__D_1.SetLocalPermutation( perm_0_2_3_1 );
	//H_me[*,D23]
DistTensor<double> H_me_perm10__D_2_3__S( dist__S__D_2_3, g );
H_me_perm10__D_2_3__S.SetLocalPermutation( perm_1_0 );
DistTensor<double> T_bfnj_perm1230__D_1__D_2__D_3__D_0( dist__D_0__D_1__D_2__D_3, g );
T_bfnj_perm1230__D_1__D_2__D_3__D_0.SetLocalPermutation( perm_1_2_3_0 );
	//t_fj[D03,D21]
DistTensor<double> t_fj__D_0_3__D_2_1( dist__D_0_3__D_2_1, g );
	//t_fj[D30,D12]
DistTensor<double> t_fj__D_3_0__D_1_2( dist__D_3_0__D_1_2, g );
	//t_fj[D3,D1]
DistTensor<double> t_fj__D_3__D_1( dist__D_3__D_1, g );
DistTensor<double> t_fj_perm10__S__D_0_1( dist__D_0_1__S, g );
t_fj_perm10__S__D_0_1.SetLocalPermutation( perm_1_0 );
// F_ae has 2 dims
ObjShape F_ae__D_0_1__D_2_3_tempShape( 2 );
F_ae__D_0_1__D_2_3_tempShape[ 0 ] = n_v;
F_ae__D_0_1__D_2_3_tempShape[ 1 ] = n_v;
F_ae__D_0_1__D_2_3.ResizeTo( F_ae__D_0_1__D_2_3_tempShape );
MakeUniform( F_ae__D_0_1__D_2_3 );
	//  Ftemp2__D_0__D_1__D_2__D_3
TensorDistribution dist__S__D_2__D_0__D_1__D_3 = rote::StringToTensorDist("[(),(2),(0),(1),(3)]");
TensorDistribution dist__D_0__D_1__S__D_3 = rote::StringToTensorDist("[(0),(1),(),(3)]");
TensorDistribution dist__D_2_3__S = rote::StringToTensorDist("[(2,3),()]");
TensorDistribution dist__D_2_3__D_0_1 = rote::StringToTensorDist("[(2,3),(0,1)]");
TensorDistribution dist__D_0_1__S__D_2_3 = rote::StringToTensorDist("[(0,1),(),(2,3)]");
Permutation perm_0_1_2( 3 );
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
	//G_mi[D01,*,D23]
DistTensor<double> G_mi__D_0_1__S__D_2_3( dist__D_0_1__S__D_2_3, g );
	//G_mi[D0,D2,D3,D1]
DistTensor<double> G_mi__D_0__D_2__D_3__D_1( dist__D_0__D_2__D_3__D_1, g );
	//G_mi[*,D2,D0,D1,D3]
DistTensor<double> G_mi__S__D_2__D_0__D_1__D_3( dist__S__D_2__D_0__D_1__D_3, g );
	//Gtemp1[D0,D1,D2,D3]
DistTensor<double> Gtemp1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Gtemp1[D0,D1,*,D3]
DistTensor<double> Gtemp1_perm2013__S__D_0__D_1__D_3( dist__D_0__D_1__S__D_3, g );
Gtemp1_perm2013__S__D_0__D_1__D_3.SetLocalPermutation( perm_2_0_1_3 );
	//Gtemp2[D0,D1,D2,D3]
DistTensor<double> Gtemp2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> Gtemp2_perm0231__D_0__D_2__D_3__D_1( dist__D_0__D_1__D_2__D_3, g );
Gtemp2_perm0231__D_0__D_2__D_3__D_1.SetLocalPermutation( perm_0_2_3_1 );
DistTensor<double> T_bfnj_perm0132__D_0__D_1__D_3__D_2( dist__D_0__D_1__D_2__D_3, g );
T_bfnj_perm0132__D_0__D_1__D_3__D_2.SetLocalPermutation( perm_0_1_3_2 );
	//t_fj[D23,D01]
DistTensor<double> t_fj__D_2_3__D_0_1( dist__D_2_3__D_0_1, g );
	//t_fj[D23,*]
DistTensor<double> t_fj__D_2_3__S( dist__D_2_3__S, g );
// G_mi has 2 dims
ObjShape G_mi__D_0_1__D_2_3_tempShape( 2 );
G_mi__D_0_1__D_2_3_tempShape[ 0 ] = n_o;
G_mi__D_0_1__D_2_3_tempShape[ 1 ] = n_o;
G_mi__D_0_1__D_2_3.ResizeTo( G_mi__D_0_1__D_2_3_tempShape );
MakeUniform( G_mi__D_0_1__D_2_3 );
	//  Gtemp1__D_0__D_1__D_2__D_3
TensorDistribution dist__D_0__S__D_2__D_3__D_1 = rote::StringToTensorDist("[(0),(),(2),(3),(1)]");
TensorDistribution dist__D_2__D_3__S__D_1 = rote::StringToTensorDist("[(2),(3),(),(1)]");
Permutation perm_0_1_4_2_3 = {0,1,4,2,3};
Permutation perm_1_0_3_4_2 = {1,0,3,4,2};
IndexArray indices_aiem( 4 );
indices_aiem[0] = 'a';
indices_aiem[1] = 'i';
indices_aiem[2] = 'e';
indices_aiem[3] = 'm';
IndexArray indices_aimef( 5 );
indices_aimef[0] = 'a';
indices_aimef[1] = 'i';
indices_aimef[2] = 'm';
indices_aimef[3] = 'e';
indices_aimef[4] = 'f';
IndexArray indices_amef( 4 );
indices_amef[0] = 'a';
indices_amef[1] = 'm';
indices_amef[2] = 'e';
indices_amef[3] = 'f';
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
IndexArray indices_mefi( 4 );
indices_mefi[0] = 'm';
indices_mefi[1] = 'e';
indices_mefi[2] = 'f';
indices_mefi[3] = 'i';
IndexArray indices_mnea( 4 );
indices_mnea[0] = 'm';
indices_mnea[1] = 'n';
indices_mnea[2] = 'e';
indices_mnea[3] = 'a';
	//G_mi[*,D23]
DistTensor<double> G_mi_perm10__D_2_3__S( dist__S__D_2_3, g );
G_mi_perm10__D_2_3__S.SetLocalPermutation( perm_1_0 );
	//H_me[D03,D21]
DistTensor<double> H_me__D_0_3__D_2_1( dist__D_0_3__D_2_1, g );
	//H_me[D30,D12]
DistTensor<double> H_me__D_3_0__D_1_2( dist__D_3_0__D_1_2, g );
	//H_me[D3,D1]
DistTensor<double> H_me_perm10__D_1__D_3( dist__D_3__D_1, g );
H_me_perm10__D_1__D_3.SetLocalPermutation( perm_1_0 );
DistTensor<double> T_bfnj_perm2310__D_2__D_3__D_1__D_0( dist__D_0__D_1__D_2__D_3, g );
T_bfnj_perm2310__D_2__D_3__D_1__D_0.SetLocalPermutation( perm_2_3_1_0 );
	//Tau_efmn[D2,D3,*,D1]
DistTensor<double> Tau_efmn_perm3012__D_1__D_2__D_3__S( dist__D_2__D_3__S__D_1, g );
Tau_efmn_perm3012__D_1__D_2__D_3__S.SetLocalPermutation( perm_3_0_1_2 );
	//U_mnie[D1,D0,D2,D3]
DistTensor<double> U_mnie__D_1__D_0__D_2__D_3( dist__D_1__D_0__D_2__D_3, g );
	//z_ai[D01,D23]
DistTensor<double> z_ai__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//z_ai[D0,D2,D1,D3]
DistTensor<double> z_ai__D_0__D_2__D_1__D_3( dist__D_0__D_2__D_1__D_3, g );
	//z_ai[D0,D2,D3,D1]
DistTensor<double> z_ai__D_0__D_2__D_3__D_1( dist__D_0__D_2__D_3__D_1, g );
	//z_ai[D0,*,D2,D3,D1]
DistTensor<double> z_ai_perm01423__D_0__S__D_1__D_2__D_3( dist__D_0__S__D_2__D_3__D_1, g );
z_ai_perm01423__D_0__S__D_1__D_2__D_3.SetLocalPermutation( perm_0_1_4_2_3 );
	//z_ai[D0,*,D1,D2,D3]
DistTensor<double> z_ai_perm10342__S__D_0__D_2__D_3__D_1( dist__D_0__S__D_1__D_2__D_3, g );
z_ai_perm10342__S__D_0__D_2__D_3__D_1.SetLocalPermutation( perm_1_0_3_4_2 );
DistTensor<double> z_ai_perm10__D_2_3__D_0_1( dist__D_0_1__D_2_3, g );
z_ai_perm10__D_2_3__D_0_1.SetLocalPermutation( perm_1_0 );
	//ztemp2[D0,D1,D2,D3]
DistTensor<double> ztemp2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//ztemp3[D0,D1,D2,D3]
DistTensor<double> ztemp3__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> ztemp3_perm0213__D_0__D_2__D_1__D_3( dist__D_0__D_1__D_2__D_3, g );
ztemp3_perm0213__D_0__D_2__D_1__D_3.SetLocalPermutation( perm_0_2_1_3 );
	//ztemp4[D0,D1,D2,D3]
DistTensor<double> ztemp4__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> ztemp4_perm0231__D_0__D_2__D_3__D_1( dist__D_0__D_1__D_2__D_3, g );
ztemp4_perm0231__D_0__D_2__D_3__D_1.SetLocalPermutation( perm_0_2_3_1 );
	//ztemp5[D0,D1,D2,D3]
DistTensor<double> ztemp5__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//ztemp5[D2,D3,D0,D1]
DistTensor<double> ztemp5__D_2__D_3__D_0__D_1( dist__D_2__D_3__D_0__D_1, g );
	//ztemp5[D2,D3,*,D1]
DistTensor<double> ztemp5_perm2013__S__D_2__D_3__D_1( dist__D_2__D_3__S__D_1, g );
ztemp5_perm2013__S__D_2__D_3__D_1.SetLocalPermutation( perm_2_0_1_3 );
// z_ai has 2 dims
ObjShape z_ai__D_0_1__D_2_3_tempShape( 2 );
z_ai__D_0_1__D_2_3_tempShape[ 0 ] = n_v;
z_ai__D_0_1__D_2_3_tempShape[ 1 ] = n_o;
z_ai__D_0_1__D_2_3.ResizeTo( z_ai__D_0_1__D_2_3_tempShape );
MakeUniform( z_ai__D_0_1__D_2_3 );
	//  ztemp2__D_0__D_1__D_2__D_3
TensorDistribution dist__S__S__D_2__D_3 = rote::StringToTensorDist("[(),(),(2),(3)]");
TensorDistribution dist__S__D_2 = rote::StringToTensorDist("[(),(2)]");
TensorDistribution dist__D_0__S__D_2__S = rote::StringToTensorDist("[(0),(),(2),()]");
TensorDistribution dist__D_0__D_1__S__S__D_2__D_3 = rote::StringToTensorDist("[(0),(1),(),(),(2),(3)]");
TensorDistribution dist__D_1__S__S__D_3 = rote::StringToTensorDist("[(1),(),(),(3)]");
TensorDistribution dist__D_1__S__D_3__S = rote::StringToTensorDist("[(1),(),(3),()]");
Permutation perm_2_0_3_1 = {2,0,3,1};
Permutation perm_3_2_0_1 = {3,2,0,1};
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
IndexArray indices_abje( 4 );
indices_abje[0] = 'a';
indices_abje[1] = 'b';
indices_abje[2] = 'j';
indices_abje[3] = 'e';
IndexArray indices_abjie( 5 );
indices_abjie[0] = 'a';
indices_abjie[1] = 'b';
indices_abjie[2] = 'j';
indices_abjie[3] = 'i';
indices_abjie[4] = 'e';
IndexArray indices_ae( 2 );
indices_ae[0] = 'a';
indices_ae[1] = 'e';
IndexArray indices_bija( 4 );
indices_bija[0] = 'b';
indices_bija[1] = 'i';
indices_bija[2] = 'j';
indices_bija[3] = 'a';
IndexArray indices_bijm( 4 );
indices_bijm[0] = 'b';
indices_bijm[1] = 'i';
indices_bijm[2] = 'j';
indices_bijm[3] = 'm';
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
IndexArray indices_ijmn( 4 );
indices_ijmn[0] = 'i';
indices_ijmn[1] = 'j';
indices_ijmn[2] = 'm';
indices_ijmn[3] = 'n';
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
	//G_mi[*,D2]
DistTensor<double> G_mi_perm10__D_2__S( dist__S__D_2, g );
G_mi_perm10__D_2__S.SetLocalPermutation( perm_1_0 );
	//P_jimb[D2,D3,D0,D1]
DistTensor<double> P_jimb__D_2__D_3__D_0__D_1( dist__D_2__D_3__D_0__D_1, g );
	//P_jimb[D2,D3,*,D1]
DistTensor<double> P_jimb_perm3012__D_1__D_2__D_3__S( dist__D_2__D_3__S__D_1, g );
P_jimb_perm3012__D_1__D_2__D_3__S.SetLocalPermutation( perm_3_0_1_2 );
	//Q_mnij[*,*,D2,D3]
DistTensor<double> Q_mnij_perm2301__D_2__D_3__S__S( dist__S__S__D_2__D_3, g );
Q_mnij_perm2301__D_2__D_3__S__S.SetLocalPermutation( perm_2_3_0_1 );
	//T_bfnj[*,D1,D2,D3]
DistTensor<double> T_bfnj__S__D_1__D_2__D_3( dist__S__D_1__D_2__D_3, g );
	//T_bfnj[D0,D1,*,D3]
DistTensor<double> T_bfnj_perm2013__S__D_0__D_1__D_3( dist__D_0__D_1__S__D_3, g );
T_bfnj_perm2013__S__D_0__D_1__D_3.SetLocalPermutation( perm_2_0_1_3 );
	//T_bfnj[D0,*,*,D2]
DistTensor<double> T_bfnj_perm2103__S__S__D_0__D_2( dist__D_0__S__S__D_2, g );
T_bfnj_perm2103__S__S__D_0__D_2.SetLocalPermutation( perm_2_1_0_3 );
	//Tau_efmn[D0,D1,*,*]
DistTensor<double> Tau_efmn_perm2301__S__S__D_0__D_1( dist__D_0__D_1__S__S, g );
Tau_efmn_perm2301__S__S__D_0__D_1.SetLocalPermutation( perm_2_3_0_1 );
	//W_bmje[D1,D0,D3,D2]
DistTensor<double> W_bmje__D_1__D_0__D_3__D_2( dist__D_1__D_0__D_3__D_2, g );
	//W_bmje[D1,*,D3,*]
DistTensor<double> W_bmje_perm0213__D_1__D_3__S__S( dist__D_1__S__D_3__S, g );
W_bmje_perm0213__D_1__D_3__S__S.SetLocalPermutation( perm_0_2_1_3 );
	//X_bmej[D1,D0,D2,D3]
DistTensor<double> X_bmej__D_1__D_0__D_2__D_3( dist__D_1__D_0__D_2__D_3, g );
	//X_bmej[D1,*,*,D3]
DistTensor<double> X_bmej_perm0312__D_1__D_3__S__S( dist__D_1__S__S__D_3, g );
X_bmej_perm0312__D_1__D_3__S__S.SetLocalPermutation( perm_0_3_1_2 );
	//Z_abij[D0,D1,D2,D3]
DistTensor<double> Z_abij__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Z_abij[D0,D1,*,*,D2,D3]
DistTensor<double> Z_abij__D_0__D_1__S__S__D_2__D_3( dist__D_0__D_1__S__S__D_2__D_3, g );
DistTensor<double> Z_abij_perm2301__D_2__D_3__D_0__D_1( dist__D_0__D_1__D_2__D_3, g );
Z_abij_perm2301__D_2__D_3__D_0__D_1.SetLocalPermutation( perm_2_3_0_1 );
	//Zaccum[D0,D1,D2,D3]
DistTensor<double> Zaccum__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Zaccum[D1,D0,D3,D2]
DistTensor<double> Zaccum__D_1__D_0__D_3__D_2( dist__D_1__D_0__D_3__D_2, g );
	//Zaccum[D2,D3,*,D1,D0]
DistTensor<double> Zaccum_perm01324__D_2__D_3__D_1__S__D_0( dist__D_2__D_3__S__D_1__D_0, g );
Zaccum_perm01324__D_2__D_3__D_1__S__D_0.SetLocalPermutation( perm_0_1_3_2_4 );
DistTensor<double> Zaccum_perm1230__D_1__D_2__D_3__D_0( dist__D_0__D_1__D_2__D_3, g );
Zaccum_perm1230__D_1__D_2__D_3__D_0.SetLocalPermutation( perm_1_2_3_0 );
DistTensor<double> Zaccum_perm1302__D_1__D_3__D_0__D_2( dist__D_0__D_1__D_2__D_3, g );
Zaccum_perm1302__D_1__D_3__D_0__D_2.SetLocalPermutation( perm_1_3_0_2 );
DistTensor<double> Zaccum_perm2013__D_2__D_0__D_1__D_3( dist__D_0__D_1__D_2__D_3, g );
Zaccum_perm2013__D_2__D_0__D_1__D_3.SetLocalPermutation( perm_2_0_1_3 );
	//Zaccum_temp[D02,D1,*,D3]
DistTensor<double> Zaccum_temp__D_0_2__D_1__S__D_3( dist__D_0_2__D_1__S__D_3, g );
	//Zaccum_temp[D0,D1,D2,D3]
DistTensor<double> Zaccum_temp__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Zaccum_temp[D20,D3,*,D1]
DistTensor<double> Zaccum_temp__D_2_0__D_3__S__D_1( dist__D_2_0__D_3__S__D_1, g );
	//Ztemp1[D0,D1,D2,D3]
DistTensor<double> Ztemp1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Ztemp1[D0,D1,D3,D2]
DistTensor<double> Ztemp1__D_0__D_1__D_3__D_2( dist__D_0__D_1__D_3__D_2, g );
DistTensor<double> Ztemp1_perm1302__D_1__D_3__D_0__D_2( dist__D_0__D_1__D_2__D_3, g );
Ztemp1_perm1302__D_1__D_3__D_0__D_2.SetLocalPermutation( perm_1_3_0_2 );
	//Ztemp2[D0,D1,D2,D3]
DistTensor<double> Ztemp2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Ztemp2[D0,*,D2,*]
DistTensor<double> Ztemp2_perm3102__S__S__D_0__D_2( dist__D_0__S__D_2__S, g );
Ztemp2_perm3102__S__S__D_0__D_2.SetLocalPermutation( perm_3_1_0_2 );
DistTensor<double> r_bmfe_perm2310__D_2__D_3__D_1__D_0( dist__D_0__D_1__D_2__D_3, g );
r_bmfe_perm2310__D_2__D_3__D_1__D_0.SetLocalPermutation( perm_2_3_1_0 );
	//y_abef[D0,D1,D2,D3]
DistTensor<double> y_abef__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> X_bmej_lvl0_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> X_bmej_lvl0_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> T_bfnj_lvl0_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> T_bfnj_lvl0_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> X_bmej_lvl0_part2_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> X_bmej_lvl0_part2_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> X_bmej_lvl0_part2_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> T_bfnj_lvl0_part1_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> T_bfnj_lvl0_part1_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> T_bfnj_lvl0_part1_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> X_bmej_lvl0_part2_1_lvl1_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> X_bmej_lvl0_part2_1_lvl1_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> T_bfnj_lvl0_part1_1_lvl1_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> T_bfnj_lvl0_part1_1_lvl1_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> X_bmej_lvl0_part2_1_lvl1_part1_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> X_bmej_lvl0_part2_1_lvl1_part1_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> X_bmej_lvl0_part2_1_lvl1_part1_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> T_bfnj_lvl0_part1_1_lvl1_part2_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> T_bfnj_lvl0_part1_1_lvl1_part2_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> T_bfnj_lvl0_part1_1_lvl1_part2_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> T_bfnj_lvl0_part1_1_lvl1_part2_1__D_0__D_1__D_3__D_2( dist__D_0__D_1__D_3__D_2, g );
DistTensor<double> T_bfnj_lvl0_part1_1_lvl1_part2_1_perm2103__S__S__D_0__D_2( dist__D_0__S__S__D_2, g );
T_bfnj_lvl0_part1_1_lvl1_part2_1_perm2103__S__S__D_0__D_2.SetLocalPermutation( perm_2_1_0_3 );
DistTensor<double> X_bmej_lvl0_part2_1_lvl1_part1_1__D_1__D_0__D_2__D_3( dist__D_1__D_0__D_2__D_3, g );
DistTensor<double> X_bmej_lvl0_part2_1_lvl1_part1_1_perm0312__D_1__D_3__S__S( dist__D_1__S__S__D_3, g );
X_bmej_lvl0_part2_1_lvl1_part1_1_perm0312__D_1__D_3__S__S.SetLocalPermutation( perm_0_3_1_2 );
DistTensor<double> W_bmje_lvl0_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> W_bmje_lvl0_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> Ztemp2_lvl0_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> Ztemp2_lvl0_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> W_bmje_lvl0_part3_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> W_bmje_lvl0_part3_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> W_bmje_lvl0_part3_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> Ztemp2_lvl0_part1_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> Ztemp2_lvl0_part1_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> Ztemp2_lvl0_part1_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> W_bmje_lvl0_part3_1_lvl1_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> W_bmje_lvl0_part3_1_lvl1_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> Ztemp2_lvl0_part1_1_lvl1_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> Ztemp2_lvl0_part1_1_lvl1_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> W_bmje_lvl0_part3_1_lvl1_part1_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> W_bmje_lvl0_part3_1_lvl1_part1_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> W_bmje_lvl0_part3_1_lvl1_part1_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> Ztemp2_lvl0_part1_1_lvl1_part3_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> Ztemp2_lvl0_part1_1_lvl1_part3_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> Ztemp2_lvl0_part1_1_lvl1_part3_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> Ztemp2_lvl0_part1_1_lvl1_part3_1_perm3102__S__S__D_0__D_2( dist__D_0__S__D_2__S, g );
Ztemp2_lvl0_part1_1_lvl1_part3_1_perm3102__S__S__D_0__D_2.SetLocalPermutation( perm_3_1_0_2 );
DistTensor<double> W_bmje_lvl0_part3_1_lvl1_part1_1__D_1__D_0__D_3__D_2( dist__D_1__D_0__D_3__D_2, g );
DistTensor<double> W_bmje_lvl0_part3_1_lvl1_part1_1_perm0213__D_1__D_3__S__S( dist__D_1__S__D_3__S, g );
W_bmje_lvl0_part3_1_lvl1_part1_1_perm0213__D_1__D_3__S__S.SetLocalPermutation( perm_0_2_1_3 );
DistTensor<double> Z_abij_lvl0_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> Z_abij_lvl0_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> Z_abij_lvl0_part2_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> Z_abij_lvl0_part2_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> Z_abij_lvl0_part2_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> Z_abij_lvl0_part2_1_lvl1_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> Z_abij_lvl0_part2_1_lvl1_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> Z_abij_lvl0_part2_1_lvl1_part3_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> Z_abij_lvl0_part2_1_lvl1_part3_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> Z_abij_lvl0_part2_1_lvl1_part3_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> Z_abij_lvl0_part2_1_lvl1_part3_1__D_0__D_1__S__S__D_2__D_3( dist__D_0__D_1__S__S__D_2__D_3, g );
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
	//  Ztemp1__D_0__D_1__D_2__D_3
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
		////High water: 0


	Tau_efmn__D_0__D_1__D_2__D_3 = T_bfnj__D_0__D_1__D_2__D_3;


//****
//**** (out of 1)
//**** Is real	0 shadows
	//Outputs:
	//  Tau_efmn__D_0__D_1__D_2__D_3
		////High water: 546120


	   // t_fj[D0,D2] <- t_fj[D01,D23]
	t_fj__D_0__D_2.AlignModesWith( modes_0_1, Tau_efmn__D_0__D_1__D_2__D_3, modes_0_2 );
	t_fj__D_0__D_2.AllGatherRedistFrom( t_fj__D_0_1__D_2_3, modes_1_3 );

	   // t_fj[D10,D32] <- t_fj[D01,D23]
	t_fj__D_1_0__D_3_2.PermutationRedistFrom( t_fj__D_0_1__D_2_3, modes_0_1_2_3 );

	   // t_fj[D1,D3] <- t_fj[D10,D32]
	t_fj__D_1__D_3.AlignModesWith( modes_0_1, Tau_efmn__D_0__D_1__D_2__D_3, modes_1_3 );
	t_fj__D_1__D_3.AllGatherRedistFrom( t_fj__D_1_0__D_3_2, modes_0_2 );
	t_fj__D_1_0__D_3_2.EmptyData();

	Permute( Tau_efmn__D_0__D_1__D_2__D_3, Tau_efmn_perm0213__D_0__D_2__D_1__D_3 );

	   // 1.0 * t_fj[D0,D2]_em * t_fj[D1,D3]_fn + 1.0 * Tau_efmn[D0,D1,D2,D3]_emfn
	LocalContractAndLocalEliminate(1.0, t_fj__D_0__D_2.LockedTensor(), indices_em, false,
		t_fj__D_1__D_3.LockedTensor(), indices_fn, false,
		1.0, Tau_efmn_perm0213__D_0__D_2__D_1__D_3.Tensor(), indices_emfn, false);
	t_fj__D_1__D_3.EmptyData();
	t_fj__D_0__D_2.EmptyData();

	Permute( Tau_efmn_perm0213__D_0__D_2__D_1__D_3, Tau_efmn__D_0__D_1__D_2__D_3 );
	Tau_efmn_perm0213__D_0__D_2__D_1__D_3.EmptyData();


//****



















//**** (out of 1)
//**** Is real	0 shadows
	//Outputs:
	//  Wtemp3__D_0__D_1__D_2__D_3
		////High water: 239112


	   // u_mnje[D1,D0,D2,D3] <- u_mnje[D0,D1,D2,D3]
	u_mnje__D_1__D_0__D_2__D_3.AlignModesWith( modes_0_1_2_3, u_mnje__D_0__D_1__D_2__D_3, modes_1_0_2_3 );
	u_mnje__D_1__D_0__D_2__D_3.PermutationRedistFrom( u_mnje__D_0__D_1__D_2__D_3, modes_0_1 );

	tempShape = u_mnje__D_0__D_1__D_2__D_3.Shape();
	Wtemp3__D_0__D_1__D_2__D_3.ResizeTo( tempShape );

	YAxpPx( -1.0, u_mnje__D_0__D_1__D_2__D_3, 2.0, u_mnje__D_1__D_0__D_2__D_3, perm_1_0_2_3, Wtemp3__D_0__D_1__D_2__D_3 );
	u_mnje__D_1__D_0__D_2__D_3.EmptyData();


//****
//**** (out of 1)
//**** Is real	0 shadows
	//Outputs:
	//  Wtemp4__D_0__D_1__D_2__D_3
		////High water: 19909026


	   // r_bmfe[D0,D1,D3,D2] <- r_bmfe[D0,D1,D2,D3]
	r_bmfe__D_0__D_1__D_3__D_2.AlignModesWith( modes_0_1_2_3, r_bmfe__D_0__D_1__D_2__D_3, modes_0_1_3_2 );
	r_bmfe__D_0__D_1__D_3__D_2.PermutationRedistFrom( r_bmfe__D_0__D_1__D_2__D_3, modes_2_3 );

	tempShape = r_bmfe__D_0__D_1__D_2__D_3.Shape();
	Wtemp4__D_0__D_1__D_2__D_3.ResizeTo( tempShape );

	YAxpPx( -1.0, r_bmfe__D_0__D_1__D_2__D_3, 2.0, r_bmfe__D_0__D_1__D_3__D_2, perm_0_1_3_2, Wtemp4__D_0__D_1__D_2__D_3 );
	r_bmfe__D_0__D_1__D_3__D_2.EmptyData();


//****
//**** (out of 1)
//**** Is real	0 shadows
	//Outputs:
	//  Wtemp2__D_0__D_1__D_2__D_3
		////High water: 7200666


	   // v_femn[D0,D1,D3,D2] <- v_femn[D0,D1,D2,D3]
	v_femn__D_0__D_1__D_3__D_2.AlignModesWith( modes_0_1_2_3, v_femn__D_0__D_1__D_2__D_3, modes_0_1_3_2 );
	v_femn__D_0__D_1__D_3__D_2.PermutationRedistFrom( v_femn__D_0__D_1__D_2__D_3, modes_2_3 );

	tempShape = v_femn__D_0__D_1__D_2__D_3.Shape();
	Wtemp2__D_0__D_1__D_2__D_3.ResizeTo( tempShape );

	YAxpPx( -1.0, v_femn__D_0__D_1__D_2__D_3, 2.0, v_femn__D_0__D_1__D_3__D_2, perm_0_1_3_2, Wtemp2__D_0__D_1__D_2__D_3 );
	v_femn__D_0__D_1__D_3__D_2.EmptyData();


//****
//**** (out of 1)
//**** Is real	0 shadows
	//Outputs:
	//  Wtemp1__D_0__D_1__D_2__D_3
		////High water: 7745310


	   // T_bfnj[D0,D1,D3,D2] <- T_bfnj[D0,D1,D2,D3]
	T_bfnj__D_0__D_1__D_3__D_2.AlignModesWith( modes_0_1_2_3, T_bfnj__D_0__D_1__D_2__D_3, modes_0_1_3_2 );
	T_bfnj__D_0__D_1__D_3__D_2.PermutationRedistFrom( T_bfnj__D_0__D_1__D_2__D_3, modes_2_3 );

	tempShape = T_bfnj__D_0__D_1__D_2__D_3.Shape();
	Wtemp1__D_0__D_1__D_2__D_3.ResizeTo( tempShape );

	ZAxpBypPx( 0.5, T_bfnj__D_0__D_1__D_2__D_3, -1.0, Tau_efmn__D_0__D_1__D_2__D_3, T_bfnj__D_0__D_1__D_3__D_2, perm_0_1_3_2, Wtemp1__D_0__D_1__D_2__D_3 );
	T_bfnj__D_0__D_1__D_3__D_2.EmptyData();


//****
//**** (out of 1)
//**** Is real	0 shadows
	//Outputs:
	//  W_bmje__D_0__D_1__D_2__D_3
		////High water: 7745310


	   // x_bmej[D0,D1,D3,D2] <- x_bmej[D0,D1,D2,D3]
	x_bmej__D_0__D_1__D_3__D_2.AlignModesWith( modes_0_1_2_3, W_bmje__D_0__D_1__D_2__D_3, modes_0_1_3_2 );
	x_bmej__D_0__D_1__D_3__D_2.PermutationRedistFrom( x_bmej__D_0__D_1__D_2__D_3, modes_2_3 );

	YAxpPx( 2.0, w_bmje__D_0__D_1__D_2__D_3, -1.0, x_bmej__D_0__D_1__D_3__D_2, perm_0_1_3_2, W_bmje__D_0__D_1__D_2__D_3 );
	x_bmej__D_0__D_1__D_3__D_2.EmptyData();


//****
//**** (out of 1)
//**** Is real	0 shadows
	//Outputs:
	//  W_bmje__D_0__D_1__D_2__D_3
		////High water: 99652878

	Permute( W_bmje__D_0__D_1__D_2__D_3, W_bmje_perm1302__D_1__D_3__D_0__D_2 );

//BEGINLOOPIFY
//NVARS 2
//VAR Wtemp2__D_0__D_1__D_2__D_3 0 3
//VAR Wtemp1__D_0__D_1__D_2__D_3 1 2
PartitionDown(Wtemp2__D_0__D_1__D_2__D_3, Wtemp2_lvl0_part0T__D_0__D_1__D_2__D_3, Wtemp2_lvl0_part0B__D_0__D_1__D_2__D_3, 0, 0);
PartitionDown(Wtemp1__D_0__D_1__D_2__D_3, Wtemp1_lvl0_part1T__D_0__D_1__D_2__D_3, Wtemp1_lvl0_part1B__D_0__D_1__D_2__D_3, 1, 0);
while(Wtemp2_lvl0_part0T__D_0__D_1__D_2__D_3.Dimension(0) < Wtemp2__D_0__D_1__D_2__D_3.Dimension(0))
{
	RepartitionDown
	( Wtemp2_lvl0_part0T__D_0__D_1__D_2__D_3, Wtemp2_lvl0_part0_0__D_0__D_1__D_2__D_3,
	  /**/ /**/
		Wtemp2_lvl0_part0_1__D_0__D_1__D_2__D_3,
		Wtemp2_lvl0_part0B__D_0__D_1__D_2__D_3,Wtemp2_lvl0_part0_2__D_0__D_1__D_2__D_3, 0, 10*blkSize);
	RepartitionDown
	( Wtemp1_lvl0_part1T__D_0__D_1__D_2__D_3, Wtemp1_lvl0_part1_0__D_0__D_1__D_2__D_3,
	  /**/ /**/
		Wtemp1_lvl0_part1_1__D_0__D_1__D_2__D_3,
		Wtemp1_lvl0_part1B__D_0__D_1__D_2__D_3,Wtemp1_lvl0_part1_2__D_0__D_1__D_2__D_3, 1, 10*blkSize);
	PartitionDown(Wtemp2_lvl0_part0_1__D_0__D_1__D_2__D_3, Wtemp2_lvl0_part0_1_lvl1_part3T__D_0__D_1__D_2__D_3, Wtemp2_lvl0_part0_1_lvl1_part3B__D_0__D_1__D_2__D_3, 3, 0);
	PartitionDown(Wtemp1_lvl0_part1_1__D_0__D_1__D_2__D_3, Wtemp1_lvl0_part1_1_lvl1_part2T__D_0__D_1__D_2__D_3, Wtemp1_lvl0_part1_1_lvl1_part2B__D_0__D_1__D_2__D_3, 2, 0);
	while(Wtemp2_lvl0_part0_1_lvl1_part3T__D_0__D_1__D_2__D_3.Dimension(3) < Wtemp2_lvl0_part0_1__D_0__D_1__D_2__D_3.Dimension(3))
	{
		RepartitionDown
		( Wtemp2_lvl0_part0_1_lvl1_part3T__D_0__D_1__D_2__D_3, Wtemp2_lvl0_part0_1_lvl1_part3_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
			Wtemp2_lvl0_part0_1_lvl1_part3_1__D_0__D_1__D_2__D_3,
			Wtemp2_lvl0_part0_1_lvl1_part3B__D_0__D_1__D_2__D_3,Wtemp2_lvl0_part0_1_lvl1_part3_2__D_0__D_1__D_2__D_3, 3, blkSize);
		RepartitionDown
		( Wtemp1_lvl0_part1_1_lvl1_part2T__D_0__D_1__D_2__D_3, Wtemp1_lvl0_part1_1_lvl1_part2_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
			Wtemp1_lvl0_part1_1_lvl1_part2_1__D_0__D_1__D_2__D_3,
			Wtemp1_lvl0_part1_1_lvl1_part2B__D_0__D_1__D_2__D_3,Wtemp1_lvl0_part1_1_lvl1_part2_2__D_0__D_1__D_2__D_3, 2, blkSize);
/*--------------------------------------------*/
		/***************/
		/* NEWVARS
		 * DistTensor<double> Wtemp2_lvl0_part0T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> Wtemp2_lvl0_part0B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> Wtemp1_lvl0_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> Wtemp1_lvl0_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> Wtemp2_lvl0_part0_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> Wtemp2_lvl0_part0_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> Wtemp2_lvl0_part0_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> Wtemp1_lvl0_part1_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> Wtemp1_lvl0_part1_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> Wtemp1_lvl0_part1_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> Wtemp2_lvl0_part0_1_lvl1_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> Wtemp2_lvl0_part0_1_lvl1_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> Wtemp1_lvl0_part1_1_lvl1_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> Wtemp1_lvl0_part1_1_lvl1_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> Wtemp2_lvl0_part0_1_lvl1_part3_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> Wtemp2_lvl0_part0_1_lvl1_part3_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> Wtemp2_lvl0_part0_1_lvl1_part3_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> Wtemp1_lvl0_part1_1_lvl1_part2_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> Wtemp1_lvl0_part1_1_lvl1_part2_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> Wtemp1_lvl0_part1_1_lvl1_part2_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> Wtemp2_lvl0_part0_1_lvl1_part3_1__D_0__D_3__D_1__D_2( dist__D_0__D_3__D_1__D_2, g );
		 * DistTensor<double> Wtemp1_lvl0_part1_1_lvl1_part2_1__D_0__D_1__D_3__D_2( dist__D_0__D_1__D_3__D_2, g );
		 * DistTensor<double> Wtemp1_lvl0_part1_1_lvl1_part2_1_perm1203__S__S__D_0__D_2( dist__D_0__S__S__D_2, g );
		 * DistTensor<double> Wtemp2_lvl0_part0_1_lvl1_part3_1_perm2103__D_1__D_3__S__S( dist__S__D_3__D_1__S, g );
		   END NEWVARS */
		/***************/

			   // Wtemp1[D0,D1,D3,D2] <- Wtemp1[D0,D1,D2,D3]
			Wtemp1_lvl0_part1_1_lvl1_part2_1__D_0__D_1__D_3__D_2.PermutationRedistFrom( Wtemp1_lvl0_part1_1_lvl1_part2_1__D_0__D_1__D_2__D_3, modes_2_3 );
		//	Wtemp1_lvl0_part1_1_lvl1_part2_1__D_0__D_1__D_2__D_3.EmptyData();

			   // Wtemp1[D0,*,*,D2] <- Wtemp1[D0,D1,D3,D2]
			Wtemp1_lvl0_part1_1_lvl1_part2_1_perm1203__S__S__D_0__D_2.AlignModesWith( modes_0_3, W_bmje__D_0__D_1__D_2__D_3, modes_0_2 );
			Wtemp1_lvl0_part1_1_lvl1_part2_1_perm1203__S__S__D_0__D_2.AllGatherRedistFrom( Wtemp1_lvl0_part1_1_lvl1_part2_1__D_0__D_1__D_3__D_2, modes_1_3 );
			Wtemp1_lvl0_part1_1_lvl1_part2_1__D_0__D_1__D_3__D_2.EmptyData();

			   // Wtemp2[D0,D3,D1,D2] <- Wtemp2[D0,D1,D2,D3]
			Wtemp2_lvl0_part0_1_lvl1_part3_1__D_0__D_3__D_1__D_2.PermutationRedistFrom( Wtemp2_lvl0_part0_1_lvl1_part3_1__D_0__D_1__D_2__D_3, modes_1_2_3 );
		//	Wtemp2_lvl0_part0_1_lvl1_part3_1__D_0__D_1__D_2__D_3.EmptyData();

			   // Wtemp2[*,D3,D1,*] <- Wtemp2[D0,D3,D1,D2]
			Wtemp2_lvl0_part0_1_lvl1_part3_1_perm2103__D_1__D_3__S__S.AlignModesWith( modes_1_2, W_bmje__D_0__D_1__D_2__D_3, modes_3_1 );
			Wtemp2_lvl0_part0_1_lvl1_part3_1_perm2103__D_1__D_3__S__S.AllGatherRedistFrom( Wtemp2_lvl0_part0_1_lvl1_part3_1__D_0__D_3__D_1__D_2, modes_0_2 );
			Wtemp2_lvl0_part0_1_lvl1_part3_1__D_0__D_3__D_1__D_2.EmptyData();

			   // 1.0 * Wtemp2[*,D3,D1,*]_mefn * Wtemp1[D0,*,*,D2]_fnbj + 1.0 * W_bmje[D0,D1,D2,D3]_mebj
			LocalContractAndLocalEliminate(1.0, Wtemp2_lvl0_part0_1_lvl1_part3_1_perm2103__D_1__D_3__S__S.LockedTensor(), indices_mefn, false,
				Wtemp1_lvl0_part1_1_lvl1_part2_1_perm1203__S__S__D_0__D_2.LockedTensor(), indices_fnbj, false,
				1.0, W_bmje_perm1302__D_1__D_3__D_0__D_2.Tensor(), indices_mebj, false);
			Wtemp1_lvl0_part1_1_lvl1_part2_1_perm1203__S__S__D_0__D_2.EmptyData();
			Wtemp2_lvl0_part0_1_lvl1_part3_1_perm2103__D_1__D_3__S__S.EmptyData();

/*--------------------------------------------*/
		SlidePartitionDown
		( Wtemp2_lvl0_part0_1_lvl1_part3T__D_0__D_1__D_2__D_3, Wtemp2_lvl0_part0_1_lvl1_part3_0__D_0__D_1__D_2__D_3,
			Wtemp2_lvl0_part0_1_lvl1_part3_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
			Wtemp2_lvl0_part0_1_lvl1_part3B__D_0__D_1__D_2__D_3,Wtemp2_lvl0_part0_1_lvl1_part3_2__D_0__D_1__D_2__D_3, 3);
		SlidePartitionDown
		( Wtemp1_lvl0_part1_1_lvl1_part2T__D_0__D_1__D_2__D_3, Wtemp1_lvl0_part1_1_lvl1_part2_0__D_0__D_1__D_2__D_3,
			Wtemp1_lvl0_part1_1_lvl1_part2_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
			Wtemp1_lvl0_part1_1_lvl1_part2B__D_0__D_1__D_2__D_3,Wtemp1_lvl0_part1_1_lvl1_part2_2__D_0__D_1__D_2__D_3, 2);
	}
	SlidePartitionDown
	( Wtemp2_lvl0_part0T__D_0__D_1__D_2__D_3, Wtemp2_lvl0_part0_0__D_0__D_1__D_2__D_3,
		Wtemp2_lvl0_part0_1__D_0__D_1__D_2__D_3,
	  /**/ /**/
		Wtemp2_lvl0_part0B__D_0__D_1__D_2__D_3,Wtemp2_lvl0_part0_2__D_0__D_1__D_2__D_3, 0);
	SlidePartitionDown
	( Wtemp1_lvl0_part1T__D_0__D_1__D_2__D_3, Wtemp1_lvl0_part1_0__D_0__D_1__D_2__D_3,
		Wtemp1_lvl0_part1_1__D_0__D_1__D_2__D_3,
	  /**/ /**/
		Wtemp1_lvl0_part1B__D_0__D_1__D_2__D_3,Wtemp1_lvl0_part1_2__D_0__D_1__D_2__D_3, 1);
}
//ENDLOOPIFY

	   // Wtemp3[D1,D0,D2,D3] <- Wtemp3[D0,D1,D2,D3]
	Wtemp3__D_1__D_0__D_2__D_3.PermutationRedistFrom( Wtemp3__D_0__D_1__D_2__D_3, modes_0_1 );
	Wtemp3__D_0__D_1__D_2__D_3.EmptyData();

	   // Wtemp3[D1,*,D2,D3] <- Wtemp3[D1,D0,D2,D3]
	Wtemp3_perm0231__D_1__D_2__D_3__S.AlignModesWith( modes_0_2_3, W_bmje__D_0__D_1__D_2__D_3, modes_1_2_3 );
	Wtemp3_perm0231__D_1__D_2__D_3__S.AllGatherRedistFrom( Wtemp3__D_1__D_0__D_2__D_3, modes_0 );
	Wtemp3__D_1__D_0__D_2__D_3.EmptyData();

	   // t_fj[D01,*] <- t_fj[D01,D23]
	t_fj__D_0_1__S.AllGatherRedistFrom( t_fj__D_0_1__D_2_3, modes_2_3 );

	   // t_fj[D0,*] <- t_fj[D01,*]
	t_fj_perm10__S__D_0.AlignModesWith( modes_0, W_bmje__D_0__D_1__D_2__D_3, modes_0 );
	t_fj_perm10__S__D_0.AllGatherRedistFrom( t_fj__D_0_1__S, modes_1 );

	Permute( W_bmje_perm1302__D_1__D_3__D_0__D_2, W_bmje_perm1230__D_1__D_2__D_3__D_0 );
	W_bmje_perm1302__D_1__D_3__D_0__D_2.EmptyData();

	   // -1.0 * Wtemp3[D1,*,D2,D3]_mjen * t_fj[D0,*]_nb + 1.0 * W_bmje[D0,D1,D2,D3]_mjeb
	LocalContractAndLocalEliminate(-1.0, Wtemp3_perm0231__D_1__D_2__D_3__S.LockedTensor(), indices_mjen, false,
		t_fj_perm10__S__D_0.LockedTensor(), indices_nb, false,
		1.0, W_bmje_perm1230__D_1__D_2__D_3__D_0.Tensor(), indices_mjeb, false);
	t_fj_perm10__S__D_0.EmptyData();
	Wtemp3_perm0231__D_1__D_2__D_3__S.EmptyData();

	Permute( W_bmje_perm1230__D_1__D_2__D_3__D_0, W_bmje__D_0__D_1__D_2__D_3 );
	W_bmje_perm1230__D_1__D_2__D_3__D_0.EmptyData();

	   // t_fj[D013,*] <- t_fj[D01,*]
	t_fj__D_0_1_3__S.LocalRedistFrom( t_fj__D_0_1__S );
	t_fj__D_0_1__S.EmptyData();

	   // t_fj[D301,*] <- t_fj[D013,*]
	t_fj__D_3_0_1__S.PermutationRedistFrom( t_fj__D_0_1_3__S, modes_0_1_3 );
	t_fj__D_0_1_3__S.EmptyData();

	   // t_fj[D3,*] <- t_fj[D301,*]
	t_fj__D_3__S.AlignModesWith( modes_0, Wtemp4__D_0__D_1__D_2__D_3, modes_3 );
	t_fj__D_3__S.AllGatherRedistFrom( t_fj__D_3_0_1__S, modes_0_1 );
	t_fj__D_3_0_1__S.EmptyData();

	W_bmje_perm01324__D_0__D_1__D_2__S__D_3.AlignModesWith( modes_0_1_3, Wtemp4__D_0__D_1__D_2__D_3, modes_0_1_2 );
	tempShape = W_bmje__D_0__D_1__D_2__D_3.Shape();
	tempShape.push_back( g.Shape()[3] );
	W_bmje_perm01324__D_0__D_1__D_2__S__D_3.ResizeTo( tempShape );

	   // 1.0 * Wtemp4[D0,D1,D2,D3]_bmef * t_fj[D3,*]_fj + 0.0 * W_bmje[D0,D1,*,D2,D3]_bmejf
	LocalContract(1.0, Wtemp4__D_0__D_1__D_2__D_3.LockedTensor(), indices_bmef, false,
		t_fj__D_3__S.LockedTensor(), indices_fj, false,
		0.0, W_bmje_perm01324__D_0__D_1__D_2__S__D_3.Tensor(), indices_bmejf, false);
	t_fj__D_3__S.EmptyData();
	Wtemp4__D_0__D_1__D_2__D_3.EmptyData();

	W_bmje_temp__D_0__D_1__S__D_2_3.AlignModesWith( modes_0_1_2_3, W_bmje__D_0__D_1__D_2__D_3, modes_0_1_2_3 );
	tempShape = W_bmje__D_0__D_1__D_2__D_3.Shape();
	W_bmje_temp__D_0__D_1__S__D_2_3.ResizeTo( tempShape );

	   // W_bmje_temp[D0,D1,*,D23] <- W_bmje[D0,D1,*,D2,D3] (with SumScatter on D3)
	W_bmje_temp__D_0__D_1__S__D_2_3.ReduceScatterRedistFrom( W_bmje_perm01324__D_0__D_1__D_2__S__D_3, 4 );
	W_bmje_perm01324__D_0__D_1__D_2__S__D_3.EmptyData();

	   // W_bmje_temp[D0,D1,*,D32] <- W_bmje_temp[D0,D1,*,D23]
	W_bmje_temp__D_0__D_1__S__D_3_2.PermutationRedistFrom( W_bmje_temp__D_0__D_1__S__D_2_3, modes_2_3 );
	W_bmje_temp__D_0__D_1__S__D_2_3.EmptyData();

	   // W_bmje_temp[D0,D1,D2,D3] <- W_bmje_temp[D0,D1,*,D32]
	W_bmje_temp__D_0__D_1__D_2__D_3.AlignModesWith( modes_0_1_2_3, W_bmje__D_0__D_1__D_2__D_3, modes_0_1_2_3 );
	W_bmje_temp__D_0__D_1__D_2__D_3.AllToAllRedistFrom( W_bmje_temp__D_0__D_1__S__D_3_2, modes_2 );
	W_bmje_temp__D_0__D_1__S__D_3_2.EmptyData();

	YxpBy( W_bmje_temp__D_0__D_1__D_2__D_3, 1.0, W_bmje__D_0__D_1__D_2__D_3 );
	W_bmje_temp__D_0__D_1__D_2__D_3.EmptyData();


//****

















//**** (out of 1)
//**** Is real	0 shadows
	//Outputs:
	//  X_bmej__D_0__D_1__D_2__D_3
		////High water: 0


	X_bmej__D_0__D_1__D_2__D_3 = x_bmej__D_0__D_1__D_2__D_3;


//****
//**** (out of 1)
//**** Is real	0 shadows
	//Outputs:
	//  Xtemp1__D_0__D_1__D_2__D_3
		////High water: 544644


	tempShape = T_bfnj__D_0__D_1__D_2__D_3.Shape();
	Xtemp1__D_0__D_1__D_2__D_3.ResizeTo( tempShape );

	ZAxpBy( 1.0, Tau_efmn__D_0__D_1__D_2__D_3, -0.5, T_bfnj__D_0__D_1__D_2__D_3, Xtemp1__D_0__D_1__D_2__D_3 );


//****
//**** (out of 1)
//**** Is real	0 shadows
	//Outputs:
	//  X_bmej__D_0__D_1__D_2__D_3
		////High water: 94630788

	Permute( X_bmej__D_0__D_1__D_2__D_3, X_bmej_perm1203__D_1__D_2__D_0__D_3 );

//BEGINLOOPIFY
//NVARS 2
//VAR v_femn__D_0__D_1__D_2__D_3 0 3
//VAR Xtemp1__D_0__D_1__D_2__D_3 1 2
PartitionDown(v_femn__D_0__D_1__D_2__D_3, v_femn_lvl0_part0T__D_0__D_1__D_2__D_3, v_femn_lvl0_part0B__D_0__D_1__D_2__D_3, 0, 0);
PartitionDown(Xtemp1__D_0__D_1__D_2__D_3, Xtemp1_lvl0_part1T__D_0__D_1__D_2__D_3, Xtemp1_lvl0_part1B__D_0__D_1__D_2__D_3, 1, 0);
while(v_femn_lvl0_part0T__D_0__D_1__D_2__D_3.Dimension(0) < v_femn__D_0__D_1__D_2__D_3.Dimension(0))
{
	RepartitionDown
	( v_femn_lvl0_part0T__D_0__D_1__D_2__D_3, v_femn_lvl0_part0_0__D_0__D_1__D_2__D_3,
	  /**/ /**/
		v_femn_lvl0_part0_1__D_0__D_1__D_2__D_3,
		v_femn_lvl0_part0B__D_0__D_1__D_2__D_3,v_femn_lvl0_part0_2__D_0__D_1__D_2__D_3, 0, 10*blkSize);
	RepartitionDown
	( Xtemp1_lvl0_part1T__D_0__D_1__D_2__D_3, Xtemp1_lvl0_part1_0__D_0__D_1__D_2__D_3,
	  /**/ /**/
		Xtemp1_lvl0_part1_1__D_0__D_1__D_2__D_3,
		Xtemp1_lvl0_part1B__D_0__D_1__D_2__D_3,Xtemp1_lvl0_part1_2__D_0__D_1__D_2__D_3, 1, 10*blkSize);
	PartitionDown(v_femn_lvl0_part0_1__D_0__D_1__D_2__D_3, v_femn_lvl0_part0_1_lvl1_part3T__D_0__D_1__D_2__D_3, v_femn_lvl0_part0_1_lvl1_part3B__D_0__D_1__D_2__D_3, 3, 0);
	PartitionDown(Xtemp1_lvl0_part1_1__D_0__D_1__D_2__D_3, Xtemp1_lvl0_part1_1_lvl1_part2T__D_0__D_1__D_2__D_3, Xtemp1_lvl0_part1_1_lvl1_part2B__D_0__D_1__D_2__D_3, 2, 0);
	while(v_femn_lvl0_part0_1_lvl1_part3T__D_0__D_1__D_2__D_3.Dimension(3) < v_femn_lvl0_part0_1__D_0__D_1__D_2__D_3.Dimension(3))
	{
		RepartitionDown
		( v_femn_lvl0_part0_1_lvl1_part3T__D_0__D_1__D_2__D_3, v_femn_lvl0_part0_1_lvl1_part3_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
			v_femn_lvl0_part0_1_lvl1_part3_1__D_0__D_1__D_2__D_3,
			v_femn_lvl0_part0_1_lvl1_part3B__D_0__D_1__D_2__D_3,v_femn_lvl0_part0_1_lvl1_part3_2__D_0__D_1__D_2__D_3, 3, blkSize);
		RepartitionDown
		( Xtemp1_lvl0_part1_1_lvl1_part2T__D_0__D_1__D_2__D_3, Xtemp1_lvl0_part1_1_lvl1_part2_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
			Xtemp1_lvl0_part1_1_lvl1_part2_1__D_0__D_1__D_2__D_3,
			Xtemp1_lvl0_part1_1_lvl1_part2B__D_0__D_1__D_2__D_3,Xtemp1_lvl0_part1_1_lvl1_part2_2__D_0__D_1__D_2__D_3, 2, blkSize);
/*--------------------------------------------*/
		/***************/
		/* NEWVARS
		 * DistTensor<double> v_femn_lvl0_part0T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> v_femn_lvl0_part0B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> Xtemp1_lvl0_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> Xtemp1_lvl0_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> v_femn_lvl0_part0_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> v_femn_lvl0_part0_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> v_femn_lvl0_part0_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> Xtemp1_lvl0_part1_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> Xtemp1_lvl0_part1_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> Xtemp1_lvl0_part1_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> v_femn_lvl0_part0_1_lvl1_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> v_femn_lvl0_part0_1_lvl1_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> Xtemp1_lvl0_part1_1_lvl1_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> Xtemp1_lvl0_part1_1_lvl1_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> v_femn_lvl0_part0_1_lvl1_part3_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> v_femn_lvl0_part0_1_lvl1_part3_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> v_femn_lvl0_part0_1_lvl1_part3_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> Xtemp1_lvl0_part1_1_lvl1_part2_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> Xtemp1_lvl0_part1_1_lvl1_part2_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> Xtemp1_lvl0_part1_1_lvl1_part2_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> v_femn_lvl0_part0_1_lvl1_part3_1_perm2103__D_1__D_2__S__S( dist__S__D_2__D_1__S, g );
		 * DistTensor<double> Xtemp1_lvl0_part1_1_lvl1_part2_1_perm1203__S__S__D_0__D_3( dist__D_0__S__S__D_3, g );
		 * DistTensor<double> v_femn_lvl0_part0_1_lvl1_part3_1__D_0__D_2__D_1__D_3( dist__D_0__D_2__D_1__D_3, g );
		   END NEWVARS */
		/***************/
			   // Xtemp1[D0,*,*,D3] <- Xtemp1[D0,D1,D2,D3]
			Xtemp1_lvl0_part1_1_lvl1_part2_1_perm1203__S__S__D_0__D_3.AlignModesWith( modes_0_3, X_bmej__D_0__D_1__D_2__D_3, modes_0_3 );
			Xtemp1_lvl0_part1_1_lvl1_part2_1_perm1203__S__S__D_0__D_3.AllGatherRedistFrom( Xtemp1_lvl0_part1_1_lvl1_part2_1__D_0__D_1__D_2__D_3, modes_1_2 );
			//Xtemp1_lvl0_part1_1_lvl1_part2_1__D_0__D_1__D_2__D_3.EmptyData();

			   // v_femn[D0,D2,D1,D3] <- v_femn[D0,D1,D2,D3]
			v_femn_lvl0_part0_1_lvl1_part3_1__D_0__D_2__D_1__D_3.PermutationRedistFrom( v_femn_lvl0_part0_1_lvl1_part3_1__D_0__D_1__D_2__D_3, modes_1_2 );

			   // v_femn[*,D2,D1,*] <- v_femn[D0,D2,D1,D3]
			v_femn_lvl0_part0_1_lvl1_part3_1_perm2103__D_1__D_2__S__S.AlignModesWith( modes_1_2, X_bmej__D_0__D_1__D_2__D_3, modes_2_1 );
			v_femn_lvl0_part0_1_lvl1_part3_1_perm2103__D_1__D_2__S__S.AllGatherRedistFrom( v_femn_lvl0_part0_1_lvl1_part3_1__D_0__D_2__D_1__D_3, modes_0_3 );
			v_femn_lvl0_part0_1_lvl1_part3_1__D_0__D_2__D_1__D_3.EmptyData();

			   // -1.0 * v_femn[*,D2,D1,*]_mefn * Xtemp1[D0,*,*,D3]_fnbj + 1.0 * X_bmej[D0,D1,D2,D3]_mebj
			LocalContractAndLocalEliminate(-1.0, v_femn_lvl0_part0_1_lvl1_part3_1_perm2103__D_1__D_2__S__S.LockedTensor(), indices_mefn, false,
				Xtemp1_lvl0_part1_1_lvl1_part2_1_perm1203__S__S__D_0__D_3.LockedTensor(), indices_fnbj, false,
				1.0, X_bmej_perm1203__D_1__D_2__D_0__D_3.Tensor(), indices_mebj, false);
			Xtemp1_lvl0_part1_1_lvl1_part2_1_perm1203__S__S__D_0__D_3.EmptyData();
			v_femn_lvl0_part0_1_lvl1_part3_1_perm2103__D_1__D_2__S__S.EmptyData();
/*--------------------------------------------*/
		SlidePartitionDown
		( v_femn_lvl0_part0_1_lvl1_part3T__D_0__D_1__D_2__D_3, v_femn_lvl0_part0_1_lvl1_part3_0__D_0__D_1__D_2__D_3,
			v_femn_lvl0_part0_1_lvl1_part3_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
			v_femn_lvl0_part0_1_lvl1_part3B__D_0__D_1__D_2__D_3,v_femn_lvl0_part0_1_lvl1_part3_2__D_0__D_1__D_2__D_3, 3);
		SlidePartitionDown
		( Xtemp1_lvl0_part1_1_lvl1_part2T__D_0__D_1__D_2__D_3, Xtemp1_lvl0_part1_1_lvl1_part2_0__D_0__D_1__D_2__D_3,
			Xtemp1_lvl0_part1_1_lvl1_part2_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
			Xtemp1_lvl0_part1_1_lvl1_part2B__D_0__D_1__D_2__D_3,Xtemp1_lvl0_part1_1_lvl1_part2_2__D_0__D_1__D_2__D_3, 2);
	}
	SlidePartitionDown
	( v_femn_lvl0_part0T__D_0__D_1__D_2__D_3, v_femn_lvl0_part0_0__D_0__D_1__D_2__D_3,
		v_femn_lvl0_part0_1__D_0__D_1__D_2__D_3,
	  /**/ /**/
		v_femn_lvl0_part0B__D_0__D_1__D_2__D_3,v_femn_lvl0_part0_2__D_0__D_1__D_2__D_3, 0);
	SlidePartitionDown
	( Xtemp1_lvl0_part1T__D_0__D_1__D_2__D_3, Xtemp1_lvl0_part1_0__D_0__D_1__D_2__D_3,
		Xtemp1_lvl0_part1_1__D_0__D_1__D_2__D_3,
	  /**/ /**/
		Xtemp1_lvl0_part1B__D_0__D_1__D_2__D_3,Xtemp1_lvl0_part1_2__D_0__D_1__D_2__D_3, 1);
}
//ENDLOOPIFY
	   // t_fj[D01,*] <- t_fj[D01,D23]
	t_fj__D_0_1__S.AllGatherRedistFrom( t_fj__D_0_1__D_2_3, modes_2_3 );

	   // t_fj[D013,*] <- t_fj[D01,*]
	t_fj__D_0_1_3__S.LocalRedistFrom( t_fj__D_0_1__S );

	   // t_fj[D301,*] <- t_fj[D013,*]
	t_fj__D_3_0_1__S.PermutationRedistFrom( t_fj__D_0_1_3__S, modes_0_1_3 );
	t_fj__D_0_1_3__S.EmptyData();

	   // t_fj[D3,*] <- t_fj[D301,*]
	t_fj__D_3__S.AlignModesWith( modes_0, r_bmfe__D_0__D_1__D_2__D_3, modes_3 );
	t_fj__D_3__S.AllGatherRedistFrom( t_fj__D_3_0_1__S, modes_0_1 );
	t_fj__D_3_0_1__S.EmptyData();

	   // u_mnje[D1,D0,D3,D2] <- u_mnje[D0,D1,D2,D3]
	u_mnje__D_1__D_0__D_3__D_2.PermutationRedistFrom( u_mnje__D_0__D_1__D_2__D_3, modes_0_1_2_3 );

	   // u_mnje[D1,*,D3,D2] <- u_mnje[D1,D0,D3,D2]
	u_mnje_perm0321__D_1__D_2__D_3__S.AlignModesWith( modes_0_2_3, X_bmej__D_0__D_1__D_2__D_3, modes_1_3_2 );
	u_mnje_perm0321__D_1__D_2__D_3__S.AllGatherRedistFrom( u_mnje__D_1__D_0__D_3__D_2, modes_0 );
	u_mnje__D_1__D_0__D_3__D_2.EmptyData();

	   // t_fj[D0,*] <- t_fj[D01,*]
	t_fj_perm10__S__D_0.AlignModesWith( modes_0, X_bmej__D_0__D_1__D_2__D_3, modes_0 );
	t_fj_perm10__S__D_0.AllGatherRedistFrom( t_fj__D_0_1__S, modes_1 );
	t_fj__D_0_1__S.EmptyData();

	Permute( X_bmej_perm1203__D_1__D_2__D_0__D_3, X_bmej_perm1230__D_1__D_2__D_3__D_0 );
	X_bmej_perm1203__D_1__D_2__D_0__D_3.EmptyData();

	   // -1.0 * u_mnje[D1,*,D3,D2]_mejn * t_fj[D0,*]_nb + 1.0 * X_bmej[D0,D1,D2,D3]_mejb
	LocalContractAndLocalEliminate(-1.0, u_mnje_perm0321__D_1__D_2__D_3__S.LockedTensor(), indices_mejn, false,
		t_fj_perm10__S__D_0.LockedTensor(), indices_nb, false,
		1.0, X_bmej_perm1230__D_1__D_2__D_3__D_0.Tensor(), indices_mejb, false);
	t_fj_perm10__S__D_0.EmptyData();
	u_mnje_perm0321__D_1__D_2__D_3__S.EmptyData();

	Permute( X_bmej_perm1230__D_1__D_2__D_3__D_0, X_bmej__D_0__D_1__D_2__D_3 );
	X_bmej_perm1230__D_1__D_2__D_3__D_0.EmptyData();

	X_bmej__D_0__D_1__D_2__S__D_3.AlignModesWith( modes_0_1_2, r_bmfe__D_0__D_1__D_2__D_3, modes_0_1_2 );
	tempShape = X_bmej__D_0__D_1__D_2__D_3.Shape();
	tempShape.push_back( g.Shape()[3] );
	X_bmej__D_0__D_1__D_2__S__D_3.ResizeTo( tempShape );

	   // 1.0 * r_bmfe[D0,D1,D2,D3]_bmef * t_fj[D3,*]_fj + 0.0 * X_bmej[D0,D1,D2,*,D3]_bmejf
	LocalContract(1.0, r_bmfe__D_0__D_1__D_2__D_3.LockedTensor(), indices_bmef, false,
		t_fj__D_3__S.LockedTensor(), indices_fj, false,
		0.0, X_bmej__D_0__D_1__D_2__S__D_3.Tensor(), indices_bmejf, false);
	t_fj__D_3__S.EmptyData();

	   // X_bmej[D0,D1,D2,D3] <- X_bmej[D0,D1,D2,*,D3] (with SumScatter on D3)
	X_bmej__D_0__D_1__D_2__D_3.ReduceScatterUpdateRedistFrom( X_bmej__D_0__D_1__D_2__S__D_3, 1.0, 4 );
	X_bmej__D_0__D_1__D_2__S__D_3.EmptyData();


//****









//**** (out of 1)
//**** Is real	0 shadows
	//Outputs:
	//  U_mnie__D_0__D_1__D_2__D_3
		////High water: 0


	U_mnie__D_0__D_1__D_2__D_3 = u_mnje__D_0__D_1__D_2__D_3;


//****
//**** (out of 1)
//**** Is real	0 shadows
	//Outputs:
	//  U_mnie__D_0__D_1__D_2__D_3
		////High water: 1055340


	   // t_fj[D0,*] <- t_fj[D01,D23]
	t_fj__D_0__S.AlignModesWith( modes_0, v_femn__D_0__D_1__D_2__D_3, modes_0 );
	t_fj__D_0__S.AllGatherRedistFrom( t_fj__D_0_1__D_2_3, modes_1_2_3 );

	Permute( v_femn__D_0__D_1__D_2__D_3, v_femn_perm2310__D_2__D_3__D_1__D_0 );

	U_mnie_perm01324__D_2__D_3__D_1__S__D_0.AlignModesWith( modes_0_1_3, v_femn__D_0__D_1__D_2__D_3, modes_2_3_1 );
	tempShape = U_mnie__D_0__D_1__D_2__D_3.Shape();
	tempShape.push_back( g.Shape()[0] );
	U_mnie_perm01324__D_2__D_3__D_1__S__D_0.ResizeTo( tempShape );

	   // 1.0 * v_femn[D0,D1,D2,D3]_mnef * t_fj[D0,*]_fi + 0.0 * U_mnie[D2,D3,*,D1,D0]_mneif
	LocalContract(1.0, v_femn_perm2310__D_2__D_3__D_1__D_0.LockedTensor(), indices_mnef, false,
		t_fj__D_0__S.LockedTensor(), indices_fi, false,
		0.0, U_mnie_perm01324__D_2__D_3__D_1__S__D_0.Tensor(), indices_mneif, false);
	t_fj__D_0__S.EmptyData();
	v_femn_perm2310__D_2__D_3__D_1__D_0.EmptyData();

	U_mnie_temp__D_2_0__D_3__S__D_1.AlignModesWith( modes_0_1_2_3, U_mnie__D_0__D_1__D_2__D_3, modes_0_1_2_3 );
	tempShape = U_mnie__D_0__D_1__D_2__D_3.Shape();
	U_mnie_temp__D_2_0__D_3__S__D_1.ResizeTo( tempShape );

	   // U_mnie_temp[D20,D3,*,D1] <- U_mnie[D2,D3,*,D1,D0] (with SumScatter on D0)
	U_mnie_temp__D_2_0__D_3__S__D_1.ReduceScatterRedistFrom( U_mnie_perm01324__D_2__D_3__D_1__S__D_0, 4 );
	U_mnie_perm01324__D_2__D_3__D_1__S__D_0.EmptyData();

	   // U_mnie_temp[D02,D1,*,D3] <- U_mnie_temp[D20,D3,*,D1]
	U_mnie_temp__D_0_2__D_1__S__D_3.PermutationRedistFrom( U_mnie_temp__D_2_0__D_3__S__D_1, modes_2_0_3_1 );
	U_mnie_temp__D_2_0__D_3__S__D_1.EmptyData();

	   // U_mnie_temp[D0,D1,D2,D3] <- U_mnie_temp[D02,D1,*,D3]
	U_mnie_temp__D_0__D_1__D_2__D_3.AlignModesWith( modes_0_1_2_3, U_mnie__D_0__D_1__D_2__D_3, modes_0_1_2_3 );
	U_mnie_temp__D_0__D_1__D_2__D_3.AllToAllRedistFrom( U_mnie_temp__D_0_2__D_1__S__D_3, modes_2 );
	U_mnie_temp__D_0_2__D_1__S__D_3.EmptyData();

	YxpBy( U_mnie_temp__D_0__D_1__D_2__D_3, 1.0, U_mnie__D_0__D_1__D_2__D_3 );
	U_mnie_temp__D_0__D_1__D_2__D_3.EmptyData();


//****















//**** (out of 1)
//**** Is real	0 shadows
	//Outputs:
	//  Qtemp1__D_0__D_1__D_2__D_3
		////High water: 107892


	   // t_fj[D01,D2] <- t_fj[D01,D23]
	t_fj__D_0_1__D_2.AllGatherRedistFrom( t_fj__D_0_1__D_2_3, modes_3 );

	   // t_fj[D013,D2] <- t_fj[D01,D2]
	t_fj__D_0_1_3__D_2.LocalRedistFrom( t_fj__D_0_1__D_2 );
	t_fj__D_0_1__D_2.EmptyData();

	   // t_fj[D301,D2] <- t_fj[D013,D2]
	t_fj__D_3_0_1__D_2.PermutationRedistFrom( t_fj__D_0_1_3__D_2, modes_0_1_3 );
	t_fj__D_0_1_3__D_2.EmptyData();

	   // t_fj[D3,*] <- t_fj[D301,D2]
	t_fj__D_3__S.AlignModesWith( modes_0, u_mnje__D_0__D_1__D_2__D_3, modes_3 );
	t_fj__D_3__S.AllGatherRedistFrom( t_fj__D_3_0_1__D_2, modes_0_1_2 );
	t_fj__D_3_0_1__D_2.EmptyData();

	Q_mnij__D_0__D_1__D_2__S__D_3.AlignModesWith( modes_0_1_2, u_mnje__D_0__D_1__D_2__D_3, modes_0_1_2 );
	tempShape = Q_mnij__D_0__D_1__D_2__D_3.Shape();
	tempShape.push_back( g.Shape()[3] );
	Q_mnij__D_0__D_1__D_2__S__D_3.ResizeTo( tempShape );

	   // 1.0 * u_mnje[D0,D1,D2,D3]_mnie * t_fj[D3,*]_ej + 0.0 * Q_mnij[D0,D1,D2,*,D3]_mnije
	LocalContract(1.0, u_mnje__D_0__D_1__D_2__D_3.LockedTensor(), indices_mnie, false,
		t_fj__D_3__S.LockedTensor(), indices_ej, false,
		0.0, Q_mnij__D_0__D_1__D_2__S__D_3.Tensor(), indices_mnije, false);
	t_fj__D_3__S.EmptyData();

	tempShape = Q_mnij__D_0__D_1__D_2__D_3.Shape();
	Qtemp1__D_0__D_1__D_2__D_3.ResizeTo( tempShape );

	   // Qtemp1[D0,D1,D2,D3] <- Q_mnij[D0,D1,D2,*,D3] (with SumScatter on D3)
	Qtemp1__D_0__D_1__D_2__D_3.ReduceScatterRedistFrom( Q_mnij__D_0__D_1__D_2__S__D_3, 4 );
	Q_mnij__D_0__D_1__D_2__S__D_3.EmptyData();


//****
//**** (out of 1)
//**** Is real	0 shadows
	//Outputs:
	//  Q_mnij__D_0__D_1__D_2__D_3
		////High water: 26244


	   // Qtemp1[D1,D0,D3,D2] <- Qtemp1[D0,D1,D2,D3]
	Qtemp1__D_1__D_0__D_3__D_2.AlignModesWith( modes_0_1_2_3, Q_mnij__D_0__D_1__D_2__D_3, modes_1_0_3_2 );
	Qtemp1__D_1__D_0__D_3__D_2.PermutationRedistFrom( Qtemp1__D_0__D_1__D_2__D_3, modes_0_1_2_3 );

	YAxpPx( 1.0, Qtemp1__D_0__D_1__D_2__D_3, 1.0, Qtemp1__D_1__D_0__D_3__D_2, perm_1_0_3_2, Q_mnij__D_0__D_1__D_2__D_3 );
	Qtemp1__D_1__D_0__D_3__D_2.EmptyData();
	Qtemp1__D_0__D_1__D_2__D_3.EmptyData();


//****
//**** (out of 1)
//**** Is real	0 shadows
	//Outputs:
	//  Q_mnij__D_0__D_1__D_2__D_3
		////High water: 57362444

//BEGINLOOPIFY
//NVARS 2
//VAR v_femn__D_0__D_1__D_2__D_3 2 3
//VAR Q_mnij__D_0__D_1__D_2__D_3 0 1
PartitionDown(v_femn__D_0__D_1__D_2__D_3, v_femn_lvl0_part2T__D_0__D_1__D_2__D_3, v_femn_lvl0_part2B__D_0__D_1__D_2__D_3, 2, 0);
PartitionDown(Q_mnij__D_0__D_1__D_2__D_3, Q_mnij_lvl0_part0T__D_0__D_1__D_2__D_3, Q_mnij_lvl0_part0B__D_0__D_1__D_2__D_3, 0, 0);
while(v_femn_lvl0_part2T__D_0__D_1__D_2__D_3.Dimension(2) < v_femn__D_0__D_1__D_2__D_3.Dimension(2))
{
	RepartitionDown
	( v_femn_lvl0_part2T__D_0__D_1__D_2__D_3, v_femn_lvl0_part2_0__D_0__D_1__D_2__D_3,
	  /**/ /**/
		v_femn_lvl0_part2_1__D_0__D_1__D_2__D_3,
		v_femn_lvl0_part2B__D_0__D_1__D_2__D_3,v_femn_lvl0_part2_2__D_0__D_1__D_2__D_3, 2, blkSize);
	RepartitionDown
	( Q_mnij_lvl0_part0T__D_0__D_1__D_2__D_3, Q_mnij_lvl0_part0_0__D_0__D_1__D_2__D_3,
	  /**/ /**/
		Q_mnij_lvl0_part0_1__D_0__D_1__D_2__D_3,
		Q_mnij_lvl0_part0B__D_0__D_1__D_2__D_3,Q_mnij_lvl0_part0_2__D_0__D_1__D_2__D_3, 0, blkSize);
	PartitionDown(v_femn_lvl0_part2_1__D_0__D_1__D_2__D_3, v_femn_lvl0_part2_1_lvl1_part3T__D_0__D_1__D_2__D_3, v_femn_lvl0_part2_1_lvl1_part3B__D_0__D_1__D_2__D_3, 3, 0);
	PartitionDown(Q_mnij_lvl0_part0_1__D_0__D_1__D_2__D_3, Q_mnij_lvl0_part0_1_lvl1_part1T__D_0__D_1__D_2__D_3, Q_mnij_lvl0_part0_1_lvl1_part1B__D_0__D_1__D_2__D_3, 1, 0);
	while(v_femn_lvl0_part2_1_lvl1_part3T__D_0__D_1__D_2__D_3.Dimension(3) < v_femn_lvl0_part2_1__D_0__D_1__D_2__D_3.Dimension(3))
	{
		RepartitionDown
		( v_femn_lvl0_part2_1_lvl1_part3T__D_0__D_1__D_2__D_3, v_femn_lvl0_part2_1_lvl1_part3_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
			v_femn_lvl0_part2_1_lvl1_part3_1__D_0__D_1__D_2__D_3,
			v_femn_lvl0_part2_1_lvl1_part3B__D_0__D_1__D_2__D_3,v_femn_lvl0_part2_1_lvl1_part3_2__D_0__D_1__D_2__D_3, 3, blkSize);
		RepartitionDown
		( Q_mnij_lvl0_part0_1_lvl1_part1T__D_0__D_1__D_2__D_3, Q_mnij_lvl0_part0_1_lvl1_part1_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
			Q_mnij_lvl0_part0_1_lvl1_part1_1__D_0__D_1__D_2__D_3,
			Q_mnij_lvl0_part0_1_lvl1_part1B__D_0__D_1__D_2__D_3,Q_mnij_lvl0_part0_1_lvl1_part1_2__D_0__D_1__D_2__D_3, 1, blkSize);
/*--------------------------------------------*/
		/***************/
		/* NEWVARS
		 * DistTensor<double> v_femn_lvl0_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> v_femn_lvl0_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> Q_mnij_lvl0_part0T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> Q_mnij_lvl0_part0B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> v_femn_lvl0_part2_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> v_femn_lvl0_part2_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> v_femn_lvl0_part2_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> Q_mnij_lvl0_part0_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> Q_mnij_lvl0_part0_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> Q_mnij_lvl0_part0_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> v_femn_lvl0_part2_1_lvl1_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> v_femn_lvl0_part2_1_lvl1_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> Q_mnij_lvl0_part0_1_lvl1_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> Q_mnij_lvl0_part0_1_lvl1_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> v_femn_lvl0_part2_1_lvl1_part3_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> v_femn_lvl0_part2_1_lvl1_part3_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> v_femn_lvl0_part2_1_lvl1_part3_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> Q_mnij_lvl0_part0_1_lvl1_part1_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> Q_mnij_lvl0_part0_1_lvl1_part1_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> Q_mnij_lvl0_part0_1_lvl1_part1_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> Q_mnij_lvl0_part0_1_lvl1_part1_1__S__S__D_2__D_3__D_0__D_1( dist__S__S__D_2__D_3__D_0__D_1, g );
		 * DistTensor<double> v_femn_lvl0_part2_1_lvl1_part3_1_perm2301__S__S__D_0__D_1( dist__D_0__D_1__S__S, g );
		   END NEWVARS */
		/***************/
			   // v_femn[D0,D1,*,*] <- v_femn[D0,D1,D2,D3]
			v_femn_lvl0_part2_1_lvl1_part3_1_perm2301__S__S__D_0__D_1.AlignModesWith( modes_0_1, Tau_efmn__D_0__D_1__D_2__D_3, modes_0_1 );
			v_femn_lvl0_part2_1_lvl1_part3_1_perm2301__S__S__D_0__D_1.AllGatherRedistFrom( v_femn_lvl0_part2_1_lvl1_part3_1__D_0__D_1__D_2__D_3, modes_2_3 );

			Q_mnij_lvl0_part0_1_lvl1_part1_1__S__S__D_2__D_3__D_0__D_1.AlignModesWith( modes_2_3, Tau_efmn__D_0__D_1__D_2__D_3, modes_2_3 );
			tempShape = Q_mnij_lvl0_part0_1_lvl1_part1_1__D_0__D_1__D_2__D_3.Shape();
			tempShape.push_back( g.Shape()[0] );
			tempShape.push_back( g.Shape()[1] );
			Q_mnij_lvl0_part0_1_lvl1_part1_1__S__S__D_2__D_3__D_0__D_1.ResizeTo( tempShape );

			   // 1.0 * v_femn[D0,D1,*,*]_mnef * Tau_efmn[D0,D1,D2,D3]_efij + 0.0 * Q_mnij[*,*,D2,D3,D0,D1]_mnijef
			LocalContract(1.0, v_femn_lvl0_part2_1_lvl1_part3_1_perm2301__S__S__D_0__D_1.LockedTensor(), indices_mnef, false,
				Tau_efmn__D_0__D_1__D_2__D_3.LockedTensor(), indices_efij, false,
				0.0, Q_mnij_lvl0_part0_1_lvl1_part1_1__S__S__D_2__D_3__D_0__D_1.Tensor(), indices_mnijef, false);
			v_femn_lvl0_part2_1_lvl1_part3_1_perm2301__S__S__D_0__D_1.EmptyData();

			   // Q_mnij[D0,D1,D2,D3] <- Q_mnij[*,*,D2,D3,D0,D1] (with SumScatter on (D0)(D1))
			Q_mnij_lvl0_part0_1_lvl1_part1_1__D_0__D_1__D_2__D_3.ReduceScatterUpdateRedistFrom( Q_mnij_lvl0_part0_1_lvl1_part1_1__S__S__D_2__D_3__D_0__D_1, 1.0, modes_5_4 );
			Q_mnij_lvl0_part0_1_lvl1_part1_1__S__S__D_2__D_3__D_0__D_1.EmptyData();
/*--------------------------------------------*/
		SlidePartitionDown
		( v_femn_lvl0_part2_1_lvl1_part3T__D_0__D_1__D_2__D_3, v_femn_lvl0_part2_1_lvl1_part3_0__D_0__D_1__D_2__D_3,
			v_femn_lvl0_part2_1_lvl1_part3_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
			v_femn_lvl0_part2_1_lvl1_part3B__D_0__D_1__D_2__D_3,v_femn_lvl0_part2_1_lvl1_part3_2__D_0__D_1__D_2__D_3, 3);
		SlidePartitionDown
		( Q_mnij_lvl0_part0_1_lvl1_part1T__D_0__D_1__D_2__D_3, Q_mnij_lvl0_part0_1_lvl1_part1_0__D_0__D_1__D_2__D_3,
			Q_mnij_lvl0_part0_1_lvl1_part1_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
			Q_mnij_lvl0_part0_1_lvl1_part1B__D_0__D_1__D_2__D_3,Q_mnij_lvl0_part0_1_lvl1_part1_2__D_0__D_1__D_2__D_3, 1);
	}
	SlidePartitionDown
	( v_femn_lvl0_part2T__D_0__D_1__D_2__D_3, v_femn_lvl0_part2_0__D_0__D_1__D_2__D_3,
		v_femn_lvl0_part2_1__D_0__D_1__D_2__D_3,
	  /**/ /**/
		v_femn_lvl0_part2B__D_0__D_1__D_2__D_3,v_femn_lvl0_part2_2__D_0__D_1__D_2__D_3, 2);
	SlidePartitionDown
	( Q_mnij_lvl0_part0T__D_0__D_1__D_2__D_3, Q_mnij_lvl0_part0_0__D_0__D_1__D_2__D_3,
		Q_mnij_lvl0_part0_1__D_0__D_1__D_2__D_3,
	  /**/ /**/
		Q_mnij_lvl0_part0B__D_0__D_1__D_2__D_3,Q_mnij_lvl0_part0_2__D_0__D_1__D_2__D_3, 0);
}
//ENDLOOPIFY


//****
//**** (out of 1)
//**** Is real	0 shadows
	//Outputs:
	//  Q_mnij__D_0__D_1__D_2__D_3
		////High water: 0


	YAxpy( 1.0, q_mnij__D_0__D_1__D_2__D_3, Q_mnij__D_0__D_1__D_2__D_3 );


//****

















//**** (out of 1)
//**** Is real	0 shadows
	//Outputs:
	//  P_jimb__D_0__D_1__D_2__D_3
		////High water: 0


	P_jimb__D_0__D_1__D_2__D_3 = u_mnje__D_0__D_1__D_2__D_3;


//****
//**** (out of 1)
//**** Is real	0 shadows
	//Outputs:
	//  P_jimb__D_0__D_1__D_2__D_3
		////High water: 57907803


	   // t_fj[D01,*] <- t_fj[D01,D23]
	t_fj__D_0_1__S.AllGatherRedistFrom( t_fj__D_0_1__D_2_3, modes_2_3 );

	   // t_fj[D012,*] <- t_fj[D01,*]
	t_fj__D_0_1_2__S.LocalRedistFrom( t_fj__D_0_1__S );

	   // t_fj[D201,*] <- t_fj[D012,*]
	t_fj__D_2_0_1__S.PermutationRedistFrom( t_fj__D_0_1_2__S, modes_0_1_2 );
	t_fj__D_0_1_2__S.EmptyData();

	   // t_fj[D2,*] <- t_fj[D201,*]
	t_fj__D_2__S.AlignModesWith( modes_0, x_bmej__D_0__D_1__D_2__D_3, modes_2 );
	t_fj__D_2__S.AllGatherRedistFrom( t_fj__D_2_0_1__S, modes_0_1 );
	t_fj__D_2_0_1__S.EmptyData();

	Permute( x_bmej__D_0__D_1__D_2__D_3, x_bmej_perm3102__D_3__D_1__D_0__D_2 );

	P_jimb_perm02314__D_3__D_1__D_0__S__D_2.AlignModesWith( modes_0_2_3, x_bmej__D_0__D_1__D_2__D_3, modes_3_1_0 );
	tempShape = P_jimb__D_0__D_1__D_2__D_3.Shape();
	tempShape.push_back( g.Shape()[2] );
	P_jimb_perm02314__D_3__D_1__D_0__S__D_2.ResizeTo( tempShape );

	   // 1.0 * x_bmej[D0,D1,D2,D3]_jmbe * t_fj[D2,*]_ei + 0.0 * P_jimb[D3,*,D1,D0,D2]_jmbie
	LocalContract(1.0, x_bmej_perm3102__D_3__D_1__D_0__D_2.LockedTensor(), indices_jmbe, false,
		t_fj__D_2__S.LockedTensor(), indices_ei, false,
		0.0, P_jimb_perm02314__D_3__D_1__D_0__S__D_2.Tensor(), indices_jmbie, false);
	t_fj__D_2__S.EmptyData();
	x_bmej_perm3102__D_3__D_1__D_0__D_2.EmptyData();

	P_jimb_temp__D_3__S__D_1_2__D_0.AlignModesWith( modes_0_1_2_3, P_jimb__D_0__D_1__D_2__D_3, modes_0_1_2_3 );
	tempShape = P_jimb__D_0__D_1__D_2__D_3.Shape();
	P_jimb_temp__D_3__S__D_1_2__D_0.ResizeTo( tempShape );

	   // P_jimb_temp[D3,*,D12,D0] <- P_jimb[D3,*,D1,D0,D2] (with SumScatter on D2)
	P_jimb_temp__D_3__S__D_1_2__D_0.ReduceScatterRedistFrom( P_jimb_perm02314__D_3__D_1__D_0__S__D_2, 4 );
	P_jimb_perm02314__D_3__D_1__D_0__S__D_2.EmptyData();

	   // P_jimb_temp[D0,*,D21,D3] <- P_jimb_temp[D3,*,D12,D0]
	P_jimb_temp__D_0__S__D_2_1__D_3.PermutationRedistFrom( P_jimb_temp__D_3__S__D_1_2__D_0, modes_3_0_1_2 );
	P_jimb_temp__D_3__S__D_1_2__D_0.EmptyData();

	   // P_jimb_temp[D0,D1,D2,D3] <- P_jimb_temp[D0,*,D21,D3]
	P_jimb_temp__D_0__D_1__D_2__D_3.AlignModesWith( modes_0_1_2_3, P_jimb__D_0__D_1__D_2__D_3, modes_0_1_2_3 );
	P_jimb_temp__D_0__D_1__D_2__D_3.AllToAllRedistFrom( P_jimb_temp__D_0__S__D_2_1__D_3, modes_1 );
	P_jimb_temp__D_0__S__D_2_1__D_3.EmptyData();

	YxpBy( P_jimb_temp__D_0__D_1__D_2__D_3, 1.0, P_jimb__D_0__D_1__D_2__D_3 );
	P_jimb_temp__D_0__D_1__D_2__D_3.EmptyData();

	   // t_fj[D013,*] <- t_fj[D01,*]
	t_fj__D_0_1_3__S.LocalRedistFrom( t_fj__D_0_1__S );
	t_fj__D_0_1__S.EmptyData();

	   // t_fj[D301,*] <- t_fj[D013,*]
	t_fj__D_3_0_1__S.PermutationRedistFrom( t_fj__D_0_1_3__S, modes_0_1_3 );
	t_fj__D_0_1_3__S.EmptyData();

	   // t_fj[D3,*] <- t_fj[D301,*]
	t_fj__D_3__S.AlignModesWith( modes_0, w_bmje__D_0__D_1__D_2__D_3, modes_3 );
	t_fj__D_3__S.AllGatherRedistFrom( t_fj__D_3_0_1__S, modes_0_1 );
	t_fj__D_3_0_1__S.EmptyData();

	P_jimb_perm32104__D_0__D_1__D_2__S__D_3.AlignModesWith( modes_1_2_3, w_bmje__D_0__D_1__D_2__D_3, modes_2_1_0 );
	tempShape = P_jimb__D_0__D_1__D_2__D_3.Shape();
	tempShape.push_back( g.Shape()[3] );
	P_jimb_perm32104__D_0__D_1__D_2__S__D_3.ResizeTo( tempShape );

	   // 1.0 * w_bmje[D0,D1,D2,D3]_bmie * t_fj[D3,*]_ej + 0.0 * P_jimb[*,D2,D1,D0,D3]_bmije
	LocalContract(1.0, w_bmje__D_0__D_1__D_2__D_3.LockedTensor(), indices_bmie, false,
		t_fj__D_3__S.LockedTensor(), indices_ej, false,
		0.0, P_jimb_perm32104__D_0__D_1__D_2__S__D_3.Tensor(), indices_bmije, false);
	t_fj__D_3__S.EmptyData();

	P_jimb_temp__S__D_2__D_1__D_0_3.AlignModesWith( modes_0_1_2_3, P_jimb__D_0__D_1__D_2__D_3, modes_0_1_2_3 );
	tempShape = P_jimb__D_0__D_1__D_2__D_3.Shape();
	P_jimb_temp__S__D_2__D_1__D_0_3.ResizeTo( tempShape );

	   // P_jimb_temp[*,D2,D1,D03] <- P_jimb[*,D2,D1,D0,D3] (with SumScatter on D3)
	P_jimb_temp__S__D_2__D_1__D_0_3.ReduceScatterRedistFrom( P_jimb_perm32104__D_0__D_1__D_2__S__D_3, 4 );
	P_jimb_perm32104__D_0__D_1__D_2__S__D_3.EmptyData();

	   // P_jimb_temp[*,D1,D2,D30] <- P_jimb_temp[*,D2,D1,D03]
	P_jimb_temp__S__D_1__D_2__D_3_0.PermutationRedistFrom( P_jimb_temp__S__D_2__D_1__D_0_3, modes_2_1_0_3 );
	P_jimb_temp__S__D_2__D_1__D_0_3.EmptyData();

	   // P_jimb_temp[D0,D1,D2,D3] <- P_jimb_temp[*,D1,D2,D30]
	P_jimb_temp__D_0__D_1__D_2__D_3.AlignModesWith( modes_0_1_2_3, P_jimb__D_0__D_1__D_2__D_3, modes_0_1_2_3 );
	P_jimb_temp__D_0__D_1__D_2__D_3.AllToAllRedistFrom( P_jimb_temp__S__D_1__D_2__D_3_0, modes_0 );
	P_jimb_temp__S__D_1__D_2__D_3_0.EmptyData();

	YxpBy( P_jimb_temp__D_0__D_1__D_2__D_3, 1.0, P_jimb__D_0__D_1__D_2__D_3 );
	P_jimb_temp__D_0__D_1__D_2__D_3.EmptyData();

//BEGINLOOPIFY
//NVARS 2
//VAR Tau_efmn__D_0__D_1__D_2__D_3 2 3
//VAR P_jimb__D_0__D_1__D_2__D_3 1 0
PartitionDown(Tau_efmn__D_0__D_1__D_2__D_3, Tau_efmn_lvl0_part2T__D_0__D_1__D_2__D_3, Tau_efmn_lvl0_part2B__D_0__D_1__D_2__D_3, 2, 0);
PartitionDown(P_jimb__D_0__D_1__D_2__D_3, P_jimb_lvl0_part1T__D_0__D_1__D_2__D_3, P_jimb_lvl0_part1B__D_0__D_1__D_2__D_3, 1, 0);
while(Tau_efmn_lvl0_part2T__D_0__D_1__D_2__D_3.Dimension(2) < Tau_efmn__D_0__D_1__D_2__D_3.Dimension(2))
{
	RepartitionDown
	( Tau_efmn_lvl0_part2T__D_0__D_1__D_2__D_3, Tau_efmn_lvl0_part2_0__D_0__D_1__D_2__D_3,
	  /**/ /**/
		Tau_efmn_lvl0_part2_1__D_0__D_1__D_2__D_3,
		Tau_efmn_lvl0_part2B__D_0__D_1__D_2__D_3,Tau_efmn_lvl0_part2_2__D_0__D_1__D_2__D_3, 2, blkSize);
	RepartitionDown
	( P_jimb_lvl0_part1T__D_0__D_1__D_2__D_3, P_jimb_lvl0_part1_0__D_0__D_1__D_2__D_3,
	  /**/ /**/
		P_jimb_lvl0_part1_1__D_0__D_1__D_2__D_3,
		P_jimb_lvl0_part1B__D_0__D_1__D_2__D_3,P_jimb_lvl0_part1_2__D_0__D_1__D_2__D_3, 1, blkSize);
	PartitionDown(Tau_efmn_lvl0_part2_1__D_0__D_1__D_2__D_3, Tau_efmn_lvl0_part2_1_lvl1_part3T__D_0__D_1__D_2__D_3, Tau_efmn_lvl0_part2_1_lvl1_part3B__D_0__D_1__D_2__D_3, 3, 0);
	PartitionDown(P_jimb_lvl0_part1_1__D_0__D_1__D_2__D_3, P_jimb_lvl0_part1_1_lvl1_part0T__D_0__D_1__D_2__D_3, P_jimb_lvl0_part1_1_lvl1_part0B__D_0__D_1__D_2__D_3, 0, 0);
	while(Tau_efmn_lvl0_part2_1_lvl1_part3T__D_0__D_1__D_2__D_3.Dimension(3) < Tau_efmn_lvl0_part2_1__D_0__D_1__D_2__D_3.Dimension(3))
	{
		RepartitionDown
		( Tau_efmn_lvl0_part2_1_lvl1_part3T__D_0__D_1__D_2__D_3, Tau_efmn_lvl0_part2_1_lvl1_part3_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
			Tau_efmn_lvl0_part2_1_lvl1_part3_1__D_0__D_1__D_2__D_3,
			Tau_efmn_lvl0_part2_1_lvl1_part3B__D_0__D_1__D_2__D_3,Tau_efmn_lvl0_part2_1_lvl1_part3_2__D_0__D_1__D_2__D_3, 3, blkSize);
		RepartitionDown
		( P_jimb_lvl0_part1_1_lvl1_part0T__D_0__D_1__D_2__D_3, P_jimb_lvl0_part1_1_lvl1_part0_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
			P_jimb_lvl0_part1_1_lvl1_part0_1__D_0__D_1__D_2__D_3,
			P_jimb_lvl0_part1_1_lvl1_part0B__D_0__D_1__D_2__D_3,P_jimb_lvl0_part1_1_lvl1_part0_2__D_0__D_1__D_2__D_3, 0, blkSize);
/*--------------------------------------------*/
		/***************/
		/* NEWVARS
		 * DistTensor<double> Tau_efmn_lvl0_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> Tau_efmn_lvl0_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> P_jimb_lvl0_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> P_jimb_lvl0_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> Tau_efmn_lvl0_part2_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> Tau_efmn_lvl0_part2_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> Tau_efmn_lvl0_part2_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> P_jimb_lvl0_part1_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> P_jimb_lvl0_part1_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> P_jimb_lvl0_part1_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> Tau_efmn_lvl0_part2_1_lvl1_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> Tau_efmn_lvl0_part2_1_lvl1_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> P_jimb_lvl0_part1_1_lvl1_part0T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> P_jimb_lvl0_part1_1_lvl1_part0B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> Tau_efmn_lvl0_part2_1_lvl1_part3_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> Tau_efmn_lvl0_part2_1_lvl1_part3_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> Tau_efmn_lvl0_part2_1_lvl1_part3_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> P_jimb_lvl0_part1_1_lvl1_part0_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> P_jimb_lvl0_part1_1_lvl1_part0_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> P_jimb_lvl0_part1_1_lvl1_part0_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> P_jimb_temp_lvl0_part1_1_lvl1_part0_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> P_jimb_lvl0_part1_1_lvl1_part0_1_perm321045__D_0__D_1__S__S__D_2__D_3( dist__S__S__D_1__D_0__D_2__D_3, g );
		 * DistTensor<double> Tau_efmn_lvl0_part2_1_lvl1_part3_1__D_2__D_3__S__S( dist__D_2__D_3__S__S, g );
		 * DistTensor<double> Tau_efmn_lvl0_part2_1_lvl1_part3_1__D_2__D_3__D_0__D_1( dist__D_2__D_3__D_0__D_1, g );
		 * DistTensor<double> P_jimb_temp_lvl0_part1_1_lvl1_part0_1__S__S__D_1_2__D_0_3( dist__S__S__D_1_2__D_0_3, g );
		 * DistTensor<double> P_jimb_temp_lvl0_part1_1_lvl1_part0_1__S__D_1__D_2__D_3_0( dist__S__D_1__D_2__D_3_0, g );
		 * DistTensor<double> P_jimb_temp_lvl0_part1_1_lvl1_part0_1__S__S__D_2_1__D_3_0( dist__S__S__D_2_1__D_3_0, g );
		   END NEWVARS */
		/***************/
			   // Tau_efmn[D2,D3,D0,D1] <- Tau_efmn[D0,D1,D2,D3]
			Tau_efmn_lvl0_part2_1_lvl1_part3_1__D_2__D_3__D_0__D_1.PermutationRedistFrom( Tau_efmn_lvl0_part2_1_lvl1_part3_1__D_0__D_1__D_2__D_3, modes_0_2_1_3 );

			   // Tau_efmn[D2,D3,*,*] <- Tau_efmn[D2,D3,D0,D1]
			Tau_efmn_lvl0_part2_1_lvl1_part3_1__D_2__D_3__S__S.AlignModesWith( modes_0_1, r_bmfe__D_0__D_1__D_2__D_3, modes_2_3 );
			Tau_efmn_lvl0_part2_1_lvl1_part3_1__D_2__D_3__S__S.AllGatherRedistFrom( Tau_efmn_lvl0_part2_1_lvl1_part3_1__D_2__D_3__D_0__D_1, modes_0_1 );
			Tau_efmn_lvl0_part2_1_lvl1_part3_1__D_2__D_3__D_0__D_1.EmptyData();

			P_jimb_lvl0_part1_1_lvl1_part0_1_perm321045__D_0__D_1__S__S__D_2__D_3.AlignModesWith( modes_2_3, r_bmfe__D_0__D_1__D_2__D_3, modes_1_0 );
			tempShape = P_jimb_lvl0_part1_1_lvl1_part0_1__D_0__D_1__D_2__D_3.Shape();
			tempShape.push_back( g.Shape()[2] );
			tempShape.push_back( g.Shape()[3] );
			P_jimb_lvl0_part1_1_lvl1_part0_1_perm321045__D_0__D_1__S__S__D_2__D_3.ResizeTo( tempShape );

			   // 1.0 * r_bmfe[D0,D1,D2,D3]_bmef * Tau_efmn[D2,D3,*,*]_efij + 0.0 * P_jimb[*,*,D1,D0,D2,D3]_bmijef
			LocalContract(1.0, r_bmfe__D_0__D_1__D_2__D_3.LockedTensor(), indices_bmef, false,
				Tau_efmn_lvl0_part2_1_lvl1_part3_1__D_2__D_3__S__S.LockedTensor(), indices_efij, false,
				0.0, P_jimb_lvl0_part1_1_lvl1_part0_1_perm321045__D_0__D_1__S__S__D_2__D_3.Tensor(), indices_bmijef, false);
			Tau_efmn_lvl0_part2_1_lvl1_part3_1__D_2__D_3__S__S.EmptyData();

			P_jimb_temp_lvl0_part1_1_lvl1_part0_1__S__S__D_1_2__D_0_3.AlignModesWith( modes_0_1_2_3, P_jimb_lvl0_part1_1_lvl1_part0_1__D_0__D_1__D_2__D_3, modes_0_1_2_3 );
			tempShape = P_jimb_lvl0_part1_1_lvl1_part0_1__D_0__D_1__D_2__D_3.Shape();
			P_jimb_temp_lvl0_part1_1_lvl1_part0_1__S__S__D_1_2__D_0_3.ResizeTo( tempShape );

			   // P_jimb_temp[*,*,D12,D03] <- P_jimb[*,*,D1,D0,D2,D3] (with SumScatter on (D2)(D3))
			P_jimb_temp_lvl0_part1_1_lvl1_part0_1__S__S__D_1_2__D_0_3.ReduceScatterRedistFrom( P_jimb_lvl0_part1_1_lvl1_part0_1_perm321045__D_0__D_1__S__S__D_2__D_3, modes_5_4 );
			P_jimb_lvl0_part1_1_lvl1_part0_1_perm321045__D_0__D_1__S__S__D_2__D_3.EmptyData();

			   // P_jimb_temp[*,*,D21,D30] <- P_jimb_temp[*,*,D12,D03]
			P_jimb_temp_lvl0_part1_1_lvl1_part0_1__S__S__D_2_1__D_3_0.PermutationRedistFrom( P_jimb_temp_lvl0_part1_1_lvl1_part0_1__S__S__D_1_2__D_0_3, modes_1_2_0_3 );
			P_jimb_temp_lvl0_part1_1_lvl1_part0_1__S__S__D_1_2__D_0_3.EmptyData();

			   // P_jimb_temp[*,D1,D2,D30] <- P_jimb_temp[*,*,D21,D30]
			P_jimb_temp_lvl0_part1_1_lvl1_part0_1__S__D_1__D_2__D_3_0.AllToAllRedistFrom( P_jimb_temp_lvl0_part1_1_lvl1_part0_1__S__S__D_2_1__D_3_0, modes_1 );
			P_jimb_temp_lvl0_part1_1_lvl1_part0_1__S__S__D_2_1__D_3_0.EmptyData();

			   // P_jimb_temp[D0,D1,D2,D3] <- P_jimb_temp[*,D1,D2,D30]
			P_jimb_temp_lvl0_part1_1_lvl1_part0_1__D_0__D_1__D_2__D_3.AlignModesWith( modes_0_1_2_3, P_jimb_lvl0_part1_1_lvl1_part0_1__D_0__D_1__D_2__D_3, modes_0_1_2_3 );
			P_jimb_temp_lvl0_part1_1_lvl1_part0_1__D_0__D_1__D_2__D_3.AllToAllRedistFrom( P_jimb_temp_lvl0_part1_1_lvl1_part0_1__S__D_1__D_2__D_3_0, modes_0 );
			P_jimb_temp_lvl0_part1_1_lvl1_part0_1__S__D_1__D_2__D_3_0.EmptyData();

			YxpBy( P_jimb_temp_lvl0_part1_1_lvl1_part0_1__D_0__D_1__D_2__D_3, 1.0, P_jimb_lvl0_part1_1_lvl1_part0_1__D_0__D_1__D_2__D_3 );
			P_jimb_temp_lvl0_part1_1_lvl1_part0_1__D_0__D_1__D_2__D_3.EmptyData();
/*--------------------------------------------*/
		SlidePartitionDown
		( Tau_efmn_lvl0_part2_1_lvl1_part3T__D_0__D_1__D_2__D_3, Tau_efmn_lvl0_part2_1_lvl1_part3_0__D_0__D_1__D_2__D_3,
			Tau_efmn_lvl0_part2_1_lvl1_part3_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
			Tau_efmn_lvl0_part2_1_lvl1_part3B__D_0__D_1__D_2__D_3,Tau_efmn_lvl0_part2_1_lvl1_part3_2__D_0__D_1__D_2__D_3, 3);
		SlidePartitionDown
		( P_jimb_lvl0_part1_1_lvl1_part0T__D_0__D_1__D_2__D_3, P_jimb_lvl0_part1_1_lvl1_part0_0__D_0__D_1__D_2__D_3,
			P_jimb_lvl0_part1_1_lvl1_part0_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
			P_jimb_lvl0_part1_1_lvl1_part0B__D_0__D_1__D_2__D_3,P_jimb_lvl0_part1_1_lvl1_part0_2__D_0__D_1__D_2__D_3, 0);
	}
	SlidePartitionDown
	( Tau_efmn_lvl0_part2T__D_0__D_1__D_2__D_3, Tau_efmn_lvl0_part2_0__D_0__D_1__D_2__D_3,
		Tau_efmn_lvl0_part2_1__D_0__D_1__D_2__D_3,
	  /**/ /**/
		Tau_efmn_lvl0_part2B__D_0__D_1__D_2__D_3,Tau_efmn_lvl0_part2_2__D_0__D_1__D_2__D_3, 2);
	SlidePartitionDown
	( P_jimb_lvl0_part1T__D_0__D_1__D_2__D_3, P_jimb_lvl0_part1_0__D_0__D_1__D_2__D_3,
		P_jimb_lvl0_part1_1__D_0__D_1__D_2__D_3,
	  /**/ /**/
		P_jimb_lvl0_part1B__D_0__D_1__D_2__D_3,P_jimb_lvl0_part1_2__D_0__D_1__D_2__D_3, 1);
}
//ENDLOOPIFY

//****







//**** (out of 1)
//**** Is real	0 shadows
	//Outputs:
	//  Htemp1__D_0__D_1__D_2__D_3
		////High water: 2178576


	   // v_femn[D0,D1,D3,D2] <- v_femn[D0,D1,D2,D3]
	v_femn__D_0__D_1__D_3__D_2.AlignModesWith( modes_0_1_2_3, v_femn__D_0__D_1__D_2__D_3, modes_0_1_3_2 );
	v_femn__D_0__D_1__D_3__D_2.PermutationRedistFrom( v_femn__D_0__D_1__D_2__D_3, modes_2_3 );

	tempShape = v_femn__D_0__D_1__D_2__D_3.Shape();
	Htemp1__D_0__D_1__D_2__D_3.ResizeTo( tempShape );

	YAxpPx( 2.0, v_femn__D_0__D_1__D_2__D_3, -1.0, v_femn__D_0__D_1__D_3__D_2, perm_0_1_3_2, Htemp1__D_0__D_1__D_2__D_3 );
	v_femn__D_0__D_1__D_3__D_2.EmptyData();


//****
//**** (out of 1)
//**** Is real	0 shadows
	//Outputs:
	//  H_me__D_0_1__D_2_3
		////High water: 1090026


	   // t_fj[D10,D32] <- t_fj[D01,D23]
	t_fj__D_1_0__D_3_2.PermutationRedistFrom( t_fj__D_0_1__D_2_3, modes_0_1_2_3 );

	   // t_fj[D1,D3] <- t_fj[D10,D32]
	t_fj__D_1__D_3.AlignModesWith( modes_0_1, Htemp1__D_0__D_1__D_2__D_3, modes_1_3 );
	t_fj__D_1__D_3.AllGatherRedistFrom( t_fj__D_1_0__D_3_2, modes_0_2 );
	t_fj__D_1_0__D_3_2.EmptyData();

	Permute( Htemp1__D_0__D_1__D_2__D_3, Htemp1_perm2013__D_2__D_0__D_1__D_3 );
	Htemp1__D_0__D_1__D_2__D_3.EmptyData();

	H_me__D_2__D_0__D_1__D_3.AlignModesWith( modes_0_1, Htemp1__D_0__D_1__D_2__D_3, modes_2_0 );
	tempShape = H_me__D_0_1__D_2_3.Shape();
	tempShape.push_back( g.Shape()[1] );
	tempShape.push_back( g.Shape()[3] );
	H_me__D_2__D_0__D_1__D_3.ResizeTo( tempShape );

	   // 1.0 * Htemp1[D0,D1,D2,D3]_mefn * t_fj[D1,D3]_fn + 0.0 * H_me[D2,D0,D1,D3]_mefn
	LocalContract(1.0, Htemp1_perm2013__D_2__D_0__D_1__D_3.LockedTensor(), indices_mefn, false,
		t_fj__D_1__D_3.LockedTensor(), indices_fn, false,
		0.0, H_me__D_2__D_0__D_1__D_3.Tensor(), indices_mefn, false);
	t_fj__D_1__D_3.EmptyData();
	Htemp1_perm2013__D_2__D_0__D_1__D_3.EmptyData();

	H_me__D_2_1__D_0_3.AlignModesWith( modes_0_1, H_me__D_0_1__D_2_3, modes_0_1 );
	tempShape = H_me__D_0_1__D_2_3.Shape();
	H_me__D_2_1__D_0_3.ResizeTo( tempShape );

	   // H_me[D21,D03] <- H_me[D2,D0,D1,D3] (with SumScatter on (D1)(D3))
	H_me__D_2_1__D_0_3.ReduceScatterRedistFrom( H_me__D_2__D_0__D_1__D_3, modes_3_2 );
	H_me__D_2__D_0__D_1__D_3.EmptyData();

	   // H_me[D12,D30] <- H_me[D21,D03]
	H_me__D_1_2__D_3_0.PermutationRedistFrom( H_me__D_2_1__D_0_3, modes_2_1_0_3 );
	H_me__D_2_1__D_0_3.EmptyData();

	   // H_me[D10,D32] <- H_me[D12,D30]
	H_me__D_1_0__D_3_2.PermutationRedistFrom( H_me__D_1_2__D_3_0, modes_2_0 );
	H_me__D_1_2__D_3_0.EmptyData();

	   // H_me[D01,D23] <- H_me[D10,D32]
	H_me__D_0_1__D_2_3.AlignModesWith( modes_0_1, H_me__D_0_1__D_2_3, modes_0_1 );
	H_me__D_0_1__D_2_3.PermutationRedistFrom( H_me__D_1_0__D_3_2, modes_1_0_3_2 );
	H_me__D_1_0__D_3_2.EmptyData();


//****













//**** (out of 1)
//**** Is real	0 shadows
	//Outputs:
	//  Ftemp2__D_0__D_1__D_2__D_3
		////High water: 19849248


	   // r_bmfe[D0,D1,D3,D2] <- r_bmfe[D0,D1,D2,D3]
	r_bmfe__D_0__D_1__D_3__D_2.AlignModesWith( modes_0_1_2_3, r_bmfe__D_0__D_1__D_2__D_3, modes_0_1_3_2 );
	r_bmfe__D_0__D_1__D_3__D_2.PermutationRedistFrom( r_bmfe__D_0__D_1__D_2__D_3, modes_2_3 );

	tempShape = r_bmfe__D_0__D_1__D_2__D_3.Shape();
	Ftemp2__D_0__D_1__D_2__D_3.ResizeTo( tempShape );

	YAxpPx( 2.0, r_bmfe__D_0__D_1__D_2__D_3, -1.0, r_bmfe__D_0__D_1__D_3__D_2, perm_0_1_3_2, Ftemp2__D_0__D_1__D_2__D_3 );
	r_bmfe__D_0__D_1__D_3__D_2.EmptyData();


//****
//**** (out of 1)
//**** Is real	0 shadows
	//Outputs:
	//  Ftemp1__D_0__D_1__D_2__D_3
		////High water: 7140888


	   // v_femn[D0,D1,D3,D2] <- v_femn[D0,D1,D2,D3]
	v_femn__D_0__D_1__D_3__D_2.AlignModesWith( modes_0_1_2_3, v_femn__D_0__D_1__D_2__D_3, modes_0_1_3_2 );
	v_femn__D_0__D_1__D_3__D_2.PermutationRedistFrom( v_femn__D_0__D_1__D_2__D_3, modes_2_3 );

	tempShape = v_femn__D_0__D_1__D_2__D_3.Shape();
	Ftemp1__D_0__D_1__D_2__D_3.ResizeTo( tempShape );

	YAxpPx( 2.0, v_femn__D_0__D_1__D_2__D_3, -1.0, v_femn__D_0__D_1__D_3__D_2, perm_0_1_3_2, Ftemp1__D_0__D_1__D_2__D_3 );
	v_femn__D_0__D_1__D_3__D_2.EmptyData();


//****
//**** (out of 1)
//**** Is real	0 shadows
	//Outputs:
	//  F_ae__D_0_1__D_2_3
		////High water: 14686200


	   // Ftemp1[*,D1,D2,D3] <- Ftemp1[D0,D1,D2,D3]
	Ftemp1__S__D_1__D_2__D_3.AlignModesWith( modes_1_2_3, T_bfnj__D_0__D_1__D_2__D_3, modes_1_2_3 );
	Ftemp1__S__D_1__D_2__D_3.AllGatherRedistFrom( Ftemp1__D_0__D_1__D_2__D_3, modes_0 );
	Ftemp1__D_0__D_1__D_2__D_3.EmptyData();

	Permute( T_bfnj__D_0__D_1__D_2__D_3, T_bfnj_perm1230__D_1__D_2__D_3__D_0 );

	F_ae_perm10234__S__D_0__D_1__D_2__D_3.AlignModesWith( modes_0, T_bfnj__D_0__D_1__D_2__D_3, modes_0 );
	tempShape = F_ae__D_0_1__D_2_3.Shape();
	tempShape.push_back( g.Shape()[1] );
	tempShape.push_back( g.Shape()[2] );
	tempShape.push_back( g.Shape()[3] );
	F_ae_perm10234__S__D_0__D_1__D_2__D_3.ResizeTo( tempShape );

	   // -1.0 * Ftemp1[*,D1,D2,D3]_efmn * T_bfnj[D0,D1,D2,D3]_fmna + 0.0 * F_ae[D0,*,D1,D2,D3]_eafmn
	LocalContract(-1.0, Ftemp1__S__D_1__D_2__D_3.LockedTensor(), indices_efmn, false,
		T_bfnj_perm1230__D_1__D_2__D_3__D_0.LockedTensor(), indices_fmna, false,
		0.0, F_ae_perm10234__S__D_0__D_1__D_2__D_3.Tensor(), indices_eafmn, false);
	T_bfnj_perm1230__D_1__D_2__D_3__D_0.EmptyData();
	Ftemp1__S__D_1__D_2__D_3.EmptyData();

	   // F_ae[D01,D23] <- F_ae[D0,*,D1,D2,D3] (with SumScatter on (D1)(D2)(D3))
	F_ae__D_0_1__D_2_3.ReduceScatterRedistFrom( F_ae_perm10234__S__D_0__D_1__D_2__D_3, modes_4_3_2 );
	F_ae_perm10234__S__D_0__D_1__D_2__D_3.EmptyData();


//****
//**** (out of 1)
//**** Is real	0 shadows
	//Outputs:
	//  F_ae__D_0_1__D_2_3
		////High water: 9925362


	   // t_fj[D03,D21] <- t_fj[D01,D23]
	t_fj__D_0_3__D_2_1.PermutationRedistFrom( t_fj__D_0_1__D_2_3, modes_1_3 );

	   // t_fj[D30,D12] <- t_fj[D03,D21]
	t_fj__D_3_0__D_1_2.PermutationRedistFrom( t_fj__D_0_3__D_2_1, modes_0_3_2_1 );
	t_fj__D_0_3__D_2_1.EmptyData();

	   // t_fj[D3,D1] <- t_fj[D30,D12]
	t_fj__D_3__D_1.AlignModesWith( modes_0_1, Ftemp2__D_0__D_1__D_2__D_3, modes_3_1 );
	t_fj__D_3__D_1.AllGatherRedistFrom( t_fj__D_3_0__D_1_2, modes_0_2 );
	t_fj__D_3_0__D_1_2.EmptyData();

	Permute( Ftemp2__D_0__D_1__D_2__D_3, Ftemp2_perm0231__D_0__D_2__D_3__D_1 );
	Ftemp2__D_0__D_1__D_2__D_3.EmptyData();

	F_ae__D_0__D_2__D_3__D_1.AlignModesWith( modes_0_1, Ftemp2__D_0__D_1__D_2__D_3, modes_0_2 );
	tempShape = F_ae__D_0_1__D_2_3.Shape();
	tempShape.push_back( g.Shape()[3] );
	tempShape.push_back( g.Shape()[1] );
	F_ae__D_0__D_2__D_3__D_1.ResizeTo( tempShape );

	   // 1.0 * Ftemp2[D0,D1,D2,D3]_aefm * t_fj[D3,D1]_fm + 0.0 * F_ae[D0,D2,D3,D1]_aefm
	LocalContract(1.0, Ftemp2_perm0231__D_0__D_2__D_3__D_1.LockedTensor(), indices_aefm, false,
		t_fj__D_3__D_1.LockedTensor(), indices_fm, false,
		0.0, F_ae__D_0__D_2__D_3__D_1.Tensor(), indices_aefm, false);
	t_fj__D_3__D_1.EmptyData();
	Ftemp2_perm0231__D_0__D_2__D_3__D_1.EmptyData();

	   // F_ae[D01,D23] <- F_ae[D0,D2,D3,D1] (with SumScatter on (D3)(D1))
	F_ae__D_0_1__D_2_3.ReduceScatterUpdateRedistFrom( F_ae__D_0__D_2__D_3__D_1, 1.0, modes_3_2 );
	F_ae__D_0__D_2__D_3__D_1.EmptyData();

	   // H_me[*,D23] <- H_me[D01,D23]
	H_me_perm10__D_2_3__S.AlignModesWith( modes_1, F_ae__D_0_1__D_2_3, modes_1 );
	H_me_perm10__D_2_3__S.AllGatherRedistFrom( H_me__D_0_1__D_2_3, modes_0_1 );

	   // t_fj[D01,*] <- t_fj[D01,D23]
	t_fj_perm10__S__D_0_1.AlignModesWith( modes_0, F_ae__D_0_1__D_2_3, modes_0 );
	t_fj_perm10__S__D_0_1.AllGatherRedistFrom( t_fj__D_0_1__D_2_3, modes_2_3 );

	Permute( F_ae__D_0_1__D_2_3, F_ae_perm10__D_2_3__D_0_1 );

	   // -1.0 * H_me[*,D23]_em * t_fj[D01,*]_ma + 1.0 * F_ae[D01,D23]_ea
	LocalContractAndLocalEliminate(-1.0, H_me_perm10__D_2_3__S.LockedTensor(), indices_em, false,
		t_fj_perm10__S__D_0_1.LockedTensor(), indices_ma, false,
		1.0, F_ae_perm10__D_2_3__D_0_1.Tensor(), indices_ea, false);
	t_fj_perm10__S__D_0_1.EmptyData();
	H_me_perm10__D_2_3__S.EmptyData();

	Permute( F_ae_perm10__D_2_3__D_0_1, F_ae__D_0_1__D_2_3 );
	F_ae_perm10__D_2_3__D_0_1.EmptyData();


//****













//**** (out of 1)
//**** Is real	0 shadows
	//Outputs:
	//  Gtemp1__D_0__D_1__D_2__D_3
		////High water: 2178576


	   // v_femn[D0,D1,D3,D2] <- v_femn[D0,D1,D2,D3]
	v_femn__D_0__D_1__D_3__D_2.AlignModesWith( modes_0_1_2_3, v_femn__D_0__D_1__D_2__D_3, modes_0_1_3_2 );
	v_femn__D_0__D_1__D_3__D_2.PermutationRedistFrom( v_femn__D_0__D_1__D_2__D_3, modes_2_3 );

	tempShape = v_femn__D_0__D_1__D_2__D_3.Shape();
	Gtemp1__D_0__D_1__D_2__D_3.ResizeTo( tempShape );

	YAxpPx( 2.0, v_femn__D_0__D_1__D_2__D_3, -1.0, v_femn__D_0__D_1__D_3__D_2, perm_0_1_3_2, Gtemp1__D_0__D_1__D_2__D_3 );
	v_femn__D_0__D_1__D_3__D_2.EmptyData();


//****
//**** (out of 1)
//**** Is real	0 shadows
	//Outputs:
	//  G_mi__D_0_1__D_2_3
		////High water: 8956368


	   // Gtemp1[D0,D1,*,D3] <- Gtemp1[D0,D1,D2,D3]
	Gtemp1_perm2013__S__D_0__D_1__D_3.AlignModesWith( modes_0_1_3, T_bfnj__D_0__D_1__D_2__D_3, modes_0_1_3 );
	Gtemp1_perm2013__S__D_0__D_1__D_3.AllGatherRedistFrom( Gtemp1__D_0__D_1__D_2__D_3, modes_2 );
	Gtemp1__D_0__D_1__D_2__D_3.EmptyData();

	Permute( T_bfnj__D_0__D_1__D_2__D_3, T_bfnj_perm0132__D_0__D_1__D_3__D_2 );

	G_mi__S__D_2__D_0__D_1__D_3.AlignModesWith( modes_1, T_bfnj__D_0__D_1__D_2__D_3, modes_2 );
	tempShape = G_mi__D_0_1__D_2_3.Shape();
	tempShape.push_back( g.Shape()[0] );
	tempShape.push_back( g.Shape()[1] );
	tempShape.push_back( g.Shape()[3] );
	G_mi__S__D_2__D_0__D_1__D_3.ResizeTo( tempShape );

	   // 1.0 * Gtemp1[D0,D1,*,D3]_mefn * T_bfnj[D0,D1,D2,D3]_efni + 0.0 * G_mi[*,D2,D0,D1,D3]_miefn
	LocalContract(1.0, Gtemp1_perm2013__S__D_0__D_1__D_3.LockedTensor(), indices_mefn, false,
		T_bfnj_perm0132__D_0__D_1__D_3__D_2.LockedTensor(), indices_efni, false,
		0.0, G_mi__S__D_2__D_0__D_1__D_3.Tensor(), indices_miefn, false);
	T_bfnj_perm0132__D_0__D_1__D_3__D_2.EmptyData();
	Gtemp1_perm2013__S__D_0__D_1__D_3.EmptyData();

	   // G_mi[D01,D23] <- G_mi[*,D2,D0,D1,D3] (with SumScatter on (D0)(D1)(D3))
	G_mi__D_0_1__D_2_3.ReduceScatterRedistFrom( G_mi__S__D_2__D_0__D_1__D_3, modes_4_3_2 );
	G_mi__S__D_2__D_0__D_1__D_3.EmptyData();


//****
//**** (out of 1)
//**** Is real	0 shadows
	//Outputs:
	//  Gtemp2__D_0__D_1__D_2__D_3
		////High water: 239112


	   // u_mnje[D1,D0,D2,D3] <- u_mnje[D0,D1,D2,D3]
	u_mnje__D_1__D_0__D_2__D_3.AlignModesWith( modes_0_1_2_3, u_mnje__D_0__D_1__D_2__D_3, modes_1_0_2_3 );
	u_mnje__D_1__D_0__D_2__D_3.PermutationRedistFrom( u_mnje__D_0__D_1__D_2__D_3, modes_0_1 );

	tempShape = u_mnje__D_0__D_1__D_2__D_3.Shape();
	Gtemp2__D_0__D_1__D_2__D_3.ResizeTo( tempShape );

	YAxpPx( 2.0, u_mnje__D_0__D_1__D_2__D_3, -1.0, u_mnje__D_1__D_0__D_2__D_3, perm_1_0_2_3, Gtemp2__D_0__D_1__D_2__D_3 );
	u_mnje__D_1__D_0__D_2__D_3.EmptyData();


//****
//**** (out of 1)
//**** Is real	0 shadows
	//Outputs:
	//  G_mi__D_0_1__D_2_3
		////High water: 120294


	   // t_fj[D03,D21] <- t_fj[D01,D23]
	t_fj__D_0_3__D_2_1.PermutationRedistFrom( t_fj__D_0_1__D_2_3, modes_1_3 );

	   // t_fj[D30,D12] <- t_fj[D03,D21]
	t_fj__D_3_0__D_1_2.PermutationRedistFrom( t_fj__D_0_3__D_2_1, modes_0_3_2_1 );
	t_fj__D_0_3__D_2_1.EmptyData();

	   // t_fj[D3,D1] <- t_fj[D30,D12]
	t_fj__D_3__D_1.AlignModesWith( modes_0_1, Gtemp2__D_0__D_1__D_2__D_3, modes_3_1 );
	t_fj__D_3__D_1.AllGatherRedistFrom( t_fj__D_3_0__D_1_2, modes_0_2 );
	t_fj__D_3_0__D_1_2.EmptyData();

	Permute( Gtemp2__D_0__D_1__D_2__D_3, Gtemp2_perm0231__D_0__D_2__D_3__D_1 );
	Gtemp2__D_0__D_1__D_2__D_3.EmptyData();

	G_mi__D_0__D_2__D_3__D_1.AlignModesWith( modes_0_1, Gtemp2__D_0__D_1__D_2__D_3, modes_0_2 );
	tempShape = G_mi__D_0_1__D_2_3.Shape();
	tempShape.push_back( g.Shape()[3] );
	tempShape.push_back( g.Shape()[1] );
	G_mi__D_0__D_2__D_3__D_1.ResizeTo( tempShape );

	   // 1.0 * Gtemp2[D0,D1,D2,D3]_mien * t_fj[D3,D1]_en + 0.0 * G_mi[D0,D2,D3,D1]_mien
	LocalContract(1.0, Gtemp2_perm0231__D_0__D_2__D_3__D_1.LockedTensor(), indices_mien, false,
		t_fj__D_3__D_1.LockedTensor(), indices_en, false,
		0.0, G_mi__D_0__D_2__D_3__D_1.Tensor(), indices_mien, false);
	t_fj__D_3__D_1.EmptyData();
	Gtemp2_perm0231__D_0__D_2__D_3__D_1.EmptyData();

	   // G_mi[D01,D23] <- G_mi[D0,D2,D3,D1] (with SumScatter on (D3)(D1))
	G_mi__D_0_1__D_2_3.ReduceScatterUpdateRedistFrom( G_mi__D_0__D_2__D_3__D_1, 1.0, modes_3_2 );
	G_mi__D_0__D_2__D_3__D_1.EmptyData();


//****
//**** (out of 1)
//**** Is real	0 shadows
	//Outputs:
	//  G_mi__D_0_1__D_2_3
		////High water: 1474


	   // t_fj[D23,D01] <- t_fj[D01,D23]
	t_fj__D_2_3__D_0_1.PermutationRedistFrom( t_fj__D_0_1__D_2_3, modes_0_1_2_3 );

	   // t_fj[D23,*] <- t_fj[D23,D01]
	t_fj__D_2_3__S.AlignModesWith( modes_0, H_me__D_0_1__D_2_3, modes_1 );
	t_fj__D_2_3__S.AllGatherRedistFrom( t_fj__D_2_3__D_0_1, modes_0_1 );
	t_fj__D_2_3__D_0_1.EmptyData();

	G_mi__D_0_1__S__D_2_3.AlignModesWith( modes_0, H_me__D_0_1__D_2_3, modes_0 );
	tempShape = G_mi__D_0_1__D_2_3.Shape();
	tempShape.push_back( g.Shape()[2] * g.Shape()[3] );
	G_mi__D_0_1__S__D_2_3.ResizeTo( tempShape );

	   // 1.0 * H_me[D01,D23]_me * t_fj[D23,*]_ei + 0.0 * G_mi[D01,*,D23]_mie
	LocalContract(1.0, H_me__D_0_1__D_2_3.LockedTensor(), indices_me, false,
		t_fj__D_2_3__S.LockedTensor(), indices_ei, false,
		0.0, G_mi__D_0_1__S__D_2_3.Tensor(), indices_mie, false);
	t_fj__D_2_3__S.EmptyData();

	   // G_mi[D01,D23] <- G_mi[D01,*,D23] (with SumScatter on D23)
	G_mi__D_0_1__D_2_3.ReduceScatterUpdateRedistFrom( G_mi__D_0_1__S__D_2_3, 1.0, 2 );
	G_mi__D_0_1__S__D_2_3.EmptyData();


//****




















//**** (out of 1)
//**** Is real	0 shadows
	//Outputs:
	//  ztemp2__D_0__D_1__D_2__D_3
		////High water: 19849252


	   // r_bmfe[D0,D1,D3,D2] <- r_bmfe[D0,D1,D2,D3]
	r_bmfe__D_0__D_1__D_3__D_2.AlignModesWith( modes_0_1_2_3, r_bmfe__D_0__D_1__D_2__D_3, modes_0_1_3_2 );
	r_bmfe__D_0__D_1__D_3__D_2.PermutationRedistFrom( r_bmfe__D_0__D_1__D_2__D_3, modes_2_3 );

	tempShape = r_bmfe__D_0__D_1__D_2__D_3.Shape();
	ztemp2__D_0__D_1__D_2__D_3.ResizeTo( tempShape );

	YAxpPx( 2.0, r_bmfe__D_0__D_1__D_2__D_3, -1.0, r_bmfe__D_0__D_1__D_3__D_2, perm_0_1_3_2, ztemp2__D_0__D_1__D_2__D_3 );
	r_bmfe__D_0__D_1__D_3__D_2.EmptyData();


//****
//**** (out of 1)
//**** Is real	0 shadows
	//Outputs:
	//  z_ai__D_0_1__D_2_3
		////High water: 13918684


	   // Tau_efmn[D2,D3,D0,D1] <- Tau_efmn[D0,D1,D2,D3]
	Tau_efmn__D_2__D_3__D_0__D_1.PermutationRedistFrom( Tau_efmn__D_0__D_1__D_2__D_3, modes_0_2_1_3 );

	   // Tau_efmn[D2,D3,*,D1] <- Tau_efmn[D2,D3,D0,D1]
	Tau_efmn_perm3012__D_1__D_2__D_3__S.AlignModesWith( modes_0_1_3, ztemp2__D_0__D_1__D_2__D_3, modes_2_3_1 );
	Tau_efmn_perm3012__D_1__D_2__D_3__S.AllGatherRedistFrom( Tau_efmn__D_2__D_3__D_0__D_1, modes_0 );
	Tau_efmn__D_2__D_3__D_0__D_1.EmptyData();

	z_ai_perm01423__D_0__S__D_1__D_2__D_3.AlignModesWith( modes_0, ztemp2__D_0__D_1__D_2__D_3, modes_0 );
	tempShape = z_ai__D_0_1__D_2_3.Shape();
	tempShape.push_back( g.Shape()[2] );
	tempShape.push_back( g.Shape()[3] );
	tempShape.push_back( g.Shape()[1] );
	z_ai_perm01423__D_0__S__D_1__D_2__D_3.ResizeTo( tempShape );

	   // 1.0 * ztemp2[D0,D1,D2,D3]_amef * Tau_efmn[D2,D3,*,D1]_mefi + 0.0 * z_ai[D0,*,D2,D3,D1]_aimef
	LocalContract(1.0, ztemp2__D_0__D_1__D_2__D_3.LockedTensor(), indices_amef, false,
		Tau_efmn_perm3012__D_1__D_2__D_3__S.LockedTensor(), indices_mefi, false,
		0.0, z_ai_perm01423__D_0__S__D_1__D_2__D_3.Tensor(), indices_aimef, false);
	Tau_efmn_perm3012__D_1__D_2__D_3__S.EmptyData();
	ztemp2__D_0__D_1__D_2__D_3.EmptyData();

	   // z_ai[D01,D23] <- z_ai[D0,*,D2,D3,D1] (with SumScatter on (D2)(D3)(D1))
	z_ai__D_0_1__D_2_3.ReduceScatterRedistFrom( z_ai_perm01423__D_0__S__D_1__D_2__D_3, modes_4_3_2 );
	z_ai_perm01423__D_0__S__D_1__D_2__D_3.EmptyData();


//****
//**** (out of 1)
//**** Is real	0 shadows
	//Outputs:
	//  ztemp4__D_0__D_1__D_2__D_3
		////High water: 2178580


	   // x_bmej[D0,D1,D3,D2] <- x_bmej[D0,D1,D2,D3]
	x_bmej__D_0__D_1__D_3__D_2.AlignModesWith( modes_0_1_2_3, w_bmje__D_0__D_1__D_2__D_3, modes_0_1_3_2 );
	x_bmej__D_0__D_1__D_3__D_2.PermutationRedistFrom( x_bmej__D_0__D_1__D_2__D_3, modes_2_3 );

	tempShape = w_bmje__D_0__D_1__D_2__D_3.Shape();
	ztemp4__D_0__D_1__D_2__D_3.ResizeTo( tempShape );

	YAxpPx( 2.0, w_bmje__D_0__D_1__D_2__D_3, -1.0, x_bmej__D_0__D_1__D_3__D_2, perm_0_1_3_2, ztemp4__D_0__D_1__D_2__D_3 );
	x_bmej__D_0__D_1__D_3__D_2.EmptyData();


//****
//**** (out of 1)
//**** Is real	0 shadows
	//Outputs:
	//  ztemp3__D_0__D_1__D_2__D_3
		////High water: 2723224


	   // T_bfnj[D0,D1,D3,D2] <- T_bfnj[D0,D1,D2,D3]
	T_bfnj__D_0__D_1__D_3__D_2.AlignModesWith( modes_0_1_2_3, T_bfnj__D_0__D_1__D_2__D_3, modes_0_1_3_2 );
	T_bfnj__D_0__D_1__D_3__D_2.PermutationRedistFrom( T_bfnj__D_0__D_1__D_2__D_3, modes_2_3 );

	tempShape = T_bfnj__D_0__D_1__D_2__D_3.Shape();
	ztemp3__D_0__D_1__D_2__D_3.ResizeTo( tempShape );

	YAxpPx( 2.0, T_bfnj__D_0__D_1__D_2__D_3, -1.0, T_bfnj__D_0__D_1__D_3__D_2, perm_0_1_3_2, ztemp3__D_0__D_1__D_2__D_3 );
	T_bfnj__D_0__D_1__D_3__D_2.EmptyData();


//****
//**** (out of 1)
//**** Is real	0 shadows
	//Outputs:
	//  z_ai__D_0_1__D_2_3
		////High water: 1636150


	   // t_fj[D03,D21] <- t_fj[D01,D23]
	t_fj__D_0_3__D_2_1.PermutationRedistFrom( t_fj__D_0_1__D_2_3, modes_1_3 );

	   // t_fj[D30,D12] <- t_fj[D03,D21]
	t_fj__D_3_0__D_1_2.PermutationRedistFrom( t_fj__D_0_3__D_2_1, modes_0_3_2_1 );
	t_fj__D_0_3__D_2_1.EmptyData();

	   // t_fj[D3,D1] <- t_fj[D30,D12]
	t_fj__D_3__D_1.AlignModesWith( modes_0_1, ztemp4__D_0__D_1__D_2__D_3, modes_3_1 );
	t_fj__D_3__D_1.AllGatherRedistFrom( t_fj__D_3_0__D_1_2, modes_0_2 );
	t_fj__D_3_0__D_1_2.EmptyData();

	   // H_me[D03,D21] <- H_me[D01,D23]
	H_me__D_0_3__D_2_1.PermutationRedistFrom( H_me__D_0_1__D_2_3, modes_1_3 );

	   // H_me[D30,D12] <- H_me[D03,D21]
	H_me__D_3_0__D_1_2.PermutationRedistFrom( H_me__D_0_3__D_2_1, modes_0_3_2_1 );
	H_me__D_0_3__D_2_1.EmptyData();

	   // H_me[D3,D1] <- H_me[D30,D12]
	H_me_perm10__D_1__D_3.AlignModesWith( modes_0_1, ztemp3__D_0__D_1__D_2__D_3, modes_3_1 );
	H_me_perm10__D_1__D_3.AllGatherRedistFrom( H_me__D_3_0__D_1_2, modes_0_2 );
	H_me__D_3_0__D_1_2.EmptyData();

	z_ai__D_0__D_2__D_1__D_3.AlignModesWith( modes_0_1, ztemp3__D_0__D_1__D_2__D_3, modes_0_2 );
	tempShape = z_ai__D_0_1__D_2_3.Shape();
	tempShape.push_back( g.Shape()[1] );
	tempShape.push_back( g.Shape()[3] );
	z_ai__D_0__D_2__D_1__D_3.ResizeTo( tempShape );

	Permute( ztemp3__D_0__D_1__D_2__D_3, ztemp3_perm0213__D_0__D_2__D_1__D_3 );
	ztemp3__D_0__D_1__D_2__D_3.EmptyData();

	   // 1.0 * ztemp3[D0,D1,D2,D3]_aiem * H_me[D3,D1]_em + 0.0 * z_ai[D0,D2,D1,D3]_aiem
	LocalContract(1.0, ztemp3_perm0213__D_0__D_2__D_1__D_3.LockedTensor(), indices_aiem, false,
		H_me_perm10__D_1__D_3.LockedTensor(), indices_em, false,
		0.0, z_ai__D_0__D_2__D_1__D_3.Tensor(), indices_aiem, false);
	H_me_perm10__D_1__D_3.EmptyData();
	ztemp3_perm0213__D_0__D_2__D_1__D_3.EmptyData();

	   // z_ai[D01,D23] <- z_ai[D0,D2,D1,D3] (with SumScatter on (D1)(D3))
	z_ai__D_0_1__D_2_3.ReduceScatterUpdateRedistFrom( z_ai__D_0__D_2__D_1__D_3, 1.0, modes_3_2 );
	z_ai__D_0__D_2__D_1__D_3.EmptyData();

	z_ai__D_0__D_2__D_3__D_1.AlignModesWith( modes_0_1, ztemp4__D_0__D_1__D_2__D_3, modes_0_2 );
	tempShape = z_ai__D_0_1__D_2_3.Shape();
	tempShape.push_back( g.Shape()[3] );
	tempShape.push_back( g.Shape()[1] );
	z_ai__D_0__D_2__D_3__D_1.ResizeTo( tempShape );

	Permute( ztemp4__D_0__D_1__D_2__D_3, ztemp4_perm0231__D_0__D_2__D_3__D_1 );
	ztemp4__D_0__D_1__D_2__D_3.EmptyData();

	   // 1.0 * ztemp4[D0,D1,D2,D3]_aiem * t_fj[D3,D1]_em + 0.0 * z_ai[D0,D2,D3,D1]_aiem
	LocalContract(1.0, ztemp4_perm0231__D_0__D_2__D_3__D_1.LockedTensor(), indices_aiem, false,
		t_fj__D_3__D_1.LockedTensor(), indices_em, false,
		0.0, z_ai__D_0__D_2__D_3__D_1.Tensor(), indices_aiem, false);
	t_fj__D_3__D_1.EmptyData();
	ztemp4_perm0231__D_0__D_2__D_3__D_1.EmptyData();

	   // z_ai[D01,D23] <- z_ai[D0,D2,D3,D1] (with SumScatter on (D3)(D1))
	z_ai__D_0_1__D_2_3.ReduceScatterUpdateRedistFrom( z_ai__D_0__D_2__D_3__D_1, 1.0, modes_3_2 );
	z_ai__D_0__D_2__D_3__D_1.EmptyData();


//****
//**** (out of 1)
//**** Is real	0 shadows
	//Outputs:
	//  ztemp5__D_0__D_1__D_2__D_3
		////High water: 239116


	   // U_mnie[D1,D0,D2,D3] <- U_mnie[D0,D1,D2,D3]
	U_mnie__D_1__D_0__D_2__D_3.AlignModesWith( modes_0_1_2_3, U_mnie__D_0__D_1__D_2__D_3, modes_1_0_2_3 );
	U_mnie__D_1__D_0__D_2__D_3.PermutationRedistFrom( U_mnie__D_0__D_1__D_2__D_3, modes_0_1 );

	tempShape = U_mnie__D_0__D_1__D_2__D_3.Shape();
	ztemp5__D_0__D_1__D_2__D_3.ResizeTo( tempShape );

	YAxpPx( 2.0, U_mnie__D_0__D_1__D_2__D_3, -1.0, U_mnie__D_1__D_0__D_2__D_3, perm_1_0_2_3, ztemp5__D_0__D_1__D_2__D_3 );
	U_mnie__D_1__D_0__D_2__D_3.EmptyData();


//****
//**** (out of 1)
//**** Is real	0 shadows
	//Outputs:
	//  z_ai__D_0_1__D_2_3
		////High water: 983020


	   // ztemp5[D2,D3,D0,D1] <- ztemp5[D0,D1,D2,D3]
	ztemp5__D_2__D_3__D_0__D_1.PermutationRedistFrom( ztemp5__D_0__D_1__D_2__D_3, modes_0_2_1_3 );
	ztemp5__D_0__D_1__D_2__D_3.EmptyData();

	   // ztemp5[D2,D3,*,D1] <- ztemp5[D2,D3,D0,D1]
	ztemp5_perm2013__S__D_2__D_3__D_1.AlignModesWith( modes_0_1_3, T_bfnj__D_0__D_1__D_2__D_3, modes_2_3_1 );
	ztemp5_perm2013__S__D_2__D_3__D_1.AllGatherRedistFrom( ztemp5__D_2__D_3__D_0__D_1, modes_0 );
	ztemp5__D_2__D_3__D_0__D_1.EmptyData();

	Permute( T_bfnj__D_0__D_1__D_2__D_3, T_bfnj_perm2310__D_2__D_3__D_1__D_0 );

	z_ai_perm10342__S__D_0__D_2__D_3__D_1.AlignModesWith( modes_0, T_bfnj__D_0__D_1__D_2__D_3, modes_0 );
	tempShape = z_ai__D_0_1__D_2_3.Shape();
	tempShape.push_back( g.Shape()[1] );
	tempShape.push_back( g.Shape()[2] );
	tempShape.push_back( g.Shape()[3] );
	z_ai_perm10342__S__D_0__D_2__D_3__D_1.ResizeTo( tempShape );

	   // -1.0 * ztemp5[D2,D3,*,D1]_imne * T_bfnj[D0,D1,D2,D3]_mnea + 0.0 * z_ai[D0,*,D1,D2,D3]_iamne
	LocalContract(-1.0, ztemp5_perm2013__S__D_2__D_3__D_1.LockedTensor(), indices_imne, false,
		T_bfnj_perm2310__D_2__D_3__D_1__D_0.LockedTensor(), indices_mnea, false,
		0.0, z_ai_perm10342__S__D_0__D_2__D_3__D_1.Tensor(), indices_iamne, false);
	T_bfnj_perm2310__D_2__D_3__D_1__D_0.EmptyData();
	ztemp5_perm2013__S__D_2__D_3__D_1.EmptyData();

	   // z_ai[D01,D23] <- z_ai[D0,*,D1,D2,D3] (with SumScatter on (D1)(D2)(D3))
	z_ai__D_0_1__D_2_3.ReduceScatterUpdateRedistFrom( z_ai_perm10342__S__D_0__D_2__D_3__D_1, 1.0, modes_4_3_2 );
	z_ai_perm10342__S__D_0__D_2__D_3__D_1.EmptyData();


//****
//**** (out of 1)
//**** Is real	0 shadows
	//Outputs:
	//  z_ai__D_0_1__D_2_3
		////High water: 1582


	   // G_mi[*,D23] <- G_mi[D01,D23]
	G_mi_perm10__D_2_3__S.AlignModesWith( modes_1, z_ai__D_0_1__D_2_3, modes_1 );
	G_mi_perm10__D_2_3__S.AllGatherRedistFrom( G_mi__D_0_1__D_2_3, modes_0_1 );

	   // t_fj[D01,*] <- t_fj[D01,D23]
	t_fj_perm10__S__D_0_1.AlignModesWith( modes_0, z_ai__D_0_1__D_2_3, modes_0 );
	t_fj_perm10__S__D_0_1.AllGatherRedistFrom( t_fj__D_0_1__D_2_3, modes_2_3 );

	Permute( z_ai__D_0_1__D_2_3, z_ai_perm10__D_2_3__D_0_1 );

	   // -1.0 * G_mi[*,D23]_im * t_fj[D01,*]_ma + 1.0 * z_ai[D01,D23]_ia
	LocalContractAndLocalEliminate(-1.0, G_mi_perm10__D_2_3__S.LockedTensor(), indices_im, false,
		t_fj_perm10__S__D_0_1.LockedTensor(), indices_ma, false,
		1.0, z_ai_perm10__D_2_3__D_0_1.Tensor(), indices_ia, false);
	t_fj_perm10__S__D_0_1.EmptyData();
	G_mi_perm10__D_2_3__S.EmptyData();

	Permute( z_ai_perm10__D_2_3__D_0_1, z_ai__D_0_1__D_2_3 );
	z_ai_perm10__D_2_3__D_0_1.EmptyData();


//****




























//**** (out of 1)
//**** Is real	0 shadows
	//Outputs:
	//  Ztemp1__D_0__D_1__D_2__D_3
		////High water: 95175432

	tempShape = Z_abij__D_0__D_1__D_2__D_3.Shape();
	Ztemp1_perm1302__D_1__D_3__D_0__D_2.ResizeTo( tempShape );

	Scal( 0.0, Ztemp1_perm1302__D_1__D_3__D_0__D_2 );


//BEGINLOOPIFY
//NVARS 2
//VAR X_bmej__D_0__D_1__D_2__D_3 2 1
//VAR T_bfnj__D_0__D_1__D_2__D_3 1 2
PartitionDown(X_bmej__D_0__D_1__D_2__D_3, X_bmej_lvl0_part2T__D_0__D_1__D_2__D_3, X_bmej_lvl0_part2B__D_0__D_1__D_2__D_3, 2, 0);
PartitionDown(T_bfnj__D_0__D_1__D_2__D_3, T_bfnj_lvl0_part1T__D_0__D_1__D_2__D_3, T_bfnj_lvl0_part1B__D_0__D_1__D_2__D_3, 1, 0);
while(X_bmej_lvl0_part2T__D_0__D_1__D_2__D_3.Dimension(2) < X_bmej__D_0__D_1__D_2__D_3.Dimension(2))
{
	RepartitionDown
	( X_bmej_lvl0_part2T__D_0__D_1__D_2__D_3, X_bmej_lvl0_part2_0__D_0__D_1__D_2__D_3,
	  /**/ /**/
		X_bmej_lvl0_part2_1__D_0__D_1__D_2__D_3,
		X_bmej_lvl0_part2B__D_0__D_1__D_2__D_3,X_bmej_lvl0_part2_2__D_0__D_1__D_2__D_3, 2, 10*blkSize);
	RepartitionDown
	( T_bfnj_lvl0_part1T__D_0__D_1__D_2__D_3, T_bfnj_lvl0_part1_0__D_0__D_1__D_2__D_3,
	  /**/ /**/
		T_bfnj_lvl0_part1_1__D_0__D_1__D_2__D_3,
		T_bfnj_lvl0_part1B__D_0__D_1__D_2__D_3,T_bfnj_lvl0_part1_2__D_0__D_1__D_2__D_3, 1, 10*blkSize);
	PartitionDown(X_bmej_lvl0_part2_1__D_0__D_1__D_2__D_3, X_bmej_lvl0_part2_1_lvl1_part1T__D_0__D_1__D_2__D_3, X_bmej_lvl0_part2_1_lvl1_part1B__D_0__D_1__D_2__D_3, 1, 0);
	PartitionDown(T_bfnj_lvl0_part1_1__D_0__D_1__D_2__D_3, T_bfnj_lvl0_part1_1_lvl1_part2T__D_0__D_1__D_2__D_3, T_bfnj_lvl0_part1_1_lvl1_part2B__D_0__D_1__D_2__D_3, 2, 0);
	while(X_bmej_lvl0_part2_1_lvl1_part1T__D_0__D_1__D_2__D_3.Dimension(1) < X_bmej_lvl0_part2_1__D_0__D_1__D_2__D_3.Dimension(1))
	{
		RepartitionDown
		( X_bmej_lvl0_part2_1_lvl1_part1T__D_0__D_1__D_2__D_3, X_bmej_lvl0_part2_1_lvl1_part1_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
			X_bmej_lvl0_part2_1_lvl1_part1_1__D_0__D_1__D_2__D_3,
			X_bmej_lvl0_part2_1_lvl1_part1B__D_0__D_1__D_2__D_3,X_bmej_lvl0_part2_1_lvl1_part1_2__D_0__D_1__D_2__D_3, 1, blkSize);
		RepartitionDown
		( T_bfnj_lvl0_part1_1_lvl1_part2T__D_0__D_1__D_2__D_3, T_bfnj_lvl0_part1_1_lvl1_part2_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
			T_bfnj_lvl0_part1_1_lvl1_part2_1__D_0__D_1__D_2__D_3,
			T_bfnj_lvl0_part1_1_lvl1_part2B__D_0__D_1__D_2__D_3,T_bfnj_lvl0_part1_1_lvl1_part2_2__D_0__D_1__D_2__D_3, 2, blkSize);
/*--------------------------------------------*/
		/***************/
		/* NEWVARS
		 * DistTensor<double> X_bmej_lvl0_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> X_bmej_lvl0_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> T_bfnj_lvl0_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> T_bfnj_lvl0_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> X_bmej_lvl0_part2_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> X_bmej_lvl0_part2_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> X_bmej_lvl0_part2_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> T_bfnj_lvl0_part1_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> T_bfnj_lvl0_part1_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> T_bfnj_lvl0_part1_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> X_bmej_lvl0_part2_1_lvl1_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> X_bmej_lvl0_part2_1_lvl1_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> T_bfnj_lvl0_part1_1_lvl1_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> T_bfnj_lvl0_part1_1_lvl1_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> X_bmej_lvl0_part2_1_lvl1_part1_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> X_bmej_lvl0_part2_1_lvl1_part1_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> X_bmej_lvl0_part2_1_lvl1_part1_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> T_bfnj_lvl0_part1_1_lvl1_part2_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> T_bfnj_lvl0_part1_1_lvl1_part2_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> T_bfnj_lvl0_part1_1_lvl1_part2_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> T_bfnj_lvl0_part1_1_lvl1_part2_1__D_0__D_1__D_3__D_2( dist__D_0__D_1__D_3__D_2, g );
		 * DistTensor<double> T_bfnj_lvl0_part1_1_lvl1_part2_1_perm2103__S__S__D_0__D_2( dist__D_0__S__S__D_2, g );
		 * DistTensor<double> X_bmej_lvl0_part2_1_lvl1_part1_1__D_1__D_0__D_2__D_3( dist__D_1__D_0__D_2__D_3, g );
		 * DistTensor<double> X_bmej_lvl0_part2_1_lvl1_part1_1_perm0312__D_1__D_3__S__S( dist__D_1__S__S__D_3, g );
		   END NEWVARS */
		/***************/

			   // X_bmej[D1,D0,D2,D3] <- X_bmej[D0,D1,D2,D3]
			X_bmej_lvl0_part2_1_lvl1_part1_1__D_1__D_0__D_2__D_3.PermutationRedistFrom( X_bmej_lvl0_part2_1_lvl1_part1_1__D_0__D_1__D_2__D_3, modes_0_1 );

			   // X_bmej[D1,*,*,D3] <- X_bmej[D1,D0,D2,D3]
			X_bmej_lvl0_part2_1_lvl1_part1_1_perm0312__D_1__D_3__S__S.AlignModesWith( modes_0_3, Z_abij__D_0__D_1__D_2__D_3, modes_1_3 );
			X_bmej_lvl0_part2_1_lvl1_part1_1_perm0312__D_1__D_3__S__S.AllGatherRedistFrom( X_bmej_lvl0_part2_1_lvl1_part1_1__D_1__D_0__D_2__D_3, modes_0_2 );
			X_bmej_lvl0_part2_1_lvl1_part1_1__D_1__D_0__D_2__D_3.EmptyData();

			   // T_bfnj[D0,D1,D3,D2] <- T_bfnj[D0,D1,D2,D3]
			T_bfnj_lvl0_part1_1_lvl1_part2_1__D_0__D_1__D_3__D_2.PermutationRedistFrom( T_bfnj_lvl0_part1_1_lvl1_part2_1__D_0__D_1__D_2__D_3, modes_2_3 );

			   // T_bfnj[D0,*,*,D2] <- T_bfnj[D0,D1,D3,D2]
			T_bfnj_lvl0_part1_1_lvl1_part2_1_perm2103__S__S__D_0__D_2.AlignModesWith( modes_0_3, Z_abij__D_0__D_1__D_2__D_3, modes_0_2 );
			T_bfnj_lvl0_part1_1_lvl1_part2_1_perm2103__S__S__D_0__D_2.AllGatherRedistFrom( T_bfnj_lvl0_part1_1_lvl1_part2_1__D_0__D_1__D_3__D_2, modes_1_3 );
			T_bfnj_lvl0_part1_1_lvl1_part2_1__D_0__D_1__D_3__D_2.EmptyData();

			   // 1.0 * X_bmej[D1,*,*,D3]_bjme * T_bfnj[D0,*,*,D2]_meai + 0.0 * Ztemp1[D0,D1,D2,D3]_bjai
			LocalContractAndLocalEliminate(1.0, X_bmej_lvl0_part2_1_lvl1_part1_1_perm0312__D_1__D_3__S__S.LockedTensor(), indices_bjme, false,
				T_bfnj_lvl0_part1_1_lvl1_part2_1_perm2103__S__S__D_0__D_2.LockedTensor(), indices_meai, false,
				1.0, Ztemp1_perm1302__D_1__D_3__D_0__D_2.Tensor(), indices_bjai, false);
			T_bfnj_lvl0_part1_1_lvl1_part2_1_perm2103__S__S__D_0__D_2.EmptyData();
			X_bmej_lvl0_part2_1_lvl1_part1_1_perm0312__D_1__D_3__S__S.EmptyData();

/*--------------------------------------------*/
		SlidePartitionDown
		( X_bmej_lvl0_part2_1_lvl1_part1T__D_0__D_1__D_2__D_3, X_bmej_lvl0_part2_1_lvl1_part1_0__D_0__D_1__D_2__D_3,
			X_bmej_lvl0_part2_1_lvl1_part1_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
			X_bmej_lvl0_part2_1_lvl1_part1B__D_0__D_1__D_2__D_3,X_bmej_lvl0_part2_1_lvl1_part1_2__D_0__D_1__D_2__D_3, 1);
		SlidePartitionDown
		( T_bfnj_lvl0_part1_1_lvl1_part2T__D_0__D_1__D_2__D_3, T_bfnj_lvl0_part1_1_lvl1_part2_0__D_0__D_1__D_2__D_3,
			T_bfnj_lvl0_part1_1_lvl1_part2_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
			T_bfnj_lvl0_part1_1_lvl1_part2B__D_0__D_1__D_2__D_3,T_bfnj_lvl0_part1_1_lvl1_part2_2__D_0__D_1__D_2__D_3, 2);
	}
	SlidePartitionDown
	( X_bmej_lvl0_part2T__D_0__D_1__D_2__D_3, X_bmej_lvl0_part2_0__D_0__D_1__D_2__D_3,
		X_bmej_lvl0_part2_1__D_0__D_1__D_2__D_3,
	  /**/ /**/
		X_bmej_lvl0_part2B__D_0__D_1__D_2__D_3,X_bmej_lvl0_part2_2__D_0__D_1__D_2__D_3, 2);
	SlidePartitionDown
	( T_bfnj_lvl0_part1T__D_0__D_1__D_2__D_3, T_bfnj_lvl0_part1_0__D_0__D_1__D_2__D_3,
		T_bfnj_lvl0_part1_1__D_0__D_1__D_2__D_3,
	  /**/ /**/
		T_bfnj_lvl0_part1B__D_0__D_1__D_2__D_3,T_bfnj_lvl0_part1_2__D_0__D_1__D_2__D_3, 1);
}
//ENDLOOPIFY

	Permute( Ztemp1_perm1302__D_1__D_3__D_0__D_2, Ztemp1__D_0__D_1__D_2__D_3 );
	Ztemp1_perm1302__D_1__D_3__D_0__D_2.EmptyData();


//****
//**** (out of 1)
//**** Is real	0 shadows
	//Outputs:
	//  Zaccum__D_0__D_1__D_2__D_3
		////High water: 2723220


	   // Ztemp1[D0,D1,D3,D2] <- Ztemp1[D0,D1,D2,D3]
	Ztemp1__D_0__D_1__D_3__D_2.AlignModesWith( modes_0_1_2_3, Z_abij__D_0__D_1__D_2__D_3, modes_0_1_3_2 );
	Ztemp1__D_0__D_1__D_3__D_2.PermutationRedistFrom( Ztemp1__D_0__D_1__D_2__D_3, modes_2_3 );

	tempShape = Z_abij__D_0__D_1__D_2__D_3.Shape();
	Zaccum__D_0__D_1__D_2__D_3.ResizeTo( tempShape );

	YAxpPx( -0.5, Ztemp1__D_0__D_1__D_2__D_3, -1.0, Ztemp1__D_0__D_1__D_3__D_2, perm_0_1_3_2, Zaccum__D_0__D_1__D_2__D_3 );
	Ztemp1__D_0__D_1__D_3__D_2.EmptyData();
	Ztemp1__D_0__D_1__D_2__D_3.EmptyData();


//****
//**** (out of 1)
//**** Is real	0 shadows
	//Outputs:
	//  Ztemp2__D_0__D_1__D_2__D_3
		////High water: 2723220


	   // T_bfnj[D0,D1,D3,D2] <- T_bfnj[D0,D1,D2,D3]
	T_bfnj__D_0__D_1__D_3__D_2.AlignModesWith( modes_0_1_2_3, T_bfnj__D_0__D_1__D_2__D_3, modes_0_1_3_2 );
	T_bfnj__D_0__D_1__D_3__D_2.PermutationRedistFrom( T_bfnj__D_0__D_1__D_2__D_3, modes_2_3 );

	tempShape = T_bfnj__D_0__D_1__D_2__D_3.Shape();
	Ztemp2__D_0__D_1__D_2__D_3.ResizeTo( tempShape );

	YAxpPx( 2.0, T_bfnj__D_0__D_1__D_2__D_3, -1.0, T_bfnj__D_0__D_1__D_3__D_2, perm_0_1_3_2, Ztemp2__D_0__D_1__D_2__D_3 );
	T_bfnj__D_0__D_1__D_3__D_2.EmptyData();


//****
//**** (out of 1)
//**** Is real	0 shadows
	//Outputs:
	//  Zaccum__D_0__D_1__D_2__D_3
		////High water: 95175432

Permute( Zaccum__D_0__D_1__D_2__D_3, Zaccum_perm1302__D_1__D_3__D_0__D_2 );

//BEGINLOOPIFY
//NVARS 2
//VAR W_bmje__D_0__D_1__D_2__D_3 3 1
//VAR Ztemp2__D_0__D_1__D_2__D_3 1 3
PartitionDown(W_bmje__D_0__D_1__D_2__D_3, W_bmje_lvl0_part3T__D_0__D_1__D_2__D_3, W_bmje_lvl0_part3B__D_0__D_1__D_2__D_3, 3, 0);
PartitionDown(Ztemp2__D_0__D_1__D_2__D_3, Ztemp2_lvl0_part1T__D_0__D_1__D_2__D_3, Ztemp2_lvl0_part1B__D_0__D_1__D_2__D_3, 1, 0);
while(W_bmje_lvl0_part3T__D_0__D_1__D_2__D_3.Dimension(3) < W_bmje__D_0__D_1__D_2__D_3.Dimension(3))
{
	RepartitionDown
	( W_bmje_lvl0_part3T__D_0__D_1__D_2__D_3, W_bmje_lvl0_part3_0__D_0__D_1__D_2__D_3,
	  /**/ /**/
		W_bmje_lvl0_part3_1__D_0__D_1__D_2__D_3,
		W_bmje_lvl0_part3B__D_0__D_1__D_2__D_3,W_bmje_lvl0_part3_2__D_0__D_1__D_2__D_3, 3, 10*blkSize);
	RepartitionDown
	( Ztemp2_lvl0_part1T__D_0__D_1__D_2__D_3, Ztemp2_lvl0_part1_0__D_0__D_1__D_2__D_3,
	  /**/ /**/
		Ztemp2_lvl0_part1_1__D_0__D_1__D_2__D_3,
		Ztemp2_lvl0_part1B__D_0__D_1__D_2__D_3,Ztemp2_lvl0_part1_2__D_0__D_1__D_2__D_3, 1, 10*blkSize);
	PartitionDown(W_bmje_lvl0_part3_1__D_0__D_1__D_2__D_3, W_bmje_lvl0_part3_1_lvl1_part1T__D_0__D_1__D_2__D_3, W_bmje_lvl0_part3_1_lvl1_part1B__D_0__D_1__D_2__D_3, 1, 0);
	PartitionDown(Ztemp2_lvl0_part1_1__D_0__D_1__D_2__D_3, Ztemp2_lvl0_part1_1_lvl1_part3T__D_0__D_1__D_2__D_3, Ztemp2_lvl0_part1_1_lvl1_part3B__D_0__D_1__D_2__D_3, 3, 0);
	while(W_bmje_lvl0_part3_1_lvl1_part1T__D_0__D_1__D_2__D_3.Dimension(1) < W_bmje_lvl0_part3_1__D_0__D_1__D_2__D_3.Dimension(1))
	{
		RepartitionDown
		( W_bmje_lvl0_part3_1_lvl1_part1T__D_0__D_1__D_2__D_3, W_bmje_lvl0_part3_1_lvl1_part1_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
			W_bmje_lvl0_part3_1_lvl1_part1_1__D_0__D_1__D_2__D_3,
			W_bmje_lvl0_part3_1_lvl1_part1B__D_0__D_1__D_2__D_3,W_bmje_lvl0_part3_1_lvl1_part1_2__D_0__D_1__D_2__D_3, 1, blkSize);
		RepartitionDown
		( Ztemp2_lvl0_part1_1_lvl1_part3T__D_0__D_1__D_2__D_3, Ztemp2_lvl0_part1_1_lvl1_part3_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
			Ztemp2_lvl0_part1_1_lvl1_part3_1__D_0__D_1__D_2__D_3,
			Ztemp2_lvl0_part1_1_lvl1_part3B__D_0__D_1__D_2__D_3,Ztemp2_lvl0_part1_1_lvl1_part3_2__D_0__D_1__D_2__D_3, 3, blkSize);
/*--------------------------------------------*/
		/***************/
		/* NEWVARS
		 * DistTensor<double> W_bmje_lvl0_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> W_bmje_lvl0_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> Ztemp2_lvl0_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> Ztemp2_lvl0_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> W_bmje_lvl0_part3_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> W_bmje_lvl0_part3_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> W_bmje_lvl0_part3_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> Ztemp2_lvl0_part1_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> Ztemp2_lvl0_part1_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> Ztemp2_lvl0_part1_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> W_bmje_lvl0_part3_1_lvl1_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> W_bmje_lvl0_part3_1_lvl1_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> Ztemp2_lvl0_part1_1_lvl1_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> Ztemp2_lvl0_part1_1_lvl1_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> W_bmje_lvl0_part3_1_lvl1_part1_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> W_bmje_lvl0_part3_1_lvl1_part1_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> W_bmje_lvl0_part3_1_lvl1_part1_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> Ztemp2_lvl0_part1_1_lvl1_part3_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> Ztemp2_lvl0_part1_1_lvl1_part3_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> Ztemp2_lvl0_part1_1_lvl1_part3_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> Ztemp2_lvl0_part1_1_lvl1_part3_1_perm3102__S__S__D_0__D_2( dist__D_0__S__D_2__S, g );
		 * DistTensor<double> W_bmje_lvl0_part3_1_lvl1_part1_1__D_1__D_0__D_3__D_2( dist__D_1__D_0__D_3__D_2, g );
		 * DistTensor<double> W_bmje_lvl0_part3_1_lvl1_part1_1_perm0213__D_1__D_3__S__S( dist__D_1__S__D_3__S, g );
		   END NEWVARS */
		/***************/

			   // Ztemp2[D0,*,D2,*] <- Ztemp2[D0,D1,D2,D3]
			Ztemp2_lvl0_part1_1_lvl1_part3_1_perm3102__S__S__D_0__D_2.AlignModesWith( modes_0_2, Zaccum__D_0__D_1__D_2__D_3, modes_0_2 );
			Ztemp2_lvl0_part1_1_lvl1_part3_1_perm3102__S__S__D_0__D_2.AllGatherRedistFrom( Ztemp2_lvl0_part1_1_lvl1_part3_1__D_0__D_1__D_2__D_3, modes_1_3 );
//			Ztemp2_lvl0_part1_1_lvl1_part3_1__D_0__D_1__D_2__D_3.EmptyData();

			   // W_bmje[D1,D0,D3,D2] <- W_bmje[D0,D1,D2,D3]
			W_bmje_lvl0_part3_1_lvl1_part1_1__D_1__D_0__D_3__D_2.PermutationRedistFrom( W_bmje_lvl0_part3_1_lvl1_part1_1__D_0__D_1__D_2__D_3, modes_0_1_2_3 );

			   // W_bmje[D1,*,D3,*] <- W_bmje[D1,D0,D3,D2]
			W_bmje_lvl0_part3_1_lvl1_part1_1_perm0213__D_1__D_3__S__S.AlignModesWith( modes_0_2, Zaccum__D_0__D_1__D_2__D_3, modes_1_3 );
			W_bmje_lvl0_part3_1_lvl1_part1_1_perm0213__D_1__D_3__S__S.AllGatherRedistFrom( W_bmje_lvl0_part3_1_lvl1_part1_1__D_1__D_0__D_3__D_2, modes_0_2 );
			W_bmje_lvl0_part3_1_lvl1_part1_1__D_1__D_0__D_3__D_2.EmptyData();

			   // 0.5 * W_bmje[D1,*,D3,*]_bjme * Ztemp2[D0,*,D2,*]_meai + 1.0 * Zaccum[D0,D1,D2,D3]_bjai
			LocalContractAndLocalEliminate(0.5, W_bmje_lvl0_part3_1_lvl1_part1_1_perm0213__D_1__D_3__S__S.LockedTensor(), indices_bjme, false,
				Ztemp2_lvl0_part1_1_lvl1_part3_1_perm3102__S__S__D_0__D_2.LockedTensor(), indices_meai, false,
				1.0, Zaccum_perm1302__D_1__D_3__D_0__D_2.Tensor(), indices_bjai, false);
			Ztemp2_lvl0_part1_1_lvl1_part3_1_perm3102__S__S__D_0__D_2.EmptyData();
			W_bmje_lvl0_part3_1_lvl1_part1_1_perm0213__D_1__D_3__S__S.EmptyData();

/*--------------------------------------------*/
		SlidePartitionDown
		( W_bmje_lvl0_part3_1_lvl1_part1T__D_0__D_1__D_2__D_3, W_bmje_lvl0_part3_1_lvl1_part1_0__D_0__D_1__D_2__D_3,
			W_bmje_lvl0_part3_1_lvl1_part1_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
			W_bmje_lvl0_part3_1_lvl1_part1B__D_0__D_1__D_2__D_3,W_bmje_lvl0_part3_1_lvl1_part1_2__D_0__D_1__D_2__D_3, 1);
		SlidePartitionDown
		( Ztemp2_lvl0_part1_1_lvl1_part3T__D_0__D_1__D_2__D_3, Ztemp2_lvl0_part1_1_lvl1_part3_0__D_0__D_1__D_2__D_3,
			Ztemp2_lvl0_part1_1_lvl1_part3_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
			Ztemp2_lvl0_part1_1_lvl1_part3B__D_0__D_1__D_2__D_3,Ztemp2_lvl0_part1_1_lvl1_part3_2__D_0__D_1__D_2__D_3, 3);
	}
	SlidePartitionDown
	( W_bmje_lvl0_part3T__D_0__D_1__D_2__D_3, W_bmje_lvl0_part3_0__D_0__D_1__D_2__D_3,
		W_bmje_lvl0_part3_1__D_0__D_1__D_2__D_3,
	  /**/ /**/
		W_bmje_lvl0_part3B__D_0__D_1__D_2__D_3,W_bmje_lvl0_part3_2__D_0__D_1__D_2__D_3, 3);
	SlidePartitionDown
	( Ztemp2_lvl0_part1T__D_0__D_1__D_2__D_3, Ztemp2_lvl0_part1_0__D_0__D_1__D_2__D_3,
		Ztemp2_lvl0_part1_1__D_0__D_1__D_2__D_3,
	  /**/ /**/
		Ztemp2_lvl0_part1B__D_0__D_1__D_2__D_3,Ztemp2_lvl0_part1_2__D_0__D_1__D_2__D_3, 1);
}
//ENDLOOPIFY

	   // G_mi[*,D2] <- G_mi[D01,D23]
	G_mi_perm10__D_2__S.AlignModesWith( modes_1, Zaccum__D_0__D_1__D_2__D_3, modes_2 );
	G_mi_perm10__D_2__S.AllGatherRedistFrom( G_mi__D_0_1__D_2_3, modes_0_1_3 );

	   // T_bfnj[D0,D1,*,D3] <- T_bfnj[D0,D1,D2,D3]
	T_bfnj_perm2013__S__D_0__D_1__D_3.AlignModesWith( modes_0_1_3, Zaccum__D_0__D_1__D_2__D_3, modes_0_1_3 );
	T_bfnj_perm2013__S__D_0__D_1__D_3.AllGatherRedistFrom( T_bfnj__D_0__D_1__D_2__D_3, modes_2 );

	Permute( Zaccum_perm1302__D_1__D_3__D_0__D_2, Zaccum_perm2013__D_2__D_0__D_1__D_3 );
	Zaccum_perm1302__D_1__D_3__D_0__D_2.EmptyData();

	   // -1.0 * G_mi[*,D2]_im * T_bfnj[D0,D1,*,D3]_mabj + 1.0 * Zaccum[D0,D1,D2,D3]_iabj
	LocalContractAndLocalEliminate(-1.0, G_mi_perm10__D_2__S.LockedTensor(), indices_im, false,
		T_bfnj_perm2013__S__D_0__D_1__D_3.LockedTensor(), indices_mabj, false,
		1.0, Zaccum_perm2013__D_2__D_0__D_1__D_3.Tensor(), indices_iabj, false);
	T_bfnj_perm2013__S__D_0__D_1__D_3.EmptyData();
	G_mi_perm10__D_2__S.EmptyData();

	   // P_jimb[D2,D3,D0,D1] <- P_jimb[D0,D1,D2,D3]
	P_jimb__D_2__D_3__D_0__D_1.PermutationRedistFrom( P_jimb__D_0__D_1__D_2__D_3, modes_0_2_1_3 );

	   // P_jimb[D2,D3,*,D1] <- P_jimb[D2,D3,D0,D1]
	P_jimb_perm3012__D_1__D_2__D_3__S.AlignModesWith( modes_0_1_3, Zaccum__D_0__D_1__D_2__D_3, modes_2_3_1 );
	P_jimb_perm3012__D_1__D_2__D_3__S.AllGatherRedistFrom( P_jimb__D_2__D_3__D_0__D_1, modes_0 );
	P_jimb__D_2__D_3__D_0__D_1.EmptyData();

	   // t_fj[D01,*] <- t_fj[D01,D23]
	t_fj__D_0_1__S.AllGatherRedistFrom( t_fj__D_0_1__D_2_3, modes_2_3 );

	   // t_fj[D0,*] <- t_fj[D01,*]
	t_fj_perm10__S__D_0.AlignModesWith( modes_0, Zaccum__D_0__D_1__D_2__D_3, modes_0 );
	t_fj_perm10__S__D_0.AllGatherRedistFrom( t_fj__D_0_1__S, modes_1 );

	   // F_ae[D0,*] <- F_ae[D01,D23]
	F_ae__D_0__S.AlignModesWith( modes_0, Zaccum__D_0__D_1__D_2__D_3, modes_0 );
	F_ae__D_0__S.AllGatherRedistFrom( F_ae__D_0_1__D_2_3, modes_1_2_3 );

	   // T_bfnj[*,D1,D2,D3] <- T_bfnj[D0,D1,D2,D3]
	T_bfnj__S__D_1__D_2__D_3.AlignModesWith( modes_1_2_3, Zaccum__D_0__D_1__D_2__D_3, modes_1_2_3 );
	T_bfnj__S__D_1__D_2__D_3.AllGatherRedistFrom( T_bfnj__D_0__D_1__D_2__D_3, modes_0 );

	Permute( Zaccum_perm2013__D_2__D_0__D_1__D_3, Zaccum__D_0__D_1__D_2__D_3 );
	Zaccum_perm2013__D_2__D_0__D_1__D_3.EmptyData();

	   // 1.0 * F_ae[D0,*]_ae * T_bfnj[*,D1,D2,D3]_ebij + 1.0 * Zaccum[D0,D1,D2,D3]_abij
	LocalContractAndLocalEliminate(1.0, F_ae__D_0__S.LockedTensor(), indices_ae, false,
		T_bfnj__S__D_1__D_2__D_3.LockedTensor(), indices_ebij, false,
		1.0, Zaccum__D_0__D_1__D_2__D_3.Tensor(), indices_abij, false);
	T_bfnj__S__D_1__D_2__D_3.EmptyData();
	F_ae__D_0__S.EmptyData();

	Permute( Zaccum__D_0__D_1__D_2__D_3, Zaccum_perm1230__D_1__D_2__D_3__D_0 );

	   // -1.0 * P_jimb[D2,D3,*,D1]_bijm * t_fj[D0,*]_ma + 1.0 * Zaccum[D0,D1,D2,D3]_bija
	LocalContractAndLocalEliminate(-1.0, P_jimb_perm3012__D_1__D_2__D_3__S.LockedTensor(), indices_bijm, false,
		t_fj_perm10__S__D_0.LockedTensor(), indices_ma, false,
		1.0, Zaccum_perm1230__D_1__D_2__D_3__D_0.Tensor(), indices_bija, false);
	t_fj_perm10__S__D_0.EmptyData();
	P_jimb_perm3012__D_1__D_2__D_3__S.EmptyData();

	Permute( Zaccum_perm1230__D_1__D_2__D_3__D_0, Zaccum__D_0__D_1__D_2__D_3 );
	Zaccum_perm1230__D_1__D_2__D_3__D_0.EmptyData();

	   // t_fj[D0,*] <- t_fj[D01,*]
	t_fj__D_0__S.AlignModesWith( modes_0, r_bmfe__D_0__D_1__D_2__D_3, modes_0 );
	t_fj__D_0__S.AllGatherRedistFrom( t_fj__D_0_1__S, modes_1 );
	t_fj__D_0_1__S.EmptyData();

	Permute( r_bmfe__D_0__D_1__D_2__D_3, r_bmfe_perm2310__D_2__D_3__D_1__D_0 );

	Zaccum_perm01324__D_2__D_3__D_1__S__D_0.AlignModesWith( modes_0_1_3, r_bmfe__D_0__D_1__D_2__D_3, modes_2_3_1 );
	tempShape = Zaccum__D_0__D_1__D_2__D_3.Shape();
	tempShape.push_back( g.Shape()[0] );
	Zaccum_perm01324__D_2__D_3__D_1__S__D_0.ResizeTo( tempShape );

	   // 1.0 * r_bmfe[D0,D1,D2,D3]_abje * t_fj[D0,*]_ei + 0.0 * Zaccum[D2,D3,*,D1,D0]_abjie
	LocalContract(1.0, r_bmfe_perm2310__D_2__D_3__D_1__D_0.LockedTensor(), indices_abje, false,
		t_fj__D_0__S.LockedTensor(), indices_ei, false,
		0.0, Zaccum_perm01324__D_2__D_3__D_1__S__D_0.Tensor(), indices_abjie, false);
	t_fj__D_0__S.EmptyData();
	r_bmfe_perm2310__D_2__D_3__D_1__D_0.EmptyData();

	Zaccum_temp__D_2_0__D_3__S__D_1.AlignModesWith( modes_0_1_2_3, Zaccum__D_0__D_1__D_2__D_3, modes_0_1_2_3 );
	tempShape = Zaccum__D_0__D_1__D_2__D_3.Shape();
	Zaccum_temp__D_2_0__D_3__S__D_1.ResizeTo( tempShape );

	   // Zaccum_temp[D20,D3,*,D1] <- Zaccum[D2,D3,*,D1,D0] (with SumScatter on D0)
	Zaccum_temp__D_2_0__D_3__S__D_1.ReduceScatterRedistFrom( Zaccum_perm01324__D_2__D_3__D_1__S__D_0, 4 );
	Zaccum_perm01324__D_2__D_3__D_1__S__D_0.EmptyData();

	   // Zaccum_temp[D02,D1,*,D3] <- Zaccum_temp[D20,D3,*,D1]
	Zaccum_temp__D_0_2__D_1__S__D_3.PermutationRedistFrom( Zaccum_temp__D_2_0__D_3__S__D_1, modes_2_0_3_1 );
	Zaccum_temp__D_2_0__D_3__S__D_1.EmptyData();

	   // Zaccum_temp[D0,D1,D2,D3] <- Zaccum_temp[D02,D1,*,D3]
	Zaccum_temp__D_0__D_1__D_2__D_3.AlignModesWith( modes_0_1_2_3, Zaccum__D_0__D_1__D_2__D_3, modes_0_1_2_3 );
	Zaccum_temp__D_0__D_1__D_2__D_3.AllToAllRedistFrom( Zaccum_temp__D_0_2__D_1__S__D_3, modes_2 );
	Zaccum_temp__D_0_2__D_1__S__D_3.EmptyData();

	YxpBy( Zaccum_temp__D_0__D_1__D_2__D_3, 1.0, Zaccum__D_0__D_1__D_2__D_3 );
	Zaccum_temp__D_0__D_1__D_2__D_3.EmptyData();


//****
//**** (out of 1)
//**** Is real	0 shadows
	//Outputs:
	//  Z_abij__D_0__D_1__D_2__D_3
		////High water: 2178576


	   // Zaccum[D1,D0,D3,D2] <- Zaccum[D0,D1,D2,D3]
	Zaccum__D_1__D_0__D_3__D_2.AlignModesWith( modes_0_1_2_3, Z_abij__D_0__D_1__D_2__D_3, modes_1_0_3_2 );
	Zaccum__D_1__D_0__D_3__D_2.PermutationRedistFrom( Zaccum__D_0__D_1__D_2__D_3, modes_0_1_2_3 );

	YAxpPx( 1.0, Zaccum__D_0__D_1__D_2__D_3, 1.0, Zaccum__D_1__D_0__D_3__D_2, perm_1_0_3_2, Z_abij__D_0__D_1__D_2__D_3 );
	Zaccum__D_1__D_0__D_3__D_2.EmptyData();
	Zaccum__D_0__D_1__D_2__D_3.EmptyData();


//****
//**** (out of 1)
//**** Is real	0 shadows
	//Outputs:
	//  Z_abij__D_0__D_1__D_2__D_3
		////High water: 57907088


//BEGINLOOPIFY
//NVARS 2
//VAR Tau_efmn__D_0__D_1__D_2__D_3 2 3
//VAR Z_abij__D_0__D_1__D_2__D_3 2 3
PartitionDown(Tau_efmn__D_0__D_1__D_2__D_3, Tau_efmn_lvl0_part2T__D_0__D_1__D_2__D_3, Tau_efmn_lvl0_part2B__D_0__D_1__D_2__D_3, 2, 0);
PartitionDown(Z_abij__D_0__D_1__D_2__D_3, Z_abij_lvl0_part2T__D_0__D_1__D_2__D_3, Z_abij_lvl0_part2B__D_0__D_1__D_2__D_3, 2, 0);
while(Tau_efmn_lvl0_part2T__D_0__D_1__D_2__D_3.Dimension(2) < Tau_efmn__D_0__D_1__D_2__D_3.Dimension(2))
{
	RepartitionDown
	( Tau_efmn_lvl0_part2T__D_0__D_1__D_2__D_3, Tau_efmn_lvl0_part2_0__D_0__D_1__D_2__D_3,
	  /**/ /**/
		Tau_efmn_lvl0_part2_1__D_0__D_1__D_2__D_3,
		Tau_efmn_lvl0_part2B__D_0__D_1__D_2__D_3,Tau_efmn_lvl0_part2_2__D_0__D_1__D_2__D_3, 2, blkSize);
	RepartitionDown
	( Z_abij_lvl0_part2T__D_0__D_1__D_2__D_3, Z_abij_lvl0_part2_0__D_0__D_1__D_2__D_3,
	  /**/ /**/
		Z_abij_lvl0_part2_1__D_0__D_1__D_2__D_3,
		Z_abij_lvl0_part2B__D_0__D_1__D_2__D_3,Z_abij_lvl0_part2_2__D_0__D_1__D_2__D_3, 2, blkSize);
	PartitionDown(Tau_efmn_lvl0_part2_1__D_0__D_1__D_2__D_3, Tau_efmn_lvl0_part2_1_lvl1_part3T__D_0__D_1__D_2__D_3, Tau_efmn_lvl0_part2_1_lvl1_part3B__D_0__D_1__D_2__D_3, 3, 0);
	PartitionDown(Z_abij_lvl0_part2_1__D_0__D_1__D_2__D_3, Z_abij_lvl0_part2_1_lvl1_part3T__D_0__D_1__D_2__D_3, Z_abij_lvl0_part2_1_lvl1_part3B__D_0__D_1__D_2__D_3, 3, 0);
	while(Tau_efmn_lvl0_part2_1_lvl1_part3T__D_0__D_1__D_2__D_3.Dimension(3) < Tau_efmn_lvl0_part2_1__D_0__D_1__D_2__D_3.Dimension(3))
	{
		RepartitionDown
		( Tau_efmn_lvl0_part2_1_lvl1_part3T__D_0__D_1__D_2__D_3, Tau_efmn_lvl0_part2_1_lvl1_part3_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
			Tau_efmn_lvl0_part2_1_lvl1_part3_1__D_0__D_1__D_2__D_3,
			Tau_efmn_lvl0_part2_1_lvl1_part3B__D_0__D_1__D_2__D_3,Tau_efmn_lvl0_part2_1_lvl1_part3_2__D_0__D_1__D_2__D_3, 3, blkSize);
		RepartitionDown
		( Z_abij_lvl0_part2_1_lvl1_part3T__D_0__D_1__D_2__D_3, Z_abij_lvl0_part2_1_lvl1_part3_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
			Z_abij_lvl0_part2_1_lvl1_part3_1__D_0__D_1__D_2__D_3,
			Z_abij_lvl0_part2_1_lvl1_part3B__D_0__D_1__D_2__D_3,Z_abij_lvl0_part2_1_lvl1_part3_2__D_0__D_1__D_2__D_3, 3, blkSize);
/*--------------------------------------------*/
		/***************/
		/* NEWVARS
		 * DistTensor<double> Tau_efmn_lvl0_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> Tau_efmn_lvl0_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> Z_abij_lvl0_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> Z_abij_lvl0_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> Tau_efmn_lvl0_part2_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> Tau_efmn_lvl0_part2_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> Tau_efmn_lvl0_part2_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> Z_abij_lvl0_part2_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> Z_abij_lvl0_part2_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> Z_abij_lvl0_part2_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> Tau_efmn_lvl0_part2_1_lvl1_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> Tau_efmn_lvl0_part2_1_lvl1_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> Z_abij_lvl0_part2_1_lvl1_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> Z_abij_lvl0_part2_1_lvl1_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> Tau_efmn_lvl0_part2_1_lvl1_part3_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> Tau_efmn_lvl0_part2_1_lvl1_part3_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> Tau_efmn_lvl0_part2_1_lvl1_part3_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> Z_abij_lvl0_part2_1_lvl1_part3_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> Z_abij_lvl0_part2_1_lvl1_part3_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> Z_abij_lvl0_part2_1_lvl1_part3_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
		 * DistTensor<double> Tau_efmn_lvl0_part2_1_lvl1_part3_1__D_2__D_3__S__S( dist__D_2__D_3__S__S, g );
		 * DistTensor<double> Tau_efmn_lvl0_part2_1_lvl1_part3_1__D_2__D_3__D_0__D_1( dist__D_2__D_3__D_0__D_1, g );
		 * DistTensor<double> Z_abij_lvl0_part2_1_lvl1_part3_1__D_0__D_1__S__S__D_2__D_3( dist__D_0__D_1__S__S__D_2__D_3, g );
		   END NEWVARS */
		/***************/
			   // Tau_efmn[D2,D3,D0,D1] <- Tau_efmn[D0,D1,D2,D3]
			Tau_efmn_lvl0_part2_1_lvl1_part3_1__D_2__D_3__D_0__D_1.PermutationRedistFrom( Tau_efmn_lvl0_part2_1_lvl1_part3_1__D_0__D_1__D_2__D_3, modes_0_2_1_3 );

			   // Tau_efmn[D2,D3,*,*] <- Tau_efmn[D2,D3,D0,D1]
			Tau_efmn_lvl0_part2_1_lvl1_part3_1__D_2__D_3__S__S.AlignModesWith( modes_0_1, y_abef__D_0__D_1__D_2__D_3, modes_2_3 );
			Tau_efmn_lvl0_part2_1_lvl1_part3_1__D_2__D_3__S__S.AllGatherRedistFrom( Tau_efmn_lvl0_part2_1_lvl1_part3_1__D_2__D_3__D_0__D_1, modes_0_1 );
			Tau_efmn_lvl0_part2_1_lvl1_part3_1__D_2__D_3__D_0__D_1.EmptyData();

			Z_abij_lvl0_part2_1_lvl1_part3_1__D_0__D_1__S__S__D_2__D_3.AlignModesWith( modes_0_1, y_abef__D_0__D_1__D_2__D_3, modes_0_1 );
			tempShape = Z_abij_lvl0_part2_1_lvl1_part3_1__D_0__D_1__D_2__D_3.Shape();
			tempShape.push_back( g.Shape()[2] );
			tempShape.push_back( g.Shape()[3] );
			Z_abij_lvl0_part2_1_lvl1_part3_1__D_0__D_1__S__S__D_2__D_3.ResizeTo( tempShape );

			   // 1.0 * y_abef[D0,D1,D2,D3]_abef * Tau_efmn[D2,D3,*,*]_efij + 0.0 * Z_abij[D0,D1,*,*,D2,D3]_abijef
			LocalContract(1.0, y_abef__D_0__D_1__D_2__D_3.LockedTensor(), indices_abef, false,
				Tau_efmn_lvl0_part2_1_lvl1_part3_1__D_2__D_3__S__S.LockedTensor(), indices_efij, false,
				0.0, Z_abij_lvl0_part2_1_lvl1_part3_1__D_0__D_1__S__S__D_2__D_3.Tensor(), indices_abijef, false);
			Tau_efmn_lvl0_part2_1_lvl1_part3_1__D_2__D_3__S__S.EmptyData();

			   // Z_abij[D0,D1,D2,D3] <- Z_abij[D0,D1,*,*,D2,D3] (with SumScatter on (D2)(D3))
			Z_abij_lvl0_part2_1_lvl1_part3_1__D_0__D_1__D_2__D_3.ReduceScatterUpdateRedistFrom( Z_abij_lvl0_part2_1_lvl1_part3_1__D_0__D_1__S__S__D_2__D_3, 1.0, modes_5_4 );
			Z_abij_lvl0_part2_1_lvl1_part3_1__D_0__D_1__S__S__D_2__D_3.EmptyData();
/*--------------------------------------------*/
		SlidePartitionDown
		( Tau_efmn_lvl0_part2_1_lvl1_part3T__D_0__D_1__D_2__D_3, Tau_efmn_lvl0_part2_1_lvl1_part3_0__D_0__D_1__D_2__D_3,
			Tau_efmn_lvl0_part2_1_lvl1_part3_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
			Tau_efmn_lvl0_part2_1_lvl1_part3B__D_0__D_1__D_2__D_3,Tau_efmn_lvl0_part2_1_lvl1_part3_2__D_0__D_1__D_2__D_3, 3);
		SlidePartitionDown
		( Z_abij_lvl0_part2_1_lvl1_part3T__D_0__D_1__D_2__D_3, Z_abij_lvl0_part2_1_lvl1_part3_0__D_0__D_1__D_2__D_3,
			Z_abij_lvl0_part2_1_lvl1_part3_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
			Z_abij_lvl0_part2_1_lvl1_part3B__D_0__D_1__D_2__D_3,Z_abij_lvl0_part2_1_lvl1_part3_2__D_0__D_1__D_2__D_3, 3);
	}
	SlidePartitionDown
	( Tau_efmn_lvl0_part2T__D_0__D_1__D_2__D_3, Tau_efmn_lvl0_part2_0__D_0__D_1__D_2__D_3,
		Tau_efmn_lvl0_part2_1__D_0__D_1__D_2__D_3,
	  /**/ /**/
		Tau_efmn_lvl0_part2B__D_0__D_1__D_2__D_3,Tau_efmn_lvl0_part2_2__D_0__D_1__D_2__D_3, 2);
	SlidePartitionDown
	( Z_abij_lvl0_part2T__D_0__D_1__D_2__D_3, Z_abij_lvl0_part2_0__D_0__D_1__D_2__D_3,
		Z_abij_lvl0_part2_1__D_0__D_1__D_2__D_3,
	  /**/ /**/
		Z_abij_lvl0_part2B__D_0__D_1__D_2__D_3,Z_abij_lvl0_part2_2__D_0__D_1__D_2__D_3, 2);
}
//ENDLOOPIFY

	   // Q_mnij[*,*,D2,D3] <- Q_mnij[D0,D1,D2,D3]
	Q_mnij_perm2301__D_2__D_3__S__S.AlignModesWith( modes_2_3, Z_abij__D_0__D_1__D_2__D_3, modes_2_3 );
	Q_mnij_perm2301__D_2__D_3__S__S.AllGatherRedistFrom( Q_mnij__D_0__D_1__D_2__D_3, modes_0_1 );

	   // Tau_efmn[D0,D1,*,*] <- Tau_efmn[D0,D1,D2,D3]
	Tau_efmn_perm2301__S__S__D_0__D_1.AlignModesWith( modes_0_1, Z_abij__D_0__D_1__D_2__D_3, modes_0_1 );
	Tau_efmn_perm2301__S__S__D_0__D_1.AllGatherRedistFrom( Tau_efmn__D_0__D_1__D_2__D_3, modes_2_3 );

	Permute( Z_abij__D_0__D_1__D_2__D_3, Z_abij_perm2301__D_2__D_3__D_0__D_1 );

	   // 1.0 * Q_mnij[*,*,D2,D3]_ijmn * Tau_efmn[D0,D1,*,*]_mnab + 1.0 * Z_abij[D0,D1,D2,D3]_ijab
	LocalContractAndLocalEliminate(1.0, Q_mnij_perm2301__D_2__D_3__S__S.LockedTensor(), indices_ijmn, false,
		Tau_efmn_perm2301__S__S__D_0__D_1.LockedTensor(), indices_mnab, false,
		1.0, Z_abij_perm2301__D_2__D_3__D_0__D_1.Tensor(), indices_ijab, false);
	Tau_efmn_perm2301__S__S__D_0__D_1.EmptyData();
	Q_mnij_perm2301__D_2__D_3__S__S.EmptyData();

	Permute( Z_abij_perm2301__D_2__D_3__D_0__D_1, Z_abij__D_0__D_1__D_2__D_3 );
	Z_abij_perm2301__D_2__D_3__D_0__D_1.EmptyData();


//****
//**** (out of 1)
//**** Is real	0 shadows
	//Outputs:
	//  Z_abij__D_0__D_1__D_2__D_3
		////High water: 0


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
