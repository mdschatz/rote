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
TensorDistribution dist__S__S__D_2__D_3__D_0__D_1 = tmen::StringToTensorDist("[(),(),(2),(3),(0),(1)]");
TensorDistribution dist__S__D_3 = tmen::StringToTensorDist("[(),(3)]");
TensorDistribution dist__D_0__D_1__S__S = tmen::StringToTensorDist("[(0),(1),(),()]");
TensorDistribution dist__D_0__D_1__D_2__S = tmen::StringToTensorDist("[(0),(1),(2),()]");
TensorDistribution dist__D_0__D_1__D_2__D_3 = tmen::StringToTensorDist("[(0),(1),(2),(3)]");
TensorDistribution dist__D_1__D_0__D_2__D_3 = tmen::StringToTensorDist("[(1),(0),(2),(3)]");
TensorDistribution dist__D_1__D_0__D_3__D_2 = tmen::StringToTensorDist("[(1),(0),(3),(2)]");
TensorDistribution dist__D_0_1__D_2_3 = tmen::StringToTensorDist("[(0,1),(2,3)]");
TensorDistribution dist__D_0_1__D_3 = tmen::StringToTensorDist("[(0,1),(3)]");
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
Permutation perm_1_0_3_2;
perm_1_0_3_2.push_back(1);
perm_1_0_3_2.push_back(0);
perm_1_0_3_2.push_back(3);
perm_1_0_3_2.push_back(2);
Permutation perm_2_3_0_1;
perm_2_3_0_1.push_back(2);
perm_2_3_0_1.push_back(3);
perm_2_3_0_1.push_back(0);
perm_2_3_0_1.push_back(1);
ModeArray modes_0_1;
modes_0_1.push_back(0);
modes_0_1.push_back(1);
ModeArray modes_0_1_2;
modes_0_1_2.push_back(0);
modes_0_1_2.push_back(1);
modes_0_1_2.push_back(2);
ModeArray modes_0_1_2_3;
modes_0_1_2_3.push_back(0);
modes_0_1_2_3.push_back(1);
modes_0_1_2_3.push_back(2);
modes_0_1_2_3.push_back(3);
ModeArray modes_1;
modes_1.push_back(1);
ModeArray modes_1_0_3_2;
modes_1_0_3_2.push_back(1);
modes_1_0_3_2.push_back(0);
modes_1_0_3_2.push_back(3);
modes_1_0_3_2.push_back(2);
ModeArray modes_2_3;
modes_2_3.push_back(2);
modes_2_3.push_back(3);
ModeArray modes_3;
modes_3.push_back(3);
ModeArray modes_5_4;
modes_5_4.push_back(5);
modes_5_4.push_back(4);
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
	//Q_mnij_part0_1_part1_1[*,*,D2,D3,D0,D1]
DistTensor<double> Q_mnij_part0_1_part1_1__S__S__D_2__D_3__D_0__D_1( dist__S__S__D_2__D_3__D_0__D_1, g );
	//Q_mnij_part0_1_part1_2[D0,D1,D2,D3]
DistTensor<double> Q_mnij_part0_1_part1_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Q_mnij_part0_1_part2B[D0,D1,D2,D3]
DistTensor<double> Q_mnij_part0_1_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Q_mnij_part0_1_part2T[D0,D1,D2,D3]
DistTensor<double> Q_mnij_part0_1_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Q_mnij_part0_1_part2_0[D0,D1,D2,D3]
DistTensor<double> Q_mnij_part0_1_part2_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Q_mnij_part0_1_part2_1[D0,D1,D2,D3]
DistTensor<double> Q_mnij_part0_1_part2_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Q_mnij_part0_1_part2_2[D0,D1,D2,D3]
DistTensor<double> Q_mnij_part0_1_part2_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Q_mnij_part0_2[D0,D1,D2,D3]
DistTensor<double> Q_mnij_part0_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//Tau_efmn[D0,D1,D2,D3]
DistTensor<double> Tau_efmn__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//q_mnij[D0,D1,D2,D3]
DistTensor<double> q_mnij__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//t_fj[D01,D23]
DistTensor<double> t_fj__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//t_fj[D01,D3]
DistTensor<double> t_fj__D_0_1__D_3( dist__D_0_1__D_3, g );
	//t_fj[*,D3]
DistTensor<double> t_fj__S__D_3( dist__S__D_3, g );
	//temp1[D0,D1,D2,D3]
DistTensor<double> temp1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp1_part0B[D0,D1,D2,D3]
DistTensor<double> temp1_part0B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp1_part0T[D0,D1,D2,D3]
DistTensor<double> temp1_part0T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp1_part0_0[D0,D1,D2,D3]
DistTensor<double> temp1_part0_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp1_part0_1[D0,D1,D2,D3]
DistTensor<double> temp1_part0_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp1_part0_1_part1B[D0,D1,D2,D3]
DistTensor<double> temp1_part0_1_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp1_part0_1_part1T[D0,D1,D2,D3]
DistTensor<double> temp1_part0_1_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp1_part0_1_part1_0[D0,D1,D2,D3]
DistTensor<double> temp1_part0_1_part1_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp1_part0_1_part1_1[D0,D1,D2,D3]
DistTensor<double> temp1_part0_1_part1_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp1_part0_1_part1_2[D0,D1,D2,D3]
DistTensor<double> temp1_part0_1_part1_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp1_part0_1_part2B[D0,D1,D2,D3]
DistTensor<double> temp1_part0_1_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp1_part0_1_part2T[D0,D1,D2,D3]
DistTensor<double> temp1_part0_1_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp1_part0_1_part2_0[D0,D1,D2,D3]
DistTensor<double> temp1_part0_1_part2_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp1_part0_1_part2_1[D0,D1,D2,D3]
DistTensor<double> temp1_part0_1_part2_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp1_part0_1_part2_2[D0,D1,D2,D3]
DistTensor<double> temp1_part0_1_part2_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp1_part0_2[D0,D1,D2,D3]
DistTensor<double> temp1_part0_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp1_part1B[D0,D1,D2,D3]
DistTensor<double> temp1_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp1_part1T[D0,D1,D2,D3]
DistTensor<double> temp1_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp1_part1_0[D0,D1,D2,D3]
DistTensor<double> temp1_part1_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp1_part1_1[D0,D1,D2,D3]
DistTensor<double> temp1_part1_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp1_part1_1_part3B[D0,D1,D2,D3]
DistTensor<double> temp1_part1_1_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp1_part1_1_part3T[D0,D1,D2,D3]
DistTensor<double> temp1_part1_1_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp1_part1_1_part3_0[D0,D1,D2,D3]
DistTensor<double> temp1_part1_1_part3_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp1_part1_1_part3_1[D0,D1,D2,D3]
DistTensor<double> temp1_part1_1_part3_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp1_part1_1_part3_1[D1,D0,D2,D3]
DistTensor<double> temp1_part1_1_part3_1__D_1__D_0__D_2__D_3( dist__D_1__D_0__D_2__D_3, g );
	//temp1_part1_1_part3_1[D1,D0,D3,D2]
DistTensor<double> temp1_part1_1_part3_1__D_1__D_0__D_3__D_2( dist__D_1__D_0__D_3__D_2, g );
	//temp1_part1_1_part3_2[D0,D1,D2,D3]
DistTensor<double> temp1_part1_1_part3_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp1_part1_2[D0,D1,D2,D3]
DistTensor<double> temp1_part1_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
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
	//u_mnje_part0_1_part1_1[D0,D1,D2,*]
DistTensor<double> u_mnje_part0_1_part1_1__D_0__D_1__D_2__S( dist__D_0__D_1__D_2__S, g );
	//u_mnje_part0_1_part1_2[D0,D1,D2,D3]
DistTensor<double> u_mnje_part0_1_part1_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//u_mnje_part0_2[D0,D1,D2,D3]
DistTensor<double> u_mnje_part0_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
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
	//v_femn_part2_1_part3_1[D0,D1,*,*]
DistTensor<double> v_femn_part2_1_part3_1_perm2301__S__S__D_0__D_1( dist__D_0__D_1__S__S, g );
v_femn_part2_1_part3_1_perm2301__S__S__D_0__D_1.SetLocalPermutation( perm_2_3_0_1 );
	//v_femn_part2_1_part3_2[D0,D1,D2,D3]
DistTensor<double> v_femn_part2_1_part3_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//v_femn_part2_2[D0,D1,D2,D3]
DistTensor<double> v_femn_part2_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
// v_femn has 4 dims
//	Starting distribution: [D0,D1,D2,D3] or _D_0__D_1__D_2__D_3
ObjShape v_femn__D_0__D_1__D_2__D_3_tempShape;
v_femn__D_0__D_1__D_2__D_3_tempShape.push_back( n_v );
v_femn__D_0__D_1__D_2__D_3_tempShape.push_back( n_v );
v_femn__D_0__D_1__D_2__D_3_tempShape.push_back( n_o );
v_femn__D_0__D_1__D_2__D_3_tempShape.push_back( n_o );
v_femn__D_0__D_1__D_2__D_3.ResizeTo( v_femn__D_0__D_1__D_2__D_3_tempShape );
MakeUniform( v_femn__D_0__D_1__D_2__D_3 );
// Tau_efmn has 4 dims
//	Starting distribution: [D0,D1,D2,D3] or _D_0__D_1__D_2__D_3
ObjShape Tau_efmn__D_0__D_1__D_2__D_3_tempShape;
Tau_efmn__D_0__D_1__D_2__D_3_tempShape.push_back( n_v );
Tau_efmn__D_0__D_1__D_2__D_3_tempShape.push_back( n_v );
Tau_efmn__D_0__D_1__D_2__D_3_tempShape.push_back( n_o );
Tau_efmn__D_0__D_1__D_2__D_3_tempShape.push_back( n_o );
Tau_efmn__D_0__D_1__D_2__D_3.ResizeTo( Tau_efmn__D_0__D_1__D_2__D_3_tempShape );
MakeUniform( Tau_efmn__D_0__D_1__D_2__D_3 );
// u_mnje has 4 dims
//	Starting distribution: [D0,D1,D2,D3] or _D_0__D_1__D_2__D_3
ObjShape u_mnje__D_0__D_1__D_2__D_3_tempShape;
u_mnje__D_0__D_1__D_2__D_3_tempShape.push_back( n_o );
u_mnje__D_0__D_1__D_2__D_3_tempShape.push_back( n_o );
u_mnje__D_0__D_1__D_2__D_3_tempShape.push_back( n_o );
u_mnje__D_0__D_1__D_2__D_3_tempShape.push_back( n_v );
u_mnje__D_0__D_1__D_2__D_3.ResizeTo( u_mnje__D_0__D_1__D_2__D_3_tempShape );
MakeUniform( u_mnje__D_0__D_1__D_2__D_3 );
// t_fj has 2 dims
//	Starting distribution: [D01,D23] or _D_0_1__D_2_3
ObjShape t_fj__D_0_1__D_2_3_tempShape;
t_fj__D_0_1__D_2_3_tempShape.push_back( n_v );
t_fj__D_0_1__D_2_3_tempShape.push_back( n_o );
t_fj__D_0_1__D_2_3.ResizeTo( t_fj__D_0_1__D_2_3_tempShape );
MakeUniform( t_fj__D_0_1__D_2_3 );
// Q_mnij has 4 dims
//	Starting distribution: [D0,D1,D2,D3] or _D_0__D_1__D_2__D_3
ObjShape Q_mnij__D_0__D_1__D_2__D_3_tempShape;
Q_mnij__D_0__D_1__D_2__D_3_tempShape.push_back( n_o );
Q_mnij__D_0__D_1__D_2__D_3_tempShape.push_back( n_o );
Q_mnij__D_0__D_1__D_2__D_3_tempShape.push_back( n_o );
Q_mnij__D_0__D_1__D_2__D_3_tempShape.push_back( n_o );
Q_mnij__D_0__D_1__D_2__D_3.ResizeTo( Q_mnij__D_0__D_1__D_2__D_3_tempShape );
MakeUniform( Q_mnij__D_0__D_1__D_2__D_3 );
// q_mnij has 4 dims
//	Starting distribution: [D0,D1,D2,D3] or _D_0__D_1__D_2__D_3
ObjShape q_mnij__D_0__D_1__D_2__D_3_tempShape;
q_mnij__D_0__D_1__D_2__D_3_tempShape.push_back( n_o );
q_mnij__D_0__D_1__D_2__D_3_tempShape.push_back( n_o );
q_mnij__D_0__D_1__D_2__D_3_tempShape.push_back( n_o );
q_mnij__D_0__D_1__D_2__D_3_tempShape.push_back( n_o );
q_mnij__D_0__D_1__D_2__D_3.ResizeTo( q_mnij__D_0__D_1__D_2__D_3_tempShape );
MakeUniform( q_mnij__D_0__D_1__D_2__D_3 );
tempShape = Q_mnij__D_0__D_1__D_2__D_3.Shape();
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
check.ResizeTo(Q_mnij__D_0__D_1__D_2__D_3.Shape());
Read(v_femn__D_0__D_1__D_2__D_3, "ccsd_terms/term_v_small", BINARY_FLAT, false);
Read(u_mnje__D_0__D_1__D_2__D_3, "ccsd_terms/term_u_small", BINARY_FLAT, false);
Read(q_mnij__D_0__D_1__D_2__D_3, "ccsd_terms/term_q_small", BINARY_FLAT, false);
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
fullName << "ccsd_terms/term_Q_iter" << testIter;
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
		//  temp1__D_0__D_1__D_2__D_3
	PartitionDown(u_mnje__D_0__D_1__D_2__D_3, u_mnje_part0T__D_0__D_1__D_2__D_3, u_mnje_part0B__D_0__D_1__D_2__D_3, 0, 0);
	PartitionDown(temp1__D_0__D_1__D_2__D_3, temp1_part0T__D_0__D_1__D_2__D_3, temp1_part0B__D_0__D_1__D_2__D_3, 0, 0);
	while(temp1_part0T__D_0__D_1__D_2__D_3.Dimension(0) < temp1__D_0__D_1__D_2__D_3.Dimension(0))
	{
		RepartitionDown
		( u_mnje_part0T__D_0__D_1__D_2__D_3,  u_mnje_part0_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       u_mnje_part0_1__D_0__D_1__D_2__D_3,
		  u_mnje_part0B__D_0__D_1__D_2__D_3, u_mnje_part0_2__D_0__D_1__D_2__D_3, 0, blkSize );
		RepartitionDown
		( temp1_part0T__D_0__D_1__D_2__D_3,  temp1_part0_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       temp1_part0_1__D_0__D_1__D_2__D_3,
		  temp1_part0B__D_0__D_1__D_2__D_3, temp1_part0_2__D_0__D_1__D_2__D_3, 0, blkSize );

		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  temp1_part0_1__D_0__D_1__D_2__D_3
		PartitionDown(u_mnje_part0_1__D_0__D_1__D_2__D_3, u_mnje_part0_1_part1T__D_0__D_1__D_2__D_3, u_mnje_part0_1_part1B__D_0__D_1__D_2__D_3, 1, 0);
		PartitionDown(temp1_part0_1__D_0__D_1__D_2__D_3, temp1_part0_1_part1T__D_0__D_1__D_2__D_3, temp1_part0_1_part1B__D_0__D_1__D_2__D_3, 1, 0);
		while(temp1_part0_1_part1T__D_0__D_1__D_2__D_3.Dimension(1) < temp1_part0_1__D_0__D_1__D_2__D_3.Dimension(1))
		{
			RepartitionDown
			( u_mnje_part0_1_part1T__D_0__D_1__D_2__D_3,  u_mnje_part0_1_part1_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       u_mnje_part0_1_part1_1__D_0__D_1__D_2__D_3,
			  u_mnje_part0_1_part1B__D_0__D_1__D_2__D_3, u_mnje_part0_1_part1_2__D_0__D_1__D_2__D_3, 1, blkSize );
			RepartitionDown
			( temp1_part0_1_part1T__D_0__D_1__D_2__D_3,  temp1_part0_1_part1_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       temp1_part0_1_part1_1__D_0__D_1__D_2__D_3,
			  temp1_part0_1_part1B__D_0__D_1__D_2__D_3, temp1_part0_1_part1_2__D_0__D_1__D_2__D_3, 1, blkSize );

			   // u_mnje_part0_1_part1_1[D0,D1,D2,*] <- u_mnje_part0_1_part1_1[D0,D1,D2,D3]
			u_mnje_part0_1_part1_1__D_0__D_1__D_2__S.AlignModesWith( modes_0_1_2, temp1_part0_1_part1_1__D_0__D_1__D_2__D_3, modes_0_1_2 );
			u_mnje_part0_1_part1_1__D_0__D_1__D_2__S.AllGatherRedistFrom( u_mnje_part0_1_part1_1__D_0__D_1__D_2__D_3, modes_3 );
			   // t_fj[D01,D3] <- t_fj[D01,D23]
			t_fj__D_0_1__D_3.AlignModesWith( modes_1, temp1_part0_1_part1_1__D_0__D_1__D_2__D_3, modes_3 );
			t_fj__D_0_1__D_3.AllToAllRedistFrom( t_fj__D_0_1__D_2_3, modes_2_3 );
			   // t_fj[*,D3] <- t_fj[D01,D3]
			t_fj__S__D_3.AlignModesWith( modes_1, temp1_part0_1_part1_1__D_0__D_1__D_2__D_3, modes_3 );
			t_fj__S__D_3.AllGatherRedistFrom( t_fj__D_0_1__D_3, modes_0_1 );
			   // 1.0 * u_mnje_part0_1_part1_1[D0,D1,D2,*]_mnie * t_fj[*,D3]_ej + 0.0 * temp1_part0_1_part1_1[D0,D1,D2,D3]_mnij
			LocalContractAndLocalEliminate(1.0, u_mnje_part0_1_part1_1__D_0__D_1__D_2__S.LockedTensor(), indices_mnie, false,
				t_fj__S__D_3.LockedTensor(), indices_ej, false,
				0.0, temp1_part0_1_part1_1__D_0__D_1__D_2__D_3.Tensor(), indices_mnij, false);
			t_fj__S__D_3.EmptyData();
			u_mnje_part0_1_part1_1__D_0__D_1__D_2__S.EmptyData();
			t_fj__D_0_1__D_3.EmptyData();

			SlidePartitionDown
			( u_mnje_part0_1_part1T__D_0__D_1__D_2__D_3,  u_mnje_part0_1_part1_0__D_0__D_1__D_2__D_3,
			       u_mnje_part0_1_part1_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  u_mnje_part0_1_part1B__D_0__D_1__D_2__D_3, u_mnje_part0_1_part1_2__D_0__D_1__D_2__D_3, 1 );
			SlidePartitionDown
			( temp1_part0_1_part1T__D_0__D_1__D_2__D_3,  temp1_part0_1_part1_0__D_0__D_1__D_2__D_3,
			       temp1_part0_1_part1_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  temp1_part0_1_part1B__D_0__D_1__D_2__D_3, temp1_part0_1_part1_2__D_0__D_1__D_2__D_3, 1 );

		}
		//****

		SlidePartitionDown
		( u_mnje_part0T__D_0__D_1__D_2__D_3,  u_mnje_part0_0__D_0__D_1__D_2__D_3,
		       u_mnje_part0_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  u_mnje_part0B__D_0__D_1__D_2__D_3, u_mnje_part0_2__D_0__D_1__D_2__D_3, 0 );
		SlidePartitionDown
		( temp1_part0T__D_0__D_1__D_2__D_3,  temp1_part0_0__D_0__D_1__D_2__D_3,
		       temp1_part0_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  temp1_part0B__D_0__D_1__D_2__D_3, temp1_part0_2__D_0__D_1__D_2__D_3, 0 );

	}
	Print(temp1__D_0__D_1__D_2__D_3, "temp1");
	u_mnje__D_0__D_1__D_2__D_3.EmptyData();
	t_fj__D_0_1__D_2_3.EmptyData();
	u_mnje__D_0__D_1__D_2__D_3.EmptyData();
	t_fj__D_0_1__D_2_3.EmptyData();
	//****
	//**** (out of 1)
	//**** Is real	0 shadows
		//Outputs:
		//  Q_mnij__D_0__D_1__D_2__D_3
	PartitionDown(v_femn__D_0__D_1__D_2__D_3, v_femn_part2T__D_0__D_1__D_2__D_3, v_femn_part2B__D_0__D_1__D_2__D_3, 2, 0);
	PartitionDown(temp1__D_0__D_1__D_2__D_3, temp1_part0T__D_0__D_1__D_2__D_3, temp1_part0B__D_0__D_1__D_2__D_3, 0, 0);
	PartitionDown(temp1__D_0__D_1__D_2__D_3, temp1_part1T__D_0__D_1__D_2__D_3, temp1_part1B__D_0__D_1__D_2__D_3, 1, 0);
	PartitionDown(Q_mnij__D_0__D_1__D_2__D_3, Q_mnij_part0T__D_0__D_1__D_2__D_3, Q_mnij_part0B__D_0__D_1__D_2__D_3, 0, 0);
	while(Q_mnij_part0T__D_0__D_1__D_2__D_3.Dimension(0) < Q_mnij__D_0__D_1__D_2__D_3.Dimension(0))
	{
		RepartitionDown
		( v_femn_part2T__D_0__D_1__D_2__D_3,  v_femn_part2_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       v_femn_part2_1__D_0__D_1__D_2__D_3,
		  v_femn_part2B__D_0__D_1__D_2__D_3, v_femn_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );
		RepartitionDown
		( temp1_part0T__D_0__D_1__D_2__D_3,  temp1_part0_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       temp1_part0_1__D_0__D_1__D_2__D_3,
		  temp1_part0B__D_0__D_1__D_2__D_3, temp1_part0_2__D_0__D_1__D_2__D_3, 0, blkSize );
		RepartitionDown
		( temp1_part1T__D_0__D_1__D_2__D_3,  temp1_part1_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       temp1_part1_1__D_0__D_1__D_2__D_3,
		  temp1_part1B__D_0__D_1__D_2__D_3, temp1_part1_2__D_0__D_1__D_2__D_3, 1, blkSize );
		RepartitionDown
		( Q_mnij_part0T__D_0__D_1__D_2__D_3,  Q_mnij_part0_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       Q_mnij_part0_1__D_0__D_1__D_2__D_3,
		  Q_mnij_part0B__D_0__D_1__D_2__D_3, Q_mnij_part0_2__D_0__D_1__D_2__D_3, 0, blkSize );

		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  Q_mnij_part0_1__D_0__D_1__D_2__D_3
		PartitionDown(temp1_part0_1__D_0__D_1__D_2__D_3, temp1_part0_1_part2T__D_0__D_1__D_2__D_3, temp1_part0_1_part2B__D_0__D_1__D_2__D_3, 2, 0);
		PartitionDown(temp1_part1_1__D_0__D_1__D_2__D_3, temp1_part1_1_part3T__D_0__D_1__D_2__D_3, temp1_part1_1_part3B__D_0__D_1__D_2__D_3, 3, 0);
		PartitionDown(Q_mnij_part0_1__D_0__D_1__D_2__D_3, Q_mnij_part0_1_part2T__D_0__D_1__D_2__D_3, Q_mnij_part0_1_part2B__D_0__D_1__D_2__D_3, 2, 0);
		while(Q_mnij_part0_1_part2T__D_0__D_1__D_2__D_3.Dimension(2) < Q_mnij_part0_1__D_0__D_1__D_2__D_3.Dimension(2))
		{
			RepartitionDown
			( temp1_part0_1_part2T__D_0__D_1__D_2__D_3,  temp1_part0_1_part2_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       temp1_part0_1_part2_1__D_0__D_1__D_2__D_3,
			  temp1_part0_1_part2B__D_0__D_1__D_2__D_3, temp1_part0_1_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );
			RepartitionDown
			( temp1_part1_1_part3T__D_0__D_1__D_2__D_3,  temp1_part1_1_part3_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       temp1_part1_1_part3_1__D_0__D_1__D_2__D_3,
			  temp1_part1_1_part3B__D_0__D_1__D_2__D_3, temp1_part1_1_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );
			RepartitionDown
			( Q_mnij_part0_1_part2T__D_0__D_1__D_2__D_3,  Q_mnij_part0_1_part2_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       Q_mnij_part0_1_part2_1__D_0__D_1__D_2__D_3,
			  Q_mnij_part0_1_part2B__D_0__D_1__D_2__D_3, Q_mnij_part0_1_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );

			   // temp1_part1_1_part3_1[D1,D0,D2,D3] <- temp1_part1_1_part3_1[D0,D1,D2,D3]
			temp1_part1_1_part3_1__D_1__D_0__D_2__D_3.AlignModesWith( modes_0_1_2_3, temp1_part0_1_part2_1__D_0__D_1__D_2__D_3, modes_1_0_3_2 );
			temp1_part1_1_part3_1__D_1__D_0__D_2__D_3.AllToAllRedistFrom( temp1_part1_1_part3_1__D_0__D_1__D_2__D_3, modes_0_1 );
			   // temp1_part1_1_part3_1[D1,D0,D3,D2] <- temp1_part1_1_part3_1[D1,D0,D2,D3]
			temp1_part1_1_part3_1__D_1__D_0__D_3__D_2.AlignModesWith( modes_0_1_2_3, temp1_part0_1_part2_1__D_0__D_1__D_2__D_3, modes_1_0_3_2 );
			temp1_part1_1_part3_1__D_1__D_0__D_3__D_2.AllToAllRedistFrom( temp1_part1_1_part3_1__D_1__D_0__D_2__D_3, modes_2_3 );
			YAxpPx( 1.0, temp1_part0_1_part2_1__D_0__D_1__D_2__D_3, 1.0, temp1_part1_1_part3_1__D_1__D_0__D_3__D_2, perm_1_0_3_2, Q_mnij_part0_1_part2_1__D_0__D_1__D_2__D_3 );
			temp1_part1_1_part3_1__D_1__D_0__D_3__D_2.EmptyData();
			temp1_part1_1_part3_1__D_1__D_0__D_2__D_3.EmptyData();

			SlidePartitionDown
			( temp1_part0_1_part2T__D_0__D_1__D_2__D_3,  temp1_part0_1_part2_0__D_0__D_1__D_2__D_3,
			       temp1_part0_1_part2_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  temp1_part0_1_part2B__D_0__D_1__D_2__D_3, temp1_part0_1_part2_2__D_0__D_1__D_2__D_3, 2 );
			SlidePartitionDown
			( temp1_part1_1_part3T__D_0__D_1__D_2__D_3,  temp1_part1_1_part3_0__D_0__D_1__D_2__D_3,
			       temp1_part1_1_part3_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  temp1_part1_1_part3B__D_0__D_1__D_2__D_3, temp1_part1_1_part3_2__D_0__D_1__D_2__D_3, 3 );
			SlidePartitionDown
			( Q_mnij_part0_1_part2T__D_0__D_1__D_2__D_3,  Q_mnij_part0_1_part2_0__D_0__D_1__D_2__D_3,
			       Q_mnij_part0_1_part2_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  Q_mnij_part0_1_part2B__D_0__D_1__D_2__D_3, Q_mnij_part0_1_part2_2__D_0__D_1__D_2__D_3, 2 );

		}
		Print(Q_mnij__D_0__D_1__D_2__D_3, "Q after p");
		//****
		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  Q_mnij_part0_1__D_0__D_1__D_2__D_3
		PartitionDown(v_femn_part2_1__D_0__D_1__D_2__D_3, v_femn_part2_1_part3T__D_0__D_1__D_2__D_3, v_femn_part2_1_part3B__D_0__D_1__D_2__D_3, 3, 0);
		PartitionDown(Q_mnij_part0_1__D_0__D_1__D_2__D_3, Q_mnij_part0_1_part1T__D_0__D_1__D_2__D_3, Q_mnij_part0_1_part1B__D_0__D_1__D_2__D_3, 1, 0);
		while(Q_mnij_part0_1_part1T__D_0__D_1__D_2__D_3.Dimension(1) < Q_mnij_part0_1__D_0__D_1__D_2__D_3.Dimension(1))
		{
			RepartitionDown
			( v_femn_part2_1_part3T__D_0__D_1__D_2__D_3,  v_femn_part2_1_part3_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       v_femn_part2_1_part3_1__D_0__D_1__D_2__D_3,
			  v_femn_part2_1_part3B__D_0__D_1__D_2__D_3, v_femn_part2_1_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );
			RepartitionDown
			( Q_mnij_part0_1_part1T__D_0__D_1__D_2__D_3,  Q_mnij_part0_1_part1_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       Q_mnij_part0_1_part1_1__D_0__D_1__D_2__D_3,
			  Q_mnij_part0_1_part1B__D_0__D_1__D_2__D_3, Q_mnij_part0_1_part1_2__D_0__D_1__D_2__D_3, 1, blkSize );

			tempShape = Q_mnij_part0_1_part1_1__D_0__D_1__D_2__D_3.Shape();
			tempShape.push_back( g.Shape()[0] );
			tempShape.push_back( g.Shape()[1] );
			Q_mnij_part0_1_part1_1__S__S__D_2__D_3__D_0__D_1.ResizeTo( tempShape );
			   // v_femn_part2_1_part3_1[D0,D1,*,*] <- v_femn_part2_1_part3_1[D0,D1,D2,D3]
			v_femn_part2_1_part3_1_perm2301__S__S__D_0__D_1.AlignModesWith( modes_0_1, Tau_efmn__D_0__D_1__D_2__D_3, modes_0_1 );
			v_femn_part2_1_part3_1_perm2301__S__S__D_0__D_1.AllGatherRedistFrom( v_femn_part2_1_part3_1__D_0__D_1__D_2__D_3, modes_2_3 );
			   // 1.0 * v_femn_part2_1_part3_1[D0,D1,*,*]_mnef * Tau_efmn[D0,D1,D2,D3]_efij + 0.0 * Q_mnij_part0_1_part1_1[*,*,D2,D3,D0,D1]_mnijef
			LocalContract(1.0, v_femn_part2_1_part3_1_perm2301__S__S__D_0__D_1.LockedTensor(), indices_mnef, false,
				Tau_efmn__D_0__D_1__D_2__D_3.LockedTensor(), indices_efij, false,
				0.0, Q_mnij_part0_1_part1_1__S__S__D_2__D_3__D_0__D_1.Tensor(), indices_mnijef, false);
			   // Q_mnij_part0_1_part1_1[D0,D1,D2,D3] <- Q_mnij_part0_1_part1_1[*,*,D2,D3,D0,D1] (with SumScatter on (D0)(D1))
			Q_mnij_part0_1_part1_1__D_0__D_1__D_2__D_3.ReduceScatterUpdateRedistFrom( Q_mnij_part0_1_part1_1__S__S__D_2__D_3__D_0__D_1, 1.0, modes_5_4 );
			Q_mnij_part0_1_part1_1__S__S__D_2__D_3__D_0__D_1.EmptyData();
			v_femn_part2_1_part3_1_perm2301__S__S__D_0__D_1.EmptyData();

			SlidePartitionDown
			( v_femn_part2_1_part3T__D_0__D_1__D_2__D_3,  v_femn_part2_1_part3_0__D_0__D_1__D_2__D_3,
			       v_femn_part2_1_part3_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  v_femn_part2_1_part3B__D_0__D_1__D_2__D_3, v_femn_part2_1_part3_2__D_0__D_1__D_2__D_3, 3 );
			SlidePartitionDown
			( Q_mnij_part0_1_part1T__D_0__D_1__D_2__D_3,  Q_mnij_part0_1_part1_0__D_0__D_1__D_2__D_3,
			       Q_mnij_part0_1_part1_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  Q_mnij_part0_1_part1B__D_0__D_1__D_2__D_3, Q_mnij_part0_1_part1_2__D_0__D_1__D_2__D_3, 1 );

		}
		//****

		SlidePartitionDown
		( v_femn_part2T__D_0__D_1__D_2__D_3,  v_femn_part2_0__D_0__D_1__D_2__D_3,
		       v_femn_part2_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  v_femn_part2B__D_0__D_1__D_2__D_3, v_femn_part2_2__D_0__D_1__D_2__D_3, 2 );
		SlidePartitionDown
		( Q_mnij_part0T__D_0__D_1__D_2__D_3,  Q_mnij_part0_0__D_0__D_1__D_2__D_3,
		       Q_mnij_part0_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  Q_mnij_part0B__D_0__D_1__D_2__D_3, Q_mnij_part0_2__D_0__D_1__D_2__D_3, 0 );
		SlidePartitionDown
		( temp1_part0T__D_0__D_1__D_2__D_3,  temp1_part0_0__D_0__D_1__D_2__D_3,
		       temp1_part0_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  temp1_part0B__D_0__D_1__D_2__D_3, temp1_part0_2__D_0__D_1__D_2__D_3, 0 );
		SlidePartitionDown
		( temp1_part1T__D_0__D_1__D_2__D_3,  temp1_part1_0__D_0__D_1__D_2__D_3,
		       temp1_part1_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  temp1_part1B__D_0__D_1__D_2__D_3, temp1_part1_2__D_0__D_1__D_2__D_3, 1 );

	}
	Print(Q_mnij__D_0__D_1__D_2__D_3, "Q after tau");
	v_femn__D_0__D_1__D_2__D_3.EmptyData();
	Tau_efmn__D_0__D_1__D_2__D_3.EmptyData();
	temp1__D_0__D_1__D_2__D_3.EmptyData();
	temp1__D_0__D_1__D_2__D_3.EmptyData();
	v_femn__D_0__D_1__D_2__D_3.EmptyData();
	Tau_efmn__D_0__D_1__D_2__D_3.EmptyData();
	temp1__D_0__D_1__D_2__D_3.EmptyData();
	temp1__D_0__D_1__D_2__D_3.EmptyData();
	//****

v_femn__D_0__D_1__D_2__D_3.EmptyData();
Tau_efmn__D_0__D_1__D_2__D_3.EmptyData();
u_mnje__D_0__D_1__D_2__D_3.EmptyData();
t_fj__D_0_1__D_2_3.EmptyData();
temp1__D_0__D_1__D_2__D_3.EmptyData();
//****
//**** (out of 1)

	YAxpy( 1.0, q_mnij__D_0__D_1__D_2__D_3, Q_mnij__D_0__D_1__D_2__D_3 );

	q_mnij__D_0__D_1__D_2__D_3.EmptyData();

q_mnij__D_0__D_1__D_2__D_3.EmptyData();
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
    Diff(check, Q_mnij__D_0__D_1__D_2__D_3, diff);
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


