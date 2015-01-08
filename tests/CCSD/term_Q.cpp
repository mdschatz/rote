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
}

template<typename T>
void Load_Tensor_Helper(ifstream& fid, Mode mode, const Location& curLoc,
        DistTensor<T>& A) {
    Unsigned i;
    Unsigned dim = A.Dimension(mode);
    Location newCurLoc = curLoc;
    for (i = 0; i < dim; i++) {
        newCurLoc[mode] = i;
        if (mode == 0) {
            char* valS = new char[8];
            fid.read(valS, 8);
            double val = *reinterpret_cast<double*>(valS);
//          std::cout << "val: " << val << std::endl;
//          std::memcpy(&val, &(valS[0]), sizeof(double));
//          printf("newVal %.03f\n", val);
            A.Set(newCurLoc, val);
        } else {
            if (mode == 3)
                printf("loading mode 3 index: %d\n", i);
            Load_Tensor_Helper(fid, mode - 1, newCurLoc, A);
        }
    }
}

template<typename T>
void Load_Tensor(DistTensor<T>& A, const std::string& filename) {
    printf("Loading tensor\n");
    PrintVector(A.Shape(), "of size");
    Unsigned order = A.Order();
    ifstream fid;
    fid.open(filename.c_str(), std::ifstream::binary);
    //Skip 4 bytes of Fortran
    fid.seekg(4);
    Location zeros(order, 0);
    Load_Tensor_Helper(fid, order - 1, zeros, A);
    fid.close();
}

template<typename T>
void Load_Tensor_efgh_Helper(ifstream& fid, Mode mode, const Location& curLoc,
        DistTensor<T>& A) {
    Unsigned i;
    Unsigned dim = A.Dimension(mode);
    Location newCurLoc = curLoc;
    for (i = 0; i < dim; i++) {
        if (mode == 3)
            newCurLoc[2] = i;
        else if (mode == 2)
            newCurLoc[3] = i;
        else
            newCurLoc[mode] = i;
        if (mode == 0) {
            char* valS = new char[8];
            fid.read(valS, 8);
            double val = *reinterpret_cast<double*>(valS);
//          PrintVector(newCurLoc, "Setting loc");
//          std::cout << "to val: " << val << std::endl;
//          std::cout << "val: " << val << std::endl;
//          std::memcpy(&val, &(valS[0]), sizeof(double));
//          printf("newVal %.03f\n", val);
            A.Set(newCurLoc, -val);
        } else {
            Load_Tensor_efgh_Helper(fid, mode - 1, newCurLoc, A);
        }
    }
}

template<typename T>
void Load_Tensor_efgh(DistTensor<T>& A, const std::string& filename) {
    printf("Loading tensor\n");
    PrintVector(A.Shape(), "of size");
    Unsigned order = A.Order();
    ifstream fid;
    fid.open(filename.c_str(), std::ifstream::binary);
    //Skip 4 bytes of Fortran
    fid.seekg(4);
    Location zeros(order, 0);
    Load_Tensor_efgh_Helper(fid, order - 1, zeros, A);
    fid.close();
}

template<typename T>
void Load_Tensor_aijb_Helper(ifstream& fid, Mode mode, const Location& curLoc,
        DistTensor<T>& A) {
    Unsigned i;
    Unsigned dim;
    if (mode == 3)
        dim = A.Dimension(0);
    else if (mode == 2)
        dim = A.Dimension(2);
    else if (mode == 1)
        dim = A.Dimension(3);
    else if (mode == 0)
        dim = A.Dimension(1);
    Location newCurLoc = curLoc;
    for (i = 0; i < dim; i++) {
        if (mode == 3)
            newCurLoc[0] = i;
        else if (mode == 2)
            newCurLoc[2] = i;
        else if (mode == 1)
            newCurLoc[3] = i;
        else if (mode == 0)
            newCurLoc[1] = i;
        if (mode == 0) {
            char* valS = new char[8];
            fid.read(valS, 8);
            double val = *reinterpret_cast<double*>(valS);
//          PrintVector(newCurLoc, "Setting loc");
//          std::cout << "to val: " << val << std::endl;
//          std::cout << "val: " << val << std::endl;
//          std::memcpy(&val, &(valS[0]), sizeof(double));
//          printf("newVal %.03f\n", val);
            A.Set(newCurLoc, val);
        } else {
            Load_Tensor_aijb_Helper(fid, mode - 1, newCurLoc, A);
        }
    }
}

template<typename T>
void Load_Tensor_aijb(DistTensor<T>& A, const std::string& filename) {
    printf("Loading tensor\n");
    PrintVector(A.Shape(), "of size");
    Unsigned order = A.Order();
    ifstream fid;
    fid.open(filename.c_str(), std::ifstream::binary);
    //Skip 4 bytes of Fortran
    fid.seekg(4);
    Location zeros(order, 0);
    Load_Tensor_aijb_Helper(fid, order - 1, zeros, A);
    fid.close();
}

template<typename T>
void Form_D_abij_Helper(const DistTensor<T>& epsilonA,
        const DistTensor<T>& epsilonB, Mode mode, const Location& loc,
        DistTensor<T>& D_abij) {
    Unsigned i;
    Unsigned dim = D_abij.Dimension(mode);
    Location newCurLoc = loc;
    for (i = 0; i < dim; i++) {
        newCurLoc[mode] = i;
        if (mode == 0) {
            Location epsLoc(1);
            epsLoc[0] = newCurLoc[0];
            double e_a = epsilonA.Get(epsLoc);

            epsLoc[0] = newCurLoc[1];
            double e_b = epsilonA.Get(epsLoc);

            epsLoc[0] = newCurLoc[2];
            double e_i = epsilonB.Get(epsLoc);

            epsLoc[0] = newCurLoc[3];
            double e_j = epsilonB.Get(epsLoc);
            double val = -1.0 / (e_a + e_b - e_i - e_j);
            D_abij.Set(newCurLoc, val);
        } else {
            Form_D_abij_Helper(epsilonA, epsilonB, mode - 1, newCurLoc, D_abij);
        }
    }
}

template<typename T>
void Form_D_abij(const DistTensor<T>& epsilonA, const DistTensor<T>& epsilonB,
        DistTensor<T>& D_abij) {
    Unsigned order = D_abij.Order();

    Location zeros(order, 0);
    Form_D_abij_Helper(epsilonA, epsilonB, order - 1, zeros, D_abij);
}

template<typename T>
void DistTensorTest(const Grid& g, Unsigned n_o, Unsigned n_v,
        Unsigned blkSize) {
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
TensorDistribution dist__D_0__D_1__D_3__D_2 = tmen::StringToTensorDist("[(0),(1),(3),(2)]");
TensorDistribution dist__D_0__D_2 = tmen::StringToTensorDist("[(0),(2)]");
TensorDistribution dist__D_1__D_0__D_3__D_2 = tmen::StringToTensorDist("[(1),(0),(3),(2)]");
TensorDistribution dist__D_1__D_3 = tmen::StringToTensorDist("[(1),(3)]");
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
Permutation perm_0_2_1_3;
perm_0_2_1_3.push_back(0);
perm_0_2_1_3.push_back(2);
perm_0_2_1_3.push_back(1);
perm_0_2_1_3.push_back(3);
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
ModeArray modes_0_2;
modes_0_2.push_back(0);
modes_0_2.push_back(2);
ModeArray modes_1;
modes_1.push_back(1);
ModeArray modes_1_0_3_2;
modes_1_0_3_2.push_back(1);
modes_1_0_3_2.push_back(0);
modes_1_0_3_2.push_back(3);
modes_1_0_3_2.push_back(2);
ModeArray modes_1_3;
modes_1_3.push_back(1);
modes_1_3.push_back(3);
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
IndexArray indices_ei( 2 );
indices_ei[0] = 'e';
indices_ei[1] = 'i';
IndexArray indices_eifj( 4 );
indices_eifj[0] = 'e';
indices_eifj[1] = 'i';
indices_eifj[2] = 'f';
indices_eifj[3] = 'j';
IndexArray indices_ej( 2 );
indices_ej[0] = 'e';
indices_ej[1] = 'j';
IndexArray indices_fj( 2 );
indices_fj[0] = 'f';
indices_fj[1] = 'j';
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
	//T_bfnj[D0,D1,D2,D3]
DistTensor<double> T_bfnj__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//q_mnij[D0,D1,D2,D3]
DistTensor<double> q_mnij__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//t_fj[D01,D23]
DistTensor<double> t_fj__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//t_fj[D01,D3]
DistTensor<double> t_fj__D_0_1__D_3( dist__D_0_1__D_3, g );
	//t_fj[*,D3]
DistTensor<double> t_fj__S__D_3( dist__S__D_3, g );
	//t_fj_part1B[D01,D23]
DistTensor<double> t_fj_part1B__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//t_fj_part1T[D01,D23]
DistTensor<double> t_fj_part1T__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//t_fj_part1_0[D01,D23]
DistTensor<double> t_fj_part1_0__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//t_fj_part1_1[D01,D23]
DistTensor<double> t_fj_part1_1__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//t_fj_part1_1[D01,D3]
DistTensor<double> t_fj_part1_1__D_0_1__D_3( dist__D_0_1__D_3, g );
	//t_fj_part1_1[D0,D2]
DistTensor<double> t_fj_part1_1__D_0__D_2( dist__D_0__D_2, g );
	//t_fj_part1_1[D1,D3]
DistTensor<double> t_fj_part1_1__D_1__D_3( dist__D_1__D_3, g );
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
	//temp1_part2_1_part3_1[D0,D1,D2,D3]
DistTensor<double> temp1_part2_1_part3_1_perm0213__D_0__D_2__D_1__D_3( dist__D_0__D_1__D_2__D_3, g );
temp1_part2_1_part3_1_perm0213__D_0__D_2__D_1__D_3.SetLocalPermutation( perm_0_2_1_3 );
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
	//temp2_part0_1_part2B[D0,D1,D2,D3]
DistTensor<double> temp2_part0_1_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp2_part0_1_part2T[D0,D1,D2,D3]
DistTensor<double> temp2_part0_1_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp2_part0_1_part2_0[D0,D1,D2,D3]
DistTensor<double> temp2_part0_1_part2_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp2_part0_1_part2_1[D0,D1,D2,D3]
DistTensor<double> temp2_part0_1_part2_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp2_part0_1_part2_2[D0,D1,D2,D3]
DistTensor<double> temp2_part0_1_part2_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp2_part0_2[D0,D1,D2,D3]
DistTensor<double> temp2_part0_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp2_part1B[D0,D1,D2,D3]
DistTensor<double> temp2_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp2_part1T[D0,D1,D2,D3]
DistTensor<double> temp2_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp2_part1_0[D0,D1,D2,D3]
DistTensor<double> temp2_part1_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp2_part1_1[D0,D1,D2,D3]
DistTensor<double> temp2_part1_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp2_part1_1_part3B[D0,D1,D2,D3]
DistTensor<double> temp2_part1_1_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp2_part1_1_part3T[D0,D1,D2,D3]
DistTensor<double> temp2_part1_1_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp2_part1_1_part3_0[D0,D1,D2,D3]
DistTensor<double> temp2_part1_1_part3_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp2_part1_1_part3_1[D0,D1,D2,D3]
DistTensor<double> temp2_part1_1_part3_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp2_part1_1_part3_1[D0,D1,D3,D2]
DistTensor<double> temp2_part1_1_part3_1__D_0__D_1__D_3__D_2( dist__D_0__D_1__D_3__D_2, g );
	//temp2_part1_1_part3_1[D1,D0,D3,D2]
DistTensor<double> temp2_part1_1_part3_1__D_1__D_0__D_3__D_2( dist__D_1__D_0__D_3__D_2, g );
	//temp2_part1_1_part3_2[D0,D1,D2,D3]
DistTensor<double> temp2_part1_1_part3_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp2_part1_2[D0,D1,D2,D3]
DistTensor<double> temp2_part1_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//u_mnie[D0,D1,D2,D3]
DistTensor<double> u_mnie__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//u_mnie_part0B[D0,D1,D2,D3]
DistTensor<double> u_mnie_part0B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//u_mnie_part0T[D0,D1,D2,D3]
DistTensor<double> u_mnie_part0T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//u_mnie_part0_0[D0,D1,D2,D3]
DistTensor<double> u_mnie_part0_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//u_mnie_part0_1[D0,D1,D2,D3]
DistTensor<double> u_mnie_part0_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//u_mnie_part0_1_part1B[D0,D1,D2,D3]
DistTensor<double> u_mnie_part0_1_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//u_mnie_part0_1_part1T[D0,D1,D2,D3]
DistTensor<double> u_mnie_part0_1_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//u_mnie_part0_1_part1_0[D0,D1,D2,D3]
DistTensor<double> u_mnie_part0_1_part1_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//u_mnie_part0_1_part1_1[D0,D1,D2,D3]
DistTensor<double> u_mnie_part0_1_part1_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//u_mnie_part0_1_part1_1[D0,D1,D2,*]
DistTensor<double> u_mnie_part0_1_part1_1__D_0__D_1__D_2__S( dist__D_0__D_1__D_2__S, g );
	//u_mnie_part0_1_part1_2[D0,D1,D2,D3]
DistTensor<double> u_mnie_part0_1_part1_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//u_mnie_part0_2[D0,D1,D2,D3]
DistTensor<double> u_mnie_part0_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
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
v_femn__D_0__D_1__D_2__D_3_tempShape.push_back( 500 );
v_femn__D_0__D_1__D_2__D_3_tempShape.push_back( 500 );
v_femn__D_0__D_1__D_2__D_3_tempShape.push_back( 50 );
v_femn__D_0__D_1__D_2__D_3_tempShape.push_back( 50 );
v_femn__D_0__D_1__D_2__D_3.ResizeTo( v_femn__D_0__D_1__D_2__D_3_tempShape );
MakeUniform( v_femn__D_0__D_1__D_2__D_3 );
// T_bfnj has 4 dims
//	Starting distribution: [D0,D1,D2,D3] or _D_0__D_1__D_2__D_3
ObjShape T_bfnj__D_0__D_1__D_2__D_3_tempShape;
T_bfnj__D_0__D_1__D_2__D_3_tempShape.push_back( 500 );
T_bfnj__D_0__D_1__D_2__D_3_tempShape.push_back( 500 );
T_bfnj__D_0__D_1__D_2__D_3_tempShape.push_back( 50 );
T_bfnj__D_0__D_1__D_2__D_3_tempShape.push_back( 50 );
T_bfnj__D_0__D_1__D_2__D_3.ResizeTo( T_bfnj__D_0__D_1__D_2__D_3_tempShape );
MakeUniform( T_bfnj__D_0__D_1__D_2__D_3 );
// t_fj has 2 dims
//	Starting distribution: [D01,D23] or _D_0_1__D_2_3
ObjShape t_fj__D_0_1__D_2_3_tempShape;
t_fj__D_0_1__D_2_3_tempShape.push_back( 500 );
t_fj__D_0_1__D_2_3_tempShape.push_back( 50 );
t_fj__D_0_1__D_2_3.ResizeTo( t_fj__D_0_1__D_2_3_tempShape );
MakeUniform( t_fj__D_0_1__D_2_3 );
tempShape = T_bfnj__D_0__D_1__D_2__D_3.Shape();
temp1__D_0__D_1__D_2__D_3.ResizeTo( tempShape );
// u_mnie has 4 dims
//	Starting distribution: [D0,D1,D2,D3] or _D_0__D_1__D_2__D_3
ObjShape u_mnie__D_0__D_1__D_2__D_3_tempShape;
u_mnie__D_0__D_1__D_2__D_3_tempShape.push_back( 50 );
u_mnie__D_0__D_1__D_2__D_3_tempShape.push_back( 50 );
u_mnie__D_0__D_1__D_2__D_3_tempShape.push_back( 50 );
u_mnie__D_0__D_1__D_2__D_3_tempShape.push_back( 500 );
u_mnie__D_0__D_1__D_2__D_3.ResizeTo( u_mnie__D_0__D_1__D_2__D_3_tempShape );
MakeUniform( u_mnie__D_0__D_1__D_2__D_3 );
// Q_mnij has 4 dims
//	Starting distribution: [D0,D1,D2,D3] or _D_0__D_1__D_2__D_3
ObjShape Q_mnij__D_0__D_1__D_2__D_3_tempShape;
Q_mnij__D_0__D_1__D_2__D_3_tempShape.push_back( 50 );
Q_mnij__D_0__D_1__D_2__D_3_tempShape.push_back( 50 );
Q_mnij__D_0__D_1__D_2__D_3_tempShape.push_back( 50 );
Q_mnij__D_0__D_1__D_2__D_3_tempShape.push_back( 50 );
Q_mnij__D_0__D_1__D_2__D_3.ResizeTo( Q_mnij__D_0__D_1__D_2__D_3_tempShape );
MakeUniform( Q_mnij__D_0__D_1__D_2__D_3 );
// q_mnij has 4 dims
//	Starting distribution: [D0,D1,D2,D3] or _D_0__D_1__D_2__D_3
ObjShape q_mnij__D_0__D_1__D_2__D_3_tempShape;
q_mnij__D_0__D_1__D_2__D_3_tempShape.push_back( 50 );
q_mnij__D_0__D_1__D_2__D_3_tempShape.push_back( 50 );
q_mnij__D_0__D_1__D_2__D_3_tempShape.push_back( 50 );
q_mnij__D_0__D_1__D_2__D_3_tempShape.push_back( 50 );
q_mnij__D_0__D_1__D_2__D_3.ResizeTo( q_mnij__D_0__D_1__D_2__D_3_tempShape );
MakeUniform( q_mnij__D_0__D_1__D_2__D_3 );
tempShape = Q_mnij__D_0__D_1__D_2__D_3.Shape();
temp2__D_0__D_1__D_2__D_3.ResizeTo( tempShape );
//**** (out of 1)

//******************************
//* Load tensors
//******************************
////////////////////////////////
//Performance testing
////////////////////////////////
#ifdef CORRECTNESS
    DistTensor<T> epsilonA( tmen::StringToTensorDist("[(0)]|()"), g);
    ObjShape epsilonAShape;
    epsilonAShape.push_back(n_v);
    epsilonA.ResizeTo(epsilonAShape);
    std::string epsilonAFilename = "data/ea";
    printf("loading epsilonA\n");
    Load_Tensor(epsilonA, epsilonAFilename);
    //Print(epsilonA, "eps_a");

    DistTensor<T> epsilonB( tmen::StringToTensorDist("[(0)]|()"), g);
    ObjShape epsilonBShape;
    epsilonBShape.push_back(n_o);
    epsilonB.ResizeTo(epsilonBShape);
    std::string epsilonBFilename = "data/ei";
    printf("loading epsilonB\n");
    Load_Tensor(epsilonB, epsilonBFilename);
    //Print(epsilonB, "eps_b");

    DistTensor<T> D_abij( tmen::StringToTensorDist("[(0),(1),(2),(3)]|()"), g);
    ObjShape D_abijShape;
    D_abijShape.push_back(n_v);
    D_abijShape.push_back(n_v);
    D_abijShape.push_back(n_o);
    D_abijShape.push_back(n_o);
    D_abij.ResizeTo(D_abijShape);

    DistTensor<T> V_abij( tmen::StringToTensorDist("[(0),(1),(2),(3)]|()"), g);
    V_abij.ResizeTo(D_abijShape);
    std::string v_abijFilename = "data/abij";
    printf("loading V_abij\n");
    Load_Tensor(V_abij, v_abijFilename);
    //Print(V_abij, "v_abij");

    std::string v_opmnFilename = "data/ijkl";
    printf("loading v_opmn\n");
    Load_Tensor(v_opmn__D_0__D_1__D_2__D_3, v_opmnFilename);
    //Print(v_opmn__D_0__D_1__D_2__D_3, "v_opmn");

    printf("loading 4\n");
    std::string v_oegmFilename = "data/aijb";
    printf("loading v_oegm\n");
    Load_Tensor_aijb(v_oegm__D_0__D_1__D_2__D_3, v_oegmFilename);
    //Print(v_oegm__D_0__D_1__D_2__D_3, "v_oegm");

    printf("loading 5\n");
    std::string v2_oegmFilename = "data/aibj";
    printf("loading v2_oegm\n");
    Load_Tensor_aijb(v2_oegm__D_0__D_1__D_2__D_3, v2_oegmFilename);
    //Print(v2_oegm__D_0__D_1__D_2__D_3, "v2_oegm");

    printf("loading 3\n");
    std::string v_efghFilename = "data/abcd";
    printf("loading v_efgh\n");
    Load_Tensor(v_efgh__D_0__D_1__D_2__D_3, v_efghFilename);
    //Print(v_efgh__D_0__D_1__D_2__D_3, "v_efgh");

    printf("elemScaling\n");
    Form_D_abij(epsilonA, epsilonB, D_abij);
    tmen::ElemScal(V_abij, D_abij, t_efmn__D_0__D_1__D_2__D_3);
//  Print(t_efmn__D_0__D_1__D_2__D_3, "t_efmn");
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
	PartitionDown(t_fj__D_0_1__D_2_3, t_fj_part1T__D_0_1__D_2_3, t_fj_part1B__D_0_1__D_2_3, 1, 0);
	PartitionDown(temp1__D_0__D_1__D_2__D_3, temp1_part2T__D_0__D_1__D_2__D_3, temp1_part2B__D_0__D_1__D_2__D_3, 2, 0);
	while(temp1_part2T__D_0__D_1__D_2__D_3.Dimension(2) < temp1__D_0__D_1__D_2__D_3.Dimension(2))
	{
		RepartitionDown
		( t_fj_part1T__D_0_1__D_2_3,  t_fj_part1_0__D_0_1__D_2_3,
		  /**/ /**/
		       t_fj_part1_1__D_0_1__D_2_3,
		  t_fj_part1B__D_0_1__D_2_3, t_fj_part1_2__D_0_1__D_2_3, 1, 32 );
		RepartitionDown
		( temp1_part2T__D_0__D_1__D_2__D_3,  temp1_part2_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       temp1_part2_1__D_0__D_1__D_2__D_3,
		  temp1_part2B__D_0__D_1__D_2__D_3, temp1_part2_2__D_0__D_1__D_2__D_3, 2, 32 );

		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  temp1_part2_1__D_0__D_1__D_2__D_3
		PartitionDown(t_fj__D_0_1__D_2_3, t_fj_part1T__D_0_1__D_2_3, t_fj_part1B__D_0_1__D_2_3, 1, 0);
		PartitionDown(temp1_part2_1__D_0__D_1__D_2__D_3, temp1_part2_1_part3T__D_0__D_1__D_2__D_3, temp1_part2_1_part3B__D_0__D_1__D_2__D_3, 3, 0);
		while(temp1_part2_1_part3T__D_0__D_1__D_2__D_3.Dimension(3) < temp1_part2_1__D_0__D_1__D_2__D_3.Dimension(3))
		{
			RepartitionDown
			( t_fj_part1T__D_0_1__D_2_3,  t_fj_part1_0__D_0_1__D_2_3,
			  /**/ /**/
			       t_fj_part1_1__D_0_1__D_2_3,
			  t_fj_part1B__D_0_1__D_2_3, t_fj_part1_2__D_0_1__D_2_3, 1, 32 );
			RepartitionDown
			( temp1_part2_1_part3T__D_0__D_1__D_2__D_3,  temp1_part2_1_part3_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       temp1_part2_1_part3_1__D_0__D_1__D_2__D_3,
			  temp1_part2_1_part3B__D_0__D_1__D_2__D_3, temp1_part2_1_part3_2__D_0__D_1__D_2__D_3, 3, 32 );

			   // t_fj_part1_1[D0,D2] <- t_fj_part1_1[D01,D23]
			t_fj_part1_1__D_0__D_2.AlignModesWith( modes_0_1, temp1_part2_1_part3_1__D_0__D_1__D_2__D_3, modes_0_2 );
			t_fj_part1_1__D_0__D_2.AllGatherRedistFrom( t_fj_part1_1__D_0_1__D_2_3, modes_1_3 );
			   // t_fj_part1_1[D01,D3] <- t_fj_part1_1[D01,D23]
			t_fj_part1_1__D_0_1__D_3.AlignModesWith( modes_0_1, temp1_part2_1_part3_1__D_0__D_1__D_2__D_3, modes_1_3 );
			t_fj_part1_1__D_0_1__D_3.AllToAllRedistFrom( t_fj_part1_1__D_0_1__D_2_3, modes_2_3 );
			   // t_fj_part1_1[D1,D3] <- t_fj_part1_1[D01,D3]
			t_fj_part1_1__D_1__D_3.AlignModesWith( modes_0_1, temp1_part2_1_part3_1__D_0__D_1__D_2__D_3, modes_1_3 );
			t_fj_part1_1__D_1__D_3.AllToAllRedistFrom( t_fj_part1_1__D_0_1__D_3, modes_0_1 );
			t_fj_part1_1__D_0_1__D_3.EmptyData();
			Permute( temp1_part2_1_part3_1__D_0__D_1__D_2__D_3, temp1_part2_1_part3_1_perm0213__D_0__D_2__D_1__D_3 );
			   // 1.0 * t_fj_part1_1[D0,D2]_ei * t_fj_part1_1[D1,D3]_fj + 0.0 * temp1_part2_1_part3_1[D0,D1,D2,D3]_eifj
			LocalContractAndLocalEliminate(1.0, t_fj_part1_1__D_0__D_2.LockedTensor(), indices_ei, false,
				t_fj_part1_1__D_1__D_3.LockedTensor(), indices_fj, false,
				0.0, temp1_part2_1_part3_1_perm0213__D_0__D_2__D_1__D_3.Tensor(), indices_eifj, false);
			t_fj_part1_1__D_0__D_2.EmptyData();
			t_fj_part1_1__D_1__D_3.EmptyData();
			Permute( temp1_part2_1_part3_1_perm0213__D_0__D_2__D_1__D_3, temp1_part2_1_part3_1__D_0__D_1__D_2__D_3 );
			temp1_part2_1_part3_1_perm0213__D_0__D_2__D_1__D_3.EmptyData();

			SlidePartitionDown
			( t_fj_part1T__D_0_1__D_2_3,  t_fj_part1_0__D_0_1__D_2_3,
			       t_fj_part1_1__D_0_1__D_2_3,
			  /**/ /**/
			  t_fj_part1B__D_0_1__D_2_3, t_fj_part1_2__D_0_1__D_2_3, 1 );
			SlidePartitionDown
			( temp1_part2_1_part3T__D_0__D_1__D_2__D_3,  temp1_part2_1_part3_0__D_0__D_1__D_2__D_3,
			       temp1_part2_1_part3_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  temp1_part2_1_part3B__D_0__D_1__D_2__D_3, temp1_part2_1_part3_2__D_0__D_1__D_2__D_3, 3 );

		}
		//****

		SlidePartitionDown
		( t_fj_part1T__D_0_1__D_2_3,  t_fj_part1_0__D_0_1__D_2_3,
		       t_fj_part1_1__D_0_1__D_2_3,
		  /**/ /**/
		  t_fj_part1B__D_0_1__D_2_3, t_fj_part1_2__D_0_1__D_2_3, 1 );
		SlidePartitionDown
		( temp1_part2T__D_0__D_1__D_2__D_3,  temp1_part2_0__D_0__D_1__D_2__D_3,
		       temp1_part2_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  temp1_part2B__D_0__D_1__D_2__D_3, temp1_part2_2__D_0__D_1__D_2__D_3, 2 );

	}
	//****


//****
//**** (out of 1)

	YAxpy( 1.0, T_bfnj__D_0__D_1__D_2__D_3, temp1__D_0__D_1__D_2__D_3 );
	T_bfnj__D_0__D_1__D_2__D_3.EmptyData();


//****
//**** (out of 1)

	//**** (out of 1)
	//**** Is real	0 shadows
		//Outputs:
		//  temp2__D_0__D_1__D_2__D_3
	PartitionDown(u_mnie__D_0__D_1__D_2__D_3, u_mnie_part0T__D_0__D_1__D_2__D_3, u_mnie_part0B__D_0__D_1__D_2__D_3, 0, 0);
	PartitionDown(temp2__D_0__D_1__D_2__D_3, temp2_part0T__D_0__D_1__D_2__D_3, temp2_part0B__D_0__D_1__D_2__D_3, 0, 0);
	while(temp2_part0T__D_0__D_1__D_2__D_3.Dimension(0) < temp2__D_0__D_1__D_2__D_3.Dimension(0))
	{
		RepartitionDown
		( u_mnie_part0T__D_0__D_1__D_2__D_3,  u_mnie_part0_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       u_mnie_part0_1__D_0__D_1__D_2__D_3,
		  u_mnie_part0B__D_0__D_1__D_2__D_3, u_mnie_part0_2__D_0__D_1__D_2__D_3, 0, 32 );
		RepartitionDown
		( temp2_part0T__D_0__D_1__D_2__D_3,  temp2_part0_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       temp2_part0_1__D_0__D_1__D_2__D_3,
		  temp2_part0B__D_0__D_1__D_2__D_3, temp2_part0_2__D_0__D_1__D_2__D_3, 0, 32 );

		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  temp2_part0_1__D_0__D_1__D_2__D_3
		PartitionDown(u_mnie_part0_1__D_0__D_1__D_2__D_3, u_mnie_part0_1_part1T__D_0__D_1__D_2__D_3, u_mnie_part0_1_part1B__D_0__D_1__D_2__D_3, 1, 0);
		PartitionDown(temp2_part0_1__D_0__D_1__D_2__D_3, temp2_part0_1_part1T__D_0__D_1__D_2__D_3, temp2_part0_1_part1B__D_0__D_1__D_2__D_3, 1, 0);
		while(temp2_part0_1_part1T__D_0__D_1__D_2__D_3.Dimension(1) < temp2_part0_1__D_0__D_1__D_2__D_3.Dimension(1))
		{
			RepartitionDown
			( u_mnie_part0_1_part1T__D_0__D_1__D_2__D_3,  u_mnie_part0_1_part1_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       u_mnie_part0_1_part1_1__D_0__D_1__D_2__D_3,
			  u_mnie_part0_1_part1B__D_0__D_1__D_2__D_3, u_mnie_part0_1_part1_2__D_0__D_1__D_2__D_3, 1, 32 );
			RepartitionDown
			( temp2_part0_1_part1T__D_0__D_1__D_2__D_3,  temp2_part0_1_part1_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       temp2_part0_1_part1_1__D_0__D_1__D_2__D_3,
			  temp2_part0_1_part1B__D_0__D_1__D_2__D_3, temp2_part0_1_part1_2__D_0__D_1__D_2__D_3, 1, 32 );

			   // u_mnie_part0_1_part1_1[D0,D1,D2,*] <- u_mnie_part0_1_part1_1[D0,D1,D2,D3]
			u_mnie_part0_1_part1_1__D_0__D_1__D_2__S.AlignModesWith( modes_0_1_2, temp2_part0_1_part1_1__D_0__D_1__D_2__D_3, modes_0_1_2 );
			u_mnie_part0_1_part1_1__D_0__D_1__D_2__S.AllGatherRedistFrom( u_mnie_part0_1_part1_1__D_0__D_1__D_2__D_3, modes_3 );
			   // t_fj[D01,D3] <- t_fj[D01,D23]
			t_fj__D_0_1__D_3.AlignModesWith( modes_1, temp2_part0_1_part1_1__D_0__D_1__D_2__D_3, modes_3 );
			t_fj__D_0_1__D_3.AllToAllRedistFrom( t_fj__D_0_1__D_2_3, modes_2_3 );
			   // t_fj[*,D3] <- t_fj[D01,D3]
			t_fj__S__D_3.AlignModesWith( modes_1, temp2_part0_1_part1_1__D_0__D_1__D_2__D_3, modes_3 );
			t_fj__S__D_3.AllGatherRedistFrom( t_fj__D_0_1__D_3, modes_0_1 );
			   // 1.0 * u_mnie_part0_1_part1_1[D0,D1,D2,*]_mnie * t_fj[*,D3]_ej + 0.0 * temp2_part0_1_part1_1[D0,D1,D2,D3]_mnij
			LocalContractAndLocalEliminate(1.0, u_mnie_part0_1_part1_1__D_0__D_1__D_2__S.LockedTensor(), indices_mnie, false,
				t_fj__S__D_3.LockedTensor(), indices_ej, false,
				0.0, temp2_part0_1_part1_1__D_0__D_1__D_2__D_3.Tensor(), indices_mnij, false);
			t_fj__S__D_3.EmptyData();
			u_mnie_part0_1_part1_1__D_0__D_1__D_2__S.EmptyData();
			t_fj__D_0_1__D_3.EmptyData();

			SlidePartitionDown
			( u_mnie_part0_1_part1T__D_0__D_1__D_2__D_3,  u_mnie_part0_1_part1_0__D_0__D_1__D_2__D_3,
			       u_mnie_part0_1_part1_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  u_mnie_part0_1_part1B__D_0__D_1__D_2__D_3, u_mnie_part0_1_part1_2__D_0__D_1__D_2__D_3, 1 );
			SlidePartitionDown
			( temp2_part0_1_part1T__D_0__D_1__D_2__D_3,  temp2_part0_1_part1_0__D_0__D_1__D_2__D_3,
			       temp2_part0_1_part1_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  temp2_part0_1_part1B__D_0__D_1__D_2__D_3, temp2_part0_1_part1_2__D_0__D_1__D_2__D_3, 1 );

		}
		//****

		SlidePartitionDown
		( u_mnie_part0T__D_0__D_1__D_2__D_3,  u_mnie_part0_0__D_0__D_1__D_2__D_3,
		       u_mnie_part0_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  u_mnie_part0B__D_0__D_1__D_2__D_3, u_mnie_part0_2__D_0__D_1__D_2__D_3, 0 );
		SlidePartitionDown
		( temp2_part0T__D_0__D_1__D_2__D_3,  temp2_part0_0__D_0__D_1__D_2__D_3,
		       temp2_part0_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  temp2_part0B__D_0__D_1__D_2__D_3, temp2_part0_2__D_0__D_1__D_2__D_3, 0 );

	}
	//****
	//**** (out of 1)
	//**** Is real	0 shadows
		//Outputs:
		//  Q_mnij__D_0__D_1__D_2__D_3
	PartitionDown(v_femn__D_0__D_1__D_2__D_3, v_femn_part2T__D_0__D_1__D_2__D_3, v_femn_part2B__D_0__D_1__D_2__D_3, 2, 0);
	PartitionDown(temp2__D_0__D_1__D_2__D_3, temp2_part0T__D_0__D_1__D_2__D_3, temp2_part0B__D_0__D_1__D_2__D_3, 0, 0);
	PartitionDown(temp2__D_0__D_1__D_2__D_3, temp2_part1T__D_0__D_1__D_2__D_3, temp2_part1B__D_0__D_1__D_2__D_3, 1, 0);
	PartitionDown(Q_mnij__D_0__D_1__D_2__D_3, Q_mnij_part0T__D_0__D_1__D_2__D_3, Q_mnij_part0B__D_0__D_1__D_2__D_3, 0, 0);
	while(Q_mnij_part0T__D_0__D_1__D_2__D_3.Dimension(0) < Q_mnij__D_0__D_1__D_2__D_3.Dimension(0))
	{
		RepartitionDown
		( v_femn_part2T__D_0__D_1__D_2__D_3,  v_femn_part2_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       v_femn_part2_1__D_0__D_1__D_2__D_3,
		  v_femn_part2B__D_0__D_1__D_2__D_3, v_femn_part2_2__D_0__D_1__D_2__D_3, 2, 32 );
		RepartitionDown
		( temp2_part0T__D_0__D_1__D_2__D_3,  temp2_part0_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       temp2_part0_1__D_0__D_1__D_2__D_3,
		  temp2_part0B__D_0__D_1__D_2__D_3, temp2_part0_2__D_0__D_1__D_2__D_3, 0, 32 );
		RepartitionDown
		( temp2_part1T__D_0__D_1__D_2__D_3,  temp2_part1_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       temp2_part1_1__D_0__D_1__D_2__D_3,
		  temp2_part1B__D_0__D_1__D_2__D_3, temp2_part1_2__D_0__D_1__D_2__D_3, 1, 32 );
		RepartitionDown
		( Q_mnij_part0T__D_0__D_1__D_2__D_3,  Q_mnij_part0_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       Q_mnij_part0_1__D_0__D_1__D_2__D_3,
		  Q_mnij_part0B__D_0__D_1__D_2__D_3, Q_mnij_part0_2__D_0__D_1__D_2__D_3, 0, 32 );

		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  Q_mnij_part0_1__D_0__D_1__D_2__D_3
		PartitionDown(temp2_part0_1__D_0__D_1__D_2__D_3, temp2_part0_1_part2T__D_0__D_1__D_2__D_3, temp2_part0_1_part2B__D_0__D_1__D_2__D_3, 2, 0);
		PartitionDown(temp2_part1_1__D_0__D_1__D_2__D_3, temp2_part1_1_part3T__D_0__D_1__D_2__D_3, temp2_part1_1_part3B__D_0__D_1__D_2__D_3, 3, 0);
		PartitionDown(Q_mnij_part0_1__D_0__D_1__D_2__D_3, Q_mnij_part0_1_part2T__D_0__D_1__D_2__D_3, Q_mnij_part0_1_part2B__D_0__D_1__D_2__D_3, 2, 0);
		while(Q_mnij_part0_1_part2T__D_0__D_1__D_2__D_3.Dimension(2) < Q_mnij_part0_1__D_0__D_1__D_2__D_3.Dimension(2))
		{
			RepartitionDown
			( temp2_part0_1_part2T__D_0__D_1__D_2__D_3,  temp2_part0_1_part2_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       temp2_part0_1_part2_1__D_0__D_1__D_2__D_3,
			  temp2_part0_1_part2B__D_0__D_1__D_2__D_3, temp2_part0_1_part2_2__D_0__D_1__D_2__D_3, 2, 32 );
			RepartitionDown
			( temp2_part1_1_part3T__D_0__D_1__D_2__D_3,  temp2_part1_1_part3_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       temp2_part1_1_part3_1__D_0__D_1__D_2__D_3,
			  temp2_part1_1_part3B__D_0__D_1__D_2__D_3, temp2_part1_1_part3_2__D_0__D_1__D_2__D_3, 3, 32 );
			RepartitionDown
			( Q_mnij_part0_1_part2T__D_0__D_1__D_2__D_3,  Q_mnij_part0_1_part2_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       Q_mnij_part0_1_part2_1__D_0__D_1__D_2__D_3,
			  Q_mnij_part0_1_part2B__D_0__D_1__D_2__D_3, Q_mnij_part0_1_part2_2__D_0__D_1__D_2__D_3, 2, 32 );

			   // temp2_part1_1_part3_1[D0,D1,D3,D2] <- temp2_part1_1_part3_1[D0,D1,D2,D3]
			temp2_part1_1_part3_1__D_0__D_1__D_3__D_2.AlignModesWith( modes_0_1_2_3, temp2_part0_1_part2_1__D_0__D_1__D_2__D_3, modes_1_0_3_2 );
			temp2_part1_1_part3_1__D_0__D_1__D_3__D_2.AllToAllRedistFrom( temp2_part1_1_part3_1__D_0__D_1__D_2__D_3, modes_2_3 );
			   // temp2_part1_1_part3_1[D1,D0,D3,D2] <- temp2_part1_1_part3_1[D0,D1,D3,D2]
			temp2_part1_1_part3_1__D_1__D_0__D_3__D_2.AlignModesWith( modes_0_1_2_3, temp2_part0_1_part2_1__D_0__D_1__D_2__D_3, modes_1_0_3_2 );
			temp2_part1_1_part3_1__D_1__D_0__D_3__D_2.AllToAllRedistFrom( temp2_part1_1_part3_1__D_0__D_1__D_3__D_2, modes_0_1 );
			YAxpPx( 1.0, temp2_part0_1_part2_1__D_0__D_1__D_2__D_3, 1.0, temp2_part1_1_part3_1__D_1__D_0__D_3__D_2, perm_1_0_3_2, Q_mnij_part0_1_part2_1__D_0__D_1__D_2__D_3 );
			temp2_part1_1_part3_1__D_1__D_0__D_3__D_2.EmptyData();
			temp2_part1_1_part3_1__D_0__D_1__D_3__D_2.EmptyData();

			SlidePartitionDown
			( temp2_part0_1_part2T__D_0__D_1__D_2__D_3,  temp2_part0_1_part2_0__D_0__D_1__D_2__D_3,
			       temp2_part0_1_part2_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  temp2_part0_1_part2B__D_0__D_1__D_2__D_3, temp2_part0_1_part2_2__D_0__D_1__D_2__D_3, 2 );
			SlidePartitionDown
			( temp2_part1_1_part3T__D_0__D_1__D_2__D_3,  temp2_part1_1_part3_0__D_0__D_1__D_2__D_3,
			       temp2_part1_1_part3_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  temp2_part1_1_part3B__D_0__D_1__D_2__D_3, temp2_part1_1_part3_2__D_0__D_1__D_2__D_3, 3 );
			SlidePartitionDown
			( Q_mnij_part0_1_part2T__D_0__D_1__D_2__D_3,  Q_mnij_part0_1_part2_0__D_0__D_1__D_2__D_3,
			       Q_mnij_part0_1_part2_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  Q_mnij_part0_1_part2B__D_0__D_1__D_2__D_3, Q_mnij_part0_1_part2_2__D_0__D_1__D_2__D_3, 2 );

		}
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
			  v_femn_part2_1_part3B__D_0__D_1__D_2__D_3, v_femn_part2_1_part3_2__D_0__D_1__D_2__D_3, 3, 32 );
			RepartitionDown
			( Q_mnij_part0_1_part1T__D_0__D_1__D_2__D_3,  Q_mnij_part0_1_part1_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       Q_mnij_part0_1_part1_1__D_0__D_1__D_2__D_3,
			  Q_mnij_part0_1_part1B__D_0__D_1__D_2__D_3, Q_mnij_part0_1_part1_2__D_0__D_1__D_2__D_3, 1, 32 );

			tempShape = Q_mnij_part0_1_part1_1__D_0__D_1__D_2__D_3.Shape();
			tempShape.push_back( g.Shape()[0] );
			tempShape.push_back( g.Shape()[1] );
			Q_mnij_part0_1_part1_1__S__S__D_2__D_3__D_0__D_1.ResizeTo( tempShape );
			   // v_femn_part2_1_part3_1[D0,D1,*,*] <- v_femn_part2_1_part3_1[D0,D1,D2,D3]
			v_femn_part2_1_part3_1_perm2301__S__S__D_0__D_1.AlignModesWith( modes_0_1, temp1__D_0__D_1__D_2__D_3, modes_0_1 );
			v_femn_part2_1_part3_1_perm2301__S__S__D_0__D_1.AllGatherRedistFrom( v_femn_part2_1_part3_1__D_0__D_1__D_2__D_3, modes_2_3 );
			   // 1.0 * v_femn_part2_1_part3_1[D0,D1,*,*]_mnef * temp1[D0,D1,D2,D3]_efij + 0.0 * Q_mnij_part0_1_part1_1[*,*,D2,D3,D0,D1]_mnijef
			LocalContract(1.0, v_femn_part2_1_part3_1_perm2301__S__S__D_0__D_1.LockedTensor(), indices_mnef, false,
				temp1__D_0__D_1__D_2__D_3.LockedTensor(), indices_efij, false,
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
		( temp2_part0T__D_0__D_1__D_2__D_3,  temp2_part0_0__D_0__D_1__D_2__D_3,
		       temp2_part0_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  temp2_part0B__D_0__D_1__D_2__D_3, temp2_part0_2__D_0__D_1__D_2__D_3, 0 );
		SlidePartitionDown
		( temp2_part1T__D_0__D_1__D_2__D_3,  temp2_part1_0__D_0__D_1__D_2__D_3,
		       temp2_part1_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  temp2_part1B__D_0__D_1__D_2__D_3, temp2_part1_2__D_0__D_1__D_2__D_3, 1 );

	}
	//****


//****
//**** (out of 1)

	YAxpy( 1.0, Q_mnij__D_0__D_1__D_2__D_3, q_mnij__D_0__D_1__D_2__D_3 );
	Q_mnij__D_0__D_1__D_2__D_3.EmptyData();


//****

/*****************************************/

    /*****************************************/
    mpi::Barrier(g.OwningComm());
    runTime = mpi::Time() - startTime;
    double flops = pow(n_o, 2) * pow(n_v, 2) * (11 + 2 * pow(n_o + n_v, 2));
    gflops = flops / (1e9 * runTime);

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
        DistTensorTest<double>(g, args.n_o, args.n_v, args.blkSize);

    } catch (std::exception& e) {
        ReportException(e);
    }

    Finalize();
    //printf("Completed\n");
    return 0;
}


