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
TensorDistribution dist__S__D_2__D_1__S = tmen::StringToTensorDist("[(),(2),(1),()]");
TensorDistribution dist__D_0__S__S__D_3 = tmen::StringToTensorDist("[(0),(),(),(3)]");
TensorDistribution dist__D_0__S = tmen::StringToTensorDist("[(0),()]");
TensorDistribution dist__D_0__D_1__D_2__S__D_3 = tmen::StringToTensorDist("[(0),(1),(2),(),(3)]");
TensorDistribution dist__D_0__D_1__D_2__D_3 = tmen::StringToTensorDist("[(0),(1),(2),(3)]");
TensorDistribution dist__D_0__D_1__D_3__D_2 = tmen::StringToTensorDist("[(0),(1),(3),(2)]");
TensorDistribution dist__D_0__D_2__D_1__D_3 = tmen::StringToTensorDist("[(0),(2),(1),(3)]");
TensorDistribution dist__D_0__D_2 = tmen::StringToTensorDist("[(0),(2)]");
TensorDistribution dist__D_1__S__D_3__D_2 = tmen::StringToTensorDist("[(1),(),(3),(2)]");
TensorDistribution dist__D_1__D_3 = tmen::StringToTensorDist("[(1),(3)]");
TensorDistribution dist__D_3__S = tmen::StringToTensorDist("[(3),()]");
TensorDistribution dist__D_3__D_2 = tmen::StringToTensorDist("[(3),(2)]");
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
Permutation perm_0_1_2_3_4;
perm_0_1_2_3_4.push_back(0);
perm_0_1_2_3_4.push_back(1);
perm_0_1_2_3_4.push_back(2);
perm_0_1_2_3_4.push_back(3);
perm_0_1_2_3_4.push_back(4);
Permutation perm_0_2_1_3;
perm_0_2_1_3.push_back(0);
perm_0_2_1_3.push_back(2);
perm_0_2_1_3.push_back(1);
perm_0_2_1_3.push_back(3);
Permutation perm_0_2_3_1;
perm_0_2_3_1.push_back(0);
perm_0_2_3_1.push_back(2);
perm_0_2_3_1.push_back(3);
perm_0_2_3_1.push_back(1);
Permutation perm_1_0;
perm_1_0.push_back(1);
perm_1_0.push_back(0);
Permutation perm_1_2_0_3;
perm_1_2_0_3.push_back(1);
perm_1_2_0_3.push_back(2);
perm_1_2_0_3.push_back(0);
perm_1_2_0_3.push_back(3);
Permutation perm_1_3_2_0;
perm_1_3_2_0.push_back(1);
perm_1_3_2_0.push_back(3);
perm_1_3_2_0.push_back(2);
perm_1_3_2_0.push_back(0);
Permutation perm_2_1_0_3;
perm_2_1_0_3.push_back(2);
perm_2_1_0_3.push_back(1);
perm_2_1_0_3.push_back(0);
perm_2_1_0_3.push_back(3);
Permutation perm_3_0_2_1;
perm_3_0_2_1.push_back(3);
perm_3_0_2_1.push_back(0);
perm_3_0_2_1.push_back(2);
perm_3_0_2_1.push_back(1);
ModeArray modes_0;
modes_0.push_back(0);
ModeArray modes_0_1;
modes_0_1.push_back(0);
modes_0_1.push_back(1);
ModeArray modes_0_1_3;
modes_0_1_3.push_back(0);
modes_0_1_3.push_back(1);
modes_0_1_3.push_back(3);
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
ModeArray modes_1_3_2;
modes_1_3_2.push_back(1);
modes_1_3_2.push_back(3);
modes_1_3_2.push_back(2);
ModeArray modes_2;
modes_2.push_back(2);
ModeArray modes_2_1;
modes_2_1.push_back(2);
modes_2_1.push_back(1);
ModeArray modes_2_3;
modes_2_3.push_back(2);
modes_2_3.push_back(3);
ModeArray modes_3;
modes_3.push_back(3);
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
IndexArray indices_bn( 2 );
indices_bn[0] = 'b';
indices_bn[1] = 'n';
IndexArray indices_bnfj( 4 );
indices_bnfj[0] = 'b';
indices_bnfj[1] = 'n';
indices_bnfj[2] = 'f';
indices_bnfj[3] = 'j';
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
	//X_bmej[D0,D1,D2,D3]
DistTensor<double> X_bmej__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//X_bmej_part1B[D0,D1,D2,D3]
DistTensor<double> X_bmej_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//X_bmej_part1T[D0,D1,D2,D3]
DistTensor<double> X_bmej_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//X_bmej_part1_0[D0,D1,D2,D3]
DistTensor<double> X_bmej_part1_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//X_bmej_part1_1[D0,D1,D2,D3]
DistTensor<double> X_bmej_part1_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//X_bmej_part1_1_part3B[D0,D1,D2,D3]
DistTensor<double> X_bmej_part1_1_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//X_bmej_part1_1_part3T[D0,D1,D2,D3]
DistTensor<double> X_bmej_part1_1_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//X_bmej_part1_1_part3_0[D0,D1,D2,D3]
DistTensor<double> X_bmej_part1_1_part3_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//X_bmej_part1_1_part3_1[D0,D1,D2,D3]
DistTensor<double> X_bmej_part1_1_part3_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//X_bmej_part1_1_part3_1[D0,D1,D2,*,D3]
DistTensor<double> X_bmej_part1_1_part3_1__D_0__D_1__D_2__S__D_3( dist__D_0__D_1__D_2__S__D_3, g );
	//X_bmej_part1_1_part3_2[D0,D1,D2,D3]
DistTensor<double> X_bmej_part1_1_part3_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//X_bmej_part1_1[D0,D1,D2,D3]
DistTensor<double> X_bmej_part1_1_perm1320__D_1__D_3__D_2__D_0( dist__D_0__D_1__D_2__D_3, g );
X_bmej_part1_1_perm1320__D_1__D_3__D_2__D_0.SetLocalPermutation( perm_1_3_2_0 );
	//X_bmej_part1_2[D0,D1,D2,D3]
DistTensor<double> X_bmej_part1_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//X_bmej[D0,D1,D2,D3]
DistTensor<double> X_bmej_perm2103__D_2__D_1__D_0__D_3( dist__D_0__D_1__D_2__D_3, g );
X_bmej_perm2103__D_2__D_1__D_0__D_3.SetLocalPermutation( perm_2_1_0_3 );
	//r_bmfe[D0,D1,D2,D3]
DistTensor<double> r_bmfe__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//r_bmfe_part1B[D0,D1,D2,D3]
DistTensor<double> r_bmfe_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//r_bmfe_part1T[D0,D1,D2,D3]
DistTensor<double> r_bmfe_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//r_bmfe_part1_0[D0,D1,D2,D3]
DistTensor<double> r_bmfe_part1_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//r_bmfe_part1_1[D0,D1,D2,D3]
DistTensor<double> r_bmfe_part1_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//r_bmfe_part1_2[D0,D1,D2,D3]
DistTensor<double> r_bmfe_part1_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
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
	//t_fj_part1_1[D01,D3]
DistTensor<double> t_fj_part1_1__D_0_1__D_3( dist__D_0_1__D_3, g );
	//t_fj_part1_1[D0,D2]
DistTensor<double> t_fj_part1_1__D_0__D_2( dist__D_0__D_2, g );
	//t_fj_part1_1[D1,D3]
DistTensor<double> t_fj_part1_1__D_1__D_3( dist__D_1__D_3, g );
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
	//temp1_part1_1_part2_1[D0,*,*,D3]
DistTensor<double> temp1_part1_1_part2_1_perm1203__S__S__D_0__D_3( dist__D_0__S__S__D_3, g );
temp1_part1_1_part2_1_perm1203__S__S__D_0__D_3.SetLocalPermutation( perm_1_2_0_3 );
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
	//temp1_part2_1_part3_1[D0,D1,D2,D3]
DistTensor<double> temp1_part2_1_part3_1_perm0213__D_0__D_2__D_1__D_3( dist__D_0__D_1__D_2__D_3, g );
temp1_part2_1_part3_1_perm0213__D_0__D_2__D_1__D_3.SetLocalPermutation( perm_0_2_1_3 );
	//temp1_part2_1_part3_2[D0,D1,D2,D3]
DistTensor<double> temp1_part2_1_part3_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//temp1_part2_2[D0,D1,D2,D3]
DistTensor<double> temp1_part2_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
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
	//u_mnje_part0_1_part1_1[D0,D1,D3,D2]
DistTensor<double> u_mnje_part0_1_part1_1__D_0__D_1__D_3__D_2( dist__D_0__D_1__D_3__D_2, g );
	//u_mnje_part0_1_part1_1[D1,*,D3,D2]
DistTensor<double> u_mnje_part0_1_part1_1_perm0231__D_1__D_3__D_2__S( dist__D_1__S__D_3__D_2, g );
u_mnje_part0_1_part1_1_perm0231__D_1__D_3__D_2__S.SetLocalPermutation( perm_0_2_3_1 );
	//u_mnje_part0_1_part1_2[D0,D1,D2,D3]
DistTensor<double> u_mnje_part0_1_part1_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//u_mnje_part0_2[D0,D1,D2,D3]
DistTensor<double> u_mnje_part0_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//v_femn[D0,D1,D2,D3]
DistTensor<double> v_femn__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//v_femn_part0B[D0,D1,D2,D3]
DistTensor<double> v_femn_part0B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//v_femn_part0T[D0,D1,D2,D3]
DistTensor<double> v_femn_part0T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//v_femn_part0_0[D0,D1,D2,D3]
DistTensor<double> v_femn_part0_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//v_femn_part0_1[D0,D1,D2,D3]
DistTensor<double> v_femn_part0_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//v_femn_part0_1_part3B[D0,D1,D2,D3]
DistTensor<double> v_femn_part0_1_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//v_femn_part0_1_part3T[D0,D1,D2,D3]
DistTensor<double> v_femn_part0_1_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//v_femn_part0_1_part3_0[D0,D1,D2,D3]
DistTensor<double> v_femn_part0_1_part3_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//v_femn_part0_1_part3_1[D0,D1,D2,D3]
DistTensor<double> v_femn_part0_1_part3_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//v_femn_part0_1_part3_1[D0,D2,D1,D3]
DistTensor<double> v_femn_part0_1_part3_1__D_0__D_2__D_1__D_3( dist__D_0__D_2__D_1__D_3, g );
	//v_femn_part0_1_part3_1[*,D2,D1,*]
DistTensor<double> v_femn_part0_1_part3_1_perm1203__D_2__D_1__S__S( dist__S__D_2__D_1__S, g );
v_femn_part0_1_part3_1_perm1203__D_2__D_1__S__S.SetLocalPermutation( perm_1_2_0_3 );
	//v_femn_part0_1_part3_2[D0,D1,D2,D3]
DistTensor<double> v_femn_part0_1_part3_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//v_femn_part0_2[D0,D1,D2,D3]
DistTensor<double> v_femn_part0_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//x_bmej[D0,D1,D2,D3]
DistTensor<double> x_bmej__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
// r_bmfe has 4 dims
//	Starting distribution: [D0,D1,D2,D3] or _D_0__D_1__D_2__D_3
ObjShape r_bmfe__D_0__D_1__D_2__D_3_tempShape;
r_bmfe__D_0__D_1__D_2__D_3_tempShape.push_back( 500 );
r_bmfe__D_0__D_1__D_2__D_3_tempShape.push_back( 50 );
r_bmfe__D_0__D_1__D_2__D_3_tempShape.push_back( 500 );
r_bmfe__D_0__D_1__D_2__D_3_tempShape.push_back( 500 );
r_bmfe__D_0__D_1__D_2__D_3.ResizeTo( r_bmfe__D_0__D_1__D_2__D_3_tempShape );
MakeUniform( r_bmfe__D_0__D_1__D_2__D_3 );
// t_fj has 2 dims
//	Starting distribution: [D01,D23] or _D_0_1__D_2_3
ObjShape t_fj__D_0_1__D_2_3_tempShape;
t_fj__D_0_1__D_2_3_tempShape.push_back( 500 );
t_fj__D_0_1__D_2_3_tempShape.push_back( 50 );
t_fj__D_0_1__D_2_3.ResizeTo( t_fj__D_0_1__D_2_3_tempShape );
MakeUniform( t_fj__D_0_1__D_2_3 );
// u_mnje has 4 dims
//	Starting distribution: [D0,D1,D2,D3] or _D_0__D_1__D_2__D_3
ObjShape u_mnje__D_0__D_1__D_2__D_3_tempShape;
u_mnje__D_0__D_1__D_2__D_3_tempShape.push_back( 50 );
u_mnje__D_0__D_1__D_2__D_3_tempShape.push_back( 50 );
u_mnje__D_0__D_1__D_2__D_3_tempShape.push_back( 50 );
u_mnje__D_0__D_1__D_2__D_3_tempShape.push_back( 500 );
u_mnje__D_0__D_1__D_2__D_3.ResizeTo( u_mnje__D_0__D_1__D_2__D_3_tempShape );
MakeUniform( u_mnje__D_0__D_1__D_2__D_3 );
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
tempShape = T_bfnj__D_0__D_1__D_2__D_3.Shape();
temp1__D_0__D_1__D_2__D_3.ResizeTo( tempShape );
// x_bmej has 4 dims
//	Starting distribution: [D0,D1,D2,D3] or _D_0__D_1__D_2__D_3
ObjShape x_bmej__D_0__D_1__D_2__D_3_tempShape;
x_bmej__D_0__D_1__D_2__D_3_tempShape.push_back( 500 );
x_bmej__D_0__D_1__D_2__D_3_tempShape.push_back( 50 );
x_bmej__D_0__D_1__D_2__D_3_tempShape.push_back( 500 );
x_bmej__D_0__D_1__D_2__D_3_tempShape.push_back( 50 );
x_bmej__D_0__D_1__D_2__D_3.ResizeTo( x_bmej__D_0__D_1__D_2__D_3_tempShape );
MakeUniform( x_bmej__D_0__D_1__D_2__D_3 );
// X_bmej has 4 dims
//	Starting distribution: [D0,D1,D2,D3] or _D_0__D_1__D_2__D_3
ObjShape X_bmej__D_0__D_1__D_2__D_3_tempShape;
X_bmej__D_0__D_1__D_2__D_3_tempShape.push_back( 500 );
X_bmej__D_0__D_1__D_2__D_3_tempShape.push_back( 50 );
X_bmej__D_0__D_1__D_2__D_3_tempShape.push_back( 500 );
X_bmej__D_0__D_1__D_2__D_3_tempShape.push_back( 50 );
X_bmej__D_0__D_1__D_2__D_3.ResizeTo( X_bmej__D_0__D_1__D_2__D_3_tempShape );
MakeUniform( X_bmej__D_0__D_1__D_2__D_3 );
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
			   // 1.0 * t_fj_part1_1[D0,D2]_bn * t_fj_part1_1[D1,D3]_fj + 0.0 * temp1_part2_1_part3_1[D0,D1,D2,D3]_bnfj
			LocalContractAndLocalEliminate(1.0, t_fj_part1_1__D_0__D_2.LockedTensor(), indices_bn, false,
				t_fj_part1_1__D_1__D_3.LockedTensor(), indices_fj, false,
				0.0, temp1_part2_1_part3_1_perm0213__D_0__D_2__D_1__D_3.Tensor(), indices_bnfj, false);
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

	x_bmej__D_0__D_1__D_2__D_3 = X_bmej__D_0__D_1__D_2__D_3;
	x_bmej__D_0__D_1__D_2__D_3.EmptyData();


//****
//**** (out of 1)

	YAxpy( 0.5, T_bfnj__D_0__D_1__D_2__D_3, temp1__D_0__D_1__D_2__D_3 );
	T_bfnj__D_0__D_1__D_2__D_3.EmptyData();


//****
//**** (out of 1)

	Scal( 0.0, X_bmej__D_0__D_1__D_2__D_3 );
	Permute( X_bmej__D_0__D_1__D_2__D_3, X_bmej_perm2103__D_2__D_1__D_0__D_3 );
	//**** (out of 1)
	//**** Is real	0 shadows
		//Outputs:
		//  X_bmej_perm2103__D_2__D_1__D_0__D_3
	PartitionDown(v_femn__D_0__D_1__D_2__D_3, v_femn_part0T__D_0__D_1__D_2__D_3, v_femn_part0B__D_0__D_1__D_2__D_3, 0, 0);
	PartitionDown(temp1__D_0__D_1__D_2__D_3, temp1_part1T__D_0__D_1__D_2__D_3, temp1_part1B__D_0__D_1__D_2__D_3, 1, 0);
	while(v_femn_part0T__D_0__D_1__D_2__D_3.Dimension(0) < v_femn__D_0__D_1__D_2__D_3.Dimension(0))
	{
		RepartitionDown
		( v_femn_part0T__D_0__D_1__D_2__D_3,  v_femn_part0_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       v_femn_part0_1__D_0__D_1__D_2__D_3,
		  v_femn_part0B__D_0__D_1__D_2__D_3, v_femn_part0_2__D_0__D_1__D_2__D_3, 0, 32 );
		RepartitionDown
		( temp1_part1T__D_0__D_1__D_2__D_3,  temp1_part1_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       temp1_part1_1__D_0__D_1__D_2__D_3,
		  temp1_part1B__D_0__D_1__D_2__D_3, temp1_part1_2__D_0__D_1__D_2__D_3, 1, 32 );

		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  X_bmej_perm2103__D_2__D_1__D_0__D_3
		PartitionDown(v_femn_part0_1__D_0__D_1__D_2__D_3, v_femn_part0_1_part3T__D_0__D_1__D_2__D_3, v_femn_part0_1_part3B__D_0__D_1__D_2__D_3, 3, 0);
		PartitionDown(temp1_part1_1__D_0__D_1__D_2__D_3, temp1_part1_1_part2T__D_0__D_1__D_2__D_3, temp1_part1_1_part2B__D_0__D_1__D_2__D_3, 2, 0);
		while(v_femn_part0_1_part3T__D_0__D_1__D_2__D_3.Dimension(3) < v_femn_part0_1__D_0__D_1__D_2__D_3.Dimension(3))
		{
			RepartitionDown
			( v_femn_part0_1_part3T__D_0__D_1__D_2__D_3,  v_femn_part0_1_part3_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       v_femn_part0_1_part3_1__D_0__D_1__D_2__D_3,
			  v_femn_part0_1_part3B__D_0__D_1__D_2__D_3, v_femn_part0_1_part3_2__D_0__D_1__D_2__D_3, 3, 32 );
			RepartitionDown
			( temp1_part1_1_part2T__D_0__D_1__D_2__D_3,  temp1_part1_1_part2_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       temp1_part1_1_part2_1__D_0__D_1__D_2__D_3,
			  temp1_part1_1_part2B__D_0__D_1__D_2__D_3, temp1_part1_1_part2_2__D_0__D_1__D_2__D_3, 2, 32 );

			   // v_femn_part0_1_part3_1[D0,D2,D1,D3] <- v_femn_part0_1_part3_1[D0,D1,D2,D3]
			v_femn_part0_1_part3_1__D_0__D_2__D_1__D_3.AlignModesWith( modes_1_2, X_bmej__D_0__D_1__D_2__D_3, modes_2_1 );
			v_femn_part0_1_part3_1__D_0__D_2__D_1__D_3.AllToAllRedistFrom( v_femn_part0_1_part3_1__D_0__D_1__D_2__D_3, modes_1_2 );
			   // v_femn_part0_1_part3_1[*,D2,D1,*] <- v_femn_part0_1_part3_1[D0,D2,D1,D3]
			v_femn_part0_1_part3_1_perm1203__D_2__D_1__S__S.AlignModesWith( modes_1_2, X_bmej__D_0__D_1__D_2__D_3, modes_2_1 );
			v_femn_part0_1_part3_1_perm1203__D_2__D_1__S__S.AllGatherRedistFrom( v_femn_part0_1_part3_1__D_0__D_2__D_1__D_3, modes_0_3 );
			v_femn_part0_1_part3_1__D_0__D_2__D_1__D_3.EmptyData();
			   // temp1_part1_1_part2_1[D0,*,*,D3] <- temp1_part1_1_part2_1[D0,D1,D2,D3]
			temp1_part1_1_part2_1_perm1203__S__S__D_0__D_3.AlignModesWith( modes_0_3, X_bmej__D_0__D_1__D_2__D_3, modes_0_3 );
			temp1_part1_1_part2_1_perm1203__S__S__D_0__D_3.AllGatherRedistFrom( temp1_part1_1_part2_1__D_0__D_1__D_2__D_3, modes_1_2 );
			   // -1.0 * v_femn_part0_1_part3_1[*,D2,D1,*]_emfn * temp1_part1_1_part2_1[D0,*,*,D3]_fnbj + 1.0 * X_bmej[D0,D1,D2,D3]_embj
			LocalContractAndLocalEliminate(-1.0, v_femn_part0_1_part3_1_perm1203__D_2__D_1__S__S.LockedTensor(), indices_emfn, false,
				temp1_part1_1_part2_1_perm1203__S__S__D_0__D_3.LockedTensor(), indices_fnbj, false,
				1.0, X_bmej_perm2103__D_2__D_1__D_0__D_3.Tensor(), indices_embj, false);
			temp1_part1_1_part2_1_perm1203__S__S__D_0__D_3.EmptyData();
			v_femn_part0_1_part3_1_perm1203__D_2__D_1__S__S.EmptyData();

			SlidePartitionDown
			( v_femn_part0_1_part3T__D_0__D_1__D_2__D_3,  v_femn_part0_1_part3_0__D_0__D_1__D_2__D_3,
			       v_femn_part0_1_part3_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  v_femn_part0_1_part3B__D_0__D_1__D_2__D_3, v_femn_part0_1_part3_2__D_0__D_1__D_2__D_3, 3 );
			SlidePartitionDown
			( temp1_part1_1_part2T__D_0__D_1__D_2__D_3,  temp1_part1_1_part2_0__D_0__D_1__D_2__D_3,
			       temp1_part1_1_part2_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  temp1_part1_1_part2B__D_0__D_1__D_2__D_3, temp1_part1_1_part2_2__D_0__D_1__D_2__D_3, 2 );

		}
		//****

		SlidePartitionDown
		( v_femn_part0T__D_0__D_1__D_2__D_3,  v_femn_part0_0__D_0__D_1__D_2__D_3,
		       v_femn_part0_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  v_femn_part0B__D_0__D_1__D_2__D_3, v_femn_part0_2__D_0__D_1__D_2__D_3, 0 );
		SlidePartitionDown
		( temp1_part1T__D_0__D_1__D_2__D_3,  temp1_part1_0__D_0__D_1__D_2__D_3,
		       temp1_part1_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  temp1_part1B__D_0__D_1__D_2__D_3, temp1_part1_2__D_0__D_1__D_2__D_3, 1 );

	}
	//****
	Permute( X_bmej_perm2103__D_2__D_1__D_0__D_3, X_bmej__D_0__D_1__D_2__D_3 );
	X_bmej_perm2103__D_2__D_1__D_0__D_3.EmptyData();
	//**** (out of 1)
	//**** Is real	0 shadows
		//Outputs:
		//  X_bmej__D_0__D_1__D_2__D_3
	PartitionDown(r_bmfe__D_0__D_1__D_2__D_3, r_bmfe_part1T__D_0__D_1__D_2__D_3, r_bmfe_part1B__D_0__D_1__D_2__D_3, 1, 0);
	PartitionDown(u_mnje__D_0__D_1__D_2__D_3, u_mnje_part0T__D_0__D_1__D_2__D_3, u_mnje_part0B__D_0__D_1__D_2__D_3, 0, 0);
	PartitionDown(X_bmej__D_0__D_1__D_2__D_3, X_bmej_part1T__D_0__D_1__D_2__D_3, X_bmej_part1B__D_0__D_1__D_2__D_3, 1, 0);
	while(X_bmej_part1T__D_0__D_1__D_2__D_3.Dimension(1) < X_bmej__D_0__D_1__D_2__D_3.Dimension(1))
	{
		RepartitionDown
		( r_bmfe_part1T__D_0__D_1__D_2__D_3,  r_bmfe_part1_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       r_bmfe_part1_1__D_0__D_1__D_2__D_3,
		  r_bmfe_part1B__D_0__D_1__D_2__D_3, r_bmfe_part1_2__D_0__D_1__D_2__D_3, 1, 32 );
		RepartitionDown
		( u_mnje_part0T__D_0__D_1__D_2__D_3,  u_mnje_part0_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       u_mnje_part0_1__D_0__D_1__D_2__D_3,
		  u_mnje_part0B__D_0__D_1__D_2__D_3, u_mnje_part0_2__D_0__D_1__D_2__D_3, 0, 32 );
		RepartitionDown
		( X_bmej_part1T__D_0__D_1__D_2__D_3,  X_bmej_part1_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       X_bmej_part1_1__D_0__D_1__D_2__D_3,
		  X_bmej_part1B__D_0__D_1__D_2__D_3, X_bmej_part1_2__D_0__D_1__D_2__D_3, 1, 32 );

		Permute( X_bmej_part1_1__D_0__D_1__D_2__D_3, X_bmej_part1_1_perm1320__D_1__D_3__D_2__D_0 );
		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  X_bmej_part1_1_perm1320__D_1__D_3__D_2__D_0
		PartitionDown(u_mnje_part0_1__D_0__D_1__D_2__D_3, u_mnje_part0_1_part1T__D_0__D_1__D_2__D_3, u_mnje_part0_1_part1B__D_0__D_1__D_2__D_3, 1, 0);
		PartitionDown(t_fj__D_0_1__D_2_3, t_fj_part1T__D_0_1__D_2_3, t_fj_part1B__D_0_1__D_2_3, 1, 0);
		while(u_mnje_part0_1_part1T__D_0__D_1__D_2__D_3.Dimension(1) < u_mnje_part0_1__D_0__D_1__D_2__D_3.Dimension(1))
		{
			RepartitionDown
			( u_mnje_part0_1_part1T__D_0__D_1__D_2__D_3,  u_mnje_part0_1_part1_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       u_mnje_part0_1_part1_1__D_0__D_1__D_2__D_3,
			  u_mnje_part0_1_part1B__D_0__D_1__D_2__D_3, u_mnje_part0_1_part1_2__D_0__D_1__D_2__D_3, 1, 32 );
			RepartitionDown
			( t_fj_part1T__D_0_1__D_2_3,  t_fj_part1_0__D_0_1__D_2_3,
			  /**/ /**/
			       t_fj_part1_1__D_0_1__D_2_3,
			  t_fj_part1B__D_0_1__D_2_3, t_fj_part1_2__D_0_1__D_2_3, 1, 32 );

			   // u_mnje_part0_1_part1_1[D0,D1,D3,D2] <- u_mnje_part0_1_part1_1[D0,D1,D2,D3]
			u_mnje_part0_1_part1_1__D_0__D_1__D_3__D_2.AlignModesWith( modes_0_2_3, X_bmej_part1_1__D_0__D_1__D_2__D_3, modes_1_3_2 );
			u_mnje_part0_1_part1_1__D_0__D_1__D_3__D_2.AllToAllRedistFrom( u_mnje_part0_1_part1_1__D_0__D_1__D_2__D_3, modes_2_3 );
			   // u_mnje_part0_1_part1_1[D1,*,D3,D2] <- u_mnje_part0_1_part1_1[D0,D1,D3,D2]
			u_mnje_part0_1_part1_1_perm0231__D_1__D_3__D_2__S.AlignModesWith( modes_0_2_3, X_bmej_part1_1__D_0__D_1__D_2__D_3, modes_1_3_2 );
			u_mnje_part0_1_part1_1_perm0231__D_1__D_3__D_2__S.AllToAllRedistFrom( u_mnje_part0_1_part1_1__D_0__D_1__D_3__D_2, modes_0_1 );
			u_mnje_part0_1_part1_1__D_0__D_1__D_3__D_2.EmptyData();
			   // t_fj_part1_1[D0,*] <- t_fj_part1_1[D01,D23]
			t_fj_part1_1_perm10__S__D_0.AlignModesWith( modes_0, X_bmej_part1_1__D_0__D_1__D_2__D_3, modes_0 );
			t_fj_part1_1_perm10__S__D_0.AllGatherRedistFrom( t_fj_part1_1__D_0_1__D_2_3, modes_1_2_3 );
			   // -1.0 * u_mnje_part0_1_part1_1[D1,*,D3,D2]_mjen * t_fj_part1_1[D0,*]_nb + 1.0 * X_bmej_part1_1[D0,D1,D2,D3]_mjeb
			LocalContractAndLocalEliminate(-1.0, u_mnje_part0_1_part1_1_perm0231__D_1__D_3__D_2__S.LockedTensor(), indices_mjen, false,
				t_fj_part1_1_perm10__S__D_0.LockedTensor(), indices_nb, false,
				1.0, X_bmej_part1_1_perm1320__D_1__D_3__D_2__D_0.Tensor(), indices_mjeb, false);
			t_fj_part1_1_perm10__S__D_0.EmptyData();
			u_mnje_part0_1_part1_1_perm0231__D_1__D_3__D_2__S.EmptyData();

			SlidePartitionDown
			( u_mnje_part0_1_part1T__D_0__D_1__D_2__D_3,  u_mnje_part0_1_part1_0__D_0__D_1__D_2__D_3,
			       u_mnje_part0_1_part1_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  u_mnje_part0_1_part1B__D_0__D_1__D_2__D_3, u_mnje_part0_1_part1_2__D_0__D_1__D_2__D_3, 1 );
			SlidePartitionDown
			( t_fj_part1T__D_0_1__D_2_3,  t_fj_part1_0__D_0_1__D_2_3,
			       t_fj_part1_1__D_0_1__D_2_3,
			  /**/ /**/
			  t_fj_part1B__D_0_1__D_2_3, t_fj_part1_2__D_0_1__D_2_3, 1 );

		}
		//****
		Permute( X_bmej_part1_1_perm1320__D_1__D_3__D_2__D_0, X_bmej_part1_1__D_0__D_1__D_2__D_3 );
		X_bmej_part1_1_perm1320__D_1__D_3__D_2__D_0.EmptyData();
		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  X_bmej_part1_1__D_0__D_1__D_2__D_3
		PartitionDown(t_fj__D_0_1__D_2_3, t_fj_part1T__D_0_1__D_2_3, t_fj_part1B__D_0_1__D_2_3, 1, 0);
		PartitionDown(X_bmej_part1_1__D_0__D_1__D_2__D_3, X_bmej_part1_1_part3T__D_0__D_1__D_2__D_3, X_bmej_part1_1_part3B__D_0__D_1__D_2__D_3, 3, 0);
		while(X_bmej_part1_1_part3T__D_0__D_1__D_2__D_3.Dimension(3) < X_bmej_part1_1__D_0__D_1__D_2__D_3.Dimension(3))
		{
			RepartitionDown
			( t_fj_part1T__D_0_1__D_2_3,  t_fj_part1_0__D_0_1__D_2_3,
			  /**/ /**/
			       t_fj_part1_1__D_0_1__D_2_3,
			  t_fj_part1B__D_0_1__D_2_3, t_fj_part1_2__D_0_1__D_2_3, 1, 32 );
			RepartitionDown
			( X_bmej_part1_1_part3T__D_0__D_1__D_2__D_3,  X_bmej_part1_1_part3_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       X_bmej_part1_1_part3_1__D_0__D_1__D_2__D_3,
			  X_bmej_part1_1_part3B__D_0__D_1__D_2__D_3, X_bmej_part1_1_part3_2__D_0__D_1__D_2__D_3, 3, 32 );

			tempShape = X_bmej_part1_1_part3_1__D_0__D_1__D_2__D_3.Shape();
			tempShape.push_back( g.Shape()[3] );
			X_bmej_part1_1_part3_1__D_0__D_1__D_2__S__D_3.ResizeTo( tempShape );
			   // t_fj_part1_1[D3,D2] <- t_fj_part1_1[D01,D23]
			t_fj_part1_1__D_3__D_2.AlignModesWith( modes_0, r_bmfe_part1_1__D_0__D_1__D_2__D_3, modes_3 );
			t_fj_part1_1__D_3__D_2.AllToAllRedistFrom( t_fj_part1_1__D_0_1__D_2_3, modes_0_1_3 );
			   // t_fj_part1_1[D3,*] <- t_fj_part1_1[D3,D2]
			t_fj_part1_1__D_3__S.AlignModesWith( modes_0, r_bmfe_part1_1__D_0__D_1__D_2__D_3, modes_3 );
			t_fj_part1_1__D_3__S.AllGatherRedistFrom( t_fj_part1_1__D_3__D_2, modes_2 );
			   // 1.0 * r_bmfe_part1_1[D0,D1,D2,D3]_bmef * t_fj_part1_1[D3,*]_fj + 0.0 * X_bmej_part1_1_part3_1[D0,D1,D2,*,D3]_bmejf
			LocalContract(1.0, r_bmfe_part1_1__D_0__D_1__D_2__D_3.LockedTensor(), indices_bmef, false,
				t_fj_part1_1__D_3__S.LockedTensor(), indices_fj, false,
				0.0, X_bmej_part1_1_part3_1__D_0__D_1__D_2__S__D_3.Tensor(), indices_bmejf, false);
			   // X_bmej_part1_1_part3_1[D0,D1,D2,D3] <- X_bmej_part1_1_part3_1[D0,D1,D2,*,D3] (with SumScatter on D3)
			X_bmej_part1_1_part3_1__D_0__D_1__D_2__D_3.ReduceScatterUpdateRedistFrom( X_bmej_part1_1_part3_1__D_0__D_1__D_2__S__D_3, 1.0, 4 );
			X_bmej_part1_1_part3_1__D_0__D_1__D_2__S__D_3.EmptyData();
			t_fj_part1_1__D_3__S.EmptyData();
			t_fj_part1_1__D_3__D_2.EmptyData();

			SlidePartitionDown
			( t_fj_part1T__D_0_1__D_2_3,  t_fj_part1_0__D_0_1__D_2_3,
			       t_fj_part1_1__D_0_1__D_2_3,
			  /**/ /**/
			  t_fj_part1B__D_0_1__D_2_3, t_fj_part1_2__D_0_1__D_2_3, 1 );
			SlidePartitionDown
			( X_bmej_part1_1_part3T__D_0__D_1__D_2__D_3,  X_bmej_part1_1_part3_0__D_0__D_1__D_2__D_3,
			       X_bmej_part1_1_part3_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  X_bmej_part1_1_part3B__D_0__D_1__D_2__D_3, X_bmej_part1_1_part3_2__D_0__D_1__D_2__D_3, 3 );

		}
		//****

		SlidePartitionDown
		( r_bmfe_part1T__D_0__D_1__D_2__D_3,  r_bmfe_part1_0__D_0__D_1__D_2__D_3,
		       r_bmfe_part1_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  r_bmfe_part1B__D_0__D_1__D_2__D_3, r_bmfe_part1_2__D_0__D_1__D_2__D_3, 1 );
		SlidePartitionDown
		( X_bmej_part1T__D_0__D_1__D_2__D_3,  X_bmej_part1_0__D_0__D_1__D_2__D_3,
		       X_bmej_part1_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  X_bmej_part1B__D_0__D_1__D_2__D_3, X_bmej_part1_2__D_0__D_1__D_2__D_3, 1 );
		SlidePartitionDown
		( u_mnje_part0T__D_0__D_1__D_2__D_3,  u_mnje_part0_0__D_0__D_1__D_2__D_3,
		       u_mnje_part0_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  u_mnje_part0B__D_0__D_1__D_2__D_3, u_mnje_part0_2__D_0__D_1__D_2__D_3, 0 );

	}
	//****


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


