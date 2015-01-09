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
TensorDistribution dist__D_0__S = tmen::StringToTensorDist("[(0),()]");
TensorDistribution dist__D_0__D_1__D_2__D_3 = tmen::StringToTensorDist("[(0),(1),(2),(3)]");
TensorDistribution dist__D_2_0__D_1__S__D_3 = tmen::StringToTensorDist("[(2,0),(1),(),(3)]");
TensorDistribution dist__D_2_0__D_3__S__D_1 = tmen::StringToTensorDist("[(2,0),(3),(),(1)]");
TensorDistribution dist__D_2__D_3__S__D_1__D_0 = tmen::StringToTensorDist("[(2),(3),(),(1),(0)]");
TensorDistribution dist__D_0_1__D_2_3 = tmen::StringToTensorDist("[(0,1),(2,3)]");
Permutation perm_0_1;
perm_0_1.push_back(0);
perm_0_1.push_back(1);
Permutation perm_0_1_2_3;
perm_0_1_2_3.push_back(0);
perm_0_1_2_3.push_back(1);
perm_0_1_2_3.push_back(2);
perm_0_1_2_3.push_back(3);
Permutation perm_1_2_3_0;
perm_1_2_3_0.push_back(1);
perm_1_2_3_0.push_back(2);
perm_1_2_3_0.push_back(3);
perm_1_2_3_0.push_back(0);
Permutation perm_3_0_1_2_4;
perm_3_0_1_2_4.push_back(3);
perm_3_0_1_2_4.push_back(0);
perm_3_0_1_2_4.push_back(1);
perm_3_0_1_2_4.push_back(2);
perm_3_0_1_2_4.push_back(4);
ModeArray modes_0;
modes_0.push_back(0);
ModeArray modes_0_1_2_3;
modes_0_1_2_3.push_back(0);
modes_0_1_2_3.push_back(1);
modes_0_1_2_3.push_back(2);
modes_0_1_2_3.push_back(3);
ModeArray modes_0_2;
modes_0_2.push_back(0);
modes_0_2.push_back(2);
ModeArray modes_1_2_3;
modes_1_2_3.push_back(1);
modes_1_2_3.push_back(2);
modes_1_2_3.push_back(3);
ModeArray modes_1_3;
modes_1_3.push_back(1);
modes_1_3.push_back(3);
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
	//U_mnie_part1B[D0,D1,D2,D3]
DistTensor<double> U_mnie_part1B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//U_mnie_part1T[D0,D1,D2,D3]
DistTensor<double> U_mnie_part1T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//U_mnie_part1_0[D0,D1,D2,D3]
DistTensor<double> U_mnie_part1_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//U_mnie_part1_1[D0,D1,D2,D3]
DistTensor<double> U_mnie_part1_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//U_mnie_part1_1_part2B[D0,D1,D2,D3]
DistTensor<double> U_mnie_part1_1_part2B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//U_mnie_part1_1_part2T[D0,D1,D2,D3]
DistTensor<double> U_mnie_part1_1_part2T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//U_mnie_part1_1_part2_0[D0,D1,D2,D3]
DistTensor<double> U_mnie_part1_1_part2_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//U_mnie_part1_1_part2_1[D0,D1,D2,D3]
DistTensor<double> U_mnie_part1_1_part2_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//U_mnie_part1_1_part2_1[D2,D3,*,D1,D0]
DistTensor<double> U_mnie_part1_1_part2_1_perm30124__D_1__D_2__D_3__S__D_0( dist__D_2__D_3__S__D_1__D_0, g );
U_mnie_part1_1_part2_1_perm30124__D_1__D_2__D_3__S__D_0.SetLocalPermutation( perm_3_0_1_2_4 );
	//U_mnie_part1_1_part2_1_temp[D0,D1,D2,D3]
DistTensor<double> U_mnie_part1_1_part2_1_temp__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//U_mnie_part1_1_part2_1_temp[D20,D1,*,D3]
DistTensor<double> U_mnie_part1_1_part2_1_temp__D_2_0__D_1__S__D_3( dist__D_2_0__D_1__S__D_3, g );
	//U_mnie_part1_1_part2_1_temp[D20,D3,*,D1]
DistTensor<double> U_mnie_part1_1_part2_1_temp__D_2_0__D_3__S__D_1( dist__D_2_0__D_3__S__D_1, g );
	//U_mnie_part1_1_part2_2[D0,D1,D2,D3]
DistTensor<double> U_mnie_part1_1_part2_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//U_mnie_part1_2[D0,D1,D2,D3]
DistTensor<double> U_mnie_part1_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
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
	//t_fj_part1_1[D0,*]
DistTensor<double> t_fj_part1_1__D_0__S( dist__D_0__S, g );
	//t_fj_part1_2[D01,D23]
DistTensor<double> t_fj_part1_2__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//u_mnje[D0,D1,D2,D3]
DistTensor<double> u_mnje__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//v_femn[D0,D1,D2,D3]
DistTensor<double> v_femn__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//v_femn_part3B[D0,D1,D2,D3]
DistTensor<double> v_femn_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//v_femn_part3T[D0,D1,D2,D3]
DistTensor<double> v_femn_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//v_femn_part3_0[D0,D1,D2,D3]
DistTensor<double> v_femn_part3_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//v_femn_part3_1[D0,D1,D2,D3]
DistTensor<double> v_femn_part3_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//v_femn_part3_1[D0,D1,D2,D3]
DistTensor<double> v_femn_part3_1_perm1230__D_1__D_2__D_3__D_0( dist__D_0__D_1__D_2__D_3, g );
v_femn_part3_1_perm1230__D_1__D_2__D_3__D_0.SetLocalPermutation( perm_1_2_3_0 );
	//v_femn_part3_2[D0,D1,D2,D3]
DistTensor<double> v_femn_part3_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
// v_femn has 4 dims
//	Starting distribution: [D0,D1,D2,D3] or _D_0__D_1__D_2__D_3
ObjShape v_femn__D_0__D_1__D_2__D_3_tempShape;
v_femn__D_0__D_1__D_2__D_3_tempShape.push_back( n_v );
v_femn__D_0__D_1__D_2__D_3_tempShape.push_back( n_v );
v_femn__D_0__D_1__D_2__D_3_tempShape.push_back( n_o );
v_femn__D_0__D_1__D_2__D_3_tempShape.push_back( n_o );
v_femn__D_0__D_1__D_2__D_3.ResizeTo( v_femn__D_0__D_1__D_2__D_3_tempShape );
MakeUniform( v_femn__D_0__D_1__D_2__D_3 );
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
// U_mnie has 4 dims
//	Starting distribution: [D0,D1,D2,D3] or _D_0__D_1__D_2__D_3
ObjShape U_mnie__D_0__D_1__D_2__D_3_tempShape;
U_mnie__D_0__D_1__D_2__D_3_tempShape.push_back( n_o );
U_mnie__D_0__D_1__D_2__D_3_tempShape.push_back( n_o );
U_mnie__D_0__D_1__D_2__D_3_tempShape.push_back( n_o );
U_mnie__D_0__D_1__D_2__D_3_tempShape.push_back( n_v );
U_mnie__D_0__D_1__D_2__D_3.ResizeTo( U_mnie__D_0__D_1__D_2__D_3_tempShape );
MakeUniform( U_mnie__D_0__D_1__D_2__D_3 );
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
check.ResizeTo(U_mnie__D_0__D_1__D_2__D_3.Shape());
Read(v_femn__D_0__D_1__D_2__D_3, "ccsd_terms/term_v_small", BINARY_FLAT, false);
Read(u_mnie__D_0__D_1__D_2__D_3, "ccsd_terms/term_u_small", BINARY_FLAT, false);
fullName.str("");
fullName.clear();
fullName << "ccsd_terms/term_t_small_iter" << testIter;
Read(t_fj__D_0_1__D_2_3, fullName.str(), BINARY_FLAT, false);
fullName.str("");
fullName.clear();
fullName << "ccsd_terms/term_U_iter" << testIter + 1;
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


	U_mnie__D_0__D_1__D_2__D_3 = u_mnje__D_0__D_1__D_2__D_3;
	u_mnje__D_0__D_1__D_2__D_3.EmptyData();


u_mnje__D_0__D_1__D_2__D_3.EmptyData();
//****
//**** (out of 1)

	//**** (out of 1)
	//**** Is real	0 shadows
		//Outputs:
		//  U_mnie__D_0__D_1__D_2__D_3
	PartitionDown(v_femn__D_0__D_1__D_2__D_3, v_femn_part3T__D_0__D_1__D_2__D_3, v_femn_part3B__D_0__D_1__D_2__D_3, 3, 0);
	PartitionDown(U_mnie__D_0__D_1__D_2__D_3, U_mnie_part1T__D_0__D_1__D_2__D_3, U_mnie_part1B__D_0__D_1__D_2__D_3, 1, 0);
	while(U_mnie_part1T__D_0__D_1__D_2__D_3.Dimension(1) < U_mnie__D_0__D_1__D_2__D_3.Dimension(1))
	{
		RepartitionDown
		( v_femn_part3T__D_0__D_1__D_2__D_3,  v_femn_part3_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       v_femn_part3_1__D_0__D_1__D_2__D_3,
		  v_femn_part3B__D_0__D_1__D_2__D_3, v_femn_part3_2__D_0__D_1__D_2__D_3, 3, blkSize );
		RepartitionDown
		( U_mnie_part1T__D_0__D_1__D_2__D_3,  U_mnie_part1_0__D_0__D_1__D_2__D_3,
		  /**/ /**/
		       U_mnie_part1_1__D_0__D_1__D_2__D_3,
		  U_mnie_part1B__D_0__D_1__D_2__D_3, U_mnie_part1_2__D_0__D_1__D_2__D_3, 1, blkSize );

		Permute( v_femn_part3_1__D_0__D_1__D_2__D_3, v_femn_part3_1_perm1230__D_1__D_2__D_3__D_0 );
		//**** (out of 1)
		//**** Is real	0 shadows
			//Outputs:
			//  U_mnie_part1_1__D_0__D_1__D_2__D_3
		PartitionDown(t_fj__D_0_1__D_2_3, t_fj_part1T__D_0_1__D_2_3, t_fj_part1B__D_0_1__D_2_3, 1, 0);
		PartitionDown(U_mnie_part1_1__D_0__D_1__D_2__D_3, U_mnie_part1_1_part2T__D_0__D_1__D_2__D_3, U_mnie_part1_1_part2B__D_0__D_1__D_2__D_3, 2, 0);
		while(U_mnie_part1_1_part2T__D_0__D_1__D_2__D_3.Dimension(2) < U_mnie_part1_1__D_0__D_1__D_2__D_3.Dimension(2))
		{
			RepartitionDown
			( t_fj_part1T__D_0_1__D_2_3,  t_fj_part1_0__D_0_1__D_2_3,
			  /**/ /**/
			       t_fj_part1_1__D_0_1__D_2_3,
			  t_fj_part1B__D_0_1__D_2_3, t_fj_part1_2__D_0_1__D_2_3, 1, blkSize );
			RepartitionDown
			( U_mnie_part1_1_part2T__D_0__D_1__D_2__D_3,  U_mnie_part1_1_part2_0__D_0__D_1__D_2__D_3,
			  /**/ /**/
			       U_mnie_part1_1_part2_1__D_0__D_1__D_2__D_3,
			  U_mnie_part1_1_part2B__D_0__D_1__D_2__D_3, U_mnie_part1_1_part2_2__D_0__D_1__D_2__D_3, 2, blkSize );

			tempShape = U_mnie_part1_1_part2_1__D_0__D_1__D_2__D_3.Shape();
			tempShape.push_back( g.Shape()[0] );
			U_mnie_part1_1_part2_1_perm30124__D_1__D_2__D_3__S__D_0.ResizeTo( tempShape );
			tempShape = U_mnie_part1_1_part2_1__D_0__D_1__D_2__D_3.Shape();
			U_mnie_part1_1_part2_1_temp__D_2_0__D_3__S__D_1.ResizeTo( tempShape );
			   // t_fj_part1_1[D0,*] <- t_fj_part1_1[D01,D23]
			t_fj_part1_1__D_0__S.AlignModesWith( modes_0, v_femn_part3_1__D_0__D_1__D_2__D_3, modes_0 );
			t_fj_part1_1__D_0__S.AllGatherRedistFrom( t_fj_part1_1__D_0_1__D_2_3, modes_1_2_3 );
			   // 1.0 * v_femn_part3_1[D0,D1,D2,D3]_emnf * t_fj_part1_1[D0,*]_fi + 0.0 * U_mnie_part1_1_part2_1[D2,D3,*,D1,D0]_emnif
			LocalContract(1.0, v_femn_part3_1_perm1230__D_1__D_2__D_3__D_0.LockedTensor(), indices_emnf, false,
				t_fj_part1_1__D_0__S.LockedTensor(), indices_fi, false,
				0.0, U_mnie_part1_1_part2_1_perm30124__D_1__D_2__D_3__S__D_0.Tensor(), indices_emnif, false);
			   // U_mnie_part1_1_part2_1_temp[D20,D3,*,D1] <- U_mnie_part1_1_part2_1[D2,D3,*,D1,D0] (with SumScatter on D0)
			U_mnie_part1_1_part2_1_temp__D_2_0__D_3__S__D_1.ReduceScatterRedistFrom( U_mnie_part1_1_part2_1_perm30124__D_1__D_2__D_3__S__D_0, 4 );
			U_mnie_part1_1_part2_1_perm30124__D_1__D_2__D_3__S__D_0.EmptyData();
			t_fj_part1_1__D_0__S.EmptyData();
			   // U_mnie_part1_1_part2_1_temp[D20,D1,*,D3] <- U_mnie_part1_1_part2_1_temp[D20,D3,*,D1]
			U_mnie_part1_1_part2_1_temp__D_2_0__D_1__S__D_3.AlignModesWith( modes_0_1_2_3, U_mnie_part1_1_part2_1__D_0__D_1__D_2__D_3, modes_0_1_2_3 );
			U_mnie_part1_1_part2_1_temp__D_2_0__D_1__S__D_3.AllToAllRedistFrom( U_mnie_part1_1_part2_1_temp__D_2_0__D_3__S__D_1, modes_1_3 );
			U_mnie_part1_1_part2_1_temp__D_2_0__D_3__S__D_1.EmptyData();
			   // U_mnie_part1_1_part2_1_temp[D0,D1,D2,D3] <- U_mnie_part1_1_part2_1_temp[D20,D1,*,D3]
			U_mnie_part1_1_part2_1_temp__D_0__D_1__D_2__D_3.AlignModesWith( modes_0_1_2_3, U_mnie_part1_1_part2_1__D_0__D_1__D_2__D_3, modes_0_1_2_3 );
			U_mnie_part1_1_part2_1_temp__D_0__D_1__D_2__D_3.AllToAllRedistFrom( U_mnie_part1_1_part2_1_temp__D_2_0__D_1__S__D_3, modes_0_2 );
			YxpBy( U_mnie_part1_1_part2_1_temp__D_0__D_1__D_2__D_3, 1.0, U_mnie_part1_1_part2_1__D_0__D_1__D_2__D_3 );
			U_mnie_part1_1_part2_1_temp__D_0__D_1__D_2__D_3.EmptyData();
			U_mnie_part1_1_part2_1_temp__D_2_0__D_1__S__D_3.EmptyData();

			SlidePartitionDown
			( t_fj_part1T__D_0_1__D_2_3,  t_fj_part1_0__D_0_1__D_2_3,
			       t_fj_part1_1__D_0_1__D_2_3,
			  /**/ /**/
			  t_fj_part1B__D_0_1__D_2_3, t_fj_part1_2__D_0_1__D_2_3, 1 );
			SlidePartitionDown
			( U_mnie_part1_1_part2T__D_0__D_1__D_2__D_3,  U_mnie_part1_1_part2_0__D_0__D_1__D_2__D_3,
			       U_mnie_part1_1_part2_1__D_0__D_1__D_2__D_3,
			  /**/ /**/
			  U_mnie_part1_1_part2B__D_0__D_1__D_2__D_3, U_mnie_part1_1_part2_2__D_0__D_1__D_2__D_3, 2 );

		}
		v_femn_part3_1_perm1230__D_1__D_2__D_3__D_0.EmptyData();
		v_femn_part3_1_perm1230__D_1__D_2__D_3__D_0.EmptyData();
		//****

		SlidePartitionDown
		( v_femn_part3T__D_0__D_1__D_2__D_3,  v_femn_part3_0__D_0__D_1__D_2__D_3,
		       v_femn_part3_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  v_femn_part3B__D_0__D_1__D_2__D_3, v_femn_part3_2__D_0__D_1__D_2__D_3, 3 );
		SlidePartitionDown
		( U_mnie_part1T__D_0__D_1__D_2__D_3,  U_mnie_part1_0__D_0__D_1__D_2__D_3,
		       U_mnie_part1_1__D_0__D_1__D_2__D_3,
		  /**/ /**/
		  U_mnie_part1B__D_0__D_1__D_2__D_3, U_mnie_part1_2__D_0__D_1__D_2__D_3, 1 );

	}
	v_femn__D_0__D_1__D_2__D_3.EmptyData();
	t_fj__D_0_1__D_2_3.EmptyData();
	v_femn__D_0__D_1__D_2__D_3.EmptyData();
	t_fj__D_0_1__D_2_3.EmptyData();
	//****


v_femn__D_0__D_1__D_2__D_3.EmptyData();
t_fj__D_0_1__D_2_3.EmptyData();
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
    Diff(check, U_mnie__D_0__D_1__D_2__D_3, diff);
    norm = Norm(diff);
#endif;

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


