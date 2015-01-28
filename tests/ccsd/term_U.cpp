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
ObjShape overwrite_tmpShape_U;
TensorDistribution dist__D_0__S = tmen::StringToTensorDist("[(0),()]");
TensorDistribution dist__D_0__D_1__D_2__D_3 = tmen::StringToTensorDist("[(0),(1),(2),(3)]");
TensorDistribution dist__D_2_0__D_1__S__D_3 = tmen::StringToTensorDist("[(2,0),(1),(),(3)]");
TensorDistribution dist__D_2_0__D_3__S__D_1 = tmen::StringToTensorDist("[(2,0),(3),(),(1)]");
TensorDistribution dist__D_2__D_3__S__D_1__D_0 = tmen::StringToTensorDist("[(2),(3),(),(1),(0)]");
TensorDistribution dist__D_0_1__D_2_3 = tmen::StringToTensorDist("[(0,1),(2,3)]");
Permutation perm_0_1( 2 );
perm_0_1[0] = 0;
perm_0_1[1] = 1;
Permutation perm_0_1_2_3( 4 );
perm_0_1_2_3[0] = 0;
perm_0_1_2_3[1] = 1;
perm_0_1_2_3[2] = 2;
perm_0_1_2_3[3] = 3;
Permutation perm_1_2_3_0( 4 );
perm_1_2_3_0[0] = 1;
perm_1_2_3_0[1] = 2;
perm_1_2_3_0[2] = 3;
perm_1_2_3_0[3] = 0;
Permutation perm_3_0_1_2_4( 5 );
perm_3_0_1_2_4[0] = 3;
perm_3_0_1_2_4[1] = 0;
perm_3_0_1_2_4[2] = 1;
perm_3_0_1_2_4[3] = 2;
perm_3_0_1_2_4[4] = 4;
ModeArray modes_0( 1 );
modes_0[0] = 0;
ModeArray modes_0_1_2_3( 4 );
modes_0_1_2_3[0] = 0;
modes_0_1_2_3[1] = 1;
modes_0_1_2_3[2] = 2;
modes_0_1_2_3[3] = 3;
ModeArray modes_0_1_3( 3 );
modes_0_1_3[0] = 0;
modes_0_1_3[1] = 1;
modes_0_1_3[2] = 3;
ModeArray modes_0_2( 2 );
modes_0_2[0] = 0;
modes_0_2[1] = 2;
ModeArray modes_1_2_3( 3 );
modes_1_2_3[0] = 1;
modes_1_2_3[1] = 2;
modes_1_2_3[2] = 3;
ModeArray modes_1_3( 2 );
modes_1_3[0] = 1;
modes_1_3[1] = 3;
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
	//t_fj_lvl2_part1_1[D0,*]
DistTensor<double> t_fj_lvl2_part1_1__D_0__S( dist__D_0__S, g );
	//t_fj_lvl2_part1_2[D01,D23]
DistTensor<double> t_fj_lvl2_part1_2__D_0_1__D_2_3( dist__D_0_1__D_2_3, g );
	//u_mnje[D0,D1,D2,D3]
DistTensor<double> u_mnje__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//v_femn[D0,D1,D2,D3]
DistTensor<double> v_femn__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//v_femn_lvl0_part3B[D0,D1,D2,D3]
DistTensor<double> v_femn_lvl0_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//v_femn_lvl0_part3T[D0,D1,D2,D3]
DistTensor<double> v_femn_lvl0_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//v_femn_lvl1_part3B[D0,D1,D2,D3]
DistTensor<double> v_femn_lvl1_part3B__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//v_femn_lvl1_part3T[D0,D1,D2,D3]
DistTensor<double> v_femn_lvl1_part3T__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//v_femn_lvl1_part3_0[D0,D1,D2,D3]
DistTensor<double> v_femn_lvl1_part3_0__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
	//v_femn_lvl1_part3_1[D0,D1,D2,D3]
DistTensor<double> v_femn_lvl1_part3_1__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
DistTensor<double> v_femn_lvl1_part3_1_perm1230__D_1__D_2__D_3__D_0( dist__D_0__D_1__D_2__D_3, g );
v_femn_lvl1_part3_1_perm1230__D_1__D_2__D_3__D_0.SetLocalPermutation( perm_1_2_3_0 );
	//v_femn_lvl1_part3_2[D0,D1,D2,D3]
DistTensor<double> v_femn_lvl1_part3_2__D_0__D_1__D_2__D_3( dist__D_0__D_1__D_2__D_3, g );
// v_femn has 4 dims
//	Starting distribution: [D0,D1,D2,D3] or _D_0__D_1__D_2__D_3
ObjShape v_femn__D_0__D_1__D_2__D_3_tmpShape_U( 4 );
v_femn__D_0__D_1__D_2__D_3_tmpShape_U[ 0 ] = n_v;
v_femn__D_0__D_1__D_2__D_3_tmpShape_U[ 1 ] = n_v;
v_femn__D_0__D_1__D_2__D_3_tmpShape_U[ 2 ] = n_o;
v_femn__D_0__D_1__D_2__D_3_tmpShape_U[ 3 ] = n_o;
v_femn__D_0__D_1__D_2__D_3.ResizeTo( v_femn__D_0__D_1__D_2__D_3_tmpShape_U );
MakeUniform( v_femn__D_0__D_1__D_2__D_3 );
// t_fj has 2 dims
//	Starting distribution: [D01,D23] or _D_0_1__D_2_3
ObjShape t_fj__D_0_1__D_2_3_tmpShape_U( 2 );
t_fj__D_0_1__D_2_3_tmpShape_U[ 0 ] = n_v;
t_fj__D_0_1__D_2_3_tmpShape_U[ 1 ] = n_o;
t_fj__D_0_1__D_2_3.ResizeTo( t_fj__D_0_1__D_2_3_tmpShape_U );
MakeUniform( t_fj__D_0_1__D_2_3 );
// u_mnje has 4 dims
ObjShape u_mnje__D_0__D_1__D_2__D_3_tmpShape_U( 4 );
u_mnje__D_0__D_1__D_2__D_3_tmpShape_U[ 0 ] = n_o;
u_mnje__D_0__D_1__D_2__D_3_tmpShape_U[ 1 ] = n_o;
u_mnje__D_0__D_1__D_2__D_3_tmpShape_U[ 2 ] = n_o;
u_mnje__D_0__D_1__D_2__D_3_tmpShape_U[ 3 ] = n_v;
u_mnje__D_0__D_1__D_2__D_3.ResizeTo( u_mnje__D_0__D_1__D_2__D_3_tmpShape_U );
MakeUniform( u_mnje__D_0__D_1__D_2__D_3 );
// U_mnie has 4 dims
ObjShape U_mnie__D_0__D_1__D_2__D_3_tmpShape_U( 4 );
U_mnie__D_0__D_1__D_2__D_3_tmpShape_U[ 0 ] = n_o;
U_mnie__D_0__D_1__D_2__D_3_tmpShape_U[ 1 ] = n_o;
U_mnie__D_0__D_1__D_2__D_3_tmpShape_U[ 2 ] = n_o;
U_mnie__D_0__D_1__D_2__D_3_tmpShape_U[ 3 ] = n_v;
U_mnie__D_0__D_1__D_2__D_3.ResizeTo( U_mnie__D_0__D_1__D_2__D_3_tmpShape_U );
MakeUniform( U_mnie__D_0__D_1__D_2__D_3 );
//END_DECL

//******************************
//* Load tensors
//******************************
////////////////////////////////
//Performance testing
////////////////////////////////
std::stringstream fullName;
#ifdef CORRECTNESS
DistTensor<T> check_U(dist__D_0__D_1__D_2__D_3, g);
check_U.ResizeTo(U_mnie__D_0__D_1__D_2__D_3.Shape());
Read(u_mnje__D_0__D_1__D_2__D_3, "ccsd_terms/term_u_small", BINARY_FLAT, false);
Read(v_femn__D_0__D_1__D_2__D_3, "ccsd_terms/term_v_small", BINARY_FLAT, false);
fullName.str("");
fullName.clear();
fullName << "ccsd_terms/term_t_small_iter" << testIter;
Read(t_fj__D_0_1__D_2_3, fullName.str(), BINARY_FLAT, false);
fullName.str("");
fullName.clear();
fullName << "ccsd_terms/term_U_iter" << testIter;
Read(check_U, fullName.str(), BINARY_FLAT, false);
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
if(commRank == 0){
flops += 2*v_femn_lvl1_part3_1_perm1230__D_1__D_2__D_3__D_0.Dimension(0)*v_femn_lvl1_part3_1_perm1230__D_1__D_2__D_3__D_0.Dimension(2)*v_femn_lvl1_part3_1_perm1230__D_1__D_2__D_3__D_0.Dimension(3)*v_femn_lvl1_part3_1_perm1230__D_1__D_2__D_3__D_0.Dimension(1)*t_fj_lvl2_part1_1__D_0__S.Dimension(1);
}
			LocalContract(1.0, v_femn_lvl1_part3_1_perm1230__D_1__D_2__D_3__D_0.LockedTensor(), indices_emnf, false,
				t_fj_lvl2_part1_1__D_0__S.LockedTensor(), indices_fi, false,
				0.0, U_mnie_lvl1_part1_1_lvl2_part2_1_perm30124__D_1__D_2__D_3__S__D_0.Tensor(), indices_emnif, false);
			t_fj_lvl2_part1_1__D_0__S.EmptyData();
			   // U_mnie_lvl1_part1_1_lvl2_part2_1_U_temp[D20,D3,*,D1] <- U_mnie_lvl1_part1_1_lvl2_part2_1[D2,D3,*,D1,D0] (with SumScatter on D0)
			U_mnie_lvl1_part1_1_lvl2_part2_1_U_temp__D_2_0__D_3__S__D_1.ReduceScatterRedistFrom( U_mnie_lvl1_part1_1_lvl2_part2_1_perm30124__D_1__D_2__D_3__S__D_0, 4 );
			U_mnie_lvl1_part1_1_lvl2_part2_1_perm30124__D_1__D_2__D_3__S__D_0.EmptyData();
			   // U_mnie_lvl1_part1_1_lvl2_part2_1_U_temp[D20,D1,*,D3] <- U_mnie_lvl1_part1_1_lvl2_part2_1_U_temp[D20,D3,*,D1]
			U_mnie_lvl1_part1_1_lvl2_part2_1_U_temp__D_2_0__D_1__S__D_3.AlignModesWith( modes_0_1_2_3, U_mnie_lvl1_part1_1_lvl2_part2_1__D_0__D_1__D_2__D_3, modes_0_1_2_3 );
			U_mnie_lvl1_part1_1_lvl2_part2_1_U_temp__D_2_0__D_1__S__D_3.AllToAllRedistFrom( U_mnie_lvl1_part1_1_lvl2_part2_1_U_temp__D_2_0__D_3__S__D_1, modes_1_3 );
			U_mnie_lvl1_part1_1_lvl2_part2_1_U_temp__D_2_0__D_3__S__D_1.EmptyData();
			   // U_mnie_lvl1_part1_1_lvl2_part2_1_U_temp[D0,D1,D2,D3] <- U_mnie_lvl1_part1_1_lvl2_part2_1_U_temp[D20,D1,*,D3]
			U_mnie_lvl1_part1_1_lvl2_part2_1_U_temp__D_0__D_1__D_2__D_3.AlignModesWith( modes_0_1_2_3, U_mnie_lvl1_part1_1_lvl2_part2_1__D_0__D_1__D_2__D_3, modes_0_1_2_3 );
			U_mnie_lvl1_part1_1_lvl2_part2_1_U_temp__D_0__D_1__D_2__D_3.AllToAllRedistFrom( U_mnie_lvl1_part1_1_lvl2_part2_1_U_temp__D_2_0__D_1__S__D_3, modes_0_2 );
			U_mnie_lvl1_part1_1_lvl2_part2_1_U_temp__D_2_0__D_1__S__D_3.EmptyData();
if(commRank == 0){
flops += 2*prod(U_mnie_lvl1_part1_1_lvl2_part2_1_U_temp__D_0__D_1__D_2__D_3.Shape());
}
			YxpBy( U_mnie_lvl1_part1_1_lvl2_part2_1_U_temp__D_0__D_1__D_2__D_3, 1.0, U_mnie_lvl1_part1_1_lvl2_part2_1__D_0__D_1__D_2__D_3 );
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


//END_CODE

    /*****************************************/
    mpi::Barrier(g.OwningComm());
    runTime = mpi::Time() - startTime;
    gflops = flops / (1e9 * runTime);
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


