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
#ifndef RELEASE
    CallStackEntry entry("DistTensorTest");
#endif
    Unsigned i;
    const Int commRank = mpi::CommRank(mpi::COMM_WORLD);

//START_DECL
ObjShape tempShape;
TensorDistribution dist__D_0__D_1__D_2__D_3 = rote::StringToTensorDist("[(0),(1),(2),(3)]");
TensorDistribution dist__D_0__D_2 = rote::StringToTensorDist("[(0),(2)]");
TensorDistribution dist__D_1__D_3 = rote::StringToTensorDist("[(1),(3)]");
TensorDistribution dist__D_0_1__D_2_3 = rote::StringToTensorDist("[(0,1),(2,3)]");
TensorDistribution dist__D_0_1__D_3_2 = rote::StringToTensorDist("[(0,1),(3,2)]");
TensorDistribution dist__D_1_0__D_3_2 = rote::StringToTensorDist("[(1,0),(3,2)]");
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


//END_CODE

    /*****************************************/
    mpi::Barrier(g.OwningComm());
    runTime = mpi::Time() - startTime;
    gflops = flops / (1e9 * runTime);
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


