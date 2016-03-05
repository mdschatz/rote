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
// NOTE: It is possible to simply include "rote.hpp" instead
#include "rote.hpp"

#ifdef PROFILE
#ifdef BGQ
#include <spi/include/kernel/memory.h>
#endif
#endif

using namespace rote;
using namespace std;

void Usage() {
    std::cout << "./DistTensor <gridDim0> <gridDim1> ... \n";
    std::cout << "<gridDimK>   : dimension of mode-K of grid\n";
}

typedef struct Arguments {
    ObjShape gridShape;
    int nProcs;
    Unsigned n_o;
    Unsigned n_v;
    Unsigned blkSize;
    Unsigned testIter;
} Params;

void ProcessInput(int argc, char** const argv, Params& args) {
    int argCount = 0;
    if (argCount + 1 >= argc) {
        std::cerr << "Missing required gridOrder argument\n";
        Usage();
        throw ArgException();
    }

    Unsigned gridOrder = atoi(argv[++argCount]);

    if (argCount + gridOrder >= argc) {
        std::cerr << "Missing required grid dimensions\n";
        Usage();
        throw ArgException();
    }

    args.gridShape.resize(gridOrder);
    args.nProcs = 1;
    for (int i = 0; i < gridOrder; i++) {
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
    const Int commRank = mpi::CommRank(mpi::COMM_WORLD);

//START_DECL
//Indices needed
IndexArray indices_em = {'e', 'm'};
IndexArray indices_fn = {'f', 'm'};
IndexArray indices_bn = {'b', 'n'};
IndexArray indices_fj = {'f', 'j'};
IndexArray indices_ej = {'e', 'j'};
IndexArray indices_ei = {'e', 'i'};
IndexArray indices_me = {'m', 'e'};
IndexArray indices_am = {'a', 'm'};
IndexArray indices_ae = {'a', 'e'};
IndexArray indices_en = {'e', 'n'};
IndexArray indices_mi = {'m', 'i'};
IndexArray indices_ai = {'a', 'i'};
IndexArray indices_emfn = {'e', 'm', 'f', 'n'};
IndexArray indices_fenm = {'f', 'e', 'n', 'm'};
IndexArray indices_bfnj = {'b', 'f', 'n', 'j'};
IndexArray indices_bmje = {'b', 'm', 'j', 'e'};
IndexArray indices_nmje = {'n', 'm', 'j', 'e'};
IndexArray indices_bmfe = {'b', 'm', 'f', 'e'};
IndexArray indices_femn = {'f', 'e', 'm', 'n'};
IndexArray indices_bmej = {'b', 'm', 'e', 'j'};
IndexArray indices_mnje = {'m', 'n', 'j', 'e'};
IndexArray indices_bmef = {'b', 'm', 'e', 'f'};
IndexArray indices_mnie = {'m', 'n', 'i', 'e'};
IndexArray indices_efmn = {'e', 'f', 'm', 'n'};
IndexArray indices_efij = {'e', 'f', 'i', 'j'};
IndexArray indices_mnij = {'m', 'n', 'i', 'j'};
IndexArray indices_jimb = {'j', 'i', 'm', 'b'};
IndexArray indices_bmie = {'b', 'm', 'i', 'e'};
IndexArray indices_amef = {'a', 'm', 'e', 'f'};
IndexArray indices_afmn = {'a', 'f', 'm', 'n'};
IndexArray indices_efin = {'e', 'f', 'i', 'n'};
IndexArray indices_amie = {'a', 'm', 'i', 'e'};
IndexArray indices_aemn = {'a', 'e', 'm', 'n'};
IndexArray indices_aeim = {'a', 'e', 'i', 'm'};
IndexArray indices_efim = {'e', 'f', 'i', 'm'};
IndexArray indices_aemi = {'a', 'e', 'm', 'i'};
IndexArray indices_abij = {'a', 'b', 'i', 'j'};
IndexArray indices_abmj = {'a', 'b', 'm', 'j'};
IndexArray indices_ebij = {'e', 'b', 'i', 'j'};
IndexArray indices_abmn = {'a', 'b', 'm', 'n'};
IndexArray indices_ijmb = {'i', 'j', 'm', 'b'};
IndexArray indices_ejab = {'e', 'j', 'a', 'b'};
IndexArray indices_abef = {'a', 'b', 'e', 'f'};

Permutation perm_1_0_2_3 = {1, 0, 2, 3};
Permutation perm_0_1_3_2 = {0, 1, 3, 2};
Permutation perm_1_0_3_2 = {1, 0, 3, 2};

	//T_bfnj[D0,D1,D2,D3]
DistTensor<double> T_bfnj( "[(0),(1),(2),(3)]", g );
	//Tau_efmn[D0,D1,D2,D3]
DistTensor<double> Tau_efmn( "[(0),(1),(2),(3)]", g );
	//t_fj[D01,D23]
DistTensor<double> t_fj( "[(0,1),(2,3)]", g );
	//t_fj_temp[D01,D23]
DistTensor<double> t_fj_temp( "[(0,1),(2,3)]", g );
//	Starting distribution: [D01,D23] or _D_0_1__D_2_3
ObjShape t_fj_tempShape = {n_v, n_o};
t_fj.ResizeTo( t_fj_tempShape );
MakeUniform( t_fj );
t_fj_temp.ResizeTo( t_fj_tempShape );
MakeUniform( t_fj_temp );
// T_bfnj has 4 dims
//	Starting distribution: [D0,D1,D2,D3] or _D_0__D_1__D_2__D_3
ObjShape T_bfnj_tempShape = {n_v, n_v, n_o, n_o};
T_bfnj.ResizeTo( T_bfnj_tempShape );
MakeUniform( T_bfnj );
// Tau_efmn has 4 dims
ObjShape Tau_efmn_tempShape = {n_v, n_v, n_o, n_o};
Tau_efmn.ResizeTo( Tau_efmn_tempShape );
MakeUniform( Tau_efmn );

	//W_bmje[D0,D1,D2,D3]
DistTensor<double> W_bmje( "[(0),(1),(2),(3)]", g );
	//Wtemp1[D0,D1,D2,D3]
DistTensor<double> Wtemp1( "[(0),(1),(2),(3)]", g );
	//Wtemp2[D0,D1,D2,D3]
DistTensor<double> Wtemp2( "[(0),(1),(2),(3)]", g );
	//Wtemp3[D0,D1,D2,D3]
DistTensor<double> Wtemp3( "[(0),(1),(2),(3)]", g );
	//Wtemp4[D0,D1,D2,D3]
DistTensor<double> Wtemp4( "[(0),(1),(2),(3)]", g );
	//r_bmfe[D0,D1,D2,D3]
DistTensor<double> r_bmfe( "[(0),(1),(2),(3)]", g );
	//u_mnje[D0,D1,D2,D3]
DistTensor<double> u_mnje( "[(0),(1),(2),(3)]", g );
	//v_femn[D0,D1,D2,D3]
DistTensor<double> v_femn( "[(0),(1),(2),(3)]", g );
	//w_bmje[D0,D1,D2,D3]
DistTensor<double> w_bmje( "[(0),(1),(2),(3)]", g );
	//x_bmej[D0,D1,D2,D3]
DistTensor<double> x_bmej( "[(0),(1),(2),(3)]", g );
// r_bmfe has 4 dims
ObjShape r_bmfe_tempShape = {n_v, n_o, n_v, n_v};
r_bmfe.ResizeTo( r_bmfe_tempShape );
MakeUniform( r_bmfe );
// u_mnje has 4 dims
ObjShape u_mnje_tempShape = {n_o, n_o, n_v, n_v};
u_mnje.ResizeTo( u_mnje_tempShape );
MakeUniform( u_mnje );
// v_femn has 4 dims
ObjShape v_femn_tempShape = {n_v, n_v, n_o, n_o};
v_femn.ResizeTo( v_femn_tempShape );
MakeUniform( v_femn );
// w_bmje has 4 dims
ObjShape w_bmje_tempShape = {n_v, n_o, n_o, n_v};
w_bmje.ResizeTo( w_bmje_tempShape );
MakeUniform( w_bmje );
// x_bmej has 4 dims
ObjShape x_bmej_tempShape = {n_v, n_o, n_v, n_o};
x_bmej.ResizeTo( x_bmej_tempShape );
MakeUniform( x_bmej );
// W_bmje has 4 dims
ObjShape W_bmje_tempShape = {n_v, n_o, n_o, n_v};
W_bmje.ResizeTo( W_bmje_tempShape );
MakeUniform( W_bmje );
	//X_bmej[D0,D1,D2,D3]
DistTensor<double> X_bmej( "[(0),(1),(2),(3)]", g );
	//Xtemp1[D0,D1,D2,D3]
DistTensor<double> Xtemp1( "[(0),(1),(2),(3)]", g );
// X_bmej has 4 dims
ObjShape X_bmej_tempShape = {n_v, n_o, n_v, n_o};
X_bmej.ResizeTo( X_bmej_tempShape );
MakeUniform( X_bmej );

	//U_mnie[D0,D1,D2,D3]
DistTensor<double> U_mnie( "[(0),(1),(2),(3)]", g );
// U_mnie has 4 dims
ObjShape U_mnie_tempShape = {n_o, n_o, n_o, n_v};
U_mnie.ResizeTo( U_mnie_tempShape );
MakeUniform( U_mnie );

	//Q_mnij[D0,D1,D2,D3]
DistTensor<double> Q_mnij( "[(0),(1),(2),(3)]", g );
	//Qtemp1[D0,D1,D2,D3]
DistTensor<double> Qtemp1( "[(0),(1),(2),(3)]", g );
	//q_mnij[D0,D1,D2,D3]
DistTensor<double> q_mnij( "[(0),(1),(2),(3)]", g );
// q_mnij has 4 dims
ObjShape q_mnij_tempShape = {n_o, n_o, n_o, n_o};
q_mnij.ResizeTo( q_mnij_tempShape );
MakeUniform( q_mnij );
// Q_mnij has 4 dims
ObjShape Q_mnij_tempShape = {n_o, n_o, n_o, n_o};
Q_mnij.ResizeTo( Q_mnij_tempShape );
MakeUniform( Q_mnij );

	//P_jimb[D0,D1,D2,D3]
DistTensor<double> P_jimb( "[(0),(1),(2),(3)]", g );
// P_jimb has 4 dims
ObjShape P_jimb_tempShape = {n_o, n_o, n_o, n_v};
P_jimb.ResizeTo( P_jimb_tempShape );
MakeUniform( P_jimb );
	//H_me[D01,D23]
DistTensor<double> H_me( "[(0,1),(2,3)]", g );
	//Htemp1[D0,D1,D2,D3]
DistTensor<double> Htemp1( "[(0),(1),(2),(3)]", g );
// H_me has 2 dims
ObjShape H_me_tempShape = {n_o, n_v};
H_me.ResizeTo( H_me_tempShape );
MakeUniform( H_me );

	//F_ae[D01,D23]
DistTensor<double> F_ae( "[(0,1),(2,3)]", g );
	//Ftemp1[D0,D1,D2,D3]
DistTensor<double> Ftemp1( "[(0),(1),(2),(3)]", g );
	//Ftemp2[D0,D1,D2,D3]
DistTensor<double> Ftemp2( "[(0),(1),(2),(3)]", g );
// F_ae has 2 dims
ObjShape F_ae_tempShape = {n_v, n_v};
F_ae.ResizeTo( F_ae_tempShape );
MakeUniform( F_ae );

	//G_mi[D01,D23]
DistTensor<double> G_mi( "[(0,1),(2,3)]", g );
	//Gtemp1[D0,D1,D2,D3]
DistTensor<double> Gtemp1( "[(0),(1),(2),(3)]", g );
	//Gtemp2[D0,D1,D2,D3]
DistTensor<double> Gtemp2( "[(0),(1),(2),(3)]", g );
// G_mi has 2 dims
ObjShape G_mi_tempShape = {n_o, n_o};
G_mi.ResizeTo( G_mi_tempShape );
MakeUniform( G_mi );

	//z_ai[D01,D23]
DistTensor<double> z_ai( "[(0,1),(2,3)]", g );
	//ztemp2[D0,D1,D2,D3]
DistTensor<double> ztemp2( "[(0),(1),(2),(3)]", g );
	//ztemp3[D0,D1,D2,D3]
DistTensor<double> ztemp3( "[(0),(1),(2),(3)]", g );
	//ztemp4[D0,D1,D2,D3]
DistTensor<double> ztemp4( "[(0),(1),(2),(3)]", g );
	//ztemp5[D0,D1,D2,D3]
DistTensor<double> ztemp5( "[(0),(1),(2),(3)]", g );
// z_ai has 2 dims
ObjShape z_ai_tempShape = {n_v, n_o};
z_ai.ResizeTo( z_ai_tempShape );
MakeUniform( z_ai );

	//Z_abij[D0,D1,D2,D3]
DistTensor<double> Z_abij( "[(0),(1),(2),(3)]", g );
	//Zaccum[D0,D1,D2,D3]
DistTensor<double> Zaccum( "[(0),(1),(2),(3)]", g );
	//Ztemp1[D0,D1,D2,D3]
DistTensor<double> Ztemp1( "[(0),(1),(2),(3)]", g );
	//Ztemp2[D0,D1,D2,D3]
DistTensor<double> Ztemp2( "[(0),(1),(2),(3)]", g );
	//y_abef[D0,D1,D2,D3]
DistTensor<double> y_abef( "[(0),(1),(2),(3)]", g );
// y_abef has 4 dims
ObjShape y_abef_tempShape = {n_v, n_v, n_v, n_v};
y_abef.ResizeTo( y_abef_tempShape );
MakeUniform( y_abef );
// Z_abij has 4 dims
ObjShape Z_abij_tempShape = {n_v, n_v, n_o, n_o};
Z_abij.ResizeTo( Z_abij_tempShape );
MakeUniform( Z_abij );
//END_DECL

//******************************
//* Load tensors
//******************************
////////////////////////////////
//Performance testing
////////////////////////////////
std::stringstream fullName;
#ifdef CORRECTNESS
DistTensor<T> check_Tau("[(0),(1),(2),(3)]", g);
check_Tau.ResizeTo(Tau_efmn.Shape());
fullName.str("");
fullName.clear();
fullName << "ccsd_terms/term_t_small_iter" << testIter;
Read(t_fj, fullName.str(), BINARY_FLAT, false);
fullName.str("");
fullName.clear();
fullName << "ccsd_terms/term_T_iter" << testIter;
Read(T_bfnj, fullName.str(), BINARY_FLAT, false);
fullName.str("");
fullName.clear();
fullName << "ccsd_terms/term_Tau_iter" << testIter;
Read(check_Tau, fullName.str(), BINARY_FLAT, false);
DistTensor<T> check_W("[(0),(1),(2),(3)", g);
check_W.ResizeTo(W_bmje.Shape());
Read(r_bmfe, "ccsd_terms/term_r_small", BINARY_FLAT, false);
Read(u_mnje, "ccsd_terms/term_u_small", BINARY_FLAT, false);
Read(v_femn, "ccsd_terms/term_v_small", BINARY_FLAT, false);
Read(w_bmje, "ccsd_terms/term_w_small", BINARY_FLAT, false);
Read(x_bmej, "ccsd_terms/term_x_small", BINARY_FLAT, false);
fullName.str("");
fullName.clear();
fullName << "ccsd_terms/term_W_iter" << testIter;
Read(check_W, fullName.str(), BINARY_FLAT, false);
DistTensor<T> check_X("[(0),(1),(2),(3)]", g);
check_X.ResizeTo(X_bmej.Shape());
fullName.str("");
fullName.clear();
fullName << "ccsd_terms/term_X_iter" << testIter;
Read(check_X, fullName.str(), BINARY_FLAT, false);
DistTensor<T> check_U("[(0),(1),(2),(3)]", g);
check_U.ResizeTo(U_mnie.Shape());
fullName.str("");
fullName.clear();
fullName << "ccsd_terms/term_U_iter" << testIter;
Read(check_U, fullName.str(), BINARY_FLAT, false);
DistTensor<T> check_Q("[(0),(1),(2),(3)]", g);
check_Q.ResizeTo(Q_mnij.Shape());
Read(q_mnij, "ccsd_terms/term_q_small", BINARY_FLAT, false);
fullName.str("");
fullName.clear();
fullName << "ccsd_terms/term_Q_iter" << testIter;
Read(check_Q, fullName.str(), BINARY_FLAT, false);
DistTensor<T> check_P("[(0),(1),(2),(3)]", g);
check_P.ResizeTo(P_jimb.Shape());
fullName.str("");
fullName.clear();
fullName << "ccsd_terms/term_P_iter" << testIter;
Read(check_P, fullName.str(), BINARY_FLAT, false);
DistTensor<T> check_H("[(0,1),(2,3)]", g);
check_H.ResizeTo(H_me.Shape());
fullName.str("");
fullName.clear();
fullName << "ccsd_terms/term_H_iter" << testIter;
Read(check_H, fullName.str(), BINARY_FLAT, false);
DistTensor<T> check_F("[(0,1),(2,3)]", g);
check_F.ResizeTo(F_ae.Shape());
fullName.str("");
fullName.clear();
fullName << "ccsd_terms/term_F_iter" << testIter;
Read(check_F, fullName.str(), BINARY_FLAT, false);
DistTensor<T> check_G("[(0,1),(2,3)]", g);
check_G.ResizeTo(G_mi.Shape());
fullName.str("");
fullName.clear();
fullName << "ccsd_terms/term_G_iter" << testIter;
Read(check_G, fullName.str(), BINARY_FLAT, false);
DistTensor<T> check_z_small("[(0,1),(2,3)]", g);
check_z_small.ResizeTo(z_ai.Shape());
fullName.str("");
fullName.clear();
fullName << "ccsd_terms/term_z_small_iter" << testIter;
Read(check_z_small, fullName.str(), BINARY_FLAT, false);
DistTensor<T> check_Z("[(0),(1),(2),(3)]", g);
check_Z.ResizeTo(Z_abij.Shape());
Read(y_abef, "ccsd_terms/term_y_small", BINARY_FLAT, false);
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


    //Tau_emfn = T_emfn + t_em*t_fn
	Tau_efmn = T_bfnj;
	GenContract(1.0, t_fj, indices_em, t_fj, indices_fn, 1.0, Tau_efmn, indices_emfn);

	//W_nmje = 2u_nmje - u_mnje
	Wtemp3.ResizeTo(u_mnje);
	GenYAxpPx(-1.0, u_mnje, 2.0, perm_1_0_2_3, Wtemp3);

	//W_fenm = 2v_fenm - v_femn
	Wtemp2.ResizeTo(v_femn);
	GenYAxpPx(-1.0, v_femn, 2.0, perm_0_1_3_2, Wtemp2);

	Wtemp1.ResizeTo(T_bfnj);
	GenZAxpBypPx(0.5, T_bfnj, -1.0, Tau_efmn, perm_0_1_3_2, Wtemp1 );

	//W_bmje = 2w_bmje - x_bmej
	GenYAxpPx( 2.0, w_bmje, -1.0, perm_0_1_3_2, W_bmje );

	//W_bmje += \sum_fn (2v_fenm - v_femn)*(0.5T_bfnj - Tau_bfnj + T_bfjn)
	GenContract(1.0, Wtemp2, indices_fenm, Wtemp1, indices_bfnj, 1.0, W_bmje, indices_bmje);

	//W_bmje -= \sum_n (2u_nmje - u_mnje) * t_bn
	GenContract(-1.0, Wtemp3, indices_nmje, t_fj, indices_bn, 1.0, W_bmje, indices_bmje);

	//W_bmef = 2r_bmfe - r_bmef
	GenYAxpPx( -1.0, r_bmfe, 2.0, perm_0_1_3_2, Wtemp4);

	//W_bmje += \sum_f (2r_bmfe - r_bmef)*t_fj
	GenContract(1.0, Wtemp4, indices_bmfe, t_fj, indices_fj, 1.0, W_bmje, indices_bmje);

	//X_bfnj = tau_bfnj - 0.5 * T_bfnj
	ZAxpBy(1.0, Tau_efmn, -0.5, T_bfnj, Xtemp1);

	//X_bmej = x_bmej;
	X_bmej = x_bmej;

	//X_bmej = \sum_fn v_femn (tau_bfnj - 0.5 * T_bfnj)
	GenContract(-0.5, v_femn, indices_femn, Xtemp1, indices_bfnj, 0.0, X_bmej, indices_bmej);

	//X_bmej -= \sum_n u_mnje * t_bn
	GenContract(-1.0, u_mnje, indices_mnje, t_fj, indices_bn, 1.0, X_bmej, indices_bmej);

	//X_bmej += \sum_f r_bmef * t_fj
	GenContract(1.0, r_bmfe, indices_bmef, t_fj, indices_fj, 1.0, X_bmej, indices_bmej);

	//U_mnie = u_mnie;
	U_mnie = u_mnje;

	//U_mnie += \sum_f v_femn * t_fj
	GenContract(1.0, v_femn, indices_femn, t_fj, indices_fj, 1.0, U_mnie, indices_mnie);

	//Q_mnij = \sum_ef v_efmn * tau_efij
	GenContract(1.0, v_femn, indices_efmn, Tau_efmn, indices_efij, 0.0, Qtemp1, indices_mnij);

	//Q_mnij = (1 + P_mnij) (\sum_ef v_efmn * tau_efij)
	GenYAxpPx(1.0, Qtemp1, 1.0, perm_1_0_3_2, Q_mnij);

	//Q_mnij += \sum_ef v_efmn * tau_efij
	GenContract(1.0, v_femn, indices_efmn, Tau_efmn, indices_efij, 1.0, Q_mnij, indices_mnij);

	//Q_mnij += q_mnij
	YAxpBy(1.0, q_mnij, 1.0, Q_mnij);

	//P_jimb = u_jimb
	P_jimb = u_mnje;

	//P_jimb += \sum_ef r_bmef * tau_efij
	GenContract(1.0, r_bmfe, indices_bmef, Tau_efmn, indices_efij, 1.0, P_jimb, indices_jimb);

	//P_jimb += \sum_e w_bmie * t_ej
	GenContract(1.0, w_bmje, indices_bmie, t_fj, indices_ej, 1.0, P_jimb, indices_jimb);

	//P_jimb += \sum_e x_bmej * t_ei
	GenContract(1.0, x_bmej, indices_bmej, t_fj, indices_ei, 1.0, P_jimb, indices_jimb);

	//tmp = (2v_efmn - v_efnm)
	GenYAxpPx(2.0, v_femn, -1.0, perm_0_1_3_2, Htemp1);

	//H_me = \sum_fn (2v_efmn - v_efnm) * t_fn
	GenContract(1.0, Htemp1, indices_efmn, t_fj, indices_fn, 1.0, H_me, indices_me);

	//F_ae = -\sum_m H_me * t_am
	GenContract(-1.0, H_me, indices_me, t_fj, indices_am, 0.0, F_ae, indices_ae);

	//tmp = (2r_amef - r_amfe)
	GenYAxpPx(2.0, r_bmfe, -1.0, perm_0_1_3_2, Ftemp1);

	//F_ae += \sum_m (2r_amef - r_amfe) * t_am
	GenContract(1.0, Ftemp1, indices_amef, t_fj, indices_am, 1.0, F_ae, indices_ae);

	//tmp = (2v_efmn - v_efnm)
	GenYAxpPx(2.0, v_femn, -1.0, perm_0_1_3_2, Ftemp2);

	//F_ae += \sum_fmn (2v_efmn - v_efnm) * T_afmn
	GenContract(1.0, Ftemp2, indices_efmn, T_bfnj, indices_afmn, 1.0, F_ae, indices_ae);

	//G_mi = \sum_e H_me * t_ei
	GenContract(1.0, H_me, indices_me, t_fj, indices_ei, 0.0, G_mi, indices_mi);

	//tmp = (2u_mnie - u_nmie)
	GenYAxpPx(2.0, u_mnje, -1.0, perm_1_0_2_3, Gtemp1);

	//G_mi += \sum_en (2u_mnie - u_nmie) * t_en
	GenContract(1.0, Gtemp1, indices_mnie, t_fj, indices_en, 1.0, G_mi, indices_mi);

	//tmp = (2v_efmn - v_efnm)
	GenYAxpPx(2.0, v_femn, -1.0, perm_0_1_3_2, Gtemp2);

	//G_mi += \sum_efn (2v_efmn - v_efnm) * T_efin
	GenContract(1.0, Gtemp2, indices_efmn, T_bfnj, indices_efin, 1.0, G_mi, indices_mi);

	//z_ai = -\sum_m G_mi * t_am
	GenContract(-1.0, G_mi, indices_mi, t_fj, indices_am, 0.0, z_ai, indices_ai);

	//tmp = (2U_mnie - U_nmie)
	GenYAxpPx(2.0, U_mnie, -1.0, perm_1_0_2_3, ztemp5);

	//z_ai -= \sum_emn (2U_mnie - U_nmie) * T_aemn
	GenContract(-1.0, ztemp5, indices_mnie, T_bfnj, indices_aemn, -1.0, z_ai, indices_ai);

	//tmp = (2w_amie - x_amei)
	ZAxpBy(2.0, w_bmje, -1.0, x_bmej, ztemp4);

	//z_ai += \sum_em (2w_amie - x_amei) * t_em
	GenContract(1.0, ztemp4, indices_amie, t_fj, indices_em, 1.0, z_ai, indices_ai);

	//tmp = (2T_aeim - T_aemi)
	GenYAxpPx(2.0, T_bfnj, -1.0, perm_0_1_3_2, ztemp3);

	//z_ai += \sum_em (2T_aeim - T_aemi) * H_me
	GenContract(-1.0, ztemp3, indices_aeim, H_me, indices_me, 1.0, z_ai, indices_ai);

	//tmp = (2r_amef - r_amfe)
	GenYAxpPx(2.0, r_bmfe, -1.0, perm_0_1_3_2, ztemp2);

	//z_ai += \sum_efm (2r_amef - r_amfe) * tau_efim
	GenContract(-1.0, ztemp2, indices_amef, Tau_efmn, indices_efim, 1.0, z_ai, indices_ai);

	//tmp = \sum_em X_bmej * T_aemi
	GenContract(1.0, X_bmej, indices_bmej, T_bfnj, indices_aemi, 0.0, Ztemp1, indices_abij);

	//tmpAccum = -(0.5 + P_ij) (\sum_em X_bmej * T_aemi)
	GenYAxpPx(-0.5, Ztemp1, -1.0, perm_0_1_3_2, Zaccum);

	//tmp2 = (2T_aeim - T_aemi)
	GenYAxpPx(2.0, T_bfnj, -1.0, perm_0_1_3_2, Ztemp2);

	//tmpAccum += 0.5 * \sum_em W_bmje * (2T_aeim - T_aemi)
	GenContract(0.5, W_bmje, indices_bmje, Ztemp2, indices_aeim, 1.0, Zaccum, indices_abij);

	//tmpAccum -= \sum_m G_mi * T_abmj
	GenContract(1.0, G_mi, indices_mi, T_bfnj, indices_abmj, -1.0, Zaccum, indices_abij);

	//tmpAccum += \sum_e F_ae * T_ebij
	GenContract(1.0, F_ae, indices_ae, T_bfnj, indices_ebij, 1.0, Zaccum, indices_abij);

	//tmpAccum -= \sum_m P_ijmb * t_am
	GenContract(1.0, P_jimb, indices_ijmb, t_fj, indices_am, -1.0, Zaccum, indices_abij);

	//tmpAccum += \sum_e r_ejab * t_ei
	GenContract(1.0, r_bmfe, indices_ejab, t_fj, indices_ei, 1.0, Zaccum, indices_abij);

	//Z_abij = (1 + P_aibj) Zaccum
	GenYAxpPx(1.0, Zaccum, 1.0, perm_1_0_3_2, Z_abij);

	//Z_abij = \sum_ef y_abef * tau_efij
	GenContract(1.0, y_abef, indices_abef, Tau_efmn, indices_efij, 1.0, Z_abij, indices_abij);

	//Z_abij = \sum_mn Q_mnij * tau_abmn
	GenContract(1.0, Q_mnij, indices_mnij, Tau_efmn, indices_abmn, 1.0, Z_abij, indices_abij);

	//Z_abij += v_abij
	YAxpBy(1.0, v_femn, 1.0, Z_abij);



//END_CODE

    /*****************************************/
    mpi::Barrier(g.OwningComm());
    runTime = mpi::Time() - startTime;
    gflops = flops / (1e9 * runTime);
#ifdef CORRECTNESS
    DistTensor<double> diff_Z("[(0),(1),(2),(3)]", g);
    diff_Z.ResizeTo(check_Z);
    Diff(check_Z, Z_abij, diff_Z);
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


