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
IndexArray indices_em(2);
indices_em[0] = 'e';
indices_em[1] = 'm';
IndexArray indices_fn(2);
indices_fn[0] = 'f';
indices_fn[1] = 'n';
IndexArray indices_emfn(4);
indices_emfn[0] = 'e';
indices_emfn[1] = 'm';
indices_emfn[2] = 'f';
indices_emfn[3] = 'n';
IndexArray indices_fenm(4);
indices_fenm[0] = 'f';
indices_fenm[1] = 'e';
indices_fenm[2] = 'n';
indices_fenm[3] = 'm';
IndexArray indices_bfnj(4);
indices_bfnj[0] = 'b';
indices_bfnj[1] = 'f';
indices_bfnj[2] = 'n';
indices_bfnj[3] = 'j';
IndexArray indices_bmje(4);
indices_bmje[0] = 'b';
indices_bmje[1] = 'm';
indices_bmje[2] = 'j';
indices_bmje[3] = 'e';
IndexArray indices_nmje(4);
indices_nmje[0] = 'n';
indices_nmje[1] = 'm';
indices_nmje[2] = 'j';
indices_nmje[3] = 'e';
IndexArray indices_bn(2);
indices_bn[0] = 'b';
indices_bn[1] = 'n';
IndexArray indices_bmfe(4);
indices_bmfe[0] = 'b';
indices_bmfe[1] = 'm';
indices_bmfe[2] = 'f';
indices_bmfe[3] = 'e';
IndexArray indices_fj(2);
indices_fj[0] = 'f';
indices_fj[1] = 'j';
IndexArray indices_femn(4);
indices_femn[0] = 'f';
indices_femn[1] = 'e';
indices_femn[2] = 'm';
indices_femn[3] = 'n';
IndexArray indices_bmej(4);
indices_bmej[0] = 'b';
indices_bmej[1] = 'm';
indices_bmej[2] = 'e';
indices_bmej[3] = 'j';
IndexArray indices_mnje(4);
indices_mnje[0] = 'm';
indices_mnje[1] = 'n';
indices_mnje[2] = 'j';
indices_mnje[3] = 'e';
IndexArray indices_bmef(4);
indices_bmef[0] = 'b';
indices_bmef[1] = 'm';
indices_bmef[2] = 'e';
indices_bmef[3] = 'f';
IndexArray indices_mnie(4);
indices_mnie[0] = 'm';
indices_mnie[1] = 'n';
indices_mnie[2] = 'i';
indices_mnie[3] = 'e';
IndexArray indices_efmn(4);
indices_efmn[0] = 'e';
indices_efmn[1] = 'f';
indices_efmn[2] = 'm';
indices_efmn[3] = 'n';
IndexArray indices_efij(4);
indices_efij[0] = 'e';
indices_efij[1] = 'f';
indices_efij[2] = 'i';
indices_efij[3] = 'j';
IndexArray indices_mnij(4);
indices_mnij[0] = 'm';
indices_mnij[1] = 'n';
indices_mnij[2] = 'i';
indices_mnij[3] = 'j';
IndexArray indices_jimb(4);
indices_jimb[0] = 'j';
indices_jimb[1] = 'i';
indices_jimb[2] = 'm';
indices_jimb[3] = 'b';
IndexArray indices_ej(2);
indices_ej[0] = 'e';
indices_ej[1] = 'j';
IndexArray indices_ei(2);
indices_ei[0] = 'e';
indices_ei[1] = 'i';
IndexArray indices_me(2);
indices_me[0] = 'm';
indices_me[1] = 'e';
IndexArray indices_am(2);
indices_am[0] = 'a';
indices_am[1] = 'm';
IndexArray indices_ae(2);
indices_ae[0] = 'a';
indices_ae[1] = 'm';
IndexArray indices_bmie(4);
indices_bmie[0] = 'b';
indices_bmie[1] = 'm';
indices_bmie[2] = 'i';
indices_bmie[3] = 'e';
IndexArray indices_amef(4);
indices_amef[0] = 'a';
indices_amef[1] = 'm';
indices_amef[2] = 'e';
indices_amef[3] = 'f';
IndexArray indices_afmn(4);
indices_afmn[0] = 'a';
indices_afmn[1] = 'f';
indices_afmn[2] = 'm';
indices_afmn[3] = 'n';
IndexArray indices_en(2);
indices_en[0] = 'e';
indices_en[1] = 'n';
IndexArray indices_mi(2);
indices_mi[0] = 'm';
indices_mi[1] = 'i';
IndexArray indices_efin(4);
indices_efin[0] = 'e';
indices_efin[1] = 'f';
indices_efin[2] = 'i';
indices_efin[3] = 'n';
IndexArray indices_amie(4);
indices_amie[0] = 'a';
indices_amie[1] = 'm';
indices_amie[2] = 'i';
indices_amie[3] = 'e';
IndexArray indices_ai(2);
indices_ai[0] = 'a';
indices_ai[1] = 'i';
IndexArray indices_aemn(4);
indices_aemn[0] = 'a';
indices_aemn[1] = 'e';
indices_aemn[2] = 'm';
indices_aemn[3] = 'n';
IndexArray indices_aeim(4);
indices_aeim[0] = 'a';
indices_aeim[1] = 'e';
indices_aeim[2] = 'i';
indices_aeim[3] = 'm';
IndexArray indices_efim(4);
indices_efim[0] = 'e';
indices_efim[1] = 'f';
indices_efim[2] = 'i';
indices_efim[3] = 'm';
IndexArray indices_aemi(4);
indices_aemi[0] = 'a';
indices_aemi[1] = 'e';
indices_aemi[2] = 'm';
indices_aemi[3] = 'i';
IndexArray indices_abij(4);
indices_abij[0] = 'a';
indices_abij[1] = 'b';
indices_abij[2] = 'i';
indices_abij[3] = 'j';
IndexArray indices_abmj(4);
indices_abmj[0] = 'a';
indices_abmj[1] = 'b';
indices_abmj[2] = 'm';
indices_abmj[3] = 'j';
IndexArray indices_ebij(4);
indices_ebij[0] = 'e';
indices_ebij[1] = 'b';
indices_ebij[2] = 'i';
indices_ebij[3] = 'j';
IndexArray indices_abmn(4);
indices_abmn[0] = 'a';
indices_abmn[1] = 'b';
indices_abmn[2] = 'm';
indices_abmn[3] = 'n';
IndexArray indices_ijmb(4);
indices_ijmb[0] = 'i';
indices_ijmb[1] = 'j';
indices_ijmb[2] = 'm';
indices_ijmb[3] = 'b';
IndexArray indices_ejab(4);
indices_ejab[0] = 'e';
indices_ejab[1] = 'j';
indices_ejab[2] = 'a';
indices_ejab[3] = 'b';
IndexArray indices_abef(4);
indices_abef[0] = 'a';
indices_abef[1] = 'b';
indices_abef[2] = 'e';
indices_abef[3] = 'f';

Permutation perm_1_0_2_3(4);
perm_1_0_2_3[0] = 1;
perm_1_0_2_3[1] = 0;
perm_1_0_2_3[2] = 2;
perm_1_0_2_3[3] = 3;
Permutation perm_0_1_3_2(4);
perm_0_1_3_2[0] = 0;
perm_0_1_3_2[1] = 1;
perm_0_1_3_2[2] = 3;
perm_0_1_3_2[3] = 2;
Permutation perm_1_0_3_2(4);
perm_1_0_3_2[0] = 1;
perm_1_0_3_2[1] = 0;
perm_1_0_3_2[2] = 3;
perm_1_0_3_2[3] = 2;

	//T_bfnj[D0,D1,D2,D3]
DistTensor<double> T_bfnj__D_0__D_1__D_2__D_3( "[(0),(1),(2),(3)]", g );
	//Tau_efmn[D0,D1,D2,D3]
DistTensor<double> Tau_efmn__D_0__D_1__D_2__D_3( "[(0),(1),(2),(3)]", g );
	//t_fj[D01,D23]
DistTensor<double> t_fj__D_0_1__D_2_3( "[(0,1),(2,3)]", g );
	//t_fj_temp[D01,D23]
DistTensor<double> t_fj_temp__D_0_1__D_2_3( "[(0,1),(2,3)]", g );
//	Starting distribution: [D01,D23] or _D_0_1__D_2_3
ObjShape t_fj__D_0_1__D_2_3_tempShape( 2 );
t_fj__D_0_1__D_2_3_tempShape[ 0 ] = n_v;
t_fj__D_0_1__D_2_3_tempShape[ 1 ] = n_o;
t_fj__D_0_1__D_2_3.ResizeTo( t_fj__D_0_1__D_2_3_tempShape );
MakeUniform( t_fj__D_0_1__D_2_3 );
t_fj_temp__D_0_1__D_2_3.ResizeTo( t_fj__D_0_1__D_2_3_tempShape );
MakeUniform( t_fj_temp__D_0_1__D_2_3 );
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

	//W_bmje[D0,D1,D2,D3]
DistTensor<double> W_bmje__D_0__D_1__D_2__D_3( "[(0),(1),(2),(3)]", g );
	//Wtemp1[D0,D1,D2,D3]
DistTensor<double> Wtemp1__D_0__D_1__D_2__D_3( "[(0),(1),(2),(3)]", g );
	//Wtemp2[D0,D1,D2,D3]
DistTensor<double> Wtemp2__D_0__D_1__D_2__D_3( "[(0),(1),(2),(3)]", g );
	//Wtemp3[D0,D1,D2,D3]
DistTensor<double> Wtemp3__D_0__D_1__D_2__D_3( "[(0),(1),(2),(3)]", g );
	//Wtemp4[D0,D1,D2,D3]
DistTensor<double> Wtemp4__D_0__D_1__D_2__D_3( "[(0),(1),(2),(3)]", g );
	//r_bmfe[D0,D1,D2,D3]
DistTensor<double> r_bmfe__D_0__D_1__D_2__D_3( "[(0),(1),(2),(3)]", g );
	//u_mnje[D0,D1,D2,D3]
DistTensor<double> u_mnje__D_0__D_1__D_2__D_3( "[(0),(1),(2),(3)]", g );
	//v_femn[D0,D1,D2,D3]
DistTensor<double> v_femn__D_0__D_1__D_2__D_3( "[(0),(1),(2),(3)]", g );
	//w_bmje[D0,D1,D2,D3]
DistTensor<double> w_bmje__D_0__D_1__D_2__D_3( "[(0),(1),(2),(3)]", g );
	//x_bmej[D0,D1,D2,D3]
DistTensor<double> x_bmej__D_0__D_1__D_2__D_3( "[(0),(1),(2),(3)]", g );
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
	//X_bmej[D0,D1,D2,D3]
DistTensor<double> X_bmej__D_0__D_1__D_2__D_3( "[(0),(1),(2),(3)]", g );
	//Xtemp1[D0,D1,D2,D3]
DistTensor<double> Xtemp1__D_0__D_1__D_2__D_3( "[(0),(1),(2),(3)]", g );
// X_bmej has 4 dims
ObjShape X_bmej__D_0__D_1__D_2__D_3_tempShape( 4 );
X_bmej__D_0__D_1__D_2__D_3_tempShape[ 0 ] = n_v;
X_bmej__D_0__D_1__D_2__D_3_tempShape[ 1 ] = n_o;
X_bmej__D_0__D_1__D_2__D_3_tempShape[ 2 ] = n_v;
X_bmej__D_0__D_1__D_2__D_3_tempShape[ 3 ] = n_o;
X_bmej__D_0__D_1__D_2__D_3.ResizeTo( X_bmej__D_0__D_1__D_2__D_3_tempShape );
MakeUniform( X_bmej__D_0__D_1__D_2__D_3 );

	//U_mnie[D0,D1,D2,D3]
DistTensor<double> U_mnie__D_0__D_1__D_2__D_3( "[(0),(1),(2),(3)]", g );
// U_mnie has 4 dims
ObjShape U_mnie__D_0__D_1__D_2__D_3_tempShape( 4 );
U_mnie__D_0__D_1__D_2__D_3_tempShape[ 0 ] = n_o;
U_mnie__D_0__D_1__D_2__D_3_tempShape[ 1 ] = n_o;
U_mnie__D_0__D_1__D_2__D_3_tempShape[ 2 ] = n_o;
U_mnie__D_0__D_1__D_2__D_3_tempShape[ 3 ] = n_v;
U_mnie__D_0__D_1__D_2__D_3.ResizeTo( U_mnie__D_0__D_1__D_2__D_3_tempShape );
MakeUniform( U_mnie__D_0__D_1__D_2__D_3 );

	//Q_mnij[D0,D1,D2,D3]
DistTensor<double> Q_mnij__D_0__D_1__D_2__D_3( "[(0),(1),(2),(3)]", g );
	//Qtemp1[D0,D1,D2,D3]
DistTensor<double> Qtemp1__D_0__D_1__D_2__D_3( "[(0),(1),(2),(3)]", g );
	//q_mnij[D0,D1,D2,D3]
DistTensor<double> q_mnij__D_0__D_1__D_2__D_3( "[(0),(1),(2),(3)]", g );
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

	//P_jimb[D0,D1,D2,D3]
DistTensor<double> P_jimb__D_0__D_1__D_2__D_3( "[(0),(1),(2),(3)]", g );
// P_jimb has 4 dims
ObjShape P_jimb__D_0__D_1__D_2__D_3_tempShape( 4 );
P_jimb__D_0__D_1__D_2__D_3_tempShape[ 0 ] = n_o;
P_jimb__D_0__D_1__D_2__D_3_tempShape[ 1 ] = n_o;
P_jimb__D_0__D_1__D_2__D_3_tempShape[ 2 ] = n_o;
P_jimb__D_0__D_1__D_2__D_3_tempShape[ 3 ] = n_v;
P_jimb__D_0__D_1__D_2__D_3.ResizeTo( P_jimb__D_0__D_1__D_2__D_3_tempShape );
MakeUniform( P_jimb__D_0__D_1__D_2__D_3 );
	//H_me[D01,D23]
DistTensor<double> H_me__D_0_1__D_2_3( "[(0,1),(2,3)]", g );
	//Htemp1[D0,D1,D2,D3]
DistTensor<double> Htemp1__D_0__D_1__D_2__D_3( "[(0),(1),(2),(3)]", g );
// H_me has 2 dims
ObjShape H_me__D_0_1__D_2_3_tempShape( 2 );
H_me__D_0_1__D_2_3_tempShape[ 0 ] = n_o;
H_me__D_0_1__D_2_3_tempShape[ 1 ] = n_v;
H_me__D_0_1__D_2_3.ResizeTo( H_me__D_0_1__D_2_3_tempShape );
MakeUniform( H_me__D_0_1__D_2_3 );

	//F_ae[D01,D23]
DistTensor<double> F_ae__D_0_1__D_2_3( "[(0,1),(2,3)]", g );
	//Ftemp1[D0,D1,D2,D3]
DistTensor<double> Ftemp1__D_0__D_1__D_2__D_3( "[(0),(1),(2),(3)]", g );
	//Ftemp2[D0,D1,D2,D3]
DistTensor<double> Ftemp2__D_0__D_1__D_2__D_3( "[(0),(1),(2),(3)]", g );
// F_ae has 2 dims
ObjShape F_ae__D_0_1__D_2_3_tempShape( 2 );
F_ae__D_0_1__D_2_3_tempShape[ 0 ] = n_v;
F_ae__D_0_1__D_2_3_tempShape[ 1 ] = n_v;
F_ae__D_0_1__D_2_3.ResizeTo( F_ae__D_0_1__D_2_3_tempShape );
MakeUniform( F_ae__D_0_1__D_2_3 );

	//G_mi[D01,D23]
DistTensor<double> G_mi__D_0_1__D_2_3( "[(0,1),(2,3)]", g );
	//Gtemp1[D0,D1,D2,D3]
DistTensor<double> Gtemp1__D_0__D_1__D_2__D_3( "[(0),(1),(2),(3)]", g );
	//Gtemp2[D0,D1,D2,D3]
DistTensor<double> Gtemp2__D_0__D_1__D_2__D_3( "[(0),(1),(2),(3)]", g );
// G_mi has 2 dims
ObjShape G_mi__D_0_1__D_2_3_tempShape( 2 );
G_mi__D_0_1__D_2_3_tempShape[ 0 ] = n_o;
G_mi__D_0_1__D_2_3_tempShape[ 1 ] = n_o;
G_mi__D_0_1__D_2_3.ResizeTo( G_mi__D_0_1__D_2_3_tempShape );
MakeUniform( G_mi__D_0_1__D_2_3 );

	//z_ai[D01,D23]
DistTensor<double> z_ai__D_0_1__D_2_3( "[(0,1),(2,3)]", g );
	//ztemp2[D0,D1,D2,D3]
DistTensor<double> ztemp2__D_0__D_1__D_2__D_3( "[(0),(1),(2),(3)]", g );
	//ztemp3[D0,D1,D2,D3]
DistTensor<double> ztemp3__D_0__D_1__D_2__D_3( "[(0),(1),(2),(3)]", g );
	//ztemp4[D0,D1,D2,D3]
DistTensor<double> ztemp4__D_0__D_1__D_2__D_3( "[(0),(1),(2),(3)]", g );
	//ztemp5[D0,D1,D2,D3]
DistTensor<double> ztemp5__D_0__D_1__D_2__D_3( "[(0),(1),(2),(3)]", g );
// z_ai has 2 dims
ObjShape z_ai__D_0_1__D_2_3_tempShape( 2 );
z_ai__D_0_1__D_2_3_tempShape[ 0 ] = n_v;
z_ai__D_0_1__D_2_3_tempShape[ 1 ] = n_o;
z_ai__D_0_1__D_2_3.ResizeTo( z_ai__D_0_1__D_2_3_tempShape );
MakeUniform( z_ai__D_0_1__D_2_3 );

	//Z_abij[D0,D1,D2,D3]
DistTensor<double> Z_abij__D_0__D_1__D_2__D_3( "[(0),(1),(2),(3)]", g );
	//Zaccum[D0,D1,D2,D3]
DistTensor<double> Zaccum__D_0__D_1__D_2__D_3( "[(0),(1),(2),(3)]", g );
	//Ztemp1[D0,D1,D2,D3]
DistTensor<double> Ztemp1__D_0__D_1__D_2__D_3( "[(0),(1),(2),(3)]", g );
	//Ztemp2[D0,D1,D2,D3]
DistTensor<double> Ztemp2__D_0__D_1__D_2__D_3( "[(0),(1),(2),(3)]", g );
	//y_abef[D0,D1,D2,D3]
DistTensor<double> y_abef__D_0__D_1__D_2__D_3( "[(0),(1),(2),(3)]", g );
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


    //Tau_emfn = T_emfn + t_em*t_fn
	Tau_efmn__D_0__D_1__D_2__D_3 = T_bfnj__D_0__D_1__D_2__D_3;
	GenContract(1.0, t_fj__D_0_1__D_2_3, indices_em, t_fj__D_0_1__D_2_3, indices_fn, 1.0, Tau_efmn__D_0__D_1__D_2__D_3, indices_emfn);

	//W_nmje = 2u_nmje - u_mnje
	Wtemp3__D_0__D_1__D_2__D_3.ResizeTo(u_mnje__D_0__D_1__D_2__D_3);
	GenYAxpPx(-1.0, u_mnje__D_0__D_1__D_2__D_3, 2.0, perm_1_0_2_3, Wtemp3__D_0__D_1__D_2__D_3);

	//W_fenm = 2v_fenm - v_femn
	Wtemp2__D_0__D_1__D_2__D_3.ResizeTo(v_femn__D_0__D_1__D_2__D_3);
	GenYAxpPx(-1.0, v_fenm__D_0__D_1__D_2__D_3, 2.0, perm_0_1_3_2, Wtemp2__D_0__D_1__D_2__D_3);

	Wtemp1__D_0__D_1__D_2__D_3.ResizeTo(T_bfnj__D_0__D_1__D_2__D_3);
	GenZAxpBypPx(0.5, T_bfnj__D_0__D_1__D_2__D_3, -1.0, Tau_efmn__D_0__D_1__D_2__D_3, perm_0_1_3_2, Wtemp1__D_0__D_1__D_2__D_3 );

	//W_bmje = 2w_bmje - x_bmej
	GenYAxpPx( 2.0, w_bmje__D_0__D_1__D_2__D_3, -1.0, perm_0_1_3_2, W_bmje__D_0__D_1__D_2__D_3 );

	//W_bmje += \sum_fn (2v_fenm - v_femn)*(0.5T_bfnj - Tau_bfnj + T_bfjn)
	GenContract(1.0, Wtemp2__D_0__D_1__D_2__D_3, indices_fenm, Wtemp1__D_0__D_1__D_2__D_3, indices_bfnj, 1.0, W_bmje__D_0__D_1__D_2__D_3, indices_bmje);

	//W_bmje -= \sum_n (2u_nmje - u_mnje) * t_bn
	GenContract(-1.0, Wtemp3__D_0__D_1__D_2__D_3, indices_nmje, t_fj__D_0_1__D_2_3, indices_bn, 1.0, W_bmje__D_0__D_1__D_2__D_3, indices_bmje);

	//W_bmef = 2r_bmfe - r_bmef
	GenYAxpPx( -1.0, r_bmfe__D_0__D_1__D_2__D_3, 2.0, perm_0_1_3_2, Wtemp4__D_0__D_1__D_2__D_3);

	//W_bmje += \sum_f (2r_bmfe - r_bmef)*t_fj
	GenContract(1.0, Wtemp4__D_0__D_1__D_2__D_3, indices_bmfe, t_fj__D_0_1__D_2_3, indices_fj, 1.0, W_bmje__D_0__D_1__D_2__D_3, indices_bmje);

	//X_bfnj = tau_bfnj - 0.5 * T_bfnj
	ZAxpBy(1.0, Tau_efmn__D_0__D_1__D_2__D_3, -0.5, T_bfnj__D_0__D_1__D_2__D_3, Xtemp1__D_0__D_1__D_2__D_3);

	//X_bmej = x_bmej;
	X_bmej__D_0__D_1__D_2__D_3 = x_bmej__D_0__D_1__D_2__D_3;

	//X_bmej = \sum_fn v_femn (tau_bfnj - 0.5 * T_bfnj)
	GenContract(-0.5, v_femn__D_0__D_1__D_2__D_3, indices_femn, Xtemp1__D_0__D_1__D_2__D_3, indices_bfnj, 0.0, X_bmej__D_0__D_1__D_2__D_3, indices_bmej);

	//X_bmej -= \sum_n u_mnje * t_bn
	GenContract(-1.0, u_mnje__D_0__D_1__D_2__D_3, indices_mnje, t_fj__D_0_1__D_2_3, indices_bn, 1.0, X_bmej__D_0__D_1__D_2__D_3, indices_bmej);

	//X_bmej += \sum_f r_bmef * t_fj
	GenContract(1.0, r_bmfe__D_0__D_1__D_2__D_3, indices_bmef, t_fj__D_0_1__D_2_3, indices_fj, 1.0, X_bmej__D_0__D_1__D_2__D_3, indices_bmej);

	//U_mnie = u_mnie;
	U_mnie__D_0__D_1__D_2__D_3 = u_mnje__D_0__D_1__D_2__D_3;

	//U_mnie += \sum_f v_femn * t_fj
	GenContract(1.0, v_femn__D_0__D_1__D_2__D_3, indices_femn, t_fj__D_0_1__D_2_3, indices_fj, 1.0, U_mnie__D_0__D_1__D_2__D_3, indices_mnie);

	//Q_mnij = \sum_ef v_efmn * tau_efij
	GenContract(1.0, v_femn__D_0__D_1__D_2__D_3, indices_efmn, Tau_efmn__D_0__D_1__D_2__D_3, indices_efij, 0.0, Qtemp1__D_0__D_1__D_2__D_3, indices_mnij);

	//Q_mnij = (1 + P_mnij) (\sum_ef v_efmn * tau_efij)
	GenYAxpPx(1.0, Qtemp1__D_0__D_1__D_2__D_3, 1.0, perm_1_0_3_2, Q_mnij__D_0__D_1__D_2__D_3);

	//Q_mnij += \sum_ef v_efmn * tau_efij
	GenContract(1.0, v_femn__D_0__D_1__D_2__D_3, indices_efmn, Tau_efmn__D_0__D_1__D_2__D_3, indices_efij, 1.0, Q_mnij__D_0__D_1__D_2__D_3, indices_mnij);

	//Q_mnij += q_mnij
	YAxpBy(1.0, q_mnij__D_0__D_1__D_2__D_3, 1.0, Q_mnij__D_0__D_1__D_2__D_3);

	//P_jimb = u_jimb
	P_jimb__D_0__D_1__D_2__D_3 = u_mnje__D_0__D_1__D_2__D_3;

	//P_jimb += \sum_ef r_bmef * tau_efij
	GenContract(1.0, r_bmfe__D_0__D_1__D_2__D_3, indices_bmef, Tau_efmn__D_0__D_1__D_2__D_3, indices_efij, 1.0, P_jimb__D_0__D_1__D_2__D_3, indices_jimb);

	//P_jimb += \sum_e w_bmie * t_ej
	GenContract(1.0, w_bmje__D_0__D_1__D_2__D_3, indices_bmie, t_fj__D_0_1__D_2_3, indices_ej, 1.0, P_jimb__D_0__D_1__D_2__D_3, indices_jimb);

	//P_jimb += \sum_e x_bmej * t_ei
	GenContract(1.0, x_bmej__D_0__D_1__D_2__D_3, indices_bmej, t_fj__D_0_1__D_2_3, indices_ei, 1.0, P_jimb__D_0__D_1__D_2__D_3, indices_jimb);

	//tmp = (2v_efmn - v_efnm)
	GenYAxpPx(2.0, v_femn__D_0__D_1__D_2__D_3, -1.0, perm_0_1_3_2, Htemp1__D_0__D_1__D_2__D_3);

	//H_me = \sum_fn (2v_efmn - v_efnm) * t_fn
	GenContract(1.0, Htemp1__D_0__D_1__D_2__D_3, indices_efmn, t_fj__D_0_1__D_2_3, indices_fn, 1.0, H_me__D_0_1__D_2_3, indices_me);

	//F_ae = -\sum_m H_me * t_am
	GenContract(-1.0, H_me__D_0_1__D_2_3, indices_me, t_fj__D_0_1__D_2_3, indices_am, 0.0, F_ae__D_0_1__D_2_3, indices_ae);

	//tmp = (2r_amef - r_amfe)
	GenYAxpPx(2.0, r_bmfe__D_0__D_1__D_2__D_3, -1.0, perm_0_1_3_2, Ftemp1__D_0__D_1__D_2__D_3);

	//F_ae += \sum_m (2r_amef - r_amfe) * t_am
	GenContract(1.0, Ftemp1__D_0__D_1__D_2__D_3, indices_amef, t_fj__D_0_1__D_2_3, indices_am, 1.0, F_ae__D_0_1__D_2_3, indices_ae);

	//tmp = (2v_efmn - v_efnm)
	GenYAxpPx(2.0, v_femn__D_0__D_1__D_2__D_3, -1.0, perm_0_1_3_2, Ftemp2__D_0__D_1__D_2__D_3);

	//F_ae += \sum_fmn (2v_efmn - v_efnm) * T_afmn
	GenContract(1.0, Ftemp2__D_0__D_1__D_2__D_3, indices_efmn, T_bfnj__D_0__D_1__D_2__D_3, indices_afmn, 1.0, F_ae__D_0_1__D_2_3, indices_ae);

	//G_mi = \sum_e H_me * t_ei
	GenContract(1.0, H_me__D_0_1__D_2_3, indices_me, t_fj__D_0_1__D_2_3, indices_ei, 0.0, G_mi__D_0_1__D_2_3, indices_mi);

	//tmp = (2u_mnie - u_nmie)
	GenYAxpPx(2.0, u_mnje__D_0__D_1__D_2__D_3, -1.0, perm_1_0_2_3, Gtemp1__D_0__D_1__D_2__D_3);

	//G_mi += \sum_en (2u_mnie - u_nmie) * t_en
	GenContract(1.0, Gtemp1__D_0__D_1__D_2__D_3, indices_mnie, t_fj__D_0_1__D_2_3, indices_en, 1.0, G_mi__D_0_1__D_2_3, indices_mi);

	//tmp = (2v_efmn - v_efnm)
	GenYAxpPx(2.0, v_femn__D_0__D_1__D_2__D_3, -1.0, perm_0_1_3_2, Gtemp2__D_0__D_1__D_2__D_3);

	//G_mi += \sum_efn (2v_efmn - v_efnm) * T_efin
	GenContract(1.0, Gtemp2__D_0__D_1__D_2__D_3, indices_efmn, T_bfnj__D_0__D_1__D_2__D_3, indices_efin, 1.0, G_mi__D_0_1__D_2_3, indices_mi);

	//z_ai = -\sum_m G_mi * t_am
	GenContract(-1.0, G_mi__D_0_1__D_2_3, indices_mi, t_fj__D_0_1__D_2_3, indices_am, 0.0, z_ai__D_0_1__D_2_3, indices_ai);

	//tmp = (2U_mnie - U_nmie)
	GenYAxpPx(2.0, U_mnie__D_0__D_1__D_2__D_3, -1.0, perm_1_0_2_3, ztemp5__D_0__D_1__D_2__D_3);

	//z_ai -= \sum_emn (2U_mnie - U_nmie) * T_aemn
	GenContract(-1.0, ztemp5__D_0__D_1__D_2__D_3, indices_mnie, T_bfnj__D_0__D_1__D_2__D_3, indices_aemn, -1.0, z_ai__D_0_1__D_2_3, indices_ai);

	//tmp = (2w_amie - x_amei)
	ZAxpBy(2.0, w_bmje__D_0__D_1__D_2__D_3, -1.0, x_bmej__D_0__D_1__D_2__D_3, ztemp4__D_0__D_1__D_2__D_3);

	//z_ai += \sum_em (2w_amie - x_amei) * t_em
	GenContract(1.0, ztemp4__D_0__D_1__D_2__D_3, indices_amie, t_fj__D_0_1__D_2_3, indices_em, 1.0, z_ai__D_0_1__D_2_3, indices_ai);

	//tmp = (2T_aeim - T_aemi)
	GenYAxpPx(2.0, T_bfnj__D_0__D_1__D_2__D_3, -1.0, perm_0_1_3_2, ztemp3__D_0__D_1__D_2__D_3);

	//z_ai += \sum_em (2T_aeim - T_aemi) * H_me
	GenContract(-1.0, ztemp3__D_0__D_1__D_2__D_3, indices_aeim, H_me__D_0_1__D_2_3, indices_me, 1.0, z_ai__D_0_1__D_2_3, indices_ai);

	//tmp = (2r_amef - r_amfe)
	GenYAxpPx(2.0, r_bmfe__D_0__D_1__D_2__D_3, -1.0, perm_0_1_3_2, ztemp2__D_0__D_1__D_2__D_3);

	//z_ai += \sum_efm (2r_amef - r_amfe) * tau_efim
	GenContract(-1.0, ztemp2__D_0__D_1__D_2__D_3, indices_amef, Tau_efmn__D_0__D_1__D_2__D_3, indices_efim, 1.0, z_ai__D_0_1__D_2_3, indices_ai);

	//tmp = \sum_em X_bmej * T_aemi
	GenContract(1.0, X_bmej__D_0__D_1__D_2__D_3, indices_bmej, T_bfnj__D_0__D_1__D_2__D_3, indices_aemi, 0.0, Ztemp1__D_0__D_1__D_2__D_3, indices_abij);

	//tmpAccum = -(0.5 + P_ij) (\sum_em X_bmej * T_aemi)
	GenYAxpPx(-0.5, Ztemp1__D_0__D_1__D_2__D_3, -1.0, perm_0_1_3_2, Zaccum__D_0__D_1__D_2__D_3);

	//tmp2 = (2T_aeim - T_aemi)
	GenYAxpPx(2.0, T_bfnj__D_0__D_1__D_2__D_3, -1.0, perm_0_1_3_2, Ztemp2__D_0__D_1__D_2__D_3);

	//tmpAccum += 0.5 * \sum_em W_bmje * (2T_aeim - T_aemi)
	GenContract(0.5, W_bmje__D_0__D_1__D_2__D_3, indices_bmje, Ztemp2__D_0__D_1__D_2__D_3, indices_aeim, 1.0, Zaccum__D_0__D_1__D_2__D_3, indices_abij);

	//tmpAccum -= \sum_m G_mi * T_abmj
	GenContract(1.0, G_mi__D_0_1__D_2_3, indices_mi, T_bfnj__D_0__D_1__D_2__D_3, indices_abmj, -1.0, Zaccum__D_0__D_1__D_2__D_3, indices_abij);

	//tmpAccum += \sum_e F_ae * T_ebij
	GenContract(1.0, F_ae__D_0_1__D_2_3, indices_ae, T_bfnj__D_0__D_1__D_2__D_3, indices_ebij, 1.0, Zaccum__D_0__D_1__D_2__D_3, indices_abij);

	//tmpAccum -= \sum_m P_ijmb * t_am
	GenContract(1.0, P_jimb__D_0__D_1__D_2__D_3, indices_ijmb, t_fj__D_0_1__D_2_3, indices_am, -1.0, Zaccum__D_0__D_1__D_2__D_3, indices_abij);

	//tmpAccum += \sum_e r_ejab * t_ei
	GenContract(1.0, r_bmfe__D_0__D_1__D_2__D_3, indices_ejab, t_fj__D_0_1__D_2_3, indices_ei, 1.0, Zaccum__D_0__D_1__D_2__D_3, indices_abij);

	//Z_abij = (1 + P_aibj) Zaccum
	GenYAxpPx(1.0, Zaccum__D_0__D_1__D_2__D_3, 1.0, perm_1_0_3_2, Z_abij__D_0__D_1__D_2__D_3);

	//Z_abij = \sum_ef y_abef * tau_efij
	GenContract(1.0, y_abef__D_0__D_1__D_2__D_3, indices_abef, Tau_efmn__D_0__D_1__D_2__D_3, indices_efij, 1.0, Z_abij__D_0__D_1__D_2__D_3, indices_abij);

	//Z_abij = \sum_mn Q_mnij * tau_abmn
	GenContract(1.0, Q_mnij__D_0__D_1__D_2__D_3, indices_mnij, Tau_efmn__D_0__D_1__D_2__D_3, indices_abmn, 1.0, Z_abij__D_0__D_1__D_2__D_3, indices_abij);

	//Z_abij += v_abij
	YAxpBy(1.0, v_femn__D_0__D_1__D_2__D_3, 1.0, Z_abij__D_0__D_1__D_2__D_3);



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


