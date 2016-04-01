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
void PrintNorm(const DistTensor<T>& check, const DistTensor<T>& actual, const char* msg){
	int commRank = mpi::CommRank(MPI_COMM_WORLD);

	std::string dist4 = "[(0),(1),(2),(3)]";
	std::string dist2 = "[(0,1),(2,3)]";
	std::string dist;
	if(check.Order() == 4)
		dist = dist4;
	else
		dist = dist2;

	DistTensor<double> diff(dist.c_str(), check.Grid());
	diff.ResizeTo(check);
	Diff(check, actual, diff);
	double norm = 1.0;
	norm = Norm(diff);
	if (commRank == 0){
	  std::cout << "NORM " << msg << " " << norm << std::endl;
	}
}

template<typename T>
void InitCheckTensor(const std::string& data_path, int testIter, DistTensor<T>& TT){
	stringstream fullName;
	fullName.str("");
	fullName.clear();
	fullName << data_path << testIter;
	Read(TT, fullName.str(), BINARY_FLAT, false);
}

template<typename T>
void ComputeTW(const DistTensor<T>& Twsa, const DistTensor<T>& Trsa, const DistTensor<T>& Tusa, const DistTensor<T>& Tvsa, const DistTensor<T>& Ttau2sa, const DistTensor<T>& Tt, DistTensor<T>& TW, Unsigned blkSize){
	//W_bmje = (2w_amie - x_amei)
	YAxpBy(1.0, Twsa, 0.0, TW);

	//W_bmje += \sum_f (2r_bmfe - r_bmef)*t_fj
	GenContract(1.0, Trsa, "bmfe", Tt, "fj", 1.0, TW, "bmje");

	//W_bmje += -\sum_n (2u_nmje - u_mnje) * t_bn
	GenContract(-1.0, Tusa, "nmje", Tt, "bn", 1.0, TW, "bmje");

	const std::vector<Unsigned> blkSizes = {30*blkSize, 3*blkSize};
	//W_bmje += \sum_fn (2v_fenm - v_femn)*(0.5T_bfnj - Tau_bfnj + T_bfjn)
	GenContract(1.0, Tvsa, "fenm", Ttau2sa, "bfnj", 1.0, TW, "bmje", blkSizes);
}

template<typename T>
void ComputeTX(const DistTensor<T>& Tx, const DistTensor<T>& Tv, const DistTensor<T>& Ttau2, const DistTensor<T>& Tu, const DistTensor<T>& Tt, const DistTensor<T>& Tr, DistTensor<T>& TX, Unsigned blkSize){
	//X_bmej = x_bmej;
	YAxpBy(1.0, Tx, 0.0, TX);

	const std::vector<Unsigned> blkSizes = {30*blkSize, 3*blkSize};
	//X_bmej += -\sum_fn v_femn (tau_bfnj - 0.5 * T_bfnj)
	GenContract(-1.0, Tv, "femn", Ttau2, "bfnj", 1.0, TX, "bmej", blkSizes);

	//X_bmej -= \sum_n u_mnje * t_bn
	GenContract(-1.0, Tu, "mnje", Tt, "bn", 1.0, TX, "bmej");

	//X_bmej += \sum_f r_bmef * t_fj
	GenContract(1.0, Tr, "bmef", Tt, "fj", 1.0, TX, "bmej");
}

template<typename T>
void ComputeTU(const DistTensor<T>& Tu, const DistTensor<T>& Tv, const DistTensor<T>& Tt, DistTensor<T>& TU){
	//U_mnie = u_mnie;
	YAxpBy(1.0, Tu, 0.0, TU);

	//U_mnie += \sum_f v_femn * t_fi
	GenContract(1.0, Tv, "femn", Tt, "fi", 1.0, TU, "mnie");
}

template<typename T>
void ComputeTQ(const DistTensor<T>& Tu, const DistTensor<T>& Tt, const DistTensor<T>& Tv, const DistTensor<T>& Ttau, const DistTensor<T>& Tq, DistTensor<T>& TQ, Unsigned blkSize){
	DistTensor<double> TQtemp1( TQ.Shape(), "[(0),(1),(2),(3)]", TQ.Grid() );
	Permutation perm_1_0_3_2 = {1,0,3,2};

	//tmp = \sum_e u_mnie * t_ej
	GenContract(1.0, Tu, "mnie", Tt, "ej", 0.0, TQtemp1, "mnij");

	//Q_mnij = (1 + P_mnij) (\sum_e u_mnie * t_ej)
	GenYAxpPx(1.0, TQtemp1, 1.0, perm_1_0_3_2, TQ);

	std::vector<Unsigned> blkSizes = {4*blkSize, 4*blkSize};
	//Q_mnij += \sum_ef v_efmn * tau_efij
	GenContract(1.0, Tv, "efmn", Ttau, "efij", 1.0, TQ, "mnij");

	//Q_mnij += q_mnij
	YAxpBy(1.0, Tq, 1.0, TQ);
}

template<typename T>
void ComputeTP(const DistTensor<T>& Tu, const DistTensor<T>& Tr, const DistTensor<T>& Ttau, const DistTensor<T>& Tw, const DistTensor<T>& Tt, const DistTensor<T>& Tx, DistTensor<T>& TP, Unsigned blkSize){
	//P_jimb = u_jimb
	YAxpBy(1.0, Tu, 0.0, TP);

	std::vector<Unsigned> blkSizes = {4*blkSize, 4*blkSize};
	//P_jimb += \sum_ef r_bmef * tau_efij
	GenContract(1.0, Tr, "bmef", Ttau, "efij", 1.0, TP, "jimb", blkSizes);

	//P_jimb += \sum_e w_bmie * t_ej
	GenContract(1.0, Tw, "bmie", Tt, "ej", 1.0, TP, "jimb");

	//P_jimb += \sum_e x_bmej * t_ei
	GenContract(1.0, Tx, "bmej", Tt, "ei", 1.0, TP, "jimb");
}

template<typename T>
void ComputeTH(const DistTensor<T>& Tvsa, const DistTensor<T>& Tt, DistTensor<T>& TH){
	//H_me = \sum_fn (2v_efmn - v_efnm) * t_fn
	GenContract(1.0, Tvsa, "efmn", Tt, "fn", 0.0, TH, "me");
}

template<typename T>
void ComputeTF(const DistTensor<T>& TH, const DistTensor<T>& Tt, const DistTensor<T>& Trsa, const DistTensor<T>& Tvsa, const DistTensor<T>& TT, DistTensor<T>& TF){
	//F_ae = -\sum_m H_me * t_am
	GenContract(-1.0, TH, "me", Tt, "am", 0.0, TF, "ae");

	//F_ae += \sum_fm (2r_amef - r_amfe) * t_fm
	GenContract(1.0, Trsa, "amef", Tt, "fm", 1.0, TF, "ae");

	//F_ae += -\sum_fmn (2v_efmn - v_efnm) * T_afmn
	GenContract(-1.0, Tvsa, "efmn", TT, "afmn", 1.0, TF, "ae");
}

template<typename T>
void ComputeTG(const DistTensor<T>& TH, const DistTensor<T>& Tt, const DistTensor<T>& Tusa, const DistTensor<T>& Tvsa, const DistTensor<T>& TT, DistTensor<T>& TG){
	//G_mi = \sum_e H_me * t_ei
	GenContract(1.0, TH, "me", Tt, "ei", 0.0, TG, "mi");

	//G_mi += \sum_en (2u_mnie - u_nmie) * t_en
	GenContract(1.0, Tusa, "mnie", Tt, "en", 1.0, TG, "mi");

	//G_mi += \sum_efn (2v_efmn - v_efnm) * T_efin
	GenContract(1.0, Tvsa, "efmn", TT, "efin", 1.0, TG, "mi");
}

template<typename T>
void ComputeTz(const DistTensor<T>& TG, const DistTensor<T>& Tt, const DistTensor<T>& TUsa, const DistTensor<T>& TT, const DistTensor<T>& Twsa, const DistTensor<T>& TTsa, const DistTensor<T>& TH, const DistTensor<T>& Trsa, const DistTensor<T>& Ttau, DistTensor<T>& Tz){
	//z_ai = -\sum_m G_mi * t_am
	GenContract(-1.0, TG, "mi", Tt, "am", 0.0, Tz, "ai");

	//z_ai += -\sum_emn (2U_mnie - U_nmie) * T_aemn
	GenContract(-1.0, TUsa, "mnie", TT, "aemn", 1.0, Tz, "ai");

	//z_ai += \sum_em (2w_amie - x_amei) * t_em
	GenContract(1.0, Twsa, "amie", Tt, "em", 1.0, Tz, "ai");

	//z_ai += \sum_em (2T_aeim - T_aemi) * H_me
	GenContract(1.0, TTsa, "aeim", TH, "me", 1.0, Tz, "ai");

	//z_ai += \sum_efm (2r_amef - r_amfe) * tau_efim
	GenContract(1.0, Trsa, "amef", Ttau, "efim", 1.0, Tz, "ai");
}

template<typename T>
void ComputeTZ(const DistTensor<T>& TX, const DistTensor<T>& TT, const DistTensor<T>& TW, const DistTensor<T>& TTsa, const DistTensor<T>& TG, const DistTensor<T>& TF, const DistTensor<T>& TP, const DistTensor<T>& Tt, const DistTensor<T>& Tr, const DistTensor<T>& Ty, const DistTensor<T>& Ttau, const DistTensor<T>& TQ, const DistTensor<T>& Tv, DistTensor<T>& TZ, Unsigned blkSize){
	DistTensor<T> TZtemp1(TZ.Shape(), TZ.TensorDist(), TZ.Grid());
	DistTensor<T> TZaccum(TZ.Shape(), TZ.TensorDist(), TZ.Grid());
	Permutation perm_0_1_3_2 = {0,1,3,2};
	Permutation perm_1_0_3_2 = {1,0,3,2};

	std::vector<Unsigned> blkSizes = {1*blkSize, 10*blkSize};
	//tmp = \sum_em X_bmej * T_aemi
	GenContract(1.0, TX, "bmej", TT, "aemi", 0.0, TZtemp1, "abij", blkSizes);

	//tmpAccum = -(0.5 + P_ij) (\sum_em X_bmej * T_aemi)
	GenYAxpPx(-0.5, TZtemp1, -1.0, perm_0_1_3_2, TZaccum);

	std::vector<Unsigned> blkSizes2 = {1*blkSize, 10*blkSize};
	//tmpAccum += 0.5 * \sum_em W_bmje * (2T_aeim - T_aemi)
	GenContract(0.5, TW, "bmje", TTsa, "aeim", 1.0, TZaccum, "abij", blkSizes2);

	//tmpAccum += -\sum_m G_mi * T_abmj
	GenContract(-1.0, TG, "mi", TT, "abmj", 1.0, TZaccum, "abij");

	//tmpAccum += \sum_e F_ae * T_ebij
	GenContract(1.0, TF, "ae", TT, "ebij", 1.0, TZaccum, "abij");

	//tmpAccum += -\sum_m P_ijmb * t_am
	GenContract(-1.0, TP, "ijmb", Tt, "am", 1.0, TZaccum, "abij");

	//tmpAccum += \sum_e r_ejab * t_ei
	GenContract(1.0, Tr, "ejab", Tt, "ei", 1.0, TZaccum, "abij");

	//Z_abij = (1 + P_aibj) TZaccum
	GenYAxpPx(1.0, TZaccum, 1.0, perm_1_0_3_2, TZ);

	std::vector<Unsigned> blkSizes3 = {3*blkSize, 3*blkSize};
	//Z_abij = \sum_ef y_abef * tau_efij
	GenContract(1.0, Ty, "abef", Ttau, "efij", 1.0, TZ, "abij", blkSizes3);

	std::vector<Unsigned> blkSizes4 = {4*blkSize, 4*blkSize};
	//Z_abij = \sum_mn Q_mnij * tau_abmn
	GenContract(1.0, TQ, "mnij", Ttau, "abmn", 1.0, TZ, "abij", blkSizes4);

	//Z_abij += v_abij
	YAxpBy(1.0, Tv, 1.0, TZ);
}

template<typename T>
void DistTensorTest(const Grid& g, Unsigned n_o, Unsigned n_v,
        Unsigned blkSize, Unsigned testIter) {
#ifndef RELEASE
    CallStackEntry entry("DistTensorTest");
#endif
    const Int commRank = mpi::CommRank(mpi::COMM_WORLD);

//START_DECL
Permutation perm_1_0_2_3 = {1, 0, 2, 3};
Permutation perm_0_1_3_2 = {0, 1, 3, 2};

//ObjShapes
ObjShape oo = {n_o, n_o};
ObjShape ov = {n_o, n_v};
ObjShape vo = {n_v, n_o};
ObjShape vv = {n_v, n_v};
ObjShape vvoo = {n_v, n_v, n_o, n_o};
ObjShape vovv = {n_v, n_o, n_v, n_v};
ObjShape ooov = {n_o, n_o, n_o, n_v};
ObjShape voov = {n_v, n_o, n_o, n_v};
ObjShape vovo = {n_v, n_o, n_v, n_o};
ObjShape oooo = {n_o, n_o, n_o, n_o};
ObjShape vvvv = {n_v, n_v, n_v, n_v};

//Input tensors
DistTensor<double> Tv(   vvoo,   "[(0),(1),(2),(3)]", g );
DistTensor<double> Tq(   oooo,   "[(0),(1),(2),(3)]", g );
DistTensor<double> Tu(   ooov,   "[(0),(1),(2),(3)]", g );
DistTensor<double> Tw(   voov,   "[(0),(1),(2),(3)]", g );
DistTensor<double> Tx(   vovo,   "[(0),(1),(2),(3)]", g );
DistTensor<double> Tr(   vovv,   "[(0),(1),(2),(3)]", g );
DistTensor<double> Ty(   vvvv,   "[(0),(1),(2),(3)]", g );

//Output tensors
DistTensor<double> TW(   voov,   "[(0),(1),(2),(3)]", g );
DistTensor<double> TX(   vovo,   "[(0),(1),(2),(3)]", g );
DistTensor<double> TU(   ooov,   "[(0),(1),(2),(3)]", g );
DistTensor<double> TQ(   oooo,   "[(0),(1),(2),(3)]", g );
DistTensor<double> TP(   ooov,   "[(0),(1),(2),(3)]", g );
DistTensor<double> TH(   ov,     "[(0,1),(2,3)]", g );
DistTensor<double> TF(   vv,     "[(0,1),(2,3)]", g );
DistTensor<double> TG(   oo,     "[(0,1),(2,3)]", g );
DistTensor<double> Tz(   vo,     "[(0,1),(2,3)]", g );
DistTensor<double> TZ(   vvoo,   "[(0),(1),(2),(3)]", g );

//Temporaries
DistTensor<double> Tt(   vo,   "[(0,1),(2,3)]", g );
DistTensor<double> TT(   vvoo, "[(0),(1),(2),(3)]", g );
DistTensor<double> Ttau( vvoo, "[(0),(1),(2),(3)]", g );
DistTensor<double> Ttau2(      "[(0),(1),(2),(3)]", g );

DistTensor<double> Twsa(       "[(0),(1),(2),(3)]", g );
DistTensor<double> Tvsa(       "[(0),(1),(2),(3)]", g );
DistTensor<double> Trsa(       "[(0),(1),(2),(3)]", g );
DistTensor<double> Tusa(       "[(0),(1),(2),(3)]", g );
DistTensor<double> TTsa(       "[(0),(1),(2),(3)]", g );
DistTensor<double> Ttau2sa(    "[(0),(1),(2),(3)]", g );
DistTensor<double> TUsa(       "[(0),(1),(2),(3)]", g );

//Init to random
MakeUniform( Tv );
MakeUniform( Tq );
MakeUniform( Tu );
MakeUniform( Tw );
MakeUniform( Tx );
MakeUniform( Tr );
MakeUniform( Ty );

MakeUniform( TW );
MakeUniform( TX );
MakeUniform( TU );
MakeUniform( TQ );
MakeUniform( TP );
MakeUniform( TH );
MakeUniform( TF );
MakeUniform( TG );
MakeUniform( Tz );
MakeUniform( TZ );

MakeUniform( Tt );
MakeUniform( TT );

//END_DECL

//******************************
//* Load tensors
//******************************
#ifdef CORRECTNESS
Read(Tv, "ccsd_terms/term_v_small", BINARY_FLAT, false);
Read(Tq, "ccsd_terms/term_q_small", BINARY_FLAT, false);
Read(Tu, "ccsd_terms/term_u_small", BINARY_FLAT, false);
Read(Tw, "ccsd_terms/term_w_small", BINARY_FLAT, false);
Read(Tx, "ccsd_terms/term_x_small", BINARY_FLAT, false);
Read(Tr, "ccsd_terms/term_r_small", BINARY_FLAT, false);
Read(Ty, "ccsd_terms/term_y_small", BINARY_FLAT, false);

DistTensor<T> check_Tau("[(0),(1),(2),(3)]", g);
DistTensor<T> check_W("[(0),(1),(2),(3)", g);
DistTensor<T> check_X("[(0),(1),(2),(3)]", g);
DistTensor<T> check_U("[(0),(1),(2),(3)]", g);
DistTensor<T> check_Q("[(0),(1),(2),(3)]", g);
DistTensor<T> check_P("[(0),(1),(2),(3)]", g);
DistTensor<T> check_H("[(0,1),(2,3)]", g);
DistTensor<T> check_F("[(0,1),(2,3)]", g);
DistTensor<T> check_G("[(0,1),(2,3)]", g);
DistTensor<T> check_z_small("[(0,1),(2,3)]", g);
DistTensor<T> check_Z("[(0),(1),(2),(3)]", g);

check_Tau.ResizeTo(Ttau.Shape());
check_W.ResizeTo(TW.Shape());
check_X.ResizeTo(TX.Shape());
check_U.ResizeTo(TU.Shape());
check_Q.ResizeTo(TQ.Shape());
check_P.ResizeTo(TP.Shape());
check_H.ResizeTo(TH.Shape());
check_F.ResizeTo(TF.Shape());
check_G.ResizeTo(TG.Shape());
check_z_small.ResizeTo(Tz.Shape());
check_Z.ResizeTo(TZ.Shape());

InitCheckTensor("ccsd_terms/term_t_small_iter", testIter, Tt);
InitCheckTensor("ccsd_terms/term_T_iter", testIter, TT);
InitCheckTensor("ccsd_terms/term_Tau_iter", testIter, check_Tau);
InitCheckTensor("ccsd_terms/term_W_iter", testIter, check_W);
InitCheckTensor("ccsd_terms/term_X_iter", testIter, check_X);
InitCheckTensor("ccsd_terms/term_U_iter", testIter, check_U);
InitCheckTensor("ccsd_terms/term_Q_iter", testIter, check_Q);
InitCheckTensor("ccsd_terms/term_P_iter", testIter, check_P);
InitCheckTensor("ccsd_terms/term_H_iter", testIter, check_H);
InitCheckTensor("ccsd_terms/term_F_iter", testIter, check_F);
InitCheckTensor("ccsd_terms/term_G_iter", testIter, check_G);
InitCheckTensor("ccsd_terms/term_z_small_iter", testIter, check_z_small);
InitCheckTensor("ccsd_terms/term_Z_iter", testIter, check_Z);
#endif

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

    //Setup temporaries
    //Tau_emfn = T_emfn + t_em*t_fn
	Ttau = TT;
	GenContract(1.0, Tt, "em", Tt, "fn", 1.0, Ttau, "efmn");

	//X_bfnj = tau_bfnj - 0.5 * T_bfnj
	ZAxpBy(1.0, Ttau, -0.5, TT, Ttau2);

	//wsa = (2w_amie - x_amei)
	GenZAxpBPy( 2.0, Tw, -1.0, Tx, perm_0_1_3_2, Twsa );

	//tmp = 2v_fenm - v_femn
	GenYAxpPx(2.0, Tv, -1.0, perm_0_1_3_2, Tvsa);

	//tmp = 2r_bmfe - r_bmef
	GenYAxpPx( 2.0, Tr, -1.0, perm_0_1_3_2, Trsa);

	//tmp = 2u_nmje - u_mnje
	GenYAxpPx(2.0, Tu, -1.0, perm_1_0_2_3, Tusa);

	//tmp = (2T_aeim - T_aemi)
	GenYAxpPx(2.0, TT, -1.0, perm_0_1_3_2, TTsa);

	//tmp = (T_bfjn + 0.5*T_bfnj - Tau_bfnj)
	GenZAxpBypPx(0.5, TT, -1.0, Ttau, perm_0_1_3_2, Ttau2sa );

	//BEGIN COMPUTATION
	ComputeTW(Twsa, Trsa, Tusa, Tvsa, Ttau2sa, Tt, TW, blkSize);
	ComputeTX(Tx, Tv, Ttau2, Tu, Tt, Tr, TX, blkSize);
	ComputeTU(Tu, Tv, Tt, TU);

	//tmp = (2U_mnie - U_nmie)
	GenYAxpPx(2.0, TU, -1.0, perm_1_0_2_3, TUsa);

	ComputeTQ(Tu, Tt, Tv, Ttau, Tq, TQ, blkSize);
	ComputeTP(Tu, Tr, Ttau, Tw, Tt, Tx, TP, blkSize);
	ComputeTH(Tvsa, Tt, TH);
	ComputeTF(TH, Tt, Trsa, Tvsa, TT, TF);
	ComputeTG(TH, Tt, Tusa, Tvsa, TT, TG);
	ComputeTz(TG, Tt, TUsa, TT, Twsa, TTsa, TH, Trsa, Ttau, Tz);
	ComputeTZ(TX, TT, TW, TTsa, TG, TF, TP, Tt, Tr, Ty, Ttau, TQ, Tv, TZ, blkSize);

	//END COMPUTATION

    mpi::Barrier(g.OwningComm());
    runTime = mpi::Time() - startTime;
#ifdef CORRECTNESS
    PrintNorm(check_W, TW, "W");
    PrintNorm(check_X, TX, "X");
    PrintNorm(check_U, TU, "U");
    PrintNorm(check_Q, TQ, "Q");
    PrintNorm(check_P, TP, "P");
    PrintNorm(check_H, TH, "H");
    PrintNorm(check_G, TG, "G");
    PrintNorm(check_z_small, Tz, "z");
    PrintNorm(check_Z, TZ, "Z");
#endif

#ifdef PROFILE
    if (commRank == 0)
        Timer::printTimers();
#endif

    if (commRank == 0) {
        std::cout << "RUNTIME " << runTime << std::endl;
    }
}

int main(int argc, char* argv[]) {
    Initialize(argc, argv);
    mpi::Comm comm = mpi::COMM_WORLD;
    const Int commRank = mpi::CommRank(comm);
    const Int commSize = mpi::CommSize(comm);
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

        const Grid g(comm, args.gridShape);
        DistTensorTest<double>(g, args.n_o, args.n_v, args.blkSize, args.testIter);

    } catch (std::exception& e) {
        ReportException(e);
    }

    Finalize();
    return 0;
}


