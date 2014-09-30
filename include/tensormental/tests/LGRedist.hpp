#pragma once
#ifndef TMEN_TESTS_LGREDIST_HPP
#define TMEN_TESTS_LGREDIST_HPP

#include "tensormental/tests/AllRedists.hpp"
using namespace tmen;


typedef std::pair< std::pair<ModeArray, std::vector<ModeArray>>, TensorDistribution> LGTest;

template<typename T>
TensorDistribution
DetermineResultingDistributionLocalG(const DistTensor<T>& A, const ModeArray& lModes, const std::vector<ModeArray>& gridRedistModes){
    Unsigned i;
    TensorDistribution ret = A.TensorDist();
    for(i = 0; i < lModes.size(); i++){
        ret[lModes[i]].insert(ret[lModes[i]].end(), gridRedistModes[i].begin(), gridRedistModes[i].end());
    }
    return ret;
}

template<typename T>
void
TestLGRedist( const DistTensor<T>& A, const ModeArray& lModes, const std::vector<ModeArray>& gridRedistModes, const TensorDistribution& resDist )
{
#ifndef RELEASE
    CallStackEntry entry("TestLGRedist");
#endif
    Unsigned i;
    Unsigned order = A.Order();
    const Int commRank = mpi::CommRank( mpi::COMM_WORLD );
    const Grid& g = A.Grid();

    TensorDistribution distB = resDist;

    DistTensor<T> B(A.Shape(), distB, g);
    B.AlignWith(A);
    B.ResizeTo(A);
    B.SetDistribution(distB);

    TensorDistribution lModeDist = A.TensorDist();

    Permutation defaultPerm(order);
    for(i = 0; i < order; i++)
        defaultPerm[i] = i;
    DistTensor<T> check(A.Shape(), distB, g);
    check.SetLocalPermutation(defaultPerm);
    check.ResizeLocalUnderPerm(defaultPerm);
    Set(check);

    if(commRank == 0){
        printf("Locally redistributing modes (");
        if(lModes.size() > 0)
            printf("%d", lModes[0]);
        for(i = 1; i < lModes.size(); i++)
            printf(", %d", lModes[i]);
        printf("): %s <-- %s\n", (tmen::TensorDistToString(B.TensorDist())).c_str(), (tmen::TensorDistToString(A.TensorDist())).c_str());
    }

//    Print(A, "A before local redist");
//    B.LocalRedistFrom(A, lModes, gridRedistModes);

    Permutation perm(order);
    for(i = 0; i < order; i++)
        perm[i] = i;

    do{
        B.SetLocalPermutation(perm);
        B.ResizeLocalUnderPerm(perm);
        B.LocalRedistFromWithPermutation(A, lModes, gridRedistModes);
        CheckResult(B, check);
    }while(next_permutation(perm.begin(), perm.end()));
}


template<typename T>
void
CreateLGTestsHelper(const DistTensor<T>& A, const ModeArray& lModes, Unsigned pos, const std::vector<std::vector<ModeArray>>& commGroups, const std::vector<ModeArray>& pieceComms, std::vector<LGTest>& tests){

//    printf("n: %d, p: %d\n", modesFrom.size(), pos);
    if(pos == lModes.size()){
//        printf("pushing\n");
        ModeArray testLModes = lModes;
        std::pair<ModeArray, std::vector<ModeArray> > t1(testLModes, pieceComms);
        TensorDistribution resDist = DetermineResultingDistributionLocalG(A, lModes, pieceComms);
        LGTest test(t1, resDist);
        tests.push_back(test);
//        printf("done\n");
    }else{
//        printf("recurring\n");
        Unsigned i;
        std::vector<ModeArray> modeCommGroups = commGroups[pos];
//        printf("modeCommGroups size: %d\n", modeCommGroups.size());

        for(i = 0; i < modeCommGroups.size(); i++){
//            printf("ping\n");
            std::vector<ModeArray> newPieceComm = pieceComms;
            newPieceComm[pos] = modeCommGroups[i];
            CreateLGTestsHelper(A, lModes, pos + 1, commGroups, newPieceComm, tests);
        }
//        printf("done recurring\n");
    }
}

template<typename T>
void
CreateLGCommGroupsHelper(const DistTensor<T>& A, const Unsigned& nlModes, Unsigned pos, const ModeArray& freeModes, std::vector<ModeArray>& piece, std::vector<std::vector<ModeArray>>& commGroups){

//    printf("n: %d, p: %d\n", modesFrom.size(), pos);
    if(pos == nlModes - 1){
        std::vector<ModeArray> newPiece = piece;
        piece.push_back(freeModes);
        commGroups.push_back(piece);
    }else{
//        printf("recurring\n");
        Unsigned i, j, k;
//        printf("modeCommGroups size: %d\n", modeCommGroups.size());

        for(i = 0; i <= freeModes.size(); i++){
            std::vector<ModeArray> lModeCombos = AllCombinations(freeModes, i);
            if(i == 0){
                ModeArray lModeCombo(0);
                std::vector<ModeArray> newPiece = piece;
                newPiece.push_back(lModeCombo);
                CreateLGCommGroupsHelper(A, nlModes, pos + 1, freeModes, newPiece, commGroups);
            }else{
                for(j = 0; j < lModeCombos.size(); j++){
                    ModeArray lModeCombo = lModeCombos[j];
                    std::vector<ModeArray> newPiece = piece;
                    newPiece.push_back(lModeCombo);
                    ModeArray newFreeModes = freeModes;
                    for(k = 0; k < lModeCombo.size(); k++){
                        newFreeModes.erase(std::find(newFreeModes.begin(), newFreeModes.end(), lModeCombo[k]));
                    }
                    CreateLGCommGroupsHelper(A, nlModes, pos + 1, newFreeModes, newPiece, commGroups);
                }
            }
        }
    }
}

template<typename T>
std::vector<LGTest>
CreateLGTests(const DistTensor<T>& A, const Params& args){
    Unsigned i, j;
    std::vector<LGTest > ret;

    const Unsigned order = A.Order();
    const TensorDistribution distA = A.TensorDist();
    ModeArray tensorModes(order);
    for(i = 0; i < tensorModes.size(); i++)
        tensorModes[i] = i;

    for(i = 1; i <= order; i++){
        std::vector<ModeArray> lModeCombos = AllCombinations(tensorModes, i);
        for(j = 0; j < lModeCombos.size(); j++){
            ModeArray lModes = lModeCombos[j];

            std::vector<std::vector<ModeArray> > commGroups(0);

            ModeArray freeModes = A.GetGridView().FreeModes();

            std::vector<ModeArray> piece;
            CreateLGCommGroupsHelper(A, lModes.size(), 0, freeModes, piece, commGroups);

            for(Unsigned k = 0; k < commGroups.size(); k++){
                ModeArray testLModes = lModes;
                std::pair<ModeArray, std::vector<ModeArray> > t1(testLModes, commGroups[k]);
                TensorDistribution resDist = DetermineResultingDistributionLocalG(A, lModes, commGroups[k]);
                LGTest test(t1, resDist);
                ret.push_back(test);
            }
        }
    }

//    AGTest test(1, DetermineResultingDistributionAG(A, 1));
//    ret.push_back(test);
    return ret;
}

#endif // ifndef TMEN_TESTS_LGREDIST_HPP
