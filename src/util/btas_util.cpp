#include "tensormental/util/btas_util.hpp"


namespace tmen{

std::vector<ModeArray> DetermineContractModes(const IndexArray& indicesA, const IndexArray& indicesB, const IndexArray& indicesC){
    Unsigned i;
    std::vector<ModeArray> ret(3);

    //Arrays which will hold the indices corresponding to m,k,n of matrix indices in contraction order
    IndexArray indM;
    IndexArray indN;
    IndexArray indK;

    for(i = 0; i < indicesA.size(); i++){
        Index index = indicesA[i];
        if(!Contains(indicesB, index)){
            indM.push_back(index);
        }else{
            indK.push_back(index);
        }
    }

    for(i = 0; i < indicesB.size(); i++){
        Index index = indicesB[i];
        if(!Contains(indicesA, index)){
            indN.push_back(index);
        }
    }

    //Convert the M,N,K indices to the modes in each tensor
    for(i = 0; i < indM.size(); i++){
        ret[0].push_back((Mode)(IndexOf(indicesA, indM[i])));
        ret[2].push_back((Mode)(IndexOf(indicesC, indM[i])));
    }
    for(i = 0; i < indK.size(); i++){
        ret[0].push_back((Mode)(IndexOf(indicesA, indK[i])));
        ret[1].push_back((Mode)(IndexOf(indicesB, indK[i])));
    }
    for(i = 0; i < indN.size(); i++){
        ret[1].push_back((Mode)(IndexOf(indicesB, indN[i])));
        ret[2].push_back((Mode)(IndexOf(indicesC, indN[i])));
    }
    for(i = 0; i < indK.size(); i++){
        ret[2].push_back((Mode)(IndexOf(indicesC, indK[i])));
    }

    return ret;
}

IndexArray DetermineContractIndices(const IndexArray& indicesA, const IndexArray& indicesB){
    Unsigned i;
    IndexArray contractIndices;
    for(i = 0; i < indicesA.size(); i++)
        if(Contains(indicesB, indicesA[i]))
            contractIndices.push_back(indicesA[i]);
    SortVector(contractIndices);
    return contractIndices;
}
}
