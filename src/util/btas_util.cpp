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
        if(std::find(indicesB.begin(), indicesB.end(), index) == indicesB.end()){
            indM.push_back(index);
        }else{
            indK.push_back(index);
        }
    }

    for(i = 0; i < indicesB.size(); i++){
        Index index = indicesB[i];
        if(std::find(indicesA.begin(), indicesA.end(), index) == indicesA.end()){
            indN.push_back(index);
        }
    }

    //Convert the M,N,K indices to the modes in each tensor
    for(i = 0; i < indM.size(); i++){
        ret[0].push_back((Mode)(std::find(indicesA.begin(), indicesA.end(), indM[i]) - indicesA.begin()));
        ret[2].push_back((Mode)(std::find(indicesC.begin(), indicesC.end(), indM[i]) - indicesC.begin()));
    }
    for(i = 0; i < indK.size(); i++){
        ret[0].push_back((Mode)(std::find(indicesA.begin(), indicesA.end(), indK[i]) - indicesA.begin()));
        ret[1].push_back((Mode)(std::find(indicesB.begin(), indicesB.end(), indK[i]) - indicesB.begin()));
    }
    for(i = 0; i < indN.size(); i++){
        ret[1].push_back((Mode)(std::find(indicesB.begin(), indicesB.end(), indN[i]) - indicesB.begin()));
        ret[2].push_back((Mode)(std::find(indicesC.begin(), indicesC.end(), indN[i]) - indicesC.begin()));
    }
    for(i = 0; i < indK.size(); i++){
        ret[2].push_back((Mode)(std::find(indicesC.begin(), indicesC.end(), indK[i]) - indicesC.begin()));
    }

    return ret;
}

IndexArray DetermineContractIndices(const IndexArray& indicesA, const IndexArray& indicesB){
    Unsigned i;
    IndexArray contractIndices;
    for(i = 0; i < indicesA.size(); i++)
        if(std::find(indicesB.begin(), indicesB.end(), indicesA[i]) != indicesB.end())
            contractIndices.push_back(indicesA[i]);
    std::sort(contractIndices.begin(), contractIndices.end());
    return contractIndices;
}
}
