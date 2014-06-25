#include "tensormental/util/btas_util.hpp"


namespace tmen{

//TODO: Generalize so that the largest is not permuted
template<typename T>
std::vector<ModeArray> DetermineContractModes(const Tensor<T>& A, const Tensor<T>& B, const Tensor<T>& C, const std::vector<IndexArray>& indices){
    Unsigned i, j;
    std::vector<ModeArray> ret(3);

    //Arrays which will hold the indices corresponding to m,k,n of matrix indices in contraction order
    IndexArray indM;
    IndexArray indN;
    IndexArray indK;

    //Indices
    IndexArray indicesA = indices[0];
    IndexArray indicesB = indices[1];
    IndexArray indicesC = indices[2];

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

#define PROTO(T) \
    template std::vector<ModeArray> DetermineContractModes(const Tensor<T>& A, const Tensor<T>& B, const Tensor<T>& C, const std::vector<IndexArray>& indices);

PROTO(int)
PROTO(float)
PROTO(double)
}
