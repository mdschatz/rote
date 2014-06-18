#include "tensormental/util/btas_util.hpp"


namespace tmen{

//TODO: Generalize so that the largest is not permuted
template<typename T>
std::vector<ModeArray> DetermineContractModes(const Tensor<T>& A, const Tensor<T>& B, const Tensor<T>& C, const std::vector<IndexArray>& indices){
    Unsigned i, j;
    std::vector<ModeArray> ret(3);

    //Arrays which will hold the indices corresponding to m,k,n of matrix indices
    ModeArray arrM;
    ModeArray arrK;
    ModeArray arrN;

    //Indices
    IndexArray indicesA = indices[0];
    IndexArray indicesB = indices[1];
    IndexArray indicesC = indices[2];


    //Get the M modes
    for(i = 0; i < indicesC.size(); i++){
        Index index = indicesC[i];
        ret[2].push_back(i);
        for(j = 0; j < indicesA.size(); j++){
            if(index == indicesA[j])
                ret[0].push_back(j);
        }
    }
    //Get the K modes
    for(i = 0; i < indicesA.size(); i++){
        Index index = indicesA[i];
        for(j = 0; j < indicesB.size(); j++){
            if(index == indicesB[j]){
                ret[0].push_back(i);
                ret[1].push_back(j);
            }
        }
    }
    //Get the N modes
    for(i = 0; i < indicesC.size(); i++){
        Index index = indicesC[i];
        ret[2].push_back(i);
        for(j = 0; j < indicesB.size(); j++){
            if(index == indicesB[j])
                ret[1].push_back(j);
        }
    }
    return ret;
}

#define PROTO(T) \
    template std::vector<ModeArray> DetermineContractModes(const Tensor<T>& A, const Tensor<T>& B, const Tensor<T>& C, const std::vector<IndexArray>& indices);

PROTO(int)
PROTO(float)
PROTO(double)
}
