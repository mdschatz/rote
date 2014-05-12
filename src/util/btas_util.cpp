#include "tensormental/util/btas_util.hpp"


namespace tmen{

template<typename T>
std::vector<IndexArray> DetermineContractIndices(const Tensor<T>& A, const Tensor<T>& B, const Tensor<T>& C){
    Unsigned i;
    std::vector<IndexArray> ret;

    //Arrays which will hold the indices corresponding to m,k,n of matrix indices
    IndexArray arrM;
    IndexArray arrK;
    IndexArray arrN;

    //Indices
    IndexArray indicesA = A.Indices();
    IndexArray indicesB = B.Indices();
    IndexArray indicesC = C.Indices();

    //Get the M,N indices
    for(i = 0; i < indicesC.size(); i++){
        Index index = indicesC[i];
        //index is M index
        if(std::find(indicesA.begin(), indicesA.end(), index) != indicesA.end())
            arrM.push_back(index);
        //index is N index
        else if(std::find(indicesB.begin(), indicesB.end(), index) != indicesB.end())
            arrN.push_back(index);
    }
    for(i = 0; i < indicesA.size(); i++){
        Index index = indicesA[i];
        //index is K index
        if(std::find(indicesB.begin(), indicesB.end(), index) != indicesB.end())
            arrK.push_back(index);
    }
    ret.push_back(arrM);
    ret.push_back(arrK);
    ret.push_back(arrN);
    return ret;
}

#define PROTO(T) \
    template std::vector<IndexArray> DetermineContractIndices(const Tensor<T>& A, const Tensor<T>& B, const Tensor<T>& C);

PROTO(int)
PROTO(float)
PROTO(double)
}
