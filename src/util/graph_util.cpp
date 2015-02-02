#include "tensormental/util/graph_util.hpp"
#include "tensormental.hpp"
#include "tensormental/io/Print.hpp"

namespace tmen{

//Modification of Tarjan's algorithm for determining strongly connected components.
//https://en.wikipedia.org/wiki/Tarjan%27s_strongly_connected_components_algorithm
//NOTE: std::vectors abused here.  The stack variables should really be stacks...
void StrongConnect(Unsigned& minIndex, TarjanVertex& v, std::vector<TarjanVertex>& S,
                   std::map<Mode, TarjanVertex>& mode2TarjanVertexMap, const ModeArray& commModes, const std::vector<std::pair<Mode, Mode> >& tensorModeFromTo,
                   ModeArray& p2pModes){
    int commRank = mpi::CommRank(MPI_COMM_WORLD);
    Unsigned i, j, k;
    // Set the depth index for v to the smallest unused index
    v.index = minIndex;
    v.lowlink = minIndex;
    mode2TarjanVertexMap[v.id] = v;
    minIndex += 1;
    S.push_back(v);

    //Consider successors of v
    TarjanVertex w;
//    printf("looking for successor of v.id: %d\n", v.id);
    for(i = 0; i < tensorModeFromTo.size(); i++){
        Mode tensorModeFrom = tensorModeFromTo[i].first;
        Mode tensorModeTo = tensorModeFromTo[i].second;
        if(tensorModeFrom == v.id){
//            printf("found an edge: %d --> %d\n", tensorModeFrom, tensorModeTo);
            std::map<Mode, TarjanVertex>::iterator it;
//            printf("mode2vertexMap:\n");
//            for(it = mode2TarjanVertexMap.begin(); it != mode2TarjanVertexMap.end(); it++){
//                printf("mode: %d maps to {id: %d, index: %d, lowlink: %d}\n", it->first, (it->second).id, (it->second).index, (it->second).lowlink);
//            }
//            printf("\n");
            if(mode2TarjanVertexMap.find(tensorModeTo) != mode2TarjanVertexMap.end()){
                w = mode2TarjanVertexMap[tensorModeTo];
//                printf("retrieved vertex from map: {id: %d, index: %d, lowlink: %d}\n", w.id, w.index, w.lowlink);
                if(w.index == -1){
//                    printf("recurring on v.id = %d and parent id: %d\n", w.id, v.id);
                    StrongConnect(minIndex, w, S,
                                  mode2TarjanVertexMap, commModes, tensorModeFromTo,
                                  p2pModes);
                    v.lowlink = Min(v.lowlink, w.lowlink);
                    mode2TarjanVertexMap[v.id] = v;
                }else{
//                    printf("searching for w.id: %d through stack of size: %d\n", w.id, S.size());
                    for(j = 0; j < S.size(); j++){
                        if(S[j].id == w.id){
//                            printf("found w: %d already in S\n", w.id);
                            v.lowlink = Min(v.lowlink, w.index);
                            mode2TarjanVertexMap[v.id] = v;
                        }
                    }
                }
            }
        }
    }

    // If v is a root node, pop the stack and generate an SCC
    if(v.lowlink == v.index){
//        printf("found root node with id: %d\n", v.id);

//        printf("Stack contains:");
//        for(i = 0; i < S.size(); i++)
//            printf(" %d", S[i].id);
//        printf("\n");

        //Create SCC
        std::vector<TarjanVertex> sccVertices;
        do{
            if(S.size() > 0){
                w = S[S.size() - 1];
                sccVertices.push_back(w);
                S.pop_back();
            }
        }while(w.id != v.id);
        //output SCC

//        printf("modes in SCC:");
//        for(i = 0; i < sccVertices.size(); i++)
//            printf(" %d", sccVertices[i].id);
//        printf("\n");

        if(sccVertices.size() > 1){
            Mode edgeSrc = sccVertices[sccVertices.size() - 1].id;
            Mode edgeDst = sccVertices[0].id;
            for(i = 0; i < tensorModeFromTo.size(); i++){
                std::pair<Mode, Mode> possEdge = tensorModeFromTo[i];
                if(possEdge.first == edgeSrc && possEdge.second == edgeDst){
                    p2pModes.push_back(commModes[i]);
                }
            }
            for(i = 1; i < sccVertices.size(); i++){
                edgeSrc = sccVertices[i-1].id;
                edgeDst = sccVertices[i].id;
                for(j = 0; j < tensorModeFromTo.size(); j++){
                    std::pair<Mode, Mode> possEdge = tensorModeFromTo[j];
                    if(possEdge.first == edgeSrc && possEdge.second == edgeDst){
                        p2pModes.push_back(commModes[j]);
                    }
                }
            }
        }
//        PrintVector(p2pModes, "p2pModes after found SCC");
    }
//    printf("returning\n");
}

//Modification of Tarjan's algorithm for determining strongly connected components.
//https://en.wikipedia.org/wiki/Tarjan%27s_strongly_connected_components_algorithm
//NOTE: std::vectors are abused here.  The stack variables should really be stacks...
void
DetermineSCC(const ModeArray& commModes, const std::vector<std::pair<Mode, Mode> >& tensorModeFromTo, ModeArray& p2pModes){
    Unsigned i;
    Unsigned minIndex = 0;
    std::vector<TarjanVertex > S;

    int commRank = mpi::CommRank(MPI_COMM_WORLD);
    //Create the list of tensor mode start points (Tarjan vertices)
    std::vector<Mode> vertices;
    std::vector<TarjanVertex > tarjanVertices;
    std::map<Mode, TarjanVertex> mode2TarjanVertexMap;
    for(i = 0; i < tensorModeFromTo.size(); i++){
        Mode possVertex = tensorModeFromTo[i].first;
        if(possVertex != -1){
            if(std::find(vertices.begin(), vertices.end(), possVertex) == vertices.end()){
                vertices.push_back(possVertex);

                TarjanVertex v;
                v.id = possVertex;
                v.index = -1;
                v.lowlink = -1;
                tarjanVertices.push_back(v);
                mode2TarjanVertexMap[possVertex] = v;
            }
        }
    }

    TarjanVertex root;
    root.id = -1;
    root.index = -1;
    root.lowlink = -1;

    if(commRank == 0){
//        printf("Created %d tarjan vertices\n", tarjanVertices.size());
        for(i = 0; i < tarjanVertices.size(); i++){
            TarjanVertex v = tarjanVertices[i];
//            printf("v.id: %d, v.index: %d, v.lowlink: %d\n", v.id, v.index, v.lowlink);
        }
    }

    //Perform Tarjan's algorithm
    for(i = 0; i < tarjanVertices.size(); i++){
        TarjanVertex v = tarjanVertices[i];
        if(v.index == -1){
            std::vector<std::pair<Mode, Mode> > ES;

//            printf("Running SCC on vertex id: %d\n", v.id);
            StrongConnect(minIndex, v, S,
                          mode2TarjanVertexMap, commModes, tensorModeFromTo,
                          p2pModes);
        }
    }
}

TensorDistribution
CreatePrefixDistribution(const TensorDistribution& inDist, const TensorDistribution& outDist){
    TensorDistribution ret;
    Unsigned i, j;
    Unsigned order = inDist.size();
    for(i = 0; i < order; i++){
        ModeDistribution modeDistA = inDist[i];
        ModeDistribution modeDistB = outDist[i];
        ModeDistribution modeDistBase;
        for(j = 0; j < modeDistA.size() && j < modeDistB.size(); j++){
            if(modeDistA[j] == modeDistB[j]){
                modeDistBase.push_back(modeDistA[j]);
            }else{
                break;
            }
        }
        ret.push_back(modeDistBase);
    }
    return ret;
}

TensorDistribution
CreatePrefixA2ADistribution(const TensorDistribution& prefixDist, const TensorDistribution& inDist, const ModeArray& a2aModes){
    Unsigned i, j;
    TensorDistribution ret = prefixDist;
    for(i = 0; i < a2aModes.size(); i++){
        Mode a2aMode = a2aModes[i];
        for(j = 0; j < inDist.size(); j++){
            ModeDistribution modeDist = inDist[j];
            if(std::find(modeDist.begin(), modeDist.end(), a2aMode) != modeDist.end()){
                ret[j].push_back(a2aMode);
                break;
            }
        }
    }
    return ret;
}

TensorDistribution
CreateA2AOptDist1(const TensorDistribution& prefixA2ADist, const TensorDistribution& inDist, const ModeArray& p2pModes){
    Unsigned i, j;
    TensorDistribution ret = prefixA2ADist;
    for(i = 0; i < p2pModes.size(); i++){
        Mode p2pMode = p2pModes[i];
        for(j = 0; j < prefixA2ADist.size(); j++){
            ModeDistribution modeDist = inDist[j];
            if(std::find(modeDist.begin(), modeDist.end(), p2pMode) != modeDist.end()){
                ret[j].push_back(p2pMode);
                break;
            }
        }
    }

    return ret;
}

TensorDistribution
CreateA2AOptDist2(const TensorDistribution& prefixA2ADist, const TensorDistribution& outDist, const ModeArray& p2pModes){
    Unsigned i, j;
    TensorDistribution ret = prefixA2ADist;

    for(i = 0; i < p2pModes.size(); i++){
        Mode p2pMode = p2pModes[i];
        for(j = 0; j < outDist.size(); j++){
            ModeDistribution modeDist = outDist[j];
            if(std::find(modeDist.begin(), modeDist.end(), p2pMode) != modeDist.end()){
                ret[j].push_back(p2pMode);
                break;
            }
        }
    }

    return ret;
}

TensorDistribution
CreatePrefixP2PDistribution(const TensorDistribution& prefixDist, const TensorDistribution& outDist, const ModeArray& p2pModes){
    Unsigned i, j;
    TensorDistribution ret = prefixDist;
    for(i = 0; i < p2pModes.size(); i++){
        Mode p2pMode = p2pModes[i];
        for(j = 0; j < outDist.size(); j++){
            ModeDistribution modeDist = outDist[j];
            if(std::find(modeDist.begin(), modeDist.end(), p2pMode) != modeDist.end()){
                ret[j].push_back(p2pMode);
                break;
            }
        }
    }
    return ret;
}

TensorDistribution
CreateA2AOptDist3(const TensorDistribution& prefixP2PDist, const TensorDistribution& inDist, const ModeArray& a2aModes){
    Unsigned i, j;
    TensorDistribution ret = prefixP2PDist;

    for(i = 0; i < a2aModes.size(); i++){
        Mode a2aMode = a2aModes[i];
        for(j = 0; j < inDist.size(); j++){
            ModeDistribution modeDist = inDist[j];
            if(std::find(modeDist.begin(), modeDist.end(), a2aMode) != modeDist.end()){
                ret[j].push_back(a2aMode);
                break;
            }
        }
    }

    return ret;
}

TensorDistribution
CreateA2AOptDist4(const TensorDistribution& prefixP2PDist, const TensorDistribution& outDist, const ModeArray& a2aModes){
    Unsigned i, j;
    TensorDistribution ret = prefixP2PDist;

    for(i = 0; i < a2aModes.size(); i++){
        Mode a2aMode = a2aModes[i];
        for(j = 0; j < outDist.size(); j++){
            ModeDistribution modeDist = outDist[j];
            if(std::find(modeDist.begin(), modeDist.end(), a2aMode) != modeDist.end()){
                ret[j].push_back(a2aMode);
                break;
            }
        }
    }

    return ret;
}

}
