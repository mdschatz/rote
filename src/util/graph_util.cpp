#include "tensormental/util/graph_util.hpp"


namespace tmen{

//Modification of Tarjan's algorithm for determining strongly connected components.
//https://en.wikipedia.org/wiki/Tarjan%27s_strongly_connected_components_algorithm
//NOTE: std::vectors abused here.  The stack variables should really be stacks...
void StrongConnect(const Unsigned minIndex, TarjanVertex& v, std::vector<TarjanVertex>& S,
                   const TarjanVertex& parent, std::vector<std::pair<Mode, Mode> >& ES,
                   std::map<Mode, TarjanVertex>& mode2TarjanVertexMap, const ModeArray& commModes, const std::vector<std::pair<Mode, Mode> >& tensorModeFromTo,
                   ModeArray& p2pModes){
    Unsigned i;
    // Set the depth index for v to the smallest unused index
    v.index = minIndex;
    v.lowlink = minIndex;
    mode2TarjanVertexMap[v.id] = v;
    S.push_back(v);
    if(parent.id != -1){
        //Find the edge and add it to ES
        for(i = 0; i < tensorModeFromTo.size(); i++){
            std::pair<Mode, Mode> possEdge = tensorModeFromTo[i];
            if(possEdge.first == parent.id && possEdge.second == v.id){
                ES.push_back(possEdge);
                break;
            }
        }
    }

    //Consider successors of v
    TarjanVertex w;
    for(i = 0; i < tensorModeFromTo.size(); i++){
        Mode tensorModeFrom = tensorModeFromTo[i].first;
        Mode tensorModeTo = tensorModeFromTo[i].second;
        if(tensorModeFrom == v.id){
            w = mode2TarjanVertexMap[tensorModeTo];
            if(w.index == -1){
                StrongConnect(minIndex + 1, w, S,
                              v, ES,
                              mode2TarjanVertexMap, commModes, tensorModeFromTo,
                              p2pModes);
                v.lowlink = Min(v.lowlink, w.lowlink);
            }else{
                for(i = 0; i < S.size(); i++){
                    if(S[i].id == w.id)
                        v.lowlink = Min(v.lowlink, w.index);
                }
            }
            mode2TarjanVertexMap[v.id] = v;
        }
    }

    // If v is a root node, pop the stack and generate an SCC
    if(v.lowlink == v.index){
        std::vector<TarjanVertex> sccVertices;
        std::vector<std::pair<Mode, Mode> > sccEdges;
        do{
            w = S[S.size() - 1];
            sccVertices.push_back(w);
            S.pop_back();
            std::pair<Mode, Mode> edge = ES[ES.size() - 1];
            sccEdges.push_back(edge);
            ES.pop_back();
        }while(w.id != v.id);
        //output
        while(sccEdges.size() != 0){
            std::pair<Mode, Mode> sccEdge = sccEdges[sccEdges.size() - 1];
            for(i = 0; i < tensorModeFromTo.size(); i++){
                std::pair<Mode, Mode> possEdge = tensorModeFromTo[i];
                if(possEdge.first == sccEdge.first && possEdge.second == sccEdge.second){
                    p2pModes.push_back(commModes[i]);
                }
            }
            sccEdges.pop_back();
        }
    }
}

//Modification of Tarjan's algorithm for determining strongly connected components.
//https://en.wikipedia.org/wiki/Tarjan%27s_strongly_connected_components_algorithm
//NOTE: std::vectors are abused here.  The stack variables should really be stacks...
void
DetermineSCC(const ModeArray& commModes, const std::vector<std::pair<Mode, Mode> >& tensorModeFromTo, ModeArray& p2pModes){
    Unsigned i;
    Unsigned minIndex = 0;
    std::vector<TarjanVertex > S;

    //Create the list of tensor mode start points (Tarjan vertices)
    std::vector<Mode> vertices;
    std::vector<TarjanVertex > tarjanVertices;
    std::map<Mode, TarjanVertex> mode2TarjanVertexMap;
    for(i = 0; i < tensorModeFromTo.size(); i++){
        Mode possVertex = tensorModeFromTo[i].first;
        if(std::find(vertices.begin(), vertices.end(), possVertex) == vertices.end()){
            vertices.push_back(possVertex);

            TarjanVertex v;
            v.id = possVertex;
            v.index = -1;
            v.lowlink = -1;
            mode2TarjanVertexMap[possVertex] = v;
        }
    }

    TarjanVertex root;
    root.id = -1;
    root.index = -1;
    root.lowlink = -1;

    //Perform Tarjan's algorithm
    for(i = 0; i < tarjanVertices.size(); i++){
        TarjanVertex v = tarjanVertices[i];
        if(v.index == -1){
            std::vector<std::pair<Mode, Mode> > ES;
            StrongConnect(0, v, S,
                          root, ES,
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
            if(std::find(inDist.begin(), inDist.end(), a2aMode) != inDist.end()){
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
            if(std::find(inDist.begin(), inDist.end(), p2pMode) != inDist.end()){
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
            if(std::find(outDist.begin(), outDist.end(), p2pMode) != outDist.end()){
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
            if(std::find(outDist.begin(), outDist.end(), p2pMode) != outDist.end()){
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
            if(std::find(inDist.begin(), inDist.end(), a2aMode) != inDist.end()){
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
            if(std::find(outDist.begin(), outDist.end(), a2aMode) != outDist.end()){
                ret[j].push_back(a2aMode);
                break;
            }
        }
    }

    return ret;
}

}
