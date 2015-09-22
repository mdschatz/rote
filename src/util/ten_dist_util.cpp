
#include "tensormental/util/vec_util.hpp"
#include "tensormental/util/ten_dist_util.hpp"
#include "tensormental/core/error_decl.hpp"
#include <algorithm>
#include <numeric>
#include <functional>

namespace tmen{

bool CheckOrder(const Unsigned& outOrder, const Unsigned& inOrder){
	if(outOrder != inOrder){
        LogicError("Invalid redistribution: Objects being redistributed must be of same order");
    }
	return true;
}

bool CheckSameNonDist(const TensorDistribution& outDist, const TensorDistribution& inDist){
	Unsigned nonDistMode = outDist.size() - 1;
	if(outDist[nonDistMode].size() != inDist[nonDistMode].size() || !EqualUnderPermutation(outDist[nonDistMode], inDist[nonDistMode])){
    	std::stringstream msg;
		msg << "Invalid redistribution\n"
			<< tmen::TensorDistToString(outDist)
			<< " <-- "
			<< tmen::TensorDistToString(inDist)
			<< std::endl
			<< "Output Non-distributed mode distribution must be same"
			<< std::endl;
		LogicError(msg.str());
    }
	return true;
}

bool CheckSameCommModes(const TensorDistribution& outDist, const TensorDistribution& inDist){
	Unsigned i;
	Unsigned order = outDist.size();
    ModeDistribution commModesOut;
    ModeDistribution commModesIn;

    for(i = 0; i < order; i++){
		if(IsPrefix(inDist[i], outDist[i])){
			commModesOut = ConcatenateVectors(commModesOut, GetSuffix(inDist[i], outDist[i]));
		}
		else if(IsPrefix(outDist[i], inDist[i])){
			commModesIn = ConcatenateVectors(commModesIn, GetSuffix(outDist[i], inDist[i]));
		}
    }

    if(commModesOut.size() != commModesIn.size() || !EqualUnderPermutation(commModesOut, commModesIn)){
    	std::stringstream msg;
    	msg << "Invalid redistribution\n"
			<< tmen::TensorDistToString(outDist)
			<< " <-- "
			<< tmen::TensorDistToString(inDist)
			<< std::endl
			<< "Cannot determine modes communicated over"
			<< std::endl;
    	LogicError(msg.str());
    }
	return true;
}

bool CheckPartition(const TensorDistribution& outDist, const TensorDistribution& inDist){
	Unsigned i;
	Unsigned j = 0;
	Unsigned objOrder = outDist.size() - 1;
	ModeArray sourceModes;
	ModeArray sinkModes;

    for(i = 0; i < objOrder; i++){
    	if(inDist[i].size() != outDist[i].size()){
			if(IsPrefix(inDist[i], outDist[i])){
				sinkModes.push_back(i);
			}
			else if(IsPrefix(outDist[i], inDist[i])){
				sourceModes.push_back(i);
			}
    	}else if(!IsSame(inDist[i], outDist[i])){
			std::stringstream msg;
			msg << "Invalid redistribution\n"
				<< tmen::TensorDistToString(outDist)
				<< " <-- "
				<< tmen::TensorDistToString(inDist)
				<< std::endl
				<< "Cannot form partition"
				<< std::endl;
			LogicError(msg.str());
		}
    }

    for(i = 0; i < sourceModes.size() && j < sinkModes.size(); i++){
    	if(sourceModes[i] == sinkModes[j]){
        	std::stringstream msg;
    		msg << "Invalid redistribution\n"
    			<< tmen::TensorDistToString(outDist)
    			<< " <-- "
    			<< tmen::TensorDistToString(inDist)
    			<< std::endl
    			<< "Cannot form partition"
    			<< std::endl;
    		LogicError(msg.str());
    	}else if(sourceModes[i] > sinkModes[j]){
    		i--;
    		j++;
    	}else if(sourceModes[i] < sinkModes[j]){
    		continue;
    	}
    }

	return true;
}

bool CheckInIsPrefix(const TensorDistribution& outDist, const TensorDistribution& inDist){
	Unsigned objOrder = outDist.size() - 1;

    for(Unsigned i = 0; i < objOrder; i++){
    	if(!(IsPrefix(inDist[i], outDist[i]))){
    		std::stringstream msg;
    		msg << "Invalid redistribution:\n"
    		    << tmen::TensorDistToString(outDist)
    		    << " <-- "
    		    << tmen::TensorDistToString(inDist)
    		    << std::endl
    		    << "Input mode-" << i << " mode distribution must be prefix of output mode distribution"
    			<< std::endl;
    		LogicError(msg.str());
    	}
    }
    return true;
}

bool CheckOutIsPrefix(const TensorDistribution& outDist, const TensorDistribution& inDist){
	Unsigned objOrder = outDist.size() - 1;

    for(Unsigned i = 0; i < objOrder; i++){
    	if(!(IsPrefix(outDist[i], inDist[i]))){
    		std::stringstream msg;
    		msg << "Invalid redistribution\n"
    		    << tmen::TensorDistToString(outDist)
    		    << " <-- "
    		    << tmen::TensorDistToString(inDist)
    		    << std::endl
    		    << "Output mode-" << i << " mode distribution must be prefix of input mode distribution"
    			<< std::endl;
    		LogicError(msg.str());
    	}
    }
    return true;
}

bool CheckSameGridViewShape(const ObjShape& outShape, const ObjShape& inShape){
    if(AnyElemwiseNotEqual(outShape, inShape)){
		std::stringstream msg;
		msg << "Invalid redistribution: Mode distributions must correspond to equal logical grid dimensions"
			<< std::endl;
		LogicError(msg.str());
    }
    return true;
}

}
