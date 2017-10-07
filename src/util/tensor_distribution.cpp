#include "rote.hpp"

namespace rote{

bool CheckOrder(const Unsigned& outOrder, const Unsigned& inOrder){
	if(outOrder != inOrder){
        LogicError("Invalid redistribution: Objects being redistributed must be of same order");
    }
	return true;
}

bool CheckNonDistOutIsPrefix(const TensorDistribution& outDist, const TensorDistribution& inDist){
	Unsigned nonDistMode = outDist.size() - 1;
	if(!(outDist[nonDistMode] <= inDist[nonDistMode])){
    	std::stringstream msg;
		msg << "Invalid redistribution\n"
			<< outDist
			<< " <-- "
			<< inDist
			<< std::endl
			<< "Output Non-distributed mode distribution must be prefix"
			<< std::endl;
		LogicError(msg.str());
    }
	return true;
}

bool CheckNonDistInIsPrefix(const TensorDistribution& outDist, const TensorDistribution& inDist){
	Unsigned nonDistMode = outDist.size() - 1;
	if(!(inDist[nonDistMode] <= outDist[nonDistMode])){
    	std::stringstream msg;
		msg << "Invalid redistribution\n"
			<< outDist
			<< " <-- "
			<< inDist
			<< std::endl
			<< "Input Non-distributed mode distribution must be prefix"
			<< std::endl;
		LogicError(msg.str());
    }
	return true;
}

bool CheckSameNonDist(const TensorDistribution& outDist, const TensorDistribution& inDist){
	Unsigned nonDistMode = outDist.size() - 1;
	if(!outDist[nonDistMode].SameModesAs(inDist[nonDistMode])){
    	std::stringstream msg;
		msg << "Invalid redistribution\n"
			<< outDist
			<< " <-- "
			<< inDist
			<< std::endl
			<< "Output Non-distributed mode distribution must be same"
			<< std::endl;
		LogicError(msg.str());
    }
	return true;
}

bool CheckSameCommModes(const TensorDistribution& outDist, const TensorDistribution& inDist){
    TensorDistribution commonPrefix = GetCommonPrefix(inDist, outDist);
    TensorDistribution remainderIn = inDist - commonPrefix;
    TensorDistribution remainderOut = outDist - commonPrefix;

    if(!remainderIn.UsedModes().SameModesAs(remainderOut.UsedModes())){
    	std::stringstream msg;
    	msg << "Invalid redistribution\n"
			<< outDist
			<< " <-- "
			<< inDist
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
			if(inDist[i] <= outDist[i]){
				sinkModes.push_back(i);
			}
			else if(outDist[i] <= inDist[i]){
				sourceModes.push_back(i);
			}
    	}else if(inDist[i] != outDist[i]){
			std::stringstream msg;
			msg << "Invalid redistribution\n"
				<< outDist
				<< " <-- "
				<< inDist
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
    			<< outDist
    			<< " <-- "
    			<< inDist
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
    	if(!(inDist[i] <= outDist[i])){
    		std::stringstream msg;
    		msg << "Invalid redistribution:\n"
    		    << outDist
    		    << " <-- "
    		    << inDist
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
    	if(!(outDist[i] <= inDist[i])){
    		std::stringstream msg;
    		msg << "Invalid redistribution\n"
    		    << outDist
    		    << " <-- "
    		    << inDist
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

bool CheckIsValidPermutation(const Unsigned& order, const Permutation& perm){
	return perm.size() == order;
}

//TODO: Error checking
ModeArray GetModeDistOfGridMode(const ModeArray& gridModes, const TensorDistribution& tenDist){
	Unsigned i;
	ModeArray ret(gridModes.size());
	for(i = 0; i < gridModes.size(); i++)
		ret[i] = GetModeDistOfGridMode(gridModes[i], tenDist);
	return ret;
}

Mode GetModeDistOfGridMode(const Mode& mode, const TensorDistribution& tenDist){
	Unsigned i;
	for(i = 0; i < tenDist.size(); i++){
		ModeDistribution modeDist = tenDist[i];
		if(modeDist.Contains(mode))
			return i;
	}
	return tenDist.size();
}

}
