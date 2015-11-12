#pragma once
#ifndef TMEN_TESTS_ALLREDISTS_HPP
#define TMEN_TESTS_ALLREDISTS_HPP

using namespace rote;

typedef struct Arguments{
  Unsigned gridOrder;
  Unsigned tenOrder;
  Unsigned nProcs;
  ObjShape gridShape;
  ObjShape tensorShape;
  TensorDistribution tensorDist;
} Params;

typedef std::pair< TensorDistribution, ModeArray> RedistTest;

template<typename T>
void
Set(Tensor<T>& A)
{
    Unsigned order = A.Order();
    Location loc(order, 0);

    Unsigned ptr = 0;
    Unsigned counter = 0;
    bool stop = !ElemwiseLessThan(loc, A.Shape());

    while(!stop){
        A.Set(loc, counter);

        //Update
        counter++;
        if(order == 0)
            break;
        loc[ptr]++;
        while(loc[ptr] == A.Dimension(ptr)){
            loc[ptr] = 0;
            ptr++;
            if(ptr == order){
                stop = true;
                break;
            }else{
                loc[ptr]++;
            }
        }
        ptr = 0;
    }
}

template<typename T>
void
SetWith(DistTensor<T>& A, const T* buf)
{
    Unsigned order = A.Order();
    Location loc(order, 0);

    Unsigned ptr = 0;
    Unsigned counter = 0;
    bool stop = !ElemwiseLessThan(loc, A.Shape());

    while(!stop){
        A.Set(loc, buf[counter]);

        //Update
        counter++;
        if(order == 0)
            break;
        loc[ptr]++;
        while(loc[ptr] == A.Dimension(ptr)){
            loc[ptr] = 0;
            ptr++;
            if(ptr == order){
                stop = true;
                break;
            }else{
                loc[ptr]++;
            }
        }
        ptr = 0;
    }
}

template<typename T>
void
Set(DistTensor<T>& A)
{
    Unsigned order = A.Order();
    Location loc(order, 0);

    Unsigned ptr = 0;
    Unsigned counter = 0;
    bool stop = !ElemwiseLessThan(loc, A.Shape());

    while(!stop){
        A.Set(loc, counter);

        //Update
        counter++;
        if(order == 0)
            break;
        loc[ptr]++;
        while(loc[ptr] == A.Dimension(ptr)){
            loc[ptr] = 0;
            ptr++;
            if(ptr == order){
                stop = true;
                break;
            }else{
                loc[ptr]++;
            }
        }
        ptr = 0;
    }
}

template<typename T>
bool CheckResult(const DistTensor<T>& A){
#ifndef RELEASE
    CallStackEntry entry("CheckResult");
#endif
    mpi::Barrier(mpi::COMM_WORLD);
    const Int commRank = mpi::CommRank( mpi::COMM_WORLD );

    const Unsigned order = A.Order();
    const std::vector<Unsigned> myFirstElem = A.ModeShifts();
    const Location myGridLoc = A.Grid().Loc();
    const TensorDistribution dist = A.TensorDist();

	DistTensor<T> check(A.Shape(), A.TensorDist(), A.Grid());
	check.SetLocalPermutation(A.LocalPermutation());
	Set(check);

    //Check that all entries are what they should be
    const std::vector<Unsigned> strides = A.ModeStrides();
    Unsigned res = 1;

    bool shouldbeParticipating = !AnyPositiveElem(FilterVector(myGridLoc, dist[order]));
//    printf("shouldBeParticipating: %s\n", (shouldbeParticipating ? "true" : "false"));
    if(shouldbeParticipating && !A.Participating())
        res = 1;

    if(!shouldbeParticipating){
        if(A.AllocatedMemory() > 1)
            res = 1;
    }else{
    	const Tensor<T> tensorA = A.LockedTensor();
    	const Tensor<T> tensorActual = check.LockedTensor();
    	const Permutation permCheckToA = DeterminePermutation(check.LocalPermutation(), A.LocalPermutation());
    	const ObjShape checkShape = check.Shape();
        Location checkLoc(order, 0);

        Unsigned modePtr = 0;
        bool stop = !ElemwiseLessThan(checkLoc, checkShape);

        if(stop && A.AllocatedMemory() > 1){
            res = 0;
        }

//        Print(actual, "actual");
//        printf("dist: %s\n", tmen::TensorDistToString(A.TensorDist()).c_str());
//        PrintVector(strides, "strides");
//        PrintVector(myGridLoc, "myGridLoc");
//        PrintVector(myFirstElem, "myFirstLoc");
        while(!stop){
//            PrintVector(checkLoc, "checking Loc");
            Location locA = PermuteVector(checkLoc, permCheckToA);

            if(A.Get(locA) != check.Get(checkLoc)){
                printf("val: %d checkVal: %d\n", A.Get(locA), check.Get(checkLoc));
                res = 0;
                break;
            }

            if(order == 0)
                break;

            //Update
            checkLoc[modePtr]++;
            while(checkLoc[modePtr] >= checkShape[modePtr]){
                checkLoc[modePtr] = 0;
                modePtr++;
                if(modePtr == order){
                    stop = true;
                    break;
                }else{
                    checkLoc[modePtr]++;
                }
            }
            modePtr = 0;
        }
    }
    Unsigned recv;

    mpi::AllReduce(&res, &recv, 1, mpi::LOGICAL_AND, mpi::COMM_WORLD);
    if(recv == 0)
        LogicError("redist bug: some process not assigned correct data");

    if(commRank == 0){
        std::cout << "PASS" << std::endl;
        return true;
    }
    return false;
}

template<typename T>
bool CheckReduceResult(const DistTensor<T>& A, const DistTensor<T>& orig, const ModeArray& commModes){
#ifndef RELEASE
    CallStackEntry entry("CheckResult");
#endif
    mpi::Barrier(mpi::COMM_WORLD);
    const Int commRank = mpi::CommRank( mpi::COMM_WORLD );

    Unsigned i;
    const Unsigned order = A.Order();
    const std::vector<Unsigned> myFirstElem = A.ModeShifts();
    const Location myGridLoc = A.Grid().Loc();
    const TensorDistribution dist = A.TensorDist();

    Tensor<T> origTen(orig.Shape());
    Set(origTen);

    ObjShape origResShape = orig.Shape();
    for(i = 0; i < commModes.size(); i++)
    	origResShape[commModes[i]] = 1;

    Tensor<T> origRes(origResShape);
    Zero(origRes);

    LocalReduce(origTen, origRes, commModes);

	DistTensor<T> check(A.Shape(), A.TensorDist(), A.Grid());
	check.SetLocalPermutation(A.LocalPermutation());
	SetWith(check, origRes.LockedBuffer());

//	Print(check, "check");

    //Check that all entries are what they should be
    const std::vector<Unsigned> strides = A.ModeStrides();
    Unsigned res = 1;

    bool shouldbeParticipating = !AnyPositiveElem(FilterVector(myGridLoc, dist[order]));
//    printf("shouldBeParticipating: %s\n", (shouldbeParticipating ? "true" : "false"));
    if(shouldbeParticipating && !A.Participating())
        res = 1;

    if(!shouldbeParticipating){
        if(A.AllocatedMemory() > 1)
            res = 1;
    }else{
    	const Tensor<T> tensorA = A.LockedTensor();
    	const Tensor<T> tensorActual = check.LockedTensor();
    	const Permutation permCheckToA = DeterminePermutation(check.LocalPermutation(), A.LocalPermutation());
    	const ObjShape checkShape = check.Shape();
        Location checkLoc(order, 0);

        Unsigned modePtr = 0;
        bool stop = !ElemwiseLessThan(checkLoc, checkShape);

        if(stop && A.AllocatedMemory() > 1){
            res = 0;
        }

//        Print(actual, "actual");
//        printf("dist: %s\n", tmen::TensorDistToString(A.TensorDist()).c_str());
//        PrintVector(strides, "strides");
//        PrintVector(myGridLoc, "myGridLoc");
//        PrintVector(myFirstElem, "myFirstLoc");
        while(!stop){
//            PrintVector(checkLoc, "checking Loc");
            Location locA = PermuteVector(checkLoc, permCheckToA);

            if(A.Get(locA) != check.Get(checkLoc)){
                printf("val: %d checkVal: %d\n", A.Get(locA), check.Get(checkLoc));
                res = 0;
                break;
            }

            if(order == 0)
                break;

            //Update
            checkLoc[modePtr]++;
            while(checkLoc[modePtr] >= checkShape[modePtr]){
                checkLoc[modePtr] = 0;
                modePtr++;
                if(modePtr == order){
                    stop = true;
                    break;
                }else{
                    checkLoc[modePtr]++;
                }
            }
            modePtr = 0;
        }
    }
    Unsigned recv;

    mpi::AllReduce(&res, &recv, 1, mpi::LOGICAL_AND, mpi::COMM_WORLD);
    if(recv == 0)
        LogicError("redist bug: some process not assigned correct data");

    if(commRank == 0){
        std::cout << "PASS" << std::endl;
        return true;
    }
    return false;
}

void AllCombinationsHelper(const ModeArray& input, Unsigned arrPos, Unsigned k, Unsigned prevLoc, ModeArray& piece, std::vector<ModeArray>& combinations){
    Unsigned i;

    if(arrPos == k){
    	SortVector(piece);
        combinations.push_back(piece);
    }
    else{
        for(i = prevLoc+1; i < input.size(); i++){
            ModeArray newPiece = piece;
            newPiece[arrPos] = input[i];
            AllCombinationsHelper(input, arrPos+1, k, i, newPiece, combinations);
        }
    }
}

std::vector<ModeArray> AllCombinations(const ModeArray& input, Unsigned k){
    Unsigned i;
    ModeArray start(k);
    std::vector<ModeArray> ret;
    if(k == 0){
        ret.push_back(start);
        return ret;
    }

    for(i = 0; i < input.size(); i++){
        ModeArray newPiece(k);
        newPiece[0] = input[i];
        AllCombinationsHelper(input, 1, k, i, newPiece, ret);
    }

    return ret;
}


#endif // ifndef TMEN_TESTS_ALLREDISTS_HPP
