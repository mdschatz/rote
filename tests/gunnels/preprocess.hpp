#pragma once
#ifndef ROTE_TESTS_GUNNELS_PREPROCESS_HPP
#define ROTE_TESTS_GUNNELS_PREPROCESS_HPP

#include "rote.hpp"
using namespace rote;
using namespace std;

typedef struct Arguments {
    int nProcs;
    int exNum;
    int tensorDim;
    ObjShape gridShape;
    int blkSize;
} Params;

void Usage() {
    std::cout << "./example <ex_num> <tensor_dim> <grid_order> <grid_dim_1> ... <blk_size>\n";
    std::cout << "<ex_num>      : example number (1-3)\n";
    std::cout << "<tensor_dim>  : tensor dimension\n";
    std::cout << "<grid_order>  : virtual grid order (number of dimensions)\n";
    std::cout << "<grid_dim_k>  : dimension of grid mode k\n";
    std::cout << "<blk_size>    : block size\n";
}

void ProcessInput(int argc, char** const argv, Params& args) {
    std::cout << "Processing input\n";
    int argCount = 0;
    if (argCount + 1 >= argc) {
        std::cerr << "Missing required argument: example number\n";
        Usage();
        throw ArgException();
    }

    args.exNum = atoi(argv[++argCount]);
    if (args.exNum > 6) {
        std::cerr << "example number must be between 1 and 3";
        Usage();
        throw ArgException();
    }

    if (argCount + 1 >= argc) {
        std::cerr << "Missing required argument: tensor dim\n";
        Usage();
        throw ArgException();
    }
    args.tensorDim = atoi(argv[++argCount]);

    if (argCount + 1 >= argc) {
        std::cerr << "Missing required argument: grid order\n";
        Usage();
        throw ArgException();
    }

    Unsigned gridOrder = atoi(argv[++argCount]);

    if (argCount + gridOrder >= argc) {
        std::cerr << "Missing required grid dimensions\n";
        Usage();
        throw ArgException();
    }

    args.gridShape.resize(gridOrder);
    args.nProcs = 1;
    for (int i = 0; i < gridOrder; i++) {
        int gridDim = atoi(argv[++argCount]);
        if (gridDim <= 0) {
            std::cerr << "Grid dim must be greater than 0\n";
            Usage();
            throw ArgException();
        }
        args.nProcs *= gridDim;
        args.gridShape[i] = gridDim;
    }

    args.blkSize = argCount >= argc ? 32 : atoi(argv[++argCount]);
}

template<typename T>
void PrintNorm(const DistTensor<T>& check, const DistTensor<T>& actual, const char* msg){
	int commRank = mpi::CommRank(MPI_COMM_WORLD);

	std::string dist16 = "[(0),(1),(2),(3),(4),(5),(6),(7),(8),(9),(10),(11),(12),(13),(14),(15)]";

	DistTensor<double> diff(dist16.c_str(), check.Grid());
	diff.ResizeTo(check);
	Diff(check, actual, diff);
	double norm = 1.0;
	norm = Norm(diff);
	if (commRank == 0){
	  std::cout << "NORM " << msg << " " << norm << std::endl;
	}
}
#endif
