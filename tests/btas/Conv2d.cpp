#include "rote.hpp"
using namespace rote;

void Usage(){
    std::cout << "./Conv2d <variant> <N> <K> <H> <W> <C> <Fh> <Fw> <gridDim0> <gridDim1>\n";
    std::cout << "   Computes a basic convolution operation on a set of images\n";
    std::cout << "<variant>   : Algorithmic stationary variant to test (weights,input,output)\n";
    std::cout << "<N>         : Number of batches\n";
    std::cout << "<K>         : Output channel dimension\n";
    std::cout << "<H>         : Image height\n";
    std::cout << "<W>         : Image width\n";
    std::cout << "<C>         : Channel dimension\n";
    std::cout << "<Fh>        : Filter height\n";
    std::cout << "<Fw>        : Filter width\n";
    std::cout << "<gridOrder> : Order of processing mesh (>0)\n";
    std::cout << "<gridDimK>  : Grid mode-K dimension (>0)\n";
}

void InvalidArgs(const std::string& msg) {
  std::cerr << msg << "\n";
  Usage();
  throw ArgException();
}

typedef struct Arguments{
  Unsigned variant;
  Unsigned N;
  Unsigned K;
  Unsigned H;
  Unsigned W;
  Unsigned C;
  Unsigned Fh;
  Unsigned Fw;
  ObjShape gridShape;
} Params;

void ProcessInput(Unsigned argc,  char** const argv, Params& args){
  Unsigned i;
  Unsigned n_problem_args = 10;
  if(argc < n_problem_args) {
    InvalidArgs("Missing required arguments");
  }

  std::string variant = argv[1];
  if (variant.compare("weights") == 0) {
    args.variant = 0;
  } else if (variant.compare("input") == 0) {
    args.variant = 1;
  } else if (variant.compare("output") == 0) {
    args.variant = 2;
  } else {
    InvalidArgs("Variant argument must be weight or output");
  }

  args.N = atoi(argv[2]);
  if (args.N < 0) {
    InvalidArgs("N must be non-negative");
  }

  args.K = atoi(argv[3]);
  if (args.K < 0) {
    InvalidArgs("K must be non-negative");
  }

  args.H = atoi(argv[4]);
  if (args.H < 0) {
    InvalidArgs("H must be non-negative");
  }

  args.W = atoi(argv[5]);
  if (args.W < 0) {
    InvalidArgs("W must be non-negative");
  }

  args.C = atoi(argv[6]);
  if (args.C < 0) {
    InvalidArgs("C must be non-negative");
  }

  args.Fh = atoi(argv[7]);
  if (args.Fh < 0) {
    InvalidArgs("Fh must be non-negative");
  }

  args.Fw = atoi(argv[8]);
  if (args.Fw < 0) {
    InvalidArgs("Fw must be non-negative");
  }

  args.gridShape.resize(atoi(argv[9]));

  if(argc < n_problem_args + args.gridShape.size()) {
    InvalidArgs("Missing required arguments");
  }

  for(i = 0; i < args.gridShape.size(); i++) {
    args.gridShape[i] = atoi(argv[n_problem_args + i]);
    if (args.gridShape[i] < 0) {
      InvalidArgs("Grid dimension must be positive");
    }
  }
}

template<typename T>
void Conv2dTest( const mpi::Comm& comm, const Params& args) {
  const Grid g(comm, args.gridShape);
  ObjShape weightsShape = {args.K, args.Fh, args.Fw, args.C};
  ObjShape inActShape = {args.N, args.C, args.H + args.Fh - 1, args.W + args.Fw - 1};
  ObjShape outActShape = {args.N, args.K, args.H, args.W};

  if (args.variant == 0) {
    // Stationary-weights variant
    DistTensor<T> weights(weightsShape, "[(0);();();(1)]", g);
    DistTensor<T> inAct(inActShape, "[();(1);();()]", g);
    DistTensor<T> outAct(outActShape, "[();(0);();()]", g);

    MakeUniform(weights);
    MakeUniform(inAct);
    MakeUniform(outAct);

    Conv2D<T>::runStatWeights(weights, inAct, outAct);
    Print(outAct, "output_activations");
  } else if (args.variant == 1) {
    // Stationary-input variant
    DistTensor<T> weights(weightsShape, "[();();();(1)]", g);
    DistTensor<T> inAct(inActShape, "[(0);(1);();()]", g);
    DistTensor<T> outAct(outActShape, "[(0);();();()]", g);

    MakeUniform(weights);
    MakeUniform(inAct);
    MakeUniform(outAct);

    Conv2D<T>::runStatInputActivations(weights, inAct, outAct);
    Print(outAct, "output_activations");
  } else if (args.variant == 2) {
    // Stationary-output variant
    DistTensor<T> weights(weightsShape, "[();();();(1)]", g);
    DistTensor<T> inAct(inActShape, "[(0);();();()]", g);
    DistTensor<T> outAct(outActShape, "[(0);(1);();()]", g);

    MakeUniform(weights);
    MakeUniform(inAct);
    MakeUniform(outAct);

    Conv2D<T>::runStatOutputActivations(weights, inAct, outAct);
    Print(outAct, "output_activations");
  }
}

int main(int argc, char* argv[]) {
  Initialize(argc, argv);
  mpi::Comm comm = mpi::COMM_WORLD;
  const Int commSize = mpi::CommSize(comm);
  try {
    Params args;
    ProcessInput(argc, argv, args);

    if ((Unsigned)mpi::CommSize(comm) != rote::prod(args.gridShape)) {
      InvalidArgs("Program not started with correct number of processes");
    }

    Conv2dTest<double>(comm, args);

  } catch(std::exception& e) {ReportException(e);}

  Finalize();
  return 0;
}
