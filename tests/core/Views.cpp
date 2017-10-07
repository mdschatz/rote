/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
// NOTE: It is possible to simply include "rote.hpp" instead
#include "rote.hpp"
using namespace rote;

void Usage(){
    std::cout << "./DistTensor <gridOrder> <gridDim0> <gridDim1> ... <tenOrder> <tenDim0> <tenDim1> ... \"<tensorDist>\"\n";
    std::cout << "<gridOrder>  : order of the grid ( >0 )\n";
    std::cout << "<gridDimK>   : dimension of mode-K of grid\n";
    std::cout << "<tenOrder>   : order of the tensor ( >0 )\n";
    std::cout << "<tenDimK>    : dimension of mode-K of tensor\n";
    std::cout << "<tensorDist> : distribution of tensor(Must be in quotes)\n";
}

typedef struct Arguments{
  Unsigned gridOrder;
  Unsigned tenOrder;
  Unsigned nProcs;
  ObjShape gridShape;
  ObjShape tensorShape;
  TensorDistribution tensorDist;
} Params;

void ProcessInput(Unsigned argc,  char** const argv, Params& args){
    Unsigned i;
    Unsigned argCount = 0;
    if(argCount + 1 >= argc){
        std::cerr << "Missing required gridOrder argument\n";
        Usage();
        throw ArgException();
    }

    Unsigned gridOrder = atoi(argv[++argCount]);
    args.gridOrder = gridOrder;
    if(gridOrder <= 0){
        std::cerr << "Grid order must be greater than 0\n";
        Usage();
        throw ArgException();
    }

    if(argCount + gridOrder >= argc){
        std::cerr << "Missing required grid dimensions\n";
        Usage();
        throw ArgException();
    }

    args.gridShape.resize(gridOrder);
    for(i = 0; i < gridOrder; i++){
        int gridDim = atoi(argv[++argCount]);
        if(gridDim <= 0){
            std::cerr << "Grid dim must be greater than 0\n";
            Usage();
            throw ArgException();
        }
        args.gridShape[i] = gridDim;
    }
    args.nProcs = rote::prod(args.gridShape);

    if(argCount + 1 >= argc){
        std::cerr << "Missing required tenOrder argument\n";
        Usage();
        throw ArgException();
    }
    Unsigned tenOrder = atoi(argv[++argCount]);
    args.tenOrder = tenOrder;

    if(argCount + tenOrder >= argc){
        std::cerr << "Missing required tensor dimensions\n";
        Usage();
        throw ArgException();
    }

    args.tensorShape.resize(tenOrder);
    for(i = 0; i < tenOrder; i++){
        Unsigned tensorDim = atoi(argv[++argCount]);
        if(tensorDim == 0){
            std::cerr << "Tensor dimension must be greater than 0\n";
            Usage();
            throw ArgException();
        }
        args.tensorShape[i] = tensorDim;
    }

    if(argCount + 1 >= argc){
        std::cerr << "Missing tensor distribution argument\n";
        Usage();
        throw ArgException();
    }

    std::string tensorDist(argv[++argCount]);
    args.tensorDist = rote::StringToTensorDist(tensorDist);

    if(args.tensorDist.size() != args.tensorShape.size() + 1){
        std::cerr << "Tensor distribution must be of same order as tensor\n";
        Usage();
        throw ArgException();
    }
}

template<typename T>
void
PrintLocalView(const Tensor<T>& A){
    Unsigned i;
    const Unsigned order = A.Order();
    printf("        Local tensor info:\n");
    printf("          shape:");
    if(order > 0)
        printf(" %d", A.Dimension(0));
    for(i = 1; i < order; i++)
        printf(" x %d", A.Dimension(i));
    printf("\n");

    Print(A, "        data: ");

}

template<typename T>
void
PrintView(const char* msg, const DistTensor<T>& A){
    Unsigned i;
    const Unsigned order = A.Order();
    printf("    Info for: %s\n", msg);
    printf("      shape:");
    if(order > 0)
        printf(" %d", A.Dimension(0));
    for(i = 1; i < order; i++)
        printf(" x %d", A.Dimension(i));
    printf("\n");

    printf("      alignments:");
    if(order > 0)
        printf(" %d", A.ModeAlignment(0));
    for(i = 1; i < order; i++)
        printf(" x %d", A.ModeAlignment(i));
    printf("\n");

    PrintLocalView(A.LockedTensor());
}

template<typename T>
void
TestConstViews(DistTensor<T>& A){
#ifndef RELEASE
    CallStackEntry entry("TestConstViews");
#endif
    const rote::Grid& g = A.Grid();
    Unsigned i;
    const Unsigned order = A.Order();
    Location start(order, 0);

    const ObjShape shape = A.Shape();

    DistTensor<T> AT(order, g), AB(order, g), A0(order, g), A1(order, g), A2(order, g);

    for(i = 0; i < order; i++){
        Mode mode = i;
        printf("Iterating over mode: %d\n", mode);

        LockedPartitionDown(A, AT, AB, mode, 0);

        Unsigned count = 0;
        while(AT.Dimension(mode) < A.Dimension(mode)){
            printf("  iteration: %d\n", count);
            LockedRepartitionDown(AT,   A0,
                                  A1,
                           /**/  /**/
                            AB,   A2, mode, 1);
            /////////////////////////////////
            PrintView("A0", A0);
            PrintView("A1", A1);
            PrintView("A2", A2);
            count++;
            /////////////////////////////////
            SlideLockedPartitionDown(AT,  A0,
                                    A1,
                              /**/ /**/
                               AB,  A2, mode);
        }
    }

    //Perform lower/higher order view changes (only applies to Local Tensors)
    Tensor<T> ATensor(order);
    ATensor = A.Tensor();
    Tensor<T> ALO(order), AHO(order);
    //View as lower order object
    Unsigned startMode, nMergeModes;
    for(startMode = 0; startMode < order; startMode++){
        for(nMergeModes = 2; startMode + nMergeModes <= order; nMergeModes++){
            std::vector<ModeArray> oldModes(1);
            ModeArray modesToMerge(nMergeModes);
            for(i = 0; i < nMergeModes; i++){
                modesToMerge[i] = startMode + i;
            }
            oldModes[0] = modesToMerge;
            printf("Merging modes (%d", modesToMerge[0]);
            for(i = 1; i < nMergeModes; i++)
                printf(" %d", modesToMerge[i]);
            printf(") of dimension (%d", ATensor.Dimension(startMode));
            for(i = 1; i < nMergeModes; i++)
                printf(" %d", ATensor.Dimension(startMode + i));
            printf(")\n");
            LockedViewAsLowerOrder(ALO, ATensor, oldModes);
            PrintLocalView(ALO);
        }
    }

    //TODO: Fix Test
//    //Try to view as higher order object
//    ObjShape ATensorShape = ATensor.Shape();
//    //See if we can split any mode into two
//    for(i = 0; i < order; i++){
//        Unsigned modeDim = ATensorShape[i];
//        for(j = 2; j < modeDim; j++){
//            if(modeDim % j == 0){
//                Mode modeToSplit = i;
//                Unsigned newMode1Dim = modeDim / j;
//                Unsigned newMode2Dim = j;
//                ModeArray oldModes(1);
//                oldModes[0] = modeToSplit;
//                std::vector<ObjShape> splitShape(1);
//                ObjShape newShape(2);
//                newShape[0] = newMode1Dim;
//                newShape[1] = newMode2Dim;
//                splitShape[0] = newShape;
//                printf("Splitting mode %d of dimension %d into shape (%d, %d)\n", modeToSplit, ATensor.Dimension(modeToSplit), newMode1Dim, newMode2Dim);
//                LockedViewAsHigherOrder(AHO, ATensor, oldModes, splitShape);
//                PrintLocalView(AHO);
//            }
//        }
//    }
}

template<typename T>
void
TestNonConstViews(DistTensor<T>& A){
#ifndef RELEASE
    CallStackEntry entry("TestNonConstViews");
#endif
    const rote::Grid& g = A.Grid();
    Unsigned i;
    const Unsigned order = A.Order();
    Location start(order, 0);

    const ObjShape shape = A.Shape();

    DistTensor<T> AT(order, g), AB(order, g), A0(order, g), A1(order, g), A2(order, g);

    for(i = 0; i < order; i++){
        Mode mode = i;
        printf("Iterating over mode: %d\n", mode);

        PartitionDown(A, AT, AB, mode, 0);

        Unsigned count = 0;
        while(AT.Dimension(mode) < A.Dimension(mode)){
            printf("  iteration: %d\n", count);
            RepartitionDown(AT,   A0,
                                  A1,
                           /**/  /**/
                            AB,   A2, mode, 1);
            /////////////////////////////////
            PrintView("A0", A0);
            PrintView("A1", A1);
            PrintView("A2", A2);
            count++;
            /////////////////////////////////
            SlidePartitionDown(AT,  A0,
                                    A1,
                              /**/ /**/
                               AB,  A2, mode);
        }
    }

    //Perform lower/higher order view changes (only applies to Local Tensors)
    Tensor<T> ATensor(order);
    ATensor = A.Tensor();
    Tensor<T> ALO(order), AHO(order);
    Unsigned startMode, nMergeModes;
    for(startMode = 0; startMode < order; startMode++){
        for(nMergeModes = 2; startMode + nMergeModes <= order; nMergeModes++){
            std::vector<ModeArray> oldModes(1);
            ModeArray modesToMerge(nMergeModes);
            for(i = 0; i < nMergeModes; i++){
                modesToMerge[i] = startMode + i;
            }
            oldModes[0] = modesToMerge;
            printf("Merging modes (%d", modesToMerge[0]);
            for(i = 1; i < nMergeModes; i++)
                printf(" %d", modesToMerge[i]);
            printf(") of dimension (%d", ATensor.Dimension(startMode));
            for(i = 1; i < nMergeModes; i++)
                printf(" %d", ATensor.Dimension(startMode + i));
            printf(")\n");
            ViewAsLowerOrder(ALO, ATensor, oldModes);
            PrintLocalView(ALO);
        }
    }

    //TODO: FIX TEST
//    //Try to view as higher order object
//    ObjShape ATensorShape = ATensor.Shape();
//    //See if we can split any index into two
//    for(i = 0; i < order; i++){
//        Unsigned modeDim = ATensorShape[i];
//        for(j = 2; j < modeDim; j++){
//            if(modeDim % j == 0){
//                Mode modeToSplit = i;
//                Unsigned newMode1Dim = modeDim / j;
//                Unsigned newMode2Dim = j;
//                ModeArray oldModes(1);
//                oldModes[0] = modeToSplit;
//                std::vector<ObjShape> splitShape(1);
//                ObjShape newShape(2);
//                newShape[0] = newMode1Dim;
//                newShape[1] = newMode2Dim;
//                splitShape[0] = newShape;
//                printf("Splitting mode %d of dimension %d into shape (%d, %d)\n", modeToSplit, ATensor.Dimension(modeToSplit), newMode1Dim, newMode2Dim);
//                ViewAsHigherOrder(AHO, ATensor, oldModes, splitShape);
//                PrintLocalView(AHO);
//            }
//        }
//    }
}

template<typename T>
void
Set(DistTensor<T>& A)
{
    MakeUniform(A);
//    Unsigned order = A.Order();
//    Location loc(order, 0);

//    Unsigned ptr = 0;
//    Unsigned counter = 0;
//    bool stop = false;
//
//    while(!stop){
//        A.Set(loc, counter);
//
//        //Update
//        counter++;
//        loc[ptr]++;
//        while(loc[ptr] == A.Dimension(ptr)){
//            loc[ptr] = 0;
//            ptr++;
//            if(ptr == order){
//                stop = true;
//                break;
//            }else{
//                loc[ptr]++;
//            }
//        }
//        ptr = 0;
//    }
}

template<typename T>
void
DistTensorTest( const Params& args, const Grid& g )
{
#ifndef RELEASE
    CallStackEntry entry("DistTensorTest");
#endif
    const Int commRank = mpi::CommRank( mpi::COMM_WORLD );

    DistTensor<T> A(args.tensorShape, args.tensorDist, g);

    Set(A);
    Print(A, "Random A");

    if(commRank == 0){
        printf("Performing Const tests\n");
    }

    TestConstViews(A);

    if(commRank == 0){
        printf("Performing Non-Const tests\n");
    }

    TestNonConstViews(A);
}

int 
main( int argc, char* argv[] )
{
    Initialize( argc, argv );
    Unsigned i;
    mpi::Comm comm = mpi::COMM_WORLD;
    const Int commRank = mpi::CommRank( comm );
    const Int commSize = mpi::CommSize( comm );
    printf("My Rank: %d\n", commRank);
    try
    {
        Params args;

        ProcessInput(argc, argv, args);

        if(commRank == 0 && args.nProcs != ((Unsigned)commSize)){
            std::cerr << "program not started with correct number of processes\n";
            Usage();
            throw ArgException();
        }

        if(commRank == 0){
            printf("Creating %d", args.gridShape[0]);
            for(i = 1; i < args.gridOrder; i++)
                printf(" x %d", args.gridShape[i]);
            printf(" grid\n");

            printf("Creating [%d", args.tensorShape[0]);
            for(i = 1; i < args.tenOrder; i++)
                printf(", %d", args.tensorShape[i]);
            printf("] tensor\n");
        }

        const Grid g( comm, args.gridShape );

        DistTensorTest<double>( args, g );

    }
    catch( std::exception& e ) { ReportException(e); }

    Finalize();
    //printf("Completed\n");
    return 0;
}
