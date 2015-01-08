#include "tensormental.hpp"

using namespace tmen;

void TestRS(mpi::Comm comm){
    //Set up the Grid object
    ObjShape gridShape(4);
    gridShape[0] = 3;
    gridShape[1] = 2;
    gridShape[2] = 3;
    gridShape[3] = 3;
    //Takes the MPI communicator we want to build the grid over and the shape of the grid
    Grid g(comm, gridShape);

    //Set up an input DistTensor
    ObjShape in_shape(5);
    in_shape[0] = 4;
    in_shape[1] = 5;
    in_shape[2] = 4;
    in_shape[3] = 6;
    in_shape[4] = 4;
    //Takes the shape of the global object, the distribution, and the grid
    DistTensor<double> input(in_shape, "[(0),(2),(1,3),(),()]", g);
    //Give input some random values
    MakeUniform(input);


    //Do the same for an output DistTensor
    ObjShape out(3);
    out[0] = in_shape[0];
    out[1] = in_shape[1];
    out[2] = 2;
    DistTensor<double> output(out, "[(0,3),(2),(1)]", g);

    Zero(output);
    ModeArray rModes(3);
    rModes[0] = 2;
    rModes[1] = 3;
    rModes[2] = 4;

    PrintData(input, "in_data");
//        PrintData(output, "out_data");
    Print(input, "in_shape");
    output.ReduceScatterRedistFrom(input, rModes);
//        PrintData(output, "out_data_after");
    Print(output, "out");
}

void TestA2A(mpi::Comm comm){
    //Set up the Grid object
    ObjShape gridShape(4);
    gridShape[0] = 3;
    gridShape[1] = 2;
    gridShape[2] = 3;
    gridShape[3] = 3;
    //Takes the MPI communicator we want to build the grid over and the shape of the grid
    Grid g(comm, gridShape);

    //Set up an input DistTensor
    ObjShape in_shape(4);
    in_shape[0] = 4;
    in_shape[1] = 7;
    in_shape[2] = 2;
    in_shape[3] = 3;
    //Takes the shape of the global object, the distribution, and the grid
    DistTensor<double> input(in_shape, "[(0),(1),(2),(3)]", g);
    //Give input some random values
    MakeUniform(input);


    //Do the same for an output DistTensor
    DistTensor<double> output(in_shape, "[(0),(1),(3),()]", g);

    PrintData(input, "in_data");
//        PrintData(output, "out_data");
    Print(input, "in_shape");
    ModeArray commModes(2);
    commModes[0] = 2;
    commModes[1] = 3;

    output.AllToAllRedistFrom(input, commModes);
//        PrintData(output, "out_data_after");
    Print(output, "out");
}

void TestP(mpi::Comm comm){
    //Set up the Grid object
    ObjShape gridShape(4);
    gridShape[0] = 3;
    gridShape[1] = 2;
    gridShape[2] = 3;
    gridShape[3] = 3;
    //Takes the MPI communicator we want to build the grid over and the shape of the grid
    Grid g(comm, gridShape);

    //Set up an input DistTensor
    ObjShape in_shape(4);
    in_shape[0] = 4;
    in_shape[1] = 5;
    in_shape[2] = 4;
    in_shape[3] = 6;
    //Takes the shape of the global object, the distribution, and the grid
    DistTensor<double> input(in_shape, "[(0),(),(),(2,3)]", g);
    //Give input some random values
    MakeUniform(input);


    //Do the same for an output DistTensor
    DistTensor<double> output(in_shape, "[(0),(),(),(3,2)]", g);

    PrintData(input, "in_data");
//        PrintData(output, "out_data");
    Print(input, "in_shape");
    ModeArray commModes(2);
    commModes[0] = 2;
    commModes[1] = 3;

    output.PermutationRedistFrom(input, commModes);
//        PrintData(output, "out_data_after");
    Print(output, "out");
}

void TestL(mpi::Comm comm){
    //Set up the Grid object
    ObjShape gridShape(4);
    gridShape[0] = 3;
    gridShape[1] = 2;
    gridShape[2] = 3;
    gridShape[3] = 3;
    //Takes the MPI communicator we want to build the grid over and the shape of the grid
    Grid g(comm, gridShape);

    //Set up an input DistTensor
    ObjShape in_shape(4);
    in_shape[0] = 4;
    in_shape[1] = 5;
    in_shape[2] = 4;
    in_shape[3] = 6;
    //Takes the shape of the global object, the distribution, and the grid
    DistTensor<double> input(in_shape, "[(0),(),(),(2,3)]", g);
    //Give input some random values
    MakeUniform(input);


    //Do the same for an output DistTensor
    DistTensor<double> output(in_shape, "[(0),(),(1),(2,3)]", g);

    PrintData(input, "in_data");
//        PrintData(output, "out_data");
    Print(input, "in_shape");

    output.LocalRedistFrom(input);
//        PrintData(output, "out_data_after");
    Print(output, "out");
}

void TestRead(const std::string& filename){
    //Set up the Grid object
    ObjShape gridShape(4);
    gridShape[0] = 2;
    gridShape[1] = 3;
    gridShape[2] = 2;
    gridShape[3] = 2;
    //Takes the MPI communicator we want to build the grid over and the shape of the grid
    Grid g(MPI_COMM_WORLD, gridShape);

    //Takes the shape of the global object, the distribution, and the grid
    DistTensor<double> input("[(1,0),(2,3)]", g);
    ObjShape shape;
    shape.push_back(8);
    shape.push_back(3);
    input.ResizeTo(shape);
    //Give input some random values
    Read(input, filename, BINARY_FLAT, true);

    PrintData(input, "in_data");
//        PrintData(output, "out_data");
    Print(input, "in_shape");
}

int main( int argc, char* argv[] ) {
    Initialize( argc, argv );
    Unsigned i;
    mpi::Comm comm = mpi::COMM_WORLD;
    const Int commRank = mpi::CommRank( comm );
    const Int commSize = mpi::CommSize( comm );

    try
    {
        TestRead("data_8_3_bin_flat");


    }
    catch( std::exception& e ) { ReportException(e); }

    Finalize();
}
