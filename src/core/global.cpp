/*
   Copyright (c) 2009-2013, Jack Poulson
                      2013, Jeff Hammond
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#include "rote.hpp"

namespace {

rote::Int numElemInits = 0;
bool roteInitializedMpi = false;
#ifdef HAVE_QT5
bool roteInitializedQt = false;
bool roteOpenedWindow = false;
QCoreApplication* coreApp;
bool haveMinRealWindowVal=false, haveMaxRealWindowVal=false,
     haveMinImagWindowVal=false, haveMaxImagWindowVal=false;
double minRealWindowVal, maxRealWindowVal,
       minImagWindowVal, maxImagWindowVal;
#endif
std::stack<rote::Int> blocksizeStack;
rote::Grid* defaultGrid = 0;
rote::mpi::CommMap* defaultCommMap = 0;
rote::Args* args = 0;

// A common Mersenne twister configuration
//std::mt19937 generator;

// Debugging
#ifndef RELEASE
std::stack<std::string> callStack;
#endif

// Tuning parameters for basic routines
//rote::Int localSymvFloatBlocksize = 64;
//rote::Int localSymvDoubleBlocksize = 64;
//rote::Int localSymvComplexFloatBlocksize = 64;
//rote::Int localSymvComplexDoubleBlocksize = 64;

//rote::Int localTrr2kFloatBlocksize = 64;
//rote::Int localTrr2kDoubleBlocksize = 64;
//rote::Int localTrr2kComplexFloatBlocksize = 64;
//rote::Int localTrr2kComplexDoubleBlocksize = 64;

//rote::Int localTrrkFloatBlocksize = 64;
//rote::Int localTrrkDoubleBlocksize = 64;
//rote::Int localTrrkComplexFloatBlocksize = 64;
//rote::Int localTrrkComplexDoubleBlocksize = 64;

// Tuning parameters for advanced routines
using namespace rote;
//HermitianTridiagApproach tridiagApproach = HERMITIAN_TRIDIAG_DEFAULT;
//GridOrder gridOrder = ROW_MAJOR;
}

namespace rote {

void PrintVersion( std::ostream& os )
{
    os << "ROTE version information:\n"
       << "  Git revision: " << GIT_SHA1 << "\n"
       << "  Version:      " << ROTE_VERSION_MAJOR << "."
                             << ROTE_VERSION_MINOR << "\n"
       << "  Build type:   " << CMAKE_BUILD_TYPE << "\n"
       << std::endl;
}

void PrintConfig( std::ostream& os )
{
	os << "Config information:\n"
	   << "Default"
	   << std::endl;
}

void PrintCCompilerInfo( std::ostream& os )
{
	os << "Compiler information:\n"
	   << "Default"
	   << std::endl;
}

void PrintCxxCompilerInfo( std::ostream& os )
{
	os << "CXX Compiler information:\n"
	   << "Default"
	   << std::endl;
}

bool Initialized()
{ return ::numElemInits > 0; }

void Initialize( int& argc, char**& argv )
{
    if( ::numElemInits > 0 )
    {
        ++::numElemInits;
        return;
    }

    ::args = new Args( argc, argv );

    ::numElemInits = 1;
    if( !mpi::Initialized() )
    {
        if( mpi::Finalized() )
        {
            LogicError
            ("Cannot initialize rote after finalizing MPI");
        }
#ifdef HAVE_OPENMP
        const Int provided = 
            mpi::InitializeThread
            ( argc, argv, mpi::THREAD_MULTIPLE );
        const Int commRank = mpi::CommRank( mpi::COMM_WORLD );
        if( provided != mpi::THREAD_MULTIPLE && commRank == 0 )
        {
            std::cerr << "WARNING: Could not achieve THREAD_MULTIPLE support."
                      << std::endl;
        }
#else
        mpi::Initialize( argc, argv );
#endif
        ::roteInitializedMpi = true;
    }
    else
    {
#ifdef HAVE_OPENMP
        const Int provided = mpi::QueryThread();
        if( provided != mpi::THREAD_MULTIPLE )
        {
            throw std::runtime_error
            ("MPI initialized with inadequate thread support for Elemental");
        }
#endif
    }


    // Queue a default algorithmic blocksize
    while( ! ::blocksizeStack.empty() )
        ::blocksizeStack.pop();
    ::blocksizeStack.push( 128 );

    // Build the default grid
    //defaultGrid = new Grid( mpi::COMM_WORLD );
    defaultCommMap = new mpi::CommMap();

    // Create the types and ops needed for ValueInt
    mpi::CreateValueIntType<Int>();
    mpi::CreateValueIntType<float>();
    mpi::CreateValueIntType<double>();
    mpi::CreateMaxLocOp<Int>();
    mpi::CreateMaxLocOp<float>();
    mpi::CreateMaxLocOp<double>();

    // Do the same for ValueIntPair
    mpi::CreateValueIntPairType<Int>();
    mpi::CreateValueIntPairType<float>();
    mpi::CreateValueIntPairType<double>();
    mpi::CreateMaxLocPairOp<Int>();
    mpi::CreateMaxLocPairOp<float>();
    mpi::CreateMaxLocPairOp<double>();

    // Seed the random number generators using Katzgrabber's approach
    // from "Random Numbers in Scientific Computing: An Introduction"
    // NOTE: srand no longer needed after C++11
    const unsigned rank = mpi::CommRank( mpi::COMM_WORLD );
    const long secs = time(NULL);
    const long seed = abs(((secs*181)*((rank-83)*359))%104729);
//    ::generator.seed( seed );
    srand(seed);
}

void Finalize()
{
#ifndef RELEASE
    CallStackEntry entry("Finalize");
#endif
    if( ::numElemInits <= 0 )
        LogicError("Finalized Elemental more than initialized");
    --::numElemInits;

    if( mpi::Finalized() )
    {
        std::cerr << "Warning: MPI was finalized before Elemental." 
                  << std::endl;
    }
    if( ::numElemInits == 0 )
    {
        delete ::args;
        ::args = 0;

        if( ::roteInitializedMpi )
        {
            // Destroy the types and ops needed for ValueInt
            mpi::DestroyValueIntType<Int>();
            mpi::DestroyValueIntType<float>();
            mpi::DestroyValueIntType<double>();
            mpi::DestroyMaxLocOp<Int>();
            mpi::DestroyMaxLocOp<float>();
            mpi::DestroyMaxLocOp<double>();

            // Do the same for ValueIntPair
            mpi::DestroyValueIntPairType<Int>();
            mpi::DestroyValueIntPairType<float>();
            mpi::DestroyValueIntPairType<double>();
            mpi::DestroyMaxLocPairOp<Int>();
            mpi::DestroyMaxLocPairOp<float>();
            mpi::DestroyMaxLocPairOp<double>();

            // Delete the default grid
            delete ::defaultGrid;
            ::defaultGrid = 0;
            delete ::defaultCommMap;
            ::defaultCommMap = 0;

            mpi::Finalize();
        }

        ::defaultGrid = 0;
        ::defaultCommMap = 0;
        while( ! ::blocksizeStack.empty() )
            ::blocksizeStack.pop();
    }
}

Args& GetArgs()
{ 
    if( args == 0 )
        throw std::runtime_error("No available instance of Args");
    return *::args; 
}

Int Blocksize()
{ return ::blocksizeStack.top(); }

void SetBlocksize( Int blocksize )
{ ::blocksizeStack.top() = blocksize; }

void PushBlocksizeStack( Int blocksize )
{ ::blocksizeStack.push( blocksize ); }

void PopBlocksizeStack()
{ ::blocksizeStack.pop(); }

ModeArray OrderedModes(Unsigned order)
{
#ifndef RELEASE
    CallStackEntry entry("OrderedModes");
#endif
    Unsigned i;
    ModeArray ret(order);
    for(i = 0; i < order; i++)
        ret[i] = i;
    return ret;
}

const Grid& DefaultGrid()
{
#ifndef RELEASE
    CallStackEntry entry("DefaultGrid");
    if( ::defaultGrid == 0 )
        LogicError
        ("Attempted to return a non-existant default grid. Please ensure that "
         "Elemental is initialized before creating a DistMatrix.");
#endif
    return *::defaultGrid;
}

mpi::CommMap& DefaultCommMap()
{
#ifndef RELEASE
    CallStackEntry entry("DefaultCommMap");
    if( ::defaultCommMap == 0 )
        LogicError
        ("Attempted to return a non-existant default grid. Please ensure that "
         "Elemental is initialized before creating a DistMatrix.");
#endif
    return *::defaultCommMap;
}

//std::mt19937& Generator()
//{ return ::generator; }

// If we are not in RELEASE mode, then implement wrappers for a CallStack
#ifndef RELEASE
void PushCallStack( std::string s )
{ 
#ifdef HAVE_OPENMP
    if( omp_get_thread_num() != 0 )
        return;
#endif // HAVE_OPENMP
    ::callStack.push(s); 
}

void PopCallStack()
{ 
#ifdef HAVE_OPENMP
    if( omp_get_thread_num() != 0 )
        return;
#endif // HAVE_OPENMP
    ::callStack.pop(); 
}

void DumpCallStack( std::ostream& os )
{
    std::ostringstream msg;
    while( ! ::callStack.empty() )
    {
        msg << "[" << ::callStack.size() << "]: " << ::callStack.top() << "\n";
        ::callStack.pop();
    }
    os << msg.str();
    os.flush();
}
#endif // RELEASE

} // namespace rote
