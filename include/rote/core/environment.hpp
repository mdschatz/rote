/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef ROTE_CORE_ENVIRONMENT_DECL_HPP
#define ROTE_CORE_ENVIRONMENT_DECL_HPP

namespace rote {

#define MAX_ELEM_PER_PROC 10000000
#define NOT_USED(x) ( (void)(x) )

void PrintVersion( std::ostream& os=std::cout );
void PrintConfig( std::ostream& os=std::cout );
void PrintCCompilerInfo( std::ostream& os=std::cout );
void PrintCxxCompilerInfo( std::ostream& os=std::cout );

// For initializing and finalizing Elemental
void Initialize( int& argc, char**& argv );
void Finalize();
bool Initialized();

// For getting the MPI argument instance (for internal usage)
class Args : public choice::MpiArgs
{
public:
    Args
    ( int argc, char** argv,
      mpi::Comm comm=mpi::COMM_WORLD, std::ostream& error=std::cerr )
    : choice::MpiArgs(argc,argv,comm,error)
    { }
    ~Args() { }
protected:
    void HandleVersion( std::ostream& os=std::cout ) const{
        std::string version = "--version";
        char** arg = std::find( argv_, argv_+argc_, version );
        const bool foundVersion = ( arg != argv_+argc_ );
        if( foundVersion )
        {
            if( mpi::WorldRank() == 0 )
                PrintVersion(os);
            throw ArgException();
        }
    };

    void HandleBuild( std::ostream& os=std::cout ) const{
        std::string build = "--build";
        char** arg = std::find( argv_, argv_+argc_, build );
        const bool foundBuild = ( arg != argv_+argc_ );
        if( foundBuild )
        {
            if( mpi::WorldRank() == 0 )
            {
                PrintVersion(os);
                //PrintConfig();
                PrintCCompilerInfo(os);
                PrintCxxCompilerInfo(os);
            }
            throw ArgException();
        }
    };
};
Args& GetArgs();

// For processing command-line arguments
template<typename T>
T Input( std::string name, std::string desc )
{ return GetArgs().Input<T>( name, desc ); };

template<typename T>
inline
T Input( std::string name, std::string desc, T defaultVal )
{ return GetArgs().Input( name, desc, defaultVal ); };

inline
void ProcessInput(){ GetArgs().Process(); };

inline
void PrintInputReport(){ GetArgs().PrintReport(); };

// For getting and setting the algorithmic blocksize
Int Blocksize();
void SetBlocksize( Int blocksize );

// For manipulating the algorithmic blocksize as a stack
void PushBlocksizeStack( Int blocksize );
void PopBlocksizeStack();

//std::mt19937& Generator();

inline Unsigned IntCeil(Unsigned m, Unsigned n)
{
    return ( m > 0 ? (m - 1)/n + 1 : 0 );
}

inline double Ceil(double m)
{ return std::ceil(m); }

inline Int Max( Int m, Int n )
{ return std::max(m,n); }

inline Int Min( Int m, Int n )
{ return std::min(m,n); }

// Replacement for std::memcpy, which is known to often be suboptimal.
// Notice the sizeof(T) is no longer required.
template<typename T>
inline
void MemCopy( T* dest, const T* source, std::size_t numEntries ){
    // This can be optimized/generalized later
    std::memcpy( dest, source, numEntries*sizeof(T) );
};

template<typename T>
inline
void MemSwap( T* a, T* b, T* temp, std::size_t numEntries ){
    // temp := a
    std::memcpy( temp, a, numEntries*sizeof(T) );
    // a := b
    std::memcpy( a, b, numEntries*sizeof(T) );
    // b := temp
    std::memcpy( b, temp, numEntries*sizeof(T) );
};

// Replacement for std::memset, which is likely suboptimal and hard to extend
// to non-POD datatypes. Notice that sizeof(T) is no longer required.
template<typename T>
inline
void MemZero( T* buffer, std::size_t numEntries ){
    // This can be optimized/generalized later
    std::memset( buffer, 0, numEntries*sizeof(T) );
};

// TODO: Remove CallStackEntry
#ifndef RELEASE
void PushCallStack( std::string s );
void PopCallStack();
void DumpCallStack( std::ostream& os=std::cerr );

class CallStackEntry
{
public:
    CallStackEntry( std::string s )
    {
        if( !std::uncaught_exception() )
            PushCallStack(s);
    }
    ~CallStackEntry()
    {
        if( !std::uncaught_exception() )
            PopCallStack();
    }
};
#endif // ifndef RELEASE

inline
void ReportException( const std::exception& e, std::ostream& os=std::cerr ){
    if( std::string(e.what()) != "" )
    {
        os << "Process " << mpi::WorldRank() << " caught error message:\n"
           << e.what() << std::endl;
    }
#ifndef RELEASE
    DumpCallStack( os );
#endif
};

class ArgException;

inline
void ComplainIfDebug(){
#ifndef RELEASE
    if( mpi::WorldRank() == 0 )
    {
        std::cout << "==========================================\n"
                  << " In debug mode! Performance will be poor! \n"
                  << "==========================================" << std::endl;
    }
#endif
};

} // namespace rote

#endif // ifndef ROTE_CORE_ENVIRONMENT_DECL_HPP
