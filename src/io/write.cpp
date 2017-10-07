/*
   Copyright (c) 2009-2015, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#include "rote.hpp"

// TODO: Reimplement write to file
namespace rote {

	template<typename T>
	inline void
	Write( const Tensor<T>& A, std::string title, std::string filename )
	{
	#ifndef RELEASE
	    CallStackEntry entry("Write");
	#endif
	    std::ofstream file( filename.c_str() );
	    file.setf( std::ios::scientific );
	    Print( A, title, false );
	    file.close();
	}

	// If already in [* ,* ] or [o ,o ] distributions, no copy is needed
	template<typename T>
	inline void
	Write
	( const DistTensor<T>& A, std::string title,
	  std::string filename )
	{
	#ifndef RELEASE
	    CallStackEntry entry("Write");
	#endif
	    if( A.Grid().LinearRank() == 0 )
	        Write( A.LockedTensor(), title, filename );
	}

#define FULL(T) \
  template void \
		Write \
		( const DistTensor<T>& A, std::string title="", \
		  std::string filename="DistTensor" ); \
	template void \
		Write \
		( const Tensor<T>& A, std::string title="", \
		  std::string filename="DistTensor" );

FULL(Int)
#ifndef DISABLE_FLOAT
FULL(float)
#endif
FULL(double)

#ifndef DISABLE_COMPLEX
#ifndef DISABLE_FLOAT
FULL(std::complex<float>)
#endif
FULL(std::complex<double>)
#endif

} // namespace rote
