/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef TMEN_BTAS_NORM_HPP
#define TMEN_BTAS_NORM_HPP

#include "tensormental/util/vec_util.hpp"
#include "tensormental/util/btas_util.hpp"
#include "tensormental/core/view_decl.hpp"

namespace tmen{

////////////////////////////////////
// Local routines
////////////////////////////////////

template <typename T>
BASE(T) Norm(const Tensor<T>& A){
#ifndef RELEASE
    CallStackEntry("Norm");
#endif
    Unsigned i;
    BASE(T) norm = 0;
    const T* buf = A.LockedBuffer();
    for(i = 0; i < prod(A.Shape()); i++){
        norm += buf[i] * buf[i];
    }

    return Sqrt(norm);
}

////////////////////////////////////
// Global routines
////////////////////////////////////
template<typename T>
BASE(T) Norm(const DistTensor<T>& A){
#ifndef RELEASE
    CallStackEntry("Norm");
#endif
    BASE(T) local_norm = Norm(A.LockedTensor());
    BASE(T) global_norm = mpi::AllReduce(local_norm, mpi::SUM, A.GetParticipatingComm());
    return global_norm;
}

} // namespace tmen

#endif // ifndef TMEN_BTAS_NORM_HPP
