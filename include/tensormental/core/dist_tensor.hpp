/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef TMEN_CORE_DISTTENSOR_HPP
#define TMEN_CORE_DISTTENSOR_HPP

namespace tmen {
struct DistData
{
    Distribution modeDistributions;
    std::vector<Int> modeAlignments; 
//    Int root;     // only relevant for [o ,o ]
//    Int diagPath; // only relevant for [MD,* ]/[* ,MD] distributions
    const Grid* grid;
};
} // namespace tmen

#include "tensormental/core/dist_tensor/abstract.hpp"
//#include "tensormental/core/dist_tensor/circ_circ.hpp"
#include "tensormental/core/dist_tensor/mc_mr.hpp"
//#include "tensormental/core/dist_tensor/mc_star.hpp"
//#include "tensormental/core/dist_tensor/md_star.hpp"
//#include "tensormental/core/dist_tensor/mr_mc.hpp"
//#include "tensormental/core/dist_tensor/mr_star.hpp"
//#include "tensormental/core/dist_tensor/star_mc.hpp"
//#include "tensormental/core/dist_tensor/star_md.hpp"
//#include "tensormental/core/dist_tensor/star_mr.hpp"
//#include "tensormental/core/dist_tensor/star_star.hpp"
//#include "tensormental/core/dist_tensor/star_vc.hpp"
//#include "tensormental/core/dist_tensor/star_vr.hpp"
//#include "tensormental/core/dist_tensor/vc_star.hpp"
//#include "tensormental/core/dist_tensor/vr_star.hpp"

#endif // ifndef TMEN_CORE_DISTTENSOR_HPP
