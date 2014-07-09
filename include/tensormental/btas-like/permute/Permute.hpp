/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef TMEN_BTAS_PERMUTE_HPP
#define TMEN_BTAS_PERMUTE_HPP

namespace tmen{


//Loop based permutation
template<typename T>
void Permute(Tensor<T>& B, const Tensor<T>& A, const Permutation& perm){
    Unsigned i;
    Unsigned order = A.Order();

    T* dstBuf = B.Buffer();
    const T* srcBuf = A.LockedBuffer();

    if(A.Order() == 0){
        MemCopy(&(dstBuf[0]), &(srcBuf[0]), 1);
        return;
    }

    Location curLoc(order, 0);
    Permutation invperm = DetermineInversePermutation(perm);

    ObjShape shapeA = A.Shape();
    ObjShape shapeB = B.Shape();
    std::vector<Unsigned> strideA = A.LDims();
    std::vector<Unsigned> strideB = B.LDims();

    Unsigned linLocDst = 0;
    Unsigned linLocSrc = 0;
    Unsigned updatePtr = 0;
    bool done = !ElemwiseLessThan(curLoc,shapeA);


//    printf("shapeA: [%d", shapeA[0]);
//    for(i = 1; i < order; i++)
//        printf(" %d", shapeA[i]);
//    printf("]\n");
//    printf("shapeB: [%d", shapeB[0]);
//    for(i = 1; i < order; i++)
//        printf(" %d", shapeB[i]);
//    printf("]\n");
//
//    printf("strideA: [%d", strideA[0]);
//    for(i = 1; i < order; i++)
//        printf(" %d", strideA[i]);
//    printf("]\n");
//    printf("strideB: [%d", strideB[0]);
//    for(i = 1; i < order; i++)
//        printf(" %d", strideB[i]);
//    printf("]\n");

    while(!done){
//        printf("curLoc: [%d", curLoc[0]);
//        for(i = 1; i < order; i++)
//            printf(" %d", curLoc[i]);
//        printf("]\n");
//        printf("linLocDst: %d\n", linLocDst);
//        printf("linLocSrc: %d\n", linLocSrc);
        dstBuf[linLocDst] = srcBuf[linLocSrc];

        //Update info
        curLoc[updatePtr]++;
        linLocSrc += strideA[updatePtr];
        linLocDst += strideB[invperm[updatePtr]];

        //Loop update
        while(updatePtr < order && curLoc[updatePtr] == shapeA[updatePtr]){
            curLoc[updatePtr] = 0;
            linLocSrc -= strideA[updatePtr]*shapeA[updatePtr];
            linLocDst -= strideB[invperm[updatePtr]] * shapeB[invperm[updatePtr]];
            updatePtr++;

            if(updatePtr >= order){
                done = true;
                break;
            }else{
                curLoc[updatePtr]++;
                linLocSrc += strideA[updatePtr];
                linLocDst += strideB[invperm[updatePtr]];
            }
        }
        updatePtr = 0;
    }
}

} // namespace tmen

#endif // ifndef TMEN_BTAS_PERMUTE_HPP
