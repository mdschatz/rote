/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#ifndef ROTE_CORE_STRUCTS_HPP
#define ROTE_CORE_STRUCTS_HPP

#include "permutation.hpp"
#include "tensor_distribution.hpp"

namespace rote {

struct Conv2DInfo {
	std::vector<Unsigned> blkSizes;
};

struct BlkContractStatCInfo
{
	// Stat A
	ModeArray reduceTensorModes;
	TensorDistribution distT;
	ModeArray alignModesT;
	ModeArray alignModesTTo;
	Permutation permT;

	// Stat C
	ModeArray partModesA;
	TensorDistribution distIntA;
	ModeArray alignModesA;
	ModeArray alignModesATo;
	Permutation permA;

	// Stat A+C
	ModeArray partModesB;
	TensorDistribution distIntB;
	ModeArray alignModesB;
	ModeArray alignModesBTo;
	Permutation permB;

	Permutation permC; // Stat C
	ModeArray partModesC; // Stat A
	std::vector<Unsigned> blkSizes;
};

struct BlkHadamardStatCInfo
{
	ModeArray partModesACA;
	ModeArray partModesACC;

	ModeArray partModesBCB;
	ModeArray partModesBCC;

	TensorDistribution distIntA;
	ModeArray alignModesA;
	ModeArray alignModesATo;
	Permutation permA;

	TensorDistribution distIntB;
	ModeArray alignModesB;
	ModeArray alignModesBTo;
	Permutation permB;

	TensorDistribution distIntC;
	ModeArray alignModesC;
	ModeArray alignModesCTo;
	Permutation permC;

	std::vector<Unsigned> blkSizes;
	bool isStatC;
};

//Pack data structs
struct PackData
{
    ObjShape loopShape;
    std::vector<Unsigned> srcBufStrides;
    std::vector<Unsigned> dstBufStrides;
    Permutation permutation;
};

struct YAxpPxData{
    ObjShape loopShape;
    std::vector<Unsigned> srcStrides;
    std::vector<Unsigned> permSrcStrides;
    std::vector<Unsigned> dstStrides;
};

struct YAxpByData{
    ObjShape loopShape;
    std::vector<Unsigned> srcStrides;
    std::vector<Unsigned> dstStrides;
};

struct ScalData{
    ObjShape loopShape;
    std::vector<Unsigned> srcStrides;
};

struct DiffData{
    ObjShape loopShape;
    std::vector<Unsigned> src1Strides;
    std::vector<Unsigned> src2Strides;
    std::vector<Unsigned> dstStrides;
};

struct ElemScalData{
    ObjShape loopShape;
    std::vector<Unsigned> src1Strides;
    std::vector<Unsigned> src2Strides;
    std::vector<Unsigned> dstStrides;
};

struct HadamardScalData{
    ObjShape loopShapeAC;
    std::vector<Unsigned> stridesACA;
    std::vector<Unsigned> stridesACC;

		ObjShape loopShapeBC;
    std::vector<Unsigned> stridesBCB;
    std::vector<Unsigned> stridesBCC;

		ObjShape loopShapeABC;
    std::vector<Unsigned> stridesABCA;
		std::vector<Unsigned> stridesABCB;
    std::vector<Unsigned> stridesABCC;
};

struct ZAxpByData{
    ObjShape loopShape;
    std::vector<Unsigned> src1Strides;
    std::vector<Unsigned> src2Strides;
    std::vector<Unsigned> dstStrides;
};

struct ZAxpBypPxData{
    ObjShape loopShape;
    std::vector<Unsigned> src1Strides;
    std::vector<Unsigned> src2Strides;
    std::vector<Unsigned> permSrcStrides;
    std::vector<Unsigned> dstStrides;
};
}

#endif // ifndef ROTE_CORE_STRUCTS_HPP
