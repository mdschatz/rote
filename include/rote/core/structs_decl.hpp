/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#ifndef ROTE_CORE_STRUCTS_DECL_HPP
#define ROTE_CORE_STRUCTS_DECL_HPP

#include <complex>
#include <vector>

namespace rote {

//Structs for performing general redistributions
struct RedistPlanInfo
{
	ModeArray tenModesReduced;
	ModeArray gridModesAppeared;
	ModeArray gridModesAppearedSinks;
	ModeArray gridModesRemoved;
	ModeArray gridModesRemovedSrcs;
	ModeArray gridModesMoved;
	ModeArray gridModesMovedSrcs;
	ModeArray gridModesMovedSinks;
};

struct Redist
{
	RedistType redistType;
	TensorDistribution dist;
	ModeArray modes;
};

typedef std::vector<Redist> RedistPlan;

struct 	BlkContractStatAInfo
{
	ModeArray reduceTensorModes;
	TensorDistribution distT;
	ModeArray alignModesT;
	ModeArray alignModesTTo;
	Permutation permT;

	ModeArray partModesB;
	TensorDistribution distIntB;
	ModeArray alignModesB;
	ModeArray alignModesBTo;
	Permutation permB;

	ModeArray partModesC;

	Permutation permA;
	std::vector<Unsigned> blkSizes;
};

struct 	BlkContractStatCInfo
{
	ModeArray partModesA;
	TensorDistribution distIntA;
	ModeArray alignModesA;
	ModeArray alignModesATo;
	Permutation permA;

	ModeArray partModesB;
	TensorDistribution distIntB;
	ModeArray alignModesB;
	ModeArray alignModesBTo;
	Permutation permB;

	Permutation permC;
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

	std::vector<Unsigned> blkSizes;
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
#endif
