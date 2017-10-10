/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef ROTE_BTAS_GEN_CONTRACT_HPP
#define ROTE_BTAS_GEN_CONTRACT_HPP

namespace rote{

////////////////////////////////////
// Utils
////////////////////////////////////

template<typename T>
void SetBlkContractStatCInfo(
	const DistTensor<T>& A, const IndexArray& indicesA,
	const DistTensor<T>& B, const IndexArray& indicesB,
	const DistTensor<T>& C, const IndexArray& indicesC,
	const std::vector<Unsigned>& blkSizes, bool isStatC,
	BlkContractStatCInfo& contractInfo
);

////////////////////////////////////
// DistContract Workhorse
////////////////////////////////////

template <typename T>
void ContractStat(T alpha, const DistTensor<T>& A, const IndexArray& indicesA, const DistTensor<T>& B, const IndexArray& indicesB, T beta, DistTensor<T>& C, const IndexArray& indicesC, bool isStatC, const std::vector<Unsigned>& blkSizes = std::vector<Unsigned>(0));

template <typename T>
void GenContract(T alpha, const DistTensor<T>& A, const IndexArray& indicesA, const DistTensor<T>& B, const IndexArray& indicesB, T beta, DistTensor<T>& C, const IndexArray& indicesC, const std::vector<Unsigned>& blkSizes = std::vector<Unsigned>(0));

template <typename T>
void GenContract(T alpha, const DistTensor<T>& A, const std::string& indicesA, const DistTensor<T>& B, const std::string& indicesB, T beta, DistTensor<T>& C, const std::string& indicesC, const std::vector<Unsigned>& blkSizes = std::vector<Unsigned>(0));

template<typename T>
class Contract {
public:
	// Main interface
	static void run(
		T alpha,
		const DistTensor<T>& A, const std::string& indicesA,
    const DistTensor<T>& B, const std::string& indicesB,
    T beta,
		      DistTensor<T>& C, const std::string& indicesC,
    const std::vector<Unsigned>& blkSizes
	);

private:
	//Struct interface
	static void setContractInfo(
		const DistTensor<T>& A, const IndexArray& indicesA,
    const DistTensor<T>& B, const IndexArray& indicesB,
    const DistTensor<T>& C, const IndexArray& indicesC,
    const std::vector<Unsigned>& blkSizes, bool isStatC,
          BlkContractStatCInfo& contractInfo
  );

	// Partition helpers
	static void runHelperPartitionAB(
		Unsigned depth, BlkContractStatCInfo& contractInfo,
		T alpha,
		const DistTensor<T>& A, const IndexArray& indicesA,
		const DistTensor<T>& B, const IndexArray& indicesB,
		T beta,
					DistTensor<T>& C, const IndexArray& indicesC
	);

	static void runHelperPartitionBC(
		Unsigned depth, BlkContractStatCInfo& contractInfo,
		T alpha,
		const DistTensor<T>& A, const IndexArray& indicesA,
		const DistTensor<T>& B, const IndexArray& indicesB,
		T beta,
					DistTensor<T>& C, const IndexArray& indicesC
	);

	// Internal interface
	static void run(
		T alpha,
		const DistTensor<T>& A, const IndexArray& indicesA,
		const DistTensor<T>& B, const IndexArray& indicesB,
		T beta,
					DistTensor<T>& C, const IndexArray& indicesC,
		const std::vector<Unsigned>& blkSizes,
		bool isStatC
	);

	// Local interface
	static void run(
		T alpha,
		const Tensor<T>& A, const IndexArray& indicesA,
		const Tensor<T>& B, const IndexArray& indicesB,
		T beta,
					Tensor<T>& C, const IndexArray& indicesC,
		bool  doEliminate, bool doPermute
	);
};

} // namespace rote

#endif // ifndef ROTE_BTAS_GEN_CONTRACT_HPP
