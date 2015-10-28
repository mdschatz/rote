
#include "tensormental/util/vec_util.hpp"
#include "tensormental/core/error_decl.hpp"
#include <algorithm>
#include <numeric>
#include <functional>

namespace tmen{

template<typename T>
T sum(const std::vector<T>& src, const Unsigned startIndex){
  if (src.size() == 0)
    return 0;
  return std::accumulate(src.begin() + startIndex, src.end(), T(0), std::plus<T>());
}

template<typename T>
T sum(const std::vector<T>& src, const Unsigned startIndex, const Unsigned nElem){
  if (src.size() == 0)
    return 0;
  return std::accumulate(src.begin() + startIndex, src.begin() + nElem, T(0), std::plus<T>());
}

template<typename T>
T prod(const std::vector<T>& src, const Unsigned startIndex){
  if (src.size() == 0)
    return 0;
  return std::accumulate(src.begin() + startIndex, src.end(), T(1), std::multiplies<T>());
}

template<typename T>
T prod(const std::vector<T>& src, const Unsigned startIndex, const Unsigned nElem){
  if (src.size() == 0)
    return 0;
  return std::accumulate(src.begin() + startIndex, src.begin() + nElem, T(1), std::multiplies<T>());
}

template<typename T>
void ElemwiseSum(const std::vector<T>& src1, const std::vector<T>& src2, std::vector<T>& out){
  std::transform(src1.begin(), src1.end(), src2.begin(), out.begin(), std::plus<T>());
}

template<typename T>
std::vector<T> ElemwiseSum(const std::vector<T>& src1, const std::vector<T>& src2){
    std::vector<T> ret(src1.size());
    ElemwiseSum(src1, src2, ret);
    return ret;
}

template<typename T>
void ElemwiseSubtract(const std::vector<T>& src1, const std::vector<T>& src2, std::vector<T>& out){
  std::transform(src1.begin(), src1.end(), src2.begin(), out.begin(), std::minus<T>());
}

template<typename T>
std::vector<T> ElemwiseSubtract(const std::vector<T>& src1, const std::vector<T>& src2){
  std::vector<T> ret(src1.size());
  ElemwiseSubtract(src1, src2, ret);
  return ret;
}

template<typename T>
void ElemwiseProd(const std::vector<T>& src1, const std::vector<T>& src2, std::vector<T>& out){
  std::transform(src1.begin(), src1.end(), src2.begin(), out.begin(), std::multiplies<T>());
}

template<typename T>
std::vector<T> ElemwiseProd(const std::vector<T>& src1, const std::vector<T>& src2){
  std::vector<T> ret(src1.size());
  ElemwiseProd(src1, src2, ret);
  return ret;
}

template<typename T>
void ElemwiseDivide(const std::vector<T>& src1, const std::vector<T>& src2, std::vector<T>& out){
  std::transform(src1.begin(), src1.end(), src2.begin(), out.begin(), std::divides<T>());
}

template<typename T>
std::vector<T> ElemwiseDivide(const std::vector<T>& src1, const std::vector<T>& src2){
  std::vector<T> ret(src1.size());
  ElemwiseDivide(src1, src2, ret);
  return ret;
}

template<typename T>
void ElemwiseMod(const std::vector<T>& src1, const std::vector<T>& src2, std::vector<T>& out){
  std::transform(src1.begin(), src1.end(), src2.begin(), out.begin(), std::modulus<T>());
}

template<typename T>
std::vector<T> ElemwiseMod(const std::vector<T>& src1, const std::vector<T>& src2){
  std::vector<T> ret(src1.size());
  ElemwiseMod(src1, src2, ret);
  return ret;
}

template<typename T>
bool AnyNonNegativeElem(const std::vector<T>& vec){
  Unsigned i;
  for(i = 0; i < vec.size(); i++)
    if(vec[i] >= 0)
      return true;
  return false;
}

template<typename T>
bool AnyNonPositiveElem(const std::vector<T>& vec){
  Unsigned i;
  for(i = 0; i < vec.size(); i++)
    if(vec[i] <= 0)
      return true;
  return false;
}

template<typename T>
bool AnyPositiveElem(const std::vector<T>& vec){
  Unsigned i;
  for(i = 0; i < vec.size(); i++)
    if(vec[i] > 0)
      return true;
  return false;
}

template<typename T>
bool AnyNegativeElem(const std::vector<T>& vec){
  Unsigned i;
  for(i = 0; i < vec.size(); i++)
    if(vec[i] < 0)
      return true;
  return false;
}

template<typename T>
bool ElemwiseLessThan(const std::vector<T>& vec1, const std::vector<T>& vec2){
  if(vec1.size() != vec2.size())
    tmen::LogicError("Vector element-wise comparison must have matching sizes");

  Unsigned i;
  for(i = 0; i < vec1.size(); i++)
    if(vec1[i] >= vec2[i])
      return false;
  return true;
}

template<typename T>
bool ElemwiseLessThanEqualTo(const std::vector<T>& vec1, const std::vector<T>& vec2){
  if(vec1.size() != vec2.size())
    tmen::LogicError("Vector element-wise comparison must have matching sizes");

  Unsigned i;
  for(i = 0; i < vec1.size(); i++)
    if(vec1[i] > vec2[i])
      return false;
  return true;
}

template<typename T>
bool AnyElemwiseGreaterThan(const std::vector<T>& vec1, const std::vector<T>& vec2){
	if(vec1.size() != vec2.size())
		tmen::LogicError("Vector element-wise comparison must have matching sizes");

	Unsigned i;
	for(i = 0; i < vec1.size(); i++)
		if(vec1[i] > vec2[i])
			return true;
	return false;
}

template<typename T>
bool AnyElemwiseGreaterThanEqualTo(const std::vector<T>& vec1, const std::vector<T>& vec2){
    if(vec1.size() != vec2.size())
        tmen::LogicError("Vector element-wise comparison must have matching sizes");

    Unsigned i;
    for(i = 0; i < vec1.size(); i++)
        if(vec1[i] >= vec2[i])
            return true;
    return false;
}

template<typename T>
bool AnyZeroElem(const std::vector<T>& vec){
  Unsigned i;
  for(i = 0; i < vec.size(); i++)
    if(vec[i] == 0)
      return true;
  return false;
}

bool AnyFalseElem(const std::vector<bool>& vec){
  Unsigned i;
  for(i = 0; i < vec.size(); i++)
    if(vec[i] == false)
      return true;
  return false;
}

template<typename T>
bool AnyElemwiseNotEqual(const std::vector<T>& vec1, const std::vector<T>& vec2){
  if(vec1.size() != vec2.size())
	tmen::LogicError("Vector element-wise comparison must have matching sizes");

  Unsigned i;
  for(i = 0; i < vec1.size(); i++)
    if(vec1[i] != vec2[i])
      return true;
  return false;
}

template<typename T>
bool EqualUnderPermutation(const std::vector<T>& vec1, const std::vector<T>& vec2){
    if(vec1.size() != vec2.size())
        tmen::LogicError("Vector Permutation check must have same sized vectors");

    Unsigned i;
    for(i = 0; i < vec1.size(); i++){
        if(std::find(vec2.begin(), vec2.end(), vec1[i]) == vec2.end())
            LogicError("EqualUnderPermutation: element in vec1 not found in vec2");
    }
    for(i = 0; i < vec2.size(); i++){
        if(std::find(vec1.begin(), vec1.end(), vec2[i]) == vec1.end())
            LogicError("EqualUnderPermutation: element in vec2 not found in vec1");
    }
    return true;
}

template<typename T>
std::vector<T> PermuteVector(const std::vector<T>& vec, const Permutation& perm){
    return FilterVector(vec, perm);
}

template<typename T>
std::vector<T> FilterVector(const std::vector<T>& vec, const std::vector<Unsigned>& filter){
	Unsigned i;
	std::vector<T> ret;
	for(i = 0; i < filter.size(); i++){
		if(filter[i] < 0 || filter[i] >= vec.size())
			continue;
		else
			ret.push_back(vec[filter[i]]);
	}
	return ret;
}

template<typename T>
std::vector<T> NegFilterVector(const std::vector<T>& vec, const std::vector<Unsigned>& filter){
    Unsigned i;
    std::vector<T> ret;
    std::vector<Unsigned> sortedFilter = filter;
    std::sort(sortedFilter.begin(), sortedFilter.end());
    Unsigned whichFilter = 0;
    for(i = 0; i < vec.size(); i++){
        if(whichFilter >= sortedFilter.size() || i != sortedFilter[whichFilter])
            ret.push_back(vec[i]);
        else
            whichFilter++;
    }
    return ret;
}

template<typename T>
bool IsSame(const std::vector<T>& vec1, const std::vector<T>& vec2){
    Unsigned i;

    if(vec1.size() != vec2.size())
        return false;
    for(i = 0; i < vec1.size(); i++){
        if(vec1[i] != vec2[i])
            return false;
    }
    return true;
}

template<typename T>
bool IsSuffix(const std::vector<T>& vec1, const std::vector<T>& vec2){
    Unsigned i;
    Unsigned vec1Len = vec1.size();
    Unsigned vec2Len = vec2.size();

    if(vec1Len > vec2Len)
        return false;
    for(i = 0; i < vec1Len; i++){
        if(vec1[vec1Len - i] != vec2[vec2Len - i])
            return false;
    }
    return true;
}

template<typename T>
bool IsPrefix(const std::vector<T>& vec1, const std::vector<T>& vec2){
    Unsigned i;

    if(vec1.size() > vec2.size())
        return false;
    for(i = 0; i < vec1.size(); i++){
        if(vec1[i] != vec2[i])
            return false;
    }
    return true;
}

template<typename T>
bool Contains(const std::vector<T>& vec, const T& val){
	return std::find(vec.begin(), vec.end(), val) != vec.end();
}

template<typename T>
int IndexOf(const std::vector<T>& vec, const T& val){
	Unsigned i;
	int ret = -1;
	for(i = 0; i < vec.size(); i++)
		if(vec[i] == val)
			return i;
	return ret;
}

template<typename T>
std::vector<T> Unique(const std::vector<T>& vec){
	std::vector<T> ret = vec;
	SortVector(ret);
	ret.erase(std::unique(ret.begin(), ret.end()), ret.end());
	return ret;
}

template<typename T>
std::vector<T> GetSuffix_(const std::vector<T>& vec1, const std::vector<T>& vec2){
	Unsigned i = 0;
    std::vector<T> ret;
    if(vec1.size() != 0){
		for(i = 0; i < vec1.size(); i++)
			if(vec1[i] != vec2[i])
				break;
    }
    ret.insert(ret.end(), vec2.begin() + i, vec2.end());

    return ret;
}

template<typename T>
std::vector<T> GetSuffix(const std::vector<T>& vec1, const std::vector<T>& vec2){
	if(vec1.size() <= vec2.size())
		return GetSuffix_(vec1, vec2);
	return GetSuffix(vec2, vec1);
}

template<typename T>
std::vector<T> ConcatenateVectors(const std::vector<T>& vec1, const std::vector<T>& vec2){
    std::vector<T> ret(vec1);
    ret.insert(ret.end(), vec2.begin(), vec2.end());

    return ret;
}

template<typename T>
T Min(const std::vector<T>& vec){
    return *std::min_element(vec.begin(), vec.end());
}

template<typename T>
T Max(const std::vector<T>& vec){
    return *std::max_element(vec.begin(), vec.end());
}

template<typename T>
Permutation DeterminePermutation(const std::vector<T>& ref, const std::vector<T>& vec){
    if(ref.size() != vec.size())
        LogicError("reference vector and permuted vector are of different sizes");
    Permutation ret(ref.size());
    typename std::vector<T>::const_iterator begin = ref.begin();
    typename std::vector<T>::const_iterator end = ref.end();

    Unsigned i;
    for(i = 0; i < vec.size(); i++){
        ret[i] = std::find(begin, end, vec[i]) - begin;
    }

    return ret;
}

Permutation DetermineInversePermutation(const Permutation& perm){
    Unsigned i;
    Permutation basePerm(perm.size());
    for(i = 0; i < basePerm.size(); i++)
        basePerm[i] = i;
    return DeterminePermutation(perm, basePerm);
}

template<typename T>
std::vector<T> DiffVector(const std::vector<T>& vec1, const std::vector<T>& vec2){
	Unsigned i;
	std::vector<T> ret = vec1;
	for(i = 0; i < vec2.size(); i++)
		ret.erase(std::remove(ret.begin(), ret.end(), vec2[i]), ret.end());
	return ret;
}

template<typename T>
void SortVector(std::vector<T>& vec1){
	std::sort(vec1.begin(), vec1.end());
}

//Non-template functions
//bool AnyFalseElem(const std::vector<bool>& vec);
#define PROTO(T) \
	template T sum(const std::vector<T>& src, const Unsigned startIndex); \
	template T sum(const std::vector<T>& src, const Unsigned startIndex, const Unsigned endIndex); \
	template T prod(const std::vector<T>& src, const Unsigned startIndex); \
    template T prod(const std::vector<T>& src, const Unsigned startIndex, const Unsigned endIndex); \
	template void ElemwiseSum(const std::vector<T>& src1, const std::vector<T>& src2, std::vector<T>& out); \
	template std::vector<T> ElemwiseSum(const std::vector<T>& src1, const std::vector<T>& src2); \
	template void ElemwiseSubtract(const std::vector<T>& src1, const std::vector<T>& src2, std::vector<T>& out); \
	template std::vector<T> ElemwiseSubtract(const std::vector<T>& src1, const std::vector<T>& src2); \
	template void ElemwiseProd(const std::vector<T>& src1, const std::vector<T>& src2, std::vector<T>& out); \
	template std::vector<T> ElemwiseProd(const std::vector<T>& src1, const std::vector<T>& src2); \
	template void ElemwiseDivide(const std::vector<T>& src1, const std::vector<T>& src2, std::vector<T>& out); \
	template std::vector<T> ElemwiseDivide(const std::vector<T>& src1, const std::vector<T>& src2); \
	template bool AnyNonNegativeElem(const std::vector<T>& vec); \
	template bool AnyNonPositiveElem(const std::vector<T>& vec); \
	template bool AnyPositiveElem(const std::vector<T>& vec); \
	template bool AnyNegativeElem(const std::vector<T>& vec); \
	template bool ElemwiseLessThan(const std::vector<T>& vec1, const std::vector<T>& vec2); \
	template bool ElemwiseLessThanEqualTo(const std::vector<T>& vec1, const std::vector<T>& vec2); \
	template bool AnyElemwiseGreaterThan(const std::vector<T>& vec1, const std::vector<T>& vec2); \
	template bool AnyElemwiseGreaterThanEqualTo(const std::vector<T>& vec1, const std::vector<T>& vec2); \
	template bool AnyZeroElem(const std::vector<T>& vec); \
	template bool AnyElemwiseNotEqual(const std::vector<T>& vec1, const std::vector<T>& vec2); \
	template bool EqualUnderPermutation(const std::vector<T>& vec1, const std::vector<T>& vec2); \
	template std::vector<T> PermuteVector(const std::vector<T>& vec, const std::vector<Unsigned>& filter); \
	template std::vector<T> FilterVector(const std::vector<T>& vec, const std::vector<Unsigned>& filter); \
	template std::vector<T> NegFilterVector(const std::vector<T>& vec, const std::vector<Unsigned>& filter); \
	template bool IsSame(const std::vector<T>& vec1, const std::vector<T>& vec2); \
	template bool IsSuffix(const std::vector<T>& vec1, const std::vector<T>& vec2); \
	template bool IsPrefix(const std::vector<T>& vec1, const std::vector<T>& vec2); \
	template int  IndexOf(const std::vector<T>& vec, const T& val); \
	template bool Contains(const std::vector<T>& vec, const T& val); \
	template std::vector<T> Unique(const std::vector<T>& vec); \
	template std::vector<T> GetSuffix_(const std::vector<T>& vec1, const std::vector<T>& vec2); \
	template std::vector<T> GetSuffix(const std::vector<T>& vec1, const std::vector<T>& vec2); \
	template std::vector<T> ConcatenateVectors(const std::vector<T>& vec1, const std::vector<T>& vec2); \
	template T Min(const std::vector<T>& vec); \
	template T Max(const std::vector<T>& vec); \
    template Permutation DeterminePermutation(const std::vector<T>& ref, const std::vector<T>& vec); \
    template std::vector<T> DiffVector(const std::vector<T>& vec1, const std::vector<T>& vec2); \
	template void SortVector(std::vector<T>& vec1);

PROTO(Unsigned)
PROTO(Int)
PROTO(float)
PROTO(double)
PROTO(char)

#define PROTOMOD(T) \
        template void ElemwiseMod(const std::vector<T>& src1, const std::vector<T>& src2, std::vector<T>& out); \
        template std::vector<T> ElemwiseMod(const std::vector<T>& src1, const std::vector<T>& src2);
PROTOMOD(Unsigned)
PROTOMOD(Int)


}
