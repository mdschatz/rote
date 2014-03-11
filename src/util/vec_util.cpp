
#include "tensormental/util/vec_util.hpp"
#include "tensormental/core/error_decl.hpp"
#include <algorithm>
#include <numeric>
#include <functional>

namespace tmen{

template<typename T>
T prod(const std::vector<T>& src, const int startIndex){
  if (src.size() == 0)
    return 0;
  return std::accumulate(src.begin() + startIndex, src.end(), T(1), std::multiplies<T>());
}

template<typename T>
T prod(const std::vector<T>& src, const int startIndex, const int endIndex){
  if (src.size() == 0)
    return 0;
  return std::accumulate(src.begin() + startIndex, src.begin() + endIndex, T(1), std::multiplies<T>());
}

template<typename T>
void ElemwiseSum(const std::vector<T>& src1, const std::vector<T>& src2, std::vector<T>& out){
  std::transform(src1.begin(), src1.end(), src2.begin(), out.begin(), std::plus<T>());
}

template<typename T>
bool AnyNonNegativeElem(const std::vector<T>& vec){
  for(int i = 0; i < vec.size(); i++)
    if(vec[i] >= 0)
      return true;
  return false;
}

template<typename T>
bool AnyNonPositiveElem(const std::vector<T>& vec){
  for(int i = 0; i < vec.size(); i++)
    if(vec[i] <= 0)
      return true;
  return false;
}

template<typename T>
bool AnyNegativeElem(const std::vector<T>& vec){
  for(int i = 0; i < vec.size(); i++)
    if(vec[i] < 0)
      return true;
  return false;
}

template<typename T>
bool ElemwiseLessThan(const std::vector<T>& vec1, const std::vector<T>& vec2){
  if(vec1.size() != vec2.size())
    tmen::LogicError("Vector element-wise comparison must have matching sizes");

  for(int i = 0; i < vec1.size(); i++)
    if(vec1[i] >= vec2[i])
      return false;
  return true;
}

template<typename T>
bool AnyElemwiseGreaterThan(const std::vector<T>& vec1, const std::vector<T>& vec2){
	if(vec1.size() != vec2.size())
		tmen::LogicError("Vector element-wise comparison must have matching sizes");

	for(int i = 0; i < vec1.size(); i++)
		if(vec1[i] > vec2[i])
			return true;
	return false;
}

template<typename T>
bool AnyZeroElem(const std::vector<T>& vec){
  for(int i = 0; i < vec.size(); i++)
    if(vec[i] == 0)
      return true;
  return false;
}

template<typename T>
bool AnyElemwiseNotEqual(const std::vector<T>& vec1, const std::vector<T>& vec2){
  if(vec1.size() != vec2.size())
	tmen::LogicError("Vector element-wise comparison must have matching sizes");

  for(int i = 0; i < vec1.size(); i++)
    if(vec1[i] != vec2[i])
      return true;
  return false;
}

template<typename T>
bool EqualUnderPermutation(const std::vector<T>& vec1, const std::vector<T>& vec2){
    if(vec1.size() != vec2.size())
        tmen::LogicError("Vector Permutation check must have same sized vectors");

    for(int i = 0; i < vec1.size(); i++){
        if(std::find(vec2.begin(), vec2.end(), vec1[i]) == vec2.end())
            LogicError("EqualUnderPermutation: element in vec1 not found in vec2");
    }
    for(int i = 0; i < vec2.size(); i++){
        if(std::find(vec1.begin(), vec1.end(), vec2[i]) == vec1.end())
            LogicError("EqualUnderPermutation: element in vec2 not found in vec1");
    }
    return true;
}

template<typename T>
std::vector<T> FilterVector(const std::vector<T>& vec, const std::vector<int>& filter){
	int i;
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
std::vector<T> DeterminePermutation(const std::vector<T>& ref, const std::vector<T>& vec){
    if(ref.size() != vec.size())
        LogicError("reference vector and permuted vector are of different sizes");
    std::vector<T> ret(ref.size());
    typename std::vector<T>::const_iterator begin = ref.begin();
    typename std::vector<T>::const_iterator end = ref.end();

    for(int i = 0; i < vec.size(); i++){
        ret[i] = std::find(begin, end, vec[i]) - begin;
    }

    return ret;
}

#define PROTO(T) \
	template T prod(const std::vector<T>& src, const int startIndex); \
    template T prod(const std::vector<T>& src, const int startIndex, const int endIndex); \
	template void ElemwiseSum(const std::vector<T>& src1, const std::vector<T>& src2, std::vector<T>& out); \
	template bool AnyNonNegativeElem(const std::vector<T>& vec); \
	template bool AnyNonPositiveElem(const std::vector<T>& vec); \
	template bool AnyNegativeElem(const std::vector<T>& vec); \
	template bool ElemwiseLessThan(const std::vector<T>& vec1, const std::vector<T>& vec2); \
	template bool AnyElemwiseGreaterThan(const std::vector<T>& vec1, const std::vector<T>& vec2); \
	template bool AnyZeroElem(const std::vector<T>& vec); \
	template bool AnyElemwiseNotEqual(const std::vector<T>& vec1, const std::vector<T>& vec2); \
	template bool EqualUnderPermutation(const std::vector<T>& vec1, const std::vector<T>& vec2); \
	template std::vector<T> FilterVector(const std::vector<T>& vec, const std::vector<int>& filter); \
	template std::vector<T> DeterminePermutation(const std::vector<T>& ref, const std::vector<T>& vec);

PROTO(int)
PROTO(float)
PROTO(double)

}
