
#include "tensormental/util/vec_util.hpp"
#include "tensormental/core/error_decl.hpp"
#include <algorithm>
#include <numeric>
#include <functional>

namespace tmen{

template<typename T>
T prod(const std::vector<T>& src){
  if (src.size() == 0)
    return 0;
  return std::accumulate(src.begin(), src.end(), T(1), std::multiplies<T>());
}

template<typename T>
void ElemwiseSum(const std::vector<T>& src1, const std::vector<T>& src2, std::vector<T>& out){
  std::transform(src1.begin(), src1.end(), src2.begin(), out.begin(), std::plus<T>());
}

template<typename T>
bool AnyNonNegativeElem(const std::vector<T>& vec){
  bool test;
  for(int i = 0; i < vec.size(); i++)
    if(vec[i] >= 0)
      return true;
  return false;
}

template<typename T>
bool AnyNegativeElem(const std::vector<T>& vec){
  bool test;
  for(int i = 0; i < vec.size(); i++)
    if(vec[i] < 0)
      return true;
  return false;
}

template<typename T>
bool ElemwiseLessThan(const std::vector<T>& vec1, const std::vector<T>& vec2){
  if(vec1.size() != vec2.size())
    tmen::LogicError("Vector element-wise comparison must have matching sizes");

  bool test;
  for(int i = 0; i < vec1.size(); i++)
    if(vec1[i] >= vec2[i])
      return false;
  return true;
}

template<typename T>
bool AnyElemwiseGreaterThan(const std::vector<T>& vec1, const std::vector<T>& vec2){
	if(vec1.size() != vec2.size())
		tmen::LogicError("Vector element-wise comparison must have matching sizes");

	bool test;
	for(int i = 0; i < vec1.size(); i++)
		if(vec1[i] > vec2[i])
			return true;
	return false;
}

template<typename T>
bool AnyZeroElem(const std::vector<T>& vec){
  bool test;
  for(int i = 0; i < vec.size(); i++)
    if(vec[i] == 0)
      return true;
  return false;
}

template<typename T>
bool AnyElemwiseNotEqual(const std::vector<T>& vec1, const std::vector<T>& vec2){
  if(vec1.size() != vec2.size())
	tmen::LogicError("Vector element-wise comparison must have matching sizes");
  bool test;
  for(int i = 0; i < vec1.size(); i++)
    if(vec1[i] != vec2[i])
      return true;
  return false;
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

#define PROTO(T) \
	template T prod(const std::vector<T>& src); \
	template void ElemwiseSum(const std::vector<T>& src1, const std::vector<T>& src2, std::vector<T>& out); \
	template bool AnyNonNegativeElem(const std::vector<T>& vec); \
	template bool AnyNegativeElem(const std::vector<T>& vec); \
	template bool ElemwiseLessThan(const std::vector<T>& vec1, const std::vector<T>& vec2); \
	template bool AnyElemwiseGreaterThan(const std::vector<T>& vec1, const std::vector<T>& vec2); \
	template bool AnyZeroElem(const std::vector<T>& vec); \
	template bool AnyElemwiseNotEqual(const std::vector<T>& vec1, const std::vector<T>& vec2); \
	template std::vector<T> FilterVector(const std::vector<T>& vec, const std::vector<int>& filter);

PROTO(int)
PROTO(float)
PROTO(double)

}
