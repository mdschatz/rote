#include "rote.hpp"

namespace rote{

template<typename T>
T sum(const std::vector<T>& src) {
  if (src.size() == 0)
    return 0;
  return std::accumulate(src.begin(), src.end(), T(0), std::plus<T>());
}

template<typename T>
T prod(const std::vector<T>& src, const Unsigned startIndex){
  if (src.size() == 0)
    return 0;
  return std::accumulate(src.begin() + startIndex, src.end(), T(1), std::multiplies<T>());
}

template<typename T>
std::vector<T> ElemwiseSum(const std::vector<T>& src1, const std::vector<T>& src2){
    std::vector<T> ret(src1.size());
    std::transform(src1.begin(), src1.end(), src2.begin(), ret.begin(), std::plus<T>());
    return ret;
}

template<typename T>
std::vector<T> ElemwiseSubtract(const std::vector<T>& src1, const std::vector<T>& src2){
  std::vector<T> ret(src1.size());
  std::transform(src1.begin(), src1.end(), src2.begin(), ret.begin(), std::minus<T>());
  return ret;
}

template<typename T>
std::vector<T> ElemwiseProd(const std::vector<T>& src1, const std::vector<T>& src2){
  std::vector<T> ret(src1.size());
  std::transform(src1.begin(), src1.end(), src2.begin(), ret.begin(), std::multiplies<T>());
  return ret;
}

template<typename T>
std::vector<T> ElemwiseDivide(const std::vector<T>& src1, const std::vector<T>& src2){
  std::vector<T> ret(src1.size());
  std::transform(src1.begin(), src1.end(), src2.begin(), ret.begin(), std::divides<T>());
  return ret;
}

template<typename T>
std::vector<T> ElemwiseMod(const std::vector<T>& src1, const std::vector<T>& src2){
  std::vector<T> ret(src1.size());
  std::transform(src1.begin(), src1.end(), src2.begin(), ret.begin(), std::modulus<T>());
  return ret;
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
    rote::LogicError("Vector element-wise comparison must have matching sizes");

  Unsigned i;
  for(i = 0; i < vec1.size(); i++)
    if(vec1[i] >= vec2[i])
      return false;
  return true;
}

template<typename T>
bool ElemwiseLessThanEqualTo(const std::vector<T>& vec1, const std::vector<T>& vec2){
  if(vec1.size() != vec2.size())
    rote::LogicError("Vector element-wise comparison must have matching sizes");

  Unsigned i;
  for(i = 0; i < vec1.size(); i++)
    if(vec1[i] > vec2[i])
      return false;
  return true;
}

template<typename T>
bool AnyElemwiseGreaterThan(const std::vector<T>& vec1, const std::vector<T>& vec2){
	if(vec1.size() != vec2.size())
		rote::LogicError("Vector element-wise comparison must have matching sizes");

	Unsigned i;
	for(i = 0; i < vec1.size(); i++)
		if(vec1[i] > vec2[i])
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

template<typename T>
bool AnyElemwiseNotEqual(const std::vector<T>& vec1, const std::vector<T>& vec2){
  if(vec1.size() != vec2.size())
	rote::LogicError("Vector element-wise comparison must have matching sizes");

  Unsigned i;
  for(i = 0; i < vec1.size(); i++)
    if(vec1[i] != vec2[i])
      return true;
  return false;
}

template<typename T>
std::vector<T> PermuteVector(const std::vector<T>& vec, const Permutation& perm){
    return FilterVector(vec, perm.Entries());
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
    SortVector(sortedFilter);
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
std::vector<Unsigned> IndicesOf(const std::vector<T>& vec, const std::vector<T>& vals) {
	std::vector<Unsigned> ret(vals.size());
	for(Unsigned i = 0; i < vals.size(); i++) {
    int index = IndexOf(vec, vals[i]);
    if (index < 0) {
      LogicError("val not found in vec");
    }
    ret[i] = index;
  }
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
std::vector<T> GetSuffix(const std::vector<T>& vec1, const std::vector<T>& vec2) {
  if (vec1.size() > vec2.size()) {
    return GetSuffix(vec2, vec2);
  }

  // TODO: This code is weird.  If vec1.size() == 0 then we return all of vec2?
  Unsigned i = 0;
  std::vector<T> ret;
  if(vec1.size() != 0){
    Unsigned i;
    for(i = vec1.size() - 1; i < vec1.size(); i--)
      if(vec1[i] != vec2[i])
        break;
  }
  ret.insert(ret.end(), vec2.begin() + i, vec2.end());

  return ret;
}

template<typename T>
std::vector<T> GetPrefix(const std::vector<T>& vec1, const std::vector<T>& vec2){
  if (vec1.size() > vec2.size()) {
    return GetPrefix(vec2, vec2);
  }

  // TODO: This code is weird.  If vec1.size() == 0 then we return all of vec2?
  Unsigned i = 0;
  std::vector<T> ret;
  if(vec1.size() != 0){
    for(i = 0; i < vec1.size(); i++)
      if(vec1[i] != vec2[i])
        break;
  }
  ret.insert(ret.end(), vec2.begin(), vec2.begin() + i);

  return ret;
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
    std::vector<Unsigned> permVec(ref.size());

    Unsigned i;
    for(i = 0; i < vec.size(); i++){
        permVec[i] = IndexOf(ref, vec[i]);
    }

    Permutation ret(permVec);
    return ret;
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
std::vector<T> IsectVector(const std::vector<T>& vec1, const std::vector<T>& vec2){
	Unsigned i;
	std::vector<T> ret;
  for(i = 0; i < vec1.size(); i++) {
    T val = vec1[i];
    if(Contains(vec2, val)) {
      ret.push_back(val);
    }
  }
	return ret;
}

template<typename T>
void SortVector(std::vector<T>& vec1) {
	std::sort(vec1.begin(), vec1.end());
}

//Non-template functions
#define PROTO(T) \
	template T sum(const std::vector<T>& src); \
	template T prod(const std::vector<T>& src, const Unsigned startIndex); \
	template std::vector<T> ElemwiseSum(const std::vector<T>& src1, const std::vector<T>& src2); \
	template std::vector<T> ElemwiseSubtract(const std::vector<T>& src1, const std::vector<T>& src2); \
	template std::vector<T> ElemwiseProd(const std::vector<T>& src1, const std::vector<T>& src2); \
	template std::vector<T> ElemwiseDivide(const std::vector<T>& src1, const std::vector<T>& src2); \
	template bool AnyPositiveElem(const std::vector<T>& vec); \
	template bool AnyNegativeElem(const std::vector<T>& vec); \
	template bool ElemwiseLessThan(const std::vector<T>& vec1, const std::vector<T>& vec2); \
	template bool ElemwiseLessThanEqualTo(const std::vector<T>& vec1, const std::vector<T>& vec2); \
	template bool AnyElemwiseGreaterThan(const std::vector<T>& vec1, const std::vector<T>& vec2); \
	template bool AnyZeroElem(const std::vector<T>& vec); \
	template bool AnyElemwiseNotEqual(const std::vector<T>& vec1, const std::vector<T>& vec2); \
	template std::vector<T> PermuteVector(const std::vector<T>& vec, const Permutation& filter); \
	template std::vector<T> FilterVector(const std::vector<T>& vec, const std::vector<Unsigned>& filter); \
	template std::vector<T> NegFilterVector(const std::vector<T>& vec, const std::vector<Unsigned>& filter); \
	template int  IndexOf(const std::vector<T>& vec, const T& val); \
  template std::vector<Unsigned> IndicesOf(const std::vector<T>& vec, const std::vector<T>& vals); \
	template bool Contains(const std::vector<T>& vec, const T& val); \
	template std::vector<T> Unique(const std::vector<T>& vec); \
	template std::vector<T> GetSuffix(const std::vector<T>& vec1, const std::vector<T>& vec2); \
	template std::vector<T> GetPrefix(const std::vector<T>& vec1, const std::vector<T>& vec2); \
	template std::vector<T> ConcatenateVectors(const std::vector<T>& vec1, const std::vector<T>& vec2); \
	template T Min(const std::vector<T>& vec); \
	template T Max(const std::vector<T>& vec); \
  template Permutation DeterminePermutation(const std::vector<T>& ref, const std::vector<T>& vec); \
  template std::vector<T> DiffVector(const std::vector<T>& vec1, const std::vector<T>& vec2); \
  template std::vector<T> IsectVector(const std::vector<T>& vec1, const std::vector<T>& vec2); \
	template void SortVector(std::vector<T>& vec1);

PROTO(Unsigned)
PROTO(Int)
PROTO(float)
PROTO(double)
PROTO(char)

#define PROTOMOD(T) \
  template std::vector<T> ElemwiseMod(const std::vector<T>& src1, const std::vector<T>& src2);
PROTOMOD(Unsigned)
PROTOMOD(Int)


}
