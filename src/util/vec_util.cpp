
#include "tensormental/util/vec_util.hpp"
#include "tensormental/core/error_decl.hpp"
#include <numeric>
#include <functional>

namespace tmen{

template<typename T>
T prod(const std::vector<T>& src){
  return std::accumulate(src.begin(), src.end(), T(1), std::multiplies<T>());
}

//template Int prod(const std::vector<Int>& src);
template int prod(const std::vector<int>& src);
template float prod(const std::vector<float>& src);
template double prod(const std::vector<double>& src);

template<typename T>
void ElemwiseSum(const std::vector<T>& src1, const std::vector<T>& src2, std::vector<T>& out){
  transform(src1.begin(), src1.end(), src2.begin(), out.begin(), std::plus<T>());
}


//template Int prod(const std::vector<Int>& src);
template void ElemwiseSum(const std::vector<int>& src1, const std::vector<int>& src2, std::vector<int>& out);
template void ElemwiseSum(const std::vector<float>& src1, const std::vector<float>& src2, std::vector<float>& out);
template void ElemwiseSum(const std::vector<double>& src1, const std::vector<double>& src2, std::vector<double>& out);

template<typename T>
bool AnyNonNegativeElem(const std::vector<T>& vec){
  bool test;
  for(int i = 0; i < vec.size(); i++)
    if(vec[i] >= 0)
      return true;
  return false;
}

template bool AnyNonNegativeElem(const std::vector<int>& vec);
template bool AnyNonNegativeElem(const std::vector<float>& vec);
template bool AnyNonNegativeElem(const std::vector<double>& vec);

template<typename T>
bool AnyNegativeElem(const std::vector<T>& vec){
  bool test;
  for(int i = 0; i < vec.size(); i++)
    if(vec[i] < 0)
      return true;
  return false;
}

template bool AnyNegativeElem(const std::vector<int>& vec);
template bool AnyNegativeElem(const std::vector<float>& vec);
template bool AnyNegativeElem(const std::vector<double>& vec);

template<typename T>
bool ElemwiseLessThan(const std::vector<T>& vec1, const std::vector<T>& vec2){
  if(vec1.size() != vec2.size())
    tmen::LogicError("Vector tmenent-wise comparison must have matching sizes");

  bool test;
  for(int i = 0; i < vec1.size(); i++)
    if(vec1[i] >= vec2[i])
      return false;
  return true;
}

template bool ElemwiseLessThan(const std::vector<int>& vec1, const std::vector<int>& vec2);
template bool ElemwiseLessThan(const std::vector<float>& vec1, const std::vector<float>& vec2);
template bool ElemwiseLessThan(const std::vector<double>& vec1, const std::vector<double>& vec2);

template<typename T>
bool AnyZeroElem(const std::vector<T>& vec){
  bool test;
  for(int i = 0; i < vec.size(); i++)
    if(vec[i] == 0)
      return true;
  return false;
}

template bool AnyZeroElem(const std::vector<int>& vec);
template bool AnyZeroElem(const std::vector<float>& vec);
template bool AnyZeroElem(const std::vector<double>& vec);

template<typename T>
bool AnyElemwiseNotEqual(const std::vector<T>& vec1, const std::vector<T>& vec2){
  //TODO: Error checking
  bool test;
  for(int i = 0; i < vec1.size(); i++)
    if(vec1[i] != vec2[i])
      return true;
  return false;
}

template bool AnyElemwiseNotEqual(const std::vector<int>& vec1, const std::vector<int>& vec2);
template bool AnyElemwiseNotEqual(const std::vector<float>& vec1, const std::vector<float>& vec2);
template bool AnyElemwiseNotEqual(const std::vector<double>& vec1, const std::vector<double>& vec2);
}
