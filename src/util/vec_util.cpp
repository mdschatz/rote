
#include "tensormental/util/vec_util.hpp"
#include <numeric>

namespace elem{

template<typename T>
T prod(std::vector<T>& src){
  return std::accumulate(src.begin(), src.end(), T(1), std::multiplies<T>());
}

template int prod(std::vector<int>& src);
template float prod(std::vector<float>& src);
template double prod(std::vector<double>& src);

}
