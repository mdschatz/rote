#include "rote.hpp"
namespace rote{

Permutation::Permutation()
: perm_()
{ }

Permutation::Permutation(Unsigned order)
: perm_(order)
{
	for(Unsigned i = 0; i < order; i++)
		perm_[i] = i;
}
Permutation::Permutation(std::initializer_list<Unsigned> list)
: perm_(list)
{ CheckIsValid(); }

Permutation::Permutation(const std::vector<Unsigned>& perm)
: perm_(perm)
{ CheckIsValid(); }

Permutation::Permutation(const Permutation& perm)
: perm_(perm.perm_)
{ }

Permutation::~Permutation()
{ }

Permutation& Permutation::operator=(const Permutation& perm)
{
	if (this != &perm){
		perm_ = perm.perm_;
	}
	return *this;
}

void Permutation::CheckIsValid() const
{
	std::vector<Unsigned> maxVal(perm_.size(), perm_.size() - 1);
	if(AnyElemwiseGreaterThan(perm_, maxVal))
		LogicError("Could not construct permutation (value to large)");

	std::vector<Unsigned> unique = Unique(perm_);
	if(perm_.size() != unique.size())
		LogicError("Could not construct permutation (duplicate values exist");
}

const Unsigned& Permutation::operator[](std::size_t index) const
{ return perm_[index]; }

std::ostream& operator<<(std::ostream& o, const Permutation &perm)
{
	o << "perm:";
	for(int i = 0; i < perm.size(); i++)
		o << perm[i];
	o << std::endl;
	return o;
}

bool operator!=(const Permutation& lhs, const Permutation& rhs)
{
	return lhs.perm_ != rhs.perm_;
}

Permutation Permutation::InversePermutation() const
{
	std::vector<Unsigned> vals(perm_.size());
	for(int i = 0; i < perm_.size(); i++)
		vals[perm_[i]] = i;
	Permutation ret(vals);
	return ret;
}

Permutation Permutation::PermutationTo(const Permutation& perm) const
{
	Permutation ret(perm_, perm.perm_);
	return ret;
}

std::vector<Unsigned> Permutation::Entries() const
{ return perm_; }

Unsigned Permutation::size() const
{ return perm_.size(); }

}
//
// #define FULL(T) \
//     template Permutation::Permutation<T>;
//
// FULL(int)
// #ifndef DISABLE_FLOAT
// FULL(float)
// #endif
// FULL(double)
//
// #ifndef DISABLE_COMPLEX
// #ifndef DISABLE_FLOAT
// FULL(std::complex<float>)
// #endif
// FULL(std::complex<double>)
// #endif
