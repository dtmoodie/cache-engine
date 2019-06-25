#include "output.hpp"

namespace ce
{
size_t HashedBase::hash() const
{
    return m_hash;
}

void HashedBase::setHash(size_t val)
{
    m_hash = val;
}
}
