#include "output.hpp"

namespace ce
{

    void setHash(size_t hash, HashedBase& obj)
    {
        obj.setHash(hash);
    }

    size_t HashedBase::hash() const
    {
        return m_hash;
    }

    void HashedBase::setHash(size_t val)
    {
        m_hash = val;
    }
}
