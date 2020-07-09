#include "output.hpp"

namespace ce
{

    void setHash(HashedBase& obj, size_t hash)
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
} // namespace ce
