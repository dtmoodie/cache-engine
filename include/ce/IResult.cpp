#include <ce/IResult.hpp>

namespace ce
{
    IResult::~IResult()
    {
    }

    size_t IResult::hash() const
    {
        return m_hash;
    }

    void IResult::setHash(size_t v)
    {
        m_hash = v;
    }
}
