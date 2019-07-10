#pragma once
#include "export.hpp"

namespace ce
{
    struct CE_EXPORT IResult
    {
        virtual ~IResult();

        size_t hash() const;
        void setHash(size_t);

      private:
        size_t m_hash;
    };
}
