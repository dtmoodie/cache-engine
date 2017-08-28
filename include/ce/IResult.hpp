#pragma once
#include "export.hpp"

namespace ce {
struct CE_EXPORT IResult {
    virtual ~IResult();
    virtual size_t getDynamicSize() const;
};

}