#pragma once
#include <ce/OutputPack.hpp>

namespace ce
{
    template <class... Args>
    constexpr int countOutputs(const Args&...)
    {
        return OutputPack<void, Args...>::OUTPUT_COUNT;
    }
}
