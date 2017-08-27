#pragma once
#include <ce/OutputPack.hpp>

namespace ce{
    template<class R, class ... FArgs, class... Args>
    constexpr int countOutputs(R(*func)(FArgs...), const Args&... args){
        return OutputPack<R(FArgs...), Args...>::OUTPUT_COUNT;
    }
    template<class T, class R, class... FArgs, class... Args>
    constexpr int countOutputs(R(T::*func)(FArgs...), const Args&... args){
        return OutputPack<R(FArgs...), Args...>::OUTPUT_COUNT;
    }
}