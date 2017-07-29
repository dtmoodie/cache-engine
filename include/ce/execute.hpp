#pragma once
#include <ce/export.hpp>
#include <ce/OutputPack.hpp>
namespace ce {

template<class ... FArgs, class... Args>
typename std::enable_if<OutputPack<void, std::remove_reference_t<Args>...>::OUTPUT_COUNT != 0>::type exec(void(*func)(FArgs...), Args&&...args);

template<class R, class ... FArgs, class... Args>
HashedOutput<R> exec(R(*func)(FArgs...), Args&&...args);

}
#include "detail/execute.hpp"