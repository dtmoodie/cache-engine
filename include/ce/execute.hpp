#pragma once
#include <ce/ICacheEngine.hpp>
#include <ce/OutputPack.hpp>
#include <ce/TResult.hpp>
#include <ce/export.hpp>
#include <ce/hash.hpp>
#include <ce/input.hpp>
#include <ce/output.hpp>

#include <ct/type_traits.hpp>
namespace ce
{
    template <class... FArgs, class... Args>
    typename std::enable_if<OutputPack<void, ct::remove_reference_t<Args>...>::OUTPUT_COUNT != 0>::type
    exec(void (*func)(FArgs...), Args&&... args)
    {
        ICacheEngine* eng = ICacheEngine::instance();
        if (eng)
        {
            eng->exec(func, std::forward<Args>(args)...);
        }
        else
        {
            return func(ce::get(std::forward<Args>(args))...);
        }
    }

    template <class R, class... FArgs, class... Args>
    HashedOutput<R> exec(R (*func)(FArgs...), Args&&... args)
    {
        ICacheEngine* eng = ICacheEngine::instance();
        if (eng)
        {
            return eng->exec(func, std::forward<Args>(args)...);
        }
        R ret = func(ce::get(std::forward<Args>(args))...);
        return HashedOutput<R>(std::move(ret), 0);
    }
}
