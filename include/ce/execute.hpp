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
    typename std::enable_if<OutputPack<ct::remove_reference_t<Args>...>::OUTPUT_COUNT != 0>::type
    exec(void (*func)(FArgs...), Args&&... args)
    {
        auto eng = ICacheEngine::instance();
        if (eng)
        {
            return eng->exec(func, std::forward<Args>(args)...);
        }
        return func(ce::get(std::forward<Args>(args))...);
    }

    template <class R, class... FArgs, class... Args>
    ReturnSelect<R> exec(R (*func)(FArgs...), Args&&... args)
    {
        auto eng = ICacheEngine::instance();
        if (eng)
        {
            return eng->exec(func, std::forward<Args>(args)...);
        }
        return func(ce::get(std::forward<Args>(args))...);
    }

    template <class R, class T, class U, class... FARGS, class... ARGS>
    ReturnSelect<R> exec(R (T::*func)(FARGS...) const, const U& obj, ARGS&&... args)
    {
        auto eng = ICacheEngine::instance();
        if (eng)
        {
            return eng->exec(func, obj, std::forward<ARGS>(args)...);
        }
        return (getObjectRef(obj).*func)(std::forward<ARGS>(args)...);
    }

    template <class T, class U, class... FARGS, class... ARGS>
    void exec(void (T::*func)(FARGS...), U& obj, ARGS&&... args)
    {
        if (std::is_base_of<HashedBase, U>::value)
        {
            auto eng = ICacheEngine::instance();
            if (eng)
            {
                return eng->exec(func, obj, std::forward<ARGS>(args)...);
            }
        }
        return (getObjectRef(obj).*func)(std::forward<ARGS>(args)...);
    }
}
