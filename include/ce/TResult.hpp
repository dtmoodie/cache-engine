#pragma once
#include <ce/IResult.hpp>
#include <utility>

namespace ce
{
    template <class... T>
    struct TResult : IResult
    {
        TResult() = default;
        TResult(std::tuple<T...>&& arg)
            : values(std::move(arg))
        {
        }
        TResult(T&&... args)
            : values(std::forward<T>(args)...)
        {
        }

        std::tuple<T...> values;
    };

    template <class... T>
    struct TResult<std::tuple<T...>> : IResult
    {

        TResult() = default;
        TResult(std::tuple<T...>&& arg)
            : values(std::move(arg))
        {
        }
        TResult(T&&... args)
            : values(std::forward<T>(args)...)
        {
        }

        std::tuple<T...> values;
    };
}
