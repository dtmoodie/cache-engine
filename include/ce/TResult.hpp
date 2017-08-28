#pragma once
#include <ce/IResult.hpp>
#include <utility>
#include <ce/type_traits.hpp>
namespace ce {
template<class ... T> 
struct TResult : IResult {
    TResult(std::tuple<T...>&& arg) :
        values(std::move(arg)) {}
    TResult(T&&... args) :
        values(std::forward<T>(args)...) {
    }

    virtual size_t getDynamicSize() const{
        return ce::type_traits::argument_specializations::DynamicSize<T...>::getDynamicSize(values);
    }

    std::tuple<T...> values;
};

template<class...T> 
struct TResult<std::tuple<T...>> : IResult {
    TResult(std::tuple<T...>&& arg) :
        values(std::move(arg)) {}
    TResult(T&&... args) :
        values(std::forward<T>(args)...) {

    }
    virtual size_t getDynamicSize() const {
        return ce::type_traits::argument_specializations::DynamicSize<std::tuple<T...>>::getDynamicSize(values);
    }
    std::tuple<T...> values;
};
}