#pragma once
#include <utility>

namespace ce {
template<typename... Args>
struct variadic_typedef {};

template<typename... Args>
struct convert_in_tuple {
    //Leaving this empty will cause the compiler
    //to complain if you try to access a "type" member.
    //You may also be able to do something like:
    //static_assert(std::is_same<>::value, "blah")
    //if you know something about the types.
};

template<typename... Args>
struct convert_in_tuple< variadic_typedef<Args...> > {
    //use Args normally
    typedef std::tuple<Args...> type;
};

template<typename T, class T2>
struct append_to_tupple {};

template<typename T, typename... Args>
struct append_to_tupple<T, variadic_typedef<Args...> > {
    typedef variadic_typedef<T, Args...> type;
    typedef std::tuple<T, Args...> tuple_type;
};
}